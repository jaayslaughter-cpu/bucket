"""Wave 3 manual paper-research layer.

Surfaces model vs market for human decisions; logs and grades bets the user
places manually. Never places wagers or sizes bankroll from the model.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.models.arbitration import arbitrate_probabilities
from src.models.edge_grades import research_edge_letter_grade
from src.models.projection_card import build_projection_card
from src.quant.contracts import MarketContext, PropMarketSnapshot, market_ev_gate
from src.quant.decision_board import line_can_push, price_sides, side_probabilities
from src.quant.historical_store import BetLifecycleRecord, HistoricalStore
from src.quant.line_diff import pickem_vs_book_line_diff
from src.utils.timezones import DISPLAY_TZ_NAME, format_pacific_iso, now_pacific

PLACEMENT_MODE = "MANUAL_ONLY"
WAVE3_DISCLAIMER = (
    "MANUAL_ONLY paper research — PropIQ does not place bets or size bankroll. "
    "You wager outside the system; this layer logs, grades, and audits."
)

BetSide = Literal["over", "under"]


class ManualBetInput(BaseModel):
    """User-entered bet after they placed (or intend to place) it manually."""

    game_id: str
    player_id: str | None = None
    player_name: str | None = None
    prop_stat: str
    line: float
    bet_side: BetSide
    taken_odds_american: int
    # model_prob is P(OVER), always — that is what the store records and what
    # the calibration reads. model_prob_side is P(the side you took); supply
    # it and nothing downstream has to reconstruct it with a complement that
    # is wrong on a whole line.
    model_prob: float
    model_prob_side: float | None = None
    bookmaker: str | None = None
    unit_stake: float = 1.0
    season: str | None = None
    opening_odds_american: int | None = None
    fair_prob_at_bet: float | None = None
    ev_at_bet_time: float | None = None
    confidence_tier: str | None = None
    edge_letter_grade: str | None = None
    model_name: str | None = None
    notes: str = "MANUAL_PAPER_LOG"
    source: str = "manual_user"


class ResearchSlateRow(BaseModel):
    """One research board row for a player-market (no auto-bet)."""

    slate_date: str
    event_id: str
    player_id: str
    player_name: str | None = None
    player_team: str | None = None
    opponent: str | None = None
    target_market: str
    research_line: float | None = None
    prediction_mean: float | None = None
    prediction_std: float | None = None
    model_p_over: float | None = None
    # P(under) and P(push) are carried explicitly because on a WHOLE line
    # P(under) is not 1 - P(over) — the difference is the push mass. Dropping
    # them forces every downstream consumer to reconstruct the under with a
    # complement that is wrong exactly when it matters.
    model_p_under: float | None = None
    model_p_push: float | None = None
    model_name: str | None = None
    confidence_tier: str | None = None
    edge_letter_grade: str | None = None
    over_under_meter: float | None = None
    hot_hand_status: str | None = None
    book_line: float | None = None
    book_over_american: int | None = None
    book_under_american: int | None = None
    book_status: str = "DATA_NOT_AVAILABLE"
    book_ev_over: float | None = None
    book_ev_under: float | None = None
    pickem_line: float | None = None
    pickem_source: str | None = None
    pickem_line_diff: float | None = None
    placement_mode: str = PLACEMENT_MODE
    timezone_display: str = DISPLAY_TZ_NAME
    disclaimer: str = WAVE3_DISCLAIMER
    warnings: list[str] = Field(default_factory=list)


def _as_probability(value: Any) -> float | None:
    """Keep a value only if it really is a probability."""
    try:
        p = float(value)
    except (TypeError, ValueError):
        return None
    return p if np.isfinite(p) and 0.0 <= p <= 1.0 else None


def enrich_row_with_book(
    row: ResearchSlateRow,
    market: MarketContext | PropMarketSnapshot | None,
    *,
    ev_threshold: float = 0.0,
) -> ResearchSlateRow:
    """
    Attach VALID two-way EV when a source provides it; else DATA_NOT_AVAILABLE.

    Source-neutral by design: PropLine is the primary feed and OddsPapi the
    fallback (see ``decision_board.resolve_market``), so this takes whichever
    snapshot was resolved rather than naming a vendor.
    """
    out = row.model_copy(deep=True)
    if market is None:
        out.book_status = "DATA_NOT_AVAILABLE"
        out.warnings.append("No priced market attached (PropLine or OddsPapi)")
        return out

    ctx = market.to_market_context() if isinstance(market, PropMarketSnapshot) else market
    gate = market_ev_gate(ctx)
    if isinstance(market, PropMarketSnapshot):
        out.book_line = market.line
        out.book_over_american = market.over_odds_american
        out.book_under_american = market.under_odds_american
    else:
        out.book_line = market.total
        out.book_over_american = market.over_odds_american
        out.book_under_american = market.under_odds_american

    if gate["status"] != "READY_FOR_EVALUATION":
        out.book_status = "DATA_NOT_AVAILABLE"
        out.warnings.append(str(gate.get("reason") or "Book not VALID"))
        return out

    out.book_status = "VALID"
    if out.model_p_over is None or out.book_over_american is None or out.book_under_american is None:
        out.warnings.append("Model P(over) or American odds missing for EV")
        return out

    # NOT 1 - P(over). On a whole line that complement is P(under) + P(push),
    # so it books every push as an under win and inflates the under's EV by
    # exactly the push mass. side_probabilities refuses it when a push is
    # possible and says so by name.
    probs = side_probabilities(
        p_over=out.model_p_over,
        p_under=out.model_p_under,
        p_push=out.model_p_push,
        line=out.book_line,
    )
    p_under, under_refusal = probs["under"]
    if p_under is None:
        out.warnings.append(str(under_refusal))

    priced, price_warning = price_sides(
        probs,
        over_american=out.book_over_american,
        under_american=out.book_under_american,
        game_id=out.event_id,
        line=out.book_line,
        min_ev=ev_threshold,
        p_push=out.model_p_push,
    )
    if "over" in priced:
        out.book_ev_over = priced["over"].ev
    if "under" in priced:
        out.book_ev_under = priced["under"].ev
    if price_warning and price_warning != under_refusal:
        out.warnings.append(price_warning)
    return out


def enrich_row_with_pickem(
    row: ResearchSlateRow,
    *,
    pickem_line: float | None,
    pickem_source: str | None,
    book_market: MarketContext | PropMarketSnapshot | None = None,
) -> ResearchSlateRow:
    """Attach pick'em line; line-diff only when book is VALID."""
    out = row.model_copy(deep=True)
    out.pickem_line = pickem_line
    out.pickem_source = pickem_source
    if pickem_line is None:
        return out
    if book_market is None:
        out.warnings.append("Pick'em line present; book VALID line-diff unavailable")
        return out
    diff = pickem_vs_book_line_diff(pickem_line, book_market, side="over")
    if diff.status == "OK":
        out.pickem_line_diff = diff.line_diff
    else:
        out.warnings.append(diff.reason or "pickem line-diff unavailable")
    return out


def research_slate_from_predictions(
    detail_rows: list[dict[str, Any]],
    *,
    slate_date: str,
    preferred_model: str | None = None,
) -> list[ResearchSlateRow]:
    """
    Collapse compare-models detail rows into one board row per player-market.

    Prefers ``preferred_model`` when present; else uses arbitration mean lean.
    """
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for r in detail_rows:
        key = (r.get("event_id"), r.get("player_id"), r.get("target_market"))
        groups.setdefault(key, []).append(r)

    board: list[ResearchSlateRow] = []
    for (event_id, player_id, market), rows in groups.items():
        probs = {str(r.get("model_name")): r.get("probability_over_raw") for r in rows}
        arb = arbitrate_probabilities(probs)
        chosen = None
        if preferred_model:
            chosen = next((r for r in rows if r.get("model_name") == preferred_model), None)
        if chosen is None:
            # Prefer distribution for mean/σ, else first row
            chosen = next((r for r in rows if r.get("model_name") == "distribution"), rows[0])

        p_over = chosen.get("probability_over_raw")
        if p_over is None and arb.get("mean_p_over") is not None:
            p_over = arb["mean_p_over"]

        card = build_projection_card(
            target_market=str(market),
            prop_line=chosen.get("prop_line"),
            prediction_mean=chosen.get("prediction_mean"),
            prediction_std=chosen.get("prediction_std_or_dispersion"),
            probability_over=p_over if p_over is None else float(p_over),
            probability_under=chosen.get("probability_under_raw"),
            player_id=str(player_id) if player_id is not None else None,
            player_name=chosen.get("player_name"),
            confidence_tier=arb.get("confidence_tier") or chosen.get("confidence_tier"),
            hot_hand_status=chosen.get("hot_hand_status"),
        )
        grade = research_edge_letter_grade(
            prediction_mean=chosen.get("prediction_mean"),
            prop_line=chosen.get("prop_line"),
            prediction_std=chosen.get("prediction_std_or_dispersion"),
            probability_over=None if p_over is None else float(p_over),
        )
        board.append(
            ResearchSlateRow(
                slate_date=slate_date,
                event_id=str(event_id or ""),
                player_id=str(player_id or ""),
                player_name=chosen.get("player_name"),
                player_team=chosen.get("player_team"),
                opponent=chosen.get("opponent"),
                target_market=str(market),
                research_line=chosen.get("prop_line"),
                prediction_mean=chosen.get("prediction_mean"),
                prediction_std=chosen.get("prediction_std_or_dispersion"),
                model_p_over=None if p_over is None else float(p_over),
                model_p_under=_as_probability(chosen.get("probability_under_raw")),
                model_p_push=_as_probability(chosen.get("probability_push_raw")),
                model_name=chosen.get("model_name"),
                confidence_tier=arb.get("confidence_tier"),
                edge_letter_grade=grade.get("edge_letter_grade"),
                over_under_meter=card.over_under_meter,
                hot_hand_status=chosen.get("hot_hand_status"),
            )
        )
    return board


def log_manual_bet(
    store: HistoricalStore,
    bet: ManualBetInput,
    *,
    actuals: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Log a user-placed (or paper-intended) bet. Grade pending first.

    ``unit_stake`` is **user-supplied** for research ROI — never computed by Kelly.
    """
    rec = BetLifecycleRecord(
        game_id=bet.game_id,
        player_id=bet.player_id,
        player_name=bet.player_name,
        prop_stat=bet.prop_stat,
        line=bet.line,
        bet_side=bet.bet_side,
        taken_odds_american=bet.taken_odds_american,
        model_prob=bet.model_prob,
        model_prob_side=bet.model_prob_side,
        bookmaker=bet.bookmaker,
        unit_stake=float(bet.unit_stake),
        season=bet.season,
        opening_odds_american=bet.opening_odds_american,
        fair_prob_at_bet=bet.fair_prob_at_bet,
        ev_at_bet_time=bet.ev_at_bet_time,
        confidence_tier=bet.confidence_tier,
        edge_letter_grade=bet.edge_letter_grade,
        model_name=bet.model_name,
        notes=f"{WAVE3_DISCLAIMER} | {bet.notes}",
        source=bet.source,
        market_type="player_prop",
    )
    result = store.append_after_grading([rec], actuals=actuals)
    result["placement_mode"] = PLACEMENT_MODE
    result["bet_id"] = result["appended_ids"][0] if result.get("appended_ids") else None
    result["disclaimer"] = WAVE3_DISCLAIMER
    return result


def paper_improvement_report(store: HistoricalStore) -> dict[str, Any]:
    """
    Aggregate paper-book stats for model improvement (not a profitability claim).
    """
    df = store.load_frame()
    base = {
        "report_timestamp_pt": format_pacific_iso(now_pacific()),
        "timezone_display": DISPLAY_TZ_NAME,
        "placement_mode": PLACEMENT_MODE,
        "disclaimer": WAVE3_DISCLAIMER,
        "roi_summary": store.roi_summary(),
    }
    if df.empty:
        base["status"] = "DATA_NOT_AVAILABLE"
        base["reason"] = "No paper bets logged yet"
        return base

    settled = df[df["bet_result"].isin(["WIN", "LOSS", "PUSH"])].copy()
    pending_n = int((df["bet_result"] == "PENDING").sum())
    by_stat: dict[str, Any] = {}
    if not settled.empty and "prop_stat" in settled.columns:
        for stat, g in settled.groupby(settled["prop_stat"].fillna("UNKNOWN")):
            stake = float(g["unit_stake"].fillna(1.0).sum())
            pnl = float(g["profit_loss"].fillna(0.0).sum())
            by_stat[str(stat)] = {
                "n": int(len(g)),
                "wins": int((g["bet_result"] == "WIN").sum()),
                "losses": int((g["bet_result"] == "LOSS").sum()),
                "roi": round(pnl / stake, 6) if stake else None,
            }

    # Simple probability calibration on settled overs/unders when model_prob present
    calib = None
    if not settled.empty and "model_prob" in settled.columns:
        hit = []
        probs = []
        skipped_push_ambiguous = 0
        for _, r in settled.iterrows():
            if r["bet_result"] == "PUSH":
                continue
            side = str(r.get("bet_side", "")).lower()
            won = r["bet_result"] == "WIN"

            # Prefer the recorded P(taken side). It is the only value that is
            # correct for an under bet on a whole line.
            p_side = _as_probability(r.get("model_prob_side"))
            if p_side is not None:
                probs.append(p_side)
                hit.append(1.0 if won else 0.0)
                continue

            p = _as_probability(r.get("model_prob"))
            if p is None:
                continue
            if side in {"over", "o"}:
                probs.append(p)
                hit.append(1.0 if won else 0.0)
            elif side in {"under", "u"}:
                # 1 - P(over) is P(under) + P(push). On a whole line that
                # overstates P(under) by the push mass, so the row is skipped
                # rather than scored against a number we know is too high.
                if line_can_push(r.get("line")):
                    skipped_push_ambiguous += 1
                    continue
                probs.append(1.0 - p)
                hit.append(1.0 if won else 0.0)
        if len(probs) >= 10:
            y = np.asarray(hit)
            p = np.clip(np.asarray(probs), 1e-6, 1 - 1e-6)
            brier = float(np.mean((p - y) ** 2))
            calib = {
                "n": int(len(y)),
                "brier": round(brier, 6),
                "mean_model_prob": round(float(p.mean()), 4),
                "skipped_push_ambiguous": skipped_push_ambiguous,
            }
        elif skipped_push_ambiguous:
            calib = {
                "status": "DATA_NOT_AVAILABLE",
                "reason": (
                    f"{skipped_push_ambiguous} under bets on whole lines carry no "
                    "recorded P(under); 1-P(over) would count push mass as an "
                    "under win. Log --model-prob-side going forward."
                ),
                "skipped_push_ambiguous": skipped_push_ambiguous,
            }

    base.update(
        {
            "status": "OK",
            "n_total": int(len(df)),
            "n_pending": pending_n,
            "n_settled": int(len(settled)),
            "by_prop_stat": by_stat,
            "probability_calibration": calib,
            "note": "Research audit of manual paper book — not a live P&L guarantee",
        }
    )
    return base


def slate_to_dataframe(rows: list[ResearchSlateRow]) -> pd.DataFrame:
    return pd.DataFrame([r.model_dump() for r in rows])


def write_slate_csv(rows: list[ResearchSlateRow], path: Any) -> int:
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df = slate_to_dataframe(rows)
    # Flatten warnings
    if "warnings" in df.columns:
        df["warnings"] = df["warnings"].apply(
            lambda w: "|".join(w) if isinstance(w, list) else w
        )
    df.to_csv(p, index=False)
    return len(df)
