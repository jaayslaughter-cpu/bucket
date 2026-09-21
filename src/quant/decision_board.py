"""
src/quant/decision_board.py — MANUAL_ONLY betting decision layer.

Status: RESEARCH_ONLY · MANUAL_ONLY.

This module ranks Over and Under so a person can choose. It does not place
wagers, does not size stakes, and has no code path to a sportsbook or DFS
order endpoint. The ranking is the deliverable; the wager happens outside
PropIQ, by hand.

SOURCE PRECEDENCE: PROPLINE IS PRIMARY, ODDSPAPI IS THE FALLBACK.

``resolve_market`` walks ``SOURCE_PRECEDENCE`` and takes the first source
whose market clears the EV gate. When PropLine is present but unusable —
most often because the row is a PrizePicks/Underdog pick'em board, which
publishes a payout multiplier rather than a two-way price — the resolution
falls through to OddsPapi and records ``fallback_used`` with the reason.
That is not a silent substitution: which source priced a row changes what
the EV means, so the source travels with every row.

FOUR DECISION BASES, and only one of them is a price:

- ``book_ev``          two-way American odds cleared the gate; EV is real
- ``model_lean``       no priced market; the model leans, and a lean is not
                       an edge because nothing says what it costs
- ``pickem_line_only`` a pick'em board posted a line; a payout multiplier
                       cannot be de-vigged, so EV stays undefined
- ``unavailable``      nothing to say

Rows are banded by basis before they are scored, so a model lean can never
outrank a priced edge. Ranking mixed bases on one scale would put a number
with no price behind it next to a number with one.

THE PUSH RULE. On a whole-number line N the bet has three outcomes: over
(> N), under (< N) and push (= N). So P(under) is NOT 1 − P(over) — the
difference is the push mass, and using the complement silently counts every
push as an under win. This module refuses the complement whenever a push is
possible, including when the line is unknown (an unknown line cannot be
shown to be a half-line). Half-lines cannot push, so there the complement
is exact and is used.

The whole layer is deliberately quiet: every refusal carries a named
reason, because "no EV shown" must never be mistaken for "no edge found".
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

import pandas as pd
from pydantic import BaseModel, Field

from src.quant.contracts import (
    GATE_READY,
    PropMarketSnapshot,
    market_ev_gate,
)
from src.quant.ev_engine import EvEngine, EvSide
from src.quant.odds_math import (
    american_to_implied_probability,
    american_to_profit_multiple,
    expected_value_per_unit,
    multiplicative_devig,
)
from src.utils.timezones import DISPLAY_TZ_NAME

logger = logging.getLogger(__name__)

PLACEMENT_MODE = "MANUAL_ONLY"
RESEARCH_STATUS = "RESEARCH_ONLY"

DECISION_DISCLAIMER = (
    "RESEARCH_ONLY · MANUAL_ONLY — this is a research ranking for your "
    "judgment. PropIQ does not place wagers, does not size stakes, and this "
    "is not live P&L. You bet outside the system, or not at all."
)

# PropLine first, OddsPapi second. Order is the whole point of this tuple.
SOURCE_PRECEDENCE: tuple[str, ...] = ("propline", "oddspapi")

DecisionStatus = Literal["CONSIDER", "ABSTAIN"]
DecisionBasis = Literal["book_ev", "model_lean", "pickem_line_only", "unavailable"]

# Ranking bands. A priced edge always sorts above an unpriced lean.
BASIS_ORDER: dict[str, int] = {
    "book_ev": 0,
    "model_lean": 1,
    "pickem_line_only": 2,
    "unavailable": 3,
}

SIDES: tuple[str, str] = ("over", "under")

# Never emitted in any user-facing string this module produces. A ranking is
# a ranking; language that promises an outcome is not research.
FORBIDDEN_CLAIM_WORDS: frozenset[str] = frozenset({
    "lock", "locks", "guaranteed", "guarantee", "best bet", "bestbet",
    "sure thing", "can't lose", "cant lose", "free money", "max bet",
})

# How far a model probability may drift from summing to one before we say so.
PROBABILITY_SUM_TOLERANCE = 0.02


class DecisionBoardError(RuntimeError):
    """Raised when the board cannot be built from what was supplied."""


# ---------------------------------------------------------------------------
# source resolution: PropLine primary, OddsPapi fallback
# ---------------------------------------------------------------------------


def normalise_source(name: Any) -> str:
    """'OddsPapi ' -> 'oddspapi'. Unknown names pass through, lowercased."""
    return str(name or "").strip().lower().replace("-", "").replace("_", "").replace(" ", "")


def _source_rank(name: Any, precedence: Sequence[str]) -> int:
    """Position in the precedence list; unlisted sources sort last."""
    key = normalise_source(name)
    for idx, candidate in enumerate(precedence):
        if key == normalise_source(candidate):
            return idx
    return len(precedence)


def propline_row_to_snapshot(row: Any, *, game_id: str | None = None) -> PropMarketSnapshot:
    """
    Bridge a ``PropLineRow`` into the quant layer's ``PropMarketSnapshot``.

    Read by attribute rather than by import so the quant layer does not take
    a hard dependency on the ingestion layer (and on ``requests`` with it).
    A row missing the fields below is refused by name rather than silently
    producing a snapshot full of None.
    """
    required = ("source", "player_name", "market", "line", "status")
    missing = [f for f in required if not hasattr(row, f)]
    if missing:
        raise DecisionBoardError(
            f"Not a PropLine row: missing {missing}. Expected the dataclass from "
            "src/ingestion/propline.py."
        )
    resolved_game = game_id or getattr(row, "nba_game_id", None) or ""
    return PropMarketSnapshot(
        game_id=str(resolved_game),
        market=getattr(row, "market", None),
        player_name=getattr(row, "player_name", None),
        player_id=getattr(row, "nba_player_id", None),
        line=getattr(row, "line", None),
        over_odds_american=getattr(row, "over_odds_american", None),
        under_odds_american=getattr(row, "under_odds_american", None),
        payout_multiplier=getattr(row, "payout_multiplier", None),
        is_pickem=bool(getattr(row, "is_pickem", False)),
        bookmaker=getattr(row, "source", None),
        # The BOOK is in `bookmaker`; `source` is the feed that carried it,
        # which is what precedence is about.
        source="propline",
        status=getattr(row, "status", "DATA_NOT_AVAILABLE"),
        captured_at_utc=getattr(row, "captured_at_utc", None),
    )


@dataclass(frozen=True)
class MarketResolution:
    """Which source priced this market, and what was passed over to get there."""

    snapshot: PropMarketSnapshot | None = None
    source: str | None = None
    fallback_used: bool = False
    considered: tuple[str, ...] = ()
    skipped: tuple[tuple[str, str], ...] = ()
    pickem_snapshot: PropMarketSnapshot | None = None
    reason: str | None = None

    @property
    def is_priced(self) -> bool:
        return self.snapshot is not None

    @property
    def skipped_summary(self) -> str:
        return "; ".join(f"{src}: {why}" for src, why in self.skipped)


def resolve_market(
    candidates: Iterable[PropMarketSnapshot] | Mapping[str, PropMarketSnapshot] | None,
    *,
    precedence: Sequence[str] = SOURCE_PRECEDENCE,
) -> MarketResolution:
    """
    Pick the market to price from, PropLine first and OddsPapi as fallback.

    A source is usable only when ``market_ev_gate`` returns
    READY_FOR_EVALUATION for it. Everything skipped on the way down is
    recorded with the gate's own reason, so a fallback is always explainable
    and a pick'em row is never mistaken for a missing one.
    """
    if candidates is None:
        return MarketResolution(reason="No market candidates supplied")

    items: list[PropMarketSnapshot] = (
        list(candidates.values()) if isinstance(candidates, Mapping) else list(candidates)
    )
    if not items:
        return MarketResolution(reason="No market candidates supplied")

    ordered = sorted(items, key=lambda s: _source_rank(s.source or s.bookmaker, precedence))
    considered = tuple(normalise_source(s.source or s.bookmaker) for s in ordered)

    skipped: list[tuple[str, str]] = []
    pickem: PropMarketSnapshot | None = None

    for snap in ordered:
        name = normalise_source(snap.source or snap.bookmaker)
        gate = market_ev_gate(snap.to_market_context())
        if gate["status"] == GATE_READY:
            return MarketResolution(
                snapshot=snap,
                source=name,
                # A fallback is "used" only when something ahead of it was
                # present and unusable, not merely because the list is short.
                fallback_used=bool(skipped),
                considered=considered,
                skipped=tuple(skipped),
                pickem_snapshot=pickem,
                reason=None,
            )
        why = str(gate.get("reason") or "not usable")
        skipped.append((name, why))
        if pickem is None and (snap.is_pickem or snap.payout_multiplier is not None):
            pickem = snap

    return MarketResolution(
        source=None,
        considered=considered,
        skipped=tuple(skipped),
        pickem_snapshot=pickem,
        reason=(
            "; ".join(f"{src}: {why}" for src, why in skipped)
            or "No source cleared the EV gate"
        ),
    )


# ---------------------------------------------------------------------------
# the push rule
# ---------------------------------------------------------------------------


def line_can_push(line: Any) -> bool:
    """
    True when the posted number can be landed on exactly.

    A whole line (24.0) can push; a half-line (24.5) cannot. An unknown line
    is treated as CAN push, because a line we cannot see cannot be shown to
    be a half-line, and the safe default is the one that refuses.
    """
    if line is None:
        return True
    try:
        value = float(line)
    except (TypeError, ValueError):
        return True
    if not math.isfinite(value):
        return True
    return float(value).is_integer()


def _is_probability(value: Any) -> bool:
    try:
        p = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(p) and 0.0 <= p <= 1.0


def side_probabilities(
    *,
    p_over: Any,
    p_under: Any = None,
    p_push: Any = None,
    line: Any = None,
) -> dict[str, tuple[float | None, str | None]]:
    """
    Model probability per side, refusing the complement when a push is possible.

    Returns ``{side: (probability, refusal_reason)}``. Exactly one of the two
    is None for each side.

    The under side is the whole point. ``1 - P(over)`` is the sum of P(under)
    and P(push); on a whole line that counts every push as an under win, which
    inflates the under's EV by the push mass. So the complement is used only
    on a half-line, where the push mass is zero by construction.
    """
    out: dict[str, tuple[float | None, str | None]] = {}

    over = float(p_over) if _is_probability(p_over) else None
    out["over"] = (over, None if over is not None else "Model P(over) unavailable")

    if _is_probability(p_under):
        under = float(p_under)
        out["under"] = (under, None)
    elif line is None:
        out["under"] = (
            None,
            "No P(under) and no line: a line we cannot see cannot be shown to be "
            "a half-line, so 1-P(over) may be counting push mass as an under win",
        )
    elif line_can_push(line):
        out["under"] = (
            None,
            f"Whole line {float(line):g} can push: P(under) is not 1-P(over), "
            "and the difference is the push mass. Supply an explicit P(under).",
        )
    elif over is None:
        out["under"] = (None, "Model P(over) unavailable, so its complement is too")
    else:
        # Half-line: over and under are exhaustive, so the complement is exact.
        out["under"] = (1.0 - over, None)

    if _is_probability(p_over) and _is_probability(p_under):
        total = float(p_over) + float(p_under) + (float(p_push) if _is_probability(p_push) else 0.0)
        if abs(total - 1.0) > PROBABILITY_SUM_TOLERANCE:
            logger.warning(
                "Model over/under/push probabilities sum to %.4f, not 1. Reported "
                "as given rather than rescaled — the gap is a model property.",
                total,
            )
    return out


# ---------------------------------------------------------------------------
# the board
# ---------------------------------------------------------------------------


class DecisionRow(BaseModel):
    """One side of one player-market, ranked for a human to choose from."""

    slate_date: str = ""
    event_id: str = ""
    player_id: str = ""
    player_name: str | None = None
    target_market: str = ""
    line: float | None = None

    side: str = "over"
    decision_status: DecisionStatus = "ABSTAIN"
    decision_basis: DecisionBasis = "unavailable"
    book_ev: float | None = None
    preferred_side: str | None = None
    why: str = ""
    rank: int | None = None

    # What the decision rests on, so a row can be audited without the board.
    model_prob_side: float | None = None
    fair_prob_side: float | None = None
    edge_vs_devig: float | None = None
    side_odds_american: int | None = None
    lean_score: float | None = None

    book_source: str | None = None
    book_bookmaker: str | None = None
    book_status: str = "DATA_NOT_AVAILABLE"
    fallback_used: bool = False
    sources_considered: str = ""
    sources_skipped: str = ""
    line_can_push: bool = True

    pickem_line: float | None = None
    pickem_source: str | None = None

    placement_mode: str = PLACEMENT_MODE
    research_status: str = RESEARCH_STATUS
    timezone_display: str = DISPLAY_TZ_NAME
    disclaimer: str = DECISION_DISCLAIMER
    warnings: list[str] = Field(default_factory=list)


def _assert_no_claims(text: str) -> str:
    """Guard the vocabulary. A ranking never promises an outcome."""
    lowered = str(text).lower()
    hit = next((w for w in FORBIDDEN_CLAIM_WORDS if w in lowered), None)
    if hit:
        raise DecisionBoardError(
            f"Refusing to emit {text!r}: {hit!r} states an outcome this layer "
            "cannot support. The board ranks candidates; it does not promise."
        )
    return text


def price_sides(
    probs: Mapping[str, tuple[float | None, str | None]],
    *,
    over_american: int | None,
    under_american: int | None,
    game_id: str = "",
    line: Any = None,
    min_ev: float = 0.0,
    p_push: Any = None,
    market_id: str | None = None,
) -> tuple[dict[str, EvSide], str | None]:
    """
    Price whichever sides have both a price and a model probability.

    A refused P(under) must not cost the OVER side its EV. The de-vig needs
    both PRICES (which the gate has already guaranteed), but each side's EV
    needs only its own probability — so a whole line with no explicit
    P(under) still yields a real over EV, and the under simply abstains.

    When both probabilities are present the arithmetic goes through
    ``EvEngine`` so there is one EV path and one sum-to-one warning; the
    per-side branch reuses the same ``odds_math`` primitives and produces
    identical numbers.

    The per-side branch is also taken when the model supplied push mass.
    ``EvEngine`` knows only two sides, so a correct over/under/push triple
    would trip its "probabilities sum to 0.92, not 1" warning — an alarm
    about the one case that is actually right.
    """
    if over_american is None or under_american is None:
        # Unreachable via the gate, which requires both; guarded anyway so a
        # future caller that skips the gate fails loudly rather than pricing
        # one side against a missing counterpart.
        return {}, "Two-way American odds required to de-vig; one side is missing"

    p_over, over_refusal = probs["over"]
    p_under, under_refusal = probs["under"]

    has_push = _is_probability(p_push) and float(p_push) > 0.0

    if p_over is not None and p_under is not None and not has_push:
        evaluation = EvEngine(ev_threshold=min_ev).evaluate_two_way(
            game_id=game_id,
            american_a=int(over_american),
            american_b=int(under_american),
            model_prob_a=p_over,
            model_prob_b=p_under,
            label_a="over",
            label_b="under",
            line=line,
            market_type="player_prop",
            market_id=market_id,
        )
        if evaluation.status == "OK":
            return {s.side: s for s in evaluation.sides}, None
        return {}, evaluation.reason or evaluation.status

    if p_over is None and p_under is None:
        return {}, over_refusal or under_refusal or "Model probabilities unavailable"

    try:
        fair = multiplicative_devig(int(over_american), int(under_american))
    except (TypeError, ValueError) as exc:
        return {}, f"Cannot de-vig this market: {exc}"

    priced: dict[str, EvSide] = {}
    for side, american, model_p, fair_p in (
        ("over", int(over_american), p_over, fair.fair_prob_a),
        ("under", int(under_american), p_under, fair.fair_prob_b),
    ):
        if model_p is None:
            continue
        priced[side] = EvSide(
            side=side,
            american=american,
            model_prob=float(model_p),
            fair_prob=fair_p,
            implied_prob=american_to_implied_probability(american),
            ev=expected_value_per_unit(float(model_p), american),
            # Edge against the DE-VIGGED price, never the posted one.
            edge=float(model_p) - fair_p,
            profit_multiple=american_to_profit_multiple(american),
        )
    return priced, (over_refusal if p_over is None else under_refusal)


def expand_sides(
    *,
    slate_date: str = "",
    event_id: str = "",
    player_id: str = "",
    player_name: str | None = None,
    target_market: str = "",
    model_p_over: Any = None,
    model_p_under: Any = None,
    model_p_push: Any = None,
    research_line: Any = None,
    resolution: MarketResolution | None = None,
    pickem_line: Any = None,
    pickem_source: str | None = None,
    min_ev: float = 0.0,
    min_lean: float = 0.0,
) -> list[DecisionRow]:
    """
    Expand one player-market into BOTH sides, always. Never just the lean.

    A board that shows only the side the model likes has already made the
    choice it claims to be leaving to you, and hides the price on the other
    side — which is where the edge usually is when the model is wrong.
    """
    resolution = resolution or MarketResolution(reason="No market candidates supplied")
    snapshot = resolution.snapshot
    line = snapshot.line if snapshot is not None and snapshot.line is not None else research_line

    probs = side_probabilities(
        p_over=model_p_over, p_under=model_p_under, p_push=model_p_push, line=line,
    )

    if snapshot is None:
        ev_by_side: dict[str, EvSide] = {}
        ev_warning: str | None = None
    else:
        ev_by_side, ev_warning = price_sides(
            probs,
            over_american=snapshot.over_odds_american,
            under_american=snapshot.under_odds_american,
            game_id=event_id or snapshot.game_id,
            line=line,
            min_ev=min_ev,
            p_push=model_p_push,
            market_id=snapshot.market_key,
        )
    # "Both priced" means both sides have a model probability AND a price.
    # A refused under leaves the over priced and the pair incomparable.
    preferred: str | None = None
    if {"over", "under"} <= set(ev_by_side):
        preferred = max(ev_by_side.values(), key=lambda s: s.ev).side

    pickem_snapshot = resolution.pickem_snapshot
    resolved_pickem_line = pickem_line
    if resolved_pickem_line is None and pickem_snapshot is not None:
        resolved_pickem_line = pickem_snapshot.line
    resolved_pickem_source = pickem_source
    if resolved_pickem_source is None and pickem_snapshot is not None:
        resolved_pickem_source = pickem_snapshot.bookmaker or pickem_snapshot.source

    rows: list[DecisionRow] = []
    for side in SIDES:
        prob, refusal = probs[side]
        ev_side = ev_by_side.get(side)
        warnings: list[str] = []

        basis: DecisionBasis
        status: DecisionStatus = "ABSTAIN"
        why: str

        if ev_side is not None:
            basis = "book_ev"
            if ev_side.ev >= min_ev:
                status = "CONSIDER"
                why = (
                    f"{resolution.source} two-way price de-vigged; {side} EV "
                    f"{ev_side.ev:+.4f} clears {min_ev:+.4f}"
                )
            else:
                why = (
                    f"{resolution.source} two-way price de-vigged; {side} EV "
                    f"{ev_side.ev:+.4f} below {min_ev:+.4f}"
                )
        elif refusal is not None and snapshot is not None:
            basis = "unavailable"
            why = refusal
        elif resolved_pickem_line is not None and snapshot is None:
            basis = "pickem_line_only"
            why = (
                "Pick'em board posts a payout multiplier, not a two-way price, "
                "so EV is undefined — line research only"
            )
            if refusal:
                warnings.append(refusal)
        elif prob is not None:
            basis = "model_lean"
            lean = prob - 0.5
            if lean > min_lean:
                status = "CONSIDER"
                why = (
                    f"No priced market ({resolution.reason or 'no source'}); model "
                    f"leans {side} at P={prob:.3f} — a lean, not an edge, because "
                    "nothing here says what it costs"
                )
            else:
                why = f"Model does not lean {side} (P={prob:.3f})"
        else:
            basis = "unavailable"
            why = refusal or resolution.reason or "No model probability and no priced market"

        if ev_warning and basis != "book_ev":
            warnings.append(ev_warning)
        if refusal and basis in {"model_lean", "book_ev"}:
            warnings.append(refusal)

        rows.append(DecisionRow(
            slate_date=slate_date,
            event_id=str(event_id or ""),
            player_id=str(player_id or ""),
            player_name=player_name,
            target_market=str(target_market or ""),
            line=None if line is None else float(line),
            side=side,
            decision_status=status,
            decision_basis=basis,
            book_ev=None if ev_side is None else float(ev_side.ev),
            preferred_side=preferred,
            why=_assert_no_claims(why),
            model_prob_side=prob,
            fair_prob_side=None if ev_side is None else float(ev_side.fair_prob),
            edge_vs_devig=None if ev_side is None else float(ev_side.edge),
            side_odds_american=None if ev_side is None else int(ev_side.american),
            lean_score=None if prob is None else float(prob) - 0.5,
            book_source=resolution.source,
            book_bookmaker=None if snapshot is None else snapshot.bookmaker,
            book_status="VALID" if snapshot is not None else "DATA_NOT_AVAILABLE",
            fallback_used=resolution.fallback_used,
            sources_considered="|".join(resolution.considered),
            sources_skipped=resolution.skipped_summary,
            line_can_push=line_can_push(line),
            pickem_line=None if resolved_pickem_line is None else float(resolved_pickem_line),
            pickem_source=resolved_pickem_source,
            warnings=warnings,
        ))
    return rows


def _score(row: DecisionRow) -> float:
    """Within a basis band: EV for priced rows, lean for unpriced ones."""
    if row.decision_basis == "book_ev" and row.book_ev is not None:
        return float(row.book_ev)
    if row.lean_score is not None:
        return float(row.lean_score)
    return float("-inf")


def rank_decisions(
    rows: Sequence[DecisionRow],
    *,
    require_valid_book: bool = False,
    consider_only: bool = False,
    top_n: int | None = None,
) -> list[DecisionRow]:
    """
    Order the board: CONSIDER first, then by band, then by score.

    The band (``BASIS_ORDER``) sits between status and score deliberately.
    Sorting a model lean and a de-vigged EV on one numeric scale would imply
    the two numbers are comparable; they are not, because only one of them
    has a price behind it.
    """
    out = [r.model_copy(deep=True) for r in rows]

    if require_valid_book:
        for row in out:
            if row.decision_basis != "book_ev" and row.decision_status == "CONSIDER":
                row.decision_status = "ABSTAIN"
                row.why = _assert_no_claims(
                    f"{row.why} [--require-valid-book: no two-way price, so no EV]"
                )

    if consider_only:
        out = [r for r in out if r.decision_status == "CONSIDER"]

    out.sort(
        key=lambda r: (
            0 if r.decision_status == "CONSIDER" else 1,
            BASIS_ORDER.get(r.decision_basis, len(BASIS_ORDER)),
            -_score(r),
            r.player_name or "",
            r.target_market,
            r.side,
        )
    )
    for idx, row in enumerate(out, start=1):
        row.rank = idx

    if top_n is not None and top_n >= 0:
        out = out[:top_n]
    return out


def _field_reader(row: Any):
    """Read a slate row whether it is a mapping or a pydantic/dataclass model."""
    if isinstance(row, Mapping):
        return lambda key, default=None: row.get(key, default)
    return lambda key, default=None: getattr(row, key, default)


def decision_board_from_slate(
    slate_rows: Sequence[Any],
    *,
    markets: Mapping[Any, Any] | None = None,
    min_ev: float = 0.0,
    min_lean: float = 0.0,
    require_valid_book: bool = False,
    consider_only: bool = False,
    top_n: int | None = None,
) -> list[DecisionRow]:
    """
    Build the ranked board from research slate rows.

    ``markets`` maps ``(event_id, player_id, target_market)`` — or any key the
    caller also uses on the slate rows — to the market candidates for that
    player-market. Candidates may be ``PropMarketSnapshot`` objects or already
    resolved; PropLine is preferred over OddsPapi by ``resolve_market``.

    With no markets supplied, every row falls to ``model_lean`` or
    ``unavailable``. That is the honest state of this repository today: no
    archived PropLine pull exists yet, so nothing here is priced.
    """
    expanded: list[DecisionRow] = []
    for row in slate_rows:
        get = _field_reader(row)
        key = (get("event_id"), get("player_id"), get("target_market"))
        candidates = None
        if markets:
            candidates = markets.get(key)
            if candidates is None:
                candidates = markets.get(get("player_id"))
        if isinstance(candidates, MarketResolution):
            resolution = candidates
        elif isinstance(candidates, PropMarketSnapshot):
            resolution = resolve_market([candidates])
        else:
            resolution = resolve_market(candidates)

        expanded.extend(expand_sides(
            slate_date=str(get("slate_date") or ""),
            event_id=str(get("event_id") or ""),
            player_id=str(get("player_id") or ""),
            player_name=get("player_name"),
            target_market=str(get("target_market") or ""),
            model_p_over=get("model_p_over"),
            model_p_under=get("model_p_under"),
            model_p_push=get("model_p_push"),
            research_line=get("research_line"),
            resolution=resolution,
            pickem_line=get("pickem_line"),
            pickem_source=get("pickem_source"),
            min_ev=min_ev,
            min_lean=min_lean,
        ))

    return rank_decisions(
        expanded,
        require_valid_book=require_valid_book,
        consider_only=consider_only,
        top_n=top_n,
    )


# ---------------------------------------------------------------------------
# handoff to the manual log
# ---------------------------------------------------------------------------


def decision_log_fields(row: DecisionRow) -> dict[str, Any]:
    """
    The fields to hand ``log-manual-bet`` after YOU place the bet.

    ``model_prob`` is P(OVER) — that is what the store records and what
    ``paper_improvement_report`` reads, regardless of which side was taken.
    ``model_prob_side`` is P(the side you took), carried separately so the
    calibration never has to reconstruct it with a complement that would be
    wrong on a whole line.

    No stake is returned. Sizing is yours; the model has no opinion it has
    earned the right to express.
    """
    p_side = row.model_prob_side
    if row.side == "over":
        p_over = p_side
    elif p_side is None:
        p_over = None
    elif row.line_can_push:
        # Cannot invert P(under) into P(over) without the push mass.
        p_over = None
    else:
        p_over = 1.0 - float(p_side)

    return {
        "game_id": row.event_id,
        "player_id": row.player_id or None,
        "player_name": row.player_name,
        "prop_stat": row.target_market,
        "line": row.line,
        "side": row.side,
        "model_prob": p_over,
        "model_prob_side": p_side,
        "odds": row.side_odds_american,
        "bookmaker": row.book_bookmaker,
        "source": row.book_source,
        "ev_at_bet_time": row.book_ev,
        "fair_prob_at_bet": row.fair_prob_side,
        "placement_mode": PLACEMENT_MODE,
        "note": (
            "Place the wager yourself. --unit-stake is YOUR choice; PropIQ "
            "never sizes it."
        ),
    }


def board_to_dataframe(rows: Sequence[DecisionRow]) -> pd.DataFrame:
    frame = pd.DataFrame([r.model_dump() for r in rows])
    if "warnings" in frame.columns:
        frame["warnings"] = frame["warnings"].apply(
            lambda w: "|".join(w) if isinstance(w, list) else w
        )
    return frame


def write_decision_board_csv(rows: Sequence[DecisionRow], path: Any) -> int:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame = board_to_dataframe(rows)
    frame.to_csv(target, index=False)
    return len(frame)


def board_summary(rows: Sequence[DecisionRow]) -> dict[str, Any]:
    """Counts a human can sanity-check the board against."""
    by_basis: dict[str, int] = {}
    for row in rows:
        by_basis[row.decision_basis] = by_basis.get(row.decision_basis, 0) + 1
    priced = [r for r in rows if r.decision_basis == "book_ev"]
    return {
        "placement_mode": PLACEMENT_MODE,
        "research_status": RESEARCH_STATUS,
        "rows": len(rows),
        "consider": sum(1 for r in rows if r.decision_status == "CONSIDER"),
        "abstain": sum(1 for r in rows if r.decision_status == "ABSTAIN"),
        "by_basis": by_basis,
        "priced_rows": len(priced),
        "sources_used": sorted({r.book_source for r in priced if r.book_source}),
        "fallback_rows": sum(1 for r in rows if r.fallback_used),
        "disclaimer": DECISION_DISCLAIMER,
    }
