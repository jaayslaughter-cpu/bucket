"""
src/quant/decision_board.py — the recommendation layer.

Expands each market into Over and Under candidates and RECOMMENDS the side the
evidence supports, where there is evidence. It does not place the wager: there
is no order API in this project and none is planned, so the last step is always
yours.

WHAT "RECOMMENDED" IS WORTH IS THE WHOLE QUESTION, and the row says so rather
than leaving it to be inferred. A recommendation on ``book_ev`` has a real
two-way price behind it, so its EV is measured against a de-vigged market
consensus. One on ``model_lean`` has no price at all.

BOTH STILL REST ON THE MODEL'S OWN PROBABILITY. A book_ev row compares THE
MODEL's probability against the market's; the price is what it is measured
AGAINST, not the number being used. So every row on this board is model-sourced,
and until the model has graded results behind it every row is an opinion with a
confident label. ``quant.publication_gate`` is what withholds them from a
channel — applied by the caller at dispatch (``notify-discord``, and
``build_decision_board_embed``'s ``publication`` argument), not inside this
module, which only builds rows.

A RECOMMENDATION IS STILL NOT A PROMISE. The vocabulary guard below stays: no
"lock", no "guaranteed", no "best bet". Recommending a side and promising an
outcome are different acts, and only the first one is supportable.

SOURCE PRECEDENCE: PROPLINE IS THE ONLY SOURCE.

``resolve_market`` walks ``SOURCE_PRECEDENCE`` and takes the first source whose
market clears the EV gate — by precedence, not by arrival order — recording
every source it passed over with the gate's own reason. Which source priced a
row changes what its EV means, so the source travels with the row.

``SOURCE_PRECEDENCE`` holds ONE name. It listed a second, OddsPapi, which had
no client in this repository at all; see the note on the tuple. With one entry
``fallback_used`` can only ever be False in production, and that is the honest
answer: when PropLine posts a payout multiplier rather than a two-way price —
most often a PrizePicks or Underdog pick'em row — there is nothing to fall
through to, so the row abstains and says why.

FOUR DECISION BASES, and only one of them is a price:

- ``book_ev``          two-way American odds cleared the gate; EV is real
- ``model_lean``       no priced market; the model leans, and a lean is not
                       an edge because nothing says what it costs
- ``pickem_line_only`` only a pick'em line exists; a payout multiplier
                       cannot be de-vigged, so EV stays undefined
- ``unavailable``      nothing to say

Rows are BANDED by basis before they are scored, so a model lean can never
outrank a priced edge however large the lean. Ranking mixed bases on one
numeric scale would put a number with no price behind it next to one with a
price behind it.

THE PUSH RULE lives in ``paper_research.resolve_two_way_model_probs``: on a
whole-number line P(under) is not 1 - P(over), the difference is the push
mass, and the complement is refused whenever a push is possible — including
when the line is unknown, since an unseen line cannot be shown to be a
half-line.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping, Sequence

import pandas as pd
from pydantic import BaseModel, Field

from src.quant.contracts import GATE_READY, PropMarketSnapshot, market_ev_gate
from src.quant.paper_research import (
    PLACEMENT_MODE,
    WAVE3_DISCLAIMER,
    ResearchSlateRow,
    enrich_row_with_book,
    is_whole_number_line,
    model_prob_for_side,
    resolve_two_way_model_probs,
)
from src.utils.timezones import DISPLAY_TZ_NAME

logger = logging.getLogger(__name__)

RESEARCH_STATUS = "RESEARCH_ONLY"

DecisionStatus = Literal["RECOMMENDED", "ABSTAIN"]
DecisionBasis = Literal["book_ev", "model_lean", "pickem_line_only", "unavailable"]

BOARD_DISCLAIMER = (
    "Recommendations, not a promise of an outcome. Each row names the basis it "
    "rests on: book_ev is a de-vigged two-way price, model_lean is the model's "
    "opinion with no price behind it. PropIQ does not place the wager. "
    + WAVE3_DISCLAIMER
)

# ONE SOURCE, AND IT IS STILL A TUPLE ON PURPOSE.
#
# This read ("propline", "oddspapi") until 2026-10-04. OddsPapi had no client
# module, no key reader and no ingestion path anywhere in this repository: it
# was a string in this tuple and prose in a dozen docstrings. A precedence
# naming a feed that cannot be fetched is not a safety net, it is a claim that
# a fallback exists — so the next person debugging an unpriced row looks for a
# fallback that was never there.
#
# The WALKER stays generic. `resolve_market` takes any `precedence` sequence,
# ranks unlisted sources last, and records every source it passed over with the
# gate's own reason. Adding a second feed is appending its name here once its
# client exists — and `tests/test_decision_board.py` holds that mechanism open
# by walking a two-source precedence built in the test, rather than by this
# repository pretending to have two.
SOURCE_PRECEDENCE: tuple[str, ...] = ("propline",)

# Ranking bands. A priced edge always sorts above an unpriced lean.
BASIS_ORDER: dict[str, int] = {
    "book_ev": 0,
    "model_lean": 1,
    "pickem_line_only": 2,
    "unavailable": 3,
}

# How far past P=0.50 an unpriced lean must sit before it is worth surfacing.
DEFAULT_MIN_LEAN = 0.03

# Never emitted in any user-facing string this module produces. A ranking is
# a ranking; language that promises an outcome is not research.
FORBIDDEN_CLAIM_WORDS: frozenset[str] = frozenset({
    "lock", "locks", "guaranteed", "guarantee", "best bet", "bestbet",
    "sure thing", "can't lose", "cant lose", "free money", "max bet",
})


class DecisionBoardError(RuntimeError):
    """Raised when the board cannot be built from what was supplied."""


def line_can_push(line: Any) -> bool:
    """A whole line (or an unknown one) can push; a half-line cannot."""
    return is_whole_number_line(line)


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


# ---------------------------------------------------------------------------
# source resolution by precedence — PropLine is the only listed source
# ---------------------------------------------------------------------------


def normalise_source(name: Any) -> str:
    """'Prop Line ' -> 'propline'. Unknown names pass through, lowercased."""
    return (
        str(name or "").strip().lower()
        .replace("-", "").replace("_", "").replace(" ", "")
    )


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
        # The BOOK is in `bookmaker`; `source` is the FEED that carried it,
        # which is what precedence reads.
        bookmaker=getattr(row, "source", None),
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
    Pick the market to price from, by precedence rather than arrival order.

    A source is usable only when ``market_ev_gate`` returns
    READY_FOR_EVALUATION for it. Everything skipped on the way down is
    recorded with the gate's own reason, so a fallback is always explainable
    and a pick'em row is never mistaken for a missing one.

    THIS RANKS, IT DOES NOT ALLOW-LIST. ``_source_rank`` sorts a source that is
    absent from ``precedence`` LAST; it does not exclude it. A snapshot from an
    unlisted source is still priced, after every listed one. Worth stating
    because removing OddsPapi from the tuple removed a promise of a fallback
    and did not add a filter — none is needed while nothing can fetch it.
    ``tests/test_decision_board.py`` pins this either way.
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
            )
        why = str(gate.get("reason") or "not usable")
        skipped.append((name, why))
        if pickem is None and (snap.is_pickem or snap.payout_multiplier is not None):
            pickem = snap

    return MarketResolution(
        considered=considered,
        skipped=tuple(skipped),
        pickem_snapshot=pickem,
        reason=(
            "; ".join(f"{src}: {why}" for src, why in skipped)
            or "No source cleared the EV gate"
        ),
    )


def enrich_row_with_resolved_market(
    row: ResearchSlateRow,
    candidates: Iterable[PropMarketSnapshot] | None,
    *,
    ev_threshold: float = 0.0,
    precedence: Sequence[str] = SOURCE_PRECEDENCE,
) -> tuple[ResearchSlateRow, MarketResolution]:
    """
    Resolve the source by precedence, then price the row from it.

    This is the entry point that makes the precedence real rather than a
    docstring: ``enrich_row_with_book`` prices whatever snapshot it is given,
    and this decides which snapshot that is.

    ``precedence`` is passed through rather than fixed, because ``resolve_market``
    has always accepted it and this caller did not — so the fallback path could
    not be exercised end to end at all. Adding a second feed is a one-line
    change here, once its client exists.
    """
    resolution = resolve_market(candidates, precedence=precedence)
    out = enrich_row_with_book(row, resolution.snapshot, ev_threshold=ev_threshold)
    out.book_source = resolution.source
    out.book_fallback_used = resolution.fallback_used
    out.book_sources_skipped = resolution.skipped_summary
    if resolution.pickem_snapshot is not None and out.pickem_line is None:
        out.pickem_line = resolution.pickem_snapshot.line
        out.pickem_source = (
            resolution.pickem_snapshot.bookmaker or resolution.pickem_snapshot.source
        )
    if resolution.snapshot is None and resolution.reason:
        out.warnings.append(resolution.reason)
    return out, resolution

# ---------------------------------------------------------------------------
# candidates
# ---------------------------------------------------------------------------


class BettingDecisionCandidate(BaseModel):
    """One actionable (or abstained) side for human wager selection."""

    rank: int = 0
    slate_date: str
    # Carried from ResearchSlateRow.game_date -- see the long note there. This
    # is the date of the GAME; slate_date is when the board was built. Both
    # reach the CSV and the Discord embed, because a reader who sees only one
    # of them cannot tell a projection from a backtest row.
    game_date: str | None = None
    event_id: str
    player_id: str
    player_name: str | None = None
    player_team: str | None = None
    opponent: str | None = None
    target_market: str
    side: Literal["over", "under"]
    line: float | None = None
    model_prob: float | None = None
    model_p_over: float | None = None
    model_p_under: float | None = None
    model_p_push: float | None = None
    prediction_mean: float | None = None
    prediction_std: float | None = None
    american_odds: int | None = None
    book_status: str = "DATA_NOT_AVAILABLE"
    book_ev: float | None = None
    book_source: str | None = None
    book_fallback_used: bool = False
    book_sources_skipped: str = ""
    line_can_push: bool = True
    edge_letter_grade: str | None = None
    confidence_tier: str | None = None
    over_under_meter: float | None = None
    preferred_side: str | None = None
    is_preferred_side: bool = False
    pickem_line: float | None = None
    pickem_source: str | None = None
    pickem_line_diff: float | None = None
    # How many prior games this player had before the row's date. Exported so a
    # reader can see WHY a row abstained on cold start rather than taking the
    # reason on faith.
    prior_games: int | None = None
    decision_status: DecisionStatus = "ABSTAIN"
    decision_basis: DecisionBasis = "unavailable"
    rank_score: float | None = None
    why: str = "DATA_NOT_AVAILABLE"
    placement_mode: str = PLACEMENT_MODE
    research_status: str = RESEARCH_STATUS
    timezone_display: str = DISPLAY_TZ_NAME
    disclaimer: str = BOARD_DISCLAIMER
    warnings: list[str] = Field(default_factory=list)
def _side_ev(row: ResearchSlateRow, side: str) -> float | None:
    return row.book_ev_over if side == "over" else row.book_ev_under


def _side_odds(row: ResearchSlateRow, side: str) -> int | None:
    return row.book_over_american if side == "over" else row.book_under_american


def _side_grade(row: ResearchSlateRow, side: str) -> str | None:
    if side == "over":
        return row.edge_letter_grade
    return row.edge_letter_grade_under or row.edge_letter_grade


def expand_row_to_candidates(
    row: ResearchSlateRow,
    *,
    min_ev: float = 0.0,
    min_lean: float = DEFAULT_MIN_LEAN,
    require_valid_book: bool = False,
) -> list[BettingDecisionCandidate]:
    """
    Turn one market row into Over + Under decision candidates. Always both.

    A board that shows only the side the model likes has already made the
    choice it claims to be leaving to you, and hides the price on the other
    side — which is where the edge usually is when the model is wrong.
    """
    line = row.book_line if row.book_line is not None else row.research_line
    po, pu, pp, resolve_warn = resolve_two_way_model_probs(
        p_over=row.model_p_over,
        p_under=row.model_p_under,
        p_push=row.model_p_push,
        line=line,
    )

    out: list[BettingDecisionCandidate] = []
    for side in ("over", "under"):
        warnings = list(row.warnings)
        if resolve_warn:
            warnings.append(resolve_warn)

        p_side = model_prob_for_side(
            bet_side=side,  # type: ignore[arg-type]
            p_over=po, p_under=pu, p_push=pp, line=line,
        )
        ev = _side_ev(row, side)
        odds = _side_odds(row, side)
        status: DecisionStatus = "ABSTAIN"
        basis: DecisionBasis = "unavailable"
        score: float | None = None
        why = "DATA_NOT_AVAILABLE"

        book_ok = row.book_status == "VALID" and odds is not None and ev is not None

        if book_ok and p_side is not None:
            basis = "book_ev"
            score = float(ev)
            source = row.book_source or "book"
            if float(ev) > float(min_ev):
                status = "RECOMMENDED"
                why = (
                    f"{side.upper()} EV={float(ev):+.4f} vs line {line} @ {odds} "
                    f"(model P={p_side:.3f}, {source} two-way de-vigged)"
                )
            else:
                why = (
                    f"{side.upper()} EV={float(ev):+.4f} does not clear "
                    f"min_ev={min_ev}; pass, or take it only if you disagree"
                )
        elif require_valid_book:
            why = (
                "ABSTAIN: --require-valid-book and no VALID two-way price "
                "(PropLine is the only source)"
            )
            if p_side is None:
                warnings.append(f"P({side}) unavailable")
        elif p_side is not None:
            # Model lean only — no EV claim without VALID two-way odds.
            basis = "model_lean"
            lean = float(p_side) - 0.5
            score = lean
            if lean > float(min_lean):
                status = "RECOMMENDED"
                why = (
                    f"{side.upper()} on the model alone, P={p_side:.3f} — no VALID "
                    "two-way price, so nothing says what this costs and the "
                    "recommendation rests entirely on the model's calibration"
                )
            else:
                why = (
                    f"{side.upper()} model lean weak (P={p_side:.3f}); "
                    "no VALID book EV"
                )
            meter = row.over_under_meter
            if meter is not None and (
                (side == "over" and meter < 0) or (side == "under" and meter > 0)
            ):
                warnings.append("Meter leans opposite side")
            if row.pickem_line is not None:
                warnings.append(
                    f"Pick'em {row.pickem_source or 'source'} line "
                    f"{row.pickem_line} — not a substitute for VALID two-way EV"
                )
        elif row.pickem_line is not None:
            # A pick'em line is all we have. It is a number, not a price, so
            # it can never justify a recommendation on its own.
            basis = "pickem_line_only"
            why = (
                f"Pick'em {row.pickem_source or 'source'} line {row.pickem_line}: "
                "a payout multiplier is not a two-way price and cannot be "
                "de-vigged, so EV is undefined — line research only"
            )
        else:
            why = f"DATA_NOT_AVAILABLE: cannot resolve P({side})"

        # COLD START OVERRIDES EVERYTHING ABOVE, and it is applied last on
        # purpose. A thin-history row can clear min_ev on a real price and still
        # be a probability the model had almost nothing to learn from, so this
        # cannot sit inside the book_ev branch — it has to be able to take a
        # RECOMMENDED row back.
        #
        # The abstention is specific to RECOMMENDING. The prediction is still
        # recorded and still graded: a thin-history row is the evidence where
        # the model is weakest, and dropping it from the ledger would bias the
        # calibration toward the easy cases.
        # Carried onto the candidate independently of how the row was built.
        # research_slate_from_predictions also folds these into row.warnings,
        # but a row constructed any other way would then lose them, and the
        # reason a board row abstained should not depend on its provenance.
        for warning in row.eligibility_warnings:
            if warning not in warnings:
                warnings.append(warning)

        if row.eligibility_warnings and status == "RECOMMENDED":
            status = "ABSTAIN"
            why = (
                "ABSTAIN on cold start: "
                + "; ".join(row.eligibility_warnings)
                + f". The model's basis for this row was {basis}, and that "
                "number stands — what is withheld is the recommendation."
            )

        out.append(
            BettingDecisionCandidate(
                slate_date=row.slate_date,
                game_date=row.game_date,
                event_id=row.event_id,
                player_id=row.player_id,
                player_name=row.player_name,
                player_team=row.player_team,
                opponent=row.opponent,
                target_market=row.target_market,
                side=side,  # type: ignore[arg-type]
                line=line,
                model_prob=p_side,
                model_p_over=po,
                model_p_under=pu,
                model_p_push=pp,
                prediction_mean=row.prediction_mean,
                prediction_std=row.prediction_std,
                american_odds=odds,
                book_status=row.book_status,
                book_ev=ev,
                book_source=row.book_source,
                book_fallback_used=row.book_fallback_used,
                book_sources_skipped=row.book_sources_skipped,
                line_can_push=line_can_push(line),
                edge_letter_grade=_side_grade(row, side),
                confidence_tier=row.confidence_tier,
                over_under_meter=row.over_under_meter,
                prior_games=row.prior_games,
                preferred_side=row.preferred_side,
                is_preferred_side=row.preferred_side == side,
                pickem_line=row.pickem_line,
                pickem_source=row.pickem_source,
                pickem_line_diff=row.pickem_line_diff,
                decision_status=status,
                decision_basis=basis,
                rank_score=score,
                why=_assert_no_claims(why),
                warnings=warnings,
            )
        )
    return out


#: How old the slate's market data may be before the board refuses outright.
#: An hour, following the rule this came from. Overridable, and None disables
#: the gate -- but see `build_decision_board` for why a filtered subset is the
#: wrong alternative.
DEFAULT_MAX_MARKET_AGE_MINUTES = 60
ENV_MAX_MARKET_AGE = "PROPIQ_MAX_MARKET_AGE_MINUTES"


def market_age_limit_minutes(explicit: int | None = None) -> int | None:
    """
    The staleness limit: the caller's, else ``$PROPIQ_MAX_MARKET_AGE_MINUTES``,
    else an hour. ``None`` means no gate.

    The env var is read HERE rather than existing only as a name in a log
    line. `tests/test_scheduler_worker` caught that: a documented variable
    nothing reads is a knob that does nothing, which is worse than no knob --
    an operator sets it, sees no change, and concludes the gate is broken.

    Set it to ``0`` or ``off`` to disable the gate deliberately.
    """
    if explicit is not None:
        return explicit
    raw = (os.environ.get(ENV_MAX_MARKET_AGE) or "").strip().lower()
    if not raw:
        return DEFAULT_MAX_MARKET_AGE_MINUTES
    if raw in ("0", "off", "none", "disabled"):
        return None
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        logger.warning(
            "$%s is %r, which is not a number of minutes — using the %d-minute "
            "default rather than guessing.",
            ENV_MAX_MARKET_AGE, raw, DEFAULT_MAX_MARKET_AGE_MINUTES,
        )
        return DEFAULT_MAX_MARKET_AGE_MINUTES
    return value if value > 0 else None


def _recommended_key(cand: "BettingDecisionCandidate") -> tuple[str, str]:
    """What counts as "the same opinion" for the one-pick rule."""
    return (str(cand.player_id or cand.player_name or ""), str(cand.target_market))


def one_recommendation_per_player_market(
    cands: list["BettingDecisionCandidate"],
) -> tuple[list["BettingDecisionCandidate"], int]:
    """
    Keep the best RECOMMENDED candidate per (player, market); abstain on the rest.

    TWO FLAGGED SIDES ON ONE PLAYER AND MARKET IS TWO OPINIONS WHERE ONLY ONE
    CAN BE RIGHT. Books post alternate ladders -- the same player can carry
    Over 24.5 and Over 27.5, and `normalize_event_props` pairs on the point
    precisely so both survive as distinct rows -- so a board could recommend
    Over 24.5 AND Under 27.5 on one player: contradictory advice, presented
    as two independent findings, and a reader taking both is hedged into the
    hold.

    The loser ABSTAINS, it is not deleted. Its numbers are still the honest
    result for that line and belong on the board; what it loses is the flag.
    `cands` must already be sorted, so the first of a pair is the better one
    by the board's own ranking.

    ABSTAIN rather than a new third status: `DecisionStatus` is
    ``Literal["RECOMMENDED", "ABSTAIN"]`` and inventing a "CONSIDER" here
    would put a value outside that type on the model -- which pydantic would
    not catch without validate_assignment, and which every consumer
    branching on the two legal values would then read as a surprise.
    """
    seen: set[tuple[str, str]] = set()
    demoted = 0
    for cand in cands:
        if cand.decision_status != "RECOMMENDED":
            continue
        key = _recommended_key(cand)
        if key in seen:
            cand.decision_status = "ABSTAIN"
            note = (
                "ABSTAINED: another line on this player and market is already "
                "recommended, and two flagged sides of one player/market are "
                "two opinions where only one can be right."
            )
            cand.warnings = [*(cand.warnings or []), note]
            demoted += 1
            continue
        seen.add(key)
    return cands, demoted


def market_age_minutes(market_as_of: Any, now: Any = None) -> float | None:
    """Minutes between the slate's market observation and ``now``, or None."""
    if market_as_of is None:
        return None
    try:
        observed = pd.Timestamp(market_as_of)
    except (TypeError, ValueError):
        return None
    if observed is pd.NaT:
        return None
    current = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
    if observed.tzinfo is None and current.tzinfo is not None:
        observed = observed.tz_localize("UTC")
    elif observed.tzinfo is not None and current.tzinfo is None:
        current = current.tz_localize("UTC")
    return float((current - observed).total_seconds()) / 60.0


def build_decision_board(
    rows: Sequence[ResearchSlateRow],
    *,
    min_ev: float = 0.0,
    min_lean: float = DEFAULT_MIN_LEAN,
    require_valid_book: bool = False,
    consider_only: bool = False,
    top_n: int | None = None,
    market_as_of: Any = None,
    max_market_age_minutes: int | None = None,
) -> list[BettingDecisionCandidate]:
    """
    Rank Over/Under candidates, recommended ones first.

    Sort: RECOMMENDED first, then BASIS BAND, then score, then the book's
    preferred side as a tie-break.

    The band sits between status and score deliberately. Sorting a model
    lean and a de-vigged EV on one numeric scale would imply the two numbers
    are comparable; they are not, because only one of them has a price
    behind it. A 0.49 lean must not outrank a 0.08 edge.

    TWO REFUSALS, both of which can only shrink the board:

    * A STALE SLATE YIELDS ZERO CANDIDATES, not a filtered subset. Pass
      ``market_as_of`` (the slate's market observation time) and
      ``max_market_age_minutes``. A stale row that still clears the bar is the
      dangerous case: it looks actionable and is not. Filtering to the rows
      that happen to look fresh is worse than refusing, because it leaves a
      short, confident-looking board built on a market nobody can vouch for.
      Omit either argument and the gate does not run -- it cannot invent an
      observation time it was not given, and guessing `now()` would make every
      slate look fresh.

    * ONE RECOMMENDATION PER PLAYER AND MARKET. See
      `one_recommendation_per_player_market`; the losers are demoted to
      ABSTAIN, not dropped.

    Both rules come from `Kevocado/NBA_Predictor`'s `odds/edge_gate.py`,
    read 2026-10-18. Its third rule -- an edge must be same book, same point,
    both sides -- this project already satisfies: `normalize_event_props`
    pairs on ``(book, market, player, point)``.
    """
    # THE GATE NEEDS BOTH A LIMIT AND AN OBSERVATION TIME. The limit has a
    # default; the timestamp cannot, so a caller that passes no
    # `market_as_of` is unaffected -- which is every caller today, so this
    # changes nothing until one is wired.
    age_limit = market_age_limit_minutes(max_market_age_minutes)
    if max_market_age_minutes is not None and market_as_of is None:
        # The caller asked for a check and gave nothing to check against.
        # Worth saying, where the same situation under the ENV DEFAULT is
        # just "not wired yet" and warning on it would be noise on every
        # call.
        logger.warning(
            "Market staleness NOT CHECKED: a limit of %s minute(s) was "
            "passed but no market_as_of came with it, so the board is built "
            "unguarded. This is not a pass.",
            max_market_age_minutes,
        )
    if age_limit is not None and market_as_of is not None:
        age = market_age_minutes(market_as_of)
        if age is None:
            logger.warning(
                "Market staleness NOT CHECKED: a limit of %s minute(s) was "
                "set but no usable market_as_of was given, so the board is "
                "built unguarded. This is not a pass.",
                age_limit,
            )
        elif age > float(age_limit):
            logger.error(
                "STALE SLATE — ZERO CANDIDATES: the market data is %.0f "
                "minute(s) old, over the %s-minute limit ($%s). Returning an "
                "empty board rather than the subset that happens to look "
                "fresh: a stale row clearing the bar looks actionable and is "
                "not.",
                age, age_limit, ENV_MAX_MARKET_AGE,
            )
            return []

    cands: list[BettingDecisionCandidate] = []
    for row in rows:
        cands.extend(expand_row_to_candidates(
            row, min_ev=min_ev, min_lean=min_lean,
            require_valid_book=require_valid_book,
        ))
    if consider_only:
        cands = [c for c in cands if c.decision_status == "RECOMMENDED"]

    def _sort_key(c: BettingDecisionCandidate):
        return (
            0 if c.decision_status == "RECOMMENDED" else 1,
            BASIS_ORDER.get(c.decision_basis, len(BASIS_ORDER)),
            -(float(c.rank_score) if c.rank_score is not None else float("-inf")),
            0 if c.is_preferred_side else 1,
            c.player_name or "",
            c.target_market,
            c.side,
        )

    cands.sort(key=_sort_key)

    # AFTER the sort, so "the first of a pair" means "the better one" by the
    # board's own ranking rather than by input order.
    cands, abstained = one_recommendation_per_player_market(cands)
    if abstained:
        logger.info(
            "Abstained on %d candidate(s): another line on the same player "
            "and market was already recommended.", abstained,
        )

    if top_n is not None and top_n > 0:
        cands = cands[: int(top_n)]
    for i, c in enumerate(cands, start=1):
        c.rank = i
    return cands


def decision_board_summary(cands: Sequence[BettingDecisionCandidate]) -> dict[str, Any]:
    consider = [c for c in cands if c.decision_status == "RECOMMENDED"]
    with_ev = [c for c in consider if c.decision_basis == "book_ev"]
    lean_only = [c for c in consider if c.decision_basis == "model_lean"]
    priced = [c for c in cands if c.decision_basis == "book_ev"]
    return {
        "placement_mode": PLACEMENT_MODE,
        "research_status": RESEARCH_STATUS,
        "disclaimer": BOARD_DISCLAIMER,
        "timezone_display": DISPLAY_TZ_NAME,
        "n_candidates": len(cands),
        "n_consider": len(consider),
        "n_abstain": len(cands) - len(consider),
        "n_consider_with_book_ev": len(with_ev),
        "n_consider_model_lean_only": len(lean_only),
        "sources_used": sorted({c.book_source for c in priced if c.book_source}),
        "n_fallback_rows": sum(1 for c in cands if c.book_fallback_used),
        "top": [
            {
                "rank": c.rank,
                "player": c.player_name or c.player_id,
                "market": c.target_market,
                "side": c.side,
                "line": c.line,
                "ev": c.book_ev,
                "model_prob": c.model_prob,
                "odds": c.american_odds,
                "source": c.book_source,
                "status": c.decision_status,
                "basis": c.decision_basis,
                "why": c.why,
            }
            for c in cands[:15]
        ],
    }


def board_to_dataframe(cands: Sequence[BettingDecisionCandidate]) -> pd.DataFrame:
    frame = pd.DataFrame([c.model_dump() for c in cands])
    if "warnings" in frame.columns:
        frame["warnings"] = frame["warnings"].apply(
            lambda w: "|".join(w) if isinstance(w, list) else w
        )
    return frame


def write_decision_board_csv(cands: Sequence[BettingDecisionCandidate], path: Any) -> int:
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    frame = board_to_dataframe(cands)
    frame.to_csv(p, index=False)
    return len(frame)


def candidate_to_manual_bet_fields(c: BettingDecisionCandidate) -> dict[str, Any]:
    """
    Map a RECOMMENDED candidate into fields for ``log-manual-bet``.

    ``model_prob`` is P(THE SIDE YOU TOOK) — the same quantity the store
    records and the calibration reads, so nothing downstream has to
    reconstruct it with a complement that is wrong on a whole line.

    NO STAKE IS RETURNED HERE, and that is not the old "we do not size"
    restriction — ``quant.advisory_sizing`` will recommend a size. It is that
    this function feeds the LEDGER, which must record what you actually staked.
    Writing a recommended size into it would grade the recommendation against a
    bet that was never placed at that size.
    """
    if c.decision_status != "RECOMMENDED":
        return {
            "status": "ABSTAIN",
            "reason": c.why,
            "placement_mode": PLACEMENT_MODE,
        }
    if c.model_prob is None:
        return {
            "status": "DATA_NOT_AVAILABLE",
            "reason": f"model_prob missing for the {c.side} side",
            "placement_mode": PLACEMENT_MODE,
        }
    return {
        "status": "READY_TO_LOG_AFTER_YOU_BET",
        "placement_mode": PLACEMENT_MODE,
        "game_id": c.event_id,
        "player_id": c.player_id,
        "player_name": c.player_name,
        "prop_stat": c.target_market,
        "line": c.line,
        "side": c.side,
        "model_prob": c.model_prob,
        "suggested_odds_american": c.american_odds,
        "book_ev": c.book_ev,
        "book_source": c.book_source,
        "edge_letter_grade": c.edge_letter_grade,
        "confidence_tier": c.confidence_tier,
        "note": (
            "unit_stake is what YOU actually staked. For a recommended size see "
            "quant.advisory_sizing; this field is the ledger's record, not the "
            "recommendation."
        ),
        "disclaimer": BOARD_DISCLAIMER,
    }
