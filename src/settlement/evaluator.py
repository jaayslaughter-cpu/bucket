"""
src/settlement/evaluator.py — prop settlement (WIN / LOSS / PUSH / VOID).

SCOPE: NBA only.

WHY Decimal AND NOT float
-------------------------
Being precise about this, because the usual "floats are inexact" hand-wave
doesn't actually apply to the PUSH comparison itself:

For standard NBA props, float comparison would in fact work — stats are
integers and lines are .0 or .5, and `25 == 25.0` is exactly True in
binary float. So Decimal is NOT load-bearing for basic push detection.

Decimal IS load-bearing for two real reasons, both verified:

1. ACCUMULATED P/L DRIFT. Summing a -110 payout (0.909090...) across
   1,000 settled props:
       float  -> 909.0909090909022
       Decimal-> 909.090909090909000
   The float ledger is wrong by ~2e-12 per thousand rows and the error
   compounds. A P/L ledger must not drift, hence NUMERIC in the schema
   and Decimal here.

2. NON-HALF-POINT LINES. Alt/quarter lines (25.25) and any line that
   arrives via a computed value rather than a literal can land off by an
   ULP. `0.1 + 0.2 != 0.3` is the canonical case. The engine should not
   silently mis-grade those if the market ever offers them.

Every line and result is coerced via `Decimal(str(x))` — never
`Decimal(float)`, which would inherit the float's error before the
conversion could help.

PUSH IS ONLY POSSIBLE ON WHOLE-NUMBER LINES
-------------------------------------------
A 25.5 line cannot tie — points are integers. Half-point lines exist
precisely to eliminate the push. The engine therefore refuses to emit a
PUSH against a fractional line (it raises, rather than silently
producing an impossible row that the DB CHECK would reject anyway).

STAT COMBINATION
----------------
Combined markets (PRA, PR, PA, RA) are summed from their real component
box-score fields, never approximated.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import StrEnum

from src.quant.composite_markets import (
    FPTS,
    SUM_COMPOSITES,
    CompositeMarketError,
    fantasy_weights,
)
from typing import Any

logger = logging.getLogger(__name__)


class Outcome(StrEnum):
    WIN = "WIN"
    LOSS = "LOSS"
    PUSH = "PUSH"
    PENDING = "PENDING"
    VOID = "VOID"


class Side(StrEnum):
    OVER = "OVER"
    UNDER = "UNDER"


#: PANEL STAT NAME -> the box-score field settlement actually reads. The
#: panel, `espn_game` and `quant.composite_markets` all speak in "PTS"; the
#: CDN's own field names are what `extract_player_stats` emits and what this
#: module grades from. The translation lives here, once, because this module
#: is what defines the box-score vocabulary -- `espn_boxscore.PANEL_TO_CDN`
#: builds on this rather than restating the same seven pairs.
PANEL_TO_BOXSCORE: dict[str, str] = {
    "PTS":  "points",
    "REB":  "reboundsTotal",
    "AST":  "assists",
    "FG3M": "threePointersMade",
    "STL":  "steals",
    "BLK":  "blocks",
    "TOV":  "turnovers",
}

# Canonical NBA prop markets -> the box-score fields they sum.
# Keys match the `market` column; values are keys produced by
# src.settlement.boxscore_fetcher.extract_player_stats().
MARKET_COMPONENTS: dict[str, tuple[str, ...]] = {
    # Single-stat markets: the market IS its component.
    "PTS":  ("points",),
    "REB":  ("reboundsTotal",),
    "AST":  ("assists",),
    "FG3M": ("threePointersMade",),
    "STL":  ("steals",),
    "BLK":  ("blocks",),
    "TOV":  ("turnovers",),
    "STOCKS": ("steals", "blocks"),
}

# THE COMPOSITES ARE DERIVED, NOT RESTATED. PRA was spelled out here AND in
# `features.builder` and the two definitions "agreed on every row of the
# archive panel", so the divergence was invisible on the only data available.
# `quant.composite_markets.SUM_COMPOSITES` is now the single decomposition and
# this reads it, so PA/PR/RA/PRA cannot be summed one way for the projection
# and another way for the grade. A test used to assert the two tables matched;
# agreeing by construction is strictly better than agreeing by assertion.
for _market, _panel_components in SUM_COMPOSITES.items():
    MARKET_COMPONENTS[_market] = tuple(
        PANEL_TO_BOXSCORE[_name] for _name in _panel_components
    )
del _market, _panel_components


class SettlementError(ValueError):
    """Raised when a prop cannot be graded safely."""


@dataclass(frozen=True)
class Settlement:
    outcome: Outcome
    actual_result: Decimal | None
    stake_units: Decimal | None
    profit_units: Decimal | None
    note: str


def _to_decimal(value: Any, field: str) -> Decimal:
    """Coerce to Decimal via str — Decimal(float) inherits the float's error."""
    if value is None:
        raise SettlementError(f"{field} is None")
    try:
        decimal_value = Decimal(str(value))
    except (InvalidOperation, TypeError) as exc:
        raise SettlementError(f"{field}={value!r} is not numeric") from exc
    # Decimal accepts 'nan' and 'inf' from a string. Left through, a NaN line
    # compares False against everything and grades as a confident LOSS, while
    # an infinite stake poisons the P/L ledger.
    if not decimal_value.is_finite():
        raise SettlementError(f"{field}={value!r} is not finite")
    return decimal_value


def is_whole_number_line(line: Decimal) -> bool:
    """True when a PUSH is arithmetically possible on this line."""
    return line == line.to_integral_value()


def _component(player_stats: dict[str, Any], field: str, market_key: str) -> Decimal:
    """One box-score component, or a refusal. A missing stat is not a zero."""
    if field not in player_stats or player_stats[field] is None:
        raise SettlementError(
            f"Box score missing component {field!r} required for market {market_key}. "
            f"Refusing to grade — a missing stat is not a zero."
        )
    return _to_decimal(player_stats[field], field)


def fantasy_points(
    player_stats: dict[str, Any], *, platform: Any
) -> Decimal:
    """
    The graded fantasy score for one player, on ONE OPERATOR'S terms.

    FPTS COULD NOT BE SETTLED AT ALL before this. `MARKET_COMPONENTS` maps a
    market to fields it SUMS, and a fantasy score is not a sum: it is
    ``1.0*PTS + 1.2*REB + 1.5*AST + 3.0*STL + 3.0*BLK - 1.0*TOV``, weights
    included and one of them negative. So an FPTS prop reached `combine_stat`,
    found no entry and raised "Unknown market" — the projection side was
    built and the ledger could never close the loop on it, which also means no
    FPTS calibration could ever be measured.

    THE PLATFORM IS REQUIRED AND IS NOT DEFAULTED. The multipliers are an
    operator's terms, not a property of basketball;
    `config/fantasy_scoring.yaml` says so and holds one block per operator
    even where they agree today. Grading a PrizePicks entry on Underdog's
    matrix would produce an entirely plausible wrong number, so an unknown
    platform raises instead.

    DECIMAL, with the weights from the one table. `composite_markets` owns
    WHAT the weights are; this owns HOW a graded number is computed, in
    Decimal, because the P/L ledger must not drift. The two arithmetics are
    pinned equal by test.
    """
    if not str(platform or "").strip():
        raise SettlementError(
            "FPTS cannot be graded without a platform: the multipliers are an "
            "operator's terms, not a property of basketball. Pass the book the "
            "line came from. See config/fantasy_scoring.yaml."
        )
    try:
        weights = fantasy_weights(str(platform))
    except CompositeMarketError as exc:
        # Surfaced as a settlement refusal so the runner leaves the prop
        # PENDING with a named reason rather than guessing an operator.
        raise SettlementError(f"FPTS weights unavailable: {exc}") from exc

    total = Decimal(0)
    for panel_name, weight in weights.items():
        field = PANEL_TO_BOXSCORE.get(panel_name)
        if field is None:
            raise SettlementError(
                f"FPTS weight names {panel_name!r}, which is not a box-score "
                f"field settlement reads ({sorted(PANEL_TO_BOXSCORE)})"
            )
        total += _to_decimal(weight, f"FPTS weight for {panel_name}") * _component(
            player_stats, field, FPTS
        )
    return total


def combine_stat(
    player_stats: dict[str, Any], market: str, *, platform: Any = None
) -> Decimal:
    """
    The real box-score value of a market: a sum, or a weighted fantasy score.

    Raises rather than defaulting a missing component to 0 — a missing
    field means the box score was incomplete, and silently treating that
    as zero would mis-grade the prop as a LOSS on the Over.
    """
    market_key = market.upper().strip()
    if market_key == FPTS:
        return fantasy_points(player_stats, platform=platform)

    components = MARKET_COMPONENTS.get(market_key)
    if components is None:
        raise SettlementError(
            f"Unknown market {market!r}. Known: "
            f"{sorted([*MARKET_COMPONENTS, FPTS])}"
        )

    total = Decimal(0)
    for field in components:
        total += _component(player_stats, field, market_key)
    return total


def american_to_profit_per_unit(odds: int) -> Decimal:
    """
    Profit on a 1-unit WIN at American odds. Stake is returned separately,
    so this is profit only (+150 -> 1.50, -110 -> 0.909090...).
    """
    odds_d = _to_decimal(odds, "odds")
    if odds_d == 0:
        raise SettlementError("American odds of 0 are invalid")
    if odds_d > 0:
        return odds_d / Decimal(100)
    return Decimal(100) / abs(odds_d)


def settle_prop(
    *,
    market: str,
    predicted_line: Any,
    predicted_side: str,
    player_stats: dict[str, Any] | None,
    odds: int | None = None,
    did_not_play: bool = False,
    minutes_played: Any = None,
    stake_units: Any = Decimal(1),
    platform: Any = None,
) -> Settlement:
    """
    Grade one prop against real box-score data.

    VOID (not LOSS, not PUSH) when the player did not appear. A scratch
    is not a losing bet — books void these, and counting them as losses
    would understate the model's real strike rate.

    Returns profit_units = None when `odds` is None (pick'em multiplier
    rather than a two-way American price). The row is still graded
    W/L/P — only ROI abstains.

    `platform` is the operator the line came from, and is required ONLY by
    FPTS, whose multipliers are an operator's terms rather than arithmetic.
    Every other market ignores it.
    """
    line = _to_decimal(predicted_line, "predicted_line")
    try:
        side = Side(str(predicted_side).upper().strip())
    except (ValueError, AttributeError, TypeError) as exc:
        raise SettlementError(
            f"predicted_side must be OVER/UNDER, got {predicted_side!r}"
        ) from exc

    if did_not_play or player_stats is None:
        return Settlement(
            outcome=Outcome.VOID,
            actual_result=None,
            stake_units=None,
            profit_units=None,
            note="Player did not play — prop voided, excluded from W-L and ROI.",
        )

    if minutes_played is not None:
        minutes = _to_decimal(minutes_played, "minutes_played")
        if minutes == 0:
            return Settlement(
                outcome=Outcome.VOID,
                actual_result=None,
                stake_units=None,
                profit_units=None,
                note="Zero minutes played — prop voided.",
            )

    actual = combine_stat(player_stats, market, platform=platform)

    # --- PUSH: exact tie. Only reachable on a whole-number line. --------
    if actual == line:
        if not is_whole_number_line(line):
            raise SettlementError(
                f"Computed a PUSH on fractional line {line} (actual={actual}). "
                f"This is arithmetically impossible for an integer stat and "
                f"indicates a data or parsing bug — refusing to record it."
            )
        return Settlement(
            outcome=Outcome.PUSH,
            actual_result=actual,
            stake_units=_to_decimal(stake_units, "stake_units") if odds is not None else None,
            # Push returns the stake: zero profit, not a loss.
            profit_units=Decimal(0) if odds is not None else None,
            note=f"PUSH: {market} {actual} exactly equals line {line}.",
        )

    won = (actual > line) if side is Side.OVER else (actual < line)
    outcome = Outcome.WIN if won else Outcome.LOSS

    if odds is None:
        return Settlement(
            outcome=outcome,
            actual_result=actual,
            stake_units=None,
            profit_units=None,
            note=(
                f"{outcome}: {market} {actual} vs {side} {line}. "
                f"No two-way American odds (pick'em payout multiplier) — "
                f"ROI not computed for this row."
            ),
        )

    stake = _to_decimal(stake_units, "stake_units")
    # A negative stake flips the sign of every outcome — wins book as losses
    # and losses as gains — which corrupts ROI silently rather than loudly.
    if stake <= 0:
        raise SettlementError(f"stake_units must be strictly positive, got {stake}")
    profit = stake * american_to_profit_per_unit(odds) if won else -stake

    return Settlement(
        outcome=outcome,
        actual_result=actual,
        stake_units=stake,
        profit_units=profit,
        note=f"{outcome}: {market} {actual} vs {side} {line} @ {odds:+d}.",
    )


# --------------------------------------------------------------------------
# Closing line value — a MARKET-QUALITY signal, explicitly not profit.
# --------------------------------------------------------------------------

def compute_clv(
    *,
    predicted_line: Any,
    predicted_side: str,
    closing_line: Any | None,
    bet_odds: int | None = None,
    closing_odds: int | None = None,
) -> dict[str, Decimal | None]:
    """
    Two separate CLV measures, never summed together:

    - clv_line_points: how many points better your LINE was than the
      close, signed so positive always favours the bettor (an Over taken
      at 24.5 that closed 26.5 is +2.0; the same number on an Under is
      -2.0).
    - clv_prob_points: implied-probability difference from the PRICE.
      Requires both prices; None otherwise.

    A line move and a price move are different things. Reporting one as
    the other is the most common CLV error, so they stay separate here.
    """
    out: dict[str, Decimal | None] = {"clv_line_points": None, "clv_prob_points": None}

    if closing_line is not None:
        line = _to_decimal(predicted_line, "predicted_line")
        close = _to_decimal(closing_line, "closing_line")
        try:
            side = Side(str(predicted_side).upper().strip())
        except (ValueError, AttributeError, TypeError) as exc:
            raise SettlementError(
                f"predicted_side must be OVER/UNDER, got {predicted_side!r}"
            ) from exc
        # Over wants the line to rise after you take it; Under wants it to fall.
        out["clv_line_points"] = (close - line) if side is Side.OVER else (line - close)

    if bet_odds is not None and closing_odds is not None:
        bet_prob = _american_to_implied_prob(bet_odds)
        close_prob = _american_to_implied_prob(closing_odds)
        # Positive = the market moved toward your side after you bet it.
        out["clv_prob_points"] = close_prob - bet_prob

    return out


def _american_to_implied_prob(odds: int) -> Decimal:
    odds_d = _to_decimal(odds, "odds")
    if odds_d == 0:
        raise SettlementError("American odds of 0 are invalid")
    if odds_d > 0:
        return Decimal(100) / (odds_d + Decimal(100))
    return abs(odds_d) / (abs(odds_d) + Decimal(100))
