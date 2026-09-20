"""Market contracts and the EV abstention gate.

The gate exists to make silence explicit. Without it, "no EV shown" reads
like "no edge found", when the truth is usually "we were never allowed to
compute one". Every refusal returns a named reason.

EV for a two-way prop requires genuine two-way American odds. A pick'em
board publishes a payout multiplier instead, which is not a price and
cannot be de-vigged, so those rows abstain by design rather than by
accident.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Literal

logger = logging.getLogger(__name__)

MarketStatus = Literal["VALID", "DATA_NOT_AVAILABLE"]

GATE_READY = "READY_FOR_EVALUATION"
GATE_ABSTAIN = "DATA_NOT_AVAILABLE"


@dataclass(frozen=True)
class MarketContext:
    """Everything the gate needs to decide whether EV may be computed."""

    game_id: str
    status: MarketStatus = "DATA_NOT_AVAILABLE"
    market: str | None = None
    player_name: str | None = None
    line: float | None = None
    over_odds_american: int | None = None
    under_odds_american: int | None = None
    payout_multiplier: float | None = None
    is_pickem: bool = False
    source: str | None = None
    captured_at_utc: Any | None = None


def american_to_implied_probability(odds: int) -> float:
    """Implied probability of a single American price, vig included."""
    odds = int(odds)
    if odds == 0:
        raise ValueError("American odds of 0 are not a price")
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return -odds / (-odds + 100.0)


def devig_two_way(over_odds: int, under_odds: int) -> dict[str, float]:
    """
    Remove the vig from a two-way market by normalising both sides.

    Returns the fair probabilities and the hold. This is the simple
    multiplicative method; it assumes the vig is spread proportionally,
    which is standard but not universally true.
    """
    p_over_raw = american_to_implied_probability(over_odds)
    p_under_raw = american_to_implied_probability(under_odds)
    booked = p_over_raw + p_under_raw
    if booked <= 0:
        raise ValueError("Degenerate two-way market")
    return {
        "fair_probability_over": p_over_raw / booked,
        "fair_probability_under": p_under_raw / booked,
        "hold": booked - 1.0,
    }


def market_ev_gate(context: MarketContext) -> dict[str, Any]:
    """
    Decide whether EV may be computed for this market.

    Returns ``status`` of READY_FOR_EVALUATION or DATA_NOT_AVAILABLE, and
    ``ev`` which is always None here — the gate grants permission, it does
    not price anything. A caller that needs EV computes it only after
    seeing READY_FOR_EVALUATION.
    """
    verdict: dict[str, Any] = {
        "status": GATE_ABSTAIN,
        "ev": None,
        "game_id": context.game_id,
        "market": context.market,
        "reason": None,
        "fair_probability_over": None,
        "fair_probability_under": None,
        "hold": None,
    }

    if context.status != "VALID":
        verdict["reason"] = f"Market status is {context.status}, not VALID"
        return verdict

    # Pick'em first, and unconditionally. Checking odds first would let a
    # pick'em row that happens to carry two odds fields through the gate,
    # and a payout multiplier is not a two-way price however it is labelled.
    if context.is_pickem or context.payout_multiplier is not None:
        verdict["reason"] = (
            "Pick'em board: a payout multiplier is not a two-way price and "
            "cannot be de-vigged, so EV is undefined here"
        )
        return verdict

    if context.over_odds_american is None or context.under_odds_american is None:
        verdict["reason"] = "Two-way American odds required; one or both sides are missing"
        return verdict

    # EV is a claim about a probability at a specific number. Without the
    # line there is nothing for the probability to be "at".
    if context.line is None or not math.isfinite(float(context.line)):
        verdict["reason"] = "A finite posted line is required before EV can be evaluated"
        return verdict

    try:
        fair = devig_two_way(context.over_odds_american, context.under_odds_american)
    except (ValueError, TypeError, OverflowError, ArithmeticError) as exc:
        verdict["reason"] = f"Could not de-vig: {exc}"
        return verdict

    verdict.update(
        status=GATE_READY,
        reason="Two-way American odds present and de-vigged",
        **fair,
    )
    return verdict
