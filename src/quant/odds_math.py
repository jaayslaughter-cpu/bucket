"""
src/quant/odds_math.py — odds conversions and de-vigging.

Written for this repository because waves 3-5 import it and no pack ever
contained it. It is deliberately a THIN layer over
``src.quant.contracts``: the American-odds conversion and the
multiplicative de-vig already live there and are already tested, so this
module returns them in the shape those waves expect rather than
reimplementing the arithmetic. Two copies of a de-vig would drift, and
the one that drifted would be the one nobody was reading.

RESEARCH_ONLY. Converting a price into a probability is not a
recommendation to take it.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from src.quant.contracts import american_to_implied_probability, devig_two_way

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DevigResult:
    """Fair two-way probabilities with the vig removed."""

    fair_prob_a: float
    fair_prob_b: float
    hold: float
    implied_prob_a: float
    implied_prob_b: float
    method: str = "multiplicative"

    def as_dict(self) -> dict[str, float | str]:
        return {
            "fair_prob_a": round(self.fair_prob_a, 6),
            "fair_prob_b": round(self.fair_prob_b, 6),
            "hold": round(self.hold, 6),
            "implied_prob_a": round(self.implied_prob_a, 6),
            "implied_prob_b": round(self.implied_prob_b, 6),
            "method": self.method,
        }


def american_to_decimal(american: int) -> float:
    """American price to decimal odds (stake included)."""
    odds = int(american)
    if odds == 0:
        raise ValueError("American odds of 0 are not a price")
    return 1.0 + (odds / 100.0 if odds > 0 else 100.0 / -odds)


def american_to_profit_multiple(american: int) -> float:
    """
    Net profit per unit staked — decimal odds MINUS the returned stake.

    This is the number EV and Kelly need. Using decimal odds where net
    profit belongs overstates the payout by exactly one unit, which on a
    +150 price is a 67% overstatement of the win branch. A reference
    implementation reviewed earlier in this project made precisely that
    substitution, so it is spelled out here rather than left implicit.
    """
    return american_to_decimal(american) - 1.0


def decimal_to_american(decimal_odds: float) -> int:
    """Decimal odds back to the nearest American price."""
    d = float(decimal_odds)
    if d <= 1.0:
        raise ValueError("Decimal odds must exceed 1.0")
    return round((d - 1.0) * 100.0) if d >= 2.0 else round(-100.0 / (d - 1.0))


def probability_to_american(probability: float) -> int:
    """Fair American price implied by a probability."""
    p = float(probability)
    if not 0.0 < p < 1.0:
        raise ValueError("Probability must be strictly between 0 and 1")
    return decimal_to_american(1.0 / p)


def multiplicative_devig(american_a: int, american_b: int) -> DevigResult:
    """
    Remove the vig by normalising both sides to sum to one.

    Delegates the arithmetic to ``contracts.devig_two_way`` so there is
    exactly one de-vig in this codebase.

    The multiplicative method assumes the vig is spread proportionally
    across both sides. That is the standard assumption and it is not
    always true — books often load the side the public prefers — so a
    favourite-longshot skew will not be corrected here.
    """
    fair = devig_two_way(int(american_a), int(american_b))
    return DevigResult(
        fair_prob_a=fair["fair_probability_over"],
        fair_prob_b=fair["fair_probability_under"],
        hold=fair["hold"],
        implied_prob_a=american_to_implied_probability(int(american_a)),
        implied_prob_b=american_to_implied_probability(int(american_b)),
    )


def expected_value_per_unit(probability: float, american: int) -> float:
    """
    EV per unit staked at a price, given a win probability.

    EV = p * net_profit - (1 - p) * 1. Note the loss branch is the stake
    itself, and the win branch is NET profit, not the decimal return.
    """
    p = float(probability)
    if not 0.0 <= p <= 1.0 or not math.isfinite(p):
        raise ValueError(f"Probability out of range: {probability}")
    return p * american_to_profit_multiple(int(american)) - (1.0 - p)


def breakeven_probability(american: int) -> float:
    """The win rate at which a price is exactly break-even."""
    return 1.0 / american_to_decimal(int(american))
