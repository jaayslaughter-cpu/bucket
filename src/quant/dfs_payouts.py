"""DFS pick'em payout structures -> breakeven and expected value.

RESEARCH_ONLY. Nothing here places a wager or sizes a stake.

WHY THIS IS A SEPARATE CONCEPT FROM ``ev_engine``, AND MUST STAY ONE.
``contracts.market_ev_gate`` refuses a pick'em board unconditionally:

    "Pick'em board: a payout multiplier is not a two-way price and cannot be
     de-vigged, so EV is undefined here"

That is correct and this module does NOT go around it. A sportsbook quotes two
prices, the pair carries the hold, and de-vigging recovers a market consensus
probability to measure a model against. A DFS pick'em quotes ONE fixed payout
and no opposing price, so there is no consensus to recover and no vig to
remove. ``market_ev_gate`` is right that two-way EV is undefined here.

What IS defined is different, and is what this module computes: a fixed payout
implies an exact BREAKEVEN probability, and comparing a model's own probability
to that threshold is arithmetic on a known payout. A 2-pick play returning
3.0x needs a joint probability above 1/3.

THE DISTINCTION THAT MATTERS, stated plainly because it is easy to lose:

    two-way EV      measured against the market's own de-vigged opinion. The
                    market is evidence; a disagreement is an edge claim about
                    a price.
    payout EV here  measured against a fixed threshold. There is NO market
                    opinion in it. Its validity rests ENTIRELY on the model's
                    calibration -- if the model's probabilities are 5 points
                    optimistic, every number this module returns is wrong by
                    about that much, and nothing in the payout can reveal it.

So a positive figure from this module is not evidence of profitability. It is
the model's own claim, restated in payout terms. Read it beside the
calibration diagnostics (``prob_calibration``, ``paper_calibration``, the
gated ECE in ``compare``), never instead of them.

NO PAYOUT TABLE IS SHIPPED. Multipliers differ by platform, pick count, market
and promotion, and they change. Hardcoding a table would be inventing
sportsbook data, so the caller supplies the structure from a source they can
point at. There is deliberately no default.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Status vocabulary, deliberately distinct from GATE_READY/GATE_ABSTAIN so a
# payout-implied figure can never be mistaken downstream for a two-way EV that
# cleared market_ev_gate.
PAYOUT_EV_READY = "PAYOUT_EV_READY"
PAYOUT_EV_ABSTAIN = "DATA_NOT_AVAILABLE"

DISCLAIMER = (
    "Payout-implied EV, not market EV: a DFS pick'em carries no opposing price "
    "to de-vig, so this figure rests entirely on the model's own calibration "
    "and is not evidence of profit."
)


class DfsPayoutError(ValueError):
    """A payout structure that does not describe a real product."""


@dataclass(frozen=True)
class DfsPayoutStructure:
    """
    One platform's payout for one pick count.

    ``payouts`` maps HITS -> total returned per 1 staked, gross (so an
    all-or-nothing 2-pick paying 3x is ``{2: 3.0}``, and a miss is simply
    absent rather than written as 0.0 -- an absent key and a 0.0 payout mean
    the same thing here, and requiring the zeros would invite typos).

    ``source`` is required and free-text: where the numbers came from. A
    structure with no provenance is indistinguishable from an invented one.
    """

    n_picks: int
    payouts: dict[int, float]
    source: str
    label: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.n_picks, int) or self.n_picks < 2:
            raise DfsPayoutError(
                f"n_picks must be an integer of at least 2, got {self.n_picks!r}"
            )
        if not self.payouts:
            raise DfsPayoutError("payouts is empty, so this structure pays nothing")
        if not str(self.source).strip():
            raise DfsPayoutError(
                "source is required: a payout table with no provenance cannot be "
                "distinguished from an invented one"
            )
        for hits, multiple in self.payouts.items():
            if not isinstance(hits, int) or not 0 <= hits <= self.n_picks:
                raise DfsPayoutError(
                    f"payout key {hits!r} is not a hit count in 0..{self.n_picks}"
                )
            if not (isinstance(multiple, (int, float)) and math.isfinite(multiple)):
                raise DfsPayoutError(f"payout for {hits} hits is not a number: {multiple!r}")
            if multiple < 0:
                raise DfsPayoutError(
                    f"payout for {hits} hits is {multiple}, and a negative return "
                    "is not a payout"
                )
        if self.n_picks not in self.payouts:
            raise DfsPayoutError(
                f"no payout for a perfect {self.n_picks}-of-{self.n_picks}; a "
                "structure that does not pay when every pick lands is not a "
                "product, it is a transcription error"
            )

    @property
    def is_all_or_nothing(self) -> bool:
        """True when only a perfect card pays — a 'power play'."""
        return set(self.payouts) == {self.n_picks}

    def payout_for(self, hits: int) -> float:
        """Gross return per 1 staked at this hit count. Absent means nothing back."""
        return float(self.payouts.get(int(hits), 0.0))

    def breakeven_joint_probability(self) -> float | None:
        """
        The joint probability a perfect card needs to break even.

        Defined only for an all-or-nothing structure: with partial payouts the
        breakeven is a surface over the whole count distribution, not a single
        probability, so returning one number would be a category error rather
        than an approximation.
        """
        if not self.is_all_or_nothing:
            return None
        multiple = self.payout_for(self.n_picks)
        if multiple <= 0:
            return None
        return 1.0 / multiple


@dataclass
class PayoutEvaluation:
    """What a payout structure is worth against a model's count distribution."""

    status: str
    n_picks: int
    structure_label: str
    structure_source: str
    expected_value: float | None = None
    expected_value_stderr: float | None = None
    probability_all_hit: float | None = None
    breakeven_joint_probability: float | None = None
    edge_vs_breakeven: float | None = None
    count_probabilities: list[float] = field(default_factory=list)
    reason: str | None = None
    disclaimer: str = DISCLAIMER

    def as_dict(self) -> dict[str, Any]:
        return {
            "STATUS": self.status,
            "N_PICKS": self.n_picks,
            "STRUCTURE": self.structure_label,
            "STRUCTURE_SOURCE": self.structure_source,
            "PAYOUT_EV": self.expected_value,
            "PAYOUT_EV_STDERR": self.expected_value_stderr,
            "P_ALL_HIT": self.probability_all_hit,
            "BREAKEVEN_JOINT_PROB": self.breakeven_joint_probability,
            "EDGE_VS_BREAKEVEN": self.edge_vs_breakeven,
            "REASON": self.reason,
            "DISCLAIMER": self.disclaimer,
        }


def _abstain(structure: DfsPayoutStructure, reason: str) -> PayoutEvaluation:
    return PayoutEvaluation(
        status=PAYOUT_EV_ABSTAIN,
        n_picks=structure.n_picks,
        structure_label=structure.label,
        structure_source=structure.source,
        reason=reason,
    )


def evaluate_payout(
    structure: DfsPayoutStructure,
    count_probabilities: Sequence[float],
    count_stderrs: Sequence[float] | None = None,
) -> PayoutEvaluation:
    """
    EV per 1 staked: ``sum_k P(k hits) * payout(k) - 1``.

    ``count_probabilities`` is indexed by hit count, length ``n_picks + 1`` --
    exactly what ``parlay.hit_count_distribution`` returns. Passing only
    P(all hit) would silently price a flex as a power play, so the full
    distribution is required and its length is checked.

    Abstains rather than guessing when the distribution does not match the
    structure or does not sum to 1.
    """
    probabilities = np.asarray(list(count_probabilities), dtype=float)
    expected_len = structure.n_picks + 1

    if probabilities.size != expected_len:
        return _abstain(
            structure,
            f"count distribution has {probabilities.size} cell(s), expected "
            f"{expected_len} for {structure.n_picks} picks. A distribution of "
            "the wrong length cannot be aligned to hit counts.",
        )
    if not np.all(np.isfinite(probabilities)):
        return _abstain(structure, "count distribution contains a non-finite cell")
    if np.any(probabilities < 0.0):
        return _abstain(structure, "count distribution contains a negative cell")
    total = float(probabilities.sum())
    if not math.isclose(total, 1.0, abs_tol=1e-6):
        return _abstain(
            structure,
            f"count distribution sums to {total:.6f}, not 1. A distribution that "
            "does not sum to 1 is not a distribution, and rescaling it here "
            "would hide whatever produced it.",
        )

    multiples = np.array(
        [structure.payout_for(k) for k in range(expected_len)], dtype=float
    )
    gross = float(np.dot(probabilities, multiples))
    ev = gross - 1.0

    stderr: float | None = None
    if count_stderrs is not None:
        errs = np.asarray(list(count_stderrs), dtype=float)
        if errs.size == expected_len and np.all(np.isfinite(errs)):
            # Cells are negatively dependent (they share the same draws), so
            # summing variances OVERSTATES the spread. Reported anyway, as an
            # upper bound and labelled one, because an unstated Monte Carlo
            # error gets read as zero.
            stderr = float(np.sqrt(np.sum((errs * multiples) ** 2)))

    breakeven = structure.breakeven_joint_probability()
    p_all = float(probabilities[-1])
    edge = (p_all - breakeven) if breakeven is not None else None

    return PayoutEvaluation(
        status=PAYOUT_EV_READY,
        n_picks=structure.n_picks,
        structure_label=structure.label,
        structure_source=structure.source,
        expected_value=ev,
        expected_value_stderr=stderr,
        probability_all_hit=p_all,
        breakeven_joint_probability=breakeven,
        edge_vs_breakeven=edge,
        count_probabilities=[float(x) for x in probabilities],
    )


def structure_from_mapping(payload: dict[str, Any]) -> DfsPayoutStructure:
    """
    Build a structure from config or a user-supplied CSV row.

    Hit-count keys arrive as strings from YAML and JSON, so they are coerced;
    a key that is not an integer is an error rather than a skipped row, because
    silently dropping a payout tier would misprice every ticket using it.
    """
    raw = payload.get("payouts")
    if not isinstance(raw, dict):
        raise DfsPayoutError("payouts must be a mapping of hit count to multiple")
    payouts: dict[int, float] = {}
    for key, value in raw.items():
        try:
            hits = int(key)
        except (TypeError, ValueError) as exc:
            raise DfsPayoutError(f"payout key {key!r} is not a hit count") from exc
        payouts[hits] = float(value)
    return DfsPayoutStructure(
        n_picks=int(payload.get("n_picks", 0) or 0),
        payouts=payouts,
        source=str(payload.get("source", "")),
        label=str(payload.get("label", "")),
    )
