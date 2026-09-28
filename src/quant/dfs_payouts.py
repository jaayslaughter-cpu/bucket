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

WHERE THE LEG PROBABILITIES COME FROM DECIDES WHAT THE NUMBER MEANS. This is
the whole of it, and an earlier version of this docstring got it wrong by
asserting that a payout-implied figure "rests entirely on the model's
calibration". That is true of ONE of the two sources below and not the other:

    p from a SHARP BENCHMARK  de-vig a two-way prop at Pinnacle / Circa /
                              FanDuel / DraftKings for the same player, market
                              and line, and p_i is a market consensus. EV
                              against the pick'em payout is then genuine
                              market-grounded EV, and a disagreement is an
                              edge claim about a price. The de-vig happens on
                              the BENCHMARK, never on the pick'em operator --
                              that is the distinction the gate was groping
                              for and stated too broadly.

    p from THIS MODEL         no market opinion enters. The figure is the
                              model's own claim restated in payout terms, and
                              if its probabilities run five points optimistic
                              every number is wrong by about that much with
                              nothing in the payout able to reveal it. Read it
                              beside the calibration diagnostics
                              (``prob_calibration``, ``paper_calibration``,
                              the gated ECE in ``compare``), never instead.

``ProbabilitySource`` records which was used and every evaluation carries it,
because the two are not interchangeable and a consumer cannot tell them apart
from the number alone.

NO PAYOUT TABLE IS SHIPPED. Multipliers differ by platform, pick count, market
and promotion, and they change. Hardcoding a table would be inventing
sportsbook data, so the caller supplies the structure from a source they can
point at. There is deliberately no default.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

import numpy as np

from src.quant.odds_math import multiplicative_devig, probability_to_american

logger = logging.getLogger(__name__)

# Status vocabulary, deliberately distinct from GATE_READY/GATE_ABSTAIN so a
# payout-implied figure can never be mistaken downstream for a two-way EV that
# cleared market_ev_gate.
PAYOUT_EV_READY = "PAYOUT_EV_READY"
PAYOUT_EV_ABSTAIN = "DATA_NOT_AVAILABLE"

class ProbabilitySource(str, Enum):
    """Where each leg's win probability came from. Not cosmetic — see below."""

    SHARP_BENCHMARK = "SHARP_BENCHMARK"
    MODEL = "MODEL"
    MIXED = "MIXED"
    UNSPECIFIED = "UNSPECIFIED"


DISCLAIMER_BY_SOURCE: dict[ProbabilitySource, str] = {
    ProbabilitySource.SHARP_BENCHMARK: (
        "Market-grounded payout EV: leg probabilities were de-vigged from a "
        "two-way sharp benchmark, so this measures the pick'em payout against a "
        "market consensus. Its validity rests on the benchmark being sharp, on "
        "the line matching exactly, and on the snapshot being current."
    ),
    ProbabilitySource.MODEL: (
        "Model-grounded payout EV: no market opinion enters, so this figure "
        "rests entirely on the model's own calibration and is not evidence of "
        "profit. Read it beside the calibration diagnostics."
    ),
    ProbabilitySource.MIXED: (
        "Mixed-source payout EV: some legs came from a sharp benchmark and some "
        "from the model, so the entry is only as market-grounded as its weakest "
        "leg. Prefer a single source per entry."
    ),
    ProbabilitySource.UNSPECIFIED: (
        "Payout EV with an unrecorded probability source. A consumer cannot "
        "tell market-grounded from model-grounded here; record the source."
    ),
}


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

    def per_leg_breakeven_probability(self) -> float | None:
        """
        The per-leg probability an EQUAL-LEG independent power play needs: M^(-1/N).

        Distinct from ``breakeven_joint_probability`` and consistent with it --
        raising this to the Nth power gives 1/M. The joint form is the threshold
        for the whole card; this one is what a downstream rule expecting a
        per-contract price needs.

        ASSUMES equal and independent legs. With unequal probabilities there is
        no single per-leg breakeven (any set whose product exceeds 1/M clears
        it), and with correlated legs the product is not the joint probability
        at all, so this is a reference price rather than a decision input. The
        decision should use ``evaluate_payout`` on the real distribution.
        """
        if not self.is_all_or_nothing:
            return None
        multiple = self.payout_for(self.n_picks)
        if multiple <= 0:
            return None
        return float(multiple ** (-1.0 / self.n_picks))

    def per_leg_synthetic_american(self) -> int | None:
        """
        The per-leg breakeven expressed as American odds, for contract comparison.

        SYNTHETIC: no book offers this price. It exists so a service that only
        speaks American odds can compare a pick'em leg against a sportsbook one.
        A 3-leg power play at 6x gives 6^(-1/3) = 55.03% -> -122 (unrounded
        -122.4), which is the documented worked example.
        """
        p_be = self.per_leg_breakeven_probability()
        if p_be is None or not 0.0 < p_be < 1.0:
            return None
        return int(round(probability_to_american(p_be)))

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
    probability_source: ProbabilitySource = ProbabilitySource.UNSPECIFIED
    per_leg_breakeven_probability: float | None = None
    per_leg_synthetic_american: int | None = None

    @property
    def disclaimer(self) -> str:
        """The caveat that applies to THIS evaluation's probability source."""
        return DISCLAIMER_BY_SOURCE[self.probability_source]

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
            "PER_LEG_BREAKEVEN_PROB": self.per_leg_breakeven_probability,
            "PER_LEG_SYNTHETIC_AMERICAN": self.per_leg_synthetic_american,
            "PROBABILITY_SOURCE": self.probability_source.value,
            "REASON": self.reason,
            "DISCLAIMER": self.disclaimer,
        }


def _abstain(
    structure: DfsPayoutStructure,
    reason: str,
    source: ProbabilitySource = ProbabilitySource.UNSPECIFIED,
) -> PayoutEvaluation:
    return PayoutEvaluation(
        status=PAYOUT_EV_ABSTAIN,
        n_picks=structure.n_picks,
        structure_label=structure.label,
        structure_source=structure.source,
        reason=reason,
        probability_source=source,
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


# ---------------------------------------------------------------------------
# Leg probabilities from a sharp two-way benchmark
# ---------------------------------------------------------------------------

def benchmark_fair_probability(
    over_american: int, under_american: int, *, side: str = "over"
) -> float:
    """
    De-vig a SHARP BENCHMARK's two-way prop into a fair probability for one side.

    This is the step that makes pick'em EV market-grounded. The de-vig happens on
    the benchmark -- Pinnacle, Circa, a major book's closing line -- never on the
    pick'em operator, which posts no opposing price. ``contracts.market_ev_gate``
    is right to refuse the latter and was wrong only in concluding that EV is
    therefore undefined.

    Delegates to ``odds_math.multiplicative_devig`` so there stays exactly one
    de-vig in this codebase, and inherits its documented caveat: multiplicative
    de-vigging spreads the vig proportionally and will not correct a
    favourite-longshot skew.

    THE LINE MUST MATCH. A benchmark priced at 25.5 does not give the fair
    probability of a pick'em leg at 24.5, and substituting one for the other is
    the most likely way to produce a confidently wrong edge. Matching is the
    caller's responsibility; this function cannot see the lines.
    """
    chosen = str(side).strip().lower()
    if chosen not in {"over", "under"}:
        raise DfsPayoutError(f"side must be 'over' or 'under', got {side!r}")
    fair = multiplicative_devig(int(over_american), int(under_american))
    return float(fair.fair_prob_a if chosen == "over" else fair.fair_prob_b)


# ---------------------------------------------------------------------------
# Exact count distribution for INDEPENDENT legs
# ---------------------------------------------------------------------------

def independent_hit_count_distribution(
    leg_probabilities: Sequence[float],
) -> np.ndarray:
    """
    Exact P(exactly k hits) for independent, heterogeneous legs.

    The Poisson-binomial distribution, computed by the standard exact recursion
    rather than by simulation: for independent legs there is no reason to accept
    Monte Carlo error. Verified against ``scipy.stats.poisson_binom`` to 1e-12
    and against ``parlay.hit_count_distribution`` to simulation noise.

    USE THE COPULA INSTEAD WHEN LEGS ARE CORRELATED.
    ``parlay.hit_count_distribution`` takes a correlation matrix; this does not,
    and cannot. Teammate legs and game-script stacks are exactly the correlated
    case, and treating them as independent OVERSTATES the probability of a
    perfect card, which overstates EV in the direction that loses money. This
    function is for genuinely unrelated legs, or as an exact reference for the
    simulation.
    """
    probabilities = [float(p) for p in leg_probabilities]
    if not probabilities:
        raise DfsPayoutError("no leg probabilities supplied")
    for value in probabilities:
        if not (math.isfinite(value) and 0.0 <= value <= 1.0):
            raise DfsPayoutError(f"leg probability {value!r} is not in [0, 1]")

    # dp[k] = P(exactly k hits so far). One leg folded in at a time.
    dp = np.zeros(len(probabilities) + 1, dtype=float)
    dp[0] = 1.0
    for i, prob in enumerate(probabilities, start=1):
        # Iterate downwards so dp[k - 1] is still the previous round's value.
        for k in range(i, 0, -1):
            dp[k] = dp[k] * (1.0 - prob) + dp[k - 1] * prob
        dp[0] *= 1.0 - prob
    return dp


def evaluate_pickem_entry(
    structure: DfsPayoutStructure,
    leg_probabilities: Sequence[float],
    *,
    source: ProbabilitySource = ProbabilitySource.UNSPECIFIED,
    correlation: "np.ndarray | None" = None,
    n_sims: int = 200_000,
    seed: int = 20240115,
) -> PayoutEvaluation:
    """
    Entry-level EV for a pick'em slip from per-leg win probabilities.

    The routing the gate should have done instead of refusing: the operator
    supplies the payout matrix, the leg probabilities come from elsewhere, and
    the entry is evaluated against the matrix.

    ``correlation`` absent  -> exact Poisson-binomial, no simulation error.
    ``correlation`` present -> Gaussian copula via ``parlay.hit_count_distribution``.

    Abstains when the leg count does not match the structure, because pricing a
    4-leg slip against a 3-pick payout table is silently wrong in whichever
    direction the tables happen to differ.
    """
    probabilities = [float(p) for p in leg_probabilities]
    if len(probabilities) != structure.n_picks:
        return _abstain(
            structure,
            f"{len(probabilities)} leg probability(ies) against a "
            f"{structure.n_picks}-pick structure; the slip and the payout table "
            "describe different products",
            source,
        )

    stderrs: Sequence[float] | None = None
    if correlation is None:
        try:
            counts = independent_hit_count_distribution(probabilities)
        except DfsPayoutError as exc:
            return _abstain(structure, str(exc), source)
    else:
        from src.quant.parlay import ParlayError, ParlayLeg, hit_count_distribution

        legs = [
            ParlayLeg(leg_id=f"leg{i}", model_prob=prob, game_id=f"leg{i}")
            for i, prob in enumerate(probabilities)
        ]
        try:
            counts, errs = hit_count_distribution(
                legs, correlation, n_sims=n_sims, seed=seed
            )
        except (ParlayError, ValueError) as exc:
            return _abstain(structure, f"correlated evaluation refused: {exc}", source)
        stderrs = errs

    out = evaluate_payout(structure, counts, stderrs)
    out.probability_source = source
    out.per_leg_breakeven_probability = structure.per_leg_breakeven_probability()
    out.per_leg_synthetic_american = structure.per_leg_synthetic_american()
    return out
