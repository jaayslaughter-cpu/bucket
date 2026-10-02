"""DFS pick'em payout structures -> breakeven and expected value.

RESEARCH_ONLY. Nothing here places a wager or sizes a stake.

WHY THIS IS A SEPARATE CONCEPT FROM ``ev_engine``, AND MUST STAY ONE.
``contracts.market_ev_gate`` refuses to price a pick'em board as a two-way
market, and routes it here instead (``route = PICKEM_ENTRY_ROUTE``, read by
``src.quant.dfs_entry``).

That refusal is correct and this module does NOT go around it. A sportsbook
quotes two prices, the pair carries the hold, and de-vigging recovers a market
consensus probability to measure a model against. A DFS pick'em quotes ONE fixed
payout and no opposing price, so there is no consensus to recover and no vig to
remove. TWO-WAY EV on that row is undefined; pick'em EV is not, and the gate's
original wording conflated the two.

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
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from src.quant.odds_math import probability_to_american

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

    ``frozen=True`` only stops the ATTRIBUTE being rebound; it does nothing to
    the mapping behind it, so a validated structure whose caller still held the
    original dict could be mutated into an unvalidated one after the fact. The
    mapping is therefore copied and wrapped read-only in ``__post_init__``, and
    validation runs on that copy.
    """

    n_picks: int
    payouts: Mapping[int, float]
    source: str
    label: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.n_picks, bool) or not isinstance(self.n_picks, int) or self.n_picks < 2:
            raise DfsPayoutError(
                f"n_picks must be an integer of at least 2, got {self.n_picks!r}"
            )
        if not isinstance(self.payouts, Mapping):
            raise DfsPayoutError(
                f"payouts must be a mapping of hit count to multiple, got "
                f"{type(self.payouts).__name__}"
            )
        # Copy first: validating the caller's dict and then keeping a reference
        # to it would leave the checks below describing a mapping the caller can
        # still change.
        object.__setattr__(self, "payouts", MappingProxyType(dict(self.payouts)))
        if not self.payouts:
            raise DfsPayoutError("payouts is empty, so this structure pays nothing")
        if not isinstance(self.source, str) or not self.source.strip():
            raise DfsPayoutError(
                "source is required and must be a non-empty string: a payout "
                "table with no provenance cannot be distinguished from an "
                f"invented one (got {self.source!r})"
            )
        for hits, multiple in self.payouts.items():
            if isinstance(hits, bool) or not isinstance(hits, int) or not 0 <= hits <= self.n_picks:
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

    def payout_multiples(self) -> list[float]:
        """
        Gross returns indexed 0..n_picks, zero-filled where a tier does not pay.

        The alignment ``evaluate_payout`` and ``advisory_sizing`` both expect.
        Exposed as a method so those two cannot disagree about how an absent
        tier is represented: a caller assembling the list by hand is one
        off-by-one away from sizing a flex as a power play.
        """
        return [self.payout_for(k) for k in range(self.n_picks + 1)]

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


def _exact_int(value: Any, what: str) -> int:
    """
    Coerce to int only when the value IS that integer.

    ``int()`` alone truncates: a YAML key of ``2.9`` would silently become the
    2-hit tier, and ``n_picks: 3.7`` a 3-pick structure. Truncation here
    misprices every ticket built from the structure and leaves no trace, so a
    non-integral value is refused instead.
    """
    if isinstance(value, bool):
        raise DfsPayoutError(f"{what} is a boolean ({value!r}), not a number")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise DfsPayoutError(
                f"{what} is {value!r}, which is not a whole number; refusing to "
                "truncate it"
            )
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        try:
            return int(text)
        except ValueError as exc:
            raise DfsPayoutError(
                f"{what} is {value!r}, which is not a whole number"
            ) from exc
    raise DfsPayoutError(f"{what} is {value!r}, which is not a whole number")


def structure_from_mapping(payload: dict[str, Any]) -> DfsPayoutStructure:
    """
    Build a structure from config or a user-supplied CSV row.

    Hit-count keys arrive as strings from YAML and JSON, so they are coerced;
    a key that is not an integer is an error rather than a skipped row, because
    silently dropping a payout tier would misprice every ticket using it. The
    coercion is exact -- see ``_exact_int`` -- because a truncated key lands on
    a real, wrong tier.
    """
    raw = payload.get("payouts")
    if not isinstance(raw, Mapping):
        raise DfsPayoutError("payouts must be a mapping of hit count to multiple")
    payouts: dict[int, float] = {}
    for key, value in raw.items():
        hits = _exact_int(key, f"payout key {key!r}")
        if hits in payouts:
            raise DfsPayoutError(
                f"payout key {key!r} repeats hit count {hits}; two multiples for "
                "one tier means one of them is being discarded"
            )
        try:
            payouts[hits] = float(value)
        except (TypeError, ValueError) as exc:
            raise DfsPayoutError(
                f"payout for {hits} hits is not a number: {value!r}"
            ) from exc
    source = payload.get("source")
    if source is None:
        raise DfsPayoutError(
            "source is missing: a payout table with no provenance cannot be "
            "distinguished from an invented one. (A null in config is not a "
            "source -- str(None) would pass the check as the text 'None'.)"
        )
    label = payload.get("label")
    return DfsPayoutStructure(
        n_picks=_exact_int(payload.get("n_picks", 0), "n_picks"),
        payouts=payouts,
        source=source if isinstance(source, str) else str(source),
        label="" if label is None else str(label),
    )


# ---------------------------------------------------------------------------
# Leg probabilities from a sharp two-way benchmark
# ---------------------------------------------------------------------------

def benchmark_fair_probability(
    over_american: int,
    under_american: int,
    *,
    side: str = "over",
    method: str = "multiplicative",
) -> float:
    """
    De-vig a SHARP BENCHMARK's two-way prop into a fair probability for one side.

    This is the step that makes pick'em EV market-grounded. The de-vig happens on
    the benchmark -- Pinnacle, Circa, a major book's closing line -- never on the
    pick'em operator, which posts no opposing price. ``contracts.market_ev_gate``
    is right to refuse the latter and was wrong only in concluding that EV is
    therefore undefined.

    Delegates to ``devig_methods.devig_two_way``, which for the default method
    delegates in turn to ``odds_math.multiplicative_devig`` -- so there stays
    exactly one implementation of the default de-vig in this codebase.

    ``method`` defaults to multiplicative and nothing changes unless a caller
    asks. Its documented caveat still applies at the default: multiplicative
    de-vigging spreads the vig proportionally and will not correct a
    favourite-longshot skew. ``devig_methods`` measures what that costs -- under
    a third of a percentage point on the prices a prop board normally quotes,
    but one to two points on a heavy favourite, which on such a leg is the same
    order as the edge being claimed. Use ``devig_methods.method_spread`` to see
    whether the choice matters on a given price before arguing about it.

    THE LINE MUST MATCH. A benchmark priced at 25.5 does not give the fair
    probability of a pick'em leg at 24.5, and substituting one for the other is
    the most likely way to produce a confidently wrong edge. Matching is the
    caller's responsibility; this function cannot see the lines.
    """
    from src.quant.devig_methods import DevigMethodError, devig_two_way

    chosen = str(side).strip().lower()
    if chosen not in {"over", "under"}:
        raise DfsPayoutError(f"side must be 'over' or 'under', got {side!r}")
    try:
        fair = devig_two_way(int(over_american), int(under_american), method=method)
    except DevigMethodError as exc:
        raise DfsPayoutError(f"benchmark could not be de-vigged: {exc}") from exc
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
    Monte Carlo error. Verified against brute-force enumeration of all 2**n
    outcomes to 1e-15, against ``parlay.hit_count_distribution`` to simulation
    noise, and — where the installed scipy is new enough to have it (1.15+) —
    against ``scipy.stats.poisson_binom`` to 1e-12.

    USE THE COPULA INSTEAD WHEN LEGS ARE CORRELATED.
    ``parlay.hit_count_distribution`` takes a correlation matrix; this does not,
    and cannot. Teammate legs and game-script stacks are exactly the correlated
    case.

    WHICH WAY THE ERROR RUNS DEPENDS ON THE SIGN, and an earlier version of this
    docstring asserted a single direction, which was wrong:

      positive correlation  P(all hit) EXCEEDS the product of the legs, so
                            assuming independence UNDERSTATES a perfect card
                            (two overs on teammates in the same blowout land
                            together more often than independence allows).
      negative correlation  P(all hit) falls BELOW the product, so assuming
                            independence OVERSTATES it (two players splitting
                            one team's shot attempts).

    For a power play that is the whole story, since only the top cell pays. For
    a flex it is not even that simple: correlation moves probability mass out of
    the middle counts toward both tails, so the sign of the EV error also
    depends on the payout curve, and a flex can lose EV from positive
    correlation while a power play gains. Either way the direction is not
    knowable without the matrix, which is why the correlated path exists rather
    than a correction factor. This function is for genuinely unrelated legs, or
    as an exact reference for the simulation.
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


# ---------------------------------------------------------------------------
# The shipped catalogue (config/dfs_payouts.yaml)
# ---------------------------------------------------------------------------

DEFAULT_CATALOG_PATH = Path("config/dfs_payouts.yaml")


def load_payout_catalog(
    path: str | Path = DEFAULT_CATALOG_PATH,
) -> dict[str, DfsPayoutStructure]:
    """
    Load ``config/dfs_payouts.yaml`` into ``{"platform:variant:n": structure}``.

    Keys look like ``underdog:power:3``. The file's top-level ``source`` and
    ``as_of`` are folded into every structure's ``source``, because provenance
    that lives only in a comment at the top of a config file does not travel
    with the number once one entry is evaluated on its own.

    A malformed entry is refused with its key named rather than skipped: a
    catalogue silently missing the tier a caller asked for reads as "that
    product does not exist" instead of "that row is malformed".

    Raises ``DfsPayoutError`` when the file is absent. There is deliberately no
    fallback table — see the module docstring.
    """
    import yaml

    resolved = Path(path)
    if not resolved.exists():
        raise DfsPayoutError(
            f"payout catalogue {resolved} not found, and no table is hardcoded: "
            "supply one whose numbers you can point at"
        )
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise DfsPayoutError(f"{resolved} does not contain a mapping")

    as_of = str(payload.get("as_of") or "unknown date")
    file_source = str(payload.get("source") or "").strip()
    if not file_source:
        raise DfsPayoutError(
            f"{resolved} has no top-level `source`; a payout catalogue with no "
            "provenance cannot be distinguished from an invented one"
        )

    platforms = payload.get("platforms")
    if not isinstance(platforms, Mapping) or not platforms:
        raise DfsPayoutError(f"{resolved} declares no `platforms`")

    catalog: dict[str, DfsPayoutStructure] = {}
    for platform, variants in platforms.items():
        if not isinstance(variants, Mapping):
            raise DfsPayoutError(f"platform {platform!r} is not a mapping of variants")
        for variant, by_count in variants.items():
            if not isinstance(by_count, Mapping):
                raise DfsPayoutError(
                    f"{platform}:{variant} is not a mapping of pick count to payouts"
                )
            for count, payouts in by_count.items():
                key = f"{platform}:{variant}:{count}"
                try:
                    catalog[key] = structure_from_mapping({
                        "n_picks": count,
                        "payouts": payouts,
                        "label": key,
                        "source": f"{file_source} (as_of {as_of})",
                    })
                except DfsPayoutError as exc:
                    raise DfsPayoutError(f"{key}: {exc}") from exc
    if not catalog:
        raise DfsPayoutError(f"{resolved} defines no payout structures")
    return catalog
