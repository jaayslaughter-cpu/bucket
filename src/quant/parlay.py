"""
src/quant/parlay.py — correlation-aware parlay evaluation (RESEARCH_ONLY).

Status: RESEARCH_ONLY · MANUAL_ONLY. This module prices a parlay someone is
considering. It does not select tickets, does not place them, and does not
size them.

WHY THIS EXISTS. A parlay's probability is NOT the product of its legs'
probabilities unless the legs are independent, and legs from the same game
never are. A player's points and his team's total move together; two
teammates' rebounds compete for the same misses. Books price same-game
parlays with correlation built in precisely because the naive product is
wrong, and it is wrong in the direction that makes a ticket look good:

    positively correlated legs -> the naive product UNDERSTATES the parlay
    negatively correlated legs -> the naive product OVERSTATES the parlay

So ``evaluate_parlay`` REFUSES to price a multi-leg ticket whose legs share
a game unless it is given a correlation matrix. Assuming independence there
would not be a simplification; it would be a different bet.

THE JOINT PROBABILITY uses a Gaussian copula. Each leg i wins when a latent
normal Z_i falls below c_i = Phi^-1(p_i), and Z ~ MVN(0, R). With R = I this
reduces exactly to the product, so the independent case is not a special
path. R is the TETRACHORIC correlation of the leg outcomes — the latent
normal correlation, not the observed correlation of the 0/1 indicators,
which is always smaller in magnitude.

THE CALIBRATION WARNING that matters more than any of the above: parlay EV
multiplies the model's error. If each leg's probability is off by a small
factor, an N-leg ticket's probability is off by roughly that factor to the
Nth power. ``calibration_amplification`` makes that concrete. Until the
model is shown calibrated on leakage-safe forward data, a parlay is the
worst available way to express an edge, not the best — it is where an
uncalibrated model's error compounds fastest.

NOTHING HERE INVENTS A PRICE. A leg with no American odds cannot be priced
and the evaluation abstains with a named reason, exactly as the single-leg
gate does.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import norm

from src.quant.decision_board import FORBIDDEN_CLAIM_WORDS
from src.quant.odds_math import american_to_decimal
from src.quant.paper_research import is_whole_number_line

logger = logging.getLogger(__name__)

PLACEMENT_MODE = "MANUAL_ONLY"
RESEARCH_STATUS = "RESEARCH_ONLY"

PARLAY_DISCLAIMER = (
    "RESEARCH_ONLY · MANUAL_ONLY — a parlay evaluation, not a recommendation. "
    "PropIQ does not select, place or size tickets. Parlay EV compounds model "
    "error; an uncalibrated model's parlay number is decoration."
)

DEFAULT_SIMULATIONS = 200_000
DEFAULT_SEED = 20260921

# Below this the Monte Carlo estimate is too noisy relative to the estimate
# itself to report as a probability.
MAX_RELATIVE_STANDARD_ERROR = 0.05


class ParlayError(RuntimeError):
    """Raised when a parlay cannot be evaluated from what was supplied."""


@dataclass(frozen=True)
class ParlayLeg:
    """One leg of a ticket. ``model_prob`` is P(THIS side wins)."""

    leg_id: str
    model_prob: float
    american: int | None = None
    game_id: str | None = None
    player_name: str | None = None
    market: str | None = None
    line: float | None = None
    side: str | None = None
    model_push_prob: float | None = None

    def validate(self) -> None:
        if not (isinstance(self.model_prob, (int, float)) and math.isfinite(self.model_prob)):
            raise ParlayError(f"{self.leg_id}: model_prob is not a number")
        if not 0.0 < float(self.model_prob) < 1.0:
            raise ParlayError(
                f"{self.leg_id}: model_prob must be strictly inside (0, 1), "
                f"got {self.model_prob}"
            )


@dataclass(frozen=True)
class ParlayEvaluation:
    """The result of pricing a ticket, or the named reason it was refused."""

    status: str = "DATA_NOT_AVAILABLE"
    reason: str | None = None
    n_legs: int = 0
    joint_probability: float | None = None
    joint_probability_stderr: float | None = None
    independent_probability: float | None = None
    correlation_effect: float | None = None
    decimal_price: float | None = None
    american_price: int | None = None
    breakeven_probability: float | None = None
    expected_value_per_unit: float | None = None
    method: str = "none"
    legs: tuple[str, ...] = ()
    placement_mode: str = PLACEMENT_MODE
    research_status: str = RESEARCH_STATUS
    disclaimer: str = PARLAY_DISCLAIMER
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "n_legs": self.n_legs,
            "joint_probability": self.joint_probability,
            "joint_probability_stderr": self.joint_probability_stderr,
            "independent_probability": self.independent_probability,
            "correlation_effect": self.correlation_effect,
            "decimal_price": self.decimal_price,
            "american_price": self.american_price,
            "breakeven_probability": self.breakeven_probability,
            "expected_value_per_unit": self.expected_value_per_unit,
            "method": self.method,
            "legs": list(self.legs),
            "placement_mode": self.placement_mode,
            "research_status": self.research_status,
            "disclaimer": self.disclaimer,
            "warnings": list(self.warnings),
        }


def _abstain(reason: str, legs: Sequence[ParlayLeg], **kw: Any) -> ParlayEvaluation:
    lowered = reason.lower()
    hit = next((w for w in FORBIDDEN_CLAIM_WORDS if w in lowered), None)
    if hit:
        raise ParlayError(f"Refusing to emit {reason!r}: contains {hit!r}")
    return ParlayEvaluation(
        status="DATA_NOT_AVAILABLE",
        reason=reason,
        n_legs=len(legs),
        legs=tuple(leg.leg_id for leg in legs),
        **kw,
    )


# ---------------------------------------------------------------------------
# joint probability
# ---------------------------------------------------------------------------


def independent_joint_probability(legs: Sequence[ParlayLeg]) -> float:
    """
    The naive product. Correct ONLY when the legs are independent.

    Exposed so the correlation effect can be reported against it, not so it
    can be used as the answer for same-game legs.
    """
    probability = 1.0
    for leg in legs:
        leg.validate()
        probability *= float(leg.model_prob)
    return probability


def correlation_matrix(
    leg_ids: Sequence[str],
    pairs: Mapping[tuple[str, str], float] | None = None,
) -> np.ndarray:
    """
    Build a tetrachoric correlation matrix from pairwise entries.

    Unlisted pairs are 0 (independent). The result is checked for positive
    semi-definiteness by the caller — an inconsistent set of pairwise
    correlations describes no joint distribution at all, and silently
    "repairing" it would invent dependence nobody supplied.
    """
    size = len(leg_ids)
    matrix = np.eye(size, dtype=float)
    if not pairs:
        return matrix
    index = {leg_id: i for i, leg_id in enumerate(leg_ids)}
    for (a, b), rho in pairs.items():
        if a not in index or b not in index:
            raise ParlayError(f"Correlation given for unknown leg pair ({a!r}, {b!r})")
        if not (-1.0 < float(rho) < 1.0):
            raise ParlayError(f"Correlation for ({a}, {b}) must be in (-1, 1), got {rho}")
        i, j = index[a], index[b]
        matrix[i, j] = matrix[j, i] = float(rho)
    return matrix


def copula_joint_probability(
    legs: Sequence[ParlayLeg],
    correlation: np.ndarray | None = None,
    *,
    n_sims: int = DEFAULT_SIMULATIONS,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float]:
    """
    P(all legs win) under a Gaussian copula. Returns (probability, stderr).

    Leg i wins when Z_i <= Phi^-1(p_i) with Z ~ MVN(0, R). At R = I this
    reproduces the product exactly, so independence is a point in the same
    model rather than a separate code path.

    The standard error is returned because this is Monte Carlo: a parlay
    probability quoted to four decimals from 200k draws has real uncertainty
    in the third, and a caller that cannot see it will over-read the number.
    """
    for leg in legs:
        leg.validate()
    thresholds = norm.ppf([float(leg.model_prob) for leg in legs])

    if correlation is None:
        correlation = np.eye(len(legs), dtype=float)
    if correlation.shape != (len(legs), len(legs)):
        raise ParlayError(
            f"Correlation matrix is {correlation.shape}, expected "
            f"{(len(legs), len(legs))} for {len(legs)} legs"
        )

    # Cholesky is not a validator. It reads only the LOWER triangle, so an
    # asymmetric matrix factorises happily while silently discarding whatever
    # the caller wrote above the diagonal; it accepts a non-unit diagonal,
    # which breaks the marginals the thresholds above were built from; and a
    # NaN entry propagates to a joint probability of exactly 0.0 with no error
    # at all, which reads as "this parlay cannot win".
    #
    # Measured on two legs at model_prob 0.55: asymmetric (0.0 upper, 0.6
    # lower) returned the rho=0.6 answer to machine precision, a diagonal of
    # 2.0 returned 0.33454 where the correct value is 0.40433, and NaN
    # returned 0.0. Each is a wrong number rather than a refusal, so the
    # matrix is checked here instead.
    correlation = np.asarray(correlation, dtype=float)
    if not np.all(np.isfinite(correlation)):
        bad = int((~np.isfinite(correlation)).sum())
        raise ParlayError(
            f"Correlation matrix has {bad} non-finite entr{'y' if bad == 1 else 'ies'} "
            "(NaN or inf). A NaN would otherwise yield a joint probability of "
            "0.0, which reads as an impossible parlay rather than a missing "
            "correlation. Supply a correlation or omit the pair."
        )
    if not np.allclose(correlation, correlation.T, atol=1e-9):
        raise ParlayError(
            "Correlation matrix is not symmetric. Cholesky uses only the lower "
            "triangle, so the values above the diagonal would be discarded "
            "without warning and the answer would describe a different matrix "
            "than the one supplied."
        )
    diagonal = np.diag(correlation)
    if not np.allclose(diagonal, 1.0, atol=1e-9):
        raise ParlayError(
            f"Correlation matrix diagonal is {np.round(diagonal, 6).tolist()}, "
            "not all 1.0. The thresholds above are standard-normal quantiles of "
            "each leg's model_prob, so a non-unit variance silently changes "
            "every leg's marginal and the joint probability stops answering the "
            "question asked."
        )
    off_diagonal = correlation[~np.eye(len(legs), dtype=bool)]
    if off_diagonal.size and np.abs(off_diagonal).max() > 1.0 + 1e-9:
        raise ParlayError(
            f"Correlation matrix has an off-diagonal entry of "
            f"{float(np.abs(off_diagonal).max()):.6f}, outside [-1, 1]. That is "
            "not a correlation."
        )

    try:
        chol = np.linalg.cholesky(correlation)
    except np.linalg.LinAlgError as exc:
        raise ParlayError(
            "Correlation matrix is not positive semi-definite, so it describes "
            f"no joint distribution: {exc}. Check the pairwise values against "
            "each other rather than adjusting one in isolation."
        ) from exc

    rng = np.random.default_rng(seed)
    draws = rng.standard_normal((int(n_sims), len(legs))) @ chol.T
    wins = np.all(draws <= thresholds, axis=1)
    probability = float(wins.mean())
    stderr = float(math.sqrt(max(probability * (1.0 - probability), 0.0) / int(n_sims)))
    return probability, stderr


# ---------------------------------------------------------------------------
# pricing
# ---------------------------------------------------------------------------


def parlay_decimal_price(legs: Sequence[ParlayLeg]) -> float:
    """Product of the legs' decimal prices. Raises if any leg has no price."""
    price = 1.0
    for leg in legs:
        if leg.american is None:
            raise ParlayError(f"{leg.leg_id}: no American odds, so it cannot be priced")
        price *= american_to_decimal(int(leg.american))
    return price


def evaluate_parlay(
    legs: Sequence[ParlayLeg],
    *,
    correlation: np.ndarray | Mapping[tuple[str, str], float] | None = None,
    n_sims: int = DEFAULT_SIMULATIONS,
    seed: int = DEFAULT_SEED,
    allow_independence_across_games: bool = True,
) -> ParlayEvaluation:
    """
    Price a ticket, or abstain with a named reason. Never invents an input.

    Refuses when:

    - fewer than two legs, or any leg carries no American price
    - two legs share a ``game_id`` and no correlation was supplied — the
      naive product is not a conservative simplification there, it is a
      different bet, and books price the difference on purpose
    - a leg's line can push and the push mass was not given. A push VOIDS
      that leg and re-prices the whole ticket at the remaining legs' odds,
      which this two-outcome model does not represent
    - the correlation matrix is not positive semi-definite
    - the Monte Carlo estimate is too noisy relative to its own size
    """
    legs = list(legs)
    if len(legs) < 2:
        return _abstain("A parlay needs at least two legs", legs)

    leg_ids = [leg.leg_id for leg in legs]
    if len(set(leg_ids)) != len(leg_ids):
        return _abstain("Leg ids must be unique to correlate them", legs)

    try:
        for leg in legs:
            leg.validate()
    except ParlayError as exc:
        return _abstain(str(exc), legs)

    unpriced = [leg.leg_id for leg in legs if leg.american is None]
    if unpriced:
        return _abstain(
            f"No American odds for {unpriced}. EV is a claim about a price; "
            "without one there is nothing to be right or wrong about.",
            legs,
        )

    # A push voids the leg and re-prices the ticket. Two outcomes cannot say that.
    pushable = [
        leg.leg_id for leg in legs
        if is_whole_number_line(leg.line)
        and (leg.model_push_prob is None or float(leg.model_push_prob) > 0.0)
    ]
    if pushable:
        return _abstain(
            f"Legs {pushable} sit on whole (or unknown) lines that can push. A "
            "push voids the leg and re-prices the parlay at the remaining "
            "legs' odds; this win/lose model does not represent that. Use "
            "half-lines, or supply model_push_prob=0.0 where a push is "
            "genuinely impossible.",
            legs,
        )

    warnings: list[str] = []

    # Same-game legs without a correlation are a refusal, not an assumption.
    by_game: dict[str, list[str]] = {}
    for leg in legs:
        if leg.game_id:
            by_game.setdefault(str(leg.game_id), []).append(leg.leg_id)
    shared = {g: ids for g, ids in by_game.items() if len(ids) > 1}

    if isinstance(correlation, Mapping):
        try:
            matrix = correlation_matrix(leg_ids, correlation)
        except ParlayError as exc:
            return _abstain(str(exc), legs)
    else:
        matrix = correlation

    if shared and matrix is None:
        return _abstain(
            f"Legs share a game ({shared}) and no correlation was supplied. "
            "Same-game legs are not independent — a player's points and his "
            "team's total move together — so the product of the legs is a "
            "different bet, not a simpler one. This repository has no fitted "
            "leg correlations yet (see estimate_tetrachoric_correlation).",
            legs,
        )

    if matrix is None:
        if not allow_independence_across_games:
            return _abstain("No correlation supplied and independence is disallowed", legs)
        matrix = np.eye(len(legs), dtype=float)
        warnings.append(
            "No correlation supplied; legs are in different games and were "
            "treated as independent. Cross-game legs still share pace, "
            "officiating and blowout risk — independence is an assumption, "
            "not a fact."
        )

    try:
        joint, stderr = copula_joint_probability(
            legs, matrix, n_sims=n_sims, seed=seed,
        )
    except ParlayError as exc:
        return _abstain(str(exc), legs)

    if joint <= 0.0:
        return _abstain(
            f"Monte Carlo produced no winning draws in {n_sims:,} simulations; "
            "the ticket is rarer than this sample can measure.",
            legs,
        )
    if stderr / joint > MAX_RELATIVE_STANDARD_ERROR:
        return _abstain(
            f"Monte Carlo standard error {stderr:.5f} is {stderr / joint:.1%} of "
            f"the estimate {joint:.5f}; raise n_sims before reading this number.",
            legs,
        )

    independent = independent_joint_probability(legs)
    decimal = parlay_decimal_price(legs)
    breakeven = 1.0 / decimal
    ev = joint * decimal - 1.0

    from src.quant.odds_math import decimal_to_american

    return ParlayEvaluation(
        status="OK",
        reason=None,
        n_legs=len(legs),
        joint_probability=joint,
        joint_probability_stderr=stderr,
        independent_probability=independent,
        correlation_effect=joint - independent,
        decimal_price=decimal,
        american_price=decimal_to_american(decimal),
        breakeven_probability=breakeven,
        expected_value_per_unit=ev,
        method="gaussian_copula",
        legs=tuple(leg_ids),
        warnings=tuple(warnings),
    )


# ---------------------------------------------------------------------------
# where the correlations would have to come from
# ---------------------------------------------------------------------------


def estimate_tetrachoric_correlation(
    outcomes_a: Iterable[Any],
    outcomes_b: Iterable[Any],
    *,
    min_observations: int = 200,
) -> float:
    """
    Estimate the latent correlation between two binary leg outcomes.

    This is the input ``evaluate_parlay`` refuses to guess. It needs realised
    outcomes for both legs over many shared games, which requires the player
    game logs this repository does not have yet.

    Raises rather than returning a shaky number from a small sample: a
    correlation fitted on a handful of games is noise, and noise in R moves
    the parlay probability in the direction that flatters the ticket.
    """
    a = np.asarray(list(outcomes_a), dtype=float)
    b = np.asarray(list(outcomes_b), dtype=float)
    if a.shape != b.shape:
        raise ParlayError(f"Outcome arrays differ in length: {a.shape} vs {b.shape}")
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < int(min_observations):
        raise ParlayError(
            f"Only {len(a)} paired observations; need at least {min_observations}. "
            "A correlation fitted on a small sample is noise, and noise here "
            "moves the parlay probability in the direction that flatters it."
        )
    if len(set(a.tolist())) < 2 or len(set(b.tolist())) < 2:
        raise ParlayError("One of the legs never varies; correlation is undefined")

    # Pearson correlation on the 0/1 indicators (phi), then the standard
    # normal-copula inversion to the latent scale the copula expects.
    phi = float(np.corrcoef(a, b)[0, 1])
    p_a, p_b = float(a.mean()), float(b.mean())
    joint = float(((a > 0.5) & (b > 0.5)).mean())
    # Solve for rho such that the bivariate normal orthant matches `joint`.
    lo, hi = -0.999, 0.999
    z_a, z_b = norm.ppf(p_a), norm.ppf(p_b)
    for _ in range(80):
        mid = (lo + hi) / 2.0
        legs = (
            ParlayLeg(leg_id="a", model_prob=p_a),
            ParlayLeg(leg_id="b", model_prob=p_b),
        )
        matrix = np.array([[1.0, mid], [mid, 1.0]])
        modelled, _ = copula_joint_probability(legs, matrix, n_sims=40_000, seed=DEFAULT_SEED)
        if modelled < joint:
            lo = mid
        else:
            hi = mid
    rho = (lo + hi) / 2.0
    logger.info(
        "tetrachoric: n=%d p_a=%.3f p_b=%.3f phi=%.3f -> rho=%.3f (z_a=%.2f z_b=%.2f)",
        len(a), p_a, p_b, phi, rho, z_a, z_b,
    )
    return float(rho)


def calibration_amplification(
    n_legs: int,
    *,
    per_leg_probability: float,
    per_leg_bias: float,
) -> dict[str, float]:
    """
    Show how a per-leg calibration error compounds across a ticket.

    A model that is a little optimistic on one leg is badly optimistic on
    five. This is the argument against expressing an unproven edge as a
    parlay, expressed in the only terms that settle it — numbers.
    """
    if not 0.0 < per_leg_probability < 1.0:
        raise ParlayError("per_leg_probability must be in (0, 1)")
    believed = per_leg_probability ** int(n_legs)
    true_leg = per_leg_probability * (1.0 - per_leg_bias)
    if not 0.0 < true_leg < 1.0:
        raise ParlayError("per_leg_bias implies an impossible true probability")
    true_parlay = true_leg ** int(n_legs)
    return {
        "n_legs": float(n_legs),
        "believed_leg_probability": per_leg_probability,
        "true_leg_probability": true_leg,
        "believed_parlay_probability": believed,
        "true_parlay_probability": true_parlay,
        "relative_overstatement": (believed / true_parlay) - 1.0,
    }
