"""Alternative two-way de-vig methods. RESEARCH_ONLY.

WHY MORE THAN ONE METHOD, AND WHY THE DEFAULT DOES NOT CHANGE.
``odds_math.multiplicative_devig`` spreads the vig proportionally across both
sides. Its own docstring already concedes the weakness — "a favourite-longshot
skew will not be corrected here" — and this module is what makes that concession
measurable rather than rhetorical. It does NOT replace the default. Every
existing caller keeps the multiplicative answer until it asks for another.

HOW MUCH IT ACTUALLY MOVES, measured on this repository's own conversion
functions rather than asserted (see tests/test_devig_methods.py, which pins
these):

    two-way price        multiplicative   shin     spread vs multiplicative
    -110 / -110               0.5000     0.5000        0.00 pp
    -115 / -105               0.5108     0.5113        0.08 pp
    -130 / +110               0.5427     0.5445        0.27 pp
    -200 / +165               0.6386     0.6447        0.92 pp
    -300 / +240               0.7183     0.7279        1.48 pp
    -450 / +340               0.7826     0.7955        2.01 pp

READ THE SHAPE OF THAT TABLE BEFORE REACHING FOR A METHOD. On the prices a
player-prop board actually quotes -- roughly -140 to +120 -- the choice is worth
a quarter of a percentage point, which is well inside the noise of the
probability being de-vigged and not worth an argument. It becomes material only
on heavy favourites: a star's low points or rebounds line at -300 or beyond,
where multiplicative understates the favourite by one to two points. That is the
same order as the edge such a bet would be claimed on, so on those prices the
method is not a detail.

WHAT EACH ONE ASSUMES, since none of them is "the true probability":

  multiplicative   the vig is a constant proportion of each side's implied
  (equal margin)   probability. Simple, symmetric, and wrong in the specific way
                   that books load the side the public prefers.

  shin             a fraction z of the money is from insiders, and the book
                   prices to protect against them. z is solved for, and it is
                   the one method here with a story about WHY the skew exists
                   rather than just a curve that removes it.

  odds_ratio       the fair and quoted odds differ by a constant odds ratio.
                   A curve fit, no mechanism claimed.

  logarithmic      the fair probability is the implied one raised to a power.
                   Also a curve fit, and the most aggressive of the four on
                   long prices.

MPTO ("margin proportional to odds") IS DELIBERATELY ABSENT. It is the fifth
method in the reference this module was checked against, and it can return a
NEGATIVE probability on long prices -- that reference hides its output whenever
negative odds are present, which is most of a prop board. For a two-way market
it also agrees with shin to four decimal places, so it adds a footgun and no
information.

SOLVED BY BISECTION, NOT BY THE REFERENCE'S NEWTON STEP. Each method's root
function is monotone in its parameter for a two-way market, so bisection cannot
diverge and cannot divide by zero. The reference implementation uses an
unbounded finite-difference Newton iteration inside `while` loops with no
iteration cap: a non-converging input spins forever, and its step divides by a
difference that can be zero. In a scheduled unattended worker that is a hang
rather than an error, which is why this does not copy it.

Nothing here recommends a bet or sizes a stake.
"""

from __future__ import annotations

import logging
import math
from typing import Callable, Sequence

from src.quant.odds_math import DevigResult, american_to_decimal, multiplicative_devig

logger = logging.getLogger(__name__)

MULTIPLICATIVE = "multiplicative"
SHIN = "shin"
ODDS_RATIO = "odds_ratio"
LOGARITHMIC = "logarithmic"

METHODS = (MULTIPLICATIVE, SHIN, ODDS_RATIO, LOGARITHMIC)

# Bisection is linear in the bit count, so this is generous: 200 halvings of any
# finite bracket is far past double precision. It exists to make a
# non-converging input an ERROR rather than a hang.
MAX_ITERATIONS = 200
TOLERANCE = 1e-12


class DevigMethodError(ValueError):
    """A de-vig that could not be solved, or a method that does not exist."""


def _implied(decimals: Sequence[float]) -> list[float]:
    return [1.0 / d for d in decimals]


def _bisect(
    total: Callable[[float], float], low: float, high: float, what: str
) -> float:
    """
    Solve ``total(c) == 1`` on a bracket where ``total`` is monotone.

    Returns the parameter, or raises. Raising rather than returning the last
    iterate is deliberate: a de-vig that did not converge is not a slightly
    worse probability, it is no probability, and the caller's own abstention
    path is better than a number with no meaning behind it.
    """
    f_low, f_high = total(low) - 1.0, total(high) - 1.0
    if not (math.isfinite(f_low) and math.isfinite(f_high)):
        raise DevigMethodError(f"{what}: the root function is not finite at the bracket")
    if f_low == 0.0:
        return low
    if f_high == 0.0:
        return high
    if f_low * f_high > 0.0:
        raise DevigMethodError(
            f"{what}: no solution inside the bracket [{low:g}, {high:g}] "
            f"(f={f_low:+.3e}, {f_high:+.3e}). The prices are outside what this "
            "method can describe."
        )

    for _ in range(MAX_ITERATIONS):
        mid = 0.5 * (low + high)
        f_mid = total(mid) - 1.0
        if abs(f_mid) < TOLERANCE or (high - low) < TOLERANCE:
            return mid
        if f_low * f_mid < 0.0:
            high = mid
        else:
            low, f_low = mid, f_mid
    raise DevigMethodError(f"{what}: did not converge in {MAX_ITERATIONS} iterations")


def _shin_probabilities(implied: Sequence[float]) -> list[float]:
    """
    Shin (1993): solve for the insider fraction z, then back out fair p.

    ``booked`` is the overround (sum of implied probabilities). The closed form
    each side takes, given z, is

        p_i = ( sqrt(z^2 + 4(1 - z) * x_i^2 / booked) - z ) / (2(1 - z))

    and z is whatever makes the p sum to one. The sum is monotone decreasing in
    z on [0, 1), so bisection is exact here.
    """
    booked = sum(implied)
    if booked <= 0.0:
        raise DevigMethodError("shin: the implied probabilities do not sum above zero")

    def probabilities(z: float) -> list[float]:
        out = []
        for x in implied:
            inner = z * z + 4.0 * (1.0 - z) * (x * x) / booked
            if inner < 0.0:
                raise DevigMethodError("shin: negative discriminant")
            out.append((math.sqrt(inner) - z) / (2.0 * (1.0 - z)))
        return out

    # z = 0 reduces to the multiplicative answer, which for an overround book
    # sums above 1; z -> 1 drives the sum down. The bracket is that interval.
    z = _bisect(lambda c: sum(probabilities(c)), 0.0, 1.0 - 1e-12, "shin")
    return probabilities(z)


def _odds_ratio_probabilities(implied: Sequence[float]) -> list[float]:
    """Cheung's odds-ratio method: p = x / (c + x - c*x), solved for c."""

    def probabilities(c: float) -> list[float]:
        return [x / (c + x - c * x) for x in implied]

    c = _bisect(lambda c: sum(probabilities(c)), 1.0, 1e6, "odds_ratio")
    return probabilities(c)


def _logarithmic_probabilities(implied: Sequence[float]) -> list[float]:
    """Power method: p = x ** c, solved for c. Each x < 1, so the sum falls in c."""
    if any(x >= 1.0 for x in implied):
        raise DevigMethodError(
            "logarithmic: a side's implied probability is at or above 1, so "
            "raising it to a power cannot reduce the total"
        )

    def probabilities(c: float) -> list[float]:
        return [x**c for x in implied]

    c = _bisect(lambda c: sum(probabilities(c)), 1.0, 500.0, "logarithmic")
    return probabilities(c)


_SOLVERS = {
    SHIN: _shin_probabilities,
    ODDS_RATIO: _odds_ratio_probabilities,
    LOGARITHMIC: _logarithmic_probabilities,
}


def devig_two_way(
    american_a: int, american_b: int, *, method: str = MULTIPLICATIVE
) -> DevigResult:
    """
    De-vig a two-way price by the named method.

    ``method=MULTIPLICATIVE`` delegates to ``odds_math.multiplicative_devig``
    rather than reimplementing it, so the default path has exactly one
    implementation and this module cannot drift away from it.

    Raises ``DevigMethodError`` on an unknown method or a price the method
    cannot describe. It does not fall back to multiplicative: a caller that
    asked for Shin and silently got equal-margin would be reading a number the
    label says it is not.
    """
    chosen = str(method).strip().lower()
    if chosen == MULTIPLICATIVE:
        return multiplicative_devig(int(american_a), int(american_b))
    if chosen not in _SOLVERS:
        raise DevigMethodError(
            f"{method!r} is not a de-vig method. Known: {', '.join(METHODS)}"
        )

    decimals = [american_to_decimal(int(american_a)), american_to_decimal(int(american_b))]
    implied = _implied(decimals)
    booked = sum(implied)
    if booked <= 1.0:
        # No vig to remove, or an arbitrage. These methods all solve for a
        # parameter that SHRINKS the total to one; with nothing to shrink there
        # is no root, and silently returning the raw prices would label an
        # untouched number as de-vigged.
        raise DevigMethodError(
            f"the two prices imply {booked:.6f}, which is not above 1, so there "
            "is no overround for this method to remove. Check the pair is really "
            "two sides of one market."
        )

    fair = _SOLVERS[chosen](implied)
    return DevigResult(
        fair_prob_a=float(fair[0]),
        fair_prob_b=float(fair[1]),
        hold=booked - 1.0,
        implied_prob_a=float(implied[0]),
        implied_prob_b=float(implied[1]),
        method=chosen,
    )


def method_spread(american_a: int, american_b: int) -> dict[str, float | None]:
    """
    Every method's fair probability for side A, and how far apart they are.

    A DIAGNOSTIC, not an input to a decision. Its purpose is to answer "does the
    method matter on this price" before anyone argues about which to use — on a
    -110/-110 prop the spread is zero and the argument is empty, and on a -400
    favourite it is worth more than the edge being claimed.

    A method that cannot describe the price is reported as None rather than
    omitted, so the caller can see which failed.
    """
    out: dict[str, float | None] = {}
    for name in METHODS:
        try:
            out[name] = devig_two_way(american_a, american_b, method=name).fair_prob_a
        except (DevigMethodError, ValueError, ZeroDivisionError, OverflowError) as exc:
            logger.debug("de-vig %s failed on %s/%s: %s", name, american_a, american_b, exc)
            out[name] = None

    solved = [v for v in out.values() if v is not None]
    baseline = out.get(MULTIPLICATIVE)
    out["max_spread_vs_multiplicative"] = (
        max(abs(v - baseline) for v in solved)
        if solved and baseline is not None else None
    )
    out["spread"] = (max(solved) - min(solved)) if len(solved) > 1 else None
    return out
