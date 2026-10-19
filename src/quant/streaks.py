"""
src/quant/streaks.py — Underdog Streaks, with the stake the engine actually risks.

WHAT A STREAK IS. A mandatory 2-pick gatekeeper, then single legs added one at
a time up to 11, paid on a fixed tier curve rather than by compounding
per-leg odds. A correct pick advances a tier; ONE MISS FORFEITS THE WHOLE
STREAK. From leg 3 a "Win Now" cash-out is offered.

THE CORRECTION THIS MODULE EXISTS TO MAKE. The brief proposed

    EV_extension = (p_new * marginal_multiplier) - 1.0

with ``marginal_multiplier = M[k+1] / M[k]``, read as "10/6 = 1.667x, so
break-even is 60%". That formula prices a 1-unit stake. **Adding a leg to a
live streak does not risk 1 unit — it risks everything already accumulated.**
At 3 correct picks you hold a claim worth 6x; a 4th pick either takes you to
10x or takes you to ZERO. So the comparison is not against 1.0, it is against
what you could have walked away with:

    extend iff  p * V(k+1)  >  cashout(k)

Measured on the supplied curve, the consequences of getting this wrong are not
small:

  * Break-even to ride all the way to 11 is p = 0.534, NOT 0.60. The brief's
    one-step rule is CONSERVATIVE -- it declines legs that are in fact +EV.
  * The 60% figure only coincides with the correct answer when the cash-out
    is the FULL tier value. Books discount cash-outs; at 85% of tier the
    break-even to extend from 3 drops to about 0.18 once option value is
    counted.
  * "+EV versus a -110 single needing 52.38%" compares two different things.
    The -110 single risks 1 unit. A streak leg risks the accumulated streak,
    and at p = 0.60 the chance of ever reaching 11 picks is 0.36%. The
    expected value is real and it is almost entirely in a lottery tail.

AND THE SENSITIVITY THAT DECIDES WHETHER ANY OF IT IS REAL. Entry EV per unit
at a flat per-leg p, cash-out at full tier:

    p = 0.600  ->  +2.63      p = 0.550  ->  +0.39
    p = 0.580  ->  +1.50      p = 0.534  ->  +0.01  (break-even)
    p = 0.570  ->  +1.06      p = 0.530  ->  -0.07

A model believing 0.60 when the truth is 0.53 -- 11.7% optimistic -- turns
+2.63 into a LOSS. `quant.parlay.calibration_amplification` already makes this
argument for ordinary tickets; a streak is the same argument with eleven legs
and a 1000x tail, which is to say the worst place to express an edge that has
not been measured. This module computes the numbers; it does not claim the
inputs are right.

NOT A WAGERING ENGINE. Every function here returns a figure and a reason. None
places, sizes, or cashes out anything, and the cash-out verdict is ADVISORY --
`quant.advisory_sizing` already draws that line and this follows it.

RESEARCH_ONLY.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

#: The tier curve lives with the other operator terms, dated and unverified.
STREAK_CONFIG = Path("config/streaks.yaml")

GATEKEEPER_PICKS = 2


class StreakError(ValueError):
    """A streak figure could not be computed from what was supplied."""


@dataclass(frozen=True)
class StreakCurve:
    """Picks -> gross multiple returned per 1 staked. Sorted, contiguous."""

    multipliers: dict[int, float]
    as_of: str | None = None
    source: str | None = None

    @property
    def max_picks(self) -> int:
        return max(self.multipliers)

    def multiple(self, picks: int) -> float:
        if picks not in self.multipliers:
            raise StreakError(
                f"No tier for {picks} pick(s). Known: {sorted(self.multipliers)}"
            )
        return self.multipliers[picks]


@dataclass(frozen=True)
class StreakVerdict:
    """What to do at one tier, and the arithmetic behind it."""

    picks: int
    cashout_value: float
    hold_value: float
    recommendation: str          # "HOLD" or "CASH_OUT"
    break_even_probability: float
    reason: str


def load_streak_curve(
    platform: str = "underdog", *, path: Path | None = None
) -> StreakCurve:
    """
    The streak tier curve from `config/streaks.yaml`.

    Config, not a constant, for the reason that file states about every
    multiple in it: operators change these, and a stale curve misprices every
    entry built on it in whichever direction the change went.
    """
    import yaml

    config_path = path or STREAK_CONFIG
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except OSError as exc:
        raise StreakError(
            f"DATA_NOT_AVAILABLE: cannot read {config_path}: {exc}"
        ) from exc

    block = ((raw.get("platforms") or {}).get(str(platform).lower()) or {})
    streaks = block.get("streaks")
    if not streaks:
        raise StreakError(
            f"DATA_NOT_AVAILABLE: {config_path} has no `streaks` curve for "
            f"{platform!r}."
        )
    multipliers = {int(k): float(v) for k, v in streaks.items()}
    if GATEKEEPER_PICKS not in multipliers:
        raise StreakError(
            f"{platform!r} streak curve has no {GATEKEEPER_PICKS}-pick tier, "
            f"which is the mandatory gatekeeper."
        )
    tiers = sorted(multipliers)
    if tiers != list(range(tiers[0], tiers[-1] + 1)):
        raise StreakError(
            f"{platform!r} streak curve skips tiers ({tiers}); a streak advances "
            f"one pick at a time, so a gap means a tier is missing rather than "
            f"unreachable."
        )
    return StreakCurve(multipliers, raw.get("as_of"), raw.get("source"))


def streak_values(
    curve: StreakCurve,
    *,
    per_leg_probability: float,
    cashout_fraction: float = 1.0,
) -> dict[int, float]:
    """
    V(k): the value of a LIVE streak at k correct picks, in base-stake units.

    BACKWARD INDUCTION, NOT ONE-STEP LOOKAHEAD. V(k) depends on V(k+1), which
    depends on V(k+2): holding at a tier carries OPTION VALUE because you may
    continue from it. Comparing only against the next tier's multiple
    systematically undervalues holding, and the brief's "Hold EV =
    p_next * V_next_tier" is that one-step form.

        V(max)  = the top tier's multiple -- the streak is complete and paid
        V(k)    = max( cashout(k), p * V(k+1) )

    ``cashout_fraction`` is what the operator actually offers as a fraction of
    the tier value. It defaults to 1.0 to reproduce the brief's assumption,
    and it is a parameter because that assumption is the one most likely to be
    wrong: a real "Win Now" is discounted, and the discount moves every
    threshold below.
    """
    if not 0.0 < per_leg_probability < 1.0:
        raise StreakError("per_leg_probability must be in (0, 1)")
    if not 0.0 <= cashout_fraction <= 1.0:
        raise StreakError("cashout_fraction must be in [0, 1]")

    top = curve.max_picks
    values: dict[int, float] = {top: curve.multiple(top)}
    for picks in range(top - 1, GATEKEEPER_PICKS - 1, -1):
        cash = curve.multiple(picks) * cashout_fraction
        hold = per_leg_probability * values[picks + 1]
        values[picks] = max(cash, hold)
    return values


def entry_expected_value(
    curve: StreakCurve,
    *,
    per_leg_probability: float,
    cashout_fraction: float = 1.0,
) -> float:
    """
    EV per 1 staked of ENTERING, in base units.

    BOTH GATEKEEPER LEGS MUST HIT before any tier is reached, so the entry is
    discounted by p^2 -- not by p. Entering is the only point at which 1 unit
    is genuinely the stake, which is why this is the one EV here expressed
    against 1.0.
    """
    values = streak_values(
        curve, per_leg_probability=per_leg_probability,
        cashout_fraction=cashout_fraction,
    )
    reached = per_leg_probability ** GATEKEEPER_PICKS
    return reached * values[GATEKEEPER_PICKS] - 1.0


def extension_verdict(
    curve: StreakCurve,
    *,
    picks_so_far: int,
    next_leg_probability: float,
    continuation_probability: float | None = None,
    cashout_fraction: float = 1.0,
) -> StreakVerdict:
    """
    Hold or cash out at ``picks_so_far``, and the break-even that decides it.

    ``next_leg_probability`` is the specific leg on offer. ``continuation_
    probability`` is what later legs are assumed to be worth and defaults to
    the same number -- separated because the leg in front of you is a known
    candidate while the ones after it are an assumption, and conflating a
    measured p with an assumed one is how option value gets overstated.
    """
    if picks_so_far < GATEKEEPER_PICKS:
        raise StreakError(
            f"A streak only exists from {GATEKEEPER_PICKS} correct picks; "
            f"got {picks_so_far}. Use `entry_expected_value` for the entry."
        )
    if picks_so_far >= curve.max_picks:
        raise StreakError(
            f"{picks_so_far} picks is the top of the curve ({curve.max_picks}); "
            f"there is no leg to add and nothing to decide."
        )
    if not 0.0 < next_leg_probability < 1.0:
        raise StreakError("next_leg_probability must be in (0, 1)")

    later = (
        next_leg_probability if continuation_probability is None
        else continuation_probability
    )
    values = streak_values(
        curve, per_leg_probability=later, cashout_fraction=cashout_fraction,
    )
    cashout_value = curve.multiple(picks_so_far) * cashout_fraction
    next_tier_value = values[picks_so_far + 1]
    hold_value = next_leg_probability * next_tier_value

    break_even = cashout_value / next_tier_value if next_tier_value else 1.0
    hold = hold_value > cashout_value
    reason = (
        f"At {picks_so_far} pick(s) the streak is worth "
        f"{cashout_value:.2f}x if cashed. Adding a leg at p="
        f"{next_leg_probability:.3f} is worth {hold_value:.2f}x "
        f"({next_leg_probability:.3f} x {next_tier_value:.2f}x), because a miss "
        f"forfeits the whole streak rather than one unit. Break-even p is "
        f"{break_even:.3f}."
    )
    return StreakVerdict(
        picks=picks_so_far,
        cashout_value=cashout_value,
        hold_value=hold_value,
        recommendation="HOLD" if hold else "CASH_OUT",
        break_even_probability=break_even,
        reason=reason,
    )


def break_even_for_full_ride(curve: StreakCurve, picks: int | None = None) -> float:
    """
    The flat per-leg p at which riding to ``picks`` is exactly break-even.

    ``M[k] * p^k = 1``, so ``p = M[k]^(-1/k)``. On the supplied curve the full
    11-pick ride breaks even at 0.534 -- the single number the brief's 60%
    most needs next to it.
    """
    target = picks if picks is not None else curve.max_picks
    return float(curve.multiple(target) ** (-1.0 / target))
