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
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

logger = logging.getLogger(__name__)

#: The tier curve lives with the other operator terms, dated and unverified.
STREAK_CONFIG = Path("config/streaks.yaml")

GATEKEEPER_PICKS = 2

#: THE FIRST TIER THAT CAN BE CASHED. The operator's terms as supplied: "after
#: every correct pick FROM LEG 3 ONWARDS, Underdog presents a 'Win Now'
#: cash-out offer". Two correct picks is a live streak, not a cashable
#: position -- there is no offer there to accept or decline.
FIRST_CASHOUT_PICKS = GATEKEEPER_PICKS + 1


class StreakError(ValueError):
    """A streak figure could not be computed from what was supplied."""


@dataclass(frozen=True)
class StreakCurve:
    """Picks -> gross multiple returned per 1 staked. Sorted, contiguous."""

    multipliers: Mapping[int, float]
    as_of: str | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        # FROZEN STOPS REBINDING, NOT MUTATION. `curve.multipliers[11] = 1e9`
        # silently repriced every valuation taken from this curve, and the
        # loader is the kind of thing a caller holds for a whole run, so the
        # edit outlives whatever made it. Read-only, and `load_streak_curve`
        # is the one door that validates what goes in.
        object.__setattr__(
            self, "multipliers", MappingProxyType(dict(self.multipliers))
        )

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
    #: None where the operator offers no cash-out at this tier.
    cashout_value: float | None
    hold_value: float
    recommendation: str          # "HOLD" or "CASH_OUT"
    break_even_probability: float
    #: WHICH QUESTION `break_even_probability` ANSWERS, because there are two
    #: and they are different numbers. "next_leg" is the threshold for THIS
    #: leg holding the continuation assumption fixed -- exact, and the right
    #: reading when `continuation_probability` was supplied. "flat_all_legs"
    #: is the self-consistent fixed point for a player who assumes the same p
    #: for every remaining leg, which is the only one a flat p can be
    #: compared against. "no_offer" means there is nothing to decline.
    break_even_basis: str
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
    # EVERY VALUE CHECKED HERE, BECAUSE NOTHING DOWNSTREAM CAN. `{int(k):
    # float(v)}` raised a bare ValueError on a typo, and accepted `.nan`,
    # `.inf`, zero and negatives without comment -- a NaN tier makes every
    # comparison in `streak_values` False so the induction silently prefers
    # whichever branch NaN loses to, and a zero tier values a live streak at
    # nothing. This is the only door into a StreakCurve, so it is the only
    # place the curve can be made trustworthy.
    if not isinstance(streaks, Mapping):
        raise StreakError(
            f"{config_path}: {platform!r} `streaks` must be a mapping of "
            f"correct picks -> multiple, got {type(streaks).__name__}"
        )

    multipliers: dict[int, float] = {}
    for key, value in streaks.items():
        try:
            picks = int(str(key).strip())
        except (TypeError, ValueError) as exc:
            raise StreakError(
                f"{config_path}: {platform!r} has a tier keyed {key!r}, which "
                f"is not a number of correct picks"
            ) from exc
        try:
            multiple = float(str(value).strip())
        except (TypeError, ValueError) as exc:
            raise StreakError(
                f"{config_path}: {platform!r} tier {picks} pays {value!r}, "
                f"which is not a multiple"
            ) from exc
        if picks < 1:
            raise StreakError(
                f"{config_path}: {platform!r} has a tier at {picks} pick(s); a "
                f"streak starts at one."
            )
        if not math.isfinite(multiple) or multiple <= 0:
            raise StreakError(
                f"{config_path}: {platform!r} tier {picks} pays {value!r}. A "
                f"tier returns a positive, finite multiple of the stake; a "
                f"non-finite one makes every comparison in the valuation "
                f"meaningless and a zero one values a live streak at nothing."
            )
        if picks in multipliers:
            raise StreakError(
                f"{config_path}: {platform!r} gives tier {picks} twice "
                f"({multipliers[picks]} and {multiple}); which one is the "
                f"operator's term is not something this can guess."
            )
        multipliers[picks] = multiple

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
    # STRICTLY INCREASING, because each tier is the one before it plus a leg
    # that could have lost everything. A tier paying no more than its
    # predecessor is a transposed digit in config (2.0 for 20.0), and the
    # induction would read it as a position worth cashing immediately and
    # always -- a real recommendation built on a typo.
    for earlier, later in zip(tiers, tiers[1:]):
        if multipliers[later] <= multipliers[earlier]:
            raise StreakError(
                f"{platform!r} tier {later} pays {multipliers[later]}, which is "
                f"no more than tier {earlier}'s {multipliers[earlier]}. Adding a "
                f"leg risks the whole streak, so a tier that does not pay more "
                f"is a config error rather than a bad bet."
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
        V(k)    = max( cashout(k), p * V(k+1) )   for k >= FIRST_CASHOUT_PICKS
        V(k)    = p * V(k+1)                      below it

    THE GATEKEEPER IS NOT A CASHABLE POSITION, and the first version of this
    took the max at every tier including 2. The operator's terms say "Win Now"
    is offered from LEG 3; at two correct picks there is no offer, so there is
    no alternative to compare holding against. Taking the max there floors the
    value of a live 2-pick streak at the 3x tier it cannot actually collect,
    which overstates entry EV and -- the live consequence -- lets
    `extension_verdict` return CASH_OUT at a tier where cashing out is not a
    thing the operator lets you do.

    MEASURED, so the size of it is on the record rather than assumed: on the
    supplied curve the floor only binds below about p = 0.52, because above
    that the hold value at two picks already exceeds 3x. The entry EVs in this
    module's header (p = 0.60 -> +2.63, break-even 0.534) are unchanged by the
    correction. At p = 0.45 it was +0.06 units of pure fiction, at p = 0.40
    +0.10, at p = 0.30 +0.11.

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
        hold = per_leg_probability * values[picks + 1]
        if picks < FIRST_CASHOUT_PICKS:
            # No offer exists at this tier, so holding is not a choice that
            # beat something -- it is the only thing the position can do.
            values[picks] = hold
            continue
        values[picks] = max(curve.multiple(picks) * cashout_fraction, hold)
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

    assumed_continuation = continuation_probability is None
    later = (
        next_leg_probability if assumed_continuation
        else continuation_probability
    )
    values = streak_values(
        curve, per_leg_probability=later, cashout_fraction=cashout_fraction,
    )
    next_tier_value = values[picks_so_far + 1]
    hold_value = next_leg_probability * next_tier_value

    if picks_so_far < FIRST_CASHOUT_PICKS:
        # NOTHING TO DECLINE, SO NOTHING TO DECIDE. "Win Now" starts at leg
        # 3; a 2-pick streak cannot be collected, so the only thing this
        # position can do is continue. Reporting it as a HOLD that beat a
        # cash-out worth 3x would be inventing the alternative -- and the
        # earlier version did exactly that, returning CASH_OUT below p = 0.52
        # and advising an action the operator does not offer.
        return StreakVerdict(
            picks=picks_so_far,
            cashout_value=None,
            hold_value=hold_value,
            recommendation="HOLD",
            break_even_probability=0.0,
            break_even_basis="no_offer",
            reason=(
                f"At {picks_so_far} pick(s) there is no cash-out: the "
                f"operator offers 'Win Now' from {FIRST_CASHOUT_PICKS} picks "
                f"onwards. The position continues or it is forfeited, so any "
                f"p above zero is better than the alternative and there is no "
                f"decision to take. Continuing is worth {hold_value:.2f}x "
                f"({next_leg_probability:.3f} x {next_tier_value:.2f}x)."
            ),
        )

    cashout_value = curve.multiple(picks_so_far) * cashout_fraction
    hold = hold_value > cashout_value

    # TWO DIFFERENT BREAK-EVENS, AND THE RATIO IS ONLY ONE OF THEM.
    # `cashout / V(k+1)` is exact for THIS leg while the continuation stays
    # fixed -- correct when the caller supplied `continuation_probability`,
    # because then V(k+1) does not move when p does. When the continuation
    # DEFAULTS to the same p it does move, so the ratio is a function of the
    # very number it is meant to be a threshold for: feed p = 0.55 and get
    # back a "break-even" computed at a continuation of 0.55, feed 0.60 and
    # get a different one. Nobody can compare a flat p against that. For the
    # flat case this solves the fixed point instead, so the answer is the p
    # at which a player assuming p throughout is genuinely indifferent.
    if assumed_continuation:
        break_even = _flat_break_even(
            curve, picks_so_far, cashout_fraction=cashout_fraction
        )
        basis = "flat_all_legs"
    else:
        break_even = cashout_value / next_tier_value if next_tier_value else 1.0
        basis = "next_leg"

    reason = (
        f"At {picks_so_far} pick(s) the streak is worth "
        f"{cashout_value:.2f}x if cashed. Adding a leg at p="
        f"{next_leg_probability:.3f} is worth {hold_value:.2f}x "
        f"({next_leg_probability:.3f} x {next_tier_value:.2f}x), because a miss "
        f"forfeits the whole streak rather than one unit. Break-even p is "
        f"{break_even:.3f} ({basis})."
    )
    return StreakVerdict(
        picks=picks_so_far,
        cashout_value=cashout_value,
        hold_value=hold_value,
        recommendation="HOLD" if hold else "CASH_OUT",
        break_even_probability=break_even,
        break_even_basis=basis,
        reason=reason,
    )


def _flat_break_even(
    curve: StreakCurve, picks: int, *, cashout_fraction: float,
) -> float:
    """
    The flat per-leg p at which holding at ``picks`` exactly ties cashing out.

    THE FIXED POINT, NOT A RATIO. Solves ``p * V_p(picks+1) = cashout(picks)``
    where V_p is the whole backward induction run at that same p, so the
    answer does not depend on a continuation assumption the caller has to
    make up. ``p * V_p(k+1)`` is strictly increasing in p -- every tier of the
    induction is -- so the root is unique and bisection finds it.

    Returns 0.0 when the operator offers nothing to cash (``cashout_fraction``
    of zero): holding then wins at any p above zero, so the threshold is zero.
    """
    cashout = curve.multiple(picks) * cashout_fraction
    if cashout <= 0:
        return 0.0

    def excess(probability: float) -> float:
        values = streak_values(
            curve, per_leg_probability=probability,
            cashout_fraction=cashout_fraction,
        )
        return probability * values[picks + 1] - cashout

    # At p -> 1 the continuation is worth at least the next tier, which pays
    # strictly more than this one, so the root is bracketed.
    low, high = 1e-12, 1.0 - 1e-12
    if excess(high) <= 0:
        return 1.0
    for _ in range(200):
        mid = (low + high) / 2.0
        if excess(mid) > 0:
            high = mid
        else:
            low = mid
    return (low + high) / 2.0


def break_even_for_full_ride(curve: StreakCurve, picks: int | None = None) -> float:
    """
    The flat per-leg p at which riding to ``picks`` is exactly break-even.

    ``M[k] * p^k = 1``, so ``p = M[k]^(-1/k)``. On the supplied curve the full
    11-pick ride breaks even at 0.534 -- the single number the brief's 60%
    most needs next to it.
    """
    target = picks if picks is not None else curve.max_picks
    return float(curve.multiple(target) ** (-1.0 / target))
