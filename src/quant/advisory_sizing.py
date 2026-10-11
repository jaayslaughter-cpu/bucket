"""Recommended stake from fractional Kelly. NEVER EXECUTION.

WHAT THIS IS AND IS NOT. ``recommended_units`` is a RECOMMENDED SIZE — this
module is allowed to say how much, and it does. What it is not is an executed
order: nothing here places a wager, reads a bankroll, returns a currency amount,
or is imported by any execution or dispatch path. Recommending a size and placing
a bet are different acts, and this project does only the first.

Three properties keep that true, and each is enforced by a test:

  - no bankroll is an input, so no output can be a dollar amount. "Units" here
    means PERCENT OF BANKROLL: f* x kelly_fraction x 100. One unit is one
    percent of whatever the reader's bankroll is, which the reader knows and
    this module does not.
  - a non-positive Kelly fraction returns 0.0 rather than a negative number.
    A negative f* means "bet the other side", which on a pick'em entry is not
    an available action, so presenting it as a size would be meaningless.
  - the cap is applied last and is hard. Fractional Kelly is already a
    volatility concession; the cap is a second one, for the case where a
    miscalibrated probability produces a confident f*.

WHY THE BINARY FORMULA IS NOT ENOUGH, which is the substantive correction here.
The closed form

    f* = (b p - q) / b

is Kelly for a bet with TWO outcomes: win b per unit staked, or lose the stake.
A power play is exactly that. A FLEX IS NOT. A 6-pick flex returning 25x on 6
hits, 2.6x on 5 and 0.25x on 4 has four distinct outcomes, and there is no
single (p, b) to substitute. Applying the binary formula to it -- by passing the
top multiple and P(all hit), say -- answers a different question and understates
the size, because it discards the partial-hit tiers that are part of the return.

So the general case maximises expected log wealth over the whole payout
distribution,

    maximise  sum_k P(k) * log(1 - f + f * M_k)

which is concave in f on (0, 1) and solved numerically. It reduces to the closed
form when there are two outcomes, which a test pins so the general path cannot
silently disagree with the formula it generalises.

The agreement is to about 1.3e-9, and the test asserts 1e-8. That is the bounded
optimiser's floor, not a tolerance chosen for comfort: measured at xatol 1e-10,
1e-12 and 1e-14 the error was 1.31e-9, 1.28e-9 and 7.92e-9 -- it stops improving
and then degrades on floating point. An earlier version of this docstring
claimed 1e-9, which is wrong by a factor of about 1.3.

A RECOMMENDED SIZE IS STILL NOT A PREDICTION OF PROFIT, and this is the caveat
that survives the permission to recommend. Kelly is optimal only if the
probabilities are right, and it is MOST sensitive to error exactly where the
edge looks largest. On a model-sourced entry these numbers inherit the model's
calibration error; on a benchmark-sourced one they inherit the benchmark's
sharpness and the line match. ``dfs_payouts.ProbabilitySource`` records which,
and a caller must carry the source alongside any size it shows.

``quant.publication_gate`` is what withholds a model-sourced figure until there
is graded evidence behind it, but note WHERE that happens: at the dispatch
surfaces (``notify-discord``, ``dfs-entry --discord``), not inside this module.
Nothing stops a Python caller computing a size on an uncalibrated model — the
gate governs publication, not arithmetic.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import minimize_scalar

logger = logging.getLogger(__name__)

# Quarter Kelly. Full Kelly is optimal for log wealth and far too aggressive in
# practice: it is maximally sensitive to exactly the probability error these
# estimates carry, so the standard concession is a fraction of it.
DEFAULT_KELLY_FRACTION = 0.25

# Units are percent of bankroll, so 3.0 is 3%. A hard ceiling regardless of how
# confident f* is, because a confident f* is usually a miscalibration.
DEFAULT_MAX_CAP_UNITS = 3.0

# f is searched on (0, 1) exclusive: at f = 1 a single loss is ruin and
# log(1 - f + f*M_k) goes to -inf for any tier paying 0, so the optimiser must
# not evaluate the endpoint.
_F_UPPER = 1.0 - 1e-9


@dataclass(frozen=True)
class AdvisorySize:
    """A suggested size and everything needed to discount it."""

    recommended_units: float
    full_kelly_fraction: float
    kelly_fraction_applied: float
    capped: bool
    method: str
    reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "RECOMMENDED_UNITS": self.recommended_units,
            "FULL_KELLY_FRACTION": self.full_kelly_fraction,
            "KELLY_FRACTION_APPLIED": self.kelly_fraction_applied,
            "CAPPED": self.capped,
            "SIZING_METHOD": self.method,
            "SIZING_NOTE": self.reason,
            # A recommended size, not an order. "ADVISORY_ONLY" said the same
            # thing but now reads as "do not take this seriously", which is the
            # wrong hedge: the number IS the recommendation. What stays true is
            # that nothing in this project places it.
            "AUTO_PLACED": False,
        }


def _unusable_controls(kelly_fraction: float, max_cap_units: float) -> str | None:
    """
    Why these sizing controls cannot produce a size, or None if they can.

    A NEGATIVE kelly_fraction or cap used to flow straight through: f* is
    positive on a real edge, so `f* * -0.25 * 100` is a NEGATIVE
    recommended_units, and the `suggested > cap` test does not catch it. That
    contradicts this module's own stated invariant -- "a non-positive Kelly
    fraction returns 0.0 rather than a negative number" -- which was written
    about f* and quietly did not hold for the multiplier applied to it.

    Refused by returning an abstaining AdvisorySize rather than raising,
    because every other bad input in this module returns one, and a caller that
    handles "no size, here is why" for a miscalibrated probability should not
    have to handle an exception for a mistyped argument.
    """
    if not math.isfinite(kelly_fraction) or kelly_fraction < 0.0:
        return (
            f"kelly_fraction {kelly_fraction!r} is not a finite non-negative "
            "fraction; a negative one would return a negative size on a real edge"
        )
    if not math.isfinite(max_cap_units) or max_cap_units < 0.0:
        return (
            f"max_cap_units {max_cap_units!r} is not a finite non-negative cap"
        )
    return None


def _finalise(
    f_star: float,
    kelly_fraction: float,
    max_cap_units: float,
    method: str,
    reason: str | None = None,
) -> AdvisorySize:
    unusable = _unusable_controls(kelly_fraction, max_cap_units)
    if unusable is not None:
        return AdvisorySize(
            recommended_units=0.0,
            full_kelly_fraction=float(f_star) if math.isfinite(f_star) else 0.0,
            kelly_fraction_applied=float(kelly_fraction)
            if math.isfinite(kelly_fraction) else 0.0,
            capped=False,
            method=method,
            reason=unusable,
        )
    if not math.isfinite(f_star) or f_star <= 0.0:
        return AdvisorySize(
            recommended_units=0.0,
            full_kelly_fraction=float(f_star) if math.isfinite(f_star) else 0.0,
            kelly_fraction_applied=float(kelly_fraction),
            capped=False,
            method=method,
            reason=reason or "Kelly fraction is not positive, so there is no edge to size",
        )
    suggested = f_star * float(kelly_fraction) * 100.0
    capped = suggested > float(max_cap_units)
    # ROUND FIRST, THEN CAP. The other order lets rounding push the answer back
    # over a cap that is not a whole number of cents: a cap of 2.555 with a
    # larger suggestion gives min(...) = 2.555, which rounds to 2.56 -- above
    # the hard limit. The cap is the last thing applied, by definition of being
    # hard.
    return AdvisorySize(
        recommended_units=min(round(suggested, 2), float(max_cap_units)),
        full_kelly_fraction=float(f_star),
        kelly_fraction_applied=float(kelly_fraction),
        capped=capped,
        method=method,
        reason=(
            f"capped at {max_cap_units} unit(s); uncapped suggestion was "
            f"{suggested:.2f}"
        ) if capped else reason,
    )


def recommended_units_binary(
    p_fair: float,
    decimal_odds: float,
    *,
    kelly_fraction: float = DEFAULT_KELLY_FRACTION,
    max_cap_units: float = DEFAULT_MAX_CAP_UNITS,
) -> AdvisorySize:
    """
    Fractional Kelly for a TWO-OUTCOME bet: f* = (b p - q) / b.

    Correct for a power play, a single prop, or any all-or-nothing ticket.
    ``decimal_odds`` is the GROSS return per unit staked, so a 3x power play is
    3.0 and ``b`` is 2.0.

    Refuses rather than guesses on an out-of-range probability: a p of 1.05 would
    otherwise produce a confident size from an impossible input.
    """
    p = float(p_fair)
    if not (math.isfinite(p) and 0.0 < p < 1.0):
        return AdvisorySize(
            0.0, 0.0, float(kelly_fraction), False, "binary",
            f"p_fair {p_fair!r} is not a probability strictly inside (0, 1)",
        )
    d = float(decimal_odds)
    if not (math.isfinite(d) and d > 1.0):
        return AdvisorySize(
            0.0, 0.0, float(kelly_fraction), False, "binary",
            f"decimal_odds {decimal_odds!r} must exceed 1.0 to return a profit",
        )

    b = d - 1.0
    f_star = (b * p - (1.0 - p)) / b
    return _finalise(f_star, kelly_fraction, max_cap_units, "binary")


def recommended_units_multi_outcome(
    count_probabilities: Sequence[float],
    payout_multiples: Sequence[float],
    *,
    kelly_fraction: float = DEFAULT_KELLY_FRACTION,
    max_cap_units: float = DEFAULT_MAX_CAP_UNITS,
) -> AdvisorySize:
    """
    Fractional Kelly for a TIERED payout, by maximising expected log wealth.

    ``count_probabilities[k]`` is P(exactly k hits) and ``payout_multiples[k]``
    the gross return at that hit count — the same indexing
    ``dfs_payouts.evaluate_payout`` uses, so a flex is priced and sized from one
    distribution.

    Returns 0.0 units when the entry's expected value is not positive, since
    Kelly on a non-positive-EV bet is a non-positive fraction.
    """
    probabilities = np.asarray(list(count_probabilities), dtype=float)
    multiples = np.asarray(list(payout_multiples), dtype=float)

    if probabilities.size != multiples.size or probabilities.size == 0:
        return AdvisorySize(
            0.0, 0.0, float(kelly_fraction), False, "multi_outcome",
            f"{probabilities.size} probability(ies) against {multiples.size} "
            "payout(s); they must be aligned by hit count",
        )
    if not (np.all(np.isfinite(probabilities)) and np.all(np.isfinite(multiples))):
        return AdvisorySize(
            0.0, 0.0, float(kelly_fraction), False, "multi_outcome",
            "a probability or payout is not finite",
        )
    if np.any(probabilities < 0.0) or np.any(multiples < 0.0):
        return AdvisorySize(
            0.0, 0.0, float(kelly_fraction), False, "multi_outcome",
            "a negative probability or payout is not a distribution or a return",
        )
    total = float(probabilities.sum())
    if not math.isclose(total, 1.0, abs_tol=1e-6):
        return AdvisorySize(
            0.0, 0.0, float(kelly_fraction), False, "multi_outcome",
            f"probabilities sum to {total:.6f}, not 1",
        )

    expected_value = float(np.dot(probabilities, multiples)) - 1.0
    if expected_value <= 0.0:
        return AdvisorySize(
            0.0, 0.0, float(kelly_fraction), False, "multi_outcome",
            f"entry EV is {expected_value:+.4f}, so Kelly is not positive",
        )

    def negative_log_growth(f: float) -> float:
        wealth = 1.0 - f + f * multiples
        # A tier paying 0 sends log to -inf at f = 1; clipping keeps the
        # optimiser inside the domain instead of returning nan.
        return -float(np.dot(probabilities, np.log(np.clip(wealth, 1e-12, None))))

    result = minimize_scalar(
        negative_log_growth, bounds=(1e-9, _F_UPPER), method="bounded",
        options={"xatol": 1e-10},
    )
    if not result.success:
        return AdvisorySize(
            0.0, 0.0, float(kelly_fraction), False, "multi_outcome",
            f"log-growth maximisation did not converge: {result.message}",
        )
    return _finalise(float(result.x), kelly_fraction, max_cap_units, "multi_outcome")


def recommended_units_for_entry(
    evaluation: object,
    payout_multiples: Sequence[float],
    *,
    kelly_fraction: float = DEFAULT_KELLY_FRACTION,
    max_cap_units: float = DEFAULT_MAX_CAP_UNITS,
) -> AdvisorySize:
    """
    Size a ``dfs_payouts.PayoutEvaluation``, choosing the right Kelly for it.

    An all-or-nothing entry takes the binary closed form; a tiered one takes the
    multi-outcome solver. The choice is made from the payout structure rather
    than left to the caller, because getting it wrong understates a flex.

    Abstains when the evaluation itself abstained: there is no size for an entry
    that could not be priced.
    """
    status = getattr(evaluation, "status", None)
    if status != "PAYOUT_EV_READY":
        return AdvisorySize(
            0.0, 0.0, float(kelly_fraction), False, "none",
            f"entry was not priced (status {status!r}), so it cannot be sized",
        )

    probabilities = list(getattr(evaluation, "count_probabilities", []) or [])
    multiples = list(payout_multiples)

    # CHECKED BEFORE ROUTING, not inside one branch. The multi-outcome path
    # validated this and the binary one did not, so a TRUNCATED payout vector
    # looked like a top-tier-only table: probabilities[-1] is P(all hit) while
    # multiples[-1] is then some middle tier's return, and the entry is sized
    # against a payout it does not have. An empty probabilities list also made
    # probabilities[-1] an IndexError rather than an abstention.
    if not probabilities or len(probabilities) != len(multiples):
        return AdvisorySize(
            0.0, 0.0, float(kelly_fraction), False, "none",
            f"{len(probabilities)} probability(ies) against {len(multiples)} "
            "payout(s); they must be aligned by hit count, so this entry cannot "
            "be sized without guessing which tier each return belongs to",
        )

    paying = [i for i, m in enumerate(multiples) if m > 0.0]

    if len(paying) == 1 and paying[0] == len(multiples) - 1:
        # All-or-nothing: only a perfect card returns anything.
        return recommended_units_binary(
            float(probabilities[-1]), float(multiples[-1]),
            kelly_fraction=kelly_fraction, max_cap_units=max_cap_units,
        )
    return recommended_units_multi_outcome(
        probabilities, multiples,
        kelly_fraction=kelly_fraction, max_cap_units=max_cap_units,
    )


def advisory_size_for_candidate(
    candidate: object,
    *,
    kelly_fraction: float = DEFAULT_KELLY_FRACTION,
    max_cap_units: float = DEFAULT_MAX_CAP_UNITS,
) -> AdvisorySize:
    """
    Size one ``decision_board.BettingDecisionCandidate``, or refuse with a reason.

    WHY THIS EXISTS. The sizer was reachable from exactly one place --
    ``scripts/nba_model_cli.py``, behind a ``--advisory-sizing`` flag on a
    hand-run command -- so the nightly board carried no recommended size at
    all. Neither ``pipeline.slate_board`` nor ``scheduler_worker`` referenced
    sizing. A board row that passes the publication gate and carries no size
    makes the operator do arithmetic this project already does better.
    `docs/autonomy_audit_2026-10-10.md` §3.2.

    ADVISORY, AND THAT IS THE WHOLE CONTRACT. The returned number is read-only
    metadata: ``as_dict`` carries ``AUTO_PLACED: False``, nothing in this
    project places or sizes a wager, and no execution path consumes this.

    DUCK-TYPED on purpose, like ``recommended_units_for_entry`` above: taking
    the candidate by ``getattr`` keeps this module free of a
    ``decision_board`` import, which would be a cycle.

    IT REFUSES ON FOUR CONDITIONS, each of which would otherwise produce a
    confident number from an input that does not support one:

      1. not RECOMMENDED -- the board already declined to recommend it, and
         sizing a row it withheld would contradict the gate that withheld it;
      2. no VALID book -- ``f* = (bp - q)/b`` needs a price, and the house rule
         is that nothing is ranked or priced without valid odds. A model
         probability alone sizes nothing;
      3. no usable model probability;
      4. PUSH MASS WITHOUT THE TWO SIDES TO REMOVE IT -- see below, which is
         the part that is easy to get wrong.

    THE PUSH. Binary Kelly is a two-outcome formula: ``q`` is P(lose). On a
    whole-number line a push is a third outcome and the stake is RETURNED, so
    ``q = 1 - p`` counts returned money as lost and understates the size. The
    correct reduction is the one this project already applies when comparing a
    model against a two-way price (``line_diff._over_given_no_push``): a
    two-way American price de-vigs to a TWO-outcome fair probability because a
    push is voided rather than priced, so the model's probability must be taken
    CONDITIONAL ON NO PUSH --
    ``model_prob / (model_p_over + model_p_under)``.

    When the line can push and the two sides are not both available, there is
    nothing to normalise with and it refuses rather than sizing on a
    probability that means something different from the price.
    """
    def _get(name: str) -> object:
        return getattr(candidate, name, None)

    def _refuse(reason: str) -> AdvisorySize:
        return AdvisorySize(
            0.0, 0.0, float(kelly_fraction), False, "none", reason,
        )

    status = _get("decision_status")
    if status != "RECOMMENDED":
        return _refuse(
            f"not sized: the board's decision_status is {status!r}, not "
            f"RECOMMENDED, so sizing it would contradict the gate that "
            f"withheld it"
        )

    book_status = _get("book_status")
    american = _get("american_odds")
    if book_status != "VALID" or american is None:
        return _refuse(
            f"not sized: book_status is {book_status!r} and american_odds is "
            f"{american!r}. Fractional Kelly needs a price; a model "
            f"probability on its own sizes nothing, and this project does not "
            f"rank or price without valid odds"
        )

    model_prob = _get("model_prob")
    try:
        p_raw = float(model_prob)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return _refuse(f"not sized: model_prob is {model_prob!r}")
    if not (math.isfinite(p_raw) and 0.0 < p_raw < 1.0):
        return _refuse(
            f"not sized: model_prob {p_raw!r} is not a probability in (0, 1)"
        )

    # --- the push reduction ------------------------------------------------
    p_push = _get("model_p_push")
    pushable = bool(_get("line_can_push")) and bool(p_push)
    p_fair = p_raw
    method_note = ""
    if pushable:
        over, under = _get("model_p_over"), _get("model_p_under")
        try:
            decided = float(over) + float(under)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            decided = 0.0
        if not (math.isfinite(decided) and decided > 0.0):
            return _refuse(
                f"not sized: the line can push (p_push={p_push!r}) and "
                f"model_p_over/model_p_under are {over!r}/{under!r}, so there "
                f"is nothing to condition on. A two-way price is a TWO-outcome "
                f"fair probability -- sizing against a three-outcome one "
                f"compares different quantities"
            )
        p_fair = p_raw / decided
        if not (math.isfinite(p_fair) and 0.0 < p_fair < 1.0):
            return _refuse(
                f"not sized: P(side | no push) came out {p_fair!r} from "
                f"model_prob={p_raw!r} and decided mass {decided!r}"
            )
        method_note = (
            f" P(side) taken conditional on no push: {p_raw:.4f} / "
            f"{decided:.4f} = {p_fair:.4f}, because a two-way price voids a "
            f"push rather than pricing it."
        )

    try:
        from src.quant.odds_math import american_to_decimal

        decimal_odds = american_to_decimal(int(american))  # type: ignore[arg-type]
    except Exception as exc:  # noqa: BLE001 — a bad price is a refusal, not a crash
        return _refuse(f"not sized: american_odds {american!r} is not a price ({exc})")

    sized = recommended_units_binary(
        p_fair, decimal_odds,
        kelly_fraction=kelly_fraction, max_cap_units=max_cap_units,
    )
    if method_note and sized.reason:
        return AdvisorySize(
            sized.recommended_units, sized.full_kelly_fraction,
            sized.kelly_fraction_applied, sized.capped, sized.method,
            sized.reason + method_note,
        )
    if method_note:
        return AdvisorySize(
            sized.recommended_units, sized.full_kelly_fraction,
            sized.kelly_fraction_applied, sized.capped, sized.method,
            method_note.strip(),
        )
    return sized
