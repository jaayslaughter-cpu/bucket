"""Pick'em vs sportsbook line-diff helper (Wave 4).

RESEARCH_ONLY. Line-diff math runs only when the sportsbook side is a
``MarketContext`` / ``PropMarketSnapshot`` with ``status=VALID`` and verified
two-way American odds. Pick'em multipliers never unlock EV.

Source-neutral: it takes whatever snapshot it is handed and asks the EV gate.
The docstrings here named OddsPapi until 2026-10-04, which was misleading twice
over — the helper never cared which feed it was, and OddsPapi had no client in
this repository.

HOW A LINE DIFFERENCE BECOMES A PROBABILITY DIFFERENCE. Moving a line by a
point does not move the fair probability by a fixed amount: the answer depends
on where the line sits relative to the projection and on how dispersed the
stat is. A point off a 4.5-rebound line is worth far more than a point off a
28.5-point line. So when a fitted ``CountDispersion`` is supplied this module
does the honest thing — it INVERTS the distribution at the book's line to
recover the mean the book's own price implies, then re-evaluates at the
pick'em line. Nothing about the model's projection enters: the quantity being
transported is the BOOK's view, moved to a different line, and mixing the
model's mean into a field named ``book_fair_prob_over`` would be a category
error rather than an improvement.

Without a dispersion it falls back to a flat per-point heuristic, and
``LineDiffResult.method`` says which of the two produced the number. That
distinction is the point: a heuristic that cannot be told apart from a
calibrated figure is worse than one that announces itself.

WHAT THIS DOES NOT CLAIM TO HAVE FIXED. ``adjusted_fair_prob_over`` is read by
nothing in this repository outside this module's own tests, and
``paper_research.enrich_row_with_pickem`` — the only caller of
``pickem_vs_book_line_diff`` — has no caller of its own. So the flat 0.03 this
replaced was not mispricing anything in production; it was wrong in a dormant
path. The correctness is worth having before something reaches for it, which
is why it was done, but it moved no number anybody sees.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from src.models.residuals import CountDispersion, over_under_push_from_dispersion
from src.quant.contracts import MarketContext, PropMarketSnapshot, market_ev_gate
from src.quant.odds_math import multiplicative_devig


class LineDiffResult(BaseModel):
    status: str
    pickem_line: float | None = None
    book_line: float | None = None
    line_diff: float | None = None  # pickem - book (negative ⇒ pickem easier for OVER)
    side: str = "over"
    book_fair_prob_over: float | None = None
    adjusted_fair_prob_over: float | None = None
    # WHICH ARITHMETIC PRODUCED adjusted_fair_prob_over: "dispersion" for the
    # inverted distribution evaluation, "heuristic" for the flat per-point
    # shift. Never None on an OK result -- a consumer that cannot tell the two
    # apart would read a fudge factor as a calibrated probability, which is
    # the defect this field exists to prevent.
    method: str | None = None
    #: Why the heuristic ran, when it did. None on the dispersion path.
    method_reason: str | None = None
    #: The mean the book's own price implies at its own line. Dispersion path
    #: only; this is the quantity the inversion recovers.
    implied_mean: float | None = None
    distribution: str | None = None
    #: Push mass at the PICK'EM line. Non-zero only at a whole number, and
    #: reported rather than folded in: adjusted_fair_prob_over is conditional
    #: on no push, to match what a two-way book price means.
    push_prob_at_pickem_line: float | None = None
    reason: str | None = None
    notes: str = Field(
        default=(
            "RESEARCH_ONLY line-diff. Pick'em is not VALID two-way American odds; "
            "no stake from this helper."
        )
    )


def _book_line_and_odds(
    market: MarketContext | PropMarketSnapshot,
) -> tuple[float | None, int | None, int | None]:
    if isinstance(market, PropMarketSnapshot):
        # Prefer ``line``; some older snapshots also populate ``total``.
        line = market.line if market.line is not None else market.total
        return line, market.over_odds_american, market.under_odds_american
    return market.line, market.over_odds_american, market.under_odds_american


#: Bisection settles when the probability is this close to the target. The
#: distribution helper rounds its outputs to 6dp, so asking for more than that
#: would be chasing rounding rather than precision.
_INVERT_PROB_TOL = 1e-6
#: Hard ceiling on the mean the inversion will consider. A fair probability so
#: extreme that no mean below this reproduces it is reported as
#: uninvertible rather than clamped to the bracket's edge, which would return a
#: confident number for a price the family cannot express.
_INVERT_MEAN_CEILING = 500.0
_INVERT_MAX_ITER = 200


def _over_given_no_push(result: dict) -> float | None:
    """
    P(over) CONDITIONAL ON NO PUSH, which is what a two-way price means.

    A sportsbook's two-way American odds de-vig to a two-outcome fair
    probability; a push is voided rather than priced. The distribution helper,
    correctly, reports three outcomes. Comparing a two-outcome fair
    probability against a three-outcome ``probability_over`` would understate
    the book's view at every whole-number line by exactly the push mass.

    Half-point lines cannot push, so the two coincide there — which is the
    common case and is why this is easy to get wrong without noticing.
    """
    if result.get("status") != "OK":
        return None
    over = result.get("probability_over")
    under = result.get("probability_under")
    if over is None or under is None:
        return None
    decided = float(over) + float(under)
    if decided <= 0:
        return None
    return float(over) / decided


def implied_mean_from_fair_prob(
    fair_over: float,
    line: float,
    dispersion: CountDispersion,
) -> dict[str, object]:
    """
    The mean whose distribution reproduces ``fair_over`` at ``line``.

    P(over) is monotone non-decreasing in the mean for all four fitted
    families, so a bisection is exact to tolerance rather than approximate.
    This is the step that makes the line adjustment a distribution evaluation
    instead of a fudge factor: it is how the book's price, which is all this
    helper is given, becomes a quantity that can be re-evaluated elsewhere.

    Returns a status dict rather than a bare float. A fair probability of
    0.999 at a 30.5-point line may have NO mean under a fitted negative
    binomial, and clamping silently to the bracket's edge would hand back a
    confident mean for a price the family cannot express.
    """
    target = float(fair_over)
    if not np.isfinite(target) or not 0.0 < target < 1.0:
        return {
            "status": "DATA_NOT_AVAILABLE",
            "reason": f"fair probability {fair_over!r} is not strictly inside (0, 1)",
            "implied_mean": None,
        }
    if not np.isfinite(line):
        return {
            "status": "DATA_NOT_AVAILABLE",
            "reason": "finite line required",
            "implied_mean": None,
        }

    def p_over(mu: float) -> float | None:
        return _over_given_no_push(
            over_under_push_from_dispersion(mu, float(line), dispersion)
        )

    # The initial bracket is CAPPED BY THE CEILING, not merely grown up to it.
    # Without the min() a line above half the ceiling started the search
    # already past it, the growth loop never ran, and the guard below could
    # not fire — so an uninvertible price came back as a confident mean. Found
    # by a test that asserted the refusal and got an answer instead.
    lo = 1e-6
    hi = min(max(2.0 * (abs(float(line)) + 1.0), 4.0), _INVERT_MEAN_CEILING)
    p_hi = p_over(hi)
    while p_hi is not None and p_hi < target and hi < _INVERT_MEAN_CEILING:
        hi = min(hi * 2.0, _INVERT_MEAN_CEILING)
        p_hi = p_over(hi)
    if p_hi is None or p_hi < target:
        return {
            "status": "DATA_NOT_AVAILABLE",
            "reason": (
                f"no mean at or below {_INVERT_MEAN_CEILING:g} reproduces "
                f"P(over)={target:.6f} at line {float(line):g} under the fitted "
                f"{dispersion.family} family"
            ),
            "implied_mean": None,
        }
    p_lo = p_over(lo)
    if p_lo is None:
        return {
            "status": "DATA_NOT_AVAILABLE",
            "reason": "the fitted family is degenerate at the lower bracket",
            "implied_mean": None,
        }
    if p_lo > target:
        return {
            "status": "DATA_NOT_AVAILABLE",
            "reason": (
                f"P(over)={target:.6f} at line {float(line):g} is below what the "
                f"fitted {dispersion.family} family produces at any positive mean"
            ),
            "implied_mean": None,
        }

    for _ in range(_INVERT_MAX_ITER):
        mid = 0.5 * (lo + hi)
        p_mid = p_over(mid)
        if p_mid is None:
            return {
                "status": "DATA_NOT_AVAILABLE",
                "reason": "the fitted family went degenerate inside the bracket",
                "implied_mean": None,
            }
        if abs(p_mid - target) <= _INVERT_PROB_TOL:
            return {"status": "OK", "reason": None, "implied_mean": float(mid)}
        if p_mid < target:
            lo = mid
        else:
            hi = mid
    return {"status": "OK", "reason": None, "implied_mean": float(0.5 * (lo + hi))}


def _line_adjust_fair_prob(
    fair_over: float,
    *,
    line_diff: float,
    pts_per_prob: float = 0.03,
    book_line: float | None = None,
    pickem_line: float | None = None,
    dispersion: CountDispersion | None = None,
) -> dict[str, object]:
    """
    The book's fair probability, transported to the pick'em line.

    TWO PATHS, AND THE RESULT SAYS WHICH RAN.

    ``dispersion`` — invert the fitted distribution at the book's line to get
    the mean the book's price implies, then evaluate at the pick'em line. A
    point of line is worth whatever the distribution says it is worth there,
    which is the entire reason this path exists.

    ``heuristic`` — a flat ``pts_per_prob`` per point, which is what this
    function did unconditionally before. Kept as the fallback because a
    dispersion is not always on hand, and reported by name so it is never
    mistaken for the calibrated figure.

    THE DOCSTRING THIS REPLACED SAID "soft log-ish shift". The code was, and
    the fallback still is, strictly LINEAR in the line difference —
    ``fair_over - line_diff * pts_per_prob``, clipped. There is no logarithm
    anywhere in it. A comment describing arithmetic the code does not perform
    is worse than no comment: it tells a reader the sharp edges have been
    thought about when they have not.

    Sign, in both paths: ``line_diff`` is pickem - book, so a negative diff
    means the pick'em line is lower and P(over) at it is higher.

    ``side`` NO LONGER CHANGES THE NUMBER, and that is a fix rather than a
    simplification. The previous version flipped the shift's sign for the
    under side, so a pick'em line one point BELOW the book's returned
    ``adjusted_fair_prob_over`` of 0.4808 when asked about the under and
    0.5408 when asked about the over — from the same two lines and the same
    price. P(over) at a given line is a property of the line, not of which
    side a reader has in mind. The field is named ``adjusted_fair_prob_over``
    and now always holds exactly that; a consumer wanting the under takes
    ``1 - adjusted_fair_prob_over`` (less the push mass, which is reported
    separately). ``side`` is still carried on the result because it records
    what was being considered.

    The other possible reading — that the field meant "fair probability of the
    chosen side" — is not what the code did either: it started from
    ``fair_over`` on both sides and only flipped the delta, which is neither
    quantity. No caller passes ``side`` at all; the one call site hardcodes
    "over", so nothing in production changes.
    """

    if dispersion is not None and book_line is not None and pickem_line is not None:
        inverted = implied_mean_from_fair_prob(fair_over, book_line, dispersion)
        if inverted["status"] == "OK":
            mu = float(inverted["implied_mean"])  # type: ignore[arg-type]
            at_pickem = over_under_push_from_dispersion(
                mu, float(pickem_line), dispersion
            )
            adjusted = _over_given_no_push(at_pickem)
            if adjusted is not None:
                return {
                    "adjusted_fair_prob_over": float(np.clip(adjusted, 0.01, 0.99)),
                    "method": "dispersion",
                    "method_reason": None,
                    "implied_mean": round(mu, 4),
                    "distribution": at_pickem.get("distribution"),
                    "push_prob_at_pickem_line": at_pickem.get("probability_push"),
                }
            fallback_reason = (
                at_pickem.get("reason")
                or "the fitted family produced no usable probability at the "
                   "pick'em line"
            )
        else:
            fallback_reason = inverted["reason"]
    elif dispersion is not None:
        fallback_reason = (
            "a dispersion was supplied but both lines are needed to invert it"
        )
    else:
        fallback_reason = "no fitted dispersion supplied"

    delta = -float(line_diff) * float(pts_per_prob)
    return {
        "adjusted_fair_prob_over": float(np.clip(fair_over + delta, 0.01, 0.99)),
        "method": "heuristic",
        "method_reason": str(fallback_reason),
        "implied_mean": None,
        "distribution": None,
        "push_prob_at_pickem_line": None,
    }


def pickem_vs_book_line_diff(
    pickem_line: float | None,
    market: MarketContext | PropMarketSnapshot,
    *,
    side: str = "over",
    pts_per_prob: float = 0.03,
    dispersion: CountDispersion | None = None,
) -> LineDiffResult:
    """
    Compare an approved pick'em line to a VALID two-way sportsbook line.

    Returns DATA_NOT_AVAILABLE when the book side is not VALID or lines missing.
    Does not invent odds or mark pick'em as VALID for stake math.

    ``dispersion`` is the fitted ``CountDispersion`` for this market, when the
    caller has one — a trained artifact carries it (``CountDispersion.from_dict``
    on the saved metadata). Supplying it moves ``adjusted_fair_prob_over`` from
    a flat per-point heuristic onto an inverted distribution evaluation, and
    ``result.method`` records which ran. It is OPTIONAL rather than required
    because no caller in this repository has one to pass today; making it
    required would have meant either breaking the one call site or inventing a
    family, and an invented family is a worse answer than an announced
    heuristic.
    """
    ctx = market.to_market_context() if isinstance(market, PropMarketSnapshot) else market
    gate = market_ev_gate(ctx)
    if gate["status"] != "READY_FOR_EVALUATION":
        return LineDiffResult(
            status="DATA_NOT_AVAILABLE",
            pickem_line=float(pickem_line) if pickem_line is not None else None,
            reason=gate.get("reason") or "Book market is not a VALID two-way price",
        )

    book_line, over_a, under_a = _book_line_and_odds(market)
    if pickem_line is None or not np.isfinite(pickem_line):
        return LineDiffResult(
            status="DATA_NOT_AVAILABLE",
            book_line=float(book_line) if book_line is not None else None,
            reason="Pick'em line missing",
        )
    if book_line is None or not np.isfinite(book_line):
        return LineDiffResult(
            status="DATA_NOT_AVAILABLE",
            pickem_line=float(pickem_line),
            reason="Book line missing on VALID quote",
        )
    if over_a is None or under_a is None:
        return LineDiffResult(
            status="DATA_NOT_AVAILABLE",
            pickem_line=float(pickem_line),
            book_line=float(book_line),
            reason="Two-way American odds required",
        )

    fair = multiplicative_devig(int(over_a), int(under_a))
    fair_over = float(fair.fair_prob_a)
    diff = float(pickem_line) - float(book_line)
    adj = _line_adjust_fair_prob(
        fair_over,
        line_diff=diff,
        pts_per_prob=pts_per_prob,
        book_line=float(book_line),
        pickem_line=float(pickem_line),
        dispersion=dispersion,
    )
    return LineDiffResult(
        status="OK",
        pickem_line=round(float(pickem_line), 4),
        book_line=round(float(book_line), 4),
        line_diff=round(diff, 4),
        side=side.lower(),
        book_fair_prob_over=round(fair_over, 4),
        adjusted_fair_prob_over=round(
            float(adj["adjusted_fair_prob_over"]), 4  # type: ignore[arg-type]
        ),
        method=str(adj["method"]),
        method_reason=adj["method_reason"],  # type: ignore[arg-type]
        implied_mean=adj["implied_mean"],  # type: ignore[arg-type]
        distribution=adj["distribution"],  # type: ignore[arg-type]
        push_prob_at_pickem_line=adj["push_prob_at_pickem_line"],  # type: ignore[arg-type]
        reason=None,
    )
