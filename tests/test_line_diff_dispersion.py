"""
Tests for src/quant/line_diff.py's dispersion-based line adjustment.

The thing being replaced was a flat 0.03 of fair probability per point of
line, with a docstring calling it a "soft log-ish shift" — it was strictly
linear and there was no logarithm in it. A point of line is worth whatever the
distribution says it is worth AT THAT LINE, which is the whole content of the
fix, and the numbers below are the evidence: the same one-point move is worth
+0.162 off a 4.5 line and +0.069 off a 24.5 line.

THE STRONGEST ASSERTION HERE IS THE IDENTITY. Inverting the book's own price
at the book's own line and then evaluating at that same line must return the
price unchanged. Nothing else catches an inversion that is subtly solving for
the wrong quantity, because a wrong-but-monotone transport still moves in the
right direction and still looks plausible at every line.

RESEARCH_ONLY. No stake, no wager; these are probabilities at lines.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.models.residuals import CountDispersion, over_under_push_from_dispersion
from src.quant.contracts import PropMarketSnapshot
from src.quant.line_diff import (
    _INVERT_MEAN_CEILING,
    _line_adjust_fair_prob,
    implied_mean_from_fair_prob,
    pickem_vs_book_line_diff,
)

NEGBIN = CountDispersion(
    family="negbin", phi=1.35, n_train_rows=5000, selection_scores={}
)


def snap(line: float, over: int = -115, under: int = -105) -> PropMarketSnapshot:
    return PropMarketSnapshot(
        game_id="g1",
        bookmaker="x",
        market_id="m1",
        line=line,
        over_odds_american=over,
        under_odds_american=under,
        status="VALID",
        captured_at_utc=datetime.now(timezone.utc),
    )


# --- the identity ---------------------------------------------------------

@pytest.mark.parametrize("line", [4.5, 9.5, 24.5, 31.5])
def test_transporting_the_price_to_its_own_line_returns_the_price(line):
    """
    The one assertion a wrong-but-monotone inversion cannot pass. If the
    implied mean solves for the wrong quantity, this round trip lands
    somewhere else while every other test in this file still looks right.
    """
    out = pickem_vs_book_line_diff(line, snap(line), dispersion=NEGBIN)
    assert out.method == "dispersion"
    assert out.adjusted_fair_prob_over == pytest.approx(
        out.book_fair_prob_over, abs=5e-4
    )


@pytest.mark.parametrize("family,kwargs", [
    ("poisson", {"phi": 1.0}),
    ("negbin", {"phi": 1.6}),
    ("zip", {"phi": 1.0, "zero_inflation": 0.2}),
    ("normal", {"phi": 1.0, "sigma_scale": 1.4}),
])
def test_the_identity_holds_for_every_fitted_family(family, kwargs):
    """
    `over_under_push_from_dispersion` gives each family its own pmf and cdf,
    and records in its own comment that an `else` sweeping non-negbin families
    into Poisson would price a fitted ZIP with the wrong distribution. The
    inversion has to work against all four, not just the common one.
    """
    disp = CountDispersion(
        family=family, n_train_rows=4000, selection_scores={}, **kwargs
    )
    out = pickem_vs_book_line_diff(14.5, snap(14.5), dispersion=disp)
    assert out.method == "dispersion", out.method_reason
    assert out.adjusted_fair_prob_over == pytest.approx(
        out.book_fair_prob_over, abs=5e-4
    )


# --- what the fix is actually for -----------------------------------------

def test_a_point_of_line_is_worth_more_at_a_low_line_than_a_high_one():
    """
    The defect the flat 0.03 embodied, in one assertion. Same price, same
    one-point move, and the distribution says the low line's point is worth
    more than twice the high line's — while the heuristic says 0.03 for both.
    """
    low = pickem_vs_book_line_diff(3.5, snap(4.5), dispersion=NEGBIN)
    high = pickem_vs_book_line_diff(23.5, snap(24.5), dispersion=NEGBIN)

    low_move = low.adjusted_fair_prob_over - low.book_fair_prob_over
    high_move = high.adjusted_fair_prob_over - high.book_fair_prob_over
    assert low_move > high_move * 2.0, (low_move, high_move)

    flat_low = pickem_vs_book_line_diff(3.5, snap(4.5))
    flat_high = pickem_vs_book_line_diff(23.5, snap(24.5))
    assert flat_low.adjusted_fair_prob_over - flat_low.book_fair_prob_over == (
        pytest.approx(flat_high.adjusted_fair_prob_over - flat_high.book_fair_prob_over)
    ), "the heuristic is supposed to be line-blind; that is the point"


def test_a_lower_pickem_line_raises_p_over_and_a_higher_one_lowers_it():
    probs = [
        pickem_vs_book_line_diff(p, snap(24.5), dispersion=NEGBIN).adjusted_fair_prob_over
        for p in (22.5, 23.5, 24.5, 25.5, 27.5)
    ]
    assert probs == sorted(probs, reverse=True), probs


# --- the method is always named -------------------------------------------

def test_without_a_dispersion_the_heuristic_runs_and_says_so():
    """
    A heuristic that cannot be told apart from a calibrated figure is worse
    than one that announces itself — a consumer would read a fudge factor as a
    probability.
    """
    out = pickem_vs_book_line_diff(23.5, snap(24.5))
    assert out.method == "heuristic"
    assert "no fitted dispersion" in (out.method_reason or "")
    assert out.implied_mean is None
    assert out.distribution is None


def test_with_a_dispersion_the_result_names_the_distribution_and_the_mean():
    out = pickem_vs_book_line_diff(23.5, snap(24.5), dispersion=NEGBIN)
    assert out.method == "dispersion"
    assert out.method_reason is None
    assert out.distribution == "NegativeBinomial"
    assert out.implied_mean is not None
    # The implied mean must sit near the book's line, since the book's fair
    # probability here is близко to a coin flip. A mean of 3 or 300 would mean
    # the inversion solved something else.
    assert 20.0 < out.implied_mean < 30.0, out.implied_mean


def test_an_uninvertible_price_falls_back_rather_than_clamping():
    """
    A fair probability no mean below the ceiling can reproduce must not come
    back as a confident number from the bracket's edge. Contrived on purpose:
    no NBA stat line approaches the ceiling, so this guards the pathological
    case rather than a live one.
    """
    line = _INVERT_MEAN_CEILING * 3
    out = pickem_vs_book_line_diff(line - 1.0, snap(line), dispersion=NEGBIN)
    assert out.method == "heuristic"
    assert "no mean at or below" in (out.method_reason or "")

    direct = implied_mean_from_fair_prob(0.51, line, NEGBIN)
    assert direct["status"] == "DATA_NOT_AVAILABLE"
    assert direct["implied_mean"] is None


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.2, 1.4, float("nan")])
def test_a_fair_probability_outside_the_open_unit_interval_is_refused(bad):
    out = implied_mean_from_fair_prob(bad, 24.5, NEGBIN)
    assert out["status"] == "DATA_NOT_AVAILABLE"
    assert "strictly inside" in str(out["reason"])


# --- push mass, and the two-way convention --------------------------------

def test_a_whole_pickem_line_reports_push_mass_separately():
    """
    A two-way book price de-vigs to a two-outcome fair probability — a push is
    voided, not priced. The distribution helper correctly reports three
    outcomes, so the comparison is made CONDITIONAL ON NO PUSH. Folding the
    push into either side would understate the book's view at every
    whole-number line by exactly the push mass.
    """
    out = pickem_vs_book_line_diff(24.0, snap(24.5), dispersion=NEGBIN)
    assert out.method == "dispersion"
    assert out.push_prob_at_pickem_line is not None
    assert out.push_prob_at_pickem_line > 0.0
    # Conditional on no push, so it is NOT 1 - P(under) - P(push).
    assert 0.0 < out.adjusted_fair_prob_over < 1.0


def test_the_adjusted_probability_is_conditional_on_no_push_not_raw():
    """
    THE ASSERTION THE IDENTITY TEST CANNOT MAKE, and the gap was real: the
    round trip holds under BOTH conventions, because inverting and evaluating
    with the same (wrong) convention cancels. Deleting the no-push
    conditioning left every other test in this file green.

    What discriminates is the arithmetic at a whole line. A two-way book price
    de-vigs to a two-outcome probability, so the comparable quantity is
    over / (over + under) — strictly GREATER than the raw three-outcome
    over whenever push carries mass.
    """
    out = pickem_vs_book_line_diff(24.0, snap(24.5), dispersion=NEGBIN)
    assert out.method == "dispersion"
    assert out.implied_mean is not None

    raw = over_under_push_from_dispersion(out.implied_mean, 24.0, NEGBIN)
    assert raw["probability_push"] > 0.0, "a whole line must carry push mass"

    conditional = raw["probability_over"] / (
        raw["probability_over"] + raw["probability_under"]
    )
    assert out.adjusted_fair_prob_over == pytest.approx(conditional, abs=5e-4)
    assert out.adjusted_fair_prob_over > raw["probability_over"] + 1e-6, (
        "the raw three-outcome over was returned, which understates the "
        "book's two-way view by exactly the push mass"
    )


def test_a_half_point_pickem_line_has_no_push_mass():
    out = pickem_vs_book_line_diff(23.5, snap(24.5), dispersion=NEGBIN)
    assert out.push_prob_at_pickem_line == pytest.approx(0.0)


# --- the side fix ---------------------------------------------------------

def test_p_over_at_the_pickem_line_does_not_depend_on_the_side_asked_about():
    """
    THE DEFECT THIS PINS, which was in the function being replaced rather than
    in the thing being ported. `side="under"` flipped the shift's sign, so one
    pair of lines and one price produced adjusted_fair_prob_over of 0.5408 for
    the over and 0.4808 for the under. P(over) at a line is a property of the
    line. A consumer wanting the under takes 1 - this, less the push mass.
    """
    over = pickem_vs_book_line_diff(23.5, snap(24.5), side="over")
    under = pickem_vs_book_line_diff(23.5, snap(24.5), side="under")
    assert over.adjusted_fair_prob_over == under.adjusted_fair_prob_over
    # And the side is still recorded, because it says what was asked.
    assert (over.side, under.side) == ("over", "under")

    # Same on the dispersion path.
    d_over = pickem_vs_book_line_diff(23.5, snap(24.5), side="over", dispersion=NEGBIN)
    d_under = pickem_vs_book_line_diff(23.5, snap(24.5), side="under", dispersion=NEGBIN)
    assert d_over.adjusted_fair_prob_over == d_under.adjusted_fair_prob_over
    # A lower pick'em line raises P(over) whichever side was asked about.
    assert d_under.adjusted_fair_prob_over > d_under.book_fair_prob_over


# --- the heuristic is still linear, and the docstring now says so ---------

def test_the_fallback_is_linear_in_the_line_difference():
    """
    The replaced docstring called it a "soft log-ish shift". It is not, and
    this is what "linear" means as an assertion: doubling the line difference
    doubles the probability move exactly.
    """
    one = _line_adjust_fair_prob(0.50, line_diff=-1.0, pts_per_prob=0.03)
    two = _line_adjust_fair_prob(0.50, line_diff=-2.0, pts_per_prob=0.03)
    assert one["adjusted_fair_prob_over"] == pytest.approx(0.53)
    assert two["adjusted_fair_prob_over"] == pytest.approx(0.56)
    assert (two["adjusted_fair_prob_over"] - 0.50) == pytest.approx(
        2.0 * (one["adjusted_fair_prob_over"] - 0.50)
    )


def test_no_docstring_in_this_module_claims_a_logarithm_it_does_not_perform():
    """
    The specific wording that was wrong, pinned so it cannot come back with
    the arithmetic unchanged. There is no `log` call in this module.
    """
    import inspect

    import src.quant.line_diff as module

    source = inspect.getsource(module)
    # No logarithm is computed anywhere. This is the assertion that matters;
    # the phrase "log-ish" DOES still appear, in the note recording that it
    # was wrong, and a test banning the string would have forced that note to
    # be deleted to stay green.
    assert "np.log" not in source
    assert "math.log" not in source
    assert "log(" not in source
    # And the correction is on the record rather than silently applied.
    assert "strictly LINEAR" in source


# --- the caller can pass one ---------------------------------------------

def test_the_board_enricher_passes_a_dispersion_through():
    """
    The helper being correct is worth nothing if the parameter stops at the
    call site. No caller HAS a dispersion today, which is recorded in the
    module docstring; what is pinned here is that one could.
    """
    import inspect

    from src.quant import paper_research

    source = inspect.getsource(paper_research.enrich_row_with_pickem)
    assert "dispersion=dispersion" in source
