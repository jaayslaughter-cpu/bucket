"""Alternative two-way de-vig methods.

The question these answer is not "which method is right" — none of them recovers
a true probability — but "does the choice change the answer enough to argue
about". On this repository's own conversion functions the answer is no for a
normal prop price and yes for a heavy favourite, and the table below pins that
so the module docstring cannot drift away from what the code does.

The prices are illustrative two-way quotes, not any book's posted line.
"""

from __future__ import annotations

import pytest

from src.quant.devig_methods import (
    LOGARITHMIC,
    METHODS,
    MULTIPLICATIVE,
    ODDS_RATIO,
    SHIN,
    DevigMethodError,
    devig_two_way,
    method_spread,
)
from src.quant.odds_math import multiplicative_devig

# (price_a, price_b, multiplicative, shin) — the table in the module docstring.
MEASURED = [
    (-110, -110, 0.5000, 0.5000),
    (-115, -105, 0.5108, 0.5113),
    (-130, 110, 0.5427, 0.5445),
    (-200, 165, 0.6386, 0.6447),
    (-300, 240, 0.7183, 0.7279),
    (-450, 340, 0.7826, 0.7955),
]


# --- the documented numbers ---------------------------------------------

@pytest.mark.parametrize("a,b,expected_mult,expected_shin", MEASURED)
def test_the_documented_table_is_what_the_code_produces(a, b, expected_mult, expected_shin):
    assert devig_two_way(a, b).fair_prob_a == pytest.approx(expected_mult, abs=5e-5)
    assert devig_two_way(a, b, method=SHIN).fair_prob_a == pytest.approx(
        expected_shin, abs=5e-5
    )


def test_the_method_is_worth_nothing_on_a_symmetric_price():
    """
    -110/-110 is the textbook prop, and there every method agrees exactly. Any
    argument about method selection has to start past this point.
    """
    for method in METHODS:
        assert devig_two_way(-110, -110, method=method).fair_prob_a == pytest.approx(
            0.5, abs=1e-9
        )
    assert method_spread(-110, -110)["max_spread_vs_multiplicative"] == pytest.approx(
        0.0, abs=1e-9
    )


def test_the_method_is_worth_more_than_the_edge_on_a_heavy_favourite():
    """
    Two percentage points at -450. This pipeline hunts edges of two or three, so
    on prices like these the method is not a detail.
    """
    spread = method_spread(-450, 340)["max_spread_vs_multiplicative"]
    assert spread > 0.015

    # and stays negligible where the board actually quotes
    assert method_spread(-115, -105)["max_spread_vs_multiplicative"] < 0.002


def test_the_alternatives_all_give_the_favourite_more_than_multiplicative_does():
    """
    The favourite-longshot correction has a direction: every alternative here
    moves probability TOWARD the favourite. A method that moved it the other way
    would not be correcting the skew the default is criticised for.
    """
    baseline = devig_two_way(-300, 240).fair_prob_a
    for method in (SHIN, ODDS_RATIO, LOGARITHMIC):
        assert devig_two_way(-300, 240, method=method).fair_prob_a > baseline


# --- properties every method must hold ----------------------------------

@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("a,b", [(-110, -110), (-130, 110), (-200, 165), (-450, 340)])
def test_both_sides_sum_to_one(method, a, b):
    result = devig_two_way(a, b, method=method)
    assert result.fair_prob_a + result.fair_prob_b == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("method", METHODS)
def test_every_probability_is_inside_the_unit_interval(method):
    result = devig_two_way(-450, 340, method=method)
    assert 0.0 < result.fair_prob_a < 1.0
    assert 0.0 < result.fair_prob_b < 1.0


@pytest.mark.parametrize("method", METHODS)
def test_the_hold_reported_is_the_books_overround_whichever_method_ran(method):
    """The hold is a property of the PRICES, not of how they were de-vigged."""
    result = devig_two_way(-130, 110, method=method)
    assert result.hold == pytest.approx(
        result.implied_prob_a + result.implied_prob_b - 1.0, abs=1e-12
    )
    assert result.hold > 0.0


@pytest.mark.parametrize("method", METHODS)
def test_the_result_records_which_method_produced_it(method):
    """A probability whose method is not recorded cannot be compared to another."""
    assert devig_two_way(-130, 110, method=method).method == method


# --- refusals ------------------------------------------------------------

def test_an_unknown_method_raises_rather_than_falling_back():
    """
    Silently returning equal-margin to a caller that asked for Shin would label
    a number as something it is not.
    """
    with pytest.raises(DevigMethodError, match="not a de-vig method"):
        devig_two_way(-130, 110, method="kelly")


@pytest.mark.parametrize("method", [SHIN, ODDS_RATIO, LOGARITHMIC])
def test_a_pair_with_no_overround_is_refused_rather_than_passed_through(method):
    """
    These methods all solve for a parameter that shrinks the total to 1. With
    nothing to shrink there is no root, and returning the raw prices would label
    an untouched number as de-vigged.
    """
    with pytest.raises(DevigMethodError, match="not above 1"):
        devig_two_way(200, 200, method=method)   # implies 0.667, an arbitrage


def test_the_multiplicative_path_still_works_on_a_pair_with_no_overround():
    """
    The default is unchanged by this module, including where it is permissive.
    Over-unit and under-unit pairs are the engine's existing policy, and this
    module does not get to alter it from the side.
    """
    assert devig_two_way(200, 200).fair_prob_a == pytest.approx(0.5)


def test_a_failed_method_is_reported_as_none_rather_than_omitted():
    """A missing key reads as 'not tried'; None reads as 'tried and could not'."""
    spread = method_spread(200, 200)
    assert set(METHODS) <= set(spread)
    assert spread[MULTIPLICATIVE] is not None
    assert spread[SHIN] is None


# --- it does not reimplement the default --------------------------------

def test_the_default_delegates_instead_of_carrying_a_second_copy():
    """Two de-vigs would drift, and the one that drifted is the one nobody reads."""
    import inspect

    import src.quant.devig_methods as mod

    source = inspect.getsource(mod.devig_two_way)
    assert "multiplicative_devig" in source

    for a, b in [(-110, -110), (-130, 110), (-450, 340)]:
        assert devig_two_way(a, b).fair_prob_a == multiplicative_devig(a, b).fair_prob_a


def test_the_solver_cannot_loop_forever():
    """
    The reference implementation this was checked against uses an unbounded
    `while` with a finite-difference Newton step. In a scheduled unattended
    worker a non-converging input is a hang rather than an error, which is the
    reason this uses a capped bisection instead.
    """
    import inspect

    import src.quant.devig_methods as mod

    source = inspect.getsource(mod._bisect)
    assert "for _ in range(MAX_ITERATIONS)" in source
    assert mod.MAX_ITERATIONS <= 1000


def test_mpto_is_not_offered():
    """
    The fifth method in the reference can return a negative probability on long
    prices, and for a two-way market it agrees with shin to four decimals. It is
    a footgun with no information in it.
    """
    assert "mpto" not in METHODS
    with pytest.raises(DevigMethodError):
        devig_two_way(-130, 110, method="mpto")
