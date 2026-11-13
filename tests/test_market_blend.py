"""
tests/test_market_blend.py -- the blend nests both endpoints, or it is not one.

WHAT THIS IS FOR. `src/quant/market_blend.py` exists because the served
probabilities are measurably worse than saying nothing: on 533 graded rows the
mean stated probability was 0.7077 against an observed 0.4765, and the served
Brier of 0.3011 is worse than the 0.2500 a constant 0.5 would have scored.
A monotone recalibration cannot fix the ordering; anchoring on the market can
change which quantity is being calibrated at all.

The whole claim of the logit form is that it NESTS the market and the model,
so that `c` is a measurement rather than a dial. Most of what follows tests
exactly that, because a "blend" that cannot reproduce its own endpoints is an
arbitrary third number wearing the name of a compromise.

RESEARCH_ONLY. Probabilities. No wager.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pytest

from src.quant.market_blend import (
    MIN_ROWS_FOR_FIT,
    MIN_SLATES_FOR_FIT,
    STATUS_DEGENERATE,
    STATUS_INSUFFICIENT_DATA,
    STATUS_OK,
    MarketBlendCoefficients,
    MarketBlendError,
    blend_probability,
    expit,
    fit_market_blend,
    logit,
)


#: First slate of the synthetic sample; 80 consecutive days follow it.
BASE_SLATE = date(2026, 1, 1)


def coef(a: float, b: float, c: float, status: str = STATUS_OK) -> MarketBlendCoefficients:
    """A usable coefficient set, for the arithmetic tests."""
    return MarketBlendCoefficients(
        a=a, b=b, c=c, market="PTS", n_rows=1000, n_slates=60, status=status,
    )


# ---------------------------------------------------------------------------
# logit / expit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("p", [0.01, 0.1, 0.5, 0.75, 0.9, 0.99])
def test_logit_and_expit_round_trip(p: float) -> None:
    assert expit(logit(p)) == pytest.approx(p, abs=1e-9)


def test_logit_of_a_certainty_is_finite_rather_than_infinite() -> None:
    """
    THE CLIP, which is the only reason a valid input cannot produce a nan.

    A de-vigged probability of exactly 1.0 is reachable from a real board --
    a one-sided market with a huge favourite price rounds there -- and
    `math.log(1/0)` raises before any downstream arithmetic sees it.
    """
    assert math.isfinite(logit(0.0))
    assert math.isfinite(logit(1.0))
    assert logit(0.0) < -10.0
    assert logit(1.0) > 10.0


def test_expit_does_not_overflow_on_a_large_negative() -> None:
    """`1/(1+exp(-z))` overflows for z = -800; the branch in expit does not."""
    assert expit(-800.0) == pytest.approx(0.0, abs=1e-12)
    assert expit(800.0) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("bad", [-0.01, 1.01, float("nan"), float("inf")])
def test_a_probability_outside_the_unit_interval_is_refused(bad: float) -> None:
    with pytest.raises(MarketBlendError):
        logit(bad)


# ---------------------------------------------------------------------------
# THE NESTING PROPERTIES -- the reason the form was chosen
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("p_market", [0.35, 0.5, 0.62, 0.8])
@pytest.mark.parametrize("p_model", [0.2, 0.5, 0.9])
def test_c_zero_returns_exactly_the_market(p_market: float, p_model: float) -> None:
    """a=0, b=1, c=0 must ignore the model entirely, whatever it says."""
    out = blend_probability(p_market, p_model, coef(0.0, 1.0, 0.0))
    assert out == pytest.approx(p_market, abs=1e-9)


@pytest.mark.parametrize("p_market", [0.35, 0.5, 0.8])
@pytest.mark.parametrize("p_model", [0.2, 0.5, 0.9])
def test_c_one_returns_exactly_the_model(p_market: float, p_model: float) -> None:
    """a=0, b=1, c=1 collapses to logit(p_model), i.e. the market drops out."""
    out = blend_probability(p_market, p_model, coef(0.0, 1.0, 1.0))
    assert out == pytest.approx(p_model, abs=1e-9)


def test_an_intermediate_c_lands_between_the_two_and_moves_with_c() -> None:
    """
    The property that makes `c` readable as "how much our disagreement is
    worth": the output is monotone in it, and bracketed by the endpoints.
    """
    p_market, p_model = 0.55, 0.80
    outs = [
        blend_probability(p_market, p_model, coef(0.0, 1.0, c))
        for c in (0.0, 0.2, 0.5, 0.8, 1.0)
    ]
    assert outs == sorted(outs), outs
    assert outs[0] == pytest.approx(p_market, abs=1e-9)
    assert outs[-1] == pytest.approx(p_model, abs=1e-9)
    for middle in outs[1:-1]:
        assert p_market < middle < p_model


def test_a_model_that_agrees_with_the_market_is_unchanged_by_any_c() -> None:
    """
    d = 0 when the two agree, so c multiplies nothing. This is what makes the
    blend safe on a board where the model simply republishes the price.
    """
    for c in (0.0, 0.17, 0.5, 1.0, 3.0):
        out = blend_probability(0.62, 0.62, coef(0.0, 1.0, c))
        assert out == pytest.approx(0.62, abs=1e-9)


def test_the_blend_shrinks_an_overconfident_model_toward_the_market() -> None:
    """
    THE MEASURED PROBLEM THIS ADDRESSES, in one assertion.

    The served board states 0.7077 on average and wins 0.4765 of the time.
    With the market at a realistic 0.52 and a small fitted c, the blended
    number must sit far nearer the market than the model's 0.95 -- that
    shrinkage IS the intended effect, and if a future change inverts the sign
    of `c` this is the test that notices.
    """
    blended = blend_probability(0.52, 0.95, coef(0.0, 1.0, 0.2))
    # Computed, not guessed: logit(0.52) + 0.2 * (logit(0.95) - logit(0.52))
    # = 0.6529, and expit(0.6529) = 0.6577.
    assert blended == pytest.approx(0.6577, abs=1e-4), blended
    assert abs(blended - 0.52) < abs(blended - 0.95), (
        "the blended number must sit nearer the market than the model; if "
        "this inverts, the sign of c has been broken"
    )


# ---------------------------------------------------------------------------
# refusal rather than silent fallback
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", [STATUS_INSUFFICIENT_DATA, STATUS_DEGENERATE])
def test_an_unusable_coefficient_set_refuses_instead_of_falling_back(status: str) -> None:
    """
    A silent fallback to the market would make a blended board and an
    unblended board indistinguishable in the output, which is exactly what an
    A/B measurement cannot tolerate.
    """
    with pytest.raises(MarketBlendError, match=status):
        blend_probability(0.55, 0.8, coef(0.0, 1.0, 0.2, status=status))


def test_the_reason_travels_with_the_refusal() -> None:
    bad = MarketBlendCoefficients(
        a=float("nan"), b=float("nan"), c=float("nan"), market="PTS",
        n_rows=12, n_slates=2, status=STATUS_INSUFFICIENT_DATA,
        reason="12 usable rows, 400 required",
    )
    with pytest.raises(MarketBlendError, match="400 required"):
        blend_probability(0.55, 0.8, bad)


# ---------------------------------------------------------------------------
# the fitter
# ---------------------------------------------------------------------------


def synthetic_rows(
    a: float, b: float, c: float, *, n: int = 4000, seed: int = 7,
) -> list[dict[str, object]]:
    """
    Rows generated BY the blend, so a correct fitter recovers (a, b, c).

    This is the only way to test a fitter without asserting its own output
    back at it: the truth is known because the data was built from it.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for i in range(n):
        p_market = float(rng.uniform(0.25, 0.75))
        p_model = float(rng.uniform(0.05, 0.95))
        m = math.log(p_market / (1 - p_market))
        d = math.log(p_model / (1 - p_model)) - m
        truth = 1.0 / (1.0 + math.exp(-(a + b * m + c * d)))
        rows.append({
            "p_market": p_market,
            "p_model": p_model,
            "outcome": float(rng.random() < truth),
            # Exactly 80 distinct slates, so the slate gate is satisfied
            # honestly rather than by an accident of the date arithmetic --
            # two earlier attempts at this line produced 93 and 84.
            "game_date": (BASE_SLATE + timedelta(days=i % 80)).isoformat(),
        })
    return rows


def test_the_fitter_recovers_the_coefficients_it_was_generated_from() -> None:
    """The arithmetic is right, demonstrated against a known truth."""
    fitted = fit_market_blend(synthetic_rows(0.0, 1.0, 0.25), market="PTS")
    assert fitted.status == STATUS_OK, fitted.reason
    assert fitted.b == pytest.approx(1.0, abs=0.25), fitted.as_dict()
    assert fitted.c == pytest.approx(0.25, abs=0.15), fitted.as_dict()
    assert fitted.a == pytest.approx(0.0, abs=0.2), fitted.as_dict()


def test_a_model_with_no_information_fits_c_near_zero() -> None:
    """
    THE FINDING THIS MODULE IS BUILT TO BE ABLE TO REPORT.

    When the model's disagreement with the market is pure noise, the honest
    answer is c ~ 0, i.e. "our disagreement is worth nothing". A fitter that
    could not produce that answer would be unable to tell us the one thing we
    most need to know.
    """
    fitted = fit_market_blend(synthetic_rows(0.0, 1.0, 0.0), market="PTS")
    assert fitted.status == STATUS_OK, fitted.reason
    assert abs(fitted.c) < 0.12, fitted.as_dict()


def test_too_few_rows_is_refused_and_says_how_many_it_had() -> None:
    fitted = fit_market_blend(synthetic_rows(0.0, 1.0, 0.3, n=50), market="PTS")
    assert fitted.status == STATUS_INSUFFICIENT_DATA
    assert not fitted.usable
    assert "50 usable rows" in (fitted.reason or "")
    assert str(MIN_ROWS_FOR_FIT) in (fitted.reason or "")


def test_enough_rows_drawn_from_too_few_slates_is_still_refused() -> None:
    """
    THE GATE THAT MATTERS MOST ON THIS PROJECT'S DATA.

    `leg_correlation` learned this the expensive way: 25 leg rows from ONE
    game produce 2,700 pairs, and six buckets cleared a 200-pair gate on that
    alone with rho that was noise wearing an n of 600. Rows from a single
    slate share pace, officiating and injury news. The independent unit is the
    slate, so a sample that clears the row gate on three nights must not pass.
    """
    rows = synthetic_rows(0.0, 1.0, 0.3, n=2000)
    for row in rows:
        row["game_date"] = "2026-11-13"
    fitted = fit_market_blend(rows, market="PTS")
    assert fitted.status == STATUS_INSUFFICIENT_DATA
    assert fitted.n_rows == 2000
    assert fitted.n_slates == 1
    assert str(MIN_SLATES_FOR_FIT) in (fitted.reason or "")


def test_one_sided_outcomes_are_degenerate_rather_than_fitted() -> None:
    rows = synthetic_rows(0.0, 1.0, 0.3)
    for row in rows:
        row["outcome"] = 1.0
    fitted = fit_market_blend(rows, market="PTS")
    assert fitted.status == STATUS_DEGENERATE
    assert "same outcome" in (fitted.reason or "")


def test_a_model_that_never_disagrees_cannot_identify_c() -> None:
    """
    d is identically zero, so no value of c changes the likelihood. Reporting
    a c of 0.0 here would be indistinguishable from the measured finding that
    disagreement is worthless, and those are very different statements.
    """
    rows = synthetic_rows(0.0, 1.0, 0.3)
    for row in rows:
        row["p_model"] = row["p_market"]
    fitted = fit_market_blend(rows, market="PTS")
    assert fitted.status == STATUS_DEGENERATE
    assert "never disagrees" in (fitted.reason or "")


def test_unusable_rows_are_counted_out_rather_than_imputed() -> None:
    rows = synthetic_rows(0.0, 1.0, 0.25)
    rows[0]["p_market"] = None
    rows[1]["p_model"] = float("nan")
    rows[2]["outcome"] = None
    rows[3]["outcome"] = 0.5          # not a graded result
    rows[4]["p_market"] = 1.4         # outside [0, 1]
    fitted = fit_market_blend(rows, market="PTS")
    assert fitted.n_rows == len(rows) - 5, fitted.as_dict()
    assert fitted.status == STATUS_OK


def test_the_fit_reports_the_last_date_it_saw() -> None:
    fitted = fit_market_blend(synthetic_rows(0.0, 1.0, 0.25), market="PTS")
    assert fitted.fitted_through == (BASE_SLATE + timedelta(days=79)).isoformat()


def test_the_coefficients_carry_their_own_evidence() -> None:
    """
    A coefficient in one place and its sample size in another is how a figure
    fitted on 40 rows comes to be quoted as though fitted on 4,000.
    """
    fitted = fit_market_blend(synthetic_rows(0.0, 1.0, 0.25), market="REB")
    payload = fitted.as_dict()
    assert payload["market"] == "REB"
    assert payload["n_rows"] == 4000
    assert payload["n_slates"] == 80
    assert payload["research_status"] == "RESEARCH_ONLY"


# ---------------------------------------------------------------------------
# the one contract that cannot be guarded in code
# ---------------------------------------------------------------------------


def test_mixing_the_sides_produces_a_wrong_number_silently() -> None:
    """
    THE DOCUMENTED FOOTGUN, asserted so it is not discovered in production.

    Pass the market's P(over) alongside the model's P(under) and `d` holds a
    side flip rather than a disagreement. Nothing in two floats can reveal
    that, so this test exists to state the size of the error: the blend reads
    a model that actually agrees with the market as disagreeing violently.
    """
    p_market_over, p_model_over = 0.60, 0.62
    agreeing = blend_probability(p_market_over, p_model_over, coef(0.0, 1.0, 0.5))
    assert agreeing == pytest.approx(0.61, abs=0.01)

    # Same model opinion, expressed as the UNDER, wrongly passed as the over.
    mixed = blend_probability(p_market_over, 1.0 - p_model_over, coef(0.0, 1.0, 0.5))
    assert abs(mixed - agreeing) > 0.1, (
        "a side mismatch must visibly corrupt the blend; if this ever reads "
        "as a small error, the footgun has become undetectable"
    )
