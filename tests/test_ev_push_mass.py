"""EV on a pushable line must not charge push mass as a loss.

``expected_value_per_unit`` computes ``p * profit - (1 - p)``, which is right
when the only two outcomes are win and lose. On a whole-number line there is
a third: a push returns the stake, contributing 0 to EV. Treating it as a
loss understates EV by exactly the push mass, on BOTH sides.

Worked example — line 25.0, P(over)=0.45, P(under)=0.50, P(push)=0.05, -110:

    correct   0.45 * 0.9091 - 0.50  = -0.0909
    complement 0.45 * 0.9091 - 0.55 = -0.1409   (understated by 0.05)

Enough to move a play across an EV threshold, so it is not cosmetic.
"""

from __future__ import annotations

import pytest

from src.quant.ev_engine import EvEngine
from src.quant.odds_math import (
    american_to_profit_multiple,
    expected_value_per_unit,
    expected_value_two_way,
)


def test_two_outcome_case_still_matches_the_old_formula():
    """With no push mass the two must agree, or this is a behaviour change."""
    for p, american in ((0.55, -110), (0.48, 120), (0.30, 250), (0.72, -300)):
        assert expected_value_two_way(p, 1.0 - p, american) == pytest.approx(
            expected_value_per_unit(p, american)
        )


def test_push_mass_is_not_charged_as_a_loss():
    profit = american_to_profit_multiple(-110)
    ev = expected_value_two_way(0.45, 0.50, -110)
    assert ev == pytest.approx(0.45 * profit - 0.50)
    # the complement form understates by exactly the push mass
    assert ev - expected_value_per_unit(0.45, -110) == pytest.approx(0.05)


def test_engine_uses_push_aware_ev_on_a_whole_line():
    engine = EvEngine()
    out = engine.evaluate_two_way(
        game_id="g1",
        american_a=-110, american_b=-110,
        model_prob_a=0.45, model_prob_b=0.50,   # 0.05 push mass
        line=25.0,
    )
    assert out.status == "OK"
    profit = american_to_profit_multiple(-110)
    by_side = {s.side: s.ev for s in out.sides}
    assert by_side["over"] == pytest.approx(0.45 * profit - 0.50)
    assert by_side["under"] == pytest.approx(0.50 * profit - 0.45)


def test_half_point_line_is_unchanged():
    """A fractional line cannot push, so nothing about it may move."""
    engine = EvEngine()
    out = engine.evaluate_two_way(
        game_id="g1",
        american_a=-110, american_b=-110,
        model_prob_a=0.55, model_prob_b=0.45,
        line=25.5,
    )
    by_side = {s.side: s.ev for s in out.sides}
    assert by_side["over"] == pytest.approx(expected_value_per_unit(0.55, -110))
    assert by_side["under"] == pytest.approx(expected_value_per_unit(0.45, -110))


def test_over_unit_pair_is_computed_as_given_not_rejected():
    """Matches EvEngine policy: an over-unit pair is surfaced, not rescaled.

    tests/test_ev_engine.py asserts the engine returns OK and warns for a
    0.60/0.60 pair. A validation error here would break that contract, which
    is how this was caught.
    """
    profit = american_to_profit_multiple(-110)
    assert expected_value_two_way(0.60, 0.60, -110) == pytest.approx(
        0.60 * profit - 0.60
    )
