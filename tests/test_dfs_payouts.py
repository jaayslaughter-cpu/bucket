"""DFS pick'em payout structures: breakeven and payout-implied EV.

The arithmetic is exact and checkable by hand, so most of these assert closed
forms rather than approximations. The structures below are TEST FIXTURES with
made-up multiples, labelled as such in their `source`; no real platform's
payout table is shipped or asserted anywhere.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.quant.dfs_payouts import (
    PAYOUT_EV_ABSTAIN,
    PAYOUT_EV_READY,
    DfsPayoutError,
    DfsPayoutStructure,
    evaluate_payout,
    structure_from_mapping,
)
from src.quant.parlay import ParlayLeg, hit_count_distribution

FIXTURE = "TEST FIXTURE — not a real platform's payout table"


def power(n: int, multiple: float) -> DfsPayoutStructure:
    return DfsPayoutStructure(n, {n: multiple}, source=FIXTURE, label=f"{n}-pick power")


# --- structure validation -----------------------------------------------

def test_a_structure_needs_provenance():
    with pytest.raises(DfsPayoutError, match="provenance"):
        DfsPayoutStructure(2, {2: 3.0}, source="  ")


def test_a_structure_must_pay_a_perfect_card():
    """Otherwise it is a transcription error, not a product."""
    with pytest.raises(DfsPayoutError, match="transcription error"):
        DfsPayoutStructure(3, {2: 1.25}, source=FIXTURE)


@pytest.mark.parametrize("n", [0, 1, -3, 2.5])
def test_fewer_than_two_picks_is_not_a_parlay(n):
    with pytest.raises(DfsPayoutError, match="at least 2"):
        DfsPayoutStructure(n, {2: 3.0}, source=FIXTURE)


def test_a_negative_payout_is_refused():
    with pytest.raises(DfsPayoutError, match="not a payout"):
        DfsPayoutStructure(2, {2: 3.0, 1: -1.0}, source=FIXTURE)


def test_a_hit_count_outside_the_card_is_refused():
    with pytest.raises(DfsPayoutError, match="hit count"):
        DfsPayoutStructure(2, {2: 3.0, 5: 1.0}, source=FIXTURE)


def test_no_default_payout_table_exists():
    """Shipping one would be inventing sportsbook data."""
    import src.quant.dfs_payouts as mod

    suspicious = [
        n for n in dir(mod)
        if n.isupper() and any(
            word in n for word in ("PRIZE", "UNDERDOG", "SLEEPER", "MULTIPLIER_TABLE")
        )
    ]
    assert not suspicious, f"a payout table is shipped: {suspicious}"


# --- breakeven ----------------------------------------------------------

@pytest.mark.parametrize("multiple,expected", [(3.0, 1 / 3), (6.0, 1 / 6), (2.0, 0.5)])
def test_breakeven_is_the_reciprocal_of_an_all_or_nothing_payout(multiple, expected):
    assert power(2, multiple).breakeven_joint_probability() == pytest.approx(expected)


def test_a_flex_has_no_single_breakeven_probability():
    """With partial payouts the breakeven is a surface, not a number.

    Returning one anyway would be a category error rather than an
    approximation, so None is the honest answer.
    """
    flex = DfsPayoutStructure(3, {3: 2.25, 2: 1.25}, source=FIXTURE, label="3-pick flex")
    assert flex.is_all_or_nothing is False
    assert flex.breakeven_joint_probability() is None


def test_an_all_or_nothing_structure_is_recognised():
    assert power(2, 3.0).is_all_or_nothing is True


# --- expected value ----------------------------------------------------

def test_power_play_ev_matches_the_closed_form():
    """Two independent legs at 0.60: P(all) = 0.36, EV = 0.36*3 - 1."""
    probs = [0.16, 0.48, 0.36]          # (1-p)^2, 2p(1-p), p^2
    out = evaluate_payout(power(2, 3.0), probs)
    assert out.status == PAYOUT_EV_READY
    assert out.expected_value == pytest.approx(0.36 * 3.0 - 1.0)
    assert out.probability_all_hit == pytest.approx(0.36)
    assert out.edge_vs_breakeven == pytest.approx(0.36 - 1 / 3)


def test_a_flex_is_not_priced_as_a_power_play():
    """The whole reason the full distribution is required.

    3-pick flex paying 2.25x on 3/3 and 1.25x on 2/3, legs at 0.60:
      EV = 0.216*2.25 + 0.432*1.25 - 1
    Pricing it as a power play would use only the 3/3 cell and understate it.
    """
    probs = [0.064, 0.288, 0.432, 0.216]
    flex = DfsPayoutStructure(3, {3: 2.25, 2: 1.25}, source=FIXTURE)
    out = evaluate_payout(flex, probs)

    expected = 0.216 * 2.25 + 0.432 * 1.25 - 1.0
    assert out.expected_value == pytest.approx(expected)

    as_power = evaluate_payout(power(3, 2.25), probs).expected_value
    assert out.expected_value > as_power, (
        "the partial-hit tier contributed nothing, so the flex was priced as a "
        "power play"
    )


def test_a_missing_hit_tier_pays_nothing_rather_than_raising():
    probs = [0.064, 0.288, 0.432, 0.216]
    out = evaluate_payout(power(3, 2.0), probs)
    assert out.expected_value == pytest.approx(0.216 * 2.0 - 1.0)


def test_ev_is_exactly_zero_at_the_breakeven_probability():
    probs = [0.0, 0.0, 1 / 3, 2 / 3]
    probs = [0.0, 0.0, 1 - 1 / 3, 1 / 3]
    out = evaluate_payout(power(3, 3.0), probs)
    assert out.expected_value == pytest.approx(0.0, abs=1e-12)


# --- abstentions --------------------------------------------------------

def test_the_wrong_length_distribution_abstains():
    """Passing only P(all hit) must not be silently accepted."""
    out = evaluate_payout(power(3, 2.25), [0.216])
    assert out.status == PAYOUT_EV_ABSTAIN
    assert "expected 4" in out.reason
    assert out.expected_value is None


def test_a_distribution_that_does_not_sum_to_one_abstains():
    out = evaluate_payout(power(2, 3.0), [0.1, 0.1, 0.1])
    assert out.status == PAYOUT_EV_ABSTAIN
    assert "not 1" in out.reason


def test_a_non_finite_cell_abstains():
    out = evaluate_payout(power(2, 3.0), [0.5, float("nan"), 0.5])
    assert out.status == PAYOUT_EV_ABSTAIN
    assert "non-finite" in out.reason


def test_a_negative_cell_abstains():
    out = evaluate_payout(power(2, 3.0), [-0.1, 0.6, 0.5])
    assert out.status == PAYOUT_EV_ABSTAIN
    assert "negative" in out.reason


# --- the honesty properties --------------------------------------------

def test_every_evaluation_carries_the_calibration_caveat():
    """A positive figure here is the model's claim, not evidence of profit."""
    out = evaluate_payout(power(2, 3.0), [0.16, 0.48, 0.36])
    assert "calibration" in out.disclaimer
    assert "not evidence of profit" in out.disclaimer
    assert out.as_dict()["DISCLAIMER"] == out.disclaimer


def test_the_status_cannot_be_confused_with_a_cleared_market_gate():
    """market_ev_gate's READY must not be reusable as this module's."""
    from src.quant.contracts import GATE_READY

    out = evaluate_payout(power(2, 3.0), [0.16, 0.48, 0.36])
    assert out.status == PAYOUT_EV_READY
    assert out.status != GATE_READY, (
        "a payout-implied figure shares a status string with a de-vigged "
        "two-way EV, so downstream code cannot tell them apart"
    )


def test_no_stake_or_sizing_is_produced():
    out = evaluate_payout(power(2, 3.0), [0.16, 0.48, 0.36]).as_dict()
    assert not [
        k for k in out
        if any(w in k.upper() for w in ("STAKE", "SIZE", "KELLY", "BANKROLL", "WAGER"))
    ], f"a sizing field reached the caller: {sorted(out)}"


# --- integration with the copula ---------------------------------------

def test_it_consumes_hit_count_distribution_directly():
    legs = [ParlayLeg(f"l{i}", 0.6, game_id=f"g{i}") for i in range(3)]
    probs, errs = hit_count_distribution(legs)
    flex = DfsPayoutStructure(3, {3: 2.25, 2: 1.25}, source=FIXTURE)
    out = evaluate_payout(flex, probs, errs)

    assert out.status == PAYOUT_EV_READY
    assert out.expected_value is not None
    assert out.expected_value_stderr is not None and out.expected_value_stderr > 0
    assert out.probability_all_hit == pytest.approx(0.216, abs=0.01)


def test_correlated_legs_change_the_ev():
    """Correlation is why the count distribution is simulated, not multiplied."""
    legs = [ParlayLeg(f"l{i}", 0.6, game_id="same") for i in range(3)]
    rho = np.array([[1.0, 0.5, 0.5], [0.5, 1.0, 0.5], [0.5, 0.5, 1.0]])

    independent = evaluate_payout(power(3, 2.25), hit_count_distribution(legs)[0])
    correlated = evaluate_payout(
        power(3, 2.25), hit_count_distribution(legs, rho)[0]
    )
    assert correlated.expected_value != pytest.approx(
        independent.expected_value, abs=1e-6
    ), "positive correlation left an all-or-nothing EV unchanged"
    # positive correlation raises P(all hit) on same-side legs
    assert correlated.probability_all_hit > independent.probability_all_hit


# --- config loading ----------------------------------------------------

def test_string_keys_from_yaml_are_coerced():
    structure = structure_from_mapping({
        "n_picks": 3, "payouts": {"3": 2.25, "2": 1.25},
        "source": FIXTURE, "label": "3-flex",
    })
    assert structure.payouts == {3: 2.25, 2: 1.25}


def test_a_non_integer_payout_key_is_an_error_not_a_skipped_tier():
    """Dropping a tier silently would misprice every ticket using it."""
    with pytest.raises(DfsPayoutError, match="not a hit count"):
        structure_from_mapping({
            "n_picks": 3, "payouts": {"three": 2.25}, "source": FIXTURE,
        })
