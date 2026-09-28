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
    ProbabilitySource,
    benchmark_fair_probability,
    evaluate_payout,
    evaluate_pickem_entry,
    independent_hit_count_distribution,
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

def test_the_caveat_follows_the_probability_source_not_the_payout():
    """The caveat is source-specific, which is the whole correction.

    An earlier version asserted "calibration" on EVERY evaluation. That was the
    overclaim: it is true of a model-sourced figure and false of one whose legs
    were de-vigged from a sharp benchmark, where the market IS the evidence.
    """
    model = evaluate_pickem_entry(
        power(2, 3.0), [0.6, 0.6], source=ProbabilitySource.MODEL
    )
    assert "calibration" in model.disclaimer
    assert "not evidence of profit" in model.disclaimer

    benchmark = evaluate_pickem_entry(
        power(2, 3.0), [0.6, 0.6], source=ProbabilitySource.SHARP_BENCHMARK
    )
    assert "Market-grounded" in benchmark.disclaimer
    assert "not evidence of profit" not in benchmark.disclaimer

    mixed = evaluate_pickem_entry(
        power(2, 3.0), [0.6, 0.6], source=ProbabilitySource.MIXED
    )
    assert "weakest leg" in mixed.disclaimer

    assert model.as_dict()["DISCLAIMER"] == model.disclaimer
    assert model.as_dict()["PROBABILITY_SOURCE"] == "MODEL"


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


# --- the per-leg synthetic price (the documented worked example) ---------

def test_per_leg_breakeven_is_the_nth_root_and_agrees_with_the_joint_one():
    """A 3-leg power play at 6x: 6^(-1/3) = 55.03%, and 0.5503^3 = 1/6."""
    s3 = power(3, 6.0)
    per_leg = s3.per_leg_breakeven_probability()
    assert per_leg == pytest.approx(6 ** (-1 / 3))
    assert per_leg == pytest.approx(0.550321, abs=1e-6)
    # consistent with the joint threshold, not an alternative to it
    assert per_leg ** 3 == pytest.approx(s3.breakeven_joint_probability())
    assert per_leg ** 3 == pytest.approx(1 / 6)


def test_the_synthetic_american_price_matches_the_worked_example():
    """55.03% -> -122 (unrounded -122.4)."""
    assert power(3, 6.0).per_leg_synthetic_american() == -122


def test_a_flex_has_no_per_leg_breakeven_either():
    flex = DfsPayoutStructure(3, {3: 2.25, 2: 1.25}, source=FIXTURE)
    assert flex.per_leg_breakeven_probability() is None
    assert flex.per_leg_synthetic_american() is None


# --- exact Poisson binomial --------------------------------------------

def test_the_exact_distribution_matches_scipy_poisson_binom():
    from scipy.stats import poisson_binom

    probs = [0.58, 0.61, 0.545, 0.62]
    mine = independent_hit_count_distribution(probs)
    theirs = np.array([poisson_binom(probs).pmf(k) for k in range(len(probs) + 1)])
    assert np.allclose(mine, theirs, atol=1e-12)
    assert mine.sum() == pytest.approx(1.0)


def test_the_exact_distribution_reduces_to_the_binomial_when_legs_are_equal():
    from math import comb

    p, n = 0.6, 3
    mine = independent_hit_count_distribution([p] * n)
    for k in range(n + 1):
        assert mine[k] == pytest.approx(comb(n, k) * p**k * (1 - p) ** (n - k))


def test_the_exact_distribution_agrees_with_the_copula_at_independence():
    """Same question, two methods: exact has no Monte Carlo error."""
    from src.quant.parlay import ParlayLeg, hit_count_distribution

    probs = [0.58, 0.61, 0.545]
    exact = independent_hit_count_distribution(probs)
    legs = [ParlayLeg(f"l{i}", p, game_id=f"g{i}") for i, p in enumerate(probs)]
    simulated, _ = hit_count_distribution(legs, n_sims=400_000, seed=7)
    assert np.allclose(exact, simulated, atol=5e-3)


@pytest.mark.parametrize("bad", [[1.5], [-0.1], [float("nan")]])
def test_a_probability_outside_zero_one_is_refused(bad):
    with pytest.raises(DfsPayoutError, match="not in"):
        independent_hit_count_distribution(bad)


# --- benchmark de-vig ---------------------------------------------------

def test_the_devig_happens_on_the_benchmark_and_removes_the_hold():
    """-110/-110 implies 52.38% each; fair is 50/50."""
    assert benchmark_fair_probability(-110, -110) == pytest.approx(0.5, abs=1e-9)
    assert benchmark_fair_probability(-110, -110, side="under") == pytest.approx(0.5)


def test_an_asymmetric_benchmark_gives_asymmetric_fair_probabilities():
    over = benchmark_fair_probability(-140, 120, side="over")
    under = benchmark_fair_probability(-140, 120, side="under")
    assert over > 0.5 > under
    assert over + under == pytest.approx(1.0, abs=1e-9)


def test_an_unknown_side_is_refused():
    with pytest.raises(DfsPayoutError, match="over"):
        benchmark_fair_probability(-110, -110, side="middle")


def test_it_reuses_the_single_devig_rather_than_reimplementing_it():
    import inspect

    import src.quant.dfs_payouts as mod

    source = inspect.getsource(mod.benchmark_fair_probability)
    assert "multiplicative_devig" in source, (
        "a second de-vig implementation would drift from contracts.devig_two_way"
    )


# --- the entry evaluator (the routing the gate now points at) ----------

def test_a_benchmark_sourced_entry_is_market_grounded_and_says_so():
    structure = power(3, 6.0)
    probs = [
        benchmark_fair_probability(-130, 110),
        benchmark_fair_probability(-125, 105),
        benchmark_fair_probability(-140, 120),
    ]
    out = evaluate_pickem_entry(
        structure, probs, source=ProbabilitySource.SHARP_BENCHMARK
    )
    assert out.status == PAYOUT_EV_READY
    assert out.expected_value == pytest.approx(np.prod(probs) * 6.0 - 1.0, abs=1e-9)
    assert out.probability_source is ProbabilitySource.SHARP_BENCHMARK
    assert "Market-grounded" in out.disclaimer
    assert "calibration" not in out.disclaimer


def test_a_model_sourced_entry_carries_the_calibration_caveat_instead():
    out = evaluate_pickem_entry(
        power(2, 3.0), [0.6, 0.6], source=ProbabilitySource.MODEL
    )
    assert "rests entirely on the model" in out.disclaimer
    assert out.expected_value == pytest.approx(0.36 * 3.0 - 1.0)


def test_an_unrecorded_source_is_flagged_rather_than_assumed():
    out = evaluate_pickem_entry(power(2, 3.0), [0.6, 0.6])
    assert out.probability_source is ProbabilitySource.UNSPECIFIED
    assert "unrecorded" in out.disclaimer


def test_the_entry_carries_the_synthetic_per_leg_price():
    out = evaluate_pickem_entry(
        power(3, 6.0), [0.6, 0.6, 0.6], source=ProbabilitySource.SHARP_BENCHMARK
    )
    assert out.per_leg_breakeven_probability == pytest.approx(0.550321, abs=1e-6)
    assert out.per_leg_synthetic_american == -122


def test_a_leg_count_mismatch_abstains_rather_than_mispricing():
    out = evaluate_pickem_entry(power(3, 6.0), [0.6, 0.6])
    assert out.status == PAYOUT_EV_ABSTAIN
    assert "different products" in out.reason


def test_correlation_lowers_ev_relative_to_assuming_independence():
    """Treating correlated legs as independent overstates a perfect card.

    Negative correlation is the case that matters for the warning in the
    docstring: with positively correlated same-side legs P(all) RISES, so the
    danger is the opposite direction — a parlay of legs that are negatively
    related is overstated by the independence assumption.
    """
    structure = power(3, 6.0)
    probs = [0.6, 0.6, 0.6]
    independent = evaluate_pickem_entry(structure, probs)

    rho = np.array([[1.0, -0.3, -0.3], [-0.3, 1.0, -0.3], [-0.3, -0.3, 1.0]])
    correlated = evaluate_pickem_entry(structure, probs, correlation=rho)

    assert correlated.status == PAYOUT_EV_READY
    assert correlated.probability_all_hit < independent.probability_all_hit
    assert correlated.expected_value < independent.expected_value


def test_a_non_psd_correlation_abstains_with_a_reason():
    bad = np.array([[1.0, 0.99, -0.99], [0.99, 1.0, 0.99], [-0.99, 0.99, 1.0]])
    out = evaluate_pickem_entry(power(3, 6.0), [0.6, 0.6, 0.6], correlation=bad)
    assert out.status == PAYOUT_EV_ABSTAIN
    assert "refused" in out.reason


# --- the gate now routes rather than dead-ending -----------------------

def test_the_gate_names_the_pickem_route_instead_of_calling_ev_undefined():
    from src.quant.contracts import PICKEM_ENTRY_ROUTE, MarketContext, market_ev_gate

    verdict = market_ev_gate(MarketContext(
        game_id="g1", status="VALID", is_pickem=True, payout_multiplier=3.0,
    ))
    # still refuses to price THIS ROW as a two-way market, which is correct
    assert verdict["status"] == "DATA_NOT_AVAILABLE"
    # but no longer claims EV does not exist. Checking for the substring
    # "undefined" alone is not enough: the new reason contains the phrase
    # "EV is not undefined", which an earlier version of this assertion
    # matched and failed on.
    assert "EV is undefined" not in verdict["reason"]
    assert "is not undefined" in verdict["reason"]
    assert "dfs_payouts" in verdict["reason"]
    assert verdict["route"] == PICKEM_ENTRY_ROUTE


def test_a_two_way_row_is_unaffected_and_carries_no_route():
    from src.quant.contracts import MarketContext, market_ev_gate

    verdict = market_ev_gate(MarketContext(
        game_id="g1", status="VALID", over_odds_american=-110,
        under_odds_american=-110, line=25.5,
    ))
    assert verdict["route"] is None
