"""
Tests for src/quant/parlay.py.

The load-bearing property is that this module refuses more often than it
answers: no price, no ticket; same-game legs with no fitted correlation, no
ticket; a line that can push, no ticket. A parlay evaluator that always
returns a number is the failure mode, not the goal.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.quant.parlay import (
    ParlayError,
    ParlayLeg,
    calibration_amplification,
    copula_joint_probability,
    correlation_matrix,
    estimate_tetrachoric_correlation,
    evaluate_parlay,
    independent_joint_probability,
    parlay_decimal_price,
)

A = ParlayLeg("a", 0.60, -110, game_id="G1", line=24.5, market="PTS", side="over")
B = ParlayLeg("b", 0.55, -110, game_id="G2", line=7.5, market="AST", side="over")
SAME_GAME = ParlayLeg("c", 0.55, -110, game_id="G1", line=8.5, market="REB", side="over")


# --- the copula ----------------------------------------------------------


def test_copula_reduces_to_the_product_under_independence():
    """R = I must reproduce the naive product, so independence is a point in
    the same model rather than a separate code path."""
    joint, stderr = copula_joint_probability([A, B], None)
    product = independent_joint_probability([A, B])
    assert product == pytest.approx(0.33)
    assert abs(joint - product) < 4 * stderr


def test_correlation_moves_the_joint_probability_the_right_way():
    """
    Positive correlation makes both legs land together more often than
    independence; negative correlation less. Getting this backwards is the
    whole reason a naive same-game parlay is mispriced.
    """
    results = {}
    for rho in (-0.4, 0.0, 0.4):
        matrix = correlation_matrix(["a", "c"], {("a", "c"): rho})
        results[rho], _ = copula_joint_probability([A, SAME_GAME], matrix)
    assert results[-0.4] < results[0.0] < results[0.4]


def test_monte_carlo_is_deterministic_for_a_given_seed():
    first, _ = copula_joint_probability([A, B], None, n_sims=50_000, seed=7)
    second, _ = copula_joint_probability([A, B], None, n_sims=50_000, seed=7)
    assert first == second


def test_a_probability_at_the_boundary_is_refused():
    with pytest.raises(ParlayError, match="strictly inside"):
        ParlayLeg("x", 1.0, -110).validate()
    with pytest.raises(ParlayError, match="strictly inside"):
        ParlayLeg("x", 0.0, -110).validate()


# --- the refusals --------------------------------------------------------


def test_same_game_legs_without_a_correlation_are_refused():
    """
    Assuming independence for same-game legs is not a simplification; it is
    a different bet, and it is wrong in the direction that flatters the
    ticket for positively correlated legs.
    """
    result = evaluate_parlay([A, SAME_GAME])
    assert result.status == "DATA_NOT_AVAILABLE"
    assert "share a game" in result.reason
    assert result.expected_value_per_unit is None


def test_same_game_legs_price_once_a_correlation_is_supplied():
    result = evaluate_parlay([A, SAME_GAME], correlation={("a", "c"): 0.25})
    assert result.status == "OK"
    # Positive correlation lifts the ticket above the naive product, and the
    # difference is reported rather than folded in silently.
    assert result.joint_probability > result.independent_probability
    assert result.correlation_effect == pytest.approx(
        result.joint_probability - result.independent_probability
    )


def test_an_unpriced_leg_abstains():
    result = evaluate_parlay([ParlayLeg("a", 0.60, None, game_id="G1", line=24.5), B])
    assert result.status == "DATA_NOT_AVAILABLE"
    assert "No American odds" in result.reason


def test_a_line_that_can_push_is_refused():
    """A push voids the leg and re-prices the ticket; win/lose cannot say that."""
    whole = ParlayLeg("d", 0.55, -110, game_id="G2", line=8.0)
    assert "push" in evaluate_parlay([A, whole]).reason

    unknown = ParlayLeg("e", 0.55, -110, game_id="G2", line=None)
    assert "push" in evaluate_parlay([A, unknown]).reason

    # Explicitly impossible push clears it.
    declared = ParlayLeg("f", 0.55, -110, game_id="G2", line=8.0, model_push_prob=0.0)
    assert evaluate_parlay([A, declared]).status == "OK"


def test_an_inconsistent_correlation_set_is_refused_not_repaired():
    third = ParlayLeg("e", 0.55, -110, game_id="G3", line=5.5)
    result = evaluate_parlay(
        [A, SAME_GAME, third],
        correlation={("a", "c"): 0.9, ("a", "e"): 0.9, ("c", "e"): -0.9},
    )
    assert result.status == "DATA_NOT_AVAILABLE"
    assert "positive semi-definite" in result.reason


def test_structural_refusals():
    assert "at least two legs" in evaluate_parlay([A]).reason
    dupe = ParlayLeg("a", 0.55, -110, game_id="G2", line=7.5)
    assert "unique" in evaluate_parlay([A, dupe]).reason
    with pytest.raises(ParlayError, match="unknown leg pair"):
        correlation_matrix(["a", "b"], {("a", "zzz"): 0.2})


def test_cross_game_independence_is_stated_as_an_assumption():
    result = evaluate_parlay([A, B])
    assert result.status == "OK"
    assert any("independence is an assumption" in w for w in result.warnings)


# --- the arithmetic ------------------------------------------------------


def test_price_and_expected_value_arithmetic():
    result = evaluate_parlay([A, B])
    decimal = parlay_decimal_price([A, B])
    assert decimal == pytest.approx(1.909090909 ** 2, rel=1e-6)
    assert result.decimal_price == pytest.approx(decimal)
    assert result.breakeven_probability == pytest.approx(1.0 / decimal)
    assert result.expected_value_per_unit == pytest.approx(
        result.joint_probability * decimal - 1.0
    )


def test_a_ticket_below_its_breakeven_has_negative_expected_value():
    """Sanity: the sign of EV must follow P vs breakeven, not the price."""
    weak = [
        ParlayLeg("a", 0.50, -110, game_id="G1", line=24.5),
        ParlayLeg("b", 0.50, -110, game_id="G2", line=7.5),
    ]
    result = evaluate_parlay(weak)
    assert result.joint_probability < result.breakeven_probability
    assert result.expected_value_per_unit < 0


# --- the warning that matters most --------------------------------------


def test_calibration_error_compounds_with_leg_count():
    """
    A model a little optimistic on one leg is badly optimistic on five. This
    is the argument against expressing an unproven edge as a parlay.
    """
    over = [
        calibration_amplification(n, per_leg_probability=0.58, per_leg_bias=0.05)[
            "relative_overstatement"
        ]
        for n in (2, 3, 5)
    ]
    assert over == sorted(over)
    assert over[-1] > over[0]
    assert over[-1] > 0.25


def test_correlation_estimation_refuses_a_small_sample():
    """A correlation fitted on a handful of games is noise."""
    rng = np.random.default_rng(0)
    a = rng.integers(0, 2, 30).astype(float)
    b = rng.integers(0, 2, 30).astype(float)
    with pytest.raises(ParlayError, match="at least 200"):
        estimate_tetrachoric_correlation(a, b)

    with pytest.raises(ParlayError, match="differ in length"):
        estimate_tetrachoric_correlation([1.0] * 300, [1.0] * 299)


def test_correlation_estimation_recovers_a_planted_dependence():
    """With enough paired outcomes it finds the latent correlation."""
    rng = np.random.default_rng(11)
    n, rho = 20_000, 0.5
    chol = np.linalg.cholesky(np.array([[1.0, rho], [rho, 1.0]]))
    z = rng.standard_normal((n, 2)) @ chol.T
    a = (z[:, 0] <= 0.0).astype(float)
    b = (z[:, 1] <= 0.0).astype(float)
    assert estimate_tetrachoric_correlation(a, b) == pytest.approx(rho, abs=0.08)


def test_module_cannot_place_or_size_a_bet():
    import ast
    import inspect

    from src.quant import parlay

    tree = ast.parse(inspect.getsource(parlay))
    imported: set[str] = set()
    identifiers: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
        elif isinstance(node, ast.Name):
            identifiers.add(node.id.lower())
        elif isinstance(node, ast.Attribute):
            identifiers.add(node.attr.lower())

    for network in ("requests", "http", "httpx", "urllib", "socket"):
        assert network not in imported
    for banned in ("kelly", "stake_size", "place_bet", "submit_order"):
        assert not [i for i in identifiers if banned in i]


# --- cubic review, PR #3: the copula's matrix must be validated -------------


def _two_legs(prob: float = 0.55):
    return [
        ParlayLeg(leg_id="a", model_prob=prob, american=-110),
        ParlayLeg(leg_id="b", model_prob=prob, american=-110),
    ]


def test_an_asymmetric_correlation_matrix_is_refused():
    """Cholesky reads only the LOWER triangle, so an asymmetric matrix
    factorises happily and silently discards whatever the caller wrote above
    the diagonal. Measured: an asymmetric (0.0 upper, 0.6 lower) matrix
    returned the rho=0.6 answer to machine precision."""
    with pytest.raises(ParlayError, match="not symmetric"):
        copula_joint_probability(
            _two_legs(), correlation=np.array([[1.0, 0.0], [0.6, 1.0]]), n_sims=2000,
        )


def test_a_non_unit_diagonal_is_refused():
    """The thresholds are standard-normal quantiles of each leg's model_prob,
    so a non-unit variance changes every marginal. Measured: a diagonal of 2.0
    returned 0.33454 where the correct answer is 0.40433."""
    with pytest.raises(ParlayError, match="diagonal"):
        copula_joint_probability(
            _two_legs(), correlation=np.array([[2.0, 0.6], [0.6, 2.0]]), n_sims=2000,
        )


def test_a_non_finite_correlation_is_refused_rather_than_priced_at_zero():
    """The dangerous one: a NaN propagated to a joint probability of exactly
    0.0 with no error, which reads as an impossible parlay rather than a
    missing correlation."""
    for bad in (np.nan, np.inf):
        with pytest.raises(ParlayError, match="non-finite"):
            copula_joint_probability(
                _two_legs(), correlation=np.array([[1.0, bad], [bad, 1.0]]), n_sims=2000,
            )


def test_an_out_of_range_correlation_is_refused():
    with pytest.raises(ParlayError, match=r"outside \[-1, 1\]"):
        copula_joint_probability(
            _two_legs(), correlation=np.array([[1.0, 1.5], [1.5, 1.0]]), n_sims=2000,
        )


def test_valid_matrices_still_price_and_independence_matches_the_product():
    """The guard must not reject the cases it exists to protect."""
    legs = _two_legs()
    correlated, _ = copula_joint_probability(
        legs, correlation=np.array([[1.0, 0.6], [0.6, 1.0]]), n_sims=200_000, seed=1,
    )
    independent, se = copula_joint_probability(
        legs, correlation=np.eye(2), n_sims=200_000, seed=1,
    )

    assert correlated > independent, "positive rho must raise the joint probability"
    # Independence is the product, within Monte Carlo error.
    assert independent == pytest.approx(0.55 * 0.55, abs=4 * se)
