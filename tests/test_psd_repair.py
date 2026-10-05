"""Repairing a correlation matrix that describes no joint distribution.

`parlay.correlation_matrix` assembles pairwise estimates fitted on separate
buckets. Nothing makes a set of pairwise numbers mutually consistent, so
`_validated_cholesky` raised and the ticket could not be priced — and the
refusal asked a human to reconcile the estimates by hand.

The repair is off by default, reported when used, and bounded: the tests below
spend most of their effort on the bound, because a projection that has to move
an entry by 0.3 is not cleaning up noise.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.quant.parlay import ParlayError, ParlayLeg, copula_joint_probability
from src.quant.psd_repair import (
    DEFAULT_MAX_SHIFT,
    PsdRepairError,
    is_psd,
    nearest_correlation,
)

#: Three legs, each pairwise plausible, jointly impossible.
IMPOSSIBLE = np.array([
    [1.00, 0.75, 0.75],
    [0.75, 1.00, -0.40],
    [0.75, -0.40, 1.00],
])
#: The same shape but mild enough that a projection barely moves it.
MILD = np.array([
    [1.00, 0.60, 0.60],
    [0.60, 1.00, -0.30],
    [0.60, -0.30, 1.00],
])


def test_the_premise_an_ordinary_pairwise_matrix_is_not_a_distribution():
    """Not a contrived input: three individually reasonable numbers."""
    assert np.linalg.eigvalsh(IMPOSSIBLE).min() < 0
    assert not is_psd(IMPOSSIBLE)
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.cholesky(IMPOSSIBLE)


def test_a_valid_matrix_is_returned_untouched_and_says_so():
    valid = np.array([[1.0, 0.3], [0.3, 1.0]])
    out = nearest_correlation(valid)
    assert out.was_psd and not out.repaired
    assert out.max_entry_shift == 0.0
    np.testing.assert_allclose(out.matrix, valid)


def test_a_mild_inconsistency_is_repaired_and_becomes_factorisable():
    out = nearest_correlation(MILD)
    assert not out.was_psd
    assert out.repaired
    assert out.min_eigenvalue_before < 0 < out.min_eigenvalue_after
    np.linalg.cholesky(out.matrix)          # the whole point
    assert out.max_entry_shift < DEFAULT_MAX_SHIFT


def test_the_repair_keeps_a_unit_diagonal_and_stays_symmetric():
    out = nearest_correlation(MILD)
    np.testing.assert_allclose(np.diag(out.matrix), 1.0)
    np.testing.assert_allclose(out.matrix, out.matrix.T)
    assert np.abs(out.matrix[~np.eye(3, dtype=bool)]).max() < 1.0


def test_a_large_repair_abstains_rather_than_papering_over_it():
    """
    The bound. This matrix needs an entry moved 0.1554, which is a real
    disagreement between estimates and not floating-point noise.
    """
    with pytest.raises(PsdRepairError, match="moves an entry by"):
        nearest_correlation(IMPOSSIBLE)
    out = nearest_correlation(IMPOSSIBLE, max_shift=None)
    assert out.max_entry_shift == pytest.approx(0.1554, abs=1e-3)


def test_the_refusal_names_the_number_and_what_to_do():
    try:
        nearest_correlation(IMPOSSIBLE)
    except PsdRepairError as exc:
        message = str(exc)
    assert "0.15" in message
    assert "eigenvalue" in message
    assert "refit" in message or "drop a leg" in message


def test_shrinkage_is_counted_as_movement_which_is_counter_intuitive():
    """
    Pinned because it surprises: shrinkage moves every off-diagonal toward
    zero, and the shift is measured against what was SUPPLIED, so asking for
    less trust reports a larger distance.
    """
    plain = nearest_correlation(MILD)
    shrunk = nearest_correlation(MILD, shrinkage=0.10)
    assert shrunk.max_entry_shift > plain.max_entry_shift
    assert shrunk.shrinkage == 0.10
    assert shrunk.min_eigenvalue_after > plain.min_eigenvalue_after


def test_the_result_is_positive_definite_not_merely_semi_definite():
    """`cholesky` wants definite; clipping to exactly 0 can still fail later."""
    out = nearest_correlation(MILD)
    assert out.min_eigenvalue_after > 0.0


def test_a_non_finite_entry_is_a_missing_estimate_not_a_correlation():
    bad = np.array([[1.0, np.nan], [np.nan, 1.0]])
    with pytest.raises(PsdRepairError, match="non-finite"):
        nearest_correlation(bad)


def test_a_non_square_input_is_refused():
    with pytest.raises(PsdRepairError, match="square"):
        nearest_correlation(np.zeros((2, 3)))


def test_shrinkage_outside_zero_to_one_is_refused():
    with pytest.raises(PsdRepairError, match="shrinkage"):
        nearest_correlation(MILD, shrinkage=1.5)


# --- through the parlay path ------------------------------------------------

def _legs(n: int = 3):
    return [ParlayLeg(leg_id=c, model_prob=0.55) for c in "ABCDE"[:n]]


def test_nothing_repairs_by_default():
    """
    `correlation_matrix`'s stance is unchanged: an inconsistent set describes
    no joint distribution and silently repairing would invent dependence.
    """
    with pytest.raises(ParlayError, match="not positive semi-definite"):
        copula_joint_probability(_legs(), IMPOSSIBLE, n_sims=2_000)


def test_the_refusal_now_names_the_way_out():
    try:
        copula_joint_probability(_legs(), IMPOSSIBLE, n_sims=2_000)
    except ParlayError as exc:
        assert "repair=True" in str(exc)


def test_an_opt_in_repair_prices_a_mildly_inconsistent_ticket():
    probability, stderr = copula_joint_probability(
        _legs(), MILD, n_sims=40_000, seed=7, repair=True
    )
    assert 0.0 < probability < 1.0
    assert stderr > 0.0
    independent, _ = copula_joint_probability(_legs(), None, n_sims=40_000, seed=7)
    assert probability > independent, (
        "positive dependence should raise P(all win) above the product"
    )


def test_an_opt_in_repair_still_refuses_a_large_one():
    with pytest.raises(ParlayError, match="cannot be repaired"):
        copula_joint_probability(_legs(), IMPOSSIBLE, n_sims=2_000, repair=True)


def test_hit_count_distribution_takes_the_same_switch():
    import inspect

    from src.quant.parlay import hit_count_distribution

    assert "repair" in inspect.signature(hit_count_distribution).parameters
    counts, _stderr = hit_count_distribution(
        _legs(), MILD, n_sims=20_000, seed=3, repair=True
    )
    assert len(counts) == 4
    assert counts.sum() == pytest.approx(1.0, abs=1e-6)


def test_a_valid_matrix_is_unaffected_by_asking_for_repair():
    valid = np.array([[1.0, 0.2, 0.1], [0.2, 1.0, 0.15], [0.1, 0.15, 1.0]])
    off, _ = copula_joint_probability(_legs(), valid, n_sims=30_000, seed=11)
    on, _ = copula_joint_probability(_legs(), valid, n_sims=30_000, seed=11, repair=True)
    assert off == pytest.approx(on, abs=1e-12)
