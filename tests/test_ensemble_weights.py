"""Tests for the fitted ensemble weights."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.fit_ensemble_weights import optimise_weights


def _probs(n: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n).astype(float)
    # good: tracks the label. noise: independent of it.
    good = np.clip(0.5 + (y - 0.5) * 0.6 + rng.normal(0, 0.12, n), 0.02, 0.98)
    noise = np.clip(rng.normal(0.5, 0.12, n), 0.02, 0.98)
    return pd.DataFrame({"good": good, "noise": noise}), y


def test_a_useless_component_is_given_almost_no_weight():
    probs, y = _probs()
    weights, scores = optimise_weights(probs, y)
    assert weights["good"] > 0.85
    assert weights["noise"] < 0.15
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-3)


def test_the_fitted_blend_beats_equal_weighting():
    probs, y = _probs()
    _, scores = optimise_weights(probs, y)
    assert scores["fitted_log_loss"] < scores["equal_weight_log_loss"]


def test_two_equally_good_but_independent_components_are_both_used():
    """Diversity is why a weaker model can still earn weight."""
    rng = np.random.default_rng(3)
    n = 4000
    y = rng.integers(0, 2, n).astype(float)
    a = np.clip(0.5 + (y - 0.5) * 0.5 + rng.normal(0, 0.20, n), 0.02, 0.98)
    b = np.clip(0.5 + (y - 0.5) * 0.5 + rng.normal(0, 0.20, n), 0.02, 0.98)
    weights, _ = optimise_weights(pd.DataFrame({"a": a, "b": b}), y)
    assert min(weights.values()) > 0.25, weights


def test_weights_are_non_negative_and_sum_to_one():
    probs, y = _probs()
    probs["third"] = np.clip(probs["noise"] * 0.9, 0.02, 0.98)
    weights, _ = optimise_weights(probs, y)
    assert all(v >= 0 for v in weights.values())
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-3)


def test_too_few_complete_rows_is_refused_rather_than_fitted():
    probs, y = _probs(n=100)
    with pytest.raises(ValueError, match="DATA_NOT_AVAILABLE"):
        optimise_weights(probs, y)


def test_rows_missing_any_component_are_dropped_not_filled():
    """A blend needs every member to have answered; filling one invents a
    probability nobody produced."""
    probs, y = _probs()
    probs.loc[:1199, "noise"] = np.nan
    weights, scores = optimise_weights(probs, y)
    assert scores["n_rows"] == 800
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-3)


# --- the shipped config -----------------------------------------------------


def test_line_aware_carries_weight_in_the_shipped_config():
    from src.models.compare import load_comparison_config

    weights = load_comparison_config()["ensemble_weights"]
    assert "line_aware" in weights, "line_aware fits but was excluded from the blend"
    assert weights["line_aware"] > 0
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)


def test_every_weighted_component_can_actually_be_built():
    """A weight for a component build_components never produces would be
    renormalised away silently."""
    from src.models.compare import build_components, load_comparison_config

    cfg = load_comparison_config()
    feature_cols = ["PTS_L5", "PTS_L10", "MIN_L5"]
    built = set(build_components("PTS", feature_cols, cfg))
    weighted = {k for k, v in cfg["ensemble_weights"].items() if v > 0}
    missing = sorted(weighted - built)
    assert not missing, f"weighted but never built: {missing}"
