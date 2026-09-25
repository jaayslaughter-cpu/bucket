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


# --- cubic review, PR #3: line-aware OOF sample identity --------------------


def _oof(frame, sample=None):
    from src.models.oof import OutOfFoldPredictions

    return OutOfFoldPredictions(frame, n_folds=3, sample=sample)


class _Stub:
    def __init__(self, frame, sample=None):
        self.oof = _oof(frame, sample)


def _source_frame(prob: float, n: int = 100):
    y = np.tile([0.0, 1.0], n // 2)
    return pd.DataFrame({"prob_over": np.full(n, prob), "y_over": y},
                        index=pd.RangeIndex(n))


def _augmented_frame(n: int = 100, offsets: int = 9):
    """What line_aware produces: one row per (source row, candidate line),
    reset to a fresh RangeIndex — which collides with the source-row one."""
    y = np.tile([0.0, 1.0], n // 2)
    return pd.DataFrame(
        {"prob_over": np.linspace(0.01, 0.99, n * offsets),
         "y_over": np.repeat(y, offsets)},
        index=pd.RangeIndex(n * offsets),
    )


def test_ensemble_declines_to_blend_oof_frames_over_different_samples():
    """line_aware trains on source x candidate-line pairs and resets to a
    fresh RangeIndex, so its labels collide with the source-row index every
    other component carries — both start at 0 over different universes.
    Intersecting by label paired augmented row i with source row i and built a
    blend whose probabilities and labels came from different rows."""
    from src.models.ensemble import EnsemblePropModel

    ens = EnsemblePropModel(
        {"xgboost": _Stub(_source_frame(0.6)),
         "catboost": _Stub(_source_frame(0.4)),
         "line_aware": _Stub(_augmented_frame(), "augmented_lines:PTS")},
        weights={"xgboost": 0.5, "catboost": 0.3, "line_aware": 0.2},
    )

    assert ens.oof is None, "blended mismatched samples instead of declining"


def test_ensemble_still_blends_when_every_component_shares_the_sample():
    """The guard must not disable the fast path for the ordinary case."""
    from src.models.ensemble import EnsemblePropModel

    ens = EnsemblePropModel(
        {"xgboost": _Stub(_source_frame(0.6)), "catboost": _Stub(_source_frame(0.4))},
        weights={"xgboost": 0.5, "catboost": 0.5},
    )

    result = ens.oof
    assert result is not None
    assert len(result.frame) == 100
    assert result.frame["prob_over"].iloc[0] == pytest.approx(0.5)
    assert result.sample is None


def test_line_aware_labels_its_oof_as_an_augmented_sample():
    """The ensemble guard only works because line_aware says what its rows are.
    Without the label the frames look interchangeable."""
    from src.models.line_aware import LineAwarePropModel

    class _Inner:
        oof = None

    model = LineAwarePropModel.__new__(LineAwarePropModel)
    model.stat = "PTS"
    model.model = _Inner()
    assert model.oof is None, "no inner oof must stay None"

    model.model.oof = _oof(_augmented_frame())
    labelled = model.oof
    assert labelled.sample == "augmented_lines:PTS"
    assert labelled.n_folds == 3, "wrapping must not lose the fold count"
