"""
Tests for src/models/oof.py and the shared out-of-fold calibration path.

Dispersion and calibration both need predictions a model did not train on,
and they were computing them separately AND disagreeing about what
out-of-fold meant. These pin the shared pass and the two properties that
make it safe: a row no fold predicted stays NaN, and a fold that fails
loses its own rows rather than the run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.oof import (
    MIN_USABLE_OOF_ROWS,
    OutOfFoldPredictions,
    chronological_oof_probabilities,
)


def _frame(n: int = 300) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(7)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = (X["a"] + rng.normal(scale=0.5, size=n) > 0).astype(float).to_numpy()
    return X, y


def _honest_fold(X_tr, y_tr, X_va):
    """A stand-in learner: predicts from the training base rate plus signal."""
    base = float(np.mean(y_tr))
    return np.clip(base + 0.2 * np.sign(X_va["a"].to_numpy()), 0.01, 0.99)


# --- the pass itself -----------------------------------------------------


def test_rows_no_fold_predicted_stay_nan():
    """
    TimeSeriesSplit never predicts the first block. Filling it would mean
    handing the calibrator in-sample predictions, which is the failure the
    whole module exists to prevent.
    """
    X, y = _frame()
    oof = chronological_oof_probabilities(_honest_fold, X, y, n_folds=3)

    assert oof.n_folds == 3
    assert oof.frame["prob_over"].isna().any()          # the first block
    assert oof.n_usable < len(X)
    assert oof.usable

    predicted = oof.frame["prob_over"].notna()
    # Everything predicted sits after the first unpredicted row: folds are
    # chronological, so the gap is a prefix and not scattered.
    first_predicted = int(np.argmax(predicted.to_numpy()))
    assert predicted.to_numpy()[first_predicted:].all()


def test_arrays_line_up_labels_with_predictions():
    X, y = _frame()
    oof = chronological_oof_probabilities(_honest_fold, X, y, n_folds=3)
    y_out, p_out = oof.arrays()

    assert len(y_out) == len(p_out) == oof.n_usable
    assert np.isfinite(p_out).all() and np.isfinite(y_out).all()
    ok = oof.frame["prob_over"].notna()
    np.testing.assert_allclose(y_out, oof.frame.loc[ok, "y_over"].to_numpy())


def test_a_failing_fold_loses_its_own_rows_not_the_run(caplog):
    calls = {"n": 0}

    def _flaky(X_tr, y_tr, X_va):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("fold blew up")
        return _honest_fold(X_tr, y_tr, X_va)

    X, y = _frame()
    oof = chronological_oof_probabilities(_flaky, X, y, n_folds=3, market="PTS")
    assert oof.n_folds == 2                 # one fold lost, two kept
    assert oof.n_usable > 0
    assert "stay NaN" in caplog.text


def test_a_fold_returning_the_wrong_number_of_rows_is_discarded(caplog):
    def _wrong_length(X_tr, y_tr, X_va):
        return np.full(len(X_va) + 3, 0.5)

    X, y = _frame()
    oof = chronological_oof_probabilities(_wrong_length, X, y, n_folds=3)
    assert oof.n_folds == 0
    assert oof.usable is False
    assert "discarded" in caplog.text


def test_too_few_rows_abstains_by_name():
    X, y = _frame(n=20)
    oof = chronological_oof_probabilities(_honest_fold, X, y)
    assert oof.n_folds == 0
    assert oof.usable is False
    assert "too few" in (oof.reason or "")


def test_usability_needs_enough_resolved_rows():
    frame = pd.DataFrame({
        "prob_over": [0.5] * (MIN_USABLE_OOF_ROWS - 1) + [np.nan],
        "y_over": [1.0] * MIN_USABLE_OOF_ROWS,
    })
    assert OutOfFoldPredictions(frame, 3).usable is False
    frame.loc[len(frame) - 1, "prob_over"] = 0.5
    assert OutOfFoldPredictions(frame, 3).usable is True


# --- the calibration path it feeds --------------------------------------


@pytest.fixture(scope="module")
def comparison():
    import logging
    import warnings

    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)
    from src.features.builder import build_feature_matrix
    from src.models.compare import compare_models_on_panel, load_comparison_config
    from src.models.data_audit import make_demo_panel

    panel = build_feature_matrix(make_demo_panel())
    result = compare_models_on_panel(
        panel, markets=["PTS"], train_end="2025-01-15",
        validation_end="2025-02-15", cfg=load_comparison_config(),
    )
    logging.disable(logging.NOTSET)
    return result


def test_the_boosted_models_calibrate_from_the_shared_folds(comparison):
    rows = {r["model_name"]: r for r in comparison["summary"]}
    for name in ("xgboost", "catboost"):
        assert rows[name]["calibration_source"] == "shared_out_of_fold"


def test_the_shared_path_sees_more_rows_than_the_old_holdout(comparison):
    """
    The old path refitted on a 70/30 split and calibrated on the last 30%
    only. Chronological folds cover everything after the first block.
    """
    rows = {r["model_name"]: r for r in comparison["summary"]}
    shared = rows["xgboost"]["calibration_rows"]
    holdout = rows["distribution"]["calibration_rows"]
    assert shared > holdout * 2


def test_the_ensemble_calibrates_without_refitting_every_component(comparison):
    """Rebuilding the blend to calibrate it was the most expensive refit."""
    rows = {r["model_name"]: r for r in comparison["summary"]}
    assert rows["ensemble"]["calibration_source"] == "shared_out_of_fold"


def test_calibrated_metrics_are_reported_alongside_the_raw_ones(comparison):
    """
    The harness fitted a calibrator, applied it to the exported predictions,
    and then scored the RAW probability — so nothing could say whether
    calibration helped.
    """
    rows = {r["model_name"]: r for r in comparison["summary"]}
    xgb = rows["xgboost"]
    for field in ("brier_score_calibrated", "log_loss_calibrated",
                  "calibration_error_calibrated"):
        assert xgb[field] is not None

    # On this panel calibration is doing real work on the boosted classifier.
    assert xgb["calibration_error_calibrated"] < xgb["calibration_error"]
    assert xgb["brier_score_calibrated"] < xgb["brier_score"]
