"""Tests for the end-to-end pipeline audit and the fit-failure record."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts.audit_pipeline import (
    AuditFinding,
    check_calibrated_metrics_exported,
    check_join_preserves_rows,
    check_metrics_reproduce,
    check_no_component_vanished,
    check_pbp_completeness,
    check_roi_is_not_invented,
)


def _panel(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        # Unique (game, player) pairs: a real panel has one row per player
        # per game, and the audit's probe must not invent duplicates.
        "PLAYER_ID": (np.arange(n) % 12).astype(str),
        "GAME_ID": (100 + np.arange(n) // 12).astype(str),
        "GAME_DATE": pd.date_range("2025-10-21", periods=n, freq="6h"),
        "FGA": rng.integers(5, 20, n).astype(float),
    })


# --- ingestion --------------------------------------------------------------


def test_join_check_catches_a_reordering_join(monkeypatch):
    """Values staying attached to their rows is not enough: a join that
    returns them in a different order corrupts any positional assignment."""
    import src.features.pbp as pbp_mod

    real = pbp_mod.attach_pbp_rolling_features
    assert check_join_preserves_rows(_panel())

    def _shuffling(panel, summaries, **kw):
        out = real(panel, summaries, **kw)
        return out.iloc[::-1]

    monkeypatch.setattr("src.features.pbp.attach_pbp_rolling_features", _shuffling)
    with pytest.raises(AuditFinding, match="reordered"):
        check_join_preserves_rows(_panel())


def test_join_check_catches_a_duplicating_join(monkeypatch):
    import src.features.pbp as pbp_mod

    real = pbp_mod.attach_pbp_rolling_features

    def _duplicating(panel, summaries, **kw):
        out = real(panel, summaries, **kw)
        return pd.concat([out, out.head(1)])

    monkeypatch.setattr("src.features.pbp.attach_pbp_rolling_features", _duplicating)
    with pytest.raises(AuditFinding, match="row count changed"):
        check_join_preserves_rows(_panel())


def test_pbp_completeness_fails_on_a_thinned_event_log():
    panel = _panel()
    events = []
    for gid, g in panel.groupby("GAME_ID"):
        for _ in range(int(g["FGA"].sum())):
            events.append({"gameId": gid, "actionType": "2pt"})
    full = pd.DataFrame(events)
    assert check_pbp_completeness(panel, full)

    keep = full.groupby("gameId").cumcount() < (
        full.groupby("gameId")["gameId"].transform("size") * 0.7
    )
    thinned = full[keep].reset_index(drop=True)
    with pytest.raises(AuditFinding, match="incomplete"):
        check_pbp_completeness(panel, thinned)


# --- silent component loss --------------------------------------------------


def _write_predictions(tmp_path, models):
    df = pd.DataFrame({
        "target_market": ["PTS"] * len(models) * 10,
        "model_name": [m for m in models for _ in range(10)],
        "actual_stat_value": np.tile(np.arange(10, dtype=float), len(models)),
        "prediction_mean": np.tile(np.arange(10, dtype=float), len(models)),
    })
    df.to_parquet(tmp_path / "predictions_detailed.parquet", index=False)
    return df


def test_a_weighted_component_that_produced_nothing_is_a_failure(tmp_path):
    """This is exactly how CatBoost ran at 0.50 of the configured weight while
    contributing nothing."""
    cfg = {"ensemble_weights": {"catboost": 0.5, "xgboost": 0.3, "distribution": 0.2}}
    _write_predictions(tmp_path, ["xgboost", "distribution", "ensemble"])
    with pytest.raises(AuditFinding, match="catboost"):
        check_no_component_vanished(tmp_path, cfg)

    _write_predictions(tmp_path, ["catboost", "xgboost", "distribution", "ensemble"])
    assert check_no_component_vanished(tmp_path, cfg)


def test_a_fitted_but_unweighted_component_is_reported_not_failed(tmp_path):
    cfg = {"ensemble_weights": {"xgboost": 0.6, "distribution": 0.4}}
    _write_predictions(tmp_path, ["xgboost", "distribution", "line_aware", "ensemble"])
    detail = check_no_component_vanished(tmp_path, cfg)
    assert "line_aware" in detail


# --- output integrity -------------------------------------------------------


def test_metrics_check_catches_a_summary_that_does_not_reproduce(tmp_path):
    _write_predictions(tmp_path, ["xgboost"])
    good = pd.DataFrame({"target_market": ["PTS"], "model_name": ["xgboost"],
                         "mae": [0.0], "rmse": [0.0]})
    good.to_csv(tmp_path / "model_comparison_summary.csv", index=False)
    assert check_metrics_reproduce(tmp_path)

    bad = good.assign(mae=[1.23], rmse=[4.56])
    bad.to_csv(tmp_path / "model_comparison_summary.csv", index=False)
    with pytest.raises(AuditFinding, match="do not reproduce"):
        check_metrics_reproduce(tmp_path)


def test_calibrated_columns_must_be_exported(tmp_path):
    pd.DataFrame({"target_market": [], "model_name": [], "brier_score": []}).to_csv(
        tmp_path / "model_comparison_summary.csv", index=False
    )
    with pytest.raises(AuditFinding, match="omits"):
        check_calibrated_metrics_exported(tmp_path)

    pd.DataFrame({
        "brier_score": [], "brier_score_calibrated": [],
        "calibration_error_calibrated": [],
    }).to_csv(tmp_path / "model_comparison_summary.csv", index=False)
    assert check_calibrated_metrics_exported(tmp_path)


def test_an_roi_column_in_the_comparison_outputs_is_refused(tmp_path):
    """The comparison has no odds and no settled bets, so any ROI it reports
    was invented."""
    assert check_roi_is_not_invented(tmp_path)
    pd.DataFrame({"model_name": ["xgboost"], "roi": [0.07]}).to_csv(
        tmp_path / "pocket.csv", index=False
    )
    with pytest.raises(AuditFinding, match="fabricated"):
        check_roi_is_not_invented(tmp_path)


# --- the pipeline's own record ----------------------------------------------


def test_a_failed_component_is_recorded_not_just_logged():
    """Skipping silently is how a run loses half its configured ensemble
    weight without leaving a trace in the outputs."""
    from src.models.compare import compare_models_on_panel, load_comparison_config

    rng = np.random.default_rng(1)
    n = 900
    df = pd.DataFrame({
        "PLAYER_ID": rng.integers(0, 30, n).astype(str),
        "GAME_ID": np.arange(n).astype(str),
        "GAME_DATE": pd.date_range("2024-10-01", periods=n, freq="6h"),
        "TEAM_ABBREVIATION": rng.choice(["AAA", "BBB"], n),
        "OPPONENT_ABBREVIATION": rng.choice(["CCC", "DDD"], n),
        "SEASON": "2024-25",
        "PTS_L5": rng.normal(15, 4, n), "PTS_L10": rng.normal(15, 4, n),
        "PTS_SEASON": rng.normal(15, 4, n), "PTS_BASELINE": rng.normal(15, 4, n),
        "PTS_L2": rng.normal(15, 4, n), "MIN_L5": rng.normal(28, 6, n),
        "MIN_L10": rng.normal(28, 6, n), "MIN_SEASON": rng.normal(28, 6, n),
        "IS_HOME": rng.integers(0, 2, n).astype(float),
    })
    df["PTS"] = df["PTS_L10"] + rng.normal(0, 5, n)
    cut = df["GAME_DATE"].iloc[int(n * 0.7)]
    result = compare_models_on_panel(
        df, markets=["PTS"], train_end=str(cut.date()),
        validation_end=str(df["GAME_DATE"].max().date()),
        cfg=load_comparison_config(),
    )
    assert "fit_failures" in result
    assert "ensemble_composition" in result
    for entry in result["ensemble_composition"]:
        assert set(entry) >= {
            "target_market", "configured", "effective",
            "failed_to_fit", "fitted_but_unweighted",
        }
        # The effective blend sums to 1 when anything carries weight.
        if entry["effective"]:
            assert sum(entry["effective"].values()) == pytest.approx(1.0, abs=1e-5)


def test_the_export_records_failures_in_the_manifest(tmp_path):
    from src.models.exports import write_comparison_exports

    result = {
        "summary": [{"target_market": "PTS", "model_name": "xgboost"}],
        "predictions": [], "feature_importance": [], "calibration": [],
        "model_vs_line": [], "edge_buckets": [], "confidence_verdicts": [],
        "winners": {},
        "fit_failures": [{
            "target_market": "PTS", "model_name": "catboost",
            "configured_ensemble_weight": 0.5, "error": "ValueError: boom",
        }],
        "ensemble_composition": [{
            "target_market": "PTS",
            "configured": {"catboost": 0.5, "xgboost": 0.3, "distribution": 0.2},
            "effective": {"xgboost": 0.6, "distribution": 0.4},
            "failed_to_fit": ["catboost"], "fitted_but_unweighted": [],
        }],
    }
    manifest = write_comparison_exports(result, output_dir=tmp_path)
    assert manifest["components_failed_to_fit"] == ["PTS/catboost"]
    assert manifest["ensemble_matches_config"] is False
    recorded = json.loads((tmp_path / "fit_failures.json").read_text())
    assert recorded[0]["model_name"] == "catboost"
    assert recorded[0]["configured_ensemble_weight"] == 0.5
