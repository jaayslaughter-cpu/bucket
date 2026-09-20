"""Unit tests for model comparison plumbing (no fabricated real analysis)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.ensemble import renormalize_weights
from src.models.exports import DETAIL_COLS, SUMMARY_COLS, write_comparison_exports
from src.models.line_probs import classifier_over_under, discrete_over_under_push, is_whole_number_line
from src.models.walk_forward import expanding_window_splits, fixed_cutoff_split, sort_by_game_date


def test_whole_number_push_and_half_line():
    assert is_whole_number_line(25.0)
    assert not is_whole_number_line(25.5)

    whole = discrete_over_under_push(25.0, 25.0, family="poisson")
    assert whole["status"] == "OK"
    assert whole["probability_push"] is not None and whole["probability_push"] > 0
    total = whole["probability_over"] + whole["probability_under"] + whole["probability_push"]
    assert total == pytest.approx(1.0, abs=1e-5)

    half = discrete_over_under_push(25.0, 25.5, family="poisson")
    assert half["probability_push"] == 0.0
    assert half["probability_over"] + half["probability_under"] == pytest.approx(1.0, abs=1e-5)


def test_classifier_under_complements_over():
    res = classifier_over_under(0.62, 24.5)
    assert res["probability_over"] == pytest.approx(0.62)
    assert res["probability_under"] == pytest.approx(0.38)
    assert res["probability_push"] is None


def test_chronological_split_no_leakage():
    dates = pd.date_range("2024-11-01", periods=200, freq="D")
    df = pd.DataFrame({"GAME_DATE": dates, "x": range(200)})
    split = fixed_cutoff_split(df, train_end="2025-02-01", validation_end="2025-03-01")
    train = df.loc[split.train_idx]
    val = df.loc[split.validation_idx]
    assert train["GAME_DATE"].max() <= pd.Timestamp("2025-02-01")
    assert val["GAME_DATE"].min() > pd.Timestamp("2025-02-01")
    assert val["GAME_DATE"].max() <= pd.Timestamp("2025-03-01")


def test_expanding_windows_are_ordered():
    dates = pd.date_range("2024-10-01", periods=400, freq="D")
    df = pd.DataFrame({"GAME_DATE": dates})
    splits = expanding_window_splits(df, min_train_rows=50, validation_days=20, step_days=30, holdout_days=30)
    assert splits
    for s in splits:
        assert s.train_end < s.validation_start


def test_ensemble_weight_renormalization():
    w, warnings = renormalize_weights({"catboost": 0.5, "xgboost": 0.3, "distribution": 0.2}, {"catboost", "distribution"})
    assert set(w) == {"catboost", "distribution"}
    assert sum(w.values()) == pytest.approx(1.0)
    assert warnings


def test_export_schema_and_no_secrets(tmp_path):
    result = {
        "summary": [{"target_market": "PTS", "model_name": "distribution", "n_predictions": 1, "brier_score": 0.2}],
        "predictions": [{"event_id": "1", "player_id": "2", "model_name": "distribution", "target_market": "PTS"}],
        "feature_importance": [],
        "calibration": [],
        "winners": {"PTS": {"winner": "distribution", "brier_score": 0.2}},
    }
    manifest = write_comparison_exports(result, output_dir=tmp_path, demo=True)
    demo = tmp_path / "demo"
    assert (demo / "model_comparison_summary.csv").exists()
    assert (demo / "predictions_detailed.csv").exists()
    assert (demo / "live_prediction_template.csv").exists()
    assert (demo / "download_manifest.json").exists()
    text = (demo / "model_comparison_summary.csv").read_text(encoding="utf-8")
    assert "API_KEY" not in text
    assert "PASSWORD" not in text
    summary = pd.read_csv(demo / "model_comparison_summary.csv")
    for c in SUMMARY_COLS:
        assert c in summary.columns
    detail = pd.read_csv(demo / "predictions_detailed.csv")
    for c in DETAIL_COLS:
        assert c in detail.columns
    assert "prediction_timestamp_pt" in detail.columns
    assert "game_start_pt" in detail.columns
    assert manifest["demo_mode"] is True
    assert manifest["timezone_display"] == "America/Los_Angeles"


def test_pacific_display_helpers():
    from datetime import datetime, timezone

    from src.utils.timezones import DISPLAY_TZ_NAME, format_pacific_iso, to_pacific

    utc = datetime(2026, 1, 15, 8, 0, 0, tzinfo=timezone.utc)
    pt = to_pacific(utc)
    assert DISPLAY_TZ_NAME == "America/Los_Angeles"
    assert pt.hour == 0  # 08:00 UTC = 00:00 PST
    assert format_pacific_iso(utc) is not None
    assert "-08:00" in format_pacific_iso(utc) or "-07:00" in format_pacific_iso(utc)


def test_shift_features_not_same_game_target():
    """Rolling builder must expose LAST_INCLUDED_GAME_DATE before GAME_DATE."""
    from src.features.builder import assert_no_lookahead, build_feature_matrix
    from src.models.data_audit import make_demo_panel

    raw = make_demo_panel(n_players=3, n_games=15)
    feats = build_feature_matrix(raw)
    assert "fatigue_multiplier" in feats.columns
    assert "PTS_L2" in feats.columns
    assert_no_lookahead(feats)


def test_distribution_probs_sum_to_one():
    from src.features.builder import build_feature_matrix
    from src.models.data_audit import make_demo_panel
    from src.models.distribution_adapter import DistributionPropModel
    from src.models.labels import attach_research_over_labels

    panel = attach_research_over_labels(build_feature_matrix(make_demo_panel(n_players=2, n_games=20)), stat="PTS")
    model = DistributionPropModel(target_market="PTS")
    model.fit(panel)
    preds = model.predict_rows(panel.dropna(subset=["RESEARCH_LINE", "PTS_L2"]).head(10))
    for p in preds:
        if p.probability_over is None:
            continue
        push = p.probability_push or 0.0
        assert p.probability_over + p.probability_under + push == pytest.approx(1.0, abs=1e-4)
