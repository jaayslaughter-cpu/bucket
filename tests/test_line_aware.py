"""
Line-aware classifiers — DATA_GAPS item 13.

The defect: `over_hit` is P(stat > RESEARCH_LINE) and the line is not a
model input, so a fitted classifier returns the identical probability at
every line. These tests pin the fix and, just as importantly, the leak it
would be easy to introduce while making it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.builder import build_feature_matrix
from src.models.data_audit import make_demo_panel
from src.models.line_aware import (
    DEFAULT_LINE_OFFSETS,
    LINE_FEATURE_COLS,
    SOURCE_ROW_COL,
    LineAwarePropModel,
    LineLeakageError,
    assert_lines_are_pregame,
    assert_no_augmented_row_straddles,
    attach_line_features,
    attach_pregame_scale,
    augment_lines,
    label_at_line,
)

BASE_COLS = ["PTS_L5", "PTS_L10", "PTS_SEASON", "MIN_L5", "MIN_L10"]


@pytest.fixture(scope="module")
def panel():
    return build_feature_matrix(make_demo_panel(n_players=20, n_games=60))


@pytest.fixture(scope="module")
def augmented(panel):
    return augment_lines(panel, "PTS")


# --------------------------------------------------------------------------
# The leak this fix could easily introduce
# --------------------------------------------------------------------------

def test_generated_lines_are_pregame(augmented):
    """Lines anchored to a pregame centre must not track the outcome."""
    report = assert_lines_are_pregame(augmented, "PTS")
    assert abs(report["line_vs_actual_corr"]) <= abs(report["centre_vs_actual_corr"]) + 0.05


def test_lines_derived_from_the_outcome_are_rejected(augmented):
    """The whole point of the guard.

    A line built from the realised stat makes validation look superb and a
    live slate fail, because at scoring time the book cannot see the result
    either. Nothing downstream would reveal it.
    """
    leaked = augmented.copy()
    leaked["LINE"] = leaked["PTS"] - 0.5
    with pytest.raises(LineLeakageError, match="encode the outcome"):
        assert_lines_are_pregame(leaked, "PTS")


def test_noisily_leaked_lines_are_also_rejected(augmented):
    """A leak with noise on top is still a leak."""
    rng = np.random.default_rng(0)
    leaked = augmented.copy()
    leaked["LINE"] = leaked["PTS"] + rng.normal(0, 2.0, size=len(leaked))
    with pytest.raises(LineLeakageError):
        assert_lines_are_pregame(leaked, "PTS")


def test_augmented_copies_must_not_straddle_a_split(augmented):
    """All copies of one game share a single realised outcome."""
    half = len(augmented) // 2
    bad_train = augmented.iloc[:half]
    bad_val = augmented.iloc[half:]
    # A row-index split deliberately tears copies apart.
    shared = set(bad_train[SOURCE_ROW_COL]) & set(bad_val[SOURCE_ROW_COL])
    if shared:
        with pytest.raises(LineLeakageError, match="BOTH train"):
            assert_no_augmented_row_straddles(bad_train, bad_val)

    # A chronological split keeps them together, which is why we use one.
    cutoff = augmented["GAME_DATE"].quantile(0.7)
    good_train = augmented[augmented["GAME_DATE"] <= cutoff]
    good_val = augmented[augmented["GAME_DATE"] > cutoff]
    assert_no_augmented_row_straddles(good_train, good_val)


# --------------------------------------------------------------------------
# Augmentation mechanics
# --------------------------------------------------------------------------

def test_augmentation_produces_a_probability_gradient(augmented):
    """Without variation in the label across lines there is nothing to learn."""
    rates = augmented.dropna(subset=["over_hit"]).groupby("line_offset")["over_hit"].mean()
    assert rates.index.min() < 0 < rates.index.max()
    # A line below the baseline must hit over more often than one above it.
    assert rates.loc[rates.index.min()] > rates.loc[rates.index.max()] + 0.3
    assert rates.is_monotonic_decreasing


def test_augmentation_attaches_every_line_feature(augmented):
    for col in LINE_FEATURE_COLS:
        assert col in augmented.columns
    assert augmented["LINE"].gt(0).all(), "a non-positive line was generated"


def test_half_point_lines_cannot_push(augmented):
    half = augmented[augmented["LINE_IS_WHOLE"] == 0.0]
    if not half.empty:
        actual = pd.to_numeric(half["PTS"], errors="coerce")
        assert (actual == half["LINE"]).sum() == 0


def test_labels_drop_pushes_rather_than_grading_them():
    frame = pd.DataFrame({
        "PTS": [20.0, 25.0, 25.0],
        "LINE": [25.0, 25.0, 20.0],   # under, PUSH, over
    })
    out = label_at_line(frame, "PTS")
    assert out["over_hit"].iloc[0] == 0.0
    assert pd.isna(out["over_hit"].iloc[1]), "an exact tie was graded"
    assert out["over_hit"].iloc[2] == 1.0


def test_pregame_scale_does_not_cross_players_or_read_the_current_game():
    rows = []
    for pid, value in (("A", 100.0), ("B", 1.0)):
        for i in range(15):
            rows.append({
                "PLAYER_ID": pid, "SEASON": "2024-25",
                "GAME_DATE": pd.Timestamp("2025-01-01") + pd.Timedelta(days=i),
                "PTS": value,
            })
    out = attach_pregame_scale(pd.DataFrame(rows), "PTS")
    b = out[out["PLAYER_ID"] == "B"].sort_values("GAME_DATE")
    assert pd.isna(b.iloc[0]["PTS_PREGAME_SD"]), "B's debut had a scale from nowhere"
    # A's values are constant, so its own prior sd is 0 — never A's spread
    # leaking into B or vice versa.
    a = out[out["PLAYER_ID"] == "A"].sort_values("GAME_DATE")
    assert a["PTS_PREGAME_SD"].dropna().eq(0).all()


def test_line_features_are_relative_to_the_player():
    """The raw line alone cannot generalise across players."""
    frame = pd.DataFrame({
        "PLAYER_ID": ["A", "B"], "SEASON": ["2024-25"] * 2,
        "GAME_DATE": pd.to_datetime(["2025-01-10", "2025-01-10"]),
        "PTS": [30.0, 10.0], "PTS_BASELINE": [30.0, 10.0],
        "PTS_PREGAME_SD": [5.0, 5.0],
        "LINE": [32.0, 12.0],
    })
    out = attach_line_features(frame, "PTS")
    # Both lines sit two points above the player's own baseline, so the
    # relative features must agree even though the raw lines differ.
    assert out["LINE_MINUS_BASELINE"].nunique() == 1
    assert out["LINE_Z"].nunique() == 1
    assert out["LINE"].nunique() == 2


# --------------------------------------------------------------------------
# The property that proves the fix
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fitted(panel):
    pytest.importorskip("xgboost")
    from src.models.xgb_adapter import XGBoostAdapter

    split = int(len(panel) * 0.7)
    model = LineAwarePropModel(
        lambda cols: XGBoostAdapter(
            cols, target_market="PTS", model_params={"n_estimators": 120}
        ),
        stat="PTS", base_feature_cols=BASE_COLS,
    )
    model.fit(panel.iloc[:split].copy())
    return model, panel.iloc[split:].copy()


def test_probability_responds_to_the_line(fitted):
    """The defect was a flat response. This is the fix, measured."""
    model, scoring = fitted
    curve = model.probability_curve(scoring, [10.5, 12.5, 14.5, 16.5])
    values = curve["mean_probability_over"].dropna().to_numpy()

    assert len(values) >= 3
    assert values.max() - values.min() > 0.2, (
        f"P(over) barely moved across lines (range {values.max() - values.min():.4f}) "
        "— the model is still effectively line-blind"
    )


def test_probability_never_rises_with_the_line(fitted):
    """P(stat > L) is a survival function; an increase is self-contradiction."""
    model, scoring = fitted
    report = model.monotonicity_report(scoring, [10.5, 12.5, 14.5, 16.5, 18.5])
    assert report["monotonic_non_increasing"], report


def test_a_line_blind_model_is_flat_by_comparison(panel):
    """Pins the defect itself, so the contrast cannot quietly disappear."""
    pytest.importorskip("xgboost")
    from src.models.labels import attach_research_over_labels
    from src.models.xgb_adapter import XGBoostAdapter

    split = int(len(panel) * 0.7)
    train = attach_research_over_labels(panel.iloc[:split].copy(), stat="PTS")
    train = train.loc[train["over_hit"].notna()]
    scoring = panel.iloc[split:].copy()

    blind = XGBoostAdapter(BASE_COLS, target_market="PTS", model_params={"n_estimators": 120})
    blind.fit(train)
    blind.line_aware = True  # bypass the mask to observe the raw response

    values = [
        float(blind.predict_probability_over(scoring, line).mean())
        for line in (10.5, 12.5, 14.5, 16.5)
    ]
    assert max(values) - min(values) < 1e-9, (
        "a line-blind classifier should be exactly flat across lines"
    )


def test_model_abstains_outside_the_trained_line_range(fitted):
    """A boosted tree extrapolates badly; abstain rather than guess."""
    model, scoring = fitted
    assert model.trained_z_range is not None

    absurd = model.predict_probability_over(scoring, 500.0)
    assert absurd.isna().all(), "the model answered at a line it never saw"

    sane = model.predict_probability_over(scoring, 12.5)
    assert sane.notna().sum() > 0


def test_fit_refuses_when_line_features_are_missing(panel):
    """A model without the line features is not line-aware at all."""
    pytest.importorskip("xgboost")
    from src.models.xgb_adapter import XGBoostAdapter

    class _DropsLineCols(XGBoostAdapter):
        pass

    model = LineAwarePropModel(
        lambda cols: _DropsLineCols(cols, target_market="PTS"),
        stat="PTS", base_feature_cols=BASE_COLS,
    )
    # Remove the centre so attach_line_features cannot build the features.
    stripped = panel.drop(
        columns=[c for c in ("PTS_BASELINE", "PTS_L10", "PTS_L5", "PTS_SEASON")
                 if c in panel.columns]
    )
    with pytest.raises(ValueError, match="DATA_NOT_AVAILABLE"):
        model.fit(stripped)


def test_default_offsets_span_both_sides_of_the_baseline():
    assert min(DEFAULT_LINE_OFFSETS) < 0 < max(DEFAULT_LINE_OFFSETS)
    assert 0.0 in DEFAULT_LINE_OFFSETS
