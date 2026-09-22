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


# ---------------------------------------------------------------------------
# Wiring into the comparison pipeline
# ---------------------------------------------------------------------------


def _fitted_pair(panel):
    """A fitted line-aware model and its line-blind counterpart."""
    from src.models.compare import build_components, load_comparison_config, prepare_market_panel
    from src.models.labels import attach_research_over_labels, default_feature_cols
    from src.models.walk_forward import fixed_cutoff_split

    work = attach_research_over_labels(prepare_market_panel(panel, "PTS"), stat="PTS")
    split = fixed_cutoff_split(work, train_end="2025-01-15", validation_end="2025-02-15")
    train = work.loc[split.train_idx].reset_index(drop=True)
    valid = work.loc[split.validation_idx].reset_index(drop=True)

    cfg = load_comparison_config()
    cols = [c for c in default_feature_cols("PTS") if c in work.columns]
    model = build_components("PTS", cols, cfg)["line_aware"]
    model.fit(train, valid)
    return model, valid


def test_source_row_identity_is_not_the_positional_index():
    """
    Train and validation are augmented separately and each resets its own
    index, so positional ids collide across the two frames: different
    player-games get the same number. That fired the straddle guard on a
    clean chronological split, and would equally have MISSED a real straddle
    whenever the positions happened not to line up.
    """
    import pandas as pd

    from src.models.line_aware import SOURCE_ROW_COL, augment_lines

    def _frame(player_ids, game_ids):
        return pd.DataFrame({
            "PLAYER_ID": player_ids, "GAME_ID": game_ids,
            "PLAYER_NAME": player_ids,
            "GAME_DATE": pd.to_datetime(["2025-01-01"] * len(player_ids)),
            "PTS": [20.0] * len(player_ids), "PTS_L10": [18.0] * len(player_ids),
            "PTS_SEASON": [19.0] * len(player_ids),
        })

    left = augment_lines(_frame(["p1", "p2"], ["g1", "g1"]), "PTS")
    right = augment_lines(_frame(["p3", "p4"], ["g2", "g2"]), "PTS")

    # Positionally these are rows 0 and 1 on both sides; by identity they
    # share nothing.
    assert set(left[SOURCE_ROW_COL]) & set(right[SOURCE_ROW_COL]) == set()
    assert all("@" in str(v) for v in left[SOURCE_ROW_COL])


def test_a_clean_chronological_split_passes_the_straddle_guard(panel):
    from src.models.compare import prepare_market_panel
    from src.models.line_aware import (
        assert_no_augmented_row_straddles,
        augment_lines,
    )
    from src.models.walk_forward import fixed_cutoff_split

    work = prepare_market_panel(panel, "PTS")
    split = fixed_cutoff_split(work, train_end="2025-01-15", validation_end="2025-02-15")
    train = augment_lines(work.loc[split.train_idx].reset_index(drop=True), "PTS")
    valid = augment_lines(work.loc[split.validation_idx].reset_index(drop=True), "PTS")

    assert_no_augmented_row_straddles(train, valid)     # must not raise


def test_the_model_answers_differently_at_different_lines(panel):
    """
    The whole point. A line-blind classifier returns one number whatever it
    is asked; this must produce a survival curve that falls as the line rises.
    """
    model, valid = _fitted_pair(panel)
    row = valid.head(1)
    lines = [8.0, 11.0, 14.0, 17.0]
    probs = [float(model.predict_probability_over(row, line).iloc[0]) for line in lines]

    assert all(pd.notna(p) for p in probs)
    assert max(probs) - min(probs) > 0.20          # genuinely line-dependent
    assert probs == sorted(probs, reverse=True)    # monotone, as a survival fn must be


def test_predict_rows_satisfies_the_contract_compare_py_calls(panel):
    """
    compare.py calls .predict_rows — the method the wrapper did not have,
    which is why 511 tested lines sat disconnected from the pipeline.
    """
    from src.models.prediction_schema import ModelPrediction

    model, valid = _fitted_pair(panel)
    rows = model.predict_rows(valid.head(20), line_col="RESEARCH_LINE")

    assert len(rows) == 20
    assert all(isinstance(r, ModelPrediction) for r in rows)
    assert {r.model_name for r in rows} == {"line_aware"}
    assert all("line_aware" in r.model_version for r in rows)

    answered = [r for r in rows if r.probability_over is not None]
    assert answered, "every row abstained — the wrapper is not answering at all"
    for r in answered:
        total = r.probability_over + r.probability_under + (r.probability_push or 0.0)
        assert total == pytest.approx(1.0, abs=1e-3)


def test_an_out_of_support_line_abstains_rather_than_extrapolating(panel):
    """
    A boosted tree asked outside its trained line range can return a
    probability that RISES with the line, which no survival function does.
    None with a named warning beats a number the fit cannot support.
    """
    model, valid = _fitted_pair(panel)
    absurd = valid.head(5).copy()
    absurd["RESEARCH_LINE"] = 400.0

    rows = model.predict_rows(absurd, line_col="RESEARCH_LINE")
    assert all(r.probability_over is None for r in rows)
    assert all(
        any("outside the range" in w for w in r.warnings) for r in rows
    )


def test_the_mean_is_a_property_of_the_game_not_of_the_line(panel):
    """The projection must not move when only the asked line moves."""
    model, valid = _fitted_pair(panel)
    row = valid.head(3)
    first = model.predict_mean(row.assign(RESEARCH_LINE=10.0))
    second = model.predict_mean(row.assign(RESEARCH_LINE=30.0))
    pd.testing.assert_series_equal(first, second, check_names=False)
