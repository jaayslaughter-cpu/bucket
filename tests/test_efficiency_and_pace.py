"""
Scoring efficiency, measured pace, and the postgame-column guard.

scoring_efficiency was referenced by wave5a's builder and never
delivered; pace was previously a fabricated 1.0. Both are now real
measurements, which introduces a new trap these tests cover: TS_PCT and
SHOT_VOLUME read like engineered features and are raw same-game values.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.builder import build_feature_matrix
from src.features.scoring_efficiency import (
    FREE_THROW_POSSESSION_FACTOR,
    attach_box_ts_features,
    true_shooting_percentage,
)
from src.models.compare import PostgameFeatureError, resolve_feature_cols
from src.models.data_audit import make_demo_panel


@pytest.fixture(scope="module")
def features():
    return build_feature_matrix(make_demo_panel(n_players=8, n_games=40))


# --------------------------------------------------------------------------
# True Shooting is a formula, not a heuristic
# --------------------------------------------------------------------------

def test_true_shooting_matches_the_published_formula():
    """TS% = PTS / (2 * (FGA + 0.44 * FTA))."""
    ts = true_shooting_percentage(
        pd.Series([25.0]), pd.Series([15.0]), pd.Series([6.0])
    )
    expected = 25.0 / (2 * (15.0 + FREE_THROW_POSSESSION_FACTOR * 6.0))
    assert ts.iloc[0] == pytest.approx(expected)
    assert FREE_THROW_POSSESSION_FACTOR == 0.44


def test_no_shots_gives_undefined_not_zero_efficiency():
    """Zero would drag rolling averages down with a 'did not play' value."""
    ts = true_shooting_percentage(pd.Series([0.0]), pd.Series([0.0]), pd.Series([0.0]))
    assert pd.isna(ts.iloc[0])


def test_efficiency_layer_abstains_without_shot_volume():
    """TS% cannot be derived from points alone."""
    frame = pd.DataFrame({"PTS": [20.0], "PLAYER_ID": ["a"]})
    out = attach_box_ts_features(frame)
    assert "TS_PCT" not in out.columns
    assert list(out.columns) == list(frame.columns)


def test_efficiency_features_never_read_the_current_game(features):
    first = features.sort_values(["PLAYER_ID", "GAME_DATE"]).groupby("PLAYER_ID").head(1)
    for col in ("TS_PCT_L5", "SHOT_VOLUME_L5", "FGA_L5"):
        assert col in features.columns
        assert first[col].isna().all(), f"{col} had a value on a player's first game"


# --------------------------------------------------------------------------
# Pace is measured, and unknown pace stays unknown
# --------------------------------------------------------------------------

def test_pace_is_measured_and_never_a_fabricated_neutral(features):
    """The old code wrote 1.0 whenever pace was unavailable."""
    assert "PACE_MULTIPLIER" in features.columns
    pace = features["PACE_MULTIPLIER"]
    assert pace.notna().any(), "pace was never measured"
    assert not (pace == 1.0).any(), (
        "an exactly-neutral pace is the fabricated value this replaced"
    )


def test_pace_is_absent_rather_than_neutral_when_inputs_are_missing():
    from src.features.builder import attach_team_pace

    panel = make_demo_panel(n_players=3, n_games=10).drop(columns=["FGA"])
    out = attach_team_pace(panel)
    assert "PACE_MULTIPLIER" not in out.columns


def test_layer_two_is_published_as_two_columns(features):
    """L2 must stay defined everywhere; only the pace-adjusted form is null.

    Collapsing them would force a choice between inventing a neutral pace
    and discarding every early-season row.
    """
    assert features["PTS_L2"].notna().any()
    assert "PTS_L2_PACE" in features.columns

    # L2 carries no pace claim, so it is defined wherever BASELINE is.
    baseline_known = features["PTS_BASELINE"].notna()
    assert features.loc[baseline_known, "PTS_L2"].notna().all()

    # The pace-adjusted form is null exactly where pace was not measured.
    pace_unknown = features["PACE_MULTIPLIER"].isna()
    assert features.loc[pace_unknown, "PTS_L2_PACE"].isna().all()


# --------------------------------------------------------------------------
# The trap the new columns introduce
# --------------------------------------------------------------------------

@pytest.mark.parametrize("column", ["TS_PCT", "SHOT_VOLUME", "FT_RATE", "PTS", "FGA"])
def test_postgame_columns_are_refused_as_features(column):
    """These hold THIS game's result. TS_PCT is the dangerous one — it
    reads like an engineered feature and is a raw box-score quantity."""
    frame = pd.DataFrame({column: [1.0], "PTS_L5": [2.0]})
    with pytest.raises(PostgameFeatureError, match="same-game outcome"):
        resolve_feature_cols(frame, ["PTS_L5", column])


def test_shifted_forms_are_allowed():
    frame = pd.DataFrame({"TS_PCT_L5": [0.5], "PTS_L5": [20.0], "SHOT_VOLUME_L10": [14.0]})
    present, absent = resolve_feature_cols(
        frame, ["TS_PCT_L5", "PTS_L5", "SHOT_VOLUME_L10"]
    )
    assert len(present) == 3
    assert absent == []


# --------------------------------------------------------------------------
# The schema version must track what was attached
# --------------------------------------------------------------------------

def test_schema_version_reflects_the_attached_layers(features):
    """Two runs with different layers must not both claim fs_v1_shift1_l2."""
    from src.features.builder import (
        FEATURE_SCHEMA_VERSION,
        resolved_feature_schema_version,
    )

    version = features["FEATURE_SCHEMA_VERSION"].iloc[0]
    assert version.startswith(FEATURE_SCHEMA_VERSION)
    assert version != FEATURE_SCHEMA_VERSION, "layers attached but version unchanged"

    assert resolved_feature_schema_version([]) == FEATURE_SCHEMA_VERSION
    assert resolved_feature_schema_version(["a"]) != resolved_feature_schema_version(["b"])
    assert resolved_feature_schema_version(["a", "b"]) == resolved_feature_schema_version(["b", "a"])


# --------------------------------------------------------------------------
# ECE
# --------------------------------------------------------------------------

def test_ece_detects_overconfidence():
    import numpy as np

    from src.models.prob_calibration import expected_calibration_error

    rng = np.random.default_rng(3)
    p = rng.uniform(0.05, 0.95, 4000)
    y = (rng.random(4000) < p).astype(int)
    calibrated = expected_calibration_error(y, p, n_bins=10)

    overconfident = np.clip((p - 0.5) * 2.2 + 0.5, 0.01, 0.99)
    skewed = expected_calibration_error(y, overconfident, n_bins=10)

    assert calibrated["gate_passed"] and skewed["gate_passed"]
    assert skewed["ece"] > calibrated["ece"] * 3


def test_ece_refuses_a_sparse_reliability_diagram():
    """Predictions clustered in two bins would score flatteringly well."""
    import numpy as np

    from src.models.prob_calibration import expected_calibration_error

    rng = np.random.default_rng(3)
    narrow = rng.uniform(0.49, 0.51, 300)
    y = (rng.random(300) < narrow).astype(int)

    result = expected_calibration_error(y, narrow, n_bins=10)
    assert result["gate_passed"] is False
    assert result["ece"] is None, "a gated ECE must not be usable for selection"
    assert result["ece_ungated"] is not None, "the raw value should still be visible"
