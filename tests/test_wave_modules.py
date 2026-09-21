"""
Wave 1/2/4/4b modules merged into the repo.

These pin the properties that make each module safe to rely on: the
feature modules must not leak, the display modules must not imply a
wager, and the registry must not persist a credential.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.builder import build_feature_matrix
from src.features.halflife import attach_halflife_shrink_features
from src.features.hot_hand import attach_hot_hand_features
from src.features.teammate_cascade import attach_teammate_cascade_stub
from src.models.data_audit import make_demo_panel


@pytest.fixture(scope="module")
def feature_matrix():
    return build_feature_matrix(make_demo_panel(n_players=6, n_games=25))


def _two_player_panel():
    """Player A scores 100 every game; player B scores 0. Any A-sized number
    on B's debut row is a cross-player leak."""
    rows = []
    for pid, value in (("A", 100.0), ("B", 0.0)):
        for i in range(14):
            rows.append({
                "PLAYER_ID": pid, "SEASON": "2024-25",
                "GAME_DATE": pd.Timestamp("2025-01-01") + pd.Timedelta(days=i),
                "GAME_ID": f"g{i:03d}", "TEAM_ABBREVIATION": "LAL",
                "PTS": value, "REB": value / 10, "AST": value / 10,
                "FG3M": value / 50, "STL": 1.0, "BLK": 1.0, "MIN": 30.0,
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Feature modules must not leak
# --------------------------------------------------------------------------

def test_halflife_features_do_not_cross_player_boundaries():
    """A rolling window that spans players gives a debut its predecessor's form."""
    panel = _two_player_panel()
    out = attach_halflife_shrink_features(panel)

    new_cols = [c for c in out.columns if c not in panel.columns]
    assert new_cols, "no features were attached"

    debut = out[out["PLAYER_ID"] == "B"].sort_values("GAME_DATE").iloc[0]
    for col in new_cols:
        value = debut[col]
        if pd.notna(value) and isinstance(value, (int, float, np.floating)):
            assert abs(float(value)) < 1.0, (
                f"{col} = {value} on B's debut — that is A's signal leaking across"
            )


def test_halflife_features_do_not_read_the_current_game():
    """The first row of a player has no prior game, so nothing to average."""
    panel = _two_player_panel()
    out = attach_halflife_shrink_features(panel)
    new_cols = [c for c in out.columns if c not in panel.columns]

    first = out[out["PLAYER_ID"] == "A"].sort_values("GAME_DATE").iloc[0]
    for col in new_cols:
        value = first[col]
        if pd.notna(value) and isinstance(value, (int, float, np.floating)):
            assert abs(float(value)) < 1.0, (
                f"{col} = {value} on A's first game — it read its own box score"
            )


def test_hot_hand_abstains_by_name_without_the_season_baseline():
    """It used to fail deep in numpy with a TypeError naming nothing useful."""
    raw = pd.DataFrame({
        "PLAYER_ID": ["A"] * 5, "SEASON": ["2024-25"] * 5,
        "GAME_DATE": pd.date_range("2025-01-01", periods=5), "PTS": [10.0] * 5,
    })
    out = attach_hot_hand_features(raw)  # must not raise
    assert "PTS_HOT_Z" not in out.columns


def test_hot_hand_attaches_on_a_real_feature_matrix(feature_matrix):
    out = attach_hot_hand_features(feature_matrix.copy())
    assert "PTS_HOT_Z" in out.columns
    assert "PTS_HOT_HAND_STATUS" in out.columns


def test_cascade_never_invents_a_usage_multiplier(feature_matrix):
    """Redistributing usage from an absence needs verified pairwise history."""
    out = attach_teammate_cascade_stub(feature_matrix.copy())
    assert out["CASCADE_USAGE_MULT"].isna().all(), (
        "a usage multiplier was invented without verified pairwise data"
    )
    assert set(out["CASCADE_STATUS"]) <= {
        "DATA_NOT_AVAILABLE", "NEEDS_VERIFIED_PAIRWISE", "NO_TEAMMATE_OUTS",
    }


def test_cascade_abstains_when_injury_flags_are_absent(feature_matrix):
    out = attach_teammate_cascade_stub(feature_matrix.copy())
    assert (out["CASCADE_STATUS"] == "DATA_NOT_AVAILABLE").all()
    assert "never invents injuries" in out["CASCADE_NOTES"].iloc[0]


# --------------------------------------------------------------------------
# Display modules must not imply a wager
# --------------------------------------------------------------------------

def test_edge_grade_carries_a_research_only_disclaimer():
    from src.models.edge_grades import DISPLAY_DISCLAIMER, research_edge_letter_grade

    assert "not a bet recommendation" in DISPLAY_DISCLAIMER
    assert "stake" in DISPLAY_DISCLAIMER.lower()

    graded = research_edge_letter_grade(
        probability_over=0.62, fair_market_probability=0.50,
    )
    assert graded["edge_letter_grade"] in {"A+", "A", "B", "C", "D", "F", "N/A"}
    # Every grade must carry the disclaimer with it — a letter travelling
    # without one reads as advice.
    assert graded["disclaimer"] == DISPLAY_DISCLAIMER
    assert graded["grade_basis"] == "model_minus_fair_market"
    assert graded["edge_probability"] == pytest.approx(0.12, abs=1e-9)


def test_edge_grade_names_the_basis_it_actually_used():
    """A research-separation grade must not be mistaken for a market edge."""
    from src.models.edge_grades import research_edge_letter_grade

    market = research_edge_letter_grade(
        probability_over=0.62, fair_market_probability=0.50,
    )
    assert market["grade_basis"] == "model_minus_fair_market"

    # Without a fair market probability it must fall back and SAY so.
    research = research_edge_letter_grade(
        prediction_mean=28.0, prop_line=25.5, prediction_std=5.0,
    )
    assert research["grade_basis"] != "model_minus_fair_market"
    assert research["disclaimer"]

    # Nothing usable at all: N/A, never a confident letter.
    nothing = research_edge_letter_grade()
    assert nothing["edge_letter_grade"] == "N/A"
    assert nothing["grade_basis"] == "unavailable"


# --------------------------------------------------------------------------
# The registry must never persist a credential
# --------------------------------------------------------------------------

def test_artifact_registry_strips_secrets(tmp_path):
    """Registry rows land on disk and are often committed."""
    from src.models.artifact_registry import append_registry, build_registry_record

    record = build_registry_record(
        model_name="catboost", model_version="cb_v1", target_market="PTS",
        artifact_path=tmp_path / "cb_PTS.cbm",
        feature_schema_version="fs_v1_shift1_l2",
        extra={
            "api_key": "SHOULD-NOT-PERSIST",
            "PROPLINE_API_KEY": "SHOULD-NOT-PERSIST",
            "password": "SHOULD-NOT-PERSIST",
            "auth_token": "SHOULD-NOT-PERSIST",
            "client_secret": "SHOULD-NOT-PERSIST",
            "discord_webhook": "https://hooks.example/SHOULD-NOT-PERSIST",
            "train_rows": 1234,
        },
    )
    written = append_registry(tmp_path, record)
    blob = str(record) + written.read_text(encoding="utf-8")

    assert "SHOULD-NOT-PERSIST" not in blob
    assert "1234" in blob, "a harmless field was dropped along with the secrets"
    assert record["research_only"] is True


def test_run_manifest_also_strips_secrets(tmp_path):
    """The same rule has to hold on the other writer, not just the registry."""
    from src.models.artifact_registry import write_run_manifest

    manifest = write_run_manifest(
        tmp_path,
        run_id="run-1",
        meta={"api_key": "SHOULD-NOT-PERSIST", "rows": 99},
    )
    assert "SHOULD-NOT-PERSIST" not in str(manifest)
    assert "99" in str(manifest)


# --------------------------------------------------------------------------
# Eligibility
# --------------------------------------------------------------------------

def test_prior_game_counts_are_strictly_prior():
    from src.models.eligibility import prior_game_counts

    panel = _two_player_panel()
    counts = prior_game_counts(panel)

    first_b = panel[panel["PLAYER_ID"] == "B"].sort_values("GAME_DATE").index[0]
    assert counts.loc[first_b] == 0, "a debut showed prior games"
    assert counts.max() == 13
