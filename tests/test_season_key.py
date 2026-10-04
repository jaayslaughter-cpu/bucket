"""The season key: one definition, and nobody else's column to write.

Three additive layers each carried their own

    out["SEASON"] = pd.to_datetime(out["GAME_DATE"], errors="coerce").dt.year

and both halves of that line were wrong.

  1. ``dt.year`` cuts at 1 January, in the middle of every NBA season, so a
     player's group restarted on New Year's Day and every expanding mean,
     rolling window and streak counter reset with it.
  2. It wrote the guess into the returned frame as the public ``SEASON``.
     builder applies layers as ``df = attach(df)`` and halflife runs FIRST, so
     one layer's fallback became the grouping key for every layer after it —
     and silently defeated ``minutes_weighted``'s abstention, which exists
     precisely to refuse this guess.

Each test below fails without the fix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.halflife import attach_halflife_shrink_features
from src.features.hot_hand import attach_hot_hand_features
from src.features.season import (
    SEASON_KEY_COL,
    drop_season_key,
    player_season_keys,
    season_start_year,
)
from src.features.sports_ev_features import attach_form_streaks

LAYERS = (
    ("halflife", attach_halflife_shrink_features),
    ("hot_hand", attach_hot_hand_features),
    ("form_streaks", attach_form_streaks),
)


def new_year_panel(n: int = 14, *, player: str = "A") -> pd.DataFrame:
    """One player's games straddling 1 January inside ONE season."""
    dates = pd.to_datetime(
        [pd.Timestamp("2025-12-20") + pd.Timedelta(days=2 * i) for i in range(n)]
    )
    return pd.DataFrame({
        "PLAYER_ID": [player] * n,
        "GAME_DATE": dates,
        "GAME_ID": [f"002250{i:04d}" for i in range(n)],
        "TEAM_ABBREVIATION": ["LAL"] * n,
        "OPPONENT_ABBREVIATION": ["BOS"] * n,
        "MIN": [30.0] * n,
        "PTS": [20.0 + i for i in range(n)],
        "REB": [5.0] * n,
        "AST": [4.0] * n,
        "FG3M": [2.0] * n,
        "STL": [1.0] * n,
        "BLK": [0.5] * n,
        "TOV": [2.0] * n,
        "FGA": [15.0] * n,
        "FTA": [4.0] * n,
    })


# --- the boundary -------------------------------------------------------

def test_the_season_does_not_end_on_new_year_s_eve():
    """The whole point: a plain dt.year would answer 2025 and 2026 here."""
    dates = pd.Series(pd.to_datetime(["2025-12-30", "2026-01-02"]))
    keys = season_start_year(dates)
    assert list(keys) == [2025, 2025]
    assert keys.iloc[0] == keys.iloc[1]


def test_august_is_the_cut_because_the_nba_does_not_play_in_august():
    dates = pd.Series(pd.to_datetime(["2025-07-15", "2025-10-21", "2026-06-10"]))
    assert list(season_start_year(dates)) == [2024, 2025, 2025]


def test_an_unreadable_date_has_no_season_rather_than_a_wrong_one():
    keys = season_start_year(pd.Series([pd.NaT, pd.Timestamp("2025-11-01")]))
    assert pd.isna(keys.iloc[0])
    assert keys.iloc[1] == 2025


# --- nobody else's column to write --------------------------------------

@pytest.mark.parametrize("name,attach", LAYERS)
def test_a_layer_never_hands_back_a_season_it_invented(name, attach):
    """
    The contamination test. Every one of these used to return a SEASON column
    the caller never supplied, and the next layer read it as the panel's own.
    """
    out = attach(new_year_panel())
    assert "SEASON" not in out.columns, f"{name} fabricated a public SEASON"
    assert SEASON_KEY_COL not in out.columns, f"{name} leaked its private key"


@pytest.mark.parametrize("name,attach", LAYERS)
def test_a_layer_still_uses_the_panel_s_own_season_when_it_has_one(name, attach):
    panel = new_year_panel()
    panel["SEASON"] = "2025-26"
    out = attach(panel)
    assert list(out["SEASON"]) == ["2025-26"] * len(panel)


@pytest.mark.parametrize("name,attach", LAYERS)
def test_the_derived_key_groups_exactly_as_an_explicit_season_would(name, attach):
    """
    Absence of a column is not correctness. This measures the VALUES: with the
    1 January restart, the derived-key run and the explicit-SEASON run
    disagree on every row after New Year's Day.
    """
    bare = new_year_panel()
    labelled = new_year_panel()
    labelled["SEASON"] = "2025-26"

    got = attach(bare)
    want = attach(labelled)
    cols = [
        c for c in want.columns
        if c in got.columns
        and c != "SEASON"
        and pd.api.types.is_numeric_dtype(want[c])
    ]
    assert cols, "no numeric output columns to compare"
    for col in cols:
        pd.testing.assert_series_equal(
            got[col].reset_index(drop=True),
            want[col].reset_index(drop=True),
            check_names=False,
            obj=f"{name}.{col}",
        )


def test_the_restart_is_what_this_would_have_caught():
    """
    Pin the failure mode itself, so the test above cannot pass for a reason
    unrelated to the boundary: grouping by calendar year splits this panel.
    """
    panel = new_year_panel()
    by_calendar_year = panel.groupby(
        [panel["PLAYER_ID"], panel["GAME_DATE"].dt.year]
    ).ngroups
    _, keys = player_season_keys(panel)
    by_season_key = panel.assign(
        **{SEASON_KEY_COL: season_start_year(panel["GAME_DATE"])}
    ).groupby(keys).ngroups
    assert by_calendar_year == 2
    assert by_season_key == 1


# --- the helper's own contract ------------------------------------------

def test_the_panel_s_season_is_preferred_and_the_frame_is_untouched():
    panel = new_year_panel()
    panel["SEASON"] = "2025-26"
    out, keys = player_season_keys(panel)
    assert keys == ["PLAYER_ID", "SEASON"]
    assert out is panel, "no copy needed when the panel already has one"
    assert SEASON_KEY_COL not in out.columns


def test_a_derived_key_does_not_mutate_the_caller_s_frame():
    panel = new_year_panel()
    out, keys = player_season_keys(panel)
    assert keys == ["PLAYER_ID", SEASON_KEY_COL]
    assert SEASON_KEY_COL in out.columns
    assert SEASON_KEY_COL not in panel.columns
    assert "SEASON" not in out.columns


def test_dropping_the_key_is_a_no_op_when_there_was_none():
    panel = new_year_panel()
    assert list(drop_season_key(panel).columns) == list(panel.columns)


def test_defense_shares_the_one_definition():
    """It had the right form first; two copies are how they drift apart."""
    from src.features.defense import _season_key

    dates = pd.Series(pd.to_datetime(["2025-12-30", "2026-01-02", "2025-07-15"]))
    assert list(_season_key(dates)) == list(season_start_year(dates))


# --- the builder: minutes_weighted's abstention is reachable again -------

def test_the_builder_no_longer_publishes_an_invented_season():
    from src.features.builder import build_feature_matrix

    feat = build_feature_matrix(new_year_panel())
    assert "SEASON" not in feat.columns
    assert SEASON_KEY_COL not in feat.columns


def test_minutes_weighted_abstains_on_a_season_less_panel_as_documented():
    """
    It refuses to guess a season. That refusal was unreachable through the
    builder: halflife ran first and SEASON was never missing by the time
    minutes_weighted looked.
    """
    from src.features.builder import build_feature_matrix

    feat = build_feature_matrix(new_year_panel())
    assert feat.attrs["minutes_weighted_status"] == "DATA_NOT_AVAILABLE"
    assert "SEASON" in feat.attrs["minutes_weighted_reason"]
    assert feat["PTS_MW_L5"].isna().all()


def test_a_labelled_panel_still_gets_minutes_weighted_values():
    """The abstention must be about the missing column, not about the layer."""
    from src.features.builder import build_feature_matrix

    panel = new_year_panel()
    panel["SEASON"] = "2025-26"
    feat = build_feature_matrix(panel)
    assert feat.attrs["minutes_weighted_status"] == "OK"
    assert feat["PTS_MW_L5"].notna().any()
    assert np.isfinite(feat["PTS_MW_L5"].dropna()).all()
