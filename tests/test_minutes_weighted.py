"""Minutes-weighted recent averages — the layer that had no test at all.

It was 134 lines with no production import AND no test, emitting `{STAT}_MW_L5`
columns nothing read. Wiring it surfaced three defects, and each has a test
below that fails without the fix:

  1. the combo aliases were hardcoded to `_L5` while the window was a parameter,
     so window=10 produced a PR_MW_L5 holding ten-game data;
  2. a missing SEASON fell back to GAME_DATE.dt.year, which splits an NBA season
     at 1 January and restarts every player's window on New Year's Day;
  3. the layer returned a RE-SORTED frame, and the builder applies layers as
     `df = attach(df)` — so it silently reordered the whole feature matrix for
     every later layer and for the caller.

The columns are produced but deliberately absent from the feature contract;
that decision is measured, and the last test here pins it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.minutes_weighted import (
    DEFAULT_WINDOW,
    HIGH_WEIGHT,
    LOW_WEIGHT,
    MID_WEIGHT,
    _emitted_columns,
    _minutes_weights,
    attach_minutes_weighted_features,
)


def panel(n: int = 12, *, player: str = "A", season: str = "2025-26", minutes=None):
    """One player's season, chronological, with a steady 30-minute baseline."""
    return pd.DataFrame({
        "PLAYER_ID": [player] * n,
        "SEASON": [season] * n,
        "GAME_DATE": pd.date_range("2025-10-21", periods=n, freq="2D"),
        "GAME_ID": [f"002250{i:04d}" for i in range(n)],
        "MIN": list(minutes) if minutes is not None else [30.0] * n,
        "PTS": [20.0 + i for i in range(n)],
        "REB": [5.0] * n,
        "AST": [4.0] * n,
        "STL": [1.0] * n,
        "BLK": [0.5] * n,
        "FG3M": [2.0] * n,
    })


# --- defect 1: the combo aliases must follow the window -----------------

def test_the_combo_aliases_are_named_from_the_window():
    """
    window=10 used to produce PTS_MW_L10 beside a PR_MW_L5 holding ten-game
    data — a column whose name contradicted its contents.
    """
    out = attach_minutes_weighted_features(panel(20), window=10)
    assert "PR_MW_L10" in out.columns
    assert "PRA_MW_L10" in out.columns
    assert "PR_MW_L5" not in out.columns, "an L5-named column from an L10 window"


def test_the_default_window_still_emits_the_l5_names():
    out = attach_minutes_weighted_features(panel())
    for column in _emitted_columns(DEFAULT_WINDOW):
        assert column in out.columns
    assert "PTS_MW_L5" in out.columns and "PRA_MW_L5" in out.columns


def test_the_abstain_path_emits_exactly_the_same_columns():
    """
    The two paths disagreed: stats were named from `window`, combos were
    hardcoded. A consumer that found PR_MW_L10 on a good frame and PR_MW_L5 on
    an abstaining one would see a different schema per run.
    """
    good = attach_minutes_weighted_features(panel(20), window=10)
    bad = attach_minutes_weighted_features(
        panel(20).drop(columns=["MIN"]), window=10
    )
    emitted = set(_emitted_columns(10))
    assert emitted <= set(good.columns)
    assert emitted <= set(bad.columns)
    assert bad.attrs["minutes_weighted_status"] == "DATA_NOT_AVAILABLE"
    assert bad[sorted(emitted)].isna().all().all()


# --- defect 2: SEASON is required, never derived ------------------------

def test_a_missing_season_abstains_rather_than_using_the_calendar_year():
    """
    GAME_DATE.dt.year splits an NBA season at 1 January, so the per-player
    grouping would restart mid-season and reset both the expanding minutes
    baseline and the rolling window for every player on New Year's Day.
    """
    out = attach_minutes_weighted_features(panel().drop(columns=["SEASON"]))
    assert out.attrs["minutes_weighted_status"] == "DATA_NOT_AVAILABLE"
    assert "SEASON" in out.attrs["minutes_weighted_reason"]
    assert out["PTS_MW_L5"].isna().all()


@pytest.mark.parametrize("dropped", ["PLAYER_ID", "GAME_DATE", "MIN", "SEASON"])
def test_every_required_column_is_named_when_it_is_missing(dropped):
    out = attach_minutes_weighted_features(panel().drop(columns=[dropped]))
    assert out.attrs["minutes_weighted_status"] == "DATA_NOT_AVAILABLE"
    assert dropped in out.attrs["minutes_weighted_reason"]


def test_a_season_boundary_does_not_restart_the_window():
    """
    The behaviour the derived-year fallback would have broken: games either side
    of 1 January inside ONE season stay in one group.
    """
    frame = panel(10)
    frame["GAME_DATE"] = pd.to_datetime([
        "2025-12-26", "2025-12-28", "2025-12-30", "2026-01-02", "2026-01-04",
        "2026-01-06", "2026-01-08", "2026-01-10", "2026-01-12", "2026-01-14",
    ])
    out = attach_minutes_weighted_features(frame)
    # a restart would leave the January rows with too few prior games to fill
    # min_periods, so they would be NaN
    assert out["PTS_MW_L5"].iloc[-1] == pytest.approx(
        out["PTS_MW_L5"].iloc[-1]
    )
    assert out["PTS_MW_L5"].tail(5).notna().all()


# --- defect 3: the caller's row order survives --------------------------

def test_the_layer_returns_the_caller_s_row_order():
    """
    builder applies layers as `df = attach(df)`. A layer that hands back a
    re-sorted frame reorders the whole feature matrix for every later layer and
    for the caller — and (player, season, date) is player-major, which is NOT
    chronological, while the comparison path's folds are positional over a
    date-sorted frame.
    """
    frame = pd.concat([panel(8, player="B"), panel(8, player="A")], ignore_index=True)
    shuffled = frame.sample(frac=1.0, random_state=3).reset_index(drop=True)

    out = attach_minutes_weighted_features(shuffled)
    assert list(out["GAME_ID"]) == list(shuffled["GAME_ID"])
    assert list(out["PLAYER_ID"]) == list(shuffled["PLAYER_ID"])
    assert out.index.equals(shuffled.index)


def test_the_values_are_per_player_regardless_of_input_order():
    """Restoring the order must not mean the numbers were computed on it."""
    frame = pd.concat([panel(8, player="B"), panel(8, player="A")], ignore_index=True)
    ordered = attach_minutes_weighted_features(frame)
    shuffled_in = frame.sample(frac=1.0, random_state=11)
    shuffled_out = attach_minutes_weighted_features(shuffled_in)

    rejoined = shuffled_out.set_index("GAME_ID")["PTS_MW_L5"]
    expected = ordered.set_index("GAME_ID")["PTS_MW_L5"]
    pd.testing.assert_series_equal(
        rejoined.sort_index(), expected.sort_index(), check_names=False
    )


def test_no_helper_columns_leak_into_the_frame():
    out = attach_minutes_weighted_features(panel())
    assert not [c for c in out.columns if c.startswith("_mw") or c == "_MW_WEIGHT"]


# --- the weight rule ----------------------------------------------------

def test_the_weight_rule_brackets():
    base = pd.Series([30.0] * 4)
    mins = pd.Series([20.0, 22.0, 26.0, 29.0])   # 0.67, 0.73, 0.87, 0.97 of base
    w = _minutes_weights(mins, base)
    assert list(w) == [LOW_WEIGHT, MID_WEIGHT, HIGH_WEIGHT, HIGH_WEIGHT]


def test_an_unknown_minute_count_weighs_nothing_rather_than_one():
    """
    A missing input must not be silently treated as a normal game: NaN keeps the
    row out of both the numerator and the denominator.
    """
    w = _minutes_weights(pd.Series([np.nan, 30.0]), pd.Series([30.0, np.nan]))
    assert w.isna().all()


def test_a_short_minutes_outlier_is_down_weighted_against_a_plain_mean():
    """
    The point of the layer. A 40-point game played on 15 minutes against a
    30-minute baseline should count for less than it would in a flat average.

    The outlier sits at index 5 and the assertion reads index 7: it has to be a
    PRIOR game to enter the window at all. An earlier version of this test put
    it on the row being read, where it cannot affect its own average — so the
    test passed while comparing 10.0 against 10.0 and measured nothing.
    """
    minutes = [30.0] * 8
    minutes[5] = 15.0
    frame = panel(8, minutes=minutes)
    frame["PTS"] = [10.0] * 8
    frame.loc[5, "PTS"] = 40.0

    out = attach_minutes_weighted_features(frame)
    plain = frame["PTS"].shift(1).rolling(5, min_periods=2).mean()

    assert plain.iloc[-1] == pytest.approx(16.0)        # (10+10+10+40+10)/5
    assert out["PTS_MW_L5"].iloc[-1] == pytest.approx(80.0 / 6.5)   # ~12.31
    assert out["PTS_MW_L5"].iloc[-1] < plain.iloc[-1]


# --- leakage ------------------------------------------------------------

def test_the_first_game_of_a_season_has_no_value():
    """Nothing prior exists, so there is nothing to average."""
    out = attach_minutes_weighted_features(panel())
    assert pd.isna(out["PTS_MW_L5"].iloc[0])


def test_the_current_game_never_enters_its_own_average():
    """
    The leakage test that matters. Row i's value must be unchanged when row i's
    own PTS is altered.
    """
    frame = panel(10)
    baseline = attach_minutes_weighted_features(frame)["PTS_MW_L5"]

    tampered = frame.copy()
    tampered.loc[tampered.index[-1], "PTS"] = 999.0
    after = attach_minutes_weighted_features(tampered)["PTS_MW_L5"]

    assert after.iloc[-1] == pytest.approx(baseline.iloc[-1], nan_ok=True)


def test_the_combo_alias_is_the_sum_of_its_weighted_parts():
    """Identical denominators, so the sum of the means is the mean of the sum."""
    out = attach_minutes_weighted_features(panel(10))
    row = out.iloc[-1]
    assert row["PRA_MW_L5"] == pytest.approx(
        row["PTS_MW_L5"] + row["REB_MW_L5"] + row["AST_MW_L5"]
    )
    assert row["PR_MW_L5"] == pytest.approx(row["PTS_MW_L5"] + row["REB_MW_L5"])


# --- produced, but deliberately not in the contract ---------------------

def test_the_columns_are_absent_from_the_feature_contract_on_purpose():
    """
    |r| 0.976-0.992 against {STAT}_L5 on the 214,381-row panel puts these inside
    the band the halflife family was excluded for, where the A/B measured Brier
    getting worse on every fold. They are built so feature_ab can test them,
    not shipped. Promoting them should follow a measurement, not an edit.
    """
    from src.models.labels import default_feature_cols

    for market in ("PTS", "REB", "AST"):
        assert not [c for c in default_feature_cols(market) if "_MW_" in c]


def test_the_builder_registers_the_layer_so_the_columns_exist():
    """Absent from the contract is not the same as absent from the matrix."""
    from src.features.builder import _ADDITIVE_FEATURE_LAYERS

    assert "minutes_weighted" in [label for label, _ in _ADDITIVE_FEATURE_LAYERS]


def test_feature_ab_can_measure_the_layer():
    from scripts.feature_ab import LAYERS

    layer = LAYERS["minutes_weighted"]
    assert set(_emitted_columns(DEFAULT_WINDOW)) == set(layer.columns)
    assert "wire-under-test" in layer.note


def test_the_builder_attaches_each_player_s_values_to_that_player_s_rows():
    """
    Registered is not the same as correct through the builder.

    build_feature_matrix sorts the panel to (PLAYER_ID, GAME_DATE) and the
    layer loop swallows ANY exception from a layer, so a layer can be
    registered, run, and still hand back values attached to the wrong rows
    with nothing failing.

    Each player here scores a different constant, so every non-null value is
    that player's own number whatever the window or the minutes baseline
    contains. A misalignment across the builder's re-sort reads B's 50 on one
    of A's rows. Comparing against the layer called on its own would not work:
    the builder supplies MIN_SEASON and the bare layer has to build a minutes
    baseline from scratch at min_periods=3, so the two paths legitimately
    average different numbers of weighted games early in a season.
    """
    from src.features.builder import build_feature_matrix

    a = panel(10, player="A")
    a["PTS"] = 10.0
    b = panel(10, player="B")
    b["PTS"] = 50.0
    frame = pd.concat([b, a], ignore_index=True)
    frame["TEAM_ABBREVIATION"] = "LAL"
    frame["OPPONENT_ABBREVIATION"] = "BOS"
    frame["GAME_ID"] = [f"g{i:03d}" for i in range(len(frame))]
    shuffled = frame.sample(frac=1.0, random_state=19).reset_index(drop=True)

    built = build_feature_matrix(shuffled)
    values = built[["PLAYER_ID", "PTS_MW_L5"]].dropna()
    assert len(values) >= 8, "no values to check alignment on"
    assert set(values.loc[values["PLAYER_ID"] == "A", "PTS_MW_L5"]) == {10.0}
    assert set(values.loc[values["PLAYER_ID"] == "B", "PTS_MW_L5"]) == {50.0}


def test_the_measured_correlation_is_recorded_where_the_others_are():
    """An undocumented exclusion reads as an oversight."""
    import pathlib

    labels = (
        pathlib.Path(__file__).parent.parent / "src" / "models" / "labels.py"
    ).read_text()
    assert "_MW_L5" in labels
    assert "minutes_weighted" in labels
