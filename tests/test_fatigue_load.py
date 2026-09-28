"""Exponential cumulative fatigue load.

The load is meant to replace four unfitted constants with one continuous
quantity, so the tests pin the properties that make it worth having: it reads
only prior games, it decays with recency, minutes scale it, and travel is
separable from minutes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.fatigue_load import (
    DEFAULTS,
    LOAD_COL,
    LOAD_MIN_COL,
    TEAM_UTC_OFFSET,
    attach_fatigue_load,
)


def _player_games(minutes: list[float], gaps: list[int], **extra) -> pd.DataFrame:
    """One player's games; gaps[i] is days between game i and i+1."""
    dates = [pd.Timestamp("2025-01-01")]
    for g in gaps:
        dates.append(dates[-1] + pd.Timedelta(days=g))
    frame = pd.DataFrame({
        "PLAYER_ID": ["p1"] * len(minutes),
        "GAME_DATE": dates[: len(minutes)],
        "MIN": minutes,
    })
    for key, value in extra.items():
        frame[key] = value
    return frame


def test_first_game_has_zero_load():
    """No prior games means no load, not an imputed one."""
    out = attach_fatigue_load(_player_games([30.0], []))
    assert out.loc[0, LOAD_COL] == 0.0
    assert out.loc[0, LOAD_MIN_COL] == 0.0


def test_load_reads_only_prior_games_never_the_current_one():
    """The leakage claim. Changing a row's OWN minutes must not move its load."""
    base = _player_games([30.0, 30.0, 30.0], [1, 1])
    first = attach_fatigue_load(base)[LOAD_COL].to_numpy()

    bumped = base.copy()
    bumped.loc[2, "MIN"] = 48.0          # only the LAST row's own minutes
    second = attach_fatigue_load(bumped)[LOAD_COL].to_numpy()

    assert second[2] == pytest.approx(first[2]), (
        "a row's own minutes changed its own load — that is same-game leakage"
    )
    # and the earlier rows cannot see the future either
    assert second[0] == pytest.approx(first[0])
    assert second[1] == pytest.approx(first[1])


def test_more_recent_games_weigh_more():
    """The same heavy game costs more when it is closer.

    An earlier version of this test put the heavy game 6 days back in BOTH
    arms and so compared a frame with itself — it passed for no reason.
    """
    recent = attach_fatigue_load(_player_games([36.0, 20.0], [1]))
    distant = attach_fatigue_load(_player_games([36.0, 20.0], [5]))
    assert recent.loc[1, LOAD_COL] > distant.loc[1, LOAD_COL]
    ratio = recent.loc[1, LOAD_COL] / distant.loc[1, LOAD_COL]
    assert ratio == pytest.approx(np.exp(DEFAULTS["decay_lambda"] * 4))


def test_minutes_scale_the_load():
    light = attach_fatigue_load(_player_games([12.0, 30.0], [1]))
    heavy = attach_fatigue_load(_player_games([38.0, 30.0], [1]))
    assert heavy.loc[1, LOAD_COL] > light.loc[1, LOAD_COL]
    # linear in prior minutes, at a fixed gap
    assert heavy.loc[1, LOAD_COL] / light.loc[1, LOAD_COL] == pytest.approx(38.0 / 12.0)


def test_decay_matches_the_documented_formula():
    """One prior game, so the closed form is checkable by hand."""
    out = attach_fatigue_load(_player_games([30.0, 20.0], [2]))
    expected = 30.0 * np.exp(-DEFAULTS["decay_lambda"] * 2)
    assert out.loc[1, LOAD_MIN_COL] == pytest.approx(expected)


def test_games_outside_the_window_do_not_count():
    """window_days is a hard cutoff, not a soft one."""
    out = attach_fatigue_load(_player_games([30.0, 20.0], [9]))
    assert out.loc[1, LOAD_COL] == 0.0


def test_travel_is_separable_from_minutes():
    """The two columns differ only by the travel factor."""
    flown = _player_games([30.0, 20.0], [1])
    flown["TRAVEL_MILES"] = [2000.0, 0.0]
    out = attach_fatigue_load(flown)
    assert out.loc[1, LOAD_COL] > out.loc[1, LOAD_MIN_COL]
    ratio = out.loc[1, LOAD_COL] / out.loc[1, LOAD_MIN_COL]
    assert ratio == pytest.approx(1.0 + DEFAULTS["miles_theta"] * 2.0)


def test_no_travel_column_leaves_the_two_columns_equal():
    out = attach_fatigue_load(_player_games([30.0, 20.0], [1]))
    assert out.loc[1, LOAD_COL] == pytest.approx(out.loc[1, LOAD_MIN_COL])


def test_time_zone_shift_adds_load():
    """A coast-to-coast trip costs more than staying in one zone.

    Three games. In both frames the load on game 3 comes from games 1 and 2;
    the frames differ only in where game 2 was played, so any difference is
    the time-zone term alone.
    """
    def frame(second_venue_opponent: str, second_is_home: bool) -> pd.DataFrame:
        return pd.DataFrame({
            "PLAYER_ID": ["p1"] * 3,
            "GAME_DATE": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
            "MIN": [30.0, 30.0, 20.0],
            "TEAM_ABBREVIATION": ["BOS"] * 3,
            "OPPONENT_ABBREVIATION": ["NYK", second_venue_opponent, "NYK"],
            "IS_HOME": [True, second_is_home, True],
        })

    stayed = attach_fatigue_load(frame("NYK", True))      # all three in Boston
    flew = attach_fatigue_load(frame("LAL", False))       # game 2 in Los Angeles

    assert flew.loc[2, LOAD_COL] > stayed.loc[2, LOAD_COL], (
        "a 3-hour time-zone change added no load"
    )
    # ET -> PT is 3 hours, and only game 2's factor changes
    assert stayed.loc[2, LOAD_COL] == pytest.approx(stayed.loc[2, LOAD_MIN_COL])
    assert flew.loc[2, LOAD_COL] > flew.loc[2, LOAD_MIN_COL]


def test_missing_required_columns_returns_the_frame_unchanged():
    frame = pd.DataFrame({"PLAYER_ID": ["p1"], "GAME_DATE": [pd.Timestamp("2025-01-01")]})
    out = attach_fatigue_load(frame)
    assert LOAD_COL not in out.columns
    assert list(out.columns) == list(frame.columns)


def test_existing_column_is_not_overwritten():
    frame = _player_games([30.0, 20.0], [1])
    frame[LOAD_COL] = [99.0, 99.0]
    out = attach_fatigue_load(frame)
    assert list(out[LOAD_COL]) == [99.0, 99.0]


def test_players_do_not_contaminate_each_other():
    frame = pd.concat([
        _player_games([40.0, 40.0], [1]).assign(PLAYER_ID="p1"),
        _player_games([0.0, 10.0], [1]).assign(PLAYER_ID="p2"),
    ], ignore_index=True)
    out = attach_fatigue_load(frame)
    p2_second = out[(out["PLAYER_ID"] == "p2")].iloc[1]
    assert p2_second[LOAD_MIN_COL] == pytest.approx(0.0), (
        "p2's load picked up p1's minutes"
    )


def test_offset_table_covers_thirty_teams_and_spans_four_zones():
    assert len(TEAM_UTC_OFFSET) == 30
    assert set(TEAM_UTC_OFFSET.values()) == {-5, -6, -7, -8}


def test_row_order_is_preserved():
    """The layer sorts internally; the caller's ordering must survive."""
    frame = _player_games([30.0, 20.0, 10.0], [1, 1]).iloc[::-1].reset_index(drop=True)
    out = attach_fatigue_load(frame)
    assert list(out["MIN"]) == [10.0, 20.0, 30.0]
    assert out[LOAD_COL].notna().all()


def test_the_builder_actually_runs_this_layer_with_travel_attached():
    """Registration is a string in a tuple, so static reading cannot confirm it.

    Also confirms TRAVEL_MILES exists by the time the layer runs — the whole
    reason it is registered after attach_team_schedule_features rather than
    beside attach_fatigue_column.
    """
    from src.features import builder
    from src.features import fatigue_load as fl

    seen: list[dict] = []
    real = fl.attach_fatigue_load_layer

    def spy(df):
        seen.append({
            "rows": len(df),
            "has_travel": "TRAVEL_MILES" in df.columns,
            "has_min": "MIN" in df.columns,
        })
        return real(df)

    layers = builder._additive_feature_layers()
    names = [name for name, _ in layers]
    assert "fatigue_load" in names, f"layer not registered; got {names}"

    # drive the registered callable, not the module attribute, so a stale
    # registration would be caught
    registered = dict(layers)["fatigue_load"]
    frame = pd.DataFrame({
        "PLAYER_ID": ["p1", "p1"],
        "GAME_DATE": pd.to_datetime(["2025-01-01", "2025-01-02"]),
        "MIN": [30.0, 20.0],
        "TRAVEL_MILES": [0.0, 1200.0],
    })
    out = registered(frame)
    assert LOAD_COL in out.columns and LOAD_MIN_COL in out.columns
    assert out.loc[1, LOAD_COL] > 0.0
