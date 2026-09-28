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
    """p2's dates are staggered AFTER p1's, and p2's first row must be zero.

    The earlier version gave p2 the same dates as p1 and a zero-minute prior
    game, so the assertion held whether or not the shift was grouped by player
    — it could not have failed. Now p1 has heavy minutes immediately before
    p2's first game, so an ungrouped shift would hand p2 p1's load.
    """
    p1 = pd.DataFrame({
        "PLAYER_ID": ["p1"] * 2,
        "GAME_DATE": pd.to_datetime(["2025-01-01", "2025-01-02"]),
        "MIN": [40.0, 40.0],
    })
    p2 = pd.DataFrame({
        "PLAYER_ID": ["p2"] * 2,
        "GAME_DATE": pd.to_datetime(["2025-01-03", "2025-01-04"]),
        "MIN": [10.0, 10.0],
    })
    out = attach_fatigue_load(pd.concat([p1, p2], ignore_index=True))
    p2_rows = out[out["PLAYER_ID"] == "p2"].reset_index(drop=True)

    assert p2_rows.loc[0, LOAD_MIN_COL] == pytest.approx(0.0), (
        "p2's first game has no prior game of its own, so any load is p1's"
    )
    # and p2's second row sees only p2's own 10-minute game
    assert p2_rows.loc[1, LOAD_MIN_COL] == pytest.approx(
        10.0 * np.exp(-DEFAULTS["decay_lambda"])
    )


def test_teammates_on_the_same_team_game_get_the_same_travel_load():
    """A player-game panel repeats each TEAM-game once per player.

    Shifting the venue within the team over PLAYER rows credits the trip to
    whichever teammate sorts first. Measured before the fix on this exact
    frame: 48.355812 for one teammate and 44.671524 for the other.
    """
    rows = []
    for day, opponent, is_home in (
        ("2025-01-01", "NYK", True),
        ("2025-01-02", "LAL", False),   # Boston -> Los Angeles, 3 hours
        ("2025-01-03", "NYK", True),
    ):
        for player in ("p1", "p2"):
            rows.append({
                "PLAYER_ID": player,
                "GAME_DATE": pd.Timestamp(day),
                "MIN": 30.0,
                "TEAM_ABBREVIATION": "BOS",
                "OPPONENT_ABBREVIATION": opponent,
                "IS_HOME": is_home,
            })
    out = attach_fatigue_load(pd.DataFrame(rows))

    third = out[out["GAME_DATE"] == pd.Timestamp("2025-01-03")]
    loads = third[LOAD_COL].round(9).unique()
    assert len(loads) == 1, f"teammates travelled together but got {loads}"
    # and the trip did register, so this is not passing by both being zero
    assert third[LOAD_COL].iloc[0] > third[LOAD_MIN_COL].iloc[0]


def test_a_neutral_site_game_does_not_invent_a_trip():
    """The nominal home team's arena is not the venue of a neutral-site game."""
    def frame(neutral: bool) -> pd.DataFrame:
        return pd.DataFrame({
            "PLAYER_ID": ["p1"] * 3,
            "GAME_DATE": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
            "MIN": [30.0, 30.0, 20.0],
            "TEAM_ABBREVIATION": ["BOS"] * 3,
            "OPPONENT_ABBREVIATION": ["NYK", "LAL", "NYK"],
            "IS_HOME": [True, False, True],
            "IS_NEUTRAL_SITE": [False, neutral, False],
        })

    real_trip = attach_fatigue_load(frame(neutral=False))
    neutral_site = attach_fatigue_load(frame(neutral=True))

    assert real_trip.loc[2, LOAD_COL] > real_trip.loc[2, LOAD_MIN_COL]
    assert neutral_site.loc[2, LOAD_COL] == pytest.approx(
        neutral_site.loc[2, LOAD_MIN_COL]
    ), "a neutral-site game was charged the nominal home arena's time zone"


def test_a_duplicated_index_is_handled_positionally():
    """A caller's index may legitimately repeat; labels would misalign."""
    frame = _player_games([30.0, 20.0, 10.0], [1, 1])
    frame.index = [7, 7, 7]
    out = attach_fatigue_load(frame)
    assert list(out["MIN"]) == [30.0, 20.0, 10.0]
    assert out[LOAD_COL].notna().all()
    assert out[LOAD_COL].iloc[0] == 0.0
    assert out[LOAD_COL].iloc[1] > 0.0


def test_offset_table_covers_thirty_teams_and_spans_four_zones():
    assert len(TEAM_UTC_OFFSET) == 30
    assert set(TEAM_UTC_OFFSET.values()) == {-5, -6, -7, -8}


def test_row_order_is_preserved():
    """The layer sorts internally; the caller's ordering must survive."""
    frame = _player_games([30.0, 20.0, 10.0], [1, 1]).iloc[::-1].reset_index(drop=True)
    out = attach_fatigue_load(frame)
    assert list(out["MIN"]) == [10.0, 20.0, 30.0]
    assert out[LOAD_COL].notna().all()


def test_the_builder_runs_this_layer_after_travel_features_are_attached():
    """Drives build_feature_matrix and asserts what the layer actually received.

    The earlier version built a spy and then never installed it — it called the
    registered callable directly on a frame it had pre-populated with
    TRAVEL_MILES, so it could not have caught an ordering regression. This one
    patches the module attribute the builder resolves at call time, runs the
    real builder on the demo panel, and asserts TRAVEL_MILES was present when
    the layer ran. Reordering fatigue_load before
    attach_team_schedule_features makes it fail.
    """
    import src.features.fatigue_load as fl
    from src.features.builder import build_feature_matrix
    from src.models.data_audit import make_demo_panel

    seen: list[dict] = []
    real = fl.attach_fatigue_load

    def spy(df, cfg=None):
        seen.append({
            "rows": len(df),
            "has_travel": "TRAVEL_MILES" in df.columns,
            "travel_non_null": (
                int(pd.to_numeric(df["TRAVEL_MILES"], errors="coerce").notna().sum())
                if "TRAVEL_MILES" in df.columns else 0
            ),
        })
        return real(df, cfg)

    original = fl.attach_fatigue_load
    fl.attach_fatigue_load = spy
    try:
        built = build_feature_matrix(make_demo_panel())
    finally:
        fl.attach_fatigue_load = original

    assert seen, "the builder never invoked the fatigue_load layer"
    assert seen[0]["has_travel"], (
        "fatigue_load ran BEFORE attach_team_schedule_features, so its "
        "distance term was silently zero"
    )
    assert seen[0]["travel_non_null"] > 0, (
        "TRAVEL_MILES was present but entirely null when the layer ran"
    )
    assert LOAD_COL in built.columns and LOAD_MIN_COL in built.columns
    assert built[LOAD_COL].notna().all()
    assert float(built[LOAD_COL].sum()) > 0.0, "every load came out zero"


def test_the_layer_is_registered_under_its_expected_label():
    from src.features import builder

    names = [name for name, _ in builder._additive_feature_layers()]
    assert "fatigue_load" in names, f"layer not registered; got {names}"
