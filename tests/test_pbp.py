"""Tests for the play-by-play feature layer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.pbp import (
    PBP_DIAGNOSTIC_COLS,
    PBP_RATE_COLS,
    PbpFeatureError,
    attach_pbp_rolling_features,
    elapsed_seconds,
    parse_clock_seconds,
    prepare_events,
    reconstruct_on_court,
    summarise_shots,
    validate_on_court_against_minutes,
)


def _events() -> pd.DataFrame:
    """One tiny game: two players, a few shots, one substitution each."""
    rows = [
        # (actionType, subType, person, period, clock, dist, result, assist, home, away)
        ("period", "start", None, 1, "PT12M00.00S", None, None, None, 0, 0),
        ("2pt", "Layup", "p1", 1, "PT11M00.00S", 2.0, "Made", "p2", 2, 0),
        ("3pt", "Jump Shot", "p2", 1, "PT10M00.00S", 25.0, "Missed", None, 2, 0),
        ("2pt", "Jump Shot", "p1", 1, "PT09M00.00S", 18.0, "Made", None, 4, 0),
        ("substitution", "out", "p1", 1, "PT06M00.00S", None, None, None, 4, 2),
        ("substitution", "in", "p3", 1, "PT06M00.00S", None, None, None, 4, 2),
        ("3pt", "Jump Shot", "p2", 2, "PT06M00.00S", 26.0, "Made", "p3", 30, 4),
        ("substitution", "in", "p1", 3, "PT12M00.00S", None, None, None, 60, 20),
        ("2pt", "DUNK", "p1", 3, "PT06M00.00S", 1.0, "Made", None, 62, 20),
        ("period", "end", None, 4, "PT00M00.00S", None, None, None, 100, 70),
    ]
    return pd.DataFrame([
        {
            "gameId": "0022500001", "actionType": a, "subType": st, "personId": p,
            "period": per, "clock": c, "shotDistance": d, "shotResult": r,
            "assistPersonId": asst, "scoreHome": h, "scoreAway": aw,
            "orderNumber": i * 10, "possession": "1610612747",
        }
        for i, (a, st, p, per, c, d, r, asst, h, aw) in enumerate(rows)
    ])


# --- the clock --------------------------------------------------------------


@pytest.mark.parametrize("text,expected", [
    ("PT12M00.00S", 720.0), ("PT00M04.30S", 4.3), ("PT05M30.00S", 330.0),
])
def test_clock_parses_to_seconds_remaining(text, expected):
    assert parse_clock_seconds(text) == pytest.approx(expected)


@pytest.mark.parametrize("bad", [None, "", "12:00", np.nan, 720])
def test_an_unparseable_clock_is_nan_not_zero(bad):
    """Zero would read as 'the period just ended', which is a real moment."""
    assert np.isnan(parse_clock_seconds(bad))


def test_elapsed_time_spans_regulation_and_overtime():
    period = pd.Series([1, 2, 4, 5])
    remaining = pd.Series([720.0, 0.0, 0.0, 0.0])
    got = elapsed_seconds(period, remaining).tolist()
    assert got == [0.0, 1440.0, 2880.0, 3180.0]   # OT adds 5 minutes, not 12


# --- shot profile -----------------------------------------------------------


def test_shot_profile_splits_rim_mid_and_three():
    shots = summarise_shots(prepare_events(_events()))
    p1 = shots.set_index("personId").loc["p1"]
    assert p1["PBP_FGA"] == 3
    assert p1["PBP_RIM_RATE"] == pytest.approx(2 / 3)       # 2 ft and 1 ft
    assert p1["PBP_MID_RATE"] == pytest.approx(1 / 3)       # the 18-footer
    assert p1["PBP_THREE_RATE"] == 0.0
    assert p1["PBP_DUNK_LAYUP_RATE"] == pytest.approx(2 / 3)


def test_assisted_rate_is_a_share_of_makes_not_of_attempts():
    shots = summarise_shots(prepare_events(_events()))
    s = shots.set_index("personId")
    # p1: three makes, one assisted.
    assert s.loc["p1", "PBP_ASSISTED_RATE"] == pytest.approx(1 / 3)
    # p2: two attempts, one make, and that make was assisted.
    assert s.loc["p2", "PBP_ASSISTED_RATE"] == pytest.approx(1.0)


def test_assisted_rate_is_undefined_without_a_make():
    ev = _events()
    ev.loc[ev["personId"] == "p1", "shotResult"] = "Missed"
    shots = summarise_shots(prepare_events(ev))
    assert np.isnan(shots.set_index("personId").loc["p1", "PBP_ASSISTED_RATE"])


def test_game_state_shares_use_the_score_margin():
    shots = summarise_shots(prepare_events(_events()))
    s = shots.set_index("personId")
    # p1's shots sit at margins of 2, 4 and 42.
    assert s.loc["p1", "PBP_CLOSE_SHOT_SHARE"] == pytest.approx(2 / 3)
    assert s.loc["p1", "PBP_GARBAGE_SHOT_SHARE"] == pytest.approx(1 / 3)


# --- on-court reconstruction ------------------------------------------------


def test_a_player_whose_first_substitution_is_out_started_the_game():
    oc = reconstruct_on_court(prepare_events(_events())).set_index("personId")
    # p1 played 0:00-6:00 of Q1 (360s) then from the start of Q3 to the end.
    assert oc.loc["p1", "PBP_SECONDS_ON_COURT"] == pytest.approx(360.0 + 1440.0)


def test_a_player_subbed_in_does_not_get_credit_from_tip():
    oc = reconstruct_on_court(prepare_events(_events())).set_index("personId")
    # p3 came in at 6:00 of Q1 and was never subbed out.
    assert oc.loc["p3", "PBP_SECONDS_ON_COURT"] == pytest.approx(2880.0 - 360.0)


def test_the_reconstruction_is_checked_against_the_box_score():
    """The starter assumption is tested, not believed."""
    oc = reconstruct_on_court(prepare_events(_events()))
    panel = pd.DataFrame({
        "GAME_ID": ["0022500001"] * 2, "PLAYER_ID": ["p1", "p3"],
        "MIN": [30.0, 42.0],
    })
    report = validate_on_court_against_minutes(oc, panel)
    assert report["n"] == 2
    assert set(report) >= {"corr", "mean_abs_error_minutes", "within_2_minutes"}


# --- leakage ----------------------------------------------------------------


def test_a_row_never_sees_its_own_game():
    """Play-by-play describes what happened DURING a game, so same-game values
    are postgame information."""
    summaries = pd.DataFrame({
        "gameId": ["g1", "g2", "g3"], "personId": ["p", "p", "p"],
        "PBP_RIM_RATE": [0.1, 0.9, 0.5], "PBP_THREE_RATE": [0.2, 0.3, 0.4],
    })
    panel = pd.DataFrame({
        "PLAYER_ID": ["p", "p", "p"], "GAME_ID": ["g1", "g2", "g3"],
        "GAME_DATE": pd.to_datetime(["2025-10-21", "2025-10-23", "2025-10-25"]),
    })
    out = attach_pbp_rolling_features(panel, summaries, windows=(2,), min_periods=1)
    rim = out.sort_values("GAME_DATE")["PBP_RIM_RATE_L2"].tolist()
    assert np.isnan(rim[0])                       # nothing before the first game
    assert rim[1] == pytest.approx(0.1)           # only g1
    assert rim[2] == pytest.approx(0.5)           # mean of g1 and g2, not g3


def test_same_game_columns_are_not_carried_onto_the_panel():
    summaries = pd.DataFrame({
        "gameId": ["g1"], "personId": ["p"], "PBP_RIM_RATE": [0.4], "PBP_FGA": [9],
    })
    panel = pd.DataFrame({
        "PLAYER_ID": ["p"], "GAME_ID": ["g1"],
        "GAME_DATE": pd.to_datetime(["2025-10-21"]),
    })
    out = attach_pbp_rolling_features(panel, summaries, windows=(3,))
    assert "PBP_RIM_RATE" not in out.columns
    assert "PBP_FGA" not in out.columns
    assert "PBP_RIM_RATE_L3" in out.columns


def test_counts_and_duplicated_minutes_are_not_offered_as_features():
    """PBP_FGA is a count; PBP_SECONDS_ON_COURT correlates with the box
    score's MIN at 0.997, and MIN is already a feature."""
    assert "PBP_FGA" in PBP_DIAGNOSTIC_COLS
    assert "PBP_SECONDS_ON_COURT" in PBP_DIAGNOSTIC_COLS
    assert not set(PBP_RATE_COLS) & set(PBP_DIAGNOSTIC_COLS)
    # Pace IS shipped: possessions per 48 on court is not in the box score.
    assert "PBP_GAME_PACE" in PBP_RATE_COLS


def test_pbp_features_never_reach_a_postgame_feature_list():
    from src.models.compare import POSTGAME_ONLY_COLS
    from src.models.labels import default_feature_cols

    for market in ("PTS", "REB", "AST", "FG3M", "PRA"):
        named = [c for c in default_feature_cols(market) if c.startswith("PBP_")]
        assert not set(named) & POSTGAME_ONLY_COLS
        # Every pbp feature named is a ROLLED one, never a same-game column.
        assert all("_L" in c for c in named), named


def test_missing_columns_are_refused_rather_than_guessed():
    with pytest.raises(PbpFeatureError, match="DATA_NOT_AVAILABLE"):
        prepare_events(pd.DataFrame({"gameId": ["g"]}))
    with pytest.raises(PbpFeatureError, match="DATA_NOT_AVAILABLE"):
        attach_pbp_rolling_features(
            pd.DataFrame({"PLAYER_ID": ["p"]}),
            pd.DataFrame({"gameId": ["g"], "personId": ["p"]}),
        )


# --- log completeness -------------------------------------------------------


def _log_and_panel(n_games: int = 40, fga_per_game: int = 180, keep: float = 1.0,
                   season: str = "2025-26"):
    """An event log and the box score it should agree with."""
    events, rows = [], []
    for g in range(n_games):
        gid = f"002250{g:04d}"
        kept = int(fga_per_game * keep)
        for i in range(kept):
            events.append({
                "gameId": gid, "actionType": "2pt", "period": 1,
                "clock": "PT06M00.00S", "orderNumber": i,
            })
        rows.append({"GAME_ID": gid, "FGA": float(fga_per_game), "SEASON": season})
    return pd.DataFrame(events), pd.DataFrame(rows)


def test_a_complete_log_passes_the_completeness_check():
    from src.features.pbp import check_log_completeness

    events, panel = _log_and_panel()
    report = check_log_completeness(prepare_events(events), panel)
    assert report["failing"] == []
    assert report["seasons"]["2025-26"]["exact_share"] == 1.0
    assert report["seasons"]["2025-26"]["median_gap"] == 0.0


def test_a_half_supplied_log_is_caught():
    """Five of about eleven parts named every game and was short by half of
    each. Nothing about its shape said 'partial' except this check."""
    from src.features.pbp import check_log_completeness

    events, panel = _log_and_panel(keep=0.5)
    report = check_log_completeness(prepare_events(events), panel)
    assert report["failing"] == ["2025-26"]
    assert report["seasons"]["2025-26"]["exact_share"] == 0.0
    assert report["seasons"]["2025-26"]["median_gap"] == 90.0


def test_completeness_is_reported_per_season_not_pooled():
    """One whole season must not mask a missing one."""
    from src.features.pbp import check_log_completeness

    good_ev, good_panel = _log_and_panel(n_games=30, season="2025-26")
    bad_ev, bad_panel = _log_and_panel(n_games=30, keep=0.5, season="2023-24")
    bad_ev["gameId"] = bad_ev["gameId"].str.replace("002250", "002230", regex=False)
    bad_panel["GAME_ID"] = bad_panel["GAME_ID"].str.replace("002250", "002230", regex=False)

    report = check_log_completeness(
        prepare_events(pd.concat([good_ev, bad_ev], ignore_index=True)),
        pd.concat([good_panel, bad_panel], ignore_index=True),
    )
    assert report["failing"] == ["2023-24"]
    assert report["seasons"]["2025-26"]["exact_share"] == 1.0


def test_summaries_can_refuse_to_build_on_an_incomplete_log():
    from src.features.pbp import PbpFeatureError, summarise_player_games

    events, panel = _log_and_panel(keep=0.5)
    events["personId"] = "p1"
    events["shotDistance"] = 10.0
    events["shotResult"] = "Made"
    panel["PLAYER_ID"] = "p1"
    panel["MIN"] = 30.0
    with pytest.raises(PbpFeatureError, match="incomplete"):
        summarise_player_games(events, panel=panel, require_complete=True)
    # Default is to warn and continue, so an exploratory run still works.
    out = summarise_player_games(events, panel=panel)
    assert out.attrs["log_completeness"]["failing"] == ["2025-26"]


def test_a_panel_without_attempts_cannot_be_checked_against():
    from src.features.pbp import PbpFeatureError, check_log_completeness

    events, panel = _log_and_panel()
    with pytest.raises(PbpFeatureError, match="DATA_NOT_AVAILABLE"):
        check_log_completeness(prepare_events(events), panel.drop(columns=["FGA"]))


def test_completeness_check_accepts_a_raw_log_with_integer_game_ids():
    """The 2021-22 upload read gameId as int64 because those ids carry no
    leading zero, and the check raised a dtype error on the merge instead of
    reporting on the log. It is the guard you run BEFORE anything else
    touches a log, so it cannot presuppose prepare_events having cast."""
    from src.features.pbp import check_log_completeness

    events, panel = _log_and_panel(keep=0.5)
    raw = events.copy()
    raw["gameId"] = raw["gameId"].str.lstrip("0").astype("int64")
    panel = panel.copy()
    panel["GAME_ID"] = panel["GAME_ID"].str.lstrip("0").astype("int64")

    report = check_log_completeness(raw, panel)

    assert report["failing"] == ["2025-26"]
    assert report["seasons"]["2025-26"]["games"] == 40.0
    assert report["seasons"]["2025-26"]["median_gap"] == 90.0


def test_prepare_events_is_idempotent():
    """Preparing twice must return the same frame, not a second copy. The
    panel build prepares once and hands the result to functions that each
    prepare defensively; on a full multi-season log a redundant copy is
    gigabytes, and the second one OOM-killed the rebuild."""
    events, _ = _log_and_panel(n_games=3, fga_per_game=10)

    once = prepare_events(events)
    twice = prepare_events(once)

    assert twice is once, "already-prepared frame was copied again"
    pd.testing.assert_frame_equal(once, twice)


def test_prepare_events_still_prepares_a_raw_frame_with_integer_game_ids():
    """The idempotence guard must not mistake a raw frame for a prepared one."""
    events, _ = _log_and_panel(n_games=3, fga_per_game=10)
    raw = events.copy()
    raw["gameId"] = raw["gameId"].str.lstrip("0").astype("int64")

    out = prepare_events(raw)

    assert pd.api.types.is_string_dtype(out["gameId"])
    assert {"clock_seconds", "elapsed", "margin_abs"}.issubset(out.columns)
    assert out["clock_seconds"].notna().all()


def _game_with_two_players_of_different_minutes():
    """One game: a starter who plays throughout, a bench player who does not."""
    rows = []
    n = 0

    def ev(**kw):
        nonlocal n
        n += 1
        rows.append({"gameId": "0022500001", "period": 1, "orderNumber": n,
                     "clock": kw.pop("clock", "PT06M00.00S"), **kw})

    # Possession alternates every event, so the count covers BOTH teams.
    for i in range(40):
        ev(actionType="2pt", personId="starter" if i % 2 else "bench",
           possession="1610612737" if i % 2 else "1610612738",
           shotResult="Made", shotDistance=5.0)
    # Both players start (each one's first substitution is an "out"), but the
    # bench player sits at the 6:00 mark while the starter plays to the buzzer.
    # reconstruct_on_court only emits a row for a player with a substitution,
    # so the starter needs one too.
    ev(actionType="substitution", subType="out", personId="bench",
       possession="1610612737", clock="PT06M00.00S")
    ev(actionType="substitution", subType="out", personId="starter",
       possession="1610612737", clock="PT00M00.00S")
    ev(actionType="period", subType="end", personId=None,
       possession="1610612737", clock="PT00M00.00S")
    return pd.DataFrame(rows)


def test_game_pace_is_constant_across_every_player_in_the_game():
    """PBP_GAME_PACE was once called PBP_PACE_ON_COURT and documented as a
    per-player measurement. The player's seconds cancel out of the arithmetic,
    so it never was one. This pins that, so nobody re-documents it as
    player-specific or reintroduces a player term believing it varies."""
    from src.features.pbp import summarise_player_games

    summary = summarise_player_games(_game_with_two_players_of_different_minutes())
    pace = summary["PBP_GAME_PACE"].dropna()
    seconds = summary["PBP_SECONDS_ON_COURT"].dropna()

    assert len(pace) >= 2, "need at least two players to compare"
    assert seconds.nunique() > 1, "the two players must differ in minutes"
    assert pace.nunique() == 1, (
        f"PBP_GAME_PACE varies within one game: {sorted(pace.unique())}. "
        "It is a game constant by construction."
    )


def test_game_pace_is_per_team_not_both_teams():
    """team_possessions counts changes of the possessing team, so its total
    covers both teams. Reporting that un-halved put 'pace' near 200, double
    the league convention, and made the column unreadable against any
    published pace figure."""
    from src.features.pbp import prepare_events, summarise_player_games, team_possessions

    events = _game_with_two_players_of_different_minutes()
    both_teams = len(team_possessions(prepare_events(events)))
    game_seconds = prepare_events(events)["elapsed"].max()

    pace = summarise_player_games(events)["PBP_GAME_PACE"].dropna().iloc[0]
    expected = (both_teams / 2.0) * 2880.0 / game_seconds

    assert pace == pytest.approx(expected), f"{pace} != {expected}"
    assert pace == pytest.approx(both_teams * 2880.0 / game_seconds / 2.0)


def test_a_log_missing_whole_games_is_caught():
    """Scoring only the games both sides carried was a hole big enough to drive
    the whole failure mode through: 234 of 2021-22's 1,230 games, each one
    individually complete, returned failing: []. A game the log never mentions
    counts as zero attempts, not as absent."""
    from src.features.pbp import check_log_completeness

    events, panel = _log_and_panel(n_games=40)
    kept = sorted(panel["GAME_ID"])[:8]
    partial = events[events["gameId"].isin(kept)]

    report = check_log_completeness(prepare_events(partial), panel)

    assert report["failing"] == ["2025-26"]
    assert report["seasons"]["2025-26"]["games"] == 40.0, "all panel games must be scored"
    assert report["seasons"]["2025-26"]["games_absent"] == 32.0
    assert report["seasons"]["2025-26"]["exact_share"] == pytest.approx(8 / 40)


def test_a_whole_log_reports_no_absent_games():
    """The fix must not turn a complete log into a failure."""
    from src.features.pbp import check_log_completeness

    events, panel = _log_and_panel(n_games=40)
    report = check_log_completeness(prepare_events(events), panel)

    assert report["failing"] == []
    assert report["seasons"]["2025-26"]["games_absent"] == 0.0
    assert report["seasons"]["2025-26"]["exact_share"] == 1.0
