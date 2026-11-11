"""
Tests for src/ingestion/basketball_reference.py.

The CSV excerpts below are a handful of rows from Basketball-Reference's
2025-26 season tables, kept only because the parser's whole job is to
survive their real quirks — the two-row header with repeated column names,
the traded-player blocks, the League Average trailer and the blank
percentage cells. Data from Basketball-Reference.com (Sports Reference
LLC); when using SR data, please cite them and provide a link and/or a
mention: https://www.basketball-reference.com/
"""

from __future__ import annotations

import io
from datetime import date

import pandas as pd
import pytest

from src.ingestion.basketball_reference import (
    SR_ATTRIBUTION,
    BasketballReferenceError,
    PriorSeasonJoinReport,
    SeasonAggregateLeakageError,
    attach_prior_season_features,
    describe_sr_csv,
    infer_table_kind,
    league_average_row,
    multi_team_report,
    normalise_player_name,
    prior_season_features,
    read_sr_season_csv,
    season_totals,
    team_splits,
)

# --- fixtures: real SR shapes -------------------------------------------

PER_GAME_CSV = (
    "Rk,Player,Age,Team,Pos,G,GS,MP,FG,FGA,FG%,3P,3PA,3P%,2P,2PA,2P%,eFG%,FT,FTA,"
    "FT%,ORB,DRB,TRB,AST,STL,BLK,TOV,PF,PTS,Awards,Player-additional\n"
    "1,Luka Dončić,26,LAL,PG,64,64,35.8,10.8,22.8,.476,4.0,10.8,.366,6.9,11.9,.575,"
    ".563,7.9,10.1,.780,0.6,7.1,7.7,8.3,1.6,0.5,4.0,2.4,33.5,MVP-4CPOY-8ASNBA1,doncilu01\n"
    "49,Jalen Duren,22,DET,C,70,70,28.2,7.5,11.5,.650,0.0,0.0,,7.5,11.5,.650,.650,"
    "4.6,6.1,.747,3.8,6.7,10.5,2.0,0.8,0.8,1.9,2.8,19.5,DPOY-11ASNBA3,durenja01\n"
    "582,Chucky Hepburn,22,TOR,PG,2,0,6.5,0.0,3.0,.000,0.0,2.5,.000,0.0,0.5,.000,"
    ".000,0.0,0.0,,0.0,0.5,0.5,1.0,0.5,0.0,0.5,1.0,0.0,,hepbuch01\n"
    ",League Average,,,,,,,,,.471,,,.360,,,.550,.546,,,.783,,,,,,,,,,,-9999\n"
)

# The play-by-play table's header is TWO rows, and the second row repeats
# 'Shoot' and 'Off.' under two different groups.
PLAY_BY_PLAY_CSV = (
    ",,,,,,,,Position Estimate,Position Estimate,Position Estimate,Position Estimate,"
    "Position Estimate,+/- Per 100 Poss,+/- Per 100 Poss,Turnovers,Turnovers,"
    "Fouls Committed,Fouls Committed,Fouls Drawn,Fouls Drawn,Misc.,Misc.,Misc.,,-additional\n"
    "Rk,Player,Age,Team,Pos,G,GS,MP,PG%,SG%,SF%,PF%,C%,OnCourt,On-Off,BadPass,LostBall,"
    "Shoot,Off.,Shoot,Off.,PGA,And1,Blkd,Awards,-9999\n"
    "1,Amen Thompson,23,HOU,PG,79,79,2953,83,16,1,0,0,6.6,5.5,92,72,80,14,191,12,1040,51,68,DPOY-8,thompam01\n"
    "23,James Harden,36,2TM,PG,70,70,2438,46,52,2,0,0,1.4,-1.5,159,66,61,15,219,3,1310,39,80,,hardeja01\n"
    "23,James Harden,36,LAC,PG,44,44,1559,34,66,1,0,0,-0.1,-2.2,106,42,38,10,156,2,844,29,60,,hardeja01\n"
    "23,James Harden,36,CLE,PG,26,26,879,68,28,5,0,0,4.0,-0.3,53,24,23,5,63,1,466,10,20,,hardeja01\n"
    "43,Nikola Jokić,30,DEN,C,65,65,2265,0,0,0,0,100,10.7,13.5,151,67,75,14,174,4,1655,45,50,MVP-2CPOY-6ASNBA1,jokicni01\n"
)

ADJUSTED_SHOOTING_CSV = (
    ",,,,,,,,Shooting %,Shooting %,Shooting %,Shooting %,Shooting %,Shooting %,Shooting %,"
    "Shooting %,League-Adjusted,League-Adjusted,League-Adjusted,League-Adjusted,"
    "League-Adjusted,League-Adjusted,League-Adjusted,League-Adjusted,Added,Added,,-additional\n"
    "Rk,Player,Age,Team,Pos,G,GS,MP,FG%,2P%,3P%,eFG%,FT%,TS%,FTr,3PAr,FG+,2P+,3P+,eFG+,"
    "FT+,TS+,FTr+,3PAr+,FG Add,TS Add,Awards,-9999\n"
    "1,Amen Thompson,23,HOU,PG,79,79,2953,.534,.573,.216,.545,.779,.594,.374,.111,113,104,"
    "60,100,100,102,142,27,-0.3,29.5,DPOY-8,thompam01\n"
    "23,James Harden,36,2TM,PG,70,70,2438,.434,.495,.375,.529,.884,.610,.467,.510,92,90,104,"
    "97,113,105,177,123,-36.5,78.0,,hardeja01\n"
    "23,James Harden,36,LAC,PG,44,44,1559,.419,.492,.347,.506,.901,.598,.485,.505,89,89,96,"
    "93,115,103,184,122,-60.4,30.1,,hardeja01\n"
    "23,James Harden,36,CLE,PG,26,26,879,.466,.500,.435,.580,.840,.639,.426,.523,99,91,121,"
    "106,107,110,162,126,23.9,47.9,,hardeja01\n"
    ",League Average,,,,,,,.471,.550,.360,.546,.783,.581,.264,.415,,,,,,,,,,,,-9999\n"
)


# Per 100 Possessions: a single-row header, and the ONLY columns separating it
# from the per-game table are ORtg/DRtg. Koloko's 2-game LAL row has blank
# shooting and a blank ORtg because he never attempted a shot.
PER_100_CSV = (
    "Rk,Player,Age,Team,Pos,G,GS,MP,FG,FGA,FG%,3P,3PA,3P%,2P,2PA,2P%,eFG%,FT,FTA,"
    "FT%,ORB,DRB,TRB,AST,STL,BLK,TOV,PF,PTS,ORtg,DRtg,Awards,Player-additional\n"
    "37,Luka Don\u010di\u0107,26,LAL,PG,64,64,2289,14.8,31.1,.476,5.4,14.8,.366,9.4,16.3,"
    ".575,.563,10.7,13.8,.780,0.9,9.7,10.6,11.3,2.2,0.7,5.4,3.3,45.7,120,114,"
    "MVP-4CPOY-8ASNBA1,doncilu01\n"
    "84,Jalen Duren,22,DET,C,70,70,1976,12.8,19.7,.650,0.0,0.0,,12.8,19.7,.650,.650,"
    "7.9,10.5,.747,6.5,11.6,18.0,3.4,1.4,1.4,3.2,4.8,33.4,134,108,DPOY-11ASNBA3,durenja01\n"
    "421,Christian Koloko,25,3TM,C,27,4,357,4.1,9.8,.419,0.1,1.2,.111,4.0,8.6,.462,.426,"
    "1.3,2.0,.667,4.1,6.6,10.7,2.4,1.9,3.0,2.1,6.4,9.7,100,114,,kolokch01\n"
    "421,Christian Koloko,25,LAL,C,2,0,6,0.0,0.0,,0.0,0.0,,0.0,0.0,,,0.0,0.0,,0.0,8.1,"
    "8.1,0.0,0.0,0.0,0.0,8.1,0.0,,120,,kolokch01\n"
    "421,Christian Koloko,25,MEM,C,11,2,195,3.4,8.5,.400,0.0,0.2,.000,3.4,8.3,.412,.400,"
    "0.2,1.0,.250,4.6,6.1,10.7,2.4,2.4,3.2,1.7,4.9,7.0,98,115,,kolokch01\n"
    "421,Christian Koloko,25,ATL,C,14,2,156,5.1,11.8,.436,0.3,2.4,.125,4.8,9.4,.516,.449,"
    "2.7,3.3,.818,3.6,7.3,10.9,2.4,1.2,3.0,2.7,8.2,13.3,102,112,,kolokch01\n"
    ",League Average,,,,,,,,,.471,,,.360,,,.550,.546,,,.783,,,,,,,,,,,,,-9999\n"
)

def _read(csv_text: str, season: str = "2024-25"):
    return read_sr_season_csv(io.StringIO(csv_text), season=season)


# --- header / shape ------------------------------------------------------


def test_repeated_column_names_are_disambiguated_by_group():
    """
    'Shoot' appears twice in the play-by-play header — once under 'Fouls
    Committed', once under 'Fouls Drawn'. A naive read gives Shoot and
    Shoot.1 and invites mapping fouls drawn onto fouls committed.
    """
    table = _read(PLAY_BY_PLAY_CSV)
    cols = set(table.frame.columns)
    assert {"FOULS_COMMITTED_SHOOT", "FOULS_DRAWN_SHOOT"} <= cols
    assert {"FOULS_COMMITTED_OFF", "FOULS_DRAWN_OFF"} <= cols
    assert "SHOOT_2" not in cols

    thompson = table.frame.set_index("PLAYER").loc["Amen Thompson"]
    assert thompson["FOULS_COMMITTED_SHOOT"] == 80
    assert thompson["FOULS_DRAWN_SHOOT"] == 191


def test_table_kinds_are_inferred():
    assert _read(PER_GAME_CSV).kind == "per_game"
    assert _read(PER_100_CSV).kind == "per_100_poss"
    assert _read(PLAY_BY_PLAY_CSV).kind == "play_by_play"
    assert _read(ADJUSTED_SHOOTING_CSV).kind == "adjusted_shooting"
    assert infer_table_kind(["RK", "PLAYER"]) == "unknown"


def test_per_100_is_not_mistaken_for_per_game():
    """
    The two tables share every counting column; only ORtg/DRtg tell them
    apart. Misread, Dončić's 45.7 points per 100 possessions lands in the
    column that holds his 33.5 points per game — same name, different
    quantity, no error raised anywhere.
    """
    per_game = prior_season_features(
        _read(PER_GAME_CSV), target_season="2025-26", columns=("PTS",)
    ).set_index("PLAYER_KEY")
    per_100 = prior_season_features(
        _read(PER_100_CSV), target_season="2025-26", columns=("PTS",)
    ).set_index("PLAYER_KEY")

    assert "SR_PG_PRIOR_PTS" in per_game.columns
    assert "SR_P100_PRIOR_PTS" in per_100.columns
    assert per_game.at["luka doncic", "SR_PG_PRIOR_PTS"] == pytest.approx(33.5)
    assert per_100.at["luka doncic", "SR_P100_PRIOR_PTS"] == pytest.approx(45.7)
    shared_keys = {"BBREF_PLAYER_ID", "PLAYER", "SR_PRIOR_SEASON"}
    overlap = (set(per_game.columns) & set(per_100.columns)) - shared_keys
    assert overlap == set(), f"stat columns collide across table kinds: {overlap}"


def test_blank_offensive_rating_is_nan_not_a_zero_rating():
    """
    Koloko's 2-game LAL row has no shot attempts, so SR leaves ORtg blank.
    A 0.0 there is not a missing value — it is the worst offensive rating
    ever recorded.
    """
    frame = _read(PER_100_CSV).frame.set_index(["PLAYER", "TEAM"])
    assert pd.isna(frame.at[("Christian Koloko", "LAL"), "ORTG"])
    assert frame.at[("Christian Koloko", "LAL"), "DRTG"] == 120
    assert pd.isna(frame.at[("Christian Koloko", "LAL"), "FG_PCT"])


def test_three_team_blocks_collapse_like_two_team_blocks():
    table = _read(PER_100_CSV)
    totals = season_totals(table)
    koloko = totals[totals["BBREF_PLAYER_ID"] == "kolokch01"]
    assert len(koloko) == 1
    assert koloko["TEAM"].iloc[0] == "3TM"
    assert koloko["G"].iloc[0] == 27
    assert team_splits(table)[
        team_splits(table)["BBREF_PLAYER_ID"] == "kolokch01"
    ]["G"].sum() == 27


def test_a_title_pasted_onto_the_header_line_is_stripped():
    """SR pages put the table's name above the export; pasting glues it on."""
    titled = "Per 100 Possessions    " + PER_100_CSV
    table = _read(titled)
    assert table.kind == "per_100_poss"
    assert "RK" in table.frame.columns


def test_column_prefixes_keep_per_game_and_total_minutes_apart():
    """MP is minutes per game in one table and season minutes in another."""
    per_game = prior_season_features(
        _read(PER_GAME_CSV), target_season="2025-26", columns=("MP",)
    )
    pbp = prior_season_features(
        _read(PLAY_BY_PLAY_CSV), target_season="2025-26", columns=("MP",)
    )
    assert "SR_PG_PRIOR_MP" in per_game.columns
    assert "SR_PBP_PRIOR_MP" in pbp.columns
    assert set(per_game.columns) & set(pbp.columns) == {
        "BBREF_PLAYER_ID", "PLAYER", "PLAYER_KEY", "SR_PRIOR_SEASON",
    }


def test_bbref_player_id_is_exposed_as_a_join_key():
    table = _read(PER_GAME_CSV)
    assert list(table.frame["BBREF_PLAYER_ID"]) == ["doncilu01", "durenja01", "hepbuch01"]


# --- the four raw-CSV hazards -------------------------------------------


def test_league_average_trailer_is_not_a_player():
    table = _read(PER_GAME_CSV)
    assert "League Average" not in set(table.frame["PLAYER"])
    assert table.dropped_rows == 1
    assert league_average_row(table)["FG_PCT"] == pytest.approx(0.471)


def test_blank_percentage_is_nan_but_a_written_zero_is_zero():
    """
    Duren's 3P% is blank because he took no threes; writing 0.0 there would
    tell the model he is a 0% shooter rather than a non-shooter. Hepburn's
    '.000' is a real zero and must survive as one.
    """
    frame = _read(PER_GAME_CSV).frame.set_index("PLAYER")
    assert pd.isna(frame.at["Jalen Duren", "3P_PCT"])
    assert frame.at["Jalen Duren", "3PA"] == 0.0
    assert frame.at["Chucky Hepburn", "3P_PCT"] == 0.0
    assert pd.isna(frame.at["Chucky Hepburn", "FT_PCT"])


def test_traded_players_never_double_count():
    table = _read(PLAY_BY_PLAY_CSV)
    assert int(table.frame["IS_MULTI_TEAM_TOTAL"].sum()) == 1

    totals = season_totals(table)
    # Five raw rows, three players: Harden's 2TM block collapses to one row.
    assert len(table.frame) == 5
    assert len(totals) == 3
    assert set(totals["BBREF_PLAYER_ID"]) == {"thompam01", "hardeja01", "jokicni01"}
    harden = totals[totals["BBREF_PLAYER_ID"] == "hardeja01"]
    assert len(harden) == 1
    assert bool(harden["IS_MULTI_TEAM_TOTAL"].iloc[0])
    assert harden["G"].iloc[0] == 70              # the season total, not a split

    splits = team_splits(table)
    assert set(splits[splits["BBREF_PLAYER_ID"] == "hardeja01"]["TEAM"]) == {"LAC", "CLE"}
    assert splits[splits["BBREF_PLAYER_ID"] == "hardeja01"]["G"].sum() == 70


def test_multi_team_report_flags_a_games_mismatch():
    truncated = PLAY_BY_PLAY_CSV.replace(
        "23,James Harden,36,CLE,PG,26,26,879,68,28,5,0,0,4.0,-0.3,53,24,23,5,63,1,466,10,20,,hardeja01\n",
        "",
    )
    report = multi_team_report(_read(truncated))
    assert not bool(report["MATCHES"].all())


def test_season_totals_refuses_to_return_a_duplicated_player():
    """A block whose TOTAL row is missing would otherwise pass two rows through."""
    no_total = PLAY_BY_PLAY_CSV.replace(
        "23,James Harden,36,2TM,PG,70,70,2438,46,52,2,0,0,1.4,-1.5,159,66,61,15,219,3,1310,39,80,,hardeja01\n",
        "",
    )
    with pytest.raises(BasketballReferenceError, match="double-count"):
        season_totals(_read(no_total))


# --- scope ---------------------------------------------------------------


def test_non_nba_export_is_refused():
    college = PER_GAME_CSV.replace(",LAL,", ",DUKE,")
    with pytest.raises(BasketballReferenceError, match="NBA-only"):
        _read(college)


def test_sr_team_spellings_map_to_nba_codes():
    charlotte = PER_GAME_CSV.replace(",TOR,", ",CHO,")
    frame = _read(charlotte).frame.set_index("PLAYER")
    assert frame.at["Chucky Hepburn", "TEAM"] == "CHO"
    assert frame.at["Chucky Hepburn", "NBA_TEAM"] == "CHA"


# --- the leakage invariant ----------------------------------------------


def test_same_season_aggregate_is_refused_as_a_feature():
    table = _read(PER_GAME_CSV, season="2025-26")
    with pytest.raises(SeasonAggregateLeakageError, match="SEASON AGGREGATES"):
        prior_season_features(table, target_season="2025-26")


def test_future_season_aggregate_is_refused_as_a_feature():
    table = _read(PER_GAME_CSV, season="2025-26")
    with pytest.raises(SeasonAggregateLeakageError):
        prior_season_features(table, target_season="2024-25")


def test_attach_refuses_a_table_covering_the_panels_own_season():
    table = _read(PER_GAME_CSV, season="2025-26")
    panel = pd.DataFrame({"PLAYER_NAME": ["Luka Doncic"], "SEASON": ["2025-26"]})
    with pytest.raises(SeasonAggregateLeakageError, match="the panel is IN"):
        attach_prior_season_features(panel, [table])


def test_attach_gives_a_players_own_season_rows_no_features():
    """
    The decisive case: one panel spanning two seasons, one 2024-25 table.
    The 2025-26 rows get the prior-season value; the 2024-25 rows — the very
    games the aggregate was computed from — get NaN.
    """
    table = _read(PER_GAME_CSV, season="2024-25")
    panel = pd.DataFrame({
        "PLAYER_NAME": ["Luka Doncic", "Luka Doncic"],
        "SEASON": ["2024-25", "2025-26"],
    })
    out, report = attach_prior_season_features(panel, [table], columns=("PTS",))
    assert pd.isna(out.loc[0, "SR_PG_PRIOR_PTS"])
    assert out.loc[1, "SR_PG_PRIOR_PTS"] == pytest.approx(33.5)
    assert out.loc[1, "SR_PRIOR_SEASON"] == "2024-25"
    assert report.seasons_without_table == ("2024-25",)


def test_awards_are_excluded_unless_asked_for():
    table = _read(PER_GAME_CSV)
    default = prior_season_features(table, target_season="2025-26")
    assert not [c for c in default.columns if "AWARD" in c]

    opted_in = prior_season_features(
        table, target_season="2025-26", columns=("PTS", "AWARDS"), include_awards=True
    )
    assert opted_in["SR_PG_PRIOR_AWARDS"].iloc[0] == "MVP-4CPOY-8ASNBA1"


# --- name matching -------------------------------------------------------


def test_accented_names_normalise_to_the_nba_spelling():
    assert normalise_player_name("Nikola Jokić") == "nikola jokic"
    assert normalise_player_name("Luka Dončić") == "luka doncic"


def test_unmatched_players_keep_nan_and_are_reported():
    table = _read(PER_GAME_CSV, season="2024-25")
    panel = pd.DataFrame({
        "PLAYER_NAME": ["Luka Doncic", "Some Rookie"],
        "SEASON": ["2025-26", "2025-26"],
    })
    out, report = attach_prior_season_features(panel, [table], columns=("PTS",))
    assert isinstance(report, PriorSeasonJoinReport)
    assert pd.isna(out.loc[1, "SR_PG_PRIOR_PTS"])
    assert "some rookie" in report.unmatched_players
    assert report.match_rate == pytest.approx(0.5)


def test_mixed_table_kinds_are_refused():
    tables = [_read(PER_GAME_CSV, season="2024-25"), _read(PLAY_BY_PLAY_CSV, season="2024-25")]
    panel = pd.DataFrame({"PLAYER_NAME": ["Luka Doncic"], "SEASON": ["2025-26"]})
    with pytest.raises(BasketballReferenceError, match="one table kind at a time"):
        attach_prior_season_features(panel, tables)


# --- attribution ---------------------------------------------------------


def test_attribution_travels_with_every_frame():
    table = _read(PER_GAME_CSV)
    assert "Basketball-Reference" in table.attribution
    assert table.frame.attrs["attribution"] == SR_ATTRIBUTION
    feats = prior_season_features(table, target_season="2025-26")
    assert feats.attrs["attribution"] == SR_ATTRIBUTION
    assert describe_sr_csv(io.StringIO(PER_GAME_CSV), season="2024-25")["attribution"] == SR_ATTRIBUTION


# ---------------------------------------------------------------------------
# The bot wall. Added 2026-11-11, after measuring it.
# ---------------------------------------------------------------------------
#
# basketball-reference.com returns HTTP 403 to a client that says who it is
# and 200 to one whose User-Agent claims to be Chrome. The fetch added on
# 2026-11-10 shipped the Chrome string as its default, which made the default
# behaviour of this project "defeat the host's bot wall". These two tests are
# the record that it does not, and they fail if either half is reverted.


def test_the_default_user_agent_does_not_impersonate_a_browser() -> None:
    """
    THE DEFAULT AGENT IDENTIFIES THIS PROJECT, NOT A BROWSER.

    A 200 from this host obtained with a Chrome string is obtained by telling
    it something untrue. Whether to do that is an operator decision with
    terms-of-service consequences, reachable through
    `$PROPIQ_BBREF_USER_AGENT`; it is not this module's default. If this
    assertion ever fails, someone put the browser string back.
    """
    from src.ingestion.basketball_reference import SR_USER_AGENT

    lowered = SR_USER_AGENT.lower()
    for impersonation in (
        "mozilla", "applewebkit", "khtml", "chrome", "safari",
        "firefox", "gecko", "edg/", "opera", "webkit",
    ):
        assert impersonation not in lowered, (
            f"the default User-Agent claims to be a browser ({impersonation!r} "
            f"in {SR_USER_AGENT!r}). That is bot-wall circumvention as a "
            f"default, which this project does not ship."
        )
    assert "propiq" in lowered, (
        f"the default User-Agent should say who this is: {SR_USER_AGENT!r}"
    )


def test_the_basketball_reference_job_is_off_unless_switched_on() -> None:
    """
    A JOB THAT CANNOT SUCCEED IS NOT SCHEDULED.

    With the honest default agent the fetch is a measured 403, so leaving the
    job on would buy one Discord failure notice per night and no data. It
    returns SKIPPED unless the operator sets the flag -- and because `_flag`
    is what reads the default, the assertion runs the real job body with the
    variable absent rather than reading the constant.
    """
    import scheduler_worker

    monkey = pytest.MonkeyPatch()
    try:
        monkey.delenv(scheduler_worker.ENV_BBREF_INGEST, raising=False)
        out = scheduler_worker.run_basketball_reference()
    finally:
        monkey.undo()

    assert out["status"] == "SKIPPED", out
    assert scheduler_worker.ENV_BBREF_INGEST in out["reason"], out


# ---------------------------------------------------------------------------
# The current-season fetch. Added 2026-11-11.
# ---------------------------------------------------------------------------


def test_the_season_underway_window_opens_and_closes() -> None:
    """
    `is_current_season` over the boundaries that matter.

    The window runs from October 20th of the starting year to June 30th of
    the ending one. June 30th is still live; July 1st the season is over and
    its table is final. The named caveat in the build report -- that the
    opening gate is a date and not the league's calendar -- is accepted here
    deliberately, so this pins the date rather than pretending it is derived.
    """
    from src.ingestion.basketball_reference import is_current_season

    assert not is_current_season("2026-27", date(2026, 10, 19))
    assert is_current_season("2026-27", date(2026, 10, 20))
    assert is_current_season("2026-27", date(2027, 6, 30))
    assert not is_current_season("2026-27", date(2027, 7, 1))
    # A season two years gone is never underway.
    assert not is_current_season("2024-25", date(2026, 11, 11))


def test_the_fetch_wants_the_prior_season_and_the_live_one() -> None:
    """
    Mid-season: two seasons. Off-season: one, because nothing is underway.

    On 2026-07-01 the 2025-26 season has just ended, so the prior-season
    default names it and there is no live season to add -- a fetch list of
    one, not a silently empty one.
    """
    from src.ingestion.basketball_reference import default_fetch_seasons

    assert default_fetch_seasons(date(2026, 11, 11)) == ["2025-26", "2026-27"]
    assert default_fetch_seasons(date(2026, 7, 1)) == ["2025-26"]


def test_a_live_snapshot_is_never_served_as_the_finished_season(tmp_path) -> None:
    """
    THE DEFECT THIS REPLACED, and it was silent and permanent.

    One cache path served both a live season's snapshot and a dead season's
    final table. So: fetch 2026-27 in December, write twelve games to
    `sr_2026-27_advanced.csv`, let the season turn over. `is_current_season`
    goes false, `refresh` resolves to False, the file exists -- and from then
    on a twelve-game December snapshot is `prior_season_features`' idea of
    the completed 2026-27 season, with no request and no warning.

    The two live at different paths now, so a final read cannot see a
    partial one. Asserted on the paths rather than by driving a fetch,
    because the point is that one file is invisible to the other read.
    """
    from src.ingestion.basketball_reference import sr_cache_path

    live = sr_cache_path("2026-27", "advanced", cache_dir=tmp_path, partial=True)
    final = sr_cache_path("2026-27", "advanced", cache_dir=tmp_path)

    assert live != final, (
        "a live snapshot and a finished season's table share a cache path, so "
        "the snapshot is served as the table once the season ends"
    )
    assert live.name.endswith(".partial.csv"), live.name

    live.write_text("a twelve-game snapshot", encoding="utf-8")
    assert not final.exists(), (
        "writing the live snapshot created the finished season's table"
    )


def test_the_live_table_still_cannot_become_a_same_season_feature() -> None:
    """
    FETCHING IS NOT JOINING, and the guard is on the join.

    The build report's central safety claim: a current-season table is data
    for the A/B harness and can never be a feature of its own season. The
    guard that makes that true is `prior_season_features`, not the fetch, so
    this asserts it against a table labelled with the season that is live.
    """
    table = read_sr_season_csv(io.StringIO(PER_GAME_CSV), season="2026-27")

    with pytest.raises(SeasonAggregateLeakageError):
        prior_season_features(table, target_season="2026-27")


# ---------------------------------------------------------------------------
# Driving the fetch itself, which arrived with no tests of any kind.
# ---------------------------------------------------------------------------


def _advanced_html(*names: str) -> str:
    """The smallest page shaped like SR's advanced table."""
    keys = (
        "ranker", "name_display", "age", "team_name_abbr", "pos", "games",
        "games_started", "mp", "per", "ts_pct", "fg3a_per_fga_pct",
        "fta_per_fga_pct", "orb_pct", "drb_pct", "trb_pct", "ast_pct",
        "stl_pct", "blk_pct", "tov_pct", "usg_pct", "ows", "dws", "ws",
        "ws_per_48", "obpm", "dbpm", "bpm", "vorp", "awards",
    )
    rows = []
    for idx, name in enumerate(names, start=1):
        slug = name.lower().replace(" ", "")[:8] + "01"
        cells = []
        for key in keys:
            if key == "ranker":
                value = str(idx)
            elif key == "name_display":
                value = f'<a href="/players/{slug[0]}/{slug}.html">{name}</a>'
            elif key == "team_name_abbr":
                value = "PHI"
            elif key == "pos":
                value = "PG"
            elif key == "awards":
                value = ""
            else:
                value = "4.3"
            cells.append(f'<td data-stat="{key}">{value}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        '<html><table id="advanced"><tbody>'
        + "".join(rows)
        + "</tbody></table></html>"
    )


class _StubResponse:
    def __init__(self, text: str) -> None:
        self.status_code = 200
        self.text = text
        self.headers: dict[str, str] = {}


class _StubSession:
    """Counts requests, so 'served from cache' is a measured claim."""

    def __init__(self, text: str) -> None:
        self._text = text
        self.calls: list[str] = []

    def get(self, url, **_kw):  # noqa: ANN001, ANN003
        self.calls.append(url)
        return _StubResponse(self._text)


def test_a_live_seasons_fetch_writes_the_partial_file(tmp_path) -> None:
    """
    MUTATION THAT SURVIVED THE FIRST ATTEMPT: making `fetch_sr_season_table`
    ignore the partial split and go back to one cache path. The path test
    above asserts the two names differ; nothing asserted that the FETCH
    picks the right one, which is where the promotion bug actually lived.

    2026-27 is underway as of the date this was written, so its fetch is a
    snapshot and belongs in `.partial.csv`.
    """
    from src.ingestion.basketball_reference import (
        fetch_sr_season_table,
        sr_cache_path,
    )

    session = _StubSession(_advanced_html("Tyrese Maxey", "Joel Embiid"))
    table = fetch_sr_season_table(
        "2026-27", "advanced", session=session, cache_dir=tmp_path,
    )

    assert len(table.frame) == 2
    assert table.kind == "advanced"
    partial = sr_cache_path("2026-27", "advanced", cache_dir=tmp_path, partial=True)
    final = sr_cache_path("2026-27", "advanced", cache_dir=tmp_path)
    assert partial.exists(), "a live season's snapshot was not cached as partial"
    assert not final.exists(), (
        "a live season's snapshot was written where the finished season's "
        "table belongs, so it will be served as that table once the season ends"
    )


def test_a_finished_seasons_second_read_makes_no_request(tmp_path) -> None:
    """
    The one-request-per-table-ever claim, measured on the request count.

    A season that is over is final, so the first read fetches and the second
    reads the file. `refresh=None` is left at its default, because that
    resolution is the thing under test.
    """
    from src.ingestion.basketball_reference import fetch_sr_season_table

    session = _StubSession(_advanced_html("Nikola Jokic"))
    first = fetch_sr_season_table(
        "2024-25", "advanced", session=session, cache_dir=tmp_path,
    )
    second = fetch_sr_season_table(
        "2024-25", "advanced", session=session, cache_dir=tmp_path,
    )

    assert len(session.calls) == 1, (
        f"a finished season's table was fetched {len(session.calls)} times; "
        f"it is final, so the cache IS the table"
    )
    assert len(first.frame) == len(second.frame) == 1


def test_the_scheduled_job_asks_for_the_live_season_too(monkeypatch) -> None:
    """
    MUTATION THAT SURVIVED THE FIRST ATTEMPT: reverting `_bbref_seasons` to
    `default_prior_seasons`. Every assertion above is on the ingestion
    module, so the scheduler could go on asking for prior seasons only and
    the live table would never be fetched by anything that runs.
    """
    import scheduler_worker as worker

    monkeypatch.delenv(worker.ENV_BBREF_SEASONS, raising=False)
    seasons = worker._bbref_seasons()

    assert len(seasons) == 2, (
        f"the job asks for {seasons}; the live season is missing, so nothing "
        f"scheduled ever fetches this season's table"
    )


def test_a_live_seasons_second_read_fetches_again(tmp_path) -> None:
    """
    MUTATION THAT SURVIVED: resolving `refresh` to False for a live season.

    The mirror of the finished-season test, and the reason the partial file
    exists at all. A live season's table changes after every game night, so
    its cache is last night's snapshot and not the table -- serving it would
    mean the "current season" rows silently stopped moving on whatever night
    they were first fetched. Two reads, two requests.
    """
    from src.ingestion.basketball_reference import fetch_sr_season_table

    session = _StubSession(_advanced_html("Tyrese Maxey"))
    fetch_sr_season_table("2026-27", "advanced", session=session, cache_dir=tmp_path)
    fetch_sr_season_table("2026-27", "advanced", session=session, cache_dir=tmp_path)

    assert len(session.calls) == 2, (
        f"a live season was fetched {len(session.calls)} time(s); its cached "
        f"snapshot was served as though the season had stopped"
    )
    # And an explicit False still pins the snapshot, which is how an A/B run
    # is reproduced against one night's data.
    fetch_sr_season_table(
        "2026-27", "advanced", session=session, cache_dir=tmp_path, refresh=False,
    )
    assert len(session.calls) == 2, "refresh=False still went to the network"
