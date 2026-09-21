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
    assert _read(PLAY_BY_PLAY_CSV).kind == "play_by_play"
    assert _read(ADJUSTED_SHOOTING_CSV).kind == "adjusted_shooting"
    assert infer_table_kind(["RK", "PLAYER"]) == "unknown"


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
