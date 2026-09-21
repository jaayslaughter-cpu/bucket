"""
Tests for src/ingestion/kaggle_nba.py against the REAL archive schema.

The loader was written without access to the files, from guessed column
spellings. The data pack supplied the actual column list, and these tests
pin the three places the guess was wrong — one of which failed loudly, one
silently, and one not at all.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.ingestion.kaggle_nba import (
    ARCHIVE_TO_NBA_TEAM,
    KaggleNbaError,
    assert_abbreviations,
    describe_schema,
    load_team_crosswalk,
    normalize_player_box_scores,
)

# The genuine column list of PlayerStatistics.csv.
ARCHIVE_COLUMNS = [
    "firstName", "lastName", "personId", "gameId", "gameDateTimeEst",
    "playerteamCity", "playerteamName", "opponentteamCity", "opponentteamName",
    "gameType", "gameLabel", "gameSubLabel", "seriesGameNumber", "win", "home",
    "numMinutes", "points", "assists", "blocks", "steals",
    "fieldGoalsAttempted", "fieldGoalsMade", "fieldGoalsPercentage",
    "threePointersAttempted", "threePointersMade", "threePointersPercentage",
    "freeThrowsAttempted", "freeThrowsMade", "freeThrowsPercentage",
    "reboundsDefensive", "reboundsOffensive", "reboundsTotal",
    "foulsPersonal", "turnovers", "plusMinusPoints",
    "playerteamId", "opponentteamId", "comment", "startingPosition", "gameDate",
]

# TeamHistories.csv in its real shape: trailing whitespace on the codes, a
# multi-era franchise, SAN for San Antonio, and an All-Star roster.
TEAM_HISTORIES = pd.DataFrame([
    {"teamId": 1610612747, "teamCity": "Los Angeles", "teamName": "Lakers",
     "teamAbbrev": "LAL  ", "seasonFounded": 1948, "seasonActiveTill": 2100, "league": "NBA"},
    {"teamId": 1610612759, "teamCity": "San Antonio", "teamName": "Spurs",
     "teamAbbrev": "SAN  ", "seasonFounded": 1976, "seasonActiveTill": 2100, "league": "NBA"},
    {"teamId": 1610612737, "teamCity": "Tri-Cities", "teamName": "Blackhawks",
     "teamAbbrev": "TRI  ", "seasonFounded": 1948, "seasonActiveTill": 1950, "league": "NBA"},
    {"teamId": 1610612737, "teamCity": "Atlanta", "teamName": "Hawks",
     "teamAbbrev": "ATL  ", "seasonFounded": 1968, "seasonActiveTill": 2100, "league": "NBA"},
    {"teamId": 9039, "teamCity": "All-Star", "teamName": "Team LeBron",
     "teamAbbrev": "LBN  ", "seasonFounded": 2018, "seasonActiveTill": 2100, "league": "NBA"},
])


def _crosswalk(tmp_path):
    path = tmp_path / "TeamHistories.csv"
    TEAM_HISTORIES.to_csv(path, index=False)
    return load_team_crosswalk(path)


def _archive_rows() -> pd.DataFrame:
    return pd.DataFrame({
        "firstName": ["LeBron", "Victor"], "lastName": ["James", "Wembanyama"],
        "personId": [2544, 1641705], "gameId": [42500405, 42500405],
        "gameDate": ["2026-01-15", "2026-01-15"],
        "playerteamId": [1610612747, 1610612759],
        "opponentteamId": [1610612759, 1610612747],
        "playerteamCity": ["Los Angeles", "San Antonio"],
        "playerteamName": ["Lakers", "Spurs"],
        "opponentteamName": ["Spurs", "Lakers"],
        "home": [1, 0], "numMinutes": [35.0, 34.0], "points": [28, 30],
        "assists": [8, 4], "reboundsTotal": [7, 12],
        "reboundsOffensive": [1, 2], "reboundsDefensive": [6, 10],
        "steals": [1, 1], "blocks": [1, 3], "turnovers": [3, 2],
        "fieldGoalsMade": [10, 11], "fieldGoalsAttempted": [20, 19],
        "threePointersMade": [2, 1],
        "freeThrowsMade": [6, 7], "freeThrowsAttempted": [7, 8],
        "gameType": ["Regular Season", "Preseason"],
        "comment": [None, "DNP - Rest"],
    })


# --- 1. the failure that was loud ----------------------------------------


def test_player_name_is_composed_from_first_and_last():
    """
    The archive has firstName and lastName but no full name, so the loader
    reported PLAYER_NAME missing and refused the only complete player
    history available.
    """
    report = describe_schema(pd.DataFrame({c: [] for c in ARCHIVE_COLUMNS}))
    assert report.usable
    assert report.composed["PLAYER_NAME"] == ("firstName", "lastName")
    assert not report.missing_required


def test_composed_names_are_joined_and_cleaned(tmp_path):
    rows = _archive_rows()
    rows.loc[0, "firstName"] = "  LeBron  "
    panel = normalize_player_box_scores(rows, team_crosswalk=_crosswalk(tmp_path))
    assert set(panel["PLAYER_NAME"]) == {"LeBron James", "Victor Wembanyama"}


# --- 2. the failure that was silent --------------------------------------


def test_nicknames_in_the_abbreviation_column_are_refused():
    """
    playerteamName holds "Lakers", matches the alias table, and maps
    cleanly into a column every downstream join keys on. Market lines, Elo
    and team pace would then match nothing while looking correct.
    """
    with pytest.raises(KaggleNbaError, match="names rather than NBA abbreviations"):
        normalize_player_box_scores(_archive_rows())

    with pytest.raises(KaggleNbaError, match="rather than NBA abbreviations"):
        assert_abbreviations(pd.Series(["Lakers", "Spurs"]), column="TEAM_ABBREVIATION")

    assert_abbreviations(pd.Series(["LAL", "SAS", None]), column="TEAM_ABBREVIATION")


def test_team_ids_resolve_to_abbreviations_that_join(tmp_path):
    panel = normalize_player_box_scores(_archive_rows(), team_crosswalk=_crosswalk(tmp_path))
    assert list(panel["TEAM_ABBREVIATION"]) == ["LAL", "SAS"]
    assert list(panel["OPPONENT_ABBREVIATION"]) == ["SAS", "LAL"]

    from src.ingestion.basketball_reference import NBA_TEAM_ABBREVIATIONS

    assert set(panel["TEAM_ABBREVIATION"]) <= NBA_TEAM_ABBREVIATIONS


def test_the_archives_san_is_mapped_to_sas(tmp_path):
    """The only real franchise code that differs from every other source."""
    assert ARCHIVE_TO_NBA_TEAM == {"SAN": "SAS"}
    crosswalk = _crosswalk(tmp_path)
    assert "SAS" in set(crosswalk["abbreviation"])
    assert "SAN" not in set(crosswalk["abbreviation"])


def test_trailing_whitespace_on_codes_is_stripped(tmp_path):
    """The file writes "ATL  ", which joins to nothing."""
    crosswalk = _crosswalk(tmp_path)
    assert all(a == a.strip() for a in crosswalk["abbreviation"])


def test_all_star_rosters_are_not_franchises(tmp_path):
    crosswalk = _crosswalk(tmp_path)
    assert "LBN" not in set(crosswalk["abbreviation"])
    assert crosswalk["teamId"].str.startswith("1610612").all()


def test_the_crosswalk_is_era_aware(tmp_path):
    """
    Team 1610612737 was TRI until 1950 and ATL from 1968. A flat map would
    stamp the modern code onto every historical row.
    """
    rows = _archive_rows().head(1).copy()
    rows["playerteamId"] = 1610612737
    rows["opponentteamId"] = 1610612747

    modern = normalize_player_box_scores(rows, team_crosswalk=_crosswalk(tmp_path))
    assert modern["TEAM_ABBREVIATION"].iloc[0] == "ATL"

    historical = rows.copy()
    historical["gameDate"] = "1949-01-15"
    old = normalize_player_box_scores(historical, team_crosswalk=_crosswalk(tmp_path))
    assert old["TEAM_ABBREVIATION"].iloc[0] == "TRI"


# --- 3. what was not carried at all --------------------------------------


def test_preseason_is_flagged_rather_than_pooled(tmp_path):
    """
    Preseason minutes and rotations do not describe the same competition,
    so they must not enter a rolling average unmarked.
    """
    panel = normalize_player_box_scores(_archive_rows(), team_crosswalk=_crosswalk(tmp_path))
    assert list(panel["GAME_TYPE"]) == ["Regular Season", "Preseason"]
    assert list(panel["IS_REGULAR_SEASON"]) == [True, False]


def test_the_dnp_comment_survives(tmp_path):
    """A did-not-play is what voids a prop leg; the reason has to reach the log."""
    panel = normalize_player_box_scores(_archive_rows(), team_crosswalk=_crosswalk(tmp_path))
    resting = panel[panel["PLAYER_NAME"] == "Victor Wembanyama"].iloc[0]
    assert resting["DNP_COMMENT"] == "DNP - Rest"


def test_season_is_derived_from_the_game_date(tmp_path):
    """The archive has no season column; October starts the next season."""
    rows = _archive_rows()
    rows["gameDate"] = ["2025-11-02", "2026-03-04"]
    panel = normalize_player_box_scores(rows, team_crosswalk=_crosswalk(tmp_path))
    assert set(panel["SEASON"]) == {"2025-26"}


def test_a_missing_crosswalk_file_is_refused_by_name(tmp_path):
    bad = tmp_path / "wrong.csv"
    pd.DataFrame({"teamId": [1]}).to_csv(bad, index=False)
    with pytest.raises(KaggleNbaError, match="missing"):
        load_team_crosswalk(bad)
