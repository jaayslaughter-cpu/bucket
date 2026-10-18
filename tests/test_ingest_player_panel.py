"""
The one-off player-panel ingest, and the two filters that are not optional.

WHY IT EXISTS. `player_game_logs` is what `load_player_panel` reads, and the
only live writer for it pulls from stats.nba.com -- which answers 000 from the
deployed container. When that table is empty the slate produces nothing while
reporting success, which is the failure this repository keeps designing
against. `ingestion.kaggle_nba` is the second path to the same panel and this
is its front door.

THE FILTERS ARE THE POINT. Both exist so that what is SERVED matches what was
TRAINED, which is the only reason a rolling mean means anything:

  * REGULAR SEASON ONLY, as the live path requests. The export states the
    season type in `gameType`, so it is read rather than inferred from a
    game id.
  * NON-APPEARANCES DROPPED, matching `ingest_training_pack` decision 1 --
    a did-not-play recorded as zero points teaches a rolling mean that a
    starter scores nothing once a week. On the real export that is 47,308
    dropped and 214,381 kept, exactly the figures that pack cites.

Get either filter wrong and the panel still loads, still looks complete, and
quietly disagrees with the artifacts about what a player's recent form was.

RESEARCH_ONLY. Historical box scores into a table.
"""

from __future__ import annotations

import csv
import inspect

import pandas as pd
import pytest

import scripts.ingest_player_panel as ing

#: The real export's header, written out rather than taken from
#: `ing.EXPORT_COLUMNS`. Deriving the fixture from the thing under test means
#: a mutation to that tuple breaks the fixture instead of failing the
#: assertion about it -- caught when removing the team-id columns produced 11
#: collection errors rather than one clear failure. A fixture must be able to
#: outlive the code it tests.
EXPORT_HEADER = [
    "firstName", "lastName", "personId", "gameId", "gameDateTimeEst",
    "playerteamCity", "playerteamName", "opponentteamCity", "opponentteamName",
    "gameType", "gameLabel", "gameSubLabel", "seriesGameNumber", "win", "home",
    "numMinutes", "points", "assists", "blocks", "steals",
    "fieldGoalsAttempted", "fieldGoalsMade", "fieldGoalsPercentage",
    "threePointersAttempted", "threePointersMade", "threePointersPercentage",
    "freeThrowsAttempted", "freeThrowsMade", "freeThrowsPercentage",
    "reboundsDefensive", "reboundsOffensive", "reboundsTotal", "foulsPersonal",
    "turnovers", "plusMinusPoints", "playerteamId", "opponentteamId",
    "comment", "startingPosition", "gameDate",
]


def test_the_fixture_header_is_the_real_export_header():
    """
    The 40 columns the Actual export carries, recorded from it on
    2026-10-18. Every column this script reads must exist here, or the
    fixture is testing a file shape the provider does not send.
    """
    for column in ing.EXPORT_COLUMNS:
        assert column in EXPORT_HEADER, (
            f"{column} is read by the script but is not in the real export's "
            f"header — one of the two is wrong"
        )


def _row(first, game_id, date, game_type, minutes, pts, fga, fta, home=1.0):
    base = dict.fromkeys(EXPORT_HEADER, "")
    base.update({
        "firstName": first, "lastName": "Player", "personId": "1",
        "gameId": game_id, "gameDate": date, "gameType": game_type,
        "playerteamId": "1610612747", "playerteamName": "Lakers",
        "playerteamCity": "Los Angeles", "opponentteamId": "1610612738",
        "opponentteamName": "Celtics", "opponentteamCity": "Boston",
        "home": home, "numMinutes": minutes, "points": pts, "assists": 3,
        "blocks": 0, "steals": 1, "fieldGoalsAttempted": fga,
        "fieldGoalsMade": 5, "threePointersMade": 2,
        "freeThrowsAttempted": fta, "freeThrowsMade": 2,
        "reboundsDefensive": 4, "reboundsOffensive": 1, "reboundsTotal": 5,
        "foulsPersonal": 2, "turnovers": 1, "startingPosition": "G",
    })
    return base


@pytest.fixture
def export(tmp_path):
    """A minimal export carrying one row of every kind the filters decide on."""
    rows = [
        _row("Kept", "0022300001", "2024-01-02", "Regular Season", 30.0, 20, 14, 4),
        _row("AlsoKept", "0022300002", "2024-01-04", "Regular Season", 28.0, 18, 12, 2, home=0.0),
        _row("Playoff", "0042300001", "2024-05-02", "Playoffs", 33.0, 25, 18, 6),
        _row("Preseason", "0012300001", "2023-10-02", "Preseason", 20.0, 10, 8, 0),
        _row("TooOld", "0021700001", "2017-01-02", "Regular Season", 25.0, 15, 10, 2),
        _row("DidNotPlay", "0022300003", "2024-01-06", "Regular Season", "", 0, 0, 0),
    ]
    with (tmp_path / ing.PLAYER_FILE).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPORT_HEADER)
        writer.writeheader()
        writer.writerows(rows)
    with (tmp_path / ing.TEAM_HISTORY_FILE).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["teamId", "teamCity", "teamName", "teamAbbrev",
                         "seasonFounded", "seasonActiveTill", "league"])
        # Trailing whitespace on purpose: the real file has it and
        # `load_team_crosswalk` strips it.
        writer.writerow(["1610612747", "Los Angeles", "Lakers", "LAL  ", 1948, 2100, "NBA"])
        writer.writerow(["1610612738", "Boston", "Celtics", "BOS  ", 1946, 2100, "NBA"])
    return tmp_path


# --- the filters ------------------------------------------------------------


def test_only_regular_season_appearances_survive(export):
    panel = ing.normalised_panel(export, ing.DEFAULT_SINCE)
    assert sorted(panel["PLAYER_NAME"].tolist()) == ["AlsoKept Player", "Kept Player"]


def test_a_playoff_row_is_dropped(export):
    """The live path requests Regular Season; a panel holding playoff games
    disagrees with the artifacts about what a rolling window contains."""
    panel = ing.normalised_panel(export, ing.DEFAULT_SINCE)
    assert "Playoff Player" not in set(panel["PLAYER_NAME"])


def test_a_preseason_row_is_dropped(export):
    panel = ing.normalised_panel(export, ing.DEFAULT_SINCE)
    assert "Preseason Player" not in set(panel["PLAYER_NAME"])


def test_a_did_not_play_is_dropped(export):
    """
    `ingest_training_pack` decision 1: a non-appearance records zero points
    and rolling it into an L5 mean teaches the model that a starter scores
    nothing once a week.
    """
    panel = ing.normalised_panel(export, ing.DEFAULT_SINCE)
    assert "DidNotPlay Player" not in set(panel["PLAYER_NAME"])


def test_the_since_window_is_honoured(export):
    panel = ing.normalised_panel(export, ing.DEFAULT_SINCE)
    assert "TooOld Player" not in set(panel["PLAYER_NAME"])
    wider = ing.normalised_panel(export, "2017-01-01")
    assert "TooOld Player" in set(wider["PLAYER_NAME"])


def test_the_default_window_matches_the_shipped_artifacts():
    """Earlier history is unused, not wrong: `load_player_panel` reads a
    400-day window and the rolling features never look past it."""
    assert ing.DEFAULT_SINCE == "2018-01-01"


# --- what the panel must carry ----------------------------------------------


def test_the_panel_carries_the_attempt_columns(export):
    """
    Without FGA/FTA the PTS model abstains on every row while REB and AST
    score normally -- a panel that looks complete and is not. Migration 009
    added the columns for exactly this.
    """
    panel = ing.normalised_panel(export, ing.DEFAULT_SINCE)
    for column in ("FGA", "FTA"):
        assert column in panel.columns
        assert int(panel[column].notna().sum()) == len(panel)


def test_the_home_flag_survives_as_a_boolean(export):
    """The export encodes it as float 1.0/0.0, which `_coerce_is_home`
    returned NA for on every row until 2026-10-18."""
    panel = ing.normalised_panel(export, ing.DEFAULT_SINCE)
    assert int(panel["IS_HOME"].isna().sum()) == 0
    assert set(panel["IS_HOME"].astype(bool)) == {True, False}


def test_the_team_ids_are_read_so_abbreviations_resolve(export):
    """
    `playerteamId`/`opponentteamId` are what the era-aware crosswalk resolves
    ON. Omitting them from the read -- which is easy, they are not obviously
    needed -- leaves every abbreviation null while everything else looks fine.
    """
    assert "playerteamId" in ing.EXPORT_COLUMNS
    assert "opponentteamId" in ing.EXPORT_COLUMNS
    panel = ing.normalised_panel(export, ing.DEFAULT_SINCE)
    assert set(panel["TEAM_ABBREVIATION"]) == {"LAL"}
    assert set(panel["OPPONENT_ABBREVIATION"]) == {"BOS"}


def test_the_panel_is_tagged_with_its_own_provenance(export):
    from src.ingestion.kaggle_nba import SOURCE_NAME

    panel = ing.normalised_panel(export, ing.DEFAULT_SINCE)
    assert set(panel["SOURCE"]) == {SOURCE_NAME}


# --- refusals, and nothing written before them ------------------------------


def test_a_missing_export_is_its_own_exit_code(tmp_path, monkeypatch):
    touched: list[str] = []
    monkeypatch.setattr(ing, "_counts", lambda: touched.append("db") or {})
    assert ing.main(["--kaggle-dir", str(tmp_path / "nope")]) == ing.EXIT_NO_SOURCE
    assert touched == [], "a missing export opened a database connection"


def test_a_missing_panel_is_its_own_exit_code(tmp_path, monkeypatch):
    touched: list[str] = []
    monkeypatch.setattr(ing, "_counts", lambda: touched.append("db") or {})
    assert ing.main(["--panel", str(tmp_path / "nope.parquet")]) == ing.EXIT_NO_SOURCE
    assert touched == []


def test_a_dry_run_needs_no_database(export, monkeypatch):
    import src.db.session as session

    def _refuse(*a, **k):  # pragma: no cover - asserted by not being called
        raise AssertionError("a dry run opened a database connection")

    monkeypatch.setattr(session, "healthcheck", _refuse)
    monkeypatch.setattr(session, "get_engine", _refuse)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("PGHOST", raising=False)
    assert ing.main(["--kaggle-dir", str(export), "--dry-run"]) == ing.EXIT_OK


def test_the_panel_is_built_before_the_database_is_touched():
    """
    A bad path costs nothing and reports itself as what it is. Asserted on the
    source because the ordering is the guarantee: the panel read happens
    before `database_target` is ever imported.
    """
    body = inspect.getsource(ing.main)
    assert body.index("read_parquet") < body.index("database_target")
    assert body.index("normalised_panel") < body.index("database_target")


def test_an_unconfigured_database_is_not_an_ingest_failure(export, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("PGHOST", raising=False)
    assert ing.main(["--kaggle-dir", str(export)]) == ing.EXIT_NO_DATABASE


def test_a_panel_without_attempts_warns_rather_than_refusing(tmp_path, caplog):
    """
    It is still a usable panel -- REB and AST score from it -- so this is a
    warning, not a refusal. But it must be said, because the resulting
    database looks healthy and silently scores two markets out of three.
    """
    import logging

    frame = pd.DataFrame({
        "PLAYER_ID": [1], "PLAYER_NAME": ["A Player"], "GAME_ID": ["0022300001"],
        "GAME_DATE": ["2024-01-02"], "TEAM_ABBREVIATION": ["LAL"],
        "OPPONENT_ABBREVIATION": ["BOS"], "IS_HOME": [True], "MIN": [30.0],
        "PTS": [20], "REB": [5], "AST": [3],
    })
    path = tmp_path / "no_attempts.parquet"
    frame.to_parquet(path)
    with caplog.at_level(logging.WARNING):
        assert ing.main(["--panel", str(path), "--dry-run"]) == ing.EXIT_OK
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "FGA is absent" in text
    assert "PTS model will abstain" in text


# --- a front door, not a second implementation ------------------------------


def test_it_delegates_the_parse_and_the_upsert():
    """
    The defect this repository keeps finding is two readers of one fact. A
    one-off that parsed the export or wrote its own upsert would be a second
    definition of what a player-game row means.
    """
    source = inspect.getsource(ing)
    for expected in ("normalize_player_box_scores", "load_team_crosswalk",
                     "upsert_player_game_logs"):
        assert expected in source, f"{expected} is not delegated to"
    for forbidden in ("pg_insert", "on_conflict_do_update", "CREATE TABLE"):
        assert forbidden not in source, f"{forbidden} reimplements the write path"


def test_the_schema_check_is_shared_with_the_workbook_ingest():
    """One reader for "is the schema behind the code". This ingest needs
    migration 009 in particular: without fga/fta the rows land and PTS still
    cannot score."""
    source = inspect.getsource(ing)
    assert "from scripts.ingest_workbook import _schema_state" in source


@pytest.mark.parametrize(
    "name,value",
    [
        ("EXIT_OK", 0),
        ("EXIT_NO_DATABASE", 2),
        ("EXIT_SCHEMA_BEHIND", 3),
        ("EXIT_INGEST_FAILED", 4),
        ("EXIT_NO_SOURCE", 5),
    ],
)
def test_the_exit_codes_are_what_the_docstring_promises(name, value):
    assert getattr(ing, name) == value
    assert f"    {value}  " in ing.__doc__


def test_the_exit_codes_match_the_workbook_ingest():
    """Two one-off ingests that disagree about what 3 means is a trap for
    whoever writes the runbook."""
    import scripts.ingest_workbook as workbook

    for name in ("EXIT_OK", "EXIT_NO_DATABASE", "EXIT_SCHEMA_BEHIND",
                 "EXIT_INGEST_FAILED"):
        assert getattr(ing, name) == getattr(workbook, name)
