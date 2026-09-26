"""
Kaggle NBA export loader — schema discovery, not schema assumption.

This loader was written without network access to Kaggle, so its column
names were never verified against the real export. These tests pin the
behaviour that makes that safe: it maps what it recognises, refuses what
it cannot identify, and never guesses which column holds which stat.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.kaggle_nba import (
    KaggleNbaError,
    describe_schema,
    load_local_export,
    normalize_player_box_scores,
)


def _frame(**overrides):
    base = {
        "personId": [201939, 201939, 203999],
        "playerName": ["Stephen Curry", "Stephen Curry", "Nikola Jokic"],
        "gameId": ["0022400001", "0022400002", "0022400001"],
        "gameDate": ["2025-01-10", "2025-01-12", "2025-01-10"],
        "playerteamName": ["GSW", "GSW", "DEN"],
        "numMinutes": [34.5, 31.0, 36.2],
        "points": [30, 22, 28],
        "reboundsTotal": [5, 6, 13],
        "assists": [7, 9, 11],
        "threePointersMade": [6, 3, 1],
        "steals": [1, 2, 0],
        "blocks": [0, 1, 1],
        "turnovers": [3, 2, 4],
    }
    base.update(overrides)
    return pd.DataFrame(base)


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------

def test_schema_discovery_reports_what_it_mapped():
    report = describe_schema(_frame())

    assert report.usable
    assert report.mapped["PLAYER_ID"] == "personId"
    assert report.mapped["PLAYER_NAME"] == "playerName"
    assert report.mapped["PTS"] == "points"
    assert report.mapped["REB"] == "reboundsTotal"
    assert report.mapped["FG3M"] == "threePointersMade"
    assert "usable" in report.as_dict()


def test_unrecognised_columns_are_listed_not_silently_dropped():
    report = describe_schema(_frame(someExoticMetric=[1, 2, 3]))
    assert "someExoticMetric" in report.unmapped_source


def test_a_column_is_never_claimed_twice():
    """Two targets must not both map onto the same source column."""
    report = describe_schema(_frame())
    sources = list(report.mapped.values())
    assert len(sources) == len(set(sources))


# --------------------------------------------------------------------------
# Refusal beats guessing
# --------------------------------------------------------------------------

def test_missing_required_column_raises_with_what_it_found():
    """A panel without its target stat cannot produce features."""
    frame = _frame().drop(columns=["points"])
    with pytest.raises(KaggleNbaError, match="PTS") as excinfo:
        normalize_player_box_scores(frame)
    # The error must show the columns that WERE present, so the fix is
    # obvious without re-running anything.
    assert "playerName" in str(excinfo.value)


def test_empty_export_raises():
    with pytest.raises(KaggleNbaError, match="empty"):
        normalize_player_box_scores(pd.DataFrame())


def test_ncaa_data_is_refused():
    """NCAA is explicitly out of scope for this project."""
    with pytest.raises(KaggleNbaError, match="NCAA"):
        normalize_player_box_scores(_frame(ncaaTeamId=[1, 2, 3]))

    frame = _frame()
    frame["league"] = ["NCAA", "NCAA", "NCAA"]
    with pytest.raises(KaggleNbaError, match="not NBA"):
        normalize_player_box_scores(frame)


def test_an_nba_league_column_is_accepted():
    frame = _frame()
    frame["league"] = ["NBA", "NBA", "NBA"]
    assert len(normalize_player_box_scores(frame)) == 3


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------

def test_normalises_onto_the_panel_contract():
    panel = normalize_player_box_scores(_frame())

    for col in ("PLAYER_ID", "PLAYER_NAME", "GAME_ID", "GAME_DATE",
                "SEASON", "MIN", "PTS", "REB", "AST"):
        assert col in panel.columns

    assert len(panel) == 3
    assert panel["PTS"].sum() == 80
    assert str(panel["GAME_DATE"].dtype).startswith("datetime64")


def test_float_ids_lose_their_decimal_tail():
    """201939.0 would never join a panel keyed on '201939'."""
    frame = _frame(personId=[201939.0, 201939.0, 203999.0])
    panel = normalize_player_box_scores(frame)
    assert set(panel["PLAYER_ID"]) == {"201939", "203999"}


def test_season_is_derived_when_absent_and_respects_the_october_boundary():
    frame = _frame(gameDate=["2025-01-10", "2025-11-12", "2025-01-10"])
    panel = normalize_player_box_scores(frame)
    by_date = dict(zip(panel["GAME_DATE"].dt.strftime("%Y-%m-%d"), panel["SEASON"]))
    assert by_date["2025-01-10"] == "2024-25"   # January belongs to last season
    assert by_date["2025-11-12"] == "2025-26"   # November starts the new one


def test_home_flag_never_defaults_an_unknown_value_to_away():
    """Defaulting would label every away game a home game — a wrong feature."""
    frame = _frame(homeAway=["home", "AWAY", "???"])
    panel = normalize_player_box_scores(frame).sort_values("GAME_DATE")
    values = list(panel["IS_HOME"])
    # Count explicitly rather than using `in` or `is`: pandas' boolean dtype
    # yields np.True_ (which is not the True singleton) and pd.NA (whose
    # truthiness raises), so both shortcuts give the wrong answer here.
    assert sum(1 for v in values if not pd.isna(v) and bool(v)) == 1
    assert sum(1 for v in values if not pd.isna(v) and not bool(v)) == 1
    assert sum(1 for v in values if pd.isna(v)) == 1, (
        "an unreadable home/away value became a real flag"
    )


def test_unparseable_dates_are_dropped_not_coerced():
    frame = _frame(gameDate=["2025-01-10", "not-a-date", "2025-01-10"])
    panel = normalize_player_box_scores(frame)
    assert len(panel) == 2


def test_duplicate_player_games_are_dropped():
    frame = pd.concat([_frame(), _frame()], ignore_index=True)
    panel = normalize_player_box_scores(frame)
    assert len(panel) == 3


# --------------------------------------------------------------------------
# Optional dependency and local path
# --------------------------------------------------------------------------

def test_local_csv_round_trips(tmp_path):
    path = tmp_path / "box.csv"
    _frame().to_csv(path, index=False)
    panel = normalize_player_box_scores(load_local_export(path))
    assert len(panel) == 3


def test_missing_local_file_raises():
    with pytest.raises(KaggleNbaError, match="DATA_NOT_AVAILABLE"):
        load_local_export("/nonexistent/export.csv")


def test_kagglehub_absence_names_the_offline_alternative(monkeypatch):
    """kagglehub is optional; the pipeline must not require it."""
    import builtins

    from src.ingestion import kaggle_nba

    real_import = builtins.__import__

    def _blocked(name, *args, **kwargs):
        if name.startswith("kagglehub"):
            raise ImportError("blocked for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    with pytest.raises(KaggleNbaError, match="load_local_export"):
        kaggle_nba.load_from_kagglehub(file_path="x.csv")


def test_kagglehub_requires_an_explicit_file():
    """A Kaggle dataset holds several files; picking one is not our call."""
    from src.ingestion import kaggle_nba

    with pytest.raises(KaggleNbaError, match="file_path is required"):
        kaggle_nba.load_from_kagglehub(file_path="")


def test_output_feeds_the_feature_builder():
    """The real contract: this panel must build features without edits."""
    from src.features.builder import build_feature_matrix

    rows = []
    for day in range(12):
        for player, team in (("Stephen Curry", "GSW"), ("Nikola Jokic", "DEN")):
            rows.append({
                "personId": 201939 if player == "Stephen Curry" else 203999,
                "playerName": player,
                "gameId": f"002240{day:04d}",
                "gameDate": f"2025-01-{day + 1:02d}",
                "playerteamName": team,
                "numMinutes": 32.0, "points": 25, "reboundsTotal": 5,
                "assists": 6, "threePointersMade": 3, "steals": 1,
                "blocks": 0, "turnovers": 2,
            })

    panel = normalize_player_box_scores(pd.DataFrame(rows))
    features = build_feature_matrix(panel)
    assert "PTS_L5" in features.columns
    assert "PRA" in features.columns

# --- cubic review, PR #3 ------------------------------------------------------


def _era_crosswalk() -> pd.DataFrame:
    """The shape load_team_crosswalk produces."""
    return pd.DataFrame({
        "teamId": ["1610612737", "1610612738"],
        "abbreviation": ["ATL", "BOS"],
        "city": ["Atlanta", "Boston"],
        "name": ["Hawks", "Celtics"],
        "season_from": [1949.0, 1946.0],
        "season_to": [2100.0, 2100.0],
    })


def _archive_rows(dates: list[str], *, with_ids: bool = True) -> pd.DataFrame:
    n = len(dates)
    frame = pd.DataFrame({
        "firstName": list("ABCDEF")[:n],
        "lastName": ["One", "Two", "Three", "Four", "Five", "Six"][:n],
        "gameDate": dates,
        "gameId": ["0022300001"] * n,
        "playerteamCity": ["Atlanta"] * n,
        "playerteamName": ["Hawks"] * n,
        "opponentteamCity": ["Boston"] * n,
        "opponentteamName": ["Celtics"] * n,
        "points": list(range(10, 10 + n)),
        "numMinutes": list(range(30, 30 + n)),
    })
    if with_ids:
        frame["playerteamId"] = "1610612737"
        frame["opponentteamId"] = "1610612738"
    return frame


def test_a_dropped_unparseable_date_does_not_break_team_resolution():
    """out starts as pd.DataFrame(index=df.index) and drops unparseable dates
    WITHOUT resetting the index, so out.index is a subset of df's labels.
    Reading team ids straight from df while pairing them with out["SEASON"]
    raised "Length of values (3) does not match length of index (4)" — the
    documented date-drop path crashed outright whenever a crosswalk was given.
    """
    from src.ingestion.kaggle_nba import normalize_player_box_scores

    out = normalize_player_box_scores(
        _archive_rows(["2024-01-01", "NOT A DATE", "2024-01-03", "2024-01-04"]),
        team_crosswalk=_era_crosswalk(),
    )

    assert len(out) == 3, "the bad-date row should be dropped, the rest kept"
    assert out["TEAM_ABBREVIATION"].tolist() == ["ATL"] * 3
    assert out["OPPONENT_ABBREVIATION"].tolist() == ["BOS"] * 3
    assert out["PLAYER_NAME"].tolist() == ["A One", "C Three", "D Four"]


def test_the_name_fallback_also_survives_a_dropped_date():
    """The fallback indexed df with a mask built over out's rows, so it had the
    same defect on the no-team-id route."""
    from src.ingestion.kaggle_nba import normalize_player_box_scores

    out = normalize_player_box_scores(
        _archive_rows(["2024-01-01", "NOT A DATE", "2024-01-03"], with_ids=False),
        team_crosswalk=_era_crosswalk(),
    )

    assert len(out) == 2
    assert out["TEAM_ABBREVIATION"].tolist() == ["ATL", "ATL"]


def test_a_clean_archive_is_unaffected_by_the_alignment_fix():
    from src.ingestion.kaggle_nba import normalize_player_box_scores

    out = normalize_player_box_scores(
        _archive_rows(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
        team_crosswalk=_era_crosswalk(),
    )

    assert len(out) == 4
    assert out["TEAM_ABBREVIATION"].notna().all()
