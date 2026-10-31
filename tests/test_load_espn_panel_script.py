"""
tests/test_load_espn_panel_script.py — the ingest that needs no laptop.

WHAT IS THIS SCRIPT'S OWN. Not the ESPN mapping (`test_espn_panel.py`) and
not the aggregation (`test_scraper_team_games.py`). What belongs here is the
refusal that makes the path safe to run unattended: ONE ID NAMESPACE PER
PANEL. Everything else in this file exists to make sure that refusal cannot
be bypassed by accident.

RESEARCH_ONLY. No network: the fetch is stubbed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import load_espn_panel as script
from src.ingestion.espn_panel import parse_summary_payload

FIXTURE = Path(__file__).parent / "fixtures" / "espn_panel" / "summary_401909881.json"


def _frame() -> pd.DataFrame:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return pd.DataFrame(parse_summary_payload(payload, "401909881"))


@pytest.fixture
def espn(monkeypatch):
    """Stand in for the network fetch with the captured real response."""
    from src.ingestion import espn_panel
    from src.ingestion.espn_panel import EspnPanelReport

    def fake(days, **kwargs):
        report = EspnPanelReport(games_seen=1, games_parsed=1, rows_out=27, dnp_rows=8)
        return _frame(), report

    monkeypatch.setattr(espn_panel, "fetch_player_rows_for_dates", fake)


def _stub_db(monkeypatch, *, namespaces, counts=None, keys_present=None):
    import src.db.repository as repository

    counts = counts or {
        "player_game_logs": 19, "team_game_stats": 2, "game_market_lines": 0,
    }
    monkeypatch.setattr(repository, "player_id_namespaces", lambda: namespaces)
    monkeypatch.setattr(repository, "ingested_row_counts", lambda: counts)
    monkeypatch.setattr(repository, "team_game_sources", lambda: {})
    monkeypatch.setattr(repository, "upsert_player_game_logs", lambda df: len(df))
    monkeypatch.setattr(repository, "upsert_team_game_stats", lambda df: len(df))
    monkeypatch.setattr(
        repository, "ingested_keys_present",
        lambda **kw: {"team_game_stats": (
            len(set(kw.get("team_games", ()))) if keys_present is None
            else keys_present
        )},
    )


# --- the refusal this script exists for ---------------------------------


def test_it_refuses_to_add_espn_ids_to_a_panel_that_holds_nba_ids(
    espn, monkeypatch, capsys,
):
    """
    THE ONE UNRECOVERABLE MISTAKE AVAILABLE HERE. `builder.py:518` groups
    every rolling feature by PLAYER_ID. A player present as both `203507` and
    `espn:3032977` is two players to that groupby: each gets a mean over half
    a season and nothing raises. Afterwards there is no way to tell which
    rows belong together, so the check has to happen BEFORE the write.
    """
    _stub_db(monkeypatch, namespaces={"nba": 4821})
    assert script.main(["--days", "1", "--end", "2026-10-28"]) == 1
    err = capsys.readouterr().err
    assert "REFUSED" in err
    assert "nba: 4,821 row(s)" in err
    assert "SPLIT history" in err


def test_a_panel_already_in_the_espn_namespace_is_added_to(espn, monkeypatch):
    """Its own rows are not foreign to it — a re-run must work."""
    _stub_db(monkeypatch, namespaces={"espn": 350})
    assert script.main(["--days", "1", "--end", "2026-10-28"]) == 0


def test_an_empty_panel_is_not_a_mixed_one(espn, monkeypatch):
    _stub_db(monkeypatch, namespaces={})
    assert script.main(["--days", "1", "--end", "2026-10-28"]) == 0


def test_the_override_exists_and_says_what_it_costs(espn, monkeypatch, capsys):
    """An operator who knows the windows do not overlap can proceed. The flag
    is not hidden, and its help text names the consequence."""
    _stub_db(monkeypatch, namespaces={"nba": 10})
    assert script.main(
        ["--days", "1", "--end", "2026-10-28", "--allow-mixed-ids"]
    ) == 0
    assert "REFUSED" not in capsys.readouterr().err


def test_an_unreadable_namespace_check_does_not_write(espn, monkeypatch, capsys):
    """Unknown is not the same as clean. A check that cannot run must stop the
    write, not wave it through."""
    import src.db.repository as repository

    _stub_db(monkeypatch, namespaces={})

    def boom():
        raise OSError("connection refused")

    monkeypatch.setattr(repository, "player_id_namespaces", boom)
    assert script.main(["--days", "1", "--end", "2026-10-28"]) == 1
    assert "COULD NOT READ" in capsys.readouterr().err


# --- the dry run, and the read-back -------------------------------------


def test_a_dry_run_reaches_for_no_database(espn, monkeypatch, capsys):
    import src.db.repository as repository

    def boom(*a, **kw):
        raise AssertionError("a dry run reached for the database")

    for name in ("player_id_namespaces", "ingested_row_counts",
                 "team_game_sources", "upsert_player_game_logs"):
        monkeypatch.setattr(repository, name, boom)
    assert script.main(
        ["--days", "1", "--end", "2026-10-28", "--team-games", "--dry-run"]
    ) == 0
    out = capsys.readouterr().out
    assert "nothing was written" in out.lower()
    assert "team-games out            2" in out


def test_the_write_is_read_back_not_asserted(espn, monkeypatch, capsys):
    _stub_db(monkeypatch, namespaces={}, counts={
        "player_game_logs": 0, "team_game_stats": 0, "game_market_lines": 0,
    })
    assert script.main(["--days", "1", "--end", "2026-10-28"]) == 1
    assert "player_game_logs is EMPTY" in capsys.readouterr().err


def test_team_game_keys_that_do_not_land_fail_the_run(espn, monkeypatch, capsys):
    _stub_db(monkeypatch, namespaces={}, keys_present=0)
    assert script.main(
        ["--days", "1", "--end", "2026-10-28", "--team-games"]
    ) == 1
    assert "are NOT in the table" in capsys.readouterr().err


def test_the_label_says_attempted_not_upserted(espn, monkeypatch, capsys):
    """Same rule as the other loaders: an upsert's return value is rows handed
    to the statement. Only the read-back claims persistence."""
    _stub_db(monkeypatch, namespaces={})
    assert script.main(["--days", "1", "--end", "2026-10-28"]) == 0
    out = capsys.readouterr().out
    assert "Attempted" in out and "Upserted" not in out


# --- arguments -----------------------------------------------------------


def test_a_bad_end_date_is_named_rather_than_tracebacked(capsys):
    assert script.main(["--end", "28/10/2026"]) == 2
    assert "not YYYY-MM-DD" in capsys.readouterr().err


def test_zero_days_is_refused():
    assert script.main(["--days", "0"]) == 2


def test_an_empty_window_is_a_failure_not_a_silent_success(monkeypatch, capsys):
    """No finished game is a real state, and it is not a successful ingest —
    exiting 0 would tell a scheduler the panel was filled."""
    from src.ingestion import espn_panel
    from src.ingestion.espn_panel import EspnPanelReport

    monkeypatch.setattr(
        espn_panel, "fetch_player_rows_for_dates",
        lambda days, **kw: (pd.DataFrame(), EspnPanelReport(games_seen=3)),
    )
    assert script.main(["--days", "1", "--end", "2026-10-28"]) == 1
    assert "No player rows" in capsys.readouterr().out


def test_it_never_reaches_an_odds_api():
    """Ingestion only. A full slate run would spend PropLine quota."""
    source = Path("scripts/load_espn_panel.py").read_text(encoding="utf-8")
    for forbidden in ("PropLineClient", "fetch_player_props", "run_slate",
                      "normalize_event_props"):
        assert forbidden not in source
