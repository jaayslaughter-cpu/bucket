"""
tests/test_espn_ingest_job.py — the nightly job that fills the panel.

WHAT THIS GUARDS. `run_espn_ingest` is the only ingest the deployed container
can run by itself, so it is the difference between a system that seeds itself
and one that waits for an operator. Three things about it can be wrong while
looking right, and each has a test here:

  * it runs at the wrong time relative to the jobs that READ what it writes;
  * it seeds an empty panel with a three-day window, which leaves every
    rolling feature null while the table looks ingested;
  * it mixes id namespaces, which splits a player's rolling history with no
    error anywhere.

RESEARCH_ONLY. No network: the fetch is stubbed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import scheduler_worker as worker
from src.ingestion.espn_panel import ESPN_TEAM_SOURCE, EspnPanelReport, parse_summary_payload

FIXTURE = Path(__file__).parent / "fixtures" / "espn_panel" / "summary_401909881.json"


def _frame() -> pd.DataFrame:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return pd.DataFrame(parse_summary_payload(payload, "401909881"))


@pytest.fixture
def wired(monkeypatch):
    """The job with its boundaries stubbed; returns what it was asked for."""
    import src.db.repository as repository
    from src.ingestion import espn_panel

    seen: dict = {"days": None, "panel_rows": None, "team_rows": None,
                  "team_source": None}

    def fake_fetch(days, **kwargs):
        seen["days"] = list(days)
        return _frame(), EspnPanelReport(games_seen=1, games_parsed=1, rows_out=27)

    monkeypatch.setattr(espn_panel, "fetch_player_rows_for_dates", fake_fetch)
    monkeypatch.setattr(repository, "ingested_row_counts",
                        lambda: {"player_game_logs": 19, "team_game_stats": 2,
                                 "game_market_lines": 0})
    monkeypatch.setattr(
        repository, "upsert_player_game_logs",
        lambda df: seen.__setitem__("panel_rows", len(df)) or len(df))

    real_agg = worker.__dict__.get("_noop")
    from src.ingestion import scraper_team_games

    original = scraper_team_games.aggregate_team_games

    def spy_agg(frame, **kwargs):
        seen["team_source"] = kwargs.get("team_source")
        return original(frame, **kwargs)

    monkeypatch.setattr(scraper_team_games, "aggregate_team_games", spy_agg)
    monkeypatch.setattr(
        repository, "upsert_team_game_stats",
        lambda df: seen.__setitem__("team_rows", len(df)) or len(df))
    assert real_agg is None
    return seen


# --- the schedule, relative to what reads it ----------------------------


def test_the_ingest_runs_before_settlement_and_before_the_slate():
    """
    THE ORDER IS THE DEPENDENCY. Both of the other jobs read what this one
    writes: settlement grades against the panel and the 09:00 slate builds
    features from it. An ingest scheduled after either of them fills the panel
    for TOMORROW while today's run reads yesterday's.
    """
    from apscheduler.schedulers.background import BackgroundScheduler

    jobs = {j["id"]: j for j in worker.describe(
        worker.build_scheduler(BackgroundScheduler())
    )}
    assert set(jobs) == {"espn_ingest", "settlement", "slate"}

    def hour(job_id: str) -> int:
        trigger = jobs[job_id]["trigger"]
        return int(str(trigger).split("hour='")[1].split("'")[0])

    assert hour("espn_ingest") < hour("settlement") < hour("slate"), (
        f"ingest {hour('espn_ingest')}, settle {hour('settlement')}, "
        f"slate {hour('slate')} — the readers must come after the writer"
    )


def test_the_ingest_survives_a_worker_that_was_down(monkeypatch):
    """Six hours of grace, like settlement: a worker down at 02:00 must still
    fill the panel before the 09:00 slate reads it."""
    from apscheduler.schedulers.background import BackgroundScheduler

    jobs = {j["id"]: j for j in worker.describe(
        worker.build_scheduler(BackgroundScheduler())
    )}
    assert jobs["espn_ingest"]["misfire_grace_time"] >= 6 * 60 * 60
    assert jobs["espn_ingest"]["max_instances"] == 1
    assert jobs["espn_ingest"]["coalesce"] is True


# --- self-seeding -------------------------------------------------------


def test_an_empty_panel_is_seeded_wide_not_with_the_nightly_window(
    wired, monkeypatch,
):
    """
    THREE DAYS INTO AN EMPTY PANEL IS WORSE THAN NOTHING. `PTS_L10` needs ten
    prior games; a three-day seed leaves every rolling feature null while
    `player_game_logs` reads non-empty, so the slate stops refusing and starts
    abstaining on every row instead — the same outcome, now invisible.
    """
    import src.db.repository as repository

    monkeypatch.setattr(repository, "player_id_namespaces", lambda: {})
    out = worker.run_espn_ingest()
    assert out["status"] == "OK"
    assert out["days"] == worker.DEFAULT_ESPN_BACKFILL_DAYS == 30
    assert len(wired["days"]) == 30


def test_a_seeded_panel_reads_only_the_nightly_window(wired, monkeypatch):
    """Re-reading thirty days every night is thirty times the fetches for
    nothing: the upsert is idempotent and the old days do not change."""
    import src.db.repository as repository

    monkeypatch.setattr(repository, "player_id_namespaces", lambda: {"espn": 4000})
    out = worker.run_espn_ingest()
    assert out["days"] == worker.DEFAULT_ESPN_LOOKBACK_DAYS == 3
    assert len(wired["days"]) == 3


def test_the_nightly_window_is_wider_than_one_day(wired, monkeypatch):
    """A game finishing after midnight Pacific lands on the next calendar day,
    and ESPN corrects a box score for a day or two afterwards."""
    import src.db.repository as repository

    monkeypatch.setattr(repository, "player_id_namespaces", lambda: {"espn": 1})
    worker.run_espn_ingest()
    assert len(wired["days"]) > 1


# --- the namespace refusal ----------------------------------------------


def test_a_mixed_panel_is_refused_and_alerted(wired, monkeypatch):
    """
    NO UNATTENDED OVERRIDE. Mixing namespaces splits a player's rolling
    history and nothing downstream can detect it afterwards, so the
    scheduled job refuses and tells someone. The operator who knows the
    windows do not overlap passes --allow-mixed-ids to the CLI by hand.
    """
    import src.db.repository as repository

    alerts: list[tuple[str, str]] = []
    monkeypatch.setattr(repository, "player_id_namespaces",
                        lambda: {"nba": 4821, "espn": 10})
    monkeypatch.setattr(
        worker, "notify_failure",
        lambda stage, detail, **kw: alerts.append((stage, detail)) or {})

    out = worker.run_espn_ingest()
    assert out["status"] == "REFUSED"
    assert wired["panel_rows"] is None, "it wrote anyway"
    assert alerts and alerts[0][0] == "espn-ingest"
    assert "4,821" in alerts[0][1]


def test_the_job_can_be_turned_off(monkeypatch):
    monkeypatch.setenv(worker.ENV_ESPN_INGEST, "false")
    out = worker.run_espn_ingest()
    assert out["status"] == "SKIPPED"


# --- provenance and the read-back ---------------------------------------


def test_team_rows_are_labelled_espn_not_scraper(wired, monkeypatch):
    """The ownership guard reads `source`. An ESPN aggregate labelled
    `scraper_agg` would be overwritten by the next scraper run."""
    import src.db.repository as repository

    monkeypatch.setattr(repository, "player_id_namespaces", lambda: {})
    worker.run_espn_ingest()
    assert wired["team_source"] == ESPN_TEAM_SOURCE == "espn_agg"


def test_an_empty_table_after_writing_is_a_failure(wired, monkeypatch):
    """An upsert returns rows handed to the statement. Only a read on a fresh
    connection says anything persisted."""
    import src.db.repository as repository

    monkeypatch.setattr(repository, "player_id_namespaces", lambda: {})
    monkeypatch.setattr(repository, "ingested_row_counts",
                        lambda: {"player_game_logs": 0, "team_game_stats": 0})
    alerts: list = []
    monkeypatch.setattr(worker, "notify_failure",
                        lambda stage, detail, **kw: alerts.append(detail) or {})
    out = worker.run_espn_ingest()
    assert out["status"] == "FAILED"
    assert alerts


def test_a_raising_ingest_does_not_take_the_worker_down(monkeypatch):
    """Every other job catches and logs. A scheduler that dies on one bad
    night stops settling and stops dispatching too."""
    import src.db.repository as repository

    def boom():
        raise RuntimeError("espn exploded")

    monkeypatch.setattr(repository, "player_id_namespaces", boom)
    monkeypatch.setattr(worker, "notify_failure", lambda *a, **k: {})
    out = worker.run_espn_ingest()
    assert out["status"] == "FAILED"
    assert "espn exploded" in out["error"]


def test_a_window_with_no_finished_game_is_named_not_called_success(
    monkeypatch,
):
    import src.db.repository as repository
    from src.ingestion import espn_panel

    monkeypatch.setattr(repository, "player_id_namespaces", lambda: {"espn": 1})
    monkeypatch.setattr(
        espn_panel, "fetch_player_rows_for_dates",
        lambda days, **kw: (pd.DataFrame(), EspnPanelReport(games_seen=4)))
    out = worker.run_espn_ingest()
    assert out["status"] == "DATA_NOT_AVAILABLE"
    assert out["games_seen"] == 4
