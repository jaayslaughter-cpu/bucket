"""
The slate refreshes `player_game_logs` before it reads the panel.

IT DID NOT, AND THAT WAS THE SHARPEST FINDING OF THE 2026-10-08 AUDIT. Step
[4] of main.py reads `player_game_logs`; the only writer was
`nba_model_cli ingest-logs --persist`, which nothing scheduled. A deployed
container therefore projected from whatever had last been ingested BY HAND,
every rolling feature aged silently, and nothing reported it — because a stale
panel is indistinguishable from a quiet one.

TWO HALVES, AND THE SECOND IS NOT OPTIONAL. The refresh can fail, be switched
off, or succeed against a season that has not started, and in each case the
run continues on an old panel. `panel_freshness` measures the consequence from
the panel itself rather than trusting the step that was supposed to fix it, so
a refresh that reported success and fetched nothing is still caught.

stats.nba.com is denied at this environment's proxy, so every fetch below is
injected. What is NOT tested here is whether the live endpoint still answers
in the shape `parse_league_game_log` expects; only the endpoint can answer
that, and `tests/test_boxscore_ingest.py` records its header list.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import main


def _logs(dates: list[str], players: int = 2) -> pd.DataFrame:
    """A frame shaped like `parse_league_game_log`'s output."""
    rows = []
    for d in dates:
        for p in range(players):
            rows.append({
                "PLAYER_ID": f"20000{p}", "PLAYER_NAME": f"Player {p}",
                "GAME_ID": f"0022500{abs(hash(d)) % 1000:03d}",
                "GAME_DATE": pd.Timestamp(d), "SEASON": "2025-26",
                "TEAM_ABBREVIATION": "LAL", "OPPONENT_ABBREVIATION": "BOS",
                "IS_HOME": True, "MIN": 32.0, "PTS": 20.0, "REB": 5.0,
                "AST": 4.0, "FG3M": 2.0, "STL": 1.0, "BLK": 0.5, "TOV": 2.0,
                "PF": 3.0,
            })
    return pd.DataFrame(rows)


@pytest.fixture()
def wired(monkeypatch):
    """Inject the fetch and capture what would be written."""
    written: dict[str, pd.DataFrame] = {}
    calls: list[object] = []

    def _fake_load(config=None):
        calls.append(config)
        return _logs(["2026-01-13", "2026-01-14", "2026-01-15", "2026-01-16"])

    def _fake_upsert(frame):
        written["frame"] = frame.copy()
        return len(frame)

    import src.db.repository as repository
    import src.ingestion.boxscores as boxscores

    monkeypatch.setattr(boxscores, "load_player_game_logs", _fake_load)
    monkeypatch.setattr(repository, "upsert_player_game_logs", _fake_upsert)
    return {"written": written, "calls": calls}


# --- the season label ----------------------------------------------------

@pytest.mark.parametrize("date,expected", [
    ("2025-10-21", "2025-26"),
    ("2026-01-15", "2025-26"),
    ("2026-07-04", "2025-26"),   # July still belongs to the season that began
    ("2026-08-01", "2026-27"),   # August is the cut
    ("2024-12-31", "2024-25"),
])
def test_the_season_label_cuts_in_august_not_in_january(date, expected):
    """
    A plain `dt.year` would call January 2026 the 2026-27 season and refresh
    the wrong one — fetching a season with no games and reporting success. The
    rule comes from `features.season.season_start_year` rather than a second
    copy here.
    """
    assert main.season_label_for(date) == expected


def test_an_unreadable_date_is_refused_rather_than_guessed():
    with pytest.raises(ValueError, match="cannot read a season"):
        main.season_label_for("not-a-date")


# --- the refresh ---------------------------------------------------------

def test_the_refresh_fetches_and_upserts(wired):
    out = main.refresh_player_logs("2026-01-15")
    assert out["status"] == "OK"
    assert out["season"] == "2025-26"
    assert out["rows_written"] == out["rows_fetched"] > 0
    assert out["newest_game_date"] == "2026-01-15"
    assert "frame" in wired["written"], "nothing was written to the database"


def test_the_fetch_bypasses_the_cache(wired):
    """
    THE ONE THAT WOULD SILENTLY UNDO THE WHOLE STEP.
    `load_player_game_logs` is cache-first, so the default config returns
    yesterday's parquet and fetches nothing — leaving exactly the staleness
    this step exists to remove while reporting a successful refresh.
    """
    main.refresh_player_logs("2026-01-15")
    config = wired["calls"][0]
    assert config is not None
    assert config.use_cache is False, (
        "the refresh is reading the cache, so it can return stale rows and "
        "call it a refresh"
    )
    assert config.seasons == ("2025-26",)


def test_games_after_the_slate_date_are_held_back(wired):
    """
    A run for a past date must not write games played after it.
    `load_player_panel` bounds the panel at query level, so this is the second
    line — but it means a backfill writes only what was knowable then rather
    than relying on every future reader to filter correctly.
    """
    out = main.refresh_player_logs("2026-01-14")
    assert out["status"] == "OK"
    assert out["rows_after_slate_held_back"] > 0
    frame = wired["written"]["frame"]
    assert pd.to_datetime(frame["GAME_DATE"]).max() <= pd.Timestamp("2026-01-14")
    assert out["newest_game_date"] == "2026-01-14"


def test_a_slate_before_the_seasons_first_game_is_skipped_not_failed(wired):
    out = main.refresh_player_logs("2025-12-01")
    assert out["status"] == "SKIPPED"
    assert "dated after" in out["reason"]
    assert "frame" not in wired["written"]


# --- failure never stops the slate ---------------------------------------

def test_a_denied_endpoint_is_reported_and_the_run_continues(monkeypatch):
    """
    Abstaining on stale data beats not running at all. What it must never do
    is continue QUIETLY, which is why the reason lands in the summary.
    """
    import src.ingestion.boxscores as boxscores

    def _denied(config=None):
        raise boxscores.BoxScoreFetchError("proxy denied nba.com")

    monkeypatch.setattr(boxscores, "load_player_game_logs", _denied)
    out = main.refresh_player_logs("2026-01-15")
    assert out["status"] == "FAILED"
    assert "proxy denied" in out["reason"]


def test_an_unexpected_exception_is_also_caught(monkeypatch):
    """A surprise in the ingest must not take the slate down with it."""
    import src.ingestion.boxscores as boxscores

    monkeypatch.setattr(
        boxscores, "load_player_game_logs",
        lambda config=None: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    out = main.refresh_player_logs("2026-01-15")
    assert out["status"] == "FAILED" and "boom" in out["reason"]


def test_a_failed_database_write_is_never_reported_as_a_refresh(monkeypatch, wired):
    import src.db.repository as repository

    monkeypatch.setattr(
        repository, "upsert_player_game_logs",
        lambda frame: (_ for _ in ()).throw(RuntimeError("no connection")),
    )
    out = main.refresh_player_logs("2026-01-15")
    assert out["status"] == "FAILED"
    assert "upsert failed" in out["reason"]
    assert "rows_written" not in out


def test_an_empty_endpoint_response_is_a_failure_not_a_quiet_success(monkeypatch):
    import src.ingestion.boxscores as boxscores

    monkeypatch.setattr(
        boxscores, "load_player_game_logs", lambda config=None: pd.DataFrame()
    )
    out = main.refresh_player_logs("2026-01-15")
    assert out["status"] == "FAILED" and "no rows" in out["reason"]


# --- the switches --------------------------------------------------------

def test_the_refresh_can_be_switched_off(monkeypatch, wired):
    monkeypatch.setenv(main.ENV_LOG_REFRESH, "0")
    out = main.refresh_player_logs("2026-01-15")
    assert out["status"] == "SKIPPED"
    assert "frame" not in wired["written"]


def test_no_db_skips_the_refresh_rather_than_fetching_for_nothing(wired):
    out = main.refresh_player_logs("2026-01-15", persist=False)
    assert out["status"] == "SKIPPED"
    assert wired["calls"] == [], "it fetched with nowhere to put the result"


def test_it_is_on_by_default(monkeypatch, wired):
    monkeypatch.delenv(main.ENV_LOG_REFRESH, raising=False)
    assert main.refresh_player_logs("2026-01-15")["status"] == "OK"


# --- panel freshness ----------------------------------------------------

def _panel(newest: str, n: int = 5) -> pd.DataFrame:
    dates = pd.date_range(end=pd.Timestamp(newest), periods=n, freq="2D")
    return pd.DataFrame({
        "PLAYER_ID": ["200001"] * n, "GAME_ID": [f"g{i}" for i in range(n)],
        "GAME_DATE": dates, "PTS": np.arange(n, dtype=float),
    })


def test_a_current_panel_reads_ok():
    out = main.panel_freshness(_panel("2026-01-14"), "2026-01-15")
    assert out["status"] == "OK"
    assert out["lag_days"] == 1
    assert out["newest_game_date"] == "2026-01-14"


def test_a_panel_trailing_the_slate_is_reported_stale(caplog):
    with caplog.at_level("WARNING"):
        out = main.panel_freshness(_panel("2025-12-20"), "2026-01-15")
    assert out["status"] == "STALE"
    assert out["lag_days"] == 26
    assert "PANEL IS STALE" in caplog.text
    # The message has to name the consequence, not just the number.
    assert "rolling feature" in caplog.text


def test_the_staleness_limit_is_configurable(monkeypatch):
    panel = _panel("2026-01-10")
    assert main.panel_freshness(panel, "2026-01-15")["status"] == "STALE"
    monkeypatch.setenv(main.ENV_MAX_PANEL_LAG, "30")
    assert main.panel_freshness(panel, "2026-01-15")["status"] == "OK"


def test_an_unusable_limit_falls_back_rather_than_raising(monkeypatch):
    monkeypatch.setenv(main.ENV_MAX_PANEL_LAG, "not-a-number")
    out = main.panel_freshness(_panel("2026-01-14"), "2026-01-15")
    assert out["max_lag_days"] == main.DEFAULT_MAX_PANEL_LAG_DAYS


@pytest.mark.parametrize("panel,expected", [
    (pd.DataFrame(), "EMPTY"),
    (pd.DataFrame({"GAME_DATE": [None, None]}), "UNDATED"),
])
def test_a_panel_with_no_usable_dates_says_which_rather_than_claiming_ok(panel, expected):
    assert main.panel_freshness(panel, "2026-01-15")["status"] == expected


# --- the wiring, and the one ordering mistake that would void it --------

def test_both_steps_are_wired_into_the_pipeline_in_the_right_order():
    """
    THE ORDERING IS THE LOAD-BEARING PART. `attach_forward_slate` adds rows
    dated ON the slate carrying no box score, so freshness measured after it
    always reads as perfect — the check would become decoration while looking
    present. And the refresh has to precede the panel load or it has nothing
    to contribute to this run.

    BY AST, NOT BY STRING OFFSETS. The first version of this test used
    `source.index(...)`, and the comment ABOVE the freshness call names
    `attach_forward_slate` — so the "forward slate comes last" ordering was
    being read off prose about the code rather than the code. The same trap as
    a comment mentioning `engine.begin()` in the migration runner and a
    docstring naming the helper in the PRA fix. Calls have line numbers;
    comments do not.
    """
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(main.main)))
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))

    lines: dict[str, int] = {}
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        name = (
            node.func.id if isinstance(node.func, ast.Name)
            else node.func.attr if isinstance(node.func, ast.Attribute)
            else None
        )
        if name in {
            "refresh_player_logs", "load_player_panel",
            "panel_freshness", "attach_forward_slate",
        }:
            lines.setdefault(name, node.lineno)

    missing = {
        "refresh_player_logs", "load_player_panel",
        "panel_freshness", "attach_forward_slate",
    } - set(lines)
    assert not missing, f"main() does not call: {sorted(missing)}"

    assert lines["refresh_player_logs"] < lines["load_player_panel"], (
        "the refresh runs after the panel is read, so it contributes nothing "
        "to this run"
    )
    assert lines["load_player_panel"] < lines["panel_freshness"], (
        "freshness is measured before the panel exists"
    )
    assert lines["panel_freshness"] < lines["attach_forward_slate"], (
        "panel_freshness runs after attach_forward_slate, which adds rows "
        "dated ON the slate carrying no box score — so it would always read "
        "as perfectly fresh while looking present"
    )


def test_both_steps_land_in_the_audit_row():
    """
    `stage_summary` is written to `pipeline_runs`, which is the difference
    between a failure that was reported and one that was logged and lost.
    """
    import inspect

    source = inspect.getsource(main.main)
    assert 'record_run(run_id, status="success", stage_summary=stage_summary)' in source
    for key in ("player_log_refresh", "panel_freshness"):
        assert f'stage_summary["{key}"]' in source
