"""
The operator-facing BigDataBall ingest wrapper.

WHAT IS WORTH TESTING. Not the parsing or the upserts — those are
`ingest_market_lines`'s and are tested where they live. What is this script's
own: that it calls that shared path rather than loading the workbook a second
way, that a wrong path is named rather than tracebacked, that --dry-run writes
nothing, and that "wrote nothing" is not reported as success.

RESEARCH_ONLY.
"""

from __future__ import annotations

import pandas as pd

from scripts import ingest_bigdataball as script


def _frames(team_rows: int = 3, market_rows: int = 2) -> tuple:
    team = pd.DataFrame({"nba_game_id": ["0022500001"] * team_rows})
    market = pd.DataFrame({
        "nba_game_id": [f"002250000{i}" for i in range(market_rows)],
        "status": ["VALID"] * market_rows,
    })
    return team, market


def test_it_calls_the_slate_s_own_ingest_and_does_not_reload_the_workbook(
    monkeypatch, tmp_path, capsys,
):
    """
    ONE LOADER. `main.ingest_market_lines` is what step [2] calls when the
    workbook exists, so this calls that — the parsing, the canonical column
    names and both upserts are shared. A second loader here is the
    two-definitions defect with the licensed export as its subject.
    """
    import main as orchestrator

    wb = tmp_path / "wb.xlsx"
    wb.write_bytes(b"not really xlsx; the loader is stubbed")
    seen: dict = {}

    def fake(path, persist=True):
        seen["path"], seen["persist"] = path, persist
        return _frames()

    monkeypatch.setattr(orchestrator, "ingest_market_lines", fake)
    assert script.main(["--xlsx", str(wb)]) == 0
    assert seen["path"] == wb
    assert seen["persist"] is True
    out = capsys.readouterr().out
    assert "team_game_stats rows : 3" in out
    assert "Written to Postgres" in out


def test_a_dry_run_persists_nothing(monkeypatch, tmp_path, capsys):
    import main as orchestrator

    wb = tmp_path / "wb.xlsx"
    wb.write_bytes(b"stub")
    seen: dict = {}

    def fake(path, persist=True):
        seen["persist"] = persist
        return _frames()

    monkeypatch.setattr(orchestrator, "ingest_market_lines", fake)
    assert script.main(["--xlsx", str(wb), "--dry-run"]) == 0
    assert seen["persist"] is False, "--dry-run must not persist"
    assert "nothing was written" in capsys.readouterr().out.lower()


def test_a_missing_workbook_is_named_rather_than_tracebacked(tmp_path, capsys):
    """
    The likeliest failure and the cheapest to report. Checked before anything
    is imported or connected, so it still works where the dependencies are
    not installed.
    """
    missing = tmp_path / "nope.xlsx"
    assert script.main(["--xlsx", str(missing)]) == 2
    err = capsys.readouterr().err
    assert str(missing) in err
    assert "never downloads or scrapes" in err


def test_writing_nothing_is_not_reported_as_success(
    monkeypatch, tmp_path, capsys,
):
    """
    THE OUTCOME THAT LOOKS LIKE SUCCESS AND IS NOT. Two empty frames write no
    rows, so step [2] would still refuse afterwards — and an operator who saw
    exit 0 would believe the blocker was cleared and go looking somewhere else.
    """
    import main as orchestrator

    wb = tmp_path / "wb.xlsx"
    wb.write_bytes(b"stub")
    monkeypatch.setattr(
        orchestrator, "ingest_market_lines",
        lambda path, persist=True: (pd.DataFrame(), pd.DataFrame()),
    )
    assert script.main(["--xlsx", str(wb)]) == 1
    assert "WROTE NOTHING" in capsys.readouterr().out


def test_a_loader_failure_suggests_the_two_things_that_cause_it(
    monkeypatch, tmp_path, capsys,
):
    """A worksheet name — one of the two ways a different export variant
    fails. Reported, not raised at an operator."""
    import main as orchestrator

    wb = tmp_path / "wb.xlsx"
    wb.write_bytes(b"stub")

    def explode(path, persist=True):
        raise ValueError("Worksheet named 'NBA-2025-26-TEAM' not found")

    monkeypatch.setattr(orchestrator, "ingest_market_lines", explode)
    assert script.main(["--xlsx", str(wb)]) == 1
    err = capsys.readouterr().err
    assert "--sheet" in err


# --- the advice must match the fault --------------------------------------
#
# MEASURED ON A REAL RUN, 2026-10-24. The first version printed the workbook
# advice for every failure, so a wrong database password was answered with
# "If this names a worksheet, pass --sheet". The operator goes looking at
# their spreadsheet while the fault is in a connection string.


def test_a_database_failure_is_not_blamed_on_the_workbook(
    monkeypatch, tmp_path, capsys,
):
    """
    THE EXACT PRODUCTION FAILURE. An unsubstituted password placeholder
    reaches Postgres and is rejected like any wrong password; the remedy is
    $DATABASE_URL, and nothing about a worksheet or a column.
    """
    import main as orchestrator

    wb = tmp_path / "wb.xlsx"
    wb.write_bytes(b"stub")

    class OperationalError(Exception):
        """Stands in for sqlalchemy.exc.OperationalError by module."""
    OperationalError.__module__ = "sqlalchemy.exc"

    def explode(path, persist=True):
        raise OperationalError(
            'connection failed: connection to server at "66.33.22.239", port '
            '27751 failed: FATAL:  password authentication failed for user '
            '"postgres"'
        )

    monkeypatch.setattr(orchestrator, "ingest_market_lines", explode)
    assert script.main(["--xlsx", str(wb)]) == 1
    err = capsys.readouterr().err
    assert "DATABASE_URL" in err
    assert "placeholder" in err
    assert "--sheet" not in err, "a connection error is not a worksheet problem"
    assert "GAME-ID" not in err, "nor a column problem"


def test_the_classifier_matches_each_fault_to_its_own_remedy():
    """
    Asserted on `advice_for` directly, so each branch is covered without
    five subprocess-shaped tests. Classified on the exception's MODULE, not
    its message: psycopg's wording is version- and locale-dependent, and this
    repository has been bitten before by a check that read prose.
    """
    class Op(Exception):
        pass
    Op.__module__ = "psycopg"

    class Cfg(Exception):
        pass
    Cfg.__name__ = "DatabaseConfigError"
    Cfg.__module__ = "src.db.session"

    assert script.advice_for(Op("whatever")) == script._DB_ADVICE
    assert script.advice_for(Cfg("no database configured")) == script._DB_ADVICE
    assert script.advice_for(
        ValueError("Worksheet named 'X' not found")
    ) == script._SHEET_ADVICE
    assert script.advice_for(KeyError("GAME-ID")) == script._COLUMN_ADVICE
    # Nothing recognisable falls back to naming both possibilities rather
    # than guessing one.
    assert script.advice_for(RuntimeError("boom")) == script._GENERIC_ADVICE


def test_the_database_advice_names_the_two_ways_the_url_is_wrong(capsys):
    """
    Both were hit for real tonight: an unsubstituted `...` placeholder, and
    the private `postgres.railway.internal` host, which does not resolve from
    outside the platform.
    """
    lines = " ".join(script._DB_ADVICE)
    assert "placeholder" in lines
    assert "postgres.railway.internal" in lines


def test_the_sheet_override_lands_where_the_default_lives(
    monkeypatch, tmp_path,
):
    """
    `sheet_name` is a default on `load_bigdataball_workbook`, not a parameter
    step [2] threads through, so an override has to be applied where the
    default lives rather than passed down.
    """
    import main as orchestrator
    from src.ingestion import bigdataball

    original = bigdataball.DEFAULT_SHEET
    try:
        wb = tmp_path / "wb.xlsx"
        wb.write_bytes(b"stub")
        monkeypatch.setattr(
            orchestrator, "ingest_market_lines",
            lambda path, persist=True: _frames(),
        )
        assert script.main(["--xlsx", str(wb), "--sheet", "MY-TAB"]) == 0
        assert bigdataball.DEFAULT_SHEET == "MY-TAB"
    finally:
        bigdataball.DEFAULT_SHEET = original


def test_it_never_reaches_an_odds_api():
    """
    Step [2] only. A full `main.py` run reaches step [3] once the ingest
    succeeds and spends a slate's worth of PropLine quota; this must not.
    """
    from pathlib import Path

    source = Path("scripts/ingest_bigdataball.py").read_text(encoding="utf-8")
    for forbidden in ("PropLineClient", "fetch_player_props", "requests.get",
                      "normalize_event_props", "run_slate"):
        assert forbidden not in source, f"{forbidden} would spend quota"
