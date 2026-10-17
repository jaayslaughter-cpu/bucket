"""
The one-off workbook ingest has to answer one question: did the rows land?

WHY A SEPARATE FRONT DOOR IS TESTED AT ALL. ``main.py --bigdataball`` performs
the same ingest at step [2] of a full slate, and then keeps going: prop lines,
player-log refresh, feature matrix, scoring. Every one of those can fail on a
database that has just been created, and the process exits non-zero AFTER the
workbook rows are committed. An operator reading that exit status learns
nothing about the ingest. So the guarantees worth asserting here are not "it
calls the loader" -- they are:

  * the exit status is about the ingest and nothing downstream of it;
  * a refusal is a refusal, i.e. nothing was written;
  * the target database is named, without its password, BEFORE the write;
  * the row counts come from the database on both sides of the call, so
    "it ran" and "it landed" stay two different statements.

RESEARCH_ONLY. Loading a licensed historical export into a table.
"""

from __future__ import annotations

import ast
import inspect

import pytest

import scripts.ingest_workbook as ing
from src.db.session import database_target

WORKBOOK = "data/external/bigdataball/2025-2026_NBA_Box_Score_Team-Stats.xlsx"


# --- the target is named, and the credential is not -----------------------


def test_the_userinfo_password_is_masked():
    out = database_target("postgresql+psycopg://u:sw0rdf1sh@db.railway.internal:5432/railway")
    assert "sw0rdf1sh" not in out
    assert "db.railway.internal" in out, "masking removed the part the operator needs"


def test_a_query_string_password_is_masked_too():
    """
    MEASURED, and it is why this function exists rather than a bare
    ``render_as_string(hide_password=True)``: SQLAlchemy's own masking covers
    the ``user:pass@`` form ONLY, and leaves ``?password=`` fully legible.
    libpq accepts both spellings, so the URL an operator pastes decides
    whether the naive version prints their password.
    """
    out = database_target("postgresql://user@host/db?password=hunter2&sslmode=require")
    assert "hunter2" not in out
    assert "sslmode=require" in out


def test_an_unparseable_url_is_not_echoed_back():
    """
    The tempting fallback -- hand the raw string back so the operator can see
    what is wrong with it -- prints the password of every URL that merely has
    a typo in its scheme.
    """
    out = database_target("postgres//u:sw0rdf1sh@host/db")
    assert "sw0rdf1sh" not in out


# --- refusals ---------------------------------------------------------------


def test_a_missing_workbook_has_its_own_exit_code_and_touches_no_database(monkeypatch):
    called: list[str] = []
    monkeypatch.setattr(ing, "_counts", lambda: called.append("counts") or {})
    assert ing.main(["--xlsx", "/no/such/file.xlsx"]) == ing.EXIT_NO_WORKBOOK
    assert called == [], "a missing workbook opened a database connection anyway"


def test_a_schema_behind_the_code_refuses_and_writes_nothing(monkeypatch):
    """
    A refusal that still wrote would be the worst outcome of the three: the
    rows land in a schema the code does not match, and the next run reports a
    clean ledger over a half-populated table.
    """
    import main as pipeline
    import src.db.migrations as migrations
    import src.db.session as session

    wrote: list[object] = []
    monkeypatch.setattr(session, "database_target", lambda *a, **k: "postgresql://x/y")
    monkeypatch.setattr(session, "healthcheck", lambda: (True, "ok"))
    monkeypatch.setattr(ing, "_counts", lambda: {})
    monkeypatch.setattr(
        migrations,
        "status",
        lambda conn, directory=None: {"pending": ["009_x.sql"], "drift": [], "orphaned_ledger_rows": []},
    )
    monkeypatch.setattr(
        session, "get_engine", lambda *a, **k: _FakeEngine()
    )
    monkeypatch.setattr(
        pipeline, "ingest_market_lines", lambda *a, **k: wrote.append(a) or (None, None)
    )

    assert ing.main(["--xlsx", WORKBOOK]) == ing.EXIT_SCHEMA_BEHIND
    assert wrote == [], "it refused on a pending migration and ingested anyway"


def test_drift_alone_is_enough_to_refuse(monkeypatch):
    """Drift is not "pending": an applied migration whose file changed means
    the database and the checkout disagree about what the schema IS."""
    import src.db.migrations as migrations
    import src.db.session as session

    monkeypatch.setattr(session, "get_engine", lambda *a, **k: _FakeEngine())
    monkeypatch.setattr(
        migrations,
        "status",
        lambda conn, directory=None: {"pending": [], "drift": ["003"], "orphaned_ledger_rows": []},
    )
    ready, problems = ing._schema_state()
    assert ready is False
    assert problems


def test_an_unreachable_database_is_not_reported_as_an_ingest_failure(monkeypatch):
    import src.db.session as session

    monkeypatch.setattr(session, "database_target", lambda *a, **k: "postgresql://x/y")
    monkeypatch.setattr(session, "healthcheck", lambda: (False, "connection refused"))
    assert ing.main(["--xlsx", WORKBOOK]) == ing.EXIT_NO_DATABASE


# --- the write path ---------------------------------------------------------


class _FakeEngine:
    def begin(self):
        class _Ctx:
            def __enter__(self_inner):
                return object()

            def __exit__(self_inner, *exc):
                return False

        return _Ctx()


def _wire_a_healthy_database(monkeypatch, counts_sequence):
    import src.db.migrations as migrations
    import src.db.session as session

    monkeypatch.setattr(session, "database_target", lambda *a, **k: "postgresql://host/db")
    monkeypatch.setattr(session, "healthcheck", lambda: (True, "ok"))
    monkeypatch.setattr(session, "get_engine", lambda *a, **k: _FakeEngine())
    monkeypatch.setattr(
        migrations,
        "status",
        lambda conn, directory=None: {"pending": [], "drift": [], "orphaned_ledger_rows": []},
    )
    seen = iter(counts_sequence)
    reads: list[int] = []
    monkeypatch.setattr(ing, "_counts", lambda: reads.append(1) or next(seen))
    return reads


def _counts_dict(team: int, market: int):
    return {
        "team_game_stats": team,
        "team_game_stats:newest": "2026-06-13",
        "game_market_lines": market,
        "game_market_lines:newest": "2026-06-13",
    }


def test_the_counts_are_read_on_both_sides_of_the_ingest(monkeypatch):
    """
    "It ran" and "it landed" are two statements. Reading the tables only
    afterwards cannot tell an ingest of 2,644 rows from a table that already
    held them, which is exactly the question an operator re-running a one-off
    is asking.
    """
    import pandas as pd

    import main as pipeline

    reads = _wire_a_healthy_database(
        monkeypatch, [_counts_dict(0, 0), _counts_dict(2644, 2644)]
    )
    monkeypatch.setattr(
        pipeline,
        "ingest_market_lines",
        lambda *a, **k: (pd.DataFrame({"x": range(2644)}), pd.DataFrame({"x": range(2644)})),
    )
    assert ing.main(["--xlsx", WORKBOOK]) == ing.EXIT_OK
    assert len(reads) == 2, f"the tables were read {len(reads)} time(s), not before and after"


def test_the_target_is_named_before_the_write_not_after(monkeypatch, caplog):
    """
    Printing the target after the rows are committed is a receipt, not a
    chance to hit ^C. The ordering is the whole value.
    """
    import logging

    import pandas as pd

    import main as pipeline

    _wire_a_healthy_database(monkeypatch, [_counts_dict(0, 0), _counts_dict(10, 10)])

    named_before_write: list[bool] = []

    def _ingest(*a, **k):
        named_before_write.append(
            any("INTO" in r.getMessage() for r in caplog.records)
        )
        return pd.DataFrame({"x": [1]}), pd.DataFrame({"x": [1]})

    monkeypatch.setattr(pipeline, "ingest_market_lines", _ingest)
    with caplog.at_level(logging.INFO):
        assert ing.main(["--xlsx", WORKBOOK]) == ing.EXIT_OK
    assert named_before_write == [True], (
        "the database was written before anything said which database it was"
    )


def test_an_ingest_that_raises_is_its_own_exit_code(monkeypatch):
    import main as pipeline

    _wire_a_healthy_database(monkeypatch, [_counts_dict(0, 0), _counts_dict(0, 0)])

    def _boom(*a, **k):
        raise RuntimeError("workbook sheet missing")

    monkeypatch.setattr(pipeline, "ingest_market_lines", _boom)
    assert ing.main(["--xlsx", WORKBOOK]) == ing.EXIT_INGEST_FAILED


def test_an_upsert_that_adds_no_new_rows_is_a_success(monkeypatch):
    """
    Re-running the ingest updates every row in place and adds none. That is
    the idempotent case, not a failure, and a script that exits non-zero on it
    teaches the operator to ignore its exit status.
    """
    import pandas as pd

    import main as pipeline

    _wire_a_healthy_database(
        monkeypatch, [_counts_dict(2644, 2644), _counts_dict(2644, 2644)]
    )
    monkeypatch.setattr(
        pipeline,
        "ingest_market_lines",
        lambda *a, **k: (pd.DataFrame({"x": [1]}), pd.DataFrame({"x": [1]})),
    )
    assert ing.main(["--xlsx", WORKBOOK]) == ing.EXIT_OK


def test_a_dry_run_needs_no_database_and_persists_nothing(monkeypatch):
    import pandas as pd

    import main as pipeline
    import src.db.session as session

    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("PGHOST", raising=False)

    def _refuse(*a, **k):  # pragma: no cover - asserted by not being called
        raise AssertionError("a dry run opened a database connection")

    monkeypatch.setattr(session, "healthcheck", _refuse)
    monkeypatch.setattr(session, "get_engine", _refuse)

    seen: list[bool] = []

    def _ingest(path, persist=True):
        seen.append(persist)
        return (
            pd.DataFrame({"game_date": ["2026-06-13"], "nba_game_id": ["1"]}),
            pd.DataFrame({"nba_game_id": ["1"]}),
        )

    monkeypatch.setattr(pipeline, "ingest_market_lines", _ingest)
    assert ing.main(["--xlsx", WORKBOOK, "--dry-run"]) == ing.EXIT_OK
    assert seen == [False], "a dry run asked the loader to persist"


# --- a front door, not a second implementation ------------------------------


def test_it_calls_the_slate_s_own_loader_and_does_not_reimplement_it():
    """
    The defect this repository keeps finding is two readers of one fact. A
    one-off ingest that parsed the workbook or built its own upsert would be a
    second definition of what the workbook means, free to drift from the one
    the slate uses.
    """
    src = inspect.getsource(ing)
    tree = ast.parse(src)
    attrs = {
        n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
    }
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert "ingest_market_lines" in attrs, "it does not call the slate's loader"
    for forbidden in ("load_bigdataball_workbook", "upsert_team_game_stats", "upsert_market_lines"):
        assert forbidden not in attrs | names, (
            f"{forbidden} is called directly -- this is a second implementation "
            f"of the ingest, not a front door onto the slate's own"
        )


def test_it_does_not_run_the_slate():
    """
    The reason for the file: the slate's later steps must not be able to
    change this exit status. Calling into `main.main` or `run_pipeline` would
    reintroduce exactly the ambiguity it removes.
    """
    src = inspect.getsource(ing)
    tree = ast.parse(src)
    calls = {
        n.func.attr
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    }
    for forbidden in ("run_pipeline", "resolve_market_frames", "ingest_prop_lines"):
        assert forbidden not in calls, f"the one-off ingest runs {forbidden}"


@pytest.mark.parametrize(
    "name,value",
    [
        ("EXIT_OK", 0),
        ("EXIT_NO_DATABASE", 2),
        ("EXIT_SCHEMA_BEHIND", 3),
        ("EXIT_INGEST_FAILED", 4),
        ("EXIT_NO_WORKBOOK", 5),
    ],
)
def test_the_exit_codes_are_what_the_docstring_promises(name, value):
    """The docstring is an operator's contract; a renumbering that leaves it
    behind is the documentation-outliving-its-facts defect this repo guards."""
    assert getattr(ing, name) == value
    assert f"    {value}  " in ing.__doc__
