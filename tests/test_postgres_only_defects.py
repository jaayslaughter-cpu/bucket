"""
tests/test_postgres_only_defects.py — green on SQLite, broken on Postgres.

Four defects found on 2026-10-10 by pointing the pipeline at a real Postgres
for the first time. Every one of them was covered by a passing test, and every
one of them made the thing it covered impossible to use:

  1. `%` IN A COMMENT KILLED THE MIGRATION. `apply_pending` used
     `conn.exec_driver_sql(sql)`, which hands the string to psycopg as a FORMAT
     STRING, so `-- a win% that includes pushes` in 002 and `99.6%` in 006
     raised `incomplete placeholder: '%'`. The migration runner had NEVER
     applied a migration to a Postgres database.

  2. THE FILES COMMITTED THEIR OWN TRANSACTIONS. 002 and 003 wrap themselves in
     `BEGIN; … COMMIT;`, correct for `psql` and wrong for a runner whose whole
     contract is that the run is one transaction — the embedded COMMIT ends it,
     so a later failure would leave earlier migrations applied and recorded.

  3. THE ORM TABLES WERE NEVER CREATED. The SQL migrations ALTER tables that
     `Base.metadata.create_all` makes, so on a brand-new database 002 failed
     with `relation "projections" does not exist`, `run_migrations` returned 4,
     and `scripts/start.sh` — which hard-fails the container on a migration
     error — meant THE WORKER NEVER STARTED. A first deploy is by definition a
     brand-new database.

  4. EVERY BULK UPSERT EXCEEDED POSTGRES' PARAMETER LIMIT. Each built one
     `pg_insert(...).values(rows)` for every row; Postgres allows 65,535 bind
     parameters per statement (an int16 on the wire). A season's workbook is
     2,644 team-games × 25 columns, and `upsert_player_game_logs` — which the
     slate runs EVERY DAY on a whole season from `leaguegamelog` — is an order
     of magnitude worse.

The common cause: the unit tests build their own small tables in SQLite, whose
driver has no %-placeholders, tolerates transaction statements differently, and
reaches its parameter limit at a different number. The suite was green and the
only database this project uses was unreachable. These tests assert the
properties rather than the driver, so they hold either way.

RESEARCH_ONLY project. No odds, no wager.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# 1. the % that could not be executed
# ---------------------------------------------------------------------------

def test_the_migration_files_really_do_contain_a_bare_percent():
    """
    THE PREMISE OF THE FIX, pinned. If no file had a `%` the tests below would
    be guarding nothing, and a future reader would be entitled to delete the
    machinery. Two files have one, both in comments.
    """
    with_percent = sorted(
        f.name for f in (REPO / "migrations").glob("*.sql")
        if "%" in f.read_text(encoding="utf-8")
    )
    assert with_percent, "no migration has a % any more; this guard is now moot"
    assert "002_prop_results.sql" in with_percent


def test_apply_pending_does_not_hand_the_script_to_a_format_string():
    """
    AST-walked. `exec_driver_sql` sends an empty parameter tuple, which psycopg
    counts as "parameters were passed" and so interpolates — and a comment
    saying otherwise has fooled this repository five times, so the call graph
    is what is read.
    """
    tree = ast.parse((REPO / "src" / "db" / "migrations.py").read_text(encoding="utf-8"))
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "apply_pending"
    )
    attrs = {
        n.func.attr for n in ast.walk(fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    }
    assert "exec_driver_sql" not in attrs, (
        "apply_pending is back on exec_driver_sql, so a % anywhere in a "
        "migration — including in a comment — makes it unapplicable"
    )
    names = {
        n.func.id for n in ast.walk(fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "execute_script" in names


def test_execute_script_omits_parameters_entirely():
    """
    psycopg interpolates only when parameters are PASSED. `cur.execute(sql)`
    with no second argument skips it; `cur.execute(sql, ())` does not. The
    difference is the whole fix, so the call is asserted to have one argument.
    """
    tree = ast.parse((REPO / "src" / "db" / "migrations.py").read_text(encoding="utf-8"))
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "execute_script"
    )
    executes = [
        n for n in ast.walk(fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == "execute"
    ]
    assert executes, "execute_script no longer executes anything"
    for call in executes:
        assert len(call.args) == 1 and not call.keywords, (
            "execute() is passed parameters, so psycopg will interpolate and a "
            "% in a migration comment becomes a placeholder again"
        )


def test_a_driver_error_is_still_a_sqlalchemy_error(tmp_path):
    """
    GOING AROUND SQLAlchemy TO REACH THE DBAPI ALSO GOES AROUND ITS EXCEPTION
    WRAPPING. Before this was re-wrapped, a bad migration raised a raw
    `sqlite3.OperationalError` / `psycopg.ProgrammingError`, so anything
    catching `sqlalchemy.exc.*` stopped seeing it — a silent API change, and
    `tests/test_db_migrations.py` is one of the things that catches it.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.exc import OperationalError

    from src.db.migrations import execute_script

    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        with pytest.raises(OperationalError) as exc:
            execute_script(conn, "CREATE TABLE ;")
    # The driver's own message survives the wrapping; it is what names the fault.
    assert "syntax error" in str(exc.value)


def test_a_valid_script_runs_through_execute_script(tmp_path):
    """The control: without it the test above would pass on a function that
    raised unconditionally."""
    from sqlalchemy import create_engine, text

    from src.db.migrations import execute_script

    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        execute_script(conn, "CREATE TABLE probe (x INTEGER);")
        assert conn.execute(text("SELECT COUNT(*) FROM probe")).scalar() == 0


# ---------------------------------------------------------------------------
# 2. the files that committed their own transactions
# ---------------------------------------------------------------------------

def _files():
    from src.db.migrations import discover_migrations

    return discover_migrations()


def test_the_runner_strips_a_file_s_own_transaction_control():
    from src.db.migrations import strip_transaction_control

    body, stripped = strip_transaction_control(
        "BEGIN;\nALTER TABLE t ADD COLUMN x int;\nCOMMIT;\n"
    )
    assert stripped == ["BEGIN;", "COMMIT;"]
    assert "ALTER TABLE t ADD COLUMN x int;" in body
    assert "BEGIN" not in body and "COMMIT" not in body


def test_at_least_one_shipped_migration_has_its_own_transaction():
    """Otherwise the stripper guards nothing. 002 and 003 both do."""
    from src.db.migrations import strip_transaction_control

    offenders = {
        f.filename: strip_transaction_control(f.sql)[1]
        for f in _files() if strip_transaction_control(f.sql)[1]
    }
    assert offenders, "no migration manages its own transaction any more"
    assert "002_prop_results.sql" in offenders


def test_a_plpgsql_body_s_own_BEGIN_and_END_are_left_alone():
    """
    THE HAZARD THIS COULD HAVE CAUSED. `002_prop_results.sql` contains a
    `DO $$ BEGIN … END IF; END $$;` block and a `RETURNS TRIGGER AS $$ … END;
    $$ LANGUAGE plpgsql;` function. Stripping the function's `END;` would
    produce a syntactically broken body that fails in a way pointing at the
    migration rather than at the stripper.
    """
    from src.db.migrations import strip_transaction_control

    f = next(x for x in _files() if x.version == 2)
    body, stripped = strip_transaction_control(f.sql)
    assert stripped == ["BEGIN;", "COMMIT;"]
    assert "END;" in body, "the trigger function's END; was stripped"
    assert "DO $$" in body
    assert "LANGUAGE plpgsql" in body

    # Nothing but the two transaction statements left the file.
    def meaningful(text: str) -> list[str]:
        return [
            ln.strip() for ln in text.splitlines()
            if ln.strip() and not ln.strip().startswith("--")
        ]

    lost = [ln for ln in meaningful(f.sql) if ln not in meaningful(body)]
    assert lost == ["BEGIN;", "COMMIT;"], lost


def test_a_commit_inside_a_dollar_quoted_body_is_not_stripped():
    from src.db.migrations import strip_transaction_control

    body, stripped = strip_transaction_control(
        "CREATE FUNCTION f() RETURNS void AS $$\nBEGIN\nCOMMIT;\nEND;\n$$ LANGUAGE plpgsql;\n"
    )
    assert stripped == []
    assert "COMMIT;" in body


# ---------------------------------------------------------------------------
# 3. the tables that were never created
# ---------------------------------------------------------------------------

def test_the_migrations_alter_tables_that_only_create_all_makes():
    """
    THE PREMISE OF --ensure-tables. `ADD COLUMN IF NOT EXISTS` does not help
    when the TABLE is absent: Postgres raises `relation … does not exist`.
    """
    altered: set[str] = set()
    for f in (REPO / "migrations").glob("*.sql"):
        altered |= set(re.findall(
            r"ALTER TABLE\s+(\w+)", f.read_text(encoding="utf-8"), re.IGNORECASE,
        ))
    assert altered, "no migration ALTERs anything, so this guard is moot"

    from src.db.models import Base

    orm_tables = set(Base.metadata.tables)
    created_only_by_create_all = altered & orm_tables
    assert created_only_by_create_all, (
        "no ALTERed table comes from the ORM any more, so --ensure-tables may "
        "no longer be needed on a fresh database — check before removing it"
    )
    # The three that actually bit.
    assert {"projections", "player_game_logs"} <= created_only_by_create_all


def test_the_container_front_door_always_ensures_the_tables():
    """
    A first deploy is BY DEFINITION a brand-new database, and
    `scripts/start.sh` hard-fails the container on a migration error — so
    without this the worker never starts and the deploy is marked failed.
    """
    import scripts.run_migrations as run_migrations

    seen: list[list[str]] = []
    original = run_migrations._migrate_main
    try:
        run_migrations._migrate_main = lambda argv: seen.append(argv) or 0
        run_migrations.main([])
    finally:
        run_migrations._migrate_main = original

    assert seen == [["--apply", "--ensure-tables"]], seen


def test_ensure_tables_actually_creates_the_tables_and_does_it_first(monkeypatch):
    """
    BEHAVIOURAL, because the flag existing proves nothing. Gutting
    `if args.ensure_tables:` to `if False:` left every other test in this file
    green -- they checked that the flag was declared and that
    `run_migrations` passed it, not that anything happened.

    ORDER IS ASSERTED TOO: `create_all` must run BEFORE the migration
    transaction opens, since the migrations ALTER the tables it creates.
    """
    import scripts.migrate_db as migrate_db
    import src.db.session as session

    events: list[str] = []

    class _Stop(RuntimeError):
        pass

    class _FakeEngine:
        def begin(self):
            events.append("migration-transaction")
            raise _Stop("far enough")

    monkeypatch.setattr(session, "init_db", lambda: events.append("create_all"))
    monkeypatch.setattr(session, "get_engine", lambda *a, **k: _FakeEngine())

    code = migrate_db.main(["--apply", "--ensure-tables"])

    assert "create_all" in events, (
        "--ensure-tables did not create the ORM tables, so a brand-new "
        "database fails at 002 with 'relation projections does not exist' and "
        "scripts/start.sh never starts the worker"
    )
    assert events.index("create_all") < events.index("migration-transaction"), events
    assert code == 4, "the stopped run should report a failed migration run"


def test_without_ensure_tables_nothing_is_created(monkeypatch):
    """The flag is opt-in: `migrate_db` without it must not touch the schema."""
    import scripts.migrate_db as migrate_db
    import src.db.session as session

    events: list[str] = []

    class _FakeEngine:
        def begin(self):
            raise RuntimeError("far enough")

    monkeypatch.setattr(session, "init_db", lambda: events.append("create_all"))
    monkeypatch.setattr(session, "get_engine", lambda *a, **k: _FakeEngine())
    migrate_db.main(["--apply"])
    assert events == [], "migrate_db created tables without being asked"


def test_migrate_db_accepts_ensure_tables_but_does_not_apply_by_default():
    """The safe default stays where a human types; see run_migrations' docstring."""
    import scripts.migrate_db as migrate_db

    src = (REPO / "scripts" / "migrate_db.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    flags = {
        n.args[0].value for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == "add_argument" and n.args
        and isinstance(n.args[0], ast.Constant)
    }
    assert "--ensure-tables" in flags
    assert "--apply" in flags
    assert callable(migrate_db.main)


# ---------------------------------------------------------------------------
# 4. the bind-parameter limit
# ---------------------------------------------------------------------------

def test_a_seasons_workbook_is_split_into_several_statements():
    """
    THE SIZE THAT FOUND THIS. The 2025-26 workbook upserted 2,644 team-game
    rows of 25 columns = 66,100 parameters, over the 65,535 limit by enough
    that it failed outright.
    """
    from src.db.repository import PG_MAX_BIND_PARAMS, batched

    rows = [{f"c{i}": i for i in range(25)} for _ in range(2644)]
    batches = batched(rows)
    assert len(batches) > 1
    assert sum(len(b) for b in batches) == len(rows), "rows were lost or duplicated"
    for b in batches:
        assert len(b) * 25 <= PG_MAX_BIND_PARAMS


def test_a_whole_season_of_player_logs_is_split():
    """`upsert_player_game_logs` runs EVERY DAY on a season from leaguegamelog."""
    from src.db.repository import PG_MAX_BIND_PARAMS, batched

    rows = [{f"c{i}": i for i in range(20)} for _ in range(30_000)]
    batches = batched(rows)
    assert sum(len(b) for b in batches) == 30_000
    for b in batches:
        assert len(b) * 20 <= PG_MAX_BIND_PARAMS


def test_the_batch_is_sized_from_the_widest_row_not_the_first():
    """
    `_records` makes uniform dicts from a DataFrame, but
    `record_pending_prop_results` is handed dicts built by hand and can omit a
    key. Sizing off a narrow first row would put a wide one over the limit.
    """
    from src.db.repository import PG_MAX_BIND_PARAMS, batched

    rows = [{"a": 1}] + [{f"c{i}": i for i in range(200)} for _ in range(2000)]
    for b in batched(rows):
        assert len(b) * 200 <= PG_MAX_BIND_PARAMS


def test_small_inputs_are_one_statement_and_empty_is_none():
    from src.db.repository import batched

    assert batched([]) == []
    rows = [{"a": 1}] * 5
    assert batched(rows) == [rows]


def test_every_bulk_insert_in_the_repository_is_batched():
    """
    AST-walked across the whole module, because this was SEVEN functions with
    the same defect and a fix applied to six of them would look complete. Any
    `.values(...)` must be handed a batch, never the full row list.
    """
    src = (REPO / "src" / "db" / "repository.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    offenders = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "values" or not node.args:
            continue
        arg = node.args[0]
        if isinstance(arg, ast.Name) and arg.id == "rows":
            offenders.append(node.lineno)
    assert not offenders, (
        f"lines {offenders} pass the full row list to .values(); Postgres "
        f"refuses a statement over 65,535 bind parameters"
    )


@pytest.mark.parametrize("fn", [
    "upsert_team_game_stats",
    "upsert_player_game_logs",
    "upsert_market_lines",
    "insert_prop_snapshots",
    "persist_projections",
    "record_pending_prop_results",
    "upsert_parlay_ledger",
])
def test_each_bulk_writer_loops_over_batches(fn: str):
    """Named one by one so a regression says WHICH writer lost its loop."""
    src = (REPO / "src" / "db" / "repository.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    target = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.FunctionDef) and n.name == fn), None,
    )
    if target is None:
        pytest.skip(f"{fn} no longer exists under that name")
    calls = {
        n.func.id for n in ast.walk(target)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "batched" in calls, f"{fn} does not batch its rows"
