"""
Tests for src/db/migrations.py — the ledger that records which
`migrations/*.sql` a database has.

WHAT WAS WRONG. Nothing recorded it. The files are applied by hand and
`docs/deploy_railway.md` carried the list — naming 002 through 005, stale from
the moment 006 landed — so learning whether a database had 007 meant querying
for the column it adds and inferring. Separately, `alembic>=1.13.0` was pinned
in two files with no `alembic.ini`, no `env.py` and no importer.

THESE RUN AGAINST IN-MEMORY SQLITE, and that needs saying because
`src/db/session.py` refuses a SQLite fallback on purpose — "silently falling
back to a local file when Postgres is misconfigured would let a 'successful'
run write nowhere the rest of the stack can read". That rule is about
PRODUCTION CONFIGURATION and is untouched here: these tests build their own
engine directly and never go through `session.py`. The ledger is an ORM model
precisely so SQLAlchemy emits its DDL per dialect and this is possible; the
repository's real `.sql` files are Postgres-specific (`NOT VALID`,
`COMMENT ON COLUMN`) and are never executed here.

So the division is: the LEDGER's behaviour — ordering, checksums, drift,
pending, the transaction — is exercised for real, and the one thing that is
not is whether Postgres accepts the hand-written DDL, which only Postgres can
answer.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, event, text
from sqlalchemy.exc import OperationalError

from src.db.migrations import (
    FILENAME_PATTERN,
    MIGRATIONS_DIR,
    MigrationError,
    applied_versions,
    apply_pending,
    checksum_bytes,
    detect_drift,
    discover_migrations,
    ensure_ledger,
    status,
)
from src.db.models import SchemaMigration


@pytest.fixture()
def engine():
    """
    A private in-memory engine. Never session.py's — see the module docstring.

    THE TWO EVENT HOOKS ARE LOAD-BEARING, and they exist to make one test able
    to prove something rather than assert it. pysqlite does not open a
    transaction for DDL, so `CREATE TABLE` autocommits and survives a
    rollback: `test_a_failing_migration_rolls_back_the_whole_run` passed its
    ledger assertion and failed its "the table is gone" one, which is how this
    was found. SQLAlchemy's documented recipe — disable pysqlite's implicit
    transaction handling, then emit BEGIN explicitly — makes DDL transactional
    here too.

    Postgres, the only production target, is transactional for DDL with no
    help at all. The hooks are scaffolding for the test, not a fix to
    anything, and they are in this fixture rather than in `session.py` for
    exactly that reason.
    """
    eng = create_engine("sqlite://")

    @event.listens_for(eng, "connect")
    def _disable_implicit_begin(dbapi_conn, _record):  # pragma: no cover - hook
        dbapi_conn.isolation_level = None

    @event.listens_for(eng, "begin")
    def _emit_begin(conn):  # pragma: no cover - hook
        conn.exec_driver_sql("BEGIN")

    yield eng
    eng.dispose()


def _write(directory, name, body="SELECT 1;\n"):
    path = directory / name
    path.write_text(body, encoding="utf-8")
    return path


@pytest.fixture()
def migrations(tmp_path):
    d = tmp_path / "migrations"
    d.mkdir()
    _write(d, "002_first.sql", "CREATE TABLE a (x INTEGER);\n")
    _write(d, "003_second.sql", "CREATE TABLE b (y INTEGER);\n")
    _write(d, "004_third.sql", "CREATE TABLE c (z INTEGER);\n")
    return d


# --- discovery and ordering ----------------------------------------------

def test_the_repositorys_own_migrations_are_all_discoverable():
    """
    The real directory, not a fixture. A pattern that failed to match a real
    file would make that migration invisible to the runner while it sits on
    disk, which is the state this whole module exists to end.
    """
    found = discover_migrations()
    names = [f.filename for f in found]
    assert names == sorted(names, key=lambda n: int(n.split("_", 1)[0]))
    assert [f.version for f in found] == sorted(f.version for f in found)
    on_disk = sorted(p.name for p in MIGRATIONS_DIR.glob("*.sql"))
    assert names == on_disk, "a .sql file in migrations/ is not being discovered"


def test_versions_order_numerically_not_lexically(tmp_path):
    """
    `010` sorts after `002` lexically only because of the zero padding, and
    the trick stops working at 100. The version integer is the ordering key.
    """
    d = tmp_path / "m"
    d.mkdir()
    for name in ("002_a.sql", "010_b.sql", "100_c.sql", "9_d.sql"):
        _write(d, name)
    assert [f.version for f in discover_migrations(d)] == [2, 9, 10, 100]


def test_a_file_that_does_not_match_the_pattern_is_refused_not_skipped(migrations):
    """
    A skipped file is invisible: the runner would report "nothing pending" for
    a database missing a migration that is sitting in the directory.
    """
    _write(migrations, "fix_the_thing.sql")
    with pytest.raises(MigrationError, match="do not match NNN_name.sql"):
        discover_migrations(migrations)


def test_two_files_sharing_a_version_are_refused(migrations):
    _write(migrations, "003_second_attempt.sql")
    with pytest.raises(MigrationError, match="share version 3"):
        discover_migrations(migrations)


def test_an_empty_or_absent_directory_is_named(tmp_path):
    with pytest.raises(MigrationError, match="no migrations directory"):
        discover_migrations(tmp_path / "nope")
    (tmp_path / "empty").mkdir()
    with pytest.raises(MigrationError, match="no migrations found"):
        discover_migrations(tmp_path / "empty")


@pytest.mark.parametrize("name,ok", [
    ("002_prop_results.sql", True),
    ("008_player_game_log_starting_position.sql", True),
    ("2_x.sql", True),
    ("002-prop.sql", False),
    ("prop_results.sql", False),
    ("002_prop results.sql", False),
    ("002_prop.SQL", False),
])
def test_the_filename_pattern_accepts_what_this_repository_uses(name, ok):
    assert bool(FILENAME_PATTERN.match(name)) is ok


# --- the ledger -----------------------------------------------------------

def test_the_runner_creates_its_own_ledger(engine):
    """
    There is deliberately no `009_schema_migrations.sql`. A ledger that has to
    be applied by hand before anything can be recorded reproduces the problem
    it was built to solve.
    """
    with engine.begin() as conn:
        ensure_ledger(conn)
        ensure_ledger(conn)  # idempotent
        rows = conn.execute(text("SELECT * FROM schema_migrations")).all()
    assert rows == []


def test_a_fresh_database_reports_everything_pending(engine, migrations):
    with engine.begin() as conn:
        report = status(conn, migrations)
    assert report["applied"] == []
    assert [f.filename for f in report["pending"]] == [
        "002_first.sql", "003_second.sql", "004_third.sql"
    ]
    assert report["drift"] == []


def test_applying_runs_each_file_and_records_it(engine, migrations):
    with engine.begin() as conn:
        result = apply_pending(conn, migrations)
    assert result["applied"] == [
        "002_first.sql", "003_second.sql", "004_third.sql"
    ]
    with engine.begin() as conn:
        # The SQL really ran: the tables each file creates exist.
        for table in ("a", "b", "c"):
            conn.execute(text(f"SELECT * FROM {table}"))
        ledger = conn.execute(
            text("SELECT version, filename, checksum FROM schema_migrations "
                 "ORDER BY version")
        ).all()
    assert [r[0] for r in ledger] == [2, 3, 4]
    assert [r[1] for r in ledger] == [
        "002_first.sql", "003_second.sql", "004_third.sql"
    ]
    expected = checksum_bytes((migrations / "002_first.sql").read_bytes())
    assert ledger[0][2] == expected


def test_a_second_run_applies_nothing(engine, migrations):
    """Idempotent, which is the minimum a ledger has to buy."""
    with engine.begin() as conn:
        apply_pending(conn, migrations)
    with engine.begin() as conn:
        again = apply_pending(conn, migrations)
    assert again["applied"] == []


def test_only_the_new_migration_is_applied_the_second_time(engine, migrations):
    with engine.begin() as conn:
        apply_pending(conn, migrations)
    _write(migrations, "005_fourth.sql", "CREATE TABLE d (w INTEGER);\n")
    with engine.begin() as conn:
        result = apply_pending(conn, migrations)
    assert result["applied"] == ["005_fourth.sql"]


def test_a_dry_run_executes_nothing_and_records_nothing(engine, migrations):
    with engine.begin() as conn:
        result = apply_pending(conn, migrations, dry_run=True)
    assert result["applied"] == []
    with engine.begin() as conn:
        assert conn.execute(
            text("SELECT COUNT(*) FROM schema_migrations")
        ).scalar() == 0
        with pytest.raises(OperationalError, match="no such table"):
            conn.execute(text("SELECT * FROM a"))


# --- drift: the failure a version number cannot see -----------------------

def test_editing_an_applied_file_is_detected(engine, migrations):
    with engine.begin() as conn:
        apply_pending(conn, migrations)
    _write(migrations, "003_second.sql", "CREATE TABLE b (y INTEGER, extra TEXT);\n")
    with engine.begin() as conn:
        report = status(conn, migrations)
    assert report["pending"] == []
    drift = report["drift"]
    assert len(drift) == 1
    assert drift[0].filename == "003_second.sql"
    assert drift[0].recorded_checksum != drift[0].current_checksum


def test_drift_stops_the_run_before_anything_is_applied(engine, migrations):
    """
    Re-running an applied migration is not the fix, and guessing which half of
    an edited file is already in place is worse. So a new PENDING migration
    does not get applied either while drift is outstanding.
    """
    with engine.begin() as conn:
        apply_pending(conn, migrations)
    _write(migrations, "003_second.sql", "CREATE TABLE b (y INTEGER, extra TEXT);\n")
    _write(migrations, "006_fifth.sql", "CREATE TABLE e (v INTEGER);\n")

    with engine.begin() as conn:
        with pytest.raises(MigrationError, match="refusing to apply anything"):
            apply_pending(conn, migrations)
    with engine.begin() as conn:
        with pytest.raises(OperationalError, match="no such table"):
            conn.execute(text("SELECT * FROM e"))


def test_drift_can_be_overridden_deliberately(engine, migrations):
    with engine.begin() as conn:
        apply_pending(conn, migrations)
    _write(migrations, "003_second.sql", "CREATE TABLE b (y INTEGER, extra TEXT);\n")
    _write(migrations, "006_fifth.sql", "CREATE TABLE e (v INTEGER);\n")
    with engine.begin() as conn:
        result = apply_pending(conn, migrations, allow_drift=True)
    assert result["applied"] == ["006_fifth.sql"]
    # The drifted file is NOT re-applied by the override.
    assert "003_second.sql" not in result["applied"]


def test_a_comment_only_edit_still_counts_as_drift(engine, migrations):
    """
    The checksum is over BYTES, not over parsed statements. Most of what is in
    these files is reasoning — 006 and 008 are each ~60 lines of it — and a
    checksum that called an edited explanation identical would be checksumming
    the wrong thing.
    """
    with engine.begin() as conn:
        apply_pending(conn, migrations)
    path = migrations / "002_first.sql"
    path.write_text("-- a new comment\n" + path.read_text(), encoding="utf-8")
    with engine.begin() as conn:
        assert len(status(conn, migrations)["drift"]) == 1


def test_a_ledger_row_with_no_file_is_reported(engine, migrations):
    """
    "Pending: none" on a database carrying a migration this checkout cannot
    show you is not a clean bill of health.
    """
    with engine.begin() as conn:
        apply_pending(conn, migrations)
        conn.execute(
            SchemaMigration.__table__.insert().values(
                version=99, filename="099_from_the_future.sql", checksum="x" * 64,
            )
        )
    with engine.begin() as conn:
        report = status(conn, migrations)
    assert report["orphaned_ledger_rows"] == [99]


def test_drift_is_not_reported_for_a_migration_that_was_never_applied(migrations):
    files = discover_migrations(migrations)
    assert detect_drift(files, {}) == []


# --- the transaction ------------------------------------------------------

def test_a_failing_migration_rolls_back_the_whole_run(engine, migrations):
    """
    A migration that applied and then failed to record would be re-run against
    a schema it had already changed. One transaction for the run means a
    failure leaves the database on the last fully applied migration, with its
    ledger row, and nothing partial.
    """
    _write(migrations, "005_broken.sql", "CREATE TABLE ;\n")  # syntax error
    with pytest.raises(OperationalError):
        with engine.begin() as conn:
            apply_pending(conn, migrations)

    with engine.begin() as conn:
        ensure_ledger(conn)
        applied = conn.execute(
            text("SELECT COUNT(*) FROM schema_migrations")
        ).scalar()
        # Nothing recorded, AND the earlier files' tables are gone. The second
        # half is the one that needs the fixture's transactional-DDL hooks;
        # without them pysqlite autocommits CREATE TABLE and this passed its
        # ledger assertion while the table survived.
        assert applied == 0
    with engine.begin() as conn:
        with pytest.raises(OperationalError, match="no such table"):
            conn.execute(text("SELECT * FROM a"))


# --- the CLI --------------------------------------------------------------

def test_the_cli_reports_without_applying_by_default():
    """
    A migration tool whose no-argument behaviour changes the schema is one typo
    away from being run against the wrong DATABASE_URL.
    """
    import inspect

    import scripts.migrate_db as cli

    source = inspect.getsource(cli.main)
    assert "if not args.apply:" in source
    assert "Nothing was applied. Pass --apply to change the schema." in source


def test_the_cli_applies_the_whole_run_in_one_transaction(engine, migrations, monkeypatch):
    """
    BEHAVIOURAL, because the first version of this was not and proved nothing.
    It asserted `"engine.begin()" in inspect.getsource(cli.main)` — and a
    COMMENT inside that function says "One transaction for the whole run:
    engine.begin() commits on clean exit", so swapping the real call to
    `engine.connect()` left the substring present and the test green. A source
    grep satisfied by prose ABOUT the code is not a test of the code.

    This drives the CLI against a real engine with a broken migration in the
    middle and requires the earlier file's table to be gone afterwards.

    The hazard it catches is a COMMIT PER FILE: 002 and 003 persist, 005
    fails, and the database is left half migrated with no record of it.
    Swapping `engine.begin()` for `engine.connect()` is a different and
    opposite defect — SQLAlchemy 2.0's connect() never commits without an
    explicit call, so the CLI would persist NOTHING while reporting success —
    and `test_the_cli_applies_a_clean_set_and_reports_it` is what catches
    that one. Both were checked by mutation; neither is caught by the other's
    test.
    """
    import scripts.migrate_db as cli
    import src.db.session as session

    monkeypatch.setattr(session, "get_engine", lambda *a, **k: engine)
    _write(migrations, "005_broken.sql", "CREATE TABLE ;\n")

    code = cli.main(["--apply", "--migrations-dir", str(migrations)])
    assert code == 4, "a failing migration must be reported, not swallowed"

    with engine.begin() as conn:
        ensure_ledger(conn)
        assert conn.execute(
            text("SELECT COUNT(*) FROM schema_migrations")
        ).scalar() == 0
    with engine.begin() as conn:
        with pytest.raises(OperationalError, match="no such table"):
            conn.execute(text("SELECT * FROM a"))


def test_the_cli_applies_a_clean_set_and_reports_it(engine, migrations, monkeypatch, capsys):
    """The other half: the happy path actually goes through the CLI."""
    import scripts.migrate_db as cli
    import src.db.session as session

    monkeypatch.setattr(session, "get_engine", lambda *a, **k: engine)
    assert cli.main(["--apply", "--migrations-dir", str(migrations)]) == 0
    out = capsys.readouterr().out
    assert "002_first.sql" in out and "Applied:" in out

    with engine.begin() as conn:
        assert sorted(applied_versions(conn)) == [2, 3, 4]

    # And a second CLI run is a no-op that says so.
    assert cli.main(["--apply", "--migrations-dir", str(migrations)]) == 0
    assert "Nothing pending." in capsys.readouterr().out


def test_the_cli_refuses_a_drifted_set_with_a_distinct_exit_code(
    engine, migrations, monkeypatch
):
    import scripts.migrate_db as cli
    import src.db.session as session

    monkeypatch.setattr(session, "get_engine", lambda *a, **k: engine)
    assert cli.main(["--apply", "--migrations-dir", str(migrations)]) == 0
    _write(migrations, "003_second.sql", "CREATE TABLE b (y INTEGER, extra TEXT);\n")
    assert cli.main(["--apply", "--migrations-dir", str(migrations)]) == 3


def test_a_missing_database_url_is_reported_not_raised():
    """`session.py` has no SQLite fallback by design; the CLI must say so
    rather than traceback."""
    import scripts.migrate_db as cli

    # No DATABASE_URL is configured in this checkout, which is the condition
    # under test; if one ever is, the assertion below still holds for the
    # non-zero path only, so guard it.
    from src.db.session import DatabaseConfigError, get_database_url

    try:
        get_database_url()
    except DatabaseConfigError:
        assert cli.main([]) == 2
    else:  # pragma: no cover — a configured database in CI
        pytest.skip("a DATABASE_URL is configured here")


# --- alembic is gone ------------------------------------------------------

def test_alembic_is_not_declared_anywhere():
    """
    The other half of the defect. A dependency declared and never used tells a
    reader the migrations are managed when they are not. There was no
    alembic.ini, no env.py and no importer.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for name in ("requirements.txt", "pyproject.toml"):
        text_body = (root / name).read_text(encoding="utf-8")
        for line in text_body.splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith('# '):
                continue
            assert "alembic" not in stripped.lower().split("#")[0], (
                f"{name} declares alembic again: {line!r}"
            )
    assert not list(root.glob("alembic*")), "an alembic tree appeared"
