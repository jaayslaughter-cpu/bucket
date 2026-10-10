"""
src/db/migrations.py — which `migrations/*.sql` files this database has.

WHY THIS EXISTS. Nothing recorded it. The files are applied by hand
(``psql "$DATABASE_URL" -f migrations/00N_*.sql``), and the authoritative list
of what to run lived in ``docs/deploy_railway.md`` — where it named 002 through
005 and went stale the moment 006 was added. So the only way to learn whether a
database had 007 was to query for the column it adds and infer. Two databases
could disagree and present identically.

THE OTHER HALF OF THE SAME DEFECT WAS A DECLARATION. ``alembic>=1.13.0`` sat in
``requirements.txt`` and ``pyproject.toml`` with no ``alembic.ini``, no
``env.py`` and no module importing it. A dependency that is declared and
unused tells a reader the migrations are managed when they are not, which is
worse than an honest absence. It is dropped; this module is what replaced it.

WHY NOT ACTUALLY WIRE ALEMBIC. It was the other option and it was considered.
These migration files are not mechanical DDL — 006 and 008 are each ~60 lines
of reasoning about why a column is nullable and why a default would be a
fabrication, and that reasoning is the valuable part. Converting them to
``op.add_column`` revisions would either lose it or duplicate it into a second
place that can drift. Alembic earns its keep when migrations are generated from
model diffs; here they are written deliberately, by hand, and what was missing
was never the authoring tool. It was the ledger.

WHAT THIS DOES NOT DO. It does not generate migrations, diff models against the
database, or roll anything back. A down-migration that has never been run is a
claim about reversibility nobody has tested; writing the forward file by hand
and restoring from a snapshot is the honest story for a project at this stage.

THE DRIFT CHECK IS THE PART WORTH READING. A version number cannot catch a file
edited AFTER it was applied: every row looks present and correct while the
database and the repository have silently diverged. So the sha256 of each
file's bytes is recorded alongside its version, and a mismatch is a refusal
rather than a warning — re-running an applied migration is not the fix, and
guessing which half of the file is already in place is worse.

RESEARCH_ONLY project; this is schema bookkeeping and touches no model, odds or
wager.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import Connection, select

from src.db.models import SchemaMigration

logger = logging.getLogger(__name__)

#: Repository root, then the directory the .sql files live in. Resolved from
#: this file rather than from the process's working directory, so the runner
#: finds the same migrations whichever directory it is invoked from.
REPO_ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS_DIR = REPO_ROOT / "migrations"

#: ``002_prop_results.sql`` -> version 2. The number is the ordering key, not
#: the filename: a lexical sort puts ``010`` before ``002`` only by accident of
#: zero-padding and breaks silently at ``100``.
FILENAME_PATTERN = re.compile(r"^(\d+)_([A-Za-z0-9_]+)\.sql$")


class MigrationError(RuntimeError):
    """Raised when the migration set or the ledger is not in a usable state."""


@dataclass(frozen=True)
class MigrationFile:
    version: int
    filename: str
    path: Path
    checksum: str

    @property
    def sql(self) -> str:
        return self.path.read_text(encoding="utf-8")


def checksum_bytes(data: bytes) -> str:
    """sha256 of the file's BYTES, not of its parsed statements.

    Bytes, because the reasoning in these files is most of their content and a
    checksum over statements alone would call an edited explanation identical.
    """
    return hashlib.sha256(data).hexdigest()


def discover_migrations(directory: Path | None = None) -> list[MigrationFile]:
    """
    Every `NNN_name.sql` in `migrations/`, ordered by version.

    A file that does not match the pattern is a REFUSAL, not a skip. A skipped
    migration is invisible: the runner would report "nothing pending" for a
    database missing a file that is sitting in the directory.
    """
    root = directory or MIGRATIONS_DIR
    if not root.is_dir():
        raise MigrationError(f"DATA_NOT_AVAILABLE: no migrations directory at {root}")

    found: dict[int, MigrationFile] = {}
    unmatched: list[str] = []
    for path in sorted(root.glob("*.sql")):
        match = FILENAME_PATTERN.match(path.name)
        if not match:
            unmatched.append(path.name)
            continue
        version = int(match.group(1))
        if version in found:
            raise MigrationError(
                f"two migrations share version {version}: "
                f"{found[version].filename} and {path.name}. The version is the "
                f"ordering key and the ledger's primary key, so a duplicate "
                f"makes the applied set ambiguous."
            )
        found[version] = MigrationFile(
            version=version,
            filename=path.name,
            path=path,
            checksum=checksum_bytes(path.read_bytes()),
        )

    if unmatched:
        raise MigrationError(
            f"{len(unmatched)} file(s) in {root} do not match NNN_name.sql and "
            f"would be silently skipped: {sorted(unmatched)}. Rename them or "
            f"move them out of the migrations directory."
        )
    if not found:
        raise MigrationError(f"DATA_NOT_AVAILABLE: no migrations found in {root}")
    return [found[v] for v in sorted(found)]


def ensure_ledger(conn: Connection) -> None:
    """Create `schema_migrations` if it is not there. Idempotent.

    The runner creates its own ledger rather than depending on a numbered
    migration having been applied by hand, which would reproduce exactly the
    problem the ledger solves.
    """
    SchemaMigration.__table__.create(bind=conn, checkfirst=True)


@dataclass(frozen=True)
class AppliedMigration:
    """One ledger row.

    A plain record rather than the ORM entity, because this module is handed a
    ``Connection`` and not a ``Session``. ``select(SchemaMigration)`` against a
    Connection returns ROWS OF COLUMNS, not instances — so ``row[0]`` is the
    integer version, and reading ``.version`` off it raises. Caught by
    ``tests/test_db_migrations.py`` on the first run; the annotation said
    ``dict[int, SchemaMigration]`` and was wrong in a way mypy would not see,
    since the error was in what SQLAlchemy returns at runtime.
    """

    version: int
    filename: str
    checksum: str


def applied_versions(conn: Connection) -> dict[int, AppliedMigration]:
    """The ledger, keyed by version. Empty on a database nobody has recorded."""
    table = SchemaMigration.__table__
    rows = conn.execute(
        select(table.c.version, table.c.filename, table.c.checksum)
    ).all()
    return {
        int(r.version): AppliedMigration(
            version=int(r.version),
            filename=str(r.filename),
            checksum=str(r.checksum),
        )
        for r in rows
    }


@dataclass(frozen=True)
class Drift:
    version: int
    filename: str
    recorded_checksum: str
    current_checksum: str


def detect_drift(
    files: list[MigrationFile], applied: dict[int, AppliedMigration]
) -> list[Drift]:
    """
    Applied migrations whose file no longer matches what was applied.

    The failure a version number cannot see. Returned rather than raised so a
    caller can report every one of them instead of only the first.
    """
    out: list[Drift] = []
    for f in files:
        row = applied.get(f.version)
        if row is None:
            continue
        if row.checksum != f.checksum:
            out.append(
                Drift(
                    version=f.version,
                    filename=f.filename,
                    recorded_checksum=str(row.checksum),
                    current_checksum=f.checksum,
                )
            )
    return out


def status(conn: Connection, directory: Path | None = None) -> dict[str, object]:
    """What this database has, what it is missing, and what has drifted."""
    files = discover_migrations(directory)
    ensure_ledger(conn)
    applied = applied_versions(conn)
    drift = detect_drift(files, applied)
    pending = [f for f in files if f.version not in applied]

    # A ledger row with no file in the repository: the database is ahead, or a
    # file was deleted. Either way it is reported, because "pending: none" on a
    # database carrying a migration nobody can read is not a clean bill.
    known = {f.version for f in files}
    orphaned = sorted(v for v in applied if v not in known)

    return {
        "migrations_dir": str(directory or MIGRATIONS_DIR),
        "files": files,
        "applied": sorted(applied),
        "pending": pending,
        "drift": drift,
        "orphaned_ledger_rows": orphaned,
    }


#: Statement-level transaction control. The RUNNER owns the transaction -- the
#: CLI wraps the whole run in ``engine.begin()`` so a mid-run failure leaves the
#: database on the last fully applied migration -- so a file that commits on its
#: own behalf silently breaks that guarantee.
_TXN_CONTROL = re.compile(
    r"^\s*(BEGIN|START\s+TRANSACTION|COMMIT|END|ROLLBACK)\s*;\s*$",
    re.IGNORECASE,
)

#: A ``$$``-quoted body. `DO $$ ... $$` blocks in these files contain SQL that
#: must not be scanned for transaction control.
_DOLLAR_QUOTE = re.compile(r"\$\$")


def strip_transaction_control(sql: str) -> tuple[str, list[str]]:
    """
    Remove whole-line BEGIN/COMMIT from a migration body.

    WHY THIS EXISTS, and it is not tidiness. `002_prop_results.sql` and
    `003_capture_vs_ingest_time.sql` each wrap themselves in `BEGIN; ...
    COMMIT;`, which is correct when the file is piped into `psql` and WRONG
    here: the embedded COMMIT ends the transaction this runner opened, so a
    failure in 004 would leave 002 and 003 applied and recorded with no way
    back -- the exact failure the one-transaction contract exists to prevent.
    Verified against a real Postgres: after such a COMMIT the connection's
    transaction_status is IDLE.

    STRIPPED RATHER THAN THE FILES EDITED, deliberately. The ledger records a
    sha256 per file and refuses to continue when an applied file has changed;
    editing 002 would make every database that applied it by hand report DRIFT
    and need an override. So the bytes stay and the runner adapts.

    Only FULL-LINE matches outside a ``$$`` body are touched, so a COMMIT
    inside a function body or a string literal is left alone -- those are the
    cases where guessing would be worse than doing nothing.
    """
    out: list[str] = []
    stripped: list[str] = []
    in_dollar = False
    for line in sql.splitlines():
        if _DOLLAR_QUOTE.search(line):
            # A line may open and close one; an odd count flips the state.
            if len(_DOLLAR_QUOTE.findall(line)) % 2 == 1:
                in_dollar = not in_dollar
            out.append(line)
            continue
        if not in_dollar and _TXN_CONTROL.match(line):
            stripped.append(line.strip())
            continue
        out.append(line)
    return "\n".join(out), stripped


def execute_script(conn: Connection, sql: str) -> None:
    """
    Run a migration body as a script, with NO parameter interpolation.

    THIS IS THE SECOND THING THAT ONLY A REAL POSTGRES SHOWS. The call here was
    ``conn.exec_driver_sql(sql)``, chosen over ``text()`` because these files
    contain ``$$`` bodies and ``:`` inside comments that SQLAlchemy's `text()`
    would read as bind parameters. Correct as far as it went -- and
    ``exec_driver_sql`` still hands the string to psycopg as a FORMAT STRING,
    so a bare ``%`` anywhere in the file, including in a comment, raises
    `incomplete placeholder: '%'` and the migration cannot be applied at all.

    Two files have one: `002_prop_results.sql` ("a win% that includes pushes")
    and `006_player_game_log_fouls.sql` ("99.6%"). Both are comments. The unit
    tests ran against SQLite, whose driver has no %-placeholders, so this was
    green everywhere and broken against the only database this project uses.

    psycopg interpolates only when parameters are PASSED; omitting them
    entirely skips it. SQLAlchemy sends an empty tuple, which still counts as
    passed, so the statement goes to the DBAPI cursor directly. It is the same
    connection SQLAlchemy has already begun a transaction on, so this stays
    inside the caller's transaction -- which the one-transaction contract
    requires.
    """
    from sqlalchemy.exc import DBAPIError

    raw = conn.connection.driver_connection
    cur = raw.cursor()
    try:
        # NOT `with raw.cursor()`: psycopg's cursor is a context manager and
        # sqlite3's is not ("'sqlite3.Cursor' object does not support the
        # context manager protocol"), and the migration tests run against
        # SQLite. try/finally is the spelling both drivers accept.
        try:
            cur.execute(sql)
        except Exception as orig:  # noqa: BLE001 — re-raised below, not swallowed
            # RE-WRAPPED so this function's failure mode is unchanged. Going
            # around SQLAlchemy to reach the DBAPI also goes around its
            # exception wrapping, so a syntax error arrived as a raw
            # `sqlite3.OperationalError` / `psycopg.ProgrammingError` instead
            # of `sqlalchemy.exc.*` -- a silent API change for anything
            # catching the latter, and the migration tests do.
            # `DBAPIError.instance` is the same classmethod SQLAlchemy uses, so
            # the subclass is the one it would have raised itself.
            raise DBAPIError.instance(
                statement=sql,
                params=None,
                orig=orig,
                dbapi_base_err=conn.dialect.loaded_dbapi.Error,
                dialect=conn.dialect,
            ) from orig
    finally:
        cur.close()


def apply_pending(
    conn: Connection,
    directory: Path | None = None,
    *,
    dry_run: bool = False,
    allow_drift: bool = False,
) -> dict[str, object]:
    """
    Apply every migration this database does not have, oldest first.

    EACH FILE AND ITS LEDGER ROW GO IN ONE TRANSACTION. A migration that
    applied and then failed to record would be re-run on the next invocation
    against a schema it had already changed; recording one that failed to apply
    is worse still. The caller supplies the ``Connection``, so a caller that
    opened a transaction gets one transaction for the whole run — which is what
    the CLI does, so a mid-run failure leaves the database on the last fully
    applied migration.

    Drift stops the run before anything is applied, unless ``allow_drift``.
    That flag exists so the state can be inspected, not because continuing is
    usually right.
    """
    report = status(conn, directory)
    drift = report["drift"]
    if drift and not allow_drift:
        lines = [
            f"  {d.filename} (version {d.version}): recorded "
            f"{d.recorded_checksum[:12]}…, file is now {d.current_checksum[:12]}…"
            for d in drift  # type: ignore[union-attr]
        ]
        raise MigrationError(
            "refusing to apply anything: "
            f"{len(drift)} applied migration(s) have been edited since they "  # type: ignore[arg-type]
            "were applied, so the repository and this database have diverged "
            "while every version row still looks correct:\n"
            + "\n".join(lines)
            + "\nRe-running an applied migration is not the fix. Write a new "
            "migration for the change, or restore the file to what was applied."
        )

    pending: list[MigrationFile] = report["pending"]  # type: ignore[assignment]
    if not pending:
        logger.info("Migrations: nothing pending; %d already applied.",
                    len(report["applied"]))  # type: ignore[arg-type]
        return {"applied": [], "dry_run": dry_run, "status": report}

    if dry_run:
        logger.info("Migrations: %d pending (dry run, nothing executed): %s",
                    len(pending), [f.filename for f in pending])
        return {"applied": [], "dry_run": True, "status": report}

    done: list[str] = []
    for f in pending:
        logger.info("Applying %s …", f.filename)
        body, stripped = strip_transaction_control(f.sql)
        if stripped:
            # Said out loud, not swallowed: the file asked for its own
            # transaction and this runner declined, because the CLI's
            # engine.begin() is what makes a mid-run failure leave the database
            # on the last fully applied migration.
            logger.info(
                "%s: ignoring its own transaction control (%s) — this run is "
                "one transaction, owned by the caller.",
                f.filename, ", ".join(stripped),
            )
        execute_script(conn, body)
        conn.execute(
            SchemaMigration.__table__.insert().values(
                version=f.version,
                filename=f.filename,
                checksum=f.checksum,
                applied_at=datetime.now(timezone.utc),
            )
        )
        done.append(f.filename)
        logger.info("Applied %s and recorded version %d.", f.filename, f.version)

    return {"applied": done, "dry_run": False, "status": status(conn, directory)}
