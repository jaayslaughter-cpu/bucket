"""
scripts/migrate_db.py — apply `migrations/*.sql` and record what was applied.

WHAT THIS REPLACED. The migrations were applied by hand and nothing recorded
which ones a database had. `docs/deploy_railway.md` carried the authoritative
list, named 002 through 005, and went stale the moment 006 was added — so
learning whether a database had 007 meant querying for the column it adds and
inferring. Two databases could disagree and present identically.

`--status` IS THE DEFAULT, and that is deliberate. A migration tool whose
no-argument behaviour changes the schema is one typo away from being run
against the wrong `DATABASE_URL`. Applying anything requires `--apply`.

THE WHOLE RUN IS ONE TRANSACTION. A failure half way leaves the database on
the last fully applied migration, with its ledger row, and nothing partial.
Each file and its ledger row commit together or not at all, so a migration
cannot be applied without being recorded or recorded without being applied.

DRIFT STOPS THE RUN. A file edited after it was applied is the one failure a
version number cannot see: every row looks present while the repository and
the database have diverged. `--apply` refuses and names the files;
`--allow-drift` exists to inspect state, not because continuing is usually
right.

RESEARCH_ONLY project. This is schema bookkeeping: no model, no odds, no
wager.

Usage:
    python -m scripts.migrate_db                     # status, changes nothing
    python -m scripts.migrate_db --apply --dry-run   # what WOULD be applied
    python -m scripts.migrate_db --apply             # apply, in one transaction
    python -m scripts.migrate_db --apply --ensure-tables   # a brand-new database
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.db.migrations import (  # noqa: E402
    MigrationError,
    apply_pending,
    status,
)

logger = logging.getLogger("migrate_db")


def _print_status(report: dict) -> None:
    files = report["files"]
    applied = set(report["applied"])
    print(f"migrations: {report['migrations_dir']}")
    for f in files:
        mark = "applied" if f.version in applied else "PENDING"
        print(f"  {f.version:>3}  {mark:<8} {f.filename}")
    if report["drift"]:
        print()
        print("  DRIFT — these were applied and the file has changed since:")
        for d in report["drift"]:
            print(f"    {d.filename}: recorded {d.recorded_checksum[:12]}…, "
                  f"file is now {d.current_checksum[:12]}…")
    if report["orphaned_ledger_rows"]:
        print()
        print(f"  LEDGER ROWS WITH NO FILE: {report['orphaned_ledger_rows']} — this "
              f"database carries a migration this checkout cannot show you.")
    pending = report["pending"]
    print()
    print(f"  {len(applied)} applied, {len(pending)} pending"
          + (f": {[f.filename for f in pending]}" if pending else ""))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__ and __doc__.splitlines()[1])
    ap.add_argument("--apply", action="store_true",
                    help="Actually apply pending migrations. Without this the "
                         "script only reports, because a migration tool that "
                         "writes by default is one typo away from the wrong "
                         "database.")
    ap.add_argument("--dry-run", action="store_true",
                    help="With --apply: name what would be applied and execute "
                         "nothing.")
    ap.add_argument("--ensure-tables", action="store_true",
                    help="Create the ORM-defined tables first, if absent. "
                         "REQUIRED ON A BRAND-NEW DATABASE: the SQL "
                         "migrations ALTER tables that create_all makes, so "
                         "002 fails with 'relation projections does not "
                         "exist' without this. Additive and idempotent -- "
                         "create_all touches nothing that exists.")
    ap.add_argument("--allow-drift", action="store_true",
                    help="With --apply: proceed although an applied migration's "
                         "file has changed. Records an override; it does not "
                         "make the drift safe.")
    ap.add_argument("--migrations-dir", default=None,
                    help="Override the directory. For tests and for applying a "
                         "subset deliberately.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose or args.apply else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    directory = Path(args.migrations_dir) if args.migrations_dir else None

    try:
        from src.db.session import get_engine
    except Exception as exc:  # noqa: BLE001 — a missing driver is the answer
        print(f"ERROR: cannot import the database layer: {exc}", file=sys.stderr)
        return 2

    try:
        engine = get_engine()
    except Exception as exc:  # noqa: BLE001 — an unset DATABASE_URL is the answer
        print(f"ERROR: no usable database connection: {exc}", file=sys.stderr)
        print("       Set DATABASE_URL (or PGHOST/PGUSER/...) and try again.",
              file=sys.stderr)
        return 2

    if args.ensure_tables:
        # BEFORE the migration transaction, and in its own. The SQL migrations
        # ALTER tables that `Base.metadata.create_all` is what creates
        # (projections, player_game_logs, prop_line_snapshots), so on a
        # brand-new database 002 fails with `relation "projections" does not
        # exist` and the whole run rolls back. Measured against a real
        # Postgres on 2026-10-10; the unit tests build their own tables, so
        # nothing caught it.
        #
        # Separate transaction so a migration failure does not roll the tables
        # back out: they are additive, every migration's ADD COLUMN is
        # IF NOT EXISTS, and a half-migrated schema with its tables present is
        # a better place to debug from than an empty database.
        try:
            from src.db.session import init_db

            init_db()
            print("Ensured the ORM-defined tables exist.")
        except Exception as exc:  # noqa: BLE001 — the driver's error is the answer
            print(f"ERROR: could not create the ORM tables: {exc}", file=sys.stderr)
            return 2

    try:
        # One transaction for the whole run: engine.begin() commits on clean
        # exit and rolls back on any exception, so a failure half way leaves
        # the database on the last fully applied migration.
        with engine.begin() as conn:
            if not args.apply:
                _print_status(status(conn, directory))
                print()
                print("Nothing was applied. Pass --apply to change the schema.")
                return 0

            result = apply_pending(
                conn, directory, dry_run=args.dry_run,
                allow_drift=args.allow_drift,
            )
            _print_status(result["status"])
            print()
            if args.dry_run:
                print("--dry-run: nothing executed.")
            elif result["applied"]:
                print(f"Applied: {result['applied']}")
            else:
                print("Nothing pending.")
            return 0
        from src.db.session import init_db
        init_db()
    except MigrationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3
    except Exception as exc:  # noqa: BLE001 — the driver's own error is the answer
        print(f"ERROR: migration run failed and was rolled back: {exc}",
              file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
