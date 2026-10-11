"""
scripts/run_migrations.py — apply pending migrations, non-interactively.

THIS IS A FRONT DOOR, NOT A SECOND IMPLEMENTATION. Every line of the work is
`src/db/migrations.py` (discovery, the checksum ledger, drift detection, one
transaction per run) and `scripts/migrate_db.py` (the argument surface and the
printing). This module adds exactly one thing: a DIFFERENT DEFAULT.

    python -m scripts.migrate_db        # reports. Changes nothing.
    python -m scripts.run_migrations    # APPLIES.

Why both. `migrate_db`'s no-argument behaviour must stay read-only: a
migration tool that writes by default is one typo away from being run against
the wrong `DATABASE_URL`, and a developer with two shells open is exactly who
makes that typo. A container start command is the opposite case — nobody is
there to pass `--apply`, the `DATABASE_URL` is the platform's own injected
value, and a boot that leaves the schema behind the code fails later, at
09:00 PT, in an insert.

So the safe default stays where humans type, and the applying default lives in
a file whose name says what it does. There is no duplicated logic to drift:
this calls `scripts.migrate_db.main` with `--apply` in front of whatever else
you passed, and its exit codes are that script's exit codes (0 applied or
nothing pending, 2 no database, 3 drift or another migration error, 4 the run
failed and was rolled back).

RESEARCH_ONLY project. This is schema bookkeeping: no model, no odds, no wager.

Usage:
    python -m scripts.run_migrations                # apply, in one transaction
    python -m scripts.run_migrations --dry-run      # name them, execute nothing
    python -m scripts.run_migrations --verbose
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.migrate_db import main as _migrate_main  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """
    `migrate_db` with `--apply` prepended.

    Prepended rather than appended so an explicit flag of yours still parses,
    and both flags are idempotent in argparse, so passing either twice is
    harmless.

    `--ensure-tables` IS NOT OPTIONAL HERE, and that was learned the hard way:
    the SQL migrations ALTER tables that `Base.metadata.create_all` creates, so
    on a brand-new database 002 fails with `relation "projections" does not
    exist`, this returns 4, and `scripts/start.sh` -- which hard-fails the
    container on a migration error -- means the worker never starts. A first
    deploy is BY DEFINITION a brand-new database, so the container's front door
    always ensures the tables. `create_all` is additive and idempotent, so on
    every later boot it does nothing.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    return _migrate_main(["--apply", "--ensure-tables", *args])


if __name__ == "__main__":
    raise SystemExit(main())
