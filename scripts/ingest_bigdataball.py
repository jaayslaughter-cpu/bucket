"""
scripts/ingest_bigdataball.py — ingest the licensed workbook, and nothing else.

WHY THIS EXISTS WHEN `main.py --bigdataball` ALREADY DOES IT. Three reasons,
all of them about the machine the licensed file actually lives on:

  1. IT IMPORTS ALMOST NOTHING. `main.main()` runs the whole pipeline, which
     eventually needs xgboost, catboost and scikit-learn. Those are the three
     dependencies least likely to have a wheel for whatever Python the
     operator happens to have (3.14 on the machine this was written for,
     against a project developed on 3.11). Measured: this path pulls pandas
     and SQLAlchemy and NONE of the heavy three, because `main`'s own
     module-level imports are stdlib and everything else there is lazy.
  2. IT COSTS NO PROPLINE QUOTA. A full `main.py` run reaches step [3] once
     step [2] succeeds and spends a slate's worth of requests. The ingest is
     step [2]. Nothing here calls an odds API.
  3. IT EXITS 0 WHEN THE INGEST WORKED. `main.py` returns 1 after a
     successful ingest because a later stage has no model yet, so the one
     signal the operator needs is buried under an unrelated failure.

IT IS THE SAME CODE PATH, NOT A SECOND ONE. `main.ingest_market_lines` is
what step [2] calls when the workbook exists -- this calls that, so the
parsing, the canonical column names and both upserts are shared. A second
loader here would be the defect this tree keeps removing.

WHAT IT NEEDS: `DATABASE_URL` (or PGHOST/PGUSER/PGPASSWORD/PGDATABASE) for
the deployment's database, and the workbook path. Through `railway run` the
first is injected for you.

    railway run python -m scripts.ingest_bigdataball --xlsx "C:\\path\\to\\wb.xlsx"
    railway run python -m scripts.ingest_bigdataball --xlsx wb.xlsx --dry-run

RESEARCH_ONLY. This moves team-game and market-line rows. It prices nothing,
places nothing and grades nothing.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger("propiq.ingest_bigdataball")

#: The libraries whose exceptions can mean the database rather than the
#: workbook. Matched on the exception's own module rather than on its message:
#: psycopg's wording is version- and locale-dependent, and this project has
#: been bitten before by a check that read prose instead of the thing itself.
_DB_MODULES = frozenset({"sqlalchemy", "psycopg", "psycopg2", "asyncpg"})

#: DBAPI's CONNECTION-shaped errors, as an ALLOWLIST rather than "every error
#: from a database module". A class this does not know about falls through to
#: the text classifier and gets the cautious answer, instead of a confident
#: wrong one -- which is the whole failure this function exists to stop.
_CONNECTION_ERRORS = frozenset({"OperationalError", "InterfaceError"})

#: ...and the ones that come from the ROWS, not the connection. A DataError is
#: a value the column's type rejected and an IntegrityError is a constraint the
#: rows violated: both are the workbook's contents, and answering them with
#: "check your connection string" is the same misdirection in the other
#: direction. cubic, PR #23.
_ROW_ERRORS = frozenset({"DataError", "IntegrityError"})

_DB_UNREACHABLE_ADVICE = (
    "This is the DATABASE, not the workbook. Check $DATABASE_URL: the password "
    "must be the real one (an unsubstituted placeholder fails exactly like a "
    "wrong password), and from outside Railway the host must be the TCP "
    "proxy's, since postgres.railway.internal only resolves inside the "
    "platform's network.",
)

#: DIFFERENT FAULT, DIFFERENT REMEDY. `DatabaseConfigError` means nothing is
#: configured at all, so telling someone their password is wrong sends them to
#: check a value that does not exist. cubic, PR #23.
_DB_UNSET_ADVICE = (
    "No database is configured at all. Set $DATABASE_URL (or "
    "PGHOST/PGUSER/PGPASSWORD/PGDATABASE) before running this without "
    "--dry-run. A dry run needs no database and would have got this far.",
)

_SHEET_ADVICE = (
    "This names a WORKSHEET, so pass --sheet with your export's tab name.",
)

#: THE COLUMNS ARE NOT RESTATED HERE. `load_bigdataball_workbook` already
#: raises "missing required columns (...)" naming every one it needs, and that
#: message is printed verbatim on the line above this advice. The first version
#: listed three of the four -- it omitted VENUE -- which is what a second
#: reader of someone else's fact always eventually does. cubic, PR #23.
_COLUMN_ADVICE = (
    "This names a COLUMN, so the export is a different variant than this "
    "loader expects. The error above lists the columns it needs.",
)

_ROW_ADVICE = (
    "The database rejected the ROWS, so this is the export's contents rather "
    "than its shape or the connection. The error above names the column and "
    "the value it refused.",
)

_GENERIC_ADVICE = (
    "If this names a worksheet, pass --sheet. If it names a column, the "
    "export is a different variant than this loader expects. If it names a "
    "host or a password, check $DATABASE_URL.",
)


def advice_for(exc: BaseException) -> tuple[str, ...]:
    """
    The remedy that matches what actually failed.

    WHY THIS EXISTS. The first version printed the workbook advice for EVERY
    failure, so a wrong database password was answered with "If this names a
    worksheet, pass --sheet" -- measured on a real run, 2026-10-24:

        ERROR: the ingest failed: OperationalError: ... password
        authentication failed for user "postgres"
        If this names a worksheet, pass --sheet. ...

    An operator reading that goes looking at their spreadsheet while the fault
    is in a connection string. Advice that fits every error tells you nothing
    about any of them.

    THE BRANCHES ARE DELIBERATELY NARROW. Each one is claimed only for errors
    whose cause is known; everything else gets `_GENERIC_ADVICE`, which names
    the possibilities instead of picking one. A wrong confident answer costs an
    operator more than an honest vague one.
    """
    name = type(exc).__name__
    root = type(exc).__module__.split(".")[0]

    if "DatabaseConfig" in name:
        return _DB_UNSET_ADVICE
    if root in _DB_MODULES:
        if name in _CONNECTION_ERRORS:
            return _DB_UNREACHABLE_ADVICE
        if name in _ROW_ERRORS:
            return _ROW_ADVICE
        # Anything else from a database library -- ProgrammingError for a
        # schema the migrations have not applied, say -- is not confidently
        # any of the above, so it falls through rather than being guessed.

    text = str(exc).lower()
    if "worksheet" in text or "sheet named" in text:
        return _SHEET_ADVICE
    if "column" in text or isinstance(exc, KeyError):
        return _COLUMN_ADVICE
    return _GENERIC_ADVICE


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Ingest the licensed BigDataBall workbook into "
                    "team_game_stats and game_market_lines. RESEARCH_ONLY."
    )
    ap.add_argument("--xlsx", required=True, metavar="PATH",
                    help="Path to the licensed export. On Windows, quote it.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Parse and report, write nothing.")
    ap.add_argument("--sheet", default=None, metavar="NAME",
                    help="Override the worksheet name. The default is "
                         "bigdataball.DEFAULT_SHEET; pass this if your "
                         "export's tab is named differently.")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)-8s %(name)s | %(message)s",
    )

    path = Path(args.xlsx).expanduser()
    if not path.exists():
        # Named before anything else is imported or connected: a wrong path is
        # the most likely failure and the cheapest to report.
        print(f"ERROR: no workbook at {path}", file=sys.stderr)
        print("This loader never downloads or scrapes the export — it is "
              "licensed data and must be supplied.", file=sys.stderr)
        return 2

    # Imported here, not at module scope, so --help and the path check above
    # work on a machine where the dependencies are not installed at all.
    try:
        import main as orchestrator
    except ImportError as exc:
        print(f"ERROR: cannot import the project: {exc}", file=sys.stderr)
        print("Run this from the repository root, with PYTHONPATH=. if needed.",
              file=sys.stderr)
        return 2

    if args.sheet:
        # The sheet name is a default on `load_bigdataball_workbook`, not a
        # parameter step [2] threads through, so an override has to be applied
        # where the default lives rather than passed down.
        from src.ingestion import bigdataball

        logger.info("Worksheet overridden: %r (default %r)",
                    args.sheet, bigdataball.DEFAULT_SHEET)
        bigdataball.DEFAULT_SHEET = args.sheet

    logger.info("Reading %s", path)
    try:
        # THE SAME CALL STEP [2] MAKES. persist=False is a real dry run: the
        # workbook is parsed and reported and neither upsert fires.
        team_games, market_lines = orchestrator.ingest_market_lines(
            path, persist=not args.dry_run,
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 — report, never traceback at an operator
        print(f"ERROR: the ingest failed: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        for line in advice_for(exc):
            print(line, file=sys.stderr)
        return 1

    valid = (
        int((market_lines["status"] == "VALID").sum())
        if "status" in market_lines.columns else len(market_lines)
    )
    print()
    print("BigDataBall ingest — RESEARCH_ONLY")
    print("=" * 60)
    print(f"  team_game_stats rows : {len(team_games):,}")
    print(f"  game_market_lines    : {len(market_lines):,}  ({valid:,} VALID)")
    if "nba_game_id" in market_lines.columns:
        print(f"  distinct games       : {market_lines['nba_game_id'].nunique():,}")
    print("=" * 60)
    if args.dry_run:
        print("DRY RUN — nothing was written. Drop --dry-run to persist.")
        return 0

    if team_games.empty and market_lines.empty:
        # Both empty is the one outcome that looks like success and is not:
        # step [2] would still refuse afterwards.
        print("WROTE NOTHING: both frames are empty. Step [2] will still "
              "refuse. Check the worksheet name and the export's contents.")
        return 1

    # READ THE ROWS BACK BEFORE CLAIMING ANYTHING. This used to print "Written
    # to Postgres." unconditionally, under the PARSE counts above, and that
    # sentence cost 2026-10-23 to 2026-10-28: an operator reported it verbatim
    # and the next three slates each logged `team_game_stats -> 0 rows`. Both
    # were true. The upserts return `len(rows)` -- rows handed to the
    # statement, not rows kept -- so even they would not have caught it. Only
    # a read on a fresh connection, which is what any other reader gets, can.
    try:
        from src.db.repository import ingested_row_counts

        counts = ingested_row_counts()
    except Exception as exc:  # noqa: BLE001 — the operator's console, not a traceback
        print(f"WROTE, BUT COULD NOT VERIFY: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        for line in advice_for(exc):
            print(line, file=sys.stderr)
        return 1

    print("Verified by reading the tables back on a fresh connection:")
    for name in ("team_game_stats", "game_market_lines", "player_game_logs"):
        print(f"  {name:<20} {counts.get(name, -1):,} row(s)")
    print()

    if not counts.get("team_game_stats"):
        print("FAILED: team_game_stats is EMPTY after a persisting run.",
              file=sys.stderr)
        print("  - The workbook parsed fine; the write did not survive.",
              file=sys.stderr)
        print("  - Most likely $DATABASE_URL points at a different database "
              "than the one the worker reads.", file=sys.stderr)
        return 1

    print("Written to Postgres and read back. The database survives a redeploy,")
    print("so no later container needs this file. Re-run when you want fresher")
    print("team stats: the rows are whatever was last ingested, and "
          "`market_frames_")
    print("freshness` logs MARKET DATA IS STALE past PROPIQ_MAX_MARKET_LAG_DAYS.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
