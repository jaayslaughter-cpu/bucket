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
        print("If this names a worksheet, pass --sheet. If it names a column, "
              "the export is a different variant than this loader expects "
              "(it needs GAME-ID, DATE and TEAM).", file=sys.stderr)
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

    print("Written to Postgres. The database survives a redeploy, so no later")
    print("container needs this file. Re-run when you want fresher team stats:")
    print("the rows are whatever was last ingested, and `market_frames_"
          "freshness` logs MARKET DATA IS STALE past PROPIQ_MAX_MARKET_LAG_DAYS.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
