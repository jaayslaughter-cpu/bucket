"""
scripts/load_scraper_panel.py — load the scraper's parquet into the panel.

WHAT THIS IS FOR. `src/ingestion/boxscores.py` fills `player_game_logs` from
stats.nba.com, which does not answer from a datacenter IP, so the deployment
has had no player panel all season. The nba-scraper-2026-27 package writes the
same box scores to parquet from a residential connection; this loads them.

IT IMPORTS ALMOST NOTHING, for the reason `scripts/ingest_bigdataball.py`
gives: this runs on the operator's own machine, and the heavy three
(xgboost, catboost, scikit-learn) are the dependencies least likely to have a
wheel for whatever Python is installed there. This path needs pandas,
pyarrow and SQLAlchemy and none of them.

IT IS NOT A SECOND WRITER. The mapping lives in
`src/ingestion/scraper_panel.py` and the write is
`repository.upsert_player_game_logs` — the same upsert the live path uses, so
the conflict handling, the Int64 casts and the neutral-site lookup are shared.

    python -m scripts.load_scraper_panel --canonical-dir ..\nba-scraper-2026-27\data --dry-run
    python -m scripts.load_scraper_panel --canonical-dir ..\nba-scraper-2026-27\data

WHAT IT NEEDS: the scraper's data directory, and for a real write
`DATABASE_URL` (or PGHOST/PGUSER/PGPASSWORD/PGDATABASE). A dry run needs no
database at all and is the way to see what would land.

RESEARCH_ONLY. This moves box-score rows. It prices nothing, places nothing
and grades nothing.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger("load_scraper_panel")

_CONNECTION_ERRORS = frozenset({"OperationalError", "InterfaceError"})
_ROW_ERRORS = frozenset({"DataError", "IntegrityError"})
_DB_MODULES = frozenset({"sqlalchemy", "psycopg", "psycopg2", "asyncpg"})

_SHAPE_ADVICE = (
    "The parquet is not the shape this mapper maps. Check the scraper wrote "
    "player_boxscore (`python run_daily.py load player_boxscore`) and that "
    "--canonical-dir points at its data directory.",
)
_CONNECTION_ADVICE = (
    "Could not reach the database. The mapping itself does not need one — "
    "re-run with --dry-run to see what WOULD be written.",
    "For a real write, set DATABASE_URL (or PGHOST/PGUSER/PGPASSWORD/"
    "PGDATABASE) in this shell. A new terminal does not inherit them.",
)
_ROW_ADVICE = (
    "The database refused a row. The error above names the column; the "
    "mapper leaves an unreported stat NULL rather than zero, so a NOT NULL "
    "constraint on one of them is the likely cause.",
)


def advice_for(exc: BaseException) -> tuple[str, ...]:
    """Branch the remedy on what actually failed, not on a guess."""
    from src.ingestion.scraper_panel import ScraperPanelError

    if isinstance(exc, ScraperPanelError):
        return _SHAPE_ADVICE
    name = type(exc).__name__
    module = type(exc).__module__.split(".")[0]
    if name in _CONNECTION_ERRORS or (module in _DB_MODULES and name not in _ROW_ERRORS):
        return _CONNECTION_ADVICE
    if name in _ROW_ERRORS:
        return _ROW_ADVICE
    return ()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m scripts.load_scraper_panel",
        description="Load nba-scraper-2026-27 canonical parquet into player_game_logs.",
    )
    ap.add_argument("--canonical-dir", required=True, metavar="PATH",
                    help="the scraper's data directory, its canonical/ "
                         "directory, or canonical/player_boxscore itself")
    ap.add_argument("--dry-run", action="store_true",
                    help="map and report, write nothing; needs no database")
    ap.add_argument("--include-dnp", action="store_true",
                    help="also write rows for players who did not play "
                         "(stats stay NULL, never zero). Off by default: a "
                         "rolling window counts rows, so DNPs change what "
                         "_L5 measures.")
    ap.add_argument("--season-type", action="append", default=None,
                    metavar="NAME",
                    help="season types to keep; repeatable. Default: "
                         "Regular Season, matching what boxscores.py fetches.")
    ap.add_argument("--verbose", action="store_true", help="debug logging")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    from src.ingestion.scraper_panel import (
        DEFAULT_SEASON_TYPES,
        read_canonical_boxscores,
        to_player_panel,
    )

    season_types = tuple(args.season_type) if args.season_type else DEFAULT_SEASON_TYPES

    try:
        frame, files = read_canonical_boxscores(Path(args.canonical_dir))
        panel, report = to_player_panel(
            frame,
            include_dnp=args.include_dnp,
            season_types=season_types,
        )
        report.files_read = files

        print()
        print(f"season types kept: {', '.join(season_types)}")
        for line in report.lines():
            print("  " + line)
        print()
        if report.rows_out:
            dates = panel["GAME_DATE"].dropna()
            if not dates.empty:
                print(f"date range: {dates.min().date()} .. {dates.max().date()}")
            print(f"distinct games:   {panel['GAME_ID'].nunique()}")
            print(f"distinct players: {panel['PLAYER_ID'].nunique()}")
            print()

        if args.dry_run:
            print("DRY RUN — nothing was written. Drop --dry-run to persist.")
            return 0

        from src.db.repository import upsert_player_game_logs

        written = upsert_player_game_logs(panel)
        print(f"Upserted {written} player game-log rows.")
        return 0

    except Exception as exc:  # noqa: BLE001 — this is the operator's console
        logger.error("%s: %s", type(exc).__name__, exc)
        tips = advice_for(exc)
        if tips:
            print(file=sys.stderr)
            for tip in tips:
                print(f"  - {tip}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
