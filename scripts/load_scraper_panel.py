r"""
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
    ap.add_argument("--team-games", action="store_true",
                    help="ALSO aggregate team-game rows from the same player "
                         "box scores and write them to team_game_stats. This "
                         "is what lets the slate get past step [2] without the "
                         "licensed workbook; see "
                         "src/ingestion/scraper_team_games.py for what it "
                         "recovers (Elo and DEF_*) and what it cannot (MKT_*).")
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

        team_frame = None
        team_report = None
        if args.team_games:
            # SAME PARQUET, read once. A second CLI over the same tree would be
            # a second reader of its layout, and these two frames are the same
            # rows summed differently.
            #
            # AND IT RUNS UNDER --dry-run TOO. It used to sit after the dry-run
            # return, so the only way to see whether the aggregation refused a
            # short roster or a one-sided game was to write to the database
            # first -- a validation step you could not take before the step it
            # was meant to validate. Flagged in review on 2026-10-29.
            # `owned_by` is the one part that needs a connection, and it is
            # asked for only on a real write; a dry run reports what it would
            # aggregate, not which keys it would decline.
            from src.ingestion.scraper_team_games import aggregate_team_games

            owned_by = None
            if not args.dry_run:
                from src.db.repository import team_game_sources

                owned_by = team_game_sources()

            print("Aggregating team-game rows from the same player box scores.")
            team_frame, team_report = aggregate_team_games(
                frame, season_types=season_types, owned_by=owned_by,
            )
            for line in team_report.lines():
                print("  " + line)
            if owned_by is None and not team_frame.empty:
                print("  (dry run: ownership of existing team_game_stats rows "
                      "was not checked, so a key a licensed ingest already "
                      "owns is still counted here)")
            print()

        if args.dry_run:
            print("DRY RUN — nothing was written. Drop --dry-run to persist.")
            return 0

        from src.db.repository import (
            ingested_keys_present,
            ingested_row_counts,
            team_game_keys,
            upsert_player_game_logs,
        )

        # BEFORE, so a delta can be reported. Every upsert returns `len(rows)`
        # -- rows handed to the statement, not rows kept -- which is the number
        # that read "Written to Postgres. 2,644 rows" for five days while the
        # table held nothing. See `ingested_row_counts`.
        try:
            before = ingested_row_counts()
        except Exception as exc:  # noqa: BLE001 — a baseline, not the write
            logger.warning("Could not read a baseline count: %s: %s",
                           type(exc).__name__, exc)
            before = None

        # "ATTEMPTED", because that is what the number is: `upsert_*` returns
        # `len(rows)`. Printed as "Upserted 2,644 rows" it is the exact
        # sentence an operator reported back as confirmation on 2026-10-25
        # while the table held nothing. Only the read-back below claims
        # persistence. Flagged in review, 2026-10-29.
        written = upsert_player_game_logs(panel)
        print(f"Attempted {written} player game-log row(s) (handed to the "
              f"statement; verified below).")

        if team_frame is not None:
            from src.db.repository import upsert_team_game_stats

            if team_frame.empty:
                print("No team-game rows to write — see the lines above for why.")
            else:
                team_written = upsert_team_game_stats(team_frame)
                print(f"Attempted {team_written} team-game stat row(s) "
                      f"(handed to the statement; verified below).")

        # READ IT BACK ON A FRESH CONNECTION. `upsert_*`'s return value is not
        # evidence of anything; only a reader that any other reader could be
        # -- the 09:00 PT slate included -- is. Added in review on 2026-10-29,
        # where the team-game write still reported success from the upsert's
        # own count.
        print()
        try:
            counts = ingested_row_counts()
        except Exception as exc:  # noqa: BLE001 — the operator's console
            print(f"WROTE, BUT COULD NOT VERIFY: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            for tip in advice_for(exc):
                print(f"  - {tip}", file=sys.stderr)
            return 1

        print("Verified by reading the tables back on a fresh connection:")
        for name in ("player_game_logs", "team_game_stats", "game_market_lines"):
            held = counts.get(name, -1)
            delta = (
                "" if before is None
                else f"  ({held - before.get(name, 0):+,} this run)"
            )
            print(f"  {name:<20} {held:,} row(s){delta}")

        failures = []
        if not counts.get("player_game_logs"):
            failures.append("player_game_logs is EMPTY after a persisting run")

        # AND THESE KEYS, for the team table. A row count passes on rows an
        # earlier run wrote while this one's write went nowhere, which is
        # every run after the first -- the same reason
        # `scripts/ingest_bigdataball` asks by key. There is no equivalent for
        # `player_game_logs` here: it is keyed on (player, game) and the panel
        # is tens of thousands of rows, so its delta plus a non-empty table is
        # what this script claims for it.
        wanted = team_game_keys(team_frame)
        if wanted:
            found = ingested_keys_present(team_games=wanted).get(
                "team_game_stats", 0
            )
            print(f"  {'':<20} {found:,} of this run's {len(wanted):,} "
                  f"team-game key(s) present")
            if found < len(wanted):
                failures.append(
                    f"{len(wanted) - found:,} of {len(wanted):,} aggregated "
                    f"team-game key(s) are NOT in the table after the write"
                )
        elif team_frame is not None and not team_frame.empty:
            failures.append("the aggregated frame carries no "
                            "(nba_game_id, team_abbr) to verify by")
        print()
        if failures:
            print("FAILED: the write did not survive.", file=sys.stderr)
            for line in failures:
                print(f"  - {line}", file=sys.stderr)
            print("  - Most likely $DATABASE_URL points at a different "
                  "database than the one the worker reads.", file=sys.stderr)
            return 1

        if team_frame is not None and not team_frame.empty:
            print("The slate's step [2] needs team_game_stats OR "
                  "game_market_lines to be non-empty, so this is what lets "
                  "it build a board without the licensed workbook. The "
                  "MKT_* and BLOWOUT_* columns stay unbuilt: those come "
                  "from an opening spread, which no box score carries.")
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
