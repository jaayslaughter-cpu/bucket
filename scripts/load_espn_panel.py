r"""
scripts/load_espn_panel.py — fill the panel from ESPN, with no laptop.

WHAT THIS IS FOR. The other two ingest paths both need something the deployed
container does not have: `scripts/ingest_bigdataball` needs a licensed file a
human downloads, and `scripts/load_scraper_panel` needs a parquet from a
scraper that only runs where nba.com answers. So the slate died at step [2] on
four consecutive days and the remedy was always "go and run something on your
own machine".

This one runs anywhere ESPN answers, which includes the Railway container --
measured 2026-10-30, against cdn.nba.com's 403 and stats.nba.com's timeout on
the same honest User-Agent.

    python -m scripts.load_espn_panel --days 7 --dry-run
    python -m scripts.load_espn_panel --days 7 --team-games

ONE NAMESPACE PER PANEL, which is the one thing this script refuses over.
ESPN ids are `espn:`-prefixed because they are not the NBA's, and
`features/builder.py:518` groups every rolling feature by PLAYER_ID -- so a
panel holding both namespaces gives each player half a history and a
confident mean over the wrong games. It will not add to a panel that already
uses the other namespace; see `repository.player_id_namespaces`.

RESEARCH_ONLY. This moves box-score rows. It prices nothing and places nothing.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - the annotation only
    import pandas as pd

logger = logging.getLogger("load_espn_panel")

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m scripts.load_espn_panel",
        description="Load ESPN box scores into player_game_logs (and team_game_stats).",
    )
    ap.add_argument("--days", type=int, default=7, metavar="N",
                    help="how many days back from --end to read (default 7)")
    ap.add_argument("--end", default=None, metavar="YYYY-MM-DD",
                    help="last date to read; default today, Pacific")
    ap.add_argument("--dry-run", action="store_true",
                    help="fetch and report, write nothing; needs no database")
    ap.add_argument("--team-games", action="store_true",
                    help="ALSO aggregate team-game rows and write them to "
                         "team_game_stats, which is what gets a slate past "
                         "step [2] without the licensed workbook")
    ap.add_argument("--allow-mixed-ids", action="store_true",
                    help="write even though the panel already holds the other "
                         "id namespace. See the module docstring: this makes "
                         "every rolling feature for the affected players wrong "
                         "and nothing downstream can detect it.")
    ap.add_argument("--verbose", action="store_true", help="debug logging")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # PACIFIC, not UTC. A slate date in this project is a Pacific calendar
    # day (`AGENTS.md` §Conventions), and taking UTC's date here would read
    # tomorrow's empty scoreboard for most of the evening.
    from src.utils.timezones import pacific_calendar_date

    try:
        end = date.fromisoformat(args.end) if args.end else pacific_calendar_date()
    except ValueError:
        print(f"ERROR: --end {args.end!r} is not YYYY-MM-DD", file=sys.stderr)
        return 2
    if args.days < 1:
        print("ERROR: --days must be at least 1", file=sys.stderr)
        return 2
    days = [end - timedelta(days=offset) for offset in range(args.days)]

    from src.ingestion.espn_panel import fetch_player_rows_for_dates

    try:
        frame, report = fetch_player_rows_for_dates(days)
    except Exception as exc:  # noqa: BLE001 — the operator's console
        logger.error("%s: %s", type(exc).__name__, exc)
        return 1

    print()
    print(f"ESPN panel — {days[-1]} .. {days[0]}  ({args.days} day(s))")
    for line in report.lines():
        print("  " + line)
    print()

    if frame.empty:
        print("No player rows. Either no game in the window has finished, or "
              "every one was refused — the lines above say which.")
        return 1

    try:
        return _map_and_write(args, frame)
    except Exception as exc:  # noqa: BLE001 — the operator's console
        # A FETCH THAT SUCCEEDS AND A WRITE THAT FAILS IS THE NORMAL SHAPE OF
        # A BAD RUN: a malformed payload the mapper does not expect, or an
        # ordinary database error. `load_scraper_panel` wraps its whole body
        # for this reason and this one did not, so an operator got a
        # traceback instead of a named error. Found in review 2026-10-31.
        logger.error("%s: %s", type(exc).__name__, exc)
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def _map_and_write(args: argparse.Namespace, frame: "pd.DataFrame") -> int:
    """The half after the fetch: map, aggregate, write, read back."""
    from src.ingestion.scraper_panel import to_player_panel

    panel, panel_report = to_player_panel(frame)
    for line in panel_report.lines():
        print("  " + line)
    print()

    team_frame = None
    if args.team_games:
        from src.ingestion.espn_panel import ESPN_TEAM_SOURCE
        from src.ingestion.scraper_team_games import aggregate_team_games

        owned_by = owned_by_date = None
        if not args.dry_run:
            from src.db.repository import (
                team_game_owners_by_date,
                team_game_sources,
            )

            owned_by = team_game_sources()
            # KEYED ON THE GAME, not on an id. An `espn:` id is a different
            # string from the NBA id for the same night, so the id view alone
            # would write a second row for a game the workbook already holds.
            owned_by_date = team_game_owners_by_date()
        team_frame, team_report = aggregate_team_games(
            frame, owned_by=owned_by, owned_by_date=owned_by_date,
            team_source=ESPN_TEAM_SOURCE,
        )
        print("Team games aggregated from the same box scores:")
        for line in team_report.lines():
            print("  " + line)
        print()

    if args.dry_run:
        print("DRY RUN — nothing was written. Drop --dry-run to persist.")
        return 0

    from src.db.repository import (
        ingested_row_counts,
        player_id_namespaces,
        upsert_player_game_logs,
    )

    # ONE NAMESPACE PER PANEL. Checked BEFORE the write, because afterwards
    # there is no way to tell which rows belong to which player.
    try:
        namespaces = player_id_namespaces()
    except Exception as exc:  # noqa: BLE001
        print(f"COULD NOT READ THE PANEL'S ID NAMESPACES: "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    foreign = {name: n for name, n in namespaces.items() if name != "espn"}
    if foreign and not args.allow_mixed_ids:
        print("REFUSED: the panel already holds non-ESPN player ids.",
              file=sys.stderr)
        for name, count in sorted(foreign.items()):
            print(f"  - {name}: {count:,} row(s)", file=sys.stderr)
        print("  Adding ESPN rows would give any player present in both a "
              "SPLIT history: `builder` groups rolling features by PLAYER_ID, "
              "so each half would be averaged as if it were a whole season, "
              "with no error anywhere.", file=sys.stderr)
        print("  Re-ingest that source for this window instead, or pass "
              "--allow-mixed-ids if you know these players do not overlap.",
              file=sys.stderr)
        return 1

    before = None
    try:
        before = ingested_row_counts()
    except Exception as exc:  # noqa: BLE001 — a baseline, not the write
        logger.warning("No baseline count: %s: %s", type(exc).__name__, exc)

    written = upsert_player_game_logs(panel)
    print(f"Attempted {written} player game-log row(s) (handed to the "
          f"statement; verified below).")

    if team_frame is not None and not team_frame.empty:
        from src.db.repository import upsert_team_game_stats

        team_written = upsert_team_game_stats(team_frame)
        print(f"Attempted {team_written} team-game stat row(s) (handed to the "
              f"statement; verified below).")

    # The read-back, for the reason `ingested_row_counts` gives: an upsert's
    # return value is rows handed to the statement, not rows kept.
    print()
    try:
        counts = ingested_row_counts()
    except Exception as exc:  # noqa: BLE001
        print(f"WROTE, BUT COULD NOT VERIFY: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 1

    print("Verified by reading the tables back on a fresh connection:")
    failures: list[str] = []
    for name in ("player_game_logs", "team_game_stats", "game_market_lines"):
        held = counts.get(name, -1)
        delta = ("" if before is None
                 else f"  ({held - before.get(name, 0):+,} this run)")
        print(f"  {name:<20} {held:,} row(s){delta}")
    if not counts.get("player_game_logs"):
        failures.append("player_game_logs is EMPTY after a persisting run")
    if team_frame is not None and not team_frame.empty:
        from src.db.repository import ingested_keys_present, team_game_keys

        wanted = team_game_keys(team_frame)
        found = ingested_keys_present(team_games=wanted).get("team_game_stats", 0)
        print(f"  {'':<20} {found:,} of this run's {len(wanted):,} "
              f"team-game key(s) present")
        if found < len(wanted):
            failures.append(
                f"{len(wanted) - found:,} of {len(wanted):,} team-game key(s) "
                f"are NOT in the table after the write"
            )
    print()

    if failures:
        print("FAILED: the write did not survive.", file=sys.stderr)
        for line in failures:
            print(f"  - {line}", file=sys.stderr)
        return 1

    print("The slate's step [2] needs team_game_stats OR game_market_lines to "
          "be non-empty. The MKT_* and BLOWOUT_* columns stay unbuilt: those "
          "come from an opening spread, which no box score carries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
