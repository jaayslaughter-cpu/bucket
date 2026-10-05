"""
scripts/pull_starting_positions.py — fill player_game_logs.starting_position.

WHAT THIS IS FOR. src/features/dvp.py was built, leakage-audited and measured
on history while abstaining on every row of a LIVE panel, because the league
game log that fills `player_game_logs` carries no position and nothing else
wrote one. This is the pass that closes it: one traditional-box-score call per
game, cached to parquet, then written onto the rows that already exist.

THE SEMANTICS GATE RUNS BEFORE ANYTHING IS WRITTEN, and it can stop the run.
Five players start a basketball game, so every team-game must carry exactly
five filled positions. A payload whose `position` column is the player's
LISTED position fills one for everyone who dressed and lands at eleven or
more: it parses, it looks plausible, and it is a different quantity from the
archive's — enough to change what POS_BUCKET means for every row downstream
while nothing looks wrong. --force exists to record an override deliberately;
it does not make the gate wrong.

IT UPDATES AND NEVER INSERTS. A position with no game log behind it would be a
row with a bucket and no statistics, which would shift an opponent's
allowed-to-bucket average while contributing nothing to it. Those are counted
and reported as `unmatched`.

stats.nba.com IS DENIED AT SOME PROXIES, including the one this repository's
CI and cloud sessions run behind ("CONNECT tunnel failed, response 403"). This
has to run where nba.com is reachable. --from-cache replays a pull made
elsewhere, so the fetch and the write can happen on different machines.

RESEARCH ONLY. No wager, no odds, no bet sizing: this writes a box-score
designation.

Usage:
    # fetch and write one season
    python -m scripts.pull_starting_positions --season 2025-26

    # fetch only, so the write can run elsewhere
    python -m scripts.pull_starting_positions --season 2025-26 --no-write

    # write from a pull made on another machine
    python -m scripts.pull_starting_positions --season 2025-26 --from-cache

    # see what would happen, touching nothing
    python -m scripts.pull_starting_positions --season 2025-26 --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.starting_positions import (  # noqa: E402
    DEFAULT_PAUSE_SECONDS,
    StartingPositionError,
    cache_path_for,
    check_starting_position_semantics,
    fetch_many_starting_positions,
    load_cached_starting_positions,
    save_starting_positions,
)

logger = logging.getLogger("pull_starting_positions")


def game_ids_for_season(season: str, limit: int | None = None) -> list[str]:
    """Distinct game ids already in `player_game_logs` for a season.

    The table is the right source rather than a schedule endpoint: these are
    exactly the rows a position can be written onto, so asking for any other
    game would guarantee an `unmatched` count.
    """
    from sqlalchemy import distinct, select

    from src.db.models import PlayerGameLog
    from src.db.session import session_scope

    with session_scope() as session:
        stmt = (
            select(distinct(PlayerGameLog.nba_game_id))
            .where(PlayerGameLog.season == season)
            .order_by(PlayerGameLog.nba_game_id)
        )
        ids = [g for (g,) in session.execute(stmt).all() if g]
    if limit:
        ids = ids[:limit]
    return ids


def _report(frame: pd.DataFrame) -> None:
    filled = frame["STARTING_POSITION"].notna()
    print(f"  rows: {len(frame):,}   filled: {int(filled.sum()):,} "
          f"({filled.mean():.1%})")
    counts = frame.loc[filled, "STARTING_POSITION"].value_counts()
    print("  mix: " + ", ".join(f"{k}={v:,}" for k, v in counts.items()))
    # The archive's own mix is 2:2:1 (35,594 G / 35,593 F / 17,799 C across
    # 214,381 rows). A pull that disagrees is worth looking at before writing.
    print("  archive mix for comparison: G 2, F 2, C 1 per team-game")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__ and __doc__.splitlines()[1])
    ap.add_argument("--season", required=True,
                    help="Season as player_game_logs records it, e.g. 2025-26.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only the first N games. For a smoke test against a "
                         "reachable endpoint before paying for a full season.")
    ap.add_argument("--from-cache", action="store_true",
                    help="Skip the fetch and write the cached parquet, so a "
                         "pull made where nba.com is reachable can be written "
                         "where the database is.")
    ap.add_argument("--no-write", action="store_true",
                    help="Fetch and cache only. The complement of --from-cache.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report and touch nothing: no cache write, no database "
                         "write.")
    ap.add_argument("--force", action="store_true",
                    help="Write even if the five-starters semantics gate fails. "
                         "Records an override; it does not make the gate wrong.")
    ap.add_argument("--pause-seconds", type=float, default=DEFAULT_PAUSE_SECONDS)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.from_cache and args.no_write:
        print("ERROR: --from-cache with --no-write would do nothing at all.",
              file=sys.stderr)
        return 2

    # --- get the rows -----------------------------------------------------
    if args.from_cache:
        frame = load_cached_starting_positions(args.season)
        if frame is None:
            print(f"ERROR: no cache at {cache_path_for(args.season)}. Run the "
                  f"fetch where stats.nba.com is reachable first.",
                  file=sys.stderr)
            return 2
        print(f"Loaded {len(frame):,} cached row(s) from "
              f"{cache_path_for(args.season)}")
    else:
        try:
            ids = game_ids_for_season(args.season, args.limit)
        except Exception as exc:  # noqa: BLE001 — a missing database is the answer
            print(f"ERROR: cannot list games for {args.season}: {exc}",
                  file=sys.stderr)
            return 2
        if not ids:
            print(f"ERROR: player_game_logs has no rows for season "
                  f"{args.season}, so there is nothing a position could be "
                  f"written onto. Ingest the game logs first.", file=sys.stderr)
            return 2
        print(f"{len(ids):,} game(s) in player_game_logs for {args.season}")
        try:
            frame, failures = fetch_many_starting_positions(
                ids, pause_seconds=args.pause_seconds
            )
        except StartingPositionError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        if failures:
            print(f"  {len(failures)} game(s) FAILED and are absent from this "
                  f"pull; first: {failures[0]['error'][:160]}")

    _report(frame)

    # --- the gate ---------------------------------------------------------
    try:
        report = check_starting_position_semantics(frame)
        print(f"  gate PASSED: {report['team_games']:,} team-game(s), exactly "
              f"5 filled position(s) each")
    except StartingPositionError as exc:
        print(f"  gate FAILED: {exc}", file=sys.stderr)
        if not args.force:
            print("  nothing written. Pass --force only if you have read the "
                  "message above and decided the override is correct.",
                  file=sys.stderr)
            return 3
        print("  --force given: writing anyway, with the failure recorded above.",
              file=sys.stderr)

    if args.dry_run:
        print("--dry-run: nothing cached, nothing written.")
        return 0

    if not args.from_cache:
        path = save_starting_positions(frame, args.season)
        print(f"  cached -> {path}")

    if args.no_write:
        print("--no-write: cache only. Re-run with --from-cache where the "
              "database is reachable.")
        return 0

    from src.db.repository import update_starting_positions

    try:
        result = update_starting_positions(frame)
    except Exception as exc:  # noqa: BLE001 — a missing database is the answer
        print(f"ERROR: database write failed: {exc}", file=sys.stderr)
        return 2
    print(f"  wrote: {result['updated']:,} updated of {result['matched']:,} "
          f"matched; {result['cleared']:,} cleared; {result['unmatched']:,} "
          f"unmatched (not inserted)")
    if result["unmatched"]:
        print("  an unmatched pair is a (game, player) the box score has and "
              "player_game_logs does not. A large count means the two "
              "endpoints disagree about ids.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
