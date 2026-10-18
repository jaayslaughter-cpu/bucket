"""
scripts/ingest_player_panel.py — fill `player_game_logs` from the Kaggle
historical export, once, and say whether it landed.

WHY THIS EXISTS. `player_game_logs` is what `load_player_panel` reads, and the
only live writer for it pulls from stats.nba.com. That host is unreachable
from some environments -- measured 2026-10-18, it answers 000 from the
deployed container -- and when the table is empty the slate produces nothing
while reporting success. `src/ingestion/kaggle_nba` is the second path to the
same panel; this is its front door.

NOT `main.py`, AND NOT A SECOND IMPLEMENTATION. The parse, the team
crosswalk and the upsert are `ingestion.kaggle_nba` and
`db.repository.upsert_player_game_logs`, unmodified. What this adds is what a
one-off backfill needs: the target named without its password before the
write, a schema check instead of an insert error after it, and row counts on
both sides of the call so the exit status is about the ingest and nothing
else. Same shape as `scripts/ingest_workbook.py`, for the same reason.

TWO FILTERS, AND THEY ARE NOT OPTIONAL. Both exist so that what is SERVED
matches what was TRAINED, which is the whole reason a rolling mean means
anything:

  * REGULAR SEASON ONLY, matching what the live path requests
    (`SeasonType=Regular+Season`). The export states the season type outright
    in `gameType`, so this is read, not inferred.
  * NON-APPEARANCES DROPPED, matching `scripts/ingest_training_pack`
    decision 1. A did-not-play recorded as zero points teaches a rolling mean
    that a starter scores nothing once a week. Measured on the 2026-10-18
    export: 47,308 dropped and 214,381 kept, which are exactly the figures
    that pack cites -- the same underlying source, filtered the same way.

THE EXPORT CARRIES FGA AND FTA, and migration 009 added the columns for them.
Without those two, `features/scoring_efficiency` abstains and the PTS model
cannot score a single row: its contract names TS_PCT_L10, TS_PCT_TREND and
FT_RATE_L10. This script reports whether they arrived, because a panel that
is complete except for those looks healthy and scores two markets out of
three.

RESEARCH_ONLY. Loads historical box scores into a table. No odds, no model,
no wager.

Usage:
    # 1. fetch the export (needs no Kaggle credentials; ~1 GB)
    pip install kagglehub
    python -c "import kagglehub; print(kagglehub.dataset_download(
        'eoinamoore/historical-nba-data-and-player-box-scores'))"

    # 2. ingest it
    DATABASE_URL=... python -m scripts.ingest_player_panel --kaggle-dir <path>

    # or, from a panel already normalised by this script
    DATABASE_URL=... python -m scripts.ingest_player_panel --panel panel.parquet
    python -m scripts.ingest_player_panel --kaggle-dir <path> --save-panel panel.parquet --dry-run

Exit codes:
    0  rows are in the database and the counts are printed
    2  no usable database (unset/invalid DATABASE_URL, or unreachable)
    3  the schema is behind the code -- migrations pending or drifted
    4  the ingest itself raised
    5  the export or panel is not where you said
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logger = logging.getLogger("ingest_player_panel")

EXIT_OK = 0
EXIT_NO_DATABASE = 2
EXIT_SCHEMA_BEHIND = 3
EXIT_INGEST_FAILED = 4
EXIT_NO_SOURCE = 5

#: Files this script reads out of the export directory.
PLAYER_FILE = "PlayerStatistics.csv"
TEAM_HISTORY_FILE = "TeamHistories.csv"

#: The season type the live path requests, spelled as the export spells it.
REGULAR_SEASON = "Regular Season"

#: Matches the window the shipped artifacts were fit on. Earlier history is
#: not wrong, just unused: `load_player_panel` reads a 400-day window and the
#: rolling features never look past it.
DEFAULT_SINCE = "2018-01-01"

#: Columns read from the export. Named rather than read-all for two reasons:
#: PlayerStatistics.csv is ~390 MB and the full frame alongside Postgres has
#: been enough to get the server OOM-killed, and `playerteamId` /
#: `opponentteamId` are easy to omit by accident -- they are what the
#: era-aware crosswalk resolves abbreviations ON, and without them every
#: TEAM_ABBREVIATION comes back null.
EXPORT_COLUMNS = (
    "firstName", "lastName", "personId", "gameId", "gameDate", "gameType",
    "playerteamId", "playerteamName", "playerteamCity",
    "opponentteamId", "opponentteamName", "opponentteamCity",
    "home", "numMinutes", "points", "assists", "blocks", "steals",
    "fieldGoalsAttempted", "fieldGoalsMade", "threePointersMade",
    "freeThrowsAttempted", "freeThrowsMade", "reboundsDefensive",
    "reboundsOffensive", "reboundsTotal", "foulsPersonal", "turnovers",
    "startingPosition",
)


def _counts() -> dict[str, object]:
    """Rows, newest game and attempt-column coverage in `player_game_logs`."""
    from sqlalchemy import func, select

    from src.db.models import PlayerGameLog
    from src.db.session import session_scope

    with session_scope() as session:
        return {
            "rows": session.execute(
                select(func.count()).select_from(PlayerGameLog)
            ).scalar_one(),
            "newest": session.execute(
                select(func.max(PlayerGameLog.game_date))
            ).scalar(),
            "with_fga": session.execute(
                select(func.count(PlayerGameLog.fga))
            ).scalar_one(),
            "with_fta": session.execute(
                select(func.count(PlayerGameLog.fta))
            ).scalar_one(),
        }


def normalised_panel(kaggle_dir: Path, since: str):
    """
    The export, mapped onto the panel contract and filtered.

    Raises `KaggleNbaError` rather than returning a partial frame: a panel
    missing its date or its target stat cannot produce leakage-safe features.
    """
    import pandas as pd

    from src.ingestion.kaggle_nba import (
        load_team_crosswalk,
        normalize_player_box_scores,
    )

    players = kaggle_dir / PLAYER_FILE
    history = kaggle_dir / TEAM_HISTORY_FILE
    missing = [str(p) for p in (players, history) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"the export is missing {missing}. Expected {PLAYER_FILE} and "
            f"{TEAM_HISTORY_FILE} under {kaggle_dir}."
        )

    raw = pd.read_csv(players, usecols=list(EXPORT_COLUMNS), low_memory=False)
    dates = pd.to_datetime(raw["gameDate"], errors="coerce")
    keep = (raw["gameType"].astype(str) == REGULAR_SEASON) & (dates >= since)
    logger.info(
        "Export: %d row(s); %d are %s on or after %s.",
        len(raw), int(keep.sum()), REGULAR_SEASON, since,
    )
    subset = raw.loc[keep].drop(columns=["gameType"]).copy()
    del raw

    panel = normalize_player_box_scores(
        subset, team_crosswalk=load_team_crosswalk(history)
    )

    minutes = pd.to_numeric(panel["MIN"], errors="coerce")
    appearances = panel.loc[~(minutes.isna() | (minutes <= 0))].copy()
    logger.info(
        "Non-appearances dropped: %d; appearances kept: %d. Dropped to match "
        "ingest_training_pack decision 1, so a rolling mean here means what "
        "it meant at fit time.",
        len(panel) - len(appearances), len(appearances),
    )
    return appearances


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fill player_game_logs from the Kaggle historical export.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--kaggle-dir", type=str, default=None,
                        help=f"directory holding {PLAYER_FILE} and {TEAM_HISTORY_FILE}")
    source.add_argument("--panel", type=str, default=None,
                        help="a panel parquet this script wrote earlier")
    parser.add_argument("--since", type=str, default=DEFAULT_SINCE,
                        help=f"earliest game date to ingest (default {DEFAULT_SINCE})")
    parser.add_argument("--save-panel", type=str, default=None,
                        help="write the normalised panel here before ingesting")
    parser.add_argument("--dry-run", action="store_true",
                        help="normalise and report; write nothing to the database")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    import pandas as pd

    from src.ingestion.kaggle_nba import KaggleNbaError

    # The panel is built BEFORE the database is touched, so a bad path or an
    # unreadable export costs nothing and reports itself as what it is.
    try:
        if args.panel:
            path = Path(args.panel)
            if not path.is_file():
                logger.error("No panel at %s.", path)
                return EXIT_NO_SOURCE
            panel = pd.read_parquet(path)
            logger.info("Panel read from %s: %d row(s).", path, len(panel))
        else:
            panel = normalised_panel(Path(args.kaggle_dir), args.since)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return EXIT_NO_SOURCE
    except KaggleNbaError as exc:
        logger.error("%s", exc)
        return EXIT_INGEST_FAILED

    for column in ("FGA", "FTA"):
        present = int(panel[column].notna().sum()) if column in panel.columns else 0
        if not present:
            logger.warning(
                "%s is absent or empty on this panel. features/"
                "scoring_efficiency requires PTS, FGA and FTA, so the PTS "
                "model will abstain on every row while REB and AST score "
                "normally — a panel that looks complete and is not.",
                column,
            )

    if args.save_panel:
        Path(args.save_panel).parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(args.save_panel)
        logger.info("Normalised panel written to %s.", args.save_panel)

    if args.dry_run:
        logger.info(
            "DRY RUN. %d row(s) ready, %s .. %s. NOTHING WAS WRITTEN.",
            len(panel),
            pd.to_datetime(panel["GAME_DATE"]).min().date(),
            pd.to_datetime(panel["GAME_DATE"]).max().date(),
        )
        return EXIT_OK

    from src.db.session import DatabaseConfigError, database_target, healthcheck

    try:
        target = database_target()
    except DatabaseConfigError as exc:
        logger.error("%s", exc)
        return EXIT_NO_DATABASE

    # Named BEFORE the write: the point of printing a target is that the
    # operator can still hit ^C.
    logger.info("Ingesting %d player-game row(s) INTO %s", len(panel), target)

    ok, message = healthcheck()
    if not ok:
        logger.error("Database not reachable: %s", message)
        return EXIT_NO_DATABASE

    # ONE READER for "is the schema behind the code", shared with
    # scripts/ingest_workbook rather than restated. This ingest needs
    # migration 009 in particular: without fga/fta the rows land and the PTS
    # model still cannot score.
    from scripts.ingest_workbook import _schema_state

    ready, problems = _schema_state()
    if not ready:
        for problem in problems:
            logger.error("SCHEMA BEHIND THE CODE: %s", problem)
        logger.error(
            "Refusing to ingest. Apply them first: "
            "DATABASE_URL=... python -m scripts.run_migrations"
        )
        return EXIT_SCHEMA_BEHIND

    before = _counts()
    try:
        from src.db.repository import upsert_player_game_logs

        written = upsert_player_game_logs(panel)
    except Exception as exc:  # noqa: BLE001 — reported, then surrendered
        logger.exception("INGEST FAILED: %s", exc)
        return EXIT_INGEST_FAILED
    after = _counts()

    logger.info(
        "player_game_logs: %s rows (was %s, +%s new); newest game %s",
        after["rows"], before["rows"],
        int(after["rows"]) - int(before["rows"]),  # type: ignore[arg-type]
        after["newest"],
    )
    # Reported separately because this is the difference between a panel that
    # scores three markets and one that scores two.
    logger.info(
        "attempt columns: fga on %s row(s), fta on %s (was %s / %s). The PTS "
        "contract's TS_PCT and FT_RATE features need these.",
        after["with_fga"], after["with_fta"],
        before["with_fga"], before["with_fta"],
    )
    logger.info("INGEST COMPLETE: %d row(s) upserted into %s.", written, target)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
