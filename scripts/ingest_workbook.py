"""
scripts/ingest_workbook.py — put the BigDataBall export into Postgres, once,
and say plainly whether it landed.

WHY THIS EXISTS. The workbook is a licensed export, and ``.gitignore`` keeps
every ``data/external/bigdataball/*.xlsx`` out of the repository, so it is
never in a deployed image. A deployed container therefore reads team-game
stats and game market lines from Postgres alone, and ``resolve_market_frames``
raises ``MarketFramesUnavailable`` -- naming this ingest in its message --
until somebody has run it once from a machine that holds both the file and the
database URL. That machine is not the container.

``python main.py --bigdataball <path>`` already performs the ingest. It
performs it as step [2] of a FULL SLATE: prop lines, player-log refresh,
feature matrix, scoring, notification. On a database that is otherwise fresh
every one of those later steps can fail, and the process then exits non-zero
AFTER the workbook rows have already been committed. The operator sees a
failure and cannot tell it from an ingest that did not happen -- so they run it
again, read the same failure, and conclude the ingest is broken when it has
succeeded twice. That ambiguity is the entire reason for this file: here the
exit status is about the ingest and nothing else.

A FRONT DOOR, NOT A SECOND IMPLEMENTATION -- the same shape as
``scripts/run_migrations.py``. The parse and both upserts are
``main.ingest_market_lines``, the function the slate itself calls, unmodified
and unwrapped. What this module adds is only what a one-off needs:

  * the target database named WITHOUT its password, before anything is written,
    because a one-off against a production URL is run by somebody with two
    shells open;
  * a schema check BEFORE the write rather than an insert error after it;
  * row counts measured on both sides of the call, so "it ran" and "it landed"
    are two different statements and this prints the second one.

Nothing here invents a row. The upserts are ``ON CONFLICT DO UPDATE`` on the
workbook's own keys, so running it twice is a no-op, not a duplicate.

RESEARCH_ONLY project. This loads a licensed historical export into a table.
No odds are fetched, no model is scored, no wager is placed or sized.

Usage:
    DATABASE_URL=... python -m scripts.ingest_workbook
    DATABASE_URL=... python -m scripts.ingest_workbook --xlsx path/to/export.xlsx
    DATABASE_URL=... python -m scripts.ingest_workbook --dry-run

Exit codes:
    0  rows are in the database and the counts are printed
    2  no usable database (unset/invalid DATABASE_URL, or it is unreachable)
    3  the schema is behind the code -- migrations are pending or have drifted
    4  the ingest itself raised
    5  no workbook at the path given
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logger = logging.getLogger("ingest_workbook")

EXIT_OK = 0
EXIT_NO_DATABASE = 2
EXIT_SCHEMA_BEHIND = 3
EXIT_INGEST_FAILED = 4
EXIT_NO_WORKBOOK = 5


def _counts() -> dict[str, object]:
    """Rows and newest game date in each table this ingest writes.

    Read through the ORM models rather than literal table names: the tables are
    renamed in one place if they are ever renamed, and a count that silently
    targets a table nobody writes is the kind of green light this script exists
    to not give.
    """
    from sqlalchemy import func, select

    from src.db.models import GameMarketLine, TeamGameStat
    from src.db.session import session_scope

    out: dict[str, object] = {}
    with session_scope() as session:
        for model in (TeamGameStat, GameMarketLine):
            name = model.__tablename__
            out[name] = session.execute(
                select(func.count()).select_from(model)
            ).scalar_one()
            out[f"{name}:newest"] = session.execute(
                select(func.max(model.game_date))
            ).scalar()
    return out


def _schema_state() -> tuple[bool, list[str]]:
    """``(ready, problems)`` for the schema, via the migration ledger."""
    from src.db.migrations import status
    from src.db.session import get_engine

    problems: list[str] = []
    with get_engine().begin() as conn:
        state = status(conn)
    pending = state.get("pending") or []
    drift = state.get("drift") or []
    orphaned = state.get("orphaned_ledger_rows") or []
    if pending:
        problems.append(
            f"{len(pending)} migration(s) pending: "
            + ", ".join(str(getattr(f, 'version', f)) for f in pending)
        )
    if drift:
        problems.append(f"{len(drift)} applied migration(s) no longer match the file")
    if orphaned:
        problems.append(
            "the ledger carries migration(s) with no file in this checkout: "
            + ", ".join(str(v) for v in orphaned)
        )
    return (not problems), problems


def main(argv: list[str] | None = None) -> int:
    import main as pipeline

    parser = argparse.ArgumentParser(
        description="Ingest the BigDataBall workbook into Postgres, once.",
    )
    parser.add_argument(
        "--xlsx",
        type=str,
        default=None,
        help=(
            "Workbook path. Default: the same one the slate uses -- "
            f"${pipeline.ENV_BIGDATABALL}, else {pipeline.DEFAULT_BIGDATABALL_XLSX}."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse the workbook and report what it holds. Writes nothing.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    import os

    xlsx = Path(
        args.xlsx
        or os.environ.get(pipeline.ENV_BIGDATABALL)
        or pipeline.DEFAULT_BIGDATABALL_XLSX
    )
    if not xlsx.is_file():
        logger.error(
            "NO WORKBOOK AT %s. This is a licensed export and is deliberately "
            "not in the repository, so it has to be on the machine you run "
            "this from. Point --xlsx (or $%s) at it.",
            xlsx,
            pipeline.ENV_BIGDATABALL,
        )
        return EXIT_NO_WORKBOOK

    if args.dry_run:
        # No database is touched, so none is required: a dry run is how you
        # check the file before you have the production URL in your shell.
        stats_df, market_df = pipeline.ingest_market_lines(xlsx, persist=False)
        logger.info(
            "DRY RUN. %s holds %d team-game row(s) and %d market-line row(s) "
            "across %d game(s); newest game %s. NOTHING WAS WRITTEN.",
            xlsx,
            len(stats_df),
            len(market_df),
            market_df["nba_game_id"].nunique() if not market_df.empty else 0,
            stats_df["game_date"].max() if not stats_df.empty else "n/a",
        )
        return EXIT_OK

    from src.db.session import DatabaseConfigError, database_target, healthcheck

    try:
        target = database_target()
    except DatabaseConfigError as exc:
        logger.error("%s", exc)
        return EXIT_NO_DATABASE

    # Named BEFORE the write, not after it: the point of printing a target is
    # that the operator can still hit ^C.
    logger.info("Ingesting %s INTO %s", xlsx, target)

    ok, message = healthcheck()
    if not ok:
        logger.error("Database not reachable: %s", message)
        return EXIT_NO_DATABASE

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
        stats_df, market_df = pipeline.ingest_market_lines(xlsx, persist=True)
    except Exception as exc:  # noqa: BLE001 -- reported, then surrendered
        logger.exception("INGEST FAILED: %s", exc)
        return EXIT_INGEST_FAILED
    after = _counts()

    # The counts are what makes this an answer rather than an absence of an
    # error. An upsert that updated every row in place adds nothing, and that
    # is a SUCCESS -- so both numbers are printed and neither is called a
    # failure on its own.
    for table in ("team_game_stats", "game_market_lines"):
        logger.info(
            "%s: %s rows (was %s, +%s new); newest game %s",
            table,
            after[table],
            before[table],
            int(after[table]) - int(before[table]),  # type: ignore[arg-type]
            after[f"{table}:newest"],
        )
    logger.info(
        "INGEST COMPLETE from %s: %d team-game and %d market-line row(s) "
        "upserted into %s.",
        xlsx,
        len(stats_df),
        len(market_df),
        target,
    )
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
