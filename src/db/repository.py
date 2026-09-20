"""
src/db/repository.py — all database reads/writes for the pipeline.

Keeping every query here means main.py never opens a session directly,
and swapping Postgres for something else later touches one file.

Upserts use PostgreSQL's ON CONFLICT DO UPDATE so re-running the
pipeline for the same slate is idempotent rather than duplicating rows.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.db.models import (
    GameMarketLine,
    PipelineRun,
    PlayerGameLog,
    Projection,
    PropLineSnapshot,
    TeamGameStat,
)
from src.db.session import session_scope

logger = logging.getLogger(__name__)


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """DataFrame -> list of dicts with NaN converted to None (Postgres NULL)."""
    return df.where(pd.notna(df), None).to_dict(orient="records")


def upsert_team_game_stats(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    rows = _records(df)
    with session_scope() as session:
        stmt = pg_insert(TeamGameStat).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["nba_game_id", "team_abbr"],
            set_={
                c: stmt.excluded[c]
                for c in (
                    "points", "pace", "off_eff", "def_eff", "poss",
                    "opponent_abbr", "is_home", "is_neutral_site",
                )
            },
        )
        session.execute(stmt)
    logger.info("Upserted %d team-game stat rows", len(rows))
    return len(rows)


def upsert_market_lines(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    rows = _records(df)
    with session_scope() as session:
        stmt = pg_insert(GameMarketLine).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["nba_game_id", "team_abbr", "source"],
            set_={
                c: stmt.excluded[c]
                for c in (
                    "opening_spread", "opening_total", "closing_spread",
                    "closing_total", "moneyline", "status",
                )
            },
        )
        session.execute(stmt)
    logger.info("Upserted %d market line rows", len(rows))
    return len(rows)


def insert_prop_snapshots(df: pd.DataFrame) -> int:
    """
    Prop snapshots are append-only by design: each capture is a distinct
    point-in-time observation, so we never overwrite an earlier one.
    That history is what makes line-movement and CLV analysis possible.
    """
    if df.empty:
        return 0
    rows = _records(df)
    with session_scope() as session:
        session.execute(pg_insert(PropLineSnapshot).values(rows))
    logger.info("Inserted %d prop line snapshots", len(rows))
    return len(rows)


def load_player_panel(slate_date: str | None = None, lookback_days: int = 400) -> pd.DataFrame:
    """
    Load the player game-log panel that feeds the feature builder.

    Both parameters are HONOURED (they were previously accepted and
    silently ignored, so every call loaded the entire table regardless of
    what the caller asked for):

    - ``slate_date``: upper bound. Only games on or before this date are
      loaded, so a backfill run for an old slate cannot see future games
      — a leakage guard at the query level, not just in the builder.
    - ``lookback_days``: lower bound, counted back from slate_date (or
      today). Keeps rolling-window features from scanning the whole
      history on every run.

    Returns an EMPTY DataFrame (not an error) when no logs match — the
    caller treats that as "nothing to project", correct off-season.
    """
    from datetime import date as _date
    from datetime import timedelta

    if slate_date:
        upper = _date.fromisoformat(slate_date)
    else:
        upper = datetime.now(timezone.utc).date()
    lower = upper - timedelta(days=lookback_days)

    with session_scope() as session:
        stmt = (
            select(PlayerGameLog)
            .where(PlayerGameLog.game_date <= upper)
            .where(PlayerGameLog.game_date >= lower)
            .order_by(PlayerGameLog.game_date)
        )
        result = session.execute(stmt).scalars().all()

    logger.info(
        "Player panel query window: %s to %s (%d day lookback)", lower, upper, lookback_days
    )

    if not result:
        return pd.DataFrame()

    df = pd.DataFrame([
        {
            "PLAYER_ID": r.nba_player_id,
            "PLAYER_NAME": r.player_name,
            "GAME_ID": r.nba_game_id,
            "GAME_DATE": r.game_date,
            "SEASON": r.season,
            "TEAM_ABBREVIATION": r.team_abbr,
            "OPPONENT_ABBREVIATION": r.opponent_abbr,
            # NEUTRAL-SITE HANDLING: fatigue_logic.attach_fatigue_column
            # applies the altitude tax when IS_HOME == False AND the
            # opponent plays at altitude (DEN/UTA). At a neutral-site game
            # neither team is home, so a naive IS_HOME=False for both would
            # tax a team that isn't actually travelling to altitude.
            # Setting IS_HOME=True for neutral rows suppresses the tax
            # (the correct behaviour — nobody is the visitor), while
            # IS_NEUTRAL_SITE is carried through so downstream home-court
            # features can distinguish it from a genuine home game.
            "IS_HOME": True if r.is_neutral_site else r.is_home,
            "IS_NEUTRAL_SITE": r.is_neutral_site,
            "MIN": r.minutes,
            "PTS": r.pts,
            "REB": r.reb,
            "AST": r.ast,
            "FG3M": r.fg3m,
            "STL": r.stl,
            "BLK": r.blk,
            "TOV": r.tov,
        }
        for r in result
    ])
    n_neutral = int(df["IS_NEUTRAL_SITE"].sum()) if "IS_NEUTRAL_SITE" in df else 0
    logger.info(
        "Loaded player panel: %d rows, %d players (%d neutral-site rows — altitude tax suppressed)",
        len(df), df["PLAYER_ID"].nunique(), n_neutral,
    )
    return df


def persist_projections(df: pd.DataFrame, run_id: str) -> int:
    if df.empty:
        return 0

    rows = []
    for _, r in df.iterrows():
        rows.append({
            "run_id": run_id,
            "nba_player_id": r.get("PLAYER_ID"),
            "player_name": r.get("PLAYER_NAME", ""),
            "nba_game_id": r.get("GAME_ID"),
            "game_date": r.get("GAME_DATE"),
            "market": r.get("MARKET", "PTS"),
            # KEYS MUST MATCH main.assemble_projections EXACTLY. Three
            # silent-null bugs lived here: BASELINE_PROJECTION and
            # FATIGUE_NOTES were never written by the assembler, so both
            # columns always persisted as NULL. PROJECTION_KEYS below is
            # the single source of truth; test_projection_roundtrip
            # enforces it.
            "baseline_projection": r.get("BASELINE"),
            "fatigue_multiplier": r.get("FATIGUE_MULTIPLIER"),
            "fatigue_notes": r.get("FATIGUE_NOTES"),
            "final_projection": r.get("FINAL_PROJECTION"),
            "line": r.get("LINE"),
            "prob_over": r.get("PROB_OVER"),
            "market_status": r.get("MARKET_STATUS", "DATA_NOT_AVAILABLE"),
        })

    with session_scope() as session:
        stmt = pg_insert(Projection).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["run_id", "nba_player_id", "market", "nba_game_id"],
            set_={c: stmt.excluded[c] for c in ("final_projection", "prob_over", "market_status")},
        )
        session.execute(stmt)
    logger.info("Persisted %d projections for run %s", len(rows), run_id)
    return len(rows)


def record_run(
    run_id: str,
    status: str,
    stage_summary: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    with session_scope() as session:
        stmt = pg_insert(PipelineRun).values(
            run_id=run_id,
            started_at_utc=datetime.now(timezone.utc),
            finished_at_utc=datetime.now(timezone.utc),
            status=status,
            stage_summary=json.loads(json.dumps(stage_summary or {}, default=str)),
            error_summary=error,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["run_id"],
            set_={"status": stmt.excluded.status, "finished_at_utc": stmt.excluded.finished_at_utc},
        )
        session.execute(stmt)
