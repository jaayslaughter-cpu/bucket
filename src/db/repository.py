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
    """DataFrame -> list of dicts with NaN converted to None (Postgres NULL).

    The astype(object) is load-bearing: on a numeric column pandas coerces
    the None straight back to NaN, so the driver receives a float NaN where
    an integer column expects NULL and the insert fails.
    """
    return df.astype(object).where(pd.notna(df), None).to_dict(orient="records")


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


def upsert_player_game_logs(df: pd.DataFrame) -> int:
    """
    Write the player panel to Postgres, keyed on (game, player).

    WHY THIS EXISTS: ``load_player_panel`` reads ``player_game_logs``, but
    nothing wrote to it. ``ingest-logs`` cached to Parquet and stopped, so
    the orchestrator's panel was permanently empty and every run reported
    ``success_no_data`` — a pipeline that looked healthy while doing
    nothing.

    Idempotent, so re-ingesting a season corrects rows rather than
    duplicating them. Neutral-site status is not in the NBA stats payload;
    it is filled from ``team_game_stats`` where that row exists and left
    False otherwise, with a count logged, because the altitude rule reads
    it and a wrong value there is a silently wrong feature.
    """
    if df.empty:
        logger.warning("No player game logs to write — refusing to report a successful ingest")
        return 0

    required = {"PLAYER_ID", "GAME_ID", "GAME_DATE", "PLAYER_NAME", "SEASON"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DATA_NOT_AVAILABLE: player logs missing {sorted(missing)}")

    work = pd.DataFrame({
        "nba_game_id": df["GAME_ID"].astype(str),
        "nba_player_id": df["PLAYER_ID"].astype(str),
        "player_name": df["PLAYER_NAME"].astype(str),
        "game_date": pd.to_datetime(df["GAME_DATE"], errors="coerce").dt.date,
        "season": df["SEASON"].astype(str),
        "team_abbr": df.get("TEAM_ABBREVIATION"),
        "opponent_abbr": df.get("OPPONENT_ABBREVIATION"),
        "is_home": df.get("IS_HOME", False),
        "minutes": pd.to_numeric(df.get("MIN"), errors="coerce"),
        "pts": pd.to_numeric(df.get("PTS"), errors="coerce"),
        "reb": pd.to_numeric(df.get("REB"), errors="coerce"),
        "ast": pd.to_numeric(df.get("AST"), errors="coerce"),
        "fg3m": pd.to_numeric(df.get("FG3M"), errors="coerce"),
        "stl": pd.to_numeric(df.get("STL"), errors="coerce"),
        "blk": pd.to_numeric(df.get("BLK"), errors="coerce"),
        "tov": pd.to_numeric(df.get("TOV"), errors="coerce"),
        "source": "nba_stats_leaguegamelog",
    })

    undated = int(work["game_date"].isna().sum())
    if undated:
        logger.warning("Dropping %d player-log rows with an unparseable GAME_DATE", undated)
        work = work.loc[work["game_date"].notna()]
    if work.empty:
        raise ValueError("DATA_NOT_AVAILABLE: no player-log rows survived date parsing")

    # Integer columns in the model; a float like 30.0 would be rejected.
    for col in ("pts", "reb", "ast", "fg3m", "stl", "blk", "tov"):
        work[col] = work[col].astype("Int64")

    work["is_neutral_site"] = _lookup_neutral_site(work)

    rows = _records(work)
    with session_scope() as session:
        stmt = pg_insert(PlayerGameLog).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["nba_game_id", "nba_player_id"],
            set_={
                c: stmt.excluded[c]
                for c in (
                    "player_name", "game_date", "season", "team_abbr",
                    "opponent_abbr", "is_home", "is_neutral_site", "minutes",
                    "pts", "reb", "ast", "fg3m", "stl", "blk", "tov", "source",
                )
            },
        )
        session.execute(stmt)
    logger.info("Upserted %d player game-log rows", len(rows))
    return len(rows)


def _lookup_neutral_site(work: pd.DataFrame) -> pd.Series:
    """Fill is_neutral_site from team_game_stats; False where unknown.

    The NBA stats player endpoint does not report it. Defaulting to False
    is the safe direction — it means the altitude tax still applies to a
    genuine road trip — but the count is logged, because an unflagged
    neutral game is a wrong feature, not a missing one.
    """
    default = pd.Series(False, index=work.index, dtype=bool)
    try:
        with session_scope() as session:
            known = session.execute(
                select(
                    TeamGameStat.nba_game_id,
                    TeamGameStat.team_abbr,
                    TeamGameStat.is_neutral_site,
                ).where(TeamGameStat.is_neutral_site.is_(True))
            ).all()
    except Exception as exc:  # noqa: BLE001 — an optional enrichment, never fatal
        logger.warning("Could not read neutral-site flags (%s) — defaulting to False", exc)
        return default

    if not known:
        logger.info("No neutral-site games recorded in team_game_stats — all rows False")
        return default

    neutral_keys = {(str(g), str(t)) for g, t, _ in known}
    flags = pd.Series(
        [
            (str(g), str(t)) in neutral_keys
            for g, t in zip(work["nba_game_id"], work["team_abbr"].astype(str))
        ],
        index=work.index,
        dtype=bool,
    )
    logger.info(
        "Marked %d of %d player-log rows as neutral-site from team_game_stats",
        int(flags.sum()), len(flags),
    )
    return flags


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
        from src.utils.timezones import pacific_calendar_date

        upper = pacific_calendar_date()
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
            # IS_HOME is reported as the source recorded it. It used to be
            # rewritten to True for neutral-site rows in order to suppress
            # the altitude tax, but IS_HOME is an active model feature and a
            # reporting field: that made every neutral game train and report
            # as a home game. The altitude tax now excludes neutral sites
            # itself, via IS_NEUTRAL_SITE, which is where that rule belongs.
            "IS_HOME": r.is_home,
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
        "Loaded player panel: %d rows, %d players (%d neutral-site rows — IS_HOME "
        "kept as recorded; the altitude tax excludes them downstream)",
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
        # Conflict on the projection's natural identity so re-running a slate
        # overwrites its own rows. Keying on run_id (a per-execution UUID)
        # meant the target never matched and every re-run doubled the table.
        # run_id is refreshed too, recording which run last wrote each row.
        stmt = stmt.on_conflict_do_update(
            index_elements=["nba_game_id", "player_name", "market"],
            set_={
                c: stmt.excluded[c]
                for c in (
                    "run_id", "baseline_projection", "fatigue_multiplier",
                    "fatigue_notes", "final_projection", "line", "prob_over",
                    "model_version", "market_status", "ev_per_dollar", "notes",
                )
            },
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
