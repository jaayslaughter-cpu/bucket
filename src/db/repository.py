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
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
from sqlalchemy import and_, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.db.models import (
    GameMarketLine,
    ParlayLegRow,
    ParlayTicketRow,
    PipelineRun,
    PlayerGameLog,
    Projection,
    PropLineSnapshot,
    PropResult,
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


def _normalised_start_positions(values: Any) -> pd.Series | None:
    """``STARTING_POSITION`` as G/F/C/None, or None when the caller has none.

    Returning None (rather than an all-null Series) when the column is absent
    keeps `pd.DataFrame` from inventing a column name, and is what
    `df.get("STARTING_POSITION")` already gives back for a panel built before
    the archive ingest asked for it.

    Normalising here and not just trusting the caller is the point: the
    archive emits G/F/C, the live writer emits G/F/C, and anything else --
    'PG', 'F-C', a stray empty string -- is a LISTED position or a bench row,
    neither of which is a starting designation. migration 008's CHECK would
    reject the first and silently accept the second.
    """
    if values is None:
        return None
    from src.features.dvp import normalise_bucket

    return pd.Series(
        [normalise_bucket(v) for v in values], index=values.index, dtype="object"
    )


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
        # Personal fouls. df.get returns None when the panel has no PF, and
        # pd.to_numeric(None) gives an all-null column -- which is the right
        # answer for a panel built before boxscores.COLUMN_MAP started asking
        # the endpoint for PF: the row leaves pf unknown rather than claiming
        # zero. See migrations/006_player_game_log_fouls.sql for why a zero
        # default would be a fabrication rather than a convenience.
        "pf": pd.to_numeric(df.get("PF"), errors="coerce"),
        # The position the player STARTED at, when the caller has one. The
        # league game log does not carry it, so on the live path this is
        # normally absent here and filled later by
        # update_starting_positions(); a panel rebuilt from the Kaggle
        # archive DOES carry it, and then it rides along with the rest.
        # Normalised rather than trusted: a listed position reaching this
        # column would change what POS_BUCKET means downstream, and
        # migration 008's CHECK only admits G, F and C.
        "starting_position": _normalised_start_positions(df.get("STARTING_POSITION")),
        "source": "nba_stats_leaguegamelog",
    })

    undated = int(work["game_date"].isna().sum())
    if undated:
        logger.warning("Dropping %d player-log rows with an unparseable GAME_DATE", undated)
        work = work.loc[work["game_date"].notna()]
    if work.empty:
        raise ValueError("DATA_NOT_AVAILABLE: no player-log rows survived date parsing")

    # Integer columns in the model; a float like 30.0 would be rejected.
    for col in ("pts", "reb", "ast", "fg3m", "stl", "blk", "tov", "pf"):
        work[col] = work[col].astype("Int64")

    work["is_neutral_site"] = _lookup_neutral_site(work)

    rows = _records(work)
    with session_scope() as session:
        stmt = pg_insert(PlayerGameLog).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["nba_game_id", "nba_player_id"],
            set_={
                c: stmt.excluded[c]
                # EVERY written column belongs here. `pf` did not, which
                # made the docstring's "re-ingesting a season corrects rows"
                # false for it alone: a season ingested before
                # boxscores.COLUMN_MAP asked for PF kept pf NULL forever, and
                # the only way to find out was to re-ingest and look. Found
                # while adding starting_position by the same route, which
                # would have inherited the same silence.
                for c in (
                    "player_name", "game_date", "season", "team_abbr",
                    "opponent_abbr", "is_home", "is_neutral_site", "minutes",
                    "pts", "reb", "ast", "fg3m", "stl", "blk", "tov", "pf",
                    "starting_position", "source",
                )
            },
        )
        session.execute(stmt)
    logger.info("Upserted %d player game-log rows", len(rows))
    return len(rows)


def update_starting_positions(frame: pd.DataFrame) -> dict[str, int]:
    """
    Write starting positions onto player-game rows that already exist.

    THIS IS THE WRITER src/features/dvp.py WAS WAITING FOR. The league game
    log that fills the rest of `player_game_logs` carries no position, so the
    traditional box score is pulled separately
    (src/ingestion/starting_positions.py) and joined on here by
    (nba_game_id, nba_player_id).

    IT UPDATES AND NEVER INSERTS, which is the whole design. A position with
    no game log behind it would be a row with a bucket and no statistics --
    enough to shift an opponent's allowed-to-bucket average while contributing
    nothing to it. Those rows are COUNTED and reported as `unmatched` rather
    than created or dropped silently, because a large unmatched count means
    the ids disagree between two endpoints and that is worth seeing.

    Returns {"matched", "updated", "unmatched", "cleared"}. `cleared` counts
    rows the caller asked to set to NULL: permitted, because a corrected
    payload that moves a player from starter to bench has to be able to say
    so, and a writer that could only ever fill would make that uncorrectable.
    """
    required = {"GAME_ID", "PLAYER_ID", "STARTING_POSITION"}
    if frame is None or frame.empty:
        logger.warning(
            "No starting positions to write — refusing to report a successful write"
        )
        return {"matched": 0, "updated": 0, "unmatched": 0, "cleared": 0}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"DATA_NOT_AVAILABLE: starting positions missing {sorted(missing)}"
        )

    work = pd.DataFrame({
        "nba_game_id": frame["GAME_ID"].astype(str),
        "nba_player_id": frame["PLAYER_ID"].astype(str),
        "starting_position": _normalised_start_positions(frame["STARTING_POSITION"]),
    }).drop_duplicates(subset=["nba_game_id", "nba_player_id"], keep="last")

    wanted = {
        (r["nba_game_id"], r["nba_player_id"]): r["starting_position"]
        for r in _records(work)
    }

    matched = updated = cleared = 0
    with session_scope() as session:
        stmt = select(PlayerGameLog).where(
            PlayerGameLog.nba_game_id.in_({g for g, _ in wanted})
        )
        for row in session.execute(stmt).scalars().all():
            key = (row.nba_game_id, row.nba_player_id)
            if key not in wanted:
                continue
            matched += 1
            value = wanted[key]
            if row.starting_position == value:
                continue
            if value is None:
                cleared += 1
            row.starting_position = value
            updated += 1

    unmatched = len(wanted) - matched
    if unmatched:
        logger.warning(
            "starting positions: %d of %d (game, player) pairs had no "
            "player_game_logs row and were NOT inserted — a bucket without a "
            "box score would move an opponent's allowed average without "
            "contributing to it",
            unmatched, len(wanted),
        )
    logger.info(
        "Starting positions: %d matched, %d updated, %d cleared, %d unmatched",
        matched, updated, cleared, unmatched,
    )
    return {
        "matched": matched,
        "updated": updated,
        "unmatched": unmatched,
        "cleared": cleared,
    }


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


def _window(slate_date: str | None, lookback_days: int) -> tuple[Any, Any]:
    """``(lower, upper)`` for a date-bounded load. Shared by the loaders below.

    Factored out when the team-stats and market-line loaders were added rather
    than copied a third time: the UPPER BOUND IS A LEAKAGE GUARD -- a backfill
    for an old slate must not see games played after it -- and three copies of
    a leakage guard is three chances for one of them to drift.
    """
    from datetime import date as _date
    from datetime import timedelta

    if slate_date:
        upper = _date.fromisoformat(slate_date)
    else:
        from src.utils.timezones import pacific_calendar_date

        upper = pacific_calendar_date()
    return upper - timedelta(days=lookback_days), upper


#: What `load_team_game_stats` selects. The DB column names, UNCHANGED, because
#: `team_strength._normalize_team_games` and `defense.REQUIRED_TEAM_COLS`
#: already read these exact names -- so this frame is interchangeable with the
#: one `bigdataball.load_bigdataball_workbook` returns, which is the point.
TEAM_GAME_STAT_COLS: tuple[str, ...] = (
    "nba_game_id", "game_date", "team_abbr", "opponent_abbr",
    "is_home", "is_neutral_site", "points",
    "fg", "fga", "fg3", "fg3a", "ft", "fta",
    "oreb", "dreb", "reb", "ast", "stl", "blk", "tov", "pf",
    "poss", "pace", "off_eff", "def_eff",
)

#: What `load_game_market_lines` selects: `market_context.PREGAME_SOURCE_COLS`
#: and NOTHING ELSE. `game_market_lines` also stores closing_spread,
#: closing_total, closing_odds_raw, halftime_raw and line_movement_1..3 -- every
#: one of which is information from AFTER the game was priced. `attach_market_
#: context` happens to project down to the pregame columns, so a `SELECT *`
#: would not leak today; selecting them anyway would mean the leakage guard was
#: one refactor away from being the only thing standing between a closing line
#: and a feature matrix. So they are never read.
MARKET_LINE_PREGAME_COLS: tuple[str, ...] = (
    "nba_game_id", "game_date", "team_abbr", "opening_spread", "opening_total",
)


def load_team_game_stats(
    slate_date: str | None = None, lookback_days: int = 400
) -> pd.DataFrame:
    """
    The team-game frame, from the database rather than the workbook.

    WHY THIS EXISTS. `main.ingest_market_lines` reads the licensed BigDataBall
    workbook, and a missing path raised `FileNotFoundError` and FAILED THE
    WHOLE SLATE. The image excludes `data/` and `*.xlsx` deliberately -- a
    licensed third-party export does not belong in an image layer -- so a
    deployed container had no workbook and every scheduled run died at step
    [2], having looked healthy at boot.

    But the workbook's contents are ALREADY IN POSTGRES: `ingest_market_lines`
    calls `upsert_team_game_stats` on every run that does find one. The data
    the feature build needs was never actually missing; only the file was. So
    this reads it back, and the Elo and defence layers get the same column
    names they already accept.

    Returns an EMPTY frame when nothing matches, so the caller can tell
    "no workbook AND no history" (a real refusal) from "no workbook" (not one).
    """
    lower, upper = _window(slate_date, lookback_days)
    with session_scope() as session:
        rows = session.execute(
            select(TeamGameStat)
            .where(TeamGameStat.game_date <= upper)
            .where(TeamGameStat.game_date >= lower)
            .order_by(TeamGameStat.game_date)
        ).scalars().all()

    logger.info(
        "Team-game stats query window: %s to %s (%d day lookback) -> %d rows",
        lower, upper, lookback_days, len(rows),
    )
    if not rows:
        return pd.DataFrame(columns=list(TEAM_GAME_STAT_COLS))
    return pd.DataFrame([
        {col: getattr(r, col) for col in TEAM_GAME_STAT_COLS} for r in rows
    ])


def load_game_market_lines(
    slate_date: str | None = None, lookback_days: int = 400
) -> pd.DataFrame:
    """
    The market-line frame, pregame columns only, from the database.

    The companion to `load_team_game_stats`; see its docstring for why either
    exists. Selects `MARKET_LINE_PREGAME_COLS` and refuses to hand back a
    closing column -- asserted here rather than trusted, because this is a
    loader that feeds a feature matrix and the whole project turns on
    `.shift(1)`-grade discipline about what a row could have known.
    """
    from src.features.market_context import assert_no_closing_lines

    lower, upper = _window(slate_date, lookback_days)
    with session_scope() as session:
        rows = session.execute(
            select(GameMarketLine)
            .where(GameMarketLine.game_date <= upper)
            .where(GameMarketLine.game_date >= lower)
            .where(GameMarketLine.status == "VALID")
            .order_by(GameMarketLine.game_date)
        ).scalars().all()

    logger.info(
        "Game market lines query window: %s to %s (%d day lookback) -> %d VALID rows",
        lower, upper, lookback_days, len(rows),
    )
    out = (
        pd.DataFrame(columns=list(MARKET_LINE_PREGAME_COLS)) if not rows
        else pd.DataFrame([
            {col: getattr(r, col) for col in MARKET_LINE_PREGAME_COLS} for r in rows
        ])
    )
    assert_no_closing_lines(out.columns)
    return out


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
            # Personal fouls, so src/features/fouls.py reaches the LIVE path
            # and not only a panel rebuilt from the archive. NULL on every row
            # written before boxscores.COLUMN_MAP began requesting PF, which
            # is what the layer's abstention is for: it can abstain on a null
            # and cannot on a zero, which is why migration 006 writes neither
            # a default nor a backfill.
            "PF": r.pf,
            # The starting position, so src/features/dvp.py reaches the LIVE
            # path and not only a panel rebuilt from the archive. NULL on
            # every row written before starting_positions.py ran, and NULL on
            # every bench appearance after it -- the layer cannot tell those
            # apart and does not need to, since both mean "no bucket from
            # this game".
            "STARTING_POSITION": r.starting_position,
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
            # NULL when the line is whole and no push mass was supplied, never
            # 1 - P(over). See the note on Projection.prob_under.
            "prob_under": r.get("PROB_UNDER"),
            "prob_push": r.get("PROB_PUSH"),
            "market_status": r.get("MARKET_STATUS", "DATA_NOT_AVAILABLE"),
            # The reason behind the status, kept beside it rather than folded
            # into `notes`: notes already carries the under/push refusal, and
            # two unrelated reasons in one free-text column cannot be read
            # apart. See Projection.market_status_reason.
            "market_status_reason": r.get("MARKET_STATUS_REASON"),
            "notes": r.get("NOTES"),
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
                    "prob_under", "prob_push",
                    "model_version", "market_status", "market_status_reason",
                    "ev_per_dollar", "notes",
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


# ---------------------------------------------------------------------------
# prop_results: the settlement ledger's writer
# ---------------------------------------------------------------------------

# Refreshed on a re-run of the same slate. Deliberately NOT here:
#   outcome_status, actual_result, minutes_played, did_not_play, profit_units,
#   stake_units, settled_at_utc, result_source, raw_boxscore_json,
#   settlement_note, closing_*, clv_*
# Every one of those is the GRADER's output. Overwriting them from a
# projections frame would un-settle a graded prop and reset it to PENDING, and
# the settlement runner would then grade it a second time.
PROP_RESULT_REFRESHABLE = (
    "run_id", "nba_player_id", "model_projection", "prob_over",
    "odds", "payout_multiplier", "is_pickem",
)


def pending_prop_result_statement(rows: list[dict[str, Any]]):
    """
    The upsert for PENDING prop_results, built but not executed.

    Separated from the execution so the conflict guard can be tested by
    compiling the SQL, which needs no database.

    ON CONFLICT DO UPDATE ... WHERE prop_results.outcome_status = 'PENDING'.
    The WHERE is the load-bearing part: without it, re-running a slate after
    settlement would overwrite a graded row's pre-settlement fields with the
    model's newer numbers, silently rewriting history in the one table whose
    job is to record what was predicted BEFORE the game.
    """
    stmt = pg_insert(PropResult).values(rows)
    return stmt.on_conflict_do_update(
        constraint="uq_prop_result",
        set_={c: stmt.excluded[c] for c in PROP_RESULT_REFRESHABLE},
        where=PropResult.outcome_status == "PENDING",
    )


def record_pending_prop_results(rows: list[dict[str, Any]]) -> int:
    """
    Write model predictions to prop_results as PENDING, for later grading.

    PREDICTIONS, NOT WAGERS: ``stake_units`` is never set here, so ROI over
    these rows stays undefined until a stake is recorded by hand. Strike rate
    and CLV do not need one.

    Returns the number of rows sent, which is not the number inserted — a row
    whose prop is already settled is skipped by the conflict guard above.
    """
    if not rows:
        return 0
    with session_scope() as session:
        session.execute(pending_prop_result_statement(rows))
    logger.info("Recorded %d pending prop result(s) for grading", len(rows))
    return len(rows)


# ---------------------------------------------------------------------------
# the parlay ledger, durable
# ---------------------------------------------------------------------------

_PARLAY_MODELS = {"tickets": ParlayTicketRow, "legs": ParlayLegRow}

# Columns lifted out of the JSONB record so the ledger is queryable from SQL.
# A key absent from a record simply leaves its column NULL; the record is the
# truth and these are a read convenience.
_PARLAY_PROMOTED = {
    "tickets": (
        "ticket_id", "slate_date", "created_at_utc", "ticket_result",
        "n_legs", "schema_version",
    ),
    "legs": (
        "ticket_id", "leg_id", "slate_date", "game_id", "player_name",
        "market", "leg_result", "schema_version",
    ),
}
_PARLAY_KEYS = {"tickets": ("ticket_id",), "legs": ("ticket_id", "leg_id")}


def _parlay_payload(table: str, record: dict[str, Any]) -> dict[str, Any]:
    clean = {k: (None if (isinstance(v, float) and pd.isna(v)) else v)
             for k, v in record.items()}
    row: dict[str, Any] = {
        c: clean.get(c) for c in _PARLAY_PROMOTED[table]
    }
    # JSONB must hold JSON-representable values. `default=str` rather than
    # dropping: a datetime that cannot serialise would otherwise take the whole
    # ticket down at write time, after the model already committed to it.
    row["record"] = json.loads(json.dumps(clean, default=str))
    return row


def load_parlay_ledger(table: str) -> pd.DataFrame:
    """Every stored row of the parlay ledger, rebuilt from its JSONB records.

    Reading back the record rather than the promoted columns is what makes the
    round-trip exact: JSONB preserves types, so the float precision and
    zero-padded ids that the CSV backend needs explicit reader settings for are
    simply not at risk here.
    """
    model = _PARLAY_MODELS[table]
    with session_scope() as session:
        records = [row.record for row in session.execute(select(model)).scalars()]
    return pd.DataFrame(records) if records else pd.DataFrame()


def upsert_parlay_ledger(table: str, frame: pd.DataFrame) -> int:
    """
    Write ledger rows, updating any that are already stored.

    Upsert rather than insert because a settlement legitimately rewrites a row
    it already wrote. Nothing is ever deleted, so this expresses both the
    CSV backend's append and its whole-frame replace.
    """
    if frame is None or frame.empty:
        return 0
    model = _PARLAY_MODELS[table]
    rows = [_parlay_payload(table, r) for r in _records(frame)]
    with session_scope() as session:
        stmt = pg_insert(model).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=list(_PARLAY_KEYS[table]),
            set_={
                c: stmt.excluded[c]
                for c in (*_PARLAY_PROMOTED[table], "record")
                if c not in _PARLAY_KEYS[table]
            },
        )
        session.execute(stmt)
    return len(rows)


def load_graded_prop_results(
    *,
    lookback_days: int | None = 180,
    market: str | None = None,
) -> list[dict[str, Any]]:
    """
    Graded prop predictions, as the rows ``settlement.calibration`` reads.

    Only the four columns calibration needs, plus the identity ones for a log
    line. Selecting the whole table would pull raw box-score JSONB for every
    row to compute one number from four of its columns.

    PENDING and VOID are filtered in SQL; PUSH is NOT, because the calibration
    report counts pushes as excluded and a board full of whole lines should be
    visible in that count rather than invisible in a WHERE clause.
    """
    conditions = [PropResult.outcome_status.in_(("WIN", "LOSS", "PUSH"))]
    if lookback_days is not None and int(lookback_days) > 0:
        cutoff = datetime.now(timezone.utc).date() - timedelta(days=int(lookback_days))
        conditions.append(PropResult.game_date >= cutoff)
    if market:
        conditions.append(PropResult.market == str(market).upper())

    with session_scope() as session:
        rows = session.execute(
            select(
                PropResult.outcome_status,
                PropResult.prob_over,
                PropResult.predicted_side,
                PropResult.predicted_line,
                PropResult.market,
                PropResult.game_date,
                PropResult.source,
            ).where(and_(*conditions))
        ).all()

    # Decimal -> float here rather than in the calibration module, so the pure
    # function never has to know the column types came from Numeric.
    out: list[dict[str, Any]] = []
    for r in rows:
        out.append({
            "outcome_status": r.outcome_status,
            "prob_over": float(r.prob_over) if r.prob_over is not None else None,
            "predicted_side": r.predicted_side,
            "predicted_line": (
                float(r.predicted_line) if r.predicted_line is not None else None
            ),
            "market": r.market,
            "game_date": r.game_date,
            "source": r.source,
        })
    logger.info("Loaded %d graded prop result(s) for calibration", len(out))
    return out
