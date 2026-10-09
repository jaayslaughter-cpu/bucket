"""
main.py — PropIQ Analytics end-to-end pipeline orchestrator.

SCOPE: NBA ONLY. No NCAA/CBB.
STATUS: RESEARCH_ONLY. No wagering, no stake instructions, no fabricated lines.

--------------------------------------------------------------------------
CORRECTIONS APPLIED (v2) after integration testing against the real repo
--------------------------------------------------------------------------
v1 of this file was written against ASSUMED interfaces and was wrong in
four ways. All four are fixed here, each verified against real source:

1. FATIGUE IS ALREADY WIRED. `features/builder.py:build_feature_matrix`
   imports and calls `attach_fatigue_column(df)` (~line 208) and folds the
   result into every `{stat}_L2`:
       df[f"{stat}_L2"] = df[f"{stat}_BASELINE"] * pace * fatigue * minutes_ratio
   This orchestrator now VERIFIES fatigue was applied instead of applying
   it again — v1 would have double-counted the multiplier.

2. REAL COLUMN NAMES. `attach_fatigue_column` emits lowercase
   `fatigue_multiplier` (+ `is_back_to_back`, `is_3_in_4`, `is_4_in_5`),
   not `FATIGUE_MULTIPLIER`. There is no `BASELINE_PROJECTION`; real
   projection columns are `{stat}_BASELINE` and `{stat}_L2`. v1's guard
   checked for a column that never exists and would have raised its own
   RuntimeError on every run.

3. XGBoost CONSTRUCTION. `XGBoostPropPipeline.__init__` REQUIRES
   `feature_cols: Sequence[str]` positionally, and `predict_proba_over`
   needs a fitted booster. v1's bare `XGBoostPropPipeline()` always
   raised and silently skipped scoring.

4. THE EV GATE NEEDS TWO-WAY ODDS. `quant/contracts.py:market_ev_gate`
   requires status == VALID *and* both `over_odds_american` and
   `under_odds_american`. BigDataBall supplies GAME spread/total/ML, not
   two-way player-prop prices, so it does NOT unlock prop EV. v1 listed
   an EV stage it never actually called; the gate is now invoked and its
   verdict recorded.

Execution order:
    [1] preflight         -> DB reachable, guideline present?
    [2] ingest_market     -> BigDataBall -> game_market_lines (GAME markets)
    [3] ingest_props      -> pick'em boards -> prop_line_snapshots
   [3b] refresh_logs      -> NBA league game log -> player_game_logs
    [4] load_player_panel -> DB -> raw panel, then panel_freshness
    [5] features          -> build_feature_matrix (fatigue applied INSIDE)
    [6] verify_fatigue    -> assert it happened; fail loud if not
    [7] score             -> XGBoost P(Over) if a fitted model exists
    [8] ev_gate           -> market_ev_gate verdict (expected: abstain)
    [9] persist           -> projections + pipeline_runs audit row

Run:
    python main.py --init-db
    python main.py --date 2026-01-15
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from src.models.labels import (
    RESEARCH_LINE_COL,
    mask_probabilities_at_unsupported_lines,
)
from src.utils.timezones import DISPLAY_TZ_NAME, now_pacific, pacific_calendar_date

load_dotenv()

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
)
logger = logging.getLogger("propiq.main")
logger.info("Display timezone=%s (storage=%s)", DISPLAY_TZ_NAME, "UTC")

# Real column names emitted by src/features/fatigue_logic.py
FATIGUE_COL = "fatigue_multiplier"
FATIGUE_FLAG_COLS = ("is_back_to_back", "is_3_in_4", "is_4_in_5")

DEFAULT_STATS = ("PTS", "REB", "AST", "FG3M")

MASTER_GUIDELINE_DEFAULT_PATH = Path("config/master_guideline_props.yaml")
#: The path this pipeline read before `resolve_model_artifact` existed. Kept as
#: the LAST resort rather than the default: nothing writes here. `train-stats`
#: writes to config/model_comparison.yaml's `artifacts_dir`, and the two paths
#: never coincided (readiness item O2).
MODEL_ARTIFACT_DEFAULT = Path("models/xgb_prop_over.json")

ENV_MODEL = "PROPIQ_MODEL"
ENV_FORWARD_SLATE = "PROPIQ_FORWARD_SLATE"
#: Refresh `player_game_logs` from the NBA stats API before building the panel.
#: On by default: the alternative is a panel frozen at whatever was last
#: ingested by hand, which is the defect this exists to close.
ENV_LOG_REFRESH = "PROPIQ_REFRESH_PLAYER_LOGS"
#: How many days the newest COMPLETED game in the panel may trail the slate
#: before the run says so. Three, because games are daily in season and the
#: All-Star break is the only legitimate longer gap.
ENV_MAX_PANEL_LAG = "PROPIQ_MAX_PANEL_LAG_DAYS"
DEFAULT_MAX_PANEL_LAG_DAYS = 3


def _flag_env(name: str, default: bool) -> bool:
    """An env flag, defaulting rather than raising on an unusable value."""
    raw = (os.environ.get(name) or "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


def _int_env(name: str, default: int, *, low: int, high: int) -> int:
    """
    A bounded integer env var, defaulting rather than raising.

    Same contract as ``scheduler_worker._int_env`` and deliberately a second
    small implementation rather than an import: main.py does not depend on the
    worker, and inverting that for four lines would make the orchestrator
    require APScheduler to parse a number.
    """
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("%s=%r is not an integer; using %d", name, raw, default)
        return default
    if not low <= value <= high:
        logger.warning(
            "%s=%d is outside [%d, %d]; using %d", name, value, low, high, default
        )
        return default
    return value


def resolve_model_artifact(
    explicit: Path | str | None = None,
    *,
    market: str | None = None,
) -> tuple[Path | None, str]:
    """
    Find the fitted artifact to score with. Returns ``(path, how)``.

    WHY THIS EXISTS (readiness item O2). ``score_prob_over`` defaulted to
    ``models/xgb_prop_over.json``; ``train-stats`` writes
    ``xgboost_{MARKET}.json`` under ``config/model_comparison.yaml``'s
    ``artifacts_dir``. The two never coincided, the ``models/`` directory does
    not exist, and ``scheduler_worker.run_slate`` calls ``main.main([])`` with
    no ``--model`` — so a scheduled run scored nothing EVEN AFTER a model was
    trained into the right place, and said only "no fitted model at ...".

    Resolution order, most explicit first:

      1. ``--model``, which still wins over everything;
      2. ``PROPIQ_MODEL``, so a deployed worker can be pointed at an artifact
         without a code change or a command-line argument;
      3. the newest ``xgboost_*.json`` in ``artifacts_dir`` that HAS its
         ``.meta.json`` sidecar — scoring without the sidecar's feature_cols is
         refused anyway, so an artifact without one is not a candidate;
      4. ``MODEL_ARTIFACT_DEFAULT``, the legacy path, last.

    Deliberately NOT read: ``src/models/artifact_registry.py``. It would be the
    right index for this, and nothing writes to it — resolving through a dead
    module is how ``oddspapi`` stayed in the source precedence for months. The
    filesystem is where training actually puts files, so that is what is read.

    ``(None, reason)`` when nothing is found, so the caller logs one reason
    rather than a missing-file message for a path nobody chose.
    """
    if explicit:
        return Path(explicit), "--model"

    from_env = (os.environ.get(ENV_MODEL) or "").strip()
    if from_env:
        return Path(from_env), ENV_MODEL

    try:
        from src.models.compare import load_comparison_config

        artifacts_dir = Path(
            str((load_comparison_config() or {}).get(
                "artifacts_dir", "data/external/model_runs/comparison"
            ))
        )
    except Exception as exc:  # noqa: BLE001 — a missing config is not fatal here
        logger.debug("Could not read artifacts_dir from the comparison config: %s", exc)
        artifacts_dir = Path("data/external/model_runs/comparison")

    pattern = f"xgboost_{market.upper()}.json" if market else "xgboost_*.json"
    if artifacts_dir.is_dir():
        candidates = [
            p for p in sorted(artifacts_dir.glob(pattern))
            if p.with_suffix(".meta.json").exists()
        ]
        if candidates:
            newest = max(candidates, key=lambda p: p.stat().st_mtime)
            return newest, f"{artifacts_dir}/{pattern}"
        unpaired = sorted(artifacts_dir.glob(pattern))
        if unpaired:
            logger.warning(
                "%d artifact(s) in %s have no .meta.json sidecar, so none can be "
                "scored with: %s. The feature_cols used at training time must be "
                "persisted beside the model.",
                len(unpaired), artifacts_dir, [p.name for p in unpaired][:4],
            )

    if MODEL_ARTIFACT_DEFAULT.exists():
        return MODEL_ARTIFACT_DEFAULT, "legacy default"
    return None, (
        f"no artifact found: {ENV_MODEL} unset, no xgboost_*.json with a sidecar "
        f"in {artifacts_dir}, and {MODEL_ARTIFACT_DEFAULT} does not exist. Train "
        f"one with `scripts/nba_model_cli.py train-stats --market PTS "
        f"--start-date ... --end-date ...` and it will be found there."
    )


# ---------------------------------------------------------------------------
# MASTER GUIDELINE INTEGRATION POINT (placeholder — file not supplied yet)
# ---------------------------------------------------------------------------

def load_master_guideline(path: Path | None = None) -> dict[str, Any] | None:
    """Return the parsed guideline, or None if absent (caller must log that)."""
    resolved = path or Path(os.environ.get("PROPIQ_MASTER_GUIDELINE", MASTER_GUIDELINE_DEFAULT_PATH))
    if not resolved.exists():
        return None
    import yaml

    with open(resolved, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# [1] preflight
# ---------------------------------------------------------------------------

def preflight(require_db: bool = True) -> dict[str, Any]:
    from src.db.session import healthcheck

    report: dict[str, Any] = {}
    ok, message = healthcheck()
    report["database"] = {"ok": ok, "message": message}
    if not ok and require_db:
        logger.error("Preflight FAILED: %s", message)
        raise SystemExit(1)
    logger.info("Preflight: database %s", "OK" if ok else "SKIPPED (--no-db)")

    guideline = load_master_guideline()
    if guideline is None:
        logger.warning(
            "MASTER GUIDELINE NOT FOUND at %s — prop-source selection falls back to "
            "config.json `pickem.approved_sources`.",
            MASTER_GUIDELINE_DEFAULT_PATH,
        )
        report["master_guideline"] = {"applied": False}
    else:
        logger.info("Master guideline loaded (%d keys)", len(guideline))
        report["master_guideline"] = {"applied": True, "keys": list(guideline)}
    return report


# ---------------------------------------------------------------------------
# [2] game market lines — GAME markets only, NOT prop EV
# ---------------------------------------------------------------------------

def ingest_market_lines(
    xlsx_path: Path, persist: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load BigDataBall. IMPORTANT: supplies GAME spread/total/moneyline. It
    does NOT supply two-way player-prop American odds, so it cannot by
    itself satisfy market_ev_gate for player props.

    Returns ``(team_games, market_lines)`` — BOTH frames, because both are
    inputs to build_feature_matrix. This function used to return only the
    market frame and discard the team one, while the feature build was called
    with neither: the workbook was read, parsed, persisted, and then the Elo,
    market-context, defence and blowout columns were left off the matrix
    entirely, because the builder omits a layer whose input is absent rather
    than inventing it. The orchestrator therefore produced a strictly narrower
    feature set than scripts/nba_model_cli.py does from the same workbook.
    """
    from src.ingestion.bigdataball import load_bigdataball_workbook

    stats_df, market_df = load_bigdataball_workbook(xlsx_path)
    logger.info(
        "Game markets: %d rows (%d VALID) across %d games — GAME spread/total/ML only, "
        "not player-prop two-way odds.",
        len(market_df), (market_df["status"] == "VALID").sum(), market_df["nba_game_id"].nunique(),
    )
    if persist:
        from src.db.repository import upsert_market_lines, upsert_team_game_stats

        upsert_team_game_stats(stats_df)
        upsert_market_lines(market_df)
    return stats_df, market_df


# ---------------------------------------------------------------------------
# [3] prop lines
# ---------------------------------------------------------------------------

def season_label_for(slate_date: str) -> str:
    """
    ``"2026-01-15"`` -> ``"2025-26"``, the label the endpoint and panel use.

    The August cut comes from ``features.season.season_start_year`` rather
    than being re-derived here: a second copy of that rule is how two parts of
    a project end up disagreeing about which season January belongs to.
    """
    from src.features.season import season_start_year

    # errors="coerce", so an unreadable date becomes NaT and is refused by the
    # message below. Letting pd.Timestamp raise would surface pandas'
    # DateParseError -- which IS a ValueError, so a caller catching ValueError
    # still works, but the reader gets "Unknown datetime string format"
    # instead of being told which input the season could not be read from.
    parsed = pd.to_datetime(pd.Series([slate_date]), errors="coerce")
    start = season_start_year(parsed).iloc[0]
    if pd.isna(start):
        raise ValueError(f"DATA_NOT_AVAILABLE: cannot read a season from {slate_date!r}")
    start = int(start)
    return f"{start}-{str(start + 1)[-2:]}"


def refresh_player_logs(slate_date: str, *, persist: bool = True) -> dict[str, Any]:
    """
    Pull this season's player game logs and upsert them before the panel loads.

    WHY THIS STEP EXISTS. It did not, and that was the sharpest finding of the
    2026-10-08 audit. Step [4] reads `player_game_logs`; the only writer was
    ``nba_model_cli ingest-logs --persist``, which nothing scheduled. So on a
    deployed container the panel sat frozen at whatever was last ingested BY
    HAND and every rolling feature aged silently, because a stale panel is
    indistinguishable from a quiet one. The slate refreshes it now.

    ONE REQUEST PER RUN. ``leaguegamelog`` returns every player-game for a
    season, so a daily refresh is a single call, not one per game. That is why
    this is affordable at the top of every slate.

    ``use_cache=False`` IS THE WHOLE POINT, and is easy to get wrong:
    ``load_player_game_logs`` is cache-first, so the default would hand back
    yesterday's parquet and fetch nothing, leaving exactly the staleness this
    step was added to remove while reporting a successful refresh. The cache
    is still WRITTEN, so it keeps working as an offline fallback for the CLI.

    BOUNDED TO THE SLATE DATE. A run for a past date must not write games
    played after it. ``load_player_panel`` already bounds the panel at query
    level, so this is the second line, not the only one — but it means a
    backfill writes only what was knowable then, rather than relying on every
    future reader to filter correctly.

    BEST EFFORT. A denied or unreachable nba.com is reported as FAILED and the
    run continues on the panel already in the database, because abstaining on
    stale data beats not running at all. What it must never do is continue
    QUIETLY: the failure lands in ``stage_summary`` and therefore in the
    ``pipeline_runs`` row, and the staleness check below measures the
    consequence independently.
    """
    if not _flag_env(ENV_LOG_REFRESH, True):
        logger.info(
            "%s is off — the panel will be whatever was last ingested by hand.",
            ENV_LOG_REFRESH,
        )
        return {"status": "SKIPPED", "reason": f"{ENV_LOG_REFRESH} is off"}
    if not persist:
        return {
            "status": "SKIPPED",
            "reason": "--no-db: there is nothing to upsert into",
        }

    try:
        season = season_label_for(slate_date)
    except ValueError as exc:
        logger.error("Player-log refresh skipped: %s", exc)
        return {"status": "FAILED", "reason": str(exc)}

    from src.ingestion.boxscores import (
        BoxScoreFetchError,
        BoxScoreLoadConfig,
        load_player_game_logs,
    )

    config = BoxScoreLoadConfig(seasons=(season,), use_cache=False)
    try:
        fetched = load_player_game_logs(config)
    except BoxScoreFetchError as exc:
        logger.error(
            "Player-log refresh FAILED for %s: %s. The run continues on the "
            "panel already in the database; see panel_freshness below for how "
            "old that is.", season, exc,
        )
        return {"status": "FAILED", "season": season, "reason": str(exc)}
    except Exception as exc:  # noqa: BLE001 — a surprise here must not stop the slate
        logger.error("Player-log refresh FAILED for %s: %r", season, exc)
        return {"status": "FAILED", "season": season, "reason": repr(exc)}

    if fetched.empty:
        return {"status": "FAILED", "season": season,
                "reason": "the endpoint returned no rows"}

    dates = pd.to_datetime(fetched["GAME_DATE"], errors="coerce")
    upper = pd.Timestamp(slate_date)
    keep = dates <= upper
    held_back = int((~keep).sum())
    bounded = fetched.loc[keep]
    if bounded.empty:
        return {
            "status": "SKIPPED", "season": season,
            "reason": (
                f"every one of the {len(fetched)} fetched row(s) is dated after "
                f"{slate_date}, so this is a slate before the season's first game"
            ),
            "rows_after_slate_held_back": held_back,
        }

    from src.db.repository import upsert_player_game_logs

    try:
        written = int(upsert_player_game_logs(bounded))
    except Exception as exc:  # noqa: BLE001 — report, never claim a write
        logger.error("Player logs fetched but the database write failed: %r", exc)
        return {"status": "FAILED", "season": season,
                "reason": f"upsert failed: {exc!r}", "rows_fetched": int(len(bounded))}

    newest = pd.to_datetime(bounded["GAME_DATE"], errors="coerce").max()
    out = {
        "status": "OK",
        "season": season,
        "rows_fetched": int(len(bounded)),
        "rows_written": written,
        "games": int(bounded["GAME_ID"].nunique()),
        "players": int(bounded["PLAYER_ID"].nunique()),
        "newest_game_date": None if pd.isna(newest) else str(newest.date()),
        "rows_after_slate_held_back": held_back,
    }
    logger.info(
        "Player logs refreshed: %d row(s) upserted for %s, newest game %s%s",
        written, season, out["newest_game_date"],
        f", {held_back} row(s) after {slate_date} held back" if held_back else "",
    )
    return out


def panel_freshness(panel: pd.DataFrame, slate_date: str) -> dict[str, Any]:
    """
    How far the newest COMPLETED game trails the slate, and whether to say so.

    THE REFRESH ABOVE IS NOT ENOUGH ON ITS OWN. It can fail, be switched off,
    or succeed against a season that has not started, and in each case the run
    proceeds on an old panel. This measures the consequence from the panel
    itself rather than trusting the step that was supposed to fix it — the
    only reading that cannot be fooled by a refresh which reported success and
    fetched nothing.

    MUST BE CALLED BEFORE ``attach_forward_slate``. Forward rows are dated ON
    the slate and carry no box score, so a panel measured after they are
    attached always looks perfectly fresh. That is the one ordering mistake
    that would turn this check into decoration.
    """
    if panel is None or panel.empty:
        return {"status": "EMPTY", "lag_days": None, "newest_game_date": None}

    dates = pd.to_datetime(panel.get("GAME_DATE"), errors="coerce")
    newest = dates.max()
    if pd.isna(newest):
        return {"status": "UNDATED", "lag_days": None, "newest_game_date": None}

    lag = int((pd.Timestamp(slate_date).normalize() - newest.normalize()).days)
    limit = _int_env(ENV_MAX_PANEL_LAG, DEFAULT_MAX_PANEL_LAG_DAYS, low=0, high=400)
    stale = lag > limit
    out = {
        "status": "STALE" if stale else "OK",
        "newest_game_date": str(newest.date()),
        "lag_days": lag,
        "max_lag_days": limit,
        "rows": int(len(panel)),
    }
    if stale:
        logger.warning(
            "PANEL IS STALE: the newest completed game is %s, %d day(s) before "
            "the %s slate, over the %d-day limit. Every rolling feature is "
            "that far out of date. A refresh that failed or is switched off is "
            "the usual cause; %s",
            out["newest_game_date"], lag, slate_date, limit,
            "see player_log_refresh in this run's summary.",
        )
    else:
        logger.info(
            "Panel freshness: newest completed game %s, %d day(s) before the "
            "slate.", out["newest_game_date"], lag,
        )
    return out


def ingest_prop_lines(guideline: dict[str, Any] | None, persist: bool = True) -> pd.DataFrame:
    """
    Pull posted NBA prop lines from PropLine.

    An absent source is not a pipeline failure — it is the same "no lines
    available" state the off-season produces, and downstream stages already
    abstain on it. Crashing here would take out feature building, scoring
    and persistence for a source that only supplies optional prop lines.
    """
    try:
        from src.ingestion.propline import pull_nba_prop_lines
    except ImportError:
        logger.warning(
            "Prop-line ingest skipped: src/ingestion/propline.py is not present. "
            "Downstream stages will abstain on prop lines; see docs/DATA_GAPS.md."
        )
        return pd.DataFrame()

    books = None
    if guideline and "approved_prop_sources" in guideline:
        books = list(guideline["approved_prop_sources"])
        logger.info("Prop books from master guideline: %s", books)

    snapshots = pull_nba_prop_lines(bookmakers=books)
    rows: list[dict[str, Any]] = []
    for snap in snapshots:
        if snap.status != "VALID":
            logger.warning("Prop source %s: %s", snap.source, snap.message or snap.status)
            continue
        for line in snap.lines:
            rows.append({
                "source": line.source,
                # Per ROW, never hardcoded. This used to be a literal True,
                # which made market_ev_gate abstain on every prop including
                # genuine two-way sportsbook prices — the gate would have
                # refused the very data it exists to evaluate.
                "is_pickem": line.is_pickem,
                # The source's own observation time, or NULL when it did not
                # report one. Never now(): see PropLineSnapshot.captured_at_utc.
                "captured_at_utc": line.captured_at_utc,
                "player_name": line.player_name,
                "nba_player_id": line.nba_player_id,
                "market": line.market,
                "line": line.line,
                "over_odds_american": line.over_odds_american,
                "under_odds_american": line.under_odds_american,
                "payout_multiplier": line.payout_multiplier,
                "game_date": line.game_date,
                "nba_game_id": line.nba_game_id,
                "status": line.status,
                "raw_json": line.raw,
            })

    df = pd.DataFrame(rows)
    logger.info("Prop lines captured: %d rows from %d source(s)", len(df), len(snapshots))
    if persist and not df.empty:
        from src.db.repository import insert_prop_snapshots

        insert_prop_snapshots(df)
    return df


# ---------------------------------------------------------------------------
# [5][6] features + FATIGUE VERIFICATION (not re-application)
# ---------------------------------------------------------------------------

def _filter_to_slate(features: pd.DataFrame, slate: str) -> tuple[pd.DataFrame, int]:
    """
    Keep only the rows whose game falls on the requested Pacific slate date.

    Called AFTER feature-building, never before: the shifted rolling windows
    need the surrounding history to read, but that history must not be
    projected or persisted as though it were today's work.

    A missing GAME_DATE column is treated as unfilterable rather than as an
    empty slate — dropping every row on a schema surprise would look exactly
    like a quiet night.
    """
    if "GAME_DATE" not in features.columns:
        logger.warning(
            "No GAME_DATE column — cannot filter to slate %s, so every panel row "
            "would be scored. Refusing to guess; returning the frame unfiltered "
            "for the caller to reject.",
            slate,
        )
        return features, len(features)

    try:
        target = pd.Timestamp(slate).normalize()
    except (TypeError, ValueError):
        logger.warning("Unparseable slate date %r — not filtering", slate)
        return features, len(features)

    dates = pd.to_datetime(features["GAME_DATE"], errors="coerce").dt.normalize()
    on_slate = features.loc[dates == target]
    logger.info(
        "Slate filter: %d of %d panel rows fall on %s; the rest are history "
        "feeding the rolling features.",
        len(on_slate), len(features), slate,
    )
    return on_slate, len(on_slate)


def build_features_and_verify_fatigue(
    player_panel: pd.DataFrame,
    *,
    team_games: pd.DataFrame | None = None,
    market_lines: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build the leakage-safe feature matrix and VERIFY fatigue was applied.

    build_feature_matrix already calls attach_fatigue_column internally and
    folds `fatigue_multiplier` into every `{stat}_L2`. Calling it again
    would double-apply the multiplier, so this stage asserts rather than
    re-applies — while still failing loudly if fatigue is ever silently
    dropped from the builder.
    """
    from src.features.builder import assert_no_lookahead, build_feature_matrix

    features = build_feature_matrix(
        player_panel, team_games=team_games, market_lines=market_lines
    )
    # EMPTY counts as absent, because that is how the builder treats it:
    # build_feature_matrix gates both layers on `is not None and not .empty`,
    # so a workbook that parsed to zero rows skips them exactly as a missing
    # frame does. Warning only on None left that case silent, which is the
    # defect this function exists to fix.
    def _absent(frame: pd.DataFrame | None) -> bool:
        return frame is None or frame.empty

    if _absent(team_games) or _absent(market_lines):
        # NAMES THE FRAMES, not a fixed list of columns: build_feature_matrix
        # gates team-strength/defence on team_games and market context on
        # market_lines INDEPENDENTLY, so one can be present while the other is
        # not. The earlier wording claimed all four groups went missing whenever
        # either frame did.
        logger.warning(
            "Workbook frames missing or EMPTY (team_games=%s, market_lines=%s) — "
            "the features that depend on whichever frame is unavailable will be "
            "ABSENT from this matrix (team Elo and opponent defence need "
            "team_games; market context and blowout need market_lines). The run "
            "is narrower, not wrong; supply the BigDataBall workbook to match "
            "what scripts/nba_model_cli.py builds.",
            "absent" if team_games is None else f"{len(team_games)} rows",
            "absent" if market_lines is None else f"{len(market_lines)} rows",
        )

    if FATIGUE_COL not in features.columns:
        raise RuntimeError(
            f"'{FATIGUE_COL}' missing from the feature matrix — fatigue was NOT "
            f"applied. build_feature_matrix is expected to call "
            f"features.fatigue_logic.attach_fatigue_column internally. Refusing to "
            f"continue with un-fatigued projections."
        )

    missing_flags = [c for c in FATIGUE_FLAG_COLS if c not in features.columns]
    if missing_flags:
        logger.warning("Fatigue applied but flag columns missing: %s", missing_flags)

    l2_cols = [c for c in features.columns if c.endswith("_L2")]
    if not l2_cols:
        raise RuntimeError(
            "No `{stat}_L2` columns found — layer-2 (fatigue/pace-adjusted) "
            "projections were not produced by build_feature_matrix."
        )

    assert_no_lookahead(features)

    logger.info(
        "Features: %d rows x %d cols | fatigue APPLIED (mean %s=%.4f, %d B2B rows) | L2: %s",
        len(features), features.shape[1], FATIGUE_COL, features[FATIGUE_COL].mean(),
        int(features.get("is_back_to_back", pd.Series(dtype=bool)).sum()), l2_cols,
    )
    return features


# ---------------------------------------------------------------------------
# [7] model scoring — real constructor signature
# ---------------------------------------------------------------------------

def score_prob_over(
    features: pd.DataFrame,
    prop_lines: pd.DataFrame,
    model_path: Path | None = None,
) -> pd.Series:
    """
    Score P(Over) with a PREVIOUSLY FITTED model.

    XGBoostPropPipeline requires feature_cols at construction and a fitted
    booster for predict_proba_over. Training is a separate offline job and this
    stage only scores, returning an all-null Series with a named reason when it
    cannot.

    ``model_path`` is resolved by ``resolve_model_artifact``, not defaulted
    here. The two paths USED TO NOT COINCIDE: train-stats writes
    ``xgboost_{MARKET}.json`` plus its ``.meta.json`` under ``artifacts_dir``
    from config/model_comparison.yaml, while this stage read
    ``models/xgb_prop_over.json`` — a directory that does not exist — so
    training a model and then running the pipeline still scored nothing, and
    the scheduled worker passes no ``--model`` at all. The resolver now finds
    the trained artifact; see its docstring for the order it tries.

    ``None`` means nothing was resolved, and the resolver has already said
    which paths it looked in.

    TRAINING IS A SEPARATE OFFLINE JOB, and its command needs all three of
    these options — a pointer naming only ``--market`` fails when followed::

        python scripts/nba_model_cli.py train-stats --market PTS \
            --start-date 2024-11-01 --end-date 2025-03-01

    That writes ``xgboost_PTS.json`` plus its ``.meta.json`` under
    ``data/external/model_runs/comparison/`` (``artifacts_dir`` in
    config/model_comparison.yaml). ``resolve_model_artifact`` looks there, so
    the artifact no longer has to be named by hand — but ``--model`` still wins
    when you want a specific one::

        python main.py --model data/external/model_runs/comparison/xgboost_PTS.json
    """
    null = pd.Series([None] * len(features), index=features.index, dtype="object")

    if prop_lines.empty:
        logger.info("P(Over) skipped: no prop lines (expected off-season).")
        return null
    if model_path is None:
        # The resolver already logged WHICH paths it tried; repeating its
        # reason here would be the second half of one message.
        logger.info("P(Over) skipped: no model artifact was resolved.")
        return null
    model_path = Path(model_path)
    if not model_path.exists():
        logger.warning(
            "P(Over) skipped: no fitted model at %s. This stage does not fit "
            "models. Train one with `scripts/nba_model_cli.py train-stats "
            "--market PTS --start-date ... --end-date ...`, which writes to "
            "data/external/model_runs/comparison/, then pass that artifact as "
            "--model — the two paths do not coincide.",
            model_path,
        )
        return null

    try:
        import json

        import xgboost as xgb

        from src.models.xgboost_pipeline import XGBoostPropPipeline

        meta_path = model_path.with_suffix(".meta.json")
        if not meta_path.exists():
            logger.warning(
                "P(Over) skipped: %s missing. The feature_cols used at training time "
                "must be persisted alongside the model — scoring with a different "
                "column order silently produces garbage.",
                meta_path,
            )
            return null

        meta = json.loads(meta_path.read_text())
        feature_cols = meta["feature_cols"]
        scored_market = meta.get("target_market")
        if not scored_market:
            logger.warning(
                "P(Over) skipped: %s has no target_market, so there is no way to tell "
                "which market these probabilities belong to.",
                meta_path,
            )
            return null
        missing = [c for c in feature_cols if c not in features.columns]
        if missing:
            logger.warning("P(Over) skipped: feature matrix missing trained columns: %s", missing[:10])
            return null

        pipeline = XGBoostPropPipeline(feature_cols)  # required positional arg
        booster = xgb.XGBClassifier()
        booster.load_model(str(model_path))

        # TRAIN/SERVE CONTRACT CHECK (audit finding R5). The booster carries the
        # column names it was trained on; nothing compared them to the sidecar's
        # feature_cols, so a .meta.json from one fit beside an artifact from
        # another was never identified as such.
        #
        # WHAT THIS ADDS, precisely: xgboost itself raises feature_names mismatch
        # on a permuted or short column list, so the broad except below already
        # abstained. It abstained with xgboost's internal message, which does not
        # say that two artifacts came from different fits. This names the
        # mismatch, the market and the fingerprint instead. It also catches a
        # sidecar that contradicts its own fingerprint, which no modelling
        # library can see — that is a property of the file.
        from src.models.feature_spec import verify_feature_contract

        try:
            booster_columns = booster.get_booster().feature_names
        except Exception:  # noqa: BLE001 — an artifact that cannot say is not a mismatch
            booster_columns = None
        spec, problem = verify_feature_contract(meta, booster_columns)
        if problem is not None:
            logger.warning(
                "P(Over) skipped: %s (market=%s fingerprint=%s)",
                problem, spec.market, spec.fingerprint(),
            )
            return null

        pipeline.model = booster

        raw = pd.Series(pipeline.predict_proba_over(features), index=features.index)
        # The classifier does not take the line as an input, so this number
        # is P(over) at the line its labels were built from. Writing it beside
        # a posted sportsbook line would present one line's probability as
        # another's. Rows scored at a different line abstain instead.
        scored, n_masked = mask_probabilities_at_unsupported_lines(
            raw, features, features.get(RESEARCH_LINE_COL, float("nan")),
            model_name=f"xgboost/{scored_market}",
        )
        usable = scored.notna().sum()
        logger.info(
            "P(Over) scored for %d of %d rows for market %s (mean %.4f); "
            "%d abstained on an unsupported line",
            usable, len(scored), scored_market,
            float(scored.mean()) if usable else float("nan"), n_masked,
        )
        scored.attrs["target_market"] = scored_market
        return scored

    except Exception as exc:  # noqa: BLE001
        logger.warning("P(Over) skipped: %s", exc)
        return null


def score_prob_over_by_market(
    features: pd.DataFrame,
    prop_lines: pd.DataFrame,
    *,
    explicit: Path | str | None = None,
    markets: tuple[str, ...] = DEFAULT_STATS,
) -> dict[str, pd.Series]:
    """
    One artifact per market, so every market can carry a probability.

    WHY THIS EXISTS. ``score_prob_over`` takes ONE ``model_path`` and
    ``assemble_projections`` writes its output only onto the market the
    artifact's sidecar names -- correctly, since copying a points model's
    P(Over) onto the rebound frame would publish one market's probability as
    another's. But the consequence was that a single run could never produce a
    probability for more than ONE market: AST, REB and FG3M rows came out
    null, and ``settlement.recorder`` skips a row with no probability, so
    three of the four markets were unrecordable and ungradeable no matter how
    many models had been trained. ``verify_wiring`` reported this as a WARN on
    every run.

    ``resolve_model_artifact`` already takes a ``market`` and globs
    ``xgboost_{MARKET}.json``; nothing called it that way. This does.

    AN EXPLICIT ARTIFACT STILL MEANS ONE MARKET. ``--model`` and
    ``$PROPIQ_MODEL`` name a specific file, and that file was fit for one
    market -- so it is resolved and scored ONCE, and the market comes from its
    own sidecar rather than from the loop. Looping an explicit path over every
    market would score the same booster four times and hand three of the
    results to markets it was not fit for, which is the exact mistake the
    sidecar check exists to prevent.

    A market with no artifact is SKIPPED, not filled. Its rows keep a null
    probability, which is what they already were.
    """
    if explicit:
        scored = score_prob_over(features, prop_lines, Path(explicit))
        market = scored.attrs.get("target_market")
        if not market:
            logger.info(
                "An explicit artifact was given but its market is unknown, so "
                "no market receives these probabilities."
            )
            return {}
        return {str(market): scored}

    out: dict[str, pd.Series] = {}
    missing: list[str] = []
    for market in markets:
        path, how = resolve_model_artifact(market=market)
        if path is None:
            missing.append(market)
            continue
        scored = score_prob_over(features, prop_lines, path)
        resolved_market = scored.attrs.get("target_market")
        if not resolved_market:
            continue
        if str(resolved_market).upper() != market.upper():
            # The glob is per market, so this should not happen -- but a
            # mislabelled sidecar would otherwise route one market's
            # probabilities to another, silently, which is the whole failure
            # this function is shaped around.
            logger.warning(
                "Artifact %s resolved for %s (via %s) but its sidecar says "
                "%s. Skipping it rather than attributing it to the wrong "
                "market.", path, market, how, resolved_market,
            )
            continue
        out[market.upper()] = scored

    if missing:
        logger.info(
            "No artifact for %s, so those rows carry no probability and the "
            "recorder will skip them. Train one per market: it is one file per "
            "market by design, not one model for all of them.", missing,
        )
    logger.info(
        "Scored %d of %d market(s): %s", len(out), len(markets), sorted(out)
    )
    return out


# ---------------------------------------------------------------------------
# [8] EV gate — call the real gate, record the verdict
# ---------------------------------------------------------------------------

#: What a projection row's MARKET_STATUS is when no prop line reached it at
#: all. Distinguished from a line the gate REFUSED, because "nobody posted a
#: price for this player" and "a price was posted and could not be de-vigged"
#: are different facts and a reader acting on either deserves the right one.
GATE_NO_LINE_REASON = (
    "No prop line was matched to this row, so there was no posted price for "
    "the gate to evaluate."
)


def evaluate_ev_gate(prop_lines: pd.DataFrame, game_markets: pd.DataFrame) -> dict[str, Any]:
    """
    Ask quant.contracts.market_ev_gate whether EV may be computed, PER ROW.

    It requires status == VALID, both over/under American odds, and a finite
    posted line. Pick'em multipliers are ROUTED rather than refused -- see
    ``contracts.PICKEM_ENTRY_ROUTE`` -- and BigDataBall game lines are not
    player props at all, so an abstention is a common and correct output here.
    Recording it explicitly stops a consumer reading "no EV shown" as "no edge
    found".

    TWO DEFECTS THIS FUNCTION HAD, both of which made its answer meaningless
    rather than merely pessimistic:

    1. IT DID NOT PASS THE LINE. ``MarketContext.line`` defaulted to None and
       this loop never set it, while ``market_ev_gate`` refuses a context with
       no finite line -- "EV is a claim about a probability at a specific
       number". So every row abstained for the one reason that was this
       function's own doing, and a perfect two-way price would have abstained
       too. ``market``, ``player_name``, ``is_pickem`` and
       ``payout_multiplier`` were dropped on the same floor, so pick'em rows
       were never routed either.

    2. IT RETURNED ONE STATUS FOR THE WHOLE SLATE. The loop counts ready and
       abstained per row and then collapses them, and
       ``assemble_projections`` wrote that single value onto EVERY row, where
       ``repository.persist_projections`` stores it per row as
       ``market_status``. One ready prop out of three hundred therefore
       labelled all three hundred READY_FOR_EVALUATION. The gate's question is
       about ONE market -- it reads one context -- so a per-row answer is the
       only kind it has; the aggregate is a summary for a log line.

    ``by_key`` is the per-row answer, keyed on ``(player_name, market)`` --
    the same key ``_attach_prop_lines`` joins LINE on, so a row that got a
    line gets that line's verdict and cannot be handed another row's.
    """
    from src.quant.contracts import MarketContext, market_ev_gate

    empty: dict[str, Any] = {
        "status": "DATA_NOT_AVAILABLE",
        "reason": "No prop lines captured.",
        "ready": 0,
        "abstained": 0,
        "by_key": {},
    }
    if prop_lines.empty:
        return empty

    ready = abstained = 0
    first_reason: str | None = None
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for _, row in prop_lines.iterrows():
        ctx = MarketContext(
            game_id=str(row.get("nba_game_id") or "unknown"),
            market=row.get("market"),
            player_name=row.get("player_name"),
            line=row.get("line"),
            source=row.get("source"),
            captured_at_utc=row.get("captured_at_utc"),
            over_odds_american=row.get("over_odds_american"),
            under_odds_american=row.get("under_odds_american"),
            payout_multiplier=row.get("payout_multiplier"),
            is_pickem=bool(row.get("is_pickem") or False),
            status=row.get("status", "DATA_NOT_AVAILABLE"),
        )
        verdict = market_ev_gate(ctx)
        if verdict["status"] == "READY_FOR_EVALUATION":
            ready += 1
        else:
            abstained += 1
            first_reason = first_reason or verdict.get("reason")

        name, market = row.get("player_name"), row.get("market")
        if name and market:
            # Last write wins, matching _attach_prop_lines' own
            # drop_duplicates(keep="last"): the freshest capture for a
            # (player, market) is the one whose line is joined, so it must
            # also be the one whose verdict is.
            by_key[(str(name), str(market))] = {
                "status": verdict["status"],
                "reason": verdict.get("reason"),
                "route": verdict.get("route"),
            }

    logger.info("EV gate: %d ready, %d abstained.%s", ready, abstained,
                f" Reason: {first_reason}" if abstained else "")
    if ready == 0:
        logger.info(
            "No prop EV computed. BigDataBall supplies GAME spread/total/ML, not "
            "two-way player-prop American odds — an approved two-way "
            "player-prop feed (PropLine) is required to satisfy the gate."
        )
    return {
        "status": "READY_FOR_EVALUATION" if ready else "DATA_NOT_AVAILABLE",
        "ready": ready,
        "abstained": abstained,
        "reason": first_reason,
        "by_key": by_key,
    }


# ---------------------------------------------------------------------------
# assemble projections
# ---------------------------------------------------------------------------

def _fatigue_notes(features: pd.DataFrame) -> pd.Series:
    """
    Build a human-readable fatigue note per row from the real flag columns
    emitted by fatigue_logic (is_4_in_5 / is_3_in_4 / is_back_to_back).

    Mirrors the precedence in assess_schedule_density: 4-in-5 > 3-in-4 >
    B2B. Previously persist_projections read a FATIGUE_NOTES key the
    assembler never produced, so the column was always NULL.
    """
    notes = pd.Series(["normal rest"] * len(features), index=features.index, dtype="object")
    if "is_back_to_back" in features.columns:
        notes = notes.mask(features["is_back_to_back"].fillna(False), "back-to-back")
    if "is_3_in_4" in features.columns:
        notes = notes.mask(features["is_3_in_4"].fillna(False), "3-in-4 density")
    if "is_4_in_5" in features.columns:
        notes = notes.mask(features["is_4_in_5"].fillna(False), "4-in-5 density")
    return notes


def assemble_projections(
    features: pd.DataFrame,
    prob_over: "pd.Series | dict[str, pd.Series]",
    ev_verdict: dict[str, Any],
    prop_lines: pd.DataFrame | None = None,
    stats: tuple[str, ...] = DEFAULT_STATS,
) -> pd.DataFrame:
    """
    Long-format projections: one row per (player, game, stat).

    Uses the REAL column names — `{stat}_BASELINE` (layer 1) and
    `{stat}_L2` (layer 2, already fatigue- and pace-adjusted by the
    builder). No extra multiplication here: doing so would apply fatigue
    twice.

    `prop_lines` is joined on EXACT (player_name, market) to populate
    LINE. Deliberately exact-match only: PropIQ has a dedicated fuzzy
    crosswalk (`ingestion/id_crosswalk.py`, rapidfuzz with soft-miss
    handling) and duplicating a naive fuzzy match here would risk
    mis-joining one player's line onto another's projection. Unmatched
    rows keep LINE = None, which is honest — no line was verified for
    that projection.

    PROB_OVER is written ONLY for the market the scoring model was trained
    for, taken from its artifact metadata. Copying one model's probability
    into every market frame published a points model's P(Over) as the
    rebound, assist and threes probability too.

    ``prob_over`` IS A MAPPING OF MARKET -> SERIES, and a bare Series is still
    accepted. The mapping is what lets more than one market carry a
    probability in a single run: with a Series, whichever single market its
    ``attrs["target_market"]`` names is the only one that can, and every other
    market's rows come out null -- which the recorder then skips. See
    ``score_prob_over_by_market``. A Series keeps working because the tests
    and ``scripts/verify_wiring`` pass one, and because an explicit --model is
    genuinely one artifact for one market.

    MARKET_STATUS IS PER ROW, from ``ev_verdict["by_key"]``, joined on the
    same ``(player_name, market)`` key as LINE. It used to be the slate-wide
    aggregate written onto every row, so one ready prop out of three hundred
    labelled all three hundred READY_FOR_EVALUATION.
    """
    if isinstance(prob_over, dict):
        by_market: dict[str, pd.Series] = {
            str(k).upper(): v for k, v in prob_over.items()
        }
    else:
        scored_market = (
            prob_over.attrs.get("target_market")
            if hasattr(prob_over, "attrs") else None
        )
        if scored_market is None and prob_over.notna().any():
            logger.warning(
                "P(Over) has no target market attached — leaving PROB_OVER null "
                "rather than attributing one market's probabilities to all of them."
            )
            by_market = {}
        elif scored_market is None:
            by_market = {}
        else:
            by_market = {str(scored_market).upper(): prob_over}

    gate_by_key: dict[tuple[str, str], dict[str, Any]] = (
        ev_verdict.get("by_key") or {}
    )

    frames = []
    for stat in stats:
        base_col, l2_col = f"{stat}_BASELINE", f"{stat}_L2"
        if l2_col not in features.columns:
            logger.debug("Skipping %s: no %s column", stat, l2_col)
            continue
        frames.append(pd.DataFrame({
            "PLAYER_ID": features.get("PLAYER_ID"),
            "PLAYER_NAME": features.get("PLAYER_NAME"),
            "GAME_ID": features.get("GAME_ID"),
            "GAME_DATE": features.get("GAME_DATE"),
            "MARKET": stat,
            "BASELINE": features.get(base_col),
            "FINAL_PROJECTION": features[l2_col],   # already fatigue-adjusted
            "FATIGUE_MULTIPLIER": features[FATIGUE_COL],
            "FATIGUE_NOTES": _fatigue_notes(features),
            "PROB_OVER": (
                by_market[stat.upper()].values
                if stat.upper() in by_market
                else [None] * len(features)
            ),
        }))

    if not frames:
        logger.warning("No `{stat}_L2` columns matched %s — no projections assembled.", stats)
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out["LINE"] = _attach_prop_lines(out, prop_lines)
    out["MARKET_STATUS"], out["MARKET_STATUS_REASON"] = _market_status_per_row(
        out, gate_by_key
    )
    # Assigned HERE rather than inside the helper so the keys are visible at
    # the assembly site: tests/test_projection_roundtrip.py reads this function
    # to check that every key persist_projections reads is actually produced,
    # and a key written inside a helper is a key that test cannot see. That is
    # the same guard that caught BASELINE_PROJECTION and FATIGUE_NOTES
    # persisting as silent NULLs, so it is worth keeping legible.
    under, push, reason = _under_push_and_reason(out)
    out["PROB_UNDER"] = under
    out["PROB_PUSH"] = push
    out["NOTES"] = reason

    matched = int(out["LINE"].notna().sum())
    scored_rows = int(pd.to_numeric(out["PROB_OVER"], errors="coerce").notna().sum())
    ready_rows = int((out["MARKET_STATUS"] == "READY_FOR_EVALUATION").sum())
    logger.info(
        "Assembled %d projection rows across %d markets (%d with a matched prop "
        "line, %d with a probability from %s, %d READY_FOR_EVALUATION per row)",
        len(out), len(frames), matched, scored_rows,
        sorted(by_market) or "no market", ready_rows,
    )
    return out


def _market_status_per_row(
    projections: pd.DataFrame,
    gate_by_key: dict[tuple[str, str], dict[str, Any]],
) -> tuple[pd.Series, pd.Series]:
    """
    Each row's OWN gate verdict, and the reason behind it.

    THE DEFECT THIS REPLACES. ``MARKET_STATUS`` was ``ev_verdict["status"]`` --
    one slate-wide value, assigned to every row, and persisted per row by
    ``repository.persist_projections`` as ``market_status``. A slate with one
    priced prop and two hundred and ninety-nine unpriced ones labelled all
    three hundred READY_FOR_EVALUATION. Anything reading the column row by row
    -- which is the only way it is stored -- was reading a claim about a
    different row.

    THREE OUTCOMES, and the distinction between the last two is the point:

      * the gate's own verdict, where a prop line was matched to this row;
      * DATA_NOT_AVAILABLE because NO LINE reached this row at all;
      * DATA_NOT_AVAILABLE because a line WAS posted and the gate refused it.

    "Nobody posted a price for this player" and "a price was posted and could
    not be de-vigged" are different facts. Collapsing them is how a reader
    concludes a market is unavailable when it is actually unpriceable, or the
    reverse.
    """
    names = projections.get("PLAYER_NAME")
    markets = projections.get("MARKET")
    statuses: list[str] = []
    reasons: list[str | None] = []
    for name, market in zip(names, markets):
        verdict = gate_by_key.get((str(name), str(market)))
        if verdict is None:
            statuses.append("DATA_NOT_AVAILABLE")
            reasons.append(GATE_NO_LINE_REASON)
            continue
        statuses.append(str(verdict.get("status") or "DATA_NOT_AVAILABLE"))
        reasons.append(verdict.get("reason"))
    return (
        pd.Series(statuses, index=projections.index, dtype="object"),
        pd.Series(reasons, index=projections.index, dtype="object"),
    )


def _under_push_and_reason(
    projections: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Return (P(under), P(push), reason) for each row. The nulls are the point.

    The classifier behind PROB_OVER is binary and does not model a push, so:

      * a HALF line cannot push -> PROB_PUSH is 0.0 and PROB_UNDER is exactly
        1 - PROB_OVER, which is lossless;
      * a WHOLE line can -> 1 - PROB_OVER is P(under OR push), not P(under),
        so both are left NULL and the refusal is returned as the reason;
      * an UNKNOWN line takes the whole-line branch, because a line nobody can
        see cannot be shown to be a half-line.

    Routed through ``paper_research.resolve_two_way_model_probs`` rather than
    reimplemented: that function already encodes this rule, is tested, and is
    what the decision board uses. A second copy here would be the one that
    drifts.
    """
    from src.quant.paper_research import resolve_two_way_model_probs

    index = projections.index
    if projections.empty:
        empty = pd.Series([], index=index, dtype="object")
        return empty.copy(), empty.copy(), empty.copy()

    unders: list[float | None] = []
    pushes: list[float | None] = []
    reasons: list[str | None] = []
    refused = 0

    for over, line in zip(projections["PROB_OVER"], projections["LINE"], strict=True):
        p_over = None if over is None or pd.isna(over) else float(over)
        p_line = None if line is None or pd.isna(line) else float(line)
        _po, p_under, p_push, warning = resolve_two_way_model_probs(
            p_over=p_over, line=p_line
        )
        unders.append(p_under)
        pushes.append(p_push)
        reasons.append(warning)
        # A row with no probability at all is not a refusal, it is an absence.
        if warning is not None and p_over is not None:
            refused += 1

    under_series = pd.Series(unders, index=index, dtype="object")
    if refused:
        logger.info(
            "P(under)/P(push): resolved on %d row(s); %d scored row(s) left null "
            "because the line is whole or unknown and a binary classifier has no "
            "push mass to split out. 1 - P(over) is not the under there.",
            int(under_series.notna().sum()), refused,
        )
    else:
        logger.info(
            "P(under)/P(push): resolved on %d row(s).",
            int(under_series.notna().sum()),
        )
    return (
        under_series,
        pd.Series(pushes, index=index, dtype="object"),
        pd.Series(reasons, index=index, dtype="object"),
    )


def _attach_prop_lines(projections: pd.DataFrame, prop_lines: pd.DataFrame | None) -> pd.Series:
    """
    Exact-match join of captured prop lines onto projections.

    THE JOIN IS STILL EXACT. What changed is that the board's names are first
    rewritten to the panel's canonical form by
    ``ingestion.id_crosswalk.resolve_board_names``, which is deterministic:
    diacritics, dots, apostrophes and hyphens are normalised away so
    ``Nikola Jokic`` resolves to ``Nikola Jokić``, while ``Jalen Williams`` and
    ``Jaylen Williams`` stay two different people. A name that resolves to two
    players is refused rather than guessed. See that module for why a fuzzy
    score cutoff cannot do this job.

    Returns an all-None Series when no prop lines exist (correct
    off-season) rather than raising — but logs the unmatched count so a
    systematic name-format mismatch is visible rather than silent.
    """
    null = pd.Series([None] * len(projections), index=projections.index, dtype="object")

    if prop_lines is None or prop_lines.empty:
        return null
    if not {"player_name", "market", "line"}.issubset(prop_lines.columns):
        logger.warning(
            "Prop lines frame missing player_name/market/line — LINE left null. Columns: %s",
            list(prop_lines.columns),
        )
        return null

    from src.ingestion.id_crosswalk import apply_name_map, resolve_board_names

    name_map, _report = resolve_board_names(prop_lines, projections)
    resolved = apply_name_map(prop_lines, name_map)

    lookup = (
        resolved.dropna(subset=["player_name", "market"])
        .drop_duplicates(subset=["player_name", "market"], keep="last")
        .set_index(["player_name", "market"])["line"]
    )

    keys = list(zip(projections["PLAYER_NAME"], projections["MARKET"]))
    values = [lookup.get(k) for k in keys]
    result = pd.Series(values, index=projections.index, dtype="object")

    unmatched = int(result.isna().sum())
    if unmatched and unmatched == len(result):
        logger.warning(
            "NO prop lines matched any projection (%d rows), even after the name "
            "crosswalk. The board and the panel are describing different players "
            "or a different slate — not a name-format problem.",
            unmatched,
        )
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """Run one slate.

    ``argv`` defaults to the process arguments, so the command line is
    unchanged; the scheduled worker passes an explicit list instead, which is
    also what makes this callable from a test without touching sys.argv.
    """
    parser = argparse.ArgumentParser(description="PropIQ Analytics pipeline (NBA only)")
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help=f"Slate date YYYY-MM-DD in {DISPLAY_TZ_NAME} (default: today Pacific)",
    )
    parser.add_argument("--init-db", action="store_true", help="Create tables then exit")
    parser.add_argument("--bigdataball", type=str,
                        default=os.environ.get(
                            "BIGDATABALL_XLSX",
                            "data/external/bigdataball/2025-2026_NBA_Box_Score_Team-Stats__1_.xlsx"))
    parser.add_argument(
        "--model", type=str, default=None,
        help="Fitted artifact to score with. Omitted: resolve_model_artifact "
             f"tries ${ENV_MODEL}, then the comparison artifacts_dir, then "
             f"{MODEL_ARTIFACT_DEFAULT}.",
    )
    parser.add_argument("--no-db", action="store_true", help="Dry run, no persistence")
    args = parser.parse_args(argv)

    # Run id stamp uses Pacific wall clock so operators reading logs see LA time.
    run_id = f"run_{now_pacific().strftime('%Y%m%dT%H%M%S%z')}_{uuid.uuid4().hex[:6]}"
    logger.info(
        "=== PropIQ pipeline %s (NBA only, RESEARCH_ONLY, display_tz=%s) ===",
        run_id,
        DISPLAY_TZ_NAME,
    )

    if args.init_db:
        from src.db.session import init_db

        init_db()
        logger.info("Tables created. Exiting.")
        return 0

    persist = not args.no_db
    stage_summary: dict[str, Any] = {}

    try:
        stage_summary["preflight"] = preflight(require_db=persist)
        guideline = load_master_guideline()

        team_games_df, market_df = ingest_market_lines(
            Path(args.bigdataball), persist=persist
        )
        stage_summary["game_markets"] = {
            "rows": len(market_df),
            "valid": int((market_df["status"] == "VALID").sum()),
            "note": "GAME spread/total/ML only — does not satisfy prop EV gate",
        }

        prop_df = ingest_prop_lines(guideline, persist=persist)
        stage_summary["prop_lines"] = {"rows": len(prop_df)}

        from src.db.repository import load_player_panel

        slate_pt = args.date or str(pacific_calendar_date())

        # [3b] REFRESH THE PANEL'S SOURCE BEFORE READING IT. Until 2026-10-09
        # this step did not exist: load_player_panel below read whatever had
        # last been ingested by hand, so a deployed container projected from a
        # frozen panel and nothing said so.
        stage_summary["player_log_refresh"] = refresh_player_logs(
            slate_pt, persist=persist
        )

        panel = load_player_panel(slate_date=slate_pt)

        # MEASURED FROM THE PANEL, AND BEFORE attach_forward_slate. Forward
        # rows are dated on the slate and carry no box score, so freshness
        # taken after they are attached always reads as perfect. This is also
        # the check that catches a refresh which reported success and fetched
        # nothing.
        stage_summary["panel_freshness"] = panel_freshness(panel, slate_pt)

        # ROWS FOR GAMES THAT HAVE NOT BEEN PLAYED (readiness item O1).
        #
        # The panel above is COMPLETED box scores, and the slate filter further
        # down keeps only rows dated on the slate. For a future slate that
        # intersection is empty, so every scheduled run exited 0 with
        # success_no_data and nothing looked wrong. The forward rows carry the
        # identity the feature builder needs and NO box-score stat, so the
        # rolling features read each player's own prior real games and the
        # forward row has nothing of its own to leak.
        #
        # Best-effort: a denied or unreachable schedule leaves the panel
        # untouched and names the reason, so the run degrades to exactly what it
        # did before rather than failing.
        if _flag_env(ENV_FORWARD_SLATE, True):
            from src.pipeline.forward_slate import attach_forward_slate

            forward = attach_forward_slate(panel, slate_date=slate_pt)
            panel = forward.panel
            stage_summary["forward_slate"] = forward.as_dict()
        else:
            logger.info(
                "%s is off — projecting only games already in the panel, which "
                "for a future slate is none.", ENV_FORWARD_SLATE,
            )
            stage_summary["forward_slate"] = {"status": "SKIPPED"}

        if panel.empty:
            logger.warning(
                "Player panel EMPTY for %s (%s) — nothing to project. Expected off-season "
                "or before box scores are ingested.",
                slate_pt,
                DISPLAY_TZ_NAME,
            )
            stage_summary["projections"] = {"rows": 0, "reason": "empty player panel"}
            if persist:
                from src.db.repository import record_run

                record_run(run_id, status="success_no_data", stage_summary=stage_summary)
            return 0

        features = build_features_and_verify_fatigue(
            panel, team_games=team_games_df, market_lines=market_df
        )

        # The panel deliberately carries ~400 days so the shift-1 rolling
        # features have history to read. Those historical rows are INPUT,
        # not output: projecting and persisting them turns a request for one
        # slate into a retrospective projection of the whole lookback. Filter
        # after feature-building so the history is used but not scored.
        slate = slate_pt
        features, slate_rows = _filter_to_slate(features, slate)
        stage_summary["slate_filter"] = {
            "slate_pt": slate,
            "panel_rows": len(panel),
            "slate_rows": slate_rows,
        }
        if features.empty:
            # Not the same condition as an empty panel, and the distinction
            # matters: the panel holds COMPLETED games, so a future slate is
            # legitimately absent from it and needs a schedule source, not
            # more box scores.
            logger.warning(
                "No rows for slate %s (%s) in a panel of %d rows. The player "
                "game log holds completed games, so a future slate will not "
                "appear here until those games are played and ingested.",
                slate, DISPLAY_TZ_NAME, len(panel),
            )
            stage_summary["projections"] = {"rows": 0, "reason": "no rows on the requested slate"}
            if persist:
                from src.db.repository import record_run

                record_run(run_id, status="success_no_data", stage_summary=stage_summary)
            return 0

        # ONE ARTIFACT PER MARKET. An explicit --model/$PROPIQ_MODEL is one
        # file fit for one market and is honoured as such; otherwise every
        # market resolves its own xgboost_{MARKET}.json, so a run is no longer
        # limited to a single market's probabilities. See
        # score_prob_over_by_market for why an explicit path is NOT looped.
        explicit = args.model or (os.environ.get(ENV_MODEL) or "").strip() or None
        if explicit:
            model_path, resolved_by = resolve_model_artifact(args.model)
            if model_path is None:
                logger.warning("P(Over) will be skipped: %s", resolved_by)
            else:
                logger.info(
                    "Scoring model: %s (resolved by %s) — one artifact, so one "
                    "market carries a probability this run.",
                    model_path, resolved_by,
                )
            stage_summary["model"] = {
                "path": str(model_path) if model_path else None,
                "resolved_by": resolved_by,
            }
        else:
            stage_summary["model"] = {"resolved_by": "per market"}

        prob_over = score_prob_over_by_market(
            features, prop_df, explicit=explicit, markets=DEFAULT_STATS
        )
        stage_summary["model"]["markets_scored"] = sorted(prob_over)
        ev_verdict = evaluate_ev_gate(prop_df, market_df)
        # by_key is one entry per priced (player, market) and is joined onto
        # the rows; it does not belong in a run summary that goes to the
        # database as JSON, where it would be hundreds of keys of duplication.
        stage_summary["ev_gate"] = {
            k: v for k, v in ev_verdict.items() if k != "by_key"
        }

        projections = assemble_projections(features, prob_over, ev_verdict, prop_lines=prop_df)
        stage_summary["projections"] = {"rows": len(projections)}

        # PRE-TIP SCRATCH FILTER (audit finding R4). The absence features are a
        # training-panel signal; nothing withheld a projection when a player was
        # ruled out AFTER it was written. ESPN's public injuries feed is the
        # reachable source — stats.nba.com is denied through this proxy.
        #
        # A FAILED CHECK LABELS, IT DOES NOT CLEAR. Rows come back UNVERIFIED and
        # are still persisted and still recorded; only OUT/DOUBTFUL is withheld.
        from src.pipeline.scratches import apply_scratch_filter

        scratches = apply_scratch_filter(projections)
        projections = scratches.projections
        stage_summary["scratch_filter"] = scratches.as_dict()
        if not scratches.verified:
            logger.warning(
                "Availability unverified (%s). Projections are labelled "
                "UNVERIFIED, not cleared.", scratches.reason,
            )

        if persist and not projections.empty:
            from src.db.repository import (
                persist_projections,
                record_pending_prop_results,
                record_run,
            )
            from src.settlement.recorder import pending_prop_result_rows

            persist_projections(projections, run_id=run_id)

            # The feedback loop's only writer. prop_results had a grader and a
            # metrics layer and nothing that ever inserted a row, so every P/L,
            # strike rate and CLV figure was an aggregate over zero rows.
            # These are PREDICTIONS recorded for forward grading, not wagers:
            # no stake is written and none can be.
            recorded = pending_prop_result_rows(
                projections, prop_df, run_id=run_id,
            )
            record_pending_prop_results(recorded.rows)
            stage_summary["prop_results"] = recorded.as_dict()

            record_run(run_id, status="success", stage_summary=stage_summary)

        logger.info("=== Pipeline complete: %d projections ===", len(projections))
        return 0

    except Exception as exc:  # noqa: BLE001
        logger.exception("Pipeline FAILED: %s", exc)
        if persist:
            try:
                from src.db.repository import record_run

                record_run(run_id, status="failed", stage_summary=stage_summary, error=str(exc))
            except Exception:  # noqa: BLE001
                logger.error("Could not record failed run to DB.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
