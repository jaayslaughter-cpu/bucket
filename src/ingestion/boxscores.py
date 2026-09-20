"""Player game-log ingestion from the NBA stats API.

This is the module that turns an empty ``player_game_logs`` table into the
panel every model reads. The BigDataBall workbook cannot do it: that
export is team-level, so its PTS/REB/AST are team totals.

SOURCE
    https://stats.nba.com/stats/leaguegamelog?PlayerOrTeam=P&Season=...

One request returns every player-game for a season, which is why this
pulls a season at a time rather than walking game ids. Results are cached
to Parquet so a re-run costs nothing and the endpoint is hit once.

NETWORK
    stats.nba.com bot-detects on missing headers — a bare requests.get is
    blocked, so browser-like headers are mandatory. Some sandboxed
    environments deny nba.com outright at the proxy; in that case this
    raises BoxScoreFetchError naming the cause rather than returning an
    empty frame that would look like "no games played".

IDs are the same zero-padded 10-char NBA game ids used by
``src/settlement/boxscore_fetcher.py`` and by the BigDataBall workbook, so
the three sources join without a crosswalk.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

LEAGUE_GAME_LOG_URL = "https://stats.nba.com/stats/leaguegamelog"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
}

# Columns kept from the endpoint, renamed to the panel's contract.
COLUMN_MAP = {
    "PLAYER_ID": "PLAYER_ID",
    "PLAYER_NAME": "PLAYER_NAME",
    "GAME_ID": "GAME_ID",
    "GAME_DATE": "GAME_DATE",
    "TEAM_ABBREVIATION": "TEAM_ABBREVIATION",
    "MIN": "MIN",
    "PTS": "PTS",
    "REB": "REB",
    "AST": "AST",
    "FG3M": "FG3M",
    "FG3A": "FG3A",
    "STL": "STL",
    "BLK": "BLK",
    "TOV": "TOV",
}


class BoxScoreFetchError(RuntimeError):
    """Raised when logs cannot be fetched. Never silently returns nothing."""


@dataclass
class BoxScoreLoadConfig:
    """How and what to pull. No credentials — this endpoint needs none."""

    seasons: tuple[str, ...] = ("2025-26",)
    season_type: str = "Regular Season"
    cache_dir: Path = field(default_factory=lambda: Path("data/external/player_logs"))
    use_cache: bool = True
    timeout: int = 30
    retry_attempts: int = 3
    retry_backoff: float = 2.0
    # stats.nba.com throttles aggressive callers.
    pause_between_seasons: float = 1.5


def _cache_path(config: BoxScoreLoadConfig, season: str) -> Path:
    slug = season.replace("/", "-")
    kind = config.season_type.replace(" ", "_").lower()
    return config.cache_dir / f"player_game_logs_{slug}_{kind}.parquet"


def _parse_matchup(matchup: str) -> tuple[str | None, bool | None]:
    """
    'HOU @ OKC' -> ('OKC', False);  'OKC vs. HOU' -> ('HOU', True).

    Returns (opponent, is_home). Neutral-site games are indistinguishable
    here and are corrected downstream from the workbook's VENUE column.
    """
    if not isinstance(matchup, str):
        return None, None
    if " @ " in matchup:
        return matchup.split(" @ ")[-1].strip(), False
    if " vs. " in matchup:
        return matchup.split(" vs. ")[-1].strip(), True
    return None, None


def fetch_season_player_logs(
    season: str,
    config: BoxScoreLoadConfig | None = None,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch one season of player game logs. Raises rather than returning empty."""
    config = config or BoxScoreLoadConfig()
    params = {
        "Counter": "0",
        "Direction": "ASC",
        "LeagueID": "00",
        "PlayerOrTeam": "P",
        "Season": season,
        "SeasonType": config.season_type,
        "Sorter": "DATE",
    }
    sess = session or requests.Session()
    last_exc: Exception | None = None

    for attempt in range(1, config.retry_attempts + 1):
        try:
            response = sess.get(
                LEAGUE_GAME_LOG_URL,
                params=params,
                headers=HEADERS,
                timeout=config.timeout,
            )
            response.raise_for_status()
            return parse_league_game_log(response.json(), season=season)
        except Exception as exc:  # noqa: BLE001 — retried, then re-raised named
            last_exc = exc
            if attempt < config.retry_attempts:
                wait = config.retry_backoff ** attempt
                logger.warning(
                    "player logs %s attempt %d/%d failed (%s); retrying in %.1fs",
                    season, attempt, config.retry_attempts, exc, wait,
                )
                time.sleep(wait)

    raise BoxScoreFetchError(
        f"Could not fetch player game logs for {season}: {last_exc}. "
        "If this is a proxy/policy denial for nba.com, run the ingest on a "
        "machine with direct network access. Refusing to return an empty "
        "frame, which would look like a season with no games."
    ) from last_exc


def parse_league_game_log(payload: dict[str, Any], *, season: str) -> pd.DataFrame:
    """Turn the endpoint's headers/rowSet payload into the panel contract."""
    result_sets = payload.get("resultSets") or []
    if not result_sets:
        raise BoxScoreFetchError(f"No resultSets in the {season} payload")

    block = result_sets[0]
    frame = pd.DataFrame(block.get("rowSet") or [], columns=block.get("headers") or [])
    if frame.empty:
        raise BoxScoreFetchError(f"{season} returned zero player-game rows")

    missing = [c for c in COLUMN_MAP if c not in frame.columns]
    if missing:
        raise BoxScoreFetchError(f"{season} payload missing expected columns {missing}")

    out = frame[list(COLUMN_MAP)].rename(columns=COLUMN_MAP).copy()

    # Game ids are zero-padded strings; JSON may hand them back unpadded.
    out["GAME_ID"] = out["GAME_ID"].astype(str).str.strip().str.zfill(10)
    out["PLAYER_ID"] = out["PLAYER_ID"].astype(str).str.strip()
    out["GAME_DATE"] = pd.to_datetime(out["GAME_DATE"], errors="coerce")
    out["SEASON"] = season

    opponents, is_home = zip(*frame["MATCHUP"].map(_parse_matchup))
    out["OPPONENT_ABBREVIATION"] = opponents
    out["IS_HOME"] = is_home

    for col in ("MIN", "PTS", "REB", "AST", "FG3M", "FG3A", "STL", "BLK", "TOV"):
        out[col] = pd.to_numeric(out[col], errors="coerce")

    unparsed = int(out["OPPONENT_ABBREVIATION"].isna().sum())
    if unparsed:
        logger.warning("%d %s rows had an unparseable MATCHUP string", unparsed, season)

    logger.info(
        "Parsed %s: %d player-game rows, %d players, %d games (%s to %s)",
        season, len(out), out["PLAYER_ID"].nunique(), out["GAME_ID"].nunique(),
        out["GAME_DATE"].min().date(), out["GAME_DATE"].max().date(),
    )
    return out


def load_player_game_logs(config: BoxScoreLoadConfig | None = None) -> pd.DataFrame:
    """
    Load player game logs for every configured season, cache-first.

    This is what ``scripts/nba_model_cli.py`` calls for the real (non-demo)
    data path.
    """
    config = config or BoxScoreLoadConfig()
    session = requests.Session()
    frames: list[pd.DataFrame] = []

    for i, season in enumerate(config.seasons):
        cache = _cache_path(config, season)
        if config.use_cache and cache.exists():
            logger.info("Using cached player logs for %s (%s)", season, cache)
            frames.append(pd.read_parquet(cache))
            continue

        if i:
            time.sleep(config.pause_between_seasons)
        season_frame = fetch_season_player_logs(season, config, session=session)

        cache.parent.mkdir(parents=True, exist_ok=True)
        season_frame.to_parquet(cache, index=False)
        logger.info("Cached %d rows to %s", len(season_frame), cache)
        frames.append(season_frame)

    panel = pd.concat(frames, ignore_index=True).sort_values(["PLAYER_ID", "GAME_DATE"])

    before = len(panel)
    panel = panel.drop_duplicates(subset=["PLAYER_ID", "GAME_ID"], keep="first")
    if before != len(panel):
        logger.warning("Dropped %d duplicate player-game rows", before - len(panel))

    return panel.reset_index(drop=True)
