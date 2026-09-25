"""
src/ingestion/nba_playbyplay.py — play-by-play event logs from the NBA CDN.

    https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{GAME_ID}.json

THE GAP THIS CLOSES. Every event log in this project arrived as a CSV someone
uploaded by hand — 41 parts, three million events, four seasons. Nothing
fetched any of it. The box-score side has had a CDN fetcher for some time
(src/settlement/boxscore_fetcher.py); this is its sibling on the same host,
with the same headers, timeout and backoff, so the two behave identically
against the same bot detection.

THE PAYLOAD IS ALREADY THE RIGHT SHAPE. The CDN returns one dict per action
under ``game.actions``, keyed exactly as src/features/pbp.py expects:
actionNumber, clock ("PT11M58.00S"), period, actionType, subType, personId,
possession, scoreHome/scoreAway, shotDistance, shotResult, assistPersonId.
Those are the same names the uploaded CSVs carry, because the CSVs were
flattened from this endpoint. A fetched frame therefore feeds
``prepare_events`` with no translation layer, and
``check_log_completeness`` still audits it against the box score afterwards.

WHAT THIS REFUSES TO DO. A game that returns no actions raises rather than
yielding an empty frame: a silent empty is indistinguishable from a game
where nothing happened, and it would pass straight through the feature
builder as a player with no shots. A partial multi-game fetch returns what it
got AND the list of failures, so a caller can tell 1,200 games from 1,190.

RESEARCH ONLY. Fetches public event data; places no bets.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Iterable, Sequence

import pandas as pd
import requests

from src.settlement.boxscore_fetcher import (
    HEADERS,
    REQUEST_TIMEOUT,
    RETRY_ATTEMPTS,
    RETRY_BACKOFF,
    normalize_game_id,
)

logger = logging.getLogger(__name__)

SOURCE_NAME = "nba_cdn_playbyplay"

CDN_PLAYBYPLAY_URL = (
    "https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{game_id}.json"
)

# Seconds between games in a multi-game pull. The CDN is a static file host and
# tolerates sequential reads, but hammering it is both rude and the fastest way
# to earn a block.
DEFAULT_PAUSE_SECONDS = 0.6

# The columns src/features/pbp.py reads. Listed so a payload that silently
# stops carrying one is caught here rather than surfacing as a feature that
# quietly became all-NaN.
# personId, shotDistance, shotResult and possession were EXPECTED (absent ->
# filled with pd.NA). The rationale for that tier -- "a game with no
# three-pointers carries no shotValue" -- does not cover them: no real NBA game
# has zero player actions or zero field-goal attempts. Absent, personId yields
# no player rows at all, shotDistance/shotResult make every shot metric NaN,
# and possession makes PBP_GAME_PACE NaN. Each is a shipped feature quietly
# becoming empty, which is exactly what this list exists to prevent.
REQUIRED_ACTION_FIELDS: tuple[str, ...] = (
    "actionNumber", "clock", "period", "actionType",
    "personId", "shotDistance", "shotResult", "possession",
)

# Everything else pbp.py or the completeness check uses when present.
EXPECTED_ACTION_FIELDS: tuple[str, ...] = (
    "orderNumber", "subType", "playerName", "playerNameI",
    "teamId", "teamTricode", "scoreHome", "scoreAway",
    "shotValue", "isFieldGoal",
    "assistPersonId", "area", "areaDetail", "x", "y", "description",
    "periodType", "timeActual",
)


class PlayByPlayError(RuntimeError):
    """Raised when an event log cannot be fetched or is unusable."""


def fetch_playbyplay(
    game_id: str | int, session: requests.Session | None = None
) -> dict[str, Any]:
    """
    Fetch one game's raw CDN payload. Raises on failure — never returns a stub.

    Mirrors fetch_boxscore's retry and error handling deliberately: the two
    hit the same host and the same bot detection, and a 403 here means what it
    means there.
    """
    gid = normalize_game_id(game_id)
    url = CDN_PLAYBYPLAY_URL.format(game_id=gid)
    sess = session or requests.Session()
    last_exc: Exception | None = None

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            resp = sess.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 403:
                raise PlayByPlayError(
                    f"403 from the NBA CDN for {gid} — bot detection. Verify the "
                    "User-Agent/Referer headers are being sent, and that the "
                    "network policy allows cdn.nba.com."
                )
            if resp.status_code == 404:
                raise PlayByPlayError(
                    f"404 for game {gid} — the game id does not exist or its "
                    "play-by-play is not posted yet."
                )
            resp.raise_for_status()
            return resp.json()
        except PlayByPlayError:
            raise
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_BACKOFF * attempt)

    raise PlayByPlayError(
        f"Failed to fetch play-by-play for {gid} after {RETRY_ATTEMPTS} attempts"
    ) from last_exc


def parse_playbyplay_payload(payload: dict[str, Any]) -> pd.DataFrame:
    """
    Flatten ``game.actions`` into the frame src/features/pbp.py consumes.

    ``gameId`` is taken from the payload and stamped on every row, because the
    action dicts do not carry it and every downstream key — the dedupe key,
    the completeness check, the panel join — is (gameId, something).
    """
    game = (payload or {}).get("game") or {}
    actions = game.get("actions")
    if not isinstance(actions, list) or not actions:
        raise PlayByPlayError(
            "DATA_NOT_AVAILABLE: payload carries no game.actions. Returning an "
            "empty frame would be indistinguishable from a game in which "
            "nothing happened."
        )

    game_id = game.get("gameId") or (payload.get("meta") or {}).get("gameId")
    if not game_id:
        raise PlayByPlayError("DATA_NOT_AVAILABLE: payload has no game.gameId")

    frame = pd.DataFrame(actions)
    missing = [c for c in REQUIRED_ACTION_FIELDS if c not in frame.columns]
    if missing:
        raise PlayByPlayError(
            f"DATA_NOT_AVAILABLE: actions are missing {missing}. The CDN schema "
            "changed, or this is not a play-by-play payload."
        )

    frame.insert(0, "gameId", str(game_id))
    absent = [c for c in EXPECTED_ACTION_FIELDS if c not in frame.columns]
    if absent:
        # Not fatal: a game with no three-pointers carries no shotValue. The
        # columns are created empty so the frame's shape is stable across
        # games, which is what concatenating a season depends on.
        for col in absent:
            frame[col] = pd.NA
        logger.debug("game %s: %d expected field(s) absent, left null: %s",
                     game_id, len(absent), absent)
    return frame


def fetch_playbyplay_frame(
    game_id: str | int, session: requests.Session | None = None
) -> pd.DataFrame:
    """One game, fetched and flattened."""
    return parse_playbyplay_payload(fetch_playbyplay(game_id, session=session))


def fetch_many_playbyplay(
    game_ids: Iterable[str | int],
    *,
    session: requests.Session | None = None,
    pause_seconds: float = DEFAULT_PAUSE_SECONDS,
    stop_on_error: bool = False,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """
    Fetch several games. Returns (events, failures).

    Failures are RETURNED, not swallowed and not raised by default. A caller
    that gets 1,190 games back needs to know whether it asked for 1,190 or for
    1,200 — the difference is a season with a hole in it, and
    check_log_completeness will otherwise report it as a data problem rather
    than a fetch problem.
    """
    sess = session or requests.Session()
    frames: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []
    ids = [normalize_game_id(g) for g in game_ids]

    for i, gid in enumerate(ids):
        try:
            frames.append(fetch_playbyplay_frame(gid, session=sess))
        except PlayByPlayError as exc:
            if stop_on_error:
                raise
            failures.append({"game_id": gid, "error": str(exc)})
            logger.warning("play-by-play failed for %s: %s", gid, exc)
        if pause_seconds and i < len(ids) - 1:
            time.sleep(pause_seconds)

    if not frames:
        raise PlayByPlayError(
            f"DATA_NOT_AVAILABLE: none of the {len(ids)} requested game(s) "
            "returned play-by-play."
        )
    events = pd.concat(frames, ignore_index=True)
    # The same key the CSV ingest dedupes on, applied here so a retried or
    # overlapping pull cannot double-count an action.
    before = len(events)
    events = events.drop_duplicates(subset=["gameId", "actionNumber"], keep="first")
    logger.info(
        "play-by-play: %d game(s) fetched, %d failed, %d events (%d duplicate "
        "action(s) dropped)",
        len(frames), len(failures), len(events), before - len(events),
    )
    return events.reset_index(drop=True), failures


def game_ids_from_panel(
    panel: pd.DataFrame, *, seasons: Sequence[str] | None = None
) -> list[str]:
    """
    The game ids a panel needs event logs for, in chronological order.

    Ordered by date on purpose: a pull interrupted halfway then covers a
    contiguous span rather than a scatter, and a contiguous span is what the
    rolling features can actually use.
    """
    if "GAME_ID" not in panel.columns:
        raise PlayByPlayError("DATA_NOT_AVAILABLE: panel has no GAME_ID column")
    work = panel
    if seasons is not None and "SEASON" in work.columns:
        work = work[work["SEASON"].isin(list(seasons))]
    if "GAME_DATE" in work.columns:
        work = work.sort_values("GAME_DATE")
    ids = work["GAME_ID"].astype(str).drop_duplicates().tolist()
    return [normalize_game_id(g) for g in ids]
