"""
src/settlement/boxscore_fetcher.py — real NBA post-game box scores.

SCOPE: NBA only.

DATA SOURCE (real, unauthenticated, no mock mode)
-------------------------------------------------
    https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{GAME_ID}.json

This is the endpoint NBA.com's own scoreboard calls client-side, and the
one `nba_api`'s live module wraps. It requires no API key. It DOES
bot-detect on missing headers, so a browser-like User-Agent is mandatory
— a bare `requests.get()` is blocked.

GAME_ID format: 10 chars, zero-padded — "0022500001".
    00 + season-type (1=pre, 2=regular, 4=playoffs, 5=play-in, 6=NBA Cup final)
       + 2-digit season + 5-digit sequence

STATUS GATE
-----------
`gameStatus` 3 == Final. Box scores for in-progress games are live and
will change, so `fetch_final_boxscore` refuses to return a non-final
game rather than letting the settlement engine grade a partial line.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

from src.utils.ids import normalise_one_game_id

logger = logging.getLogger(__name__)

CDN_BOXSCORE_URL = "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{game_id}.json"

# NBA's CDN rejects requests that look scripted.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
}

GAME_STATUS_FINAL = 3
REQUEST_TIMEOUT = 15
RETRY_ATTEMPTS = 3
#: EXPONENTIAL, as the base of `RETRY_BACKOFF ** attempt`. This was multiplied
#: by the attempt number instead -- 2s, 4s, 6s -- which `docs/autonomy_audit_
#: 2026-10-10.md` P4 recorded as the one client on a daily path with neither
#: exponential backoff nor a 429 branch, while `ingestion/espn_client` and
#: `ingestion/propline` had both.
RETRY_BACKOFF = 2.0
#: A server asking us to wait an hour is not something to obey inside a slate
#: or settlement job; cap it and let the caller fail rather than hang. Same
#: value and reasoning as `ingestion/espn_client.MAX_RETRY_AFTER_SECONDS`.
MAX_RETRY_AFTER_SECONDS = 60.0


def retry_after_seconds(headers: Any, fallback: float) -> float:
    """
    The server's ``Retry-After`` in seconds, else ``fallback``.

    ONE RULE FOR BOTH CDN CLIENTS. cubic, PR #11: each parsed the header with a
    bare ``float(...)``, which accepts ``-1`` (``time.sleep`` raises), ``nan``
    (raises) and ``inf`` (never returns) -- turning a rate limit into a crashed
    or hung fetch -- while the HTTP-date form fell through silently.

    Only the plain integer-seconds form is honoured. The date form is IGNORED
    rather than parsed approximately, because a wrong date parse sleeps for
    either no time or a very long one; `ingestion/espn_client` made the same
    call for the same reason and this is that rule, shared.
    """
    import math

    try:
        raw = headers.get("Retry-After")
    except (AttributeError, TypeError):
        return float(fallback)
    if raw is None:
        return float(fallback)
    try:
        seconds = float(str(raw).strip())
    except (TypeError, ValueError):
        return float(fallback)
    if not math.isfinite(seconds) or seconds <= 0:
        return float(fallback)
    return min(seconds, MAX_RETRY_AFTER_SECONDS)
#: The one 4xx that IS transient: the request is fine, there have just been too
#: many of them. Falling through to `raise_for_status` abandoned a night's
#: grading over rate limiting.
TOO_MANY_REQUESTS = 429


class BoxScoreError(RuntimeError):
    """Raised when a box score cannot be fetched or is not final."""


class GameNotFinalError(BoxScoreError):
    """The game exists but has not finished — do not settle against it."""


def normalize_game_id(game_id: str | int) -> str:
    """NBA game ids are zero-padded 10-char strings."""
    return normalise_one_game_id(game_id) or ""


def fetch_boxscore(game_id: str | int, session: requests.Session | None = None) -> dict[str, Any]:
    """Fetch the raw CDN box score payload. Raises on failure — never returns mock data."""
    gid = normalize_game_id(game_id)
    url = CDN_BOXSCORE_URL.format(game_id=gid)
    sess = session or requests.Session()
    last_exc: Exception | None = None

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            resp = sess.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 403:
                raise BoxScoreError(
                    f"403 from NBA CDN for {gid} — bot detection. Verify the "
                    f"User-Agent/Referer headers are being sent."
                )
            if resp.status_code == 404:
                raise BoxScoreError(
                    f"404 for game {gid} — game id does not exist or the box "
                    f"score is not yet posted."
                )
            if resp.status_code == TOO_MANY_REQUESTS:
                # Honour the server's own number when it gives one; guessing a
                # backoff against a rate limiter is how access gets revoked.
                # `espn_client` and `propline` both already do this.
                retry_after = resp.headers.get("Retry-After")
                wait = retry_after_seconds(resp.headers, RETRY_BACKOFF ** attempt)
                if attempt < RETRY_ATTEMPTS:
                    logger.warning(
                        "boxscore: 429 for %s, attempt %d/%d; waiting %.1fs%s",
                        gid, attempt, RETRY_ATTEMPTS, wait,
                        " (Retry-After)" if retry_after else "",
                    )
                    time.sleep(wait)
                    continue
                raise BoxScoreError(
                    f"NBA CDN rate-limited {gid} on all {RETRY_ATTEMPTS} "
                    f"attempts. Settlement has a six-hour misfire grace, so "
                    f"this game is graded on the next run rather than lost."
                )
            resp.raise_for_status()
            return resp.json()
        except BoxScoreError:
            raise
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            if attempt < RETRY_ATTEMPTS:
                # ** not *: 2s, 4s, 8s. Linear backoff against a rate limiter
                # that is still saturated just retries into the same wall.
                time.sleep(RETRY_BACKOFF ** attempt)

    raise BoxScoreError(f"Failed to fetch box score for {gid} after {RETRY_ATTEMPTS} attempts") from last_exc


def fetch_final_boxscore(game_id: str | int, session: requests.Session | None = None) -> dict[str, Any]:
    """
    Fetch a box score and REFUSE to return it unless the game is final.

    Grading against a live game would settle props on partial stat lines.
    """
    payload = fetch_boxscore(game_id, session=session)
    game = payload.get("game", {})
    status = game.get("gameStatus")

    if status != GAME_STATUS_FINAL:
        raise GameNotFinalError(
            f"Game {normalize_game_id(game_id)} has gameStatus={status} "
            f"({game.get('gameStatusText', 'unknown')}), not Final ({GAME_STATUS_FINAL}). "
            f"Refusing to settle against a non-final box score."
        )
    return payload


def _positive(value: Any) -> bool:
    """True only for a numeric value above zero.

    Used to decide whether a player with unreadable minutes actually
    appeared, so a non-numeric or absent field must read as 'no evidence',
    never as evidence of play.
    """
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False


def _parse_minutes(value: Any) -> float | None:
    """
    CDN reports minutes as an ISO-8601 duration: 'PT36M14.00S'.
    Returns decimal minutes, or None when absent/unparseable.
    """
    if not value or not isinstance(value, str):
        return None
    if not value.startswith("PT"):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    try:
        body = value[2:]
        minutes = seconds = 0.0
        if "M" in body:
            mins_part, _, rest = body.partition("M")
            minutes = float(mins_part)
            body = rest
        if "S" in body:
            seconds = float(body.rstrip("S"))
        return round(minutes + seconds / 60.0, 2)
    except (TypeError, ValueError):
        return None


def extract_player_stats(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """
    Flatten a CDN box score into {player_name: stats} for BOTH teams.

    The returned stat keys match MARKET_COMPONENTS in evaluator.py:
    points, reboundsTotal, assists, threePointersMade, steals, blocks,
    turnovers — the CDN's own field names, used verbatim so no renaming
    layer can drift.

    `did_not_play` is set from the CDN's `status`/`notPlayingReason`
    fields. A DNP must produce a VOID, not a zero — zeroing it would
    mis-grade every Over as a LOSS.
    """
    game = payload.get("game", {})
    out: dict[str, dict[str, Any]] = {}

    for side in ("homeTeam", "awayTeam"):
        team = game.get(side) or {}
        team_tricode = team.get("teamTricode")
        for player in team.get("players", []) or []:
            name = player.get("name") or (
                f"{player.get('firstName', '')} {player.get('familyName', '')}".strip()
            )
            if not name:
                continue

            stats = player.get("statistics") or {}
            status = (player.get("status") or "").upper()
            dnp_reason = player.get("notPlayingReason")
            minutes = _parse_minutes(stats.get("minutes"))

            # A failed minutes parse is not a DNP. Folding None into this
            # test voided every player whose minutes string the CDN changed
            # the format of — with the note "player did not play", against a
            # box score showing they had played. Those rows left W-L and ROI
            # entirely, so a parser break would have read as a quiet sample
            # shrink rather than an error.
            played_signal = any(
                _positive(stats.get(field))
                for field in (
                    "points", "reboundsTotal", "assists",
                    "threePointersMade", "steals", "blocks", "turnovers",
                )
            )
            if status == "INACTIVE" or dnp_reason:
                did_not_play = True          # the feed says so outright
                minutes_status = "DNP_DECLARED"
            elif minutes == 0.0:
                did_not_play = True          # parsed, and genuinely zero
                minutes_status = "PARSED"
            elif minutes is None:
                # Unknowable from minutes alone. Recorded stats settle it;
                # with none, DNP is both the likelier reading and the safe
                # one, since VOID invents no win or loss.
                did_not_play = not played_signal
                minutes_status = "UNAVAILABLE"
            else:
                did_not_play = False
                minutes_status = "PARSED"

            if minutes_status == "UNAVAILABLE" and played_signal:
                logger.warning(
                    "Could not parse minutes for %r (raw %r) but the box score "
                    "records stats — grading from the stat line, not voiding.",
                    name, stats.get("minutes"),
                )

            # Two players in one game can share a full name. Silently keeping
            # the last would let settlement grade one player's prop against
            # the other's box score, so the collision is recorded and both
            # entries are marked ambiguous for the matcher to refuse.
            if name in out:
                out[name]["ambiguous_name"] = True
                logger.warning(
                    "Duplicate player name %r in game %s — marked ambiguous; "
                    "settle by personId, not name.",
                    name, payload.get("gameId", "?"),
                )
                continue

            out[name] = {
                "player_name": name,
                "ambiguous_name": False,
                "nba_player_id": str(player.get("personId")) if player.get("personId") else None,
                "team_tricode": team_tricode,
                "is_home": side == "homeTeam",
                "minutes_played": minutes,
                "minutes_status": minutes_status,
                "did_not_play": did_not_play,
                "not_playing_reason": dnp_reason,
                # CDN field names, verbatim
                "points": stats.get("points"),
                "reboundsTotal": stats.get("reboundsTotal"),
                "assists": stats.get("assists"),
                "threePointersMade": stats.get("threePointersMade"),
                "steals": stats.get("steals"),
                "blocks": stats.get("blocks"),
                "turnovers": stats.get("turnovers"),
            }

    logger.info(
        "Extracted %d player stat lines from game %s (%d DNP)",
        len(out), game.get("gameId"), sum(1 for p in out.values() if p["did_not_play"]),
    )
    return out


def fetch_player_stats_for_game(
    game_id: str | int, session: requests.Session | None = None
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Convenience: final box score -> (player stats by name, raw payload)."""
    payload = fetch_final_boxscore(game_id, session=session)
    return extract_player_stats(payload), payload
