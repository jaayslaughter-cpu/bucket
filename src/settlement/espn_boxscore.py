"""
src/settlement/espn_boxscore.py — the box score we can actually reach.

WHY THIS EXISTS. `settlement/boxscore_fetcher` grades from
``cdn.nba.com/static/json/liveData/boxscore/boxscore_{game_id}.json``, and
measured on 2026-10-19, two days before the season opener:

    cdn.nba.com    403 in 0.47s  (actively rejected, with the real
                                  User-Agent/Referer/Origin headers sent)
    stats.nba.com  000 after 20s (connection never completed)

A control host through the same proxy answered, so this is the NBA side
refusing, not a local misconfiguration. With no reachable box score the
settlement loop cannot close: props stay PENDING, the W/L ledger never
fills, `prop_result_calibration_report` has nothing to read, and the
publication gate withholds every card for want of evidence it can never
acquire. The loop is self-reinforcing, which is why this is a blocker and
not an inconvenience.

``site.api.espn.com`` answers 200 and carries full player box scores —
MIN, PTS, FG, 3PT, FT, REB, AST, TO, STL, BLK, OREB, DREB, PF, +/-, plus
``didNotPlay`` and ``starter`` — and the project already reaches it for the
schedule, so it needs no new host, no new key and no bot-wall circumvention.

WHAT THIS IS NOT. It is a FALLBACK, not a replacement. The CDN is addressed
by NBA game id and is exact; ESPN is addressed by its own event id and has
to be matched to ours. Where the CDN answers, it still wins.

THE TWO THINGS ESPN CANNOT GIVE US, recorded here rather than papered over:

  * NO NBA PERSON ID. ESPN numbers athletes itself (Tobias Harris is 6440),
    so ``nba_player_id`` comes back None on this path rather than carrying an
    ESPN id in a field every reader will take for an NBA one. That is a real
    loss: `extract_player_stats`'s duplicate-name branch says "settle by
    personId, not name", and on this path there is no personId to settle by.
    Duplicate names are still DETECTED and still marked ambiguous, so the
    matcher refuses them instead of grading one player against another's line.

  * NO NBA GAME ID anywhere in the payload (searched; absent). So the event
    has to be identified, and `fetch_player_stats_for_game` does it by
    EVIDENCE: the players we are grading must appear in the event's own box
    score. See `select_event_by_players`, which declines rather than guesses.

RESEARCH_ONLY. This reads results that already happened.
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime
from typing import Any

import requests

from src.utils.http_retry import retry_after_seconds

logger = logging.getLogger(__name__)

SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"
SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"

#: ESPN's own name for a completed game. Anything else is refused, for the
#: same reason the CDN path checks gameStatus: grading a live game settles
#: props against a partial stat line.
STATUS_FINAL = "STATUS_FINAL"

REQUEST_TIMEOUT = 15
RETRY_ATTEMPTS = 3
RETRY_BACKOFF = 2.0
TOO_MANY_REQUESTS = 429

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
}

#: ESPN label -> the CDN field name the rest of settlement already reads.
#: The CDN's names are used VERBATIM downstream so no renaming layer can
#: drift (see `extract_player_stats`), which means the translation has to
#: happen here, once.
DIRECT_LABELS = {
    "PTS": "points",
    "REB": "reboundsTotal",
    "AST": "assists",
    "STL": "steals",
    "BLK": "blocks",
    "TO": "turnovers",
}

#: Labels whose value is "made-attempted" ("3-5"). 3PT feeds the market
#: component; FG and FT feed the fga/fta columns migration 009 added.
SPLIT_LABELS = {
    "3PT": ("threePointersMade", "threePointersAttempted"),
    "FG": ("fieldGoalsMade", "fieldGoalsAttempted"),
    "FT": ("freeThrowsMade", "freeThrowsAttempted"),
}


class EspnBoxScoreError(RuntimeError):
    """ESPN could not supply a usable box score. Never returns mock data."""


class EspnGameNotFinalError(EspnBoxScoreError):
    """The event exists but is not finished."""


def _int_or_none(value: Any) -> int | None:
    """ESPN sends every stat as a STRING, including '+2' and '--'."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in {"--", "-"}:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _split_made_attempted(value: Any) -> tuple[int | None, int | None]:
    """'3-5' -> (3, 5). Anything else -> (None, None), never a guess."""
    if value is None:
        return None, None
    parts = str(value).strip().split("-")
    if len(parts) != 2:
        return None, None
    return _int_or_none(parts[0]), _int_or_none(parts[1])


def _parse_minutes(value: Any) -> float | None:
    """ESPN minutes are plain ('25'), but accept 'MM:SS' rather than assume."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in {"--", "-"}:
        return None
    if ":" in text:
        mm, _, ss = text.partition(":")
        m, s = _int_or_none(mm), _int_or_none(ss)
        if m is None:
            return None
        return float(m) + (float(s) / 60.0 if s is not None else 0.0)
    try:
        return float(text)
    except ValueError:
        return None


def _get(url: str, params: dict[str, Any], session: requests.Session | None) -> dict[str, Any]:
    """One GET, with the project's single Retry-After rule."""
    sess = session or requests.Session()
    last: Exception | None = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            resp = sess.get(url, params=params, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            if resp.status_code == TOO_MANY_REQUESTS:
                wait = retry_after_seconds(resp.headers, RETRY_BACKOFF ** attempt)
                if attempt < RETRY_ATTEMPTS:
                    logger.warning(
                        "espn boxscore: 429 on %s, attempt %d/%d; waiting %.1fs",
                        url, attempt, RETRY_ATTEMPTS, wait,
                    )
                    time.sleep(wait)
                    continue
                raise EspnBoxScoreError(
                    f"ESPN rate-limited {url} on all {RETRY_ATTEMPTS} attempts. "
                    f"Settlement has a six-hour misfire grace, so this game is "
                    f"graded on the next run rather than lost."
                )
            resp.raise_for_status()
            return resp.json()
        except EspnBoxScoreError:
            raise
        except (requests.exceptions.RequestException, ValueError) as exc:
            last = exc
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_BACKOFF ** attempt)
    raise EspnBoxScoreError(f"ESPN request failed after {RETRY_ATTEMPTS} attempts: {url}") from last


def _as_yyyymmdd(game_date: str | date | datetime) -> str:
    if isinstance(game_date, datetime):
        return game_date.date().strftime("%Y%m%d")
    if isinstance(game_date, date):
        return game_date.strftime("%Y%m%d")
    text = str(game_date).strip()
    if len(text) == 8 and text.isdigit():
        return text
    return datetime.fromisoformat(text[:10]).strftime("%Y%m%d")


def final_event_ids_for_date(
    game_date: str | date | datetime, session: requests.Session | None = None
) -> list[str]:
    """Event ids of the FINAL games on a date. Non-final games are left out."""
    payload = _get(SCOREBOARD_URL, {"dates": _as_yyyymmdd(game_date)}, session)
    out: list[str] = []
    for event in payload.get("events") or []:
        event_id = event.get("id")
        if not event_id:
            continue
        competitions = event.get("competitions") or [{}]
        status = (((competitions[0].get("status") or {}).get("type") or {})
                  .get("name") or "")
        if status == STATUS_FINAL:
            out.append(str(event_id))
    return out


def fetch_event_summary(
    event_id: str, session: requests.Session | None = None
) -> dict[str, Any]:
    return _get(SUMMARY_URL, {"event": str(event_id)}, session)


def _event_status(summary: dict[str, Any]) -> str:
    header = summary.get("header") or {}
    competitions = header.get("competitions") or [{}]
    return (((competitions[0].get("status") or {}).get("type") or {}).get("name") or "")


def extract_espn_player_stats(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """
    Flatten an ESPN summary into the SAME shape `extract_player_stats` returns.

    STATS ARE READ BY LABEL, NEVER BY POSITION. ESPN ships a positional
    ``stats`` array beside a ``labels`` array; indexing the array by a
    hardcoded offset would silently read rebounds as assists the day ESPN
    inserts a column. Zipping the two is the only form that cannot drift.
    """
    status = _event_status(summary)
    if status != STATUS_FINAL:
        raise EspnGameNotFinalError(
            f"ESPN event status is {status or 'unknown'}, not {STATUS_FINAL}. "
            f"Refusing to settle against a non-final box score."
        )

    boxscore = summary.get("boxscore") or {}
    out: dict[str, dict[str, Any]] = {}
    home_abbr = _home_abbreviation(summary)

    for team_block in boxscore.get("players") or []:
        team = team_block.get("team") or {}
        tricode = team.get("abbreviation")
        for group in team_block.get("statistics") or []:
            labels = [str(x) for x in (group.get("labels") or [])]
            for athlete in group.get("athletes") or []:
                info = athlete.get("athlete") or {}
                name = (info.get("displayName") or "").strip()
                if not name:
                    continue

                raw = dict(zip(labels, athlete.get("stats") or []))
                values: dict[str, Any] = {
                    field: _int_or_none(raw.get(label))
                    for label, field in DIRECT_LABELS.items()
                }
                for label, (made_key, att_key) in SPLIT_LABELS.items():
                    made, attempted = _split_made_attempted(raw.get(label))
                    values[made_key] = made
                    values[att_key] = attempted

                minutes = _parse_minutes(raw.get("MIN"))
                declared_dnp = bool(athlete.get("didNotPlay"))
                played_signal = any(
                    v is not None and v > 0
                    for v in (values.get(f) for f in (
                        "points", "reboundsTotal", "assists",
                        "threePointersMade", "steals", "blocks", "turnovers",
                    ))
                )

                # Mirrors the CDN path's reasoning exactly: a failed minutes
                # parse is NOT a DNP, because voiding a player the box score
                # shows playing removes them from W-L entirely and reads as a
                # quiet sample shrink rather than an error.
                if declared_dnp:
                    did_not_play, minutes_status = True, "DNP_DECLARED"
                elif minutes == 0.0:
                    did_not_play, minutes_status = True, "PARSED"
                elif minutes is None:
                    did_not_play, minutes_status = not played_signal, "UNAVAILABLE"
                else:
                    did_not_play, minutes_status = False, "PARSED"

                if minutes_status == "UNAVAILABLE" and played_signal:
                    logger.warning(
                        "ESPN: could not parse minutes for %r (raw %r) but the "
                        "box score records stats — grading from the stat line, "
                        "not voiding.", name, raw.get("MIN"),
                    )

                if name in out:
                    out[name]["ambiguous_name"] = True
                    logger.warning(
                        "ESPN: duplicate player name %r in event %s — marked "
                        "ambiguous. This path has no NBA personId to settle by, "
                        "so the matcher must refuse rather than pick.",
                        name, (summary.get("header") or {}).get("id", "?"),
                    )
                    continue

                out[name] = {
                    "player_name": name,
                    "ambiguous_name": False,
                    # DELIBERATELY None: ESPN's athlete id is not an NBA
                    # personId, and a reader of this field will assume it is.
                    "nba_player_id": None,
                    "espn_athlete_id": str(info.get("id")) if info.get("id") else None,
                    "team_tricode": tricode,
                    "is_home": tricode is not None and tricode == home_abbr,
                    "minutes_played": minutes,
                    "minutes_status": minutes_status,
                    "did_not_play": did_not_play,
                    "not_playing_reason": athlete.get("reason"),
                    "source": "espn",
                    **values,
                }

    logger.info(
        "ESPN: extracted %d player stat lines from event %s (%d DNP)",
        len(out), (summary.get("header") or {}).get("id", "?"),
        sum(1 for p in out.values() if p["did_not_play"]),
    )
    return out


def _home_abbreviation(summary: dict[str, Any]) -> str | None:
    header = summary.get("header") or {}
    for competition in header.get("competitions") or []:
        for competitor in competition.get("competitors") or []:
            if competitor.get("homeAway") == "home":
                return (competitor.get("team") or {}).get("abbreviation")
    return None


def select_event_by_players(
    candidates: dict[str, dict[str, dict[str, Any]]],
    expected_players: set[str],
) -> str:
    """
    Pick the event whose box score actually contains the players being graded.

    THE EVENT ID IS DERIVED, NOT ASSUMED. ESPN's payload carries no NBA game
    id, and `games` — the one table with both an nba_game_id and a team pair —
    has no writer in this tree, so there is nothing to join on. What IS
    available is the thing settlement already matches on: player names.

    REFUSES RATHER THAN GUESSES, in both ambiguous cases: no event contains
    any of the players, or two events tie on overlap. A wrong event here would
    grade a whole game's props against a different game's box score, which is
    worse than leaving them PENDING for the next run — settlement has a
    six-hour misfire grace precisely so that waiting is cheap.
    """
    if not expected_players:
        raise EspnBoxScoreError(
            "No expected players supplied, so the ESPN event cannot be "
            "identified. This path matches on the players being graded."
        )

    wanted = {p.strip().lower() for p in expected_players if p and p.strip()}
    scores: list[tuple[int, str]] = []
    for event_id, stats in candidates.items():
        have = {name.strip().lower() for name in stats}
        scores.append((len(wanted & have), event_id))

    scores.sort(reverse=True)
    if not scores or scores[0][0] == 0:
        raise EspnBoxScoreError(
            f"None of the {len(wanted)} player(s) being graded appear in any "
            f"of the {len(candidates)} final ESPN event(s) for this date, so "
            f"the event cannot be identified."
        )
    if len(scores) > 1 and scores[1][0] == scores[0][0]:
        raise EspnBoxScoreError(
            f"Two ESPN events match {scores[0][0]} of the players being graded "
            f"equally well ({scores[0][1]}, {scores[1][1]}). Refusing to pick: "
            f"grading against the wrong game is worse than grading later."
        )
    return scores[0][1]


def fetch_espn_player_stats_for_game(
    game_date: str | date | datetime,
    expected_players: set[str],
    session: requests.Session | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """
    (player stats by name, raw summary) for the event matching these players.

    One scoreboard call plus one summary call per FINAL event on the date.
    A full slate is ~12 requests, against a host that answers.
    """
    event_ids = final_event_ids_for_date(game_date, session=session)
    if not event_ids:
        raise EspnGameNotFinalError(
            f"ESPN lists no FINAL NBA events on {game_date}, so there is "
            f"nothing to settle against yet."
        )

    candidates: dict[str, dict[str, dict[str, Any]]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for event_id in event_ids:
        try:
            summary = fetch_event_summary(event_id, session=session)
            candidates[event_id] = extract_espn_player_stats(summary)
            summaries[event_id] = summary
        except EspnGameNotFinalError:
            continue  # the scoreboard and the summary disagreed; trust the summary
        except EspnBoxScoreError as exc:
            logger.warning("ESPN: event %s unusable (%s); trying the rest.", event_id, exc)

    if not candidates:
        raise EspnBoxScoreError(
            f"ESPN returned no usable final box score among "
            f"{len(event_ids)} event(s) on {game_date}."
        )

    chosen = select_event_by_players(candidates, expected_players)
    logger.info("ESPN: graded against event %s on %s", chosen, game_date)
    return candidates[chosen], summaries[chosen]
