"""
src/settlement/espn_boxscore.py — grade from a box score we can reach.

WHY THIS EXISTS. `settlement/boxscore_fetcher` grades from the NBA CDN, and
measured on 2026-10-19, two days before the season opener:

    cdn.nba.com    403 in 0.47s  (actively rejected, with the real
                                  User-Agent/Referer/Origin headers sent)
    stats.nba.com  000 after 20s (connection never completed)

A control host through the same proxy answered, so the refusal is NBA's. With
no box score the settlement loop cannot close: props stay PENDING, the ledger
never fills, `prop_result_calibration_report` has nothing to read, and
`publication_gate` then withholds every card for want of evidence that grading
is the only source of. The loop re-arms itself.

WHAT THIS MODULE IS: AN ADAPTER, AND NOTHING MORE. The first version of this
file parsed ESPN itself -- and `src/ingestion/espn_game` already did, better.
That module handles BOTH documented summary shapes
(``boxscore.teams[].players[]`` and ``boxscore.players[]``), zips ESPN's own
header names to that block's values with ``strict=True`` so a length mismatch
RAISES instead of truncating to fit, and splits made-attempted pairs. Its
docstring states the same lesson the duplicate had rediscovered: indexing the
positional ``stats`` array puts rebounds in the assists column the moment ESPN
reorders a column, and nothing looks wrong.

So transport goes through `espn_client.get_json` (which honours
``PROPIQ_ESPN_SITE_BASE``, so a mirror or a test host is respected, and which
does not retry a permanent 4xx), the scoreboard through
`espn_schedule.fetch_scoreboard`, and the box score through
`espn_game.fetch_summary`. What is left here is the only thing genuinely new:
deciding WHICH ESPN event is the NBA game we are grading.

THAT DECISION IS MADE ON EVIDENCE. ESPN's payload carries no NBA game id
(searched; absent), and `games` -- the one table holding both an nba_game_id
and a team pair -- has no writer anywhere in this tree, so there is nothing to
join on. What is available is what settlement already matches on: the players
being graded. `select_event_by_players` requires the winning event to contain
EVERY expected player that appears anywhere on the slate, and refuses
otherwise. A unique PARTIAL overlap is not enough: if the game we want is not
final yet and some other game that night happens to contain one same-named
player, a partial rule would pick it and grade a whole game's props against a
different game's box score. That is the exact catastrophe this module exists
to avoid, so it declines and the props wait for the next run -- which is what
settlement's six-hour misfire grace is for.

RESEARCH_ONLY. This reads results that already happened.
"""

from __future__ import annotations

import logging
import math
from datetime import date, datetime
from typing import Any

import requests

from src.ingestion.espn_game import fetch_summary
from src.ingestion.espn_schedule import STATE_FINAL, fetch_scoreboard, parse_scoreboard
from src.settlement.evaluator import PANEL_TO_BOXSCORE
from src.settlement.names import normalise_player_name

logger = logging.getLogger(__name__)

#: `espn_game` emits the PANEL's stat names. Settlement reads the CDN's field
#: names verbatim (see `extract_player_stats`), so the translation happens
#: once -- in `evaluator.PANEL_TO_BOXSCORE`, which is where the box-score
#: vocabulary is defined and graded from. This EXTENDS it with the shot-volume
#: columns migration 009 added, rather than restating the seven pairs it
#: already has: two copies of this table is how a renamed field grades one way
#: and projects another. A stat ESPN omitted stays absent, never a zero.
PANEL_TO_CDN = {
    **PANEL_TO_BOXSCORE,
    # Shot volume, for the fga/fta columns migration 009 added.
    "FGM": "fieldGoalsMade",
    "FGA": "fieldGoalsAttempted",
    "FTM": "freeThrowsMade",
    "FTA": "freeThrowsAttempted",
}

#: The market components `evaluator` grades on. Used only to decide whether a
#: row with unreadable minutes nevertheless shows evidence of having played.
MARKET_COMPONENTS = (
    "points", "reboundsTotal", "assists",
    "threePointersMade", "steals", "blocks", "turnovers",
)


class EspnBoxScoreError(RuntimeError):
    """ESPN could not supply a usable box score. Never returns mock data."""


class EspnGameNotFinalError(EspnBoxScoreError):
    """No ESPN event for this game has finished."""


def _as_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.fromisoformat(str(value)[:10]).date()


def _int_or_none(value: Any) -> int | None:
    """
    `espn_game` yields floats; the CDN contract is integer counting stats.

    REFUSES A FRACTION, DOES NOT ROUND IT. This was `int(round(...))`, so a
    malformed ``3.5`` points became a confident ``4`` — and on a 3.5 line that
    is the difference between a WIN and a LOSS, decided by a rounding mode. A
    player cannot score half a point: a fractional counting stat is corrupt
    input, and the only honest reading of corrupt input is none.

    AND REFUSES A NEGATIVE, for the same reason. Every field this feeds is a
    COUNTING stat — points, rebounds, assists, steals, blocks, turnovers,
    makes and attempts — and none of them can go below zero. ``PLUS_MINUS`` is
    the one signed figure ESPN sends and it is deliberately not in
    `PANEL_TO_CDN`, so nothing legitimately negative passes through here. A
    ``-3`` is corrupt, and settling an Under on it would be a real-looking
    result built on impossible input.
    """
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or not number.is_integer() or number < 0:
        return None
    return int(number)


def _minutes_or_none(value: Any) -> float | None:
    """Minutes as a finite, non-negative float, else None for "unknown"."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0:
        return None
    return number


def final_events_for_date(
    game_date: str | date | datetime, session: requests.Session | None = None
) -> list[Any]:
    """
    The FINAL games on a date, as `espn_schedule.SlateGame`s.

    Finality comes from the SCOREBOARD's own ``state == "post"``, which is the
    convention `espn_schedule` already uses. It deliberately does not come from
    `GameSummary.status`: that field reports whether the PARSE succeeded (``OK``
    / ``DATA_NOT_AVAILABLE``), not whether the game finished, and reading it as
    finality would settle props against a game still in progress.
    """
    payload = fetch_scoreboard(_as_date(game_date), session=session)
    result = parse_scoreboard(payload)
    return [g for g in result.games if g.state == STATE_FINAL]


def settlement_rows(summary: Any, slate_game: Any = None) -> dict[str, dict[str, Any]]:
    """
    `espn_game`'s BoxScoreRows -> the dict `extract_player_stats` returns.

    TWO THINGS ESPN CANNOT GIVE, recorded rather than papered over:

      * NO NBA PERSON ID. ESPN numbers athletes itself, so ``nba_player_id``
        is None here rather than carrying an ESPN id in a field every reader
        will take for an NBA one -- a join against player_game_logs would then
        match nothing, or the wrong person. It goes in ``espn_athlete_id``.
      * Duplicate names therefore have no personId to settle by, so DETECTION
        is the only protection left. Both entries are marked ambiguous and
        `_match_player` refuses, rather than grading one player against the
        other's line.
    """
    out: dict[str, dict[str, Any]] = {}

    for row in getattr(summary, "box_score", None) or []:
        name = (row.player_name or "").strip()
        if not name:
            continue

        values = {
            cdn: _int_or_none(row.stats.get(panel))
            for panel, cdn in PANEL_TO_CDN.items()
        }
        # FINITE AND NON-NEGATIVE, OR UNKNOWN. A bare float() let ``-5`` and
        # ``inf`` through as "parsed", so malformed availability data read as a
        # row that played. Anything unusable becomes None and takes the
        # unavailable-minutes path below, which is already written for exactly
        # this uncertainty.
        minutes = _minutes_or_none(row.stats.get("MIN"))
        played_signal = any(
            values.get(field) is not None and values[field] > 0
            for field in MARKET_COMPONENTS
        )

        # A failed minutes parse is NOT a DNP. The CDN path learned this the
        # hard way: folding it in voided every player whose minutes format
        # changed, against a box score showing they had played, and those rows
        # left W-L entirely -- a parser break reading as a quiet sample shrink.
        if row.did_not_play:
            did_not_play, minutes_status = True, "DNP_DECLARED"
        elif minutes == 0.0:
            did_not_play, minutes_status = True, "PARSED"
        elif minutes is None:
            did_not_play, minutes_status = not played_signal, "UNAVAILABLE"
        else:
            did_not_play, minutes_status = False, "PARSED"

        if minutes_status == "UNAVAILABLE" and played_signal:
            logger.warning(
                "ESPN: minutes unreadable for %r but the box score records "
                "stats — grading from the stat line, not voiding.", name,
            )

        # THE GUARD HAS TO USE THE MATCHER'S OWN RULE. Keyed on the raw name,
        # "P.J. Washington" and "PJ Washington" were two separate entries --
        # and `_match_player` normalises, so it would have matched whichever
        # it reached first and graded one player against the other's line. The
        # collision the guard exists to catch was invisible to it.
        clash = [
            existing for existing in out
            if normalise_player_name(existing) == normalise_player_name(name)
        ]
        if clash:
            for existing in clash:
                out[existing]["ambiguous_name"] = True
            logger.warning(
                "ESPN: %r collides with %s in event %s once normalised — all "
                "marked ambiguous. This path has no NBA personId to settle by, "
                "so the matcher must refuse rather than pick.",
                name, ", ".join(repr(c) for c in clash), row.espn_event_id,
            )
            continue

        out[name] = {
            "player_name": name,
            "ambiguous_name": False,
            "nba_player_id": None,
            "espn_athlete_id": row.espn_athlete_id,
            # NEITHER IS AVAILABLE AND NEITHER IS READ. `espn_game` carries
            # ESPN's team id, not an NBA tricode, and no settlement consumer
            # reads these two fields (checked: only features/matchup uses the
            # name, against different data). None is the honest value; a
            # guessed tricode would be worse than an absent one.
            "team_tricode": None,
            "is_home": None,
            "minutes_played": minutes,
            "minutes_status": minutes_status,
            "did_not_play": did_not_play,
            "not_playing_reason": None,
            "source": "espn",
            **values,
        }
    return out


def select_event_by_players(
    candidates: dict[str, dict[str, dict[str, Any]]],
    expected_players: set[str],
) -> str:
    """
    The event whose box score contains EVERY player we are grading.

    A UNIQUE PARTIAL OVERLAP IS NOT ENOUGH, and the first version accepted
    one. If the game we want is not final yet, and another game that night
    contains a single same-named player, a partial rule picks that event and
    grades a whole game's props against a different game's box score.

    "Every player" means every expected player who appears ANYWHERE on the
    slate. A player absent from all of them -- someone who did not travel and
    so never reached a box score -- must not block grading for the rest of the
    game, and is reported rather than silently tolerated. But a player who
    turns up in a DIFFERENT event means the set is split across games, and
    that is a refusal: something upstream is wrong and guessing cannot fix it.

    Names are compared through `settlement.names.normalise_player_name`, the
    same rule `_match_player` settles with, so the selector cannot fail to
    recognise a player the matcher would have matched.
    """
    if not expected_players:
        raise EspnBoxScoreError(
            "No expected players supplied, so the ESPN event cannot be "
            "identified. This path matches on the players being graded."
        )

    wanted = {normalise_player_name(p) for p in expected_players if p and p.strip()}
    if not wanted:
        raise EspnBoxScoreError(
            "Every expected player name was blank once normalised, so the "
            "ESPN event cannot be identified."
        )

    have: dict[str, set[str]] = {
        event_id: {normalise_player_name(n) for n in stats}
        for event_id, stats in candidates.items()
    }
    anywhere = set().union(*have.values()) if have else set()
    reachable = wanted & anywhere
    if not reachable:
        raise EspnBoxScoreError(
            f"None of the {len(wanted)} player(s) being graded appear in any "
            f"of the {len(candidates)} final ESPN event(s) for this date, so "
            f"the event cannot be identified."
        )

    missing_everywhere = sorted(wanted - anywhere)
    complete = [eid for eid, names in have.items() if reachable <= names]

    if not complete:
        best = max(((len(reachable & names), eid) for eid, names in have.items()),
                   default=(0, "?"))
        raise EspnBoxScoreError(
            f"No single final ESPN event contains all {len(reachable)} of the "
            f"players being graded that appear on this slate (best was "
            f"{best[1]} with {best[0]}). The player set is split across "
            f"events, so settling against any one of them would grade this "
            f"game against another's box score."
        )
    if len(complete) > 1:
        raise EspnBoxScoreError(
            f"{len(complete)} final ESPN events each contain all "
            f"{len(reachable)} players being graded ({', '.join(sorted(complete))}). "
            f"Refusing to pick: grading against the wrong game is worse than "
            f"grading later."
        )

    if missing_everywhere:
        logger.warning(
            "ESPN: %d of the players being graded reached no box score on this "
            "slate (%s) — they are likely scratches who did not travel, and "
            "they do not block the rest of the game. They will not settle.",
            len(missing_everywhere), ", ".join(missing_everywhere[:5]),
        )
    return complete[0]


def fetch_espn_player_stats_for_game(
    game_date: str | date | datetime,
    expected_players: set[str],
    session: requests.Session | None = None,
    cache: dict[str, Any] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """
    (player stats by name, raw-ish summary) for the event matching these players.

    ``cache`` IS NOT AN OPTIMISATION, IT IS A RATE-LIMIT GUARD. Without it the
    whole slate is re-fetched once per game: a twelve-game CDN outage made one
    scoreboard plus twelve summary calls THIRTEEN TIMES -- about 156 requests
    against a host whose 429 would then strand the grading this exists to
    rescue. The runner creates one dict per settlement run and passes it down,
    so the slate is fetched once. It is explicit rather than module-level
    state, so two runs cannot see each other's.
    """
    key = f"espn_slate:{_as_date(game_date).isoformat()}"
    store = cache if cache is not None else {}

    if key not in store:
        slate_games = final_events_for_date(game_date, session=session)
        if not slate_games:
            raise EspnGameNotFinalError(
                f"ESPN lists no final NBA events on {game_date}, so there is "
                f"nothing to settle against yet."
            )
        built: dict[str, tuple[dict[str, dict[str, Any]], Any]] = {}
        for game in slate_games:
            event_id = str(game.espn_event_id)
            try:
                summary = fetch_summary(event_id, session=session)
                built[event_id] = (settlement_rows(summary, game), summary)
            except Exception as exc:  # noqa: BLE001 — one bad event is not the slate
                logger.warning(
                    "ESPN: event %s unusable (%s); trying the rest.", event_id, exc,
                )
        store[key] = built

    built = store[key]
    if not built:
        raise EspnBoxScoreError(
            f"ESPN returned no usable final box score on {game_date}."
        )

    candidates = {eid: rows for eid, (rows, _) in built.items()}
    chosen = select_event_by_players(candidates, expected_players)
    rows, summary = built[chosen]
    logger.info("ESPN: graded against event %s on %s", chosen, game_date)
    return rows, {"espn_event_id": chosen, "summary": summary}
