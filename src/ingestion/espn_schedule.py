"""ESPN public scoreboard -> NBA slate (RESEARCH_ONLY, NBA only).

WHY THIS EXISTS. PropLine supplies odds, not the slate: nothing in this repo
answers "which games are on tonight, and when do they tip". ``main.py`` derives
the slate from whatever is already in the feature panel, which cannot include a
game that has not been played. A pre-tip run needs the schedule first.

WHY THE JSON ENDPOINT AND NOT www.espn.com/nba/schedule. The HTML page breaks
on any restyle. ``site.api.espn.com`` returns structured JSON with a stable
documented shape, and takes ``?dates=YYYYMMDD`` for a specific day.

Endpoint and field names follow the shapes documented in
github.com/pseudo-r/public-espn-api (docs/sports/basketball.md,
docs/response_schemas.md). That is a REFERENCE ONLY -- no code or dependency
from it is used here, and the parser tolerates missing keys rather than
assuming the documented shape is guaranteed.

TWO THINGS THAT WILL BITE A CALLER, both handled explicitly:

1. AN ESPN EVENT ID IS NOT AN NBA GAME ID. ESPN uses ids like "401765432";
   this repo's panel keys on zero-padded 10-char NBA ids (see
   ``settlement.boxscore_fetcher.normalize_game_id``). They are different
   namespaces and must never be joined directly. The column is therefore named
   ``ESPN_EVENT_ID``, never ``GAME_ID``, and ``slate_join_keys`` gives the
   (date, home, away) tuple that CAN be joined.

2. ESPN TEAM CODES DIVERGE FROM NBA'S for several teams. Rather than assume
   they match, ``ESPN_TO_NBA_TEAM`` maps the known variants and anything
   unmapped passes through with a warning and is reported in
   ``unmapped_teams`` -- an unrecognised code is surfaced, never guessed into
   the nearest NBA team.

NOT VERIFIED AGAINST THE LIVE ENDPOINT. The environment this was written in
denies outbound CONNECT to site.api.espn.com (and every other data host), so
the parser is exercised against fixtures derived from the documented schema
only. Treat the first live run as the real test.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import requests

from src.utils.timezones import DISPLAY_TZ_NAME, pacific_calendar_date, to_pacific

logger = logging.getLogger(__name__)

SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
)
INJURIES_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
)

ENV_BASE_URL = "PROPIQ_ESPN_BASE_URL"

# ESPN code -> NBA code, for the teams where they differ. Only codes that are
# genuinely ambiguous are listed; an identity mapping is not an entry. Anything
# absent from this map and not already a valid NBA code is reported, not
# rewritten.
ESPN_TO_NBA_TEAM: dict[str, str] = {
    "GS": "GSW",
    "NO": "NOP",
    "NY": "NYK",
    "SA": "SAS",
    "UTAH": "UTA",
    "WSH": "WAS",
    "PHO": "PHX",
    "BRK": "BKN",
    "CHO": "CHA",
}

NBA_TEAM_CODES: frozenset[str] = frozenset({
    "ATL", "BKN", "BOS", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW",
    "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK",
    "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS",
})

# ESPN status states, from the documented status.type.state field.
STATE_PREGAME = "pre"
STATE_IN_PROGRESS = "in"
STATE_FINAL = "post"


class EspnScheduleError(RuntimeError):
    """The schedule could not be fetched. Never raised in place of empty data."""


@dataclass(frozen=True)
class EspnScheduleConfig:
    """No credentials: this endpoint is public and takes none."""

    timeout: int = 30
    retry_attempts: int = 3
    retry_backoff: float = 2.0
    user_agent: str = "PropIQ-Analytics/research (public ESPN JSON)"
    # Overridable so a test or a mirror can point elsewhere without editing code.
    base_url: str = field(
        default_factory=lambda: os.environ.get(ENV_BASE_URL, "").strip()
        or SCOREBOARD_URL
    )


@dataclass(frozen=True)
class SlateGame:
    """One scheduled game. Tip-off is stored UTC and displayed Pacific."""

    espn_event_id: str
    tipoff_utc: datetime | None
    slate_date_pt: date | None
    home_team: str | None
    away_team: str | None
    state: str | None
    status_detail: str | None
    venue: str | None
    venue_city: str | None
    is_neutral_site: bool | None
    home_team_raw: str | None = None
    away_team_raw: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "ESPN_EVENT_ID": self.espn_event_id,
            "TIPOFF_UTC": self.tipoff_utc,
            "SLATE_DATE_PT": self.slate_date_pt,
            "HOME_TEAM": self.home_team,
            "AWAY_TEAM": self.away_team,
            "STATE": self.state,
            "STATUS_DETAIL": self.status_detail,
            "VENUE": self.venue,
            "VENUE_CITY": self.venue_city,
            "IS_NEUTRAL_SITE": self.is_neutral_site,
        }


@dataclass
class SlateResult:
    """What a fetch produced, including what it could not resolve."""

    status: str
    slate_date_pt: date | None = None
    games: list[SlateGame] = field(default_factory=list)
    unmapped_teams: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    timezone_display: str = DISPLAY_TZ_NAME

    @property
    def pregame_only(self) -> list[SlateGame]:
        """Games that have not tipped — the only ones a pre-tip run may use."""
        return [g for g in self.games if g.state == STATE_PREGAME]


def normalize_team(code: Any) -> tuple[str | None, bool]:
    """
    (NBA code, was_recognised). An unknown code is returned as-is, not guessed.

    Returning the raw value rather than None keeps it visible in the output so
    a crosswalk gap shows up as a wrong-looking abbreviation instead of a
    silently dropped game.
    """
    if code is None:
        return None, False
    raw = str(code).strip().upper()
    if not raw:
        return None, False
    if raw in NBA_TEAM_CODES:
        return raw, True
    mapped = ESPN_TO_NBA_TEAM.get(raw)
    if mapped:
        return mapped, True
    return raw, False


def _parse_tipoff(value: Any) -> datetime | None:
    """ESPN dates look like 2025-03-15T00:00Z — not a format fromisoformat takes."""
    if not value:
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        logger.warning("espn_schedule: unparseable date %r — tip-off left null.", value)
        return None
    return parsed


def parse_scoreboard(payload: dict[str, Any]) -> SlateResult:
    """
    Turn a scoreboard payload into games. Tolerates absent keys throughout.

    A malformed event is skipped and counted rather than raising: one bad event
    must not cost the rest of the slate.
    """
    if not isinstance(payload, dict):
        return SlateResult(status="DATA_NOT_AVAILABLE", notes=["payload is not an object"])

    events = payload.get("events")
    if not isinstance(events, list) or not events:
        return SlateResult(
            status="DATA_NOT_AVAILABLE",
            notes=["no events in payload (an empty slate looks identical to this)"],
        )

    games: list[SlateGame] = []
    unmapped: set[str] = set()
    skipped = 0

    for event in events:
        if not isinstance(event, dict):
            skipped += 1
            continue
        event_id = event.get("id")
        if not event_id:
            skipped += 1
            continue

        tipoff = _parse_tipoff(event.get("date"))
        slate_pt = pacific_calendar_date(to_pacific(tipoff)) if tipoff else None

        status = event.get("status") or {}
        stype = status.get("type") if isinstance(status, dict) else {}
        stype = stype if isinstance(stype, dict) else {}

        competitions = event.get("competitions")
        comp = competitions[0] if isinstance(competitions, list) and competitions else {}
        comp = comp if isinstance(comp, dict) else {}

        home = away = home_raw = away_raw = None
        for competitor in comp.get("competitors") or []:
            if not isinstance(competitor, dict):
                continue
            team = competitor.get("team")
            team = team if isinstance(team, dict) else {}
            code, ok = normalize_team(team.get("abbreviation"))
            if code is not None and not ok:
                unmapped.add(code)
            side = str(competitor.get("homeAway") or "").lower()
            if side == "home":
                home, home_raw = code, team.get("abbreviation")
            elif side == "away":
                away, away_raw = code, team.get("abbreviation")

        venue = comp.get("venue")
        venue = venue if isinstance(venue, dict) else {}
        address = venue.get("address")
        address = address if isinstance(address, dict) else {}
        city = address.get("city")

        games.append(SlateGame(
            espn_event_id=str(event_id),
            tipoff_utc=tipoff,
            slate_date_pt=slate_pt,
            home_team=home,
            away_team=away,
            state=str(stype.get("state")) if stype.get("state") else None,
            status_detail=stype.get("shortDetail") or stype.get("detail"),
            venue=venue.get("fullName"),
            venue_city=city,
            # The scoreboard carries no neutral-site flag, so this stays None
            # rather than defaulting to False — an unknown is not a "no".
            is_neutral_site=None,
            home_team_raw=home_raw,
            away_team_raw=away_raw,
        ))

    notes: list[str] = []
    if skipped:
        notes.append(f"{skipped} event(s) skipped: no id or not an object")
    if unmapped:
        notes.append(
            f"{len(unmapped)} team code(s) not in the NBA set or the crosswalk: "
            f"{sorted(unmapped)} — passed through unchanged, not guessed"
        )
        logger.warning(
            "espn_schedule: unrecognised team code(s) %s. Add them to "
            "ESPN_TO_NBA_TEAM; they are NOT mapped to a nearest match.",
            sorted(unmapped),
        )

    dates = {g.slate_date_pt for g in games if g.slate_date_pt}
    return SlateResult(
        status="OK" if games else "DATA_NOT_AVAILABLE",
        slate_date_pt=next(iter(dates)) if len(dates) == 1 else None,
        games=games,
        unmapped_teams=sorted(unmapped),
        notes=notes,
    )


def fetch_scoreboard(
    slate: date | None = None,
    *,
    config: EspnScheduleConfig | None = None,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """
    Raw scoreboard payload for a Pacific calendar day. Raises on failure.

    Never returns a stub or an empty dict in place of a failed request: a
    caller must be able to tell "no games today" from "the fetch failed".
    """
    cfg = config or EspnScheduleConfig()
    params: dict[str, str] = {}
    if slate is not None:
        params["dates"] = slate.strftime("%Y%m%d")

    sess = session or requests.Session()
    last: Exception | None = None
    for attempt in range(1, cfg.retry_attempts + 1):
        try:
            response = sess.get(
                cfg.base_url,
                params=params or None,
                timeout=cfg.timeout,
                headers={"Accept": "application/json", "User-Agent": cfg.user_agent},
            )
            # A 4xx is not worth retrying — the request itself is wrong.
            if 400 <= response.status_code < 500:
                raise EspnScheduleError(
                    f"ESPN scoreboard returned {response.status_code} for "
                    f"{cfg.base_url} params={params}; retrying will not help."
                )
            response.raise_for_status()
            return response.json()
        except EspnScheduleError:
            raise
        except (requests.exceptions.RequestException, ValueError) as exc:
            last = exc
            if attempt < cfg.retry_attempts:
                wait = cfg.retry_backoff ** attempt
                logger.warning(
                    "espn_schedule: attempt %d/%d failed (%s); retrying in %.1fs",
                    attempt, cfg.retry_attempts, exc, wait,
                )
                time.sleep(wait)

    raise EspnScheduleError(
        f"ESPN scoreboard unreachable after {cfg.retry_attempts} attempts: {last}"
    )


def load_slate(
    slate: date | None = None,
    *,
    config: EspnScheduleConfig | None = None,
    session: requests.Session | None = None,
) -> SlateResult:
    """Fetch and parse one Pacific slate day."""
    day = slate or pacific_calendar_date()
    payload = fetch_scoreboard(day, config=config, session=session)
    result = parse_scoreboard(payload)
    if result.slate_date_pt is None:
        result.slate_date_pt = day
    logger.info(
        "espn_schedule: %d game(s) for %s PT (%d pre-tip); unmapped teams: %s",
        len(result.games), day, len(result.pregame_only),
        result.unmapped_teams or "none",
    )
    return result


def slate_join_keys(result: SlateResult) -> list[tuple[date | None, str | None, str | None]]:
    """
    (slate_date_pt, home, away) per game — the ONLY safe join into the panel.

    ESPN event ids live in a different namespace from NBA game ids, so joining
    on an id would silently match nothing (or, worse, something).
    """
    return [(g.slate_date_pt, g.home_team, g.away_team) for g in result.games]
