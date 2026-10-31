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

VERIFIED AGAINST THE LIVE ENDPOINT on 2026-10-30, which this docstring
previously said was impossible. It read: "The environment this was written in
denies outbound CONNECT to site.api.espn.com (and every other data host), so
the parser is exercised against fixtures derived from the documented schema
only. Treat the first live run as the real test."

That was true when written and is not true now -- site.api.espn.com answers
over the agent proxy. The first live run has been made:
``scoreboard?dates=20261028`` returned 12 events, and ``espn_game.fetch_summary``
on one of them returned a FINAL game with 27 box-score rows, 19 of them played,
carrying MIN/PTS/FGM/FGA/FG3M/FG3A/FTM/FTA/OREB/DREB/REB/AST/STL/BLK/TOV/PF --
every column ``scraper_team_games.STAT_SUMS`` sums and every one the panel
reads. The parser handled it unchanged.

WHAT THAT DOES AND DOES NOT UNLOCK, because the distinction is the whole point
of item 1 above. It gives this project a box-score source the DEPLOYED
CONTAINER can reach, where stats.nba.com and cdn.nba.com both refuse a
datacenter IP (403 and a read timeout respectively, on an honest User-Agent,
re-tested the same day). What it does not give is an NBA game id: an ESPN
event id is 401909881, and ``team_game_stats``'s season filter reads character
3 of ``nba_game_id`` and wants ``2`` (``repository.py:709``, ``utils/ids.py``).
ESPN's id has ``1`` there, so the row is not misclassified by that filter --
it is INVISIBLE to it. `db/repository`'s regular-season condition therefore
carries an explicit second branch for the ``espn:`` namespace, which is what
lets these rows fill ``team_game_stats`` as well as the player panel. The
namespace is regular season by construction because `ingestion/espn_panel`
refuses any game whose ESPN season type is not 2.

Nothing anywhere synthesises an NBA id: one that looks like the NBA's and is
not would collide with the real game the moment a licensed or scraped ingest
covers the same night.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import requests

from src.ingestion.espn_client import EspnConfig, EspnError, get_json, site_url
from src.utils.timezones import DISPLAY_TZ_NAME, pacific_calendar_date, to_pacific

logger = logging.getLogger(__name__)

SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
)
INJURIES_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
)

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

# ESPN season types, from the documented season.type field. Corroborated
# against an unrelated project whose NBA ingest runs in CI and annotates its
# own config "season_type: 2  # ESPN: 2 = regular season"
# (danieltomaro13/basketball-modelling @ config.yaml:22) -- evidence, not
# proof: this environment denies outbound CONNECT to site.api.espn.com, so
# nothing here has seen the live field and the NAME is read as well as the
# number (see ``_read_season_type``).
SEASON_TYPE_PRESEASON = 1
SEASON_TYPE_REGULAR = 2
SEASON_TYPE_POSTSEASON = 3

#: Season-type names that mean preseason, normalised to letters only.
PRESEASON_NAMES: frozenset[str] = frozenset({"preseason"})
#: And postseason, which is projected but noted -- see ``exclusion_reason``.
POSTSEASON_NAMES: frozenset[str] = frozenset({"postseason", "playoffs", "playoff"})


def _letters(value: Any) -> str:
    """Lowercase letters only, so "Pre-Season" and "pre season" compare equal."""
    return "".join(c for c in str(value or "").lower() if c.isalpha())


def _as_int(value: Any) -> int | None:
    """ESPN returns ids as strings in some blocks and integers in others."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _read_season_type(
    event: dict[str, Any], payload: dict[str, Any] | None = None
) -> tuple[int | None, str | None, str]:
    """
    ``(type_id, type_name, source)`` for one event. ``source`` is the point.

    EVERY PLAUSIBLE LOCATION, IN ORDER, because the live shape cannot be
    verified from here and a filter that turns on one key would be a filter
    that silently never engages. ``source`` reports which one answered -- the
    same reason ``utils.volume.resolve_state_root`` returns how it chose:
    "preseason, per this event" and "preseason, per the request's season block"
    are different claims, and only one of them is about this game.

    ``absent`` means NOT REPORTED, which is not the same as "not preseason".
    ``parse_scoreboard`` keeps that distinction all the way to the note.
    """
    # EVERY EVENT-LEVEL SOURCE BEFORE THE REQUEST-LEVEL ONE. cubic, PR #9
    # (P1): the flat ``event["seasonType"]`` used to be checked AFTER
    # ``payload.season``, so a request reporting season.type=2 beat an event
    # reporting seasonType=1 and that preseason game stayed projectable --
    # the exact leak this filter exists to close. The payload block describes
    # the REQUEST; an event that speaks for itself always wins.
    candidates: list[tuple[str, Any]] = []
    season = event.get("season")
    if isinstance(season, dict):
        candidates.append(("event.season", season))
    competitions = event.get("competitions")
    comp = competitions[0] if isinstance(competitions, list) and competitions else None
    if isinstance(comp, dict) and isinstance(comp.get("season"), dict):
        candidates.append(("competition.season", comp["season"]))
    if event.get("seasonType") is not None:
        # A flat scalar rather than a block; wrapped so one loop reads them all.
        candidates.append(("event.seasonType", {"type": event["seasonType"]}))
    if isinstance(payload, dict) and isinstance(payload.get("season"), dict):
        # LAST. For a single-date query this is the season type of its events,
        # but it is the request's answer and not the event's -- named as the
        # source so a reader can tell which they got.
        candidates.append(("payload.season", payload["season"]))

    for source, block in candidates:
        type_id = _as_int(block.get("type"))
        name = block.get("slug") or block.get("name")
        if type_id is None and isinstance(block.get("type"), dict):
            # Some ESPN blocks nest it: {"type": {"id": 2, "name": "..."}}.
            inner = block["type"]
            type_id = _as_int(inner.get("id"))
            name = name or inner.get("name") or inner.get("slug")
        if type_id is not None or name:
            return type_id, (str(name) if name else None), source

    return None, None, "absent"


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
    # Derived from the shared site root so PROPIQ_ESPN_SITE_BASE means the
    # same thing here as in every other ESPN module.
    base_url: str = field(default_factory=lambda: site_url("scoreboard"))


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
    # Optional so every existing positional construction still works, and
    # None-by-default so "not reported" is the honest starting state.
    season_type: int | None = None
    season_type_name: str | None = None
    season_type_source: str = "absent"

    @property
    def is_preseason(self) -> bool | None:
        """
        True / False / **None for not reported**, which is not the same as no.

        The name is consulted before the number because the number was, when
        this was written, the one thing no test here could verify against the
        live endpoint. THAT IS NO LONGER TRUE: the live summary for event
        401909881 carries ``header.season.type == 2`` for a regular-season
        game, confirmed 2026-10-30, and `ingestion/espn_panel` now refuses on
        that field -- the whole `espn:` id namespace is regular season
        because of it. The name is still consulted first because it is the
        more legible of the two and costs nothing, not because the number is
        unverified. Corrected in review 2026-10-31.
        """
        name = _letters(self.season_type_name)
        if name and name in PRESEASON_NAMES:
            return True
        if self.season_type is not None:
            return self.season_type == SEASON_TYPE_PRESEASON
        if name:
            # A name that is not preseason is a positive statement that it is
            # not one, even with no number beside it.
            return False
        return None

    @property
    def is_postseason(self) -> bool | None:
        name = _letters(self.season_type_name)
        if name and name in POSTSEASON_NAMES:
            return True
        if self.season_type is not None:
            return self.season_type == SEASON_TYPE_POSTSEASON
        if name:
            return False
        return None

    @property
    def unrecognised_teams(self) -> tuple[str, ...]:
        """Codes that came through ``normalize_team`` unmapped, away first.

        Derived from the stored codes rather than carried as a second flag:
        ``normalize_team`` returns a valid NBA code or the raw value, so
        membership in ``NBA_TEAM_CODES`` IS "was it recognised".
        """
        return tuple(
            code for code in (self.away_team, self.home_team)
            if code is not None and code not in NBA_TEAM_CODES
        )

    @property
    def exclusion_reason(self) -> str | None:
        """
        Why this fixture may not be projected, or None if it may.

        THE ONE READER for that question. A preseason test in the slate
        pipeline and another in the pre-lock arming would be the two-readers
        defect this repository has fixed five times.

        An unreported season type is NOT a reason: see ``parse_scoreboard``.
        """
        if self.is_preseason:
            return (
                "preseason (season type "
                f"{self.season_type if self.season_type is not None else '?'}"
                f", per {self.season_type_source}): the panel this model is fit "
                "on is regular-season only -- scripts/ingest_training_pack.py "
                "filters on IS_REGULAR_SEASON, and preseason minutes and "
                "rotations do not describe the same competition"
            )
        missing = [
            side for side, code in (("home", self.home_team), ("away", self.away_team))
            if code is None
        ]
        if missing:
            return (
                f"no {' or '.join(missing)} team in the event, so it cannot be "
                "joined to the panel on (date, home, away)"
            )
        unknown = self.unrecognised_teams
        if unknown:
            return (
                f"team code(s) {list(unknown)} are not NBA teams and are not in "
                "ESPN_TO_NBA_TEAM, so the DEF_*, DvP and MKT_* columns for this "
                "game have no counterparty to be computed against"
            )
        return None

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
            "SEASON_TYPE": self.season_type,
            "SEASON_TYPE_NAME": self.season_type_name,
            "SEASON_TYPE_SOURCE": self.season_type_source,
            # A filter that drops a game invisibly is not auditable: the CSV
            # and the CLI show why each one was or was not projected.
            "EXCLUSION_REASON": self.exclusion_reason,
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

    @property
    def projectable(self) -> list[SlateGame]:
        """
        The fixtures a slate run may actually project: pre-tip AND no
        ``exclusion_reason``.

        Built ON TOP of ``pregame_only`` rather than beside it, so the pre-tip
        guarantee holds by construction instead of by a second filter that can
        drift from the first.
        """
        return [g for g in self.pregame_only if g.exclusion_reason is None]

    @property
    def excluded(self) -> list[tuple[SlateGame, str]]:
        """Pre-tip games that will NOT be projected, each with its reason."""
        return [
            (g, g.exclusion_reason)
            for g in self.pregame_only
            if g.exclusion_reason is not None
        ]


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
        competitors = comp.get("competitors")
        if competitors is not None and not isinstance(competitors, list):
            # A truthy non-iterable here would raise TypeError and cost the
            # whole slate for one malformed event.
            skipped += 1
            continue
        for competitor in competitors or []:
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

        season_type, season_name, season_source = _read_season_type(event, payload)

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
            season_type=season_type,
            season_type_name=season_name,
            season_type_source=season_source,
        ))

    notes: list[str] = []
    if skipped:
        notes.append(f"{skipped} event(s) skipped: no id or not an object")
    if unmapped:
        notes.append(
            f"{len(unmapped)} team code(s) not in the NBA set or the crosswalk: "
            f"{sorted(unmapped)} — passed through unchanged, not guessed, and "
            "the games they appear in are not projectable"
        )
        logger.warning(
            "espn_schedule: unrecognised team code(s) %s. Add them to "
            "ESPN_TO_NBA_TEAM; they are NOT mapped to a nearest match.",
            sorted(unmapped),
        )

    # WHICH FIXTURES THIS SLATE MAY BE PROJECTED FOR, AND WHY NOT THE REST.
    # Found in production 2026-10-12: five PRESEASON games were armed as an
    # ordinary slate, one of them against ``LON``, a code with no NBA mapping.
    # Nothing abstained on either ground -- every row abstained for want of an
    # artifact, which is a different fault that was hiding this one.
    pregame = [g for g in games if g.state == STATE_PREGAME]
    preseason = [g for g in pregame if g.is_preseason]
    if preseason:
        notes.append(
            f"{len(preseason)} preseason game(s) excluded from the projectable "
            "slate: the panel is regular-season only "
            "(scripts/ingest_training_pack.py filters on IS_REGULAR_SEASON)"
        )
        logger.warning(
            "espn_schedule: %d of %d pre-tip game(s) are PRESEASON and will "
            "not be projected (%s). Preseason minutes and rotations do not "
            "describe the competition this model was fit on.",
            len(preseason), len(pregame),
            ", ".join(
                f"{g.away_team} @ {g.home_team}" for g in preseason
            ) or "no matchups named",
        )

    # NOT REPORTED IS NOT "NOT PRESEASON". These stay projectable on purpose:
    # the season type has never been seen on the live endpoint from here, so
    # excluding an unreported one would mean a single renamed ESPN key turns
    # every slate into an empty board that reads as a quiet night. The cost of
    # the other choice is one noisy board; the cost of this one is a line in
    # the log and a note, which is why it is both.
    unknown = [g for g in pregame if g.is_preseason is None]
    if unknown:
        notes.append(
            f"{len(unknown)} game(s) report no season type (source: "
            f"{sorted({g.season_type_source for g in unknown})}) — projected "
            "anyway rather than blanking the slate on a key that may have "
            "been renamed; a preseason fixture could be among them"
        )
        logger.error(
            "espn_schedule: %d of %d pre-tip game(s) carry NO season type, so "
            "the preseason filter cannot judge them and they are projected "
            "anyway. If this is every game, the payload shape has changed and "
            "src.ingestion.espn_schedule._read_season_type needs the new key.",
            len(unknown), len(pregame),
        )

    postseason = [g for g in pregame if g.is_postseason]
    if postseason:
        # Projected deliberately, and narrower than the preseason case: the
        # training pack filters to IS_REGULAR_SEASON too, so a playoff game is
        # also off-distribution -- but excluding it would blank every April
        # board, which is a larger decision than this filter.
        notes.append(
            f"{len(postseason)} postseason game(s) projected from a "
            "regular-season-only panel — off-distribution, deliberately "
            "included; excluding the playoffs is a separate decision"
        )

    other = [
        g for g in pregame
        if g.exclusion_reason is not None and not g.is_preseason
    ]
    if other:
        notes.append(
            f"{len(other)} pre-tip game(s) excluded for reasons other than the "
            f"season type: {sorted({g.exclusion_reason for g in other})}"
        )

    if pregame and not [g for g in pregame if g.exclusion_reason is None]:
        # A DARK BOARD MUST NEVER READ AS A QUIET NIGHT. There are games on the
        # schedule and none of them may be projected; that belongs in the
        # result, not only in a log line nobody greps.
        notes.append(
            f"NO PROJECTABLE GAMES: all {len(pregame)} pre-tip game(s) were "
            "excluded. The board will be empty for a reason, not for lack of a "
            "slate."
        )
        logger.error(
            "espn_schedule: all %d pre-tip game(s) excluded from the "
            "projectable slate. An empty board tonight is this, not a quiet "
            "schedule.", len(pregame),
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
) -> Any:
    """
    Raw scoreboard payload for a Pacific calendar day. Raises on failure.

    Transport (retry policy, 4xx handling, raise-rather-than-empty) lives in
    ``espn_client.get_json`` so the schedule, game and availability modules
    cannot drift apart on it.
    """
    cfg = config or EspnScheduleConfig()
    params: dict[str, str] = {}
    if slate is not None:
        params["dates"] = slate.strftime("%Y%m%d")

    shared = EspnConfig(
        timeout=cfg.timeout,
        retry_attempts=cfg.retry_attempts,
        retry_backoff=cfg.retry_backoff,
        user_agent=cfg.user_agent,
    )
    try:
        payload = get_json(cfg.base_url, params=params, config=shared, session=session)
    except EspnError as exc:
        raise EspnScheduleError(str(exc)) from exc
    # Returned unchanged: coercing a non-object to {} would present a malformed
    # 200 as an empty slate, and parse_scoreboard has a distinct
    # DATA_NOT_AVAILABLE reason for exactly that case.
    return payload


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
        "espn_schedule: %d game(s) for %s PT (%d pre-tip, %d projectable); "
        "unmapped teams: %s",
        len(result.games), day, len(result.pregame_only),
        len(result.projectable), result.unmapped_teams or "none",
    )
    for game, reason in result.excluded:
        logger.info(
            "espn_schedule: NOT projectable — %s @ %s (%s): %s",
            game.away_team, game.home_team, game.espn_event_id, reason,
        )
    return result


def slate_join_keys(result: SlateResult) -> list[tuple[date | None, str | None, str | None]]:
    """
    (slate_date_pt, home, away) per game — the ONLY safe join into the panel.

    ESPN event ids live in a different namespace from NBA game ids, so joining
    on an id would silently match nothing (or, worse, something).
    """
    return [(g.slate_date_pt, g.home_team, g.away_team) for g in result.games]
