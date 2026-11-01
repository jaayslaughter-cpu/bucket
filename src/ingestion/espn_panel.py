"""
src/ingestion/espn_panel.py — player box scores from ESPN, as the canonical frame.

WHY THIS EXISTS. `team_game_stats` had two writers and a deploy could reach
neither. The licensed BigDataBall workbook is a file an operator downloads by
hand; `scraper_panel`'s parquet comes from a scraper that only runs where
nba.com answers, which is not a datacenter. So the slate died at step [2] on
2026-10-26, 27, 28 and 29 with the same message, and the fix was always "go
and run something on your own machine".

ESPN is the third source and the container can already reach it. Measured
2026-10-30 from the deployment's own egress: `site.api.espn.com` answers,
`cdn.nba.com` returns 403 and `stats.nba.com` times out, both on an honest
User-Agent. `espn_schedule` is already wired for the slate and
`settlement/espn_boxscore` already grades from it -- the box score was parsed
for settlement every night and never once used for ingestion.

THIS IS A MAPPER, NOT A THIRD PANEL DEFINITION. It emits the SAME canonical
lowercase frame that `scraper_panel.to_player_panel` and
`scraper_team_games.aggregate_team_games` already consume, so both work
unchanged and there is one definition of what a player row is. The stat
parsing is `espn_game.parse_box_score`'s, called on the same payload, not
restated: it already splits "11-18" into FGM/FGA and already knows ESPN's
stat labels.

THE TWO ID NAMESPACES, which is the whole design decision.

An ESPN event id is not an NBA game id and an ESPN athlete id is not an NBA
player id. Rows from here therefore carry `espn:`-prefixed ids -- see
`utils/ids.ESPN_ID_PREFIX` for why a prefix and not a synthesised NBA id.

THE PLAYER ID IS THE DANGEROUS HALF, and more dangerous than the game id.
`features/builder.py:518` builds every rolling feature with
`df.groupby("PLAYER_ID")`. If one night of a player's season arrives as
`espn:3032977` and the next as the NBA's `203507`, that is TWO players to the
groupby: each carries half a history, `PTS_L10` is a mean over the wrong
games, and nothing raises. A mixed panel is worse than an empty one, because
an empty one stops the slate and a mixed one scores it.

So mixing is DETECTED, not hoped against: `player_id_namespaces()` in
`db/repository` counts the panel's rows per namespace and the loader refuses
to add to a panel that already holds the other one. One namespace per panel.

WHAT THIS CANNOT DO. No opening spread or total, so `game_market_lines` stays
empty and the five `MKT_*` columns and two `BLOWOUT_*` hinges stay unbuilt --
the same permanent limit the scraper path has, for the same reason: a box
score is not a betting line. Step [2] needs `team_game_stats` OR
`game_market_lines`, so team stats alone clears it.

RESEARCH_ONLY. This moves box-score rows. It prices nothing and places nothing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Iterable, Sequence

import pandas as pd

from src.ingestion.espn_client import CORE_API_BASE, EspnError, get_json, site_url
from src.ingestion.espn_game import parse_box_score
from src.ingestion.espn_schedule import normalize_team
from src.utils.ids import ESPN_ID_PREFIX, espn_game_id
from src.utils.timezones import to_pacific

logger = logging.getLogger(__name__)

#: What the `source` column says. Matches `scraper_panel.SOURCE_PRECEDENCE`'s
#: convention of naming the feed, not the file.
ESPN_SOURCE = "espn"

#: What `aggregate_team_games` writes in a team row's `source` when the rows
#: were summed from these box scores. DISTINCT FROM `scraper_agg`, because the
#: ownership guard reads that column to decide whose row it may overwrite: an
#: ESPN aggregate labelled `scraper_agg` would be taken for a scraper row and
#: silently replaced by the next scraper run. Here rather than in either
#: caller, because `scripts/load_espn_panel` and `scheduler_worker` both need
#: it and a second spelling of it is the two-definitions defect.
ESPN_TEAM_SOURCE = "espn_agg"

#: ESPN's own season-type numbering. 2 is the regular season; 1 is preseason
#: and 3 the postseason. ONLY 2 IS INGESTED, and that is what lets
#: `utils/ids.ESPN_IDS_ARE_REGULAR_SEASON` hold -- a filter downstream accepts
#: the whole `espn:` namespace as regular season because nothing else gets in.
SEASON_TYPE_REGULAR = 2

#: The status ESPN reports for a finished game. A game still in progress has a
#: box score too, and it is a PARTIAL one that looks complete: a player with
#: 8 points at half-time reads exactly like a player who scored 8 all night.
#: Writing one would put a fabricated-looking but merely premature row into a
#: rolling window that cannot tell the difference.
STATUS_FINAL = "STATUS_FINAL"

#: ESPN's stat labels -> the canonical frame's. `espn_game` has already turned
#: "11-18" into FGM/FGA by the time this runs; these are a rename only.
STAT_MAP: dict[str, str] = {
    "PTS": "pts", "FGM": "fgm", "FGA": "fga", "FG3M": "fg3m", "FG3A": "fg3a",
    "FTM": "ftm", "FTA": "fta", "OREB": "oreb", "DREB": "dreb", "REB": "reb",
    "AST": "ast", "STL": "stl", "BLK": "blk", "TOV": "tov", "PF": "pf",
}


class EspnPanelError(RuntimeError):
    """The payload is not the shape this mapper maps."""


@dataclass
class EspnPanelReport:
    """What was taken, and what was refused and why."""

    games_seen: int = 0
    games_parsed: int = 0
    rows_out: int = 0
    skipped_not_final: list[str] = field(default_factory=list)
    skipped_season_type: list[str] = field(default_factory=list)
    skipped_unmapped_team: list[str] = field(default_factory=list)
    skipped_one_sided: list[str] = field(default_factory=list)
    skipped_no_date: list[str] = field(default_factory=list)
    dropped_no_athlete_id: int = 0
    dnp_rows: int = 0
    #: Events the panel already holds, so no summary was fetched for them.
    #: Counted rather than silent: it is the difference between "this date
    #: was already complete" and "this date produced nothing", and the
    #: history backfill's whole progress report rests on telling them apart.
    skipped_already_held: int = 0
    notes: list[str] = field(default_factory=list)

    def lines(self) -> list[str]:
        out = [
            f"games seen            {self.games_seen}",
            f"games parsed          {self.games_parsed}",
            f"player rows out       {self.rows_out}",
            f"  of which DNP        {self.dnp_rows}",
        ]
        for label, items in (
            ("not final", self.skipped_not_final),
            ("not regular season", self.skipped_season_type),
            ("unmapped team code", self.skipped_unmapped_team),
            ("only one team in the payload", self.skipped_one_sided),
            ("no usable tip-off date", self.skipped_no_date),
        ):
            if items:
                out.append(
                    f"SKIPPED, {label}: {', '.join(items[:6])}"
                    + (" ..." if len(items) > 6 else "")
                )
        if self.skipped_already_held:
            out.append(
                f"already in the panel   {self.skipped_already_held} "
                f"event(s), not re-fetched"
            )
        if self.dropped_no_athlete_id:
            out.append(
                f"DROPPED, no athlete id: {self.dropped_no_athlete_id} row(s) "
                f"-- a row with no identity cannot be keyed or rolled"
            )
        out.extend(self.notes)
        return out


def espn_player_id(athlete_id: object) -> str:
    """``3032977`` -> ``espn:3032977``. See the module docstring on why."""
    raw = str(athlete_id or "").strip()
    if not raw:
        raise ValueError("DATA_NOT_AVAILABLE: an ESPN athlete id is required")
    return raw if raw.startswith(ESPN_ID_PREFIX) else f"{ESPN_ID_PREFIX}{raw}"


def _season_label(season_year: object, tipoff: date | None) -> str:
    """
    ``2027`` -> ``"2026-27"``, ESPN's season year being the ENDING one.

    Falls back to the game's own date when the header omits the season, since
    an NBA season starting in October belongs to the year it started in.
    """
    try:
        end = int(str(season_year))
        return f"{end - 1}-{str(end)[-2:]}"
    except (TypeError, ValueError):
        pass
    if tipoff is None:
        raise EspnPanelError(
            "DATA_NOT_AVAILABLE: the payload carries neither a season year nor "
            "a tip-off date, so the season cannot be labelled"
        )
    start = tipoff.year if tipoff.month >= 8 else tipoff.year - 1
    return f"{start}-{str(start + 1)[-2:]}"


def _competition(payload: dict[str, Any]) -> dict[str, Any]:
    header = payload.get("header") or {}
    comps = header.get("competitions") or []
    if not comps:
        raise EspnPanelError(
            "DATA_NOT_AVAILABLE: no header.competitions in the summary payload, "
            "so home/away and the date cannot be read"
        )
    return comps[0] or {}


def _tipoff_date(comp: dict[str, Any]) -> date | None:
    """
    The game's PACIFIC calendar date, which is the panel's unit.

    NOT THE UTC DATE, and the difference is not an edge case. A slate date in
    this project is a Pacific calendar day (`AGENTS.md` §Conventions). ESPN
    stamps the tip in UTC, and a 19:30 PT West Coast tip is 02:30Z THE NEXT
    DAY -- so taking the UTC date files roughly every late game one day
    forward. That shifts it into the wrong rolling window and breaks the
    (date, home, away) tuple `espn_schedule.slate_join_keys` exists to join
    on. Found in review 2026-10-31, after the fixture's single 23:00Z game
    happened to agree in both zones and hid it.
    """
    raw = comp.get("date")
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    return to_pacific(parsed).date()


def parse_summary_payload(
    payload: dict[str, Any],
    event_id: str,
    *,
    report: EspnPanelReport | None = None,
) -> list[dict[str, Any]]:
    """
    One game's summary payload -> canonical player rows, or none with a reason.

    PURE. Takes the payload rather than fetching it, so every refusal below is
    testable against a saved fixture and the suite needs no network.
    """
    rep = report if report is not None else EspnPanelReport()
    rep.games_seen += 1
    comp = _competition(payload)

    status = ((comp.get("status") or {}).get("type") or {}).get("name")
    if status != STATUS_FINAL:
        rep.skipped_not_final.append(f"{event_id} ({status or 'no status'})")
        return []

    season = (payload.get("header") or {}).get("season") or {}
    season_type = season.get("type")
    if season_type != SEASON_TYPE_REGULAR:
        # NOT A PREFERENCE. `utils/ids.ESPN_IDS_ARE_REGULAR_SEASON` lets a SQL
        # filter accept this whole namespace without reading a digit, and that
        # is only true while this refusal holds.
        rep.skipped_season_type.append(f"{event_id} (type {season_type!r})")
        return []

    # [1] Identity: which ESPN team id is home, and what each one is called.
    sides: dict[str, str] = {}
    abbrs: dict[str, str] = {}
    for competitor in comp.get("competitors") or []:
        team = competitor.get("team") or {}
        team_id = str(team.get("id") or "").strip()
        if not team_id:
            continue
        sides[team_id] = str(competitor.get("homeAway") or "").strip().lower()
        abbr, mapped = normalize_team(team.get("abbreviation"))
        if not mapped or not abbr:
            # Never guessed into the nearest NBA team -- `espn_schedule` makes
            # the same refusal for the same reason.
            rep.skipped_unmapped_team.append(
                f"{event_id} ({team.get('abbreviation')!r})"
            )
            return []
        abbrs[team_id] = abbr

    if len(abbrs) != 2 or set(sides.values()) != {"home", "away"}:
        # Elo and every opponent-indexed feature need both sides; a one-sided
        # payload has no opponent to name.
        rep.skipped_one_sided.append(
            f"{event_id} ({len(abbrs)} team(s), sides {sorted(set(sides.values()))})"
        )
        return []

    opponent_of = {tid: abbrs[other] for tid in abbrs for other in abbrs if other != tid}
    game_day = _tipoff_date(comp)
    if game_day is None:
        # EVERY CONSUMER IS DATE-KEYED. `player_game_logs.game_date` is NOT
        # NULL, rolling windows order by it and the slate joins on it, so a
        # row with no date is one the write rejects or the features silently
        # misplace. Refused here, where the reason can be named. Found in
        # review 2026-10-31.
        rep.skipped_no_date.append(str(event_id))
        return []
    season_label = _season_label(season.get("year"), game_day)
    game_id = espn_game_id(event_id)

    # [2] The stat rows, through `espn_game`'s own parser.
    box_rows, notes = parse_box_score(payload, event_id)
    rep.notes.extend(notes)

    out: list[dict[str, Any]] = []
    for row in box_rows:
        team_id = str(row.espn_team_id or "").strip()
        if team_id not in abbrs:
            rep.notes.append(
                f"{event_id}: a box row names team {team_id!r}, which is not "
                f"one of the two competitors; dropped"
            )
            continue
        if not str(row.espn_athlete_id or "").strip():
            rep.dropped_no_athlete_id += 1
            continue

        played = not row.did_not_play
        if not played:
            rep.dnp_rows += 1
        record: dict[str, Any] = {
            "game_id": game_id,
            "game_date": game_day,
            "season": season_label,
            "season_type": "Regular Season",
            "team_abbr": abbrs[team_id],
            "opponent_abbr": opponent_of[team_id],
            "home_away": sides[team_id],
            "player_name": row.player_name,
            "player_id_nba": espn_player_id(row.espn_athlete_id),
            "played": played,
            "source": ESPN_SOURCE,
        }
        minutes = row.stats.get("MIN") if played else None
        # SECONDS, because `scraper_panel` reads `seconds_played` and divides
        # by 60. Converting here rather than teaching that mapper a second
        # unit keeps one definition of what MIN means.
        record["seconds_played"] = (
            float(minutes) * 60.0 if minutes is not None else None
        )
        # A DNP's stats stay NULL, never zero: a zero is a player who played
        # and scored nothing, and a rolling mean cannot tell them apart.
        for espn_name, canonical in STAT_MAP.items():
            record[canonical] = row.stats.get(espn_name) if played else None
        out.append(record)

    rep.games_parsed += 1
    rep.rows_out += len(out)
    return out


def fetch_season_calendar(
    season_year: int, *, session: Any = None, config: Any = None
) -> list[date]:
    """
    Every date ESPN lists as having an NBA game in one season, Pacific.

    WHY A CALENDAR AND NOT A DATE RANGE. Backfilling history by walking
    calendar days asks ESPN about roughly 260 days per season to find the
    ~170 that had basketball, and it cannot tell "there was no game" from
    "the panel has not ingested this day yet" -- so a resumable backfill
    keyed on what the database holds would re-ask every gameless day on
    every run, for ever. ESPN answers the question directly: the scoreboard
    payload carries `leagues[0].calendar`, every date in the season with a
    game, for ONE call. Measured 2026-11-01: `dates=2026` returns 229 dates
    spanning 2025-10-02 to 2026-06-13.

    THE CALENDAR IS NOT REGULAR-SEASON-ONLY, and `seasontype=2` does not
    narrow it -- both forms return the same 229 dates, which include
    preseason in October and the playoffs through June. Passing `seasontype`
    would therefore be a parameter that reads as a filter and filters
    nothing. The per-event guard in `_player_rows_for_event` is what keeps
    non-regular-season games out of the panel, as it already did; the cost
    of the wider calendar is wasted calls on those days, which is why the
    backfill that uses this is operator-triggered rather than unconditional.

    PACIFIC, like every other date in this module. ESPN renders each calendar
    entry as UTC midnight Pacific (`2025-10-02T07:00Z`), so reading the UTC
    date is right by luck for exactly as long as the offset stays at -7;
    across the November DST change it is -8 and `08:00Z` would still be the
    2nd. `to_pacific` makes that explicit rather than lucky.

    SEASON_YEAR IS ESPN'S, which is the season's ENDING year: 2026 is the
    2025-26 season. That is the number its own payload reports in
    `leagues[0].season.year`, and inventing a different convention here
    would be one more thing a caller has to know.
    """
    payload = get_json(
        site_url("scoreboard"),
        params={"dates": str(int(season_year))},
        config=config,
        session=session,
    )
    leagues = payload.get("leagues") if isinstance(payload, dict) else None
    first = (leagues or [{}])[0] if isinstance(leagues, list) and leagues else {}
    entries = first.get("calendar") if isinstance(first, dict) else None
    if not isinstance(entries, list) or not entries:
        raise EspnError(
            f"ESPN returned no season calendar for {season_year}: "
            f"leagues[0].calendar was {type(entries).__name__}. Without it "
            f"there is no list of game dates to back-fill, and walking the "
            f"calendar by hand would re-ask every gameless day for ever."
        )
    days: list[date] = []
    seen: set[date] = set()
    unreadable: list[str] = []
    for entry in entries:
        # A nested calendar (ESPN uses one for week-based sports) is not the
        # flat day list this reads, and a dict's str() would parse as nothing.
        parsed = None
        if isinstance(entry, str):
            try:
                parsed = datetime.fromisoformat(entry.replace("Z", "+00:00"))
            except ValueError:
                parsed = None
        if parsed is None:
            unreadable.append(repr(entry)[:60])
            continue
        day = to_pacific(parsed).date()
        if day not in seen:
            seen.add(day)
            days.append(day)
    # ANY UNREADABLE ENTRY IS A REFUSAL, not a skipped day, and the
    # asymmetry is deliberate. A date dropped here never enters the
    # backfill's work list, so it is never fetched -- and because the
    # planner reports what is MISSING, a calendar quietly short by 29 dates
    # produces `remaining: 0` with 29 nights that have no rows. That is the
    # one failure mode of this whole job that looks like success. A calendar
    # is a short machine-generated list, so an entry that will not parse
    # means the payload shape changed, which is exactly when to stop rather
    # than proceed on a partial reading. Raised by review on #31.
    if unreadable:
        raise EspnError(
            f"ESPN's season calendar for {season_year} held {len(entries)} "
            f"entr(ies) and {len(unreadable)} did not parse as a date (e.g. "
            f"{unreadable[0]}). Refusing a PARTIAL calendar: the dates it "
            f"dropped would never be fetched, and the backfill would report "
            f"itself complete with those nights empty."
        )
    # NO "nothing parsed" BRANCH. `entries` is already known non-empty, and
    # every entry either parses (so `days` grew) or was recorded above (so
    # the refusal fired), which leaves `days` empty unreachable. The first
    # version had that branch and it was dead the moment the partial refusal
    # went in; unreachable code that looks like a guard is worse than none,
    # because a reader counts it as one.
    return sorted(days)


def fetch_regular_season_window(
    season_year: int, *, session: Any = None, config: Any = None
) -> tuple[date, date]:
    """
    The first and last PACIFIC date of one season's regular season.

    WHY THE BACKFILL CANNOT WORK WITHOUT THIS. `fetch_season_calendar`
    returns every date with any NBA game in a season -- preseason from
    early October and the playoffs through June included -- and the
    per-event guard refuses both, so those dates never get panel rows. A
    work list built from the calendar alone therefore reports them missing
    on every run: the backfill never reports itself complete, re-fetches
    them nightly for ever, and the operator is told to wait for a signal
    that cannot arrive. Found in review on #31, against exactly that claim
    in the job's own docstring.

    ESPN ANSWERS IT DIRECTLY rather than being guessed at. The core API's
    season/type resource carries the regular season's own bounds, and one
    call settles a question no heuristic should be deciding. Measured
    2026-11-01: 2026 -> 2025-10-21 to 2026-04-13Z, and 2025 -> 2024-10-22
    to 2025-04-14Z.

    BOTH BOUNDS READ AS PACIFIC DATES, like the calendar. ESPN renders the
    end as 06:59Z, which is 23:59 the previous evening Pacific -- so the
    Pacific end date is the last day games are played, which is what an
    intersection with the calendar wants. Reading the UTC date there would
    extend the window one day past the season.
    """
    year = int(season_year)
    payload = get_json(
        f"{CORE_API_BASE}/v2/sports/basketball/leagues/nba/seasons/"
        f"{year}/types/{SEASON_TYPE_REGULAR}",
        config=config,
        session=session,
    )
    if not isinstance(payload, dict):
        raise EspnError(
            f"ESPN returned {type(payload).__name__} for the {year} regular "
            f"season, not an object."
        )
    reported = payload.get("type")
    if reported is not None and int(reported) != SEASON_TYPE_REGULAR:
        # A payload describing the PRESEASON would hand back a window from
        # early October, which is the exact set of dates this exists to
        # exclude. Checked rather than trusted.
        raise EspnError(
            f"ESPN described season type {reported!r} for the {year} regular "
            f"season; expected {SEASON_TYPE_REGULAR}."
        )
    bounds: list[date] = []
    for key in ("startDate", "endDate"):
        raw = payload.get(key)
        try:
            parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        except (TypeError, ValueError) as exc:
            raise EspnError(
                f"ESPN's {year} regular season has no readable {key} "
                f"({raw!r}). Without both bounds the backfill cannot tell a "
                f"preseason date from a regular-season one, and would report "
                f"itself complete while re-fetching preseason for ever."
            ) from exc
        bounds.append(to_pacific(parsed).date())
    start, end = bounds
    if end < start:
        raise EspnError(
            f"ESPN's {year} regular season ends ({end}) before it starts "
            f"({start}); the window would select nothing."
        )
    return start, end


def fetch_event_ids(
    day: date, *, session: Any = None, config: Any = None
) -> list[str]:
    """Every ESPN event id on a date. Scheduling only; status is read later."""
    payload = get_json(
        site_url("scoreboard"),
        params={"dates": day.strftime("%Y%m%d")},
        config=config,
        session=session,
    )
    return [
        str(event.get("id"))
        for event in (payload.get("events") or [])
        if event.get("id")
    ]


def fetch_player_rows(
    event_ids: Iterable[str],
    *,
    session: Any = None,
    config: Any = None,
    skip_game_ids: "frozenset[str] | set[str]" = frozenset(),
) -> tuple[pd.DataFrame, EspnPanelReport]:
    """
    Canonical player rows for these events, and what was refused.

    ONE GAME'S FAILURE IS NOT THE NIGHT'S. An `EspnError` on one event is
    recorded and the rest are fetched -- a slate that loses twelve games
    because the first one 404s is the failure mode `espn_client`'s own retry
    budget exists to avoid, and it would be reintroduced here by a bare loop.

    ``skip_game_ids`` IS THE UNIT OF WORK, AND IT IS THE EVENT. The history
    backfill used to decide a DATE was done because the panel held any row
    for it, and the two guarantees above make that wrong in both
    directions: this loop records a failed event and keeps the others, so a
    night that lost one game to a 404 was written, marked done, and never
    retried -- a silent hole in a rolling window. Asking by event id heals
    that on the next pass and costs nothing for a night already complete,
    because the summary is never fetched. Found in review on #31.
    """
    report = EspnPanelReport()
    rows: list[dict[str, Any]] = []
    skip = {str(g) for g in skip_game_ids}
    for event_id in event_ids:
        if skip and espn_game_id(event_id) in skip:
            report.skipped_already_held += 1
            continue
        try:
            payload = get_json(
                site_url("summary"),
                params={"event": str(event_id)},
                config=config,
                session=session,
            )
        except EspnError as exc:
            report.notes.append(f"{event_id}: fetch failed -- {exc}")
            logger.warning("ESPN summary %s failed: %s", event_id, exc)
            continue
        try:
            rows.extend(parse_summary_payload(payload, str(event_id), report=report))
        except EspnPanelError as exc:
            report.notes.append(f"{event_id}: {exc}")
            logger.warning("ESPN summary %s unusable: %s", event_id, exc)
        except Exception as exc:  # noqa: BLE001 — one bad game, not the night
            # A nested shape this mapper does not expect raises AttributeError
            # or TypeError, neither of which is an EspnPanelError. Letting one
            # past here would abandon every later game in the window -- the
            # failure-isolation rule `espn_client`'s retry budget already
            # follows. Found in review 2026-10-31.
            report.notes.append(
                f"{event_id}: unreadable payload -- {type(exc).__name__}: {exc}"
            )
            logger.warning(
                "ESPN summary %s raised %s; the other games continue.",
                event_id, type(exc).__name__,
            )
    return pd.DataFrame(rows), report


def fetch_player_rows_for_dates(
    days: Sequence[date],
    *,
    session: Any = None,
    config: Any = None,
    skip_game_ids: "frozenset[str] | set[str]" = frozenset(),
) -> tuple[pd.DataFrame, EspnPanelReport]:
    """
    Every final regular-season game across these dates, as one frame.

    ``skip_game_ids`` is passed straight through to `fetch_player_rows`,
    where the reasoning is: the unit of work is the event, not the date.
    """
    frames: list[pd.DataFrame] = []
    combined = EspnPanelReport()
    for day in days:
        try:
            ids = fetch_event_ids(day, session=session, config=config)
        except EspnError as exc:
            combined.notes.append(f"{day}: scoreboard failed -- {exc}")
            logger.warning("ESPN scoreboard %s failed: %s", day, exc)
            continue
        frame, report = fetch_player_rows(
            ids, session=session, config=config, skip_game_ids=skip_game_ids,
        )
        frames.append(frame)
        combined.games_seen += report.games_seen
        combined.games_parsed += report.games_parsed
        combined.rows_out += report.rows_out
        combined.dropped_no_athlete_id += report.dropped_no_athlete_id
        combined.dnp_rows += report.dnp_rows
        combined.skipped_already_held += report.skipped_already_held
        for name in ("skipped_not_final", "skipped_season_type",
                     "skipped_unmapped_team", "skipped_one_sided", "notes"):
            getattr(combined, name).extend(getattr(report, name))
    out = (
        pd.concat(frames, ignore_index=True)
        if any(not f.empty for f in frames) else pd.DataFrame()
    )
    return out, combined
