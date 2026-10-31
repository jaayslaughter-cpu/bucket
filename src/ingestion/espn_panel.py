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
from datetime import date, datetime, timezone
from typing import Any, Iterable, Sequence

import pandas as pd

from src.ingestion.espn_client import EspnError, get_json, site_url
from src.ingestion.espn_game import parse_box_score
from src.ingestion.espn_schedule import normalize_team
from src.utils.ids import ESPN_ID_PREFIX, espn_game_id

logger = logging.getLogger(__name__)

#: What the `source` column says. Matches `scraper_panel.SOURCE_PRECEDENCE`'s
#: convention of naming the feed, not the file.
ESPN_SOURCE = "espn"

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
    dropped_no_athlete_id: int = 0
    dnp_rows: int = 0
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
        ):
            if items:
                out.append(
                    f"SKIPPED, {label}: {', '.join(items[:6])}"
                    + (" ..." if len(items) > 6 else "")
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
    raw = comp.get("date")
    if not raw:
        return None
    try:
        return (
            datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            .astimezone(timezone.utc)
            .date()
        )
    except ValueError:
        return None


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
) -> tuple[pd.DataFrame, EspnPanelReport]:
    """
    Canonical player rows for these events, and what was refused.

    ONE GAME'S FAILURE IS NOT THE NIGHT'S. An `EspnError` on one event is
    recorded and the rest are fetched -- a slate that loses twelve games
    because the first one 404s is the failure mode `espn_client`'s own retry
    budget exists to avoid, and it would be reintroduced here by a bare loop.
    """
    report = EspnPanelReport()
    rows: list[dict[str, Any]] = []
    for event_id in event_ids:
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
    return pd.DataFrame(rows), report


def fetch_player_rows_for_dates(
    days: Sequence[date],
    *,
    session: Any = None,
    config: Any = None,
) -> tuple[pd.DataFrame, EspnPanelReport]:
    """Every final regular-season game across these dates, as one frame."""
    frames: list[pd.DataFrame] = []
    combined = EspnPanelReport()
    for day in days:
        try:
            ids = fetch_event_ids(day, session=session, config=config)
        except EspnError as exc:
            combined.notes.append(f"{day}: scoreboard failed -- {exc}")
            logger.warning("ESPN scoreboard %s failed: %s", day, exc)
            continue
        frame, report = fetch_player_rows(ids, session=session, config=config)
        frames.append(frame)
        combined.games_seen += report.games_seen
        combined.games_parsed += report.games_parsed
        combined.rows_out += report.rows_out
        combined.dropped_no_athlete_id += report.dropped_no_athlete_id
        combined.dnp_rows += report.dnp_rows
        for name in ("skipped_not_final", "skipped_season_type",
                     "skipped_unmapped_team", "skipped_one_sided", "notes"):
            getattr(combined, name).extend(getattr(report, name))
    out = (
        pd.concat(frames, ignore_index=True)
        if any(not f.empty for f in frames) else pd.DataFrame()
    )
    return out, combined
