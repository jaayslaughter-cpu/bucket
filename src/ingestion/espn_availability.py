"""ESPN injuries and rosters -> player availability (RESEARCH_ONLY, NBA).

WHY THIS MATTERS HERE. The late-scratch gap in the readiness audit is blocked
because ``ingestion/inactive_players.py`` reads stats.nba.com, which this
environment's network policy denies. ESPN's public injuries feed is the
reachable substitute for the same question: who is unavailable before tip.

THERE IS NO LINEUPS ENDPOINT, and this module does not pretend otherwise.
Nothing in the documented ESPN API returns a starting five for an NBA game.
What exists is:

  - ``injuries``            league-wide availability status per player
  - ``teams/{id}/roster``   the roster, each athlete carrying a status
  - the box score's ``didNotPlay``, which is AFTER the fact

A projected lineup would be roster minus unavailable, which is an INFERENCE,
not data. ``projected_available`` returns exactly that and is named so the
inference is visible at the call site; it is not called a lineup.

UNKNOWN IS NOT HEALTHY. A player ESPN does not mention gets
``DATA_NOT_AVAILABLE``, never ``AVAILABLE``. Treating silence as fit to play
is how a scratched star reaches a projection.

NO MULTIPLIER IS INVENTED. This module reports status only. An availability
haircut is a modelling choice with an effect size, and asserting one here
would bury it in an ingestion layer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

import requests

from src.ingestion.espn_client import EspnConfig, as_dict, as_list, get_json, site_url

logger = logging.getLogger(__name__)

INJURIES_PATH = "/injuries"
ROSTER_PATH = "/teams/{team_id}/roster"

Availability = Literal[
    "OUT", "DOUBTFUL", "QUESTIONABLE", "PROBABLE", "DAY_TO_DAY",
    "AVAILABLE", "DATA_NOT_AVAILABLE",
]

# Statuses that keep a player off a pre-tip projection outright. DOUBTFUL is
# included deliberately: a doubtful player who plays is a missed row, while a
# doubtful player projected as a starter is a wrong one.
UNAVAILABLE: frozenset[str] = frozenset({"OUT", "DOUBTFUL"})


@dataclass
class InjuryRow:
    player_name: str
    espn_athlete_id: str | None
    espn_team_id: str | None
    team_abbreviation: str | None
    status: Availability
    status_raw: str | None
    detail: str | None
    reported_date: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "PLAYER_NAME": self.player_name,
            "ESPN_ATHLETE_ID": self.espn_athlete_id,
            "ESPN_TEAM_ID": self.espn_team_id,
            "TEAM_ABBREVIATION": self.team_abbreviation,
            "ESPN_INJURY_STATUS": self.status,
            "ESPN_INJURY_STATUS_RAW": self.status_raw,
            "ESPN_INJURY_DETAIL": self.detail,
            "ESPN_INJURY_REPORTED": self.reported_date,
        }


@dataclass
class RosterPlayer:
    player_name: str
    espn_athlete_id: str | None
    jersey: str | None
    position: str | None
    status: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "PLAYER_NAME": self.player_name,
            "ESPN_ATHLETE_ID": self.espn_athlete_id,
            "JERSEY": self.jersey,
            "POSITION": self.position,
            "ROSTER_STATUS": self.status,
        }


@dataclass
class AvailabilityReport:
    status: str
    injuries: list[InjuryRow] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def by_name(self) -> dict[str, InjuryRow]:
        """Lowercased name -> row. Later rows win, and a collision is warned."""
        out: dict[str, InjuryRow] = {}
        for row in self.injuries:
            key = row.player_name.strip().lower()
            if key in out and out[key].status != row.status:
                logger.warning(
                    "espn_availability: two different statuses for %r (%s, %s); "
                    "keeping the later one.",
                    row.player_name, out[key].status, row.status,
                )
            out[key] = row
        return out

    @property
    def unavailable_names(self) -> list[str]:
        return [r.player_name for r in self.injuries if r.status in UNAVAILABLE]


def normalize_status(raw: Any) -> Availability:
    """
    ESPN's status text to a bucket. Unrecognised text is DATA_NOT_AVAILABLE.

    Matching is on substrings because ESPN mixes forms ("Out", "Day-To-Day",
    "Game Time Decision"). Order matters: 'out' is checked last because it is a
    substring of other words, and a bare equality check would miss "Out
    (Knee)".
    """
    if raw is None:
        return "DATA_NOT_AVAILABLE"
    text = str(raw).strip().lower()
    if not text:
        return "DATA_NOT_AVAILABLE"
    if "doubt" in text:
        return "DOUBTFUL"
    if "question" in text or "game time" in text or "gtd" in text:
        return "QUESTIONABLE"
    if "probable" in text:
        return "PROBABLE"
    if "day" in text and "to" in text:
        return "DAY_TO_DAY"
    if text in {"active", "available", "healthy"}:
        return "AVAILABLE"
    if text == "out" or text.startswith("out"):
        return "OUT"
    return "DATA_NOT_AVAILABLE"


def parse_injuries(payload: Any) -> AvailabilityReport:
    """Parse the league-wide injuries payload; tolerates absent keys."""
    if not isinstance(payload, dict):
        return AvailabilityReport(
            status="DATA_NOT_AVAILABLE", notes=["payload is not an object"]
        )

    blocks = as_list(payload.get("injuries"))
    if not blocks:
        return AvailabilityReport(
            status="DATA_NOT_AVAILABLE",
            notes=["no injuries array (an injury-free league looks identical)"],
        )

    rows: list[InjuryRow] = []
    unrecognised: set[str] = set()

    for block in blocks:
        block = as_dict(block)
        team = as_dict(block.get("team"))
        team_id = team.get("id")
        team_abbr = team.get("abbreviation")
        for entry in as_list(block.get("injuries")):
            entry = as_dict(entry)
            athlete = as_dict(entry.get("athlete"))
            name = athlete.get("displayName")
            if not name:
                continue

            raw_status = entry.get("status")
            if isinstance(raw_status, dict):
                raw_status = (
                    raw_status.get("name")
                    or raw_status.get("type")
                    or raw_status.get("description")
                )
            bucket = normalize_status(raw_status)
            if bucket == "DATA_NOT_AVAILABLE" and raw_status:
                unrecognised.add(str(raw_status))

            rows.append(InjuryRow(
                player_name=str(name).strip(),
                espn_athlete_id=str(athlete.get("id")) if athlete.get("id") else None,
                espn_team_id=str(team_id) if team_id else None,
                team_abbreviation=str(team_abbr).upper() if team_abbr else None,
                status=bucket,
                status_raw=str(raw_status) if raw_status else None,
                detail=as_dict(entry.get("type")).get("name") or entry.get("longComment"),
                reported_date=entry.get("date"),
            ))

    notes: list[str] = []
    if unrecognised:
        notes.append(
            f"{len(unrecognised)} unrecognised status string(s): "
            f"{sorted(unrecognised)} — bucketed DATA_NOT_AVAILABLE, not guessed"
        )
        logger.warning(
            "espn_availability: unrecognised status text %s. Extend "
            "normalize_status; they are NOT assumed available.",
            sorted(unrecognised),
        )

    return AvailabilityReport(
        status="OK" if rows else "DATA_NOT_AVAILABLE", injuries=rows, notes=notes
    )


def parse_roster(payload: Any) -> list[RosterPlayer]:
    """Flatten ``athletes[].items[]`` — ESPN nests the roster by position group."""
    players: list[RosterPlayer] = []
    root = as_dict(payload)
    for group in as_list(root.get("athletes")):
        group = as_dict(group)
        # Some leagues return athletes[] flat rather than grouped, so handle both.
        items = as_list(group.get("items")) or [group]
        for item in items:
            item = as_dict(item)
            name = item.get("displayName")
            if not name:
                continue
            players.append(RosterPlayer(
                player_name=str(name).strip(),
                espn_athlete_id=str(item.get("id")) if item.get("id") else None,
                jersey=str(item.get("jersey")) if item.get("jersey") else None,
                position=as_dict(item.get("position")).get("abbreviation"),
                status=as_dict(item.get("status")).get("name"),
            ))
    return players


@dataclass
class ProjectedAvailability:
    """
    Three buckets, never two. ``status`` says whether this is usable at all.

    An earlier version returned (available, withheld) and folded every
    uncertainty into ``available``: a report that failed outright, or a player
    ESPN bucketed DATA_NOT_AVAILABLE, both came back as likely to play. That
    turns a fetch failure into "the whole roster is healthy", which is the
    exact inversion of the rule this module is built on.
    """

    status: str
    available: list[RosterPlayer] = field(default_factory=list)
    withheld: list[RosterPlayer] = field(default_factory=list)
    unknown: list[RosterPlayer] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def projected_available(
    roster: list[RosterPlayer], report: AvailabilityReport
) -> ProjectedAvailability:
    """
    Split a roster into likely available / withheld / unknown. An INFERENCE.

    ABSTAINS ENTIRELY when the report is not OK. Without a usable injury feed
    there is no basis for calling anyone available, so every player lands in
    ``unknown`` and ``status`` is DATA_NOT_AVAILABLE — a caller that ignores
    status gets an empty projection rather than a falsely healthy one.

    With a usable report:
      withheld   ESPN says OUT or DOUBTFUL
      unknown    ESPN has a row but could not bucket it (DATA_NOT_AVAILABLE)
      available  everyone else, INCLUDING players ESPN never mentions, because
                 most of a roster is healthy and absent from an injury feed

    Matching prefers ``espn_athlete_id`` and falls back to a lowercased name.
    Ids are stable; names collide ("Jaylen Brown"), get reformatted, and change.
    """
    if report.status != "OK":
        return ProjectedAvailability(
            status="DATA_NOT_AVAILABLE",
            unknown=list(roster),
            notes=[
                f"injury report status is {report.status!r}, so no player can be "
                "called available; every roster spot is unknown"
            ],
        )

    by_id = {r.espn_athlete_id: r for r in report.injuries if r.espn_athlete_id}
    by_name = report.by_name()

    available: list[RosterPlayer] = []
    withheld: list[RosterPlayer] = []
    unknown: list[RosterPlayer] = []
    matched_by_name = 0

    for player in roster:
        hit = by_id.get(player.espn_athlete_id) if player.espn_athlete_id else None
        if hit is None:
            hit = by_name.get(player.player_name.strip().lower())
            if hit is not None:
                matched_by_name += 1

        if hit is None:
            available.append(player)            # no injury row at all
        elif hit.status in UNAVAILABLE:
            withheld.append(player)
        elif hit.status == "DATA_NOT_AVAILABLE":
            unknown.append(player)              # listed, but unbucketable
        else:
            available.append(player)

    notes: list[str] = []
    if matched_by_name:
        notes.append(
            f"{matched_by_name} player(s) matched by name because no athlete id "
            "was available on one side; ids are preferred"
        )
    return ProjectedAvailability(
        status="OK", available=available, withheld=withheld,
        unknown=unknown, notes=notes,
    )


def fetch_injuries(
    *, config: EspnConfig | None = None, session: requests.Session | None = None
) -> AvailabilityReport:
    """League-wide injury report."""
    cfg = config or EspnConfig()
    payload = get_json(
        site_url(INJURIES_PATH, cfg), config=cfg, session=session
    )
    report = parse_injuries(payload)
    logger.info(
        "espn_availability: %d injury row(s), %d OUT/DOUBTFUL%s",
        len(report.injuries), len(report.unavailable_names),
        f"; notes: {report.notes}" if report.notes else "",
    )
    return report


def fetch_roster(
    espn_team_id: str | int,
    *,
    config: EspnConfig | None = None,
    session: requests.Session | None = None,
) -> list[RosterPlayer]:
    """One team's roster, by ESPN team id (not an NBA team id)."""
    cfg = config or EspnConfig()
    payload = get_json(
        site_url(ROSTER_PATH.format(team_id=espn_team_id), cfg),
        config=cfg, session=session,
    )
    players = parse_roster(payload)
    logger.info("espn_availability: %d roster player(s) for team %s", len(players), espn_team_id)
    return players
