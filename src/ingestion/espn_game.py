"""ESPN game summary -> box score rows and play-by-play (RESEARCH_ONLY, NBA).

``summary?event={id}`` carries both the box score and the plays, so one fetch
covers both. Shapes follow docs/response_schemas.md of
github.com/pseudo-r/public-espn-api (reference only; no code or dependency
taken from it).

THE ONE THING THAT WILL SILENTLY CORRUPT A BOX SCORE. Player stats arrive as
two parallel arrays::

    "names":  ["MIN", "FG", "3PT", "FT", "OREB", "DREB", "REB", "AST", ...]
    "stats":  ["36",  "12-24", "4-10", "4-4", "0", "5", "5", "7", ...]

Indexing ``stats`` by a position this module assumed would put rebounds in the
assists column the moment ESPN reorders or adds a stat, and nothing would look
wrong. Every row is therefore built by ZIPPING the header names ESPN sent with
that same block's values. A row whose two arrays differ in length is skipped
and counted, never truncated to fit.

WHAT THIS PLAY-BY-PLAY IS NOT. The existing PBP feature layer
(``scripts/build_pbp_panel.py``) consumes NBA-native event columns:
``actionNumber``, ``actionType``, ``subType``, ``personId``, ``teamId``,
``shotDistance``, ``shotResult``. ESPN plays carry none of those -- they have
an id, a sequence number, a period, a clock, a team id, a score value, and a
free-text ``text`` field. So this CANNOT feed ``PBP_SHOT_DIST_AVG_*`` or
``PBP_RIM_RATE_*`` without parsing distances out of English prose, which is
not done here and is not recommended. Treat ESPN plays as their own source:
scoring runs, period structure, and event ordering.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import requests

from src.ingestion.espn_client import (
    EspnConfig,
    as_dict,
    as_list,
    get_json,
    site_url,
)

logger = logging.getLogger(__name__)

SUMMARY_PATH = "/summary"

# Header labels ESPN uses for the counting stats this project models. Mapped to
# the panel's own names so a caller never has to guess which is which. Absent
# labels are simply not produced -- no zero is invented for a stat ESPN omitted.
ESPN_STAT_TO_PANEL: dict[str, str] = {
    "MIN": "MIN",
    "PTS": "PTS",
    "REB": "REB",
    "AST": "AST",
    "STL": "STL",
    "BLK": "BLK",
    "TO": "TOV",
    "PF": "PF",
    "OREB": "OREB",
    "DREB": "DREB",
    "+/-": "PLUS_MINUS",
}

# "12-24" style made-attempted pairs, expanded into two numeric columns each.
MADE_ATTEMPTED: dict[str, tuple[str, str]] = {
    "FG": ("FGM", "FGA"),
    "3PT": ("FG3M", "FG3A"),
    "FT": ("FTM", "FTA"),
}


@dataclass
class BoxScoreRow:
    espn_event_id: str
    espn_athlete_id: str | None
    player_name: str | None
    espn_team_id: str | None
    did_not_play: bool
    stats: dict[str, float | None] = field(default_factory=dict)
    raw: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "ESPN_EVENT_ID": self.espn_event_id,
            "ESPN_ATHLETE_ID": self.espn_athlete_id,
            "PLAYER_NAME": self.player_name,
            "ESPN_TEAM_ID": self.espn_team_id,
            "DID_NOT_PLAY": self.did_not_play,
        }
        out.update(self.stats)
        return out


@dataclass
class PlayRow:
    espn_event_id: str
    play_id: str | None
    sequence_number: int | None
    period: int | None
    clock_display: str | None
    espn_team_id: str | None
    text: str | None
    score_value: int | None
    scoring_play: bool | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "ESPN_EVENT_ID": self.espn_event_id,
            "PLAY_ID": self.play_id,
            "SEQUENCE_NUMBER": self.sequence_number,
            "PERIOD": self.period,
            "CLOCK_DISPLAY": self.clock_display,
            "ESPN_TEAM_ID": self.espn_team_id,
            "TEXT": self.text,
            "SCORE_VALUE": self.score_value,
            "SCORING_PLAY": self.scoring_play,
        }


@dataclass
class GameSummary:
    status: str
    espn_event_id: str
    box_score: list[BoxScoreRow] = field(default_factory=list)
    plays: list[PlayRow] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def played(self) -> list[BoxScoreRow]:
        """Rows for players who actually appeared."""
        return [r for r in self.box_score if not r.did_not_play]

    @property
    def inactive_names(self) -> list[str]:
        """Who was on the box score but did not play — the scratch signal."""
        return [r.player_name for r in self.box_score if r.did_not_play and r.player_name]


def _number(text: Any) -> float | None:
    """A stat cell as a float, or None. '+8' and '-3' are signed; '--' is absent."""
    if text is None:
        return None
    raw = str(text).strip().replace("+", "")
    if raw in {"", "-", "--", "—"}:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _split_made_attempted(text: Any) -> tuple[float | None, float | None]:
    """'12-24' -> (12.0, 24.0). Anything else -> (None, None), never a guess."""
    if text is None:
        return None, None
    parts = str(text).strip().split("-")
    if len(parts) != 2:
        return None, None
    return _number(parts[0]), _number(parts[1])


def _stats_from_block(names: list[Any], values: list[Any]) -> dict[str, float | None]:
    """Zip ESPN's own header names with that block's values."""
    out: dict[str, float | None] = {}
    for label, value in zip(names, values, strict=True):
        key = str(label).strip()
        if key in MADE_ATTEMPTED:
            made_col, att_col = MADE_ATTEMPTED[key]
            made, attempted = _split_made_attempted(value)
            out[made_col] = made
            out[att_col] = attempted
        elif key in ESPN_STAT_TO_PANEL:
            out[ESPN_STAT_TO_PANEL[key]] = _number(value)
    return out


def _player_groups(payload: dict[str, Any]) -> list[tuple[Any, dict[str, Any]]]:
    """
    (fallback team id, player group) pairs, from EITHER documented shape.

    The reference documents two layouts and this module cannot reach the live
    endpoint to settle which the summary returns:

      docs/response_schemas.md "Game Summary"  boxscore.teams[].players[]
      docs/response_schemas.md "CDN Game Package"  boxscore: {teams, players}

    Picking one and being wrong yields zero box-score rows on every response,
    silently. Both are walked instead, and a group reached twice is harmless
    because ``parse_box_score`` keys output on (athlete id, stat block).
    """
    box = as_dict(payload.get("boxscore"))
    groups: list[tuple[Any, dict[str, Any]]] = []

    # Shape A: players nested inside each team block.
    for team_block in as_list(box.get("teams")):
        team_block = as_dict(team_block)
        team_id = as_dict(team_block.get("team")).get("id")
        for group in as_list(team_block.get("players")):
            groups.append((team_id, as_dict(group)))

    # Shape B: players as a sibling of teams, each group carrying its own team.
    for group in as_list(box.get("players")):
        group = as_dict(group)
        groups.append((as_dict(group.get("team")).get("id"), group))

    return groups


def parse_box_score(payload: dict[str, Any], event_id: str) -> tuple[list[BoxScoreRow], list[str]]:
    """Per-player rows from whichever box-score layout the payload uses."""
    rows: list[BoxScoreRow] = []
    notes: list[str] = []
    mismatched = 0
    seen: set[str] = set()

    for team_id, player_block in _player_groups(payload):
        block_team = as_dict(player_block.get("team")).get("id") or team_id
        for stat_block in as_list(player_block.get("statistics")):
            stat_block = as_dict(stat_block)
            names = as_list(stat_block.get("names"))
            for athlete_entry in as_list(stat_block.get("athletes")):
                athlete_entry = as_dict(athlete_entry)
                athlete = as_dict(athlete_entry.get("athlete"))
                values = as_list(athlete_entry.get("stats"))
                dnp = bool(athlete_entry.get("didNotPlay"))

                athlete_id = str(athlete.get("id")) if athlete.get("id") else None
                if athlete_id is not None:
                    # Both layouts can reach the same group; an athlete with
                    # no id cannot be deduped and is kept, since dropping a
                    # row is worse than repeating one.
                    if athlete_id in seen:
                        continue
                    seen.add(athlete_id)

                stats: dict[str, float | None] = {}
                # An empty stats array is legitimate ONLY for a DNP. For a
                # player who appeared it means the row carries no numbers
                # while still being counted among those who played, which
                # would quietly understate everything downstream.
                if len(values) != len(names) and not (dnp and not values):
                    # Truncating to the shorter array would shift every
                    # column after the first divergence.
                    mismatched += 1
                    logger.warning(
                        "espn_game %s: %s has %d stat value(s) against %d "
                        "header name(s); stats dropped rather than aligned "
                        "by position.",
                        event_id, athlete.get("displayName"),
                        len(values), len(names),
                    )
                elif values:
                    stats = _stats_from_block(names, values)

                rows.append(BoxScoreRow(
                    espn_event_id=event_id,
                    espn_athlete_id=athlete_id,
                    player_name=athlete.get("displayName"),
                    espn_team_id=str(block_team) if block_team else None,
                    did_not_play=dnp,
                    stats=stats,
                    raw={str(n): str(v) for n, v in zip(names, values, strict=False)},
                ))

    if mismatched:
        notes.append(
            f"{mismatched} player row(s) had mismatched names/stats lengths; "
            "their stats are empty rather than positionally aligned"
        )
    return rows, notes


def parse_plays(payload: dict[str, Any], event_id: str) -> list[PlayRow]:
    """Play-by-play from ``plays[]``. See the module docstring on its limits."""
    plays: list[PlayRow] = []
    for play in as_list(payload.get("plays")):
        play = as_dict(play)
        clock = as_dict(play.get("clock"))
        period = as_dict(play.get("period"))
        seq = play.get("sequenceNumber")
        try:
            seq_int = int(seq) if seq is not None else None
        except (TypeError, ValueError):
            seq_int = None
        plays.append(PlayRow(
            espn_event_id=event_id,
            play_id=str(play.get("id")) if play.get("id") else None,
            sequence_number=seq_int,
            period=period.get("number") if isinstance(period.get("number"), int) else None,
            clock_display=clock.get("displayValue"),
            espn_team_id=str(as_dict(play.get("team")).get("id") or "") or None,
            text=play.get("text"),
            score_value=play.get("scoreValue") if isinstance(play.get("scoreValue"), int) else None,
            scoring_play=(
                bool(play.get("scoringPlay")) if play.get("scoringPlay") is not None else None
            ),
        ))
    return plays


def parse_summary(payload: Any, event_id: str) -> GameSummary:
    """Parse a summary payload. Absent sections abstain rather than raising."""
    if not isinstance(payload, dict):
        return GameSummary(
            status="DATA_NOT_AVAILABLE", espn_event_id=event_id,
            notes=["payload is not an object"],
        )

    box, notes = parse_box_score(payload, event_id)
    plays = parse_plays(payload, event_id)
    if not box and not plays:
        notes.append("payload carried neither a box score nor plays")
        return GameSummary(
            status="DATA_NOT_AVAILABLE", espn_event_id=event_id, notes=notes
        )
    if not box:
        notes.append("no box score in payload (expected before tip-off)")
    if not plays:
        notes.append("no plays in payload (expected before tip-off)")

    return GameSummary(
        status="OK", espn_event_id=event_id,
        box_score=box, plays=plays, notes=notes,
    )


def fetch_summary(
    espn_event_id: str,
    *,
    config: EspnConfig | None = None,
    session: requests.Session | None = None,
) -> GameSummary:
    """
    Box score and plays for one ESPN event id.

    The id must be an ESPN event id, not an NBA game id — see
    ``espn_schedule`` on why the two must never be interchanged.
    """
    cfg = config or EspnConfig()
    payload = get_json(
        site_url(SUMMARY_PATH, cfg),
        params={"event": str(espn_event_id)},
        config=cfg,
        session=session,
    )
    summary = parse_summary(payload, str(espn_event_id))
    logger.info(
        "espn_game %s: %d box row(s) (%d played, %d DNP), %d play(s)%s",
        espn_event_id, len(summary.box_score), len(summary.played),
        len(summary.inactive_names), len(summary.plays),
        f"; notes: {summary.notes}" if summary.notes else "",
    )
    return summary
