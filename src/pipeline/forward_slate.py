"""Rows for a slate that has not been played yet.

RESEARCH_ONLY. Produces rows to PROJECT, never a stat, a line or a wager.

THE PROBLEM THIS SOLVES (readiness item O1). ``repository.load_player_panel``
reads ``PlayerGameLog`` — completed box scores — and ``main._filter_to_slate``
keeps only rows whose GAME_DATE equals the slate date. At 09:00 PT, before any
tip, that intersection is EMPTY, so the deployed worker took the
``success_no_data`` branch and exited 0 every single day. It did not look
broken. ``main.py`` said so itself:

    the player game log holds completed games, so a future slate will not
    appear here until those games are played and ingested

So a forward slate needs rows that do not exist yet: one per (player, scheduled
game), carrying the identity the feature builder needs and NO box-score stat.

WHY THE LINEUP COMES FROM THE PANEL AND NOT FROM A ROSTER ENDPOINT.
``espn_availability.fetch_roster`` exists and would give a team's listed
players — but keyed by ESPN athlete name, which then has to be matched to the
NBA panel's PLAYER_NAME. That is the crosswalk this repository still does not
have: ``src/ingestion/id_crosswalk.py`` is named as the fix by three modules
and was never written (readiness item O5). Building the forward slate on a
roster fetch would make every slate depend on a fuzzy name match that does not
exist, and a mismatch would silently drop a player.

The panel already answers the question with data this project verified: who
has actually appeared for this team in its last few games, with their real
PLAYER_ID. That is a narrower claim than a roster — it misses a player
returning from a long absence, and it is stated rather than hidden — but it is
a TRUE claim, and the ESPN injury feed already removes the ruled-out through
``src/pipeline/scratches.py``.

LEAKAGE. A forward row carries every box-score column as NaN. That is what
makes it safe: the rolling features are ``.shift(1)`` within player-season, so
the forward row reads its own prior REAL games and has nothing of its own to
leak. Verified rather than assumed — ``tests/test_forward_slate.py`` builds a
panel plus a forward row, checks the forward row's ``{stat}_L5`` equals the
mean of the five real games before it, and runs ``assert_no_lookahead`` over
the result.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Columns a forward row must carry for the feature builder to work on it.
IDENTITY_COLUMNS = (
    "PLAYER_ID", "PLAYER_NAME", "GAME_ID", "GAME_DATE", "SEASON",
    "TEAM_ABBREVIATION", "OPPONENT_ABBREVIATION", "IS_HOME",
)
#: Every column that describes what HAPPENED. All NaN on a forward row.
BOX_SCORE_COLUMNS = (
    "MIN", "PTS", "REB", "AST", "PRA", "FG3M", "STL", "BLK", "TOV",
    "FGA", "FTA", "FGM", "FTM", "FG3A", "OREB", "DREB", "PF", "PLUS_MINUS",
)
#: How many of a team's most recent games define "has been playing".
DEFAULT_LOOKBACK_GAMES = 5
#: Marks a row this module created. Nothing else in the tree writes it.
FORWARD_FLAG = "IS_FORWARD_SLATE"


@dataclass
class ForwardSlateResult:
    """The combined panel, and an account of what could not be built."""

    panel: pd.DataFrame
    status: str = "OK"
    slate_date: str | None = None
    n_games: int = 0
    n_forward_rows: int = 0
    teams_without_history: list[str] = field(default_factory=list)
    games_skipped: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "slate_date": self.slate_date,
            "n_games": self.n_games,
            "n_forward_rows": self.n_forward_rows,
            "teams_without_history": list(self.teams_without_history),
            "games_skipped": list(self.games_skipped),
            "notes": list(self.notes),
        }


def _recent_team_players(
    panel: pd.DataFrame,
    team: str,
    *,
    lookback_games: int,
) -> pd.DataFrame:
    """
    One row per player who appeared for ``team`` in its last ``lookback_games``.

    "Appeared" means a recorded MIN above zero. A DNP row has a player on the
    bench, not on the floor, and projecting one would put a name on the board
    that the box score already said did not play.
    """
    if "TEAM_ABBREVIATION" not in panel.columns:
        return panel.iloc[0:0]
    team_rows = panel.loc[panel["TEAM_ABBREVIATION"].astype(str) == str(team)]
    if team_rows.empty:
        return team_rows

    if "GAME_DATE" in team_rows.columns:
        recent_dates = (
            pd.to_datetime(team_rows["GAME_DATE"], errors="coerce")
            .dropna()
            .drop_duplicates()
            .sort_values()
            .tail(int(lookback_games))
        )
        dates = pd.to_datetime(team_rows["GAME_DATE"], errors="coerce")
        team_rows = team_rows.loc[dates.isin(set(recent_dates))]

    if "MIN" in team_rows.columns:
        minutes = pd.to_numeric(team_rows["MIN"], errors="coerce")
        team_rows = team_rows.loc[minutes.notna() & (minutes > 0)]

    if team_rows.empty or "PLAYER_ID" not in team_rows.columns:
        return team_rows
    # The player's LATEST appearance carries the season and name to use.
    ordered = team_rows.sort_values(
        [c for c in ("GAME_DATE", "GAME_ID") if c in team_rows.columns],
        kind="mergesort",
    )
    return ordered.drop_duplicates(subset=["PLAYER_ID"], keep="last")


def _forward_row(
    source: pd.Series,
    *,
    game_id: str,
    game_date: pd.Timestamp,
    team: str,
    opponent: str,
    is_home: bool,
    is_neutral_site: bool | None,
    columns: Sequence[str],
) -> dict[str, Any]:
    """One (player, scheduled game) row: identity kept, every stat blanked."""
    row: dict[str, Any] = {c: np.nan for c in columns}
    row.update({
        "PLAYER_ID": source.get("PLAYER_ID"),
        "PLAYER_NAME": source.get("PLAYER_NAME"),
        "GAME_ID": str(game_id),
        "GAME_DATE": game_date,
        "SEASON": source.get("SEASON"),
        "TEAM_ABBREVIATION": team,
        "OPPONENT_ABBREVIATION": opponent,
        "IS_HOME": bool(is_home),
        FORWARD_FLAG: True,
    })
    if "IS_NEUTRAL_SITE" in columns:
        row["IS_NEUTRAL_SITE"] = (
            bool(is_neutral_site) if is_neutral_site is not None else False
        )
    for stat in BOX_SCORE_COLUMNS:
        if stat in row:
            row[stat] = np.nan
    return row


def build_forward_rows(
    panel: pd.DataFrame,
    games: Iterable[Any],
    *,
    slate_date: date | str,
    lookback_games: int = DEFAULT_LOOKBACK_GAMES,
) -> ForwardSlateResult:
    """
    Rows for each scheduled game, from the panel's own recent appearances.

    ``games`` are ``espn_schedule.SlateGame`` objects — read by attribute so
    this module does not import the ingestion layer, and so any object with
    ``game_id``/``home_team``/``away_team`` works in a test.

    A game whose teams have no recent history in the panel produces NO rows and
    is named in ``games_skipped``. An empty slate is a valid off-day, not a
    failure.
    """
    day = pd.Timestamp(str(slate_date)).normalize()
    result = ForwardSlateResult(panel=panel, slate_date=str(day.date()))

    game_list = list(games)
    result.n_games = len(game_list)
    if panel is None or panel.empty:
        result.status = "DATA_NOT_AVAILABLE"
        result.notes.append(
            "empty panel — a forward row's features come from the player's own "
            "prior games, so there is nothing to project from"
        )
        return result
    if not game_list:
        result.status = "EMPTY"
        result.notes.append("no scheduled games — a valid off-day")
        return result

    columns = list(panel.columns)
    if FORWARD_FLAG not in columns:
        columns.append(FORWARD_FLAG)
    missing_identity = [c for c in ("PLAYER_ID", "GAME_DATE") if c not in panel.columns]
    if missing_identity:
        result.status = "DATA_NOT_AVAILABLE"
        result.notes.append(f"panel missing {missing_identity}")
        return result

    existing_ids = (
        set(panel["GAME_ID"].astype(str)) if "GAME_ID" in panel.columns else set()
    )
    rows: list[dict[str, Any]] = []
    no_history: set[str] = set()

    for game in game_list:
        game_id = str(getattr(game, "game_id", None) or getattr(game, "espn_event_id", "") or "")
        home = getattr(game, "home_team", None)
        away = getattr(game, "away_team", None)
        neutral = getattr(game, "is_neutral_site", None)

        if not game_id or not home or not away:
            result.games_skipped.append({
                "game_id": game_id or "?",
                "reason": "game id or a team abbreviation is missing",
            })
            continue
        if game_id in existing_ids:
            # Already played and ingested, or already projected. Either way the
            # real row wins: a synthetic duplicate would double every join.
            result.games_skipped.append({
                "game_id": game_id,
                "reason": "already in the panel — the real row wins",
            })
            continue

        made = 0
        for team, opponent, is_home in ((home, away, True), (away, home, False)):
            players = _recent_team_players(panel, team, lookback_games=lookback_games)
            if players.empty:
                no_history.add(str(team))
                continue
            for _, source in players.iterrows():
                rows.append(_forward_row(
                    source, game_id=game_id, game_date=day, team=str(team),
                    opponent=str(opponent), is_home=is_home,
                    is_neutral_site=neutral, columns=columns,
                ))
                made += 1
        if made == 0:
            result.games_skipped.append({
                "game_id": game_id,
                "reason": f"neither {home} nor {away} has recent history in the panel",
            })

    result.teams_without_history = sorted(no_history)
    if not rows:
        result.status = "DATA_NOT_AVAILABLE"
        result.notes.append(
            "no forward rows built — every scheduled team is absent from the "
            "panel's recent games, so there is no history to project from"
        )
        return result

    forward = pd.DataFrame(rows, columns=columns)
    base = panel.copy()
    if FORWARD_FLAG not in base.columns:
        base[FORWARD_FLAG] = False
    else:
        base[FORWARD_FLAG] = base[FORWARD_FLAG].fillna(False)
    combined = pd.concat([base, forward], ignore_index=True)
    combined["GAME_DATE"] = pd.to_datetime(combined["GAME_DATE"], errors="coerce")

    result.panel = combined
    result.n_forward_rows = len(forward)
    result.notes.append(
        f"{len(forward)} forward row(s) across {result.n_games - len(result.games_skipped)} "
        f"game(s), from each team's last {lookback_games} game(s) in the panel"
    )
    logger.info(
        "Forward slate %s: %d row(s) for %d game(s); %d game(s) skipped%s",
        result.slate_date, result.n_forward_rows,
        result.n_games - len(result.games_skipped), len(result.games_skipped),
        f"; no panel history for {result.teams_without_history}"
        if result.teams_without_history else "",
    )
    return result


def attach_forward_slate(
    panel: pd.DataFrame,
    *,
    slate_date: date | str,
    games: Iterable[Any] | None = None,
    lookback_games: int = DEFAULT_LOOKBACK_GAMES,
) -> ForwardSlateResult:
    """
    Fetch the day's schedule if it was not supplied, then build the rows.

    The fetch is best-effort by design: a denied or unreachable ESPN endpoint
    returns the panel unchanged with a named reason, so the slate run behaves
    exactly as it did before this module existed rather than failing. Passing
    ``games`` skips the network entirely, which is how the tests run.
    """
    if games is not None:
        return build_forward_rows(
            panel, games, slate_date=slate_date, lookback_games=lookback_games
        )

    try:
        from src.ingestion.espn_schedule import load_slate
        from src.utils.timezones import parse_slate_date

        target = (
            slate_date if isinstance(slate_date, date)
            else parse_slate_date(str(slate_date))
        )
        slate = load_slate(target)
    except Exception as exc:  # noqa: BLE001 — the network is optional here
        result = ForwardSlateResult(panel=panel, status="DATA_NOT_AVAILABLE",
                                    slate_date=str(slate_date))
        result.notes.append(f"schedule unavailable: {exc}")
        logger.warning(
            "Forward slate skipped — the schedule could not be fetched (%s). The "
            "run continues on completed games only, which for a future slate "
            "means no rows.", exc,
        )
        return result

    if getattr(slate, "status", None) != "OK":
        result = ForwardSlateResult(panel=panel, status="DATA_NOT_AVAILABLE",
                                    slate_date=str(slate_date))
        result.notes.append(f"schedule status {getattr(slate, 'status', '?')}")
        result.notes.extend(list(getattr(slate, "notes", []) or [])[:3])
        return result

    # PRE-TIP GAMES ONLY. A game already under way or finished is not something
    # to project: its box score is the answer, and the panel is where it goes.
    pregame = list(getattr(slate, "pregame_only", None) or [])
    out = build_forward_rows(
        panel, pregame, slate_date=slate_date, lookback_games=lookback_games
    )
    unmapped = list(getattr(slate, "unmapped_teams", []) or [])
    if unmapped:
        out.notes.append(f"schedule had unmapped team code(s): {unmapped}")
    out.notes.append(
        f"{len(pregame)} pre-tip game(s) of {len(getattr(slate, 'games', []) or [])} "
        "on the schedule"
    )
    return out
