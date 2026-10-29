"""
src/ingestion/scraper_team_games.py — team-game rows from player box scores.

WHY THIS EXISTS. `team_game_stats` has had exactly one writer: the licensed
BigDataBall workbook, which is a file an operator downloads with their own
credentials onto their own machine. That makes it the only step in this project
that cannot run in the container, and the slate refuses without it --
`main.py:393` raises `MarketFramesUnavailable` when both the team frame and the
market frame are empty, which is what happened on 2026-10-26, 10-27 and 10-28.

A team's box score is the sum of its players' box scores, and
`src/ingestion/scraper_panel.py` already brings player box scores in from the
multi-source scraper's parquet. So the team frame can be derived from data
already arriving, and `models.TeamGameStat`'s own docstring anticipates it:
"Populated from BigDataBall exports (licensed) and/or PBP-derived box scores."

WHAT IT RECOVERS, COUNTED RATHER THAN ASSERTED. Against the 40 features a PTS
spec names: the four Elo columns (`team_strength._normalize_team_games`
requires only game id, date, team and points) and the two `DEF_*` columns
(`defense.REQUIRED_TEAM_COLS` is the same four, with the per-100 rates coming
from reb/ast/fg3/fga/tov). Six columns, taking a buildable set from 27 to 33.

WHAT IT CANNOT RECOVER: the five `MKT_*` columns and the two `BLOWOUT_*`
hinges, seven in all. Those come from the opening spread and total -- betting
lines, which no box score contains. A board built on this is therefore a board
without market context, and that is a permanent property of the source rather
than a gap to be filled later by the same route.

WHAT IT DELIBERATELY DOES NOT COMPUTE. `poss`, `pace`, `off_eff` and `def_eff`
are left NULL. Nothing reads them: `builder.attach_team_pace` owns possessions
(`POSSESSIONS_EST = FGA - OREB + TOV + 0.44 * FTA`) and `defense.py` says so in
as many words -- "already in attach_team_pace; it is not repeated here". A
second possessions formula here would be the two-definitions defect that
`attach_pra_from_components` was written to describe. `rest_days` is likewise
absent from `repository.TEAM_GAME_STAT_COLS`, so it is never read back either.

THE ONE THING THAT IS SIMPLY LOST. `is_neutral_site`. BigDataBall carries a
venue column and sets it from `_venue == "N"`; the scraper's payload has no
equivalent. It is written False here, which is wrong for about six games a
season -- the NBA Cup final and the global games -- and the consequence is
specific rather than vague: `fatigue_logic` applies its altitude tax on
`IS_HOME == False`, and this flag is the only thing that stops a neutral-site
game in Denver or Utah taxing BOTH teams. Bounded, named, and not silently
papered over.

RESEARCH_ONLY. This moves team-game rows. It prices nothing, places nothing
and grades nothing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import pandas as pd

from src.ingestion.scraper_panel import DEFAULT_SEASON_TYPES
from src.utils.ids import normalise_game_id

logger = logging.getLogger(__name__)

#: This writer's provenance. Never "bigdataball": the whole point of the
#: precedence below is that the two stay distinguishable.
TEAM_SOURCE = "scraper_agg"

#: The source whose rows this writer will not touch. Licensed, richer (it has
#: the venue and the source's own pace figures), and therefore authoritative
#: wherever it exists.
AUTHORITATIVE_SOURCE = "bigdataball"

#: Player column -> team column. The fifteen box-score fields
#: `TeamGameStat` holds, each a plain sum. No formula, no estimate.
STAT_SUMS: dict[str, str] = {
    "pts": "points",
    "fgm": "fg",
    "fga": "fga",
    "fg3m": "fg3",
    "fg3a": "fg3a",
    "ftm": "ft",
    "fta": "fta",
    "oreb": "oreb",
    "dreb": "dreb",
    "reb": "reb",
    "ast": "ast",
    "stl": "stl",
    "blk": "blk",
    "tov": "tov",
    "pf": "pf",
}

#: Columns needed to aggregate at all. A frame missing one is refused rather
#: than part-aggregated.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "game_id", "game_date", "team_abbr", "opponent_abbr", "home_away",
    "player_name", "player_id_nba", "season_type", *STAT_SUMS,
)

#: Rows that are NOT players. A "Team Totals" line summed with the players
#: DOUBLES every figure and looks entirely plausible afterwards -- the sharpest
#: failure this module can have. The scraper's Basketball-Reference parser
#: already drops these two (basketball_reference.py), but inheriting a guarantee
#: is not the same as holding one: a future source, or a hand-made frame, would
#: not.
NON_PLAYER_ROW_NAMES: frozenset[str] = frozenset({"team totals", "reserves"})

#: Fewest player rows a believable team-game has. An NBA team dresses far more,
#: and a total summed from four ingested players is arithmetically fine and
#: factually wrong -- with nothing downstream able to tell. Eight is well below
#: any real roster and well above a partial ingest.
MIN_PLAYERS_PER_TEAM_GAME = 8

#: A game needs both sides. One team's row without its opponent gives Elo a
#: result it cannot score and `defense` an opponent it cannot find.
TEAMS_PER_GAME = 2


class TeamAggregateError(RuntimeError):
    """The frame is not the shape this aggregator aggregates."""


@dataclass
class TeamAggregateReport:
    """What was built, and what was refused and why."""

    rows_in: int = 0
    team_games_out: int = 0
    dropped_non_player_rows: int = 0
    dropped_no_identity: int = 0
    dropped_wrong_season_type: int = 0
    skipped_short_roster: list[str] = field(default_factory=list)
    skipped_one_sided_games: list[str] = field(default_factory=list)
    declined_to_authoritative: list[str] = field(default_factory=list)

    def lines(self) -> list[str]:
        out = [
            f"player rows in            {self.rows_in}",
            f"  non-player rows         -{self.dropped_non_player_rows}",
            f"  no player identity      -{self.dropped_no_identity}",
            f"  wrong season type       -{self.dropped_wrong_season_type}",
            f"team-games out            {self.team_games_out}",
        ]
        if self.skipped_short_roster:
            out.append(
                f"SKIPPED, fewer than {MIN_PLAYERS_PER_TEAM_GAME} players: "
                f"{', '.join(self.skipped_short_roster[:6])}"
                + (" ..." if len(self.skipped_short_roster) > 6 else "")
            )
        if self.skipped_one_sided_games:
            out.append(
                "SKIPPED, only one team present: "
                f"{', '.join(self.skipped_one_sided_games[:6])}"
                + (" ..." if len(self.skipped_one_sided_games) > 6 else "")
            )
        if self.declined_to_authoritative:
            out.append(
                f"left to {AUTHORITATIVE_SOURCE}: "
                f"{len(self.declined_to_authoritative)} team-game(s)"
            )
        return out


def _non_player_mask(frame: pd.DataFrame) -> pd.Series:
    names = frame["player_name"].astype(str).str.strip().str.lower()
    return names.isin(NON_PLAYER_ROW_NAMES)


def _has_identity(frame: pd.DataFrame) -> pd.Series:
    """A real player row carries a canonical NBA person id.

    The scraper leaves `player_id_nba` null when its crosswalk could not
    resolve a name rather than guessing, and an unresolved row is also the
    shape an aggregate row has.
    """
    ids = frame["player_id_nba"].astype("object")
    return ids.notna() & ids.map(
        lambda v: str(v).strip() not in ("", "None", "nan")
    )


def aggregate_team_games(
    frame: pd.DataFrame,
    *,
    season_types: Sequence[str] = DEFAULT_SEASON_TYPES,
    min_players: int = MIN_PLAYERS_PER_TEAM_GAME,
    owned_by: dict[tuple[str, str], str] | None = None,
) -> tuple[pd.DataFrame, TeamAggregateReport]:
    """
    Sum player box scores into one row per team-game.

    ``owned_by`` maps ``(nba_game_id, team_abbr)`` to the source that already
    holds that row -- `repository.team_game_sources()`. Keys held by
    ``AUTHORITATIVE_SOURCE`` are declined, because the upsert's conflict clause
    refreshes only some columns and would otherwise leave a row stitched from
    two sources wearing one source's label.

    Returns the frame `repository.upsert_team_game_stats` expects -- the model's
    own lowercase column names -- and a report of everything refused.
    """
    report = TeamAggregateReport(rows_in=len(frame))
    if frame.empty:
        raise TeamAggregateError("DATA_NOT_AVAILABLE: no player rows to aggregate")

    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise TeamAggregateError(
            f"DATA_NOT_AVAILABLE: player box scores missing {missing}. "
            f"Present: {sorted(frame.columns)}"
        )

    work = frame

    # [1] Non-player rows. Doubling, and it looks right afterwards.
    is_aggregate = _non_player_mask(work)
    report.dropped_non_player_rows = int(is_aggregate.sum())
    work = work.loc[~is_aggregate]

    # [2] Identity. Also catches an aggregate row under a different name.
    keep = _has_identity(work)
    report.dropped_no_identity = int((~keep).sum())
    work = work.loc[keep]

    # [3] Season type: preseason totals are not this season's team.
    in_season = work["season_type"].isin(season_types)
    report.dropped_wrong_season_type = int((~in_season).sum())
    work = work.loc[in_season]

    if work.empty:
        raise TeamAggregateError(
            "DATA_NOT_AVAILABLE: no player rows survived filtering — "
            + "; ".join(report.lines())
        )

    work = work.assign(_game=normalise_game_id(work["game_id"]))
    work = work.loc[work["_game"].notna()]

    # [4] One row per (game, team). Sums skip NaN, which is what a DNP should
    # contribute to a team total: nothing. It must not null the whole column.
    sums = (
        work.groupby(["_game", "team_abbr"], sort=False)[list(STAT_SUMS)]
        .sum(min_count=1)
        .rename(columns=STAT_SUMS)
        .reset_index()
    )
    counts = (
        work.groupby(["_game", "team_abbr"], sort=False)
        .size()
        .rename("_players")
        .reset_index()
    )
    context = (
        work.groupby(["_game", "team_abbr"], sort=False)
        .agg(
            game_date=("game_date", "first"),
            opponent_abbr=("opponent_abbr", "first"),
            _home=("home_away", "first"),
        )
        .reset_index()
    )
    out = sums.merge(counts, on=["_game", "team_abbr"]).merge(
        context, on=["_game", "team_abbr"]
    )

    # [5] A short roster is a partial ingest, not a low-scoring team.
    short = out["_players"] < int(min_players)
    report.skipped_short_roster = [
        f"{g}/{t} ({n} players)"
        for g, t, n in zip(
            out.loc[short, "_game"], out.loc[short, "team_abbr"],
            out.loc[short, "_players"],
        )
    ]
    out = out.loc[~short]

    # [6] Both sides, or neither. Elo cannot score a one-sided result.
    per_game = out.groupby("_game")["team_abbr"].transform("size")
    one_sided = per_game != TEAMS_PER_GAME
    report.skipped_one_sided_games = sorted(set(out.loc[one_sided, "_game"]))
    out = out.loc[~one_sided]

    if out.empty:
        raise TeamAggregateError(
            "DATA_NOT_AVAILABLE: no complete team-game survived — "
            + "; ".join(report.lines())
        )

    # [7] Precedence. A licensed row is left exactly as it is.
    owned_by = owned_by or {}
    declined = [
        (g, t) for g, t in zip(out["_game"], out["team_abbr"])
        if owned_by.get((str(g), str(t))) == AUTHORITATIVE_SOURCE
    ]
    if declined:
        report.declined_to_authoritative = [f"{g}/{t}" for g, t in declined]
        declined_set = set(declined)
        mask = [
            (str(g), str(t)) not in declined_set
            for g, t in zip(out["_game"], out["team_abbr"])
        ]
        out = out.loc[pd.Series(mask, index=out.index)]
        logger.info(
            "Left %d team-game(s) to %s, which already holds them.",
            len(declined), AUTHORITATIVE_SOURCE,
        )

    if out.empty:
        # Not a failure: every row was already owned by the licensed source.
        logger.info("Nothing to write: %s already holds every team-game.",
                    AUTHORITATIVE_SOURCE)
        report.team_games_out = 0
        return pd.DataFrame(columns=_output_columns()), report

    built = pd.DataFrame({
        "nba_game_id": out["_game"].astype(str),
        "game_date": pd.to_datetime(out["game_date"], errors="coerce").dt.date,
        "team_abbr": out["team_abbr"].astype(str),
        "opponent_abbr": out["opponent_abbr"].astype(str),
        "is_home": out["_home"].astype(str).str.lower().eq("home"),
        # FALSE, AND WRONG FOR ABOUT SIX GAMES A SEASON. See the module
        # docstring: the scraper has no venue field, and the altitude tax reads
        # this to avoid taxing both teams at a neutral site.
        "is_neutral_site": False,
        "source": TEAM_SOURCE,
    })
    for team_col in STAT_SUMS.values():
        built[team_col] = pd.to_numeric(out[team_col], errors="coerce")

    # poss / pace / off_eff / def_eff are deliberately absent, not zero:
    # nothing reads them and the builder owns the one possessions formula.

    built = built.loc[built["game_date"].notna()]
    report.team_games_out = len(built)
    return built.reset_index(drop=True), report


def _output_columns() -> list[str]:
    return [
        "nba_game_id", "game_date", "team_abbr", "opponent_abbr",
        "is_home", "is_neutral_site", "source", *STAT_SUMS.values(),
    ]
