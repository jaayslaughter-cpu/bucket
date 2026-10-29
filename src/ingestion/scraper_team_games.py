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

from src.ingestion.scraper_panel import (
    DEFAULT_SEASON_TYPES,
    SOURCE_PRECEDENCE,
    _as_bool,
    _precedence_rank,
)
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
    "player_name", "player_id_nba", "season_type",
    # REQUIRED, not optional. Without it the per-source copies cannot be
    # collapsed and every team total is multiplied by the number of enabled
    # sources. A frame that cannot be deduplicated must be refused, not summed.
    "source",
    *STAT_SUMS,
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
    #: Duplicate (source, game, player) copies collapsed to one row. Non-zero
    #: whenever more than one scraper source is enabled, and the single most
    #: dangerous number here: summing them multiplies every team total.
    deduplicated_source_copies: int = 0
    skipped_short_roster: list[str] = field(default_factory=list)
    skipped_one_sided_games: list[str] = field(default_factory=list)
    #: Team-games with no usable home/away. Refused: a silent False would be a
    #: wrong IS_HOME, which the altitude tax and home advantage both read.
    skipped_unknown_side: list[str] = field(default_factory=list)
    #: (team-game, how many PLAYED players had a null stat). The total is low
    #: by those players' lines; the row is still written, but never silently.
    stat_gaps: list[str] = field(default_factory=list)
    declined_to_authoritative: list[str] = field(default_factory=list)

    def lines(self) -> list[str]:
        out = [
            f"player rows in            {self.rows_in}",
            f"  non-player rows         -{self.dropped_non_player_rows}",
            f"  no player identity      -{self.dropped_no_identity}",
            f"  wrong season type       -{self.dropped_wrong_season_type}",
            f"  duplicate source copies -{self.deduplicated_source_copies}",
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
        if self.skipped_unknown_side:
            out.append(
                "SKIPPED, home/away unknown: "
                f"{', '.join(self.skipped_unknown_side[:6])}"
                + (" ..." if len(self.skipped_unknown_side) > 6 else "")
            )
        if self.stat_gaps:
            out.append(
                "TOTALS LOW, a played player had a null stat: "
                f"{', '.join(self.stat_gaps[:6])}"
                + (" ..." if len(self.stat_gaps) > 6 else "")
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
    precedence: tuple[str, ...] = SOURCE_PRECEDENCE,
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

    # [3b] ONE ROW PER PLAYER, ACROSS SOURCES. THE DEFECT THIS WHOLE MODULE
    # WOULD OTHERWISE HAVE SHIPPED WITH. The scraper writes one row per
    # (source, game, player) -- `nba_live`, `nba_stats` and
    # `basketball_reference` each contribute a copy -- so summing the frame as
    # it arrives MULTIPLIES every team total by the number of enabled sources.
    # A `.env` with USE_NBA_STATS=1 and USE_BASKETBALL_REFERENCE=1 triples it.
    #
    # Nothing downstream could catch that, and neither could the guards below:
    # a short roster looks for FEWER rows and duplicates give MORE, and the
    # two-teams rule is satisfied either way. Twenty-seven tests passed over it
    # because the fixtures are single-source. Found in review, 2026-10-29.
    #
    # `to_player_panel` already solved this for the player panel, so the same
    # precedence and the same whole-row `drop_duplicates` are reused rather
    # than restated: a second ranking of the same sources would be the
    # two-definitions defect with provenance as its subject.
    before_dedup = len(work)
    work = (
        work.assign(_rank=_precedence_rank(work["source"], precedence))
        .sort_values(["_game", "player_id_nba", "_rank"], kind="mergesort")
        .drop_duplicates(subset=["_game", "player_id_nba"], keep="first")
        .drop(columns="_rank")
    )
    report.deduplicated_source_copies = before_dedup - len(work)
    if report.deduplicated_source_copies:
        logger.info(
            "Collapsed %d duplicate source copy/copies to one row per player "
            "(precedence %s); summing them would have multiplied every team "
            "total.",
            report.deduplicated_source_copies, ", ".join(precedence),
        )

    # [4] One row per (game, team). Sums skip NaN, which is what a DNP should
    # contribute to a team total: nothing. It must not null the whole column.
    # NULLS AMONG PLAYED ROWS ARE A DATA GAP, not a zero. `sum(min_count=1)`
    # treats one player's null as zero, which is right for a DNP (he added
    # nothing) and wrong for a played row (his points are unknown, and the
    # team total comes out silently LOW). Counted per team-game and reported;
    # the team-game is still written, because a total short by one player's
    # line is more useful than no row at all -- but it is never silent.
    # Flagged in review, 2026-10-29.
    # `_as_bool`, NOT `.astype(bool)`: every non-empty string is truthy, so a
    # `played` that arrived as the TEXT "False" would read as played. That is
    # the confusion `scraper_panel._as_bool` exists for, and borrowing it is
    # cheaper than being the second reader of the same flag that gets it
    # wrong. Its NA is False; here an UNKNOWN played flag is treated as played
    # so the gap is still reported -- a report line nobody needed costs
    # nothing, and a silently low total is the thing being guarded against.
    played_mask = (
        _as_bool(work["played"]) | work["played"].isna()
        if "played" in work.columns else pd.Series(True, index=work.index)
    )
    gap_counts = (
        work.loc[played_mask]
        .assign(_gaps=lambda d: d[list(STAT_SUMS)].isna().any(axis=1))
        .groupby(["_game", "team_abbr"], sort=False)["_gaps"]
        .sum()
        .rename("_stat_gaps")
        .reset_index()
    )

    sums = (
        work.groupby(["_game", "team_abbr"], sort=False)[list(STAT_SUMS)]
        .sum(min_count=1)
        .rename(columns=STAT_SUMS)
        .reset_index()
    )
    # PLAYED ROWS ONLY. A DNP contributes nothing to the sums, so counting one
    # toward the roster minimum lets seven played players plus one inactive
    # pass as eight and write a team total short by three players' lines --
    # which is exactly the partial ingest `min_players` exists to catch, let
    # through by the guard meant to catch it. The loader hands over the raw
    # parquet (DNPs included) because the sums want them, so this cannot be
    # left to the caller. Found in review, 2026-10-29.
    counts = (
        work.loc[played_mask]
        .groupby(["_game", "team_abbr"], sort=False)
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
    # LEFT, not inner: a team-game whose every row is a DNP has no played
    # count at all, and an inner join would make it vanish silently instead of
    # being reported as the short roster it is.
    out = (
        sums.merge(counts, on=["_game", "team_abbr"], how="left")
        .merge(context, on=["_game", "team_abbr"])
        .merge(gap_counts, on=["_game", "team_abbr"], how="left")
    )
    out["_players"] = out["_players"].fillna(0).astype(int)
    out["_stat_gaps"] = out["_stat_gaps"].fillna(0).astype(int)

    # [5] A short roster is a partial ingest, not a low-scoring team.
    short = out["_players"] < int(min_players)
    report.skipped_short_roster = [
        f"{g}/{t} ({n} played)"
        for g, t, n in zip(
            out.loc[short, "_game"], out.loc[short, "team_abbr"],
            out.loc[short, "_players"],
        )
    ]
    out = out.loc[~short]

    # [5b] An unknown home/away is refused, not recorded as away.
    # `astype(str).eq("home")` turns a missing value into the string "nan" and
    # then into False, so a team-game with no side would be written as an away
    # game. `fatigue_logic` reads IS_HOME for the altitude tax and
    # `team_strength` for home advantage, so a silent False is a wrong feature
    # rather than a missing one. Found in review, 2026-10-29.
    side = out["_home"].astype("object")
    known_side = side.notna() & side.map(
        lambda v: str(v).strip().lower() in ("home", "away")
    )
    report.skipped_unknown_side = [
        f"{g}/{t}" for g, t in zip(
            out.loc[~known_side, "_game"], out.loc[~known_side, "team_abbr"]
        )
    ]
    out = out.loc[known_side]

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

    report.stat_gaps = [
        f"{g}/{t} ({n} player(s))"
        for g, t, n in zip(out["_game"], out["team_abbr"], out["_stat_gaps"])
        if n
    ]
    if report.stat_gaps:
        logger.warning(
            "%d team-game(s) have a played player with a null stat, so those "
            "totals are LOW by that player's line: %s",
            len(report.stat_gaps), ", ".join(report.stat_gaps[:6]),
        )

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
