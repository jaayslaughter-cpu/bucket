"""
src/features/matchup.py — opponent defence from ACTUAL defensive assignments.

WHAT THIS MEASURES THAT NOTHING ELSE HERE DOES. Both existing defence layers
infer who defended whom:

  src/features/defense.py   reads TEAM totals. A team's points allowed per 100
                            is complete and unbiased, and it says nothing
                            about which defender was on the floor or whom he
                            was guarding.
  src/features/dvp.py       buckets players G/F/C and asks what a team concedes
                            to each bucket. That bucket is a PROXY for the
                            matchup, and its own docstring says so.

This layer reads the NBA's own matchup tracking: one row per (game, offensive
player, defensive player) with the minutes they spent matched up and the box
score accumulated DURING that assignment. No inference and no proxy — the
defender is named.

THE PROJECT WAS ALREADY TOLD THIS WAS THE BETTER ROUTE, and the record is
worth keeping. ``docs/external_repo_review_2026-09.md`` (2026-09-26 addendum)
says of the ``leaguedashptdefend`` endpoint: "Supersedes P1.4. Gives POSITION
directly *and* a better defender feature than position-bucketed DvP."
Position-bucketed DvP was built anyway. ``dvp.py`` is leakage-safe and
measured so it is not wasted, but its honest status is interim, and this is
the layer it was a proxy for.

WHAT IT IS AND IS NOT. It is TEAM-level, like ``defense.py``: the output is
what this opponent's defenders have been allowing, weighted by the minutes
they actually spend defending. It is NOT a per-matchup prediction, because
tonight's assignments are not known before tip — a pregame feature cannot
depend on who ends up guarding whom. What the assignment data buys is a
defensive rating that is weighted by WHO ACTUALLY DEFENDS rather than by a
team total in which a bench player's minutes count the same as a starter's.

MEASURED ON matchups_2024 (2023-24), to establish there is signal before
building on it. 2,458 team-games, a mean of 10.7 distinct defenders and 94
matchup rows per team-game, 94.7 matchup-minutes per team-game:

    points allowed per matchup minute   mean 1.2487   sd 0.1771 (per game)
    team SEASON means                   1.166 -> 1.365, sd 0.0440
    FG% allowed, team season means      0.4400 -> 0.4879, sd 0.0109

The best and worst defences differ by about 17% in points allowed per matchup
minute, so the spread is real rather than noise around a league constant.

HALF OF WHAT IT COMPUTES IS REDUNDANT, AND THAT WAS MEASURED RATHER THAN
ASSUMED. See ``MATCHUP_SHIPPABLE_COLS`` below for the table: FG% allowed from
assignments correlates 0.96 with the team-total ``DEF_FG_PCT_ALLOWED_L10`` we
already carry, while points allowed per matchup MINUTE correlates 0.77 with
``DEF_RATING_L10`` and is the column worth having.

ONE THING THAT LOOKED LIKE A WIN AND IS NOT. This export carries a
``position`` column, and ``dvp.py``'s bucket is limited by position coverage,
so the obvious move was to use it as a second source. Counted on 2023-24 it is
**46.8% of player-games (433 players)** against the archive's
``startingPosition`` at **47.1% (391 players)** — 42 more players and no more
coverage. Not worth a second data dependency, and recorded so nobody tries it
again expecting more.

LEAKAGE, in the two places it could enter. The aggregation is per (defending
team, GAME) first and the ten-game window is then ten of the defender team's
GAMES — not ten matchup rows, which would span about one game given 94 rows
per team-game. That is the same defect ``dvp.py`` documents finding in its
source implementation, and the same fix. The league baseline behind the INDEX
columns is an as-of expanding daily mean, shifted, never a season-wide mean.

RESEARCH ONLY. Nothing here is a betting signal.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.features.season import season_start_year

logger = logging.getLogger(__name__)

SOURCE_NAME = "matchup"

#: Columns read from a leagueseasonmatchups / boxscorematchups export.
REQUIRED_MATCHUP_COLS: tuple[str, ...] = (
    "game_id", "team_id", "home_team_id", "away_team_id", "team_tricode",
    "matchups_person_id", "matchup_minutes", "player_points",
    "matchup_field_goals_made", "matchup_field_goals_attempted",
)

ROLL_WINDOW = 10
#: Five of the defending team's games. Each window entry is already a sum over
#: ~94 matchup rows, so it is far less noisy than dvp.py's bucket mean — but
#: UNFITTED either way, and nothing has been measured about 3 vs 5 vs 8.
ROLL_MIN_PERIODS = 5

ALLOWED_TEMPLATE = "MU_{stat}_L10"
INDEX_TEMPLATE = "MU_{stat}_INDEX_L10"

#: What a defence allowed, per the assignments it actually made.
MATCHUP_STATS: tuple[str, ...] = ("PTS_PER_MIN", "FG_PCT")

MATCHUP_FEATURE_COLS: tuple[str, ...] = (
    *(ALLOWED_TEMPLATE.format(stat=s) for s in MATCHUP_STATS),
    *(INDEX_TEMPLATE.format(stat=s) for s in MATCHUP_STATS),
)

#: WHAT WOULD ACTUALLY SHIP, which is half of what is computed.
#:
#: Measured on the full 214,381-row panel with all nine seasons of matchup
#: data attached (92.1% coverage), max |r| against every DEF_* column the
#: contract already carries:
#:
#:     MU_PTS_PER_MIN_L10        0.7705  vs DEF_RATING_L10
#:     MU_PTS_PER_MIN_INDEX_L10  0.7358  vs DEF_RATING_INDEX_L10
#:     MU_FG_PCT_L10             0.9595  vs DEF_FG_PCT_ALLOWED_L10
#:     MU_FG_PCT_INDEX_L10       0.9102  vs DEF_FG_PCT_ALLOWED_L10
#:
#: The FG% pair is REDUNDANT -- 0.91-0.96 sits inside the 0.83-0.99 band
#: labels._EXCLUDED_AS_REDUNDANT was built from, and it makes sense: field-goal
#: percentage allowed is the same shooting outcome whether it is attributed by
#: assignment or summed over a team, and the assignment weighting adds almost
#: nothing to it.
#:
#: Points allowed per MATCHUP MINUTE is the one that is not. 0.77 against
#: DEF_RATING_L10 is correlated but outside the band, which is what one would
#: expect: per-100-possession team points and per-assigned-minute points
#: weight a bench defender's minutes differently, and that difference is the
#: whole reason to read assignment data.
#:
#: The FG% columns are still COMPUTED, so the measurement above stays
#: reproducible and so a future change of mind has the numbers rather than an
#: argument. They are diagnostics here, the way PBP_FGA and SZ_FGA are.
MATCHUP_SHIPPABLE_COLS: tuple[str, ...] = (
    ALLOWED_TEMPLATE.format(stat="PTS_PER_MIN"),
    INDEX_TEMPLATE.format(stat="PTS_PER_MIN"),
)


class MatchupFeatureError(ValueError):
    """Raised when the matchup export cannot support this layer."""


def parse_matchup_minutes(value: object) -> float:
    """
    ``"M:SS"`` to float minutes. NaN for anything else.

    The export writes matchup time as a clock string, not a number, so a
    pd.to_numeric would null the entire column and the layer would abstain on
    every row while looking like it had simply found no data.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return float("nan")
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return float("nan")
    try:
        if ":" not in text:
            return float(text)
        minutes, seconds = text.split(":", 1)
        return float(int(minutes) + float(seconds) / 60.0)
    except (TypeError, ValueError):
        return float("nan")


def build_defender_allowed(
    matchups: pd.DataFrame,
    game_dates: pd.DataFrame,
) -> pd.DataFrame:
    """
    One row per (defending team, game) carrying PREGAME allowed rates.

    ``game_dates`` supplies ``nba_game_id`` and ``game_date``: the matchup
    export carries no date, and a rolling window needs an order. Taking it
    from the panel rather than guessing keeps one definition of when a game
    was played.

    THE DEFENDING TEAM IS DERIVED, NOT READ. ``team_id`` on a row is the
    OFFENSIVE player's team, so the defender's team is the other side of
    ``home_team_id``/``away_team_id``. Reading ``team_id`` as the defender
    would produce a full column of plausible numbers describing each team's
    own offence as its defence — verified on the real export that every row's
    ``team_id`` is one of the two.
    """
    if matchups is None or matchups.empty:
        raise MatchupFeatureError("DATA_NOT_AVAILABLE: the matchup export is empty")
    missing = [c for c in REQUIRED_MATCHUP_COLS if c not in matchups.columns]
    if missing:
        raise MatchupFeatureError(
            f"DATA_NOT_AVAILABLE: matchup export missing {missing}. Expected a "
            "leagueseasonmatchups/boxscorematchups export; found: "
            f"{sorted(matchups.columns)[:12]}..."
        )

    work = matchups.copy()
    work["_GAME"] = work["game_id"].astype(str).str.strip().str.lstrip("0")
    work["_MM"] = work["matchup_minutes"].map(parse_matchup_minutes)
    for col in ("player_points", "matchup_field_goals_made",
                "matchup_field_goals_attempted"):
        work[col] = pd.to_numeric(work[col], errors="coerce")

    offense_is_home = work["team_id"] == work["home_team_id"]
    work["_DEF_TEAM_ID"] = np.where(
        offense_is_home, work["away_team_id"], work["home_team_id"]
    )

    # team_id -> tricode, built from the export itself. Both teams of a game
    # appear as an offensive team somewhere in it, so the map is complete
    # without a second source.
    codes = (
        work[["team_id", "team_tricode"]]
        .dropna()
        .drop_duplicates(subset=["team_id"], keep="last")
        .set_index("team_id")["team_tricode"]
    )
    work["_DEF_TEAM"] = work["_DEF_TEAM_ID"].map(codes)
    unmapped = int(work["_DEF_TEAM"].isna().sum())
    if unmapped:
        logger.warning(
            "Matchup layer: %d row(s) have a defending team id with no tricode "
            "in the export and are dropped rather than guessed.", unmapped,
        )
    work = work.dropna(subset=["_DEF_TEAM"])

    per_game = work.groupby(["_DEF_TEAM", "_GAME"], as_index=False).agg(
        _mm=("_MM", "sum"),
        _pts=("player_points", "sum"),
        _fgm=("matchup_field_goals_made", "sum"),
        _fga=("matchup_field_goals_attempted", "sum"),
        _defenders=("matchups_person_id", "nunique"),
    )
    # RATIOS OF TOTALS, not means of per-matchup ratios. A defender who spent
    # eleven seconds on a star would otherwise carry the same weight as one
    # who guarded him for twenty minutes.
    per_game["_rate_MU_PTS_PER_MIN_L10"] = (
        per_game["_pts"] / per_game["_mm"].where(per_game["_mm"] > 0)
    )
    per_game["_rate_MU_FG_PCT_L10"] = (
        per_game["_fgm"] / per_game["_fga"].where(per_game["_fga"] > 0)
    )

    dates = game_dates.copy()
    dates["_GAME"] = dates["nba_game_id"].astype(str).str.strip().str.lstrip("0")
    dates["game_date"] = pd.to_datetime(dates["game_date"], errors="coerce")
    dates = dates.dropna(subset=["game_date"]).drop_duplicates(subset=["_GAME"])
    merged = per_game.merge(dates[["_GAME", "game_date"]], on="_GAME", how="inner")
    if merged.empty:
        raise MatchupFeatureError(
            "DATA_NOT_AVAILABLE: no matchup game id matched a panel game id, so "
            "no window can be ordered. The export writes game_id unpadded and "
            "the panel pads it; both sides are zero-stripped here, so an empty "
            "intersection means the two cover different games."
        )
    merged["SEASON_KEY"] = season_start_year(merged["game_date"])
    merged = merged.sort_values(
        ["_DEF_TEAM", "SEASON_KEY", "game_date"], kind="mergesort"
    ).reset_index(drop=True)

    grouped = merged.groupby(["_DEF_TEAM", "SEASON_KEY"], sort=False)
    for stat in MATCHUP_STATS:
        raw = f"_rate_MU_{stat}_L10"
        merged[ALLOWED_TEMPLATE.format(stat=stat)] = grouped[raw].transform(
            lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=ROLL_MIN_PERIODS).mean()
        )
    for stat in MATCHUP_STATS:
        merged[INDEX_TEMPLATE.format(stat=stat)] = _league_relative_index(
            merged, ALLOWED_TEMPLATE.format(stat=stat)
        )

    keep = ["_GAME", "_DEF_TEAM", "game_date", "SEASON_KEY", "_defenders"]
    keep += [c for c in MATCHUP_FEATURE_COLS if c in merged.columns]
    out = merged[keep].copy()
    first = ALLOWED_TEMPLATE.format(stat=MATCHUP_STATS[0])
    known = int(out[first].notna().sum())
    logger.info(
        "Matchup layer: %d (defending team, game) rows from real assignments, "
        "mean %.1f distinct defenders; allowed rate known on %d (%.1f%%). The "
        "rest are a team's first %d games of a season.",
        len(out), float(out["_defenders"].mean()), known,
        100.0 * known / max(len(out), 1), ROLL_MIN_PERIODS,
    )
    return out


def _league_relative_index(frame: pd.DataFrame, col: str) -> pd.Series:
    """
    ``col`` over the league's as-of mean, never a season-wide one.

    The form ``defense._league_relative_index`` arrived at, and for the reason
    recorded there: a season-wide mean folds games that have not been played
    into an October index.
    """
    daily = (
        frame.groupby(["SEASON_KEY", "game_date"], as_index=False)[col]
        .mean()
        .rename(columns={col: "_day_mean"})
        .sort_values(["SEASON_KEY", "game_date"], kind="mergesort")
        .reset_index(drop=True)
    )
    daily["_asof"] = daily.groupby("SEASON_KEY", sort=False)["_day_mean"].transform(
        lambda s: s.expanding(min_periods=ROLL_MIN_PERIODS).mean().shift(1)
    )
    joined = frame.merge(
        daily[["SEASON_KEY", "game_date", "_asof"]],
        on=["SEASON_KEY", "game_date"], how="left",
    )
    baseline = joined["_asof"].where(joined["_asof"] > 0)
    return (joined[col] / baseline).to_numpy()


def attach_matchup_features(
    panel: pd.DataFrame,
    allowed: pd.DataFrame | None = None,
    *,
    required: bool = False,
) -> pd.DataFrame:
    """
    Join the OPPONENT's assignment-weighted defensive rates onto each row.

    The defending team is the player's ``OPPONENT_ABBREVIATION``, as in
    ``defense.attach_defense_features`` and for the same reason: getting it
    backwards hands a model its own team's defence and still produces a full
    column of plausible numbers.

    Unmatched rows keep NaN. A league-average fill would read as a measured
    matchup against an average defence.
    """
    if panel is None or panel.empty:
        if required:
            raise MatchupFeatureError("DATA_NOT_AVAILABLE: panel is empty")
        logger.info("Matchup layer skipped: empty panel.")
        return panel if panel is not None else pd.DataFrame()

    needed = {"GAME_ID", "OPPONENT_ABBREVIATION"}
    absent = needed - set(panel.columns)
    if absent:
        message = f"matchup features need {sorted(absent)}"
        if required:
            raise MatchupFeatureError(f"DATA_NOT_AVAILABLE: {message}")
        logger.info("Matchup layer skipped: %s", message)
        return panel

    if allowed is None or allowed.empty:
        message = (
            "no matchup table supplied. Build one with build_defender_allowed "
            "from a leagueseasonmatchups export."
        )
        if required:
            raise MatchupFeatureError(f"DATA_NOT_AVAILABLE: {message}")
        logger.info("Matchup layer skipped: %s", message)
        return panel

    out = panel.copy()
    out["_GAME"] = out["GAME_ID"].astype(str).str.strip().str.lstrip("0")
    present = [c for c in MATCHUP_FEATURE_COLS if c in allowed.columns]
    lookup = allowed[["_GAME", "_DEF_TEAM", *present]].rename(
        columns={"_DEF_TEAM": "OPPONENT_ABBREVIATION"}
    )
    lookup["OPPONENT_ABBREVIATION"] = lookup["OPPONENT_ABBREVIATION"].astype(
        out["OPPONENT_ABBREVIATION"].dtype
    )
    keys = ["_GAME", "OPPONENT_ABBREVIATION"]
    duplicated = int(lookup.duplicated(subset=keys).sum())
    if duplicated:
        raise MatchupFeatureError(
            f"Matchup lookup holds {duplicated} duplicate (game, defender) "
            "rows; joining it would silently multiply player rows."
        )

    before = len(out)
    out = out.merge(lookup, on=keys, how="left")
    if len(out) != before:
        raise MatchupFeatureError(
            f"Matchup join changed the row count ({before} -> {len(out)})."
        )
    out = out.drop(columns=["_GAME"])

    first = ALLOWED_TEMPLATE.format(stat=MATCHUP_STATS[0])
    matched = int(out[first].notna().sum()) if first in out.columns else 0
    logger.info(
        "Matchup layer: %d of %d player rows matched their opponent's prior "
        "assignment-weighted defence (%.1f%%). Unmatched rows are null, not "
        "league-average.",
        matched, len(out), 100.0 * matched / max(len(out), 1),
    )
    return out
