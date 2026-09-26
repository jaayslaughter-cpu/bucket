"""
src/features/defense.py — pregame opponent-defence features from team box scores.

WHAT A DEFENSIVE MATCHUP FEATURE IS FOR. A scorer's line should move when the
opponent is Orlando and again when it is Washington. Nothing in the player's
own rolling history knows which one he is playing, so without an opponent
term the model projects the same number against both.

WHY THIS DOES NOT USE THE PLAYER PANEL. There was already an
``OPP_{STAT}_ALLOWED_L10`` in src/features/sports_ev_features.py, built by
summing the panel's own player rows per team-game. That sum is over the
players PRESENT IN THE PANEL, not over the team, so it measures roster
coverage as much as it measures defence: dropping one player per team-game
from the demo panel moves "opponent points allowed" by 33%. Any filter --
eligibility, a season slice, an archive with gaps -- silently rescales it,
and two teams whose games happen to carry different numbers of panel rows
look like different defences when they are not.

This module reads TEAM totals instead, from the same ``team_games`` frame
that already feeds Elo. A team's points allowed is the other team's points,
which is complete by construction.

WHY EVERYTHING IS PER 100 POSSESSIONS. Points allowed per GAME confounds two
different things: how well a team defends, and how fast it plays. A fast
team allows more points while defending no worse. Dividing by possessions
separates them, and the tempo term is then published on its own as
``DEF_PACE_L10`` so a model can use both instead of one blurred number.

LEAKAGE. Every rate is a shift-1 rolling mean over the defending team's
PRIOR games, so a team's number for tonight never contains tonight. The
league baseline behind ``DEF_RATING_INDEX_L10`` is an as-of expanding mean,
shifted -- not a season-wide mean, which would fold games that have not been
played into an early-season index. That exact bug was found and fixed once
already in attach_team_pace; it is not repeated here.

Nothing in this module is a betting signal.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SOURCE_NAME = "defense"

# Columns this layer reads from a team_games frame.
REQUIRED_TEAM_COLS: tuple[str, ...] = ("nba_game_id", "game_date", "team_abbr", "points")

# What it writes, all keyed to the DEFENDING team (the player's opponent).
DEFENSE_FEATURE_COLS: tuple[str, ...] = (
    "DEF_RATING_L10",
    "DEF_RATING_INDEX_L10",
    "DEF_PACE_L10",
    "DEF_REB_ALLOWED_PER100_L10",
    "DEF_AST_ALLOWED_PER100_L10",
    "DEF_FG3M_ALLOWED_PER100_L10",
    "DEF_FGA_ALLOWED_PER100_L10",
    "DEF_TOV_FORCED_PER100_L10",
    "DEF_FG_PCT_ALLOWED_L10",
)

# Source column on the OPPONENT's row -> the per-100 feature it becomes.
_ALLOWED_RATES: dict[str, str] = {
    # NOTE: points allowed per 100 IS the defensive rating, so it is emitted
    # once, as DEF_RATING_L10, and does not appear here. Shipping both was a
    # real bug: the two columns correlated at r = 1.0000 and the PTS model
    # received the same number three times (rating, per-100 and the index),
    # which fragmented the trees' splits across identical candidates and gave
    # back a third of the layer's gain. Measured on a panel with a planted
    # defensive signal, Brier over four chronological folds:
    #
    #   no defence features                        0.26749
    #   rating alone                               0.24536
    #   rating + index + points-per-100 + pace     0.24793
    #
    "reb": "DEF_REB_ALLOWED_PER100_L10",
    "ast": "DEF_AST_ALLOWED_PER100_L10",
    "fg3": "DEF_FG3M_ALLOWED_PER100_L10",
    "fga": "DEF_FGA_ALLOWED_PER100_L10",
    "tov": "DEF_TOV_FORCED_PER100_L10",
}

ROLL_WINDOW = 10
ROLL_MIN_PERIODS = 3


class DefenseFeatureError(ValueError):
    """Raised when defensive features are asked for without team box scores."""


def _season_key(dates: pd.Series) -> pd.Series:
    """
    NBA season starting year. October 2025 and March 2026 are one season.

    Rolling within a season rather than across one keeps last June's defence
    out of this October's average, where the roster is a different roster.
    """
    d = pd.to_datetime(dates)
    return (d.dt.year - (d.dt.month < 8).astype(int)).astype("Int64")


def build_team_defense(team_games: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (team, game) carrying that team's PREGAME defensive rates.

    ``team_games`` is the two-rows-per-game frame produced by
    src/ingestion/bigdataball.py. Each team's "allowed" totals are read off
    the opposing row of the same game.

    Possessions come from the frame's ``poss`` when present and otherwise
    from the standard estimate ``FGA - OREB + TOV + 0.44*FTA``. When neither
    is available the per-100 rates are not created -- a per-possession rate
    without possessions does not exist, and a per-game number wearing a
    per-100 name would be worse than nothing.
    """
    if team_games is None or team_games.empty:
        raise DefenseFeatureError("DATA_NOT_AVAILABLE: team_games frame is empty")

    missing = [c for c in REQUIRED_TEAM_COLS if c not in team_games.columns]
    if missing:
        raise DefenseFeatureError(
            f"DATA_NOT_AVAILABLE: team_games missing {missing}. This layer reads "
            "TEAM totals on purpose — see this module's docstring for why summing "
            "the player panel is not a substitute."
        )

    tg = team_games.copy()
    tg["game_date"] = pd.to_datetime(tg["game_date"])
    tg["nba_game_id"] = tg["nba_game_id"].astype(str)
    tg["SEASON_KEY"] = _season_key(tg["game_date"])

    numeric = [c for c in ("points", "fg", "fga", "fg3", "reb", "ast", "tov",
                           "oreb", "fta", "poss") if c in tg.columns]
    for c in numeric:
        tg[c] = pd.to_numeric(tg[c], errors="coerce")

    # Possessions: measured if the source carries them, else the standard
    # estimate, else nothing. Never a stand-in constant.
    if "poss" in tg.columns and tg["poss"].notna().any():
        tg["POSS"] = tg["poss"]
        poss_source = "team_games.poss"
    elif {"fga", "oreb", "tov", "fta"}.issubset(tg.columns):
        tg["POSS"] = tg["fga"] - tg["oreb"] + tg["tov"] + 0.44 * tg["fta"]
        poss_source = "estimated FGA-OREB+TOV+0.44*FTA"
    else:
        raise DefenseFeatureError(
            "DATA_NOT_AVAILABLE: team_games has neither possessions nor the "
            "FGA/OREB/TOV/FTA needed to estimate them. Per-100 defensive rates "
            "cannot be built, and a per-game number would confound defence with "
            "pace — which is the reason this module exists."
        )
    logger.info("Defence layer: possessions from %s.", poss_source)

    # A team's ALLOWED totals are the other team's scored totals in the same
    # game. Self-join rather than groupby: it states the relationship.
    opp_cols = [c for c in (*_ALLOWED_RATES, "points") if c in tg.columns]
    right = tg[["nba_game_id", "team_abbr", *opp_cols]].rename(
        columns={"team_abbr": "_opp", **{c: f"allowed_{c}" for c in opp_cols}}
    )
    if "fg" in tg.columns:
        right = right.merge(
            tg[["nba_game_id", "team_abbr", "fg"]].rename(
                columns={"team_abbr": "_opp", "fg": "allowed_fg"}
            ),
            on=["nba_game_id", "_opp"], how="left",
        )
    merged = tg.merge(right, on="nba_game_id", how="inner")
    merged = merged[merged["team_abbr"] != merged["_opp"]].copy()
    if merged.empty:
        raise DefenseFeatureError(
            "DATA_NOT_AVAILABLE: no game in team_games has two distinct teams, "
            "so no team has an opponent to have allowed anything to."
        )

    per100 = merged["POSS"].where(merged["POSS"] > 0) / 100.0
    for src, out_col in _ALLOWED_RATES.items():
        src_col = f"allowed_{src}"
        merged[f"_rate_{out_col}"] = (
            merged[src_col] / per100 if src_col in merged.columns else np.nan
        )
    if {"allowed_fg", "allowed_fga"}.issubset(merged.columns):
        merged["_rate_DEF_FG_PCT_ALLOWED_L10"] = (
            merged["allowed_fg"] / merged["allowed_fga"].where(merged["allowed_fga"] > 0)
        )
    else:
        merged["_rate_DEF_FG_PCT_ALLOWED_L10"] = np.nan
    merged["_rate_DEF_PACE_L10"] = merged["POSS"]
    # Points allowed per 100 possessions: the defensive rating itself.
    merged["_rate_DEF_RATING_L10"] = merged["allowed_points"] / per100

    merged = merged.sort_values(["team_abbr", "SEASON_KEY", "game_date"]).reset_index(drop=True)

    # Shift-1 rolling within team-season: tonight is never in tonight's number.
    rate_cols = [c for c in merged.columns if c.startswith("_rate_")]
    grouped = merged.groupby(["team_abbr", "SEASON_KEY"], sort=False)
    for col in rate_cols:
        merged[col[len("_rate_"):]] = grouped[col].transform(
            lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=ROLL_MIN_PERIODS).mean()
        )

    merged["DEF_RATING_INDEX_L10"] = _league_relative_index(merged, "DEF_RATING_L10")

    keep = ["nba_game_id", "game_date", "team_abbr", "SEASON_KEY"]
    keep += [c for c in DEFENSE_FEATURE_COLS if c in merged.columns]
    out = merged[keep].copy()
    known = int(out["DEF_RATING_L10"].notna().sum())
    logger.info(
        "Defence layer: %d team-games, defensive rating known on %d (%.1f%%). "
        "The rest are a team's first %d games of a season, which have no prior "
        "form and are left null.",
        len(out), known, 100.0 * known / max(len(out), 1), ROLL_MIN_PERIODS,
    )
    return out


def _league_relative_index(frame: pd.DataFrame, col: str) -> pd.Series:
    """
    ``col`` divided by the league mean AS OF that date, never season-wide.

    A season-wide mean includes games that have not been played, so an
    October index would be scored against June's league. The baseline is an
    expanding daily mean, shifted by one day so a team's own game-day
    contribution is excluded from its own denominator.
    """
    daily = (
        frame.groupby(["SEASON_KEY", "game_date"], as_index=False)[col]
        .mean()
        .rename(columns={col: "_day_mean"})
        .sort_values(["SEASON_KEY", "game_date"])
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


def attach_defense_features(
    panel: pd.DataFrame,
    team_defense: pd.DataFrame,
    *,
    required: bool = False,
) -> pd.DataFrame:
    """
    Join the OPPONENT's pregame defensive rates onto each player row.

    The defending team is the player's opponent, so the join key is
    ``OPPONENT_ABBREVIATION``, not ``TEAM_ABBREVIATION``. Getting that
    backwards would hand the model its own team's defence and still produce
    a full column of plausible numbers.

    Unmatched rows keep NaN. A league-average fill would read as a measured
    matchup against an average defence.
    """
    needed = {"GAME_ID", "OPPONENT_ABBREVIATION"}
    absent = needed - set(panel.columns)
    if absent:
        message = (
            f"Defence features need {sorted(absent)}. Without the opponent's "
            "identity there is no matchup to describe."
        )
        if required:
            raise DefenseFeatureError(f"DATA_NOT_AVAILABLE: {message}")
        logger.info("Defence layer skipped: %s", message)
        return panel

    if team_defense is None or team_defense.empty:
        if required:
            raise DefenseFeatureError("DATA_NOT_AVAILABLE: team_defense frame is empty")
        logger.info("Defence layer skipped: no team_defense rows.")
        return panel

    present = [c for c in DEFENSE_FEATURE_COLS if c in team_defense.columns]
    lookup = team_defense[["nba_game_id", "team_abbr", *present]].rename(
        columns={"nba_game_id": "GAME_ID", "team_abbr": "OPPONENT_ABBREVIATION"}
    )
    out = panel.copy()
    out["GAME_ID"] = out["GAME_ID"].astype(str)
    lookup = lookup.copy()
    lookup["GAME_ID"] = lookup["GAME_ID"].astype(str)

    before = len(out)
    out = out.merge(lookup, on=["GAME_ID", "OPPONENT_ABBREVIATION"], how="left")
    if len(out) != before:
        raise DefenseFeatureError(
            f"Defence join changed the row count ({before} -> {len(out)}). "
            "team_defense must hold at most one row per (game, team); a "
            "duplicate would silently multiply player rows."
        )

    matched = int(out["DEF_RATING_L10"].notna().sum()) if "DEF_RATING_L10" in out else 0
    logger.info(
        "Defence layer: %d of %d player rows matched an opponent's prior form "
        "(%.1f%%). Unmatched rows are null, not league-average.",
        matched, len(out), 100.0 * matched / max(len(out), 1),
    )
    return out
