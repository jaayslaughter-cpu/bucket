"""Pregame-only feature matrix.

THE RULE THIS FILE EXISTS TO ENFORCE: a feature for a game may only use
games that finished BEFORE it. Every rolling statistic here is computed on
a ``.shift(1)`` of the player's own history, so the current game's box
score can never reach its own features.

``LAST_INCLUDED_GAME_DATE`` records the most recent game folded into each
row's features. ``assert_no_lookahead`` checks it is strictly earlier than
``GAME_DATE``, which fails loudly if the shift discipline is ever broken.

Two layers, matching the orchestrator's contract:

    {stat}_BASELINE  layer 1 — blended recent form, nothing situational
    {stat}_L2        layer 2 — BASELINE adjusted for fatigue and pace

``build_feature_matrix`` calls ``attach_fatigue_column`` itself and folds
the multiplier into ``{stat}_L2``. main.py therefore VERIFIES rather than
re-applies; applying it twice would compound the penalty.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.features.fatigue_logic import attach_fatigue_column
from src.features.schedule import attach_team_schedule_features
from src.features.team_strength import attach_elo_features, compute_team_elo

logger = logging.getLogger(__name__)

# Counting stats that get the full rolling treatment.
ROLLING_STATS = ("PTS", "REB", "AST", "FG3M", "STL", "BLK", "MIN")

# Layer-1 blend. Recent form dominates, season average stabilises a short
# sample. Unfitted starting weights, not estimated parameters.
BASELINE_WEIGHTS = {"L5": 0.5, "L10": 0.3, "SEASON": 0.2}

FEATURE_SCHEMA_VERSION = "fs_v1_shift1_l2"


class LookaheadError(AssertionError):
    """Raised when a feature row could see its own game or a later one."""


def _prior_window_mean(series: pd.Series, window: int) -> pd.Series:
    """Mean of the previous `window` games. Never includes the current one.

    Must be used via ``groupby(...).transform``. Calling ``.rolling()`` on
    an already-groupby-shifted Series silently rolls ACROSS players, so one
    player's debut inherits the previous player's last game — applying the
    shift and the window inside the same per-group call is what prevents
    that.
    """
    return series.shift(1).rolling(window, min_periods=1).mean()


def _expanding_prior_mean(series: pd.Series) -> pd.Series:
    """Season-to-date mean of PRIOR games only.

    Deliberately expanding rather than a whole-season average: a
    full-season mean would leak the rest of the season into October rows.
    """
    return series.shift(1).expanding().mean()


def build_feature_matrix(
    player_panel: pd.DataFrame,
    *,
    team_games: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build the leakage-safe feature matrix from a raw player game-log panel.

    Expects at minimum PLAYER_ID, GAME_DATE, and one or more of
    ``ROLLING_STATS``. Missing stats are skipped rather than invented.

    ``team_games`` is an optional team-level frame carrying final scores
    (the BigDataBall ``team_game_stats`` output, or anything with game id,
    date, team and points). When supplied, pre-game Elo ratings are joined
    on; when absent, the team-strength columns are simply not created and
    the run is narrower rather than silently filled.
    """
    if player_panel.empty:
        logger.warning("Empty player panel — returning it unchanged.")
        return player_panel.copy()

    required = {"PLAYER_ID", "GAME_DATE"}
    missing = required - set(player_panel.columns)
    if missing:
        raise ValueError(f"DATA_NOT_AVAILABLE: player panel missing {sorted(missing)}")

    df = player_panel.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)

    by_player = df.groupby("PLAYER_ID", sort=False)

    # The newest game already folded into this row's features.
    df["LAST_INCLUDED_GAME_DATE"] = by_player["GAME_DATE"].shift(1)
    df["CAREER_GAMES_PRIOR"] = by_player.cumcount()

    season_key = ["PLAYER_ID", "SEASON"] if "SEASON" in df.columns else ["PLAYER_ID"]
    by_season = df.groupby(season_key, sort=False)

    present = [s for s in ROLLING_STATS if s in df.columns]
    if not present:
        raise ValueError(
            f"DATA_NOT_AVAILABLE: none of {ROLLING_STATS} present in the panel"
        )
    for stat in present:
        df[stat] = pd.to_numeric(df[stat], errors="coerce")
        for window in (2, 5, 10):
            df[f"{stat}_L{window}"] = by_player[stat].transform(
                _prior_window_mean, window=window
            )
        df[f"{stat}_SEASON"] = by_season[stat].transform(_expanding_prior_mean)

    df = attach_fatigue_column(df)

    if {"TEAM_ABBREVIATION", "GAME_ID"}.issubset(df.columns):
        df = attach_team_schedule_features(df)
    else:
        logger.info("Team schedule features skipped: needs TEAM_ABBREVIATION and GAME_ID.")

    if team_games is not None and not team_games.empty:
        # Elo is computed over the full team history in date order, then
        # joined by pre-game value only. elo_post never reaches the panel.
        df = attach_elo_features(df, compute_team_elo(team_games))
    else:
        logger.info(
            "Team Elo skipped: no team_games frame supplied, so no opponent-strength "
            "features. Pass the BigDataBall team_game_stats frame to enable them."
        )

    if "PACE_MULTIPLIER" not in df.columns:
        # No opponent pace joined yet. 1.0 is a neutral no-op, not an
        # estimate — see docs/DATA_GAPS.md for what would populate it.
        df["PACE_MULTIPLIER"] = 1.0
        logger.info("PACE_MULTIPLIER absent — defaulting to neutral 1.0 (no pace effect).")

    for stat in present:
        blended = (
            BASELINE_WEIGHTS["L5"] * df[f"{stat}_L5"]
            + BASELINE_WEIGHTS["L10"] * df[f"{stat}_L10"]
            + BASELINE_WEIGHTS["SEASON"] * df[f"{stat}_SEASON"]
        )
        # Early-season rows have no season mean yet; fall back to what exists
        # rather than dropping the row or inventing a value.
        df[f"{stat}_BASELINE"] = blended.fillna(df[f"{stat}_L5"]).fillna(df[f"{stat}_L10"])
        df[f"{stat}_L2"] = (
            df[f"{stat}_BASELINE"] * df["fatigue_multiplier"] * df["PACE_MULTIPLIER"]
        )

    df["FEATURE_SCHEMA_VERSION"] = FEATURE_SCHEMA_VERSION

    assert_no_lookahead(df)
    logger.info(
        "Feature matrix: %d rows x %d cols | stats=%s | schema=%s",
        len(df), df.shape[1], present, FEATURE_SCHEMA_VERSION,
    )
    return df


def assert_no_lookahead(features: pd.DataFrame) -> None:
    """
    Fail loudly if any row's features could see its own game or a later one.

    A player's first game has no prior history, so a null
    LAST_INCLUDED_GAME_DATE is correct and is not a violation.
    """
    if "LAST_INCLUDED_GAME_DATE" not in features.columns:
        raise LookaheadError(
            "LAST_INCLUDED_GAME_DATE missing — cannot verify the feature matrix "
            "is pregame-safe. Refusing to certify it."
        )

    game = pd.to_datetime(features["GAME_DATE"])
    last = pd.to_datetime(features["LAST_INCLUDED_GAME_DATE"])
    violations = last.notna() & (last >= game)

    if violations.any():
        sample = features.loc[violations, ["PLAYER_ID", "GAME_DATE", "LAST_INCLUDED_GAME_DATE"]]
        raise LookaheadError(
            f"{int(violations.sum())} rows include a game on or after their own "
            f"GAME_DATE. First offenders:\n{sample.head(5)}"
        )

    logger.debug("Lookahead check passed for %d rows.", len(features))
