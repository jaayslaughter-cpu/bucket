"""
src/features/scoring_efficiency.py — True Shooting and shot-volume features.

WHY THIS FILE EXISTS: wave5a's builder imports ``attach_box_ts_features``
and no pack ever contained it, so the layer was silently absent. It is
written here rather than stubbed, because a stub returning a plausible
number would be worse than the missing column it replaced.

TRUE SHOOTING is a real formula, not a heuristic:

    TS% = PTS / (2 * (FGA + 0.44 * FTA))

The 0.44 is the standard estimate of the share of free-throw attempts
that end a possession — it accounts for and-ones and technical free
throws, which do not. It is a convention from the basketball-analytics
literature, not a parameter fitted here, and it is labelled as such.

WHY IT IS WORTH HAVING: points alone conflate volume with efficiency. A
player scoring 20 on 12 shots and one scoring 20 on 24 shots have the
same PTS and very different expectations for tomorrow. Efficiency and
volume move differently, and separating them gives the model something
points cannot express.

LEAKAGE: every feature here is a shift-1 prior-games average within
player-season, the same discipline as the rest of the builder. The
current game's shooting never reaches its own row.

ABSTENTION: FGA and FTA are optional on the panel. Without them, nothing
is attached and the reason is logged — TS% cannot be estimated from
points alone, and a version that tried would be inventing shot volume.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Share of free-throw attempts that end a possession. A published
# convention (Oliver), not a value fitted on this repository's data.
FREE_THROW_POSSESSION_FACTOR = 0.44

# Rolling windows for the prior-games averages, matching the builder.
EFFICIENCY_WINDOWS = (5, 10)

REQUIRED_COLS = ("PTS", "FGA", "FTA")


def true_shooting_percentage(
    points: pd.Series,
    field_goal_attempts: pd.Series,
    free_throw_attempts: pd.Series,
) -> pd.Series:
    """
    TS% per row. NaN where there were no shooting possessions.

    A player who took no shots has undefined efficiency, not zero
    efficiency — returning 0.0 would drag every rolling average down with
    a value that means "did not play", not "shot badly".
    """
    pts = pd.to_numeric(points, errors="coerce")
    fga = pd.to_numeric(field_goal_attempts, errors="coerce")
    fta = pd.to_numeric(free_throw_attempts, errors="coerce")

    possessions = 2.0 * (fga + FREE_THROW_POSSESSION_FACTOR * fta)
    return pts.divide(possessions.where(possessions > 0))


def _prior_window_mean(series: pd.Series, window: int) -> pd.Series:
    """Mean of the previous ``window`` games — never the current one.

    Shift and window happen inside one call so the rolling window cannot
    cross a player boundary, the same construction the builder uses.
    """
    return series.shift(1).rolling(window, min_periods=1).mean()


def attach_box_ts_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add prior-games True Shooting and shot-volume columns.

    Adds only; never rewrites an existing column. Returns the frame
    unchanged when the shooting columns are absent.
    """
    out = df.copy()

    missing = [c for c in REQUIRED_COLS if c not in out.columns]
    if missing:
        logger.info(
            "scoring_efficiency: %s absent — no True Shooting features. TS%% "
            "cannot be derived from points alone, and this layer will not "
            "invent shot volume.", missing,
        )
        return out

    if "PLAYER_ID" not in out.columns:
        logger.warning("scoring_efficiency: PLAYER_ID absent — skipping.")
        return out

    keys = ["PLAYER_ID", "SEASON"] if "SEASON" in out.columns else ["PLAYER_ID"]

    # Realised, same-game values. These are POSTGAME quantities and are not
    # features — only their shifted rolling means below are safe to model on.
    out["TS_PCT"] = true_shooting_percentage(out["PTS"], out["FGA"], out["FTA"])
    fga = pd.to_numeric(out["FGA"], errors="coerce")
    fta = pd.to_numeric(out["FTA"], errors="coerce")
    out["SHOT_VOLUME"] = fga + FREE_THROW_POSSESSION_FACTOR * fta

    grouped = out.groupby(keys, sort=False)
    for window in EFFICIENCY_WINDOWS:
        out[f"TS_PCT_L{window}"] = grouped["TS_PCT"].transform(
            _prior_window_mean, window=window
        )
        out[f"SHOT_VOLUME_L{window}"] = grouped["SHOT_VOLUME"].transform(
            _prior_window_mean, window=window
        )
        out[f"FGA_L{window}"] = grouped["FGA"].transform(
            _prior_window_mean, window=window
        )

    # Free-throw rate: how much of a player's scoring comes from the line.
    # It moves with role and with how often they attack the rim, and is
    # more stable game to game than points.
    with np.errstate(invalid="ignore", divide="ignore"):
        out["FT_RATE"] = fta.divide(fga.where(fga > 0))
    out["FT_RATE_L10"] = grouped["FT_RATE"].transform(_prior_window_mean, window=10)

    # Efficiency trend: recent shooting against the slightly longer window.
    # Positive means a player has been converting above their own baseline.
    out["TS_PCT_TREND"] = out["TS_PCT_L5"] - out["TS_PCT_L10"]

    # Same-game TS_PCT / SHOT_VOLUME / FT_RATE are postgame — drop after
    # building shift-1 rolls so they cannot enter an "all numeric cols" path.
    out = out.drop(columns=["TS_PCT", "SHOT_VOLUME", "FT_RATE"], errors="ignore")

    attached = [c for c in out.columns if c not in df.columns]
    logger.info(
        "scoring_efficiency: attached %d columns (%d rows have a prior TS%%)",
        len(attached), int(out["TS_PCT_L5"].notna().sum()),
    )
    return out
