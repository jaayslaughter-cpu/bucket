"""
src/features/box_rates.py — the rates migration 011's columns exist for.

THIS LAYER IS WHY THE MIGRATION IS ALLOWED TO EXIST. Migration 009 added
`fga`/`fta` and refused to add the made-shot columns alongside them:

    A column nothing reads is a column that rots [...] If a feature later
    needs FG%, that is its own migration with its own reason.

011 is that migration and this is that reader. The two land together, and
`tests/test_box_rates.py` fails if this layer stops reading any of the five.

FOUR RATES THE PANEL COULD NOT EXPRESS BEFORE:

    EFG_PCT     = (FGM + 0.5 * FG3M) / FGA   -- a three is worth 1.5 twos
    FT_PCT      = FTM / FTA                  -- the only pure skill rate here
    FG3A_RATE   = FG3A / FGA                 -- share of shots from range
    OREB_SHARE  = OREB / (OREB + DREB)       -- see the warning below

WHAT OREB_SHARE IS NOT, STATED HERE AND IN THE MIGRATION BECAUSE THE NAME
MATTERS. It is NOT offensive rebound percentage. True OREB% is
OREB / (OREB + opponent DREB): a share of available offensive rebounds, which
needs the opponent's defensive rebounds and therefore a team-level join no
player row can supply. What this computes is the player's own split between
offensive and defensive rebounds -- a ROLE signal (who crashes the glass)
rather than an OPPORTUNITY signal (how many were there to take). Those are
different quantities and only one of them is here, so the column says SHARE.

EVERY FEATURE IS A SHIFTED PRIOR WINDOW. The same-game rate is a POSTGAME
quantity: a model handed this game's eFG% has been told how the game went.
They are computed, used to build the rolling means, and dropped. This is
CLAUDE.md's first rule and the one the whole project rests on.

NOT IN THE MODEL MATRIX, AND NOT BY OVERSIGHT. These columns are emitted for
`scripts/feature_ab.py` to measure. A feature reaches the matrix by winning a
holdout, not by being computed -- so adding this layer changes what the panel
CARRIES and changes nothing about what the artifacts USE. The deployed
artifacts name their own `feature_cols` and extra columns are ignored.

RESEARCH_ONLY. Box-score ratios. Nothing here prices, sizes or places a wager.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Every column this layer needs. Absent any one of them it abstains rather
#: than deriving part of the set, because a frame missing `fgm` is a panel
#: written before migration 011 and the honest answer there is "no rates".
REQUIRED_COLS: tuple[str, ...] = (
    "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "OREB", "DREB",
)

#: Windows, matching `scoring_efficiency`'s so the two layers' outputs are
#: comparable in an A/B rather than differing by lookback as well as content.
RATE_WINDOWS: tuple[int, ...] = (5, 10)

#: The weight a made three carries over a made two in effective FG%. Not a
#: tunable: a three is worth 1.5 times a two, so the standard formula adds
#: half a made three to the numerator.
THREE_POINT_BONUS: float = 0.5


def effective_field_goal_percentage(
    field_goals_made: pd.Series,
    three_pointers_made: pd.Series,
    field_goal_attempts: pd.Series,
) -> pd.Series:
    """
    eFG% per row. NaN where no shots were attempted.

    Zero attempts is undefined efficiency, not zero efficiency -- the same
    reasoning `scoring_efficiency.true_shooting_percentage` gives. Returning
    0.0 would drag a rolling mean down with a value meaning "did not shoot".
    """
    fgm = pd.to_numeric(field_goals_made, errors="coerce")
    fg3m = pd.to_numeric(three_pointers_made, errors="coerce")
    fga = pd.to_numeric(field_goal_attempts, errors="coerce")
    numerator = fgm + THREE_POINT_BONUS * fg3m
    return numerator.divide(fga.where(fga > 0))


def free_throw_percentage(
    free_throws_made: pd.Series, free_throw_attempts: pd.Series,
) -> pd.Series:
    """FT% per row. NaN where the player did not reach the line."""
    ftm = pd.to_numeric(free_throws_made, errors="coerce")
    fta = pd.to_numeric(free_throw_attempts, errors="coerce")
    return ftm.divide(fta.where(fta > 0))


def three_point_attempt_rate(
    three_point_attempts: pd.Series, field_goal_attempts: pd.Series,
) -> pd.Series:
    """3PAr per row. NaN where no shots were attempted."""
    fg3a = pd.to_numeric(three_point_attempts, errors="coerce")
    fga = pd.to_numeric(field_goal_attempts, errors="coerce")
    return fg3a.divide(fga.where(fga > 0))


def offensive_rebound_share(
    offensive_rebounds: pd.Series, defensive_rebounds: pd.Series,
) -> pd.Series:
    """
    OREB / (OREB + DREB) per row. NaN where the player rebounded nothing.

    NOT offensive rebound percentage -- see the module docstring. A player
    with no rebounds has no split, which is why the denominator is masked
    rather than allowed to be zero.
    """
    oreb = pd.to_numeric(offensive_rebounds, errors="coerce")
    dreb = pd.to_numeric(defensive_rebounds, errors="coerce")
    total = oreb + dreb
    return oreb.divide(total.where(total > 0))


def _prior_window_mean(series: pd.Series, window: int) -> pd.Series:
    """
    Mean of the previous ``window`` games -- never the current one.

    Shift and window in ONE call, inside a groupby transform, so the window
    cannot cross a player boundary. Identical construction to
    `scoring_efficiency._prior_window_mean`, deliberately: two leakage-safe
    rollers that differ would be two chances to get it wrong.
    """
    return series.shift(1).rolling(window, min_periods=1).mean()


#: The same-game rates. Computed to build the shifted means, then DROPPED:
#: each is a postgame quantity and none may reach a model.
_POSTGAME = ("EFG_PCT", "FT_PCT", "FG3A_RATE", "OREB_SHARE")


def attach_box_rate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add prior-games eFG%, FT%, 3PAr and offensive-rebound-share columns.

    Adds only; never rewrites an existing column. Returns the frame unchanged
    when migration 011's columns are absent, which is every panel written
    before it and every row whose source did not report them.
    """
    out = df.copy()

    missing = [c for c in REQUIRED_COLS if c not in out.columns]
    if missing:
        logger.info(
            "box_rates: %s absent -- no eFG%%/FT%%/3PAr/OREB_SHARE features. "
            "These need migration 011's columns; a panel without them carries "
            "no made shots and no rebound split, and this layer will not "
            "invent either.", missing,
        )
        return out

    if "PLAYER_ID" not in out.columns:
        logger.warning("box_rates: PLAYER_ID absent -- skipping.")
        return out

    keys = ["PLAYER_ID", "SEASON"] if "SEASON" in out.columns else ["PLAYER_ID"]

    with np.errstate(invalid="ignore", divide="ignore"):
        out["EFG_PCT"] = effective_field_goal_percentage(
            out["FGM"], out["FG3M"], out["FGA"],
        )
        out["FT_PCT"] = free_throw_percentage(out["FTM"], out["FTA"])
        out["FG3A_RATE"] = three_point_attempt_rate(out["FG3A"], out["FGA"])
        out["OREB_SHARE"] = offensive_rebound_share(out["OREB"], out["DREB"])

    grouped = out.groupby(keys, sort=False)
    for window in RATE_WINDOWS:
        for rate in _POSTGAME:
            out[f"{rate}_L{window}"] = grouped[rate].transform(
                _prior_window_mean, window=window,
            )

    # Shooting trend, mirroring TS_PCT_TREND: recent form against the player's
    # own slightly longer baseline. Positive means converting above it.
    out["EFG_PCT_TREND"] = out["EFG_PCT_L5"] - out["EFG_PCT_L10"]

    # THE POSTGAME COLUMNS GO. They were scaffolding for the shifted means and
    # a model handed this game's eFG% has been told how the game went.
    out = out.drop(columns=[c for c in _POSTGAME if c in out.columns])

    logger.info(
        "box_rates: attached %d prior-window feature(s) from migration 011's "
        "columns over %d row(s). Not in the model matrix: a feature reaches "
        "that by winning a holdout in scripts/feature_ab.py.",
        len(RATE_WINDOWS) * len(_POSTGAME) + 1, len(out),
    )
    return out
