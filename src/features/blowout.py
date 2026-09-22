"""
src/features/blowout.py — pregame blowout-risk features from the opening spread.

THE MECHANISM. When a game is expected to be lopsided, starters sit in the
fourth quarter. Their counting stats stop accruing while the clock keeps
running, so a points or rebounds line that was fair for 36 minutes is not
fair for 29. The opening spread is the market's pregame estimate of how
lopsided the game will be, and it is posted before tip, so a projection
made that morning could have seen it.

WHAT WAS MEASURED, AND WHAT IT SAID. This layer is DISABLED BY DEFAULT, and
that is a finding rather than an oversight. Measured on 1,322 real 2025-26
games (the BigDataBall workbook), against a team-level proxy label -- a
team's fourth quarter falling below 90% of its own first-three-quarter pace:

  1. The effect is real. Teams opening as 9+ point favourites average a 4Q
     ratio of 0.933; 9+ point underdogs average 0.991. Difference 0.0586,
     se 0.0174, t = 3.38.

  2. The effect is far too small to predict with. That 0.0586 is 0.235 of
     one game's standard deviation (0.249). Walk-forward over five folds,
     RMSE on the continuous ratio: 0.24683 predicting the constant mean,
     0.24647 from spread and total, 0.24667 adding the hinges below. The
     hinge arm is worse than the plain spread arm, which is itself barely
     better than predicting the mean.

  3. For the tree models this project actually uses, the hinges are
     provably empty. A hinge is a monotone transform of the spread, so it
     offers a tree no split point that the raw spread does not already
     offer. Fitted on the real panel, XGBoost gave BLOWOUT_FAV_HINGE and
     BLOWOUT_DOG_HINGE an importance of exactly 0.0 -- not one split in
     200 trees -- and produced predictions identical to the spread-only
     arm to six decimal places. Brier and ECE deltas were 0.00000 at every
     threshold tried (6, 9, 12 points).

  Across eight configurations (two proxy labels x four thresholds) the
  Brier delta was positive -- worse -- every single time.

So the layer exists, is correct, and stays off. Turning it on is a config
change, and the A/B harness (scripts/feature_ab.py) re-runs the comparison
with and without so the decision is made from numbers rather than from this
docstring. The measurement above is a TEAM-LEVEL PROXY; the mechanism is a
player-level one, and team fourth-quarter points barely move because
substitutes replace the starters who sat. A player-level A/B on real prop
data could still come out differently, which is exactly what the harness is
for.

NO CLOSING LINES. Only MKT_OPENING_SPREAD is read. A closing spread is
known only at tip and is refused upstream by src/features/market_context.py;
nothing here reaches around that.

Nothing here is a betting signal. These are context features for a
projection.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SOURCE_NAME = "blowout"

# The spread the market quotes before blowout risk is worth modelling at
# all. Below it, the measured blowout rate is flat: on the 2025-26 panel,
# P(final margin >= 15) by opening-spread bucket runs 0.343 / 0.315 / 0.364
# / 0.343 for 0-3, 3-6, 6-9 and 9-12 points, and only reaches 0.555 at 12+.
# A threshold is therefore a real feature of the data and not a tuning knob,
# but it is also the only place a value could be fitted to the evaluation
# window, so it is a stated constant rather than something this module
# estimates.
DEFAULT_SPREAD_THRESHOLD = 9.0

SPREAD_COL = "MKT_OPENING_SPREAD"

# Two hinges, not three. The symmetric max(0, |spread| - t) is EXACTLY the
# sum of these two -- only one can be positive for a given spread -- so
# adding it would hand every model a perfectly collinear column.
BLOWOUT_FEATURE_COLS: tuple[str, ...] = (
    "BLOWOUT_FAV_HINGE",
    "BLOWOUT_DOG_HINGE",
)


class BlowoutFeatureError(ValueError):
    """Raised when blowout features are asked for without a pregame spread."""


def attach_blowout_features(
    df: pd.DataFrame,
    *,
    spread_threshold: float = DEFAULT_SPREAD_THRESHOLD,
    required: bool = False,
) -> pd.DataFrame:
    """
    Add ``BLOWOUT_FAV_HINGE`` and ``BLOWOUT_DOG_HINGE`` from the opening spread.

    By this project's sign convention a negative spread is the favourite,
    so for a team quoted at -12 with a threshold of 9, FAV_HINGE is 3 and
    DOG_HINGE is 0. Both are 0 for any game inside the threshold, which is
    most of them.

    The columns are NOT created when ``MKT_OPENING_SPREAD`` is absent. A
    zero-filled blowout feature would read as "measured, and no blowout
    risk" for every row of a panel that simply had no market lines joined,
    which is the fabrication this project removed zero-fills for. Pass
    ``required=True`` to raise instead of skipping.

    Rows whose spread is null keep NaN, for the same reason.
    """
    if SPREAD_COL not in df.columns:
        message = (
            f"Blowout features need {SPREAD_COL}, which is absent. Supply the "
            "market_lines frame to build_feature_matrix so market context is "
            "attached. Not creating zero-filled columns, which would read as a "
            "measured absence of blowout risk."
        )
        if required:
            raise BlowoutFeatureError(f"DATA_NOT_AVAILABLE: {message}")
        logger.info("Blowout layer skipped: %s", message)
        return df

    threshold = float(spread_threshold)
    if not np.isfinite(threshold) or threshold < 0:
        raise BlowoutFeatureError(
            f"spread_threshold must be a non-negative number, got {spread_threshold!r}"
        )

    out = df.copy()
    spread = pd.to_numeric(out[SPREAD_COL], errors="coerce")

    # np.maximum propagates NaN, which is what we want: an unknown spread
    # yields an unknown hinge, never a confident zero.
    out["BLOWOUT_FAV_HINGE"] = np.maximum(0.0, -spread - threshold)
    out["BLOWOUT_DOG_HINGE"] = np.maximum(0.0, spread - threshold)

    known = int(spread.notna().sum())
    engaged = int((out["BLOWOUT_FAV_HINGE"] > 0).sum() + (out["BLOWOUT_DOG_HINGE"] > 0).sum())
    logger.info(
        "Blowout layer: spread known on %d of %d rows; %d rows past the "
        "%.1f-point threshold. See this module's docstring for why the layer "
        "defaults to off.",
        known, len(out), engaged, threshold,
    )
    return out
