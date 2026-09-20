"""Research targets and feature lists for the model comparison.

IMPORTANT — WHAT ``RESEARCH_LINE`` IS AND IS NOT

It is the player's own trailing 10-game average. It is a **stand-in** used
so the comparison machinery can run before real prop lines exist. It is
NOT a sportsbook line, and a probability computed against it is NOT a
betting probability.

Because the projection is built from the same rolling history, scoring
against this line measures roughly "is recent form above medium-term
form". That is a sanity check on plumbing, not evidence of prop skill.
Every export carries ``line_type`` so this can never be misread.

When real timestamped prop lines land, join them in and pass their column
as ``line_col`` — the models take the line as an argument and need no
change.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

LAUNCH_MARKETS = ("PTS", "REB", "AST")
POST_LAUNCH_MARKETS = ("FG3M", "STL", "BLK", "PRA")

RESEARCH_LINE_COL = "RESEARCH_LINE"
RESEARCH_LINE_SOURCE = "L10"
TARGET_COL = "over_hit"


def default_feature_cols(market: str) -> list[str]:
    """
    Numeric pregame features for a market.

    Own-stat history first, then situational context. Categorical columns
    are deliberately absent: CatBoost adds them separately, while XGBoost
    stays numeric-only so its behaviour is unchanged.
    """
    market = market.upper()
    return [
        f"{market}_L2",
        f"{market}_L5",
        f"{market}_L10",
        f"{market}_SEASON",
        f"{market}_BASELINE",
        "MIN_L5",
        "MIN_L10",
        "MIN_SEASON",
        "fatigue_multiplier",
        "PACE_MULTIPLIER",
        "days_rest",
        "IS_HOME",
        "CAREER_GAMES_PRIOR",
    ]


def attach_research_over_labels(
    df: pd.DataFrame,
    *,
    stat: str = "PTS",
    line_col: str | None = None,
) -> pd.DataFrame:
    """
    Attach ``RESEARCH_LINE`` and the binary ``over_hit`` target.

    ``over_hit`` is 1 when the realised stat finished strictly above the
    line, 0 when strictly below. Exact ties on a whole-number line are a
    PUSH and are dropped from the target (left NaN), because grading a
    push as a loss would bias the label set.

    Pass ``line_col`` to label against real prop lines instead of the
    research stand-in.
    """
    stat = stat.upper()
    work = df.copy()

    if line_col is not None:
        if line_col not in work.columns:
            raise ValueError(f"DATA_NOT_AVAILABLE: line column {line_col!r} not present")
        work[RESEARCH_LINE_COL] = pd.to_numeric(work[line_col], errors="coerce")
        line_type = line_col
    else:
        source_col = f"{stat}_{RESEARCH_LINE_SOURCE}"
        if source_col not in work.columns:
            raise ValueError(
                f"DATA_NOT_AVAILABLE: {source_col} missing — cannot build a research "
                f"line for {stat}. Run build_feature_matrix first."
            )
        work[RESEARCH_LINE_COL] = pd.to_numeric(work[source_col], errors="coerce")
        line_type = f"research_{RESEARCH_LINE_SOURCE.lower()}"

    work["line_type"] = line_type

    if stat not in work.columns:
        raise ValueError(f"DATA_NOT_AVAILABLE: realised {stat} column missing — cannot label")

    actual = pd.to_numeric(work[stat], errors="coerce")
    line = work[RESEARCH_LINE_COL]

    over = actual > line
    under = actual < line
    work[TARGET_COL] = pd.Series(pd.NA, index=work.index, dtype="Float64")
    work.loc[over, TARGET_COL] = 1.0
    work.loc[under, TARGET_COL] = 0.0

    pushes = int((actual == line).sum())
    labelled = int(work[TARGET_COL].notna().sum())
    logger.info(
        "Labelled %s: %d rows (%d pushes dropped, line_type=%s)",
        stat, labelled, pushes, line_type,
    )
    if labelled == 0:
        logger.warning("No labelled rows for %s — every row lacked a line or an outcome.", stat)
    return work
