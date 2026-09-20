"""Is the projection closer to reality than the line is?

Every other metric in this repository compares models against each other or
against a rolling average. This one asks the question that actually decides
whether a prop model is worth anything:

    MAE(line, actual) - MAE(projection, actual)

Positive means the projection landed nearer the truth than the posted line
did. It needs no odds, no expected value and no bet decision, which is
exactly why it is the honest test — none of the usual ways of flattering a
model apply to it.

WHAT IT MEANS TODAY VS LATER. Run against ``RESEARCH_LINE`` (the player's
own trailing 10-game average) it answers "does the model beat a naive
recency baseline as a point estimate". That is a real question and worth
tracking, but it is NOT a market test. Point ``line_col`` at real
timestamped prop lines and the same function becomes one, with no change
to the maths. Every report carries the ``line_type`` it was computed
against so the two can never be confused.

THE SECOND TEST is whether confidence means anything. Bucket predictions by
how strongly the model leans, and the hit rate should climb across buckets.
If it does not, the model's confidence is decorative — it can still be
right on average while being unable to tell you *when* it is right, and
that distinction is invisible to Brier score alone.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Below this, a bucket's hit rate is noise. 30 coin flips has a standard
# error near 9 percentage points, which is wider than any edge worth having.
MIN_BUCKET_ROWS = 30

DEFAULT_CONFIDENCE_EDGES: tuple[float, ...] = (0.0, 0.05, 0.10, 0.20, 0.50)


def model_vs_line_report(
    predictions: pd.DataFrame,
    *,
    projection_col: str = "prediction_mean",
    line_col: str = "prop_line",
    actual_col: str = "actual_stat_value",
    group_cols: tuple[str, ...] = ("target_market", "model_name"),
) -> pd.DataFrame:
    """
    Compare the projection's error against the line's error, per group.

    ``model_beats_line_by`` is in the units of the stat: +0.4 on points
    means the projection was, on average, 0.4 points nearer the actual
    result than the line was. Negative means the line was better and the
    projection is not yet worth using as a point estimate.
    """
    needed = {projection_col, line_col, actual_col, *group_cols}
    missing = needed - set(predictions.columns)
    if missing:
        raise ValueError(f"DATA_NOT_AVAILABLE: predictions missing {sorted(missing)}")

    work = predictions.copy()
    for col in (projection_col, line_col, actual_col):
        work[col] = pd.to_numeric(work[col], errors="coerce")

    usable = work.dropna(subset=[projection_col, line_col, actual_col])
    dropped = len(work) - len(usable)
    if dropped:
        logger.info(
            "model_vs_line: %d of %d rows lack a projection, line or outcome — excluded",
            dropped, len(work),
        )
    if usable.empty:
        logger.warning("model_vs_line: no rows with all three of projection, line and outcome")
        return pd.DataFrame()

    line_type = (
        str(usable["line_type"].iloc[0]) if "line_type" in usable.columns else "unknown"
    )
    mixed_lines = "line_type" in usable.columns and usable["line_type"].nunique() > 1
    if mixed_lines:
        logger.warning(
            "model_vs_line: rows mix %d line types — comparing a research stand-in "
            "against a real market line in one number would be meaningless",
            usable["line_type"].nunique(),
        )

    rows: list[dict[str, Any]] = []
    for keys, group in usable.groupby(list(group_cols), sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        model_err = (group[projection_col] - group[actual_col]).abs()
        line_err = (group[line_col] - group[actual_col]).abs()
        model_mae, line_mae = float(model_err.mean()), float(line_err.mean())

        row = dict(zip(group_cols, keys))
        row.update({
            "n_predictions": int(len(group)),
            "model_mae": round(model_mae, 4),
            "line_mae": round(line_mae, 4),
            "model_beats_line_by": round(line_mae - model_mae, 4),
            "model_rmse": round(float(np.sqrt((model_err ** 2).mean())), 4),
            "line_rmse": round(float(np.sqrt((line_err ** 2).mean())), 4),
            "line_type": "MIXED" if mixed_lines else line_type,
            "is_market_line": bool(
                not mixed_lines and not line_type.startswith("research")
            ),
        })
        rows.append(row)

    report = pd.DataFrame(rows)
    if not report.empty:
        beat = int((report["model_beats_line_by"] > 0).sum())
        logger.info(
            "model_vs_line: projection beat the line in %d of %d groups (line_type=%s)",
            beat, len(report), "MIXED" if mixed_lines else line_type,
        )
    return report


def _favoured_side_and_confidence(
    p_over: pd.Series,
    market_p_over: pd.Series | None,
) -> tuple[pd.Series, pd.Series, str]:
    """Which side the model leans, and by how much.

    With a market probability the measure is genuine edge — model minus
    market on the side taken. Without one it is bare confidence, distance
    from a coin flip. They are not the same quantity and the report says
    which was used: confidence can be high while edge is zero, if the
    market already knows.
    """
    side_over = p_over > 0.5
    if market_p_over is None:
        return side_over, (p_over - 0.5).abs(), "confidence"

    model_side_p = p_over.where(side_over, 1.0 - p_over)
    market_side_p = market_p_over.where(side_over, 1.0 - market_p_over)
    return side_over, model_side_p - market_side_p, "edge"


def edge_bucket_report(
    predictions: pd.DataFrame,
    *,
    probability_col: str = "probability_over_raw",
    line_col: str = "prop_line",
    actual_col: str = "actual_stat_value",
    market_probability_col: str | None = None,
    bucket_edges: tuple[float, ...] = DEFAULT_CONFIDENCE_EDGES,
    group_cols: tuple[str, ...] = ("target_market", "model_name"),
) -> pd.DataFrame:
    """
    Hit rate by how strongly the model leaned, per group.

    A model whose hit rate is flat across buckets cannot tell you when it is
    right, even if its average calibration looks fine. Pushes are excluded
    from the hit rate and counted separately — grading one as a loss would
    understate every bucket it falls in.

    ``measure`` records whether buckets are true edge (a market probability
    was supplied) or bare confidence (none was).
    """
    needed = {probability_col, line_col, actual_col, *group_cols}
    missing = needed - set(predictions.columns)
    if missing:
        raise ValueError(f"DATA_NOT_AVAILABLE: predictions missing {sorted(missing)}")

    work = predictions.copy()
    numeric = [probability_col, line_col, actual_col]
    if market_probability_col:
        if market_probability_col not in work.columns:
            raise ValueError(
                f"DATA_NOT_AVAILABLE: market probability column "
                f"{market_probability_col!r} not present"
            )
        numeric.append(market_probability_col)
    for col in numeric:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    usable = work.dropna(subset=numeric)
    if usable.empty:
        logger.warning("edge_buckets: no rows with probability, line and outcome")
        return pd.DataFrame()

    market_p = usable[market_probability_col] if market_probability_col else None
    side_over, measure_value, measure_name = _favoured_side_and_confidence(
        usable[probability_col], market_p
    )
    usable = usable.assign(_side_over=side_over, _measure=measure_value)

    # Push first: on a whole-number line an exact hit is neither side's win.
    is_push = usable[actual_col] == usable[line_col]
    won_over = usable[actual_col] > usable[line_col]
    usable["_hit"] = np.where(
        is_push, np.nan, (won_over == usable["_side_over"]).astype(float)
    )

    labels = [f"{lo:.0%}-{hi:.0%}" for lo, hi in zip(bucket_edges, bucket_edges[1:])]
    usable["_bucket"] = pd.cut(
        usable["_measure"], bins=list(bucket_edges), labels=labels,
        include_lowest=True, right=False,
    )

    rows: list[dict[str, Any]] = []
    for keys, group in usable.groupby(list(group_cols), sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        for bucket in labels:
            in_bucket = group[group["_bucket"] == bucket]
            if in_bucket.empty:
                continue
            graded = in_bucket.dropna(subset=["_hit"])
            row = dict(zip(group_cols, keys))
            row.update({
                "measure": measure_name,
                "bucket": bucket,
                "n_predictions": int(len(in_bucket)),
                "n_graded": int(len(graded)),
                "n_pushes": int(len(in_bucket) - len(graded)),
                "mean_measure": round(float(in_bucket["_measure"].mean()), 4),
                "hit_rate": round(float(graded["_hit"].mean()), 4) if len(graded) else None,
                "below_min_sample": bool(len(graded) < MIN_BUCKET_ROWS),
            })
            rows.append(row)

    report = pd.DataFrame(rows)
    if not report.empty:
        logger.info(
            "edge_buckets: %d rows across %d groups (measure=%s); %d buckets below "
            "the %d-row minimum and flagged",
            len(report), report.groupby(list(group_cols)).ngroups, measure_name,
            int(report["below_min_sample"].sum()), MIN_BUCKET_ROWS,
        )
    return report


def confidence_is_informative(bucket_report: pd.DataFrame, *, group: dict[str, str]) -> dict[str, Any]:
    """
    Does the hit rate actually rise with the model's confidence?

    Returns the spread between the top and bottom usable bucket and whether
    the ordering is monotonic. Buckets under the sample minimum are excluded
    rather than allowed to set the verdict on a handful of rows.
    """
    subset = bucket_report
    for key, value in group.items():
        subset = subset[subset[key] == value]
    usable = subset[~subset["below_min_sample"] & subset["hit_rate"].notna()]

    if len(usable) < 2:
        return {
            **group,
            "verdict": "INSUFFICIENT_DATA",
            "n_usable_buckets": int(len(usable)),
            "detail": f"need 2+ buckets with {MIN_BUCKET_ROWS}+ graded rows",
        }

    ordered = usable.sort_values("mean_measure")
    rates = ordered["hit_rate"].to_numpy(dtype=float)
    counts = ordered["n_graded"].to_numpy(dtype=float)
    spread = float(rates[-1] - rates[0])
    monotonic = bool(np.all(np.diff(rates) >= 0))

    # A positive spread alone proves nothing: with an uninformative model the
    # top bucket outscores the bottom roughly half the time by chance. Require
    # the gap to clear two standard errors of the difference between the two
    # proportions before calling it signal.
    low_p, high_p = float(rates[0]), float(rates[-1])
    low_n, high_n = max(float(counts[0]), 1.0), max(float(counts[-1]), 1.0)
    standard_error = float(
        np.sqrt(low_p * (1 - low_p) / low_n + high_p * (1 - high_p) / high_n)
    )
    threshold = 2.0 * standard_error
    informative = spread > threshold

    return {
        **group,
        "verdict": "INFORMATIVE" if informative else "FLAT_OR_INVERTED",
        "n_usable_buckets": int(len(usable)),
        "lowest_bucket_hit_rate": round(low_p, 4),
        "highest_bucket_hit_rate": round(high_p, 4),
        "hit_rate_spread": round(spread, 4),
        "spread_threshold": round(threshold, 4),
        "monotonic": monotonic,
        "detail": (
            f"Hit rate rises with confidence by {spread:.1%}, clearing the "
            f"{threshold:.1%} needed to rule out chance"
            if informative
            else (
                f"Spread of {spread:.1%} is within the {threshold:.1%} expected "
                "from chance — confidence does not track being right"
            )
        ),
    }
