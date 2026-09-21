"""
src/models/recency.py — weight recent training games more heavily.

WHY: a game from three seasons ago and one from last week are not equally
informative about tonight, but an unweighted fit treats them identically.
Exponential recency weighting tilts the fit toward recent form without
throwing old games away, which is what a hard cutoff does.

Adapted from a pattern seen in a reference implementation, with two
changes:

1. **Parameterised by half-life, not a decay constant.** The reference
   used ``exp(-0.001 * days_ago)``, which is opaque — it is a half-life
   of about 1.9 years, which is not what most readers would guess from
   the number. A half-life in days says what it means.

2. **Reports effective sample size.** Weighting does not just re-rank
   rows, it discards information: an aggressive half-life can reduce
   5,000 rows to an effective few hundred, and nothing in the fit will
   say so. ``effective_sample_size`` makes that visible before it shows
   up as an unstable model.

LEAKAGE: weights are computed from the TRAINING rows' own dates. Passing
an ``as_of`` taken from the validation window would leak the split
boundary into the fit, so that is rejected rather than silently allowed.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# A season is ~165 days end to end. This default weights the current
# season well above the previous one without erasing it. It is a starting
# point, not a fitted value — tune it with fit_half_life_by_holdout.
DEFAULT_HALF_LIFE_DAYS = 240.0

MIN_HALF_LIFE_DAYS = 7.0


class RecencyWeightError(ValueError):
    """Raised when the weighting would be meaningless or leak."""


def exponential_recency_weights(
    dates: pd.Series,
    *,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    as_of: pd.Timestamp | None = None,
    normalise: bool = True,
) -> pd.Series:
    """
    Weight each row by how recent it is, halving every ``half_life_days``.

    ``as_of`` defaults to the most recent date in ``dates`` — the training
    set's own end. It may not be EARLIER than that: a row after ``as_of``
    would take a weight above 1, which means the caller passed a date from
    outside the training window.

    Weights are normalised to mean 1.0 by default so that changing the
    half-life re-balances the rows without also changing the total weight,
    which would otherwise silently alter regularisation strength.
    """
    if half_life_days < MIN_HALF_LIFE_DAYS:
        raise RecencyWeightError(
            f"half_life_days={half_life_days} is below {MIN_HALF_LIFE_DAYS}; "
            "that discards almost every game and is not a weighting scheme"
        )

    parsed = pd.to_datetime(pd.Series(dates), errors="coerce")
    if parsed.notna().sum() == 0:
        raise RecencyWeightError("DATA_NOT_AVAILABLE: no parseable dates to weight")

    latest = parsed.max()
    reference = pd.Timestamp(as_of) if as_of is not None else latest
    if reference < latest:
        raise RecencyWeightError(
            f"as_of={reference.date()} is earlier than the newest training row "
            f"({latest.date()}). Weights above 1 mean the reference date came "
            "from outside the training window, which leaks the split boundary."
        )

    days_ago = (reference - parsed).dt.total_seconds() / 86400.0
    weights = np.power(0.5, days_ago / float(half_life_days))
    # An unparseable date gets the smallest weight rather than being
    # dropped here — the caller decides what to do with the row.
    weights = pd.Series(weights, index=parsed.index).fillna(weights.min())

    if normalise:
        mean = float(weights.mean())
        if mean <= 0:
            raise RecencyWeightError("Degenerate weights — all rows weighted zero")
        weights = weights / mean
    return weights


def effective_sample_size(weights: pd.Series | np.ndarray) -> float:
    """
    Kish effective sample size: (sum w)^2 / sum(w^2).

    With equal weights this equals the row count. The more concentrated
    the weighting, the lower it goes — which is the number to look at
    before concluding a model has thousands of rows behind it.
    """
    w = np.asarray(pd.Series(weights).dropna(), dtype=float)
    if len(w) == 0 or not np.isfinite(w).any():
        return 0.0
    w = w[np.isfinite(w) & (w >= 0)]
    denominator = float(np.sum(w ** 2))
    if denominator <= 0:
        return 0.0
    return float(np.sum(w) ** 2 / denominator)


def recency_weight_report(
    dates: pd.Series,
    *,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    as_of: pd.Timestamp | None = None,
) -> dict[str, float]:
    """Weights plus the diagnostics worth seeing before trusting them."""
    weights = exponential_recency_weights(
        dates, half_life_days=half_life_days, as_of=as_of
    )
    n = int(len(weights))
    ess = effective_sample_size(weights)
    return {
        "n_rows": float(n),
        "half_life_days": float(half_life_days),
        "effective_sample_size": round(ess, 1),
        "effective_fraction": round(ess / n, 4) if n else 0.0,
        "max_weight": round(float(weights.max()), 4),
        "min_weight": round(float(weights.min()), 6),
        "weight_ratio": (
            round(float(weights.max() / weights.min()), 2)
            if float(weights.min()) > 0 else float("inf")
        ),
    }


def fit_half_life_by_holdout(
    train_predict_score,
    dates: pd.Series,
    *,
    candidates: tuple[float, ...] = (90.0, 180.0, 240.0, 365.0, 730.0, 1e6),
    holdout_fraction: float = 0.25,
) -> dict[str, float]:
    """
    Choose the half-life by held-out score instead of asserting one.

    ``train_predict_score(train_idx, score_idx, weights)`` must fit on the
    training slice with the supplied weights and return a score where
    LOWER is better. The split is chronological, and the candidate list
    includes an effectively-infinite half-life so "no weighting at all"
    can win — which it will when recency carries no signal, and that is a
    result worth being able to get.
    """
    parsed = pd.to_datetime(pd.Series(dates), errors="coerce")
    order = parsed.sort_values().index
    cut = int(len(order) * (1 - holdout_fraction))
    if cut < 30 or len(order) - cut < 15:
        return {
            "chosen_half_life_days": DEFAULT_HALF_LIFE_DAYS,
            "reason_insufficient_rows": float(len(order)),
        }

    train_idx, score_idx = order[:cut], order[cut:]
    train_dates = parsed.loc[train_idx]

    scores: dict[str, float] = {}
    best_value, best_half_life = float("inf"), DEFAULT_HALF_LIFE_DAYS
    for half_life in candidates:
        weights = exponential_recency_weights(
            train_dates, half_life_days=max(half_life, MIN_HALF_LIFE_DAYS)
        )
        try:
            value = float(train_predict_score(train_idx, score_idx, weights))
        except Exception as exc:  # noqa: BLE001 — a candidate may be unfittable
            logger.warning("half-life %.0f failed to score (%s)", half_life, exc)
            continue
        scores[f"half_life_{int(half_life)}"] = round(value, 6)
        if value < best_value:
            best_value, best_half_life = value, half_life

    logger.info(
        "recency half-life chosen by holdout: %.0f days (scores: %s)",
        best_half_life, scores,
    )
    return {
        "chosen_half_life_days": float(best_half_life),
        "chosen_score": round(best_value, 6),
        **scores,
    }
