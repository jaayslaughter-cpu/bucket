"""Strict comparison of batch and live numeric features on identical keys.

This is an additive research/testing utility, not a feature or a scorer.
Callers must supply real, archived as-of inputs to both feature builders.
Equal features do not by themselves prove those inputs were available at tip.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


class FeatureParityError(ValueError):
    """Batch and live features cannot support a parity claim."""


@dataclass(frozen=True)
class ParityResult:
    rows: int
    max_absolute_error: dict[str, float]
    compared_values: dict[str, int]


def assert_prospective_parity(
    batch: pd.DataFrame,
    live: pd.DataFrame,
    *,
    keys: Sequence[str],
    features: Sequence[str],
    absolute_tolerance: float = 1e-10,
) -> ParityResult:
    """Refuse duplicate keys, missing rows, missingness changes and infinities.

    Frames may differ in row order. Both-missing values agree in missingness
    but are not counted as numeric comparisons. A wholly missing feature
    fails instead of yielding a vacuous successful comparison.
    """
    key_columns, feature_columns = list(keys), list(features)
    if (
        not key_columns
        or not feature_columns
        or len(set(key_columns)) != len(key_columns)
        or len(set(feature_columns)) != len(feature_columns)
        or set(key_columns).intersection(feature_columns)
    ):
        raise FeatureParityError("Provide distinct, nonempty keys and feature columns")
    if (
        isinstance(absolute_tolerance, bool)
        or not math.isfinite(absolute_tolerance)
        or absolute_tolerance < 0
    ):
        raise FeatureParityError("absolute_tolerance must be finite and nonnegative")

    required = set(key_columns + feature_columns)
    indexed = []
    for name, frame in (("batch", batch), ("live", live)):
        if frame.empty or not frame.columns.is_unique:
            raise FeatureParityError(f"{name}: empty frame or duplicate columns")
        missing = required.difference(frame.columns)
        if missing:
            raise FeatureParityError(f"{name}: missing columns {sorted(missing)}")
        if frame[key_columns].isna().any().any():
            raise FeatureParityError(f"{name}: null identity key")
        if frame.duplicated(key_columns).any():
            raise FeatureParityError(f"{name}: duplicate identity key")
        indexed.append(frame.set_index(key_columns))
    left, right = indexed
    if (
        len(left) != len(right)
        or len(left.index.difference(right.index, sort=False))
        or len(right.index.difference(left.index, sort=False))
    ):
        raise FeatureParityError("Batch and live key sets differ")
    right = right.reindex(left.index)

    errors: dict[str, float] = {}
    counts: dict[str, int] = {}
    for column in feature_columns:
        try:
            a = pd.to_numeric(left[column], errors="raise").to_numpy(
                dtype=float, na_value=np.nan
            )
            b = pd.to_numeric(right[column], errors="raise").to_numpy(
                dtype=float, na_value=np.nan
            )
        except (TypeError, ValueError) as exc:
            raise FeatureParityError(f"{column}: nonnumeric feature") from exc
        if np.isinf(a).any() or np.isinf(b).any():
            raise FeatureParityError(f"{column}: infinite feature")
        missing_a, missing_b = np.isnan(a), np.isnan(b)
        if not np.array_equal(missing_a, missing_b):
            raise FeatureParityError(f"{column}: missingness differs")
        present = ~missing_a
        count = int(present.sum())
        if not count:
            raise FeatureParityError(f"{column}: no comparable observations")
        with np.errstate(over="ignore"):
            maximum = float(np.max(np.abs(a[present] - b[present])))
        if not math.isfinite(maximum) or maximum > absolute_tolerance:
            raise FeatureParityError(f"{column}: maximum absolute error {maximum}")
        errors[column], counts[column] = maximum, count
    return ParityResult(len(left), errors, counts)
