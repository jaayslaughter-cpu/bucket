"""Chronological walk-forward splits for model comparison (no random K-fold)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import pandas as pd


@dataclass(frozen=True)
class ChronoSplit:
    train_idx: pd.Index
    validation_idx: pd.Index
    holdout_idx: pd.Index
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    holdout_start: pd.Timestamp | None
    holdout_end: pd.Timestamp | None


def sort_by_game_date(df: pd.DataFrame, date_col: str = "GAME_DATE") -> pd.DataFrame:
    """Sort chronologically, RESETTING the index.

    The reset is deliberate and is part of this module's contract: the
    ``ChronoSplit`` indices returned by the split functions refer to this
    sorted, re-indexed frame, not to the caller's original ordering.
    Callers must therefore apply them to a frame prepared the same way —
    ``compare_models_on_panel`` does so by resetting before splitting.
    Preserving the caller's labels here would silently change which rows
    every existing split selects.
    """
    if date_col not in df.columns:
        raise ValueError(f"DATA_NOT_AVAILABLE: missing {date_col}")
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], utc=False)
    return out.sort_values(date_col).reset_index(drop=True)


def expanding_window_splits(
    df: pd.DataFrame,
    *,
    date_col: str = "GAME_DATE",
    min_train_rows: int = 500,
    validation_days: int = 30,
    step_days: int = 14,
    holdout_days: int = 30,
) -> list[ChronoSplit]:
    """
    Expanding train → next validation_days → optional final holdout at the end.

    Never places a later row in train than an earlier validation row.
    """
    if step_days <= 0:
        raise ValueError("CONFIG_INVALID: step_days must be > 0 or the cursor never advances")
    if validation_days <= 0:
        raise ValueError("CONFIG_INVALID: validation_days must be > 0")

    work = sort_by_game_date(df, date_col=date_col)
    if work.empty:
        return []

    dates = pd.to_datetime(work[date_col])
    min_d, max_d = dates.min(), dates.max()
    holdout_start = max_d - timedelta(days=holdout_days - 1)
    usable_end = holdout_start - timedelta(days=1)

    splits: list[ChronoSplit] = []
    cursor = min_d + timedelta(days=max(1, validation_days))
    while cursor <= usable_end:
        val_start = cursor - timedelta(days=validation_days - 1)
        val_end = cursor
        train_mask = dates < val_start
        val_mask = (dates >= val_start) & (dates <= val_end)
        hold_mask = dates >= holdout_start
        if int(train_mask.sum()) < min_train_rows or int(val_mask.sum()) == 0:
            cursor = cursor + timedelta(days=step_days)
            continue
        splits.append(
            ChronoSplit(
                train_idx=work.index[train_mask],
                validation_idx=work.index[val_mask],
                holdout_idx=work.index[hold_mask],
                train_start=dates[train_mask].min(),
                train_end=dates[train_mask].max(),
                validation_start=dates[val_mask].min(),
                validation_end=dates[val_mask].max(),
                holdout_start=dates[hold_mask].min() if hold_mask.any() else None,
                holdout_end=dates[hold_mask].max() if hold_mask.any() else None,
            )
        )
        cursor = cursor + timedelta(days=step_days)
    return splits


def fixed_cutoff_split(
    df: pd.DataFrame,
    *,
    train_end: str,
    validation_end: str,
    date_col: str = "GAME_DATE",
) -> ChronoSplit:
    """Single chronological split by inclusive date cutoffs (YYYY-MM-DD)."""
    work = sort_by_game_date(df, date_col=date_col)
    dates = pd.to_datetime(work[date_col])
    te = pd.Timestamp(train_end)
    ve = pd.Timestamp(validation_end)
    train_mask = dates <= te
    val_mask = (dates > te) & (dates <= ve)
    hold_mask = dates > ve
    if not train_mask.any() or not val_mask.any():
        raise ValueError("DATA_NOT_AVAILABLE: empty train or validation after cutoff")
    return ChronoSplit(
        train_idx=work.index[train_mask],
        validation_idx=work.index[val_mask],
        holdout_idx=work.index[hold_mask],
        train_start=dates[train_mask].min(),
        train_end=dates[train_mask].max(),
        validation_start=dates[val_mask].min(),
        validation_end=dates[val_mask].max(),
        holdout_start=dates[hold_mask].min() if hold_mask.any() else None,
        holdout_end=dates[hold_mask].max() if hold_mask.any() else None,
    )
