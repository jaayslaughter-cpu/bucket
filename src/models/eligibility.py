"""Eligibility gates and KS drift audits for Wave 1 model comparison."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def prior_game_counts(
    df: pd.DataFrame,
    *,
    player_col: str = "PLAYER_ID",
    date_col: str = "GAME_DATE",
) -> pd.Series:
    """Number of prior rows per player (strictly earlier GAME_DATE), index-aligned to ``df``."""
    work = df[[player_col, date_col]].copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    order = work.sort_values([player_col, date_col]).index
    counts = pd.Series(0, index=df.index, dtype=int)
    counts.loc[order] = (
        work.loc[order].groupby(player_col, sort=False).cumcount().to_numpy()
    )
    return counts


def eligibility_warnings_for_row(
    row: pd.Series,
    *,
    prior_games: int,
    min_prior_games: int = 10,
    min_minutes_l5: float = 12.0,
) -> list[str]:
    """Return abstain warnings; never invent injury/starter fields."""
    warnings: list[str] = []
    if prior_games < min_prior_games:
        warnings.append(
            f"ABSTAIN: prior_games={prior_games} < min_prior_games={min_prior_games} (warm-up)"
        )
    min_l5 = row.get("MIN_L5")
    if min_l5 is not None and pd.notna(min_l5) and float(min_l5) < min_minutes_l5:
        warnings.append(
            f"ABSTAIN: MIN_L5={float(min_l5):.1f} < min_minutes_l5={min_minutes_l5}"
        )
    return warnings


def attach_eligibility_warnings(
    df: pd.DataFrame,
    *,
    min_prior_games: int = 10,
    min_minutes_l5: float = 12.0,
) -> pd.DataFrame:
    """Add ``eligibility_warnings`` list column (empty list when eligible)."""
    out = df.copy()
    counts = prior_game_counts(out)
    out["prior_games"] = counts
    warns: list[list[str]] = []
    for idx, row in out.iterrows():
        warns.append(
            eligibility_warnings_for_row(
                row,
                prior_games=int(counts.loc[idx]),
                min_prior_games=min_prior_games,
                min_minutes_l5=min_minutes_l5,
            )
        )
    out["eligibility_warnings"] = warns
    return out


def ks_feature_drift(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_cols: list[str],
    *,
    p_threshold: float = 0.01,
) -> list[dict[str, Any]]:
    """
    Two-sample KS test train vs validation per numeric feature.

    Features with p < threshold are flagged ``DRIFT`` (log/report only —
    not silently dropped from the model).
    """
    rows: list[dict[str, Any]] = []
    for col in feature_cols:
        if col not in train.columns or col not in validation.columns:
            continue
        a = pd.to_numeric(train[col], errors="coerce").dropna().to_numpy()
        b = pd.to_numeric(validation[col], errors="coerce").dropna().to_numpy()
        if len(a) < 30 or len(b) < 30:
            rows.append(
                {
                    "feature_name": col,
                    "ks_statistic": None,
                    "p_value": None,
                    "status": "INSUFFICIENT_SAMPLE",
                    "train_n": int(len(a)),
                    "validation_n": int(len(b)),
                }
            )
            continue
        if np.nanstd(a) == 0 and np.nanstd(b) == 0:
            rows.append(
                {
                    "feature_name": col,
                    "ks_statistic": 0.0,
                    "p_value": 1.0,
                    "status": "OK_CONSTANT",
                    "train_n": int(len(a)),
                    "validation_n": int(len(b)),
                }
            )
            continue
        stat, pval = ks_2samp(a, b)
        status = "DRIFT" if float(pval) < p_threshold else "OK"
        rows.append(
            {
                "feature_name": col,
                "ks_statistic": round(float(stat), 6),
                "p_value": round(float(pval), 6),
                "status": status,
                "train_n": int(len(a)),
                "validation_n": int(len(b)),
            }
        )
    return rows
