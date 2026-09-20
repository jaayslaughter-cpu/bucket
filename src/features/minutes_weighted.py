"""Minutes-weighted recent averages (native steal from parlayparlor).

Interactive nba_api CLI is not used. This module adds leakage-safe columns:

  - ``{STAT}_MW_L5`` — last-5 prior games, weighted by minutes vs prior season mean
  - Combo aliases: ``PR_MW_L5``, ``PA_MW_L5``, ``RA_MW_L5``, ``PRA_MW_L5``

Weight rule (parlayparlor):
  MIN < 0.70 × prior_MIN_mean → 0.5
  MIN > 0.85 × prior_MIN_mean → 1.5
  else → 1.0

All inputs use ``.shift(1)``. RESEARCH_ONLY — never invents lines.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_STATS = ("PTS", "REB", "AST", "STL", "BLK", "FG3M")
DEFAULT_WINDOW = 5
LOW_MIN_RATIO = 0.70
HIGH_MIN_RATIO = 0.85
LOW_WEIGHT = 0.5
HIGH_WEIGHT = 1.5
MID_WEIGHT = 1.0


def _minutes_weights(
    min_prior: pd.Series,
    min_baseline: pd.Series,
) -> pd.Series:
    """Per-row weights from prior-game minutes vs prior season mean."""
    base = pd.to_numeric(min_baseline, errors="coerce")
    mins = pd.to_numeric(min_prior, errors="coerce")
    w = pd.Series(MID_WEIGHT, index=mins.index, dtype=float)
    low = mins.notna() & base.notna() & (mins < LOW_MIN_RATIO * base)
    high = mins.notna() & base.notna() & (mins > HIGH_MIN_RATIO * base)
    w = w.where(~low, LOW_WEIGHT)
    w = w.where(~high, HIGH_WEIGHT)
    w = w.where(mins.notna() & base.notna(), np.nan)
    return w


def _weighted_roll(
    df: pd.DataFrame,
    values: pd.Series,
    weights: pd.Series,
    group_keys: list[str],
    *,
    window: int,
    min_periods: int,
) -> pd.Series:
    tmp = df[group_keys].copy()
    tmp["_vw"] = values * weights
    tmp["_w"] = weights
    g = tmp.groupby(group_keys, sort=False)
    num = g["_vw"].rolling(window, min_periods=min_periods).sum()
    den = g["_w"].rolling(window, min_periods=min_periods).sum()
    num = num.reset_index(level=list(range(len(group_keys))), drop=True)
    den = den.reset_index(level=list(range(len(group_keys))), drop=True)
    out = num / den.replace(0, np.nan)
    return out


def attach_minutes_weighted_features(
    df: pd.DataFrame,
    *,
    window: int = DEFAULT_WINDOW,
) -> pd.DataFrame:
    """
    Add minutes-weighted L{window} features for counting stats + combo aliases.
    """
    out = df.copy()
    needed = {"PLAYER_ID", "GAME_DATE", "MIN"}
    if not needed.issubset(out.columns):
        for stat in _STATS:
            out[f"{stat}_MW_L{window}"] = np.nan
        out["PR_MW_L5"] = np.nan
        out["PA_MW_L5"] = np.nan
        out["RA_MW_L5"] = np.nan
        out["PRA_MW_L5"] = np.nan
        out.attrs["minutes_weighted_status"] = "DATA_NOT_AVAILABLE"
        return out

    if "SEASON" not in out.columns:
        out["SEASON"] = pd.to_datetime(out["GAME_DATE"], errors="coerce").dt.year
    group_keys = ["PLAYER_ID", "SEASON"]
    out = out.sort_values(
        [c for c in (*group_keys, "GAME_DATE", "GAME_ID") if c in out.columns]
    ).reset_index(drop=True)

    min_prior = out.groupby(group_keys, sort=False)["MIN"].shift(1)
    if "MIN_SEASON" in out.columns:
        min_base = pd.to_numeric(out["MIN_SEASON"], errors="coerce")
    else:
        tmp = out[group_keys].copy()
        tmp["_m"] = min_prior
        min_base = (
            tmp.groupby(group_keys, sort=False)["_m"]
            .expanding(min_periods=3)
            .mean()
            .reset_index(level=list(range(len(group_keys))), drop=True)
        )

    weights = _minutes_weights(min_prior, min_base)
    out["_MW_WEIGHT"] = weights

    min_periods = max(2, window // 2)
    for stat in _STATS:
        if stat not in out.columns:
            out[f"{stat}_MW_L{window}"] = np.nan
            continue
        prior = out.groupby(group_keys, sort=False)[stat].shift(1)
        out[f"{stat}_MW_L{window}"] = _weighted_roll(
            out, prior, weights, group_keys, window=window, min_periods=min_periods
        )

    # Combo aliases (parlayparlor P+R / P+A / R+A / P+R+A) from weighted components
    pts = out.get(f"PTS_MW_L{window}")
    reb = out.get(f"REB_MW_L{window}")
    ast = out.get(f"AST_MW_L{window}")
    out["PR_MW_L5"] = pts + reb if pts is not None and reb is not None else np.nan
    out["PA_MW_L5"] = pts + ast if pts is not None and ast is not None else np.nan
    out["RA_MW_L5"] = reb + ast if reb is not None and ast is not None else np.nan
    out["PRA_MW_L5"] = (
        pts + reb + ast if pts is not None and reb is not None and ast is not None else np.nan
    )

    out = out.drop(columns=["_MW_WEIGHT"], errors="ignore")
    out.attrs["minutes_weighted_status"] = "OK"
    out.attrs["minutes_weighted_window"] = int(window)
    return out
