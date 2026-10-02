"""Minutes-weighted recent averages.

Adds leakage-safe columns:

  - ``{STAT}_MW_L{w}`` — last-w prior games, weighted by minutes vs the prior
    season-to-date mean
  - Combo aliases: ``PR_MW_L{w}``, ``PA_MW_L{w}``, ``RA_MW_L{w}``, ``PRA_MW_L{w}``

Weight rule:
  MIN < 0.70 × prior_MIN_mean → 0.5
  MIN > 0.85 × prior_MIN_mean → 1.5
  else → 1.0

LEAKAGE. Every input is strictly prior: the stat and the minutes are
``.shift(1)`` within player-season, and the baseline is either the panel's
``MIN_SEASON`` — which ``builder._expanding_prior_mean`` computes as
``shift(1).expanding().mean()`` — or, absent that, an expanding mean of the
shifted minutes computed here. Verified rather than assumed, because the whole
value of this module depends on it.

PRODUCED, BUT NOT IN THE FEATURE CONTRACT, AND THAT IS A MEASURED DECISION.
Within-season |r| of each column against the panel's already-listed features,
on the real 214,381-row panel (204,529 overlapping rows, weighted by overlap —
the same methodology as the table in ``models/labels.py``):

    column        vs {STAT}_L5   vs {STAT}_BASELINE   vs {STAT}_L10
    PTS_MW_L5        0.991            0.984               0.952
    REB_MW_L5        0.989            0.980               0.943
    AST_MW_L5        0.992            0.984               0.953
    FG3M_MW_L5       0.988            0.974               0.918
    STL_MW_L5        0.976            0.949               0.830
    BLK_MW_L5        0.984            0.965               0.885

0.976 to 0.992 against ``{STAT}_L5`` puts these INSIDE the band this repository
already excluded the whole halflife family for (0.971-0.990), where the A/B
measured Brier getting worse on every fold. A 0.5/1.0/1.5 reweighting of the
same five games is a second copy of one number, not a second opinion.

So the columns are built and are registered as a ``scripts/feature_ab.py``
layer, and they are deliberately ABSENT from ``labels.default_feature_cols``.
Measure with ``feature_ab --layer minutes_weighted --wire-under-test`` before
promoting them; the correlation above is a prediction, and this project's
standard is to test it rather than argue from it.

RESEARCH_ONLY — never invents lines.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_STATS = ("PTS", "REB", "AST", "STL", "BLK", "FG3M")
_COMBOS = ("PR", "PA", "RA", "PRA")
DEFAULT_WINDOW = 5
LOW_MIN_RATIO = 0.70
HIGH_MIN_RATIO = 0.85
LOW_WEIGHT = 0.5
HIGH_WEIGHT = 1.5
MID_WEIGHT = 1.0


def _emitted_columns(window: int = DEFAULT_WINDOW) -> list[str]:
    """Every column this layer emits, so the abstain path cannot drift from the
    happy path. They disagreed before: the stats were named from ``window`` and
    the combos were hardcoded to ``L5``."""
    return [f"{stat}_MW_L{window}" for stat in _STATS] + [
        f"{combo}_MW_L{window}" for combo in _COMBOS
    ]


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
    # SEASON is required, not derived. An earlier version fell back to
    # GAME_DATE.dt.year, which splits an NBA season at 1 January: the per-player
    # grouping would restart mid-season, resetting both the expanding minutes
    # baseline and the rolling window for every player on New Year's Day. That
    # is a silently wrong answer, so it abstains instead.
    needed = {"PLAYER_ID", "GAME_DATE", "MIN", "SEASON"}
    missing = sorted(needed - set(out.columns))
    if missing:
        for column in _emitted_columns(window):
            out[column] = np.nan
        out.attrs["minutes_weighted_status"] = "DATA_NOT_AVAILABLE"
        out.attrs["minutes_weighted_reason"] = f"missing {missing}"
        return out

    group_keys = ["PLAYER_ID", "SEASON"]
    # THE CALLER'S ROW ORDER IS RESTORED BEFORE RETURNING. builder applies
    # layers as `df = attach(df)`, so a layer that hands back a re-sorted frame
    # silently reorders the whole feature matrix for every later layer and for
    # the caller. Sorting by (player, season, date) is player-major, which is
    # NOT chronological, and the comparison path's folds are positional over a
    # date-sorted frame. fatigue_load restores order for the same reason.
    out["_mw_row"] = np.arange(len(out))
    out = out.sort_values(
        [c for c in (*group_keys, "GAME_DATE", "GAME_ID") if c in out.columns],
        kind="mergesort",
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

    # Combo aliases (P+R / P+A / R+A / P+R+A) from the weighted components.
    # The denominators are identical, so the sum of the weighted means IS the
    # weighted mean of the sum.
    #
    # NAMED FROM `window`, not hardcoded to L5. An earlier version wrote
    # PR_MW_L5 whatever the window, so window=10 produced PTS_MW_L10 beside a
    # PR_MW_L5 holding ten-game data — a column whose name contradicted its
    # contents.
    pts = out.get(f"PTS_MW_L{window}")
    reb = out.get(f"REB_MW_L{window}")
    ast = out.get(f"AST_MW_L{window}")
    for alias, parts in (
        (f"PR_MW_L{window}", (pts, reb)),
        (f"PA_MW_L{window}", (pts, ast)),
        (f"RA_MW_L{window}", (reb, ast)),
        (f"PRA_MW_L{window}", (pts, reb, ast)),
    ):
        if any(part is None for part in parts):
            out[alias] = np.nan
            continue
        total = parts[0]
        for part in parts[1:]:
            total = total + part
        out[alias] = total

    out = out.drop(columns=["_MW_WEIGHT"], errors="ignore")
    out = out.sort_values("_mw_row", kind="mergesort").drop(columns=["_mw_row"])
    out.index = df.index
    out.attrs["minutes_weighted_status"] = "OK"
    out.attrs["minutes_weighted_window"] = int(window)
    return out
