"""Hot-hand / mean-reversion research flags (Wave 4b).

Thesis (betting-the-regression): when recent form (L3) is elevated vs season
baseline with stable minutes, sportsbooks often inflate the line — research
flag leans UNDER. Never invents lines; RESEARCH_ONLY display signal.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_Z_THRESHOLD = 1.0
DEFAULT_MINUTES_STABLE_RATIO = 0.15  # |MIN_L5/MIN_SEASON - 1| < this
_STAT_HOT = ("PTS", "REB", "AST", "FG3M", "STL", "BLK")


def _group_shift_roll_l3(
    df: pd.DataFrame,
    col: str,
    group_keys: list[str],
) -> pd.Series:
    shifted = df.groupby(group_keys, sort=False)[col].shift(1)
    tmp = df[group_keys].copy()
    tmp["_v"] = shifted
    out = tmp.groupby(group_keys, sort=False)["_v"].rolling(3, min_periods=2).mean()
    return out.reset_index(level=list(range(len(group_keys))), drop=True)


def _group_shift_roll_std(
    df: pd.DataFrame,
    col: str,
    group_keys: list[str],
    window: int = 15,
) -> pd.Series:
    shifted = df.groupby(group_keys, sort=False)[col].shift(1)
    tmp = df[group_keys].copy()
    tmp["_v"] = shifted
    out = tmp.groupby(group_keys, sort=False)["_v"].rolling(window, min_periods=5).std()
    return out.reset_index(level=list(range(len(group_keys))), drop=True)


def attach_hot_hand_features(
    df: pd.DataFrame,
    *,
    z_threshold: float = DEFAULT_Z_THRESHOLD,
    minutes_stable_ratio: float = DEFAULT_MINUTES_STABLE_RATIO,
) -> pd.DataFrame:
    """
    Add per-stat L3, hot z-score, minutes-stability, and fade-under flag.

    All rolling inputs are shift(1). Does not replace L5/L10/L2.
    """
    out = df.copy()
    if "PLAYER_ID" not in out.columns or "GAME_DATE" not in out.columns:
        raise ValueError("DATA_NOT_AVAILABLE: PLAYER_ID and GAME_DATE required")
    if "SEASON" not in out.columns:
        out["SEASON"] = pd.to_datetime(out["GAME_DATE"], errors="coerce").dt.year
    group_keys = ["PLAYER_ID", "SEASON"]

    # Minutes stability (shared across stats)
    if "MIN_L5" in out.columns and "MIN_SEASON" in out.columns:
        ratio = (pd.to_numeric(out["MIN_L5"], errors="coerce") / pd.to_numeric(out["MIN_SEASON"], errors="coerce")).replace(
            [np.inf, -np.inf], np.nan
        )
        out["MINUTES_STABLE"] = (ratio - 1.0).abs() < float(minutes_stable_ratio)
        out["MINUTES_TREND_RATIO"] = ratio
    else:
        out["MINUTES_STABLE"] = False
        out["MINUTES_TREND_RATIO"] = np.nan

    for stat in _STAT_HOT:
        if stat not in out.columns:
            continue
        # The season baseline is this module's reference point, and it comes
        # from build_feature_matrix. Without it, out.get() returns None and
        # the arithmetic below fails deep inside numpy with a TypeError that
        # says nothing about the real cause. Abstain by name instead.
        if f"{stat}_SEASON" not in out.columns:
            logger.warning(
                "hot_hand: %s_SEASON absent — skipping %s. Run "
                "build_feature_matrix before attach_hot_hand_features.",
                stat, stat,
            )
            continue
        out[f"{stat}_L3"] = _group_shift_roll_l3(out, stat, group_keys)
        season = pd.to_numeric(out[f"{stat}_SEASON"], errors="coerce")
        l3 = pd.to_numeric(out[f"{stat}_L3"], errors="coerce")
        sd = _group_shift_roll_std(out, stat, group_keys, window=15)
        # Floor sd so early-season rows don't explode
        sd = sd.fillna(np.sqrt(season.clip(lower=0.5))).clip(lower=0.5)
        z = (l3 - season) / sd
        out[f"{stat}_HOT_Z"] = z
        # Research fade-under when hot + stable minutes
        fade = (z > float(z_threshold)) & out["MINUTES_STABLE"].fillna(False)
        out[f"{stat}_HOT_HAND_FADE_UNDER"] = fade.astype("boolean")
        out[f"{stat}_HOT_HAND_STATUS"] = np.where(
            l3.isna() | season.isna(),
            "DATA_NOT_AVAILABLE",
            np.where(
                fade,
                "RESEARCH_FADE_UNDER",
                np.where(z < -float(z_threshold), "RESEARCH_COLD_STREAK", "NEUTRAL"),
            ),
        )

    out.attrs["hot_hand_z_threshold"] = float(z_threshold)
    out.attrs["hot_hand_minutes_stable_ratio"] = float(minutes_stable_ratio)
    return out


def hot_hand_note_for_row(row: pd.Series, market: str) -> str | None:
    """Short research note for exports / UI."""
    status = row.get(f"{market}_HOT_HAND_STATUS")
    if status is None or status == "NEUTRAL" or (isinstance(status, float) and np.isnan(status)):
        return None
    if status == "DATA_NOT_AVAILABLE":
        return None
    z = row.get(f"{market}_HOT_Z")
    z_s = f"{float(z):.2f}" if z is not None and pd.notna(z) else "?"
    if status == "RESEARCH_FADE_UNDER":
        return (
            f"HOT_HAND_FADE_UNDER z={z_s} (L3 vs season; stable minutes) — "
            "RESEARCH_ONLY, not a stake"
        )
    if status == "RESEARCH_COLD_STREAK":
        return f"COLD_STREAK z={z_s} — RESEARCH_ONLY context"
    return str(status)
