"""Selective Sports-EV-inspired feature columns (Wave 5a).

Additive, leakage-safe only:
  - USAGE_PROXY_L10 — box-score usage when FGA/FTA/TOV present; else NaN
  - STREAK_ABOVE / STREAK_BELOW vs prior season mean (per stat)
  - OPP_{STAT}_ALLOWED_L10 — opponent team allowed L10 (prior games only)

Never invents lines, vig, or market prices. RESEARCH_ONLY.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_STREAK_STATS = ("PTS", "REB", "AST", "FG3M")
_OPP_ALLOWED_STATS = ("PTS", "REB", "AST", "FG3M", "STL", "BLK")


def _group_shift_roll(
    df: pd.DataFrame,
    col: str,
    group_keys: list[str],
    *,
    window: int,
    min_periods: int,
) -> pd.Series:
    shifted = df.groupby(group_keys, sort=False)[col].shift(1)
    tmp = df[group_keys].copy()
    tmp["_v"] = shifted
    out = tmp.groupby(group_keys, sort=False)["_v"].rolling(window, min_periods=min_periods).mean()
    return out.reset_index(level=list(range(len(group_keys))), drop=True)


def attach_usage_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Player usage proxy from box attempts, then L10 of shift(1) values.

    Formula (per game): (FGA + 0.44*FTA + TOV) / team possessions.
    Without FGA/FTA/TOV → USAGE_PROXY / USAGE_PROXY_L10 left NaN (DATA_NOT_AVAILABLE).
    """
    out = df.copy()
    needed = {"FGA", "FTA", "TOV", "TEAM_ABBREVIATION", "GAME_ID", "PLAYER_ID", "SEASON", "GAME_DATE"}
    if not needed.issubset(out.columns):
        out["USAGE_PROXY"] = np.nan
        out["USAGE_PROXY_L10"] = np.nan
        out.attrs["usage_proxy_status"] = "DATA_NOT_AVAILABLE"
        return out

    for c in ("FGA", "FTA", "TOV", "OREB"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    player_poss = out["FGA"] + 0.44 * out["FTA"] + out["TOV"]
    team_agg: dict[str, tuple[str, str]] = {
        "TEAM_FGA": ("FGA", "sum"),
        "TEAM_FTA": ("FTA", "sum"),
        "TEAM_TOV": ("TOV", "sum"),
    }
    if "OREB" in out.columns:
        team_agg["TEAM_OREB"] = ("OREB", "sum")
    team = out.groupby(["SEASON", "TEAM_ABBREVIATION", "GAME_ID"], as_index=False).agg(**team_agg)
    if "TEAM_OREB" not in team.columns:
        team["TEAM_OREB"] = 0.0
    team["TEAM_POSS"] = (
        team["TEAM_FGA"] - team["TEAM_OREB"] + team["TEAM_TOV"] + 0.44 * team["TEAM_FTA"]
    ).replace(0, np.nan)

    out = out.merge(
        team[["SEASON", "TEAM_ABBREVIATION", "GAME_ID", "TEAM_POSS"]],
        on=["SEASON", "TEAM_ABBREVIATION", "GAME_ID"],
        how="left",
    )
    # Same-game raw (internal) then shift(1) so game-T never sees game-T usage
    out["_USAGE_RAW"] = (player_poss / out["TEAM_POSS"]).clip(0.0, 1.0)
    out = out.drop(columns=["TEAM_POSS"], errors="ignore")

    group_keys = ["PLAYER_ID", "SEASON"]
    out["USAGE_PROXY"] = out.groupby(group_keys, sort=False)["_USAGE_RAW"].shift(1)
    out["USAGE_PROXY_L10"] = _group_shift_roll(
        out, "_USAGE_RAW", group_keys, window=10, min_periods=3
    )
    out = out.drop(columns=["_USAGE_RAW"], errors="ignore")
    out.attrs["usage_proxy_status"] = "OK"
    return out

def attach_form_streaks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Consecutive prior games above/below the player's prior season mean.

    Uses only shift(1) outcomes vs shift(1) expanding season mean.
    """
    out = df.copy()
    if "PLAYER_ID" not in out.columns or "GAME_DATE" not in out.columns:
        return out
    if "SEASON" not in out.columns:
        out["SEASON"] = pd.to_datetime(out["GAME_DATE"], errors="coerce").dt.year

    group_keys = ["PLAYER_ID", "SEASON"]
    out = out.sort_values([c for c in (*group_keys, "GAME_DATE", "GAME_ID") if c in out.columns]).reset_index(
        drop=True
    )

    for stat in _STREAK_STATS:
        if stat not in out.columns:
            out[f"{stat}_STREAK_ABOVE"] = np.nan
            out[f"{stat}_STREAK_BELOW"] = np.nan
            continue

        season_mean_col = f"{stat}_SEASON"
        if season_mean_col in out.columns:
            baseline = pd.to_numeric(out[season_mean_col], errors="coerce")
        else:
            # Fallback: expanding mean of prior games
            shifted = out.groupby(group_keys, sort=False)[stat].shift(1)
            tmp = out[group_keys].copy()
            tmp["_v"] = shifted
            baseline = (
                tmp.groupby(group_keys, sort=False)["_v"]
                .expanding(min_periods=3)
                .mean()
                .reset_index(level=list(range(len(group_keys))), drop=True)
            )

        prior = out.groupby(group_keys, sort=False)[stat].shift(1)
        above = (prior > baseline).astype("float")
        below = (prior < baseline).astype("float")
        # NaN prior → break streak
        above = above.where(prior.notna() & baseline.notna(), other=0.0)
        below = below.where(prior.notna() & baseline.notna(), other=0.0)

        out[f"{stat}_STREAK_ABOVE"] = _run_length_within_groups(out, above, group_keys)
        out[f"{stat}_STREAK_BELOW"] = _run_length_within_groups(out, below, group_keys)

    return out


def _run_length_within_groups(
    df: pd.DataFrame,
    flag: pd.Series,
    group_keys: list[str],
) -> pd.Series:
    """Count consecutive 1.0 flags ending at each row (within group)."""
    work = df[group_keys].copy()
    work["_f"] = flag.fillna(0.0).astype(float)
    # Reset when flag==0
    work["_block"] = (work["_f"] == 0).groupby([work[k] for k in group_keys]).cumsum()
    work["_rl"] = work.groupby(group_keys + ["_block"], sort=False)["_f"].cumsum()
    # Zero-flag rows should show 0 streak
    work.loc[work["_f"] == 0, "_rl"] = 0.0
    return work["_rl"]


def attach_opp_allowed_l10(df: pd.DataFrame) -> pd.DataFrame:
    """
    Opponent points/rebounds/... allowed over prior L10 (team as defender).

    Built from team-game aggregates where this team was OPPONENT of scorers,
    then joined onto player rows by OPPONENT_ABBREVIATION + GAME_ID.
    """
    out = df.copy()
    needed = {"OPPONENT_ABBREVIATION", "TEAM_ABBREVIATION", "GAME_ID", "SEASON", "GAME_DATE"}
    if not needed.issubset(out.columns):
        for stat in _OPP_ALLOWED_STATS:
            out[f"OPP_{stat}_ALLOWED_L10"] = np.nan
        out.attrs["opp_allowed_status"] = "DATA_NOT_AVAILABLE"
        return out

    agg_map: dict[str, tuple[str, str]] = {
        "GAME_DATE": ("GAME_DATE", "first"),
        "OPPONENT_ABBREVIATION": ("OPPONENT_ABBREVIATION", "first"),
    }
    for stat in _OPP_ALLOWED_STATS:
        if stat in out.columns:
            agg_map[f"{stat}_SCORED"] = (stat, "sum")

    if len(agg_map) <= 2:
        for stat in _OPP_ALLOWED_STATS:
            out[f"OPP_{stat}_ALLOWED_L10"] = np.nan
        out.attrs["opp_allowed_status"] = "DATA_NOT_AVAILABLE"
        return out

    team = out.groupby(["SEASON", "TEAM_ABBREVIATION", "GAME_ID"], as_index=False).agg(**{
        k: v for k, v in agg_map.items()
    })

    # Flip: scoring team → defending team is OPPONENT; allowed = scored against them
    allowed = team.rename(
        columns={
            "TEAM_ABBREVIATION": "SCORING_TEAM",
            "OPPONENT_ABBREVIATION": "DEFENDING_TEAM",
        }
    )
    allowed = allowed.sort_values(["DEFENDING_TEAM", "SEASON", "GAME_DATE"]).reset_index(drop=True)

    roll_cols = []
    for stat in _OPP_ALLOWED_STATS:
        src = f"{stat}_SCORED"
        if src not in allowed.columns:
            continue
        tmp = allowed.rename(columns={"DEFENDING_TEAM": "TEAM_ABBREVIATION"})
        col = f"OPP_{stat}_ALLOWED_L10"
        tmp[col] = _group_shift_roll(
            tmp, src, ["TEAM_ABBREVIATION", "SEASON"], window=10, min_periods=3
        )
        allowed[col] = tmp[col].values
        roll_cols.append(col)

    keep = ["SEASON", "DEFENDING_TEAM", "GAME_ID", *roll_cols]
    out = out.merge(
        allowed[keep],
        left_on=["SEASON", "OPPONENT_ABBREVIATION", "GAME_ID"],
        right_on=["SEASON", "DEFENDING_TEAM", "GAME_ID"],
        how="left",
    ).drop(columns=["DEFENDING_TEAM"], errors="ignore")

    for stat in _OPP_ALLOWED_STATS:
        col = f"OPP_{stat}_ALLOWED_L10"
        if col not in out.columns:
            out[col] = np.nan

    out.attrs["opp_allowed_status"] = "OK"
    return out


def attach_sports_ev_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compose Wave 5a Sports-EV selective columns onto a feature panel."""
    out = attach_usage_proxy(df)
    out = attach_form_streaks(out)
    out = attach_opp_allowed_l10(out)
    return out
