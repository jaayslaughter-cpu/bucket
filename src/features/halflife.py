"""Recency half-life + empirical-Bayes shrink (Wave 2 additive features).

Does not replace L5 / L10 / L15 / SEASON / BASELINE / L2 — only adds columns.
All inputs are shift(1) within player-season.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.season import drop_season_key, player_season_keys

# Half-life in games for exponential recency weighting (pandas ewm halflife).
DEFAULT_HALFLIFE_GAMES = 10.0
# Prior strength for shrink toward season EWM (higher → more shrinkage).
DEFAULT_SHRINK_K = 8.0

_STAT_FOR_HL = ("PTS", "REB", "AST", "FG3M", "STL", "BLK", "MIN")


def _group_shift_ewm_halflife(
    df: pd.DataFrame,
    col: str,
    group_keys: list[str],
    *,
    halflife: float,
    min_periods: int = 1,
) -> pd.Series:
    shifted = df.groupby(group_keys, sort=False)[col].shift(1)
    tmp = df[group_keys].copy()
    tmp["_v"] = shifted
    out = (
        tmp.groupby(group_keys, sort=False)["_v"]
        .ewm(halflife=halflife, min_periods=min_periods, adjust=False)
        .mean()
    )
    return out.reset_index(level=list(range(len(group_keys))), drop=True)


def _prior_game_counts(df: pd.DataFrame, group_keys: list[str]) -> pd.Series:
    """Games already observed before this row (0 on debut)."""
    # After shift hygiene, cumcount of rows within group equals prior games for ewm input.
    return df.groupby(group_keys, sort=False).cumcount()


def attach_halflife_shrink_features(
    df: pd.DataFrame,
    *,
    halflife_games: float = DEFAULT_HALFLIFE_GAMES,
    shrink_k: float = DEFAULT_SHRINK_K,
) -> pd.DataFrame:
    """
    Add ``{stat}_HL`` and ``{stat}_HL_SHRINK`` (and optional ``{stat}_L2_HL``).

    ``HL_SHRINK`` = w * HL + (1-w) * SEASON with w = n / (n + k).
    """
    out = df.copy()
    if "PLAYER_ID" not in out.columns or "GAME_DATE" not in out.columns:
        raise ValueError("DATA_NOT_AVAILABLE: PLAYER_ID and GAME_DATE required for halflife")
    # A derived season stays PRIVATE to this layer. Writing it into the frame
    # under the public name made this layer's guess every later layer's fact —
    # halflife runs first — and defeated minutes_weighted's abstention. See
    # src/features/season.py.
    out, group_keys = player_season_keys(out)
    n_prior = _prior_game_counts(out, group_keys).astype(float)
    w = n_prior / (n_prior + float(shrink_k))

    for stat in _STAT_FOR_HL:
        if stat not in out.columns:
            continue
        src = stat
        # Prefer DNP-qualified series if present from layer1 (already dropped); use raw.
        out[f"{stat}_HL"] = _group_shift_ewm_halflife(
            out, src, group_keys, halflife=float(halflife_games), min_periods=1
        )
        season_col = f"{stat}_SEASON"
        if season_col in out.columns:
            season = pd.to_numeric(out[season_col], errors="coerce")
            hl = pd.to_numeric(out[f"{stat}_HL"], errors="coerce")
            out[f"{stat}_HL_SHRINK"] = w * hl + (1.0 - w) * season
        else:
            out[f"{stat}_HL_SHRINK"] = out[f"{stat}_HL"]

    # Additive L2 variant using shrink mean (does not overwrite {stat}_L2).
    # Do NOT fillna(1.0) on pace/fatigue — that fabricates neutral context
    # when the builder intentionally left values null.
    for stat in ("PTS", "REB", "AST", "FG3M", "STL", "BLK"):
        shrink_col = f"{stat}_HL_SHRINK"
        if shrink_col not in out.columns:
            continue
        base = pd.to_numeric(out[shrink_col], errors="coerce")
        if "fatigue_multiplier" in out.columns:
            base = base * pd.to_numeric(out["fatigue_multiplier"], errors="coerce")
        if "MIN_L5" in out.columns and "MIN_SEASON" in out.columns:
            minutes_ratio = (
                pd.to_numeric(out["MIN_L5"], errors="coerce")
                / pd.to_numeric(out["MIN_SEASON"], errors="coerce")
            ).replace([np.inf, -np.inf], np.nan).clip(0.5, 1.5)
            base = base * minutes_ratio
        out[f"{stat}_L2_HL"] = base
        if "PACE_MULTIPLIER" in out.columns:
            out[f"{stat}_L2_HL_PACE"] = base * pd.to_numeric(
                out["PACE_MULTIPLIER"], errors="coerce"
            )

    out.attrs["halflife_games"] = float(halflife_games)
    out.attrs["halflife_shrink_k"] = float(shrink_k)
    return drop_season_key(out)


#: The suffixes THIS LAYER owns: the ones no other producer makes. The core
#: set -- L5, L10, SEASON, BASELINE, L2, L2_PACE -- belongs to
#: ``builder.build_feature_matrix``, which builds PRA's members of it from the
#: components itself.
PRA_LAYER_SUFFIXES: tuple[str, ...] = ("L15", "HL", "HL_SHRINK", "L2_HL")


def attach_pra_component_rollups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Additive PRA aggregates for the HALFLIFE suffixes (no new shift logic).

    IT USED TO REWRITE FIVE COLUMNS IT DID NOT OWN. Its suffix list ran from
    L5 through L2_HL, and ``build_feature_matrix`` had already produced
    PRA_L5, PRA_L10, PRA_SEASON, PRA_BASELINE and PRA_L2 by rolling the PRA
    column -- so this layer silently replaced them with a different
    definition, in the middle of a block whose own comment says "Each of these
    ONLY adds columns; none rewrites the core L2/L5/L10/BASELINE set above".

    Both definitions agreed on every row of the archive panel (PTS, REB and
    AST are null on 0 of its 214,381) and differed on a live panel, where
    ``player_game_logs.pts/reb/ast`` are independently nullable. The visible
    damage was PRA_L2_PACE: the builder computed it from the PRA_L2 it had
    just made, this layer then replaced PRA_L2 and not PRA_L2_PACE, and the
    two disagreed on 8 of 14 rows in a fixture with one partial game --
    including rows where PACE_MULTIPLIER was exactly 1.0.

    The definition that survived is the one this layer used, for the reasons
    in ``builder.attach_pra_from_components``, which is now the single
    implementation and is called from both places. This function keeps only
    the suffixes nothing else produces, so the block's claim about itself is
    true again.

    Safe when component columns exist; otherwise leaves PRA columns absent.
    The raw ``PRA`` label is NOT written here -- the builder owns it, and
    writing it twice from two places is the defect this function is being
    narrowed to avoid repeating.
    """
    out = df.copy()
    if not {"PTS", "REB", "AST"}.issubset(out.columns):
        return out

    # Lazily imported: builder imports this module inside its own closures, so
    # a module-scope import here would close the cycle.
    from src.features.builder import attach_pra_from_components

    return attach_pra_from_components(out, PRA_LAYER_SUFFIXES)
