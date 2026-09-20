"""Teammate-absence cascade stub (Wave 4b).

Kalshi-style usage redistribution requires verified pairwise absence history.
This module never invents +usage from outs. Without verified OUT teammates on
the same game, status is DATA_NOT_AVAILABLE.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def attach_teammate_cascade_stub(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cascade audit columns; do not apply invented multipliers.

    Requires ``TEAM_ABBREVIATION`` + ``GAME_ID`` and optionally ``BBS_OUT_FLAG``.
    """
    out = df.copy()
    out["CASCADE_STATUS"] = "DATA_NOT_AVAILABLE"
    out["CASCADE_TEAMMATE_OUTS"] = 0
    out["CASCADE_USAGE_MULT"] = np.nan
    out["CASCADE_NOTES"] = (
        "No verified pairwise absence effects — will not invent usage bumps"
    )

    needed = {"TEAM_ABBREVIATION", "GAME_ID", "PLAYER_ID"}
    if not needed.issubset(out.columns):
        out["CASCADE_NOTES"] = "DATA_NOT_AVAILABLE: need TEAM_ABBREVIATION, GAME_ID, PLAYER_ID"
        return out

    if "BBS_OUT_FLAG" not in out.columns:
        out["CASCADE_NOTES"] = (
            "DATA_NOT_AVAILABLE: BBS_OUT_FLAG absent — cascade abstains "
            "(never invents injuries)"
        )
        return out

    flag = pd.to_numeric(out["BBS_OUT_FLAG"], errors="coerce").fillna(0).astype(int)
    # Count other players on same team-game marked OUT
    tmp = out[["TEAM_ABBREVIATION", "GAME_ID", "PLAYER_ID"]].copy()
    tmp["_out"] = flag
    team_outs = (
        tmp.groupby(["TEAM_ABBREVIATION", "GAME_ID"], sort=False)["_out"].transform("sum")
    )
    # Exclude self from count when this player is out
    teammate_outs = (team_outs - flag).clip(lower=0)
    out["CASCADE_TEAMMATE_OUTS"] = teammate_outs.astype(int)

    has_outs = teammate_outs > 0
    out.loc[has_outs, "CASCADE_STATUS"] = "NEEDS_VERIFIED_PAIRWISE"
    out.loc[has_outs, "CASCADE_NOTES"] = (
        "Teammate OUT flags present but no verified pairwise cascade table — "
        "usage multiplier left DATA_NOT_AVAILABLE (not invented)"
    )
    out.loc[~has_outs, "CASCADE_STATUS"] = "NO_TEAMMATE_OUTS"
    out.loc[~has_outs, "CASCADE_NOTES"] = "No teammate OUT flags on this team-game"

    # Player themselves out → zero projection context, still no cascade invent
    self_out = flag == 1
    out.loc[self_out, "CASCADE_STATUS"] = "PLAYER_OUT"
    out.loc[self_out, "CASCADE_NOTES"] = "Player BBS_OUT_FLAG=1 — cascade N/A for this row"
    return out


def cascade_warning_for_row(row: pd.Series) -> str | None:
    status = row.get("CASCADE_STATUS")
    if status in {None, "NO_TEAMMATE_OUTS"}:
        return None
    if status == "DATA_NOT_AVAILABLE":
        return None
    if status == "NEEDS_VERIFIED_PAIRWISE":
        n = int(row.get("CASCADE_TEAMMATE_OUTS") or 0)
        return f"CASCADE: {n} teammate OUT(s) — pairwise effects DATA_NOT_AVAILABLE"
    if status == "PLAYER_OUT":
        return "CASCADE: player OUT flag set"
    return None


def cascade_summary(df: pd.DataFrame) -> dict[str, Any]:
    if "CASCADE_STATUS" not in df.columns:
        return {"status": "DATA_NOT_AVAILABLE"}
    vc = df["CASCADE_STATUS"].value_counts(dropna=False).to_dict()
    return {
        "status": "OK",
        "counts": {str(k): int(v) for k, v in vc.items()},
        "note": "RESEARCH_ONLY stub — no invented usage multipliers",
    }
