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

    Requires ``TEAM_ABBREVIATION`` + ``GAME_ID``, and one of two absence inputs:

    ``BBS_TEAMMATES_OUT`` — inactive teammates per (team, game), produced by
    ``src.ingestion.inactive_players.attach_absence_features`` from the
    official pregame inactive list. PREFERRED, and the only one that works on
    this panel: the panel holds only players who APPEARED (median 10 rows per
    team-game, no row with MIN == 0), so an inactive player has no row here and
    a per-row flag cannot see him.

    ``BBS_OUT_FLAG`` — a per-row OUT flag, kept for a caller whose frame *does*
    carry rows for absent players. On a panel of appearances only, this is 0
    everywhere and the count below is 0 everywhere with it.
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

    if "BBS_TEAMMATES_OUT" in out.columns:
        # The counted path. Already per (team, game) and already excludes this
        # row's player, who appeared by construction, so nothing is subtracted.
        counted = pd.to_numeric(out["BBS_TEAMMATES_OUT"], errors="coerce")
        known = counted.notna()
        if not known.any():
            out["CASCADE_NOTES"] = (
                "DATA_NOT_AVAILABLE: BBS_TEAMMATES_OUT present but null on every "
                "row — the inactive list was not pulled for these games"
            )
            return out
        teammate_outs = counted.fillna(0).astype(int)
        flag = pd.Series(0, index=out.index, dtype=int)
        out["CASCADE_TEAMMATE_OUTS"] = teammate_outs
        out.loc[~known, "CASCADE_TEAMMATE_OUTS"] = 0
    elif "BBS_OUT_FLAG" in out.columns:
        flag = pd.to_numeric(out["BBS_OUT_FLAG"], errors="coerce").fillna(0).astype(int)
        # Count other players on same team-game marked OUT
        tmp = out[["TEAM_ABBREVIATION", "GAME_ID", "PLAYER_ID"]].copy()
        tmp["_out"] = flag
        team_outs = (
            tmp.groupby(["TEAM_ABBREVIATION", "GAME_ID"], sort=False)["_out"].transform("sum")
        )
        # Exclude self from count when this player is out
        teammate_outs = (team_outs - flag).clip(lower=0)
        known = pd.Series(True, index=out.index)
    else:
        out["CASCADE_NOTES"] = (
            "DATA_NOT_AVAILABLE: neither BBS_TEAMMATES_OUT nor BBS_OUT_FLAG "
            "present — cascade abstains (never invents injuries)"
        )
        return out
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

    # Rows whose game was never pulled are not "no teammates out" — they are
    # unknown, and they say so rather than reading as a healthy roster.
    unknown = ~known
    if unknown.any():
        out.loc[unknown, "CASCADE_STATUS"] = "DATA_NOT_AVAILABLE"
        out.loc[unknown, "CASCADE_NOTES"] = (
            "No inactive list pulled for this game — absence unknown, not absent"
        )
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
