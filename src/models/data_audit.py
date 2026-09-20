"""Data readiness audit for model comparison (no fabricated fills)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.utils.timezones import DISPLAY_TZ_NAME, format_pacific_iso, now_pacific


def audit_player_panel(df: pd.DataFrame, *, dataset_name: str = "player_panel") -> dict[str, Any]:
    total = len(df)
    dup = 0
    if {"PLAYER_ID", "GAME_ID"}.issubset(df.columns) and total:
        dup = int(df.duplicated(subset=["PLAYER_ID", "GAME_ID"]).sum())
    missing_player = int(df["PLAYER_ID"].isna().sum()) if "PLAYER_ID" in df.columns else total
    missing_event = int(df["GAME_ID"].isna().sum()) if "GAME_ID" in df.columns else total
    # Count ROWS with any missing target, not cells. Summing per column made
    # one row missing all three targets read as three rejected rows, so the
    # figure could exceed total_rows and was never a row count at all.
    target_cols = [c for c in ("PTS", "REB", "AST") if c in df.columns]
    if target_cols and total:
        any_target_missing = pd.concat(
            [pd.to_numeric(df[c], errors="coerce").isna() for c in target_cols], axis=1
        ).any(axis=1)
        missing_target = int(any_target_missing.sum())
    else:
        missing_target = total if not target_cols else 0
    missing_pregame_ts = total  # tipoff UTC not on default box panel
    leakage = 0
    if {"GAME_DATE", "LAST_INCLUDED_GAME_DATE"}.issubset(df.columns):
        g = pd.to_datetime(df["GAME_DATE"], errors="coerce")
        last = pd.to_datetime(df["LAST_INCLUDED_GAME_DATE"], errors="coerce")
        leakage = int((last.notna() & (last >= g)).sum())

    # Union of the missing-key masks, not a sum: a row lacking BOTH keys was
    # counted twice, letting rejected_rows exceed total_rows.
    if total:
        player_missing = (
            df["PLAYER_ID"].isna() if "PLAYER_ID" in df.columns
            else pd.Series(True, index=df.index)
        )
        event_missing = (
            df["GAME_ID"].isna() if "GAME_ID" in df.columns
            else pd.Series(True, index=df.index)
        )
        rejected = int((player_missing | event_missing).sum())
    else:
        rejected = 0
    valid = max(0, total - rejected)
    field_status = {
        "game_date": "AVAILABLE_AND_VERIFIED" if "GAME_DATE" in df.columns else "MISSING",
        "pregame_timestamp": "MISSING",
        "player_id": "AVAILABLE_AND_VERIFIED" if "PLAYER_ID" in df.columns else "MISSING",
        "event_id": "AVAILABLE_AND_VERIFIED" if "GAME_ID" in df.columns else "MISSING",
        "team": "AVAILABLE_AND_VERIFIED" if "TEAM_ABBREVIATION" in df.columns else "MISSING",
        "opponent": "AVAILABLE_AND_VERIFIED" if "OPPONENT_ABBREVIATION" in df.columns else "MISSING",
        "home_away": "AVAILABLE_AND_VERIFIED" if "IS_HOME" in df.columns else "MISSING",
        "minutes": "AVAILABLE_AND_VERIFIED" if "MIN" in df.columns else "MISSING",
        "points": "AVAILABLE_AND_VERIFIED" if "PTS" in df.columns else "MISSING",
        "rebounds": "AVAILABLE_AND_VERIFIED" if "REB" in df.columns else "MISSING",
        "assists": "AVAILABLE_AND_VERIFIED" if "AST" in df.columns else "MISSING",
        "fg3m": "AVAILABLE_AND_VERIFIED" if "FG3M" in df.columns else "MISSING",
        "stl": "AVAILABLE_AND_VERIFIED" if "STL" in df.columns else "MISSING",
        "blk": "AVAILABLE_AND_VERIFIED" if "BLK" in df.columns else "MISSING",
        "spread_total": "AVAILABLE_BUT_NEEDS_VALIDATION",
        "injury_starter": "AVAILABLE_BUT_NEEDS_VALIDATION" if "BBS_OUT_FLAG" in df.columns else "MISSING",
        "sportsbook_prop": "AVAILABLE_BUT_NEEDS_VALIDATION",
        "settlement": "AVAILABLE_AND_VERIFIED" if "PTS" in df.columns else "MISSING",
    }
    return {
        "report_timestamp_pt": format_pacific_iso(now_pacific()),
        "timezone_display": DISPLAY_TZ_NAME,
        "dataset_name": dataset_name,
        "total_rows": total,
        "valid_rows": valid,
        "rejected_rows": rejected,
        "duplicate_rows": dup,
        "missing_player_id_rows": missing_player,
        "missing_event_id_rows": missing_event,
        "missing_target_rows": missing_target,
        "missing_pregame_timestamp_rows": missing_pregame_ts,
        "possible_leakage_rows": leakage,
        "notes": (
            f"User-facing times are {DISPLAY_TZ_NAME}; storage remains UTC. "
            "Pregame tipoff not on default boxscore panel; RESEARCH_LINE uses L10 not sportsbook."
        ),
        "field_status": field_status,
    }


def make_demo_panel(n_players: int = 24, n_games: int = 70, seed: int = 42) -> pd.DataFrame:
    """
    DEMO ONLY synthetic panel for wiring tests.

    Clearly tagged; must never be mixed into real ``outputs/`` without the
    demo flag. Sized so a month-long validation window holds enough rows to
    exercise calibration binning — a panel too small to calibrate would
    leave that code path unverified.

    The generator draws each stat independently from a Poisson whose mean
    scales with minutes. Real NBA stats are correlated and overdispersed,
    so demo metrics say nothing about real performance. They only prove the
    wiring runs.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    teams = ["LAL", "BOS", "GSW", "NYK", "MIA", "DEN", "PHX", "DAL"]
    start = pd.Timestamp("2024-11-01")
    for p in range(n_players):
        pid = f"DEMO{p:03d}"
        team = teams[p % len(teams)]
        for g in range(n_games):
            day = start + pd.Timedelta(days=g * 2 + (p % 2))
            opp = teams[(p + 3 + g) % len(teams)]
            if opp == team:  # a team never plays itself
                opp = teams[(p + 4 + g) % len(teams)]
            mins = float(rng.uniform(18, 36))
            pts = float(rng.poisson(mins * 0.45))
            reb = float(rng.poisson(mins * 0.15))
            ast = float(rng.poisson(mins * 0.12))
            rows.append(
                {
                    "PLAYER_ID": pid,
                    "PLAYER_NAME": f"Demo Player {p}",
                    "GAME_ID": f"00DEMO{g:06d}",
                    "GAME_DATE": day,
                    "SEASON": "2024-25",
                    "TEAM_ABBREVIATION": team,
                    "OPPONENT_ABBREVIATION": opp,
                    "IS_HOME": int(g % 2 == 0),
                    "MIN": mins,
                    "PTS": pts,
                    "REB": reb,
                    "AST": ast,
                    "FG3M": float(rng.poisson(1.2)),
                    "STL": float(rng.poisson(0.8)),
                    "BLK": float(rng.poisson(0.4)),
                    "DEMO_ONLY": True,
                }
            )
    return pd.DataFrame(rows)
