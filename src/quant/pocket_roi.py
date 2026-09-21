"""BookieX-style pocket ROI board for manual paper research (Wave 5a).

Historical unit ROI by pocket (prop_stat, side, confidence, grade, book).
Never sizes stakes or places wagers — research audit only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.quant.historical_store import HistoricalStore
from src.utils.timezones import DISPLAY_TZ_NAME, format_pacific_iso, now_pacific

POCKET_DISCLAIMER = (
    "Pocket ROI is a historical research audit of YOUR manual paper log — "
    "not bankroll advice, not Kelly, not a live P&L guarantee."
)

POCKET_DIMS = (
    "prop_stat",
    "bet_side",
    "confidence_tier",
    "edge_letter_grade",
    "model_name",
    "bookmaker",
)


def _settled_frame(store: HistoricalStore) -> pd.DataFrame:
    df = store.load_frame()
    if df.empty:
        return df
    return df[df["bet_result"].isin(["WIN", "LOSS", "PUSH"])].copy()


def _pocket_row(g: pd.DataFrame, *, pocket: str, key: str) -> dict[str, Any]:
    stake = float(g["unit_stake"].fillna(1.0).sum())
    pnl = float(g["profit_loss"].fillna(0.0).sum())
    wins = int((g["bet_result"] == "WIN").sum())
    losses = int((g["bet_result"] == "LOSS").sum())
    pushes = int((g["bet_result"] == "PUSH").sum())
    n = int(len(g))
    hit_rate = (wins / (wins + losses)) if (wins + losses) else None
    return {
        "pocket": pocket,
        "key": key if key and str(key) != "nan" else "UNKNOWN",
        "n": n,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "hit_rate": round(hit_rate, 4) if hit_rate is not None else None,
        "stake_units": round(stake, 4),
        "profit_loss": round(pnl, 4),
        "roi": round(pnl / stake, 6) if stake else None,
    }


def build_pocket_roi_board(store: HistoricalStore) -> dict[str, Any]:
    """
    Aggregate settled paper bets into pockets for model improvement review.
    """
    base: dict[str, Any] = {
        "report_timestamp_pt": format_pacific_iso(now_pacific()),
        "timezone_display": DISPLAY_TZ_NAME,
        "placement_mode": "MANUAL_ONLY",
        "disclaimer": POCKET_DISCLAIMER,
        "pockets": [],
        "overall": None,
    }
    settled = _settled_frame(store)
    if settled.empty:
        base["status"] = "DATA_NOT_AVAILABLE"
        base["reason"] = "No settled paper bets yet"
        return base

    # Ensure optional research tags exist
    for col in POCKET_DIMS:
        if col not in settled.columns:
            settled[col] = None

    base["overall"] = _pocket_row(settled, pocket="ALL", key="ALL")
    pockets: list[dict[str, Any]] = []
    for dim in POCKET_DIMS:
        series = settled[dim].fillna("UNKNOWN").astype(str)
        for key, g in settled.groupby(series):
            pockets.append(_pocket_row(g, pocket=dim, key=str(key)))

    # Sort: worst ROI first within each pocket (surfaces what to improve)
    pockets.sort(key=lambda r: (r["pocket"], r["roi"] if r["roi"] is not None else 0.0))
    base["pockets"] = pockets
    base["status"] = "OK"
    base["n_settled"] = int(len(settled))
    base["note"] = "Lower-ROI pockets are candidates for model / process review"
    return base


def pocket_board_to_dataframe(board: dict[str, Any]) -> pd.DataFrame:
    rows = list(board.get("pockets") or [])
    if board.get("overall"):
        rows = [board["overall"], *rows]
    return pd.DataFrame(rows)


def write_pocket_roi_csv(store: HistoricalStore, path: Path | str) -> dict[str, Any]:
    board = build_pocket_roi_board(store)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df = pocket_board_to_dataframe(board)
    df.to_csv(p, index=False)
    return {**board, "out": str(p), "rows_written": int(len(df))}
