"""Paper-book probability calibration / reliability chart (Wave 5b).

Uses settled manual paper bets only. RESEARCH audit — not a profitability claim.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.prob_calibration import expected_calibration_error, reliability_table
from src.quant.historical_store import HistoricalStore
from src.utils.timezones import DISPLAY_TZ_NAME, format_pacific_iso, now_pacific

CALIB_DISCLAIMER = (
    "Paper calibration from YOUR manual settled log — research audit only. "
    "Not bankroll advice. Not a live P&L guarantee."
)


def _side_aligned_probs(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Convert model P(over) + bet side into P(side) vs hit indicator."""
    probs: list[float] = []
    hits: list[float] = []
    for _, r in df.iterrows():
        if r.get("bet_result") == "PUSH":
            continue
        p = r.get("model_prob")
        if p is None or (isinstance(p, float) and not np.isfinite(p)):
            continue
        side = str(r.get("bet_side", "")).lower()
        won = r.get("bet_result") == "WIN"
        if side in {"over", "o"}:
            probs.append(float(p))
            hits.append(1.0 if won else 0.0)
        elif side in {"under", "u"}:
            probs.append(1.0 - float(p))
            hits.append(1.0 if won else 0.0)
    return np.asarray(probs, dtype=float), np.asarray(hits, dtype=float)


def paper_reliability_report(
    store: HistoricalStore,
    *,
    n_bins: int = 10,
    min_bin_coverage: float = 0.5,
) -> dict[str, Any]:
    """
    Reliability diagram + ECE/Brier on settled paper bets with model_prob.
    """
    base: dict[str, Any] = {
        "report_timestamp_pt": format_pacific_iso(now_pacific()),
        "timezone_display": DISPLAY_TZ_NAME,
        "placement_mode": "MANUAL_ONLY",
        "disclaimer": CALIB_DISCLAIMER,
        "reliability_table": [],
        "chart_points": [],
    }
    df = store.load_frame()
    if df.empty:
        base["status"] = "DATA_NOT_AVAILABLE"
        base["reason"] = "No paper bets logged yet"
        return base

    settled = df[df["bet_result"].isin(["WIN", "LOSS", "PUSH"])].copy()
    if settled.empty:
        base["status"] = "DATA_NOT_AVAILABLE"
        base["reason"] = "No settled paper bets yet"
        return base

    p, y = _side_aligned_probs(settled)
    if len(p) < 5:
        base["status"] = "DATA_NOT_AVAILABLE"
        base["reason"] = f"Need >=5 graded non-push bets with model_prob (have {len(p)})"
        base["n_settled"] = int(len(settled))
        return base

    table = reliability_table(y, p, n_bins=n_bins)
    ece = expected_calibration_error(
        y, p, n_bins=n_bins, min_bin_coverage=min_bin_coverage
    )
    brier = float(np.mean((p - y) ** 2))

    chart = [
        {
            "mean_predicted": t["mean_predicted_probability"],
            "observed": t["observed_frequency"],
            "n": t["n_predictions"],
            "bin": t["probability_bin"],
        }
        for t in table
    ]

    base.update(
        {
            "status": "OK",
            "n_settled": int(len(settled)),
            "n_scored": int(len(p)),
            "brier": round(brier, 6),
            "ece": ece.get("ece"),
            "ece_ungated": ece.get("ece_ungated"),
            "bin_coverage": ece.get("bin_coverage"),
            "ece_gate_passed": ece.get("gate_passed"),
            "reliability_table": table,
            "chart_points": chart,
            "perfect_calibration_line": [
                {"mean_predicted": 0.0, "observed": 0.0},
                {"mean_predicted": 1.0, "observed": 1.0},
            ],
            "note": "Points near the diagonal are better calibrated",
        }
    )
    return base


def write_paper_reliability_csv(
    store: HistoricalStore,
    path: Path | str,
    *,
    n_bins: int = 10,
) -> dict[str, Any]:
    report = paper_reliability_report(store, n_bins=n_bins)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = list(report.get("reliability_table") or [])
    pd.DataFrame(rows).to_csv(p, index=False)
    return {**report, "out": str(p), "rows_written": int(len(rows))}
