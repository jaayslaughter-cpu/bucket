"""OOF probability calibration (isotonic / sigmoid). Never fit on eval fold."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

CalibMethod = Literal["isotonic", "sigmoid"]


class ProbabilityCalibrator:
    """Fit only on out-of-fold predictions from earlier data."""

    def __init__(self, method: CalibMethod = "isotonic") -> None:
        self.method = method
        self._iso: IsotonicRegression | None = None
        self._platt: LogisticRegression | None = None
        self.fitted = False

    def fit(self, y_true: np.ndarray, p_raw: np.ndarray) -> "ProbabilityCalibrator":
        y = np.asarray(y_true).astype(int)
        p = np.clip(np.asarray(p_raw).astype(float), 1e-6, 1 - 1e-6)
        if len(y) < 50:
            raise ValueError("DATA_NOT_AVAILABLE: need >=50 OOF rows to calibrate")
        if self.method == "isotonic":
            self._iso = IsotonicRegression(out_of_bounds="clip")
            self._iso.fit(p, y)
        else:
            self._platt = LogisticRegression(max_iter=1000)
            self._platt.fit(p.reshape(-1, 1), y)
        self.fitted = True
        return self

    def transform(self, p_raw: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Calibrator not fitted")
        p = np.clip(np.asarray(p_raw).astype(float), 1e-6, 1 - 1e-6)
        if self._iso is not None:
            return np.asarray(self._iso.transform(p), dtype=float)
        assert self._platt is not None
        return self._platt.predict_proba(p.reshape(-1, 1))[:, 1]

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"method": self.method, "iso": self._iso, "platt": self._platt}, path)

    @classmethod
    def load(cls, path: Path | str) -> "ProbabilityCalibrator":
        blob = joblib.load(path)
        obj = cls(method=blob["method"])
        obj._iso = blob["iso"]
        obj._platt = blob["platt"]
        obj.fitted = True
        return obj


def choose_calibrator(
    y_true: np.ndarray,
    p_raw: np.ndarray,
    methods: list[CalibMethod] | None = None,
) -> tuple[ProbabilityCalibrator, dict[str, float]]:
    """Pick method with best OOF Brier (fit on first 70%, score on last 30% chrono)."""
    from sklearn.metrics import brier_score_loss, log_loss

    methods = methods or ["isotonic", "sigmoid"]
    n = len(y_true)
    cut = int(n * 0.7)
    if cut < 40 or n - cut < 20:
        raise ValueError("DATA_NOT_AVAILABLE: insufficient rows to choose calibrator")
    y_fit, y_sc = y_true[:cut], y_true[cut:]
    p_fit, p_sc = p_raw[:cut], p_raw[cut:]
    best: ProbabilityCalibrator | None = None
    best_brier = float("inf")
    scores: dict[str, float] = {}
    for m in methods:
        cal = ProbabilityCalibrator(method=m)
        cal.fit(y_fit, p_fit)
        p_hat = cal.transform(p_sc)
        b = float(brier_score_loss(y_sc, p_hat))
        ll = float(log_loss(y_sc, p_hat, labels=[0, 1]))
        scores[f"{m}_brier"] = b
        scores[f"{m}_log_loss"] = ll
        if b < best_brier:
            best_brier = b
            best = cal
    assert best is not None
    # Refit on all OOF for deployment
    final = ProbabilityCalibrator(method=best.method)
    final.fit(y_true, p_raw)
    scores["chosen"] = 1.0 if best.method == "isotonic" else 0.0
    scores["chosen_method_isotonic"] = float(best.method == "isotonic")
    logger.info("calibrator chosen=%s scores=%s", best.method, scores)
    return final, scores


def reliability_table(
    y_true: np.ndarray,
    p_hat: np.ndarray,
    *,
    n_bins: int = 10,
) -> list[dict[str, Any]]:
    y = np.asarray(y_true).astype(int)
    p = np.clip(np.asarray(p_hat).astype(float), 0, 1)
    edges = np.linspace(0, 1, n_bins + 1)
    rows: list[dict[str, Any]] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi if i < n_bins - 1 else p <= hi)
        if not mask.any():
            continue
        mean_p = float(p[mask].mean())
        obs = float(y[mask].mean())
        rows.append(
            {
                "probability_bin": f"{lo:.1f}-{hi:.1f}",
                "n_predictions": int(mask.sum()),
                "mean_predicted_probability": round(mean_p, 4),
                "observed_frequency": round(obs, 4),
                "calibration_gap": round(mean_p - obs, 4),
            }
        )
    return rows


def expected_calibration_error(
    y_true: np.ndarray,
    p_hat: np.ndarray,
    *,
    n_bins: int = 10,
    min_bin_coverage: float = 0.8,
) -> dict[str, Any]:
    """
    Weighted ECE with a non-empty-bin coverage gate.

    If fewer than ``min_bin_coverage`` of the ``n_bins`` bins contain predictions,
    ``ece`` is None and ``gate_passed`` is False — sparse reliability diagrams
    must not win model selection (Wave 1 / calibration-driven selection).
    """
    table = reliability_table(y_true, p_hat, n_bins=n_bins)
    nonempty = len(table)
    coverage = nonempty / float(n_bins) if n_bins else 0.0
    gate_passed = coverage >= float(min_bin_coverage)
    if not table:
        return {
            "ece": None,
            "bin_coverage": coverage,
            "nonempty_bins": 0,
            "n_bins": n_bins,
            "gate_passed": False,
            "table": table,
        }
    ntot = sum(t["n_predictions"] for t in table)
    ece = sum(abs(t["calibration_gap"]) * t["n_predictions"] for t in table) / max(ntot, 1)
    return {
        "ece": round(float(ece), 6) if gate_passed else None,
        "ece_ungated": round(float(ece), 6),
        "bin_coverage": round(coverage, 4),
        "nonempty_bins": nonempty,
        "n_bins": n_bins,
        "gate_passed": gate_passed,
        "table": table,
    }


def select_model_dual(
    market_scores: dict[str, dict[str, Any]],
    *,
    exclude: set[str] | None = None,
) -> dict[str, Any]:
    """
    Dual winners: accuracy-driven vs calibration-driven.

    - accuracy: highest classification accuracy (ties → lower Brier)
    - calibration: lowest gated ECE (ties → lower Brier); falls back to Brier if no ECE
    """
    exclude = exclude or {"ensemble"}
    eligible = {k: v for k, v in market_scores.items() if k not in exclude and v.get("n_predictions", 0)}

    accuracy_winner = None
    if eligible:
        ranked_acc = sorted(
            eligible.items(),
            key=lambda t: (
                -(t[1].get("accuracy") if t[1].get("accuracy") is not None else -1.0),
                t[1].get("brier_score") if t[1].get("brier_score") is not None else 9.0,
            ),
        )
        accuracy_winner = {
            "winner": ranked_acc[0][0],
            "accuracy": ranked_acc[0][1].get("accuracy"),
            "brier_score": ranked_acc[0][1].get("brier_score"),
            "selection": "accuracy",
            "note": "Highest accuracy; not a profitability claim",
        }

    calib_pool = {
        k: v
        for k, v in eligible.items()
        if v.get("ece_gate_passed") and v.get("calibration_error") is not None
    }
    if calib_pool:
        ranked_cal = sorted(
            calib_pool.items(),
            key=lambda t: (
                t[1].get("calibration_error") if t[1].get("calibration_error") is not None else 9.0,
                t[1].get("brier_score") if t[1].get("brier_score") is not None else 9.0,
            ),
        )
        calibration_winner = {
            "winner": ranked_cal[0][0],
            "calibration_error": ranked_cal[0][1].get("calibration_error"),
            "brier_score": ranked_cal[0][1].get("brier_score"),
            "bin_coverage": ranked_cal[0][1].get("bin_coverage"),
            "selection": "calibration",
            "note": "Lowest gated ECE; not a profitability claim",
        }
    else:
        # Fallback: Brier among eligible
        ranked_b = sorted(
            eligible.items(),
            key=lambda t: t[1].get("brier_score") if t[1].get("brier_score") is not None else 9.0,
        )
        calibration_winner = {
            "winner": ranked_b[0][0] if ranked_b else None,
            "calibration_error": None,
            "brier_score": ranked_b[0][1].get("brier_score") if ranked_b else None,
            "selection": "calibration_fallback_brier",
            "note": "ECE gate failed for all models; fell back to lowest Brier",
        }

    return {
        "accuracy_driven": accuracy_winner,
        "calibration_driven": calibration_winner,
    }
