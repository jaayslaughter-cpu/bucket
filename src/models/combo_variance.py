"""PRA / combo joint variance and residual σ calibration (Wave 2).

Basketball-Modelling-inspired: mean = Σ component means;
variance ≈ (Σ component vars) × correlation fudge (default 1.1).

Residual bootstrap estimates an empirical σ scale from walk-forward residuals.
RESEARCH_ONLY — does not invent lines or odds.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.models.residuals import CountDispersion, over_under_push_from_dispersion

# Positive dependence among PTS/REB/AST inflates the variance of their sum
# above the sum of their variances. This default is a documented starting
# point, NOT a measurement — prefer fit_combo_variance_inflation(), which
# estimates the same quantity directly from realised data.
DEFAULT_COMBO_VAR_FUDGE = 1.1


class ComboProjection(BaseModel):
    mean: float | None = None
    variance: float | None = None
    std: float | None = None
    status: str = "OK"
    reason: str | None = None
    components: dict[str, float] = Field(default_factory=dict)
    var_fudge: float = DEFAULT_COMBO_VAR_FUDGE


class DispersionCalibration(BaseModel):
    """Empirical scale so calibrated_sd = raw_sd * sigma_scale."""

    sigma_scale: float = 1.0
    n_residuals: int = 0
    residual_std: float | None = None
    raw_pred_std_mean: float | None = None
    bootstrap_ci_low: float | None = None
    bootstrap_ci_high: float | None = None
    notes: str = "RESEARCH_ONLY residual dispersion; not a stake rule"


def _as_var(mean: float | None, sd: float | None) -> float | None:
    if mean is None or not np.isfinite(mean):
        return None
    if sd is not None and np.isfinite(sd) and sd > 0:
        return float(sd) ** 2
    # Poisson-like fallback for count means
    return max(float(mean), 0.0)


def pra_joint_projection(
    pts_mean: float | None,
    reb_mean: float | None,
    ast_mean: float | None,
    *,
    pts_sd: float | None = None,
    reb_sd: float | None = None,
    ast_sd: float | None = None,
    var_fudge: float = DEFAULT_COMBO_VAR_FUDGE,
) -> ComboProjection:
    """Combine PTS+REB+AST means with inflated independent variance."""
    comps = {"PTS": pts_mean, "REB": reb_mean, "AST": ast_mean}
    if any(v is None or not np.isfinite(v) for v in comps.values()):
        return ComboProjection(
            status="DATA_NOT_AVAILABLE",
            reason="All of PTS/REB/AST means must be finite",
            components={k: float(v) for k, v in comps.items() if v is not None and np.isfinite(v)},
            var_fudge=var_fudge,
        )
    mean = float(pts_mean) + float(reb_mean) + float(ast_mean)  # type: ignore[arg-type]
    vars_ = [
        _as_var(float(pts_mean), pts_sd),  # type: ignore[arg-type]
        _as_var(float(reb_mean), reb_sd),  # type: ignore[arg-type]
        _as_var(float(ast_mean), ast_sd),  # type: ignore[arg-type]
    ]
    if any(v is None for v in vars_):
        return ComboProjection(
            status="DATA_NOT_AVAILABLE",
            reason="Could not form component variances",
            var_fudge=var_fudge,
        )
    variance = float(sum(vars_)) * float(var_fudge)  # type: ignore[arg-type]
    return ComboProjection(
        mean=round(mean, 4),
        variance=round(variance, 4),
        std=round(float(np.sqrt(max(variance, 1e-9))), 4),
        status="OK",
        components={"PTS": float(pts_mean), "REB": float(reb_mean), "AST": float(ast_mean)},  # type: ignore[arg-type]
        var_fudge=var_fudge,
    )


def pra_line_probabilities(
    combo: ComboProjection,
    line: float,
    *,
    family: str = "normal",
) -> dict[str, Any]:
    """P(over/under/push) for a PRA line using the joint mean/std."""
    if combo.status != "OK" or combo.mean is None:
        return {
            "status": "DATA_NOT_AVAILABLE",
            "reason": combo.reason or "combo unavailable",
            "probability_over": None,
            "probability_under": None,
            "probability_push": None,
        }
    # The repo's push math parameterises the normal as
    # sd = sigma_scale * sqrt(mean), so convert the joint sd into that
    # scale. Routing through over_under_push_from_dispersion keeps ONE
    # implementation of the over/under/push rule; a second copy would
    # drift, which is why the earlier one was removed.
    sigma_scale = 1.0
    if combo.std is not None and combo.mean > 0:
        sigma_scale = float(combo.std) / float(np.sqrt(max(combo.mean, 1e-6)))

    dispersion = CountDispersion(
        family=family,
        phi=1.0,
        n_train_rows=0,
        selection_scores={},
        sigma_scale=max(sigma_scale, 0.1),
        fallback_reason=(
            "scale derived from the combo joint variance, not fitted by "
            "fit_count_dispersion"
        ),
    )
    return over_under_push_from_dispersion(combo.mean, line, dispersion)


def fit_combo_variance_inflation(
    component_values: "pd.DataFrame",
    *,
    components: tuple[str, ...] = ("PTS", "REB", "AST"),
    min_rows: int = 100,
) -> dict[str, float]:
    """
    Measure the variance inflation instead of assuming it.

    Var(A+B+C) exceeds Var(A)+Var(B)+Var(C) by exactly the covariance
    terms, so the ratio between them IS the inflation factor — no fudge
    required. Returns the measured ratio alongside the default, so a
    caller can see how far the assumption was off.

    Abstains rather than returning a number from too few rows.
    """
    missing = [c for c in components if c not in component_values.columns]
    if missing:
        return {
            "status": float("nan"),
            "reason_missing": float(len(missing)),
            "inflation": float("nan"),
        }

    frame = component_values[list(components)].apply(pd.to_numeric, errors="coerce").dropna()
    if len(frame) < min_rows:
        return {
            "inflation": float("nan"),
            "n_rows": float(len(frame)),
            "default_used": DEFAULT_COMBO_VAR_FUDGE,
        }

    independent = float(sum(frame[c].var(ddof=1) for c in components))
    joint = float(frame[list(components)].sum(axis=1).var(ddof=1))
    if independent <= 0:
        return {"inflation": float("nan"), "n_rows": float(len(frame))}

    inflation = joint / independent
    return {
        "inflation": round(inflation, 4),
        "n_rows": float(len(frame)),
        "independent_variance": round(independent, 4),
        "joint_variance": round(joint, 4),
        "default_used": DEFAULT_COMBO_VAR_FUDGE,
    }


def fit_dispersion_from_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    raw_sd: np.ndarray | None = None,
    *,
    n_bootstrap: int = 400,
    seed: int = 42,
) -> DispersionCalibration:
    """
    Estimate sigma_scale from residuals.

    If ``raw_sd`` provided: scale = std(residual / raw_sd).
    Else: scale so mean absolute residual matches scaled Poisson-ish sqrt(pred).
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[mask], yp[mask]
    if len(yt) < 30:
        return DispersionCalibration(
            sigma_scale=1.0,
            n_residuals=int(len(yt)),
            notes="DATA_NOT_AVAILABLE: need >=30 residuals; default scale=1.0",
        )
    resid = yt - yp
    resid_std = float(np.std(resid, ddof=1))
    if raw_sd is not None:
        rs = np.asarray(raw_sd, dtype=float)[mask]
        ok = np.isfinite(rs) & (rs > 1e-6)
        if ok.sum() < 30:
            return DispersionCalibration(
                sigma_scale=1.0,
                n_residuals=int(len(yt)),
                residual_std=round(resid_std, 4),
                notes="DATA_NOT_AVAILABLE: insufficient raw_sd; default scale=1.0",
            )
        z = resid[ok] / rs[ok]
        point = float(np.std(z, ddof=1))
        raw_mean = float(np.mean(rs[ok]))
    else:
        proxy = np.sqrt(np.clip(yp, 1e-6, None))
        z = resid / proxy
        point = float(np.std(z, ddof=1))
        raw_mean = float(np.mean(proxy))

    rng = np.random.default_rng(seed)
    boots: list[float] = []
    n = len(z)
    for _ in range(int(n_bootstrap)):
        idx = rng.integers(0, n, size=n)
        boots.append(float(np.std(z[idx], ddof=1)))
    lo, hi = float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))
    return DispersionCalibration(
        sigma_scale=round(max(point, 0.1), 4),
        n_residuals=int(len(yt)),
        residual_std=round(resid_std, 4),
        raw_pred_std_mean=round(raw_mean, 4),
        bootstrap_ci_low=round(lo, 4),
        bootstrap_ci_high=round(hi, 4),
        notes="RESEARCH_ONLY residual dispersion; not a stake rule",
    )


def apply_sigma_scale(std: float | None, calib: DispersionCalibration) -> float | None:
    if std is None or not np.isfinite(std):
        return None
    return float(std) * float(calib.sigma_scale)


def pra_from_feature_row(
    row: pd.Series,
    *,
    mean_suffix: str = "L2",
    var_fudge: float = DEFAULT_COMBO_VAR_FUDGE,
    sigma_scale: float = 1.0,
) -> ComboProjection:
    """Build PRA combo from a feature row using ``{STAT}_{suffix}`` means."""
    def _m(stat: str) -> float | None:
        # Prefer explicit PRA rollup when present and suffix matches
        v = row.get(f"{stat}_{mean_suffix}")
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return None
        return float(v)

    # Component SDs: sqrt(mean) as Poisson-ish raw, then scale
    pts_m, reb_m, ast_m = _m("PTS"), _m("REB"), _m("AST")
    pts_sd = (np.sqrt(max(pts_m, 0.0)) * sigma_scale) if pts_m is not None else None
    reb_sd = (np.sqrt(max(reb_m, 0.0)) * sigma_scale) if reb_m is not None else None
    ast_sd = (np.sqrt(max(ast_m, 0.0)) * sigma_scale) if ast_m is not None else None
    return pra_joint_projection(
        pts_m,
        reb_m,
        ast_m,
        pts_sd=pts_sd,
        reb_sd=reb_sd,
        ast_sd=ast_sd,
        var_fudge=var_fudge,
    )
