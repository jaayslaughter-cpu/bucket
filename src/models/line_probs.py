"""Leakage-safe over / under / push probabilities at a prop line.

Whole-number line N:
  Over  wins when stat > N
  Under wins when stat < N
  Push when stat == N

Half-point line (e.g. 25.5): push probability is 0.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import nbinom, poisson


def is_whole_number_line(line: float) -> bool:
    return float(line) == float(int(line))


def discrete_over_under_push(
    projection: float,
    line: float,
    *,
    family: str = "poisson",
    nb_var_scale: float = 1.35,
) -> dict[str, Any]:
    """
    Return P(over), P(under), P(push) for a non-negative count mean.

    Does not invent a projection — returns DATA_NOT_AVAILABLE when invalid.
    """
    if projection is None or not np.isfinite(projection) or projection < 0:
        return {
            "status": "DATA_NOT_AVAILABLE",
            "reason": "Non-negative finite projection required",
            "probability_over": None,
            "probability_under": None,
            "probability_push": None,
            "distribution": None,
        }
    if line is None or not np.isfinite(line):
        return {
            "status": "DATA_NOT_AVAILABLE",
            "reason": "Finite prop line required",
            "probability_over": None,
            "probability_under": None,
            "probability_push": None,
            "distribution": None,
        }

    mu = float(projection)
    ln = float(line)
    family_l = family.lower()

    def pmf_cdf(k: int) -> tuple[float, float]:
        if family_l in {"negbin", "negativebinomial", "nb"}:
            variance = max(mu * nb_var_scale, mu + 0.01)
            p_param = mu / variance
            r_param = (mu**2) / max(variance - mu, 1e-6)
            return float(nbinom.pmf(k, r_param, p_param)), float(nbinom.cdf(k, r_param, p_param))
        return float(poisson.pmf(k, mu)), float(poisson.cdf(k, mu))

    if is_whole_number_line(ln):
        n = int(ln)
        p_eq, cdf_n = pmf_cdf(n)
        p_under = max(0.0, cdf_n - p_eq)  # P(X < n)
        p_push = p_eq
        p_over = max(0.0, 1.0 - cdf_n)  # P(X > n)
        dist_name = "NegativeBinomial" if family_l.startswith("neg") or family_l == "nb" else "Poisson"
    else:
        k = int(np.floor(ln))
        _, cdf_k = pmf_cdf(k)
        p_under = cdf_k
        p_over = 1.0 - cdf_k
        p_push = 0.0
        dist_name = "NegativeBinomial" if family_l.startswith("neg") or family_l == "nb" else "Poisson"

    total = p_over + p_under + p_push
    if total <= 0:
        return {
            "status": "DATA_NOT_AVAILABLE",
            "reason": "Degenerate distribution",
            "probability_over": None,
            "probability_under": None,
            "probability_push": None,
            "distribution": dist_name,
        }
    p_over, p_under, p_push = p_over / total, p_under / total, p_push / total
    return {
        "status": "OK",
        "distribution": dist_name,
        "probability_over": round(p_over, 6),
        "probability_under": round(p_under, 6),
        "probability_push": round(p_push, 6),
        "projected_mean": round(mu, 4),
        "line": ln,
    }


def classifier_over_under(
    p_over: float,
    line: float,
) -> dict[str, Any]:
    """Binary classifier P(over); push not modeled (warned on whole lines)."""
    if p_over is None or not np.isfinite(p_over):
        return {
            "status": "DATA_NOT_AVAILABLE",
            "probability_over": None,
            "probability_under": None,
            "probability_push": None,
            "warnings": ["Invalid classifier probability"],
        }
    p = float(np.clip(p_over, 1e-6, 1.0 - 1e-6))
    warnings: list[str] = []
    if is_whole_number_line(line):
        warnings.append(
            "Classifier does not model push mass on whole-number lines; "
            "P(under) approximated as 1 - P(over)."
        )
    return {
        "status": "OK",
        "probability_over": round(p, 6),
        "probability_under": round(1.0 - p, 6),
        "probability_push": None,
        "warnings": warnings,
    }
