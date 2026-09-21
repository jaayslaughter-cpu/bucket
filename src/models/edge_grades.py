"""Display-only research edge letter grades (Wave 2).

Never implies stake size, bankroll, or guaranteed outcomes.
When fair market probability is absent, grades research separation
(projection vs line in σ units / P(over) vs 0.5) and labels provenance.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

Grade = Literal["A+", "A", "B", "C", "D", "F", "N/A"]

DISPLAY_DISCLAIMER = (
    "RESEARCH_DISPLAY_ONLY — letter grade is not a bet recommendation, "
    "stake size, or profitability claim."
)


def _grade_from_abs_edge(edge: float) -> Grade:
    """Map absolute probability edge to letter (UI heuristic only)."""
    e = abs(float(edge))
    if e >= 0.12:
        return "A+"
    if e >= 0.08:
        return "A"
    if e >= 0.05:
        return "B"
    if e >= 0.03:
        return "C"
    if e >= 0.015:
        return "D"
    return "F"


def _grade_from_abs_z(z: float) -> Grade:
    az = abs(float(z))
    if az >= 1.5:
        return "A+"
    if az >= 1.0:
        return "A"
    if az >= 0.7:
        return "B"
    if az >= 0.4:
        return "C"
    if az >= 0.2:
        return "D"
    return "F"


def research_edge_letter_grade(
    *,
    prediction_mean: float | None = None,
    prop_line: float | None = None,
    prediction_std: float | None = None,
    probability_over: float | None = None,
    fair_market_probability: float | None = None,
    side: str = "over",
) -> dict[str, Any]:
    """
    Return letter grade + provenance.

    Preference order:
      1. model_prob − fair_market_prob (when fair market available)
      2. (mean − line) / std research z
      3. |P(over) − 0.5| research confidence proxy
    """
    base: dict[str, Any] = {
        "edge_letter_grade": "N/A",
        "edge_probability": None,
        "research_z": None,
        "grade_basis": "unavailable",
        "side_lean": None,
        "disclaimer": DISPLAY_DISCLAIMER,
    }

    side_l = str(side).lower()
    prefer_under = side_l in {"under", "u", "less"}

    if fair_market_probability is not None and np.isfinite(fair_market_probability):
        if probability_over is None or not np.isfinite(probability_over):
            return base
        # Edge for the *requested* side (not absolute magnitude of either side).
        model_p = (
            1.0 - float(probability_over) if prefer_under else float(probability_over)
        )
        edge = model_p - float(fair_market_probability)
        grade = _grade_from_abs_edge(edge) if edge > 0 else "F"
        return {
            **base,
            "edge_letter_grade": grade,
            "edge_probability": round(edge, 4),
            "grade_basis": "model_minus_fair_market",
            "side_lean": ("under" if prefer_under else "over") if edge > 0 else (
                "over" if prefer_under else "under"
            ),
        }

    if (
        prediction_mean is not None
        and prop_line is not None
        and prediction_std is not None
        and np.isfinite(prediction_mean)
        and np.isfinite(prop_line)
        and np.isfinite(prediction_std)
        and prediction_std > 1e-6
    ):
        z = (float(prediction_mean) - float(prop_line)) / float(prediction_std)
        signed = -z if prefer_under else z
        grade = _grade_from_abs_z(signed) if signed > 0 else "F"
        return {
            **base,
            "edge_letter_grade": grade,
            "research_z": round(z, 4),
            "grade_basis": "research_z_vs_line",
            "side_lean": ("under" if prefer_under else "over") if signed > 0 else (
                "over" if prefer_under else "under"
            ),
        }

    if probability_over is not None and np.isfinite(probability_over):
        p = float(probability_over)
        edge = (1.0 - p) - 0.5 if prefer_under else p - 0.5
        grade = _grade_from_abs_edge(edge) if edge > 0 else "F"
        return {
            **base,
            "edge_letter_grade": grade,
            "edge_probability": round(edge, 4),
            "grade_basis": "research_p_over_vs_half",
            "side_lean": ("under" if prefer_under else "over") if edge > 0 else (
                "over" if prefer_under else "under"
            ),
        }

    return base
