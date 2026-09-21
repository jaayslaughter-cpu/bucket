"""Projection-card payload for research UI / exports (Wave 4b).

Display-only: mean, line, σ, O/U meter, grades. Not a stake recommendation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from src.models.edge_grades import DISPLAY_DISCLAIMER, research_edge_letter_grade

CARD_DISCLAIMER = (
    "RESEARCH_DISPLAY_ONLY projection card — not betting advice, "
    "stake size, or a guaranteed outcome."
)


class ProjectionCard(BaseModel):
    player_id: str | None = None
    player_name: str | None = None
    target_market: str
    prop_line: float | None = None
    prediction_mean: float | None = None
    prediction_std: float | None = None
    probability_over: float | None = None
    probability_under: float | None = None
    probability_push: float | None = None
    # 0–100 meter: where mean sits vs line (±2σ window), 50 = at line
    over_under_meter: float | None = None
    research_z: float | None = None
    edge_letter_grade: str | None = None
    edge_grade_basis: str | None = None
    confidence_tier: str | None = None
    hot_hand_status: str | None = None
    warnings: list[str] = Field(default_factory=list)
    disclaimer: str = CARD_DISCLAIMER


def _meter(mean: float, line: float, std: float | None) -> float:
    """Map (mean - line) into 0–100 with ~2σ ≈ full scale."""
    scale = float(std) if std is not None and np.isfinite(std) and std > 1e-6 else 5.0
    z = (float(mean) - float(line)) / (2.0 * scale)
    return float(np.clip(50.0 + 50.0 * z, 0.0, 100.0))


def build_projection_card(
    *,
    target_market: str,
    prop_line: float | None = None,
    prediction_mean: float | None = None,
    prediction_std: float | None = None,
    probability_over: float | None = None,
    probability_under: float | None = None,
    probability_push: float | None = None,
    player_id: str | None = None,
    player_name: str | None = None,
    confidence_tier: str | None = None,
    hot_hand_status: str | None = None,
    warnings: list[str] | None = None,
) -> ProjectionCard:
    grade = research_edge_letter_grade(
        prediction_mean=prediction_mean,
        prop_line=prop_line,
        prediction_std=prediction_std,
        probability_over=probability_over,
    )
    meter = None
    if (
        prediction_mean is not None
        and prop_line is not None
        and np.isfinite(prediction_mean)
        and np.isfinite(prop_line)
    ):
        meter = round(
            _meter(float(prediction_mean), float(prop_line), prediction_std),
            1,
        )
    return ProjectionCard(
        player_id=player_id,
        player_name=player_name,
        target_market=target_market,
        prop_line=prop_line,
        prediction_mean=prediction_mean,
        prediction_std=prediction_std,
        probability_over=probability_over,
        probability_under=probability_under,
        probability_push=probability_push,
        over_under_meter=meter,
        research_z=grade.get("research_z"),
        edge_letter_grade=grade.get("edge_letter_grade"),
        edge_grade_basis=grade.get("grade_basis"),
        confidence_tier=confidence_tier,
        hot_hand_status=hot_hand_status,
        warnings=list(warnings or []),
        disclaimer=f"{CARD_DISCLAIMER} {DISPLAY_DISCLAIMER}",
    )


def projection_card_from_detail_row(row: dict[str, Any]) -> dict[str, Any]:
    warns = row.get("warnings")
    wlist = str(warns).split("|") if warns else []
    card = build_projection_card(
        target_market=str(row.get("target_market") or ""),
        prop_line=row.get("prop_line"),
        prediction_mean=row.get("prediction_mean"),
        prediction_std=row.get("prediction_std_or_dispersion"),
        probability_over=row.get("probability_over_raw"),
        probability_under=row.get("probability_under_raw"),
        probability_push=row.get("probability_push_raw"),
        player_id=row.get("player_id"),
        player_name=row.get("player_name"),
        confidence_tier=row.get("confidence_tier"),
        hot_hand_status=row.get("hot_hand_status"),
        warnings=[w for w in wlist if w],
    )
    return card.model_dump()
