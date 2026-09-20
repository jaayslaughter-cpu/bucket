"""Shared prediction schema for XGBoost / CatBoost / distribution / ensemble.

RESEARCH_ONLY — does not invent lines, odds, or outcomes.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from src.utils.timezones import format_pacific_iso, now_utc, to_pacific


def utcnow() -> datetime:
    """Storage clock (UTC). Display via ``prediction_timestamp_pt`` / format helpers."""
    return now_utc()


class ModelPrediction(BaseModel):
    """One model prediction for one player-game-market at a given line.

    Timestamps are stored UTC; use ``*_pt`` properties for America/Los_Angeles display.
    """

    model_name: str
    model_version: str
    target_market: str
    event_id: str
    player_id: str
    player_name: str | None = None
    prediction_timestamp_utc: datetime = Field(default_factory=utcnow)
    prediction_mean: float | None = None
    prediction_std_or_dispersion: float | None = None
    prop_line: float | None = None
    probability_over: float | None = None
    probability_under: float | None = None
    probability_push: float | None = None
    probability_over_calibrated: float | None = None
    probability_under_calibrated: float | None = None
    data_cutoff_timestamp_utc: datetime | None = None
    feature_schema_version: str = "fs_v1_shift1_l2"
    warnings: list[str] = Field(default_factory=list)
    extras: dict[str, Any] = Field(default_factory=dict)

    @property
    def prediction_timestamp_pt(self) -> str | None:
        return format_pacific_iso(self.prediction_timestamp_utc)

    @property
    def data_cutoff_timestamp_pt(self) -> str | None:
        if self.data_cutoff_timestamp_utc is None:
            return None
        return format_pacific_iso(to_pacific(self.data_cutoff_timestamp_utc))

    def is_valid_probability(self) -> bool:
        if self.probability_over is None or self.probability_under is None:
            return False
        push = self.probability_push or 0.0
        total = float(self.probability_over) + float(self.probability_under) + float(push)
        return abs(total - 1.0) < 1e-2


class ModelMetadata(BaseModel):
    model_name: str
    model_version: str
    target_market: str
    # Every adapter passes this; without the field pydantic silently drops it
    # and the saved artifact loses the feature contract needed to reproduce
    # or audit its predictions.
    feature_schema_version: str = "fs_v1_shift1_l2"
    feature_cols: list[str] = Field(default_factory=list)
    categorical_cols: list[str] = Field(default_factory=list)
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    train_row_count: int | None = None
    validation_row_count: int | None = None
    train_start_date: str | None = None
    train_end_date: str | None = None
    validation_start_date: str | None = None
    validation_end_date: str | None = None
    data_cutoff_timestamp_utc: datetime | None = None
    random_seed: int | None = None
    package_versions: dict[str, str] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
