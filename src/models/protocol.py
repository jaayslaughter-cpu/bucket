"""Common model Protocol — adapters wrap XGBoost/CatBoost/distribution."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from src.models.prediction_schema import ModelMetadata, ModelPrediction


@runtime_checkable
class PropModel(Protocol):
    """Shared interface for research prop models."""

    def fit(self, train_data: pd.DataFrame, validation_data: pd.DataFrame | None = None) -> Any:
        ...

    def predict_mean(self, features: pd.DataFrame) -> pd.Series:
        ...

    def predict_distribution(self, features: pd.DataFrame) -> pd.DataFrame:
        ...

    def predict_probability_over(
        self,
        features: pd.DataFrame,
        line: float | pd.Series,
    ) -> pd.Series:
        ...

    def predict_rows(
        self,
        features: pd.DataFrame,
        *,
        line_col: str = "RESEARCH_LINE",
    ) -> list[ModelPrediction]:
        ...

    def save(self, path: Path | str) -> None:
        ...

    def load(self, path: Path | str) -> Any:
        ...

    def get_model_metadata(self) -> ModelMetadata:
        ...
