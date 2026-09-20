"""Distribution-based P(Over) using QuantEngine families + explicit push handling."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.models.line_probs import discrete_over_under_push
from src.models.prediction_schema import ModelMetadata, ModelPrediction

logger = logging.getLogger(__name__)

_NB_STATS = {"FG3M", "3PM", "STL", "BLK"}


class DistributionPropModel:
    """Mean from ``{stat}_L2``; probabilities from Poisson / NegBin with push."""

    model_name = "distribution"

    def __init__(
        self,
        *,
        target_market: str = "PTS",
        model_version: str = "dist_v1",
        feature_schema_version: str = "fs_v1_shift1_l2",
    ) -> None:
        self.target_market = target_market
        self.model_version = model_version
        self.feature_schema_version = feature_schema_version
        self._fitted = True

    def fit(self, train_data: pd.DataFrame, validation_data: pd.DataFrame | None = None) -> "DistributionPropModel":
        logger.info(
            "distribution model: no parameter fit (uses {stat}_L2 + discrete family); train_rows=%d",
            len(train_data),
        )
        return self

    def _mean_col(self) -> str:
        return f"{self.target_market}_L2"

    def predict_mean(self, features: pd.DataFrame) -> pd.Series:
        col = self._mean_col()
        if col not in features.columns:
            raise ValueError(f"DATA_NOT_AVAILABLE: missing {col}")
        return pd.to_numeric(features[col], errors="coerce")

    def predict_distribution(self, features: pd.DataFrame) -> pd.DataFrame:
        means = self.predict_mean(features)
        family = "negbin" if self.target_market in _NB_STATS else "poisson"
        return pd.DataFrame(
            {
                "prediction_mean": means,
                "prediction_std_or_dispersion": means.apply(
                    lambda m: (float(m) * 1.35) ** 0.5
                    if pd.notna(m) and family == "negbin"
                    else (float(m) ** 0.5 if pd.notna(m) else None)
                ),
                "method": family,
            },
            index=features.index,
        )

    def predict_probability_over(
        self,
        features: pd.DataFrame,
        line: float | pd.Series,
    ) -> pd.Series:
        means = self.predict_mean(features)
        lines = line if isinstance(line, pd.Series) else pd.Series([line] * len(features), index=features.index)
        family = "negbin" if self.target_market in _NB_STATS else "poisson"
        vals = []
        for m, ln in zip(means, lines):
            res = discrete_over_under_push(
                float(m) if pd.notna(m) else float("nan"),
                float(ln) if pd.notna(ln) else float("nan"),
                family=family,
            )
            vals.append(res.get("probability_over"))
        return pd.Series(vals, index=features.index)

    def predict_rows(
        self,
        features: pd.DataFrame,
        *,
        line_col: str = "RESEARCH_LINE",
    ) -> list[ModelPrediction]:
        means = self.predict_mean(features)
        family = "negbin" if self.target_market in _NB_STATS else "poisson"
        out: list[ModelPrediction] = []
        for i, (_, row) in enumerate(features.iterrows()):
            line = row.get(line_col)
            m = means.iloc[i]
            res = discrete_over_under_push(
                float(m) if pd.notna(m) else float("nan"),
                float(line) if pd.notna(line) else float("nan"),
                family=family,
            )
            warnings: list[str] = []
            if res.get("status") != "OK":
                warnings.append(str(res.get("reason") or "distribution unavailable"))
            out.append(
                ModelPrediction(
                    model_name=self.model_name,
                    model_version=self.model_version,
                    target_market=self.target_market,
                    event_id=str(row.get("GAME_ID", "")),
                    player_id=str(row.get("PLAYER_ID", "")),
                    player_name=row.get("PLAYER_NAME"),
                    prediction_mean=float(m) if pd.notna(m) else None,
                    prop_line=float(line) if pd.notna(line) else None,
                    probability_over=res.get("probability_over"),
                    probability_under=res.get("probability_under"),
                    probability_push=res.get("probability_push"),
                    feature_schema_version=self.feature_schema_version,
                    warnings=warnings,
                    extras={"distribution": res.get("distribution")},
                )
            )
        return out

    def save(self, path: Any) -> None:
        logger.info("distribution model has no binary artifact (%s)", path)

    def load(self, path: Any) -> "DistributionPropModel":
        return self

    def get_model_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_name=self.model_name,
            model_version=self.model_version,
            target_market=self.target_market,
            feature_cols=[self._mean_col()],
            feature_schema_version=self.feature_schema_version,
            notes=[
                "Uses leakage-safe {stat}_L2 mean",
                "Poisson for PTS/REB/AST; NegBin for FG3M/STL/BLK",
                "Push mass modeled on whole-number lines",
            ],
        )
