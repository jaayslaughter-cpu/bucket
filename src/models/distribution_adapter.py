"""Distribution baseline: rolling-average mean, data-fitted count spread.

The deliberately simple reference model. It learns no feature weights —
its mean is the player's own recent form (``{stat}_L2``) — so it shows
what the boosted models have to beat to justify themselves.

What it DOES fit is the spread. Dispersion comes from training residuals
and the family (Poisson vs Negative Binomial) is chosen by held-out score,
rather than being asserted per stat.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.models.prediction_schema import ModelMetadata, ModelPrediction
from src.models.residuals import (
    CountDispersion,
    fit_count_dispersion,
    over_under_push_from_dispersion,
)

logger = logging.getLogger(__name__)


class DistributionPropModel:
    """Mean from ``{stat}_L2``; spread fitted from training residuals."""

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
        self.dispersion: CountDispersion | None = None
        self._fitted = False

    def fit(self, train_data: pd.DataFrame, validation_data: pd.DataFrame | None = None) -> "DistributionPropModel":
        """Estimate dispersion from training rows only. No feature weights are learned."""
        stat = self.target_market
        mean_col = self._mean_col()

        if stat not in train_data.columns or mean_col not in train_data.columns:
            logger.warning(
                "distribution: need both %s and %s to fit dispersion — falling back to Poisson",
                stat, mean_col,
            )
            self._fitted = True
            return self

        work = train_data
        if "GAME_DATE" in work.columns:
            work = work.sort_values("GAME_DATE")
        actual = pd.to_numeric(work[stat], errors="coerce")
        projected = pd.to_numeric(work[mean_col], errors="coerce")
        self.dispersion = fit_count_dispersion(
            actual.to_numpy(), projected.to_numpy(), market=stat
        )
        self._fitted = True
        logger.info(
            "distribution fitted %s: family=%s phi=%.3f on %d rows",
            stat, self.dispersion.family, self.dispersion.phi, self.dispersion.n_train_rows,
        )
        return self

    def _effective_dispersion(self) -> CountDispersion:
        """Poisson is the honest default when nothing was fitted."""
        if self.dispersion is not None:
            return self.dispersion
        return CountDispersion(
            family="poisson",
            phi=1.0,
            n_train_rows=0,
            selection_scores={},
            fallback_reason="fit() was never called or lacked the columns to fit",
        )

    def _mean_col(self) -> str:
        return f"{self.target_market}_L2"

    def predict_mean(self, features: pd.DataFrame) -> pd.Series:
        col = self._mean_col()
        if col not in features.columns:
            raise ValueError(f"DATA_NOT_AVAILABLE: missing {col}")
        return pd.to_numeric(features[col], errors="coerce")

    def predict_distribution(self, features: pd.DataFrame) -> pd.DataFrame:
        means = self.predict_mean(features)
        dispersion = self._effective_dispersion()
        return pd.DataFrame(
            {
                "prediction_mean": means,
                "prediction_std_or_dispersion": means.apply(
                    lambda m: float(dispersion.variance_for(m)) ** 0.5 if pd.notna(m) else None
                ),
                "method": dispersion.family,
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
        dispersion = self._effective_dispersion()
        vals = []
        for m, ln in zip(means, lines):
            res = over_under_push_from_dispersion(
                float(m) if pd.notna(m) else float("nan"),
                float(ln) if pd.notna(ln) else float("nan"),
                dispersion,
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
        dispersion = self._effective_dispersion()
        out: list[ModelPrediction] = []
        for i, (_, row) in enumerate(features.iterrows()):
            line = row.get(line_col)
            m = means.iloc[i]
            res = over_under_push_from_dispersion(
                float(m) if pd.notna(m) else float("nan"),
                float(line) if pd.notna(line) else float("nan"),
                dispersion,
            )
            warnings: list[str] = []
            if res.get("status") != "OK":
                warnings.append(str(res.get("reason") or "distribution unavailable"))
            if dispersion.fallback_reason:
                warnings.append(f"Dispersion not fitted: {dispersion.fallback_reason}")
            out.append(
                ModelPrediction(
                    model_name=self.model_name,
                    model_version=self.model_version,
                    target_market=self.target_market,
                    event_id=str(row.get("GAME_ID", "")),
                    player_id=str(row.get("PLAYER_ID", "")),
                    player_name=row.get("PLAYER_NAME"),
                    prediction_mean=float(m) if pd.notna(m) else None,
                    prediction_std_or_dispersion=res.get("standard_deviation"),
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
                "Uses leakage-safe {stat}_L2 mean; learns no feature weights",
                (
                    f"Dispersion {self.dispersion.family} phi={self.dispersion.phi:.3f} "
                    f"fitted on {self.dispersion.n_train_rows} training rows"
                    if self.dispersion
                    else "Dispersion not fitted — Poisson fallback"
                ),
                "Push mass modelled on whole-number lines",
            ],
        )
