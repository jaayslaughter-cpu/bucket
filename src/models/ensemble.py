"""Transparent weighted ensemble of component over-probabilities."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.models.prediction_schema import ModelMetadata, ModelPrediction

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS: dict[str, float] = {
    "catboost": 0.50,
    "xgboost": 0.30,
    "distribution": 0.20,
}


def renormalize_weights(
    weights: dict[str, float],
    available: set[str],
) -> tuple[dict[str, float], list[str]]:
    """Keep only available components; renormalize to sum 1. Warn if any dropped."""
    warnings: list[str] = []
    kept = {k: float(v) for k, v in weights.items() if k in available and float(v) > 0}
    dropped = [k for k in weights if k not in kept]
    if dropped:
        warnings.append(f"Ensemble dropped unavailable/zero-weight models: {dropped}")
    total = sum(kept.values())
    if total <= 0:
        raise ValueError("DATA_NOT_AVAILABLE: no positive ensemble weights remain")
    return {k: v / total for k, v in kept.items()}, warnings


class EnsemblePropModel:
    """Blend component P(Over). Does not claim weights are optimal."""

    model_name = "ensemble"

    def __init__(
        self,
        components: dict[str, Any],
        *,
        weights: dict[str, float] | None = None,
        target_market: str = "PTS",
        model_version: str = "ens_v1",
        feature_schema_version: str = "fs_v1_shift1_l2",
    ) -> None:
        self.components = components
        self.weights = dict(weights or DEFAULT_WEIGHTS)
        self.target_market = target_market
        self.model_version = model_version
        self.feature_schema_version = feature_schema_version

    def fit(self, train_data: pd.DataFrame, validation_data: pd.DataFrame | None = None) -> "EnsemblePropModel":
        for name, model in self.components.items():
            try:
                model.fit(train_data, validation_data)
            except Exception as exc:  # noqa: BLE001
                logger.warning("ensemble component %s fit failed: %s", name, exc)
        return self

    def predict_mean(self, features: pd.DataFrame) -> pd.Series:
        """Weighted blend of component means, using the same weights as the
        probabilities. Taking one component's mean would report a blended
        probability next to an unblended projection."""
        means: dict[str, pd.Series] = {}
        for name, model in self.components.items():
            try:
                series = pd.to_numeric(model.predict_mean(features), errors="coerce")
            except Exception as exc:  # noqa: BLE001
                logger.warning("ensemble mean skip %s: %s", name, exc)
                continue
            if series.notna().any():
                means[name] = series

        if not means:
            return pd.Series([None] * len(features), index=features.index, dtype="object")

        weights, _ = renormalize_weights(self.weights, set(means))
        blended = pd.Series(0.0, index=features.index, dtype=float)
        weight_used = pd.Series(0.0, index=features.index, dtype=float)
        for name, w in weights.items():
            valid = means[name].notna()
            blended = blended.add((means[name] * w).where(valid, 0.0), fill_value=0.0)
            weight_used = weight_used.add(pd.Series(w, index=features.index).where(valid, 0.0))
        # Renormalize per row so rows where a component was null are not diluted.
        return (blended / weight_used.replace(0.0, pd.NA)).astype(float)

    def predict_distribution(self, features: pd.DataFrame) -> pd.DataFrame:
        if "distribution" in self.components:
            return self.components["distribution"].predict_distribution(features)
        return pd.DataFrame({"prediction_mean": self.predict_mean(features)}, index=features.index)

    def predict_probability_over(
        self,
        features: pd.DataFrame,
        line: float | pd.Series,
    ) -> pd.Series:
        """P(over), consistent with ``predict_rows``.

        Delegates to ``predict_rows`` rather than blending the components'
        raw probabilities. Blending directly skips the push handling that
        ``predict_rows`` applies, so the two APIs disagreed on exactly the
        whole-number lines where push mass is non-zero — and which one a
        caller happened to use decided the answer.
        """
        work = features
        line_col = "_ENSEMBLE_LINE"
        if isinstance(line, pd.Series):
            work = features.assign(**{line_col: line})
        else:
            work = features.assign(**{line_col: float(line)})

        rows = self.predict_rows(work, line_col=line_col)
        return pd.Series(
            [r.probability_over for r in rows], index=features.index, dtype="float64"
        )

    def predict_rows(
        self,
        features: pd.DataFrame,
        *,
        line_col: str = "RESEARCH_LINE",
    ) -> list[ModelPrediction]:
        component_rows: dict[str, list[ModelPrediction]] = {}
        for name, model in self.components.items():
            try:
                component_rows[name] = model.predict_rows(features, line_col=line_col)
            except Exception as exc:  # noqa: BLE001
                logger.warning("ensemble rows skip %s: %s", name, exc)

        weights, base_warnings = renormalize_weights(self.weights, set(component_rows))
        blended_means = self.predict_mean(features)
        out: list[ModelPrediction] = []
        n = len(features)
        for i in range(n):
            row = features.iloc[i]
            p_over = 0.0
            p_under = 0.0
            p_push = 0.0
            push_known = True
            contributing = 0
            extras: dict[str, Any] = {"component_weights": weights, "component_probabilities": {}}
            warnings = list(base_warnings)
            raw_mean = blended_means.iloc[i]
            mean_val = float(raw_mean) if pd.notna(raw_mean) else None
            for name, w in weights.items():
                pred = component_rows[name][i]
                extras["component_probabilities"][name] = {
                    "probability_over": pred.probability_over,
                    "probability_under": pred.probability_under,
                    "probability_push": pred.probability_push,
                }
                if pred.probability_over is None:
                    warnings.append(f"{name} missing probability_over")
                    continue
                contributing += 1
                p_over += w * float(pred.probability_over)
                if pred.probability_under is not None:
                    p_under += w * float(pred.probability_under)
                if pred.probability_push is None:
                    push_known = False
                else:
                    p_push += w * float(pred.probability_push)
                warnings.extend(pred.warnings)
            if not push_known:
                p_push_out = None
                # renormalize over/under only
                s = p_over + p_under
                if s > 0:
                    p_over, p_under = p_over / s, p_under / s
            else:
                p_push_out = p_push
                s = p_over + p_under + p_push
                if s > 0:
                    p_over, p_under, p_push_out = p_over / s, p_under / s, p_push / s

            # No component produced a probability for this row. Emitting the
            # accumulator's 0.0 would publish "certainly under" — an invented
            # certainty that downstream scoring cannot tell from a real one.
            if not contributing:
                warnings.append("No component supplied a probability for this row")
                probabilities: dict[str, float | None] = {
                    "probability_over": None,
                    "probability_under": None,
                    "probability_push": None,
                }
            else:
                probabilities = {
                    "probability_over": round(p_over, 6),
                    "probability_under": round(p_under, 6),
                    "probability_push": None if p_push_out is None else round(float(p_push_out), 6),
                }

            out.append(
                ModelPrediction(
                    model_name=self.model_name,
                    model_version=self.model_version,
                    target_market=self.target_market,
                    event_id=str(row.get("GAME_ID", "")),
                    player_id=str(row.get("PLAYER_ID", "")),
                    player_name=row.get("PLAYER_NAME"),
                    prediction_mean=mean_val,
                    prop_line=float(row[line_col]) if pd.notna(row.get(line_col)) else None,
                    **probabilities,
                    feature_schema_version=self.feature_schema_version,
                    warnings=warnings,
                    extras=extras,
                )
            )
        return out

    def save(self, path: Any) -> None:
        logger.info("ensemble save is a no-op (components saved separately): %s", path)

    def load(self, path: Any) -> "EnsemblePropModel":
        return self

    def get_model_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_name=self.model_name,
            model_version=self.model_version,
            target_market=self.target_market,
            hyperparameters={"weights": self.weights},
            feature_schema_version=self.feature_schema_version,
            notes=["Default weights are not claimed optimal", "Renormalizes when components missing"],
        )
