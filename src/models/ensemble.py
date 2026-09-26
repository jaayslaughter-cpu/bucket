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

    @property
    def oof(self):
        """
        Blend the components' out-of-fold probabilities, weight for weight.

        Rebuilding the whole ensemble on a separate split to calibrate it
        re-fits every component — by far the most expensive calibration in
        the comparison. The components already computed their own
        out-of-fold probabilities on chronological folds; blending those with
        the same weights answers the same question without refitting
        anything.

        Only rows every contributing component predicted are used. A row one
        component missed would otherwise be blended from a different mix of
        models than the weights describe.
        """
        import numpy as np
        import pandas as pd

        from src.models.oof import OutOfFoldPredictions

        usable = {
            name: model.oof
            for name, model in self.components.items()
            if getattr(model, "oof", None) is not None
            and getattr(model.oof, "usable", False)
            and float(self.weights.get(name, 0.0)) > 0.0
        }
        if len(usable) < 2:
            return None

        # Every frame must label the SAME rows. line_aware's index is a fresh
        # RangeIndex over (source row x candidate line) pairs and collides with
        # the source-row index the other components carry -- both start at 0
        # over different universes, so intersecting by label paired augmented
        # row i with source row i and produced a blend whose probabilities and
        # labels came from different rows. Declining the fast path is the only
        # safe answer: dropping the odd component out would blend a different
        # mix of models than the weights describe, which is the very thing the
        # shared-rows rule above exists to prevent.
        samples = {o.sample for o in usable.values()}
        if len(samples) > 1:
            logger.warning(
                "ensemble %s: components disagree about which rows their "
                "out-of-fold frames describe (%s), so the blended fast path is "
                "declined rather than aligning indexes that mean different "
                "things. Calibration falls back to the slower refit.",
                self.target_market,
                sorted(x or "source_rows" for x in samples),
            )
            return None

        frames = {n: o.frame for n, o in usable.items()}
        shared = None
        for frame in frames.values():
            idx = frame.index[frame["prob_over"].notna() & frame["y_over"].notna()]
            shared = idx if shared is None else shared.intersection(idx)
        if shared is None or len(shared) == 0:
            return None

        total = sum(float(self.weights.get(n, 0.0)) for n in usable)
        if total <= 0:
            return None
        blended = np.zeros(len(shared), dtype=float)
        for name, frame in frames.items():
            w = float(self.weights.get(name, 0.0)) / total
            blended += w * frame.loc[shared, "prob_over"].to_numpy(dtype=float)

        any_frame = next(iter(frames.values()))
        return OutOfFoldPredictions(
            pd.DataFrame(
                {"prob_over": blended,
                 "y_over": any_frame.loc[shared, "y_over"].to_numpy(dtype=float)},
                index=shared,
            ),
            n_folds=min(o.n_folds for o in usable.values()),
            sample=next(iter(samples)),
        )

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
