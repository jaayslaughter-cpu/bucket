"""Adapter putting XGBoostPropPipeline behind the common model interface.

The classifier's own training behaviour is untouched. A separate XGBoost
regressor is fitted alongside it for the stat mean, so XGBoost and
CatBoost are scored on the same footing: without it, MAE and RMSE would
compare a real CatBoost projection against a passthrough feature column.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.labels import mask_probabilities_at_unsupported_lines
from src.models.prediction_schema import ModelMetadata, ModelPrediction
from src.models.residuals import (
    CountDispersion,
    fit_dispersion_out_of_fold,
    over_under_push_from_dispersion,
)
from src.models.xgboost_pipeline import XGBoostPropPipeline

logger = logging.getLogger(__name__)


class XGBoostAdapter:
    """
    Wraps ``XGBoostPropPipeline`` behind the common interface.

    Does not alter XGBoost training defaults or TimeSeriesSplit behavior.
    """

    model_name = "xgboost"

    def __init__(
        self,
        feature_cols: list[str],
        *,
        target_market: str = "PTS",
        model_version: str = "xgb_v1",
        feature_schema_version: str = "fs_v1_shift1_l2",
        n_splits: int = 5,
        random_state: int = 42,
        model_params: dict[str, Any] | None = None,
        tuning: dict[str, Any] | None = None,
    ) -> None:
        self.target_market = target_market
        self.model_version = model_version
        self.feature_schema_version = feature_schema_version
        self.feature_cols = list(feature_cols)
        self.oof = None
        self._pipe = XGBoostPropPipeline(
            self.feature_cols,
            n_splits=n_splits,
            random_state=random_state,
            model_params=model_params,
            tuning=tuning,
        )
        self._meta_extra: dict[str, Any] = {}
        self._fitted = False
        self.mean_model: Any | None = None
        self.dispersion: CountDispersion | None = None
        # Set by LineAwarePropModel when the line is among the fitted
        # features. The abstention below exists BECAUSE the line is not
        # an input; once it is, masking would throw away the answer.
        self.line_aware: bool = False

    def fit(self, train_data: pd.DataFrame, validation_data: pd.DataFrame | None = None) -> "XGBoostAdapter":
        # Existing pipeline fits on train only (validation unused — preserved).
        if validation_data is not None and not validation_data.empty:
            logger.info(
                "xgboost_adapter: validation_data provided (%d rows) but "
                "XGBoostPropPipeline.fit uses train only (unchanged behavior).",
                len(validation_data),
            )
        self._pipe.fit(train_data, target_col="over_hit")
        self._fitted = True
        self._fit_mean_head(train_data)
        self._fit_out_of_fold(train_data)
        self._meta_extra = {
            "train_row_count": int(len(train_data)),
            "validation_row_count": int(len(validation_data)) if validation_data is not None else 0,
        }
        if self.dispersion is not None:
            self._meta_extra.update(self.dispersion.as_metadata())
        if "GAME_DATE" in train_data.columns:
            d = pd.to_datetime(train_data["GAME_DATE"])
            self._meta_extra["train_start_date"] = str(d.min().date())
            self._meta_extra["train_end_date"] = str(d.max().date())
        return self

    def _fit_mean_head(self, train_data: pd.DataFrame) -> None:
        """Fit the stat regressor and estimate dispersion from training residuals."""
        target = self.target_market
        if target not in train_data.columns:
            logger.warning(
                "xgboost_adapter: realised %s absent — no mean head, so MAE/RMSE "
                "are unavailable for this market.",
                target,
            )
            return

        from xgboost import XGBRegressor

        work = train_data
        if "GAME_DATE" in work.columns:
            work = work.sort_values("GAME_DATE")
        y = pd.to_numeric(work[target], errors="coerce")
        rows = work.loc[y.notna()]
        y = y.loc[rows.index]
        if rows.empty:
            logger.warning("xgboost_adapter: no labelled %s rows for the mean head", target)
            return

        params = {
            k: v
            for k, v in self._pipe.model_params.items()
            if k not in {"objective", "eval_metric"}
        }
        X = rows[self.feature_cols].apply(pd.to_numeric, errors="coerce")

        def _train_predict(X_tr, y_tr, X_va):
            fold = XGBRegressor(objective="reg:squarederror", **params)
            fold.fit(X_tr, y_tr)
            return np.clip(fold.predict(X_va), 0, None)

        # Out-of-fold residuals only: in-sample residuals from a boosted
        # tree are far too tight and produce an overconfident distribution.
        self.dispersion = fit_dispersion_out_of_fold(
            _train_predict, X, y.to_numpy(), market=target
        )

        self.mean_model = XGBRegressor(objective="reg:squarederror", **params)
        self.mean_model.fit(X, y)

    def _fit_out_of_fold(self, train_data: pd.DataFrame) -> None:
        """
        Out-of-fold P(over) over the training window, for the calibrator.

        Produced here rather than by refitting the whole component later:
        the calibrator then sees the WHOLE training window instead of its
        last 30%, and on the same chronological folds the dispersion used.
        """
        from src.models.oof import chronological_oof_probabilities

        if "over_hit" not in train_data.columns:
            self.oof = None
            return
        work = train_data
        if "GAME_DATE" in work.columns:
            work = work.sort_values("GAME_DATE")
        y = pd.to_numeric(work["over_hit"], errors="coerce")
        rows = work.loc[y.notna()]
        if rows.empty:
            self.oof = None
            return

        pipe_cls = type(self._pipe)
        # Carry the WHOLE configuration into each fold, not just model_params.
        # tuning and n_splits are separate constructor arguments, so passing
        # only model_params left every fold on DEFAULT_TUNING and n_splits=5:
        # an adapter built with n_estimators_max=300 and n_splits=3 produced
        # out-of-fold probabilities from models tuned to 2000 and 5. Those
        # probabilities are what the calibrator is fitted on, so the
        # calibrator was corrected against a differently-tuned model than the
        # one it later corrects.
        params = self._pipe.model_params
        tuning = self._pipe.tuning
        n_splits = self._pipe.n_splits

        # Whole rows travel through the folds, not just the feature matrix:
        # the pipeline reads GAME_DATE to verify its own splits are
        # chronological, and handing it a bare X made it warn that it could
        # not check the very property this pass exists to guarantee.
        def _fit_predict(rows_tr, y_tr, rows_va):
            fold = pipe_cls(
                self.feature_cols,
                model_params=params,
                tuning=tuning,
                n_splits=n_splits,
            )
            fold.fit(rows_tr, target_col="over_hit")
            return fold.predict_proba_over(rows_va)

        self.oof = chronological_oof_probabilities(
            _fit_predict, rows, y.loc[rows.index].to_numpy(),
            market=self.target_market,
        )

    def predict_mean(self, features: pd.DataFrame) -> pd.Series:
        if self.mean_model is None:
            logger.warning("xgboost_adapter: mean head unfitted — returning nulls")
            return pd.Series([None] * len(features), index=features.index, dtype="object")
        X = features[self.feature_cols].apply(pd.to_numeric, errors="coerce")
        return pd.Series(np.clip(self.mean_model.predict(X), 0, None), index=features.index, dtype=float)

    def predict_distribution(self, features: pd.DataFrame) -> pd.DataFrame:
        means = self.predict_mean(features)
        if self.dispersion is None:
            return pd.DataFrame(
                {
                    "prediction_mean": means,
                    "prediction_std_or_dispersion": None,
                    "method": "no_dispersion_fitted",
                },
                index=features.index,
            )
        std = means.apply(
            lambda m: float(np.sqrt(self.dispersion.variance_for(m))) if pd.notna(m) else None
        )
        return pd.DataFrame(
            {
                "prediction_mean": means,
                "prediction_std_or_dispersion": std,
                "method": self.dispersion.family,
            },
            index=features.index,
        )

    def predict_probability_over(
        self,
        features: pd.DataFrame,
        line: float | pd.Series,
    ) -> pd.Series:
        if not self._fitted or self._pipe.model is None:
            raise RuntimeError("XGBoost adapter is not fitted — cannot silently substitute another model")
        raw = pd.Series(self._pipe.predict_proba_over(features), index=features.index)
        if self.line_aware:
            # The line is a fitted feature, so this probability is already
            # AT the requested line. Nothing to abstain from.
            return raw
        # See mask_probabilities_at_unsupported_lines: the classifier's
        # probability is valid only at the line its labels were built from.
        masked, _ = mask_probabilities_at_unsupported_lines(
            raw,
            features,
            line,
            model_name=f"xgboost/{self.target_market}",
        )
        return masked

    def predict_rows(
        self,
        features: pd.DataFrame,
        *,
        line_col: str = "RESEARCH_LINE",
    ) -> list[ModelPrediction]:
        if line_col not in features.columns:
            raise ValueError(f"DATA_NOT_AVAILABLE: missing {line_col}")
        probs = self.predict_probability_over(features, features[line_col])
        dist = self.predict_distribution(features)
        means = dist["prediction_mean"]
        stds = dist["prediction_std_or_dispersion"]

        out: list[ModelPrediction] = []
        for i, (_, row) in enumerate(features.iterrows()):
            line = row.get(line_col)
            mean_val = means.iloc[i]
            warnings: list[str] = []
            extras: dict[str, Any] = {}

            p_over = float(np.clip(probs.iloc[i], 1e-6, 1 - 1e-6))
            p_push: float | None = None

            # Push mass comes from the fitted count distribution; a binary
            # classifier has no way to express it.
            if self.dispersion is not None and pd.notna(line) and pd.notna(mean_val):
                d = over_under_push_from_dispersion(float(mean_val), float(line), self.dispersion)
                extras["distribution_probabilities"] = {
                    "probability_over": d.get("probability_over"),
                    "probability_under": d.get("probability_under"),
                    "probability_push": d.get("probability_push"),
                    "distribution": d.get("distribution"),
                }
                if d.get("status") == "OK":
                    p_push = float(d["probability_push"])
            elif self.dispersion is None:
                warnings.append("No fitted dispersion — push mass not modelled")

            if p_push is None:
                p_under = 1.0 - p_over
                if pd.notna(line) and float(line) == int(float(line)):
                    warnings.append(
                        "Whole-number line with no push model; P(under) is 1 - P(over)"
                    )
            else:
                remaining = 1.0 - p_push
                p_under = remaining * (1.0 - p_over)
                p_over = remaining * p_over

            out.append(
                ModelPrediction(
                    model_name=self.model_name,
                    model_version=self.model_version,
                    target_market=self.target_market,
                    event_id=str(row.get("GAME_ID", "")),
                    player_id=str(row.get("PLAYER_ID", "")),
                    player_name=row.get("PLAYER_NAME"),
                    prediction_mean=float(mean_val) if pd.notna(mean_val) else None,
                    prediction_std_or_dispersion=float(stds.iloc[i]) if pd.notna(stds.iloc[i]) else None,
                    prop_line=float(line) if pd.notna(line) else None,
                    probability_over=round(p_over, 6),
                    probability_under=round(p_under, 6),
                    probability_push=None if p_push is None else round(p_push, 6),
                    feature_schema_version=self.feature_schema_version,
                    warnings=warnings,
                    extras=extras,
                )
            )
        return out

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._pipe.model is None:
            raise RuntimeError("Cannot save unfitted XGBoost model")
        model_path = path.with_suffix(".json") if path.suffix == "" else path
        self._pipe.model.save_model(str(model_path))

        # The classifier alone is not the model. The mean head supplies every
        # MAE/RMSE figure and the dispersion supplies push mass on whole-number
        # lines, so saving only the booster made a reloaded model return null
        # projections and different probabilities than the one just evaluated.
        mean_path = model_path.with_suffix(".mean.json")
        if self.mean_model is not None:
            self.mean_model.save_model(str(mean_path))
        elif mean_path.exists():
            # A stale head from an earlier fit would be reloaded as current.
            mean_path.unlink()

        meta = {
            "feature_cols": self.feature_cols,
            "target_market": self.target_market,
            "model_version": self.model_version,
            "feature_schema_version": self.feature_schema_version,
            "model_params": self._pipe.effective_params(),
            # Recorded so a reload reconstructs the same pipeline. effective_params
            # reports the tree count the search LANDED on; these are the settings
            # that produced it, and without them a reloaded artifact silently
            # reverted to DEFAULT_TUNING and n_splits=5.
            "tuning": dict(self._pipe.tuning),
            "n_splits": int(self._pipe.n_splits),
            "dispersion": None if self.dispersion is None else self.dispersion.to_dict(),
            "has_mean_head": self.mean_model is not None,
            **self._meta_extra,
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        model_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        logger.info("Saved XGBoost artifact to %s", model_path)

    def load(self, path: Path | str) -> "XGBoostAdapter":
        path = Path(path)
        meta_path = path.with_suffix(".meta.json") if path.suffix != ".meta.json" else path
        model_path = path if path.suffix == ".json" and not str(path).endswith(".meta.json") else path.with_suffix(".json")
        if not meta_path.exists():
            # allow path/foo.json + path/foo.meta.json naming
            meta_path = Path(str(model_path).replace(".json", ".meta.json"))
        if not model_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"XGBoost artifact missing at {model_path} / {meta_path} — "
                "train and save before scoring; will not substitute another model."
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.feature_cols = list(meta["feature_cols"])
        self.target_market = meta.get("target_market", self.target_market)
        self.model_version = meta.get("model_version", self.model_version)
        # Same omission as the fold path above: tuning and n_splits are their
        # own constructor arguments, so a reloaded artifact silently reverted to
        # DEFAULT_TUNING. Persisted when present; absent in older sidecars,
        # where the constructor default is the honest answer.
        self._pipe = XGBoostPropPipeline(
            self.feature_cols,
            model_params=meta.get("model_params"),
            tuning=meta.get("tuning"),
            **({"n_splits": int(meta["n_splits"])} if meta.get("n_splits") else {}),
        )
        import xgboost as xgb

        booster = xgb.XGBClassifier()
        booster.load_model(str(model_path))
        self._pipe.model = booster

        self.dispersion = CountDispersion.from_dict(meta.get("dispersion"))
        mean_path = model_path.with_suffix(".mean.json")
        if mean_path.exists():
            regressor = xgb.XGBRegressor()
            regressor.load_model(str(mean_path))
            self.mean_model = regressor
        else:
            self.mean_model = None
            if meta.get("has_mean_head"):
                logger.warning(
                    "Mean head recorded in %s but %s is absent — projections will be "
                    "null. The artifact is incomplete; re-save rather than scoring.",
                    meta_path.name, mean_path.name,
                )

        self._fitted = True
        self._meta_extra = {
            k: v for k, v in meta.items()
            if k not in {"feature_cols", "model_params", "dispersion", "has_mean_head"}
        }
        return self

    def get_model_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_name=self.model_name,
            model_version=self.model_version,
            target_market=self.target_market,
            feature_cols=self.feature_cols,
            hyperparameters=self._pipe.effective_params(),
            train_row_count=self._meta_extra.get("train_row_count"),
            validation_row_count=self._meta_extra.get("validation_row_count"),
            train_start_date=self._meta_extra.get("train_start_date"),
            train_end_date=self._meta_extra.get("train_end_date"),
            feature_schema_version=self.feature_schema_version,
            random_seed=self._pipe.random_state,
            notes=["Wraps existing XGBoostPropPipeline without behavior changes"],
        )
