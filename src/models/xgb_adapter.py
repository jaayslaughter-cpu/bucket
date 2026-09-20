"""Thin adapter around existing XGBoostPropPipeline — behavior preserved."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.models.line_probs import classifier_over_under
from src.models.prediction_schema import ModelMetadata, ModelPrediction
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
    ) -> None:
        self.target_market = target_market
        self.model_version = model_version
        self.feature_schema_version = feature_schema_version
        self.feature_cols = list(feature_cols)
        self._pipe = XGBoostPropPipeline(
            self.feature_cols,
            n_splits=n_splits,
            random_state=random_state,
            model_params=model_params,
        )
        self._meta_extra: dict[str, Any] = {}
        self._fitted = False

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
        self._meta_extra = {
            "train_row_count": int(len(train_data)),
            "validation_row_count": int(len(validation_data)) if validation_data is not None else 0,
        }
        if "GAME_DATE" in train_data.columns:
            d = pd.to_datetime(train_data["GAME_DATE"])
            self._meta_extra["train_start_date"] = str(d.min().date())
            self._meta_extra["train_end_date"] = str(d.max().date())
        return self

    def predict_mean(self, features: pd.DataFrame) -> pd.Series:
        col = f"{self.target_market}_L2"
        if col in features.columns:
            return pd.to_numeric(features[col], errors="coerce")
        logger.warning("xgboost_adapter: %s missing — mean left null", col)
        return pd.Series([None] * len(features), index=features.index, dtype="object")

    def predict_distribution(self, features: pd.DataFrame) -> pd.DataFrame:
        # Classifier has no full count distribution; expose mean + null dispersion.
        return pd.DataFrame(
            {
                "prediction_mean": self.predict_mean(features),
                "prediction_std_or_dispersion": None,
                "method": "classifier_no_count_distribution",
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
        return pd.Series(self._pipe.predict_proba_over(features), index=features.index)

    def predict_rows(
        self,
        features: pd.DataFrame,
        *,
        line_col: str = "RESEARCH_LINE",
    ) -> list[ModelPrediction]:
        if line_col not in features.columns:
            raise ValueError(f"DATA_NOT_AVAILABLE: missing {line_col}")
        probs = self.predict_probability_over(features, features[line_col])
        means = self.predict_mean(features)
        out: list[ModelPrediction] = []
        for i, (idx, row) in enumerate(features.iterrows()):
            line = row.get(line_col)
            cu = classifier_over_under(float(probs.iloc[i]), float(line) if pd.notna(line) else float("nan"))
            out.append(
                ModelPrediction(
                    model_name=self.model_name,
                    model_version=self.model_version,
                    target_market=self.target_market,
                    event_id=str(row.get("GAME_ID", "")),
                    player_id=str(row.get("PLAYER_ID", "")),
                    player_name=row.get("PLAYER_NAME"),
                    prediction_mean=float(means.iloc[i]) if pd.notna(means.iloc[i]) else None,
                    prop_line=float(line) if pd.notna(line) else None,
                    probability_over=cu.get("probability_over"),
                    probability_under=cu.get("probability_under"),
                    probability_push=cu.get("probability_push"),
                    feature_schema_version=self.feature_schema_version,
                    warnings=list(cu.get("warnings") or []),
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
        meta = {
            "feature_cols": self.feature_cols,
            "target_market": self.target_market,
            "model_version": self.model_version,
            "feature_schema_version": self.feature_schema_version,
            "model_params": self._pipe.model_params,
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
        self._pipe = XGBoostPropPipeline(self.feature_cols, model_params=meta.get("model_params"))
        import xgboost as xgb

        booster = xgb.XGBClassifier()
        booster.load_model(str(model_path))
        self._pipe.model = booster
        self._fitted = True
        self._meta_extra = {k: v for k, v in meta.items() if k not in {"feature_cols", "model_params"}}
        return self

    def get_model_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_name=self.model_name,
            model_version=self.model_version,
            target_market=self.target_market,
            feature_cols=self.feature_cols,
            hyperparameters=dict(self._pipe.model_params),
            train_row_count=self._meta_extra.get("train_row_count"),
            validation_row_count=self._meta_extra.get("validation_row_count"),
            train_start_date=self._meta_extra.get("train_start_date"),
            train_end_date=self._meta_extra.get("train_end_date"),
            feature_schema_version=self.feature_schema_version,
            random_seed=self._pipe.random_state,
            notes=["Wraps existing XGBoostPropPipeline without behavior changes"],
        )
