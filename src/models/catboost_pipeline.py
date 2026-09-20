"""CatBoost challenger for P(Over) — CPU default, chronological early stopping."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.line_probs import classifier_over_under
from src.models.prediction_schema import ModelMetadata, ModelPrediction

logger = logging.getLogger(__name__)

try:
    from catboost import CatBoostClassifier, Pool
except ImportError:  # pragma: no cover
    CatBoostClassifier = None  # type: ignore[misc, assignment]
    Pool = None  # type: ignore[misc, assignment]


class CatBoostPropPipeline:
    """Binary P(Over) challenger. Does not replace XGBoost."""

    model_name = "catboost"

    def __init__(
        self,
        feature_cols: list[str],
        *,
        target_market: str = "PTS",
        model_version: str = "cb_v1",
        feature_schema_version: str = "fs_v1_shift1_l2",
        categorical_features: list[str] | None = None,
        hyperparameters: dict[str, Any] | None = None,
        random_seed: int = 42,
    ) -> None:
        if CatBoostClassifier is None:
            raise ImportError(
                "catboost is required for the challenger model. "
                "Install with: pip install 'propiq-analytics[ml]'"
            )
        self.feature_cols = list(feature_cols)
        self.target_market = target_market
        self.model_version = model_version
        self.feature_schema_version = feature_schema_version
        self.categorical_features = [
            c for c in (categorical_features or []) if c in self.feature_cols
        ]
        defaults: dict[str, Any] = {
            "iterations": 400,
            "depth": 6,
            "learning_rate": 0.05,
            "loss_function": "Logloss",
            "eval_metric": "Logloss",
            "random_seed": random_seed,
            "task_type": "CPU",
            "verbose": False,
            "early_stopping_rounds": 40,
            "l2_leaf_reg": 3.0,
        }
        if hyperparameters:
            defaults.update(hyperparameters)
        # Force CPU unless explicitly overridden to GPU by caller
        if str(defaults.get("task_type", "CPU")).upper() != "GPU":
            defaults["task_type"] = "CPU"
        self.hyperparameters = defaults
        self.random_seed = random_seed
        self.model: Any | None = None
        self._meta_extra: dict[str, Any] = {}

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"DATA_NOT_AVAILABLE: missing features {missing}")
        work = df.copy()
        for c in self.categorical_features:
            work[c] = work[c].astype(str).fillna("MISSING")
        for c in self.feature_cols:
            if c not in self.categorical_features:
                work[c] = pd.to_numeric(work[c], errors="coerce")
        return work

    def fit(self, train_data: pd.DataFrame, validation_data: pd.DataFrame | None = None) -> "CatBoostPropPipeline":
        if "over_hit" not in train_data.columns:
            raise ValueError("DATA_NOT_AVAILABLE: missing over_hit")
        train = self._prepare(train_data).dropna(subset=self.feature_cols + ["over_hit"])
        if "GAME_DATE" in train.columns:
            train = train.sort_values("GAME_DATE")
        y_train = train["over_hit"].astype(int)
        train_pool = Pool(
            train[self.feature_cols],
            y_train,
            cat_features=self.categorical_features or None,
        )
        eval_set = None
        val_rows = 0
        if validation_data is not None and not validation_data.empty:
            val = self._prepare(validation_data).dropna(subset=self.feature_cols + ["over_hit"])
            if not val.empty:
                eval_set = Pool(
                    val[self.feature_cols],
                    val["over_hit"].astype(int),
                    cat_features=self.categorical_features or None,
                )
                val_rows = len(val)

        params = {k: v for k, v in self.hyperparameters.items() if k != "early_stopping_rounds"}
        early = int(self.hyperparameters.get("early_stopping_rounds") or 0)
        self.model = CatBoostClassifier(**params)
        fit_kwargs: dict[str, Any] = {}
        if eval_set is not None and early > 0:
            fit_kwargs["eval_set"] = eval_set
            fit_kwargs["early_stopping_rounds"] = early
        self.model.fit(train_pool, **fit_kwargs)

        self._meta_extra = {
            "train_row_count": len(train),
            "validation_row_count": val_rows,
            "data_cutoff_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        if "GAME_DATE" in train.columns:
            d = pd.to_datetime(train["GAME_DATE"])
            self._meta_extra["train_start_date"] = str(d.min().date())
            self._meta_extra["train_end_date"] = str(d.max().date())
        logger.info(
            "catboost fitted market=%s train=%d val=%d cats=%s",
            self.target_market,
            len(train),
            val_rows,
            self.categorical_features,
        )
        return self

    def predict_mean(self, features: pd.DataFrame) -> pd.Series:
        col = f"{self.target_market}_L2"
        if col in features.columns:
            return pd.to_numeric(features[col], errors="coerce")
        return pd.Series([None] * len(features), index=features.index, dtype="object")

    def predict_distribution(self, features: pd.DataFrame) -> pd.DataFrame:
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
        if self.model is None:
            raise RuntimeError("CatBoost model is not fitted")
        work = self._prepare(features)
        # Unseen categories → CatBoost handles; numeric NaNs filled with column median of train if needed
        for c in self.feature_cols:
            if c not in self.categorical_features:
                work[c] = work[c].fillna(0.0)
        proba = self.model.predict_proba(work[self.feature_cols])[:, 1]
        return pd.Series(proba, index=features.index)

    def predict_rows(
        self,
        features: pd.DataFrame,
        *,
        line_col: str = "RESEARCH_LINE",
    ) -> list[ModelPrediction]:
        probs = self.predict_probability_over(features, features[line_col])
        means = self.predict_mean(features)
        out: list[ModelPrediction] = []
        for i, (_, row) in enumerate(features.iterrows()):
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
        if self.model is None:
            raise RuntimeError("Cannot save unfitted CatBoost model")
        model_path = path if path.suffix else path.with_suffix(".cbm")
        self.model.save_model(str(model_path))
        meta = {
            "feature_cols": self.feature_cols,
            "categorical_features": self.categorical_features,
            "target_market": self.target_market,
            "model_version": self.model_version,
            "feature_schema_version": self.feature_schema_version,
            "hyperparameters": self.hyperparameters,
            "random_seed": self.random_seed,
            **self._meta_extra,
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        model_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def load(self, path: Path | str) -> "CatBoostPropPipeline":
        path = Path(path)
        model_path = path if path.suffix == ".cbm" else path.with_suffix(".cbm")
        meta_path = Path(str(model_path) + ".meta.json") if not str(model_path).endswith(".meta.json") else path
        # Prefer sibling .meta.json
        alt = model_path.with_name(model_path.stem + ".meta.json")
        if alt.exists():
            meta_path = alt
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.feature_cols = list(meta["feature_cols"])
        self.categorical_features = list(meta.get("categorical_features") or [])
        self.target_market = meta.get("target_market", self.target_market)
        self.model_version = meta.get("model_version", self.model_version)
        self.hyperparameters = meta.get("hyperparameters", self.hyperparameters)
        self.model = CatBoostClassifier()
        self.model.load_model(str(model_path))
        self._meta_extra = {k: v for k, v in meta.items() if k not in {"feature_cols", "hyperparameters"}}
        return self

    def get_model_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_name=self.model_name,
            model_version=self.model_version,
            target_market=self.target_market,
            feature_cols=self.feature_cols,
            categorical_cols=self.categorical_features,
            hyperparameters=dict(self.hyperparameters),
            train_row_count=self._meta_extra.get("train_row_count"),
            validation_row_count=self._meta_extra.get("validation_row_count"),
            train_start_date=self._meta_extra.get("train_start_date"),
            train_end_date=self._meta_extra.get("train_end_date"),
            feature_schema_version=self.feature_schema_version,
            random_seed=self.random_seed,
            notes=["CatBoost challenger — CPU default; early stopping on chrono validation"],
        )

    def feature_importance_frame(self) -> pd.DataFrame:
        if self.model is None:
            return pd.DataFrame(columns=["feature_name", "importance", "rank"])
        raw = self.model.get_feature_importance()
        pairs = sorted(zip(self.feature_cols, raw), key=lambda t: float(t[1]), reverse=True)
        return pd.DataFrame(
            [
                {
                    "target_market": self.target_market,
                    "model_version": self.model_version,
                    "feature_name": name,
                    "importance": float(imp),
                    "rank": i + 1,
                    "trained_through_date": self._meta_extra.get("train_end_date"),
                }
                for i, (name, imp) in enumerate(pairs)
            ]
        )
