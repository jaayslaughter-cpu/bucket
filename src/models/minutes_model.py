"""Separate minutes projection model (pregame features only)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.models.prediction_schema import ModelMetadata

logger = logging.getLogger(__name__)

try:
    from catboost import CatBoostRegressor, Pool
except ImportError:  # pragma: no cover
    CatBoostRegressor = None  # type: ignore[misc, assignment]
    Pool = None  # type: ignore[misc, assignment]

# Pregame-only numeric features — never same-game MIN / box stats
DEFAULT_MINUTES_FEATURES = [
    "MIN_L5",
    "MIN_L10",
    "MIN_SEASON",
    "fatigue_multiplier",
    "IS_HOME",
    "PACE_MULTIPLIER",
]

DEFAULT_MINUTES_CATS = ["TEAM_ABBREVIATION", "OPPONENT_ABBREVIATION", "SEASON"]


class MinutesModel:
    """CatBoost mean minutes + optional quantile models (P10/P50/P90)."""

    model_name = "minutes_catboost"

    def __init__(
        self,
        feature_cols: list[str] | None = None,
        *,
        categorical_features: list[str] | None = None,
        hyperparameters: dict[str, Any] | None = None,
        quantiles: list[float] | None = None,
        random_seed: int = 42,
        model_version: str = "min_v1",
    ) -> None:
        if CatBoostRegressor is None:
            raise ImportError("catboost required for MinutesModel — pip install 'propiq-analytics[ml]'")
        self.feature_cols = list(feature_cols or DEFAULT_MINUTES_FEATURES)
        self.categorical_features = list(categorical_features or DEFAULT_MINUTES_CATS)
        self.quantiles = list(quantiles or [0.1, 0.5, 0.9])
        defaults = {
            "iterations": 300,
            "depth": 6,
            "learning_rate": 0.05,
            "loss_function": "RMSE",
            "random_seed": random_seed,
            "task_type": "CPU",
            "verbose": False,
            "early_stopping_rounds": 30,
        }
        if hyperparameters:
            defaults.update(hyperparameters)
        defaults["task_type"] = "CPU" if str(defaults.get("task_type", "CPU")).upper() != "GPU" else "GPU"
        self.hyperparameters = defaults
        self.model_version = model_version
        self.random_seed = random_seed
        self.mean_model: Any | None = None
        self.quantile_models: dict[float, Any] = {}
        self._meta: dict[str, Any] = {}

    def _usable_cols(self, df: pd.DataFrame) -> tuple[list[str], list[str]]:
        feats = [c for c in self.feature_cols if c in df.columns]
        cats = [c for c in self.categorical_features if c in df.columns]
        # Include cats in feature matrix
        all_feats = list(dict.fromkeys(feats + cats))
        return all_feats, cats

    def fit(self, train_data: pd.DataFrame, validation_data: pd.DataFrame | None = None) -> "MinutesModel":
        if "MIN" not in train_data.columns:
            raise ValueError("DATA_NOT_AVAILABLE: MIN target missing")
        # Target is historical minutes of prior games in the panel row —
        # caller must ensure features are shift-safe (MIN_L* already are).
        feats, cats = self._usable_cols(train_data)
        if not feats:
            raise ValueError("DATA_NOT_AVAILABLE: no minutes feature columns present")
        train = train_data.dropna(subset=feats + ["MIN"]).copy()
        for c in cats:
            train[c] = train[c].astype(str).fillna("MISSING")
        y = pd.to_numeric(train["MIN"], errors="coerce")
        train = train.loc[y.notna()]
        y = y.loc[train.index]

        train_pool = Pool(train[feats], y, cat_features=cats or None)
        eval_set = None
        if validation_data is not None and not validation_data.empty:
            val = validation_data.dropna(subset=feats + ["MIN"]).copy()
            for c in cats:
                val[c] = val[c].astype(str).fillna("MISSING")
            if not val.empty:
                eval_set = Pool(val[feats], pd.to_numeric(val["MIN"], errors="coerce"), cat_features=cats or None)

        params = {k: v for k, v in self.hyperparameters.items() if k != "early_stopping_rounds"}
        early = int(self.hyperparameters.get("early_stopping_rounds") or 0)
        self.mean_model = CatBoostRegressor(**params)
        fit_kw: dict[str, Any] = {}
        if eval_set is not None and early > 0:
            fit_kw["eval_set"] = eval_set
            fit_kw["early_stopping_rounds"] = early
        self.mean_model.fit(train_pool, **fit_kw)

        self.quantile_models = {}
        for q in self.quantiles:
            qparams = dict(params)
            qparams["loss_function"] = f"Quantile:alpha={q}"
            qm = CatBoostRegressor(**qparams)
            qm.fit(train_pool, **fit_kw)
            self.quantile_models[q] = qm

        self.feature_cols = feats
        self.categorical_features = cats
        self._meta = {"train_row_count": len(train), "feature_cols": feats}
        logger.info("minutes model fitted rows=%d feats=%s", len(train), feats)
        return self

    def predict_mean(self, features: pd.DataFrame) -> pd.Series:
        if self.mean_model is None:
            raise RuntimeError("Minutes model not fitted")
        work = features.copy()
        for c in self.categorical_features:
            if c in work.columns:
                work[c] = work[c].astype(str).fillna("MISSING")
        for c in self.feature_cols:
            if c not in self.categorical_features:
                work[c] = pd.to_numeric(work[c], errors="coerce").fillna(0.0)
        pred = self.mean_model.predict(work[self.feature_cols])
        return pd.Series(np.clip(pred, 0, 48), index=features.index)

    def predict_quantiles(self, features: pd.DataFrame) -> pd.DataFrame:
        work = features.copy()
        for c in self.categorical_features:
            if c in work.columns:
                work[c] = work[c].astype(str).fillna("MISSING")
        for c in self.feature_cols:
            if c not in self.categorical_features:
                work[c] = pd.to_numeric(work[c], errors="coerce").fillna(0.0)
        cols = {}
        for q, model in self.quantile_models.items():
            cols[f"MIN_P{int(q * 100)}"] = np.clip(model.predict(work[self.feature_cols]), 0, 48)
        return pd.DataFrame(cols, index=features.index)

    def uncertainty(self, features: pd.DataFrame) -> pd.Series:
        q = self.predict_quantiles(features)
        if "MIN_P10" in q.columns and "MIN_P90" in q.columns:
            return (q["MIN_P90"] - q["MIN_P10"]) / 2.0
        return pd.Series([None] * len(features), index=features.index, dtype="object")

    def injury_warnings(self, features: pd.DataFrame) -> list[list[str]]:
        """Attach warnings only when injury/starter columns exist — never invent."""
        out: list[list[str]] = []
        for _, row in features.iterrows():
            w: list[str] = []
            if "BBS_OUT_FLAG" in features.columns and bool(row.get("BBS_OUT_FLAG")):
                w.append("BBS_OUT_FLAG=1 — minutes may be zero / unavailable")
            if "STARTER" in features.columns and pd.isna(row.get("STARTER")):
                w.append("STARTER status missing")
            if "INJURY_STATUS" in features.columns and pd.notna(row.get("INJURY_STATUS")):
                w.append(f"INJURY_STATUS={row.get('INJURY_STATUS')}")
            if not any(c in features.columns for c in ("BBS_OUT_FLAG", "STARTER", "INJURY_STATUS")):
                w.append("No injury/starter fields present — minutes model unguided on availability")
            out.append(w)
        return out

    def get_model_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_name=self.model_name,
            model_version=self.model_version,
            target_market="MIN",
            feature_cols=self.feature_cols,
            categorical_cols=self.categorical_features,
            hyperparameters=self.hyperparameters,
            train_row_count=self._meta.get("train_row_count"),
            random_seed=self.random_seed,
            notes=["Pregame features only", "Does not use same-game MIN as a feature"],
        )
