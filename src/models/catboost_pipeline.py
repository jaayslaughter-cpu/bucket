"""CatBoost challenger for P(Over) — CPU default, chronological early stopping."""

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

logger = logging.getLogger(__name__)

try:
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool
except ImportError:  # pragma: no cover
    CatBoostClassifier = None  # type: ignore[misc, assignment]
    CatBoostRegressor = None  # type: ignore[misc, assignment]
    Pool = None  # type: ignore[misc, assignment]


class CatBoostPropPipeline:
    """CatBoost challenger. Does not replace XGBoost.

    Fits two heads on the same features:

    - a **regressor** for the expected stat value, which is what MAE and
      RMSE score, and what the count distribution is centred on;
    - a **classifier** for P(over) at the labelled line.

    Both are needed. Without the regressor the model has no opinion about
    the stat itself and can only be compared on probability metrics.
    """

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
        self.mean_model: Any | None = None
        self.dispersion: CountDispersion | None = None
        self._meta_extra: dict[str, Any] = {}

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"DATA_NOT_AVAILABLE: missing features {missing}")
        work = df.copy()
        for c in self.categorical_features:
            work[c] = work[c].astype(str).fillna("MISSING")
        emptied = []
        for c in self.feature_cols:
            if c not in self.categorical_features:
                coerced = pd.to_numeric(work[c], errors="coerce")
                # A column that was entirely non-numeric becomes entirely NaN,
                # and the dropna below then removes every row. CatBoost reports
                # that as "Labels variable is empty", which points at the target
                # rather than at the string column that is really the problem.
                if len(work) and coerced.isna().all() and work[c].notna().any():
                    emptied.append(c)
                work[c] = coerced
        if emptied:
            raise ValueError(
                f"DATA_NOT_AVAILABLE: feature column(s) {emptied} hold no numeric "
                "values and are not declared categorical, so coercing them would "
                "drop every training row. Declare them in catboost."
                "categorical_features, or remove them from the feature list."
            )
        return work

    def _carve_early_stopping_split(
        self,
        train: pd.DataFrame,
        *,
        holdout_fraction: float = 0.15,
        min_holdout_rows: int = 50,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Split the tail off train for early stopping, chronologically.

        Returns (fit_rows, stop_rows). When train is too small to spare a
        holdout, stop_rows is None and the model trains for its full
        iteration count rather than stopping against rows it also fits.
        """
        if len(train) < min_holdout_rows * 3:
            logger.info(
                "catboost %s: %d training rows is too few to carve an early-stopping "
                "holdout — training the full iteration count instead",
                self.target_market, len(train),
            )
            return train, None
        cut = int(len(train) * (1.0 - holdout_fraction))
        return train.iloc[:cut], train.iloc[cut:]

    def fit(self, train_data: pd.DataFrame, validation_data: pd.DataFrame | None = None) -> "CatBoostPropPipeline":
        if "over_hit" not in train_data.columns:
            raise ValueError("DATA_NOT_AVAILABLE: missing over_hit")
        train = self._prepare(train_data).dropna(subset=self.feature_cols + ["over_hit"])
        if "GAME_DATE" in train.columns:
            train = train.sort_values("GAME_DATE")

        # EARLY STOPPING NEVER SEES validation_data. Handing the scoring set
        # to CatBoost as its eval_set lets early stopping pick the iteration
        # count by reading those labels, which makes every metric later
        # computed on that same set optimistic. The holdout is carved from
        # the tail of train instead; validation_data is used only to record
        # what it covered.
        fit_rows, stop_rows = self._carve_early_stopping_split(train)
        val_rows = 0
        if validation_data is not None and not validation_data.empty:
            val_rows = len(
                self._prepare(validation_data).dropna(subset=self.feature_cols + ["over_hit"])
            )

        train_pool = Pool(
            fit_rows[self.feature_cols],
            fit_rows["over_hit"].astype(int),
            cat_features=self.categorical_features or None,
        )
        eval_set = None
        if stop_rows is not None and not stop_rows.empty:
            eval_set = Pool(
                stop_rows[self.feature_cols],
                stop_rows["over_hit"].astype(int),
                cat_features=self.categorical_features or None,
            )

        params = {k: v for k, v in self.hyperparameters.items() if k != "early_stopping_rounds"}
        early = int(self.hyperparameters.get("early_stopping_rounds") or 0)
        self.model = CatBoostClassifier(**params)
        fit_kwargs: dict[str, Any] = {}
        if eval_set is not None and early > 0:
            fit_kwargs["eval_set"] = eval_set
            fit_kwargs["early_stopping_rounds"] = early
        self.model.fit(train_pool, **fit_kwargs)

        self._fit_mean_head(train)

        self._meta_extra = {
            "train_row_count": len(train),
            "validation_row_count": val_rows,
            "data_cutoff_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        if self.dispersion is not None:
            self._meta_extra.update(self.dispersion.as_metadata())
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

    def _fit_mean_head(self, train: pd.DataFrame) -> None:
        """Fit the regressor and estimate dispersion from training rows only.

        Takes no validation frame by design: dispersion comes from
        out-of-fold residuals within train, so nothing here may see the
        rows this model will later be scored on.
        """
        target = self.target_market
        if target not in train.columns:
            logger.warning(
                "catboost: realised %s column absent — no mean head, so MAE/RMSE "
                "and the count distribution are unavailable for this market.",
                target,
            )
            return

        y = pd.to_numeric(train[target], errors="coerce")
        fit_rows = train.loc[y.notna()]
        y = y.loc[fit_rows.index]
        if fit_rows.empty:
            logger.warning("catboost: no labelled %s rows for the mean head", target)
            return

        params = {
            k: v
            for k, v in self.hyperparameters.items()
            if k not in {"early_stopping_rounds", "loss_function", "eval_metric"}
        }
        params["loss_function"] = "RMSE"
        cats = self.categorical_features or None
        X = fit_rows[self.feature_cols]

        def _train_predict(X_tr, y_tr, X_va):
            fold = CatBoostRegressor(**params)
            fold.fit(Pool(X_tr, y_tr, cat_features=cats))
            return np.clip(fold.predict(X_va), 0, None)

        # Dispersion comes from out-of-fold residuals on TRAINING rows only.
        self.dispersion = fit_dispersion_out_of_fold(
            _train_predict, X, y.to_numpy(), market=target
        )

        self.mean_model = CatBoostRegressor(**params)
        self.mean_model.fit(Pool(X, y, cat_features=cats))

    def predict_mean(self, features: pd.DataFrame) -> pd.Series:
        if self.mean_model is None:
            logger.warning("catboost: mean head unfitted — returning nulls, not a fallback column")
            return pd.Series([None] * len(features), index=features.index, dtype="object")
        work = self._prepare(features)
        for c in self.feature_cols:
            if c not in self.categorical_features:
                work[c] = work[c].fillna(0.0)
        preds = np.clip(self.mean_model.predict(work[self.feature_cols]), 0, None)
        return pd.Series(preds, index=features.index, dtype=float)

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
        if self.model is None:
            raise RuntimeError("CatBoost model is not fitted")
        work = self._prepare(features)
        # Unseen categories → CatBoost handles; numeric NaNs filled with column median of train if needed
        for c in self.feature_cols:
            if c not in self.categorical_features:
                work[c] = work[c].fillna(0.0)
        proba = self.model.predict_proba(work[self.feature_cols])[:, 1]
        # The line is not a model input, so this probability answers only the
        # line the labels were built from. Asking at any other line abstains
        # rather than returning that number under a different label.
        masked, _ = mask_probabilities_at_unsupported_lines(
            pd.Series(proba, index=features.index),
            features,
            line,
            model_name=f"catboost/{self.target_market}",
        )
        return masked

    def predict_rows(
        self,
        features: pd.DataFrame,
        *,
        line_col: str = "RESEARCH_LINE",
    ) -> list[ModelPrediction]:
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

            # The classifier cannot represent a push. The fitted count
            # distribution can, so take push mass from it and let the
            # classifier split only the remaining probability.
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
        if self.model is None:
            raise RuntimeError("Cannot save unfitted CatBoost model")
        model_path = path if path.suffix else path.with_suffix(".cbm")
        self.model.save_model(str(model_path))
        # Remove a stale sidecar when this fit has no mean head, or load()
        # would resurrect a previous run's regressor as the current
        # projection without anything indicating the mismatch.
        mean_path = model_path.with_name(model_path.stem + ".mean.cbm")
        if self.mean_model is not None:
            self.mean_model.save_model(str(mean_path))
        elif mean_path.exists():
            mean_path.unlink()
            logger.info("Removed stale mean-head artifact at %s", mean_path)
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

        mean_path = model_path.with_name(model_path.stem + ".mean.cbm")
        if mean_path.exists():
            self.mean_model = CatBoostRegressor()
            self.mean_model.load_model(str(mean_path))
            if meta.get("dispersion_family"):
                self.dispersion = CountDispersion(
                    family=meta["dispersion_family"],
                    phi=float(meta.get("dispersion_phi", 1.0)),
                    n_train_rows=int(meta.get("dispersion_train_rows", 0)),
                    selection_scores=meta.get("dispersion_selection") or {},
                    fallback_reason=meta.get("dispersion_fallback_reason"),
                )
        else:
            logger.warning(
                "No mean head at %s — loaded classifier only, so this model has no "
                "stat projection and no count distribution.",
                mean_path,
            )

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
            notes=[
                "CatBoost challenger — CPU default; early stopping on chrono validation",
                "Two heads: RMSE regressor for the mean, Logloss classifier for P(over)",
                (
                    f"Dispersion {self.dispersion.family} phi={self.dispersion.phi:.3f} "
                    f"fitted on {self.dispersion.n_train_rows} training rows"
                    if self.dispersion
                    else "No dispersion fitted"
                ),
            ],
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
