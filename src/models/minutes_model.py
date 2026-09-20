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
# PACE_MULTIPLIER is deliberately absent: no ingestion path produces a
# pregame pace figure, and the feature builder no longer fabricates a
# neutral 1.0 for it. Add it back here once a verified pace source is
# joined — listing it now would make the default configuration refuse to
# train on every real panel.
DEFAULT_MINUTES_FEATURES = [
    "MIN_L5",
    "MIN_L10",
    "MIN_SEASON",
    "fatigue_multiplier",
    "IS_HOME",
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
        """Resolve the configured columns, refusing to quietly train a
        narrower model than was asked for.

        Silently dropping an absent feature produces a model that reports
        success while having been fitted on something other than its
        configuration — indistinguishable afterwards from the intended one.
        """
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"DATA_NOT_AVAILABLE: minutes features {missing} absent from the "
                f"training frame. Pass an explicit feature_cols list if a narrower "
                f"model is intended."
            )
        cats = [c for c in self.categorical_features if c in df.columns]
        absent_cats = [c for c in self.categorical_features if c not in df.columns]
        if absent_cats:
            logger.info("minutes model: categorical(s) %s absent — omitted", absent_cats)
        all_feats = list(dict.fromkeys(list(self.feature_cols) + cats))
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

        # Early stopping uses the tail of train, never validation_data: an
        # eval_set drawn from the scoring rows lets the stopping point be
        # chosen by their labels, which flatters every metric measured on
        # them afterwards.
        if "GAME_DATE" in train.columns:
            train = train.sort_values("GAME_DATE")
            y = y.loc[train.index]

        eval_set = None
        if len(train) >= 150:
            cut = int(len(train) * 0.85)
            fit_part, stop_part = train.iloc[:cut], train.iloc[cut:]
            train_pool = Pool(fit_part[feats], y.iloc[:cut], cat_features=cats or None)
            eval_set = Pool(stop_part[feats], y.iloc[cut:], cat_features=cats or None)
        else:
            train_pool = Pool(train[feats], y, cat_features=cats or None)
            logger.info(
                "minutes model: %d rows is too few to carve an early-stopping holdout",
                len(train),
            )

        if validation_data is not None and not validation_data.empty:
            logger.info(
                "minutes model: validation_data (%d rows) is recorded but not used for "
                "early stopping — it is the scoring set.",
                len(validation_data),
            )

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
        """Projected minutes, null where the inputs were never there.

        A row with no prior minutes history (a debut, or a player returning
        with a gap) used to have its features zero-filled and came back with
        a confident-looking projection built from invented zeros. Those rows
        now return null.
        """
        if self.mean_model is None:
            raise RuntimeError("Minutes model not fitted")
        work, usable = self._prepare_for_predict(features)
        out = pd.Series([np.nan] * len(features), index=features.index, dtype=float)
        if usable.any():
            pred = self.mean_model.predict(work.loc[usable, self.feature_cols])
            out.loc[usable] = np.clip(pred, 0, 48)
        if (~usable).any():
            logger.warning(
                "minutes model: %d of %d rows lack fitted features — returned null "
                "rather than a projection built from zeros",
                int((~usable).sum()), len(features),
            )
        return out

    def _prepare_for_predict(self, features: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Coerce inputs and flag which rows actually have the numeric features."""
        work = features.copy()
        for c in self.categorical_features:
            if c in work.columns:
                work[c] = work[c].astype(str).fillna("MISSING")
        numeric = [c for c in self.feature_cols if c not in self.categorical_features]
        for c in numeric:
            work[c] = pd.to_numeric(work.get(c), errors="coerce")
        usable = work[numeric].notna().all(axis=1) if numeric else pd.Series(True, index=work.index)
        return work, usable

    def predict_quantiles(self, features: pd.DataFrame) -> pd.DataFrame:
        """P10/P50/P90 minutes, null where the inputs were never there."""
        work, usable = self._prepare_for_predict(features)
        cols: dict[str, pd.Series] = {}
        for q, model in self.quantile_models.items():
            series = pd.Series([np.nan] * len(features), index=features.index, dtype=float)
            if usable.any():
                series.loc[usable] = np.clip(
                    model.predict(work.loc[usable, self.feature_cols]), 0, 48
                )
            cols[f"MIN_P{int(q * 100)}"] = series
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

    def save(self, path: Any) -> None:
        """Persist the mean head, every quantile head, and the feature contract.

        Without this, ``train-minutes`` fitted four boosters and discarded
        all of them when the process exited — the command reported success
        and left nothing behind. This was the only model in the project with
        no save/load.
        """
        import json
        from pathlib import Path

        target = Path(path)
        target = target.with_suffix("") if target.suffix else target
        target.parent.mkdir(parents=True, exist_ok=True)

        if self.mean_model is None:
            raise RuntimeError("Cannot save an unfitted MinutesModel")

        self.mean_model.save_model(str(target.with_suffix(".mean.cbm")))

        saved_quantiles = []
        for q, model in self.quantile_models.items():
            model.save_model(str(target.with_suffix(f".q{q}.cbm")))
            saved_quantiles.append(q)

        # Stale heads from a previous fit must not survive into this artifact,
        # or a reload would mix two models' quantiles.
        for old in target.parent.glob(f"{target.name}.q*.cbm"):
            q_text = old.name.rsplit(".q", 1)[-1].removesuffix(".cbm")
            try:
                if float(q_text) not in saved_quantiles:
                    old.unlink()
            except ValueError:  # not one of ours
                continue

        target.with_suffix(".meta.json").write_text(
            json.dumps(
                {
                    "model_version": self.model_version,
                    "feature_cols": self.feature_cols,
                    "categorical_features": self.categorical_features,
                    "quantiles": saved_quantiles,
                    "hyperparameters": self.hyperparameters,
                    "random_seed": self.random_seed,
                    "train_row_count": self._meta.get("train_row_count"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info("Saved minutes model to %s (%d quantile heads)", target, len(saved_quantiles))

    def load(self, path: Any) -> "MinutesModel":
        """Reload a saved minutes model. Refuses a missing or partial artifact."""
        import json
        from pathlib import Path

        target = Path(path)
        target = target.with_suffix("") if target.suffix else target
        meta_path = target.with_suffix(".meta.json")
        mean_path = target.with_suffix(".mean.cbm")

        if not meta_path.exists() or not mean_path.exists():
            raise FileNotFoundError(
                f"No minutes-model artifact at {target} — train and save before "
                "scoring; this will not substitute an unfitted model."
            )

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.model_version = meta.get("model_version", self.model_version)
        self.feature_cols = list(meta["feature_cols"])
        self.categorical_features = list(meta.get("categorical_features") or [])
        self.hyperparameters = dict(meta.get("hyperparameters") or self.hyperparameters)
        self.random_seed = meta.get("random_seed", self.random_seed)
        self._meta = {
            "train_row_count": meta.get("train_row_count"),
            "feature_cols": self.feature_cols,
        }

        self.mean_model = CatBoostRegressor()
        self.mean_model.load_model(str(mean_path))

        self.quantile_models = {}
        missing = []
        for q in meta.get("quantiles") or []:
            q_path = target.with_suffix(f".q{q}.cbm")
            if not q_path.exists():
                missing.append(q)
                continue
            head = CatBoostRegressor()
            head.load_model(str(q_path))
            self.quantile_models[float(q)] = head
        if missing:
            # Silently returning fewer quantiles would narrow every published
            # interval without saying so.
            raise FileNotFoundError(
                f"Minutes artifact at {target} is missing quantile head(s) {missing}. "
                "Re-save rather than scoring with a narrower interval than was fitted."
            )
        self.quantiles = sorted(self.quantile_models)
        logger.info("Loaded minutes model from %s (%d quantile heads)", target, len(self.quantile_models))
        return self

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
