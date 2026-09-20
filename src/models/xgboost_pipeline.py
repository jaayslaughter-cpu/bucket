"""XGBoost binary P(Over) baseline.

PROVENANCE — READ THIS BEFORE QUOTING ANY COMPARISON RESULT

This file was written fresh for this repository. It is NOT a recovered
copy of an earlier PropIQ baseline; no such file was available. Treat it
as "a reasonable XGBoost baseline", never as "the incumbent model that
CatBoost had to beat". A comparison against it says which of two models
written at the same time scores better on the same split — nothing about
a pre-existing production system.

Chronological by construction: validation uses ``TimeSeriesSplit``, never
a random K-fold, because shuffling game rows lets a model train on games
that happen after the ones it is scored on.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None  # type: ignore[misc, assignment]


DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 400,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
}


class XGBoostPropPipeline:
    """Binary classifier for P(stat > line).

    ``feature_cols`` is required positionally and is persisted alongside
    the booster: scoring with a different column order silently produces
    garbage rather than raising, so the order is part of the artifact.
    """

    model_name = "xgboost"

    def __init__(
        self,
        feature_cols: Sequence[str],
        *,
        n_splits: int = 5,
        random_state: int = 42,
        model_params: dict[str, Any] | None = None,
    ) -> None:
        if XGBClassifier is None:
            raise ImportError(
                "xgboost is required for the baseline. Install with: "
                "pip install 'propiq-analytics[ml]'"
            )
        if not feature_cols:
            raise ValueError("feature_cols must be a non-empty sequence")

        self.feature_cols = list(feature_cols)
        self.n_splits = int(n_splits)
        self.random_state = int(random_state)
        self.model_params = {**DEFAULT_PARAMS, "random_state": self.random_state}
        if model_params:
            self.model_params.update(model_params)
        self.model: Any | None = None
        self.cv_scores_: list[float] = []

    def _matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"DATA_NOT_AVAILABLE: missing feature columns {missing}")
        raw = df[self.feature_cols]
        out = raw.apply(pd.to_numeric, errors="coerce")
        # A value that was present but failed to parse is a data fault, not a
        # missing observation. Coercing it to NaN lets XGBoost score the row
        # anyway and returns a confident probability built on a silently
        # discarded value.
        unparseable = raw.notna() & out.isna()
        if unparseable.any().any():
            bad = list(raw.columns[unparseable.any(axis=0)])
            raise ValueError(
                f"DATA_NOT_AVAILABLE: non-numeric values in feature column(s) {bad}"
            )
        # Genuine NaNs are left alone: XGBoost handles them natively via its
        # default split direction, which beats imputing a value nobody chose.
        return out

    def fit(self, train_data: pd.DataFrame, target_col: str = "over_hit") -> "XGBoostPropPipeline":
        """Fit on chronologically ordered rows, reporting TimeSeriesSplit scores."""
        if target_col not in train_data.columns:
            raise ValueError(f"DATA_NOT_AVAILABLE: missing target {target_col!r}")

        work = train_data
        if "GAME_DATE" in work.columns:
            work = work.sort_values("GAME_DATE")
        else:
            logger.warning(
                "No GAME_DATE on the training frame — cannot guarantee chronological "
                "order, so TimeSeriesSplit scores may not be honest."
            )

        y_full = pd.to_numeric(work[target_col], errors="coerce")
        keep = y_full.notna()
        work, y_full = work.loc[keep], y_full.loc[keep]
        if work.empty:
            raise ValueError("DATA_NOT_AVAILABLE: no labelled rows to train on")

        X = self._matrix(work)
        # astype(int) truncates rather than rejecting: a 0.5 becomes 0 and a
        # probability-valued target would train as a silently altered label.
        invalid = ~y_full.isin((0, 1))
        if invalid.any():
            raise ValueError(
                f"DATA_NOT_AVAILABLE: target {target_col!r} must contain only 0 and 1, "
                f"found {sorted(set(y_full[invalid].unique()))[:5]}"
            )
        y = y_full.astype(int)

        if y.nunique() < 2:
            raise ValueError(
                f"DATA_NOT_AVAILABLE: target {target_col!r} has a single class — "
                "nothing to separate"
            )

        self.cv_scores_ = []
        n_splits = min(self.n_splits, max(2, len(X) // 50))
        if len(X) >= 100:
            from sklearn.metrics import log_loss

            for fold, (tr, va) in enumerate(TimeSeriesSplit(n_splits=n_splits).split(X), 1):
                if y.iloc[tr].nunique() < 2 or y.iloc[va].nunique() < 2:
                    continue
                fold_model = XGBClassifier(**self.model_params)
                fold_model.fit(X.iloc[tr], y.iloc[tr])
                p = fold_model.predict_proba(X.iloc[va])[:, 1]
                score = float(log_loss(y.iloc[va], p, labels=[0, 1]))
                self.cv_scores_.append(score)
                logger.debug("xgboost fold %d log_loss=%.4f", fold, score)
        else:
            logger.info("Only %d rows — skipping TimeSeriesSplit scoring.", len(X))

        self.model = XGBClassifier(**self.model_params)
        self.model.fit(X, y)

        logger.info(
            "xgboost fitted rows=%d features=%d folds=%d mean_cv_log_loss=%s",
            len(X),
            len(self.feature_cols),
            len(self.cv_scores_),
            f"{np.mean(self.cv_scores_):.4f}" if self.cv_scores_ else "n/a",
        )
        return self

    def predict_proba_over(self, features: pd.DataFrame) -> np.ndarray:
        """P(over) for each row. Raises rather than guessing when unfitted."""
        if self.model is None:
            raise RuntimeError(
                "XGBoost pipeline is not fitted — refusing to return probabilities. "
                "Train it, or load a persisted booster plus its feature_cols sidecar."
            )
        return np.asarray(self.model.predict_proba(self._matrix(features))[:, 1], dtype=float)
