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


# Tuning knobs, kept OUT of DEFAULT_PARAMS because they are not XGBoost
# constructor arguments in their own right: they control the search that
# CHOOSES n_estimators. Mixing them into model_params would forward them to
# XGBRegressor for the mean head, which does not accept them.
DEFAULT_TUNING: dict[str, Any] = {
    # Ceiling for the cross-validated search. Early stopping ends the fit long
    # before this on any real panel; it exists so a fold cannot run away.
    "n_estimators_max": 2000,
    # Used only when no fold could run (too few rows, or one class).
    "n_estimators_fallback": 400,
    "early_stopping_rounds": 40,
    "min_trees": 5,
    # Fold models train on less data than the final model, and the best tree
    # count grows with data: on a 4,056-row panel the per-fold best iterations
    # ran 2, 25, 33, 49, 81 as the folds got larger. Scaling the median by
    # (final rows / mean fold rows) accounts for that and measured better than
    # not scaling. See docs/xgboost_early_stopping.md.
    "scale_with_data": True,
}

DEFAULT_PARAMS: dict[str, Any] = {
    # NOT a tuned value. It is the fallback the search overrides; see
    # learned_n_estimators_ for what a fit actually used.
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


def split_xgboost_config(block: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Split a config block into XGBoost constructor args and tuning knobs.

    The knobs control the search that CHOOSES n_estimators and are not
    themselves XGBoost arguments. Forwarding one to XGBRegressor (the mean
    head reuses model_params) raises, so the split happens once, here, rather
    than being remembered at each call site.

    Unknown keys go to model_params so a new XGBoost argument can be set from
    config without a code change; XGBoost rejects a genuinely bad one.
    """
    tuning = {k: v for k, v in block.items() if k in DEFAULT_TUNING}
    params = {k: v for k, v in block.items() if k not in DEFAULT_TUNING}
    return params, tuning


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
        tuning: dict[str, Any] | None = None,
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
        self.tuning = {**DEFAULT_TUNING, **(tuning or {})}
        self.model: Any | None = None
        self.cv_scores_: list[float] = []
        # What the cross-validated search found, and what the final fit used.
        # Both are exported with the artifact: "400 trees" in a metadata file
        # that actually trained 76 is worse than no metadata.
        self.cv_best_iterations_: list[int] = []
        self.learned_n_estimators_: int | None = None
        self.n_estimators_source_: str = "configured"

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

    def fit(
        self,
        train_data: pd.DataFrame,
        target_col: str = "over_hit",
        sample_weight: "pd.Series | None" = None,
    ) -> "XGBoostPropPipeline":
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
        self.cv_best_iterations_ = []
        fold_train_rows: list[int] = []
        n_splits = min(self.n_splits, max(2, len(X) // 50))
        esr = int(self.tuning.get("early_stopping_rounds") or 0)
        if len(X) >= 100:
            from sklearn.metrics import log_loss

            for fold, (tr, va) in enumerate(TimeSeriesSplit(n_splits=n_splits).split(X), 1):
                if y.iloc[tr].nunique() < 2 or y.iloc[va].nunique() < 2:
                    continue
                # The eval set is this fold's own validation slice. It comes
                # from inside the training data and, because TimeSeriesSplit is
                # chronological, lies strictly AFTER the rows the fold trains
                # on. Using the outer validation window instead would tune the
                # tree count on the very rows the model is later scored against.
                fold_params = dict(self.model_params)
                fit_kwargs: dict[str, Any] = {}
                if esr > 0:
                    fold_params["n_estimators"] = int(self.tuning["n_estimators_max"])
                    fold_params["early_stopping_rounds"] = esr
                    fit_kwargs = {
                        "eval_set": [(X.iloc[va], y.iloc[va])],
                        "verbose": False,
                    }
                fold_model = XGBClassifier(**fold_params)
                fold_model.fit(X.iloc[tr], y.iloc[tr], **fit_kwargs)
                if esr > 0 and fold_model.best_iteration is not None:
                    self.cv_best_iterations_.append(int(fold_model.best_iteration) + 1)
                    fold_train_rows.append(len(tr))
                p = fold_model.predict_proba(X.iloc[va])[:, 1]
                score = float(log_loss(y.iloc[va], p, labels=[0, 1]))
                self.cv_scores_.append(score)
                logger.debug("xgboost fold %d log_loss=%.4f", fold, score)
        else:
            logger.info("Only %d rows — skipping TimeSeriesSplit scoring.", len(X))

        final_params = dict(self.model_params)
        learned = self._learn_n_estimators(self.cv_best_iterations_, fold_train_rows, len(X))
        if learned is not None:
            final_params["n_estimators"] = learned
            self.learned_n_estimators_ = learned
            self.n_estimators_source_ = "cross_validated_early_stopping"
            logger.info(
                "xgboost tree count LEARNED: %d (fold best iterations %s over "
                "%d rows). The configured %d was a fallback, not a tuned value.",
                learned, self.cv_best_iterations_, len(X),
                int(self.model_params.get("n_estimators", 0)),
            )
        else:
            self.n_estimators_source_ = "configured"
            logger.info(
                "xgboost tree count NOT learned (no usable fold) — falling back "
                "to the configured %d. This number is not tuned.",
                int(final_params.get("n_estimators", 0)),
            )
        self.model = XGBClassifier(**final_params)
        # Recency weights, when supplied, are aligned by index rather than
        # position: rows were dropped above for missing targets, so a
        # positional zip would silently pair each weight with the wrong game.
        weights = None
        if sample_weight is not None:
            weights = pd.Series(sample_weight).reindex(X.index)
            if weights.isna().any():
                raise ValueError(
                    "DATA_NOT_AVAILABLE: sample_weight does not cover every "
                    "training row after filtering — refusing to fit with "
                    "weights that do not line up with the rows."
                )
            self.sample_weight_summary_ = {
                "n": int(len(weights)),
                "effective_sample_size": round(
                    float(weights.sum() ** 2 / (weights ** 2).sum()), 1
                ),
            }
            logger.info(
                "xgboost fitting with recency weights: %s", self.sample_weight_summary_
            )
        self.model.fit(X, y, sample_weight=weights)

        logger.info(
            "xgboost fitted rows=%d features=%d folds=%d trees=%d (%s) "
            "mean_cv_log_loss=%s",
            len(X),
            len(self.feature_cols),
            len(self.cv_scores_),
            int(final_params.get("n_estimators", 0)),
            self.n_estimators_source_,
            f"{np.mean(self.cv_scores_):.4f}" if self.cv_scores_ else "n/a",
        )
        return self

    def effective_params(self) -> dict[str, Any]:
        """
        The parameters the fitted model ACTUALLY used, with provenance.

        ``model_params`` still carries the configured fallback, so reporting
        it after a learned fit publishes "400 trees" for a model that grew 76.
        Metadata that is confidently wrong is worse than metadata that is
        missing, so exports read this instead.
        """
        out = dict(self.model_params)
        if self.learned_n_estimators_ is not None:
            out["n_estimators"] = int(self.learned_n_estimators_)
        out["n_estimators_source"] = self.n_estimators_source_
        if self.cv_best_iterations_:
            out["cv_best_iterations"] = list(self.cv_best_iterations_)
        return out

    def _learn_n_estimators(
        self, best: list[int], fold_rows: list[int], n_final: int
    ) -> int | None:
        """
        Tree count from the fold searches, or None when none could run.

        The MEDIAN, not the minimum: a fold that stops after two trees is an
        unlucky slice, and taking the minimum underfit measurably (Brier
        0.22611 against 0.21844 for the scaled median on the same panel).
        """
        if not best:
            return None
        learned = float(np.median(best))
        if self.tuning.get("scale_with_data", True) and fold_rows:
            mean_fold_rows = float(np.mean(fold_rows))
            if mean_fold_rows > 0:
                learned *= n_final / mean_fold_rows
        ceiling = int(self.tuning["n_estimators_max"])
        floor = int(self.tuning["min_trees"])
        return int(min(max(round(learned), floor), ceiling))

    def predict_proba_over(self, features: pd.DataFrame) -> np.ndarray:
        """P(over) for each row. Raises rather than guessing when unfitted."""
        if self.model is None:
            raise RuntimeError(
                "XGBoost pipeline is not fitted — refusing to return probabilities. "
                "Train it, or load a persisted booster plus its feature_cols sidecar."
            )
        return np.asarray(self.model.predict_proba(self._matrix(features))[:, 1], dtype=float)
