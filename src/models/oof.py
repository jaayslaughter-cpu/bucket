"""
src/models/oof.py — one chronological out-of-fold pass, shared by its consumers.

Two things in this pipeline need predictions a model did not train on, and
they were computing them separately:

- DISPERSION needs out-of-fold residuals. In-sample residuals from a boosted
  tree are far too tight, and the resulting distribution is overconfident at
  exactly the lines people bet.
- CALIBRATION needs out-of-fold probabilities. A calibrator fitted on the
  predictions it later corrects learns the noise it is supposed to smooth.

They were also DISAGREEING about what out-of-fold means: dispersion used a
3-fold chronological split over the whole training window, while calibration
used a single 70/30 split and refitted the entire component. So the
calibrator was corrected against a different sample than the dispersion was
fitted on, and only the last 30% of training rows ever reached it.

One pass fixes both. The folds are chronological — every prediction is made
by a model that saw only earlier rows — and the same fold boundaries feed
both consumers.

A row that no fold predicted keeps NaN. TimeSeriesSplit never predicts the
first block, so a fraction of the training window is legitimately absent
from the out-of-fold arrays; filling it would mean handing the calibrator
in-sample predictions, which is the failure this module exists to prevent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Below this a fold is too small to say anything, so the pass abstains
# rather than producing a calibrator fitted on noise.
MIN_ROWS_PER_FOLD = 40
MIN_USABLE_OOF_ROWS = 60


@dataclass(frozen=True)
class OutOfFoldPredictions:
    """Out-of-fold probabilities aligned to the training frame that made them."""

    frame: pd.DataFrame          # index = training rows; columns prob_over, y_over
    n_folds: int = 0
    reason: str | None = None

    @property
    def n_usable(self) -> int:
        if self.frame.empty:
            return 0
        return int((self.frame["prob_over"].notna() & self.frame["y_over"].notna()).sum())

    @property
    def usable(self) -> bool:
        return self.n_usable >= MIN_USABLE_OOF_ROWS

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """(y_true, p_pred) for the rows where both are present."""
        ok = self.frame["prob_over"].notna() & self.frame["y_over"].notna()
        block = self.frame.loc[ok]
        return (
            block["y_over"].to_numpy(dtype=float),
            block["prob_over"].to_numpy(dtype=float),
        )

    def as_metadata(self) -> dict[str, Any]:
        return {
            "oof_folds": self.n_folds,
            "oof_usable_rows": self.n_usable,
            "oof_reason": self.reason,
        }


def chronological_oof_probabilities(
    fit_predict_proba: Callable[[pd.DataFrame, np.ndarray, pd.DataFrame], np.ndarray],
    X: pd.DataFrame,
    y_over: np.ndarray,
    *,
    market: str = "",
    n_folds: int = 3,
) -> OutOfFoldPredictions:
    """
    Out-of-fold P(over) from chronological folds over the training window.

    ``X`` must already be in date order — the folds are positional, so an
    unsorted frame would train on later games and predict earlier ones,
    which is the leak this is built to avoid rather than one it tolerates.

    Returns predictions aligned to ``X.index``, NaN where no fold covered
    the row.
    """
    from sklearn.model_selection import TimeSeriesSplit

    y_over = np.asarray(y_over, dtype=float)
    n = len(X)
    empty = pd.DataFrame(
        {"prob_over": np.full(n, np.nan), "y_over": y_over}, index=X.index,
    )

    if n < MIN_ROWS_PER_FOLD * 2:
        return OutOfFoldPredictions(
            empty, 0,
            f"only {n} rows — too few for out-of-fold folds of {MIN_ROWS_PER_FOLD}",
        )

    splits = min(int(n_folds), max(2, n // MIN_ROWS_PER_FOLD))
    oof = np.full(n, np.nan)
    completed = 0

    for train_idx, valid_idx in TimeSeriesSplit(n_splits=splits).split(np.arange(n)):
        if len(train_idx) < MIN_ROWS_PER_FOLD or len(valid_idx) == 0:
            continue
        try:
            preds = fit_predict_proba(
                X.iloc[train_idx], y_over[train_idx], X.iloc[valid_idx],
            )
        except Exception as exc:  # noqa: BLE001 — one bad fold must not lose the rest
            logger.warning(
                "oof %s: fold %d failed (%s) — its rows stay NaN rather than "
                "being filled with an in-sample prediction.",
                market or "?", completed + 1, exc,
            )
            continue
        preds = np.asarray(preds, dtype=float).ravel()
        if len(preds) != len(valid_idx):
            logger.warning(
                "oof %s: fold returned %d predictions for %d rows — discarded.",
                market or "?", len(preds), len(valid_idx),
            )
            continue
        oof[valid_idx] = preds
        completed += 1

    frame = pd.DataFrame({"prob_over": oof, "y_over": y_over}, index=X.index)
    result = OutOfFoldPredictions(
        frame, completed,
        None if completed else "every fold failed or was too small",
    )
    logger.info(
        "oof %s: %d fold(s), %d of %d rows predicted out-of-fold",
        market or "?", completed, result.n_usable, n,
    )
    return result
