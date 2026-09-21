"""
Recency weighting — adapted from a reference implementation, made legible.

The reference used exp(-0.001 * days_ago) and reported nothing about what
the weighting cost. These tests pin the two changes: a half-life anyone
can interpret, and an effective-sample-size number that makes the cost
visible.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.recency import (
    DEFAULT_HALF_LIFE_DAYS,
    MIN_HALF_LIFE_DAYS,
    RecencyWeightError,
    effective_sample_size,
    exponential_recency_weights,
    fit_half_life_by_holdout,
    recency_weight_report,
)

DATES = pd.Series(pd.date_range("2023-01-01", "2025-01-01", freq="D"))


# --------------------------------------------------------------------------
# The half-life must mean what it says
# --------------------------------------------------------------------------

def test_weight_halves_over_exactly_one_half_life():
    weights = exponential_recency_weights(DATES, half_life_days=365)
    newest = weights.iloc[-1]
    one_half_life_earlier = weights.iloc[-1 - 365]
    assert newest / one_half_life_earlier == pytest.approx(2.0, abs=1e-6)


def test_weights_are_normalised_to_mean_one():
    """Changing the half-life must not also change the total weight, which
    would silently alter regularisation strength."""
    for half_life in (90, 240, 730):
        weights = exponential_recency_weights(DATES, half_life_days=half_life)
        assert weights.mean() == pytest.approx(1.0, abs=1e-9)


def test_recent_rows_outweigh_old_ones():
    weights = exponential_recency_weights(DATES, half_life_days=180)
    assert weights.iloc[-1] > weights.iloc[0]
    assert weights.is_monotonic_increasing


# --------------------------------------------------------------------------
# The cost of weighting must be visible
# --------------------------------------------------------------------------

def test_effective_sample_size_equals_n_for_equal_weights():
    equal = pd.Series(np.ones(500))
    assert effective_sample_size(equal) == pytest.approx(500.0)


def test_aggressive_weighting_collapses_the_effective_sample():
    """A model 'trained on 732 games' may hold far less information."""
    aggressive = recency_weight_report(DATES, half_life_days=30)
    mild = recency_weight_report(DATES, half_life_days=730)

    assert aggressive["effective_sample_size"] < mild["effective_sample_size"]
    assert aggressive["effective_fraction"] < 0.25, (
        "a 30-day half-life should visibly discard most of the sample"
    )
    assert mild["effective_fraction"] > 0.9


def test_effective_sample_size_handles_degenerate_input():
    assert effective_sample_size(pd.Series([], dtype=float)) == 0.0
    assert effective_sample_size(pd.Series([0.0, 0.0])) == 0.0


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------

def test_as_of_from_outside_the_training_window_is_refused():
    """A reference date after some training rows gives them weight > 1,
    which means it came from the validation window."""
    with pytest.raises(RecencyWeightError, match="leaks the split boundary"):
        exponential_recency_weights(
            DATES, half_life_days=365, as_of=pd.Timestamp("2024-06-01")
        )


def test_as_of_at_or_after_the_newest_row_is_allowed():
    exponential_recency_weights(DATES, half_life_days=365, as_of=DATES.max())
    exponential_recency_weights(
        DATES, half_life_days=365, as_of=DATES.max() + pd.Timedelta(days=5)
    )


def test_absurdly_short_half_life_is_refused():
    with pytest.raises(RecencyWeightError, match="not a weighting scheme"):
        exponential_recency_weights(DATES, half_life_days=1.0)
    assert MIN_HALF_LIFE_DAYS >= 7.0


def test_unparseable_dates_do_not_crash_the_weighting():
    dates = pd.Series(["2025-01-01", "not-a-date", "2025-02-01"])
    weights = exponential_recency_weights(dates, half_life_days=90)
    assert len(weights) == 3
    assert weights.notna().all()


def test_all_unparseable_dates_abstain():
    with pytest.raises(RecencyWeightError, match="DATA_NOT_AVAILABLE"):
        exponential_recency_weights(pd.Series(["x", "y"]), half_life_days=90)


# --------------------------------------------------------------------------
# Choosing the half-life instead of asserting one
# --------------------------------------------------------------------------

def test_half_life_can_be_chosen_by_holdout():
    """No-weighting must be able to win, or the search is rigged."""
    calls: list[float] = []

    def _score(train_idx, score_idx, weights):
        # Score improves as weights approach uniform, so the search should
        # land on the longest half-life offered.
        spread = float(weights.max() / max(weights.min(), 1e-9))
        calls.append(spread)
        return spread

    result = fit_half_life_by_holdout(_score, DATES)
    assert calls, "the scorer was never called"
    assert result["chosen_half_life_days"] >= 730.0


def test_half_life_search_abstains_on_too_few_rows():
    tiny = pd.Series(pd.date_range("2025-01-01", periods=10, freq="D"))
    result = fit_half_life_by_holdout(lambda a, b, c: 1.0, tiny)
    assert result["chosen_half_life_days"] == DEFAULT_HALF_LIFE_DAYS
    assert "reason_insufficient_rows" in result


# --------------------------------------------------------------------------
# Integration with the fitters
# --------------------------------------------------------------------------

def test_weights_actually_change_the_xgboost_fit():
    pytest.importorskip("xgboost")
    from src.features.builder import build_feature_matrix
    from src.models.data_audit import make_demo_panel
    from src.models.labels import attach_research_over_labels
    from src.models.xgboost_pipeline import XGBoostPropPipeline

    panel = build_feature_matrix(make_demo_panel(n_players=12, n_games=50))
    labelled = attach_research_over_labels(panel, stat="PTS")
    labelled = labelled.loc[labelled["over_hit"].notna()].reset_index(drop=True)
    cols = ["PTS_L5", "PTS_L10", "MIN_L5"]
    split = int(len(labelled) * 0.7)
    train, scoring = labelled.iloc[:split], labelled.iloc[split:]

    plain = XGBoostPropPipeline(cols, model_params={"n_estimators": 60})
    plain.fit(train)

    weights = exponential_recency_weights(train["GAME_DATE"], half_life_days=45)
    weighted = XGBoostPropPipeline(cols, model_params={"n_estimators": 60})
    weighted.fit(train, sample_weight=weights)

    assert not np.allclose(
        plain.predict_proba_over(scoring), weighted.predict_proba_over(scoring)
    ), "the weights had no effect on the fit"
    assert weighted.sample_weight_summary_["effective_sample_size"] < len(train)


def test_misaligned_weights_are_refused_not_mispaired():
    """Rows are dropped for missing targets, so a positional zip would pair
    each weight with the wrong game."""
    pytest.importorskip("xgboost")
    from src.features.builder import build_feature_matrix
    from src.models.data_audit import make_demo_panel
    from src.models.labels import attach_research_over_labels
    from src.models.xgboost_pipeline import XGBoostPropPipeline

    panel = build_feature_matrix(make_demo_panel(n_players=8, n_games=40))
    labelled = attach_research_over_labels(panel, stat="PTS")
    labelled = labelled.loc[labelled["over_hit"].notna()].reset_index(drop=True)

    short = exponential_recency_weights(
        labelled["GAME_DATE"], half_life_days=180
    ).iloc[:5]
    with pytest.raises(ValueError, match="DATA_NOT_AVAILABLE"):
        XGBoostPropPipeline(["PTS_L5", "PTS_L10"]).fit(labelled, sample_weight=short)
