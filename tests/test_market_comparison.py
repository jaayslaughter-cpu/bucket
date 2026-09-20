"""Model-vs-line comparison and confidence bucketing.

These assert the properties that make the comparison honest: that a better
projection actually reports as better, that pushes are excluded rather than
graded as losses, and that a research stand-in line is never labelled a
market line.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.market_comparison import (
    MIN_BUCKET_ROWS,
    confidence_is_informative,
    edge_bucket_report,
    model_vs_line_report,
)


def _predictions(
    n: int = 200,
    *,
    projection_error: float = 1.0,
    line_error: float = 3.0,
    line_type: str = "research_l10",
    seed: int = 7,
) -> pd.DataFrame:
    """Actuals with a projection and a line at controlled distances."""
    rng = np.random.default_rng(seed)
    actual = rng.integers(8, 34, size=n).astype(float)
    return pd.DataFrame({
        "target_market": "PTS",
        "model_name": "catboost",
        "line_type": line_type,
        "actual_stat_value": actual,
        "prediction_mean": actual + rng.normal(0, projection_error, n),
        "prop_line": actual + rng.normal(0, line_error, n),
        "probability_over_raw": rng.uniform(0.2, 0.8, n),
    })


# --------------------------------------------------------------------------
# Model vs line
# --------------------------------------------------------------------------

def test_better_projection_reports_as_beating_the_line():
    report = model_vs_line_report(_predictions(projection_error=0.5, line_error=4.0))
    assert len(report) == 1
    assert report.iloc[0]["model_beats_line_by"] > 0
    assert report.iloc[0]["model_mae"] < report.iloc[0]["line_mae"]


def test_worse_projection_reports_as_losing_to_the_line():
    """The metric must be able to deliver bad news."""
    report = model_vs_line_report(_predictions(projection_error=5.0, line_error=0.5))
    assert report.iloc[0]["model_beats_line_by"] < 0


def test_research_line_is_not_labelled_a_market_line():
    report = model_vs_line_report(_predictions(line_type="research_l10"))
    assert not report.iloc[0]["is_market_line"]
    assert report.iloc[0]["line_type"] == "research_l10"


def test_real_line_is_labelled_a_market_line():
    report = model_vs_line_report(_predictions(line_type="draftkings_pts"))
    assert bool(report.iloc[0]["is_market_line"])


def test_mixed_line_types_are_flagged_not_averaged():
    """Averaging a research stand-in with a real line produces a meaningless number."""
    research = _predictions(n=50, line_type="research_l10")
    market = _predictions(n=50, line_type="draftkings_pts", seed=9)
    report = model_vs_line_report(pd.concat([research, market], ignore_index=True))
    assert report.iloc[0]["line_type"] == "MIXED"
    assert not report.iloc[0]["is_market_line"]


def test_rows_without_an_outcome_are_excluded():
    frame = _predictions(n=100)
    frame.loc[:49, "actual_stat_value"] = np.nan
    report = model_vs_line_report(frame)
    assert report.iloc[0]["n_predictions"] == 50


def test_no_gradable_rows_returns_empty_not_zero():
    """An empty report is honest; a zero would read as 'the model tied the line'."""
    frame = _predictions(n=20)
    frame["actual_stat_value"] = np.nan
    assert model_vs_line_report(frame).empty


def test_grouping_splits_markets_and_models():
    a = _predictions(n=60)
    b = _predictions(n=60, seed=3)
    b["model_name"] = "xgboost"
    c = _predictions(n=60, seed=4)
    c["target_market"] = "REB"
    report = model_vs_line_report(pd.concat([a, b, c], ignore_index=True))
    assert len(report) == 3


def test_missing_columns_raise():
    with pytest.raises(ValueError, match="DATA_NOT_AVAILABLE"):
        model_vs_line_report(pd.DataFrame({"target_market": ["PTS"]}))


# --------------------------------------------------------------------------
# Confidence / edge buckets
# --------------------------------------------------------------------------

def test_pushes_are_excluded_from_hit_rate_not_counted_as_losses():
    frame = _predictions(n=120)
    # Force whole-number pushes on a slice: actual exactly equals the line.
    frame.loc[:39, "prop_line"] = frame.loc[:39, "actual_stat_value"]
    report = edge_bucket_report(frame)
    assert report["n_pushes"].sum() == 40
    assert report["n_graded"].sum() == len(frame) - 40


def test_hit_rate_tracks_a_genuinely_informative_model():
    """A model that is right when confident must report as informative."""
    rng = np.random.default_rng(11)
    n = 600
    actual = rng.integers(10, 30, size=n).astype(float)
    line = actual + rng.choice([-4.0, 4.0], size=n)  # never a push
    went_over = actual > line
    # Confidence is honest: strong leans are usually correct, weak ones coin flips.
    strong = rng.random(n) < 0.5
    correct = np.where(strong, rng.random(n) < 0.85, rng.random(n) < 0.5)
    p_over = np.where(
        went_over == correct,
        np.where(strong, 0.85, 0.53),
        np.where(strong, 0.15, 0.47),
    )
    frame = pd.DataFrame({
        "target_market": "PTS", "model_name": "catboost", "line_type": "research_l10",
        "actual_stat_value": actual, "prop_line": line, "probability_over_raw": p_over,
    })
    buckets = edge_bucket_report(frame)
    verdict = confidence_is_informative(
        buckets, group={"target_market": "PTS", "model_name": "catboost"}
    )
    assert verdict["verdict"] == "INFORMATIVE"
    assert verdict["hit_rate_spread"] > 0


def test_flat_confidence_is_reported_as_uninformative():
    """A model whose confidence means nothing must not pass silently."""
    rng = np.random.default_rng(5)
    n = 600
    actual = rng.integers(10, 30, size=n).astype(float)
    line = actual + rng.choice([-4.0, 4.0], size=n)
    # Probabilities unrelated to the outcome: confidence carries no signal.
    frame = pd.DataFrame({
        "target_market": "PTS", "model_name": "noise", "line_type": "research_l10",
        "actual_stat_value": actual, "prop_line": line,
        "probability_over_raw": rng.uniform(0.05, 0.95, n),
    })
    buckets = edge_bucket_report(frame)
    verdict = confidence_is_informative(
        buckets, group={"target_market": "PTS", "model_name": "noise"}
    )
    assert verdict["verdict"] in {"FLAT_OR_INVERTED", "INSUFFICIENT_DATA"}


def test_small_buckets_are_flagged_below_minimum():
    report = edge_bucket_report(_predictions(n=40))
    assert report["below_min_sample"].any()


def test_verdict_refuses_on_too_few_usable_buckets():
    verdict = confidence_is_informative(
        edge_bucket_report(_predictions(n=35)),
        group={"target_market": "PTS", "model_name": "catboost"},
    )
    assert verdict["verdict"] == "INSUFFICIENT_DATA"
    assert str(MIN_BUCKET_ROWS) in verdict["detail"]


def test_measure_is_confidence_without_market_probabilities():
    report = edge_bucket_report(_predictions(n=120))
    assert set(report["measure"]) == {"confidence"}


def test_measure_becomes_edge_when_market_probabilities_supplied():
    """With a market price the bucket is real edge, not bare confidence."""
    frame = _predictions(n=200)
    frame["market_probability_over"] = 0.5
    report = edge_bucket_report(frame, market_probability_col="market_probability_over")
    assert set(report["measure"]) == {"edge"}


def test_edge_against_an_identical_market_is_zero():
    """If the market agrees exactly, there is no edge by construction."""
    frame = _predictions(n=200)
    frame["market_probability_over"] = frame["probability_over_raw"]
    report = edge_bucket_report(frame, market_probability_col="market_probability_over")
    assert report["mean_measure"].abs().max() == pytest.approx(0.0, abs=1e-9)


def test_missing_market_column_raises_rather_than_falling_back():
    with pytest.raises(ValueError, match="DATA_NOT_AVAILABLE"):
        edge_bucket_report(_predictions(n=50), market_probability_col="not_a_column")
