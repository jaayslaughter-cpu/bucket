"""
The middle of the recency chain was missing.

src/models/recency.py computed exponential recency weights. Both
xgboost_pipeline.py and catboost_pipeline.py accepted a ``sample_weight`` and
aligned it by index. Nothing in the repository ever passed one, so a game from
2018 and a game from last week carried identical influence in every fit, and
recency.py had no production caller at all.

These tests pin the connection, the leakage property that makes it safe, and the
fact that it stays OFF until measured.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.compare import (
    _accepts_sample_weight,
    load_comparison_config,
    recency_sample_weights,
)


def _train(n: int = 60) -> pd.DataFrame:
    return pd.DataFrame({
        "GAME_DATE": pd.date_range("2024-11-01", periods=n, freq="D"),
        "PTS": [10.0] * n,
        "over_hit": [i % 2 for i in range(n)],
    })


def test_weighting_is_off_unless_the_config_turns_it_on():
    """Weighting discards information, so it is opt-in like the blowout layer."""
    weights, report = recency_sample_weights(_train(), {}, "PTS")
    assert weights is None and report is None

    off = recency_sample_weights(_train(), {"recency": {"enabled": False}}, "PTS")
    assert off == (None, None)


def test_the_shipped_config_has_it_off():
    """It has not been A/B-ed yet. If someone enables it by default, this fails
    and they have to say so in a diff."""
    cfg = load_comparison_config()
    assert (cfg.get("recency") or {}).get("enabled") is False


def test_enabling_it_produces_weights_and_reports_what_they_cost():
    """The Kish effective sample size is the number that says how much
    information the weighting threw away. A run without it is indistinguishable
    from an unweighted one by its metrics alone."""
    train = _train()
    weights, report = recency_sample_weights(
        train, {"recency": {"enabled": True, "half_life_days": 30}}, "PTS"
    )
    assert weights is not None and report is not None
    assert len(weights) == len(train)
    assert report["market"] == "PTS"
    assert 0 < report["effective_sample_size"] < len(train), report
    assert report["effective_fraction"] < 1.0
    assert report["weight_ratio"] > 1.0


def test_the_newest_TRAINING_row_anchors_the_weights_not_the_validation_end():
    """The leakage property, stated as a measurement.

    No ``as_of`` is passed, so exponential_recency_weights anchors on the newest
    date in the TRAINING frame. Handing it a later date would leak the split
    boundary into the fit. Checked by the shape of the result: the last training
    row carries the largest weight and the first the smallest, and the ratio
    matches the half-life rather than some later reference.
    """
    train = _train(n=61)  # exactly 60 days from first row to last
    weights, _ = recency_sample_weights(
        train, {"recency": {"enabled": True, "half_life_days": 30}}, "PTS"
    )
    assert weights is not None
    assert weights.iloc[-1] == pytest.approx(weights.max())
    assert weights.iloc[0] == pytest.approx(weights.min())
    # 60 days at a 30-day half-life is two halvings: the oldest row is 1/4 the
    # newest. Anchoring on anything LATER than the training end would compress
    # this ratio, so the number is the leakage check.
    assert weights.iloc[-1] / weights.iloc[0] == pytest.approx(4.0, rel=1e-6)


def test_a_frame_with_no_dates_fits_unweighted_rather_than_inventing_an_order():
    train = _train().drop(columns=["GAME_DATE"])
    weights, report = recency_sample_weights(
        train, {"recency": {"enabled": True}}, "PTS"
    )
    assert weights is None and report is None


def test_an_impossible_half_life_abstains_instead_of_fitting_on_one_game():
    """recency.py refuses a half-life below 7 days. compare must turn that
    refusal into an unweighted fit, not a crashed run."""
    weights, report = recency_sample_weights(
        _train(), {"recency": {"enabled": True, "half_life_days": 0.5}}, "PTS"
    )
    assert weights is None and report is None


# --- the pass-through that did not exist ----------------------------------


def test_the_components_that_accept_weights_are_detected_and_the_others_are_not():
    """Inspected rather than hardcoded: DistributionPropModel estimates a
    dispersion and takes no weights, and a component added later must not start
    raising TypeError at the fit site."""
    from src.models.catboost_pipeline import CatBoostPropPipeline
    from src.models.distribution_adapter import DistributionPropModel
    from src.models.xgb_adapter import XGBoostAdapter

    assert _accepts_sample_weight(CatBoostPropPipeline(["PTS_L10"]).fit)
    assert _accepts_sample_weight(XGBoostAdapter(["PTS_L10"]).fit)
    assert not _accepts_sample_weight(DistributionPropModel(target_market="PTS").fit)
    assert not _accepts_sample_weight(object())


def test_the_xgboost_adapter_forwards_the_weights_to_its_pipeline(monkeypatch):
    """The adapter is where the chain was broken: the pipeline under it has
    accepted sample_weight since it was written, and the adapter had no
    parameter to pass one through."""
    from src.models.xgb_adapter import XGBoostAdapter

    adapter = XGBoostAdapter(["PTS_L10"], target_market="PTS")
    seen: dict[str, object] = {}

    def _spy(train_data, target_col="over_hit", sample_weight=None):
        seen["sample_weight"] = sample_weight
        return adapter._pipe

    monkeypatch.setattr(adapter._pipe, "fit", _spy)
    monkeypatch.setattr(adapter, "_fit_mean_head", lambda df: None)
    monkeypatch.setattr(adapter, "_fit_out_of_fold", lambda df: None)

    train = _train()
    train["PTS_L10"] = 12.0
    weights = pd.Series(1.5, index=train.index)
    adapter.fit(train, None, sample_weight=weights)

    assert seen["sample_weight"] is weights, "the adapter swallowed the weights"


# --- what the A/B rounds decided -----------------------------------------


def test_only_measured_markets_carry_form_columns():
    """_FORM_BY_MARKET is populated from measurement, not from the correlation
    screen that preceded it.

    AST is the reason this test exists: its entry was shipped on the screen
    alone, and when the A/B ran, every Brier delta sat at or below its own fold
    spread and line_aware got worse. A column can be genuinely new -- r < 0.45
    against everything the market already reads -- and still not help. Any market
    added back here needs its numbers in the comment beside it.
    """
    from src.models.labels import _FORM_BY_MARKET, default_feature_cols

    measured_winners = {"PTS", "REB"}
    for market, cols in _FORM_BY_MARKET.items():
        if market in measured_winners:
            assert cols, f"{market} won its A/B and should carry form columns"
        else:
            assert cols == (), (
                f"{market} has no A/B result supporting these columns: {cols}. "
                "Measure it with scripts/feature_ab.py --layer form before "
                "wiring it."
            )

    # And the decision reaches the feature list the models actually use.
    assert "AST_HOT_Z" not in default_feature_cols("AST")
    assert "PTS_HOT_Z" in default_feature_cols("PTS")
    assert "REB_HOT_Z" in default_feature_cols("REB")


def test_the_redundant_families_stay_out_of_every_feature_list():
    """halflife and usage_volume were measured with --wire-under-test and made
    Brier worse on every fold. Nothing should re-list them without new numbers.
    """
    from src.models.labels import default_feature_cols

    banned = {
        "PTS_HL", "PTS_HL_SHRINK", "PTS_L2_HL", "MIN_HL", "MIN_HL_SHRINK",
        "REB_HL", "AST_HL", "USAGE_PROXY_L10",
        "SHOT_VOLUME_L5", "SHOT_VOLUME_L10", "FGA_L5", "FGA_L10",
        "OPP_PTS_ALLOWED_L10", "OPP_REB_ALLOWED_L10", "OPP_AST_ALLOWED_L10",
    }
    for market in ("PTS", "REB", "AST", "FG3M", "STL", "BLK", "PRA"):
        listed = set(default_feature_cols(market)) & banned
        assert not listed, f"{market} re-lists measured-redundant column(s) {listed}"
