"""Tests for the cross-validated XGBoost tree count."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.xgboost_pipeline import (
    DEFAULT_TUNING,
    XGBoostPropPipeline,
    split_xgboost_config,
)


def _panel(n: int = 900, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    noise = rng.normal(size=(n, 4))
    y = (x1 * 0.8 + rng.normal(0, 1.0, n) > 0).astype(int)
    return pd.DataFrame({
        "GAME_DATE": pd.date_range("2024-10-01", periods=n, freq="6h"),
        "f1": x1, "f2": noise[:, 0], "f3": noise[:, 1],
        "f4": noise[:, 2], "f5": noise[:, 3],
        "over_hit": y,
    })


FEATS = ["f1", "f2", "f3", "f4", "f5"]


# --- the config split -------------------------------------------------------


def test_tuning_knobs_never_reach_the_xgboost_constructor():
    """The mean head builds an XGBRegressor from model_params. A knob like
    early_stopping_rounds forwarded there raises at fit time."""
    params, tuning = split_xgboost_config({
        "n_estimators": 400, "max_depth": 5, "early_stopping_rounds": 40,
        "n_estimators_max": 2000, "min_trees": 5, "scale_with_data": True,
    })
    assert params == {"n_estimators": 400, "max_depth": 5}
    assert set(tuning) == {"early_stopping_rounds", "n_estimators_max",
                           "min_trees", "scale_with_data"}
    assert not set(params) & set(DEFAULT_TUNING)


def test_an_unknown_key_goes_to_model_params():
    """So a new XGBoost argument can be set from config without a code change."""
    params, tuning = split_xgboost_config({"gamma": 0.5})
    assert params == {"gamma": 0.5}
    assert tuning == {}


def test_the_shipped_config_splits_cleanly():
    from src.models.compare import load_comparison_config

    block = load_comparison_config().get("xgboost") or {}
    assert block, "config/model_comparison.yaml has no xgboost block"
    params, tuning = split_xgboost_config(block)
    assert "early_stopping_rounds" in tuning
    from xgboost import XGBRegressor  # must accept every model_param

    XGBRegressor(**{k: v for k, v in params.items()
                    if k not in {"objective", "eval_metric"}})


# --- the learned count ------------------------------------------------------


def test_the_tree_count_is_learned_not_the_configured_fallback():
    pipe = XGBoostPropPipeline(FEATS, model_params={"n_estimators": 400})
    pipe.fit(_panel())
    assert pipe.n_estimators_source_ == "cross_validated_early_stopping"
    assert pipe.learned_n_estimators_ is not None
    assert pipe.learned_n_estimators_ < 400
    assert pipe.model.get_booster().num_boosted_rounds() == pipe.learned_n_estimators_


def test_disabling_early_stopping_falls_back_to_the_configured_count():
    pipe = XGBoostPropPipeline(
        FEATS, model_params={"n_estimators": 25},
        tuning={"early_stopping_rounds": 0},
    )
    pipe.fit(_panel())
    assert pipe.n_estimators_source_ == "configured"
    assert pipe.learned_n_estimators_ is None
    assert pipe.cv_best_iterations_ == []
    assert pipe.model.get_booster().num_boosted_rounds() == 25


def test_too_few_rows_to_cross_validate_falls_back_and_says_so():
    pipe = XGBoostPropPipeline(FEATS, model_params={"n_estimators": 30})
    pipe.fit(_panel(n=60))
    assert pipe.n_estimators_source_ == "configured"
    assert pipe.model.get_booster().num_boosted_rounds() == 30


def test_the_median_is_used_not_the_minimum():
    """A fold that stops after two trees is an unlucky slice. Taking the
    minimum underfit measurably: Brier 0.22611 against 0.21844 on the same
    panel."""
    pipe = XGBoostPropPipeline(FEATS, tuning={"scale_with_data": False})
    learned = pipe._learn_n_estimators([2, 25, 33, 49, 81], [100] * 5, 100)
    assert learned == 33


def test_the_count_is_scaled_for_the_larger_final_fit():
    """Fold models train on less data than the final model, and the best tree
    count grows with data."""
    pipe = XGBoostPropPipeline(FEATS, tuning={"scale_with_data": True})
    assert pipe._learn_n_estimators([40, 40, 40], [100, 100, 100], 200) == 80
    off = XGBoostPropPipeline(FEATS, tuning={"scale_with_data": False})
    assert off._learn_n_estimators([40, 40, 40], [100, 100, 100], 200) == 40


def test_the_learned_count_is_clamped():
    pipe = XGBoostPropPipeline(
        FEATS, tuning={"min_trees": 5, "n_estimators_max": 50, "scale_with_data": False}
    )
    assert pipe._learn_n_estimators([1, 1, 1], [10] * 3, 10) == 5
    assert pipe._learn_n_estimators([900, 900, 900], [10] * 3, 10) == 50


def test_no_folds_means_no_learned_count():
    pipe = XGBoostPropPipeline(FEATS)
    assert pipe._learn_n_estimators([], [], 500) is None


# --- honest metadata --------------------------------------------------------


def test_metadata_reports_the_trees_the_model_actually_grew():
    """Reporting the configured 400 for a model that grew 48 is worse than
    reporting nothing."""
    pipe = XGBoostPropPipeline(FEATS, model_params={"n_estimators": 400})
    pipe.fit(_panel())
    eff = pipe.effective_params()
    assert eff["n_estimators"] == pipe.learned_n_estimators_ != 400
    assert eff["n_estimators_source"] == "cross_validated_early_stopping"
    assert eff["cv_best_iterations"] == pipe.cv_best_iterations_
    # The configured fallback is left intact for reference.
    assert pipe.model_params["n_estimators"] == 400


def test_the_adapter_exports_the_learned_count():
    from src.models.xgb_adapter import XGBoostAdapter

    panel = _panel()
    panel["PTS"] = panel["f1"] * 3 + 20
    adapter = XGBoostAdapter(FEATS, target_market="PTS",
                             model_params={"n_estimators": 400})
    adapter.fit(panel, None)
    hp = adapter.get_model_metadata().hyperparameters
    assert hp["n_estimators"] == adapter._pipe.learned_n_estimators_
    assert hp["n_estimators"] != 400
    assert hp["n_estimators_source"] == "cross_validated_early_stopping"


def test_build_components_passes_the_config_through():
    from src.models.compare import build_components, load_comparison_config

    cfg = load_comparison_config()
    comps = build_components("PTS", FEATS, cfg, include_catboost=False)
    pipe = comps["xgboost"]._pipe
    assert pipe.tuning["early_stopping_rounds"] == (
        cfg["xgboost"]["early_stopping_rounds"]
    )
    assert pipe.model_params["max_depth"] == cfg["xgboost"]["max_depth"]
    assert "early_stopping_rounds" not in pipe.model_params


def test_the_mean_head_still_fits_with_the_config_applied():
    """Regression: the mean head reuses model_params for an XGBRegressor,
    which rejects the tuning knobs."""
    from src.models.compare import load_comparison_config
    from src.models.xgb_adapter import XGBoostAdapter
    from src.models.xgboost_pipeline import split_xgboost_config

    params, tuning = split_xgboost_config(load_comparison_config()["xgboost"])
    panel = _panel()
    panel["PTS"] = panel["f1"] * 3 + 20
    adapter = XGBoostAdapter(FEATS, target_market="PTS",
                             model_params=params, tuning=tuning)
    adapter.fit(panel, None)
    assert adapter.mean_model is not None
    assert np.isfinite(adapter.mean_model.predict(panel[FEATS])).all()


@pytest.mark.parametrize("esr", [10, 40])
def test_fold_searches_stay_inside_the_training_rows(esr):
    """The eval set is each fold's own later slice, so the count is chosen
    without ever seeing the window the model is scored on."""
    pipe = XGBoostPropPipeline(FEATS, tuning={"early_stopping_rounds": esr})
    train = _panel(n=900)
    pipe.fit(train)
    assert pipe.cv_best_iterations_
    # Every fold's best iteration is bounded by the search ceiling.
    assert all(1 <= b <= pipe.tuning["n_estimators_max"] for b in pipe.cv_best_iterations_)
