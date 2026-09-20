"""Leakage, dispersion, calibration-date safety, and categorical handling.

These guard the properties that make the comparison honest. If one of
these fails, published metrics are wrong — not merely worse.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.builder import LookaheadError, assert_no_lookahead, build_feature_matrix
from src.models.data_audit import make_demo_panel
from src.models.labels import attach_research_over_labels
from src.models.residuals import (
    CountDispersion,
    fit_count_dispersion,
    over_under_push_from_dispersion,
)

# --------------------------------------------------------------------------
# Rolling-feature leakage
# --------------------------------------------------------------------------

def test_rolling_features_exclude_the_current_game():
    """A player's L2 must never contain the game it is predicting."""
    panel = make_demo_panel(n_players=2, n_games=12)
    feats = build_feature_matrix(panel)

    one = feats[feats["PLAYER_ID"] == feats["PLAYER_ID"].iloc[0]].sort_values("GAME_DATE")
    # The third game's L2 is the mean of games 1 and 2 — not games 2 and 3.
    expected = one["PTS"].iloc[0:2].mean()
    assert one["PTS_L2"].iloc[2] == pytest.approx(expected)


def test_first_game_has_no_prior_history():
    """A debut row cannot have rolling features; NaN is correct, 0.0 is not."""
    feats = build_feature_matrix(make_demo_panel(n_players=2, n_games=6))
    first_rows = feats.groupby("PLAYER_ID").head(1)
    assert first_rows["PTS_L2"].isna().all()
    assert first_rows["LAST_INCLUDED_GAME_DATE"].isna().all()


def test_season_average_is_expanding_not_whole_season():
    """A whole-season mean would leak the future into early-season rows."""
    feats = build_feature_matrix(make_demo_panel(n_players=1, n_games=20))
    one = feats.sort_values("GAME_DATE")
    # Game 5's season mean is over games 1-4 only.
    assert one["PTS_SEASON"].iloc[4] == pytest.approx(one["PTS"].iloc[0:4].mean())
    # It must differ from the full-season mean, or it is not expanding.
    assert one["PTS_SEASON"].iloc[4] != pytest.approx(one["PTS"].mean())


def test_assert_no_lookahead_catches_a_violation():
    """The guard must actually fire — a guard that cannot fail is not a guard."""
    feats = build_feature_matrix(make_demo_panel(n_players=1, n_games=8))
    tampered = feats.copy()
    tampered.loc[tampered.index[3], "LAST_INCLUDED_GAME_DATE"] = tampered.loc[
        tampered.index[3], "GAME_DATE"
    ]
    with pytest.raises(LookaheadError):
        assert_no_lookahead(tampered)


def test_labels_drop_pushes_rather_than_grading_them_as_losses():
    panel = build_feature_matrix(make_demo_panel(n_players=2, n_games=12))
    labelled = attach_research_over_labels(panel, stat="PTS")
    exact = labelled["PTS"] == labelled["RESEARCH_LINE"]
    if exact.any():
        assert labelled.loc[exact, "over_hit"].isna().all()


# --------------------------------------------------------------------------
# Dispersion
# --------------------------------------------------------------------------

def test_dispersion_detects_overdispersion():
    """Negative-binomial data must not be reported as Poisson."""
    rng = np.random.default_rng(7)
    mu = np.full(600, 20.0)
    # Gamma-Poisson mixture => variance well above the mean.
    y = rng.poisson(rng.gamma(shape=4.0, scale=5.0, size=600))
    fitted = fit_count_dispersion(y, mu, market="TEST")
    assert fitted.family == "negbin"
    assert fitted.phi > 1.0


def test_dispersion_falls_back_with_a_named_reason():
    fitted = fit_count_dispersion(np.array([1.0, 2.0]), np.array([1.5, 1.5]))
    assert fitted.family == "poisson"
    assert fitted.fallback_reason is not None


def test_whole_line_push_mass_and_half_line_none():
    d = CountDispersion(family="poisson", phi=1.0, n_train_rows=500, selection_scores={})

    whole = over_under_push_from_dispersion(25.0, 25.0, d)
    assert whole["probability_push"] > 0
    total = whole["probability_over"] + whole["probability_under"] + whole["probability_push"]
    assert total == pytest.approx(1.0, abs=1e-6)

    half = over_under_push_from_dispersion(25.0, 25.5, d)
    assert half["probability_push"] == 0.0
    assert half["probability_over"] + half["probability_under"] == pytest.approx(1.0, abs=1e-6)


def test_wider_dispersion_moves_probability_off_the_mean():
    """More spread must mean less mass concentrated near the projection."""
    tight = CountDispersion(family="poisson", phi=1.0, n_train_rows=500, selection_scores={})
    wide = CountDispersion(family="negbin", phi=3.0, n_train_rows=500, selection_scores={})
    # A line well above the mean is likelier to be cleared under wide spread.
    assert (
        over_under_push_from_dispersion(20.0, 28.5, wide)["probability_over"]
        > over_under_push_from_dispersion(20.0, 28.5, tight)["probability_over"]
    )


def test_invalid_projection_refuses_to_produce_probabilities():
    d = CountDispersion(family="poisson", phi=1.0, n_train_rows=500, selection_scores={})
    for bad in (float("nan"), -1.0):
        res = over_under_push_from_dispersion(bad, 20.5, d)
        assert res["status"] == "DATA_NOT_AVAILABLE"
        assert res["probability_over"] is None


# --------------------------------------------------------------------------
# Calibration date safety
# --------------------------------------------------------------------------

def test_calibrator_is_fitted_strictly_before_the_evaluation_window():
    """A calibrator fitted on its own evaluation rows is worthless."""
    from src.models.compare import fit_calibrator_from_earlier_data, prepare_market_panel

    panel = prepare_market_panel(build_feature_matrix(make_demo_panel()), "PTS")
    panel = panel.loc[panel["over_hit"].notna()].reset_index(drop=True)

    train = panel[panel["GAME_DATE"] <= "2025-01-15"]
    evaluation = panel[panel["GAME_DATE"] > "2025-01-15"]
    if len(train) < 200 or evaluation.empty:
        pytest.skip("demo panel too small for a calibration split")

    from src.models.compare import load_comparison_config
    from src.models.labels import default_feature_cols

    cols = [c for c in default_feature_cols("PTS") if c in panel.columns]
    _cal, info = fit_calibrator_from_earlier_data(
        "distribution", "PTS", cols, cols, load_comparison_config(), train
    )
    if info.get("fit_end_date") is None:
        pytest.skip(f"calibration unavailable: {info.get('reason')}")

    assert pd.Timestamp(info["fit_end_date"]) <= pd.Timestamp("2025-01-15")
    assert pd.Timestamp(info["fit_end_date"]) < evaluation["GAME_DATE"].min()


def test_calibrator_refuses_to_fit_on_too_few_rows():
    from src.models.prob_calibration import ProbabilityCalibrator

    with pytest.raises(ValueError, match="DATA_NOT_AVAILABLE"):
        ProbabilityCalibrator().fit(np.array([1, 0, 1]), np.array([0.6, 0.4, 0.7]))


# --------------------------------------------------------------------------
# CatBoost categorical handling
# --------------------------------------------------------------------------

def test_catboost_handles_unseen_and_missing_categories():
    """A category absent at training time must not crash scoring."""
    pytest.importorskip("catboost")
    from src.models.catboost_pipeline import CatBoostPropPipeline

    panel = attach_research_over_labels(
        build_feature_matrix(make_demo_panel(n_players=6, n_games=30)), stat="PTS"
    )
    panel = panel.loc[panel["over_hit"].notna()].reset_index(drop=True)
    cols = ["PTS_L5", "PTS_L10", "TEAM_ABBREVIATION", "OPPONENT_ABBREVIATION"]

    model = CatBoostPropPipeline(
        cols,
        target_market="PTS",
        categorical_features=["TEAM_ABBREVIATION", "OPPONENT_ABBREVIATION"],
        hyperparameters={"iterations": 30},
    )
    model.fit(panel)

    unseen = panel.head(5).copy()
    unseen["TEAM_ABBREVIATION"] = "ZZZ"          # never seen in training
    unseen["OPPONENT_ABBREVIATION"] = None        # missing entirely

    probs = model.predict_probability_over(unseen, unseen["RESEARCH_LINE"])
    assert len(probs) == 5
    assert probs.between(0, 1).all()


# --------------------------------------------------------------------------
# Odds-free (DFS / pick'em) behaviour
# --------------------------------------------------------------------------

def test_ev_gate_abstains_without_two_way_odds():
    """A pick'em multiplier is not a price and must not yield EV."""
    from src.quant.contracts import MarketContext, market_ev_gate

    pickem = MarketContext(
        game_id="0022500001", status="VALID", payout_multiplier=3.0, is_pickem=True
    )
    verdict = market_ev_gate(pickem)
    assert verdict["status"] == "DATA_NOT_AVAILABLE"
    assert verdict["ev"] is None
    assert "not a two-way price" in verdict["reason"]


def test_pickem_carrying_odds_fields_still_refuses():
    """A pick'em row with two odds attached must not slip past the gate.

    Checking odds before the pick'em flag let exactly this through, because
    the odds branch returned early and the refusal was never reached.
    """
    from src.quant.contracts import MarketContext, market_ev_gate

    verdict = market_ev_gate(
        MarketContext(
            game_id="0022500001",
            status="VALID",
            line=25.5,
            is_pickem=True,
            over_odds_american=-110,
            under_odds_american=-110,
        )
    )
    assert verdict["status"] == "DATA_NOT_AVAILABLE"
    assert "not a two-way price" in verdict["reason"]


def test_gate_requires_a_finite_line():
    """EV is a claim about a probability at a number; without one there is none."""
    from src.quant.contracts import MarketContext, market_ev_gate

    for bad_line in (None, float("nan"), float("inf")):
        verdict = market_ev_gate(
            MarketContext(
                game_id="0022500001", status="VALID", line=bad_line,
                over_odds_american=-110, under_odds_american=-110,
            )
        )
        assert verdict["status"] == "DATA_NOT_AVAILABLE"


def test_ev_gate_devigs_a_real_two_way_market():
    from src.quant.contracts import MarketContext, market_ev_gate

    verdict = market_ev_gate(
        MarketContext(
            game_id="0022500001",
            status="VALID",
            line=25.5,
            over_odds_american=-110,
            under_odds_american=-110,
        )
    )
    assert verdict["status"] == "READY_FOR_EVALUATION"
    # A balanced -110/-110 market is 50/50 once the vig is removed.
    assert verdict["fair_probability_over"] == pytest.approx(0.5, abs=1e-9)
    # -110 implies 110/210 per side; the pair books to 1.047619.
    assert verdict["hold"] == pytest.approx(0.047619, abs=1e-5)


def test_exports_carry_no_odds_columns_when_none_were_supplied():
    """Odds-free research must not emit invented prices."""
    detail = pd.read_csv(Path(__file__).parent.parent / "outputs/demo/predictions_detailed.csv") \
        if (Path(__file__).parent.parent / "outputs/demo/predictions_detailed.csv").exists() else None
    if detail is None:
        pytest.skip("no demo exports generated yet")
    for banned in ("american_odds", "over_odds_american", "under_odds_american", "ev_per_dollar"):
        assert banned not in detail.columns
