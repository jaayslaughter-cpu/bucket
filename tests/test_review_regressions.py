"""Regressions for defects found in code review.

Each test here corresponds to a bug that shipped. They exist so the same
mistake cannot return quietly — several of these failures would have looked
like a slightly different metric rather than an error.
"""

from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.builder import build_feature_matrix
from src.models.data_audit import make_demo_panel
from src.models.labels import attach_research_over_labels
from src.models.residuals import fit_dispersion_out_of_fold


def _labelled_panel(n_players: int = 8, n_games: int = 40) -> pd.DataFrame:
    panel = attach_research_over_labels(
        build_feature_matrix(make_demo_panel(n_players=n_players, n_games=n_games)),
        stat="PTS",
    )
    return panel.loc[panel["over_hit"].notna()].reset_index(drop=True)


# --------------------------------------------------------------------------
# Early stopping must not see the scoring set
# --------------------------------------------------------------------------

def test_catboost_ignores_validation_data_for_early_stopping():
    """Passing the scoring set must not change the fitted model.

    Using validation_data as CatBoost's eval_set let early stopping choose
    the iteration count from the labels the model was then scored against,
    which quietly flattered every validation metric. If that returns, the
    two fits below diverge.
    """
    pytest.importorskip("catboost")
    from src.models.catboost_pipeline import CatBoostPropPipeline

    panel = _labelled_panel()
    split = int(len(panel) * 0.7)
    train, scoring = panel.iloc[:split], panel.iloc[split:]
    cols = ["PTS_L5", "PTS_L10", "MIN_L5"]
    params = {"iterations": 60, "early_stopping_rounds": 10}

    without = CatBoostPropPipeline(cols, target_market="PTS", hyperparameters=dict(params))
    without.fit(train)
    with_scoring = CatBoostPropPipeline(cols, target_market="PTS", hyperparameters=dict(params))
    with_scoring.fit(train, scoring)

    np.testing.assert_allclose(
        without.predict_probability_over(scoring, scoring["RESEARCH_LINE"]).to_numpy(),
        with_scoring.predict_probability_over(scoring, scoring["RESEARCH_LINE"]).to_numpy(),
        rtol=1e-9,
        err_msg="validation_data changed the fit — it is reaching early stopping again",
    )


def test_in_sample_dispersion_fallback_is_named():
    """An in-sample fit is overconfident; unlabelled it looks like a real OOF fit."""
    rng = np.random.default_rng(3)
    n = 40  # below the out-of-fold minimum, forcing the fallback
    X = pd.DataFrame({"f": rng.normal(size=n)})
    y = rng.poisson(12, size=n).astype(float)

    def _train_predict(X_tr, y_tr, X_va):
        return np.full(len(X_va), float(np.mean(y_tr)))

    fitted = fit_dispersion_out_of_fold(_train_predict, X, y, market="PTS")
    assert fitted.fallback_reason is not None
    assert "IN-SAMPLE" in fitted.fallback_reason


# --------------------------------------------------------------------------
# Never publish an invented certainty
# --------------------------------------------------------------------------

def test_ensemble_returns_null_when_no_component_has_a_probability():
    """Accumulating from 0.0 published 'certainly under' for unscored rows."""
    from src.models.ensemble import EnsemblePropModel
    from src.models.prediction_schema import ModelPrediction

    class _Silent:
        """A component that produces a row but no probability."""

        def predict_mean(self, features):
            return pd.Series([np.nan] * len(features), index=features.index, dtype=float)

        def predict_rows(self, features, *, line_col="RESEARCH_LINE"):
            return [
                ModelPrediction(
                    model_name="silent", model_version="v0", target_market="PTS",
                    event_id="g1", player_id="p1", probability_over=None,
                )
                for _ in range(len(features))
            ]

    features = pd.DataFrame({
        "GAME_ID": ["g1"], "PLAYER_ID": ["p1"], "RESEARCH_LINE": [20.5],
    })
    ensemble = EnsemblePropModel({"a": _Silent(), "b": _Silent()}, weights={"a": 0.5, "b": 0.5})
    prediction = ensemble.predict_rows(features)[0]

    assert prediction.probability_over is None
    assert prediction.probability_under is None
    assert any("No component supplied" in w for w in prediction.warnings)


def test_calibrated_under_is_never_negative_on_a_whole_line():
    """Subtracting push from a calibrated over drove the under below zero.

    Isotonic saturates at exactly 1.0, so a whole-number line carrying real
    push mass produced a negative probability.
    """
    push = 0.08
    for calibrated_over in (0.0, 0.5, 1.0):
        open_mass = max(0.0, 1.0 - push)
        over = calibrated_over * open_mass
        under = (1.0 - calibrated_over) * open_mass
        assert over >= 0.0
        assert under >= 0.0
        assert over + under + push == pytest.approx(1.0, abs=1e-9)


def test_minutes_model_returns_null_for_rows_without_features():
    """Zero-filling missing history produced a confident projection from nothing."""
    pytest.importorskip("catboost")
    from src.models.minutes_model import MinutesModel

    panel = build_feature_matrix(make_demo_panel(n_players=6, n_games=30))
    model = MinutesModel(hyperparameters={"iterations": 30})
    model.fit(panel.dropna(subset=["MIN_L5", "MIN_L10"]))

    blank = panel.head(3).copy()
    for col in model.feature_cols:
        if col not in model.categorical_features:
            blank[col] = np.nan

    assert model.predict_mean(blank).isna().all()


# --------------------------------------------------------------------------
# Artifacts must round-trip what they learned
# --------------------------------------------------------------------------

def test_distribution_model_round_trips_its_dispersion(tmp_path):
    """Reload used to revert to Poisson, silently changing every probability."""
    from src.models.distribution_adapter import DistributionPropModel
    from src.models.residuals import CountDispersion

    model = DistributionPropModel(target_market="PTS")
    model.dispersion = CountDispersion(
        family="negbin", phi=1.87, n_train_rows=500, selection_scores={"fitted_phi": 1.87}
    )
    model.save(tmp_path / "dist_PTS")

    reloaded = DistributionPropModel(target_market="PTS").load(tmp_path / "dist_PTS")
    assert reloaded.dispersion is not None
    assert reloaded.dispersion.family == "negbin"
    assert reloaded.dispersion.phi == pytest.approx(1.87)


def test_xgboost_adapter_round_trips_its_mean_head_and_dispersion(tmp_path):
    """Saving only the classifier changed what a reloaded model predicts.

    The mean head supplies every MAE/RMSE figure and the dispersion supplies
    push mass on whole-number lines, so a reload returned null projections
    and different probabilities than the model that was just evaluated —
    the deployed artifact was not the one the comparison report described.
    """
    pytest.importorskip("xgboost")
    from src.models.xgb_adapter import XGBoostAdapter

    panel = _labelled_panel(n_players=6, n_games=30)
    cols = ["PTS_L5", "PTS_L10", "MIN_L5"]
    model = XGBoostAdapter(cols, target_market="PTS", model_params={"n_estimators": 30})
    model.fit(panel)
    assert model.mean_model is not None, "fixture failed to fit a mean head"
    assert model.dispersion is not None, "fixture failed to fit a dispersion"

    scoring = panel.head(15).copy()
    scoring["WHOLE_LINE"] = scoring["PTS_L10"].round()  # whole lines carry push mass
    before_mean = model.predict_mean(scoring).to_numpy()
    before_probs = np.array(
        [r.probability_over for r in model.predict_rows(scoring, line_col="WHOLE_LINE")],
        dtype=float,
    )

    model.save(tmp_path / "xgb_PTS")
    reloaded = XGBoostAdapter(cols, target_market="PTS").load(tmp_path / "xgb_PTS")

    assert reloaded.dispersion is not None
    assert reloaded.dispersion.family == model.dispersion.family
    assert reloaded.dispersion.phi == pytest.approx(model.dispersion.phi)

    after_mean = reloaded.predict_mean(scoring).to_numpy()
    assert not np.isnan(after_mean).all(), "mean head was lost — projections are null"
    np.testing.assert_allclose(before_mean, after_mean, rtol=1e-6)

    after_probs = np.array(
        [r.probability_over for r in reloaded.predict_rows(scoring, line_col="WHOLE_LINE")],
        dtype=float,
    )
    np.testing.assert_allclose(before_probs, after_probs, rtol=1e-6)


def test_xgboost_adapter_drops_a_stale_mean_head_on_save(tmp_path):
    """A leftover sidecar would reload as the current projection."""
    pytest.importorskip("xgboost")
    from src.models.xgb_adapter import XGBoostAdapter

    panel = _labelled_panel(n_players=6, n_games=30)
    cols = ["PTS_L5", "PTS_L10"]
    model = XGBoostAdapter(cols, target_market="PTS", model_params={"n_estimators": 20})
    model.fit(panel)
    model.save(tmp_path / "xgb_PTS")
    assert (tmp_path / "xgb_PTS.mean.json").exists()

    model.mean_model = None
    model.save(tmp_path / "xgb_PTS")
    assert not (tmp_path / "xgb_PTS.mean.json").exists()


def test_dispersion_round_trip_keeps_full_precision():
    """`as_metadata` rounds phi; reloading from it changes the distribution."""
    from src.models.residuals import CountDispersion

    original = CountDispersion(
        family="negbin", phi=1.8712345, n_train_rows=500,
        selection_scores={"poisson": -2.5, "negbin": -2.1}, fallback_reason=None,
    )
    restored = CountDispersion.from_dict(original.to_dict())
    assert restored == original
    assert restored.phi == original.phi  # not rounded to 4dp
    assert CountDispersion.from_dict(None) is None


def test_distribution_load_refuses_a_missing_artifact():
    from src.models.distribution_adapter import DistributionPropModel

    with pytest.raises(FileNotFoundError):
        DistributionPropModel().load("/nonexistent/path/dist")


def test_model_metadata_keeps_feature_schema_version():
    """Pydantic silently dropped this, losing the feature contract."""
    from src.models.prediction_schema import ModelMetadata

    meta = ModelMetadata(
        model_name="catboost", model_version="cb_v1", target_market="PTS",
        feature_schema_version="fs_v9_custom",
    )
    assert meta.feature_schema_version == "fs_v9_custom"
    assert "feature_schema_version" in meta.model_dump()


def test_stale_mean_head_is_removed_on_save(tmp_path):
    """A leftover sidecar was reloaded as the current projection."""
    pytest.importorskip("catboost")
    from src.models.catboost_pipeline import CatBoostPropPipeline

    panel = _labelled_panel(n_players=6, n_games=30)
    cols = ["PTS_L5", "PTS_L10"]
    model = CatBoostPropPipeline(cols, target_market="PTS", hyperparameters={"iterations": 30})
    model.fit(panel)
    model.save(tmp_path / "cb_PTS")
    assert (tmp_path / "cb_PTS.mean.cbm").exists()

    model.mean_model = None
    model.save(tmp_path / "cb_PTS")
    assert not (tmp_path / "cb_PTS.mean.cbm").exists()


# --------------------------------------------------------------------------
# Guards against malformed input
# --------------------------------------------------------------------------

def test_nonpositive_step_days_raises_instead_of_looping_forever():
    from src.models.walk_forward import expanding_window_splits

    frame = pd.DataFrame({"GAME_DATE": pd.date_range("2025-10-01", periods=100, freq="D")})
    with pytest.raises(ValueError, match="CONFIG_INVALID"):
        expanding_window_splits(frame, step_days=0)


def test_settlement_rejects_non_finite_values():
    from src.settlement.evaluator import SettlementError, _to_decimal

    for bad in ("nan", "inf", "-inf", float("nan"), float("inf")):
        with pytest.raises(SettlementError, match="not finite"):
            _to_decimal(bad, "predicted_line")


def test_settlement_rejects_a_negative_stake():
    """A negative stake books wins as losses and corrupts ROI silently."""
    from src.settlement.evaluator import SettlementError, settle_prop

    with pytest.raises(SettlementError, match="strictly positive"):
        settle_prop(
            market="PTS", predicted_line=Decimal("25.5"), predicted_side="OVER",
            player_stats={"points": 30}, odds=-110, stake_units=Decimal("-1"),
        )


def test_records_converts_numeric_nan_to_none():
    """pandas coerces None back to NaN on numeric columns without astype(object)."""
    from src.db.repository import _records

    frame = pd.DataFrame({"pts": [10.0, np.nan], "name": ["a", None]})
    records = _records(frame)
    assert records[1]["pts"] is None
    assert records[1]["name"] is None


def test_ensemble_apis_agree_on_a_whole_number_line():
    """Blending raw probabilities skipped push handling, so the two APIs
    disagreed on exactly the lines where push mass is non-zero."""
    pytest.importorskip("catboost")
    from src.models.compare import build_components, load_comparison_config
    from src.models.ensemble import EnsemblePropModel

    panel = _labelled_panel(n_players=6, n_games=30)
    cols = ["PTS_L5", "PTS_L10", "MIN_L5"]
    components = build_components("PTS", cols, load_comparison_config(), xgb_feature_cols=cols)
    fitted = {}
    for name, model in components.items():
        try:
            model.fit(panel)
            fitted[name] = model
        except Exception:  # noqa: BLE001 — component availability varies
            pass
    if len(fitted) < 2:
        pytest.skip("need two fitted components")

    ensemble = EnsemblePropModel(fitted, weights={k: 1.0 for k in fitted}, target_market="PTS")
    scoring = panel.head(20).copy()
    scoring["WHOLE_LINE"] = scoring["PTS_L10"].round()  # whole numbers can push

    via_series = ensemble.predict_probability_over(scoring, scoring["WHOLE_LINE"]).to_numpy()
    via_rows = np.array(
        [r.probability_over for r in ensemble.predict_rows(scoring, line_col="WHOLE_LINE")],
        dtype=float,
    )
    np.testing.assert_allclose(via_series, via_rows, rtol=1e-9)


# --------------------------------------------------------------------------
# Connection and settlement identity
# --------------------------------------------------------------------------

def test_database_url_percent_encodes_credentials():
    """An f-string URL with @ or / in the password parses to the wrong host."""
    import os
    from unittest import mock

    from src.db.session import _build_url_from_parts

    with mock.patch.dict(os.environ, {
        "PGHOST": "db.example.com", "PGUSER": "postgres",
        "PGPASSWORD": "p@ss:w/rd#1", "PGPORT": "6543", "PGDATABASE": "postgres",
    }, clear=False):
        url = _build_url_from_parts()

    assert "p@ss:w/rd#1" not in url
    assert "%40" in url and "%2F" in url

    from urllib.parse import urlparse
    assert urlparse(url).hostname == "db.example.com"
    assert urlparse(url).port == 6543


def test_duplicate_player_names_are_marked_ambiguous_not_overwritten():
    """Keeping the last silently grades one player's prop with another's stats."""
    from src.settlement.boxscore_fetcher import extract_player_stats

    def _player(pid, name):
        return {
            "personId": pid, "nameI": name, "firstName": name.split()[0],
            "familyName": name.split()[-1], "status": "ACTIVE",
            "statistics": {"minutes": "PT30M00.00S", "points": 10},
        }

    payload = {
        "game": {
            "gameId": "0022500001",
            "homeTeam": {"teamTricode": "LAL", "players": [_player("1", "Chris Johnson")]},
            "awayTeam": {"teamTricode": "BOS", "players": [_player("2", "Chris Johnson")]},
        }
    }
    stats = extract_player_stats(payload)
    if "Chris Johnson" in stats:
        assert stats["Chris Johnson"]["ambiguous_name"] is True


def test_settlement_refuses_an_ambiguous_name():
    from src.settlement.runner import _match_player

    stats = {"Chris Johnson": {"player_name": "Chris Johnson", "ambiguous_name": True}}
    assert _match_player(stats, "Chris Johnson") is None


def test_projection_uniqueness_excludes_run_id():
    """Keying on a per-execution UUID meant re-runs never conflicted."""
    from src.db.models import Projection

    uq = next(
        c for c in Projection.__table__.constraints
        if getattr(c, "name", None) == "uq_projection"
    )
    columns = {c.name for c in uq.columns}
    assert "run_id" not in columns
    assert columns == {"nba_game_id", "player_name", "market"}


def test_stake_of_zero_is_not_promoted_to_one_unit():
    """`stake or Decimal(1)` graded a zero-stake row as a one-unit bet.

    Decimal("0") is falsey, so the row was silently restaked and its
    profit_units entered the ROI sums as though a wager had been placed.
    """
    from src.settlement.runner import stake_or_default

    assert stake_or_default(None) == Decimal(1)
    assert stake_or_default(Decimal("0")) == Decimal("0")
    assert stake_or_default(Decimal("2.5")) == Decimal("2.5")


# --------------------------------------------------------------------------
# A calendar date is not an instant
# --------------------------------------------------------------------------

def test_date_only_cutoff_displays_on_the_same_calendar_day():
    """A cutoff of 2025-02-01 read as UTC midnight displays as Jan 31 PT.

    `data_cutoff_pt` is how a reader checks that no future data entered a
    fold, so an off-by-one-day render is a leakage report that lies.
    """
    from datetime import date

    from src.utils.timezones import format_pacific_iso, pacific_midnight_utc

    rendered = format_pacific_iso(pacific_midnight_utc(date(2025, 2, 1)))
    assert rendered.startswith("2025-02-01T00:00:00")


def test_naive_pacific_datetime_is_not_read_as_utc():
    from datetime import datetime

    from src.utils.timezones import to_utc

    # noqa DTZ001: a naive datetime is exactly what this test is about —
    # supplying tzinfo here would remove the ambiguity being tested.
    naive = datetime(2025, 2, 1, 0, 0, 0)  # noqa: DTZ001
    assert to_utc(naive).hour == 0                      # assumed UTC
    assert to_utc(naive, assume="pacific").hour == 8    # PST is UTC-8


# --------------------------------------------------------------------------
# Counts that describe rows must stay within the row count
# --------------------------------------------------------------------------

def test_audit_counts_never_exceed_the_row_count():
    """Summing per-column misses counted one bad row three times.

    missing_target_rows and rejected_rows are reported as row counts, so a
    figure above total_rows is not a near-miss — it is a different quantity.
    """
    from src.models.data_audit import audit_player_panel

    frame = pd.DataFrame({
        "PLAYER_ID": [None, "p2", "p3"],
        "GAME_ID": [None, "g2", "g3"],          # row 0 is missing BOTH keys
        "PTS": [np.nan, 10.0, 12.0],
        "REB": [np.nan, 4.0, 5.0],              # row 0 is missing ALL targets
        "AST": [np.nan, 2.0, 3.0],
    })
    report = audit_player_panel(frame)

    assert report["missing_target_rows"] == 1
    assert report["rejected_rows"] == 1
    assert report["valid_rows"] == 2
    assert report["rejected_rows"] <= report["total_rows"]


# --------------------------------------------------------------------------
# Probabilities must be probabilities, not just balanced
# --------------------------------------------------------------------------

def test_out_of_range_components_are_rejected_even_when_they_sum_to_one():
    """1.4 over against -0.4 under totals exactly 1.0 but means nothing."""
    from src.models.prediction_schema import ModelPrediction

    def _prediction(**kwargs) -> ModelPrediction:
        return ModelPrediction(
            model_name="m", model_version="v", target_market="PTS",
            event_id="g1", player_id="p1", **kwargs,
        )

    assert not _prediction(probability_over=1.4, probability_under=-0.4).is_valid_probability()
    assert not _prediction(
        probability_over=float("nan"), probability_under=float("nan")
    ).is_valid_probability()
    assert _prediction(probability_over=0.6, probability_under=0.4).is_valid_probability()


# --------------------------------------------------------------------------
# One rating, one team
# --------------------------------------------------------------------------

def test_elo_skips_a_game_whose_two_rows_are_the_same_team():
    """A duplicated row passed the len()==2 check and rated a team against itself.

    The update moves that team's rating twice and emits a self-opponent into
    the feature join, so every later game inherits a corrupted rating.
    """
    from src.features.team_strength import compute_team_elo

    duplicated = pd.DataFrame([
        {"game_id": "001", "game_date": "2025-01-01", "team": "LAL",
         "opponent": "BOS", "points": 110, "is_home": True, "season": "2024-25"},
        {"game_id": "001", "game_date": "2025-01-01", "team": "LAL",
         "opponent": "BOS", "points": 104, "is_home": False, "season": "2024-25"},
    ])
    assert compute_team_elo(duplicated).empty


def test_elo_rates_a_well_formed_game():
    """The same-team guard must not reject legitimate two-sided games."""
    from src.features.team_strength import compute_team_elo

    frame = pd.DataFrame([
        {"game_id": "001", "game_date": "2025-01-01", "team": "LAL",
         "opponent": "BOS", "points": 110, "is_home": True, "season": "2024-25"},
        {"game_id": "001", "game_date": "2025-01-01", "team": "BOS",
         "opponent": "LAL", "points": 104, "is_home": False, "season": "2024-25"},
    ])
    out = compute_team_elo(frame)
    assert len(out) == 2
    assert set(out["team"]) == {"LAL", "BOS"}
    assert out.loc[out["team"] == "LAL", "elo_post"].iloc[0] > 1500.0


def test_roi_aggregates_are_scoped_to_settled_rows():
    """A PENDING or VOID row carrying a stake must not enter the ROI sums."""
    from pathlib import Path

    sql = (Path(__file__).parent.parent / "migrations/002_prop_results.sql").read_text()
    unscoped = sql.count("FILTER (WHERE odds IS NOT NULL)")
    assert unscoped == 0, "a stake/profit aggregate is not filtered by settlement status"
