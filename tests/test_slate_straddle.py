"""No fold boundary may split a calendar slate.

WHY THIS EXISTS: a positional ``TimeSeriesSplit`` cuts by row number. An NBA
slate is ~150 rows (oof.py records min 20, median 153, max 318 on the real
panel), so every boundary lands mid-slate: one calendar day ends up with some
rows training and others being predicted. A fold can then train on one game's
outcome and predict another game from the same night — information nobody has
before tip.

``oof._fold_indices`` already splits on distinct calendar dates. These tests
extend the same guarantee to the two other places that fold chronologically:
the XGBoost early-stopping CV (which selects the tree count) and the
out-of-fold dispersion fit (whose phi drives P(over)/P(under)/P(push)).

Measured on the synthetic 40-day panel below: positional folds straddle 5 days
and affect 424 validation rows; date-grouped folds straddle none.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.oof import chronological_fold_indices


def slate_panel(n_days: int = 40, seed: int = 0) -> pd.DataFrame:
    """Rows carrying realistic within-night tip-off spread, sorted by date."""
    rng = np.random.default_rng(seed)
    stamps: list[pd.Timestamp] = []
    for day in pd.date_range("2025-01-01", periods=n_days, freq="D"):
        for _ in range(int(rng.integers(20, 318))):
            stamps.append(day + pd.Timedelta(hours=int(rng.integers(19, 23))))
    frame = pd.DataFrame({"GAME_DATE": stamps})
    return frame.sort_values("GAME_DATE").reset_index(drop=True)


def straddle_report(
    folds: list[tuple[np.ndarray, np.ndarray]], days: pd.Series
) -> tuple[int, int]:
    """(distinct days straddled, validation rows on a straddled day)."""
    straddled: set[pd.Timestamp] = set()
    affected = 0
    for train_idx, valid_idx in folds:
        both = set(days.iloc[train_idx]) & set(days.iloc[valid_idx])
        straddled |= both
        affected += int(days.iloc[valid_idx].isin(both).sum())
    return len(straddled), affected


def test_the_positional_split_really_does_straddle_a_slate():
    """The hazard is real, not hypothetical — this is the control."""
    from sklearn.model_selection import TimeSeriesSplit

    panel = slate_panel()
    days = panel["GAME_DATE"].dt.normalize()
    folds = list(TimeSeriesSplit(n_splits=5).split(np.arange(len(panel))))
    n_days_straddled, n_rows = straddle_report(folds, days)
    assert n_days_straddled > 0, "control failed: nothing to fix"
    assert n_rows > 0


def test_date_grouped_folds_straddle_nothing():
    panel = slate_panel()
    days = panel["GAME_DATE"].dt.normalize()
    folds = chronological_fold_indices(
        len(panel), splits=5, market="PTS", dates=panel["GAME_DATE"]
    )
    assert straddle_report(folds, days) == (0, 0)


def test_every_row_is_accounted_for_and_train_precedes_validation():
    panel = slate_panel()
    days = panel["GAME_DATE"].dt.normalize()
    folds = chronological_fold_indices(
        len(panel), splits=5, market="PTS", dates=panel["GAME_DATE"]
    )
    assert folds, "no folds produced"
    for train_idx, valid_idx in folds:
        assert len(train_idx) and len(valid_idx)
        assert not set(train_idx) & set(valid_idx)
        # Chronology holds at CALENDAR-DAY granularity, which is the claim
        # that matters: every training day is strictly before every
        # validation day.
        assert max(days.iloc[train_idx]) < min(days.iloc[valid_idx])


def test_no_dates_falls_back_positionally_rather_than_failing():
    """A caller with no dates still gets folds, and is warned elsewhere."""
    folds = chronological_fold_indices(600, splits=3, market="PTS", dates=None)
    assert len(folds) == 3
    for train_idx, valid_idx in folds:
        assert len(train_idx) and len(valid_idx)


def test_too_few_dates_to_group_falls_back():
    two_days = pd.Series(pd.to_datetime(["2025-01-01"] * 50 + ["2025-01-02"] * 50))
    folds = chronological_fold_indices(100, splits=5, market="PTS", dates=two_days)
    assert len(folds) == 5   # positional fallback, not an exception


def test_xgboost_early_stopping_cv_groups_by_date():
    """Site A: the tree-count CV must not tune on same-day rows."""
    import inspect

    from src.models import xgboost_pipeline

    source = inspect.getsource(xgboost_pipeline.XGBoostPropPipeline.fit)
    assert "chronological_fold_indices" in source, (
        "the early-stopping CV still folds positionally, so the tree count is "
        "selected against rows from the same slate it trains on"
    )


def test_dispersion_oof_groups_by_date():
    """Site B: phi drives all three probability legs."""
    import inspect

    from src.models import residuals

    source = inspect.getsource(residuals.fit_dispersion_out_of_fold)
    assert "chronological_fold_indices" in source, (
        "the dispersion fit still folds positionally"
    )


def _dated_panel(n_days: int = 24, per_day: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    for day in pd.date_range("2025-01-01", periods=n_days, freq="D"):
        for _ in range(per_day):
            rows.append({
                "GAME_DATE": day + pd.Timedelta(hours=int(rng.integers(19, 23))),
                "PLAYER_ID": f"p{rng.integers(0, 40)}",
                "PTS": float(max(0.0, rng.normal(22, 6))),
                "PTS_L10": float(rng.normal(22, 5)),
                "MIN_L10": float(rng.normal(31, 4)),
                "RESEARCH_LINE": 22.5,
                "over_hit": int(rng.random() > 0.5),
            })
    return pd.DataFrame(rows).sort_values("GAME_DATE").reset_index(drop=True)


def _spy_on_fold_helper(monkeypatch, *modules):
    """Record every call, patching each module that bound the name itself.

    A local ``from ... import`` binds the function into the importing module,
    so patching only ``oof`` would leave those bindings untouched and the spy
    would record nothing while the test still passed. Only modules that
    actually hold the attribute are patched -- the adapters reach the folds
    through ``fit_dispersion_out_of_fold`` and never bind it -- and at least
    one must, or the spy is watching nothing.
    """
    from src.models import oof as oof_mod

    calls: list[dict] = []
    real = oof_mod.chronological_fold_indices

    def spy(n, *, splits, market="", dates=None):
        calls.append({"market": market, "dates_is_none": dates is None})
        return real(n, splits=splits, market=market, dates=dates)

    patched = 0
    for module in (oof_mod, *modules):
        if hasattr(module, "chronological_fold_indices"):
            monkeypatch.setattr(module, "chronological_fold_indices", spy)
            patched += 1
    assert patched, "spy patched nothing, so an empty call list would prove nothing"
    return calls


def test_xgb_adapter_hands_real_dates_to_the_dispersion_fit(monkeypatch):
    """The ternary in the caller passes None when GAME_DATE is absent.

    Reading the source cannot tell whether the frame at that point still
    carries GAME_DATE, so this drives the real adapter and asserts on what
    actually arrived.
    """
    from src.models import residuals, xgb_adapter  # noqa: F401

    calls = _spy_on_fold_helper(monkeypatch, residuals)
    model = xgb_adapter.XGBoostAdapter(
        ["PTS_L10", "MIN_L10"], target_market="PTS"
    )
    model.fit(_dated_panel())

    assert calls, "no chronological folds were built at all"
    offenders = [c for c in calls if c["dates_is_none"]]
    assert not offenders, f"dates=None reached the fold helper: {offenders}"


def test_catboost_pipeline_hands_real_dates_to_the_dispersion_fit(monkeypatch):
    from src.models import catboost_pipeline, residuals

    calls = _spy_on_fold_helper(monkeypatch, residuals)
    model = catboost_pipeline.CatBoostPropPipeline(
        ["PTS_L10", "MIN_L10"], target_market="PTS"
    )
    model.fit(_dated_panel())

    assert calls, "no chronological folds were built at all"
    offenders = [c for c in calls if c["dates_is_none"]]
    assert not offenders, f"dates=None reached the fold helper: {offenders}"
