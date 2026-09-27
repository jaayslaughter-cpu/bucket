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


def test_no_reference_date_is_passed_so_the_anchor_is_the_training_window(
    monkeypatch,
):
    """The leakage property, checked at the call rather than inferred from the
    weights.

    AN EARLIER VERSION OF THIS TEST PROVED NOTHING. It asserted that 60 days at a
    30-day half-life gives the oldest row exactly 1/4 the newest, and called that
    ratio the leakage check. It is not: the weights are normalised to mean 1, so a
    later ``as_of`` multiplies every weight by the same constant and the
    normalisation divides it straight back out. Measured — with the reference 45
    days past the training end, the returned weights are IDENTICAL, element for
    element. Ordering and the 4:1 ratio are invariant to the very thing the test
    claimed to detect.

    What actually protects against the leak is that recency_sample_weights passes
    no ``as_of`` at all, leaving exponential_recency_weights to anchor on the
    newest date in the frame it was handed — the training window's own end. So
    that is what this asserts, by recording the call.
    """
    # Patched on the recency MODULE, not on compare: recency_sample_weights
    # imports the function locally on each call, so the name is resolved from
    # src.models.recency at call time and a patch on compare's namespace would
    # never be seen. A spy installed in the wrong place is a test that passes
    # without observing anything.
    import src.models.recency as recency_module

    # EVERY call is recorded, not the last one. recency_weight_report calls the
    # same function a second time with an explicit as_of=None, so a spy that
    # overwrote a single slot reported that harmless call and never saw the one
    # compare makes — the first version of this test passed even when compare was
    # mutated to pass as_of=train.max(). Collect them all.
    calls: list[dict[str, object]] = []
    real = recency_module.exponential_recency_weights

    def _spy(dates, **kwargs):
        calls.append({
            "kwargs": dict(kwargs),
            "max_date": pd.to_datetime(pd.Series(dates)).max(),
        })
        return real(dates, **kwargs)

    monkeypatch.setattr(recency_module, "exponential_recency_weights", _spy)

    train = _train(n=61)
    weights, _ = recency_sample_weights(
        train, {"recency": {"enabled": True, "half_life_days": 30}}, "PTS"
    )
    assert weights is not None

    assert calls, "the weighting function was never called"
    for i, call in enumerate(calls):
        supplied = call["kwargs"].get("as_of")
        assert supplied is None, (
            f"call {i} supplied a reference date ({supplied}); the anchor must "
            "come from the training frame, not from the caller"
        )
        # And every frame handed over ends where the training window ends, so the
        # implicit anchor cannot see past it.
        assert call["max_date"] == train["GAME_DATE"].max()


def test_the_guard_in_recency_catches_an_EARLIER_reference_not_a_later_one():
    """Which direction recency.py actually guards, read from the code rather than
    assumed.

    ``exponential_recency_weights`` raises when ``as_of`` is EARLIER than the
    newest row, because that produces weights above 1 and means the reference came
    from outside the window. A LATER reference is NOT refused — and does not need
    to be, since mean-normalisation cancels it (the test below measures that).
    I first wrote this test asserting the opposite direction and it failed, which
    is the only reason the claim did not end up in a docstring.

    So the leakage protection is not a guard against reaching forward: it is that
    recency_sample_weights passes no reference at all.
    """
    from src.models.recency import RecencyWeightError, exponential_recency_weights

    dates = _train(n=61)["GAME_DATE"]
    with pytest.raises(RecencyWeightError, match="leaks the split boundary"):
        exponential_recency_weights(
            dates, half_life_days=30, as_of=dates.max() - pd.Timedelta(days=5)
        )

    # Later is accepted, and is a no-op on the returned weights.
    anchored = exponential_recency_weights(dates, half_life_days=30)
    later = exponential_recency_weights(
        dates, half_life_days=30, as_of=dates.max() + pd.Timedelta(days=45)
    )
    assert list(anchored.round(12)) == list(later.round(12))


def test_normalisation_is_why_the_old_ratio_assertion_was_vacuous():
    """Pinned so the mistake is not repeated: a later anchor changes nothing
    about the returned weights, because mean-normalisation cancels it."""
    from src.models.recency import exponential_recency_weights

    dates = _train(n=61)["GAME_DATE"]
    anchored = exponential_recency_weights(dates, half_life_days=30)
    # Reaching past the end is refused, so compare two windows that differ only
    # in where their own last row falls: the SHAPE is identical either way.
    shifted = exponential_recency_weights(
        dates + pd.Timedelta(days=45), half_life_days=30
    )
    assert list(anchored.round(9)) == list(shifted.round(9))
    assert anchored.iloc[-1] / anchored.iloc[0] == pytest.approx(4.0, rel=1e-6)


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
