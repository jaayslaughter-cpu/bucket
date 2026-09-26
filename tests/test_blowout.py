"""Tests for the blowout-risk layer and the feature A/B harness."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.blowout import (
    BLOWOUT_FEATURE_COLS,
    DEFAULT_SPREAD_THRESHOLD,
    BlowoutFeatureError,
    attach_blowout_features,
)


def _frame(spreads):
    return pd.DataFrame({"MKT_OPENING_SPREAD": spreads})


def test_hinges_use_the_project_sign_convention():
    """Negative spread is the favourite, so -12 at a 9-point threshold is a
    3-point favourite hinge and nothing on the underdog side."""
    out = attach_blowout_features(_frame([-12.0, 14.5]), spread_threshold=9.0)
    assert out["BLOWOUT_FAV_HINGE"].tolist() == [3.0, 0.0]
    assert out["BLOWOUT_DOG_HINGE"].tolist() == [0.0, 5.5]


def test_inside_the_threshold_both_hinges_are_zero():
    out = attach_blowout_features(_frame([-9.0, -4.0, 0.0, 4.0, 9.0]), spread_threshold=9.0)
    assert out["BLOWOUT_FAV_HINGE"].eq(0.0).all()
    assert out["BLOWOUT_DOG_HINGE"].eq(0.0).all()


def test_unknown_spread_stays_unknown():
    """A null spread must not become a confident zero — that would read as
    'measured, and no blowout risk' for a row that had no market line."""
    out = attach_blowout_features(_frame([np.nan, -20.0]))
    assert out["BLOWOUT_FAV_HINGE"].isna().tolist() == [True, False]
    assert out["BLOWOUT_DOG_HINGE"].isna().tolist() == [True, False]


def test_missing_spread_column_skips_rather_than_zero_filling():
    df = pd.DataFrame({"PLAYER_ID": ["A", "B"]})
    out = attach_blowout_features(df)
    assert not set(BLOWOUT_FEATURE_COLS) & set(out.columns)
    assert list(out.columns) == ["PLAYER_ID"]


def test_missing_spread_column_raises_when_required():
    with pytest.raises(BlowoutFeatureError, match="DATA_NOT_AVAILABLE"):
        attach_blowout_features(pd.DataFrame({"PLAYER_ID": ["A"]}), required=True)


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_invalid_threshold_is_refused(bad):
    with pytest.raises(BlowoutFeatureError, match="non-negative"):
        attach_blowout_features(_frame([-10.0]), spread_threshold=bad)


def test_a_symmetric_hinge_would_be_exactly_collinear():
    """Documents why there is no BLOWOUT_ABS_HINGE: max(0, |s| - t) is the
    exact sum of the two published hinges, because only one can be positive.
    Adding it would hand every model a perfectly collinear column."""
    spreads = np.arange(-25.0, 25.5, 0.5)
    out = attach_blowout_features(_frame(spreads), spread_threshold=9.0)
    symmetric = np.maximum(0.0, np.abs(spreads) - 9.0)
    total = out["BLOWOUT_FAV_HINGE"] + out["BLOWOUT_DOG_HINGE"]
    assert np.allclose(total.to_numpy(), symmetric)


def test_only_the_opening_spread_is_read():
    """A closing spread on the frame must not be reachable through this layer."""
    df = _frame([-14.0]).assign(CLOSING_SPREAD=[-20.0], MKT_OPENING_TOTAL=[230.0])
    out = attach_blowout_features(df, spread_threshold=9.0)
    # Derived from the OPENING -14, not the closing -20 (which would give 11).
    assert out["BLOWOUT_FAV_HINGE"].iloc[0] == 5.0


def test_the_input_frame_is_not_mutated():
    df = _frame([-14.0])
    attach_blowout_features(df)
    assert not set(BLOWOUT_FEATURE_COLS) & set(df.columns)


# --- wiring into the pipeline ----------------------------------------------


def test_layer_is_disabled_in_the_shipped_config():
    """The measurement in src/features/blowout.py says this layer does not
    improve accuracy, so it must not ship enabled. If someone flips it, the
    A/B numbers should be what changed their mind — not this test."""
    from src.models.compare import load_comparison_config

    assert (load_comparison_config().get("blowout") or {}).get("enabled") is False


def test_builder_attaches_the_layer_only_when_enabled(monkeypatch):
    import src.features.builder as builder

    def _cfg(enabled):
        return {"blowout": {"enabled": enabled, "spread_threshold": 9.0}}

    monkeypatch.setattr(builder, "_layer_config", lambda: _cfg(False))
    assert "blowout" not in [name for name, _ in builder._additive_feature_layers()]

    monkeypatch.setattr(builder, "_layer_config", lambda: _cfg(True))
    layers = builder._additive_feature_layers()
    assert "blowout" in [name for name, _ in layers]

    attach = dict(layers)["blowout"]
    out = attach(_frame([-15.0]))
    assert out["BLOWOUT_FAV_HINGE"].iloc[0] == 6.0


def test_market_context_columns_reach_the_feature_list():
    """These were attached to every panel and read by nothing: no feature
    list named them, so no model ever saw the market's own forecast."""
    from src.models.labels import default_feature_cols

    cols = set(default_feature_cols("PTS"))
    assert {
        "MKT_OPENING_SPREAD",
        "MKT_OPENING_TOTAL",
        "MKT_IMPLIED_TEAM_TOTAL",
        "MKT_IMPLIED_OPP_TOTAL",
        "MKT_IS_FAVORITE",
    } <= cols
    assert set(BLOWOUT_FEATURE_COLS) <= cols


def test_feature_list_carries_no_closing_line():
    from src.features.market_context import CLOSING_ONLY_COLS
    from src.models.labels import default_feature_cols

    for market in ("PTS", "REB", "AST"):
        assert not set(default_feature_cols(market)) & CLOSING_ONLY_COLS


def test_ab_harness_builds_both_arms_from_one_feature_build():
    """When the panel already carries the columns, the control is the panel
    WITHOUT them rather than a second feature build. Two builds could differ
    for reasons other than the layer under test."""
    from scripts.feature_ab import LAYERS

    panel = _frame([-12.0]).assign(
        BLOWOUT_FAV_HINGE=[3.0], BLOWOUT_DOG_HINGE=[0.0], KEEP=[1]
    )
    control, treatment, under_test = LAYERS["blowout"].arms(panel, {})
    assert sorted(under_test) == ["BLOWOUT_DOG_HINGE", "BLOWOUT_FAV_HINGE"]
    assert not set(BLOWOUT_FEATURE_COLS) & set(control.columns)
    assert set(BLOWOUT_FEATURE_COLS) <= set(treatment.columns)
    # Everything else is identical — only the layer differs.
    assert control.drop(columns=[]).equals(treatment.drop(columns=list(BLOWOUT_FEATURE_COLS)))


def test_ab_harness_attaches_when_the_panel_lacks_the_layer():
    from scripts.feature_ab import LAYERS

    panel = _frame([-15.0, 2.0])
    control, treatment, under_test = LAYERS["blowout"].arms(panel, {})
    assert sorted(under_test) == ["BLOWOUT_DOG_HINGE", "BLOWOUT_FAV_HINGE"]
    assert not set(BLOWOUT_FEATURE_COLS) & set(control.columns)
    assert treatment["BLOWOUT_FAV_HINGE"].tolist() == [6.0, 0.0]


def test_market_context_arm_is_subtractive_and_cannot_be_conjured():
    """market_context needs the market_lines frame at build time. Without it
    the harness must say so, not invent a spread."""
    from scripts.feature_ab import LAYERS

    layer = LAYERS["market_context"]
    with pytest.raises(RuntimeError, match="cannot attach"):
        layer.arms(pd.DataFrame({"PLAYER_ID": ["A"]}), {})

    panel = pd.DataFrame({
        "PLAYER_ID": ["A"], "MKT_OPENING_SPREAD": [-6.5],
        "MKT_OPENING_TOTAL": [225.5], "MKT_IMPLIED_TEAM_TOTAL": [116.0],
        "MKT_IMPLIED_OPP_TOTAL": [109.5], "MKT_IS_FAVORITE": [1.0],
    })
    control, treatment, under_test = layer.arms(panel, {})
    assert len(under_test) == 5
    assert list(control.columns) == ["PLAYER_ID"]
    assert treatment is panel


def test_ab_harness_reports_a_missing_input_instead_of_crashing(monkeypatch, capsys):
    from scripts import feature_ab

    monkeypatch.setattr(
        "scripts.nba_model_cli._load_real_or_demo",
        lambda *a, **k: (pd.DataFrame({"PLAYER_ID": ["A"]}), True),
    )
    rc = feature_ab.main(["--layer", "blowout", "--markets", "PTS", "--demo"])
    assert rc == 2
    assert "DATA_NOT_AVAILABLE" in capsys.readouterr().err


def test_default_threshold_matches_where_the_blowout_rate_actually_moves():
    """Not a magic number: on the 2025-26 panel P(margin >= 15) is flat
    across 0-3/3-6/6-9/9-12 and only climbs at 12+. The default sits inside
    that flat region deliberately, so the hinge engages before the jump
    rather than after it."""
    assert DEFAULT_SPREAD_THRESHOLD == 9.0
