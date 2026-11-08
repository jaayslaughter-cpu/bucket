"""
`LineAwarePropModel` persistence, and the fitted-vs-persisted distinction.

WHY THIS FILE EXISTS. `fit_market_artifacts` guarded its save with
`if hasattr(model, "save")` and appended to `fitted` regardless, so every run
reported four components and wrote three: `line_aware` had no `save` at all.
The report said "fitted: [distribution, xgboost, catboost, line_aware]" while
the artifacts directory held no line_aware file. An external audit surfaced it;
the guard below is so it cannot come back.

The save/load implementation was ported from that audit's bundle after review.
These tests assert the properties that made it worth taking.

RESEARCH_ONLY.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest


def _panel(n: int = 320) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    actual = rng.poisson(18.0, n).astype(float)
    return pd.DataFrame({
        "PLAYER_ID": [f"espn:{i % 30}" for i in range(n)],
        "GAME_DATE": pd.date_range("2026-01-01", periods=n, freq="h"),
        "PTS": actual,
        "PTS_L2": rng.normal(18.0, 2.0, n).clip(1.0),
        "PTS_L5": rng.normal(18.0, 2.0, n).clip(1.0),
        "PTS_L10": rng.normal(18.0, 2.0, n).clip(1.0),
        "PTS_SEASON": rng.normal(18.0, 1.5, n).clip(1.0),
        "PTS_BASELINE": rng.normal(18.0, 2.0, n).clip(1.0),
        "MIN_L5": rng.normal(30.0, 3.0, n).clip(5.0),
        "MIN_L10": rng.normal(30.0, 3.0, n).clip(5.0),
        "MIN_SEASON": rng.normal(30.0, 3.0, n).clip(5.0),
        "fatigue_multiplier": np.ones(n),
        "days_rest": rng.integers(1, 4, n).astype(float),
        "RESEARCH_LINE": rng.normal(18.0, 2.0, n).clip(1.0),
        "over_hit": (actual > 18.0).astype(float),
    })


# --- the distinction my code lost -----------------------------------------


def test_the_report_separates_fitted_from_persisted(tmp_path):
    """
    A COMPONENT THAT CANNOT PERSIST CANNOT BE SERVED, so its share of the
    ensemble is unreachable and that has to show in the report rather than be
    inferred from a directory listing.
    """
    from src.models.training import fit_market_artifacts

    report = fit_market_artifacts(_panel(), "PTS", artifacts_dir=tmp_path)

    assert set(report["persisted"]) <= set(report["fitted"])
    # Every component that claims to be fitted now also writes something.
    assert report["fitted_but_not_persisted"] == [], (
        f"{report['fitted_but_not_persisted']} fitted without writing an "
        f"artifact, so they cannot be resolved or served"
    )
    for name in report["persisted"]:
        assert list(tmp_path.glob(f"{name}_PTS*")), f"{name} wrote no file"


def test_a_component_without_save_is_named_not_silently_counted(tmp_path, caplog):
    """
    The guard itself: if a component ever loses `save` again, the report must
    say so by name instead of folding it into `fitted`.
    """
    import logging

    from src.models import compare, training

    real = compare.build_components

    def _crippled(market, cols, cfg, **kw):
        comps = real(market, cols, cfg, **kw)
        victim = comps["distribution"]

        class NoSave:
            def __init__(self, inner):
                self._inner = inner

            def fit(self, *a, **k):
                return self._inner.fit(*a, **k)

        comps["distribution"] = NoSave(victim)
        return comps

    compare.build_components = _crippled
    try:
        with caplog.at_level(logging.WARNING, logger="src.models.training"):
            report = training.fit_market_artifacts(
                _panel(), "PTS", artifacts_dir=tmp_path
            )
    finally:
        compare.build_components = real

    assert report["fitted_but_not_persisted"] == ["distribution"]
    assert "distribution" not in report["persisted"]
    assert "no save()" in caplog.text


# --- the ported persistence ------------------------------------------------


@pytest.fixture
def saved_line_aware(tmp_path):
    # THE REAL FEATURE RESOLUTION, not every column: passing the whole frame's
    # columns hands PLAYER_ID to a numeric model, which it correctly refuses.
    from src.models.compare import build_components, load_comparison_config, resolve_feature_cols
    from src.models.labels import default_feature_cols

    frame_cols, _dropped = resolve_feature_cols(
        _panel(), list(default_feature_cols("PTS"))
    )
    comps = build_components("PTS", frame_cols, load_comparison_config())
    model = comps["line_aware"]
    frame = _panel().sort_values("GAME_DATE").reset_index(drop=True)
    # SPLIT ON GAME_DATE, not on rows. `fit` raises LineLeakageError when a
    # player-game has augmented copies on both sides -- every copy shares one
    # realised outcome, so an overlap means the model has seen the answer.
    cut = frame["GAME_DATE"].quantile(0.67)
    model.fit(frame[frame["GAME_DATE"] <= cut], frame[frame["GAME_DATE"] > cut])
    target = tmp_path / "line_aware_PTS"
    model.save(target)
    return model, tmp_path / "line_aware_PTS.manifest.json"


def test_it_round_trips_and_scores_the_same(saved_line_aware):
    """Persistence that changes the probabilities is not persistence."""
    from src.models.line_aware import LineAwarePropModel

    model, manifest = saved_line_aware
    features = _panel(40)

    before = pd.to_numeric(model.predict_probability_over(features, 18.5))
    after = pd.to_numeric(
        LineAwarePropModel.load(manifest).predict_probability_over(features, 18.5)
    )
    pd.testing.assert_series_equal(before, after, check_names=False)


def test_a_tampered_component_is_refused_by_checksum(saved_line_aware):
    """
    THE PROPERTY THAT MADE THIS WORTH PORTING. Every component is SHA-256
    checksummed and re-verified on load, so a swapped booster is caught rather
    than scored with.
    """
    from src.models.line_aware import LineAwarePropModel

    _model, manifest = saved_line_aware
    meta = json.loads(manifest.read_text())
    victim = manifest.parent / meta["base_artifact"]
    victim.write_bytes(victim.read_bytes() + b" ")

    with pytest.raises(ValueError, match="checksum mismatch"):
        LineAwarePropModel.load(manifest)


def test_a_manifest_cannot_point_outside_its_own_directory(saved_line_aware):
    """A crafted manifest must not read across the filesystem."""
    from src.models.line_aware import LineAwarePropModel

    _model, manifest = saved_line_aware
    meta = json.loads(manifest.read_text())
    meta["base_artifact"] = "../escaped.json"
    manifest.write_text(json.dumps(meta))

    with pytest.raises(ValueError, match="sibling file"):
        LineAwarePropModel.load(manifest)


def test_an_unknown_format_is_refused(saved_line_aware):
    from src.models.line_aware import LineAwarePropModel

    _model, manifest = saved_line_aware
    meta = json.loads(manifest.read_text())
    meta["format"] = "something-else-v9"
    manifest.write_text(json.dumps(meta))

    with pytest.raises(ValueError, match="Unknown line-aware artifact format"):
        LineAwarePropModel.load(manifest)


def test_an_unfitted_model_refuses_to_save(tmp_path):
    """Writing an unfitted model would produce an artifact that scores."""
    from src.models.compare import build_components, load_comparison_config, resolve_feature_cols
    from src.models.labels import default_feature_cols

    cols, _dropped = resolve_feature_cols(_panel(), list(default_feature_cols("PTS")))
    comps = build_components("PTS", cols, load_comparison_config())
    with pytest.raises(RuntimeError, match="unfitted"):
        comps["line_aware"].save(tmp_path / "line_aware_PTS")


def test_the_manifest_is_published_atomically(saved_line_aware):
    """No `.tmp` left behind: the temp file is replaced, not copied."""
    _model, manifest = saved_line_aware
    assert manifest.exists()
    assert not list(manifest.parent.glob("*.tmp")), "a temp artifact survived the save"
