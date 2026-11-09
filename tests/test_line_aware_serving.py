"""
`line_aware` reachable from the resolver, and scoreable at a posted line.

THE DEFECT. Training writes `line_aware_{MARKET}.manifest.json`. The resolver
globbed `{component}_{MARKET}.json` for every component, so the one fitted
component that both takes a line AND reads a feature vector was persisted,
checksummed, verified on every nightly fit -- and unfindable. The serving
component (`distribution`) reads exactly one column, `{MARKET}_L2`, out of a
192-column matrix, so the gap mattered: every other feature the builder
produces was unreachable by anything that could answer at a book line.

WIRING IS NOT PROMOTION. These tests pin that `$PROPIQ_SCORING_COMPONENT`
still defaults to `distribution`. Making a component available is a different
act from deciding it should serve, and the second needs forward evidence this
project does not yet have.

RESEARCH_ONLY. Fixtures are synthetic; no probability here is a real forecast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import main as propiq_main
from src.models.serving_components import (
    DEFAULT_ARTIFACT_SUFFIX,
    artifact_suffix,
    is_line_taking,
    load_line_taking,
)


def _panel(n: int = 320) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    actual = rng.poisson(18.0, n).astype(float)
    return pd.DataFrame({
        "PLAYER_ID": [f"espn:{i % 30}" for i in range(n)],
        "PLAYER_NAME": [f"DEMO_{i % 30}" for i in range(n)],
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


def _prop_lines(names) -> pd.DataFrame:
    return pd.DataFrame([{
        "player_name": n, "market": "PTS", "line": 18.5, "source": "propline",
        "over_odds_american": -110, "under_odds_american": -110,
    } for n in sorted(set(names))])


@pytest.fixture
def fitted_line_aware(tmp_path):
    """A real fit, saved exactly as `fit_market_artifacts` saves it."""
    from src.models.compare import (
        build_components,
        load_comparison_config,
        resolve_feature_cols,
    )
    from src.models.labels import default_feature_cols

    cols, _dropped = resolve_feature_cols(_panel(), list(default_feature_cols("PTS")))
    model = build_components("PTS", cols, load_comparison_config())["line_aware"]
    frame = _panel().sort_values("GAME_DATE").reset_index(drop=True)
    cut = frame["GAME_DATE"].quantile(0.67)
    model.fit(frame[frame["GAME_DATE"] <= cut], frame[frame["GAME_DATE"] > cut])
    model.save(tmp_path / "line_aware_PTS")
    return tmp_path


# --- the resolver ----------------------------------------------------------

def test_the_resolver_finds_the_manifest_that_training_actually_writes(
    fitted_line_aware, monkeypatch,
):
    """
    THE DEFECT, DIRECTLY. `line_aware_PTS.manifest.json` exists on disk; the
    glob asked for `line_aware_PTS.json` and found nothing.
    """
    monkeypatch.setattr(
        "src.models.compare.load_comparison_config",
        lambda *a, **k: {"artifacts_dir": str(fitted_line_aware)},
    )
    path, how = propiq_main.resolve_model_artifact(
        market="PTS", component="line_aware",
    )
    assert path is not None, f"the fitted manifest was not resolvable: {how}"
    assert path.name == "line_aware_PTS.manifest.json"


def test_one_components_glob_cannot_match_anothers_artifact(
    fitted_line_aware, monkeypatch,
):
    """
    Asking for the distribution must never be answered with a line-aware
    manifest. A resolver that returned the wrong component would publish one
    model's probabilities under another's name.
    """
    monkeypatch.setattr(
        "src.models.compare.load_comparison_config",
        lambda *a, **k: {"artifacts_dir": str(fitted_line_aware)},
    )
    monkeypatch.setattr(propiq_main, "MODEL_ARTIFACT_DEFAULT", fitted_line_aware / "absent.json")
    path, _how = propiq_main.resolve_model_artifact(
        market="PTS", component="distribution",
    )
    assert path is None, f"a distribution lookup resolved {path}"


def test_the_suffix_is_read_from_the_class_that_writes_the_file(monkeypatch):
    """
    ONE DEFINITION. If the resolver kept its own copy of ".manifest.json",
    changing how the model names its artifact would silently unwire it again.
    """
    from src.models.line_aware import LineAwarePropModel

    assert artifact_suffix("line_aware") == LineAwarePropModel.ARTIFACT_SUFFIX
    monkeypatch.setattr(LineAwarePropModel, "ARTIFACT_SUFFIX", ".bundle.json")
    assert artifact_suffix("line_aware") == ".bundle.json"


def test_an_unknown_component_gets_the_default_rather_than_raising():
    assert artifact_suffix("not_a_component") == DEFAULT_ARTIFACT_SUFFIX


# --- scoring ---------------------------------------------------------------

def test_line_aware_scores_at_the_posted_line(fitted_line_aware, monkeypatch):
    """
    End to end: selected by the environment, resolved from the manifest,
    loaded through its classmethod, and asked at a BOOK line rather than at
    the trailing average its labels were built from.
    """
    monkeypatch.setenv("PROPIQ_SCORING_COMPONENT", "line_aware")
    monkeypatch.delenv("PROPIQ_MODEL", raising=False)
    monkeypatch.setattr(
        "src.models.compare.load_comparison_config",
        lambda *a, **k: {"artifacts_dir": str(fitted_line_aware)},
    )
    features = _panel(40)
    scored = propiq_main.score_prob_over_by_market(
        features, _prop_lines(features["PLAYER_NAME"]), markets=("PTS",),
    )
    assert "PTS" in scored, "line_aware produced no market"
    probs = pd.to_numeric(scored["PTS"], errors="coerce")
    usable = int(probs.notna().sum())
    assert usable > 0, "every row abstained; the component scored nothing"
    inside = probs.dropna()
    assert ((inside >= 0.0) & (inside <= 1.0)).all()


def test_the_probability_moves_with_the_line(fitted_line_aware):
    """
    THE WHOLE REASON THIS COMPONENT EXISTS. A line-blind classifier returns
    the same number at every line; this one must not. If these two agree, the
    model is not reading the line and wiring it bought nothing.
    """
    from src.models.line_aware import LineAwarePropModel

    model = LineAwarePropModel.load(fitted_line_aware / "line_aware_PTS.manifest.json")
    features = _panel(60)
    low = pd.to_numeric(model.predict_probability_over(features, 14.5), errors="coerce")
    high = pd.to_numeric(model.predict_probability_over(features, 22.5), errors="coerce")
    both = low.notna() & high.notna()
    assert both.any(), "no row was inside the trained line range"
    assert not np.allclose(low[both], high[both]), (
        "P(over) is identical at 14.5 and 22.5 — the line is not reaching the model"
    )


def test_both_load_shapes_are_absorbed_by_one_caller(fitted_line_aware, tmp_path):
    """
    `DistributionPropModel.load` is an instance method returning self;
    `LineAwarePropModel.load` is a classmethod. A scorer hardcoding either
    breaks on the other.
    """
    from src.models.distribution_adapter import DistributionPropModel

    dist = DistributionPropModel(target_market="PTS")
    dist.fit(_panel())
    dist.save(tmp_path / "distribution_PTS.json")

    assert load_line_taking(
        "line_aware", fitted_line_aware / "line_aware_PTS.manifest.json"
    ).target_market == "PTS"
    assert load_line_taking(
        "distribution", tmp_path / "distribution_PTS.json"
    ).target_market == "PTS"


def test_the_booster_is_not_a_line_taking_component():
    """
    XGBoost learns P(stat > RESEARCH_LINE) and the line is not an input, so it
    returns the same number at every line. Routing it through the posted-line
    scorer would publish exactly the number `mask_probabilities_at_unsupported_
    lines` exists to refuse.
    """
    assert is_line_taking("distribution") and is_line_taking("line_aware")
    assert not is_line_taking("xgboost")
    with pytest.raises(ValueError, match="posted line"):
        load_line_taking("xgboost", "irrelevant.json")


# --- wiring is not promotion ----------------------------------------------

def test_making_a_component_available_does_not_make_it_serve(monkeypatch):
    monkeypatch.delenv("PROPIQ_SCORING_COMPONENT", raising=False)
    assert propiq_main.scoring_component() == "distribution"
    assert "line_aware" in propiq_main.SCORING_COMPONENTS


def test_line_aware_is_selectable_by_name(monkeypatch):
    monkeypatch.setenv("PROPIQ_SCORING_COMPONENT", "line_aware")
    assert propiq_main.scoring_component() == "line_aware"
