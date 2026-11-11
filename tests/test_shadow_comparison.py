"""
Two components, one slate: scored on the same rows, only one published.

WHY. Every component's standing in config/model_comparison.yaml was measured
against RESEARCH-LINE labels, so the weights rank them at "over the trailing
average" rather than "over a book line" -- the question the board asks.
Nothing on record says which is better at the real question. Scoring both at
the same posted lines and grading the same outcome is how that gets answered.

THE BOUNDARY THESE TESTS DEFEND. A shadow is written to the decision record
and nowhere else. Two probabilities in `prop_results` would double the
denominator behind every strike rate, and a challenger that can publish is
not a challenger.

RESEARCH_ONLY. Synthetic fixtures; no probability here is a real forecast.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import main as propiq_main
from src.settlement.decision_record import decision_rows


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


def _slate(n: int = 24) -> pd.DataFrame:
    """One row per player, the shape `features` has after the slate filter."""
    frame = _panel(n).copy()
    frame["PLAYER_NAME"] = [f"DEMO_{i}" for i in range(n)]
    return frame


def _lines(names) -> pd.DataFrame:
    return pd.DataFrame([{
        "player_name": n, "market": "PTS", "line": 18.5, "source": "propline",
        "over_odds_american": -110, "under_odds_american": -110,
    } for n in sorted(set(names))])


@pytest.fixture
def both_components(tmp_path):
    """Fit and persist distribution AND line_aware into one artifacts dir."""
    from src.models.compare import (
        build_components,
        load_comparison_config,
        resolve_feature_cols,
    )
    from src.models.labels import default_feature_cols

    cols, _dropped = resolve_feature_cols(_panel(), list(default_feature_cols("PTS")))
    comps = build_components("PTS", cols, load_comparison_config())
    frame = _panel().sort_values("GAME_DATE").reset_index(drop=True)
    cut = frame["GAME_DATE"].quantile(0.67)
    train, val = frame[frame["GAME_DATE"] <= cut], frame[frame["GAME_DATE"] > cut]

    comps["distribution"].fit(train, val)
    comps["distribution"].save(tmp_path / "distribution_PTS.json")
    comps["line_aware"].fit(train, val)
    comps["line_aware"].save(tmp_path / "line_aware_PTS")
    return tmp_path


@pytest.fixture
def artifacts(both_components, monkeypatch):
    monkeypatch.setattr(
        "src.models.compare.load_comparison_config",
        lambda *a, **k: {"artifacts_dir": str(both_components)},
    )
    monkeypatch.delenv("PROPIQ_MODEL", raising=False)
    monkeypatch.delenv("PROPIQ_SCORING_COMPONENT", raising=False)
    monkeypatch.delenv("PROPIQ_SHADOW_COMPONENTS", raising=False)
    return both_components


# --- which components are shadowed ----------------------------------------

def test_the_serving_component_is_never_its_own_shadow(artifacts):
    """Scoring it twice would compare it with itself."""
    assert propiq_main.scoring_component() == "distribution"
    assert "distribution" not in propiq_main.shadow_components()
    assert "line_aware" in propiq_main.shadow_components()


def test_switching_what_serves_switches_what_shadows(artifacts, monkeypatch):
    """
    THE RULE, NOT A FIXED LIST. This asserted ``== ("distribution",)`` and so
    failed the moment a third line-taking component was registered -- which is
    the thing shadow scoring exists to support, not a regression. What must
    hold is that every line-taking component EXCEPT the one serving is
    shadowed, whatever the registry contains.
    """
    from src.models.serving_components import LINE_TAKING_COMPONENTS

    monkeypatch.setenv("PROPIQ_SCORING_COMPONENT", "line_aware")
    shadows = set(propiq_main.shadow_components())

    assert "line_aware" not in shadows, "the serving component shadowed itself"
    assert shadows == set(LINE_TAKING_COMPONENTS) - {"line_aware"}, (
        f"shadows {sorted(shadows)} against registry "
        f"{sorted(LINE_TAKING_COMPONENTS)}"
    )
    assert len(shadows) >= 2, (
        "the registry no longer holds enough components for this test to "
        "distinguish the rule from a hardcoded single entry"
    )


def test_it_can_be_turned_off(artifacts, monkeypatch):
    monkeypatch.setenv("PROPIQ_SHADOW_COMPONENTS", "none")
    assert propiq_main.shadow_components() == ()


def test_a_component_that_cannot_take_a_line_is_refused(artifacts, monkeypatch, caplog):
    """
    The booster answers the same number at every line. Shadowing it would
    record a comparison that says nothing about the question being asked.
    """
    monkeypatch.setenv("PROPIQ_SHADOW_COMPONENTS", "xgboost,line_aware")
    assert propiq_main.shadow_components() == ("line_aware",)


# --- the comparison itself -------------------------------------------------

def test_both_components_answer_the_same_rows_at_the_same_line(artifacts):
    features = _slate()
    shadow = propiq_main.shadow_probabilities(
        features, _lines(features["PLAYER_NAME"]), markets=("PTS",),
    )
    assert "line_aware" in shadow, "the challenger produced nothing to compare"
    served = propiq_main.score_prob_over_by_market(
        features, _lines(features["PLAYER_NAME"]), markets=("PTS",),
    )
    served_probs = pd.to_numeric(served["PTS"], errors="coerce")

    keys = {(n, "PTS") for n in features["PLAYER_NAME"]}
    assert set(shadow["line_aware"]).issubset(keys)
    assert served_probs.notna().sum() > 0
    assert len(shadow["line_aware"]) > 0


def test_the_two_components_do_not_simply_agree(artifacts):
    """
    A comparison between two identical answers is not a comparison. If these
    ever match row for row, one of them is not reading what it claims to.
    """
    features = _slate()
    lines = _lines(features["PLAYER_NAME"])
    served = pd.to_numeric(
        propiq_main.score_prob_over_by_market(features, lines, markets=("PTS",))["PTS"],
        errors="coerce",
    )
    shadow = propiq_main.shadow_probabilities(features, lines, markets=("PTS",))["line_aware"]
    pairs = [
        (served.iloc[i], shadow.get((name, "PTS")))
        for i, name in enumerate(features["PLAYER_NAME"])
    ]
    both = [(a, b) for a, b in pairs if pd.notna(a) and b is not None]
    assert both, "no row was answered by both components"
    assert any(abs(a - b) > 1e-9 for a, b in both)


# --- the boundary ----------------------------------------------------------

def test_a_shadow_never_enters_the_graded_ledger(artifacts):
    """
    `prop_results` is the denominator behind every strike rate. A second row
    per prediction, or a shadow overwriting `prob_over`, would corrupt it.
    """
    from src.settlement.recorder import pending_prop_result_rows

    projections = pd.DataFrame([{
        "PLAYER_NAME": "DEMO_0", "PLAYER_ID": "1", "GAME_ID": "0022500123",
        "GAME_DATE": pd.Timestamp("2026-11-09"), "MARKET": "PTS",
        "LINE": 18.5, "PROB_OVER": 0.62, "FINAL_PROJECTION": 19.0,
    }])
    report = pending_prop_result_rows(projections, _lines(["DEMO_0"]), run_id="r")
    assert len(report.rows) == 1
    ledger_row = report.rows[0]
    assert ledger_row["prob_over"] == pytest.approx(0.62)
    assert not any("shadow" in k for k in ledger_row)

    rows = decision_rows(report, shadow={"line_aware": {("DEMO_0", "PTS"): 0.71}})
    assert rows[0]["prob_over"] == pytest.approx(0.62), (
        "the shadow overwrote the served probability"
    )
    assert rows[0]["shadow"] == {"line_aware": 0.71}
    # Still one ledger row, unchanged by the record it fed.
    assert len(report.rows) == 1


def test_a_shadow_that_abstained_is_recorded_as_abstaining(artifacts):
    """
    Omitting it would make "did not answer" look like "was not asked", and
    the second reads as a smaller sample rather than a refusal -- which is
    exactly the distinction `line_aware` draws outside its trained range.
    """
    from src.settlement.recorder import pending_prop_result_rows

    projections = pd.DataFrame([{
        "PLAYER_NAME": "DEMO_0", "PLAYER_ID": "1", "GAME_ID": "0022500123",
        "GAME_DATE": pd.Timestamp("2026-11-09"), "MARKET": "PTS",
        "LINE": 18.5, "PROB_OVER": 0.62, "FINAL_PROJECTION": 19.0,
    }])
    report = pending_prop_result_rows(projections, _lines(["DEMO_0"]), run_id="r")
    rows = decision_rows(report, shadow={"line_aware": {}})
    assert rows[0]["shadow"] == {"line_aware": None}


def test_a_failing_shadow_does_not_fail_the_slate(artifacts, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("synthetic shadow failure")

    monkeypatch.setattr(propiq_main, "score_prob_over_at_posted_line", boom)
    features = _slate()
    assert propiq_main.shadow_probabilities(
        features, _lines(features["PLAYER_NAME"]), markets=("PTS",),
    ) == {}


def test_no_shadow_is_scored_when_there_is_nothing_to_score(artifacts):
    assert propiq_main.shadow_probabilities(
        pd.DataFrame(), pd.DataFrame(), markets=("PTS",),
    ) == {}


# --- wiring ----------------------------------------------------------------

def test_the_slate_computes_shadows_and_records_them():
    """
    AST-walked inside `main.main`. A comparison nothing calls compares
    nothing, and a shadow computed but not recorded is CPU spent for no
    evidence.
    """
    tree = ast.parse(Path("main.py").read_text(encoding="utf-8"))
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    called = {
        n.func.id for n in ast.walk(fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "shadow_probabilities" in called, "main() computes no shadow"

    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "decision_rows":
            kw = {k.arg for k in node.keywords}
            assert "shadow" in kw, "the shadow is computed but never recorded"
            return
    raise AssertionError("decision_rows is not called in main()")
