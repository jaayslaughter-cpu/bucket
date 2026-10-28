"""
tests/test_composite_features.py — rollups for the combination markets.

WHAT THIS COVERS. `src/quant/composite_markets.py` already decomposed PA, PR,
RA and PRA and already held the operator's FPTS weights, but its only consumer
was the settlement grader — so the project could GRADE a PA prop and not
PROJECT one, for want of a PA_L5 column. `src/models/composite_features.py`
attaches those columns from the component rollups.

THE PROPERTY THAT MATTERS MOST IS THE REFUSAL. A composite built from some of
its components is not a noisier estimate of that composite, it is a different
quantity wearing its name. FPTS is the sharp case: its TOV weight is NEGATIVE,
so silently dropping that term biases every projection UPWARD rather than
adding spread. So every "partial" case here asserts that NO column was written.

RESEARCH_ONLY.
"""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from src.models.composite_features import (
    COMPOSITE_SUFFIXES,
    attach_composite_rollups,
    composite_columns,
    composite_spec,
)
from src.quant.composite_markets import (
    SUM_COMPOSITES,
    CompositeMarketError,
    fantasy_weights,
)


@pytest.fixture
def rolled() -> pd.DataFrame:
    """A frame in the shape the builder returns: component rollups present."""
    return pd.DataFrame({
        "PLAYER_ID": ["1", "2"],
        "PTS_L2": [20.0, 10.0], "PTS_L5": [30.0, 12.0], "PTS_L10": [28.0, 11.0],
        "PTS_SEASON": [27.0, 10.5], "PTS_BASELINE": [29.0, 11.5],
        "REB_L2": [5.0, 8.0], "REB_L5": [6.0, 9.0], "REB_L10": [6.5, 8.5],
        "REB_SEASON": [6.2, 8.2], "REB_BASELINE": [6.1, 8.8],
        "AST_L2": [4.0, 2.0], "AST_L5": [3.0, 2.5], "AST_L10": [3.5, 2.2],
        "AST_SEASON": [3.2, 2.1], "AST_BASELINE": [3.4, 2.3],
        "STL_L2": [1.0, 0.5], "STL_L5": [1.2, 0.6], "STL_L10": [1.1, 0.7],
        "STL_SEASON": [1.0, 0.6], "STL_BASELINE": [1.1, 0.6],
        "BLK_L2": [0.5, 1.5], "BLK_L5": [0.6, 1.4], "BLK_L10": [0.7, 1.3],
        "BLK_SEASON": [0.6, 1.4], "BLK_BASELINE": [0.6, 1.4],
        "TOV_L2": [3.0, 1.0], "TOV_L5": [2.5, 1.2], "TOV_L10": [2.6, 1.1],
        "TOV_SEASON": [2.4, 1.0], "TOV_BASELINE": [2.5, 1.1],
    })


# --- the sums ------------------------------------------------------------


@pytest.mark.parametrize("market", sorted(SUM_COMPOSITES))
def test_a_sum_composite_is_exactly_the_sum_of_its_component_rollups(market, rolled):
    """Arithmetic, per suffix, per row — not "a column appeared"."""
    out, report = attach_composite_rollups(rolled, [market], overwrite=True)
    components = SUM_COMPOSITES[market]

    for suffix in COMPOSITE_SUFFIXES:
        col = f"{market}_{suffix}"
        assert col in out.columns, f"{col} was not attached"
        expected = sum(out[f"{c}_{suffix}"] for c in components)
        pd.testing.assert_series_equal(
            out[col], expected, check_names=False,
            obj=f"{col} must equal {' + '.join(f'{c}_{suffix}' for c in components)}",
        )
    assert market in report.attached


def test_the_decomposition_is_read_from_composite_markets_not_copied(rolled, monkeypatch):
    """
    THE "N READERS OF ONE FACT" TEST, done behaviourally rather than by
    grepping the source. If this module held its own copy of "PA = PTS + AST",
    changing the shared table would not change its output. So the table is
    changed and the output has to follow.
    """
    import src.quant.composite_markets as cm

    monkeypatch.setitem(cm.SUM_COMPOSITES, "PA", ("PTS", "REB"))
    out, _ = attach_composite_rollups(rolled, ["PA"], overwrite=True)
    pd.testing.assert_series_equal(
        out["PA_L5"], out["PTS_L5"] + out["REB_L5"], check_names=False,
        obj="PA followed a local copy of the decomposition, not the shared one",
    )


# --- FPTS, which is an operator's terms ----------------------------------


def test_fpts_is_the_weighted_sum_of_the_configured_weights(rolled):
    """
    The weights are read from `fantasy_weights` here too, so this test cannot
    drift from the config. Hardcoding 1.0/1.2/1.5/3.0/-1.0 would make it a
    second copy of the operator's terms — the thing
    `config/fantasy_scoring.yaml` exists to prevent.
    """
    weights = fantasy_weights("underdog")
    out, report = attach_composite_rollups(
        rolled, ["FPTS"], platform="underdog", overwrite=True
    )
    assert "FPTS" in report.attached

    for suffix in COMPOSITE_SUFFIXES:
        expected = sum(
            out[f"{name}_{suffix}"] * weight for name, weight in weights.items()
        )
        pd.testing.assert_series_equal(
            out[f"FPTS_{suffix}"], expected, check_names=False,
            obj=f"FPTS_{suffix} is not the configured weighted sum",
        )


def test_fpts_without_a_platform_refuses_rather_than_guessing(rolled, caplog):
    """
    The same refusal `settlement/evaluator.py` makes. Two operators are
    configured separately even though their weights match today, so picking
    one silently would turn "they happen to agree" into "they cannot differ".
    """
    with caplog.at_level(logging.WARNING):
        out, report = attach_composite_rollups(rolled, ["FPTS"])

    assert "FPTS" in report.skipped
    assert "DATA_NOT_AVAILABLE" in report.skipped["FPTS"]
    assert not [c for c in out.columns if c.startswith("FPTS_")], (
        "FPTS columns were written without an operator's weights"
    )


def test_the_refusal_comes_from_the_one_reader_of_the_weights(rolled):
    """
    The reason must be `fantasy_weights`' own, which names the config file and
    every configured operator. A hand-written guard here produced a thinner
    message AND re-read the YAML to list the platforms — a second reader of
    the fact that file owns. Mutation testing showed the guard was not
    load-bearing, so it was deleted; this pins what replaced it.
    """
    _, report = attach_composite_rollups(rolled, ["FPTS"])
    why = report.skipped["FPTS"]
    assert "DATA_NOT_AVAILABLE" in why
    assert "fantasy_scoring.yaml" in why, (
        f"the refusal does not say where the weights would come from: {why}"
    )
    assert "prizepicks" in why and "underdog" in why, (
        f"the refusal leaves the operator guessing which platforms exist: {why}"
    )


def test_fpts_differs_from_a_plain_sum(rolled):
    """A guard against the weights silently becoming 1.0 each: a weighted
    score must not equal the unweighted total of the same components."""
    out, _ = attach_composite_rollups(
        rolled, ["FPTS"], platform="underdog", overwrite=True
    )
    plain = sum(out[f"{c}_L5"] for c in fantasy_weights("underdog"))
    assert not out["FPTS_L5"].equals(plain)


# --- never a part sum ----------------------------------------------------


@pytest.mark.parametrize("drop", ["AST", "PTS"])
def test_a_sum_with_a_missing_component_writes_nothing(rolled, drop, caplog):
    """
    PA from PTS alone is not a worse PA. It is a number that reads as a PA and
    is not one, and nothing downstream could tell.
    """
    frame = rolled.drop(columns=[c for c in rolled.columns if c.startswith(f"{drop}_")])
    with caplog.at_level(logging.WARNING):
        out, report = attach_composite_rollups(frame, ["PA"], overwrite=True)

    assert not [c for c in out.columns if c.startswith("PA_")], (
        f"PA was part-summed with {drop} missing"
    )
    assert "PA" in report.skipped
    assert "partial sum" in report.skipped["PA"]


def test_fpts_is_skipped_when_tov_rollups_are_absent(rolled):
    """
    THE EXACT STATE THIS TREE WAS IN UNTIL 2026-10-28. `TOV` was not in
    `builder.ROLLING_STATS`, so TOV_L5 did not exist. Dropping the −1.0*TOV
    term would not have made FPTS noisier — it would have biased every
    projection UPWARD, which is the worst direction for a number used to
    decide an Over.
    """
    frame = rolled.drop(columns=[c for c in rolled.columns if c.startswith("TOV_")])
    out, report = attach_composite_rollups(frame, ["FPTS"], platform="underdog")

    assert not [c for c in out.columns if c.startswith("FPTS_")]
    assert "FPTS" in report.skipped
    assert "TOV" in report.skipped["FPTS"]


def test_tov_is_now_rolled_so_fpts_is_buildable_at_all():
    """The prerequisite, asserted against the builder rather than assumed."""
    from src.features.builder import ROLLING_STATS

    assert "TOV" in ROLLING_STATS, (
        "TOV left ROLLING_STATS, so FPTS can no longer be projected — its "
        "weights carry a negative TOV term"
    )
    for component in fantasy_weights("underdog"):
        assert component in ROLLING_STATS, (
            f"FPTS needs {component} rollups and the builder does not produce them"
        )


# --- not becoming a second writer ---------------------------------------


def test_pra_already_present_is_left_to_the_builder(rolled):
    """
    `builder.attach_pra_from_components` owns PRA and computes it identically.
    Two writers of one column is the defect that function's docstring exists
    to describe — two definitions of PRA, the second silently winning.
    """
    frame = rolled.copy()
    frame["PRA_L5"] = -999.0  # a sentinel the builder would never write
    out, report = attach_composite_rollups(frame, ["PRA"])
    assert out["PRA_L5"].iloc[0] == -999.0, "PRA_L5 was overwritten by default"
    assert "PRA_L5" not in report.attached.get("PRA", [])


def test_overwrite_is_opt_in_and_then_recomputes(rolled):
    frame = rolled.copy()
    frame["PRA_L5"] = -999.0
    out, _ = attach_composite_rollups(frame, ["PRA"], overwrite=True)
    expected = out["PTS_L5"] + out["REB_L5"] + out["AST_L5"]
    pd.testing.assert_series_equal(out["PRA_L5"], expected, check_names=False)


def test_the_input_frame_is_not_mutated(rolled):
    before = list(rolled.columns)
    attach_composite_rollups(rolled, ["PA"], overwrite=True)
    assert list(rolled.columns) == before


# --- the contract surface ------------------------------------------------


def test_an_unknown_market_raises_rather_than_returning_something_plausible():
    with pytest.raises(CompositeMarketError, match="not a composite"):
        composite_spec("POINTS_PLUS_VIBES")


def test_composite_columns_predicts_what_attach_produces(rolled):
    names = composite_columns("PA")
    out, _ = attach_composite_rollups(rolled, ["PA"], overwrite=True)
    for name in names:
        assert name in out.columns


def test_a_sum_composite_weights_every_component_at_one():
    spec = composite_spec("PA")
    assert set(spec.weights) == set(SUM_COMPOSITES["PA"])
    assert all(w == 1.0 for w in spec.weights.values())


# --- what is still NOT true, pinned so nobody assumes otherwise ---------


def test_pa_is_still_not_a_first_class_market_and_the_warning_still_fires(caplog):
    """
    THE REMAINING STEP, RECORDED RATHER THAN GLOSSED. These rollups are the
    prerequisite for a PA model, not the whole of one: `feature_spec`'s owned
    set is pinned to `builder.ROLLING_STATS`, which does not contain PA, so
    `build_prop_feature_spec("PA")` still substitutes PTS features and still
    says so. Declaring PA first-class also needs the Wave 5a layers —
    `_STREAK_STATS` is (PTS, REB, AST, FG3M), so PA_STREAK_ABOVE and PA_HOT_Z
    are not built by anything.

    If someone flips PA into the owned set without that, this test fails and
    says what is missing.
    """
    from src.models.feature_spec import build_prop_feature_spec

    with caplog.at_level(logging.WARNING):
        spec = build_prop_feature_spec("PA")

    assert spec.market == "PA"
    assert spec.feature_market == "PTS", (
        "PA was promoted to a first-class market, but the Wave 5a streak and "
        "hot-hand layers still do not build PA_STREAK_ABOVE / PA_HOT_Z"
    )
    assert "NOT about PA" in caplog.text
