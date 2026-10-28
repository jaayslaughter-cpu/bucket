"""
tests/test_feature_market_fallback.py — a spec must not wear a label it has
not earned.

WHAT WAS WRONG. `build_prop_feature_spec("PA")` returned a spec with `market="PA"`
whose feature list was the PTS one. `default_feature_cols` templates
`{MARKET}_L2`, `{MARKET}_L5`, ... so a market with no rollup columns in the
panel produces names nothing builds, and `assert_compatible` would refuse the
spec — the fallback to PTS exists to avoid that refusal. But it was SILENT,
which is the worse of the two outcomes: a model fitted that way is a points
model carrying a PA label, its probabilities are about points, and no log line,
no field and no fingerprint said so.

NOTHING IS MIS-SCORED TODAY. Every configured market -- `markets_launch` and
`markets_post_launch` in `config/model_comparison.yaml`, and
`slate_board.DEFAULT_MARKETS` -- is inside `MARKETS_WITH_OWN_FEATURES`, so the
fallback is currently unreachable in production. (I claimed otherwise before
checking, having read `compare.py`'s `POSTGAME_ONLY_COLS` as a market list; it
is a column set.) It is a trap armed for the next market requested: PA, PR, RA
and FPTS each have no rollups, and all four were asked for on 2026-10-25.

The fix does not make PA work — that needs the rollups built. It makes the
substitution SAY SO: a warning naming both markets, and `feature_market`
recorded on the spec so it reaches the `.meta.json`.

RESEARCH_ONLY.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from src.models.feature_spec import (
    FALLBACK_FEATURE_MARKET,
    MARKETS_WITH_OWN_FEATURES,
    FeatureSpec,
    build_prop_feature_spec,
    spec_from_artifact_meta,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("market", sorted(MARKETS_WITH_OWN_FEATURES))
def test_a_market_with_its_own_columns_records_itself(market, caplog):
    """No substitution, so no warning and feature_market == market."""
    with caplog.at_level(logging.WARNING):
        spec = build_prop_feature_spec(market)
    assert spec.market == market
    assert spec.feature_market == market
    assert "rollup columns" not in caplog.text


@pytest.mark.parametrize("market", ["PA", "PR", "RA", "FPTS", "FG3A", "TOV"])
def test_a_market_without_its_own_columns_says_so_loudly(market, caplog):
    """
    THE DEFECT. Each of these silently became a PTS spec. FG3A, TOV and MIN
    are in `compare.py`'s market list today, so this is live behaviour and not
    a hypothetical about combo props nobody has asked for yet.
    """
    with caplog.at_level(logging.WARNING):
        spec = build_prop_feature_spec(market)

    assert spec.market == market
    assert spec.feature_market == "PTS", (
        "the substitution happened but was not recorded, so the spec still "
        "claims to be this market's contract"
    )
    assert market in caplog.text
    assert "PTS" in caplog.text
    assert "NOT about" in caplog.text, (
        "the warning must say the probabilities are not about the asked-for "
        "market; 'using fallback features' does not tell an operator that"
    )


def test_the_substituted_features_really_are_the_other_market_s(caplog):
    """
    Asserted on the COLUMNS, not on the recorded field. A spec could record
    feature_market="PTS" while carrying something else entirely, and every
    assertion about the field would still pass.
    """
    with caplog.at_level(logging.WARNING):
        pa = build_prop_feature_spec("PA")
    pts = build_prop_feature_spec("PTS")

    assert "PTS_L5" in pa.features
    assert "PA_L5" not in pa.features, (
        "a PA column reached the spec; nothing in the panel builds one"
    )
    # The own-stat history is the PTS one, column for column.
    pts_history = [c for c in pts.features if c.startswith("PTS_")]
    assert pts_history
    assert all(c in pa.features for c in pts_history)


def test_the_owned_set_equals_what_the_builder_actually_rolls_forward():
    """
    THE CIRCULARITY THIS FILE FIRST HAD. My original version asserted that
    every configured market is "in" MARKETS_WITH_OWN_FEATURES — but that
    constant is hand-written, so the assertion proved only that I had typed
    the same list twice. The idea for grounding it came from the audit package
    uploaded on 2026-10-25, and it is strictly better: the ground truth is
    `builder.ROLLING_STATS`, the stats the panel actually rolls forward.

    It immediately found a disagreement: MIN was in ROLLING_STATS and absent
    from the owned set, so a MIN spec substituted PTS features for a market
    the builder can serve perfectly well.

    The constant is not imported from `builder` on purpose — that module pulls
    in the whole features package and this one is on every adapter's import
    path — so this test is what keeps them identical.
    """
    from src.features.builder import ROLLING_STATS

    assert set(ROLLING_STATS) == set(MARKETS_WITH_OWN_FEATURES), (
        f"the owned set and the builder's ROLLING_STATS have drifted. "
        f"only in ROLLING_STATS: {sorted(set(ROLLING_STATS) - MARKETS_WITH_OWN_FEATURES)}; "
        f"only in the owned set: {sorted(MARKETS_WITH_OWN_FEATURES - set(ROLLING_STATS))}"
    )


@pytest.mark.parametrize("market", sorted(MARKETS_WITH_OWN_FEATURES))
def test_the_builder_really_does_produce_each_owned_market_s_columns(market):
    """
    Not "it is in a tuple" — the builder is RUN, on a panel carrying that
    stat, and the own-stat columns are looked for in the output. A market in
    the owned set whose columns the builder cannot build would make the
    substitution's absence a lie.

    PRA is the documented exception: the builder refuses to roll a raw PRA
    column (two definitions existed and the raw one lost) and derives it from
    PTS+REB+AST, so its panel carries the components.
    """
    import pandas as pd

    from src.features.builder import build_feature_matrix

    frame = {
        "PLAYER_ID": ["1"] * 5,
        "GAME_DATE": pd.date_range("2026-10-01", periods=5),
    }
    if market == "PRA":
        frame.update({"PTS": [10.0, 12, 14, 16, 18],
                      "REB": [4.0, 5, 6, 7, 8],
                      "AST": [2.0, 3, 4, 5, 6]})
    else:
        frame[market] = [10.0, 12, 14, 16, 18]

    built = set(build_feature_matrix(pd.DataFrame(frame)).columns)
    for suffix in ("_L2", "_L5", "_L10"):
        assert f"{market}{suffix}" in built, (
            f"{market} is in MARKETS_WITH_OWN_FEATURES but the builder did "
            f"not produce {market}{suffix} from a panel carrying {market}"
        )


def test_no_configured_market_is_being_substituted():
    """The live question, kept now that the set itself is grounded above."""
    import yaml

    from src.models.labels import LAUNCH_MARKETS, POST_LAUNCH_MARKETS
    from src.pipeline.slate_board import DEFAULT_MARKETS

    cfg = yaml.safe_load(
        (ROOT / "config" / "model_comparison.yaml").read_text(encoding="utf-8")
    )
    configured = (
        set(cfg.get("markets_launch") or [])
        | set(cfg.get("markets_post_launch") or [])
        | set(LAUNCH_MARKETS)
        | set(POST_LAUNCH_MARKETS)
        | set(DEFAULT_MARKETS)
    )
    assert configured, "parsed no configured markets, so this proves nothing"

    substituted = sorted(configured - MARKETS_WITH_OWN_FEATURES)
    assert not substituted, (
        f"these configured markets have no rollup columns and are therefore "
        f"being scored on {FALLBACK_FEATURE_MARKET} features: {substituted}. "
        f"Build the rollups, or take them out of the config — do NOT add them "
        f"to MARKETS_WITH_OWN_FEATURES, which only silences the warning."
    )


def test_the_recorded_market_reaches_the_sidecar():
    """
    THE GAP MY OWN FIX HAD, caught by the uploaded package. `to_meta` is
    hand-written rather than a model dump, so adding the field to the model
    did not put it in the `.meta.json`. The spec knew which market's columns
    it carried and the file beside the booster did not — and that file is the
    one place an operator looks.
    """
    meta = build_prop_feature_spec("FPTS").to_meta()
    assert meta["market"] == "FPTS"
    assert meta["feature_market"] == FALLBACK_FEATURE_MARKET

    own = build_prop_feature_spec("PTS").to_meta()
    assert own["feature_market"] == "PTS"


def test_the_substitution_does_not_mix_two_markets_columns():
    """
    THE SECOND INSTANCE OF THE SAME DEFECT, found by this file's own fingerprint
    test failing. The base list came from the substituted market but the Wave
    5a extras were templated on the asked-for one, so a PA spec held
    PTS_L2..PTS_SEASON next to PA_STREAK_ABOVE, PA_STREAK_BELOW and PA_HOT_Z —
    columns nothing builds for PA. With `available_columns` those three were
    silently dropped; without it `assert_compatible` refused the spec outright.
    """
    pa = build_prop_feature_spec("PA")
    leaked = [c for c in pa.features if c.startswith("PA_")]
    assert not leaked, (
        f"the spec mixes columns from two markets: {leaked} are templated on "
        f"the asked-for market while the base list came from PTS"
    )


def test_a_fingerprint_still_separates_two_markets_sharing_one_feature_set():
    """
    `feature_market` is deliberately NOT in the fingerprint payload, so adding
    it did not invalidate existing sidecars. That is only safe because
    `market` already is: PA-with-PTS-columns and PTS-with-PTS-columns must
    still hash differently, or verify_feature_contract would accept one in
    place of the other.
    """
    pa = build_prop_feature_spec("PA")
    pts = build_prop_feature_spec("PTS")
    assert pa.features == pts.features, "this test is only meaningful if they match"
    assert pa.fingerprint() != pts.fingerprint()


def test_adding_the_field_did_not_change_an_existing_fingerprint():
    """A sidecar written before this field exists must still verify. If the
    field entered the hash payload, every stored fingerprint would break at
    once and every artifact would start abstaining."""
    spec = FeatureSpec(market="PTS", features=["PTS_L5", "PTS_L10"], categorical=[])
    without = spec.fingerprint()
    with_field = FeatureSpec(
        market="PTS", features=["PTS_L5", "PTS_L10"], categorical=[],
        feature_market="PTS",
    ).fingerprint()
    assert without == with_field


def test_a_sidecar_that_never_recorded_it_reads_empty_not_a_guess():
    """
    The `fold_basis` reasoning from #25: an artifact written before the field
    existed did not answer the question. Defaulting it to `market` would
    answer it wrongly for precisely the specs this field exists to expose — a
    PA artifact would read feature_market="PA" and look correct.
    """
    spec = spec_from_artifact_meta({
        "feature_cols": ["PTS_L5", "PTS_L10"],
        "target_market": "PA",
        "feature_spec": {"name": "prop_over_pa", "version": "x"},
    })
    assert spec.market == "PA"
    assert spec.feature_market == "", (
        "an unrecorded feature_market was filled in from `market`, which "
        "invents the one fact the field exists to carry"
    )


def test_a_sidecar_that_did_record_it_is_believed():
    spec = spec_from_artifact_meta({
        "feature_cols": ["PTS_L5"],
        "target_market": "PA",
        "feature_spec": {"feature_market": "PTS"},
    })
    assert spec.feature_market == "PTS"


# --- the Wave 5a gate ----------------------------------------------------


def test_no_market_s_spec_names_a_wave5a_column_nothing_builds():
    """
    A LIVE DEFECT FOR THREE CONFIGURED MARKETS, flagged in review on
    2026-10-28 and not a hypothetical. The extras were templated for every
    market, but `sports_ev_features.STREAK_STATS` is (PTS, REB, AST, FG3M) and
    `hot_hand.HOT_STATS` adds STL and BLK — so a STL or BLK spec named
    STL_STREAK_ABOVE, and a PRA spec named all three. PRA, STL and BLK are all
    in `markets_post_launch`.

    Which way it broke depended on the caller: `assert_compatible` refused the
    spec outright, while a caller passing `available_columns` silently got a
    narrower feature set than the spec claimed.

    The two sets are NOT the same, which is why the gate has two halves.
    """
    from src.features.hot_hand import HOT_STATS
    from src.features.sports_ev_features import STREAK_STATS

    for market in sorted(MARKETS_WITH_OWN_FEATURES):
        spec = build_prop_feature_spec(market)
        has_streak = [c for c in spec.features if c.endswith(("_STREAK_ABOVE", "_STREAK_BELOW"))]
        has_hot = [c for c in spec.features if c.endswith("_HOT_Z")]

        if market in STREAK_STATS:
            assert has_streak, f"{market} is a streak stat but its spec has no streak columns"
        else:
            assert not has_streak, (
                f"{market}'s spec names {has_streak}, which attach_form_streaks "
                f"does not build — STREAK_STATS is {STREAK_STATS}"
            )

        if market in HOT_STATS:
            assert has_hot, f"{market} is a hot stat but its spec has no HOT_Z"
        else:
            assert not has_hot, (
                f"{market}'s spec names {has_hot}, which hot_hand does not "
                f"build — HOT_STATS is {HOT_STATS}"
            )


def test_the_gate_reads_the_layers_rather_than_restating_them(monkeypatch):
    """
    Behavioural, not a source grep: narrow a layer's own tuple and the spec
    has to follow, or the gate holds a second copy of it.

    STL, NOT PTS. My first version monkeypatched `STREAK_STATS` and checked
    PTS, and it failed — because `labels.default_feature_cols` has a
    per-market table that already lists PTS_HOT_Z / PTS_STREAK_ABOVE, and
    that table is correct: it omits them for PRA, noting "hot_hand covers the
    six single stats and _STREAK_STATS covers four, neither of them the
    combo". So PTS's streak columns come from the BASE list and the gate has
    no say in them.

    STL is the market the gate actually decides: its base list carries no
    Wave 5a column at all, and `HOT_STATS` includes STL, so STL_HOT_Z exists
    only because the gate let it through.
    """
    import src.features.hot_hand as hh

    before = build_prop_feature_spec("STL")
    assert "STL_HOT_Z" in before.features, (
        "this test is only meaningful while the gate is what supplies STL_HOT_Z"
    )

    monkeypatch.setattr(hh, "HOT_STATS", ("PTS",))
    after = build_prop_feature_spec("STL")
    assert "STL_HOT_Z" not in after.features, (
        "STL kept HOT_Z after STL left HOT_STATS, so the gate is reading a "
        "local copy of the layer's stat tuple"
    )
