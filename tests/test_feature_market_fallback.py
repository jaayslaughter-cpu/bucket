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


@pytest.mark.parametrize("market", ["PA", "PR", "RA", "FPTS", "FG3A", "TOV", "MIN"])
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


def test_every_market_the_project_is_configured_for_has_its_own_columns():
    """
    THE GUARD THAT MAKES THE FALLBACK SAFE TO LEAVE IN PLACE. The substitution
    is currently unreachable because every configured market has its own
    rollups. Adding PA, PR, RA or FPTS to the config without building the
    rollups first would arm it — and this is where that shows up, rather than
    in a season of points-shaped PA probabilities.
    """
    import yaml

    from src.pipeline.slate_board import DEFAULT_MARKETS

    cfg = yaml.safe_load(
        (ROOT / "config" / "model_comparison.yaml").read_text(encoding="utf-8")
    )
    configured = set(cfg.get("markets_launch") or []) | set(
        cfg.get("markets_post_launch") or []
    )
    configured |= set(DEFAULT_MARKETS)
    assert configured, "parsed no configured markets, so this proves nothing"

    substituted = sorted(configured - MARKETS_WITH_OWN_FEATURES)
    assert not substituted, (
        f"these configured markets have no rollup columns and are therefore "
        f"being scored on PTS features: {substituted}. Build the rollups, or "
        f"take them out of the config — do not add them to "
        f"MARKETS_WITH_OWN_FEATURES, which would only silence the warning."
    )


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
