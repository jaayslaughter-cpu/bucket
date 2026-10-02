"""The train/serve feature contract (audit finding R5).

WHAT WAS ACTUALLY UNGUARDED is narrower than R5's wording. Order and dtype are
already enforced on the serving path — ``xgboost_pipeline._matrix`` selects by
``feature_cols`` in order and refuses a column whose values will not parse. What
nothing checked was whether the SIDECAR AND THE ARTIFACT AGREE, which is the
failure that returns confident probabilities from the wrong model.

The round-trip test below trains a real three-tree booster, so the claim that
XGBoost preserves its own column names through save/load is checked rather than
asserted from documentation.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.models.feature_spec import (
    META_KEY,
    FeatureSpec,
    first_order_difference,
    spec_from_artifact_meta,
    verify_feature_contract,
)

COLUMNS = ["PTS_L5", "PTS_L10", "days_rest"]


def meta(cols=COLUMNS, *, market="PTS", spec_cols=None, fingerprint=None) -> dict:
    """A sidecar whose feature_spec block can be made to disagree with itself."""
    spec = FeatureSpec(market=market, features=list(spec_cols or cols))
    block = spec.to_meta()
    if fingerprint is not None:
        block["fingerprint"] = fingerprint
    return {"feature_cols": list(cols), "target_market": market, META_KEY: block}


# --- the two checks ------------------------------------------------------

def test_a_matching_artifact_and_sidecar_pass():
    spec, problem = verify_feature_contract(meta(), COLUMNS)
    assert problem is None
    assert spec.features == COLUMNS
    assert spec.market == "PTS"


def test_a_sidecar_that_contradicts_its_own_fingerprint_is_refused():
    """A half-written or hand-edited sidecar is not a contract."""
    _, problem = verify_feature_contract(meta(fingerprint="deadbeefdeadbeef"), COLUMNS)
    assert problem is not None
    assert "contradicts its own fingerprint" in problem
    assert "deadbeefdeadbeef" in problem


def test_a_reordered_artifact_is_refused_even_though_the_columns_match():
    """
    The silent-garbage case, and the one a set comparison calls equal. Same three
    columns, different order: every row would be scored against the wrong feature.
    """
    _, problem = verify_feature_contract(meta(), ["PTS_L10", "PTS_L5", "days_rest"])
    assert problem is not None
    assert "different ORDER" in problem
    assert "position 0" in problem


def test_an_artifact_missing_a_trained_column_is_refused():
    _, problem = verify_feature_contract(meta(), ["PTS_L5", "PTS_L10"])
    assert problem is not None
    assert "absent from the artifact" in problem
    assert "days_rest" in problem


def test_an_artifact_carrying_an_extra_column_is_refused():
    _, problem = verify_feature_contract(meta(), [*COLUMNS, "SOMETHING_ELSE"])
    assert problem is not None
    assert "present only in the artifact" in problem


def test_an_empty_sidecar_has_no_contract_to_check():
    _, problem = verify_feature_contract({"feature_cols": []}, COLUMNS)
    assert problem is not None
    assert "no feature_cols" in problem


# --- migration: an older sidecar must not be refused wholesale ----------

def test_a_sidecar_predating_this_check_still_loads():
    """
    Refusing every previously trained model would be a migration, not a guard.
    An old sidecar has no feature_spec block and no fingerprint to contradict.
    """
    old = {"feature_cols": COLUMNS, "target_market": "PTS"}
    spec, problem = verify_feature_contract(old, COLUMNS)
    assert problem is None
    assert spec.features == COLUMNS


def test_an_artifact_that_cannot_name_its_columns_is_not_treated_as_a_mismatch():
    """
    A booster that does not report feature names says nothing either way, and
    "could not ask" is not "they disagree".
    """
    _, problem = verify_feature_contract(meta(), None)
    assert problem is None


# --- the fingerprint itself ---------------------------------------------

def test_the_fingerprint_changes_with_order_not_only_with_membership():
    a = FeatureSpec(market="PTS", features=["a", "b", "c"])
    b = FeatureSpec(market="PTS", features=["a", "c", "b"])
    assert a.fingerprint() != b.fingerprint()


def test_the_fingerprint_changes_with_the_market():
    a = FeatureSpec(market="PTS", features=["a", "b"])
    b = FeatureSpec(market="REB", features=["a", "b"])
    assert a.fingerprint() != b.fingerprint()


def test_the_spec_is_rebuilt_from_feature_cols_not_from_the_stored_copy():
    """
    Reading the stored features instead would verify the sidecar against itself
    in the one way that cannot fail. feature_cols is the list the pipeline
    actually selects with, so it is the list worth hashing.
    """
    payload = meta(cols=COLUMNS, spec_cols=["WRONG"])
    spec = spec_from_artifact_meta(payload)
    assert spec.features == COLUMNS
    # and the disagreement is caught, because the stored fingerprint was
    # computed over ["WRONG"]
    _, problem = verify_feature_contract(payload, COLUMNS)
    assert problem is not None
    assert "contradicts its own fingerprint" in problem


# --- first_order_difference ---------------------------------------------

def test_identical_lists_have_no_difference():
    assert first_order_difference(["a", "b"], ["a", "b"]) is None


def test_a_difference_names_the_first_offending_position():
    out = first_order_difference(["a", "b", "c"], ["a", "c", "b"])
    assert "position 1" in out


def test_set_differences_are_reported_before_order():
    """Naming a reorder when a column is missing entirely would mislead."""
    out = first_order_difference(["a", "b"], ["b"])
    assert "absent from the artifact" in out
    assert "ORDER" not in out


# --- a real booster round trip ------------------------------------------

def test_xgboost_preserves_its_column_names_through_save_and_load(tmp_path):
    """
    The premise the artifact-side check rests on, verified rather than cited.
    If this ever stops holding, the cross-check silently becomes a no-op and
    every other test here would still pass.
    """
    xgb = pytest.importorskip("xgboost")

    rng = np.random.default_rng(7)
    X = pd.DataFrame({c: rng.random(60) for c in COLUMNS})
    y = (X["PTS_L5"] > 0.5).astype(int)

    model = xgb.XGBClassifier(n_estimators=3, max_depth=2, verbosity=0)
    model.fit(X, y)
    path = tmp_path / "m.json"
    model.save_model(str(path))

    reloaded = xgb.XGBClassifier()
    reloaded.load_model(str(path))
    assert list(reloaded.get_booster().feature_names) == COLUMNS


def test_a_saved_adapter_writes_a_verifiable_contract(tmp_path):
    """
    End to end on the real save path: train, save, then verify the sidecar the
    adapter wrote against the booster it wrote beside it.
    """
    xgb = pytest.importorskip("xgboost")
    pytest.importorskip("sklearn")

    from src.models.xgb_adapter import XGBoostAdapter

    rng = np.random.default_rng(11)
    frame = pd.DataFrame({c: rng.random(80) for c in COLUMNS})
    frame["over_hit"] = (frame["PTS_L5"] > 0.5).astype(int)

    adapter = XGBoostAdapter(list(COLUMNS), target_market="PTS")
    adapter.fit(frame)
    target = tmp_path / "xgboost_PTS.json"
    adapter.save(target)

    sidecar = json.loads(target.with_suffix(".meta.json").read_text())
    assert META_KEY in sidecar, "the contract block was not written"
    assert sidecar[META_KEY]["fingerprint"]
    assert sidecar[META_KEY]["market"] == "PTS"

    booster = xgb.XGBClassifier()
    booster.load_model(str(target))
    spec, problem = verify_feature_contract(
        sidecar, booster.get_booster().feature_names
    )
    assert problem is None, problem
    assert spec.fingerprint() == sidecar[META_KEY]["fingerprint"]


def test_a_swapped_sidecar_is_caught_on_the_real_artifact(tmp_path):
    """
    The scenario the whole check exists for: metadata from one fit beside a
    booster from another. Both save() methods already delete a stale MEAN head
    for this reason; the classifier had no equivalent guard.
    """
    xgb = pytest.importorskip("xgboost")
    pytest.importorskip("sklearn")

    from src.models.xgb_adapter import XGBoostAdapter

    rng = np.random.default_rng(13)

    def train(cols, name):
        frame = pd.DataFrame({c: rng.random(80) for c in cols})
        frame["over_hit"] = (frame[cols[0]] > 0.5).astype(int)
        adapter = XGBoostAdapter(list(cols), target_market="PTS")
        adapter.fit(frame)
        path = tmp_path / f"{name}.json"
        adapter.save(path)
        return path

    run_a = train(COLUMNS, "run_a")
    run_b = train([*COLUMNS, "MINUTES_L5"], "run_b")

    # run B's sidecar, run A's booster.
    sidecar_b = json.loads(run_b.with_suffix(".meta.json").read_text())
    booster_a = xgb.XGBClassifier()
    booster_a.load_model(str(run_a))

    _, problem = verify_feature_contract(
        sidecar_b, booster_a.get_booster().feature_names
    )
    assert problem is not None
    assert "different feature" in problem
    assert "MINUTES_L5" in problem


# --- the wiring: score_prob_over must abstain, not score ----------------

def _artifact(tmp_path, cols, name):
    """Train and save a real artifact through the adapter's own save path."""
    pytest.importorskip("sklearn")
    from src.models.xgb_adapter import XGBoostAdapter

    rng = np.random.default_rng(abs(hash(name)) % 2**31)
    frame = pd.DataFrame({c: rng.random(80) for c in cols})
    frame["over_hit"] = (frame[cols[0]] > 0.5).astype(int)
    adapter = XGBoostAdapter(list(cols), target_market="PTS")
    adapter.fit(frame)
    path = tmp_path / f"{name}.json"
    adapter.save(path)
    return path


def _slate_features(cols, n=12):
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({c: rng.random(n) for c in cols})
    # The line the masking step checks the scored line against. Named from the
    # module's own constant: with the wrong name every row abstains on an
    # "unsupported line", and the matching-pair test below would pass for the
    # wrong reason.
    from src.models.labels import RESEARCH_LINE_COL

    frame[RESEARCH_LINE_COL] = 24.5
    frame["PLAYER_NAME"] = [f"DEMO_{i}" for i in range(n)]
    return frame


def test_score_prob_over_abstains_and_names_the_mismatch(tmp_path, caplog):
    """
    THE WIRING, and what it actually buys.

    xgboost would refuse this pair on its own (measured: a permuted or short
    column list raises feature_names mismatch), so the abstention is not the new
    part — the REASON is. Without the check the log carries xgboost's internal
    message, which does not say that a sidecar and an artifact came from
    different fits. This asserts the named reason, not just the NaNs, because
    asserting only the NaNs would pass with the check removed.
    """
    import logging

    pytest.importorskip("xgboost")
    import main

    run_a = _artifact(tmp_path, COLUMNS, "run_a")
    run_b = _artifact(tmp_path, [*COLUMNS, "MINUTES_L5"], "run_b")

    # Put run B's sidecar next to run A's booster — the stale-pair scenario.
    run_a.with_suffix(".meta.json").write_text(
        run_b.with_suffix(".meta.json").read_text()
    )

    features = _slate_features([*COLUMNS, "MINUTES_L5"])
    with caplog.at_level(logging.WARNING):
        scored = main.score_prob_over(
            features, pd.DataFrame([{"player_name": "DEMO_0"}]), run_a,
        )
    assert scored.isna().all()
    assert "different feature" in caplog.text, (
        "abstained, but with a reason that does not identify a mismatched pair"
    )
    assert "MINUTES_L5" in caplog.text
    assert "feature_names mismatch" not in caplog.text, (
        "xgboost's internal message reached the log instead of a named reason"
    )


def test_score_prob_over_still_scores_a_matching_pair(tmp_path):
    """The guard must not refuse the case it exists to permit."""
    pytest.importorskip("xgboost")
    import main

    path = _artifact(tmp_path, COLUMNS, "matching")
    features = _slate_features(COLUMNS)
    scored = main.score_prob_over(
        features, pd.DataFrame([{"player_name": "DEMO_0"}]), path,
    )
    assert scored.notna().any(), "a matching artifact and sidecar must score"
    assert scored.attrs.get("target_market") == "PTS"


def test_score_prob_over_abstains_on_a_sidecar_that_contradicts_itself(tmp_path):
    pytest.importorskip("xgboost")
    import main

    path = _artifact(tmp_path, COLUMNS, "tampered")
    sidecar_path = path.with_suffix(".meta.json")
    sidecar = json.loads(sidecar_path.read_text())
    sidecar[META_KEY]["fingerprint"] = "0000000000000000"
    sidecar_path.write_text(json.dumps(sidecar))

    scored = main.score_prob_over(
        _slate_features(COLUMNS), pd.DataFrame([{"player_name": "DEMO_0"}]), path,
    )
    assert scored.isna().all()
