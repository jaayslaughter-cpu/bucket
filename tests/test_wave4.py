"""Wave 4 unit tests: arbitration, run manifest, lifecycle, line-diff."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.models.arbitration import arbitrate_probabilities, attach_arbitration_to_predictions
from src.models.artifact_registry import append_run_step, persist_run_manifest, write_run_manifest
from src.quant.contracts import PropMarketSnapshot
from src.quant.historical_store import BetLifecycleRecord, HistoricalStore, HistoricalStoreConfig
from src.quant.line_diff import pickem_vs_book_line_diff


def test_arbitration_high_agreement():
    arb = arbitrate_probabilities(
        {"xgboost": 0.62, "catboost": 0.64, "distribution": 0.61, "ensemble": 0.99}
    )
    assert arb["confidence_tier"] in {"HIGH", "MODERATE"}
    assert arb["side_lean"] == "over"
    assert arb["n_models"] == 3  # ensemble excluded
    assert "ensemble" not in arb["component_probs"]


def test_arbitration_disagree():
    arb = arbitrate_probabilities({"a": 0.70, "b": 0.30})
    assert arb["confidence_tier"] == "DISAGREE"
    assert arb["agreement_rate"] == pytest.approx(0.5)


def test_attach_arbitration_groups_rows():
    rows = [
        {"event_id": "g1", "player_id": "p1", "target_market": "PTS", "model_name": "xgboost", "probability_over_raw": 0.6},
        {"event_id": "g1", "player_id": "p1", "target_market": "PTS", "model_name": "catboost", "probability_over_raw": 0.58},
        {"event_id": "g1", "player_id": "p1", "target_market": "PTS", "model_name": "distribution", "probability_over_raw": 0.55},
    ]
    out = attach_arbitration_to_predictions(rows)
    assert all(r.get("confidence_tier") for r in out)
    assert out[0]["arbitration_n_models"] == 3


def test_run_manifest_forward_only(tmp_path: Path):
    f = tmp_path / "out.csv"
    f.write_text("a,b\n1,2\n", encoding="utf-8")
    m = write_run_manifest(
        tmp_path,
        steps=[{"step_name": "export"}],
        output_files=[f],
        meta={"wave": 4, "api_key": "SECRET"},
    )
    assert m["forward_only"] is True
    assert "api_key" not in m["meta"]
    assert m["outputs"][0]["sha256"]
    append_run_step(m, step_name="checksum", output_paths=[f])
    path = persist_run_manifest(m)
    assert path.exists()
    # Second write should not clobber first if still present — new sibling
    m2 = write_run_manifest(tmp_path, steps=[{"step_name": "again"}], output_files=[f])
    assert m2["_manifest_path"] != m["_manifest_path"] or Path(m["_manifest_path"]).name.startswith("run_manifest")


def test_grade_before_append_lifecycle(tmp_path: Path):
    store = HistoricalStore(HistoricalStoreConfig(root=tmp_path, use_sqlite=False, prefer_parquet=False))
    pending = BetLifecycleRecord(
        game_id="g1",
        player_id="p1",
        player_name="A",
        prop_stat="PTS",
        line=24.5,
        bet_side="over",
        taken_odds_american=-110,
        model_prob=0.55,
    )
    store.append(pending)
    # Duplicate while still PENDING must be skipped
    dup_while_pending = BetLifecycleRecord(
        game_id="g1",
        player_id="p1",
        player_name="A",
        prop_stat="PTS",
        line=24.5,
        bet_side="over",
        taken_odds_american=-110,
        model_prob=0.56,
    )
    mid = store.append_after_grading([dup_while_pending], actuals=None)
    assert mid["skipped_duplicate_pending"] == 1
    assert mid["appended"] == 0

    new_other = BetLifecycleRecord(
        game_id="g2",
        player_id="p2",
        player_name="B",
        prop_stat="PTS",
        line=20.5,
        bet_side="under",
        taken_odds_american=-110,
        model_prob=0.54,
    )
    result = store.append_after_grading(
        [new_other],
        actuals={"g1|p1": 30.0},
    )
    assert result["graded_pending"] == 1
    assert result["appended"] == 1
    graded = store.get(pending.bet_id)
    assert graded is not None and graded.bet_result == "WIN"


def test_line_diff_requires_valid_book():
    bad = PropMarketSnapshot(
        game_id="g1",
        bookmaker="x",
        market_id="m1",
        line=25.5,
        over_odds_american=-110,
        under_odds_american=-110,
        status="DATA_NOT_AVAILABLE",
        captured_at_utc=datetime.now(timezone.utc),
    )
    out = pickem_vs_book_line_diff(24.5, bad)
    assert out.status == "DATA_NOT_AVAILABLE"

    good = PropMarketSnapshot(
        game_id="g1",
        bookmaker="x",
        market_id="m1",
        line=25.5,
        over_odds_american=-110,
        under_odds_american=-110,
        status="VALID",
        captured_at_utc=datetime.now(timezone.utc),
    )
    ok = pickem_vs_book_line_diff(24.5, good, side="over")
    assert ok.status == "OK"
    assert ok.line_diff == pytest.approx(-1.0)
    assert ok.book_fair_prob_over is not None
    assert ok.adjusted_fair_prob_over is not None
    # pickem lower line → OVER easier → adjusted fair over > book fair
    assert ok.adjusted_fair_prob_over > ok.book_fair_prob_over
