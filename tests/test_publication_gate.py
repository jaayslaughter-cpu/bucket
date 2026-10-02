"""The gate between computing a model figure and publishing it.

Every refusal here is a case where a number would otherwise have reached a
person looking indistinguishable from a verified one. The reports below are
TEST FIXTURES; no real backtest result is asserted anywhere.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.quant.dfs_payouts import ProbabilitySource
from src.quant.publication_gate import (
    DEFAULT_MAX_ECE,
    PUBLISH_ALLOWED,
    PUBLISH_WITHHELD,
    calibration_gate,
)

NOW = datetime(2026, 9, 28, 12, 0, tzinfo=timezone.utc)


def report(
    *,
    ece: float | None = 0.03,
    n_scored: int = 400,
    age_days: float = 3.0,
    gate_passed: bool = True,
    status: str = "OK",
    **extra,
) -> dict:
    stamp = NOW - timedelta(days=age_days)
    payload = {
        "status": status,
        "n_scored": n_scored,
        "ece": ece,
        "ece_ungated": ece if ece is not None else 0.02,
        "ece_gate_passed": gate_passed,
        "bin_coverage": 0.9 if gate_passed else 0.3,
        "report_timestamp_pt": stamp.isoformat(),
    }
    payload.update(extra)
    return payload


# --- the happy path is narrow and says what it rests on -----------------

def test_a_recent_dense_well_calibrated_report_allows_publication():
    verdict = calibration_gate(report(), now=NOW)
    assert verdict.status == PUBLISH_ALLOWED
    assert verdict.allowed
    assert verdict.ece == pytest.approx(0.03)
    assert verdict.n_scored == 400
    assert set(verdict.checks) >= {
        "evidence_present", "sample_size", "ece_present",
        "ece_within_threshold", "evidence_is_current",
    }
    assert "not a profitability claim" in verdict.reason.lower()


def test_passing_is_never_dressed_up_as_a_profit_claim():
    verdict = calibration_gate(report(), now=NOW)
    text = f"{verdict.reason} {verdict.disclaimer}".lower()
    for forbidden in ("guaranteed", "profitable", "lock", "certain"):
        assert forbidden not in text


# --- the source distinction ---------------------------------------------

def test_a_benchmark_sourced_figure_needs_no_model_backtest():
    """
    Market probabilities do not rest on this model's calibration, so withholding
    them for a missing model backtest would withhold on an irrelevance.
    """
    verdict = calibration_gate(
        None, probability_source=ProbabilitySource.SHARP_BENCHMARK, now=NOW
    )
    assert verdict.status == PUBLISH_ALLOWED
    assert "line matching" in verdict.reason


@pytest.mark.parametrize("source", [
    ProbabilitySource.MODEL,
    ProbabilitySource.MIXED,
    ProbabilitySource.UNSPECIFIED,
])
def test_every_model_dependent_source_is_gated(source):
    """
    MIXED included: every leg has to land, so one model leg puts the model's
    error on the whole card. UNSPECIFIED too — an unrecorded source is not
    evidence of a market-grounded one.
    """
    verdict = calibration_gate(None, probability_source=source, now=NOW)
    assert verdict.status == PUBLISH_WITHHELD
    assert verdict.probability_source is source


# --- unknown is not evidence --------------------------------------------

def test_no_evidence_at_all_withholds():
    verdict = calibration_gate(None, now=NOW)
    assert verdict.status == PUBLISH_WITHHELD
    assert "absent evidence is not evidence" in verdict.reason


def test_a_report_that_could_not_produce_a_figure_passes_its_reason_through():
    unusable = {
        "status": "DATA_NOT_AVAILABLE",
        "reason": "No settled paper bets yet",
    }
    verdict = calibration_gate(unusable, now=NOW)
    assert verdict.status == PUBLISH_WITHHELD
    assert "No settled paper bets yet" in verdict.reason


def test_a_sparse_diagram_is_withheld_and_the_ungated_figure_is_not_used():
    """
    prob_calibration sets ece=None when its own bin-coverage gate fails and puts
    the number in ece_ungated. Reading that here would defeat that gate from
    outside, so a flattering ungated figure must not rescue the entry.
    """
    sparse = report(ece=None, gate_passed=False)
    sparse["ece_ungated"] = 0.001          # would pass easily if it were read
    verdict = calibration_gate(sparse, now=NOW)
    assert verdict.status == PUBLISH_WITHHELD
    assert verdict.ece is None
    assert "too sparse" in verdict.reason


def test_a_small_sample_is_withheld_however_good_the_number_looks():
    verdict = calibration_gate(report(ece=0.001, n_scored=40), now=NOW)
    assert verdict.status == PUBLISH_WITHHELD
    assert "40 graded" in verdict.reason
    assert "luck" in verdict.reason


def test_a_report_that_does_not_say_how_many_predictions_is_withheld():
    anonymous = report()
    anonymous.pop("n_scored")
    verdict = calibration_gate(anonymous, now=NOW)
    assert verdict.status == PUBLISH_WITHHELD
    assert "how many graded predictions" in verdict.reason


def test_an_ece_that_is_not_a_number_is_withheld():
    verdict = calibration_gate(report(ece=float("nan")), now=NOW)
    assert verdict.status == PUBLISH_WITHHELD
    assert "not a number" in verdict.reason


# --- the threshold -------------------------------------------------------

def test_an_ece_above_the_threshold_is_withheld_with_both_numbers():
    verdict = calibration_gate(report(ece=0.12), now=NOW)
    assert verdict.status == PUBLISH_WITHHELD
    assert "0.1200" in verdict.reason
    assert f"{DEFAULT_MAX_ECE:.4f}" in verdict.reason


def test_the_threshold_is_an_argument_not_a_law():
    strict = calibration_gate(report(ece=0.03), max_ece=0.01, now=NOW)
    assert strict.status == PUBLISH_WITHHELD
    lenient = calibration_gate(report(ece=0.03), max_ece=0.10, now=NOW)
    assert lenient.status == PUBLISH_ALLOWED


def test_the_boundary_is_inclusive():
    """Exactly at the threshold passes; a hair above does not."""
    assert calibration_gate(report(ece=0.05), max_ece=0.05, now=NOW).allowed
    assert not calibration_gate(report(ece=0.0500001), max_ece=0.05, now=NOW).allowed


# --- staleness -----------------------------------------------------------

def test_stale_evidence_is_withheld():
    verdict = calibration_gate(report(age_days=120), now=NOW)
    assert verdict.status == PUBLISH_WITHHELD
    assert "120.0 days old" in verdict.reason
    assert verdict.evidence_age_days == pytest.approx(120.0)


def test_undated_evidence_is_withheld_rather_than_assumed_current():
    """A good ECE from a season ago is not evidence about today's rotations."""
    undated = report()
    undated.pop("report_timestamp_pt")
    verdict = calibration_gate(undated, now=NOW)
    assert verdict.status == PUBLISH_WITHHELD
    assert "no usable timestamp" in verdict.reason


def test_an_unparseable_timestamp_is_treated_as_undated():
    verdict = calibration_gate(report(report_timestamp_pt="last tuesday"), now=NOW)
    assert verdict.status == PUBLISH_WITHHELD
    assert "no usable timestamp" in verdict.reason


def test_a_future_dated_report_is_a_fault_not_fresh_evidence():
    """Otherwise the staleness check is bypassable with a wrong clock."""
    verdict = calibration_gate(report(age_days=-5), now=NOW)
    assert verdict.status == PUBLISH_WITHHELD
    assert "future" in verdict.reason


def test_a_trailing_z_timestamp_is_accepted():
    stamp = (NOW - timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    verdict = calibration_gate(report(report_timestamp_pt=stamp), now=NOW)
    assert verdict.status == PUBLISH_ALLOWED


def test_an_explicit_evidence_as_of_key_is_read():
    """A raw expected_calibration_error result carries no report timestamp."""
    raw = {
        "ece": 0.02,
        "gate_passed": True,
        "n_predictions": 500,
        "evidence_as_of": (NOW - timedelta(days=1)).isoformat(),
    }
    verdict = calibration_gate(raw, now=NOW)
    assert verdict.status == PUBLISH_ALLOWED
    assert verdict.n_scored == 500


# --- serialisation -------------------------------------------------------

def test_the_verdict_serialises_the_numbers_a_reader_needs():
    payload = calibration_gate(report(ece=0.12), now=NOW).as_dict()
    assert payload["PUBLICATION_STATUS"] == PUBLISH_WITHHELD
    assert payload["ECE"] == pytest.approx(0.12)
    assert payload["MAX_ECE"] == pytest.approx(DEFAULT_MAX_ECE)
    assert payload["PROBABILITY_SOURCE"] == "MODEL"
    assert payload["REASON"]
