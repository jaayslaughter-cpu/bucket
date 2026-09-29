"""Calibration evidence built from graded prop_results.

This is the link that was missing between grading and the publication gate. The
arithmetic is pure, so none of this needs a database.

NO PLAYS ARE AVAILABLE YET, so the abstention paths are the ones that will
actually run for a while — they get the same attention as the happy path.
"""

from __future__ import annotations

import pytest

from src.settlement.calibration import (
    calibration_from_graded_rows,
    probability_of_taken_side,
    summarise,
)


def row(**kw):
    base = {
        "outcome_status": "WIN",
        "prob_over": 0.60,
        "predicted_side": "OVER",
        "predicted_line": 25.5,
        "market": "PTS",
    }
    base.update(kw)
    return base


# --- P(the side taken) --------------------------------------------------

def test_an_over_row_uses_the_stored_probability_as_is():
    assert probability_of_taken_side(0.62, "OVER", 25.5) == pytest.approx(0.62)


def test_an_under_row_on_a_half_line_takes_the_exact_complement():
    assert probability_of_taken_side(0.62, "UNDER", 25.5) == pytest.approx(0.38)


@pytest.mark.parametrize("line", [25.0, 8.0, None])
def test_an_under_row_on_a_whole_or_unknown_line_is_not_recoverable(line):
    """
    1 - P(over) absorbs the push mass on a line that can tie, and prop_results
    stores no push probability. An unknown line cannot be shown to be a
    half-line, so it counts as able to push — the same rule the other end of the
    pipeline applies.
    """
    assert probability_of_taken_side(0.62, "UNDER", line) is None


@pytest.mark.parametrize("bad", [None, "", "high", float("nan"), 1.5, -0.2])
def test_an_unusable_probability_is_not_recoverable(bad):
    assert probability_of_taken_side(bad, "OVER", 25.5) is None


def test_an_unknown_side_is_not_recoverable():
    assert probability_of_taken_side(0.6, "MIDDLE", 25.5) is None


# --- the report while nothing is graded ---------------------------------

def test_no_rows_at_all_abstains_with_a_reason():
    report = calibration_from_graded_rows([])
    assert report["status"] == "DATA_NOT_AVAILABLE"
    assert report["n_scored"] == 0
    assert "no scorable graded predictions" in report["reason"]
    assert report["ece"] is None if "ece" in report else True


def test_only_pending_rows_abstains_and_says_so():
    """The state today: predictions written, nothing settled."""
    report = calibration_from_graded_rows([row(outcome_status="PENDING")] * 5)
    assert report["status"] == "DATA_NOT_AVAILABLE"
    assert report["excluded_by_reason"] == {"outcome_status=PENDING": 5}
    assert report["rows_considered"] == 5


def test_the_abstaining_report_is_a_shape_the_gate_understands():
    """
    A report the gate cannot read is worse than no report: it would abstain for
    the wrong reason and hide that nothing is settled.
    """
    from src.quant.publication_gate import PUBLISH_WITHHELD, calibration_gate

    report = calibration_from_graded_rows([row(outcome_status="PENDING")])
    verdict = calibration_gate(report)
    assert verdict.status == PUBLISH_WITHHELD
    assert "did not produce a figure" in verdict.reason
    assert "no scorable graded" in verdict.reason


# --- exclusions are counted, not hidden ---------------------------------

def test_voids_and_pushes_are_excluded_and_counted():
    """
    A scratch is not a wrong prediction, and a push has no cell in a two-outcome
    calibration. Both must be visible in the count rather than silently dropped.
    """
    rows = [
        row(outcome_status="WIN"), row(outcome_status="LOSS"),
        row(outcome_status="VOID"), row(outcome_status="PUSH"),
    ]
    report = calibration_from_graded_rows(rows)
    assert report["n_scored"] == 2
    assert report["excluded_by_reason"]["outcome_status=VOID"] == 1
    assert report["excluded_by_reason"]["outcome_status=PUSH"] == 1
    assert report["rows_considered"] == 4


def test_under_rows_on_whole_lines_are_counted_as_their_own_exclusion():
    """A board full of whole lines should be visible, not just a small n."""
    rows = [row(predicted_side="UNDER", predicted_line=25.0) for _ in range(4)]
    report = calibration_from_graded_rows(rows)
    assert report["n_scored"] == 0
    reason = next(iter(report["excluded_by_reason"]))
    assert "whole or unknown line" in reason
    assert report["excluded_by_reason"][reason] == 4


def test_a_market_with_no_stored_probability_is_excluded():
    """PROB_OVER is written only for the market the model was trained for."""
    report = calibration_from_graded_rows([row(prob_over=None, market="REB")])
    assert report["n_scored"] == 0
    assert report["excluded_by_reason"] == {"no model probability stored": 1}


# --- a real measurement --------------------------------------------------

def _spread_rows(n_per_bin: int = 12):
    """Rows spread across the bins, honest at every probability."""
    out = []
    for tenth in range(1, 10):
        p = tenth / 10.0
        wins = round(p * n_per_bin)
        for i in range(n_per_bin):
            out.append(row(
                prob_over=p + 0.005,
                outcome_status="WIN" if i < wins else "LOSS",
            ))
    return out


def test_a_well_calibrated_set_reports_a_small_ece_and_passes_the_gate():
    from src.quant.publication_gate import PUBLISH_ALLOWED, calibration_gate

    report = calibration_from_graded_rows(_spread_rows())
    assert report["status"] == "OK"
    assert report["n_scored"] == 108
    assert report["ece_gate_passed"] is True
    assert report["ece"] < 0.05, report["ece"]
    assert report["reliability_table"]

    verdict = calibration_gate(report, min_scored=100)
    assert verdict.status == PUBLISH_ALLOWED, verdict.reason


def test_a_miscalibrated_set_is_measured_as_such_and_is_withheld():
    """Every row claims 90% and half of them lose."""
    from src.quant.publication_gate import PUBLISH_WITHHELD, calibration_gate

    rows = [
        row(prob_over=0.9, outcome_status="WIN" if i % 2 else "LOSS")
        for i in range(200)
    ]
    report = calibration_from_graded_rows(rows)
    assert report["status"] == "OK"
    assert report["observed_strike_rate"] == pytest.approx(0.5)
    # only one bin is populated, so the coverage gate refuses to stand behind it
    assert report["ece_gate_passed"] is False
    assert report["ece"] is None
    assert report["ece_ungated"] > 0.35

    verdict = calibration_gate(report)
    assert verdict.status == PUBLISH_WITHHELD
    assert "too sparse" in verdict.reason


def test_an_under_row_is_scored_against_the_side_that_was_taken():
    """
    The direction that would be invisible if it were wrong: a 0.30 P(over) row
    recommended UNDER is a 0.70 claim, and a win for the under is a win.
    """
    rows = [row(prob_over=0.30, predicted_side="UNDER", outcome_status="WIN")]
    report = calibration_from_graded_rows(rows)
    assert report["mean_stated_probability"] == pytest.approx(0.70)
    assert report["observed_strike_rate"] == pytest.approx(1.0)


def test_the_report_carries_a_timestamp_the_gate_can_age():
    """Undated evidence is refused by the gate, so the producer must date it."""
    report = calibration_from_graded_rows(_spread_rows())
    assert report["evidence_as_of"]
    assert report["report_timestamp_pt"] == report["evidence_as_of"]

    from src.quant.publication_gate import _parse_timestamp

    assert _parse_timestamp(report["evidence_as_of"]) is not None


def test_summarise_drops_the_bulky_table_only():
    report = calibration_from_graded_rows(_spread_rows())
    small = summarise(report)
    assert "reliability_table" not in small
    assert small["ece"] == report["ece"]
    assert small["n_scored"] == report["n_scored"]


def test_it_is_never_described_as_a_profitability_measure():
    report = calibration_from_graded_rows(_spread_rows())
    assert "not a profitability measure" in report["note"].lower()
