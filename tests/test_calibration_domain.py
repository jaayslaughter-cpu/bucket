"""
The probability domain the publication gate measures coverage over.

THE DEFECT THIS FILE EXISTS FOR, reproduced before it was fixed. The gate
scores the TAKEN-SIDE probability, and `settlement.recorder` records the side
the model thinks is more likely:

    side = SIDE_OVER if prob_over > 0.5 else SIDE_UNDER

so `probability_of_taken_side` returns prob_over above 0.5 and 1 - prob_over
below it. Either way the result EXCEEDS 0.5. Binning that over [0, 1] left
half the bins unreachable by construction: coverage could never exceed 0.5,
the 0.8 requirement could never be met, and no number of graded predictions
would ever have opened the gate. Measured on 1,000 synthetic perfectly
calibrated rows: coverage 0.5, ece_ungated 0.021612 (inside the 0.05 bar),
ece None, WITHHELD.

Reported by an external review, reproduced here against this repository, and
these tests are what keep it fixed.

RESEARCH_ONLY.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from src.models.prob_calibration import (
    FULL_PROBABILITY_DOMAIN,
    expected_calibration_error,
    reliability_table,
)
from src.settlement.calibration import (
    TAKEN_SIDE_PROBABILITY_DOMAIN,
    calibration_from_graded_rows,
    probability_of_taken_side,
)


def _graded(n: int, *, seed: int = 3) -> list[dict]:
    """Rows drawn to be PERFECTLY calibrated, via the recorder's own side rule."""
    random.seed(seed)
    rows = []
    for _ in range(n):
        p_over = random.random()
        side = "OVER" if p_over > 0.5 else "UNDER"
        p_taken = p_over if side == "OVER" else 1.0 - p_over
        rows.append({
            "outcome_status": "WIN" if random.random() < p_taken else "LOSS",
            "prob_over": p_over,
            "predicted_side": side,
            "predicted_line": 20.5,
            "market": "PTS",
            "game_date": "2026-11-01",
        })
    return rows


# --- the contradiction itself ---------------------------------------------


def test_the_recorder_can_only_ever_produce_probabilities_above_one_half():
    """
    THE PREMISE THE DOMAIN RESTS ON. If this ever stops being true, the
    constant below is wrong and the gate's coverage figure becomes misleading
    — so it is asserted rather than assumed.
    """
    for p_over in (0.01, 0.2, 0.49, 0.5001, 0.7, 0.99):
        side = "OVER" if p_over > 0.5 else "UNDER"
        taken = probability_of_taken_side(p_over, side, 20.5)
        assert taken is not None
        assert taken > 0.5, f"prob_over={p_over} gave a taken-side p of {taken}"

    low, high = TAKEN_SIDE_PROBABILITY_DOMAIN
    assert (low, high) == (0.5, 1.0)


def test_a_perfectly_calibrated_thousand_rows_is_now_publishable():
    """
    THE REGRESSION TEST. Before the fix this returned ece=None with
    coverage 0.5 — a perfectly calibrated model, refused forever.
    """
    report = calibration_from_graded_rows(_graded(1000))

    assert report["bin_coverage"] == 1.0, (
        f"coverage {report['bin_coverage']} — the taken-side domain is not "
        f"being binned, so half the bins are unreachable again"
    )
    assert report["ece"] is not None, "a calibrated model was withheld"
    assert report["ece"] <= 0.05
    assert report["ece_gate_passed"] is True


def test_binning_the_full_range_is_what_made_it_unreachable():
    """
    The counterfactual, pinned: the same rows over [0, 1] cannot clear 0.8.
    This is the arithmetic, not a style preference.
    """
    rows = _graded(1000)
    y, p = [], []
    for row in rows:
        taken = probability_of_taken_side(
            row["prob_over"], row["predicted_side"], row["predicted_line"]
        )
        y.append(1 if row["outcome_status"] == "WIN" else 0)
        p.append(taken)

    full = expected_calibration_error(
        np.array(y), np.array(p), domain=FULL_PROBABILITY_DOMAIN
    )
    assert full["bin_coverage"] <= 0.5
    assert full["gate_passed"] is False
    assert full["ece"] is None
    assert full["ece_ungated"] is not None, (
        "the ungated figure must still be reported, or the operator cannot "
        "see that calibration was fine and coverage was the blocker"
    )


# --- what the fix must NOT have done --------------------------------------


def test_the_quality_bar_still_refuses_a_badly_calibrated_model():
    """
    Widening coverage must not have turned the gate off. A model whose
    probabilities are deliberately wrong has to fail on ECE.
    """
    random.seed(11)
    rows = []
    for _ in range(600):
        p_over = random.uniform(0.75, 0.99)   # claims near-certainty
        rows.append({
            "outcome_status": "WIN" if random.random() < 0.5 else "LOSS",  # coin flip
            "prob_over": p_over, "predicted_side": "OVER",
            "predicted_line": 20.5, "market": "PTS", "game_date": "2026-11-01",
        })
    report = calibration_from_graded_rows(rows)

    assert report["ece_ungated"] > 0.05
    assert report["ece"] is None or report["ece"] > 0.05


def test_the_sample_size_requirement_is_untouched():
    from src.quant.publication_gate import DEFAULT_MIN_SCORED

    assert DEFAULT_MIN_SCORED == 100, (
        "the coverage fix must not have moved the sample-size bar"
    )


def test_the_default_domain_is_unchanged_for_every_other_caller():
    """
    `models.compare` scores a RAW P(over) and `quant.paper_calibration` a
    research probability; both legitimately span [0, 1] and neither passes a
    domain. A changed default would silently re-bin model selection.
    """
    assert FULL_PROBABILITY_DOMAIN == (0.0, 1.0)
    y = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 1])
    p = np.linspace(0.05, 0.95, 10)
    table = reliability_table(y, p)
    assert table[0]["probability_bin"].startswith("0.0")
    assert expected_calibration_error(y, p)["probability_domain"] == [0.0, 1.0]


# --- the domain contract ---------------------------------------------------


def test_a_probability_outside_the_domain_is_excluded_and_counted():
    """
    EXCLUDED, NOT CLIPPED. Clipping a 0.30 into the 0.5 bin would move the ECE
    and hide that the domain assumption no longer holds; excluding it keeps
    the figure honest and makes the violation visible.
    """
    y = np.array([1, 0, 1, 0])
    p = np.array([0.6, 0.9, 0.30, 0.10])       # two below the taken-side floor
    out = expected_calibration_error(y, p, domain=TAKEN_SIDE_PROBABILITY_DOMAIN)

    assert out["out_of_domain"] == 2
    for row in out["table"]:
        low = float(row["probability_bin"].split("-")[0])
        assert low >= 0.5

    # THE OBSERVABLE THAT DISTINGUISHES EXCLUDED FROM CLIPPED, and the first
    # version of this test lacked it: clipping also leaves every bin above
    # 0.5 and still counts 2 out of domain, so both assertions above passed
    # against a `np.clip` mutation. What only exclusion gives is a table that
    # WEIGHS two rows, not four -- a clipped 0.30 would land in the bottom
    # bin and drag the ECE with it.
    weighed = sum(row["n_predictions"] for row in out["table"])
    assert weighed == 2, (
        f"{weighed} rows reached the reliability table; the two below the "
        f"domain floor were clipped into it rather than excluded"
    )


def test_an_inverted_domain_is_refused():
    y, p = np.array([1, 0]), np.array([0.6, 0.7])
    with pytest.raises(ValueError, match="increasing"):
        reliability_table(y, p, domain=(1.0, 0.5))


def test_the_settlement_report_says_how_many_rows_the_figure_excludes():
    """
    `scored` counts rows that reached the scorer; the ECE describes only the
    ones inside the taken-side domain. When those two numbers differ the report
    must say so by name, or a figure built on half the evidence would read as a
    figure built on all of it.

    The recorder takes the more likely side (recorder.py:252), so it cannot
    emit an OVER row at 0.30 today. That is exactly why the count has to be
    reported rather than assumed zero: if one appears, the producer changed.
    """
    rows = [
        {"outcome_status": "WIN", "prob_over": 0.80, "predicted_side": "OVER",
         "predicted_line": 24.5},
        {"outcome_status": "LOSS", "prob_over": 0.70, "predicted_side": "OVER",
         "predicted_line": 24.5},
        # Inside the probability range the scorer accepts, OUTSIDE the domain
        # the figure is binned over.
        {"outcome_status": "WIN", "prob_over": 0.30, "predicted_side": "OVER",
         "predicted_line": 24.5},
    ]
    report = calibration_from_graded_rows(rows)

    assert report["n_scored"] == 3, "the scorer should have accepted all three"
    assert report["out_of_domain"] == 1, (
        "the report does not say that one scored row falls outside the domain "
        "its own ECE is binned over"
    )
    weighed = sum(r["n_predictions"] for r in report["reliability_table"])
    assert weighed == 2, (
        f"{weighed} rows reached the reliability table; the sub-domain row was "
        f"clipped into it rather than excluded"
    )


def test_float_drift_at_a_domain_edge_is_not_an_exclusion():
    """
    The pre-change code did `np.clip(p, 0, 1)`, which absorbed a sigmoid's
    1.0000000002 without comment. Replacing clip with exclusion would have
    turned that drift into a DROPPED ROW -- a regression for every caller on
    the default full domain, which is model selection in `models.compare` and
    the research path in `quant.paper_calibration`.

    So drift at an edge is snapped on; a value genuinely outside is excluded.
    """
    y = np.array([1, 1, 0])
    p = np.array([1.0 + 5e-10, 0.0 - 5e-10, 0.60])
    out = expected_calibration_error(y, p)

    assert out["out_of_domain"] == 0, (
        "float drift at the edge of the full probability domain was counted as "
        "out of domain, so a legitimate row would be dropped"
    )
    assert sum(r["n_predictions"] for r in out["table"]) == 3


def test_a_value_genuinely_outside_the_domain_is_counted_and_dropped():
    y = np.array([1, 0, 1])
    p = np.array([0.60, 0.80, 1.5])
    out = expected_calibration_error(y, p)

    assert out["out_of_domain"] == 1
    weighed = sum(r["n_predictions"] for r in out["table"])
    assert weighed == 2, f"{weighed} rows weighed; the 1.5 was clipped in, not excluded"
    # THE COUNT AND THE TABLE MUST ANSWER TO ONE RULE. Before `in_domain` was
    # extracted, the exclusion count and the binning each tested membership
    # their own way, so a tolerance change in one would have made the report
    # claim it excluded a number of rows the table did not agree with.
    assert weighed + out["out_of_domain"] == len(p)
