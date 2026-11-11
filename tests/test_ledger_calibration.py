"""
tests/test_ledger_calibration.py — the calibration map fitted on the ledger.

THE TWO BUGS THESE WERE WRITTEN AFTER, both found by a smoke test against a
synthetic ledger rather than by reasoning:

  1. The calibrated probabilities were measured for ECE on the TAKEN-SIDE
     domain [0.5, 1.0]. A calibrated probability may fall below 0.5 -- that is
     the entire point of the map -- so every one of them was excluded as
     out-of-domain and `ece_calibrated` came back None from a fit that had
     worked perfectly.
  2. The rows were folded in whatever order the loader returned.
     `load_graded_prop_results` issues no ORDER BY, so that is an arbitrary
     permutation, and folding it chronologically folds nothing of the kind.

RESEARCH_ONLY. No wager, no stake, no price.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.ledger_calibration import (
    CONSTANT_HALF_BRIER,
    MIN_FIT_ROWS,
    STATUS_OK,
    STATUS_UNAVAILABLE,
    fit_ledger_calibration,
    summarise,
)

FIRST_SLATE = pd.Timestamp("2026-10-20")


def _rows(
    *,
    slates: int = 24,
    per_slate: int = 22,
    informative: bool = False,
    seed: int = 7,
    side: str = "OVER",
    line: float = 20.5,
) -> list[dict]:
    """
    A ledger shaped like `prop_results`, overconfident by construction.

    `informative=True` makes a higher stated probability genuinely win more
    often, so the suite can tell a map that relabels a real ranking from one
    that collapses noise onto the base rate.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for day in range(slates):
        for _ in range(per_slate):
            p = float(rng.uniform(0.55, 0.92))
            win = (0.35 + 0.35 * (p - 0.55) / 0.37) if informative else 0.477
            rows.append({
                "outcome_status": "WIN" if rng.random() < win else "LOSS",
                "prob_over": p,
                "predicted_side": side,
                "predicted_line": line,
                "game_date": FIRST_SLATE + pd.Timedelta(days=day),
            })
    return rows


# ---------------------------------------------------------------------------
# Bug 1: the domain a calibrated probability lives on.
# ---------------------------------------------------------------------------


def test_a_calibrated_probability_below_half_is_measured_not_discarded() -> None:
    """
    THE FIRST DRAFT REPORTED NO CALIBRATED ECE AT ALL.

    A raw taken-side probability is >= 0.5 by construction: the recorder takes
    the side it thinks more likely. A CALIBRATED one is not, because saying
    "this pick is worse than a coin flip" is a value below 0.5 -- and on this
    ledger that is most of them. Measured on the taken-side domain they were
    all excluded as out-of-domain.

    The assertion is on `out_of_domain_calibrated`, not on the ECE itself: the
    bin-coverage gate may still legitimately withhold a figure when the map
    collapses, and that is a different thing from measuring on the wrong
    interval.
    """
    report = fit_ledger_calibration(_rows())
    assert report["status"] == STATUS_OK, report
    best = report["methods"][report["chosen_method"]]

    assert best["mean_probability_calibrated"] < 0.5, (
        "this ledger observes a sub-coin-flip strike rate, so a working map "
        "must map it below 0.5; if it did not, the fixture stopped exercising "
        f"the bug: {best}"
    )
    assert best["out_of_domain_calibrated"] == 0, (
        f"{best['out_of_domain_calibrated']} calibrated probability(ies) were "
        f"excluded as out-of-domain, which is the taken-side interval being "
        f"applied to values that may legitimately fall below 0.5"
    )
    assert best["ece_calibrated_domain"] == [0.0, 1.0], best
    assert best["ece_raw_domain"] == [0.5, 1.0], best


def test_the_summary_refuses_to_present_the_two_eces_as_subtractable() -> None:
    """
    Different domains, different bin widths, so not one number minus the other.

    A reader handed "ECE raw 0.22 -> calibrated 0.04" would conclude the map
    fixed calibration by that margin. The figures are not comparable, and the
    line says so rather than leaving the arithmetic available.
    """
    text = " ".join(summarise(fit_ledger_calibration(_rows())))
    assert "DIFFERENT DOMAINS" in text, text
    assert "Brier" in text, "the comparable pair is not named"


# ---------------------------------------------------------------------------
# Bug 2: the fold order.
# ---------------------------------------------------------------------------


def test_the_fit_sorts_the_ledger_before_folding_it() -> None:
    """
    `load_graded_prop_results` HAS NO ORDER BY.

    Its row order is whatever Postgres returns, so a chronological fold over
    it folds a permutation and calls it time -- which would put later slates
    in the training side of an earlier fold, the one thing these folds exist
    to prevent. The same rows shuffled must produce the same report.
    """
    rows = _rows()
    shuffled = list(rows)
    np.random.default_rng(11).shuffle(shuffled)
    assert [r["game_date"] for r in shuffled] != [r["game_date"] for r in rows], (
        "the shuffle did not reorder anything, so this proves nothing"
    )

    ordered = fit_ledger_calibration(rows)
    jumbled = fit_ledger_calibration(shuffled)

    assert ordered["first_slate"] == jumbled["first_slate"] == "2026-10-20"
    assert ordered["last_slate"] == jumbled["last_slate"]
    for key in ("brier_raw", "brier_calibrated", "auc_raw", "n_out_of_fold"):
        a = ordered["methods"][ordered["chosen_method"]][key]
        b = jumbled["methods"][jumbled["chosen_method"]][key]
        assert a == pytest.approx(b), (
            f"{key} moved when the input order changed ({a} vs {b}): the fit "
            f"is folding row order rather than calendar time"
        )


# ---------------------------------------------------------------------------
# The diagnosis: can calibration help at all?
# ---------------------------------------------------------------------------


def test_discrimination_separates_a_relabelable_model_from_noise() -> None:
    """
    THE NUMBER THAT ANSWERS "WILL CALIBRATING THIS HELP".

    Calibration is monotone, so it cannot create an ordering that is not
    already there. A model that ranks outcomes while stating the wrong
    magnitudes is repairable by a map; one whose probabilities do not rank at
    all is not, and no amount of calibration changes that. Brier confounds the
    two, AUC separates them, so this asserts the instrument actually
    discriminates rather than trusting that it does.
    """
    noise = fit_ledger_calibration(_rows(informative=False))
    signal = fit_ledger_calibration(_rows(informative=True))

    noise_auc = noise["methods"][noise["chosen_method"]]["auc_raw"]
    signal_auc = signal["methods"][signal["chosen_method"]]["auc_raw"]

    assert noise_auc < 0.56, f"noise ranked too well to be noise: {noise_auc}"
    assert signal_auc > 0.60, f"signal did not rank: {signal_auc}"
    assert signal_auc > noise_auc + 0.08, (noise_auc, signal_auc)

    # And the consequence the AUC predicts: only the rankable one gets under
    # the score you achieve by answering 0.5 to everything.
    assert noise["beats_constant_half"] is False, noise
    assert signal["beats_constant_half"] is True, signal


def test_auc_is_none_rather_than_a_half_when_every_row_won() -> None:
    """
    AUC is undefined with one class present, and None says so.

    Returning 0.5 would be a measurement claimed from nothing -- the same
    defect as reading an absent correlation as zero, which this repository has
    now shipped twice.
    """
    rows = _rows()
    for row in rows:
        row["outcome_status"] = "WIN"
    report = fit_ledger_calibration(rows)

    best = report["methods"].get(report.get("chosen_method", ""), {})
    assert best.get("auc_raw") is None, best
    assert "undefined" in " ".join(summarise(report)).lower()


def test_a_collapsed_map_is_reported_as_evidence_about_the_projection() -> None:
    """
    The honest reading of a map that flattens: the PROJECTION has no edge.

    This is the result the live ledger is expected to produce, and the summary
    must not let it read as a calibration success. Calibrating an
    uninformative probability makes it honest, not informative.
    """
    report = fit_ledger_calibration(_rows(informative=False))
    lines = " ".join(summarise(report))

    assert "NO BETTER THAN A CONSTANT" in lines, lines
    assert "PROJECTION" in lines, lines
    assert "wired into scoring" in lines, "the A/B boundary is not stated"
    assert report["serving_unchanged"] is True


# ---------------------------------------------------------------------------
# Abstention, which is the third status and the point of having one.
# ---------------------------------------------------------------------------


def test_too_few_rows_abstains_with_a_named_reason() -> None:
    report = fit_ledger_calibration(_rows(slates=2, per_slate=4))
    assert report["status"] == STATUS_UNAVAILABLE
    assert str(MIN_FIT_ROWS) in report["reason"], report["reason"]
    assert "methods" not in report, "a verdict was built on too little evidence"


def test_too_few_slates_abstains_rather_than_straddling_one() -> None:
    """
    Enough ROWS, not enough DAYS.

    120 rows over three slates is past the row floor and cannot make four
    date-grouped folds. Folding anyway would split a slate across the
    boundary, which is precisely the leakage this fit exists to avoid, so it
    abstains and names that.
    """
    report = fit_ledger_calibration(_rows(slates=3, per_slate=40))
    assert report["n_scored"] >= MIN_FIT_ROWS, report["n_scored"]
    assert report["status"] == STATUS_UNAVAILABLE
    assert "slate" in report["reason"], report["reason"]
    assert report["distinct_slates"] == 3


def test_an_empty_ledger_abstains_and_counts_nothing() -> None:
    report = fit_ledger_calibration([])
    assert report["status"] == STATUS_UNAVAILABLE
    assert report["n_scored"] == 0
    assert report["rows_considered"] == 0


def test_unscorable_rows_are_counted_by_reason_not_dropped() -> None:
    """
    A figure over a third of its input must not report the whole count.

    The exclusions come from `settlement.calibration.taken_side_pairs`, which
    is the single reader of the `prop_results` contract; this asserts the fit
    surfaces them rather than quietly shrinking its sample.
    """
    rows = _rows()
    for row in rows[:40]:
        row["outcome_status"] = "VOID"
    for row in rows[40:60]:
        row["prob_over"] = None
    report = fit_ledger_calibration(rows)

    assert report["rows_considered"] == len(rows)
    assert report["n_scored"] == len(rows) - 60
    assert report["excluded_by_reason"]["outcome_status=VOID"] == 40
    assert report["excluded_by_reason"]["no model probability stored"] == 20


# ---------------------------------------------------------------------------
# The baseline, and what it is compared on.
# ---------------------------------------------------------------------------


def test_the_raw_baseline_is_measured_on_the_out_of_fold_rows_only() -> None:
    """
    SAME ROWS, OR THE COMPARISON IS A DIFFERENT SAMPLE.

    Chronological folds never score the earliest block, so the out-of-fold set
    is a strict subset of the ledger. A raw Brier measured over ALL rows and
    set beside a calibrated Brier measured over that subset credits or blames
    the map for rows it never saw.
    """
    rows = _rows()
    report = fit_ledger_calibration(rows)
    best = report["methods"][report["chosen_method"]]

    assert best["n_out_of_fold"] < report["n_scored"], (
        "every row was scored out of fold, so the earliest training block was "
        "also validated somewhere -- the folds are not chronological"
    )

    # The raw figure must be the one computed on the out-of-fold subset, which
    # in general differs from the whole-ledger figure.
    from src.settlement.calibration import taken_side_pairs

    pairs = taken_side_pairs(rows)
    p_all = np.asarray(pairs.probabilities)
    y_all = np.asarray(pairs.outcomes)
    whole = float(np.mean((p_all - y_all) ** 2))
    # TOLERANCE MATTERS HERE, AND THE FIRST VERSION GOT IT WRONG. `brier_raw`
    # is rounded to six places, so `!= approx(whole, abs=1e-9)` was satisfied
    # by the rounding alone and passed even when the baseline WAS the whole
    # ledger -- a mutation replacing the out-of-fold subset with every row
    # survived it. The gap must exceed what rounding can explain.
    assert abs(best["brier_raw"] - whole) > 1e-4, (
        f"the raw baseline ({best['brier_raw']}) is the whole-ledger Brier "
        f"({whole}) to within rounding, so it was not restricted to the rows "
        f"the map actually predicted"
    )


def test_the_method_chosen_is_the_one_with_the_better_brier() -> None:
    """
    MUTATION THAT SURVIVED THE FIRST ATTEMPT: `min` -> `max` on the chooser.

    Every other assertion here holds for either method, so the fit could have
    been selecting the WORSE of the two out-of-fold and nothing noticed. The
    choice is the one decision this module makes; it needs its own assertion.
    """
    report = fit_ledger_calibration(_rows())
    chosen = report["chosen_method"]
    usable = {
        m: r for m, r in report["methods"].items() if r.get("status") == STATUS_OK
    }
    assert len(usable) >= 2, (
        f"only {sorted(usable)} fitted, so this cannot tell a chooser from a "
        f"coin toss"
    )
    best = min(usable, key=lambda m: usable[m]["brier_calibrated"])
    assert chosen == best, (
        f"chose {chosen} (Brier {usable[chosen]['brier_calibrated']}) over "
        f"{best} (Brier {usable[best]['brier_calibrated']})"
    )


def test_a_fold_too_small_to_fit_is_named_rather_than_skipped() -> None:
    """
    MUTATION THAT SURVIVED: deleting the per-fold row-count guard.

    `ProbabilityCalibrator.fit` raises under 50 rows and the caller catches
    that, so removing the explicit check changes no outcome -- only the
    message. It is kept for the message: "fold 1: 38 training row(s), under
    the 50 a calibrator needs" says what to do about it, where the raised
    `DATA_NOT_AVAILABLE` text does not name the fold. So the note is the
    behaviour, and it gets an assertion.
    """
    # Five slates is the minimum that folds at all; few enough rows that the
    # earliest training block falls under the floor.
    report = fit_ledger_calibration(_rows(slates=6, per_slate=11))
    notes = " ".join(report.get("fold_notes", []))

    assert "fold 1" in notes, (
        f"the first fold's training side is too small to fit and nothing said "
        f"so: {report.get('fold_notes')}"
    )
    assert str(MIN_FIT_ROWS) in notes, notes
    assert "training row(s)" in notes, notes


def test_the_constant_half_benchmark_is_the_arithmetic_value() -> None:
    """0.25 is (0.5 - y)^2 for y in {0,1}, not a tunable threshold."""
    assert CONSTANT_HALF_BRIER == 0.25
    for y in (0.0, 1.0):
        assert (0.5 - y) ** 2 == pytest.approx(CONSTANT_HALF_BRIER)
    assert fit_ledger_calibration(_rows())["constant_half_brier"] == 0.25


# ---------------------------------------------------------------------------
# The job that runs it.
# ---------------------------------------------------------------------------


def test_a_failing_fit_does_not_cost_the_report_the_gate_reads(
    monkeypatch, tmp_path,
) -> None:
    """
    THE GATE MUST SURVIVE A BROKEN MEASUREMENT.

    `calibration.json` is what `publication_gate.calibration_gate` reads to
    decide whether anything may be published, and it is written BEFORE the
    ledger fit runs. A fit that raises must therefore cost nothing: not the
    report, not the gate, not the scheduler.
    """
    import scheduler_worker as worker

    def boom(*_a, **_k):
        raise RuntimeError("the fit exploded")

    monkeypatch.setattr(
        "src.models.ledger_calibration.fit_ledger_calibration", boom,
    )
    out = worker._run_ledger_calibration(_rows(), tmp_path / "calibration.json")

    assert out["status"] == "FAILED"
    assert "the fit exploded" in out["error"]
    assert out["out"].endswith("ledger_calibration.json")


def test_the_job_writes_the_report_and_names_the_chosen_method(
    tmp_path,
) -> None:
    """The happy path, through the real job body rather than the unit."""
    import json

    import scheduler_worker as worker

    out = worker._run_ledger_calibration(_rows(), tmp_path / "calibration.json")

    assert out["status"] == STATUS_OK, out
    assert out["chosen_method"] in ("isotonic", "sigmoid"), out
    assert out["beats_constant_half"] is False, (
        "a noise ledger scored better than a constant 0.5; the fixture no "
        "longer represents the live evidence"
    )
    written = json.loads((tmp_path / "ledger_calibration.json").read_text())
    assert written["status"] == STATUS_OK
    assert written["serving_unchanged"] is True
