"""
src/models/ledger_calibration.py — fit a calibration map on the GRADED LEDGER.

WHAT THIS IS FOR, AND WHY IT IS NOT `compare.py`'s CALIBRATION.
`src/models/compare.py` already fits a calibrator, on out-of-fold predictions
made during training. This fits one on something different and better: the
predictions this pipeline actually SERVED, graded against what the players then
did. That is forward evidence in the sense AGENTS.md §O requires — the rows were
scored before the outcome existed, by the artifact that was live at the time —
and until now nothing in this project read it as calibration data.

THE MEASUREMENT THAT MADE THIS POSSIBLE. A calibration map needs graded rows
spread across the probability range. On 2026-11-11 the ledger reached 533 scored
rows with all five taken-side bins populated (`bin_coverage: 1.0`), which is the
first time the fit below could be attempted honestly; at the gate's 100-row
floor it could not.

WHAT IT WILL PROBABLY SAY, STATED UP FRONT SO THE RESULT IS NOT A SURPRISE.
The same report measured a mean stated probability of 0.7077 against an observed
strike rate of 0.4765. A monotone map fitted on that lands most of the range
near 0.48. So a successful fit here does NOT produce an edge — it produces
probabilities that correctly say there isn't one, after which the EV gate clears
nothing and the board goes empty. That is the honest outcome and it will look
like the pipeline doing less.

NOTHING HERE CHANGES WHAT SERVES. The A/B rule: a new component reaches the
serving path by winning a holdout in `scripts/feature_ab.py`, not by being
argued for, and a calibration map is a component. This module MEASURES a
candidate map out-of-fold and reports it. Wiring it in is a separate, deliberate
act with its own evidence.

THE LEAKAGE RULE THIS MODULE EXISTS TO RESPECT. `prob_calibration.choose_calibrator`
splits at a raw row index (`cut = int(n * 0.7)`). On a graded ledger that cuts
INSIDE a slate: part of one night's props land in the fit side and part in the
scoring side, sharing a date, a model artifact and that night's conditions. The
folds here come from `oof.chronological_fold_indices`, which groups by calendar
day and never straddles one — the same guarantee the training OOF already gets.

AND THE ROWS ARE SORTED HERE, DELIBERATELY. `repository.load_graded_prop_results`
issues no ORDER BY, so its row order is whatever Postgres returns. Folding that
order chronologically would be folding an arbitrary permutation while calling it
time. Sorting is not a tidy-up; it is the precondition for the fold to mean
anything.

RESEARCH_ONLY. This measures whether stated probabilities were honest. It
prices nothing, sizes nothing and places nothing.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from src.models.prob_calibration import (
    FULL_PROBABILITY_DOMAIN,
    ProbabilityCalibrator,
    expected_calibration_error,
)
from src.settlement.calibration import (
    TAKEN_SIDE_BINS,
    TAKEN_SIDE_PROBABILITY_DOMAIN,
    taken_side_pairs,
)
from src.utils.timezones import format_pacific_iso, now_pacific

logger = logging.getLogger(__name__)

#: The methods measured. Both come from `prob_calibration`; neither is assumed
#: better, because which one wins depends on the shape of the miscalibration
#: and that is an empirical question per ledger.
CANDIDATE_METHODS: tuple[str, ...] = ("isotonic", "sigmoid")

#: Chronological folds. Four is a compromise: more folds means each validation
#: block is smaller and its Brier noisier, fewer means the earliest fit sees
#: too little. With 533 rows over ~20 slates, four blocks is ~5 slates each.
DEFAULT_SPLITS: int = 4

#: `ProbabilityCalibrator.fit` refuses under 50 rows, so a fold whose training
#: side is smaller than this cannot contribute and is recorded as skipped
#: rather than silently producing nothing.
MIN_FIT_ROWS: int = 50

#: The Brier score of answering 0.5 to everything. Not a tuning parameter — it
#: is (0.5 - y)^2 for y in {0, 1}, which is 0.25 either way. It is here because
#: it is the benchmark that exposed the problem: a probability layer scoring
#: above this is subtracting information rather than adding it, and a
#: calibration map that does not get below it has not earned anything.
CONSTANT_HALF_BRIER: float = 0.25

STATUS_OK = "OK"
STATUS_UNAVAILABLE = "DATA_NOT_AVAILABLE"


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def _discrimination(y: np.ndarray, p: np.ndarray) -> float | None:
    """
    AUC of the RAW probabilities: do they rank winners above losers at all?

    THE NUMBER THAT DECIDES WHETHER CALIBRATION CAN HELP, and the reason it is
    here. Brier confounds two different failures. A model can be badly
    miscalibrated while still ORDERING outcomes well -- it says 0.75 when it
    means 0.52, but the props it is most confident about really do win more
    often -- and for that model a calibration map is a genuine fix: it relabels
    a useful ranking with honest numbers.

    A model whose probabilities do not rank at all is a different case
    entirely, and no map repairs it. Calibration is monotone, so it cannot
    create an ordering that is not already there; shown an uninformative
    score it can only collapse the whole range onto the base rate. The
    collapse is the symptom; AUC is the diagnosis.

    0.5 is "no ranking information". Returns None when the graded rows are all
    wins or all losses, where AUC is undefined -- not 0.5, which would be a
    measurement claimed from nothing.
    """
    y_int = np.asarray(y).astype(int)
    if len(np.unique(y_int)) < 2:
        return None
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y_int, np.asarray(p, dtype=float)))


#: Bins for a calibrated probability, which spans the whole unit interval.
#: Ten, not the taken side's five, because the full domain is twice as wide and
#: five bins over it would be half the resolution of the figure it is set beside.
CALIBRATED_BINS: int = 10


def _ece_of(
    y: np.ndarray,
    p: np.ndarray,
    *,
    min_bin_coverage: float,
    calibrated: bool,
) -> dict[str, Any]:
    """
    ECE through the one gated implementation, on the domain the values live on.

    THE DOMAIN IS NOT THE SAME FOR BOTH, AND THAT MATTERS. A RAW taken-side
    probability is >= 0.5 by construction -- the recorder takes the side it
    thinks more likely -- so it belongs on `TAKEN_SIDE_PROBABILITY_DOMAIN`. A
    CALIBRATED one does not: the entire point of the map is that it can answer
    "this pick is worse than a coin flip", which is a value below 0.5 and lands
    outside that domain.

    Measuring the calibrated values on the taken-side domain -- which the first
    draft of this module did -- excluded every one of them as out-of-domain and
    reported `ece: None`, i.e. no calibrated ECE at all, from a fit that had
    worked perfectly. Caught by the first smoke test against a synthetic
    ledger.

    The consequence is that the two ECEs are NOT directly comparable: different
    domains, different bin widths. `summarise` says so rather than subtracting
    one from the other. The Brier scores ARE comparable, need no domain, and
    are what the choice between methods rests on.
    """
    return expected_calibration_error(
        y, p,
        n_bins=CALIBRATED_BINS if calibrated else TAKEN_SIDE_BINS,
        min_bin_coverage=min_bin_coverage,
        domain=FULL_PROBABILITY_DOMAIN if calibrated else TAKEN_SIDE_PROBABILITY_DOMAIN,
    )


def _out_of_fold_predictions(
    y: np.ndarray,
    p: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    *,
    method: str,
) -> tuple[np.ndarray, list[str]]:
    """
    One genuinely out-of-fold calibrated probability per validated row.

    Returns an array of NaN for rows no fold scored -- with chronological
    folds that is every row before the first validation block, and those rows
    must not be compared against anything, because no map that excluded them
    ever predicted them.
    """
    out = np.full(len(y), np.nan, dtype=float)
    notes: list[str] = []
    for i, (train_idx, val_idx) in enumerate(folds, start=1):
        if len(train_idx) < MIN_FIT_ROWS:
            notes.append(
                f"fold {i}: {len(train_idx)} training row(s), under the "
                f"{MIN_FIT_ROWS} a calibrator needs; not fitted"
            )
            continue
        try:
            cal = ProbabilityCalibrator(method=method).fit(y[train_idx], p[train_idx])
        except ValueError as exc:
            notes.append(f"fold {i}: {exc}")
            continue
        out[val_idx] = cal.transform(p[val_idx])
    return out, notes


def fit_ledger_calibration(
    rows: Iterable[Mapping[str, Any]],
    *,
    splits: int = DEFAULT_SPLITS,
    min_bin_coverage: float = 0.8,
    as_of: str | None = None,
) -> dict[str, Any]:
    """
    Measure a calibration map on the graded ledger, out of fold, and report it.

    Pure: no database, no filesystem, no network, no clock it was not given
    (`as_of`). The 04:00 job already loads these rows for the ECE report, so
    this consumes the same list rather than issuing its own read.

    THREE STATUSES, AND THE THIRD IS THE POINT. `DATA_NOT_AVAILABLE` is
    returned when there are too few rows, too few distinct slates to fold, or
    no fold could be fitted. A calibration verdict asserted on absent evidence
    is worse than no verdict, which is the lesson this repository has learned
    repeatedly: a check that cannot say "I could not tell" eventually reports a
    pass it has not earned.

    The returned dict carries, for each candidate method, the out-of-fold
    Brier and ECE of the CALIBRATED probabilities, alongside the RAW figures
    computed on exactly the same rows -- not on all rows, which would compare
    a map's predictions against a baseline measured over a larger sample.
    """
    stamp = as_of or format_pacific_iso(now_pacific())
    pairs = taken_side_pairs(rows)

    base: dict[str, Any] = {
        "report_timestamp_pt": stamp,
        "rows_considered": pairs.considered,
        "n_scored": len(pairs),
        "excluded_by_reason": dict(pairs.excluded),
        "serving_unchanged": True,
        "note": (
            "A candidate calibration map measured out of fold on served, "
            "graded predictions. Not wired into scoring: a component reaches "
            "the serving path by winning a holdout, not by being measured "
            "once. Not a profitability measure."
        ),
    }

    if len(pairs) < MIN_FIT_ROWS:
        base["status"] = STATUS_UNAVAILABLE
        base["reason"] = (
            f"{len(pairs)} scorable graded row(s), under the {MIN_FIT_ROWS} a "
            f"calibrator needs. A map fitted on fewer is a map fitted on luck."
        )
        return base

    # SORT, BECAUSE THE LOADER DOES NOT. See the module docstring: folding an
    # arbitrary row order chronologically folds a permutation and calls it time.
    day = pd.to_datetime(pd.Series(pairs.dates), errors="coerce").dt.normalize()
    undated = int(day.isna().sum())
    order = np.argsort(day.values, kind="stable")
    y = np.asarray(pairs.outcomes, dtype=float)[order]
    p = np.asarray(pairs.probabilities, dtype=float)[order]
    dates = day.iloc[order].reset_index(drop=True)

    base["undated_rows"] = undated
    base["distinct_slates"] = int(dates.nunique(dropna=True))
    base["first_slate"] = (
        None if dates.dropna().empty else str(dates.dropna().iloc[0].date())
    )
    base["last_slate"] = (
        None if dates.dropna().empty else str(dates.dropna().iloc[-1].date())
    )

    if base["distinct_slates"] < splits + 1:
        base["status"] = STATUS_UNAVAILABLE
        base["reason"] = (
            f"{base['distinct_slates']} distinct slate(s) cannot make "
            f"{splits} date-grouped fold(s). Folding anyway would split a "
            f"slate across the boundary, which is the leakage this fit exists "
            f"to avoid."
        )
        return base

    from src.models.oof import chronological_fold_indices

    folds = chronological_fold_indices(
        len(y), splits=splits, market="ledger_calibration", dates=dates,
    )

    results: dict[str, Any] = {}
    notes: list[str] = []
    for method in CANDIDATE_METHODS:
        oof, method_notes = _out_of_fold_predictions(y, p, folds, method=method)
        notes.extend(method_notes)
        scored = np.isfinite(oof)
        if not scored.any():
            results[method] = {"status": STATUS_UNAVAILABLE,
                               "reason": "no fold produced a prediction"}
            continue
        y_s, p_raw_s, p_cal_s = y[scored], p[scored], oof[scored]
        raw_ece = _ece_of(
            y_s, p_raw_s, min_bin_coverage=min_bin_coverage, calibrated=False,
        )
        cal_ece = _ece_of(
            y_s, p_cal_s, min_bin_coverage=min_bin_coverage, calibrated=True,
        )
        results[method] = {
            "status": STATUS_OK,
            "n_out_of_fold": int(scored.sum()),
            # RAW ON THE SAME ROWS. Comparing a map's out-of-fold predictions
            # against a raw figure measured over ALL rows would credit or blame
            # the map for a different sample.
            "brier_raw": round(_brier(y_s, p_raw_s), 6),
            "brier_calibrated": round(_brier(y_s, p_cal_s), 6),
            # Each ECE names the domain it was measured on, because the two
            # are not interchangeable and a reader who subtracts them is
            # comparing different bin widths over different intervals.
            "ece_raw": raw_ece.get("ece"),
            "ece_raw_domain": list(TAKEN_SIDE_PROBABILITY_DOMAIN),
            "ece_raw_bins": TAKEN_SIDE_BINS,
            "ece_calibrated": cal_ece.get("ece"),
            "ece_calibrated_domain": list(FULL_PROBABILITY_DOMAIN),
            "ece_calibrated_bins": CALIBRATED_BINS,
            "ece_raw_ungated": raw_ece.get("ece_ungated"),
            "ece_calibrated_ungated": cal_ece.get("ece_ungated"),
            "bin_coverage_raw": raw_ece.get("bin_coverage"),
            "bin_coverage_calibrated": cal_ece.get("bin_coverage"),
            "out_of_domain_calibrated": cal_ece.get("out_of_domain"),
            "mean_probability_raw": round(float(p_raw_s.mean()), 6),
            "mean_probability_calibrated": round(float(p_cal_s.mean()), 6),
            "observed_strike_rate": round(float(y_s.mean()), 6),
            # Measured on the RAW probabilities. A monotone map cannot change
            # an ordering, so the calibrated AUC is the same number by
            # construction and reporting it twice would imply otherwise.
            "auc_raw": (
                None if (auc := _discrimination(y_s, p_raw_s)) is None
                else round(auc, 6)
            ),
        }

    usable = {m: r for m, r in results.items() if r.get("status") == STATUS_OK}
    base["methods"] = results
    base["fold_notes"] = notes
    base["constant_half_brier"] = CONSTANT_HALF_BRIER

    if not usable:
        base["status"] = STATUS_UNAVAILABLE
        base["reason"] = (
            "no candidate method produced an out-of-fold prediction; see "
            "fold_notes"
        )
        return base

    chosen = min(usable, key=lambda m: usable[m]["brier_calibrated"])
    best = usable[chosen]
    base["status"] = STATUS_OK
    base["chosen_method"] = chosen
    base["brier_improvement"] = round(
        best["brier_raw"] - best["brier_calibrated"], 6
    )
    # BEATING THE RAW PROBABILITIES IS NOT THE BAR. The raw layer scores worse
    # than answering 0.5 to everything, so a map can improve on it and still
    # be worthless. Both comparisons are reported and neither is called a pass.
    base["beats_constant_half"] = bool(
        best["brier_calibrated"] < CONSTANT_HALF_BRIER
    )
    return base


def summarise(report: Mapping[str, Any]) -> list[str]:
    """
    The report as lines a person reads in a log, with no verdict invented.

    Separate from the fit for the reason `self_check.summarise` is: the shape
    of the dict is what the next reader consumes, and prose baked into it
    cannot be compared, gated or stored.
    """
    status = report.get("status")
    if status != STATUS_OK:
        return [
            f"Ledger calibration [{status}]: {report.get('reason', 'no reason given')}"
        ]
    chosen = str(report.get("chosen_method"))
    best = dict(report.get("methods", {}).get(chosen, {}))
    lines = [
        f"Ledger calibration [{status}]: {chosen} chosen on out-of-fold Brier "
        f"over {best.get('n_out_of_fold')} row(s) from "
        f"{report.get('distinct_slates')} slate(s) "
        f"({report.get('first_slate')} to {report.get('last_slate')}).",
        f"Brier raw {best.get('brier_raw')} -> calibrated "
        f"{best.get('brier_calibrated')} "
        f"(answering 0.5 to everything scores {CONSTANT_HALF_BRIER}).",
        f"ECE raw {best.get('ece_raw')} on {best.get('ece_raw_domain')} in "
        f"{best.get('ece_raw_bins')} bins; calibrated "
        f"{best.get('ece_calibrated')} on {best.get('ece_calibrated_domain')} "
        f"in {best.get('ece_calibrated_bins')} bins. DIFFERENT DOMAINS -- a "
        f"calibrated probability may fall below 0.5 and a raw taken-side one "
        f"cannot, so these two are not one number minus the other. The Brier "
        f"scores above are the comparable pair.",
        f"Mean stated probability {best.get('mean_probability_raw')} -> "
        f"{best.get('mean_probability_calibrated')}, against an observed "
        f"strike rate of {best.get('observed_strike_rate')}.",
    ]
    auc = best.get("auc_raw")
    if auc is None:
        lines.append(
            "Discrimination (AUC) is undefined on these rows: they are all "
            "wins or all losses."
        )
    else:
        lines.append(
            f"Discrimination (AUC) of the raw probabilities: {auc}. 0.5 is no "
            f"ranking information at all. A monotone map cannot invent an "
            f"ordering, so this number -- not the ECE -- is what decides "
            f"whether calibration can help: near 0.5 there is nothing to "
            f"relabel, and the projection is what needs work."
        )
    if not report.get("beats_constant_half"):
        lines.append(
            "THE CALIBRATED PROBABILITIES STILL SCORE NO BETTER THAN A "
            f"CONSTANT {CONSTANT_HALF_BRIER}. Calibrating an uninformative "
            "probability makes it honest, not informative: the map can only "
            "restate the base rate it was shown. This is evidence about the "
            "PROJECTION's edge, not about the map."
        )
    lines.append(
        "Nothing was wired into scoring. A calibration map reaches the "
        "serving path by winning a holdout, not by being measured."
    )
    return lines
