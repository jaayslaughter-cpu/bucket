"""Calibration evidence from graded ``prop_results``. RESEARCH_ONLY.

THE MISSING LINK IN THE FEEDBACK LOOP. ``settlement/recorder.py`` writes PENDING
predictions, ``settlement/runner.py`` grades them against box scores, and
``settlement/metrics.py`` aggregates W/L/PUSH out of the result. Nothing turned
those graded rows into a CALIBRATION report, so ``quant.publication_gate`` — the
thing that decides whether a model-sourced recommendation may be published — had
no producer pointed at the pipeline's own data. It could only be fed by hand from
the manual paper log. This module closes that gap.

WHAT IT MEASURES. For each graded row: did the side we recommended win, and what
probability did we give it. Feed that pair to
``models.prob_calibration.expected_calibration_error`` and the answer is how far
the stated probabilities sat from the observed frequencies.

P(THE SIDE TAKEN) IS NOT ALWAYS RECOVERABLE, AND THE COMPLEMENT IS A TRAP. The
column is ``prob_over``. For an OVER row that is already the taken side's
probability. For an UNDER row it is 1 - prob_over ONLY when the line cannot push:
on a whole line the missing mass is P(push), so 1 - P(over) overstates the under
by exactly that mass. ``prop_results`` stores no push probability, so those rows
CANNOT be scored and are excluded and counted rather than approximated. This is
the same refusal ``paper_research.resolve_two_way_model_probs`` makes at the
other end of the pipeline, for the same reason, and an unknown line counts as
able to push.

WHAT IS EXCLUDED, AND WHY EACH EXCLUSION IS NOT A CHOICE:

  PENDING        no outcome yet. Nothing to compare against.
  VOID           a scratch is not a wrong prediction. Counting it as a loss
                 would understate the model; as a win, overstate it.
  PUSH           the line was hit exactly. A two-outcome calibration has no
                 cell for it, and dropping it is standard — but it is counted
                 in the report so a board full of whole lines is visible.
  no prob_over   a market the scoring model was not trained for carries no
                 probability at all.
  UNDER on a
  whole line     see above. Refused, not approximated.

THE REPORT'S SHAPE IS THE GATE'S INPUT SHAPE. ``status``, ``n_scored``, ``ece``,
``ece_ungated``, ``bin_coverage``, ``ece_gate_passed`` and a timestamp are what
``publication_gate.calibration_gate`` reads. Producing a different shape here
would leave the gate abstaining on a report that exists, which is the most
confusing of the available failures.

NOT A PROFITABILITY MEASURE. A well-calibrated model can still lose money after
vig, and a badly calibrated one can win on variance. This answers "were the
stated probabilities honest", nothing else.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from src.models.prob_calibration import expected_calibration_error, reliability_table
from src.quant.paper_research import is_whole_number_line
from src.utils.timezones import DISPLAY_TZ_NAME, format_pacific_iso, now_pacific

logger = logging.getLogger(__name__)

# Outcomes that carry a two-outcome result. PUSH and VOID deliberately absent.
SCORABLE_OUTCOMES = frozenset({"WIN", "LOSS"})

CALIBRATION_NOTE = (
    "Calibration on graded predictions from prop_results: how far the stated "
    "probabilities sat from the observed frequencies. Not a profitability "
    "measure — a well-calibrated model can still lose after vig."
)


def probability_of_taken_side(
    prob_over: Any, predicted_side: Any, predicted_line: Any
) -> float | None:
    """
    P(the side that was recommended), or None when it cannot be recovered.

    OVER  -> prob_over as stored.
    UNDER -> 1 - prob_over, but ONLY on a half-line. On a whole (or unknown)
             line the complement silently absorbs the push mass, so this
             returns None instead.
    """
    try:
        p_over = float(prob_over)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(p_over) and 0.0 <= p_over <= 1.0):
        return None

    side = str(predicted_side or "").strip().upper()
    if side == "OVER":
        return p_over
    if side != "UNDER":
        return None
    if is_whole_number_line(predicted_line):
        return None
    return 1.0 - p_over


def calibration_from_graded_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    n_bins: int = 10,
    min_bin_coverage: float = 0.8,
    as_of: str | None = None,
) -> dict[str, Any]:
    """
    Build a ``publication_gate``-shaped report from graded prop_results rows.

    Pure: no database, no network. ``rows`` are mappings with ``outcome_status``,
    ``prob_over``, ``predicted_side`` and ``predicted_line`` — the columns
    ``db.models.PropResult`` defines.

    Abstains with ``status`` DATA_NOT_AVAILABLE and a named reason rather than
    returning a figure built on nothing. Every excluded row is counted by reason,
    because a report that silently scored a third of its input would misrepresent
    how much evidence stands behind it.
    """
    stamp = as_of or format_pacific_iso(now_pacific())
    excluded: dict[str, int] = {}
    probabilities: list[float] = []
    outcomes: list[float] = []
    considered = 0

    def drop(reason: str) -> None:
        excluded[reason] = excluded.get(reason, 0) + 1

    for row in rows:
        considered += 1
        status = str(row.get("outcome_status") or "").strip().upper()
        if status not in SCORABLE_OUTCOMES:
            drop(f"outcome_status={status or 'MISSING'}")
            continue
        if row.get("prob_over") is None:
            drop("no model probability stored")
            continue
        p_taken = probability_of_taken_side(
            row.get("prob_over"), row.get("predicted_side"), row.get("predicted_line"),
        )
        if p_taken is None:
            side = str(row.get("predicted_side") or "?").upper()
            if side == "UNDER" and is_whole_number_line(row.get("predicted_line")):
                drop("UNDER on a whole or unknown line (complement absorbs push mass)")
            else:
                drop("probability of the taken side not recoverable")
            continue
        probabilities.append(float(p_taken))
        outcomes.append(1.0 if status == "WIN" else 0.0)

    base: dict[str, Any] = {
        "report_timestamp_pt": stamp,
        "evidence_as_of": stamp,
        "timezone_display": DISPLAY_TZ_NAME,
        "rows_considered": considered,
        "n_scored": len(probabilities),
        "excluded_by_reason": excluded,
        "note": CALIBRATION_NOTE,
        "reliability_table": [],
    }

    if not probabilities:
        base["status"] = "DATA_NOT_AVAILABLE"
        base["reason"] = (
            f"no scorable graded predictions out of {considered} row(s) examined"
            + (f" — excluded: {excluded}" if excluded else "")
        )
        return base

    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    ece = expected_calibration_error(
        y, p, n_bins=n_bins, min_bin_coverage=min_bin_coverage
    )

    base.update({
        "status": "OK",
        "brier": round(float(np.mean((p - y) ** 2)), 6),
        "observed_strike_rate": round(float(y.mean()), 6),
        "mean_stated_probability": round(float(p.mean()), 6),
        # The headline number the gate reads. None when prob_calibration's own
        # bin-coverage gate refused it; `ece_ungated` carries the figure it
        # would not stand behind, and the gate is careful not to read that one.
        "ece": ece.get("ece"),
        "ece_ungated": ece.get("ece_ungated"),
        "bin_coverage": ece.get("bin_coverage"),
        "nonempty_bins": ece.get("nonempty_bins"),
        "n_bins": ece.get("n_bins"),
        "ece_gate_passed": ece.get("gate_passed"),
        "reliability_table": reliability_table(y, p, n_bins=n_bins),
    })
    return base


def prop_result_calibration_report(
    *,
    lookback_days: int | None = 180,
    market: str | None = None,
    n_bins: int = 10,
    min_bin_coverage: float = 0.8,
) -> dict[str, Any]:
    """
    Read graded rows from Postgres and build the report.

    Separated from ``calibration_from_graded_rows`` so the arithmetic is testable
    without a database — which is also the only way it gets tested in an
    environment with no DATABASE_URL.
    """
    from src.db.repository import load_graded_prop_results

    rows = load_graded_prop_results(lookback_days=lookback_days, market=market)
    report = calibration_from_graded_rows(
        rows, n_bins=n_bins, min_bin_coverage=min_bin_coverage,
    )
    report["filters"] = {"lookback_days": lookback_days, "market": market}
    return report


def summarise(report: Mapping[str, Any], *, drop: Sequence[str] = ()) -> dict[str, Any]:
    """The report without its bulky tables, for a log line or a CLI echo."""
    skip = {"reliability_table", *drop}
    return {k: v for k, v in report.items() if k not in skip}
