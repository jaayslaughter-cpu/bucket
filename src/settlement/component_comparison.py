"""
Which scoring component was actually more accurate, on the same rows.

WHAT NOTHING ELSE ANSWERS. `config/model_comparison.yaml` ranks components --
0.089 / 0.000 / 0.000 for line_aware, 0.019 / 0.008 / 0.000 for distribution --
but that was fitted against RESEARCH-LINE labels, so it ranks them at "over
the trailing average". Serving asks "over a POSTED line", which is a different
question, and no measurement in this project answers it. `score_prob_over_by_
market` publishes one component and `shadow_probabilities` records the others
on the same rows at the same lines; this is the arithmetic that reads them
back once those rows have graded.

OVER-HIT, NOT THE TAKEN SIDE. The settlement calibration report scores the
probability of the side the recorder took, which lives in (0.5, 1.0]. That is
the right measure for the publication gate and the wrong one here: a shadow
never took a side, and scoring each component on its own side would compare
two different outcomes. Every component is scored on P(OVER) against whether
the over actually hit, which is one outcome and the full [0, 1] domain.

A PUSH IS NOT A RESULT. ``actual_result == predicted_line`` is excluded rather
than counted either way, because calling it a loss for the over and a win for
the under would score the same row twice in opposite directions.

THE PAIRED SET IS THE COMPARISON. Component A on 300 rows against component B
on 180 is not a head-to-head -- the one that abstains more looks better by
choosing easier rows. The headline figures are computed over the rows where
EVERY component answered; per-component totals are reported beside them so
the abstention rate is visible rather than hidden.

IT MEASURES; IT PROMOTES NOTHING. A lower Brier here is evidence for a
promotion decision, not the decision, and it is not evidence of profit. The
publication gate is unchanged and unread by this module.

RESEARCH_ONLY.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from src.models.prob_calibration import expected_calibration_error
from src.quant.publication_gate import DEFAULT_MIN_SCORED

logger = logging.getLogger(__name__)

#: The served probability is stored under this name in a decision record row.
SERVED_COMPONENT = "served"

#: Below this many PAIRED rows the comparison reports its count and withholds
#: a winner. Taken from the publication gate rather than restated, so "enough
#: graded evidence to say something" has one definition in this project.
MIN_ROWS_FOR_A_VERDICT = DEFAULT_MIN_SCORED

#: Scored outcomes. A VOID or PENDING row is not evidence about a forecast.
SCORABLE = frozenset({"WIN", "LOSS", "PUSH"})


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def row_key(player_name: Any, market: Any, game_date: Any) -> tuple[str, str, str]:
    """
    The join key, normalised ONCE for both sides.

    A decision record carries `game_date` as whatever JSON made of a Timestamp
    ("2026-11-09 00:00:00"); the ledger carries a `date`. Normalising in two
    places is how a join silently matches nothing and reports an empty
    comparison as "no graded rows yet".
    """
    return (
        str(player_name or "").strip(),
        str(market or "").strip().upper(),
        str(game_date or "").strip()[:10],
    )


def over_hit(actual_result: Any, predicted_line: Any) -> float | None:
    """
    1.0 if the over hit, 0.0 if it did not, None for a push or missing data.

    ONE DEFINITION OF THE OUTCOME, derived from the realised stat and the line
    rather than from `predicted_side` and `outcome_status`. Those two encode
    the same fact through the recorder's side choice, and reading the outcome
    back out of them would make a shadow's score depend on which side the
    SERVED component happened to take.
    """
    actual, line = _finite(actual_result), _finite(predicted_line)
    if actual is None or line is None:
        return None
    if actual == line:
        return None
    return 1.0 if actual > line else 0.0


def _probabilities_of(record_row: dict[str, Any]) -> dict[str, float]:
    """Every component's P(over) for one recorded row, served one included."""
    out: dict[str, float] = {}
    served = _finite(record_row.get("prob_over"))
    if served is not None:
        out[SERVED_COMPONENT] = served
    for component, value in (record_row.get("shadow") or {}).items():
        number = _finite(value)
        if number is not None:
            out[str(component)] = number
    return out


def _score(probabilities: list[float], outcomes: list[float], *, n_bins: int) -> dict[str, Any]:
    """Brier and ECE for one component over one row set."""
    if not probabilities:
        return {"n": 0, "brier": None, "ece": None, "status": "NO_ROWS"}
    n = len(probabilities)
    brier = sum((p - y) ** 2 for p, y in zip(probabilities, outcomes, strict=True)) / n
    # FULL DOMAIN, deliberately. These are P(over), not taken-side
    # probabilities, so the (0.5, 1.0] domain `settlement.calibration` uses
    # would exclude every row the model thought was unlikely.
    ece = expected_calibration_error(outcomes, probabilities, n_bins=n_bins)
    return {
        "n": n,
        "brier": round(float(brier), 6),
        "ece": ece.get("ece"),
        "ece_ungated": ece.get("ece_ungated"),
        "bin_coverage": ece.get("bin_coverage"),
        "mean_predicted": round(sum(probabilities) / n, 6),
        "observed_over_rate": round(sum(outcomes) / n, 6),
        "status": "OK",
    }


def compare_components(
    graded_rows: list[dict[str, Any]],
    record_rows: list[dict[str, Any]],
    *,
    n_bins: int = 10,
    as_of: str | None = None,
) -> dict[str, Any]:
    """
    Join graded outcomes to recorded probabilities and score every component.

    Pure: no database, no filesystem, no clock it did not receive.
    """
    base: dict[str, Any] = {
        "as_of": as_of,
        "graded_rows_read": len(graded_rows),
        "record_rows_read": len(record_rows),
        "min_rows_for_a_verdict": MIN_ROWS_FOR_A_VERDICT,
        "note": (
            "P(over) scored against whether the over hit, on the rows every "
            "component answered. A lower Brier is evidence for a promotion "
            "decision, not the decision, and not a claim of profitability."
        ),
    }

    by_key: dict[tuple[str, str, str], dict[str, float]] = {}
    for row in record_rows:
        if str(row.get("decision") or "") != "RECORDED":
            continue
        probs = _probabilities_of(row)
        if probs:
            by_key[row_key(row.get("player_name"), row.get("market"), row.get("game_date"))] = probs

    excluded: dict[str, int] = {}

    def drop(reason: str) -> None:
        excluded[reason] = excluded.get(reason, 0) + 1

    matched: list[tuple[dict[str, float], float]] = []
    for row in graded_rows:
        if str(row.get("outcome_status") or "").strip().upper() not in SCORABLE:
            drop("not a scored outcome")
            continue
        if row.get("did_not_play"):
            drop("did not play (a scratch is not a forecast error)")
            continue
        key = row_key(row.get("player_name"), row.get("market"), row.get("game_date"))
        probs = by_key.get(key)
        if not probs:
            drop("no decision record for this graded row")
            continue
        outcome = over_hit(row.get("actual_result"), row.get("predicted_line"))
        if outcome is None:
            drop("push or missing result/line")
            continue
        matched.append((probs, outcome))

    base["excluded_by_reason"] = excluded
    base["matched_rows"] = len(matched)
    if not matched:
        base["status"] = "DATA_NOT_AVAILABLE"
        base["reason"] = (
            "no graded row joined a recorded probability. On a new ledger this "
            "is simply day one: rows are recorded when the slate scores them "
            "and graded after the games finish."
        )
        return base

    components = sorted({c for probs, _ in matched for c in probs})
    base["components"] = components

    # Per component, over every row IT answered.
    per_component: dict[str, Any] = {}
    for component in components:
        pairs = [(probs[component], y) for probs, y in matched if component in probs]
        scored = _score([p for p, _ in pairs], [y for _, y in pairs], n_bins=n_bins)
        scored["answered"] = len(pairs)
        scored["abstained"] = len(matched) - len(pairs)
        per_component[component] = scored
    base["per_component"] = per_component

    # The head-to-head: only rows EVERY component answered.
    paired = [(probs, y) for probs, y in matched if all(c in probs for c in components)]
    base["paired_rows"] = len(paired)
    head_to_head = {
        component: _score(
            [probs[component] for probs, _ in paired],
            [y for _, y in paired],
            n_bins=n_bins,
        )
        for component in components
    }
    base["head_to_head"] = head_to_head

    base["status"] = "OK"
    base["findings"] = _findings(head_to_head, len(paired), components)
    return base


def _findings(
    head_to_head: dict[str, Any], paired_rows: int, components: list[str]
) -> list[str]:
    """What the numbers support saying, and nothing beyond it."""
    out: list[str] = []
    if len(components) < 2:
        out.append(
            f"Only {components} answered, so there is nothing to compare. A "
            f"shadow component is recorded by the slate; check that one is "
            f"configured and that its artifact resolves."
        )
        return out
    if paired_rows < MIN_ROWS_FOR_A_VERDICT:
        out.append(
            f"{paired_rows} paired row(s) is below the {MIN_ROWS_FOR_A_VERDICT} "
            f"this project treats as enough graded evidence. The figures above "
            f"are reported; no component is called better."
        )
        return out
    scored = {c: v["brier"] for c, v in head_to_head.items() if v.get("brier") is not None}
    if len(scored) < 2:
        out.append("Fewer than two components produced a Brier score on the paired set.")
        return out
    best = min(scored, key=lambda c: scored[c])
    worst = max(scored, key=lambda c: scored[c])
    out.append(
        f"On {paired_rows} paired row(s), {best} has the lower Brier score "
        f"({scored[best]:.6f} against {worst}'s {scored[worst]:.6f})."
    )
    if abs(scored[best] - scored[worst]) < 0.001:
        out.append(
            "The gap is under 0.001, which is not a difference worth acting "
            "on. Treat these as tied and keep collecting."
        )
    out.append(
        "A Brier score measures how honest the probabilities were. It is not "
        "a profit figure, and switching what serves resets the graded "
        "evidence accumulated under the current component."
    )
    return out
