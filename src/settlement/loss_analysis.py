"""
src/settlement/loss_analysis.py — what the losses are made of.

WHAT THIS ANSWERS THAT NOTHING ELSE DOES. `metrics.get_performance_summary`
counts W-L-P, ROI and CLV; `calibration.calibration_from_graded_rows` scores
whether the probabilities were honest. Neither says WHERE the model was wrong,
and that is the question a season's feedback loop has to answer: is the error
one-directional, is it concentrated in a market, and does it shrink as a
player accumulates games this season?

That last one is the point. The early-season worry is that a projection built
mostly from last season's baseline is wrong in a way that a projection built
on twenty current games is not. This bins the residuals by HOW MANY GAMES THE
PLAYER HAD ALREADY PLAYED and reports the bias in each bin, so the question
stops being a hunch.

IT MEASURES AND RECOMMENDS; IT DOES NOT RETUNE ANYTHING. Two reasons, both
load bearing. First, `models/labels.py` already excluded the whole halflife
shrink family on measurement -- `{M}_HL_SHRINK` correlates 0.987-0.990 with
`{M}_SEASON` on the 214,381-row panel -- so "add shrinkage" is a decision
that was taken, with numbers, and reversing it needs numbers rather than a
loop that edits a feature list. Second, fitting anything to the residuals of
the same rows you then report on has no holdout in it at all; that is how a
feedback loop becomes an overfitting loop.

NOTHING HERE IS A BET, A SIZE OR A RECOMMENDATION TO ACT. It reads a ledger
of graded research predictions and reports arithmetic on them.

PURE: no database, no network, no clock beyond what the caller passes. The
database read lives in `scripts/analyze_losses.py`.

RESEARCH_ONLY.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Iterable, Mapping, Sequence

logger = logging.getLogger(__name__)

#: Prior-games-played bins, as (label, low, high_inclusive). The first three
#: are deliberately narrow: the whole early-season question lives below ten
#: games, and a single 0-9 bucket would average the interesting part away.
DEFAULT_PRIOR_GAME_BINS: tuple[tuple[str, int, int], ...] = (
    ("0-2", 0, 2),
    ("3-5", 3, 5),
    ("6-9", 6, 9),
    ("10-19", 10, 19),
    ("20+", 20, 10_000),
)

#: Below this many residuals a bin's mean bias is noise, and printing it as a
#: number invites someone to act on it. Bins under this report `n` and nothing
#: else. Chosen as the smallest count at which a 1.5-point bias in a
#: ~10-point-sigma market is distinguishable from zero at all -- it is a floor
#: against nonsense, NOT a significance test, and the report says so.
MIN_ROWS_FOR_A_BIAS = 30

GRADED = ("WIN", "LOSS", "PUSH")


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _moments(values: Sequence[float]) -> dict[str, float | None]:
    """Mean signed bias, MAE and RMSE. None for an empty set, never 0.0."""
    if not values:
        return {"n": 0, "mean_bias": None, "mae": None, "rmse": None}
    n = len(values)
    mean = sum(values) / n
    return {
        "n": n,
        "mean_bias": mean,
        "mae": sum(abs(v) for v in values) / n,
        "rmse": math.sqrt(sum(v * v for v in values) / n),
    }


def _reported(values: Sequence[float]) -> dict[str, Any]:
    """
    Moments, but withheld under `MIN_ROWS_FOR_A_BIAS`.

    A bin with four rows has a mean; it does not have a finding. Returning the
    number anyway is how a feedback loop starts chasing noise, so the count
    goes out and the figures do not.
    """
    out = _moments(list(values))
    if out["n"] < MIN_ROWS_FOR_A_BIAS:
        return {
            "n": out["n"],
            "mean_bias": None,
            "mae": None,
            "rmse": None,
            "insufficient": True,
            "needs": MIN_ROWS_FOR_A_BIAS,
        }
    return out


def residual_of(row: Mapping[str, Any]) -> float | None:
    """
    ``actual - model_projection``: positive means the player beat the model.

    None when either side is missing or unusable, which is not a defect: a
    VOID row has no actual, and a row the model never scored has no
    projection. The caller counts those by reason rather than treating a
    missing residual as a zero one.
    """
    actual = _finite(row.get("actual_result"))
    projected = _finite(row.get("model_projection"))
    if actual is None or projected is None:
        return None
    return actual - projected


def bin_for_prior_games(
    count: Any, bins: Sequence[tuple[str, int, int]] = DEFAULT_PRIOR_GAME_BINS
) -> str | None:
    """The label whose range contains this count, or None if it is unknown."""
    value = _finite(count)
    if value is None or value < 0:
        return None
    n = int(value)
    for label, low, high in bins:
        if low <= n <= high:
            return label
    return None


def analyse_losses(
    rows: Iterable[Mapping[str, Any]],
    *,
    prior_games: Mapping[Any, int] | None = None,
    bins: Sequence[tuple[str, int, int]] = DEFAULT_PRIOR_GAME_BINS,
    as_of: str | None = None,
) -> dict[str, Any]:
    """
    One report over graded prop results. See the module docstring for scope.

    ``prior_games`` maps a row key -- ``(nba_player_id, game_date)`` as the
    caller builds it, or None -- to how many games that player had already
    played in that season. Absent, the prior-games section reports that it was
    not measured rather than silently bucketing everything as zero. "Not
    measured" and "zero games" are different facts and the second one is a
    claim.
    """
    considered: list[Mapping[str, Any]] = []
    excluded: dict[str, int] = {}

    def drop(reason: str) -> None:
        excluded[reason] = excluded.get(reason, 0) + 1

    record = {"WIN": 0, "LOSS": 0, "PUSH": 0}
    for row in rows:
        status = str(row.get("outcome_status") or "").strip().upper()
        if status not in GRADED:
            drop(f"not graded ({status or 'blank'})")
            continue
        if row.get("did_not_play"):
            # A DNP is not a loss. `PropResult` records it as VOID for exactly
            # this reason, but a row can carry the flag and a graded status if
            # it was settled before the scratch was known.
            drop("did not play")
            continue
        record[status] += 1
        considered.append(row)

    decided = record["WIN"] + record["LOSS"]
    strike_rate = (record["WIN"] / decided) if decided else None

    residuals: list[float] = []
    by_market: dict[str, list[float]] = {}
    by_side: dict[str, list[float]] = {}
    by_outcome: dict[str, list[float]] = {}
    by_prior: dict[str, list[float]] = {label: [] for label, _lo, _hi in bins}
    prior_unknown = 0

    for row in considered:
        residual = residual_of(row)
        if residual is None:
            drop("no residual (missing actual or projection)")
            continue
        residuals.append(residual)
        market = str(row.get("market") or "?").upper()
        by_market.setdefault(market, []).append(residual)
        side = str(row.get("predicted_side") or "?").upper()
        by_side.setdefault(side, []).append(residual)
        by_outcome.setdefault(
            str(row.get("outcome_status") or "?").upper(), []
        ).append(residual)

        if prior_games is None:
            continue
        key = (row.get("nba_player_id"), row.get("game_date"))
        label = bin_for_prior_games(prior_games.get(key), bins)
        if label is None:
            prior_unknown += 1
        else:
            by_prior[label].append(residual)

    report: dict[str, Any] = {
        "as_of": as_of,
        "n_rows_in": len(considered) + sum(excluded.values()),
        "n_graded": len(considered),
        "n_with_residual": len(residuals),
        "excluded": excluded,
        "record": {**record, "decided": decided, "strike_rate": strike_rate},
        "residuals_overall": _reported(residuals),
        "residuals_by_market": {m: _reported(v) for m, v in sorted(by_market.items())},
        "residuals_by_side": {s: _reported(v) for s, v in sorted(by_side.items())},
        "residuals_by_outcome": {o: _reported(v) for o, v in sorted(by_outcome.items())},
    }

    if prior_games is None:
        report["residuals_by_prior_games"] = {
            "status": "NOT_MEASURED",
            "reason": (
                "no prior-game counts were supplied, so no row could be "
                "binned. This is not the same as every player having played "
                "no games."
            ),
        }
    else:
        report["residuals_by_prior_games"] = {
            "status": "OK",
            "unknown_count": prior_unknown,
            "bins": {label: _reported(by_prior[label]) for label, _lo, _hi in bins},
        }

    report["findings"] = _findings(report)
    return report


def _findings(report: Mapping[str, Any]) -> list[str]:
    """
    Plain sentences a person can act on, or a sentence saying there is nothing
    yet. Every one of these is a statement about THESE rows, not a forecast.
    """
    out: list[str] = []
    overall = report["residuals_overall"]
    if not overall.get("n"):
        return [
            "No graded row carried both an actual result and a model "
            "projection, so there is nothing to analyse. On a freshly trained "
            "model this is simply day one: rows are recorded when the slate "
            "scores them and graded after the games finish."
        ]
    if overall.get("insufficient"):
        return [
            f"{overall['n']} residual(s) is below the {MIN_ROWS_FOR_A_BIAS}-row "
            f"floor this module reports a bias at, so the figures are withheld "
            f"rather than printed as findings."
        ]

    bias = overall["mean_bias"]
    mae = overall["mae"]
    direction = "ABOVE" if bias > 0 else "BELOW"
    out.append(
        f"Across {overall['n']} graded residual(s) the model's mean error is "
        f"{bias:+.2f} and its MAE is {mae:.2f}: players finished {direction} "
        f"the projection on average. A mean much smaller than the MAE means "
        f"the error is noise rather than a standing bias."
    )

    sides = report["residuals_by_side"]
    usable = {s: v for s, v in sides.items() if v.get("mean_bias") is not None}
    if len(usable) == 2 and {"OVER", "UNDER"} <= set(usable):
        over, under = usable["OVER"]["mean_bias"], usable["UNDER"]["mean_bias"]
        if (over > 0) == (under > 0):
            out.append(
                f"Both sides lean the same way (OVER {over:+.2f}, UNDER "
                f"{under:+.2f}), which is a projection bias rather than a "
                f"side-selection problem: the level is off, not the choice."
            )
        else:
            out.append(
                f"The sides lean opposite ways (OVER {over:+.2f}, UNDER "
                f"{under:+.2f}), which is what selection looks like -- the "
                f"model is picking the side its own error favours."
            )

    prior = report["residuals_by_prior_games"]
    if prior.get("status") == "OK":
        usable_bins = {
            label: block["mean_bias"]
            for label, block in prior["bins"].items()
            if block.get("mean_bias") is not None
        }
        if len(usable_bins) >= 2:
            early = [v for k, v in usable_bins.items() if k in ("0-2", "3-5", "6-9")]
            late = [v for k, v in usable_bins.items() if k in ("10-19", "20+")]
            if early and late:
                e, latest = sum(early) / len(early), sum(late) / len(late)
                if abs(e) > abs(latest) * 1.5:
                    out.append(
                        f"Early-season rows carry the larger error (mean "
                        f"{e:+.2f} under ten prior games against {latest:+.2f} "
                        f"at ten or more). That is the pattern a stronger prior "
                        f"on last season would address -- and it is a reason to "
                        f"RE-MEASURE the halflife shrink family on these rows, "
                        f"not to re-add it: it was excluded at r=0.99 against "
                        f"{{M}}_SEASON on the full panel."
                    )
                else:
                    out.append(
                        f"The error does not shrink with games played (mean "
                        f"{e:+.2f} under ten prior games against {latest:+.2f} "
                        f"at ten or more), so these rows give no support for "
                        f"changing the early-season weighting."
                    )
        else:
            out.append(
                "Too few rows per prior-games bin to compare early against "
                "late; the bins' counts are in the report."
            )
    else:
        out.append(
            "Prior-game counts were not supplied, so the early-season question "
            "is unanswered rather than answered negatively."
        )
    return out
