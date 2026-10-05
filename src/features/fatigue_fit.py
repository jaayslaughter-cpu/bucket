"""Fit the fatigue multipliers instead of asserting them.

RESEARCH_ONLY. Produces four numbers and a report; places nothing, sizes
nothing, and invents no line.

WHY THIS EXISTS. ``features/fatigue_logic.py`` folds a multiplier into EVERY
``{stat}_L2`` the pipeline publishes, and those multipliers are guesses:

    B2B_PENALTY            0.97
    THREE_IN_FOUR_PENALTY  0.96
    FOUR_IN_FIVE_PENALTY   0.94
    ALTITUDE_PENALTY       0.98

``docs/DATA_GAPS.md`` lists them as unfitted heuristics. They are not labelled
uncertain anywhere downstream, so a 3% haircut nobody measured reaches every
projection as though it had been.

TWO METHODOLOGICAL FIXES over the reference implementation this was adapted
from (``PropIQ_JuiceReel_Local``'s ``fatigue_fit.py``; see
``docs/external_repo_review_2026-10.md`` §1.2). Both change the answer.

1. A RATIO OF TOTALS, NOT A MEAN OF RATIOS. ``mean(actual / baseline)`` is
   pulled upward by low-baseline rows: a bench player whose L10 is 2.0 points
   and who scores 6 contributes a ratio of 3.0, and no amount of clipping makes
   that an unbiased estimate of a 3% effect. ``sum(actual) / sum(baseline)``
   weights each row by its own volume, which is what a multiplier applied to a
   projection actually needs. Both are reported so the gap is visible.

2. MEASURED AGAINST THE UNFLAGGED ROWS, NOT AGAINST 1.0. The reference computes
   the B2B multiplier as the mean ratio on B2B rows, which silently assumes the
   ratio on rested rows is exactly 1.0. It is not: a ``shift(1)`` rolling
   baseline lags any trend, so the whole panel's ratio sits off 1.0 and that
   bias would be folded into the fatigue multiplier. The multiplier here is
   ``ratio(flagged) / ratio(unflagged)`` — the fatigue effect net of the
   baseline's own bias.

ABSTENTION. A flag with too few rows, or a panel missing the columns, produces
no multiplier for that flag rather than a number from thin evidence. The
existing constant stays in force and the report says which flags were fitted.
Nothing here edits ``fatigue_logic``; a caller applies the result deliberately.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: The flags fatigue_logic applies, and the constant each currently uses.
FATIGUE_FLAGS: dict[str, str] = {
    "b2b": "is_back_to_back",
    "three_in_four": "is_3_in_4",
    "four_in_five": "is_4_in_5",
}
#: Minimum usable rows overall, and per flag, before a figure is reported.
MIN_ROWS = 2_000
MIN_ROWS_PER_FLAG = 300
#: A fitted multiplier outside this range is reported but not recommended: a
#: fatigue effect beyond it is more likely a data fault than a finding.
PLAUSIBLE_RANGE = (0.85, 1.05)


@dataclass
class FlagFit:
    """One flag's fitted effect, with the numbers behind it."""

    flag: str
    n_flagged: int
    n_unflagged: int
    ratio_flagged: float | None = None
    ratio_unflagged: float | None = None
    multiplier: float | None = None
    mean_of_ratios: float | None = None
    status: str = "OK"
    reason: str | None = None

    @property
    def plausible(self) -> bool:
        if self.multiplier is None:
            return False
        low, high = PLAUSIBLE_RANGE
        return low <= self.multiplier <= high

    def as_dict(self) -> dict[str, Any]:
        return {
            "flag": self.flag,
            "status": self.status,
            "n_flagged": self.n_flagged,
            "n_unflagged": self.n_unflagged,
            "ratio_flagged": _round(self.ratio_flagged),
            "ratio_unflagged": _round(self.ratio_unflagged),
            "multiplier": _round(self.multiplier),
            "mean_of_ratios": _round(self.mean_of_ratios),
            "plausible": self.plausible,
            "reason": self.reason,
        }


@dataclass
class FatigueFitResult:
    """Every flag's fit for one stat, and what could not be fitted."""

    stat: str
    status: str = "OK"
    n_usable_rows: int = 0
    fits: dict[str, FlagFit] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def multipliers(self) -> dict[str, float]:
        """Only the flags that produced a plausible figure."""
        return {
            name: fit.multiplier
            for name, fit in self.fits.items()
            if fit.multiplier is not None and fit.plausible
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "stat": self.stat,
            "status": self.status,
            "n_usable_rows": self.n_usable_rows,
            "multipliers": {k: _round(v) for k, v in self.multipliers.items()},
            "fits": {k: v.as_dict() for k, v in self.fits.items()},
            "notes": list(self.notes),
        }


def _round(value: Any) -> float | None:
    return None if value is None else round(float(value), 4)


def _ratio_of_totals(actual: pd.Series, baseline: pd.Series) -> float | None:
    """Volume-weighted ratio. None when there is no baseline mass to divide by."""
    total_base = float(baseline.sum())
    if not np.isfinite(total_base) or total_base <= 0.0:
        return None
    return float(actual.sum()) / total_base


def fit_fatigue_multipliers(
    panel: pd.DataFrame,
    *,
    stat: str = "PTS",
    baseline_col: str | None = None,
    min_rows: int = MIN_ROWS,
    min_rows_per_flag: int = MIN_ROWS_PER_FLAG,
) -> FatigueFitResult:
    """
    Fit one multiplier per fatigue flag from realised stat against baseline.

    ``baseline_col`` defaults to ``{stat}_L10``, which is ``shift(1)`` rolling
    and so contains no information from the row being measured. Pass
    ``{stat}_BASELINE`` to fit against the blended baseline instead — but note
    that one already has the fatigue multiplier folded in by
    ``build_feature_matrix``, which would make this circular.
    """
    result = FatigueFitResult(stat=stat)
    base_name = baseline_col or f"{stat}_L10"

    if panel is None or getattr(panel, "empty", True):
        result.status = "DATA_NOT_AVAILABLE"
        result.notes.append("empty panel")
        return result
    missing = [c for c in (stat, base_name) if c not in panel.columns]
    if missing:
        result.status = "DATA_NOT_AVAILABLE"
        result.notes.append(f"panel missing {missing}")
        return result
    if baseline_col and baseline_col.endswith("_BASELINE"):
        result.notes.append(
            f"WARNING: {baseline_col} already carries the fatigue multiplier "
            "from build_feature_matrix, so this fit is circular"
        )

    actual = pd.to_numeric(panel[stat], errors="coerce")
    baseline = pd.to_numeric(panel[base_name], errors="coerce")
    usable_mask = actual.notna() & baseline.notna() & (baseline > 0.0)
    work = panel.loc[usable_mask].copy()
    work["_actual"] = actual.loc[usable_mask]
    work["_baseline"] = baseline.loc[usable_mask]
    result.n_usable_rows = len(work)

    if len(work) < int(min_rows):
        result.status = "DATA_NOT_AVAILABLE"
        result.notes.append(
            f"{len(work)} usable row(s), need {int(min_rows)}. A multiplier "
            "applied to every projection should not rest on less."
        )
        return result

    for name, column in FATIGUE_FLAGS.items():
        if column not in work.columns:
            result.fits[name] = FlagFit(
                flag=column, n_flagged=0, n_unflagged=0,
                status="DATA_NOT_AVAILABLE",
                reason=f"panel has no {column} column",
            )
            continue

        flagged = work[column].fillna(False).astype(bool)
        n_flagged, n_unflagged = int(flagged.sum()), int((~flagged).sum())
        fit = FlagFit(flag=column, n_flagged=n_flagged, n_unflagged=n_unflagged)

        if n_flagged < int(min_rows_per_flag) or n_unflagged < int(min_rows_per_flag):
            fit.status = "DATA_NOT_AVAILABLE"
            fit.reason = (
                f"{n_flagged} flagged / {n_unflagged} unflagged row(s), need "
                f"{int(min_rows_per_flag)} of each"
            )
            result.fits[name] = fit
            continue

        on = work.loc[flagged]
        off = work.loc[~flagged]
        fit.ratio_flagged = _ratio_of_totals(on["_actual"], on["_baseline"])
        fit.ratio_unflagged = _ratio_of_totals(off["_actual"], off["_baseline"])
        # Reported for comparison only — this is the reference's statistic, and
        # the gap between it and `multiplier` is the bias described at the top.
        fit.mean_of_ratios = float((on["_actual"] / on["_baseline"]).mean())

        if not fit.ratio_flagged or not fit.ratio_unflagged:
            fit.status = "DATA_NOT_AVAILABLE"
            fit.reason = "no baseline mass in one of the two groups"
            result.fits[name] = fit
            continue

        fit.multiplier = fit.ratio_flagged / fit.ratio_unflagged
        if not fit.plausible:
            fit.status = "IMPLAUSIBLE"
            fit.reason = (
                f"fitted {fit.multiplier:.4f}, outside {PLAUSIBLE_RANGE}. "
                "Reported, not recommended: an effect this large is more likely "
                "a data fault than a finding."
            )
        result.fits[name] = fit

    fitted = result.multipliers
    if not fitted:
        result.status = "DATA_NOT_AVAILABLE"
        result.notes.append("no flag produced a usable multiplier")
    logger.info(
        "Fatigue fit on %s: %d usable row(s); fitted %s",
        stat, result.n_usable_rows,
        {k: round(v, 4) for k, v in fitted.items()} or "nothing",
    )
    return result


def compare_to_constants(
    result: FatigueFitResult,
    constants: Mapping[str, float] | None = None,
) -> list[dict[str, Any]]:
    """
    The fitted figure beside the constant it would replace.

    Returned rather than logged so a caller can print it, write it to a doc, or
    assert on it. Changing ``fatigue_logic``'s constants is a separate,
    deliberate edit — this only says what the data supports.
    """
    if constants is None:
        from src.features.fatigue_logic import (
            B2B_PENALTY,
            FOUR_IN_FIVE_PENALTY,
            THREE_IN_FOUR_PENALTY,
        )

        constants = {
            "b2b": B2B_PENALTY,
            "three_in_four": THREE_IN_FOUR_PENALTY,
            "four_in_five": FOUR_IN_FIVE_PENALTY,
        }

    rows: list[dict[str, Any]] = []
    for name, current in constants.items():
        fit = result.fits.get(name)
        rows.append({
            "flag": name,
            "in_use": float(current),
            "fitted": _round(fit.multiplier) if fit else None,
            "delta": (
                _round(fit.multiplier - float(current))
                if fit and fit.multiplier is not None else None
            ),
            "n_flagged": fit.n_flagged if fit else 0,
            "status": fit.status if fit else "DATA_NOT_AVAILABLE",
        })
    return rows


def save_fatigue_fit(result: FatigueFitResult, path: Path | str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result.as_dict(), indent=2) + "\n", encoding="utf-8")
    return target


def load_fatigue_fit(path: Path | str) -> dict[str, Any] | None:
    target = Path(path)
    if not target.exists():
        return None
    return json.loads(target.read_text(encoding="utf-8"))
