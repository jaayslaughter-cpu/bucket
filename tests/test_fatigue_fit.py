"""Fitting the fatigue multipliers, and the two statistics that disagree.

`fatigue_logic` folds 0.97 / 0.96 / 0.94 into every `{stat}_L2` this pipeline
publishes, and `docs/DATA_GAPS.md` lists them as unfitted heuristics. This
module measures them instead.

The two fixes over the reference implementation both change the answer, and
each has a test that fails without it:

  1. a ratio of TOTALS, not a mean of ratios — the latter is pulled upward by
     low-baseline rows;
  2. measured against the UNFLAGGED rows, not against 1.0 — otherwise the
     baseline's own bias is folded into the fatigue effect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.fatigue_fit import (
    MIN_ROWS,
    MIN_ROWS_PER_FLAG,
    compare_to_constants,
    fit_fatigue_multipliers,
    load_fatigue_fit,
    save_fatigue_fit,
)


def panel(
    n: int = 6_000,
    *,
    flag_share: float = 0.3,
    flagged_effect: float = 0.90,
    unflagged_effect: float = 1.00,
    baseline: float = 20.0,
    seed: int = 5,
) -> pd.DataFrame:
    """
    A panel whose true effect is known, so the fit can be checked against it.

    ``flagged_effect`` is the real multiplier: actual = baseline x effect.
    """
    rng = np.random.default_rng(seed)
    flagged = rng.random(n) < flag_share
    base = np.full(n, float(baseline))
    effect = np.where(flagged, flagged_effect, unflagged_effect)
    return pd.DataFrame({
        "PTS": base * effect,
        "PTS_L10": base,
        "is_back_to_back": flagged,
        "is_3_in_4": np.zeros(n, dtype=bool),
        "is_4_in_5": np.zeros(n, dtype=bool),
    })


def test_a_known_effect_is_recovered():
    out = fit_fatigue_multipliers(panel(flagged_effect=0.90))
    assert out.status == "OK"
    assert out.fits["b2b"].multiplier == pytest.approx(0.90, abs=1e-6)


def test_no_effect_recovers_one():
    out = fit_fatigue_multipliers(panel(flagged_effect=1.00))
    assert out.fits["b2b"].multiplier == pytest.approx(1.0, abs=1e-6)


def test_the_effect_is_measured_against_the_unflagged_rows_not_against_one():
    """
    FIX 2. Here the BASELINE is biased: every row, flagged or not, outruns its
    L10 by 10%. The fatigue effect is 0.90 relative to peers, and a method that
    compares flagged rows to 1.0 would report 0.99 and call it a 1% effect.
    """
    frame = panel(flagged_effect=0.90 * 1.10, unflagged_effect=1.10)
    out = fit_fatigue_multipliers(frame)
    fit = out.fits["b2b"]
    assert fit.ratio_unflagged == pytest.approx(1.10, abs=1e-6)
    assert fit.multiplier == pytest.approx(0.90, abs=1e-6)
    naive = fit.ratio_flagged
    assert naive == pytest.approx(0.99, abs=1e-6)
    assert abs(naive - 0.90) > 0.08, "the naive statistic should be visibly wrong here"


def test_a_ratio_of_totals_is_not_a_mean_of_ratios():
    """
    FIX 1. One low-baseline row drags the mean of ratios far from the truth; the
    volume-weighted ratio barely moves. The reference used the mean.
    """
    frame = panel(n=4_000, flagged_effect=1.00)
    # a single bench row: L10 of 1.0, scored 8 -> a ratio of 8.0
    frame.loc[frame.index[0], ["PTS", "PTS_L10"]] = [8.0, 1.0]
    frame.loc[frame.index[0], "is_back_to_back"] = True

    fit = fit_fatigue_multipliers(frame).fits["b2b"]
    assert fit.multiplier == pytest.approx(1.0, abs=0.01)
    assert fit.mean_of_ratios > 1.004, (
        "the mean of ratios should be visibly inflated by the bench row"
    )
    assert abs(fit.mean_of_ratios - 1.0) > abs(fit.multiplier - 1.0)


# --- abstention ------------------------------------------------------------

def test_too_few_rows_overall_abstains_with_the_count():
    out = fit_fatigue_multipliers(panel(n=100))
    assert out.status == "DATA_NOT_AVAILABLE"
    assert str(MIN_ROWS) in " ".join(out.notes)
    assert out.multipliers == {}


def test_a_flag_with_too_few_rows_of_its_own_abstains():
    out = fit_fatigue_multipliers(panel(n=6_000, flag_share=0.001))
    fit = out.fits["b2b"]
    assert fit.status == "DATA_NOT_AVAILABLE"
    assert str(MIN_ROWS_PER_FLAG) in (fit.reason or "")
    assert "b2b" not in out.multipliers


def test_a_flag_the_panel_lacks_is_named_rather_than_skipped():
    frame = panel().drop(columns=["is_4_in_5"])
    out = fit_fatigue_multipliers(frame)
    assert out.fits["four_in_five"].status == "DATA_NOT_AVAILABLE"
    assert "is_4_in_5" in (out.fits["four_in_five"].reason or "")


def test_a_missing_stat_or_baseline_column_abstains():
    out = fit_fatigue_multipliers(panel().drop(columns=["PTS_L10"]))
    assert out.status == "DATA_NOT_AVAILABLE"
    assert "PTS_L10" in " ".join(out.notes)


def test_an_implausible_fit_is_reported_but_not_recommended():
    """A 40% effect is a data fault, not a finding, so it does not ship."""
    out = fit_fatigue_multipliers(panel(flagged_effect=0.60))
    fit = out.fits["b2b"]
    assert fit.multiplier == pytest.approx(0.60, abs=1e-6)
    assert fit.status == "IMPLAUSIBLE"
    assert not fit.plausible
    assert "b2b" not in out.multipliers, "an implausible figure must not be offered"


def test_rows_with_a_zero_or_missing_baseline_are_excluded():
    frame = panel(n=6_000)
    frame.loc[frame.index[:100], "PTS_L10"] = 0.0
    frame.loc[frame.index[100:200], "PTS_L10"] = np.nan
    out = fit_fatigue_multipliers(frame)
    assert out.n_usable_rows == len(frame) - 200


def test_fitting_against_the_blended_baseline_warns_that_it_is_circular():
    frame = panel()
    frame["PTS_BASELINE"] = frame["PTS_L10"]
    out = fit_fatigue_multipliers(frame, baseline_col="PTS_BASELINE")
    assert any("circular" in n for n in out.notes)


# --- reporting -------------------------------------------------------------

def test_the_comparison_names_the_constant_each_fit_would_replace():
    out = fit_fatigue_multipliers(panel(flagged_effect=0.95))
    rows = {r["flag"]: r for r in compare_to_constants(out)}
    assert rows["b2b"]["in_use"] == pytest.approx(0.97)
    assert rows["b2b"]["fitted"] == pytest.approx(0.95, abs=1e-3)
    assert rows["b2b"]["delta"] == pytest.approx(-0.02, abs=1e-3)
    for flag in ("b2b", "three_in_four", "four_in_five"):
        assert flag in rows, "every constant in use must appear in the comparison"


def test_the_comparison_reads_the_constants_actually_in_force():
    from src.features import fatigue_logic

    rows = {r["flag"]: r for r in compare_to_constants(fit_fatigue_multipliers(panel()))}
    assert rows["b2b"]["in_use"] == fatigue_logic.B2B_PENALTY
    assert rows["four_in_five"]["in_use"] == fatigue_logic.FOUR_IN_FIVE_PENALTY


def test_a_fit_round_trips_through_disk(tmp_path):
    out = fit_fatigue_multipliers(panel(flagged_effect=0.93))
    path = save_fatigue_fit(out, tmp_path / "fit.json")
    loaded = load_fatigue_fit(path)
    assert loaded["stat"] == "PTS"
    assert loaded["multipliers"]["b2b"] == pytest.approx(0.93, abs=1e-3)
    assert load_fatigue_fit(tmp_path / "absent.json") is None


def test_nothing_here_edits_the_constants_in_force():
    """
    Applying a fitted figure is a separate, deliberate edit. On the real panel
    the fit comes out ABOVE 1.0 — see docs/fatigue_fit.md — so an automatic
    swap would have replaced a haircut with a boost on survivorship-biased
    evidence.
    """
    from pathlib import Path

    body = (Path(__file__).parent.parent / "src" / "features" / "fatigue_fit.py").read_text()
    assert "B2B_PENALTY =" not in body, "this module must not redefine the constants"
    assert "fatigue_logic" in body, "it should still read them for the comparison"
