"""
compare.py computed its own ungated ECE while the gated one sat unused.

src/models/prob_calibration.py:134 implements ECE with a bin-coverage gate: below
80% non-empty bins it returns None, because a reliability diagram with two
occupied bins is not a calibration measurement. compare.py had an inline "simple
ECE proxy" instead, and its numbers feed model selection.

The arithmetic is the same. What the gate changes is the sparse case — and the
ranking tiebreak that consumed the result had two bugs of its own.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.compare import _winner_rank_key
from src.models.prob_calibration import expected_calibration_error, reliability_table


def _inline_proxy(y, p):
    """The implementation compare.py used to carry, kept here to prove the
    replacement does not move any number that was already trustworthy."""
    table = reliability_table(y, p)
    if not table:
        return None
    gaps = [abs(t["calibration_gap"]) * t["n_predictions"] for t in table]
    ntot = sum(t["n_predictions"] for t in table)
    return round(sum(gaps) / max(ntot, 1), 4)


def test_the_gated_ece_equals_the_old_proxy_when_the_gate_passes():
    """This is what makes the swap safe: every ECE already recorded from a dense
    window keeps its value, so the A/B results in docs/form_ab_results*.txt stay
    reproducible."""
    rng = np.random.default_rng(7)
    p = rng.uniform(0.02, 0.98, 4000)
    y = (rng.uniform(size=4000) < p).astype(int)

    gated = expected_calibration_error(y, p)
    assert gated["gate_passed"] is True
    assert gated["bin_coverage"] == 1.0
    assert _inline_proxy(y, p) == pytest.approx(round(gated["ece"], 4))


def test_a_sparse_window_reports_no_ece_instead_of_a_confident_one():
    """The case the gate exists for. Predictions crowded into two bins produced a
    number the proxy reported as an ECE; the gated version withholds it and keeps
    it visible as ece_ungated rather than discarding it."""
    rng = np.random.default_rng(7)
    p = rng.uniform(0.45, 0.55, 400)
    y = (rng.uniform(size=400) < p).astype(int)

    gated = expected_calibration_error(y, p)
    assert gated["gate_passed"] is False
    assert gated["bin_coverage"] < 0.8
    assert gated["ece"] is None
    assert gated["ece_ungated"] is not None
    # The proxy would have published exactly this figure as a measurement.
    assert _inline_proxy(y, p) == pytest.approx(round(gated["ece_ungated"], 4))


def test_compare_publishes_the_gate_alongside_the_number():
    """A withheld ECE has to be distinguishable from a missing column, so the
    row carries the gate flag and the coverage it was judged on."""
    import inspect

    from src.models import compare

    source = inspect.getsource(compare.compare_models_on_panel)
    for key in (
        "calibration_error_ungated",
        "calibration_gate_passed",
        "calibration_bin_coverage",
    ):
        assert key in source, f"{key} is not reported"
    assert "Simple ECE proxy" not in source, "the ungated proxy is still in place"


# --- the ranking tiebreak -------------------------------------------------


def test_a_perfectly_calibrated_model_is_not_ranked_as_unmeasurable():
    """`calibration_error or 9` mapped 0.0 to 9, identical to None: the best
    possible calibration and an absent measurement sorted the same."""
    assert (0.0 or 9) == 9, "the premise of this test has changed"

    perfect = {"brier_score": 0.24, "log_loss": 0.68, "calibration_error": 0.0}
    worse = {"brier_score": 0.24, "log_loss": 0.68, "calibration_error": 0.01}
    assert _winner_rank_key(perfect) < _winner_rank_key(worse)


def test_a_failed_gate_sorts_last_but_is_not_treated_as_a_score():
    """A model whose calibration could not be measured does not win a tie — that
    is not evidence of good calibration — but it is ordered by a missing-flag
    rather than by a fabricated 9."""
    measured = {"brier_score": 0.24, "log_loss": 0.68, "calibration_error": 0.05}
    ungated = {"brier_score": 0.24, "log_loss": 0.68, "calibration_error": None}
    assert _winner_rank_key(measured) < _winner_rank_key(ungated)

    key = _winner_rank_key(ungated)
    assert key[2] == 1 and key[3] == float("inf")
    assert 9 not in key, "a sentinel score is still standing in for a measurement"


def test_brier_still_decides_before_calibration_ever_matters():
    """ECE is the third key, not the first: a better-calibrated model with a
    worse Brier must not win."""
    better_brier = {"brier_score": 0.23, "log_loss": 0.70, "calibration_error": 0.09}
    better_ece = {"brier_score": 0.24, "log_loss": 0.68, "calibration_error": 0.001}
    assert _winner_rank_key(better_brier) < _winner_rank_key(better_ece)


def test_a_missing_brier_sorts_last_rather_than_first():
    """None must not read as zero, which would make an unscored model the winner."""
    scored = {"brier_score": 0.30, "log_loss": 0.9, "calibration_error": 0.2}
    unscored = {"brier_score": None, "log_loss": None, "calibration_error": None}
    assert _winner_rank_key(scored) < _winner_rank_key(unscored)
