"""Eligibility gates and KS drift, now that both have readers.

``src/models/eligibility.py`` was 129 tested lines with no caller, and both of
its config blocks — ``eligibility`` and ``drift`` — had no reader either, so
editing them did nothing. These tests cover the wiring rather than the
arithmetic (which `test_eligibility.py` already pins).

TWO DIFFERENT CONSEQUENCES, and keeping them apart is the point:

  cold start  withholds the RECOMMENDATION. The prediction is still recorded
              and still graded, because a thin-history row is the evidence
              where the model is weakest and dropping it from the ledger would
              bias the calibration toward easy cases.
  drift       is REPORTED and nothing else. Dropping a feature on a KS p-value
              would let the validation window choose the feature set, which is
              the shape of a leak.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.eligibility import (
    eligibility_warnings_for_row,
    ks_feature_drift,
    prior_game_counts,
)
from src.quant.decision_board import build_decision_board, expand_row_to_candidates
from src.quant.paper_research import ResearchSlateRow


def slate_row(**kw) -> ResearchSlateRow:
    base = dict(
        slate_date="2026-10-02",
        event_id="0022600001",
        player_id="203999",
        player_name="DEMO_A",
        target_market="PTS",
        research_line=24.5,
        model_p_over=0.62,
        model_p_under=0.38,
        book_status="VALID",
        book_line=24.5,
        book_over_american=-110,
        book_under_american=-110,
        book_ev_over=0.08,
        book_ev_under=-0.05,
        preferred_side="over",
    )
    base.update(kw)
    return ResearchSlateRow(**base)


# --- the board refuses to recommend a cold-start row --------------------

def test_a_priced_row_with_enough_history_is_recommended():
    """The control: without warnings this row recommends."""
    cands = expand_row_to_candidates(slate_row(prior_games=40), min_ev=0.0)
    over = next(c for c in cands if c.side == "over")
    assert over.decision_status == "RECOMMENDED"
    assert over.decision_basis == "book_ev"


def test_a_cold_start_row_abstains_even_though_it_clears_min_ev():
    """
    The case the override exists for. The row has a real two-way price and an
    EV of +0.08; it is withheld anyway, because the probability came from a
    model that saw almost nothing of this player.
    """
    cands = expand_row_to_candidates(
        slate_row(
            prior_games=3,
            eligibility_warnings=["ABSTAIN: prior_games=3 < min_prior_games=10 (warm-up)"],
        ),
        min_ev=0.0,
    )
    over = next(c for c in cands if c.side == "over")
    assert over.decision_status == "ABSTAIN"
    assert "cold start" in over.why
    assert "prior_games=3" in over.why


def test_the_abstention_keeps_the_basis_and_the_number_it_withheld():
    """
    Withholding a recommendation is not the same as having nothing to say, and
    the row should still show what it was going to say and on what basis.
    """
    cands = expand_row_to_candidates(
        slate_row(prior_games=2, eligibility_warnings=["ABSTAIN: prior_games=2 < 10"]),
        min_ev=0.0,
    )
    over = next(c for c in cands if c.side == "over")
    assert over.book_ev == pytest.approx(0.08), "the EV was discarded, not withheld"
    assert over.model_prob == pytest.approx(0.62)
    assert "basis for this row was book_ev" in over.why


def test_the_warning_reaches_the_candidate_warnings_too():
    row = slate_row(
        prior_games=1, eligibility_warnings=["ABSTAIN: MIN_L5=4.0 < min_minutes_l5=12.0"]
    )
    cands = expand_row_to_candidates(row, min_ev=0.0)
    assert any("MIN_L5" in w for c in cands for w in c.warnings)


def test_prior_games_is_exported_so_the_reason_can_be_checked():
    """A reader should not have to take the abstention on faith."""
    cands = expand_row_to_candidates(slate_row(prior_games=4), min_ev=0.0)
    assert all(c.prior_games == 4 for c in cands)


def test_a_cold_start_row_cannot_reach_a_recommended_only_board():
    board = build_decision_board(
        [slate_row(prior_games=2, eligibility_warnings=["ABSTAIN: prior_games=2 < 10"])],
        min_ev=0.0,
        consider_only=True,
    )
    assert board == []


def test_an_already_abstaining_row_is_not_relabelled_by_the_override():
    """
    The override only ever takes a RECOMMENDED row back. A row abstaining for
    another reason must keep that reason, or the real cause is lost.
    """
    cands = expand_row_to_candidates(
        slate_row(
            book_status="DATA_NOT_AVAILABLE",
            book_over_american=None,
            book_under_american=None,
            book_ev_over=None,
            book_ev_under=None,
            model_p_over=0.51,
            model_p_under=0.49,
            prior_games=2,
            eligibility_warnings=["ABSTAIN: prior_games=2 < 10"],
        ),
        min_ev=0.0,
        min_lean=0.05,
    )
    over = next(c for c in cands if c.side == "over")
    assert over.decision_status == "ABSTAIN"
    assert "cold start" not in over.why, "the weak-lean reason was overwritten"


def test_a_row_with_no_eligibility_information_is_unaffected():
    """Rows from an older prediction export carry neither field."""
    cands = expand_row_to_candidates(slate_row(), min_ev=0.0)
    over = next(c for c in cands if c.side == "over")
    assert over.decision_status == "RECOMMENDED"
    assert over.prior_games is None


# --- prior games must be counted over the whole frame -------------------

def test_prior_games_counted_on_a_validation_slice_alone_would_abstain_veterans():
    """
    THE TRAP the wiring note in compare.py names. prior_game_counts is a
    per-player cumcount, so handing it only the validation window restarts
    every player at zero and a 60-game veteran reads as a cold start.
    """
    dates = pd.date_range("2026-01-01", periods=60, freq="D")
    full = pd.DataFrame({"PLAYER_ID": ["A"] * 60, "GAME_DATE": dates})

    over_everything = prior_game_counts(full)
    assert over_everything.iloc[-1] == 59

    validation_only = full.tail(5).copy()
    restarted = prior_game_counts(validation_only)
    assert restarted.iloc[-1] == 4, "a slice restarts the count"

    # and that is the difference between eligible and not
    assert eligibility_warnings_for_row(
        validation_only.iloc[-1], prior_games=int(over_everything.iloc[-1]),
    ) == []
    assert eligibility_warnings_for_row(
        validation_only.iloc[-1], prior_games=int(restarted.iloc[-1]),
    ) != []


# --- the config blocks are read -----------------------------------------

def test_the_eligibility_block_is_read_by_the_comparison():
    """
    Both blocks had no reader, so editing them did nothing. Checked on the
    source rather than by running a full comparison, which needs a panel.
    """
    import inspect

    from src.models import compare

    source = inspect.getsource(compare.compare_models_on_panel)
    assert 'cfg.get("eligibility")' in source
    assert "min_prior_games" in source
    assert "min_minutes_l5" in source


def test_the_drift_block_is_read_by_the_comparison():
    import inspect

    from src.models import compare

    source = inspect.getsource(compare.compare_models_on_panel)
    assert 'cfg.get("drift")' in source
    assert "ks_p_threshold" in source
    assert "ks_feature_drift" in source


def test_the_shipped_config_still_carries_both_blocks():
    """If a block is removed, the defaults above become the silent behaviour."""
    from src.models.compare import load_comparison_config

    cfg = load_comparison_config()
    assert cfg["eligibility"]["min_prior_games"] == 10
    assert cfg["eligibility"]["min_minutes_l5"] == 12.0
    assert cfg["drift"]["ks_p_threshold"] == 0.01


# --- drift is reported, never acted on ----------------------------------

def test_a_shifted_feature_is_flagged_as_drift():
    rng = np.random.default_rng(5)
    train = pd.DataFrame({"PTS_L5": rng.normal(20, 3, 400)})
    val = pd.DataFrame({"PTS_L5": rng.normal(28, 3, 400)})
    rows = ks_feature_drift(train, val, ["PTS_L5"], p_threshold=0.01)
    assert rows[0]["status"] == "DRIFT"
    assert rows[0]["p_value"] < 0.01


def test_a_stable_feature_is_not_flagged():
    rng = np.random.default_rng(5)
    train = pd.DataFrame({"PTS_L5": rng.normal(20, 3, 400)})
    val = pd.DataFrame({"PTS_L5": rng.normal(20, 3, 400)})
    assert ks_feature_drift(train, val, ["PTS_L5"], p_threshold=0.01)[0]["status"] == "OK"


def test_a_thin_sample_reports_insufficient_rather_than_ok():
    """Calling a 10-row comparison "OK" would be the dangerous answer."""
    train = pd.DataFrame({"PTS_L5": range(10)})
    val = pd.DataFrame({"PTS_L5": range(10)})
    assert ks_feature_drift(train, val, ["PTS_L5"])[0]["status"] == "INSUFFICIENT_SAMPLE"


def test_the_comparison_returns_the_drift_report_rather_than_only_logging_it():
    """
    A model trained on one distribution and scored on another is not comparable
    to one that was not, and its metrics alone do not say which it was.
    """
    import inspect

    from src.models import compare

    source = inspect.getsource(compare.compare_models_on_panel)
    assert '"covariate_drift": drift_rows' in source


def test_drift_never_removes_a_feature():
    """
    Dropping on a KS p-value would let the validation window choose the feature
    set, which is the shape of a leak. The only feature removal in the
    comparison is the training-window-coverage one, which is a different test.
    """
    import inspect

    from src.models import compare

    source = inspect.getsource(compare.compare_models_on_panel)
    drift_section = source.split("KS COVARIATE DRIFT")[1].split("ELIGIBILITY")[0]
    assert "feature_cols = [" not in drift_section
    assert "xgb_cols = [" not in drift_section
