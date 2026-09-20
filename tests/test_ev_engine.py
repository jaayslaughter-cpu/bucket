"""
odds_math / ev_engine / PropMarketSnapshot.

These three were imported by waves 3-5 and existed in no pack, so they
were written here against the interface the call sites expect. The wave
packs' own tests (test_wave3_paper, test_wave4, test_wave5a) exercise
them from above; these cover the arithmetic directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.quant.contracts import PropMarketSnapshot, market_ev_gate
from src.quant.ev_engine import ClvResult, EvEngine, EvEvaluation, compute_clv
from src.quant.odds_math import (
    american_to_decimal,
    american_to_profit_multiple,
    breakeven_probability,
    decimal_to_american,
    expected_value_per_unit,
    multiplicative_devig,
    probability_to_american,
)

# --------------------------------------------------------------------------
# The mistake this module is built to avoid
# --------------------------------------------------------------------------

def test_profit_multiple_is_net_not_decimal():
    """Decimal odds include the returned stake; EV and Kelly need net profit.

    A reference implementation reviewed earlier in this project used
    decimal odds where net profit belongs, overstating the win branch on a
    +150 price by 67%.
    """
    assert american_to_decimal(150) == pytest.approx(2.5)
    assert american_to_profit_multiple(150) == pytest.approx(1.5)
    assert american_to_decimal(-110) == pytest.approx(1.909091, abs=1e-6)
    assert american_to_profit_multiple(-110) == pytest.approx(0.909091, abs=1e-6)


def test_ev_is_zero_at_the_breakeven_probability():
    for american in (-250, -110, 100, 150, 400):
        p = breakeven_probability(american)
        assert expected_value_per_unit(p, american) == pytest.approx(0.0, abs=1e-9)


def test_ev_matches_hand_computation():
    # 0.55 * 0.909091 - 0.45 = 0.05
    assert expected_value_per_unit(0.55, -110) == pytest.approx(0.05, abs=1e-6)


def test_odds_round_trip():
    for american in (-400, -110, -101, 100, 150, 900):
        assert decimal_to_american(american_to_decimal(american)) == american


def test_probability_to_american_inverts_breakeven():
    for american in (-300, -110, 120, 500):
        assert probability_to_american(breakeven_probability(american)) == american


def test_zero_odds_are_refused():
    for fn in (american_to_decimal, american_to_profit_multiple, breakeven_probability):
        with pytest.raises(ValueError):
            fn(0)


# --------------------------------------------------------------------------
# De-vig delegates rather than duplicating
# --------------------------------------------------------------------------

def test_devig_removes_the_hold_symmetrically():
    result = multiplicative_devig(-110, -110)
    assert result.fair_prob_a == pytest.approx(0.5)
    assert result.fair_prob_b == pytest.approx(0.5)
    assert result.hold == pytest.approx(0.047619, abs=1e-6)
    # The raw implied probabilities must still sum above 1 — that IS the vig.
    assert result.implied_prob_a + result.implied_prob_b > 1.0


def test_devig_fair_probabilities_sum_to_one():
    for pair in ((-150, 130), (-110, -110), (200, -250)):
        result = multiplicative_devig(*pair)
        assert result.fair_prob_a + result.fair_prob_b == pytest.approx(1.0, abs=1e-9)


def test_there_is_only_one_devig_implementation():
    source = (Path(__file__).parent.parent / "src/quant/odds_math.py").read_text()
    assert "devig_two_way" in source, "odds_math should delegate, not reimplement"


# --------------------------------------------------------------------------
# EvEngine
# --------------------------------------------------------------------------

def test_edge_is_measured_against_the_devigged_price():
    """Measuring against the posted price counts the book's hold as edge."""
    engine = EvEngine(ev_threshold=0.0)
    result = engine.evaluate_two_way(
        game_id="g1", american_a=-110, american_b=-110,
        model_prob_a=0.55, label_a="over", label_b="under",
    )
    assert result.status == "OK"
    over = next(s for s in result.sides if s.side == "over")
    # Fair is 0.50 after de-vig, so the edge is 0.05 — not 0.55 - 0.5238.
    assert over.fair_prob == pytest.approx(0.50)
    assert over.edge == pytest.approx(0.05, abs=1e-9)


def test_both_sides_are_priced():
    engine = EvEngine()
    result = engine.evaluate_two_way(
        game_id="g1", american_a=-120, american_b=100, model_prob_a=0.6,
    )
    assert {s.side for s in result.sides} == {"over", "under"}
    assert result.selected_side == "over"
    assert result.selected_ev > 0


def test_no_side_is_selected_below_the_threshold():
    engine = EvEngine(ev_threshold=0.10)
    result = engine.evaluate_two_way(
        game_id="g1", american_a=-110, american_b=-110, model_prob_a=0.51,
    )
    assert result.status == "OK"
    assert result.selected_side is None
    assert "threshold" in result.reason


def test_out_of_range_model_probability_abstains():
    engine = EvEngine()
    for bad in (1.4, -0.2, float("nan")):
        result = engine.evaluate_two_way(
            game_id="g1", american_a=-110, american_b=-110, model_prob_a=bad,
        )
        assert result.status != "OK"
        assert "probability" in result.reason


def test_model_probabilities_are_not_silently_renormalised(caplog):
    """A pair that does not sum to 1 is a model property, not a tidy-up job."""
    import logging

    engine = EvEngine()
    with caplog.at_level(logging.WARNING):
        result = engine.evaluate_two_way(
            game_id="g1", american_a=-110, american_b=-110,
            model_prob_a=0.60, model_prob_b=0.60,
        )
    assert result.status == "OK"
    assert next(s for s in result.sides if s.side == "over").model_prob == 0.60
    assert next(s for s in result.sides if s.side == "under").model_prob == 0.60
    assert "not 1" in caplog.text


def test_snapshot_evaluation_goes_through_the_gate():
    """The gate is the single place that decides whether EV may exist."""
    engine = EvEngine()

    priced = PropMarketSnapshot(
        game_id="g1", market="PTS", line=27.5, status="VALID",
        over_odds_american=-110, under_odds_american=-110,
    )
    assert engine.evaluate_snapshot(priced, 0.55).status == "OK"

    # A pick'em board carries a multiplier, not a price.
    pickem = PropMarketSnapshot(
        game_id="g1", market="PTS", line=27.5, status="VALID", is_pickem=True,
    )
    abstained = engine.evaluate_snapshot(pickem, 0.55)
    assert abstained.status != "OK"
    assert "Pick'em" in abstained.reason

    # An unverified market never reaches the arithmetic.
    unverified = PropMarketSnapshot(
        game_id="g1", market="PTS", line=27.5, status="DATA_NOT_AVAILABLE",
        over_odds_american=-110, under_odds_american=-110,
    )
    assert engine.evaluate_snapshot(unverified, 0.55).status != "OK"


def test_evaluation_carries_a_research_disclaimer():
    engine = EvEngine()
    result = engine.evaluate_two_way(
        game_id="g1", american_a=-110, american_b=-110, model_prob_a=0.55,
    )
    assert "not a bet recommendation" in result.disclaimer
    assert "stake size" in result.disclaimer


# --------------------------------------------------------------------------
# CLV
# --------------------------------------------------------------------------

def test_clv_is_positive_when_the_line_moves_toward_you():
    """Took +150, closed -110: the market came to your side."""
    result = compute_clv(taken_american=150, closing_american=-110)
    assert result.status == "OK"
    assert result.clv > 0
    assert result.beat_close is True


def test_clv_is_negative_when_the_line_moves_away():
    result = compute_clv(taken_american=-110, closing_american=150)
    assert result.clv < 0
    assert result.beat_close is False


def test_clv_uses_devigged_probabilities_when_both_sides_are_given():
    devigged = compute_clv(
        taken_american=-110, closing_american=-130,
        taken_other_american=-110, closing_other_american=110,
    )
    assert devigged.status == "OK"
    assert "devigged" in devigged.note

    raw = compute_clv(taken_american=-110, closing_american=-130)
    assert "raw_implied" in raw.note


def test_clv_is_never_described_as_profit():
    result = compute_clv(taken_american=150, closing_american=-110)
    assert "not profit" in result.note
    assert "never added to ROI" in result.note


def test_clv_abstains_on_a_missing_price():
    assert compute_clv(taken_american=None, closing_american=-110).status != "OK"


# --------------------------------------------------------------------------
# PropMarketSnapshot
# --------------------------------------------------------------------------

def test_snapshot_narrows_to_the_gate_input():
    snap = PropMarketSnapshot(
        game_id="g1", market="PTS", market_id="m-42", player_name="Nikola Jokic",
        line=27.5, over_odds_american=-110, under_odds_american=-110,
        status="VALID", bookmaker="draftkings",
    )
    context = snap.to_market_context()
    assert context.game_id == "g1"
    assert context.source == "draftkings"
    assert market_ev_gate(context)["status"] == "READY_FOR_EVALUATION"
    assert snap.market_key == "m-42"


def test_snapshot_knows_when_it_has_no_real_two_way_price():
    assert not PropMarketSnapshot(game_id="g", is_pickem=True).has_two_way_price()
    assert not PropMarketSnapshot(
        game_id="g", over_odds_american=-110, payout_multiplier=1.5,
    ).has_two_way_price()
    assert not PropMarketSnapshot(game_id="g", over_odds_american=-110).has_two_way_price()
    assert PropMarketSnapshot(
        game_id="g", over_odds_american=-110, under_odds_american=-110,
    ).has_two_way_price()


def test_dataclasses_expose_the_fields_the_wave_modules_read():
    """historical_store and paper_research read these by name."""
    for field in ("game_id", "market_id", "market_type", "line",
                  "selected_side", "selected_ev", "status", "reason", "sides"):
        assert hasattr(EvEvaluation(game_id="g"), field), field
    for field in ("clv", "status"):
        assert hasattr(ClvResult(), field), field
