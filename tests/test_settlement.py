"""
tests/test_settlement.py

Focus: the cases that silently produce WRONG MONEY if handled naively.
No mock box scores are used for grading logic — these are pure arithmetic
tests on the settlement engine. Network fetching is tested separately and
skipped without connectivity.
"""

import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from src.settlement.evaluator import (
    MARKET_COMPONENTS,
    Outcome,
    SettlementError,
    american_to_profit_per_unit,
    combine_stat,
    compute_clv,
    is_whole_number_line,
    settle_prop,
)


def stats(**kw):
    """Build a stat dict with all components present (missing != zero)."""
    base = {
        "points": 0, "reboundsTotal": 0, "assists": 0,
        "threePointersMade": 0, "steals": 0, "blocks": 0, "turnovers": 0,
    }
    base.update(kw)
    return base


# ---------------------------------------------------------------------------
# PUSH — the case the whole feature exists for
# ---------------------------------------------------------------------------

class TestPush:
    def test_exact_tie_on_whole_line_is_push(self):
        r = settle_prop(market="PTS", predicted_line=25, predicted_side="OVER",
                        player_stats=stats(points=25), odds=-110)
        assert r.outcome is Outcome.PUSH
        assert r.actual_result == Decimal(25)

    def test_push_returns_stake_not_profit_and_not_loss(self):
        r = settle_prop(market="PTS", predicted_line=25, predicted_side="OVER",
                        player_stats=stats(points=25), odds=-110, stake_units=1)
        assert r.profit_units == Decimal(0), "A push must be zero profit, never negative"

    def test_push_is_side_independent(self):
        over = settle_prop(market="REB", predicted_line=10, predicted_side="OVER",
                           player_stats=stats(reboundsTotal=10), odds=-110)
        under = settle_prop(market="REB", predicted_line=10, predicted_side="UNDER",
                            player_stats=stats(reboundsTotal=10), odds=-110)
        assert over.outcome is Outcome.PUSH and under.outcome is Outcome.PUSH

    def test_half_point_line_can_never_push(self):
        """25.5 cannot tie — one side must win."""
        for actual in (25, 26):
            r = settle_prop(market="PTS", predicted_line=25.5, predicted_side="OVER",
                            player_stats=stats(points=actual), odds=-110)
            assert r.outcome is not Outcome.PUSH

    def test_is_whole_number_line(self):
        assert is_whole_number_line(Decimal("25"))
        assert is_whole_number_line(Decimal("25.0"))
        assert not is_whole_number_line(Decimal("25.5"))


# ---------------------------------------------------------------------------
# FLOAT PRECISION — why Decimal is used
# ---------------------------------------------------------------------------

class TestFloatPrecision:
    def test_combined_market_tie_is_detected_exactly(self):
        """
        PRA line 35, actual 12+11+12=35. In float arithmetic a combined
        sum can land a hair off and mis-grade a PUSH as a LOSS.
        """
        r = settle_prop(market="PRA", predicted_line=35, predicted_side="OVER",
                        player_stats=stats(points=12, reboundsTotal=11, assists=12),
                        odds=-110)
        assert r.outcome is Outcome.PUSH
        assert r.actual_result == Decimal(35)

    def test_float_line_input_is_coerced_safely(self):
        """A float 25.5 passed from JSON must not corrupt the comparison."""
        r = settle_prop(market="PTS", predicted_line=25.5, predicted_side="UNDER",
                        player_stats=stats(points=25), odds=-110)
        assert r.outcome is Outcome.WIN

    def test_decimal_conversion_goes_through_str(self):
        """Decimal(0.1) != Decimal('0.1'); the engine must use the latter."""
        r = settle_prop(market="PTS", predicted_line=0.5, predicted_side="OVER",
                        player_stats=stats(points=1), odds=100)
        assert r.outcome is Outcome.WIN


# ---------------------------------------------------------------------------
# WIN / LOSS direction
# ---------------------------------------------------------------------------

class TestWinLoss:
    @pytest.mark.parametrize("side,actual,expected", [
        ("OVER", 26, Outcome.WIN),
        ("OVER", 24, Outcome.LOSS),
        ("UNDER", 24, Outcome.WIN),
        ("UNDER", 26, Outcome.LOSS),
    ])
    def test_direction(self, side, actual, expected):
        r = settle_prop(market="PTS", predicted_line=25.5, predicted_side=side,
                        player_stats=stats(points=actual), odds=-110)
        assert r.outcome is expected

    def test_win_profit_negative_odds(self):
        r = settle_prop(market="PTS", predicted_line=25.5, predicted_side="OVER",
                        player_stats=stats(points=30), odds=-110, stake_units=1)
        # -110 pays 100/110 = 0.9090...
        assert r.profit_units == pytest.approx(Decimal("0.909090909090909090909090909"), abs=Decimal("0.0001"))

    def test_win_profit_positive_odds(self):
        r = settle_prop(market="PTS", predicted_line=25.5, predicted_side="OVER",
                        player_stats=stats(points=30), odds=150, stake_units=1)
        assert r.profit_units == Decimal("1.5")

    def test_loss_is_negative_stake_exactly(self):
        r = settle_prop(market="PTS", predicted_line=25.5, predicted_side="OVER",
                        player_stats=stats(points=20), odds=-110, stake_units=Decimal("2"))
        assert r.profit_units == Decimal("-2")


# ---------------------------------------------------------------------------
# VOID — DNP must not be a LOSS
# ---------------------------------------------------------------------------

class TestVoid:
    def test_did_not_play_is_void_not_loss(self):
        r = settle_prop(market="PTS", predicted_line=25.5, predicted_side="OVER",
                        player_stats=stats(points=0), odds=-110, did_not_play=True)
        assert r.outcome is Outcome.VOID
        assert r.profit_units is None

    def test_zero_minutes_is_void(self):
        r = settle_prop(market="PTS", predicted_line=25.5, predicted_side="OVER",
                        player_stats=stats(points=0), odds=-110, minutes_played=0)
        assert r.outcome is Outcome.VOID

    def test_missing_stats_is_void(self):
        r = settle_prop(market="PTS", predicted_line=25.5, predicted_side="OVER",
                        player_stats=None, odds=-110)
        assert r.outcome is Outcome.VOID


# ---------------------------------------------------------------------------
# Missing data must never be silently zeroed
# ---------------------------------------------------------------------------

class TestMissingComponents:
    def test_missing_component_raises_not_zeroes(self):
        incomplete = {"points": 20}  # PRA needs rebounds + assists too
        with pytest.raises(SettlementError, match="missing component"):
            combine_stat(incomplete, "PRA")

    def test_none_component_raises(self):
        with pytest.raises(SettlementError, match="missing component"):
            combine_stat(stats(points=None), "PTS")

    def test_unknown_market_raises(self):
        with pytest.raises(SettlementError, match="Unknown market"):
            combine_stat(stats(points=20), "NOT_A_MARKET")


# ---------------------------------------------------------------------------
# Combined markets sum real components
# ---------------------------------------------------------------------------

class TestCombinedMarkets:
    def test_pra_sums_three_components(self):
        assert combine_stat(stats(points=20, reboundsTotal=8, assists=5), "PRA") == Decimal(33)

    def test_pr_excludes_assists(self):
        assert combine_stat(stats(points=20, reboundsTotal=8, assists=5), "PR") == Decimal(28)

    def test_stocks_is_steals_plus_blocks(self):
        assert combine_stat(stats(steals=3, blocks=2), "STOCKS") == Decimal(5)

    def test_every_market_has_components(self):
        for market, comps in MARKET_COMPONENTS.items():
            assert comps, f"{market} has no components"


# ---------------------------------------------------------------------------
# Pick'em: graded, but ROI abstains
# ---------------------------------------------------------------------------

class TestPickemAbstains:
    def test_no_odds_still_grades_win_loss(self):
        r = settle_prop(market="PTS", predicted_line=25.5, predicted_side="OVER",
                        player_stats=stats(points=30), odds=None)
        assert r.outcome is Outcome.WIN
        assert r.profit_units is None, "ROI must abstain without a two-way price"

    def test_no_odds_push_also_abstains_on_profit(self):
        r = settle_prop(market="PTS", predicted_line=25, predicted_side="OVER",
                        player_stats=stats(points=25), odds=None)
        assert r.outcome is Outcome.PUSH
        assert r.profit_units is None


# ---------------------------------------------------------------------------
# Odds conversion
# ---------------------------------------------------------------------------

class TestOdds:
    def test_plus_money(self):
        assert american_to_profit_per_unit(150) == Decimal("1.5")

    def test_minus_money(self):
        assert american_to_profit_per_unit(-200) == Decimal("0.5")

    def test_zero_odds_rejected(self):
        with pytest.raises(SettlementError):
            american_to_profit_per_unit(0)


# ---------------------------------------------------------------------------
# CLV — line movement and price movement are separate
# ---------------------------------------------------------------------------

class TestCLV:
    def test_over_benefits_from_rising_line(self):
        r = compute_clv(predicted_line=24.5, predicted_side="OVER", closing_line=26.5)
        assert r["clv_line_points"] == Decimal("2.0")

    def test_under_benefits_from_falling_line(self):
        r = compute_clv(predicted_line=26.5, predicted_side="UNDER", closing_line=24.5)
        assert r["clv_line_points"] == Decimal("2.0")

    def test_same_move_is_negative_for_opposite_side(self):
        r = compute_clv(predicted_line=24.5, predicted_side="UNDER", closing_line=26.5)
        assert r["clv_line_points"] == Decimal("-2.0")

    def test_price_clv_requires_both_prices(self):
        r = compute_clv(predicted_line=25.5, predicted_side="OVER",
                        closing_line=25.5, bet_odds=-110)
        assert r["clv_prob_points"] is None

    def test_price_clv_positive_when_market_moves_to_your_side(self):
        r = compute_clv(predicted_line=25.5, predicted_side="OVER", closing_line=25.5,
                        bet_odds=-110, closing_odds=-130)
        assert r["clv_prob_points"] > 0

    def test_line_and_price_clv_are_independent(self):
        """A line move must not leak into the price metric or vice versa."""
        r = compute_clv(predicted_line=24.5, predicted_side="OVER", closing_line=26.5,
                        bet_odds=-110, closing_odds=-110)
        assert r["clv_line_points"] == Decimal("2.0")
        assert r["clv_prob_points"] == Decimal(0)
