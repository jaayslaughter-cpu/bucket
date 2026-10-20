"""
Composite markets (PA, PR, RA, PRA, FPTS) and Underdog Streaks.

TWO DEFINITIONS OF PRA ALREADY PRODUCED A SILENT DEFECT in this tree --
`attach_pra_from_components` records it -- and they agreed on every row of the
archive panel, so the divergence was invisible on the only data available.
Four more composites by hand would be four more chances at it, which is why
the decomposition has one home and these tests pin it against the settlement
grader's own table.

RESEARCH_ONLY. Projections and figures, no wagers.
"""

from __future__ import annotations

import math
from datetime import date as _d
from pathlib import Path

import pytest

from src.quant import composite_markets as cm
from src.quant import streaks as st

# --- the decomposition, and that nothing else disagrees with it -------------


def test_the_settlement_grader_reads_this_decomposition():
    """
    ONE DEFINITION OR THE LEDGER AND THE PROJECTION DIVERGE. This used to
    assert that two hand-written tables agreed. They now agree BY
    CONSTRUCTION -- `evaluator.MARKET_COMPONENTS` is built from
    `SUM_COMPOSITES` through `PANEL_TO_BOXSCORE` -- so what is left to pin is
    the part that can still break: that the derivation happened at all, and
    that every component has a box-score field to be read from.
    """
    from src.settlement.evaluator import MARKET_COMPONENTS, PANEL_TO_BOXSCORE

    for market, components in cm.SUM_COMPOSITES.items():
        graded = MARKET_COMPONENTS.get(market)
        assert graded is not None, f"{market} projects but settlement cannot grade it"
        assert graded == tuple(PANEL_TO_BOXSCORE[c] for c in components)

    # And every component of every composite, FPTS included, is translatable.
    for name in cm.COMPONENTS:
        assert name in PANEL_TO_BOXSCORE, (
            f"{name} is a composite component with no box-score field"
        )


def test_an_fpts_prop_can_actually_be_settled():
    """
    IT COULD NOT BE, AND THAT WAS THE WHOLE POINT OF THE FINDING. A fantasy
    score is not a SUM, so an FPTS prop reached `combine_stat`, found no entry
    in a table of markets-that-sum, and raised "Unknown market". The
    projection side was built and the ledger could never close the loop on it
    — which also means no FPTS calibration could ever be measured, so the
    user's own "singles until calibration is measured" gate was unreachable
    for this market.
    """
    from decimal import Decimal

    from src.settlement.evaluator import combine_stat, settle_prop

    box = {"points": 30, "reboundsTotal": 8, "assists": 9,
           "steals": 2, "blocks": 1, "turnovers": 4}
    # 30 + 1.2*8 + 1.5*9 + 3*2 + 3*1 - 1*4
    assert combine_stat(box, "FPTS", platform="underdog") == Decimal("58.1")

    graded = settle_prop(
        market="FPTS", predicted_line=55.5, predicted_side="OVER",
        player_stats=box, odds=-110, platform="underdog",
    )
    assert str(graded.outcome) == "WIN"
    assert graded.actual_result == Decimal("58.1")


def test_fpts_grades_in_decimal_but_on_the_same_weights_as_the_projection():
    """
    TWO NUMERIC REPRESENTATIONS, ONE WEIGHT TABLE. The grader is Decimal
    throughout because the P/L ledger must not drift; this module projects in
    float. That is two arithmetics, which is exactly the shape of the
    two-definitions defect — so the weights come from one place and the two
    results are pinned equal here.
    """
    from decimal import Decimal

    from src.settlement.evaluator import fantasy_points

    box = {"PTS": 30.0, "REB": 8.0, "AST": 9.0, "STL": 2.0, "BLK": 1.0, "TOV": 4.0}
    cdn = {"points": 30, "reboundsTotal": 8, "assists": 9,
           "steals": 2, "blocks": 1, "turnovers": 4}
    projected = cm.realised_composite(cm.composite("FPTS", platform="underdog"), box)
    assert fantasy_points(cdn, platform="underdog") == pytest.approx(
        Decimal(str(projected))
    )


def test_fpts_is_never_graded_on_a_guessed_operators_terms():
    """
    The multipliers are an operator's term. Grading a PrizePicks entry on
    Underdog's matrix returns an entirely plausible wrong number, and the two
    are configured separately precisely so the first divergence is visible.
    """
    from src.settlement.evaluator import SettlementError, combine_stat

    box = {"points": 30, "reboundsTotal": 8, "assists": 9,
           "steals": 2, "blocks": 1, "turnovers": 4}
    with pytest.raises(SettlementError, match="without a platform"):
        combine_stat(box, "FPTS")
    with pytest.raises(SettlementError, match="weights unavailable"):
        combine_stat(box, "FPTS", platform="not_an_operator")


def test_a_missing_component_still_refuses_to_grade_fpts():
    """A missing stat is not a zero — the rule the sum markets already keep."""
    from src.settlement.evaluator import SettlementError, combine_stat

    incomplete = {"points": 30, "reboundsTotal": 8, "assists": 9,
                  "steals": 2, "blocks": 1}   # no turnovers
    with pytest.raises(SettlementError, match="missing component"):
        combine_stat(incomplete, "FPTS", platform="underdog")


def test_the_runner_grades_fpts_on_the_rows_own_source(monkeypatch, run_settlement):
    """
    The source column IS the scoring matrix for an FPTS row. Hardcoding one
    operator would grade every book's fantasy line on that operator's terms,
    and asserting the substring "platform=prop.source" appears in runner.py
    would be satisfied by a comment — so this grades two real rows and reads
    the answer off the ledger.

    The second row's operator has no configured matrix, which is the refusal
    that proves the source is being read rather than defaulted: it stays
    PENDING with a named reason instead of being scored on Underdog's terms.
    """
    from decimal import Decimal

    class _Row:
        def __init__(self, source):
            self.nba_game_id = "0022600001"
            self.game_date = _d(2026, 10, 21)
            self.player_name = "De'Aaron Fox"
            self.market = "FPTS"
            self.predicted_line = 55.5
            self.predicted_side = "OVER"
            self.odds = -110
            self.source = source
            self.stake_units = None
            self.closing_line = None
            self.closing_odds = None
            self.outcome_status = "PENDING"
            for field in ("actual_result", "profit_units", "minutes_played",
                          "did_not_play", "clv_line_points", "clv_prob_points",
                          "settlement_note", "result_source", "settled_at_utc",
                          "raw_boxscore_json"):
                setattr(self, field, None)

    known, unknown = _Row("underdog"), _Row("a_book_with_no_matrix")
    report = run_settlement(
        monkeypatch, [known, unknown],
        player_stats={"De'Aaron Fox": {
            "points": 30, "reboundsTotal": 8, "assists": 9, "steals": 2,
            "blocks": 1, "turnovers": 4, "minutes_played": 36.0,
            "did_not_play": False,
        }},
    )

    assert known.outcome_status == "WIN"
    assert known.actual_result == Decimal("58.1")

    assert unknown.outcome_status == "PENDING", (
        "an FPTS row was graded on an operator's terms that were never configured"
    )
    assert any("FPTS weights unavailable" in e for e in report.errors), report.errors


def test_every_component_is_a_real_panel_column():
    """A composite of a column the panel does not have can never be computed."""
    for market in list(cm.SUM_COMPOSITES) + [cm.FPTS]:
        spec = cm.composite(market, platform="underdog")
        assert set(spec.components) <= set(cm.COMPONENTS), market


# --- the fantasy matrix is an operator's term, not a constant ---------------


def test_the_fpts_weights_come_from_dated_config():
    spec = cm.composite("FPTS", platform="prizepicks")
    assert spec.weights == {
        "PTS": 1.0, "REB": 1.2, "AST": 1.5, "STL": 3.0, "BLK": 3.0, "TOV": -1.0,
    }


def test_fpts_without_a_platform_is_refused():
    """
    The multipliers are an operator's terms, not a property of basketball.
    Defaulting to one operator would price the other's lines wrongly and look
    completely normal doing it.
    """
    with pytest.raises(cm.CompositeMarketError, match="needs a platform"):
        cm.composite("FPTS")


def test_an_unknown_platform_raises_rather_than_defaulting():
    with pytest.raises(cm.CompositeMarketError, match="no fpts weights"):
        cm.composite("FPTS", platform="not_an_operator")


def test_the_two_operators_are_configured_separately():
    """
    They were supplied identical. Sharing one block would turn "they happen to
    agree today" into "they cannot differ", and the first divergence would be
    invisible.
    """
    import yaml

    raw = yaml.safe_load(Path("config/fantasy_scoring.yaml").read_text())
    assert set(raw["platforms"]) >= {"prizepicks", "underdog"}
    assert raw.get("as_of"), "an operator term with no date cannot be audited"


# --- the mean needs no correlation; the variance does -----------------------


def test_the_mean_is_the_weighted_sum():
    """E[sum w_i X_i] = sum w_i E[X_i], whatever the correlations."""
    spec = cm.composite("PRA")
    assert cm.composite_mean(spec, {"PTS": 24.0, "REB": 6.0, "AST": 5.0}) == 35.0


def test_the_fantasy_mean_applies_every_multiplier_including_the_negative():
    spec = cm.composite("FPTS", platform="underdog")
    got = cm.composite_mean(
        spec, {"PTS": 20.0, "REB": 10.0, "AST": 5.0, "STL": 1.0, "BLK": 1.0, "TOV": 3.0},
    )
    # 20 + 12 + 7.5 + 3 + 3 - 3
    assert got == pytest.approx(42.5)


def test_a_missing_component_projection_abstains():
    spec = cm.composite("PRA")
    with pytest.raises(cm.CompositeMarketError, match="DATA_NOT_AVAILABLE"):
        cm.composite_mean(spec, {"PTS": 24.0, "REB": 6.0})


def test_the_variance_refuses_to_assume_independence():
    """
    THE ASSUMPTION THAT IS CERTAINLY FALSE. PTS, REB and AST are all driven by
    minutes and usage, so dropping the covariance term understates the spread
    and makes every Over and Under on a composite look more certain than it
    is. Independence is not a safe default, so it is not a default at all.
    """
    spec = cm.composite("PRA")
    with pytest.raises(cm.CompositeMarketError, match="needs a correlation"):
        cm.composite_variance(spec, {"PTS": 6.0, "REB": 2.0, "AST": 2.0}, {})


def test_positive_correlation_widens_the_composite():
    """The whole reason the covariance term is required."""
    spec = cm.composite("PA")
    sds = {"PTS": 6.0, "AST": 2.0}
    independent = cm.composite_variance(spec, sds, {("PTS", "AST"): 0.0})
    correlated = cm.composite_variance(spec, sds, {("PTS", "AST"): 0.5})
    assert independent == pytest.approx(36.0 + 4.0)
    assert correlated == pytest.approx(40.0 + 2 * 0.5 * 6.0 * 2.0)
    assert correlated > independent, "positive correlation must widen the spread"


def test_a_correlation_is_accepted_in_either_order():
    spec = cm.composite("PA")
    sds = {"PTS": 6.0, "AST": 2.0}
    a = cm.composite_variance(spec, sds, {("PTS", "AST"): 0.4})
    b = cm.composite_variance(spec, sds, {("AST", "PTS"): 0.4})
    assert a == pytest.approx(b)


def test_a_negative_weight_reduces_the_variance_contribution():
    """
    Not a special case, which is the point of carrying weights rather than a
    component list: ``w_i w_j`` carries the sign, so turnovers being positively
    correlated with points correctly pulls the composite's variance DOWN.
    """
    spec = cm.composite("FPTS", platform="underdog")
    sds = {"PTS": 6.0, "REB": 2.0, "AST": 2.0, "STL": 1.0, "BLK": 1.0, "TOV": 1.5}
    pairs = {(a, b): 0.0 for i, a in enumerate(spec.components)
             for b in spec.components[i + 1:]}
    flat = cm.composite_variance(spec, sds, pairs)
    pairs[("PTS", "TOV")] = 0.5
    with_tov = cm.composite_variance(spec, sds, pairs)
    assert with_tov < flat, "a -1.0 weight must subtract, not add, covariance"


@pytest.mark.parametrize("rho", [1.5, -1.5, 2.0])
def test_an_impossible_correlation_is_refused(rho):
    spec = cm.composite("PA")
    with pytest.raises(cm.CompositeMarketError, match="outside"):
        cm.composite_variance(spec, {"PTS": 6.0, "AST": 2.0}, {("PTS", "AST"): rho})


def test_a_non_psd_correlation_set_is_named_not_square_rooted():
    """
    An inconsistent correlation set is not a matrix any three variables could
    produce. Computing a variance from it would hand the caller a confident
    number about an impossible world; `quant.psd_repair` exists for this.
    """
    spec = cm.composite("PRA")
    sds = {"PTS": 5.0, "REB": 5.0, "AST": 5.0}
    impossible = {("PTS", "REB"): -1.0, ("PTS", "AST"): -1.0, ("REB", "AST"): -1.0}
    with pytest.raises(cm.CompositeMarketError, match="not a consistent"):
        cm.composite_variance(spec, sds, impossible)


def test_an_impossible_set_is_refused_even_when_the_variance_comes_out_POSITIVE():
    """
    THE CHECK THAT WAS ONLY HALF A CHECK. Testing `total < 0` catches
    inconsistency only when it happens to show up in THIS composite's scalar.
    PTSxREB = +0.9, PTSxAST = +0.9, REBxAST = -0.9 cannot be realised by any
    three random variables — its smallest eigenvalue is about -0.72 — yet at
    equal sds it yields a comfortably POSITIVE variance, so the old guard
    passed it and the projection inherited the confidence.

    Pairs are fitted ONE BUCKET AT A TIME by `leg_correlation`, so nothing
    upstream makes the three agree with each other. This is reachable from
    real fitted inputs, not only from a hand-written matrix.
    """
    spec = cm.composite("PRA")
    sds = {"PTS": 5.0, "REB": 5.0, "AST": 5.0}
    impossible = {("PTS", "REB"): 0.9, ("PTS", "AST"): 0.9, ("REB", "AST"): -0.9}

    # The arithmetic itself is positive — this is why `total < 0` missed it.
    weighted = sum(25.0 for _ in range(3)) + 2.0 * 25.0 * (0.9 + 0.9 - 0.9)
    assert weighted > 0

    with pytest.raises(cm.CompositeMarketError, match="not a consistent"):
        cm.composite_variance(spec, sds, impossible)


def test_the_fitted_priors_this_project_measured_are_accepted():
    """
    The guard must refuse impossible sets WITHOUT refusing the real ones.
    These are the priors fitted from the 2018-2026 panel (214,381 rows,
    as_of 2026-03-28): PTSxREB 0.385, ASTxREB 0.279, ASTxPTS 0.253.
    """
    spec = cm.composite("PRA")
    sds = {"PTS": 6.5, "REB": 2.8, "AST": 2.1}
    fitted = {("PTS", "REB"): 0.385, ("REB", "AST"): 0.279, ("PTS", "AST"): 0.253}
    assert cm.composite_variance(spec, sds, fitted) > 0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_projection_abstains_instead_of_propagating(bad):
    """
    A NaN composite is worse than a missing one: it compares False against
    every line, so the Under grades as a confident win and nothing says the
    projection was absent. inf makes every Over certain the same way.
    """
    spec = cm.composite("PRA")
    with pytest.raises(cm.CompositeMarketError, match="not finite"):
        cm.composite_mean(spec, {"PTS": bad, "REB": 6.0, "AST": 5.0})
    with pytest.raises(cm.CompositeMarketError, match="not finite"):
        cm.composite_variance(
            spec, {"PTS": bad, "REB": 2.0, "AST": 2.0},
            {("PTS", "REB"): 0.3, ("PTS", "AST"): 0.2, ("REB", "AST"): 0.2},
        )


def test_a_spec_cannot_be_repriced_in_place():
    """
    `frozen=True` STOPS REBINDING, NOT MUTATION. The specs come from a
    module-level table and a config read, so an in-place edit outlives the
    caller that made it and reprices every later projection in the process.
    """
    spec = cm.composite("PRA")
    with pytest.raises(TypeError):
        spec.weights["REB"] = 99.0
    assert cm.composite("PRA").weights == {"PTS": 1.0, "REB": 1.0, "AST": 1.0}


def test_an_fpts_matrix_missing_a_component_is_refused(tmp_path):
    """
    A MISSING TOV DOES NOT MEAN TURNOVERS ARE FREE. It means nobody said what
    they cost, and scoring a six-component product on five returns an
    entirely plausible number for it. An operator who truly does not score a
    component writes 0.0 — that is the difference between zero and forgotten.
    """
    config = tmp_path / "fantasy.yaml"
    config.write_text(
        "platforms:\n  underdog:\n    fpts:\n      PTS: 1.0\n      REB: 1.2\n"
        "      AST: 1.5\n      STL: 3.0\n      BLK: 3.0\n",
        encoding="utf-8",
    )
    with pytest.raises(cm.CompositeMarketError, match="no FPTS weight for"):
        cm.fantasy_weights("underdog", path=config)

    # Stated as zero is a term, and is accepted.
    config.write_text(
        "platforms:\n  underdog:\n    fpts:\n      PTS: 1.0\n      REB: 1.2\n"
        "      AST: 1.5\n      STL: 3.0\n      BLK: 3.0\n      TOV: 0.0\n",
        encoding="utf-8",
    )
    assert cm.fantasy_weights("underdog", path=config)["TOV"] == 0.0


@pytest.mark.parametrize("value", [".nan", ".inf", "not_a_number"])
def test_an_unusable_fpts_weight_is_refused(tmp_path, value):
    """
    `inf * PTS` makes every FPTS line infinitely over and `nan` makes every
    comparison False, which grades as a confident Under. Both look like
    arithmetic rather than a config error everywhere downstream.
    """
    config = tmp_path / "fantasy.yaml"
    config.write_text(
        "platforms:\n  underdog:\n    fpts:\n      PTS: 1.0\n      REB: 1.2\n"
        f"      AST: 1.5\n      STL: 3.0\n      BLK: 3.0\n      TOV: {value}\n",
        encoding="utf-8",
    )
    with pytest.raises(cm.CompositeMarketError, match="not finite|not a number"):
        cm.fantasy_weights("underdog", path=config)


def test_grading_uses_the_same_arithmetic_as_projecting():
    spec = cm.composite("FPTS", platform="underdog")
    box = {"PTS": 30.0, "REB": 8.0, "AST": 9.0, "STL": 2.0, "BLK": 1.0, "TOV": 4.0}
    assert cm.realised_composite(spec, box) == cm.composite_mean(spec, box)


# --- Underdog Streaks: the stake is the streak, not one unit ----------------


def test_the_curve_loads_from_dated_config():
    curve = st.load_streak_curve("underdog")
    assert curve.multipliers[2] == 3.0 and curve.multipliers[11] == 1000.0
    assert curve.max_picks == 11


def test_a_curve_with_a_missing_tier_is_refused(tmp_path):
    """A streak advances one pick at a time, so a gap is a missing tier."""
    bad = tmp_path / "p.yaml"
    bad.write_text("platforms:\n  underdog:\n    streaks:\n      2: 3.0\n      4: 10.0\n")
    with pytest.raises(st.StreakError, match="skips tiers"):
        st.load_streak_curve("underdog", path=bad)


def test_the_full_ride_breaks_even_at_0_534_not_0_60():
    """
    THE NUMBER THE BRIEF'S 60% MOST NEEDS BESIDE IT. ``M[11] * p^11 = 1`` gives
    p = 0.534, so the one-step rule DECLINES legs that are in fact +EV. The
    brief's rule errs conservative, which is the safer direction, but it is
    still not the threshold.
    """
    curve = st.load_streak_curve("underdog")
    assert st.break_even_for_full_ride(curve) == pytest.approx(0.5337, abs=5e-4)
    assert st.break_even_for_full_ride(curve, picks=2) == pytest.approx(0.5774, abs=5e-4)


def test_holding_is_valued_by_backward_induction_not_one_step():
    """
    V(k) = max(cashout, p*V(k+1)) all the way down, so holding carries OPTION
    VALUE from being able to continue. A one-step comparison against the next
    tier's multiple systematically undervalues holding.
    """
    curve = st.load_streak_curve("underdog")
    values = st.streak_values(curve, per_leg_probability=0.60)
    assert values[11] == 1000.0
    # Option value: a live 4-pick streak is worth more than its own 10x tier.
    assert values[4] > curve.multiple(4)
    # And value must be monotone in the tier reached.
    assert all(values[k] < values[k + 1] for k in range(2, 11))


def test_the_entry_is_discounted_by_both_gatekeeper_legs():
    """
    Entering is the ONLY point at which 1 unit is genuinely the stake, and both
    gatekeeper picks must hit before any tier is reached — so p^2, not p.
    """
    curve = st.load_streak_curve("underdog")
    p = 0.60
    values = st.streak_values(curve, per_leg_probability=p)
    expected = p * p * values[2] - 1.0
    assert st.entry_expected_value(curve, per_leg_probability=p) == pytest.approx(expected)


@pytest.mark.parametrize("p,sign", [
    (0.60, 1), (0.58, 1), (0.55, 1), (0.53, -1), (0.50, -1),
])
def test_entry_ev_flips_sign_around_the_true_break_even(p, sign):
    """
    THE SENSITIVITY THAT DECIDES WHETHER ANY OF THIS IS REAL. A model believing
    0.60 when the truth is 0.53 — 11.7% optimistic — turns +2.63 per unit into
    a loss. `parlay.calibration_amplification` makes the same argument for
    ordinary tickets; a streak is that argument with eleven legs and a 1000x
    tail.
    """
    curve = st.load_streak_curve("underdog")
    ev = st.entry_expected_value(curve, per_leg_probability=p)
    assert math.copysign(1, ev) == sign, f"p={p} gave EV {ev:+.3f}"


def test_extending_is_priced_against_the_cashout_not_against_one_unit():
    """
    THE CORRECTION. Adding a leg at 3 picks risks the 6x already accumulated,
    not 1 unit: a miss forfeits the whole streak. So the comparison is
    p*V(4) vs cashout(3), and the break-even is cashout/V(next) — nothing like
    ``(p * marginal) - 1``.
    """
    curve = st.load_streak_curve("underdog")
    v = st.extension_verdict(curve, picks_so_far=3, next_leg_probability=0.60)
    assert v.cashout_value == pytest.approx(6.0)
    assert v.hold_value == pytest.approx(0.60 * st.streak_values(
        curve, per_leg_probability=0.60)[4])
    assert v.recommendation == "HOLD"
    assert "forfeits the whole streak" in v.reason


def test_a_discounted_cashout_lowers_the_bar_to_keep_going():
    """
    The 60% figure only coincides with the right answer when the cash-out is
    the FULL tier value. Books discount "Win Now", and a smaller cash-out makes
    holding easier to justify, not harder.
    """
    curve = st.load_streak_curve("underdog")
    full = st.extension_verdict(curve, picks_so_far=3, next_leg_probability=0.60,
                                cashout_fraction=1.0)
    disc = st.extension_verdict(curve, picks_so_far=3, next_leg_probability=0.60,
                                cashout_fraction=0.85)
    assert disc.break_even_probability < full.break_even_probability
    assert disc.cashout_value < full.cashout_value


def test_a_weak_leg_is_told_to_cash_out():
    curve = st.load_streak_curve("underdog")
    v = st.extension_verdict(curve, picks_so_far=10, next_leg_probability=0.20)
    assert v.recommendation == "CASH_OUT"


def test_there_is_no_decision_at_the_top_of_the_curve():
    curve = st.load_streak_curve("underdog")
    with pytest.raises(st.StreakError, match="top of the curve"):
        st.extension_verdict(curve, picks_so_far=11, next_leg_probability=0.9)


def test_a_streak_below_the_gatekeeper_is_not_a_streak():
    curve = st.load_streak_curve("underdog")
    with pytest.raises(st.StreakError, match="only exists from"):
        st.extension_verdict(curve, picks_so_far=1, next_leg_probability=0.9)


def test_nothing_here_places_or_sizes_anything():
    """
    `advisory_sizing` draws this line and this module follows it: every output
    is a figure and a reason. The cash-out verdict is a recommendation.
    """
    source = Path("src/quant/streaks.py").read_text(encoding="utf-8")
    for forbidden in ("place_bet", "submit", "AUTO_PLACED = True", "requests.post"):
        assert forbidden not in source, f"{forbidden} appeared in a research module"


def test_the_streak_curve_is_not_in_the_pickem_payout_catalogue():
    """
    A STREAK IS NOT A PICK'EM VARIANT, and the attempt to store it as one broke
    two of `test_dfs_payouts`'s own tests — correctly. That file's invariant is
    that every variant is a HITS -> MULTIPLE matrix for an entry bought at a
    fixed size; a streak is picks -> multiple, built one leg at a time, with
    cash-out and total forfeiture. Sharing the file would also have offered
    streaks to the entry builder, which prices static matrices and has no
    notion of a sequential position.
    """
    import yaml

    payouts = yaml.safe_load(Path("config/dfs_payouts.yaml").read_text())
    for platform, variants in (payouts.get("platforms") or {}).items():
        assert "streaks" not in variants, (
            f"{platform}: the streak curve is back in the pick'em catalogue"
        )
    streaks = yaml.safe_load(Path("config/streaks.yaml").read_text())
    assert streaks["platforms"]["underdog"]["streaks"]["11"
           if "11" in streaks["platforms"]["underdog"]["streaks"] else 11] == 1000.0
    assert streaks.get("as_of"), "an operator term with no date cannot be audited"


# --- the gatekeeper is not a cashable position -----------------------------


def test_there_is_no_cash_out_at_the_gatekeeper():
    """
    "WIN NOW" STARTS AT LEG 3. The operator's terms as supplied say the
    cash-out is offered after every correct pick FROM LEG 3 ONWARDS, so a
    2-pick streak cannot be collected. The first version took
    ``max(cashout, hold)`` at every tier including 2, and below p = 0.52 the
    3x floor won — so it returned CASH_OUT and recommended an action the
    operator does not offer.
    """
    curve = st.load_streak_curve("underdog")
    v = st.extension_verdict(curve, picks_so_far=2, next_leg_probability=0.45)

    assert v.cashout_value is None, "a 2-pick streak cannot be cashed"
    assert v.recommendation == "HOLD", "it recommended cashing out a position"
    assert v.break_even_basis == "no_offer"
    assert v.break_even_probability == 0.0
    assert "no cash-out" in v.reason


@pytest.mark.parametrize("p", [0.10, 0.30, 0.40, 0.45, 0.50])
def test_a_live_gatekeeper_is_not_floored_at_a_tier_it_cannot_collect(p):
    """
    V(2) = p * V(3), with no max. Flooring it at the 3x tier values the
    position at a payout that cannot be taken, and the error lands straight in
    the entry EV because the entry is p^2 * V(2).
    """
    curve = st.load_streak_curve("underdog")
    values = st.streak_values(curve, per_leg_probability=p)

    assert values[2] == pytest.approx(p * values[3])
    # `<=` rather than `<` for one real coincidence: at exactly p = 0.50 every
    # tier cashes, so V(3) = 6.0 and p*V(3) = 3.0 — the same number the floor
    # would have supplied. The two agree there and nowhere below it.
    assert values[2] <= curve.multiple(2), (
        "the gatekeeper is valued above a 3x it cannot collect"
    )
    if p < 0.50:
        assert values[2] < curve.multiple(2)

    # And the fiction is bounded: entry EV is no higher than the floored
    # version would report, and strictly lower wherever the floor bound.
    floored = p * p * max(curve.multiple(2), p * values[3]) - 1.0
    assert st.entry_expected_value(curve, per_leg_probability=p) <= floored


def test_the_correction_does_not_move_the_quoted_entry_evs():
    """
    MEASURED, NOT ASSUMED. Above about p = 0.52 the hold value at two picks
    already exceeds 3x, so the floor never bound there and this module's
    header figures stand. Pinned so a later change to the valuation cannot
    quietly rewrite the numbers that were reported.
    """
    curve = st.load_streak_curve("underdog")
    for p, expected in ((0.60, 2.628), (0.58, 1.499), (0.55, 0.393),
                        (0.534, 0.007), (0.53, -0.073)):
        got = st.entry_expected_value(curve, per_leg_probability=p)
        assert got == pytest.approx(expected, abs=1e-3), f"p={p} gave {got:+.4f}"


# --- the break-even has to be a threshold, not an echo ---------------------


def test_the_flat_break_even_does_not_move_with_the_p_it_is_given():
    """
    `cashout / V(k+1)` IS A FUNCTION OF ITS OWN INPUT when the continuation
    defaults to the same p: V(k+1) is computed at that p, so feeding 0.55 and
    0.60 returned two different "break-evens" and a flat p could not be
    compared against either. The fixed point is one number.
    """
    curve = st.load_streak_curve("underdog")
    reported = {
        st.extension_verdict(
            curve, picks_so_far=3, next_leg_probability=p
        ).break_even_probability
        for p in (0.30, 0.45, 0.52, 0.55, 0.60, 0.90)
    }
    assert len(reported) == 1, f"the break-even moved with its input: {reported}"


def test_at_the_reported_break_even_holding_and_cashing_are_equal():
    """
    The definition, checked rather than asserted: at p* a player assuming p*
    for every remaining leg is exactly indifferent.
    """
    curve = st.load_streak_curve("underdog")
    for picks in range(st.FIRST_CASHOUT_PICKS, curve.max_picks):
        star = st.extension_verdict(
            curve, picks_so_far=picks, next_leg_probability=0.5,
        ).break_even_probability
        at_star = st.extension_verdict(
            curve, picks_so_far=picks, next_leg_probability=star,
        )
        assert at_star.hold_value == pytest.approx(at_star.cashout_value, rel=1e-6), (
            f"{picks} picks: p*={star:.6f} is not the indifference point"
        )


def test_an_explicit_continuation_gets_the_exact_ratio_and_says_so():
    """
    When the caller SUPPLIES the continuation, V(k+1) no longer moves with p,
    so `cashout / V(k+1)` is exact and is the right answer for that leg. The
    two readings are different numbers and the verdict names which it gave.
    """
    curve = st.load_streak_curve("underdog")
    v = st.extension_verdict(
        curve, picks_so_far=3, next_leg_probability=0.60,
        continuation_probability=0.52,
    )
    values = st.streak_values(curve, per_leg_probability=0.52)
    assert v.break_even_basis == "next_leg"
    assert v.break_even_probability == pytest.approx(6.0 / values[4])

    flat = st.extension_verdict(curve, picks_so_far=3, next_leg_probability=0.60)
    assert flat.break_even_basis == "flat_all_legs"
    assert flat.break_even_probability != pytest.approx(v.break_even_probability)


# --- the curve reader is the only door, so it is the only guard ------------


@pytest.mark.parametrize("value,pattern", [
    (".nan", "positive, finite"),
    (".inf", "positive, finite"),
    ("0", "positive, finite"),
    ("-10.0", "positive, finite"),
    ("not_a_number", "not a multiple"),
])
def test_a_tier_that_is_not_a_multiple_is_refused(tmp_path, value, pattern):
    """
    `{int(k): float(v)}` RAISED A BARE ValueError ON A TYPO and accepted NaN,
    inf, zero and negatives without comment. A NaN tier makes every comparison
    in the induction False, so it silently takes whichever branch NaN loses
    to; a zero tier values a live streak at nothing.
    """
    bad = tmp_path / "s.yaml"
    bad.write_text(
        f"platforms:\n  underdog:\n    streaks:\n      2: 3.0\n      3: {value}\n",
        encoding="utf-8",
    )
    with pytest.raises(st.StreakError, match=pattern):
        st.load_streak_curve("underdog", path=bad)


def test_a_tier_that_pays_no_more_than_its_predecessor_is_refused(tmp_path):
    """
    Each tier is the one before it plus a leg that could have lost everything,
    so a tier paying less is a transposed digit (2.0 for 20.0) — and the
    induction would read it as a position to cash immediately and always,
    making a real recommendation out of a typo.
    """
    bad = tmp_path / "s.yaml"
    bad.write_text(
        "platforms:\n  underdog:\n    streaks:\n      2: 3.0\n      3: 6.0\n"
        "      4: 2.0\n",
        encoding="utf-8",
    )
    with pytest.raises(st.StreakError, match="no more than tier"):
        st.load_streak_curve("underdog", path=bad)


def test_a_loaded_curve_cannot_be_repriced_in_place():
    """
    `frozen=True` stops rebinding, not mutation. A curve is held for a whole
    run, so `curve.multipliers[11] = 1e9` repriced every valuation taken from
    it and outlived whatever made the edit.
    """
    curve = st.load_streak_curve("underdog")
    with pytest.raises(TypeError):
        curve.multipliers[11] = 1e9
    assert st.load_streak_curve("underdog").multipliers[11] == 1000.0


# --- the follow-up round: a fractional push is real on a fantasy score -----


def test_an_exact_fpts_tie_is_a_push_not_a_refusal():
    """
    THE GUARD'S PREMISE DID NOT HOLD FOR FPTS. `settle_prop` refuses a tie on
    a fractional line as "arithmetically impossible for an integer stat" —
    correct for every counting stat, and false for a fantasy score. 1.2*REB
    and 1.5*AST reach x.5 with entirely ordinary numbers: 20 PTS, 5 REB,
    1 AST is 27.5 EXACTLY, so a 27.5 line is a legitimate PUSH.

    Refused, it raised, the runner left the row PENDING, and every later run
    did the same while blaming a parsing bug that did not exist.
    """
    from decimal import Decimal

    from src.settlement.evaluator import settle_prop

    box = {"points": 20, "reboundsTotal": 5, "assists": 1,
           "steals": 0, "blocks": 0, "turnovers": 0}
    graded = settle_prop(
        market="FPTS", predicted_line=27.5, predicted_side="OVER",
        player_stats=box, odds=-110, platform="underdog",
    )
    assert str(graded.outcome) == "PUSH"
    assert graded.actual_result == Decimal("27.5")
    assert graded.profit_units == Decimal(0), "a push returns the stake"


def test_the_fractional_push_guard_still_bites_on_counting_stats():
    """
    The guard is not weakened, only made market-aware. A PTS prop tying a
    fractional line is still impossible and still refused loudly: that is a
    mis-parse, and recording it would launder a data bug into a graded row.
    """
    from src.settlement.evaluator import SettlementError, settle_prop

    for market, stats in (
        ("PTS", {"points": 20.5}),
        ("PRA", {"points": 20.5, "reboundsTotal": 0, "assists": 0}),
    ):
        with pytest.raises(SettlementError, match="arithmetically impossible"):
            settle_prop(
                market=market, predicted_line=20.5, predicted_side="OVER",
                player_stats=stats, odds=-110,
            )


def test_integer_fpts_weights_keep_the_guard(monkeypatch):
    """
    DERIVED FROM THE WEIGHTS, NOT THE MARKET'S NAME, and this test exists
    because the weaker version of it could not tell the difference: asserting
    only that FPTS-on-underdog is fractional passes for `return market ==
    FPTS` too, which is the "asserts the code looks right" trap cubic has
    been right about all session.

    So an operator whose matrix is ALL WHOLE NUMBERS is injected here. Its
    fantasy scores are integers, a fractional tie there really is impossible,
    and the guard must stay on — otherwise the exemption is "FPTS is special"
    rather than "these particular weights are fractional".
    """
    from src.settlement import evaluator as ev

    assert ev.fractional_values_possible("FPTS", platform="underdog") is True
    assert ev.fractional_values_possible("PTS") is False
    assert ev.fractional_values_possible("PRA") is False

    whole = {"PTS": 1.0, "REB": 1.0, "AST": 2.0,
             "STL": 3.0, "BLK": 3.0, "TOV": -1.0}
    monkeypatch.setattr(ev, "fantasy_weights", lambda _p: whole)
    assert ev.fractional_values_possible("FPTS", platform="underdog") is False, (
        "the exemption is keyed on the market's name, not on its weights"
    )

    # AND END TO END, on the case that is still a data bug. With integer
    # weights the score is an integer, so it can only tie a .5 line if a
    # COMPONENT arrived fractional — 20.5 points is a mis-parse, which is
    # exactly what the guard exists to refuse. It must still refuse it on
    # FPTS: the exemption is about fractional WEIGHTS, not about trusting
    # whatever the box score said.
    with pytest.raises(ev.SettlementError, match="arithmetically impossible"):
        ev.settle_prop(
            market="FPTS", predicted_line=27.5, predicted_side="OVER",
            player_stats={"points": 20.5, "reboundsTotal": 5, "assists": 1,
                          "steals": 0, "blocks": 0, "turnovers": 0},
            odds=-110, platform="underdog",
        )


def test_an_overflowed_variance_abstains_like_the_mean_does():
    """
    `_finite` guards each sd, but squaring and weighting six of them near the
    float ceiling overflows anyway. An sd of inf reads downstream as
    "infinitely uncertain" rather than "unusable" — every Over and Under
    equally likely, and the row looks priced instead of absent.
    """
    spec = cm.composite("PR")
    # `(1e200) ** 2` RAISES OverflowError while `1e200 * 1e200` returns inf,
    # so the accumulation fails both ways from inputs `_finite` accepted.
    # Either way the answer is "no usable spread", not a crash and not inf.
    with pytest.raises(cm.CompositeMarketError, match="not usable"):
        cm.composite_variance(spec, {"PTS": 1e200, "REB": 1e200},
                              {("PTS", "REB"): 0.4})
    with pytest.raises(cm.CompositeMarketError, match="not usable"):
        cm.composite_variance(spec, {"PTS": 1e180, "REB": 1e180},
                              {("PTS", "REB"): 0.4})


@pytest.mark.parametrize("document,pattern", [
    ("- not: a mapping\n", "does not contain a mapping"),
    ("platforms: [underdog]\n", "`platforms` is list"),
    ("platforms:\n  underdog: 3.0\n", "is float, not a"),
])
def test_malformed_streak_yaml_is_named_not_an_attribute_error(
    tmp_path, document, pattern,
):
    """
    `raw.get(...)` and `platforms.get(...)` ran before any mapping check, so a
    document that parsed as a list raised AttributeError — a traceback where
    the reader needs to be told which file is malformed and in what way.
    """
    bad = tmp_path / "s.yaml"
    bad.write_text(document, encoding="utf-8")
    with pytest.raises(st.StreakError, match=pattern):
        st.load_streak_curve("underdog", path=bad)
