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
from pathlib import Path

import pytest

from src.quant import composite_markets as cm
from src.quant import streaks as st

# --- the decomposition, and that nothing else disagrees with it -------------


def test_the_sum_composites_match_the_settlement_grader():
    """
    ONE DEFINITION OR THE LEDGER AND THE PROJECTION DIVERGE. `evaluator.
    MARKET_COMPONENTS` decides what a graded PA actually totals; this module
    decides what the projection totals. If they ever disagree, every PA row is
    projected on one definition and settled on another, and both look fine.
    """
    from src.settlement.evaluator import MARKET_COMPONENTS

    cdn_to_panel = {
        "points": "PTS", "reboundsTotal": "REB", "assists": "AST",
        "steals": "STL", "blocks": "BLK", "turnovers": "TOV",
    }
    for market, components in cm.SUM_COMPOSITES.items():
        graded = MARKET_COMPONENTS.get(market)
        assert graded is not None, f"{market} projects but settlement cannot grade it"
        assert tuple(cdn_to_panel[c] for c in graded) == components, (
            f"{market}: settlement totals {graded}, this module totals {components}"
        )


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
    A negative variance means the supplied correlations are not a consistent
    matrix. Returning it would hand the caller an imaginary standard
    deviation; `quant.psd_repair` exists for exactly this.
    """
    spec = cm.composite("PRA")
    sds = {"PTS": 5.0, "REB": 5.0, "AST": 5.0}
    impossible = {("PTS", "REB"): -1.0, ("PTS", "AST"): -1.0, ("REB", "AST"): -1.0}
    with pytest.raises(cm.CompositeMarketError, match="NEGATIVE variance"):
        cm.composite_variance(spec, sds, impossible)


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
