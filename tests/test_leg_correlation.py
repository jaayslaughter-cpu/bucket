"""
Tests for src/quant/leg_correlation.py.

The fitter's job is to turn realised games into the correlations
``evaluate_parlay`` refuses to guess. Its two disciplines are that it never
fits on the game being predicted, and that a bucket it could not fit stays
"unknown" rather than collapsing into "independent".
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from src.quant.leg_correlation import (
    OPPOSING_TEAM,
    SAME_PLAYER,
    SAME_TEAM,
    LegCorrelationError,
    classify_relationship,
    correlation_for_legs,
    fit_leg_correlations,
    realised_leg_outcomes,
)
from src.quant.parlay import ParlayLeg


def _planted_panel(rho: float = 0.45, n_games: int = 900, seed: int = 3) -> pd.DataFrame:
    """Two teammates whose latent scoring correlates at a known rho."""
    rng = np.random.default_rng(seed)
    chol = np.linalg.cholesky(np.array([[1.0, rho], [rho, 1.0]]))
    rows = []
    for g in range(n_games):
        z = rng.standard_normal(2) @ chol.T
        date = pd.Timestamp("2025-01-01") + pd.Timedelta(days=g % 120)
        for player, latent in (("A", z[0]), ("B", z[1])):
            rows.append({
                "PLAYER_ID": player, "PLAYER_NAME": player, "GAME_ID": f"G{g}",
                "GAME_DATE": date, "TEAM_ABBREVIATION": "AAA",
                "OPPONENT_ABBREVIATION": "BBB",
                "PTS": 20.0 + 6.0 * latent, "RESEARCH_LINE": 20.0,
            })
    return pd.DataFrame(rows)


def test_it_recovers_a_planted_correlation():
    priors = fit_leg_correlations(
        _planted_panel(rho=0.45), as_of="2026-01-01", markets=("PTS",), min_pairs=200,
    )
    bucket = priors.get(SAME_TEAM, "PTS", "PTS")
    assert bucket is not None and bucket.usable
    assert bucket.n_pairs == 900
    assert bucket.rho == pytest.approx(0.45, abs=0.08)


def test_as_of_is_required_and_excludes_the_slate_itself():
    """
    A correlation fitted on the game being predicted leaks into it, the same
    way a season-wide mean does.
    """
    panel = _planted_panel(n_games=300)
    with pytest.raises(LegCorrelationError, match="as_of is required"):
        realised_leg_outcomes(panel, markets=("PTS",), as_of=None)

    with pytest.raises(LegCorrelationError, match="nothing to fit on"):
        fit_leg_correlations(panel, as_of="2024-01-01", markets=("PTS",))

    cutoff = "2025-02-01"
    outcomes = realised_leg_outcomes(panel, markets=("PTS",), as_of=cutoff)
    assert outcomes["GAME_DATE"].max() < pd.Timestamp(cutoff)


def test_a_thin_bucket_is_unknown_not_independent():
    """
    The distinction matters: correlation_for_legs must be able to tell the
    caller it could not fit a pair, rather than handing back a 0 that reads
    as a measured independence.
    """
    priors = fit_leg_correlations(
        _planted_panel(n_games=30), as_of="2026-01-01", markets=("PTS",), min_pairs=200,
    )
    bucket = priors.get(SAME_TEAM, "PTS", "PTS")
    assert bucket is not None
    assert bucket.usable is False
    assert bucket.rho is None
    assert "not 'independent'" in bucket.reason

    legs = [
        ParlayLeg("L1", 0.60, -110, game_id="G1", line=24.5, market="PTS"),
        ParlayLeg("L2", 0.55, -110, game_id="G1", line=18.5, market="PTS"),
    ]
    matrix, unresolved = correlation_for_legs(
        legs, priors, leg_teams={"L1": "AAA", "L2": "AAA"},
    )
    assert matrix[0, 1] == 0.0          # nothing was measured
    assert len(unresolved) == 1          # and the caller is told so
    assert "L1xL2" in unresolved[0]


def test_fitted_priors_apply_to_a_prospective_ticket():
    priors = fit_leg_correlations(
        _planted_panel(rho=0.45), as_of="2026-01-01", markets=("PTS",), min_pairs=200,
    )
    legs = [
        ParlayLeg("L1", 0.60, -110, game_id="G1", line=24.5, market="PTS",
                  player_name="A"),
        ParlayLeg("L2", 0.55, -110, game_id="G1", line=18.5, market="PTS",
                  player_name="B"),
    ]
    matrix, unresolved = correlation_for_legs(
        legs, priors, leg_teams={"L1": "AAA", "L2": "AAA"},
    )
    assert not unresolved
    assert matrix[0, 1] == pytest.approx(priors.get(SAME_TEAM, "PTS", "PTS").rho)

    from src.quant.parlay import evaluate_parlay

    # Same-game legs, so the combined ticket price has to be the quoted one --
    # the product of the legs is not a same-game payout any book offers.
    evaluation = evaluate_parlay(legs, correlation=matrix, ticket_american=+240)
    assert evaluation.status == "OK"
    # Positive correlation lifts the ticket above the naive product.
    assert evaluation.joint_probability > evaluation.independent_probability


def test_cross_game_legs_are_left_alone():
    priors = fit_leg_correlations(
        _planted_panel(), as_of="2026-01-01", markets=("PTS",), min_pairs=200,
    )
    legs = [
        ParlayLeg("L1", 0.60, -110, game_id="G1", line=24.5, market="PTS"),
        ParlayLeg("L2", 0.55, -110, game_id="G2", line=18.5, market="PTS"),
    ]
    matrix, unresolved = correlation_for_legs(legs, priors)
    assert matrix[0, 1] == 0.0
    assert not unresolved           # different games are not an unresolved pair


def test_a_push_row_is_dropped_not_coerced():
    """
    A stat landing exactly on a whole line is a push: neither a win nor a
    loss. Coercing it to either biases every correlation fitted from it.
    """
    panel = pd.DataFrame({
        "PLAYER_ID": ["A", "B"], "PLAYER_NAME": ["A", "B"],
        "GAME_ID": ["G1", "G1"],
        "GAME_DATE": [pd.Timestamp("2025-01-01")] * 2,
        "TEAM_ABBREVIATION": ["AAA", "AAA"], "OPPONENT_ABBREVIATION": ["BBB", "BBB"],
        "PTS": [20.0, 25.0], "RESEARCH_LINE": [20.0, 20.0],
    })
    outcomes = realised_leg_outcomes(panel, markets=("PTS",), as_of="2026-01-01")
    assert len(outcomes) == 1
    assert outcomes["outcome"].iloc[0] == 1.0


def test_two_unknown_players_are_not_the_same_player():
    """
    A missing identity on both sides used to compare equal, so two different
    people were pooled into the same-player bucket — where a player's own
    PTS and REB correlate far more strongly than two people's do.
    """
    same_game = {"GAME_ID": "G1", "TEAM_ABBREVIATION": "AAA"}
    assert classify_relationship(
        {**same_game, "player_key": None}, {**same_game, "player_key": None}
    ) == SAME_TEAM


def test_relationship_classification():
    base = {"GAME_ID": "G1", "TEAM_ABBREVIATION": "AAA"}
    assert classify_relationship(
        {**base, "player_key": "A"}, {**base, "player_key": "A"}
    ) == SAME_PLAYER
    assert classify_relationship(
        {**base, "player_key": "A"}, {**base, "player_key": "B"}
    ) == SAME_TEAM
    assert classify_relationship(
        {**base, "player_key": "A"},
        {"GAME_ID": "G1", "TEAM_ABBREVIATION": "BBB", "player_key": "B"},
    ) == OPPOSING_TEAM
    # Unknown teams fall to the weaker bucket rather than inventing a pairing.
    assert classify_relationship(
        {"GAME_ID": "G1", "player_key": "A", "TEAM_ABBREVIATION": None},
        {"GAME_ID": "G1", "player_key": "B", "TEAM_ABBREVIATION": None},
    ) == OPPOSING_TEAM


def test_missing_inputs_are_refused_by_name():
    with pytest.raises(LegCorrelationError, match="missing"):
        realised_leg_outcomes(pd.DataFrame({"PTS": [1]}), markets=("PTS",), as_of="2026-01-01")

    no_line = _planted_panel(n_games=10).drop(columns=["RESEARCH_LINE"])
    with pytest.raises(LegCorrelationError, match="No usable markets"):
        realised_leg_outcomes(no_line, markets=("PTS",), as_of="2026-01-01")


def test_summary_carries_provenance():
    priors = fit_leg_correlations(
        _planted_panel(), as_of="2026-01-01", markets=("PTS",), min_pairs=200,
    )
    summary = priors.summary()
    assert summary["as_of"] == "2026-01-01"
    assert summary["n_games"] == 900
    assert summary["research_status"] == "RESEARCH_ONLY"
    # The provenance names the column ACTUALLY read, per market. This asserted
    # the literal "RESEARCH_LINE" while the field reported the requested
    # default, so it could not tell the two apart -- and the default itself was
    # a column the real panel does not have.
    assert summary["line_source"] == "PTS=RESEARCH_LINE"
    assert not priors.as_frame().empty


# --- cubic review, PR #3: rho's sign depends on each leg's side -------------


def _bucket(rho: float = 0.6):
    from src.quant.leg_correlation import CorrelationBucket

    class _Priors:
        def get(self, relationship, market_a, market_b):
            return CorrelationBucket(
                relationship=relationship, market_a=market_a, market_b=market_b,
                rho=rho, n_pairs=500, usable=True, reason=None,
            )

    return _Priors()


def _same_game_legs(side_a, side_b):
    from src.quant.parlay import ParlayLeg

    return [
        ParlayLeg(leg_id="a", model_prob=0.55, american=-110, game_id="G1",
                  player_name="A", market="PTS", side=side_a),
        ParlayLeg(leg_id="b", model_prob=0.55, american=-110, game_id="G1",
                  player_name="B", market="REB", side=side_b),
    ]


def _matrix_for(side_a, side_b, rho=0.6):
    from src.quant.leg_correlation import correlation_for_legs

    return correlation_for_legs(
        _same_game_legs(side_a, side_b), _bucket(rho),
        leg_teams={"a": "BOS", "b": "LAL"}, leg_players={"a": "A", "b": "B"},
    )


def test_a_mixed_over_under_pair_flips_the_fitted_rho():
    """The fitted buckets are OVER/OVER — realised_leg_outcomes defaults to
    side="over" — but model_prob is P(THIS side wins). Verified by simulation:
    with a latent over/over rho of 0.6 and 0.55 marginals, phi(over_a, over_b)
    = +0.4083 and phi(over_a, under_b) = -0.4083, summing to 0.000000.
    """
    same_a, _ = _matrix_for("over", "over")
    same_b, _ = _matrix_for("under", "under")
    mixed_a, _ = _matrix_for("over", "under")
    mixed_b, _ = _matrix_for("under", "over")

    assert same_a[0, 1] == pytest.approx(+0.6)
    assert same_b[0, 1] == pytest.approx(+0.6), "under/under is also same-sign"
    assert mixed_a[0, 1] == pytest.approx(-0.6)
    assert mixed_b[0, 1] == pytest.approx(-0.6), "the flip is symmetric in order"


def test_an_absent_side_reads_as_over_and_says_so(caplog):
    """An absent side is read as OVER — the side realised_leg_outcomes fits by
    default — so a caller that never populated the field keeps the behaviour it
    had. Refusing instead would turn every side-agnostic over/over ticket into
    an abstention, which fixes nothing.

    It must not be silent, though: a leg that MEANT under and omitted the field
    is signed wrongly, and saying so is the only cure.
    """
    with caplog.at_level(logging.INFO, logger="src.quant.leg_correlation"):
        matrix, unresolved = _matrix_for("over", None)

    assert not unresolved, "an absent side is not an unresolved pair"
    assert matrix[0, 1] == pytest.approx(+0.6), "read as over/over"
    assert any("declares no side" in r.message for r in caplog.records), (
        "the assumption must be logged"
    )


def test_a_declared_under_still_flips_even_when_the_other_side_is_absent():
    """The absent-side default must not swallow a side that WAS declared."""
    matrix, _ = _matrix_for(None, "under")

    assert matrix[0, 1] == pytest.approx(-0.6)


def test_side_spellings_are_recognised():
    from src.quant.leg_correlation import _side_sign

    class _Leg:
        def __init__(self, side):
            self.side = side

    assert _side_sign(_Leg("over")) == 1
    assert _side_sign(_Leg("Under")) == -1
    assert _side_sign(_Leg("O")) == 1
    assert _side_sign(_Leg("u")) == -1
    for unknown in (None, "", "whatever"):
        assert _side_sign(_Leg(unknown)) is None


# --- the two gates that decide whether a bucket is evidence --------------


def _one_game_panel(n_players: int = 25, n_games: int = 1) -> pd.DataFrame:
    """Many leg rows from FEW games — lots of pairs, almost no independence."""
    rng = np.random.default_rng(1)
    rows = []
    for g in range(n_games):
        for p in range(n_players):
            row = {
                "GAME_ID": f"G{g}", "GAME_DATE": "2024-12-01",
                "PLAYER_ID": f"p{p}", "PLAYER_NAME": f"p{p}",
                "TEAM_ABBREVIATION": "AAA", "OPPONENT_ABBREVIATION": "BBB",
            }
            for market, base in (("PTS", 20.0), ("REB", 6.0), ("AST", 4.0)):
                row[market] = base + rng.normal(scale=3.0)
                row[f"{market}_L10"] = base
            rows.append(row)
    return pd.DataFrame(rows)


def test_pairs_from_one_game_are_not_enough_however_many_there_are():
    """
    n_pairs counts pairs, and pairs inside one game reuse the same outcomes
    under that night's blowout, pace and officiating. 25 leg rows from a SINGLE
    game produce 2,700 pairs, and six buckets cleared the 200-pair gate on that
    alone with rho between -0.059 and +0.053 — noise wearing an n of 600. The
    independent unit is the game.
    """
    priors = fit_leg_correlations(_one_game_panel(), as_of="2025-01-01")
    assert priors.n_games == 1

    bucket = priors.get(SAME_TEAM, "PTS", "PTS")
    assert bucket is not None
    assert bucket.n_pairs > 200, "fixture no longer clears the pair gate"
    assert bucket.n_games == 1
    assert bucket.usable is False
    assert "distinct game" in bucket.reason
    assert "not 'independent'" in bucket.reason
    assert not any(b.usable for b in priors.buckets.values())


def test_the_games_gate_clears_once_there_are_enough_games():
    """The gate must not be unpassable: the same shape over 60 games fits."""
    priors = fit_leg_correlations(
        _one_game_panel(n_players=6, n_games=60), as_of="2025-01-01",
    )
    bucket = priors.get(SAME_TEAM, "PTS", "PTS")
    assert bucket is not None and bucket.usable, bucket.reason
    assert bucket.n_games == 60


def test_both_gates_are_reported_and_each_names_which_one_failed():
    thin_games = fit_leg_correlations(_one_game_panel(), as_of="2025-01-01")
    thin_pairs = fit_leg_correlations(
        _one_game_panel(n_players=3, n_games=60), as_of="2025-01-01", min_pairs=10_000,
    )
    assert "distinct game" in thin_games.get(SAME_TEAM, "PTS", "PTS").reason
    assert "realised pairs" in thin_pairs.get(SAME_TEAM, "PTS", "PTS").reason
    assert thin_games.summary()["min_games"] == 50
    assert thin_games.as_frame()["n_games"].max() == 1


def test_the_default_line_column_is_the_one_the_panel_actually_has():
    """
    This defaulted to a single shared "RESEARCH_LINE". The panel does not carry
    it — labels.attach_research_line derives it per market from {stat}_L10 and
    attaches it to ONE market's training frame — so the default fit skipped
    every market and raised "No usable markets" on the real 214,381-row panel:
    the layer evaluate_parlay depends on could not run at its own defaults. And
    one shared column cannot be the line for three markets anyway; it would
    grade REB legs against PTS's line.
    """
    panel = _one_game_panel(n_players=6, n_games=60)
    assert "RESEARCH_LINE" not in panel.columns

    priors = fit_leg_correlations(panel, as_of="2025-01-01")
    assert priors.line_source == "AST=AST_L10, PTS=PTS_L10, REB=REB_L10"
    assert any(b.usable for b in priors.buckets.values())

    # An explicit mapping still wins, and RESEARCH_LINE is still accepted for a
    # single market, where it is unambiguous.
    single = fit_leg_correlations(
        _planted_panel(n_games=200), as_of="2026-01-01", markets=("PTS",),
    )
    assert single.line_source == "PTS=RESEARCH_LINE"


def test_research_line_is_refused_for_a_multi_market_fit():
    """One column cannot be three markets' lines. Grading REB against PTS's
    line is a silently wrong outcome, so the market is skipped instead."""
    panel = _planted_panel(n_games=200)
    panel["REB"] = 6.0 + (panel["PTS"] - 20.0) / 4.0
    with pytest.raises(LegCorrelationError, match="No usable markets"):
        fit_leg_correlations(panel, as_of="2026-01-01", markets=("PTS", "REB"))
