"""
Tests for src/quant/leg_correlation.py.

The fitter's job is to turn realised games into the correlations
``evaluate_parlay`` refuses to guess. Its two disciplines are that it never
fits on the game being predicted, and that a bucket it could not fit stays
"unknown" rather than collapsing into "independent".
"""

from __future__ import annotations

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

    evaluation = evaluate_parlay(legs, correlation=matrix)
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
    assert summary["line_source"] == "RESEARCH_LINE"
    assert not priors.as_frame().empty
