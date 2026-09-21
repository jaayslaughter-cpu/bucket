"""
Tests for src/quant/parlay_log.py.

The schema exists to stop a betting log from quietly becoming useless. The
three ways that happens each have a test: grading a parlay as the AND of its
legs, calibrating from ticket results instead of leg results, and rewriting
the at-bet-time snapshot when the model is re-run.
"""

from __future__ import annotations

import pytest

from src.quant.parlay import ParlayLeg, evaluate_parlay
from src.quant.parlay_log import (
    SCHEMA_VERSION,
    ParlayLogError,
    ParlayLogStore,
    assert_export_safe,
    grade_parlay,
    leg_calibration_frame,
    roi_summary,
    ticket_from_evaluation,
    to_payload,
)

LEGS = [
    ParlayLeg("L1", 0.60, -110, game_id="G1", line=24.5,
              player_name="DEMO_PLAYER_A", market="PTS", side="over"),
    ParlayLeg("L2", 0.55, -115, game_id="G2", line=7.5,
              player_name="DEMO_PLAYER_B", market="AST", side="over"),
    ParlayLeg("L3", 0.58, -105, game_id="G3", line=8.5,
              player_name="DEMO_PLAYER_C", market="REB", side="over"),
]


def _ticket(unit_stake: float = 1.0):
    evaluation = evaluate_parlay(LEGS)
    assert evaluation.status == "OK"
    return ticket_from_evaluation(
        evaluation, LEGS, slate_date="2026-09-21", unit_stake=unit_stake,
        model_logic={"min_ev": 0.02, "dispersion_family": "negbin"},
    )


# --- grading: a parlay is not the AND of its legs ------------------------


def test_a_voided_leg_reprices_the_ticket_instead_of_killing_it():
    """
    A late scratch VOIDS the leg and the ticket re-prices at the remaining
    legs' odds. Grading a parlay as the AND of its legs books this as a
    loss — wrong, and the kind of wrong that poisons a backtest.
    """
    ticket, legs = _ticket()
    original_price = ticket.ticket_american_price

    legs[0].leg_result, legs[0].actual_stat = "WIN", 28.0
    legs[1].leg_result, legs[1].void_reason = "VOID", "DNP - late scratch"
    legs[2].leg_result, legs[2].actual_stat = "WIN", 11.0

    graded, _ = grade_parlay(ticket, legs)
    assert graded.ticket_result == "WIN"
    assert graded.n_legs_void == 1 and graded.n_legs_won == 2
    assert graded.settled_american_price < original_price   # shorter price
    assert graded.net_return_units == pytest.approx(
        graded.settled_decimal_price - 1.0
    )


def test_a_push_leg_drops_out_the_same_way():
    ticket, legs = _ticket()
    legs[0].leg_result = "WIN"
    legs[1].leg_result = "PUSH"
    legs[2].leg_result = "WIN"
    graded, _ = grade_parlay(ticket, legs)
    assert graded.ticket_result == "WIN"
    assert graded.n_legs_void == 1


def test_every_leg_void_returns_the_stake():
    ticket, legs = _ticket(unit_stake=2.5)
    for leg in legs:
        leg.leg_result = "VOID"
    graded, _ = grade_parlay(ticket, legs)
    assert graded.ticket_result == "VOID"
    assert graded.net_return_units == 0.0
    assert graded.settled_decimal_price == 1.0


def test_a_surviving_loss_loses_the_ticket_whatever_else_voided():
    ticket, legs = _ticket(unit_stake=3.0)
    legs[0].leg_result = "WIN"
    legs[1].leg_result = "VOID"
    legs[2].leg_result = "LOSS"
    graded, _ = grade_parlay(ticket, legs)
    assert graded.ticket_result == "LOSS"
    assert graded.net_return_units == pytest.approx(-3.0)


def test_an_unresolved_leg_leaves_the_ticket_pending():
    ticket, legs = _ticket()
    legs[0].leg_result = "WIN"
    graded, _ = grade_parlay(ticket, legs)
    assert graded.ticket_result == "PENDING"
    assert graded.net_return_units is None


def test_legs_from_another_ticket_are_refused():
    ticket, legs = _ticket()
    other, other_legs = _ticket()
    with pytest.raises(ParlayLogError, match="do not belong"):
        grade_parlay(ticket, other_legs)


# --- what may and may not be logged --------------------------------------


def test_an_unpriced_ticket_is_never_logged():
    """A ticket with no price has no EV to compare an outcome against."""
    bad = evaluate_parlay([LEGS[0], ParlayLeg("L9", 0.55, None, game_id="G9", line=7.5)])
    with pytest.raises(ParlayLogError, match="unpriced ticket"):
        ticket_from_evaluation(bad, LEGS)


def test_credentials_and_pii_are_refused():
    with pytest.raises(ParlayLogError, match="credential"):
        assert_export_safe({"api_key": "abc123"})
    with pytest.raises(ParlayLogError, match="refusing"):
        assert_export_safe({"note": "postgresql://user:pw@host/db"})
    with pytest.raises(ParlayLogError, match="refusing"):
        assert_export_safe({"contact": "someone@example.com"})
    assert_export_safe({"min_ev": 0.02, "family": "negbin"})  # clean payload passes


def test_model_logic_carrying_a_secret_is_refused_at_build_time():
    evaluation = evaluate_parlay(LEGS)
    with pytest.raises(ParlayLogError):
        ticket_from_evaluation(evaluation, LEGS, model_logic={"api_key": "x"})


# --- the store: append-only, snapshot frozen -----------------------------


def test_round_trip_and_duplicate_refusal(tmp_path):
    store = ParlayLogStore(tmp_path)
    ticket, legs = _ticket()
    store.append(ticket, legs)

    assert len(store.load_tickets()) == 1
    assert len(store.load_legs()) == 3

    with pytest.raises(ParlayLogError, match="already logged"):
        store.append(ticket, legs)


def test_at_bet_time_fields_cannot_be_rewritten(tmp_path):
    """
    Re-running the model later and overwriting model_prob grades it on
    information it never had. Settlement columns may be filled in; the
    snapshot may not move.
    """
    store = ParlayLogStore(tmp_path)
    ticket, legs = _ticket()
    store.append(ticket, legs)

    for leg in legs:
        leg.leg_result = "WIN"
    graded, graded_legs = grade_parlay(ticket, legs)
    store.update_settlement(graded, graded_legs)          # settlement is fine

    graded_legs[0].model_prob = 0.99                       # the snapshot is not
    with pytest.raises(ParlayLogError, match="frozen"):
        store.update_settlement(graded, graded_legs)


# --- the feedback loop ---------------------------------------------------


def test_calibration_uses_leg_outcomes_and_skips_unresolved_legs(tmp_path):
    """
    Ticket win/loss is one draw from a joint distribution. The model's
    probabilities are per leg, so calibration has to be too.
    """
    store = ParlayLogStore(tmp_path)
    ticket, legs = _ticket()
    legs[0].leg_result = "WIN"
    legs[1].leg_result = "VOID"
    legs[2].leg_result = "LOSS"
    graded, graded_legs = grade_parlay(ticket, legs)
    store.append(graded, graded_legs)

    frame = leg_calibration_frame(store)
    assert len(frame) == 2                       # the VOID leg is not evidence
    assert set(frame["hit"]) == {0.0, 1.0}
    assert frame["model_prob"].notna().all()


def test_roi_excludes_void_stake_and_keeps_clv_out_of_it(tmp_path):
    store = ParlayLogStore(tmp_path)

    won, won_legs = _ticket(unit_stake=1.0)
    for leg in won_legs:
        leg.leg_result = "WIN"
        leg.clv_line_points = 0.5
    store.append(*grade_parlay(won, won_legs))

    voided, void_legs = _ticket(unit_stake=1.0)
    for leg in void_legs:
        leg.leg_result = "VOID"
    store.append(*grade_parlay(voided, void_legs))

    summary = roi_summary(store)
    assert summary["n_settled"] == 2
    assert summary["n_void"] == 1
    assert summary["staked_units"] == pytest.approx(1.0)   # the void is not staked
    assert summary["mean_clv_line_points"] == pytest.approx(0.5)
    # CLV lives beside ROI, never inside it.
    assert summary["net_return_units"] == pytest.approx(won.net_return_units)


def test_empty_store_abstains(tmp_path):
    assert roi_summary(ParlayLogStore(tmp_path))["status"] == "DATA_NOT_AVAILABLE"


def test_payload_shape_is_complete():
    ticket, legs = _ticket()
    payload = to_payload(ticket, legs)

    assert payload["schema_version"] == SCHEMA_VERSION
    for key in ("metadata", "probability", "legs", "model_logic", "tracking"):
        assert key in payload
    for key in ("timestamp" if False else "created_at_utc", "slate_date",
                "ticket_american_price", "placement_mode"):
        assert key in payload["metadata"]
    leg = payload["legs"][0]
    for key in ("player_name", "market", "line", "taken_odds_american",
                "model_prob_side", "edge_vs_devig", "model_version",
                "feature_schema_version"):
        assert key in leg
    for key in ("leg_result", "closing_line", "clv_line_points", "settled_at_utc"):
        assert key in leg["tracking"]
    for key in ("ticket_result", "net_return_units", "n_legs_void"):
        assert key in payload["tracking"]
    assert payload["metadata"]["placement_mode"] == "MANUAL_ONLY"
