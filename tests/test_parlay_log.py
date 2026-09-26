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
    AT_BET_TIME_FIELDS,
    SCHEMA_VERSION,
    TICKET_AT_BET_TIME_FIELDS,
    ParlayLegRecord,
    ParlayLogError,
    ParlayLogStore,
    ParlayTicketRecord,
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


# --- the snapshot the freeze used to miss --------------------------------

_LEG_SETTLEMENT_FIELDS = frozenset({
    "leg_result", "actual_stat", "void_reason", "closing_line",
    "closing_odds_american", "clv_line_points", "clv_prob_points",
    "settled_at_utc", "schema_version",
})
_TICKET_SETTLEMENT_FIELDS = frozenset({
    "ticket_result", "n_legs_won", "n_legs_lost", "n_legs_void",
    "settled_decimal_price", "settled_american_price", "net_return_units",
    "settled_at_utc", "notes",
})


def test_every_snapshot_field_on_both_records_is_frozen():
    """The frozen sets are enumerated by hand, so a field added to a record
    and not to its set is rewritable and nothing says so. game_id and
    model_push_prob were exactly that: snapshot fields absent from
    AT_BET_TIME_FIELDS."""
    leg_gap = set(ParlayLegRecord.model_fields) - AT_BET_TIME_FIELDS - _LEG_SETTLEMENT_FIELDS
    assert not leg_gap, f"leg snapshot fields not frozen: {sorted(leg_gap)}"

    ticket_gap = (
        set(ParlayTicketRecord.model_fields)
        - TICKET_AT_BET_TIME_FIELDS
        - _TICKET_SETTLEMENT_FIELDS
    )
    assert not ticket_gap, f"ticket snapshot fields not frozen: {sorted(ticket_gap)}"


@pytest.mark.parametrize("field,value", [("game_id", "G999"), ("model_push_prob", 0.42)])
def test_a_legs_game_and_push_mass_cannot_be_rewritten(tmp_path, field, value):
    """game_id is what the same-game correlation was chosen for and what the
    leg is joined to a box score by; model_push_prob is the mass the win/lose
    model was allowed to ignore. A settlement write moved L1 from G1 to G999
    and its push mass from 0.0 to 0.42 without complaint."""
    store = ParlayLogStore(tmp_path)
    ticket, legs = _ticket()
    store.append(ticket, legs)

    for leg in legs:
        leg.leg_result = "WIN"
    graded, graded_legs = grade_parlay(ticket, legs)
    setattr(graded_legs[0], field, value)
    with pytest.raises(ParlayLogError, match=f"{field} changed"):
        store.update_settlement(graded, graded_legs)


@pytest.mark.parametrize("field,value", [
    ("joint_probability", 0.95),
    ("ev_at_bet_time", 4.2),
    ("ticket_american_price", 5000),
    ("unit_stake", 100.0),
    ("model_logic", {"rewritten": True}),
])
def test_the_tickets_own_snapshot_cannot_be_rewritten(tmp_path, field, value):
    """
    update_settlement validated every leg and then replaced the ticket row
    wholesale with no comparison at all — so the two numbers the log exists to
    compare against outcomes were the easiest things in it to rewrite. A
    settlement write moved joint_probability from 0.330515 to 0.95,
    ev_at_bet_time from 0.179664 to 4.2, and the price from +257 to +5000.
    """
    store = ParlayLogStore(tmp_path)
    ticket, legs = _ticket()
    store.append(ticket, legs)

    for leg in legs:
        leg.leg_result = "WIN"
    graded, graded_legs = grade_parlay(ticket, legs)
    setattr(graded, field, value)
    with pytest.raises(ParlayLogError, match=f"{field} changed"):
        store.update_settlement(graded, graded_legs)


def test_a_legitimate_settlement_is_not_blocked_by_float_precision(tmp_path):
    """The guard above must not reject an UNCHANGED value. pandas writes 17
    significant digits and its default parser reads back 16, so
    breakeven_probability 0.28017718715393136 returned as 0.2801771871539313
    and the frozen check saw every computed float as rewritten."""
    store = ParlayLogStore(tmp_path)
    ticket, legs = _ticket()
    store.append(ticket, legs)

    stored = store.load_tickets().iloc[0]
    for field in ("joint_probability", "breakeven_probability", "ev_at_bet_time",
                  "ticket_decimal_price"):
        assert stored[field] == getattr(ticket, field), field

    for leg in legs:
        leg.leg_result = "WIN"
    graded, graded_legs = grade_parlay(ticket, legs)
    store.update_settlement(graded, graded_legs)
    assert store.load_tickets().iloc[0]["ticket_result"] == "WIN"


def test_a_logged_ticket_can_be_read_back_into_its_record(tmp_path):
    """
    pandas serialises a dict or list with str(), which is Python repr and not
    JSON, so a round trip left leg_ids as "['L1', 'L2', 'L3']" and model_logic
    as "{'min_ev': 0.02, ...}" — strings pydantic rejects with list_type and
    dict_type. `notify-discord --source parlay` reconstructs exactly this way
    and could not reload ANY ticket the store had written.
    """
    store = ParlayLogStore(tmp_path)
    ticket, legs = _ticket()
    store.append(ticket, legs)

    row = store.load_tickets().tail(1).iloc[0].dropna().to_dict()
    reloaded = ParlayTicketRecord(**row)
    assert reloaded.leg_ids == ["L1", "L2", "L3"]
    assert reloaded.model_logic == {"min_ev": 0.02, "dispersion_family": "negbin"}
    assert reloaded.joint_probability == ticket.joint_probability

    leg_rows = store.load_legs()
    for _, r in leg_rows.iterrows():
        ParlayLegRecord(**r.dropna().to_dict())


def test_a_log_written_with_python_repr_still_reads():
    """Logs written before the JSON fix carry repr. Refusing them would make
    the fix lose the history it was protecting."""
    reloaded = ParlayTicketRecord(
        ticket_id="abc", leg_ids="['L1', 'L2']", model_logic="{'min_ev': 0.02}",
    )
    assert reloaded.leg_ids == ["L1", "L2"]
    assert reloaded.model_logic == {"min_ev": 0.02}


# --- the CSV must give back the types it was handed ----------------------


def test_an_all_digit_ticket_id_survives_the_round_trip(tmp_path):
    """
    ticket_id is uuid4().hex[:16], which is ALL DIGITS about one run in 1,150.
    Only 15 of those 16 nibbles are random: hex[12] is the uuid4 version
    nibble and is always "4", itself a digit. So the rate is (10/16)**15 =
    1 in 1,153, not (10/16)**16. Measured directly over 4,000,000 draws:
    0.000866, i.e. 1 in 1,155.

    On such a draw pandas infers int64 on read, `ticket_id not in
    set(tickets["ticket_id"])` is True for the equal string, and
    update_settlement refuses a real settlement with "is not in the log".

    This is why the suite passed 719/719 locally and failed in CI on the draw
    6204641547604808. Pinned with that exact id so it is no longer a coin flip.
    """
    store = ParlayLogStore(tmp_path)
    ticket, legs = _ticket()
    ticket.ticket_id = "6204641547604808"
    for leg in legs:
        leg.ticket_id = ticket.ticket_id
    store.append(ticket, legs)

    stored = store.load_tickets()
    assert stored["ticket_id"].iloc[0] == ticket.ticket_id
    assert ticket.ticket_id in set(stored["ticket_id"])

    for leg in legs:
        leg.leg_result = "WIN"
    graded, graded_legs = grade_parlay(ticket, legs)
    store.update_settlement(graded, graded_legs)
    assert store.load_tickets().iloc[0]["ticket_result"] == "WIN"


def test_a_zero_padded_game_id_keeps_its_padding(tmp_path):
    """
    Not probabilistic, unlike the id above: NBA game ids are zero-padded and
    always numeric, so "0022500001" was inferred as int64 and read back as
    22500001 on EVERY row — losing the padding that joins a leg to its box
    score, in the one column the at-bet-time freeze was just extended to cover.
    """
    store = ParlayLogStore(tmp_path)
    padded = [
        ParlayLeg("L1", 0.60, -110, game_id="0022500001", line=24.5,
                  market="PTS", side="over", player_name="A"),
        ParlayLeg("L2", 0.55, -115, game_id="0022500002", line=7.5,
                  market="AST", side="over", player_name="B"),
    ]
    evaluation = evaluate_parlay(padded)
    assert evaluation.status == "OK", evaluation.reason
    ticket, legs = ticket_from_evaluation(evaluation, padded, slate_date="2026-09-21")
    store.append(ticket, legs)

    read_back = store.load_legs().sort_values("leg_id")
    assert list(read_back["game_id"]) == ["0022500001", "0022500002"]


def test_the_text_columns_are_derived_from_the_records_not_a_list(tmp_path):
    """The dtype map is built from the models, so a str field added to either
    record is covered without anyone updating a hardcoded list. Pins that the
    derivation actually reaches both records and the JSON-encoded fields."""
    from src.quant.parlay_log import _string_columns

    columns = _string_columns()
    for field in ("ticket_id", "leg_id", "game_id", "player_id", "slate_date",
                  "side", "market", "leg_result", "ticket_result",
                  "leg_ids", "model_logic"):
        assert field in columns, field
    # Numeric fields must NOT be forced to text, or arithmetic downstream breaks.
    for field in ("model_prob", "joint_probability", "unit_stake", "line"):
        assert field not in columns, field
