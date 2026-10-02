"""Where the parlay ledger's rows live, and what must not move with them.

The ledger was two CSVs under `data/`, which is gitignored and, on a container,
ephemeral: a redeploy destroyed every tracked ticket along with the at-bet-time
probability and EV that are the only record of what was believed before a game.
Moving it into Postgres is that fix.

The property these tests protect is that MOVING THE ROWS MOVED NOTHING ELSE.
Every rule — a ticket id must be new, at-bet-time fields are frozen — lives in
ParlayLogStore and holds whatever the rows are kept in. A rule that lived in one
backend and not the other is how the two would come to disagree about what the
ledger means.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.quant.parlay import ParlayLeg, evaluate_parlay
from src.quant.parlay_log import (
    LEGS as LEGS_TABLE,
)
from src.quant.parlay_log import (
    TICKETS,
    CsvLedgerBackend,
    ParlayLogError,
    ParlayLogStore,
    PostgresLedgerBackend,
    grade_parlay,
    ticket_from_evaluation,
)

LEGS = [
    ParlayLeg("L1", 0.60, -110, game_id="G1", line=24.5,
              player_name="DEMO_PLAYER_A", market="PTS", side="over"),
    ParlayLeg("L2", 0.55, -115, game_id="G2", line=7.5,
              player_name="DEMO_PLAYER_B", market="AST", side="over"),
]


class InMemoryBackend:
    """A third backend, existing only to show the rules are not in a backend."""

    def __init__(self) -> None:
        self.tables: dict[str, pd.DataFrame] = {}
        self.calls: list[tuple[str, str, int]] = []

    def load(self, table: str) -> pd.DataFrame:
        self.calls.append(("load", table, 0))
        return self.tables.get(table, pd.DataFrame()).copy()

    def append(self, table: str, frame: pd.DataFrame) -> None:
        self.calls.append(("append", table, len(frame)))
        prior = self.tables.get(table)
        self.tables[table] = (
            pd.concat([prior, frame], ignore_index=True)
            if prior is not None and not prior.empty else frame.copy()
        )

    def replace(self, table: str, frame: pd.DataFrame) -> None:
        self.calls.append(("replace", table, len(frame)))
        self.tables[table] = frame.copy()


def _ticket():
    evaluation = evaluate_parlay(LEGS)
    assert evaluation.status == "OK"
    return ticket_from_evaluation(
        evaluation, LEGS, slate_date="2026-09-21", unit_stake=1.0,
    )


# --- the rules are in the store, not in a backend ------------------------

def test_a_ticket_round_trips_through_an_arbitrary_backend():
    backend = InMemoryBackend()
    store = ParlayLogStore(backend=backend)
    ticket, legs = _ticket()

    store.append(ticket, legs)
    stored = store.load_tickets()
    assert list(stored["ticket_id"]) == [ticket.ticket_id]
    assert len(store.load_legs()) == 2
    assert ("append", TICKETS, 1) in backend.calls
    assert ("append", LEGS_TABLE, 2) in backend.calls


def test_a_duplicate_ticket_is_refused_whatever_the_backend():
    store = ParlayLogStore(backend=InMemoryBackend())
    ticket, legs = _ticket()
    store.append(ticket, legs)
    with pytest.raises(ParlayLogError, match="already logged"):
        store.append(ticket, legs)


def test_the_at_bet_time_freeze_survives_the_move():
    """
    The rule the whole log exists for: a re-run of the model must not overwrite
    the probability the ticket was logged with, because grading it against a
    rewritten number grades it on information it never had.
    """
    store = ParlayLogStore(backend=InMemoryBackend())
    ticket, legs = _ticket()
    store.append(ticket, legs)

    ticket.joint_probability = 0.99
    with pytest.raises(ParlayLogError, match="frozen"):
        store.update_settlement(ticket, legs)


def test_a_settlement_updates_through_the_backend_without_touching_the_snapshot():
    backend = InMemoryBackend()
    store = ParlayLogStore(backend=backend)
    ticket, legs = _ticket()
    store.append(ticket, legs)
    logged_probability = ticket.joint_probability

    for leg in legs:
        leg.leg_result = "WIN"
        leg.actual_stat = 30.0
    settled_ticket, settled_legs = grade_parlay(ticket, legs)
    store.update_settlement(settled_ticket, settled_legs)

    stored = store.load_tickets()
    assert len(stored) == 1, "settlement duplicated the ticket"
    row = stored.iloc[0]
    assert row["ticket_result"] == "WIN"
    assert row["joint_probability"] == pytest.approx(logged_probability)
    assert ("replace", TICKETS, 1) in backend.calls


def test_a_settlement_leaves_other_tickets_alone():
    store = ParlayLogStore(backend=InMemoryBackend())
    first_ticket, first_legs = _ticket()
    second_ticket, second_legs = _ticket()
    store.append(first_ticket, first_legs)
    store.append(second_ticket, second_legs)

    for leg in first_legs:
        leg.leg_result = "WIN"
        leg.actual_stat = 30.0
    graded, graded_legs = grade_parlay(first_ticket, first_legs)
    store.update_settlement(graded, graded_legs)

    stored = store.load_tickets()
    assert set(stored["ticket_id"]) == {
        first_ticket.ticket_id, second_ticket.ticket_id,
    }
    untouched = stored[stored["ticket_id"] == second_ticket.ticket_id].iloc[0]
    assert untouched["ticket_result"] == "PENDING"
    assert len(store.load_legs()) == 4


# --- the CSV backend is still the default --------------------------------

def test_the_csv_backend_is_the_default_and_the_old_signature_still_works(tmp_path):
    store = ParlayLogStore(tmp_path / "ledger")
    assert isinstance(store.backend, CsvLedgerBackend)
    ticket, legs = _ticket()
    store.append(ticket, legs)
    assert store.tickets_path.exists()
    assert store.legs_path.exists()


def test_a_backend_with_no_files_reports_no_paths():
    """A path that is never written to reads as a file someone can go and open."""
    store = ParlayLogStore(backend=InMemoryBackend())
    assert store.tickets_path is None
    assert store.legs_path is None
    assert store.root is None


def test_an_unknown_table_is_refused_by_both_backends(tmp_path):
    with pytest.raises(ParlayLogError, match="Unknown ledger table"):
        CsvLedgerBackend(tmp_path).load("wagers")
    with pytest.raises(ParlayLogError, match="Unknown ledger table"):
        PostgresLedgerBackend().load("wagers")


# --- the Postgres payload, without a database ---------------------------

def test_the_record_round_trips_through_the_jsonb_payload():
    """
    JSONB preserves types, which is the point: the CSV backend needs explicit
    reader settings for float precision and zero-padded ids, and this does not.
    """
    from src.db.repository import _parlay_payload

    ticket, legs = _ticket()
    store = ParlayLogStore(backend=InMemoryBackend())
    store.append(ticket, legs)
    original = store.load_legs()

    payloads = [_parlay_payload("legs", r) for r in original.to_dict("records")]
    rebuilt = pd.DataFrame([p["record"] for p in payloads])

    assert list(rebuilt.columns) == list(original.columns)
    assert rebuilt["model_prob"].iloc[0] == original["model_prob"].iloc[0]
    assert rebuilt["game_id"].iloc[0] == "G1", "a zero-paddable id changed type"


def test_the_promoted_columns_are_lifted_out_of_the_record_not_instead_of_it():
    from src.db.repository import _PARLAY_PROMOTED, _parlay_payload

    ticket, legs = _ticket()
    store = ParlayLogStore(backend=InMemoryBackend())
    store.append(ticket, legs)
    row = store.load_tickets().to_dict("records")[0]

    payload = _parlay_payload("tickets", row)
    assert payload["ticket_id"] == ticket.ticket_id
    assert payload["record"]["ticket_id"] == ticket.ticket_id, (
        "the record must still carry the promoted keys; it is the truth and "
        "the columns are a read convenience"
    )
    for column in _PARLAY_PROMOTED["tickets"]:
        assert column in payload


def test_every_promoted_column_is_a_field_the_record_actually_has():
    """
    A promoted column the record never carries would be NULL on every row, and
    a query filtering on it would silently return nothing.
    """
    from src.db.repository import _PARLAY_PROMOTED
    from src.quant.parlay_log import ParlayLegRecord, ParlayTicketRecord

    for table, model in (
        ("tickets", ParlayTicketRecord), ("legs", ParlayLegRecord),
    ):
        missing = set(_PARLAY_PROMOTED[table]) - set(model.model_fields)
        assert not missing, f"{table} promotes {sorted(missing)}, which {model} lacks"


def test_a_datetime_is_serialised_rather_than_failing_the_write():
    """
    JSONB holds JSON. A datetime that raised here would take down the whole
    ticket at write time, after the model had already committed to it, so it is
    stringified instead.
    """
    from datetime import datetime, timezone

    from src.db.repository import _parlay_payload

    payload = _parlay_payload("tickets", {
        "ticket_id": "abc", "created_at_utc": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "n_legs": 2, "slate_date": "2026-01-01",
    })
    assert isinstance(payload["record"]["created_at_utc"], str)


def test_a_nan_becomes_null_rather_than_a_float_nan():
    """JSON has no NaN; a bare float('nan') produces invalid JSON."""
    import json

    from src.db.repository import _parlay_payload

    payload = _parlay_payload("legs", {
        "ticket_id": "abc", "leg_id": "L1", "model_prob": float("nan"),
    })
    assert payload["record"]["model_prob"] is None
    json.loads(json.dumps(payload["record"]))   # must be valid JSON


# --- choosing a backend --------------------------------------------------

def test_the_default_is_still_csv(tmp_path, monkeypatch):
    from src.quant.parlay_log import ENV_LEDGER_BACKEND, open_parlay_log

    monkeypatch.delenv(ENV_LEDGER_BACKEND, raising=False)
    assert isinstance(open_parlay_log(tmp_path).backend, CsvLedgerBackend)


def test_the_environment_can_move_the_ledger_into_postgres(tmp_path, monkeypatch):
    from src.quant.parlay_log import ENV_LEDGER_BACKEND, open_parlay_log

    monkeypatch.setenv(ENV_LEDGER_BACKEND, "postgres")
    assert isinstance(open_parlay_log(tmp_path).backend, PostgresLedgerBackend)


def test_a_typo_raises_rather_than_silently_using_the_ephemeral_ledger(
    tmp_path, monkeypatch
):
    """
    Falling back to CSV on a typo would look like it worked and lose the ledger
    at the next restart — the failure this whole change exists to prevent.
    """
    from src.quant.parlay_log import ENV_LEDGER_BACKEND, open_parlay_log

    monkeypatch.setenv(ENV_LEDGER_BACKEND, "postgress")
    with pytest.raises(ParlayLogError, match="not a ledger backend"):
        open_parlay_log(tmp_path)


def test_an_explicit_argument_beats_the_environment(tmp_path, monkeypatch):
    from src.quant.parlay_log import ENV_LEDGER_BACKEND, open_parlay_log

    monkeypatch.setenv(ENV_LEDGER_BACKEND, "postgres")
    store = open_parlay_log(tmp_path, ledger="csv")
    assert isinstance(store.backend, CsvLedgerBackend)
