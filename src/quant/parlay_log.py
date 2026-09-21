"""
src/quant/parlay_log.py — append-only parlay ticket log for backtesting.

Status: RESEARCH_ONLY · MANUAL_ONLY. This records tickets a person decided
to take, outside PropIQ. It never selects, places or sizes one.

WHAT THIS IS FOR: every logged ticket is a future backtest row. That makes
the schema's job preventing three specific ways a betting log quietly
becomes useless.

1. A PARLAY DOES NOT SETTLE AS THE AND OF ITS LEGS.

   For player props this is the common case, not an edge case: a late
   scratch VOIDS that leg, and the ticket re-prices at the remaining legs'
   odds. A 4-leg ticket at +900 with one DNP becomes a 3-leg ticket at
   whatever those three multiply to. A log that stores only "ticket
   won/lost" cannot reconstruct the grade, cannot tell a 3-for-4 loss from
   a 3-for-3 win after a void, and cannot be regraded when the source data
   is corrected. So legs are stored as their own rows and the ticket is
   graded FROM them.

2. LEG-LEVEL OUTCOMES ARE WHAT ACTUALLY CALIBRATE THE MODEL.

   Ticket win/loss is a single Bernoulli draw from a joint distribution and
   tells you almost nothing per ticket. The model's probabilities are
   per-leg, so calibration needs (model_prob, leg_result) pairs. A
   ticket-only log can never close the feedback loop it was built for.

3. AT-BET-TIME FIELDS MUST BE FROZEN.

   Re-running the model later and overwriting ``model_prob`` measures
   today's model against yesterday's outcomes — a backtest that grades the
   model on information it did not have. ``AT_BET_TIME_FIELDS`` is enforced
   on update: settlement columns may be filled in, the snapshot may not be
   touched.

CLV IS NOT EV AND IS NEVER SUMMED INTO ROI. They answer different
questions — EV asks whether the model was right, CLV asks whether the price
was. They are stored in separate columns and aggregated separately.

NOTHING SECRET GOES IN. ``assert_export_safe`` refuses any payload carrying
an api key, token, password, connection string or email address.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence
from uuid import uuid4

import pandas as pd
from pydantic import BaseModel, Field

from src.quant.odds_math import american_to_decimal, decimal_to_american
from src.utils.timezones import DISPLAY_TZ_NAME, format_pacific_iso, now_pacific

logger = logging.getLogger(__name__)

PLACEMENT_MODE = "MANUAL_ONLY"
RESEARCH_STATUS = "RESEARCH_ONLY"
SCHEMA_VERSION = "parlay_log_v1"

LegResult = Literal["WIN", "LOSS", "PUSH", "VOID", "PENDING"]
TicketResult = Literal["WIN", "LOSS", "VOID", "PENDING"]

# Frozen once written. Settlement may be filled in; the snapshot may not be
# rewritten, or the backtest grades the model on information it never had.
AT_BET_TIME_FIELDS: frozenset[str] = frozenset({
    "ticket_id", "leg_id", "created_at_utc", "slate_date",
    "player_name", "player_id", "market", "line", "side",
    "taken_odds_american", "model_prob", "fair_prob_at_bet",
    "ev_at_bet_time", "edge_vs_devig", "book_source", "bookmaker",
    "model_version", "feature_schema_version", "dispersion_family",
    "confidence_tier", "edge_letter_grade",
})

_SECRET_PATTERNS = (
    re.compile(r"api[_-]?key", re.I),
    re.compile(r"secret|password|passwd|token|bearer", re.I),
    re.compile(r"postgres(ql)?://|mysql://|mongodb(\+srv)?://", re.I),
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
)


class ParlayLogError(RuntimeError):
    """Raised when a ticket cannot be logged or graded from what was supplied."""


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------


class ParlayLegRecord(BaseModel):
    """One leg, with its at-bet-time snapshot and its settlement slots."""

    ticket_id: str
    leg_id: str
    created_at_utc: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    slate_date: str | None = None

    # --- at bet time (frozen) ---
    game_id: str | None = None
    player_id: str | None = None
    player_name: str | None = None
    market: str | None = None
    line: float | None = None
    side: str | None = None
    taken_odds_american: int | None = None
    # P(THE SIDE TAKEN) — never P(over) for an under leg.
    model_prob: float | None = None
    model_push_prob: float | None = None
    fair_prob_at_bet: float | None = None
    ev_at_bet_time: float | None = None
    edge_vs_devig: float | None = None
    book_source: str | None = None
    bookmaker: str | None = None
    model_version: str | None = None
    feature_schema_version: str | None = None
    dispersion_family: str | None = None
    confidence_tier: str | None = None
    edge_letter_grade: str | None = None

    # --- settlement (filled in later) ---
    leg_result: LegResult = "PENDING"
    actual_stat: float | None = None
    void_reason: str | None = None
    closing_line: float | None = None
    closing_odds_american: int | None = None
    # CLV is kept in its own columns and never added into ROI.
    clv_line_points: float | None = None
    clv_prob_points: float | None = None
    settled_at_utc: datetime | None = None

    schema_version: str = SCHEMA_VERSION


class ParlayTicketRecord(BaseModel):
    """The ticket: its price, its joint probability, and how it graded."""

    ticket_id: str = Field(default_factory=lambda: uuid4().hex[:16])
    created_at_utc: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_at_pt: str = Field(default_factory=lambda: format_pacific_iso(now_pacific()))
    slate_date: str | None = None
    timezone_display: str = DISPLAY_TZ_NAME

    n_legs: int = 0
    leg_ids: list[str] = Field(default_factory=list)

    # --- price and probability at bet time (frozen) ---
    ticket_decimal_price: float | None = None
    ticket_american_price: int | None = None
    joint_probability: float | None = None
    joint_probability_stderr: float | None = None
    independent_probability: float | None = None
    correlation_effect: float | None = None
    correlation_method: str | None = None
    correlation_matrix_json: str | None = None
    breakeven_probability: float | None = None
    ev_at_bet_time: float | None = None

    # YOUR stake. Never computed by the model, never Kelly.
    unit_stake: float = 1.0

    # --- the logic snapshot that produced the selection ---
    model_logic: dict[str, Any] = Field(default_factory=dict)

    # --- settlement ---
    ticket_result: TicketResult = "PENDING"
    n_legs_won: int = 0
    n_legs_lost: int = 0
    n_legs_void: int = 0
    settled_decimal_price: float | None = None
    settled_american_price: int | None = None
    net_return_units: float | None = None
    settled_at_utc: datetime | None = None

    placement_mode: str = PLACEMENT_MODE
    research_status: str = RESEARCH_STATUS
    schema_version: str = SCHEMA_VERSION
    notes: str = ""


# ---------------------------------------------------------------------------
# export safety
# ---------------------------------------------------------------------------


def assert_export_safe(payload: Any, *, path: str = "payload") -> None:
    """Refuse to write credentials, connection strings or email addresses."""
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            for pattern in _SECRET_PATTERNS[:2]:
                if pattern.search(str(key)):
                    raise ParlayLogError(
                        f"{path}.{key}: field name looks like a credential. Logs are "
                        "exports; secrets never go in one."
                    )
            assert_export_safe(value, path=f"{path}.{key}")
        return
    if isinstance(payload, (list, tuple)):
        for i, item in enumerate(payload):
            assert_export_safe(item, path=f"{path}[{i}]")
        return
    if isinstance(payload, str):
        for pattern in _SECRET_PATTERNS:
            if pattern.search(payload):
                raise ParlayLogError(
                    f"{path}: value matches {pattern.pattern!r} — refusing to write a "
                    "credential, connection string or email address into the log."
                )


# ---------------------------------------------------------------------------
# building a ticket from a priced evaluation
# ---------------------------------------------------------------------------


def ticket_from_evaluation(
    evaluation: Any,
    legs: Sequence[Any],
    *,
    slate_date: str | None = None,
    unit_stake: float = 1.0,
    model_logic: Mapping[str, Any] | None = None,
    leg_context: Mapping[str, Mapping[str, Any]] | None = None,
    correlation: Any = None,
) -> tuple[ParlayTicketRecord, list[ParlayLegRecord]]:
    """
    Turn a ``ParlayEvaluation`` plus its legs into log records.

    Refuses an evaluation that did not price. A ticket that was never priced
    has no EV, no joint probability and no reason to be in a dataset whose
    whole purpose is comparing those numbers to outcomes.
    """
    status = getattr(evaluation, "status", None)
    if status != "OK":
        raise ParlayLogError(
            f"Refusing to log an unpriced ticket (status={status!r}): "
            f"{getattr(evaluation, 'reason', None)}"
        )
    if unit_stake <= 0:
        raise ParlayLogError("unit_stake must be positive; it is YOUR chosen size")

    logic = dict(model_logic or {})
    assert_export_safe(logic, path="model_logic")

    matrix_json = None
    if correlation is not None:
        try:
            matrix_json = json.dumps(
                correlation.tolist() if hasattr(correlation, "tolist") else correlation,
                default=str,
            )
        except (TypeError, ValueError) as exc:  # noqa: BLE001 — recorded, not raised
            logger.warning("Could not serialise the correlation matrix: %s", exc)

    ticket = ParlayTicketRecord(
        slate_date=slate_date,
        n_legs=len(legs),
        leg_ids=[str(leg.leg_id) for leg in legs],
        ticket_decimal_price=evaluation.decimal_price,
        ticket_american_price=evaluation.american_price,
        joint_probability=evaluation.joint_probability,
        joint_probability_stderr=evaluation.joint_probability_stderr,
        independent_probability=evaluation.independent_probability,
        correlation_effect=evaluation.correlation_effect,
        correlation_method=evaluation.method,
        correlation_matrix_json=matrix_json,
        breakeven_probability=evaluation.breakeven_probability,
        ev_at_bet_time=evaluation.expected_value_per_unit,
        unit_stake=float(unit_stake),
        model_logic=logic,
    )

    context = leg_context or {}
    records: list[ParlayLegRecord] = []
    for leg in legs:
        extra = dict(context.get(str(leg.leg_id), {}))
        assert_export_safe(extra, path=f"leg_context.{leg.leg_id}")
        records.append(ParlayLegRecord(
            ticket_id=ticket.ticket_id,
            leg_id=str(leg.leg_id),
            slate_date=slate_date,
            game_id=getattr(leg, "game_id", None),
            player_name=getattr(leg, "player_name", None),
            market=getattr(leg, "market", None),
            line=getattr(leg, "line", None),
            side=getattr(leg, "side", None),
            taken_odds_american=getattr(leg, "american", None),
            model_prob=getattr(leg, "model_prob", None),
            model_push_prob=getattr(leg, "model_push_prob", None),
            **extra,
        ))
    return ticket, records


# ---------------------------------------------------------------------------
# grading
# ---------------------------------------------------------------------------


def grade_parlay(
    ticket: ParlayTicketRecord,
    legs: Sequence[ParlayLegRecord],
    *,
    settled_at_utc: datetime | None = None,
) -> tuple[ParlayTicketRecord, list[ParlayLegRecord]]:
    """
    Grade a ticket FROM its legs, re-pricing around every voided leg.

    A VOID or PUSH leg drops out and the ticket re-prices at the remaining
    legs' odds — this is the ordinary outcome for a player prop when someone
    is a late scratch, not a rare one. Grading a parlay as the AND of its
    legs books that ticket as a loss, which is both wrong and the kind of
    wrong that makes a historical dataset worse than none.

    Ticket outcomes:

    - any surviving leg LOST      -> LOSS, net -stake
    - every surviving leg WON     -> WIN, net stake * (settled_decimal - 1)
    - no legs survive             -> VOID, net 0 (stake returned)
    - anything still PENDING      -> PENDING, untouched
    """
    legs = list(legs)
    if not legs:
        raise ParlayLogError(f"Ticket {ticket.ticket_id} has no legs to grade")
    mismatched = [leg.leg_id for leg in legs if leg.ticket_id != ticket.ticket_id]
    if mismatched:
        raise ParlayLogError(
            f"Legs {mismatched} do not belong to ticket {ticket.ticket_id}"
        )

    if any(leg.leg_result == "PENDING" for leg in legs):
        ticket.ticket_result = "PENDING"
        return ticket, legs

    survivors = [leg for leg in legs if leg.leg_result in {"WIN", "LOSS"}]
    voided = [leg for leg in legs if leg.leg_result in {"VOID", "PUSH"}]

    ticket.n_legs_won = sum(1 for leg in legs if leg.leg_result == "WIN")
    ticket.n_legs_lost = sum(1 for leg in legs if leg.leg_result == "LOSS")
    ticket.n_legs_void = len(voided)
    ticket.settled_at_utc = settled_at_utc or datetime.now(timezone.utc)

    if not survivors:
        # Every leg voided: the stake comes back, the ticket is a no-action.
        ticket.ticket_result = "VOID"
        ticket.settled_decimal_price = 1.0
        ticket.settled_american_price = None
        ticket.net_return_units = 0.0
        return ticket, legs

    missing_price = [leg.leg_id for leg in survivors if leg.taken_odds_american is None]
    if missing_price:
        raise ParlayLogError(
            f"Cannot re-price ticket {ticket.ticket_id}: surviving legs "
            f"{missing_price} have no taken odds"
        )

    settled_decimal = 1.0
    for leg in survivors:
        settled_decimal *= american_to_decimal(int(leg.taken_odds_american))
    ticket.settled_decimal_price = settled_decimal
    ticket.settled_american_price = (
        decimal_to_american(settled_decimal) if settled_decimal > 1.0 else None
    )

    stake = float(ticket.unit_stake)
    if any(leg.leg_result == "LOSS" for leg in survivors):
        ticket.ticket_result = "LOSS"
        ticket.net_return_units = -stake
    else:
        ticket.ticket_result = "WIN"
        ticket.net_return_units = stake * (settled_decimal - 1.0)
    return ticket, legs


# ---------------------------------------------------------------------------
# the store
# ---------------------------------------------------------------------------


class ParlayLogStore:
    """Append-only ticket + leg log. Refuses to rewrite the at-bet-time snapshot."""

    def __init__(self, root: str | Path = "data/external/parlay_log") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def tickets_path(self) -> Path:
        return self.root / "parlay_tickets.csv"

    @property
    def legs_path(self) -> Path:
        return self.root / "parlay_legs.csv"

    def _read(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path)

    def load_tickets(self) -> pd.DataFrame:
        return self._read(self.tickets_path)

    def load_legs(self) -> pd.DataFrame:
        return self._read(self.legs_path)

    def append(
        self,
        ticket: ParlayTicketRecord,
        legs: Sequence[ParlayLegRecord],
    ) -> str:
        """Write a ticket and its legs. The ticket id must be new."""
        existing = self.load_tickets()
        if not existing.empty and ticket.ticket_id in set(existing["ticket_id"]):
            raise ParlayLogError(
                f"Ticket {ticket.ticket_id} is already logged. Use update_settlement "
                "to fill in outcomes; the at-bet-time snapshot never changes."
            )
        payload = ticket.model_dump(mode="json")
        assert_export_safe(payload, path="ticket")
        leg_rows = []
        for leg in legs:
            row = leg.model_dump(mode="json")
            assert_export_safe(row, path=f"leg.{leg.leg_id}")
            leg_rows.append(row)

        self._write(self.tickets_path, pd.DataFrame([payload]))
        self._write(self.legs_path, pd.DataFrame(leg_rows))
        logger.info(
            "logged parlay %s: %d legs, price %s, P=%s, EV=%s",
            ticket.ticket_id, ticket.n_legs, ticket.ticket_american_price,
            ticket.joint_probability, ticket.ev_at_bet_time,
        )
        return ticket.ticket_id

    def _write(self, path: Path, frame: pd.DataFrame) -> None:
        prior = self._read(path)
        combined = pd.concat([prior, frame], ignore_index=True) if not prior.empty else frame
        combined.to_csv(path, index=False)

    def update_settlement(
        self,
        ticket: ParlayTicketRecord,
        legs: Sequence[ParlayLegRecord],
    ) -> None:
        """
        Replace a ticket's settlement columns, leaving the snapshot untouched.

        Raises if an at-bet-time field changed since it was written: that is
        the difference between recording history and rewriting it.
        """
        tickets = self.load_tickets()
        if tickets.empty or ticket.ticket_id not in set(tickets["ticket_id"]):
            raise ParlayLogError(f"Ticket {ticket.ticket_id} is not in the log")

        stored_legs = self.load_legs()
        stored_for_ticket = stored_legs[stored_legs["ticket_id"] == ticket.ticket_id]
        for leg in legs:
            match = stored_for_ticket[stored_for_ticket["leg_id"] == leg.leg_id]
            if match.empty:
                raise ParlayLogError(f"Leg {leg.leg_id} is not in the log")
            stored = match.iloc[0]
            new = leg.model_dump(mode="json")
            for field in AT_BET_TIME_FIELDS & set(new):
                before, after = stored.get(field), new[field]
                if pd.isna(before) and after is None:
                    continue
                if str(before) != str(after):
                    raise ParlayLogError(
                        f"Leg {leg.leg_id}: {field} changed from {before!r} to "
                        f"{after!r}. At-bet-time fields are frozen — re-running the "
                        "model and overwriting them grades it on information it "
                        "never had."
                    )

        # Replace whole rows rather than assigning across a mask: an in-place
        # assignment has to match pandas' per-column dtypes, and a settlement
        # write legitimately turns None columns into floats.
        updated_ticket = pd.DataFrame([ticket.model_dump(mode="json")])
        tickets = pd.concat(
            [tickets[tickets["ticket_id"] != ticket.ticket_id], updated_ticket],
            ignore_index=True,
        )
        tickets.to_csv(self.tickets_path, index=False)

        touched = {(leg.ticket_id, leg.leg_id) for leg in legs}
        keep = stored_legs[
            ~stored_legs.apply(
                lambda r: (r["ticket_id"], r["leg_id"]) in touched, axis=1
            )
        ]
        replacements = pd.DataFrame([leg.model_dump(mode="json") for leg in legs])
        pd.concat([keep, replacements], ignore_index=True).to_csv(
            self.legs_path, index=False
        )


# ---------------------------------------------------------------------------
# the payload and the feedback loop
# ---------------------------------------------------------------------------


def to_payload(
    ticket: ParlayTicketRecord,
    legs: Sequence[ParlayLegRecord],
) -> dict[str, Any]:
    """The structured record: metadata, legs, logic snapshot, tracking slots."""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            "ticket_id": ticket.ticket_id,
            "created_at_utc": ticket.created_at_utc.isoformat(),
            "created_at_pt": ticket.created_at_pt,
            "timezone_display": ticket.timezone_display,
            "slate_date": ticket.slate_date,
            "placement_mode": ticket.placement_mode,
            "research_status": ticket.research_status,
            "n_legs": ticket.n_legs,
            "ticket_decimal_price": ticket.ticket_decimal_price,
            "ticket_american_price": ticket.ticket_american_price,
            "unit_stake": ticket.unit_stake,
        },
        "probability": {
            "joint_probability": ticket.joint_probability,
            "joint_probability_stderr": ticket.joint_probability_stderr,
            "independent_probability": ticket.independent_probability,
            "correlation_effect": ticket.correlation_effect,
            "correlation_method": ticket.correlation_method,
            "breakeven_probability": ticket.breakeven_probability,
            "ev_at_bet_time": ticket.ev_at_bet_time,
        },
        "legs": [
            {
                "leg_id": leg.leg_id,
                "game_id": leg.game_id,
                "player_name": leg.player_name,
                "market": leg.market,
                "line": leg.line,
                "side": leg.side,
                "taken_odds_american": leg.taken_odds_american,
                "model_prob_side": leg.model_prob,
                "fair_prob_at_bet": leg.fair_prob_at_bet,
                "edge_vs_devig": leg.edge_vs_devig,
                "ev_at_bet_time": leg.ev_at_bet_time,
                "book_source": leg.book_source,
                "bookmaker": leg.bookmaker,
                "model_version": leg.model_version,
                "feature_schema_version": leg.feature_schema_version,
                "dispersion_family": leg.dispersion_family,
                "confidence_tier": leg.confidence_tier,
                "edge_letter_grade": leg.edge_letter_grade,
                "tracking": {
                    "leg_result": leg.leg_result,
                    "actual_stat": leg.actual_stat,
                    "void_reason": leg.void_reason,
                    "closing_line": leg.closing_line,
                    "closing_odds_american": leg.closing_odds_american,
                    "clv_line_points": leg.clv_line_points,
                    "clv_prob_points": leg.clv_prob_points,
                    "settled_at_utc": (
                        leg.settled_at_utc.isoformat() if leg.settled_at_utc else None
                    ),
                },
            }
            for leg in legs
        ],
        "model_logic": ticket.model_logic,
        "tracking": {
            "ticket_result": ticket.ticket_result,
            "n_legs_won": ticket.n_legs_won,
            "n_legs_lost": ticket.n_legs_lost,
            "n_legs_void": ticket.n_legs_void,
            "settled_decimal_price": ticket.settled_decimal_price,
            "settled_american_price": ticket.settled_american_price,
            "net_return_units": ticket.net_return_units,
            "settled_at_utc": (
                ticket.settled_at_utc.isoformat() if ticket.settled_at_utc else None
            ),
        },
        "note": (
            "CLV is reported per leg and is NEVER summed into net_return_units. "
            "EV asks whether the model was right; CLV asks whether the price was."
        ),
    }
    assert_export_safe(payload)
    return payload


def leg_calibration_frame(store: ParlayLogStore) -> pd.DataFrame:
    """
    The (model_prob, hit) pairs that actually calibrate the model.

    Pushed and voided legs are excluded: a leg that never resolved is not
    evidence about a probability. Ticket-level results are deliberately not
    used here — one Bernoulli draw from a joint distribution says almost
    nothing about the per-leg numbers the model produces.
    """
    legs = store.load_legs()
    if legs.empty:
        return pd.DataFrame(columns=["model_prob", "hit"])
    resolved = legs[legs["leg_result"].isin(["WIN", "LOSS"])].copy()
    resolved = resolved[resolved["model_prob"].notna()]
    resolved["hit"] = (resolved["leg_result"] == "WIN").astype(float)
    return resolved


def roi_summary(store: ParlayLogStore) -> dict[str, Any]:
    """Ticket ROI over settled tickets. CLV is reported beside it, never inside."""
    tickets = store.load_tickets()
    legs = store.load_legs()
    base: dict[str, Any] = {
        "placement_mode": PLACEMENT_MODE,
        "research_status": RESEARCH_STATUS,
        "schema_version": SCHEMA_VERSION,
    }
    if tickets.empty:
        return {**base, "status": "DATA_NOT_AVAILABLE", "reason": "No tickets logged yet"}

    settled = tickets[tickets["ticket_result"].isin(["WIN", "LOSS", "VOID"])]
    staked = float(settled.loc[settled["ticket_result"] != "VOID", "unit_stake"].sum())
    net = float(settled["net_return_units"].fillna(0.0).sum())

    clv_mean = None
    if not legs.empty and legs["clv_line_points"].notna().any():
        clv_mean = round(float(legs["clv_line_points"].dropna().mean()), 6)

    return {
        **base,
        "status": "OK",
        "n_tickets": int(len(tickets)),
        "n_settled": int(len(settled)),
        "n_pending": int((tickets["ticket_result"] == "PENDING").sum()),
        "n_void": int((tickets["ticket_result"] == "VOID").sum()),
        "staked_units": staked,
        "net_return_units": net,
        "roi": round(net / staked, 6) if staked else None,
        "mean_clv_line_points": clv_mean,
        "n_resolved_legs": int(len(leg_calibration_frame(store))),
        "note": (
            "Research audit of a manual paper book. Not a profitability claim, "
            "and CLV is not part of ROI."
        ),
    }
