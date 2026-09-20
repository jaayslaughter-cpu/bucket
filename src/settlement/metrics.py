"""
src/settlement/metrics.py — W-L-P record, strike rate, ROI, CLV.

THREE DELIBERATE SEPARATIONS
----------------------------
1. Strike rate uses DECIDED props only (WIN + LOSS). Pushes and voids
   are reported but excluded from the denominator. Including pushes
   understates the rate; hiding the count overstates certainty — both
   numbers are always returned.

2. ROI uses PRICED props only (odds IS NOT NULL). Pick'em rows carry a
   payout multiplier, not a two-way American price, so ROI is undefined
   for them. They still count toward W-L-P. `unpriced_graded_n` makes
   the size of that exclusion visible rather than silently dropping it.

3. CLV IS NOT ROI. It is a market-quality signal — whether the line/price
   moved toward you after you took it. It correlates with long-run edge
   in the literature, but it is not profit and is never summed into the
   ROI figure. Reported in its own block.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import date
from decimal import Decimal
from typing import Any

from sqlalchemy import and_, func, select

from src.db.models import PropResult
from src.db.session import session_scope

logger = logging.getLogger(__name__)


@dataclass
class RecordBlock:
    graded_n: int = 0
    wins: int = 0
    losses: int = 0
    pushes: int = 0
    voids: int = 0
    pending: int = 0
    decided_n: int = 0
    strike_rate_pct: float | None = None
    record: str = "0-0-0"


@dataclass
class RoiBlock:
    priced_n: int = 0
    unpriced_graded_n: int = 0
    staked_units: float | None = None
    profit_units: float | None = None
    roi_pct: float | None = None
    note: str = ""


@dataclass
class ClvBlock:
    n_with_line_clv: int = 0
    n_with_price_clv: int = 0
    avg_clv_line_points: float | None = None
    avg_clv_prob_points: float | None = None
    pct_positive_line_clv: float | None = None
    note: str = (
        "CLV is a market-quality signal, not profit, and is never included "
        "in ROI. Positive CLV does not guarantee future profitability."
    )


@dataclass
class PerformanceSummary:
    record: RecordBlock = field(default_factory=RecordBlock)
    roi: RoiBlock = field(default_factory=RoiBlock)
    clv: ClvBlock = field(default_factory=ClvBlock)
    filters: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


MIN_SAMPLE_FOR_RATE = 30  # below this, a strike rate is noise, not signal


def _f(value: Any) -> float | None:
    if value is None:
        return None
    return float(value) if not isinstance(value, Decimal) else float(value)


def get_performance_summary(
    start_date: date | None = None,
    end_date: date | None = None,
    market: str | None = None,
    source: str | None = None,
    include_pickem: bool = True,
) -> PerformanceSummary:
    """
    Compute W-L-P, strike rate, ROI, and CLV over the filtered set.

    All aggregation happens in Postgres — no loading rows into Python
    just to count them.
    """
    summary = PerformanceSummary()
    summary.filters = {
        "start_date": str(start_date) if start_date else None,
        "end_date": str(end_date) if end_date else None,
        "market": market,
        "source": source,
        "include_pickem": include_pickem,
    }

    conditions = []
    if start_date:
        conditions.append(PropResult.game_date >= start_date)
    if end_date:
        conditions.append(PropResult.game_date <= end_date)
    if market:
        conditions.append(PropResult.market == market.upper())
    if source:
        conditions.append(PropResult.source == source)
    if not include_pickem:
        conditions.append(PropResult.is_pickem.is_(False))

    where = and_(*conditions) if conditions else None

    def count_where(*extra) -> Any:
        return func.count().filter(and_(*extra)) if extra else func.count()

    with session_scope() as db:
        stmt = select(
            count_where(PropResult.outcome_status.in_(("WIN", "LOSS", "PUSH"))).label("graded_n"),
            count_where(PropResult.outcome_status == "WIN").label("wins"),
            count_where(PropResult.outcome_status == "LOSS").label("losses"),
            count_where(PropResult.outcome_status == "PUSH").label("pushes"),
            count_where(PropResult.outcome_status == "VOID").label("voids"),
            count_where(PropResult.outcome_status == "PENDING").label("pending"),
            count_where(
                PropResult.odds.isnot(None),
                PropResult.outcome_status.in_(("WIN", "LOSS", "PUSH")),
            ).label("priced_n"),
            count_where(
                PropResult.odds.is_(None),
                PropResult.outcome_status.in_(("WIN", "LOSS", "PUSH")),
            ).label("unpriced_graded_n"),
            func.sum(PropResult.stake_units).filter(PropResult.odds.isnot(None)).label("staked"),
            func.sum(PropResult.profit_units).filter(PropResult.odds.isnot(None)).label("profit"),
            func.avg(PropResult.clv_line_points).label("avg_clv_line"),
            func.avg(PropResult.clv_prob_points).label("avg_clv_prob"),
            count_where(PropResult.clv_line_points.isnot(None)).label("n_line_clv"),
            count_where(PropResult.clv_prob_points.isnot(None)).label("n_price_clv"),
            count_where(PropResult.clv_line_points > 0).label("n_positive_line_clv"),
        )
        if where is not None:
            stmt = stmt.where(where)

        row = db.execute(stmt).one()

    # --- record -------------------------------------------------------
    rec = summary.record
    rec.graded_n = row.graded_n or 0
    rec.wins = row.wins or 0
    rec.losses = row.losses or 0
    rec.pushes = row.pushes or 0
    rec.voids = row.voids or 0
    rec.pending = row.pending or 0
    rec.decided_n = rec.wins + rec.losses
    rec.record = f"{rec.wins}-{rec.losses}-{rec.pushes}"
    if rec.decided_n > 0:
        rec.strike_rate_pct = round(rec.wins / rec.decided_n * 100, 2)

    # --- roi ----------------------------------------------------------
    roi = summary.roi
    roi.priced_n = row.priced_n or 0
    roi.unpriced_graded_n = row.unpriced_graded_n or 0
    roi.staked_units = _f(row.staked)
    roi.profit_units = _f(row.profit)
    if roi.staked_units:
        roi.roi_pct = round(roi.profit_units / roi.staked_units * 100, 2)
    if roi.unpriced_graded_n:
        roi.note = (
            f"{roi.unpriced_graded_n} graded prop(s) excluded from ROI — pick'em "
            f"payout multipliers are not two-way American odds, so ROI is "
            f"undefined for them. They ARE included in the W-L-P record."
        )
    elif roi.priced_n == 0:
        roi.note = "No priced props settled — ROI not computable."

    # --- clv ----------------------------------------------------------
    clv = summary.clv
    clv.n_with_line_clv = row.n_line_clv or 0
    clv.n_with_price_clv = row.n_price_clv or 0
    clv.avg_clv_line_points = round(_f(row.avg_clv_line), 4) if row.avg_clv_line is not None else None
    clv.avg_clv_prob_points = round(_f(row.avg_clv_prob), 5) if row.avg_clv_prob is not None else None
    if clv.n_with_line_clv:
        clv.pct_positive_line_clv = round((row.n_positive_line_clv or 0) / clv.n_with_line_clv * 100, 2)

    # --- warnings -----------------------------------------------------
    if 0 < rec.decided_n < MIN_SAMPLE_FOR_RATE:
        summary.warnings.append(
            f"Only {rec.decided_n} decided props — a strike rate on this sample is "
            f"noise, not signal. Treat as descriptive, not predictive."
        )
    if rec.pending:
        summary.warnings.append(f"{rec.pending} prop(s) still PENDING and excluded from all rates.")
    if rec.voids:
        summary.warnings.append(
            f"{rec.voids} prop(s) VOID (DNP/scratch) — excluded from W-L and ROI, "
            f"which is correct: a scratch is not a losing bet."
        )

    return summary


def get_performance_by_market(
    start_date: date | None = None,
    end_date: date | None = None,
) -> list[dict[str, Any]]:
    """Per-market breakdown. Same denominator rules as the overall summary."""
    conditions = []
    if start_date:
        conditions.append(PropResult.game_date >= start_date)
    if end_date:
        conditions.append(PropResult.game_date <= end_date)

    with session_scope() as db:
        stmt = select(
            PropResult.market,
            func.count().filter(PropResult.outcome_status == "WIN").label("wins"),
            func.count().filter(PropResult.outcome_status == "LOSS").label("losses"),
            func.count().filter(PropResult.outcome_status == "PUSH").label("pushes"),
            func.sum(PropResult.stake_units).filter(PropResult.odds.isnot(None)).label("staked"),
            func.sum(PropResult.profit_units).filter(PropResult.odds.isnot(None)).label("profit"),
        ).group_by(PropResult.market)
        if conditions:
            stmt = stmt.where(and_(*conditions))

        rows = db.execute(stmt).all()

    out = []
    for r in rows:
        decided = (r.wins or 0) + (r.losses or 0)
        staked = _f(r.staked)
        profit = _f(r.profit)
        out.append({
            "market": r.market,
            "record": f"{r.wins or 0}-{r.losses or 0}-{r.pushes or 0}",
            "decided_n": decided,
            "strike_rate_pct": round((r.wins or 0) / decided * 100, 2) if decided else None,
            "roi_pct": round(profit / staked * 100, 2) if staked else None,
            "low_sample": decided < MIN_SAMPLE_FOR_RATE,
        })
    return sorted(out, key=lambda x: x["decided_n"], reverse=True)
