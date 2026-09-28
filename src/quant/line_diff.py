"""Pick'em vs OddsPapi line-diff helper (Wave 4).

RESEARCH_ONLY. Line-diff math runs only when the sportsbook side is a
``MarketContext`` / ``PropMarketSnapshot`` with ``status=VALID`` and verified
two-way American odds (OddsPapi path). Pick'em multipliers never unlock EV.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from src.quant.contracts import MarketContext, PropMarketSnapshot, market_ev_gate
from src.quant.odds_math import multiplicative_devig


class LineDiffResult(BaseModel):
    status: str
    pickem_line: float | None = None
    book_line: float | None = None
    line_diff: float | None = None  # pickem - book (negative ⇒ pickem easier for OVER)
    side: str = "over"
    book_fair_prob_over: float | None = None
    adjusted_fair_prob_over: float | None = None
    reason: str | None = None
    notes: str = Field(
        default=(
            "RESEARCH_ONLY line-diff. Pick'em is not VALID two-way American odds; "
            "no stake from this helper."
        )
    )


def _book_line_and_odds(
    market: MarketContext | PropMarketSnapshot,
) -> tuple[float | None, int | None, int | None]:
    if isinstance(market, PropMarketSnapshot):
        # Prefer ``line``; some older snapshots also populate ``total``.
        line = market.line if market.line is not None else market.total
        return line, market.over_odds_american, market.under_odds_american
    return market.line, market.over_odds_american, market.under_odds_american


def _line_adjust_fair_prob(
    fair_over: float,
    *,
    line_diff: float,
    side: str,
    pts_per_prob: float = 0.03,
) -> float:
    """
    Soft log-ish shift: ~3% fair-prob per point of line difference.

    Sign: if pickem line is lower than book (line_diff < 0), OVER is easier on
    pick'em → raise fair P(over) for the pickem board comparison.
    """
    # line_diff = pickem - book; negative means pickem OVER is easier
    delta = -float(line_diff) * float(pts_per_prob)
    if side.lower() in {"under", "u", "less"}:
        delta = -delta
    return float(np.clip(fair_over + delta, 0.01, 0.99))


def pickem_vs_book_line_diff(
    pickem_line: float | None,
    market: MarketContext | PropMarketSnapshot,
    *,
    side: str = "over",
    pts_per_prob: float = 0.03,
) -> LineDiffResult:
    """
    Compare an approved pick'em line to a VALID OddsPapi two-way book line.

    Returns DATA_NOT_AVAILABLE when the book side is not VALID or lines missing.
    Does not invent odds or mark pick'em as VALID for stake math.
    """
    ctx = market.to_market_context() if isinstance(market, PropMarketSnapshot) else market
    gate = market_ev_gate(ctx)
    if gate["status"] != "READY_FOR_EVALUATION":
        return LineDiffResult(
            status="DATA_NOT_AVAILABLE",
            pickem_line=float(pickem_line) if pickem_line is not None else None,
            reason=gate.get("reason") or "Book market not VALID two-way OddsPapi",
        )

    book_line, over_a, under_a = _book_line_and_odds(market)
    if pickem_line is None or not np.isfinite(pickem_line):
        return LineDiffResult(
            status="DATA_NOT_AVAILABLE",
            book_line=float(book_line) if book_line is not None else None,
            reason="Pick'em line missing",
        )
    if book_line is None or not np.isfinite(book_line):
        return LineDiffResult(
            status="DATA_NOT_AVAILABLE",
            pickem_line=float(pickem_line),
            reason="Book line missing on VALID quote",
        )
    if over_a is None or under_a is None:
        return LineDiffResult(
            status="DATA_NOT_AVAILABLE",
            pickem_line=float(pickem_line),
            book_line=float(book_line),
            reason="Two-way American odds required",
        )

    fair = multiplicative_devig(int(over_a), int(under_a))
    fair_over = float(fair.fair_prob_a)
    diff = float(pickem_line) - float(book_line)
    adj = _line_adjust_fair_prob(
        fair_over, line_diff=diff, side=side, pts_per_prob=pts_per_prob
    )
    return LineDiffResult(
        status="OK",
        pickem_line=round(float(pickem_line), 4),
        book_line=round(float(book_line), 4),
        line_diff=round(diff, 4),
        side=side.lower(),
        book_fair_prob_over=round(fair_over, 4),
        adjusted_fair_prob_over=round(adj, 4),
        reason=None,
    )
