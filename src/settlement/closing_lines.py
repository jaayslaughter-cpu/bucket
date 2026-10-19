"""
src/settlement/closing_lines.py — what the market said at the close.

WHY CLV IS THE FAST VALIDATOR. `settlement/evaluator.compute_clv` has always
been able to measure closing line value, `prop_results` has always had the
``closing_line`` / ``closing_odds`` columns, and the Discord results card
already displays the averages. Nothing ever populated them, so
`compute_clv` returned ``{None, None}`` on every row and the whole path was
complete and inert.

That matters more than it looks. CLV needs NO history: it compares the line
YOU took against THAT SAME GAME's close, so it produces a reading on the
first game of a season with no warm-up, no prior-season data and no
calibration report. Win/loss needs roughly three thousand settled props
before a two-point edge separates from noise; CLV is a continuous measure
and starts paying out information immediately. On a season that begins in two
days, that is the difference between measuring from game one and measuring
from February.

THE LADDER PROBLEM, AND THE RULE THIS PICKS. Books post alternate points --
measured on one real event, 3 to 10 points per (book, market, player), with
Chet Holmgren offered at 10.5 through 17.5 -- so "the closing line" is a
choice, not a lookup. Liquidity would be the natural tiebreak and PropLine
reports it on 258 of 1022 outcomes, so it cannot be the rule. This uses the
MOST BALANCED TWO-WAY PRICE: the point whose over and under implied
probabilities are closest together is the one the book is pricing as the
coin-flip, which is what a main line is. Validated against a real event --
Fox's 11.5-18.5 ladder resolves to 14.5 at -115/-115, Holmgren's to 13.5 at
-120/-110, each mid-ladder and near even, which is what a main line looks
like.

AND THE ERROR THIS REFUSES TO MAKE. `compute_clv` keeps the line move and the
price move separate, because "reporting one as the other is the most common
CLV error". So ``closing_odds`` here is the price of THE SAME POINT that was
taken, never the main line's price: mixing them would let a line move leak
into ``clv_prob_points`` and be read as a price edge. If the point we took is
no longer offered at the close, ``closing_odds`` is None and the probability
measure abstains while the line measure still reports.

RESEARCH_ONLY. This measures market agreement. It is not profit, and
`compute_clv` says so where it is computed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from src.quant.contracts import american_to_implied_probability
from src.settlement.names import normalise_player_name

logger = logging.getLogger(__name__)

#: (book, market, normalised player) — the unit a closing line belongs to.
#: The BOOK is part of the key deliberately: comparing a line taken at
#: DraftKings against Bovada's close measures two shops disagreeing, not the
#: market moving, and `features/market_context` makes the same point about
#: spreads.
ClosingKey = tuple[str, str, str]


@dataclass(frozen=True)
class ClosingQuote:
    """The close for one (book, market, player), ladder and all."""

    main_line: float | None
    main_over_odds: int | None
    main_under_odds: int | None
    #: point -> (over odds, under odds), so the taken point's own price can be
    #: found without re-deriving which point was main.
    by_point: dict[float, tuple[int | None, int | None]] = field(default_factory=dict)
    n_points: int = 0


def price_imbalance(over_odds: Any, under_odds: Any) -> float | None:
    """
    How far from a coin flip this two-way price is, in probability points.

    None when either side is missing: a one-way quote cannot be judged
    balanced, and treating it as perfectly balanced would make every
    half-quoted ladder rung look like the main line.
    """
    if over_odds is None or under_odds is None:
        return None
    try:
        over = american_to_implied_probability(int(over_odds))
        under = american_to_implied_probability(int(under_odds))
    except (TypeError, ValueError):
        return None
    return abs(float(over) - float(under))


def build_closing_lookup(rows: Any) -> dict[ClosingKey, ClosingQuote]:
    """
    Closing rows (as `normalize_event_props` returns them) -> quote per key.

    REUSES THE LIVE NORMALISER, measured rather than assumed: the closing
    payload has the same bookmakers/markets/outcomes shape, and
    `normalize_event_props` reads ``price`` and ``point``, which in that
    payload ARE the closing values. 664 usable rows came back from one real
    event. A second parser here would be the defect this tree keeps removing,
    and it would also silently drop the ``line_type: "milestone"`` outcomes
    the live normaliser already refuses (``point`` is null and the side is
    "To Score 10+ Points", not over/under).
    """
    grouped: dict[ClosingKey, list[Any]] = {}
    for row in rows or []:
        player = normalise_player_name(row.player_name or "")
        if not player or row.line is None:
            continue
        key = (str(row.source or ""), str(row.market or ""), player)
        grouped.setdefault(key, []).append(row)

    out: dict[ClosingKey, ClosingQuote] = {}
    for key, candidates in grouped.items():
        by_point: dict[float, tuple[int | None, int | None]] = {}
        for row in candidates:
            by_point[float(row.line)] = (
                row.over_odds_american, row.under_odds_american,
            )

        two_way = [
            (imbalance, row)
            for row, imbalance in (
                (r, price_imbalance(r.over_odds_american, r.under_odds_american))
                for r in candidates
            )
            if imbalance is not None
        ]
        if two_way:
            _, main = min(two_way, key=lambda pair: pair[0])
            out[key] = ClosingQuote(
                main_line=float(main.line),
                main_over_odds=main.over_odds_american,
                main_under_odds=main.under_odds_american,
                by_point=by_point,
                n_points=len(by_point),
            )
        else:
            # Every rung was one-way. There is no main line to name, but the
            # taken point's own price may still be here, so the probability
            # measure can work where the line measure cannot.
            out[key] = ClosingQuote(
                main_line=None, main_over_odds=None, main_under_odds=None,
                by_point=by_point, n_points=len(by_point),
            )
    return out


def capture_for_dates(
    game_dates: Any,
    *,
    client: Any = None,
    markets: Any = None,
    session: Any = None,
) -> dict[ClosingKey, ClosingQuote]:
    """
    Build the closing lookup for every PropLine event on these slate dates.

    ONE CALL PER EVENT, ONCE PER SETTLEMENT RUN. The endpoint is per-event, so
    a twelve-game slate is twelve calls; the runner builds this once and every
    prop reads from it, the same discipline the ESPN box-score cache follows
    after re-fetching a slate per game cost ~156 requests.

    AN EVENT THAT FAILS DOES NOT TAKE THE SLATE DOWN. A closing line is a
    market-quality measure, not a settlement input: a prop with no closing
    quote still grades to WIN/LOSS/PUSH, it just carries no CLV. So a failed
    event is logged and skipped rather than raised, because the alternative
    trades a complete grading run for a complete CLV run.
    """
    from src.db.repository import propline_event_ids_for_dates
    from src.ingestion.propline import (
        DEFAULT_MARKET_MAP,
        PropLineClient,
        normalize_event_props,
    )

    api = client or PropLineClient()
    market_keys = list(markets or DEFAULT_MARKET_MAP)
    event_ids = propline_event_ids_for_dates(list(game_dates))
    if not event_ids:
        logger.info(
            "No PropLine event ids in prop_line_snapshots for %s, so there is "
            "no closing line to capture. CLV stays null and grading is "
            "unaffected.", sorted(game_dates),
        )
        return {}

    rows: list[Any] = []
    failed = 0
    for event_id in event_ids:
        try:
            rows.extend(normalize_event_props(
                api.fetch_closing_odds(event_id, market_keys)
            ))
        except Exception as exc:  # noqa: BLE001 — see the docstring
            failed += 1
            logger.warning(
                "Closing odds unavailable for PropLine event %s (%s); its props "
                "grade without CLV.", event_id, exc,
            )
    lookup = build_closing_lookup(rows)
    logger.info(
        "Closing lines: %d event(s), %d failed, %d quote(s) across %d row(s)",
        len(event_ids), failed, len(lookup), len(rows),
    )
    return lookup


def closing_for_prop(
    lookup: dict[ClosingKey, ClosingQuote],
    *,
    source: Any,
    market: Any,
    player_name: Any,
    taken_line: Any,
    taken_side: Any,
) -> tuple[float | None, int | None]:
    """
    (closing_line, closing_odds) for one prop, or (None, None) with no quote.

    ``closing_line`` is the MAIN line, which is what a line move is measured
    against. ``closing_odds`` is the price of THE SAME POINT that was taken,
    on the side that was taken -- see this module's header on why those must
    not be the same number's price.
    """
    key = (
        str(source or ""), str(market or ""),
        normalise_player_name(player_name or ""),
    )
    quote = lookup.get(key)
    if quote is None:
        return None, None

    closing_odds: int | None = None
    try:
        point = float(taken_line)
    except (TypeError, ValueError):
        point = None
    if point is not None and point in quote.by_point:
        over_odds, under_odds = quote.by_point[point]
        side = str(taken_side or "").strip().upper()
        if side == "OVER":
            closing_odds = over_odds
        elif side == "UNDER":
            closing_odds = under_odds
    return quote.main_line, closing_odds
