"""
src/ingestion/propline_history.py — normalisers for PropLine's historical feed.

The live client answers "what is priced right now". These endpoints answer
"what was priced, and what happened" — which is the half that makes
calibration and CLV possible, and the half this repository has never had.

WHAT IS IMPLEMENTED AGAINST A DOCUMENTED SHAPE, AND WHAT IS NOT.

``/odds/closing`` and ``POST /clv/grade`` have full response examples in the
provider's documentation, so they are parsed field by field. ``/results`` and
``/players/{name}/history`` are listed in the endpoint index with no response
example, so nothing about their shape is assumed: ``describe_resolution_payload``
reports the fields it actually found and refuses to nominate one as the graded
result. Guessing which key holds a settlement is how a backtest ends up
grading against the wrong column.

TWO CLV NUMBERS, AND THEY ARE NOT INTERCHANGEABLE. The provider returns both
and says so plainly: ``clv_pct`` compares price to price and is vig-blind, so
it flatters a bet taken on the juicy side of a wide market. ``ev_vs_close_pct``
scores the price against the DE-VIGGED close. That is the same rule this
project already applies to edge, so it is the one reported first here, and
neither is ever summed into a return.

WHAT IS ACTUALLY REACHABLE. Two limits stack, and together they decide
whether any of this returns NBA rows at all: the archive begins April 2026,
and the event-age cap is set by plan (hobby 30d … enterprise unlimited).
``plan_history_window`` computes the intersection so a pull that cannot
possibly contain data is refused before it spends quota.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from src.ingestion.propline import (
    ARCHIVE_START,
    TIER_EVENT_AGE_DAYS,
    PropLineError,
    _american,
    _parse_iso,
    _strip_player_namespace,
)
from src.quant.ev_engine import compute_clv

logger = logging.getLogger(__name__)

SOURCE_NAME = "propline_history"

# The documented columns of one closing-odds outcome.
CLOSING_COLS: tuple[str, ...] = (
    "event_id", "sport_key", "commence_time", "bookmaker", "bookmaker_title",
    "market", "player_name", "player_id", "side", "line",
    "closing_price", "closing_at", "closing_age_seconds", "is_stale",
    "opening_price", "opening_point", "opening_at", "opening_age_seconds",
)


class PropLineHistoryError(PropLineError):
    """Raised when a historical payload cannot be normalised as received."""


# ---------------------------------------------------------------------------
# what a plan can actually reach
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HistoryWindow:
    """The dates a given plan can actually return, and why."""

    tier: str
    earliest: date | None
    latest: date
    usable: bool
    reason: str

    @property
    def days(self) -> int:
        if self.earliest is None:
            return 0
        return max((self.latest - self.earliest).days, 0)


def plan_history_window(
    tier: str,
    *,
    as_of: date | None = None,
    season_start: date | None = None,
) -> HistoryWindow:
    """
    Intersect the plan's event-age cap with the provider's archive start.

    Both limits bind, and on the lower plans the cap is the one that bites:
    a 30-day window measured from an off-season day contains no basketball
    whatever the archive holds. Passing ``season_start`` makes that explicit
    rather than leaving it to be discovered by an empty pull.
    """
    today = as_of or datetime.now(timezone.utc).date()
    key = str(tier).strip().lower().replace("-", "_").replace(" ", "_")
    if key not in TIER_EVENT_AGE_DAYS:
        raise PropLineHistoryError(
            f"Unknown tier {tier!r}. Known: {sorted(TIER_EVENT_AGE_DAYS)}"
        )

    cap_days = TIER_EVENT_AGE_DAYS[key]
    cap_earliest = None if cap_days is None else today - timedelta(days=cap_days)
    earliest = ARCHIVE_START if cap_earliest is None else max(cap_earliest, ARCHIVE_START)

    if earliest > today:
        return HistoryWindow(
            key, None, today, False,
            f"The archive starts {ARCHIVE_START} — later than today.",
        )

    binding = (
        "the archive start" if cap_earliest is None or ARCHIVE_START >= cap_earliest
        else f"the {key} {cap_days}-day event-age cap"
    )
    reason = f"Reaches back to {earliest} ({binding})."

    if season_start is not None and earliest > season_start:
        return HistoryWindow(
            key, earliest, today, False,
            reason
            + f" That is after the {season_start} season start, so the window "
            "lands in the off-season and would return no games. A longer "
            "event-age cap is the only thing that changes this.",
        )
    return HistoryWindow(key, earliest, today, True, reason)


# ---------------------------------------------------------------------------
# /odds/closing  — documented shape
# ---------------------------------------------------------------------------


def normalize_closing_odds(payload: Mapping[str, Any]) -> pd.DataFrame:
    """
    Flatten a closing-odds payload into one row per (book, market, player, side).

    ``opening_*`` is first-observed-by-the-provider, not the book's true open,
    for anything posted before they began polling. ``opening_age_seconds`` is
    carried through unchanged so that judgement stays with the caller: a value
    in minutes rather than hours means the "open" is nothing of the kind.
    """
    if not isinstance(payload, Mapping) or not payload:
        return pd.DataFrame(columns=list(CLOSING_COLS))

    event_id = str(payload.get("id") or "") or None
    sport_key = payload.get("sport_key")
    commence = _parse_iso(payload.get("commence_time"))

    rows: list[dict[str, Any]] = []
    for book in payload.get("bookmakers") or []:
        book_key = book.get("key")
        book_title = book.get("title")
        for market in book.get("markets") or []:
            market_key = market.get("key")
            for outcome in market.get("outcomes") or []:
                rows.append({
                    "event_id": event_id,
                    "sport_key": sport_key,
                    "commence_time": commence,
                    "bookmaker": book_key,
                    "bookmaker_title": book_title,
                    "market": market_key,
                    # `description` carries the player on a prop; `name` is the side.
                    "player_name": outcome.get("description"),
                    "player_id": _strip_player_namespace(outcome.get("player_id")),
                    "side": str(outcome.get("name") or "").strip().lower() or None,
                    "line": outcome.get("point"),
                    "closing_price": _american(outcome.get("price")),
                    "closing_at": _parse_iso(outcome.get("closing_at")),
                    "closing_age_seconds": outcome.get("closing_age_seconds"),
                    "is_stale": outcome.get("is_stale"),
                    "opening_price": _american(outcome.get("opening_price")),
                    "opening_point": outcome.get("opening_point"),
                    "opening_at": _parse_iso(outcome.get("opening_at")),
                    "opening_age_seconds": outcome.get("opening_age_seconds"),
                })

    frame = pd.DataFrame(rows, columns=list(CLOSING_COLS))
    logger.info(
        "closing odds: %d outcomes across %d books for event %s",
        len(frame), frame["bookmaker"].nunique() if len(frame) else 0, event_id,
    )
    return frame


def clv_from_closing(
    closing: pd.DataFrame,
    *,
    min_opening_age_seconds: int = 3600,
) -> pd.DataFrame:
    """
    CLV per outcome, measured on DE-VIGGED probabilities where both sides exist.

    The two sides of one (book, market, player, line) are paired so the vig
    can be removed from both the taken price and the close. Where only one
    side is present the comparison falls back to raw implied probability and
    says so, because that number carries each book's hold and is noisier.

    ``min_opening_age_seconds`` drops outcomes whose "open" was first seen
    too close to tip to be an open at all. Measuring CLV from a price
    observed ten minutes before tip is measuring nothing.
    """
    if closing.empty:
        return pd.DataFrame()

    work = closing.copy()
    pairs = work.groupby(
        ["event_id", "bookmaker", "market", "player_name", "line"], dropna=False
    )

    rows: list[dict[str, Any]] = []
    for _, block in pairs:
        by_side = {
            str(r["side"]): r for _, r in block.iterrows() if r.get("side")
        }
        for side, row in by_side.items():
            other = by_side.get("under" if side == "over" else "over")
            age = row.get("opening_age_seconds")
            if age is not None and pd.notna(age) and float(age) < min_opening_age_seconds:
                rows.append({
                    **{k: row.get(k) for k in
                       ("event_id", "bookmaker", "market", "player_name", "side", "line")},
                    "status": "DATA_NOT_AVAILABLE",
                    "reason": (
                        f"opening_age_seconds={float(age):.0f} — first observed "
                        f"{float(age) / 60:.0f} minutes before tip, which is not an open"
                    ),
                    "clv": None, "clv_method": None,
                })
                continue

            # `other` is a pandas row; truth-testing one raises, so the
            # presence check is explicit rather than an `or {}` shortcut.
            other_open = other.get("opening_price") if other is not None else None
            other_close = other.get("closing_price") if other is not None else None
            result = compute_clv(
                taken_american=row.get("opening_price"),
                closing_american=row.get("closing_price"),
                taken_other_american=other_open,
                closing_other_american=other_close,
            )
            # compute_clv de-vigs only when BOTH counterpart prices are
            # present. That condition is knowable here, so the method is
            # recorded rather than left to be read out of a prose note.
            devigged = other_open is not None and other_close is not None
            rows.append({
                **{k: row.get(k) for k in
                   ("event_id", "bookmaker", "market", "player_name", "side", "line")},
                "opening_price": row.get("opening_price"),
                "closing_price": row.get("closing_price"),
                "status": result.status,
                "reason": result.reason,
                "clv": result.clv,
                # Raw-implied CLV carries each book's hold and is the noisier
                # of the two; a consumer that cannot tell them apart will
                # average them together.
                "clv_method": "devigged" if devigged else "raw_implied",
                "taken_fair_prob": result.taken_fair_prob,
                "closing_fair_prob": result.closing_fair_prob,
                "beat_close": result.beat_close,
            })

    frame = pd.DataFrame(rows)
    graded = frame[frame["status"] == "OK"] if "status" in frame else frame
    logger.info(
        "CLV from closing: %d of %d outcomes measurable; mean %.5f",
        len(graded), len(frame),
        float(graded["clv"].mean()) if len(graded) and graded["clv"].notna().any() else float("nan"),
    )
    return frame


# ---------------------------------------------------------------------------
# POST /clv/grade — documented shape
# ---------------------------------------------------------------------------


@dataclass
class GradedBets:
    """The provider's grade of bets already placed."""

    summary: dict[str, Any] = field(default_factory=dict)
    bets: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def honest_clv_column(self) -> str:
        """The vig-aware one. ``clv_pct`` flatters a wide market."""
        return "ev_vs_close_pct"


def normalize_clv_grade(payload: Mapping[str, Any]) -> GradedBets:
    """
    Parse the grade response, keeping both CLV numbers and their meanings.

    ``ev_vs_close_pct`` is ordered first because it is the de-vigged one.
    ``clv_pct`` is kept beside it rather than dropped — it is what everyone
    quotes, and hiding it would just mean someone recomputes it worse.
    """
    if not isinstance(payload, Mapping):
        raise PropLineHistoryError(f"Expected a grade object, got {type(payload).__name__}")

    bets = pd.DataFrame(payload.get("bets") or [])
    preferred = [
        "ref", "matched", "unmatched_reason", "resolution", "actual_value",
        "ev_vs_close_pct", "clv_pct", "beat_close",
        "closing_price", "closing_point", "closing_at",
        "closing_fair_prob", "fair_source", "closing_is_stale", "closing_is_final",
    ]
    ordered = [c for c in preferred if c in bets.columns]
    bets = bets[ordered + [c for c in bets.columns if c not in ordered]] if len(bets) else bets

    summary = dict(payload.get("summary") or {})
    if summary:
        logger.info(
            "CLV grade: %s bets, %s matched, mean ev_vs_close %.4f%% (clv_pct %.4f%%)",
            summary.get("bets"), summary.get("matched"),
            float(summary.get("avg_ev_vs_close_pct") or 0.0),
            float(summary.get("avg_clv_pct") or 0.0),
        )
    return GradedBets(summary=summary, bets=bets)


def bets_to_grade_payload(
    records: Iterable[Mapping[str, Any]],
    *,
    sport_key: str = "basketball_nba",
) -> list[dict[str, Any]]:
    """
    Build the ``/clv/grade`` request from logged bets.

    Reads the field names this repository already stores, so a row written by
    ``log-manual-bet`` grades without being reshaped by hand.
    """
    payload: list[dict[str, Any]] = []
    for i, record in enumerate(records):
        get = record.get
        ref = str(get("bet_id") or get("ref") or f"b{i + 1}")
        missing = [
            k for k in ("game_id", "prop_stat", "bet_side", "taken_odds_american")
            if get(k) in (None, "")
        ]
        if missing:
            raise PropLineHistoryError(f"Bet {ref} is missing {missing}")
        payload.append({
            "ref": ref,
            "sport_key": sport_key,
            "event_id": get("game_id"),
            "market": get("prop_stat"),
            "bookmaker": get("bookmaker"),
            "selection": get("player_name"),
            "side": str(get("bet_side")).capitalize(),
            "point": get("line"),
            "price": int(get("taken_odds_american")),
            "stake": float(get("unit_stake") or 1.0),
        })
    return payload


# ---------------------------------------------------------------------------
# /results and /players/{name}/history — shape NOT documented
# ---------------------------------------------------------------------------

# Field names a resolution payload plausibly uses. Used to REPORT candidates,
# never to pick one: two of these could both be present and mean different
# things, and grading against the wrong one is silent.
_RESOLUTION_HINTS: tuple[str, ...] = (
    "resolution", "result", "outcome", "status", "graded", "settled",
    "won", "hit", "actual_value", "actual", "final_value", "stat_value",
)


@dataclass
class ResolutionReport:
    """What a discovery pass found in an undocumented payload."""

    rows: int = 0
    columns: list[str] = field(default_factory=list)
    resolution_candidates: list[str] = field(default_factory=list)
    frame: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def usable(self) -> bool:
        return self.rows > 0 and len(self.resolution_candidates) == 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "columns": self.columns,
            "resolution_candidates": self.resolution_candidates,
            "usable": self.usable,
            "note": (
                "Exactly one resolution candidate means the graded column is "
                "unambiguous. Zero or several means a human picks it — this "
                "module will not guess which column settled the bet."
            ),
        }


def describe_resolution_payload(payload: Any) -> ResolutionReport:
    """
    Inspect a ``/results`` or ``/players/{name}/history`` body without assuming.

    The provider's documentation lists both endpoints but shows no response
    example, so the shape is discovered and reported. Run this once against a
    real response, then map the columns deliberately.
    """
    if isinstance(payload, Mapping):
        for key in ("results", "history", "props", "data", "items"):
            if isinstance(payload.get(key), list):
                records = payload[key]
                break
        else:
            records = [payload]
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        records = list(payload)
    else:
        raise PropLineHistoryError(
            f"Cannot inspect a {type(payload).__name__} payload"
        )

    frame = pd.json_normalize(records) if records else pd.DataFrame()
    columns = list(frame.columns)
    lowered = {c: c.lower() for c in columns}
    candidates = sorted(
        c for c, low in lowered.items()
        if any(hint == low or low.endswith(f"_{hint}") for hint in _RESOLUTION_HINTS)
    )

    report = ResolutionReport(
        rows=len(frame), columns=columns,
        resolution_candidates=candidates, frame=frame,
    )
    if not candidates:
        logger.warning(
            "No resolution-looking column among %s. The payload may not carry a "
            "graded result, or it uses a name this module has not seen.", columns[:20],
        )
    elif len(candidates) > 1:
        logger.warning(
            "Several resolution candidates %s — pick one deliberately. Grading "
            "against the wrong column fails silently.", candidates,
        )
    return report
