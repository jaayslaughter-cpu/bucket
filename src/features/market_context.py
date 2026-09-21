"""
src/features/market_context.py — pregame market context from game lines.

The BigDataBall workbook carries the market's own forecast of every game:
an opening spread and total, a closing spread and total, and a moneyline.
Two of those are pregame and two are not, and the difference is the whole
point of this module.

OPENING LINES ARE PREGAME. They are posted before tip, so a projection made
that morning could have seen them. They are admissible as features.

CLOSING LINES ARE NOT. A closing number is only known at tip, after every
late scratch and every steam move. Joining it to a projection made hours
earlier is look-ahead of the purest kind: it hands the model the market's
final answer and lets it call the result a prediction. ``CLOSING_ONLY_COLS``
is refused by ``attach_market_context`` outright, and closing values are
reachable only through ``closing_line_value``, which is settlement.

WHAT THE MARKET KNOWS THAT THE BOX SCORE DOES NOT. The implied team total —
``(total - spread) / 2`` — is the market's estimate of how many points a
team will score, priced by people with injury news, rotation news and rest
news that no rolling average contains. For a points prop that is the single
most informative pregame number available, which is also why it must not be
contaminated with its own closing value.

Nothing here is a betting signal. These are context features for a
projection, and the module refuses to derive a wager from them.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SOURCE_NAME = "market_context"

# Known only at tip. Never a feature; settlement and CLV only.
CLOSING_ONLY_COLS: frozenset[str] = frozenset({
    "closing_spread", "closing_total", "closing_odds_raw", "halftime_raw",
    "line_movement_1", "line_movement_2", "line_movement_3",
    "CLOSING_SPREAD", "CLOSING_TOTAL",
})

# The pregame columns this layer reads from a market_lines frame.
PREGAME_SOURCE_COLS: tuple[str, ...] = (
    "nba_game_id", "game_date", "team_abbr", "opening_spread", "opening_total",
)

# What it writes. Namespaced so nothing collides with a box-score column.
MARKET_FEATURE_COLS: tuple[str, ...] = (
    "MKT_OPENING_SPREAD",
    "MKT_OPENING_TOTAL",
    "MKT_IMPLIED_TEAM_TOTAL",
    "MKT_IMPLIED_OPP_TOTAL",
    "MKT_IS_FAVORITE",
)


class MarketContextError(ValueError):
    """Raised when market context cannot be attached from what was supplied."""


class ClosingLineLeakageError(MarketContextError):
    """Raised when a closing line is offered as a pregame feature."""


def assert_no_closing_lines(columns: Sequence[str]) -> None:
    """Refuse any column that is only known at tip."""
    leaking = sorted({str(c) for c in columns} & CLOSING_ONLY_COLS)
    if leaking:
        raise ClosingLineLeakageError(
            f"Columns {leaking} are CLOSING lines, known only at tip. Using one as "
            "a feature for a projection made hours earlier hands the model the "
            "market's final answer and lets it call the result a prediction. "
            "Opening lines are the pregame choice; closing lines belong to "
            "closing_line_value()."
        )


def implied_team_total(total: Any, spread: Any) -> float | None:
    """
    The market's estimate of a team's points: ``(total - spread) / 2``.

    ``spread`` is from that team's perspective, so a favourite carries a
    negative number and therefore the larger implied total. OKC at -6.5 in a
    225.5 game implies 116.0, and the opponent 109.5, which sums back to the
    posted total.
    """
    try:
        t, s = float(total), float(spread)
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(t) and np.isfinite(s)):
        return None
    return (t - s) / 2.0


def build_market_context(market_lines: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce a market_lines frame to the pregame context of one team-game.

    Closing columns are dropped here rather than carried and ignored, so
    nothing downstream can reach them by accident.
    """
    missing = [c for c in PREGAME_SOURCE_COLS if c not in market_lines.columns]
    if missing:
        raise MarketContextError(
            f"market_lines is missing {missing}. Expected the frame from "
            "src.ingestion.bigdataball.load_bigdataball_workbook."
        )

    out = market_lines.loc[:, list(PREGAME_SOURCE_COLS)].copy()
    out = out.rename(columns={
        "nba_game_id": "GAME_ID",
        "game_date": "GAME_DATE",
        "team_abbr": "TEAM_ABBREVIATION",
    })
    out["GAME_ID"] = out["GAME_ID"].astype(str).str.strip()
    out["GAME_DATE"] = pd.to_datetime(out["GAME_DATE"], errors="coerce")
    out["TEAM_ABBREVIATION"] = out["TEAM_ABBREVIATION"].astype(str).str.strip().str.upper()

    spread = pd.to_numeric(out.pop("opening_spread"), errors="coerce")
    total = pd.to_numeric(out.pop("opening_total"), errors="coerce")

    out["MKT_OPENING_SPREAD"] = spread
    out["MKT_OPENING_TOTAL"] = total
    out["MKT_IMPLIED_TEAM_TOTAL"] = (total - spread) / 2.0
    out["MKT_IMPLIED_OPP_TOTAL"] = (total + spread) / 2.0
    # Missing spread means unknown, not "pick'em". A 0.0 here would assert
    # that a game with no posted line was an even matchup.
    out["MKT_IS_FAVORITE"] = np.where(spread.notna(), (spread < 0).astype(float), np.nan)

    duplicated = out.duplicated(subset=["GAME_ID", "TEAM_ABBREVIATION"], keep=False)
    if duplicated.any():
        logger.warning(
            "market_context: %d duplicated (GAME_ID, TEAM) rows; keeping the first "
            "of each so a join cannot fan out.", int(duplicated.sum()),
        )
        out = out.drop_duplicates(subset=["GAME_ID", "TEAM_ABBREVIATION"], keep="first")

    return out.reset_index(drop=True)


def attach_market_context(
    panel: pd.DataFrame,
    market_lines: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Join pregame market context onto a player panel.

    Rows with no matching game keep NaN rather than a league-average line —
    a game the market never priced is unknown, and filling it would assert a
    forecast nobody made.
    """
    if market_lines is None or market_lines.empty:
        logger.info(
            "market_context: no market lines supplied, so no market features. The "
            "run is narrower rather than silently filled."
        )
        return panel

    assert_no_closing_lines(panel.columns)

    keys = ("GAME_ID", "TEAM_ABBREVIATION")
    missing = [k for k in keys if k not in panel.columns]
    if missing:
        logger.warning(
            "market_context: panel has no %s — cannot join market lines; skipping.",
            missing,
        )
        return panel

    context = build_market_context(market_lines)
    out = panel.copy()
    out["GAME_ID"] = out["GAME_ID"].astype(str).str.strip()
    out["TEAM_ABBREVIATION"] = out["TEAM_ABBREVIATION"].astype(str).str.strip().str.upper()

    merged = out.merge(
        context.drop(columns=["GAME_DATE"]),
        on=list(keys),
        how="left",
        validate="many_to_one",
    )
    matched = int(merged["MKT_OPENING_TOTAL"].notna().sum())
    logger.info(
        "market_context: %d of %d panel rows matched a posted opening line (%.1f%%)",
        matched, len(merged), 100.0 * matched / max(len(merged), 1),
    )
    if matched == 0:
        logger.warning(
            "market_context: nothing matched. Check that the panel's GAME_ID and "
            "TEAM_ABBREVIATION use the same spellings as the workbook."
        )
    assert_no_closing_lines(merged.columns)
    return merged


# ---------------------------------------------------------------------------
# settlement: the closing line's one legitimate use
# ---------------------------------------------------------------------------


def closing_line_value(
    market_lines: pd.DataFrame,
    *,
    taken_col: str = "opening_spread",
) -> pd.DataFrame:
    """
    Line CLV per team-game: how the spread moved after the opening number.

    This is the closing line's legitimate use. Positive means the market
    moved toward the side you would have taken at the open — for a team laid
    at -3.5 that closes -5.5, the number moved two points your way.

    CLV is reported on its own and is never added to a return. It answers a
    different question from EV: whether the PRICE was right, not whether the
    model was.
    """
    needed = {taken_col, "closing_spread", "nba_game_id", "team_abbr"}
    missing = sorted(needed - set(market_lines.columns))
    if missing:
        raise MarketContextError(f"market_lines is missing {missing} for CLV")

    out = market_lines.loc[:, ["nba_game_id", "game_date", "team_abbr"]].copy()
    taken = pd.to_numeric(market_lines[taken_col], errors="coerce")
    closing = pd.to_numeric(market_lines["closing_spread"], errors="coerce")

    out["taken_spread"] = taken
    out["closing_spread"] = closing
    # A spread moving DOWN (-3.5 -> -5.5) means the market grew more
    # confident in that team, which is value to whoever took the opener.
    out["clv_line_points"] = taken - closing
    out["status"] = np.where(
        taken.notna() & closing.notna(), "OK", "DATA_NOT_AVAILABLE",
    )
    graded = out[out["status"] == "OK"]
    logger.info(
        "closing_line_value: %d of %d team-games priced both ends; mean move "
        "%.3f points",
        len(graded), len(out),
        float(graded["clv_line_points"].mean()) if len(graded) else float("nan"),
    )
    return out
