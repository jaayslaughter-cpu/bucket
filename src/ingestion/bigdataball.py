"""
src/ingestion/bigdataball.py — licensed BigDataBall export loader.

Reads the BigDataBall "NBA Box Score Team-Stats" workbook (manual,
licensed export — NOT scraped) and produces two normalized frames:

1. team_game_stats  — box score + pace/efficiency per team-game
2. game_market_lines — opening/closing spread, total, moneyline,
   and the three line-movement columns

WHY THIS MATTERS: PropIQ's quant layer abstains from EV/CLV whenever
``MarketContext.status == DATA_NOT_AVAILABLE`` (see src/quant/contracts.py).
BigDataBall is a licensed, timestamped market source, so rows loaded here
are the first thing in the project that can legitimately flip that gate
to VALID. Rows with missing/unparseable market values keep
DATA_NOT_AVAILABLE rather than being filled with a guess.

Verified against the real 2025-26 workbook: 5 sheets, data on
``NBA-2025-26-TEAM``, 2 rows per game (one per team), 1,322 games /
2,644 rows covering 10/21/2025 onward.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_SHEET = "NBA-2025-26-TEAM"
SOURCE_NAME = "bigdataball"

# BigDataBall long/short names -> NBA.com abbreviations. Populated from
# the workbook's own TEAMS sheet at load time when available; this is the
# fallback for when the sheet is missing.
_FALLBACK_TEAM_MAP: dict[str, str] = {
    "Atlanta": "ATL", "Boston": "BOS", "Brooklyn": "BKN", "Charlotte": "CHA",
    "Chicago": "CHI", "Cleveland": "CLE", "Dallas": "DAL", "Denver": "DEN",
    "Detroit": "DET", "Golden State": "GSW", "Houston": "HOU", "Indiana": "IND",
    "LA Clippers": "LAC", "LA Lakers": "LAL", "Memphis": "MEM", "Miami": "MIA",
    "Milwaukee": "MIL", "Minnesota": "MIN", "New Orleans": "NOP", "New York": "NYK",
    "Oklahoma City": "OKC", "Orlando": "ORL", "Philadelphia": "PHI", "Phoenix": "PHX",
    "Portland": "POR", "Sacramento": "SAC", "San Antonio": "SAS", "Toronto": "TOR",
    "Utah": "UTA", "Washington": "WAS",
}


def _clean_header(value: Any) -> str:
    """BigDataBall headers contain embedded newlines (e.g. 'VENUE\\n(R/H/N)')."""
    return str(value).replace("\n", " ").replace("\r", " ").strip()


def load_team_map(xlsx_path: str | Path) -> dict[str, str]:
    """
    Build short-name -> NBA.com abbreviation map from the workbook's TEAMS sheet.

    NOTE: the TEAMS sheet has TWO abbreviation columns — 'BIGDATABALL
    INITIALS' (e.g. 'Hou', 'Gol', 'Bro') and 'NBA.com INITIALS' (e.g.
    'HOU', 'GSW', 'BKN'). We want NBA.com's, because that's what every
    other data source in PropIQ keys on. Matching loosely on 'INITIALS'
    picks the wrong column and silently produces 'Gol' instead of 'GSW'.
    """
    try:
        teams = pd.read_excel(xlsx_path, sheet_name="TEAMS")
        teams.columns = [_clean_header(c) for c in teams.columns]
        short_col = next((c for c in teams.columns if "SHORT" in c.upper()), None)
        # Prefer the NBA.com column explicitly; only fall back to a generic
        # INITIALS match if no NBA.com column exists.
        abbr_col = next((c for c in teams.columns if "NBA.COM" in c.upper()), None)
        if abbr_col is None:
            abbr_col = next(
                (c for c in teams.columns if "INITIALS" in c.upper() and "BIGDATABALL" not in c.upper()),
                None,
            )
        if short_col and abbr_col:
            mapping = dict(zip(teams[short_col].astype(str).str.strip(), teams[abbr_col].astype(str).str.strip()))
            if mapping:
                return mapping
    except Exception as exc:  # noqa: BLE001 — fall back rather than fail the load
        logger.warning("Could not read TEAMS sheet (%s); using fallback team map.", exc)
    return dict(_FALLBACK_TEAM_MAP)


def _parse_bdb_date(value: Any) -> date | None:
    """BigDataBall stores dates as text MM/DD/YYYY."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.date()
    try:
        return datetime.strptime(str(value).strip(), "%m/%d/%Y").date()
    except ValueError:
        try:
            return pd.to_datetime(value).date()
        except Exception:  # noqa: BLE001
            return None


def _to_float(value: Any) -> float | None:
    if value is None or value == "" or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    f = _to_float(value)
    return int(f) if f is not None else None


def market_row_status(closing_spread: Any, closing_total: Any) -> str:
    """VALID only when the row carries a usable closing number.

    Uses ``pd.notna``, not ``is not None``: a missing numeric arrives as
    NaN, which *is not* None, so the identity form marked every empty row
    VALID and fed blank market data downstream as though it were real.
    Missing market data stays DATA_NOT_AVAILABLE — never filled with a
    league-average or invented line.
    """
    return "VALID" if (pd.notna(closing_spread) or pd.notna(closing_total)) else "DATA_NOT_AVAILABLE"


def load_bigdataball_workbook(
    xlsx_path: str | Path,
    sheet_name: str = DEFAULT_SHEET,
    season: str = "2025-26",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the workbook and return ``(team_stats_df, market_lines_df)``.

    Both frames use the canonical column names expected by
    ``src/db/models.py`` so they can be inserted without further renaming.
    """
    xlsx_path = Path(xlsx_path)
    if not xlsx_path.exists():
        raise FileNotFoundError(
            f"BigDataBall workbook not found at {xlsx_path}. Place the licensed "
            f"export under data/external/bigdataball/ — this loader never "
            f"downloads or scrapes it."
        )

    team_map = load_team_map(xlsx_path)
    # GAME-ID must stay a STRING: NBA game ids are zero-padded 10-digit
    # values ("0022500001"). Letting pandas infer the dtype turns them
    # into ints and silently strips the leading zeros, which then fail to
    # join against nba_api / CDN / PBP sources keyed on the padded form.
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, dtype={"GAME-ID": str})
    df.columns = [_clean_header(c) for c in df.columns]

    def col(*candidates: str) -> str | None:
        for cand in candidates:
            for c in df.columns:
                if c.upper() == cand.upper():
                    return c
        return None

    c_gameid = col("GAME-ID")
    c_date = col("DATE")
    c_team = col("TEAM")
    c_venue = col("VENUE (R/H/N)", "VENUE")
    if not all([c_gameid, c_date, c_team, c_venue]):
        raise ValueError(
            f"Workbook sheet {sheet_name!r} is missing required columns "
            f"(GAME-ID / DATE / TEAM / VENUE). Found: {list(df.columns)[:15]}"
        )

    df = df[df[c_gameid].notna()].copy()

    # Re-pad in case the source ever delivers unpadded ids: NBA game ids
    # are 10 chars. zfill is a no-op on already-correct values.
    df["_game_id"] = df[c_gameid].astype(str).str.strip().str.zfill(10)
    df["_game_date"] = df[c_date].map(_parse_bdb_date)
    df["_team_abbr"] = df[c_team].astype(str).str.strip().map(lambda t: team_map.get(t, t))
    # Venue is R/H/N — NEUTRAL SITE IS REAL: the 2025-26 workbook has 12
    # neutral rows (6 games: NBA Cup final in Las Vegas, global games).
    # Collapsing N into "away" would wrongly apply road-team treatment in
    # the fatigue/home-court logic, so venue is preserved explicitly and
    # is_home stays False for BOTH teams at a neutral site.
    df["_venue"] = df[c_venue].astype(str).str.strip().str.upper()
    df["_is_home"] = df["_venue"].eq("H")
    df["_is_neutral"] = df["_venue"].eq("N")

    # Opponent = the other row sharing the same GAME-ID
    opp = df.groupby("_game_id")["_team_abbr"].apply(list).to_dict()

    def opponent_of(row: pd.Series) -> str | None:
        pair = opp.get(row["_game_id"], [])
        others = [t for t in pair if t != row["_team_abbr"]]
        return others[0] if others else None

    df["_opponent_abbr"] = df.apply(opponent_of, axis=1)

    # --- team stats frame -------------------------------------------------
    stats = pd.DataFrame({
        "nba_game_id": df["_game_id"],
        "game_date": df["_game_date"],
        "team_abbr": df["_team_abbr"],
        "opponent_abbr": df["_opponent_abbr"],
        "is_home": df["_is_home"],
        "is_neutral_site": df["_is_neutral"],
        "points": df[col("PTS")].map(_to_int) if col("PTS") else None,
        "fg": df[col("FG")].map(_to_int) if col("FG") else None,
        "fga": df[col("FGA")].map(_to_int) if col("FGA") else None,
        "fg3": df[col("3P")].map(_to_int) if col("3P") else None,
        "fg3a": df[col("3PA")].map(_to_int) if col("3PA") else None,
        "ft": df[col("FT")].map(_to_int) if col("FT") else None,
        "fta": df[col("FTA")].map(_to_int) if col("FTA") else None,
        "oreb": df[col("OR")].map(_to_int) if col("OR") else None,
        "dreb": df[col("DR")].map(_to_int) if col("DR") else None,
        "reb": df[col("TOT")].map(_to_int) if col("TOT") else None,
        "ast": df[col("A")].map(_to_int) if col("A") else None,
        "stl": df[col("ST")].map(_to_int) if col("ST") else None,
        "blk": df[col("BL")].map(_to_int) if col("BL") else None,
        "tov": df[col("TO")].map(_to_int) if col("TO") else None,
        "pf": df[col("PF")].map(_to_int) if col("PF") else None,
        "poss": df[col("POSS")].map(_to_float) if col("POSS") else None,
        "pace": df[col("PACE")].map(_to_float) if col("PACE") else None,
        "off_eff": df[col("OEFF")].map(_to_float) if col("OEFF") else None,
        "def_eff": df[col("DEFF")].map(_to_float) if col("DEFF") else None,
        "rest_days": df[col("TEAM REST DAYS")].astype(str) if col("TEAM REST DAYS") else None,
        "source": SOURCE_NAME,
    })

    # --- market lines frame -------------------------------------------------
    def opt(colname: str) -> pd.Series:
        c = col(colname)
        return df[c] if c else pd.Series([None] * len(df), index=df.index)

    market = pd.DataFrame({
        "nba_game_id": df["_game_id"],
        "game_date": df["_game_date"],
        "team_abbr": df["_team_abbr"],
        "opening_spread": opt("OPENING SPREAD").map(_to_float),
        "opening_total": opt("OPENING TOTAL").map(_to_float),
        "opening_odds_raw": opt("OPENING ODDS").astype(str).replace("nan", None),
        "closing_spread": opt("CLOSING SPREAD").map(_to_float),
        "closing_total": opt("CLOSING TOTAL").map(_to_float),
        "closing_odds_raw": opt("CLOSING ODDS").astype(str).replace("nan", None),
        "moneyline": opt("MONEYLINE").astype(str).replace("nan", None),
        "halftime_raw": opt("HALFTIME").astype(str).replace("nan", None),
        "line_movement_1": opt("LINE MOVEMENT #1").astype(str).replace("nan", None),
        "line_movement_2": opt("LINE MOVEMENT #2").astype(str).replace("nan", None),
        "line_movement_3": opt("LINE MOVEMENT #3").astype(str).replace("nan", None),
        "source": SOURCE_NAME,
    })

    market["status"] = [
        market_row_status(cs, ct)
        for cs, ct in zip(market["closing_spread"], market["closing_total"])
    ]

    logger.info(
        "Loaded BigDataBall: %d team-game rows, %d games, %d market rows VALID",
        len(stats), stats["nba_game_id"].nunique(), (market["status"] == "VALID").sum(),
    )
    return stats, market
