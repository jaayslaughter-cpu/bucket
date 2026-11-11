"""
src/ingestion/basketball_reference.py — Basketball-Reference season tables.

ATTRIBUTION IS MANDATORY. Sports Reference's terms read: "When using SR
data, please cite us and provide a link and/or a mention." Every frame this
module returns carries ``SR_ATTRIBUTION`` on its ``.attrs``, and any export
built from it must reproduce that string. The raw CSVs are licensed
third-party data and are gitignored (see data/external/basketball_reference/).

THE CENTRAL INVARIANT: THESE TABLES ARE SEASON AGGREGATES.

A row in "per game", "play-by-play" or "adjusted shooting" summarises a
player's WHOLE season. Joining such a row onto that same season's games as
a feature leaks the future into every single game: a season TS% is computed
from the game being predicted and from every game after it. The Awards
column is the most extreme case — award shares are voted at season's end,
so a model handed "MVP-4" for a November game has been told how the season
turned out.

So this module refuses that join. ``prior_season_features`` and
``attach_prior_season_features`` raise ``SeasonAggregateLeakageError``
unless the table's season strictly precedes the target season. There is no
flag to switch that off. If you want within-season form, build it from
per-game rows with the shift-1 rollers in src/features/builder.py, which
are already leakage-safe.

FOUR STRUCTURAL HAZARDS IN THE RAW CSVs, all handled here:

1. TRADED PLAYERS APPEAR TWICE OVER. A player dealt mid-season gets one
   ``2TM``/``3TM``/``4TM`` row (his season TOTAL) plus one row per team, all
   sharing the same ``Rk``. Summing the column double-counts him. Rows are
   tagged ``IS_MULTI_TEAM_TOTAL``; use ``season_totals`` or ``team_splits``,
   never the raw frame.
2. A ``League Average`` TRAILER ROW sits at the bottom with ``-9999`` in the
   id column. It is dropped, and surfaced separately by ``league_average_row``.
3. EMPTY PERCENTAGE CELLS mean "no attempts", not zero. Jalen Duren's 3P%
   is blank because he took no threes; writing 0.0 there would tell a model
   he is a 0% shooter rather than a non-shooter. Blanks become NaN.
4. DUPLICATE COLUMN NAMES UNDER DIFFERENT GROUP HEADERS. The play-by-play
   table has ``Shoot`` and ``Off.`` twice — once under "Fouls Committed",
   once under "Fouls Drawn". Reading it with a naive header gives
   ``Shoot`` and ``Shoot.1`` and invites mapping fouls drawn onto fouls
   committed. The two-row header is flattened with the group label.

JOIN KEY. The trailing ``-additional`` column carries the stable
Basketball-Reference player id (``doncilu01``), exposed as
``BBREF_PLAYER_ID``. It is NOT the NBA.com person id the panel keys on, and
no crosswalk between them exists in this repository yet. Until one does,
attaching falls back to normalised player names and REPORTS its miss rate
rather than fuzzy-matching; unmatched players keep NaN features.

SCOPE: NBA only. Team codes are validated against the NBA set, so a college
or G-League export is refused rather than silently ingested.

RESEARCH_ONLY. No credential, no wager, no bot-wall bypass -- the same line
`underdog_props.py` and `sleeper_props.py` carry, and it binds the fetch
section at the bottom of this file too. This host's wall reads the client
identity, so the scheduled fetch ships DISABLED with an honest User-Agent
rather than enabled with a browser-shaped one; see `SR_USER_AGENT`.
"""

from __future__ import annotations

import dataclasses
import io
import logging
import os
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

logger = logging.getLogger(__name__)

SOURCE_NAME = "basketball_reference"

SR_ATTRIBUTION = (
    "Data from Basketball-Reference.com (Sports Reference LLC). "
    "When using SR data, please cite us and provide a link and/or a mention: "
    "https://www.basketball-reference.com/"
)

# Rows whose Team reads like this are the player's season TOTAL across a
# trade, not a team he played for. They coexist with per-team rows.
MULTI_TEAM_CODES = frozenset({"2TM", "3TM", "4TM", "5TM", "6TM"})

# Basketball-Reference spells three franchises differently from NBA.com.
# Everything else is identical, so only the differences are listed.
SR_TO_NBA_TEAM: dict[str, str] = {"BRK": "BKN", "CHO": "CHA", "PHO": "PHX"}

NBA_TEAM_ABBREVIATIONS = frozenset({
    "ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW",
    "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK",
    "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS",
})

# Columns that are decided AFTER the season they describe. Never a
# same-season feature; admissible from a prior season only, and even then
# only on explicit opt-in because they encode voter reputation rather than
# production.
SEASON_END_COLS = frozenset({"AWARDS"})

# Identity/bookkeeping columns — carried through but never treated as stats.
IDENTITY_COLS = ("RK", "PLAYER", "AGE", "TEAM", "POS", "BBREF_PLAYER_ID", "AWARDS")

LEAGUE_AVERAGE_LABEL = "league average"
SR_NULL_SENTINEL = "-9999"

# Per-table column prefixes. Namespacing matters here: MP means minutes PER
# GAME in the per-game table and TOTAL minutes in the other two. Unprefixed,
# one would silently overwrite the other on a merge.
TABLE_PREFIXES: dict[str, str] = {
    "per_game": "SR_PG",
    "per_100_poss": "SR_P100",
    "play_by_play": "SR_PBP",
    "adjusted_shooting": "SR_ADJ",
    # The advanced table (PER, WS, BPM) — the prior-season BPM source this
    # project had no equivalent of. Distinct from SR_ADJ by one character,
    # which is deliberate: they are different tables and a shared prefix
    # would let one overwrite the other's columns on a merge.
    "advanced": "SR_ADV",
    "unknown": "SR",
}

_NAME_SUFFIXES = frozenset({"jr", "sr", "ii", "iii", "iv", "v"})


class BasketballReferenceError(RuntimeError):
    """Raised when an SR export cannot be parsed or is out of scope."""


class SeasonAggregateLeakageError(RuntimeError):
    """Raised when a season aggregate would be joined onto its own season."""


# ---------------------------------------------------------------------------
# header handling
# ---------------------------------------------------------------------------


def _to_col_name(label: str) -> str:
    """'FG%' -> 'FG_PCT', 'TS+' -> 'TS_PLUS', 'Off.' -> 'OFF'."""
    text = str(label).strip()
    text = text.replace("%", " pct").replace("+", " plus").replace("/", " per ")
    text = re.sub(r"[^0-9A-Za-z]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_").upper()


def _flatten_header(group_row: list[str] | None, name_row: list[str]) -> list[str]:
    """
    Flatten SR's two-row header, disambiguating repeats with the group label.

    Only names that actually repeat get the group prefix, so ``PTS`` stays
    ``PTS`` while the play-by-play table's two ``Shoot`` columns become
    ``FOULS_COMMITTED_SHOOT`` and ``FOULS_DRAWN_SHOOT``.
    """
    names = [str(n).strip() for n in name_row]
    groups = [str(g).strip() for g in (group_row or [""] * len(names))]
    if len(groups) < len(names):
        groups = groups + [""] * (len(names) - len(groups))

    counts: dict[str, int] = {}
    for name in names:
        key = _to_col_name(name)
        counts[key] = counts.get(key, 0) + 1

    out: list[str] = []
    seen: dict[str, int] = {}
    for idx, name in enumerate(names):
        base = _to_col_name(name)
        group = _to_col_name(groups[idx])
        if not base:
            base = group or f"COL_{idx}"
            label = base
        elif counts[base] > 1 and group:
            label = f"{group}_{base}"
        else:
            label = base
        # Belt and braces: a collision that survives the group prefix would
        # otherwise let pandas drop one of the two columns on assignment.
        seen[label] = seen.get(label, 0) + 1
        if seen[label] > 1:
            label = f"{label}_{seen[label]}"
        out.append(label)
    return out


def _find_header_row(raw: pd.DataFrame) -> tuple[int, str | None]:
    """
    SR exports may carry one or two header rows; the real one starts 'Rk'.

    Returns the row index and, when the table's title was pasted onto the
    front of that cell ("Per 100 Possessions    Rk"), the prefix that was
    stripped — so the caller can say what it dropped rather than silently
    reinterpreting the file.
    """
    for idx in range(min(len(raw), 5)):
        first = str(raw.iat[idx, 0]).strip()
        if first.lower() == "rk":
            return idx, None
        row = {str(c).strip().lower() for c in raw.iloc[idx]}
        if first.lower().endswith("rk") and "player" in row:
            return idx, first[: -2].strip()
    raise BasketballReferenceError(
        "No header row starting with 'Rk' in the first five lines. This does "
        "not look like a Basketball-Reference season table; refusing to guess "
        "which line is the header."
    )


def _identify_id_column(columns: list[str]) -> str | None:
    """The trailing '-additional' / '-9999' column holds the BBRef player id."""
    for col in reversed(columns):
        if "ADDITIONAL" in col or "9999" in col:
            return col
    return None


# ---------------------------------------------------------------------------
# table kind
# ---------------------------------------------------------------------------

_KIND_SIGNATURES: tuple[tuple[str, tuple[str, ...]], ...] = (
    # Ordered most-specific first, and the order is load-bearing. The per-100
    # and per-game tables share every counting column; only ORtg/DRtg tell
    # them apart. Checked the other way round, a per-100 table parses as
    # per_game and Dončić's 45.7 points per 100 possessions lands in the
    # column that holds 33.5 points per game.
    ("adjusted_shooting", ("TS_PLUS", "EFG_PLUS")),
    ("play_by_play", ("PG_PCT", "ONCOURT")),
    ("per_100_poss", ("ORTG", "DRTG", "PTS")),
    ("advanced", ("PER", "WS", "BPM")),
    ("per_game", ("PTS", "TRB", "FG_PCT")),
)


def infer_table_kind(columns: Iterable[str]) -> str:
    cols = set(columns)
    for kind, signature in _KIND_SIGNATURES:
        if all(sig in cols for sig in signature):
            return kind
    return "unknown"


# ---------------------------------------------------------------------------
# season helpers
# ---------------------------------------------------------------------------


def season_start_year(season: str) -> int:
    """'2025-26' -> 2025. Also accepts '2025-2026' and a bare '2025'."""
    text = str(season).strip()
    match = re.match(r"^(\d{4})(?:-\d{2,4})?$", text)
    if not match:
        raise BasketballReferenceError(
            f"Unrecognised season {season!r}. Use the NBA.com form '2025-26' so "
            "seasons can be ordered; an unorderable season cannot be checked "
            "for leakage."
        )
    return int(match.group(1))


# ---------------------------------------------------------------------------
# the parsed table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SrSeasonTable:
    """One parsed Basketball-Reference season table."""

    kind: str
    season: str
    frame: pd.DataFrame
    stat_cols: tuple[str, ...]
    source: str
    league_average: dict[str, Any] = field(default_factory=dict)
    dropped_rows: int = 0
    attribution: str = SR_ATTRIBUTION

    @property
    def start_year(self) -> int:
        return season_start_year(self.season)

    @property
    def prefix(self) -> str:
        return TABLE_PREFIXES.get(self.kind, TABLE_PREFIXES["unknown"])


def _coerce_numeric(frame: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = frame.copy()
    for col in cols:
        # errors="coerce" turns the empty percentage cells into NaN. It does
        # NOT touch '.000', which is a real zero and must survive as 0.0.
        out[col] = pd.to_numeric(out[col].replace("", pd.NA), errors="coerce")
    return out


def _assert_nba_only(teams: pd.Series, *, source: str) -> None:
    codes = {
        str(t).strip().upper()
        for t in teams.dropna().unique()
        if str(t).strip()
    }
    known = NBA_TEAM_ABBREVIATIONS | MULTI_TEAM_CODES | set(SR_TO_NBA_TEAM)
    unknown = sorted(codes - known)
    if unknown:
        raise BasketballReferenceError(
            f"{source}: team codes {unknown} are not NBA franchises. This "
            "project is NBA-only; refusing to ingest a college, G-League or "
            "international export."
        )


def read_sr_season_csv(
    path_or_buffer: str | Path | Any,
    *,
    season: str,
    kind: str | None = None,
) -> SrSeasonTable:
    """
    Parse a Basketball-Reference season CSV (per game / play-by-play /
    adjusted shooting) into a tidy frame.

    ``season`` is not inferred: the CSV does not carry it, and a wrong
    season silently defeats every leakage check downstream. The caller must
    state it.
    """
    season_start_year(season)  # validate early, before any parsing work
    source = str(getattr(path_or_buffer, "name", path_or_buffer))

    raw = pd.read_csv(
        path_or_buffer,
        header=None,
        dtype=str,
        keep_default_na=False,
        skip_blank_lines=True,
    )
    if raw.empty:
        raise BasketballReferenceError(f"{source}: file is empty")

    header_idx, title_prefix = _find_header_row(raw)
    header_cells = list(raw.iloc[header_idx])
    if title_prefix:
        logger.warning(
            "%s: header cell began with %r; treating the rest as the 'Rk' "
            "column. This happens when a table's title is pasted onto the "
            "header line.", source, title_prefix,
        )
        header_cells[0] = "Rk"
    group_row = list(raw.iloc[header_idx - 1]) if header_idx > 0 else None
    columns = _flatten_header(group_row, header_cells)

    body = raw.iloc[header_idx + 1 :].copy()
    body.columns = columns
    body = body.reset_index(drop=True)

    id_col = _identify_id_column(columns)
    if id_col is not None:
        body = body.rename(columns={id_col: "BBREF_PLAYER_ID"})
        columns = list(body.columns)

    if "PLAYER" not in body.columns:
        raise BasketballReferenceError(
            f"{source}: no Player column after flattening header {columns}"
        )

    for col in body.columns:
        body[col] = body[col].astype(str).str.strip()

    before = len(body)

    # SR repeats the header every ~20 rows in some exports.
    body = body[body["PLAYER"].str.lower() != "player"]

    # The League Average trailer: kept aside, never a player row.
    is_league_avg = body["PLAYER"].str.lower() == LEAGUE_AVERAGE_LABEL
    league_average = (
        body[is_league_avg].iloc[0].to_dict() if is_league_avg.any() else {}
    )
    body = body[~is_league_avg]

    if "BBREF_PLAYER_ID" in body.columns:
        body = body[body["BBREF_PLAYER_ID"] != SR_NULL_SENTINEL]

    body = body.reset_index(drop=True)
    dropped = before - len(body)

    if body.empty:
        raise BasketballReferenceError(
            f"{source}: no player rows left after dropping header repeats and "
            "the League Average trailer"
        )

    if "TEAM" in body.columns:
        _assert_nba_only(body["TEAM"], source=source)
        body["TEAM"] = body["TEAM"].str.upper()
        body["NBA_TEAM"] = body["TEAM"].map(lambda t: SR_TO_NBA_TEAM.get(t, t))
        body["IS_MULTI_TEAM_TOTAL"] = body["TEAM"].isin(MULTI_TEAM_CODES)
    else:
        body["IS_MULTI_TEAM_TOTAL"] = False

    stat_cols = tuple(
        c for c in body.columns
        if c not in IDENTITY_COLS
        and c not in {"NBA_TEAM", "IS_MULTI_TEAM_TOTAL"}
    )
    body = _coerce_numeric(body, stat_cols)
    if "RK" in body.columns:
        body["RK"] = pd.to_numeric(body["RK"], errors="coerce")

    body["SEASON"] = season
    body["PLAYER_KEY"] = body["PLAYER"].map(normalise_player_name)

    resolved_kind = kind or infer_table_kind(body.columns)
    if resolved_kind == "unknown":
        logger.warning(
            "%s: could not identify the SR table kind from its columns; "
            "columns will be prefixed 'SR_' rather than by table.", source,
        )

    body.attrs["attribution"] = SR_ATTRIBUTION
    body.attrs["source"] = source

    table = SrSeasonTable(
        kind=resolved_kind,
        season=season,
        frame=body,
        stat_cols=stat_cols,
        source=source,
        league_average=league_average,
        dropped_rows=dropped,
    )
    logger.info(
        "%s: parsed %d %s rows for %s (%d non-player rows dropped, "
        "%d multi-team TOTAL rows)",
        source, len(body), resolved_kind, season, dropped,
        int(body["IS_MULTI_TEAM_TOTAL"].sum()),
    )
    return table


# ---------------------------------------------------------------------------
# multi-team rows
# ---------------------------------------------------------------------------


def season_totals(table: SrSeasonTable) -> pd.DataFrame:
    """
    One row per player: the ``2TM``/``3TM`` TOTAL for traded players, the
    single row for everyone else. Never both, so nothing double-counts.
    """
    frame = table.frame
    key = "BBREF_PLAYER_ID" if "BBREF_PLAYER_ID" in frame.columns else "PLAYER_KEY"
    traded = set(frame.loc[frame["IS_MULTI_TEAM_TOTAL"], key])
    keep = (~frame[key].isin(traded)) | frame["IS_MULTI_TEAM_TOTAL"]
    out = frame[keep].copy()

    duplicated = out[key].duplicated(keep=False)
    if duplicated.any():
        raise BasketballReferenceError(
            f"{table.source}: {int(duplicated.sum())} rows remain duplicated on "
            f"{key} after collapsing multi-team blocks "
            f"({sorted(set(out.loc[duplicated, key]))[:5]}). Refusing to return "
            "a frame that would double-count a player."
        )
    return out.reset_index(drop=True)


def team_splits(table: SrSeasonTable) -> pd.DataFrame:
    """Per-team rows only. The TOTAL rows are excluded."""
    return table.frame[~table.frame["IS_MULTI_TEAM_TOTAL"]].reset_index(drop=True)


def multi_team_report(table: SrSeasonTable) -> pd.DataFrame:
    """
    Cross-check every TOTAL row against its per-team rows.

    A games-played mismatch means the block was mis-parsed or the export is
    partial. Reported rather than raised, because a hand-trimmed export is a
    legitimate reason for the split rows to be missing.
    """
    frame = table.frame
    key = "BBREF_PLAYER_ID" if "BBREF_PLAYER_ID" in frame.columns else "PLAYER_KEY"
    if "G" not in frame.columns:
        return pd.DataFrame(columns=[key, "PLAYER", "TEAM", "TOTAL_G", "SPLIT_G", "SPLIT_ROWS"])

    rows: list[dict[str, Any]] = []
    for pid, block in frame[frame[key].isin(
        set(frame.loc[frame["IS_MULTI_TEAM_TOTAL"], key])
    )].groupby(key):
        total = block[block["IS_MULTI_TEAM_TOTAL"]]
        splits = block[~block["IS_MULTI_TEAM_TOTAL"]]
        total_g = float(total["G"].iloc[0]) if len(total) else float("nan")
        split_g = float(splits["G"].sum())
        rows.append({
            key: pid,
            "PLAYER": block["PLAYER"].iloc[0],
            "TEAM": total["TEAM"].iloc[0] if len(total) else None,
            "TOTAL_G": total_g,
            "SPLIT_G": split_g,
            "SPLIT_ROWS": int(len(splits)),
            "MATCHES": bool(total_g == split_g),
        })
    report = pd.DataFrame(rows)
    if not report.empty and not report["MATCHES"].all():
        bad = report[~report["MATCHES"]]
        logger.warning(
            "%s: %d multi-team blocks where TOTAL games != sum of split games "
            "(%s). Use season_totals() or team_splits(), never both.",
            table.source, len(bad), list(bad["PLAYER"])[:5],
        )
    return report


def league_average_row(table: SrSeasonTable) -> dict[str, Any]:
    """The trailer row, coerced to floats. Empty dict when absent."""
    out: dict[str, Any] = {}
    for key, value in table.league_average.items():
        if key in {"PLAYER", "TEAM", "POS", "BBREF_PLAYER_ID", "AWARDS"}:
            continue
        numeric = pd.to_numeric(pd.Series([value]).replace("", pd.NA), errors="coerce")
        if pd.notna(numeric.iloc[0]):
            out[key] = float(numeric.iloc[0])
    return out


# ---------------------------------------------------------------------------
# player-name matching
# ---------------------------------------------------------------------------


def normalise_player_name(name: Any) -> str:
    """
    'Nikola Jokić' -> 'nikola jokic'. Accents are stripped because NBA.com
    writes the ASCII form and Basketball-Reference does not.
    """
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _strip_suffix(key: str) -> str:
    parts = key.split()
    while len(parts) > 2 and parts[-1] in _NAME_SUFFIXES:
        parts = parts[:-1]
    return " ".join(parts)


# ---------------------------------------------------------------------------
# the only sanctioned join
# ---------------------------------------------------------------------------


def _assert_prior_season(table: SrSeasonTable, target_season: str) -> None:
    target = season_start_year(target_season)
    if table.start_year >= target:
        raise SeasonAggregateLeakageError(
            f"Refusing to build features for {target_season} from the "
            f"{table.season} {table.kind} table. These rows are SEASON "
            "AGGREGATES: a season average computed over "
            f"{table.season} includes the game being predicted and every game "
            "after it, so joining it onto its own season leaks the future into "
            "every row. Use the prior season's table, or build within-season "
            "form from per-game rows with the shift-1 rollers in "
            "src/features/builder.py."
        )


def prior_season_features(
    table: SrSeasonTable,
    *,
    target_season: str,
    include_awards: bool = False,
    columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    Turn a season table into prior-season features for ``target_season``.

    Raises ``SeasonAggregateLeakageError`` unless the table's season strictly
    precedes the target. Column names are prefixed per table and carry the
    source season, so a leaked join is visible in the feature names
    themselves (``SR_PG_PRIOR_PTS``, ``SR_PRIOR_SEASON``).

    ``include_awards`` is off by default. Prior-season awards are not
    leakage — they were voted before the target season tipped off — but they
    encode voter reputation rather than production, so opting in is a
    deliberate choice.
    """
    _assert_prior_season(table, target_season)

    totals = season_totals(table)
    requested = tuple(columns) if columns is not None else table.stat_cols
    missing = [c for c in requested if c not in totals.columns]
    if missing:
        raise BasketballReferenceError(
            f"{table.source}: requested columns {missing} are not in this "
            f"{table.kind} table. Available: {sorted(totals.columns)}"
        )

    selected = [c for c in requested if c.upper() not in SEASON_END_COLS]
    prefix = table.prefix
    out = totals[["PLAYER", "PLAYER_KEY"] + list(selected)].copy()
    if "BBREF_PLAYER_ID" in totals.columns:
        out.insert(0, "BBREF_PLAYER_ID", totals["BBREF_PLAYER_ID"])
    out = out.rename(columns={c: f"{prefix}_PRIOR_{c}" for c in selected})

    if include_awards and "AWARDS" in totals.columns:
        out[f"{prefix}_PRIOR_AWARDS"] = totals["AWARDS"].replace("", pd.NA)

    out["SR_PRIOR_SEASON"] = table.season
    out.attrs["attribution"] = SR_ATTRIBUTION
    return out.reset_index(drop=True)


@dataclass(frozen=True)
class PriorSeasonJoinReport:
    """What the name-based join actually managed to match."""

    target_seasons: tuple[str, ...]
    matched_rows: int
    unmatched_rows: int
    matched_players: int
    unmatched_players: tuple[str, ...]
    seasons_without_table: tuple[str, ...]
    suffix_matched_players: tuple[str, ...] = ()
    ambiguous_players: tuple[str, ...] = ()

    @property
    def match_rate(self) -> float:
        total = self.matched_rows + self.unmatched_rows
        return float(self.matched_rows) / total if total else 0.0

    def summary(self) -> str:
        return (
            f"prior-season SR features: {self.matched_rows}/"
            f"{self.matched_rows + self.unmatched_rows} rows matched "
            f"({self.match_rate:.1%}), {self.matched_players} players matched, "
            f"{len(self.unmatched_players)} unmatched, "
            f"{len(self.seasons_without_table)} seasons without a table"
        )


def attach_prior_season_features(
    panel: pd.DataFrame,
    tables: Mapping[str, SrSeasonTable] | Iterable[SrSeasonTable],
    *,
    season_col: str = "SEASON",
    name_col: str = "PLAYER_NAME",
    include_awards: bool = False,
    columns: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, PriorSeasonJoinReport]:
    """
    Attach each panel row's PRIOR-season SR aggregate.

    For a row in season S, the table for season S-1 is used. Seasons with no
    such table, and players with no matching prior-season row (rookies,
    renames, missing crosswalk), keep NaN — they are never filled with a
    league average, which would tell the model a rookie is exactly average.

    Read the returned report before training on these columns: a low match
    rate means the name join failed, not that the league is full of rookies.
    """
    if season_col not in panel.columns:
        raise BasketballReferenceError(
            f"panel has no {season_col!r} column; cannot tell which season each "
            "row belongs to, and therefore cannot check for leakage"
        )
    if name_col not in panel.columns:
        raise BasketballReferenceError(f"panel has no {name_col!r} column")

    by_start_year: dict[int, SrSeasonTable] = {}
    table_list = list(tables.values()) if isinstance(tables, Mapping) else list(tables)
    for table in table_list:
        by_start_year[table.start_year] = table
    if not by_start_year:
        raise BasketballReferenceError("no SR tables supplied")

    kinds = {t.kind for t in table_list}
    if len(kinds) > 1:
        raise BasketballReferenceError(
            f"attach one table kind at a time; got {sorted(kinds)}. Different "
            "kinds use the same column names for different quantities (MP is "
            "per-game in one and a season total in another)."
        )

    panel_start_years = {
        season_start_year(s) for s in panel[season_col].dropna().astype(str).unique()
    }
    usable = [t for t in table_list if t.start_year + 1 in panel_start_years]
    if not usable:
        same_season = sorted(
            t.season for t in table_list if t.start_year in panel_start_years
        )
        if same_season:
            raise SeasonAggregateLeakageError(
                f"The only SR tables supplied ({same_season}) cover seasons the "
                "panel is IN, not seasons before it. Attaching them would join a "
                "season aggregate onto the very games it was computed from. "
                "Supply the prior season's table instead."
            )
        raise BasketballReferenceError(
            f"None of the supplied SR tables "
            f"({sorted(t.season for t in table_list)}) is the season before any "
            f"panel season ({sorted(panel_start_years)}); the join would produce "
            "nothing but NaN."
        )

    out = panel.copy()
    out["_sr_player_key"] = out[name_col].map(normalise_player_name)

    feature_cols: list[str] = []
    matched_mask = pd.Series(False, index=out.index)
    seasons_without: list[str] = []
    matched_players: set[str] = set()
    unmatched_players: set[str] = set()
    suffix_matched: set[str] = set()
    ambiguous: set[str] = set()

    for season in sorted(out[season_col].dropna().astype(str).unique()):
        rows = out[season_col].astype(str) == season
        prior = by_start_year.get(season_start_year(season) - 1)
        if prior is None:
            seasons_without.append(season)
            unmatched_players.update(out.loc[rows, "_sr_player_key"])
            continue

        feats = prior_season_features(
            prior,
            target_season=season,
            include_awards=include_awards,
            columns=columns,
        )
        lookup = feats.drop_duplicates(subset="PLAYER_KEY", keep=False).set_index("PLAYER_KEY")
        dupes = set(feats["PLAYER_KEY"]) - set(lookup.index)
        ambiguous.update(dupes)

        value_cols = [c for c in lookup.columns if c.startswith(("SR_", "PLAYER"))]
        value_cols = [c for c in value_cols if c not in {"PLAYER", "BBREF_PLAYER_ID"}]

        keys = out.loc[rows, "_sr_player_key"]
        hit = keys.isin(lookup.index)

        # Second pass for suffix differences ('Jabari Smith Jr.' vs 'Jabari
        # Smith'), and only where the stripped key is unambiguous on BOTH
        # sides. A stripped key that maps to two people is left unmatched
        # rather than guessed at.
        stripped_lookup: dict[str, str] = {}
        counts: dict[str, int] = {}
        for key in lookup.index:
            stripped = _strip_suffix(key)
            counts[stripped] = counts.get(stripped, 0) + 1
            stripped_lookup[stripped] = key
        resolved = keys.copy()
        for idx in keys.index[~hit]:
            stripped = _strip_suffix(keys.at[idx])
            if counts.get(stripped, 0) == 1:
                resolved.at[idx] = stripped_lookup[stripped]
                suffix_matched.add(keys.at[idx])
            elif counts.get(stripped, 0) > 1:
                ambiguous.add(keys.at[idx])

        final_hit = resolved.isin(lookup.index)
        for col in value_cols:
            if col not in out.columns:
                out[col] = pd.NA
                feature_cols.append(col)
            values = resolved[final_hit].map(lookup[col])
            out.loc[values.index, col] = values

        matched_mask.loc[resolved.index[final_hit]] = True
        matched_players.update(resolved[final_hit])
        unmatched_players.update(keys[~final_hit])

    for col in feature_cols:
        # SR_PRIOR_SEASON and the opt-in awards string are genuinely textual;
        # coercing them would blank them out. Everything else is a stat.
        if col == "SR_PRIOR_SEASON" or col.endswith("_AWARDS"):
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.drop(columns=["_sr_player_key"])
    report = PriorSeasonJoinReport(
        target_seasons=tuple(sorted(out[season_col].dropna().astype(str).unique())),
        matched_rows=int(matched_mask.sum()),
        unmatched_rows=int((~matched_mask).sum()),
        matched_players=len(matched_players),
        unmatched_players=tuple(sorted(p for p in unmatched_players if p)),
        seasons_without_table=tuple(seasons_without),
        suffix_matched_players=tuple(sorted(suffix_matched)),
        ambiguous_players=tuple(sorted(a for a in ambiguous if a)),
    )
    out.attrs["attribution"] = SR_ATTRIBUTION
    logger.info("%s", report.summary())
    if report.match_rate < 0.5 and report.matched_rows + report.unmatched_rows:
        logger.warning(
            "SR prior-season join matched under half the panel rows. Check the "
            "player-name spellings before training on these columns; the join "
            "is name-based because no NBA.com<->BBRef id crosswalk exists yet."
        )
    return out, report


# ---------------------------------------------------------------------------
# inspection
# ---------------------------------------------------------------------------


def describe_sr_csv(path_or_buffer: str | Path | Any, *, season: str) -> dict[str, Any]:
    """Parse and summarise a table without committing to any join."""
    return describe_sr_table(read_sr_season_csv(path_or_buffer, season=season))


def describe_sr_table(table: SrSeasonTable) -> dict[str, Any]:
    """Summarise an already-parsed table (no re-read, so buffers are safe)."""
    frame = table.frame
    return {
        "source": table.source,
        "kind": table.kind,
        "season": table.season,
        "rows": int(len(frame)),
        "players": int(frame["PLAYER_KEY"].nunique()),
        "columns": list(frame.columns),
        "stat_cols": list(table.stat_cols),
        "multi_team_totals": int(frame["IS_MULTI_TEAM_TOTAL"].sum()),
        "dropped_non_player_rows": int(table.dropped_rows),
        "has_league_average": bool(table.league_average),
        "has_awards": "AWARDS" in frame.columns,
        "has_bbref_id": "BBREF_PLAYER_ID" in frame.columns,
        "attribution": table.attribution,
    }


# ---------------------------------------------------------------------------
# fetching the season tables from the site (added 2026-11-10)
# ---------------------------------------------------------------------------
#
# WHY THIS IS HERE. The parser above read CSVs a person exported by hand, so
# the tables were one manual step away from being a feature at all, and
# nothing scheduled ever refreshed them. This section makes the transport a
# job instead of a human.
#
# BUT IT DOES NOT RUN BY DEFAULT, AND THE REASON IS NOT POLITENESS.
# basketball-reference.com answered a 200 in 0.16s from this environment on
# 2026-11-10 (a 2.4 MB per-game page, a 2.2 MB advanced page) -- with a
# browser-shaped User-Agent. Measured again on 2026-11-11 with an honest one:
# 403, on both the prior season's page and the live one. The host's wall reads
# the client identity, so the 200 was bought by claiming to be a browser.
# `$PROPIQ_BBREF_INGEST` therefore defaults false and `SR_USER_AGENT` is
# honest; everything below is a parser, a cache and a backoff that a
# deployment with a legitimate route to this data can switch on deliberately.
#
# NO NEW DEPENDENCY, ON PURPOSE. The HTML is parsed with the standard library
# (regex over the table's own ``data-stat`` cell attributes), not lxml or
# BeautifulSoup or pandas.read_html: this project's rule is that a feature
# another repository has does not justify a new import, and SR's markup is
# regular enough that the machine keys are the robust interface — a header
# TEXT change does not move anything, and the CSV headers emitted below are
# the same ones the hand-exported CSVs carried, so ``read_sr_season_csv``
# (and every guard in it: the League Average trailer, the multi-team blocks,
# the NBA-only assertion, the leakage refusal) runs unchanged over what the
# fetch produces. One parse path, two transports.
#
# PRIOR-SEASON TABLES ARE FINAL, SO THE CACHE IS FOREVER. A season's per-game
# and advanced tables stop changing when that season ends, so a cached CSV is
# not staleness to be re-fetched nightly — it IS the table. The job fetches
# once per (season, table) and no-ops on every later run unless
# ``refresh=True``. That is also the polite rate limit: one request per table
# per season, ever.
#
# ATTRIBUTION TRAVELS WITH THE TABLE, as SR's terms ask for.

#: kind -> (page slug, HTML table id). Verified against the live pages on
#: 2026-11-10: the per-game page carries ``per_game_stats`` and the advanced
#: page carries ``advanced`` (``advanced_post`` is the playoffs table, a
#: different season type, and is deliberately not offered here).
SR_PAGES: dict[str, tuple[str, str]] = {
    "per_game": ("per_game", "per_game_stats"),
    "advanced": ("advanced", "advanced"),
}

#: The pages' own machine keys -> the CSV headers the parser expects. The
#: keys are SR's ``data-stat`` cell attributes; the values are the headers
#: the "Get table as CSV" export produces, which is the shape
#: ``read_sr_season_csv`` was built against. ``mp`` is TOTAL minutes in the
#: advanced table and PER GAME in the per-game table — the prefixes keep
#: them apart downstream.
DATA_STAT_TO_HEADER: dict[str, dict[str, str]] = {
    "per_game": {
        "ranker": "Rk", "name_display": "Player", "age": "Age",
        "team_name_abbr": "Team", "pos": "Pos", "games": "G",
        "games_started": "GS", "mp_per_g": "MP",
        "fg_per_g": "FG", "fga_per_g": "FGA", "fg_pct": "FG%",
        "fg3_per_g": "3P", "fg3a_per_g": "3PA", "fg3_pct": "3P%",
        "fg2_per_g": "2P", "fg2a_per_g": "2PA", "fg2_pct": "2P%",
        "efg_pct": "eFG%", "ft_per_g": "FT", "fta_per_g": "FTA",
        "ft_pct": "FT%", "orb_per_g": "ORB", "drb_per_g": "DRB",
        "trb_per_g": "TRB", "ast_per_g": "AST", "stl_per_g": "STL",
        "blk_per_g": "BLK", "tov_per_g": "TOV", "pf_per_g": "PF",
        "pts_per_g": "PTS", "awards": "Awards",
    },
    "advanced": {
        "ranker": "Rk", "name_display": "Player", "age": "Age",
        "team_name_abbr": "Team", "pos": "Pos", "games": "G",
        "games_started": "GS", "mp": "MP", "per": "PER",
        "ts_pct": "TS%", "fg3a_per_fga_pct": "3PAr",
        "fta_per_fga_pct": "FTr", "orb_pct": "ORB%", "drb_pct": "DRB%",
        "trb_pct": "TRB%", "ast_pct": "AST%", "stl_pct": "STL%",
        "blk_pct": "BLK%", "tov_pct": "TOV%", "usg_pct": "USG%",
        "ows": "OWS", "dws": "DWS", "ws": "WS", "ws_per_48": "WS/48",
        "obpm": "OBPM", "dbpm": "DBPM", "bpm": "BPM", "vorp": "VORP",
        "awards": "Awards",
    },
}

#: AN HONEST CLIENT IDENTITY, AND IT GETS A 403. Measured 2026-11-11 from
#: this container: this exact User-Agent returns HTTP 403 on
#: ``NBA_2026_advanced.html``. A browser-shaped string was what the original
#: draft of this module shipped, and it is what makes the fetch return 200 --
#: which means a 200 here is obtained by telling the host this is a browser
#: when it is not. THAT IS THE BOT WALL, NOT A TRANSPORT DETAIL, so the
#: default does not do it. The job ships disabled (see
#: ``$PROPIQ_BBREF_INGEST``) and the honest agent stays the default, so the
#: failure mode out of the box is a refusal this project reports rather than a
#: circumvention it performs.
#:
#: The parser, cache and backoff below are the reusable part and are
#: exercised by ``tests/test_basketball_reference.py`` against stored HTML; a
#: deployment that has a legitimate route to this data (the site's own export,
#: a licence, or a residential egress its operator is entitled to use) sets
#: ``$PROPIQ_BBREF_USER_AGENT`` and ``$PROPIQ_BBREF_INGEST`` itself. That is
#: the operator's call to make, not this file's.
SR_USER_AGENT = "PropIQ-Research/1.0 (+non-commercial NBA research)"

#: Where the fetched CSVs are cached. The raw CSVs are licensed third-party
#: data and gitignored; the deployed path is the volume-backed state root.
SR_CACHE_DIR = Path("data/external/basketball_reference")

#: Politeness: the same backoff shape every client in this project uses.
SR_RETRY_ATTEMPTS = 3
SR_RETRY_BACKOFF_SECONDS = 2.0
SR_TIMEOUT_SECONDS = 30.0


def sr_page_url(season: str, kind: str) -> str:
    """``('2025-26', 'advanced')`` -> the season's advanced-table page URL."""
    slug, _table_id = _sr_page(kind)
    end_year = season_start_year(season) + 1
    return f"https://www.basketball-reference.com/leagues/NBA_{end_year}_{slug}.html"


def _sr_page(kind: str) -> tuple[str, str]:
    try:
        return SR_PAGES[kind]
    except KeyError as exc:
        raise BasketballReferenceError(
            f"No Basketball-Reference page mapping for {kind!r}. Fetched "
            f"kinds are {sorted(SR_PAGES)}; the play-by-play and "
            "adjusted-shooting tables stay on the hand-exported CSV path."
        ) from exc


def default_prior_seasons(now: Any | None = None) -> list[str]:
    """
    The immediately preceding NBA season, as a one-element list.

    '2025-26' for any date in the 2026-27 season. The prior season's tables
    are the only ones the leakage guard admits as features, so they are what
    a scheduled fetch wants by default.
    """
    from datetime import date

    from src.utils.timezones import pacific_calendar_date

    today = now if isinstance(now, date) else pacific_calendar_date()
    current_start = today.year if today.month >= 7 else today.year - 1
    prior_start = current_start - 1
    return [f"{prior_start}-{str(prior_start + 1)[-2:]}"]


#: The month and day the NBA regular season is underway by. The league has
#: tipped off on or before October 22nd every season since 1980, so a table
#: fetched on this date has rows; before it, the current season's page has
#: nothing to parse. The 20th trades a few days' delay in a
#: late-starting season for never fetching a table that cannot exist yet.
SR_REGULAR_SEASON_UNDERWAY = (10, 20)


def default_current_season(now: Any | None = None) -> str:
    """
    The season currently underway: ``'2026-27'`` for any date from July 2026
    through June 2027. July is the new season's first month, so a July date
    still names the season that October will tip off.
    """
    from datetime import date

    from src.utils.timezones import pacific_calendar_date

    today = now if isinstance(now, date) else pacific_calendar_date()
    start = today.year if today.month >= 7 else today.year - 1
    return f"{start}-{str(start + 1)[-2:]}"


def is_current_season(season: str, now: Any | None = None) -> bool:
    """
    Is ``season`` the NBA season currently being played?

    True only while the regular season is underway — from October 20th of
    the starting year through June 30th of the ending one. July 1st the
    season is over: the table is final and is no longer a live object.
    """
    from datetime import date

    from src.utils.timezones import pacific_calendar_date

    today = now if isinstance(now, date) else pacific_calendar_date()
    start = season_start_year(season)
    if today.year == start:
        month, day = SR_REGULAR_SEASON_UNDERWAY
        return today.month > month or (today.month == month and today.day >= day)
    if today.year == start + 1:
        return today.month <= 6
    return False


def default_fetch_seasons(now: Any | None = None) -> list[str]:
    """
    The seasons the scheduled fetch wants: the prior season's final tables
    plus the current season's while it is underway.

    The current-season table is season-to-date data, not a final table, and
    fetching it does NOT make it a feature. The leakage guard
    (``prior_season_features``) still refuses to join a season's aggregate
    onto its own season, so the fetched rows are data for the A/B
    measurement — and, the day the season ends, the cache file already IS
    next season's prior table, which the prior-only fetch then serves
    without a request.
    """
    seasons = default_prior_seasons(now)
    current = default_current_season(now)
    if is_current_season(current, now) and current not in seasons:
        seasons.append(current)
    return seasons


def sr_cache_path(season: str, kind: str, *, cache_dir: str | Path | None = None,
                  partial: bool = False) -> Path:
    """
    Where one (season, table) CSV is cached.

    TWO NAMES, BECAUSE A SNAPSHOT IS NOT A TABLE. A season that is OVER has
    one final table, cached at ``sr_<season>_<kind>.csv`` and served forever.
    A season that is UNDERWAY has only a season-to-date snapshot, and that
    goes to ``.partial.csv``.

    The split is the guard, and it replaces a one-path cache that could not
    tell the two apart: fetch 2026-27 in December, let the season turn over,
    and `is_current_season` goes false, `refresh` resolves to False, and a
    twelve-game December snapshot is served from then on as the completed
    season's table -- feeding `prior_season_features` a partial season
    labelled as a whole one, silently and permanently. A reader that cannot
    distinguish a snapshot from a final table will eventually mistake one for
    the other, so the file name distinguishes them and a final read never
    looks at a partial file.

    The cost is two requests, once, in the season after the handoff: the
    first final read of a season that has ended fetches it rather than
    promoting the last snapshot. That is the correct table instead of a
    convenient one.
    """
    root = Path(cache_dir) if cache_dir is not None else SR_CACHE_DIR
    suffix = ".partial.csv" if partial else ".csv"
    return root / f"sr_{season}_{kind}{suffix}"


def _sr_rows_to_csv(body: str, kind: str) -> str:
    """The table's rows as the CSV text ``read_sr_season_csv`` expects."""
    import csv
    import io

    mapping = DATA_STAT_TO_HEADER[kind]
    rows: list[list[str]] = [[*mapping.values(), "Player-additional"]]
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", body, re.S):
        cells = re.findall(r'<t[dh][^>]*data-stat="([^"]*)"[^>]*>(.*?)</t[dh]>', tr, re.S)
        if not cells:
            continue
        by_key: dict[str, str] = {}
        name_html = ""
        for key, html in cells:
            text = re.sub(r"<!--.*?-->", "", html, flags=re.S)
            text = re.sub(r"<[^>]+>", "", text).strip()
            by_key[key] = text
            if key == "name_display":
                name_html = html
        if "name_display" not in by_key:
            continue
        name = by_key["name_display"]
        is_league_average = name.strip().lower() == LEAGUE_AVERAGE_LABEL
        # The player id comes from the anchor inside the name cell, which is
        # the same id the CSV export's trailing '-additional' column carries.
        # The League Average row has no anchor and gets SR's own null
        # sentinel, so the parser's existing trailer handling sees the exact
        # shape its CSV path always had.
        link = re.search(r'href="/players/[a-z]/([a-z0-9]+)\.html"', name_html)
        player_id = link.group(1) if link else (SR_NULL_SENTINEL if is_league_average else "")
        rows.append([by_key.get(key, "") for key in mapping] + [player_id])

    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerows(rows)
    return buffer.getvalue()


def _sr_get(url: str, *, session: Any | None = None) -> str:
    """GET the page, retrying only what is genuinely transient."""
    import time

    import requests

    from src.utils.http_retry import retry_after_seconds

    get = (session or requests).get
    headers = {"User-Agent": os.environ.get("PROPIQ_BBREF_USER_AGENT") or SR_USER_AGENT}
    last = "no attempt made"
    for attempt in range(1, SR_RETRY_ATTEMPTS + 1):
        try:
            response = get(url, headers=headers, timeout=SR_TIMEOUT_SECONDS)
        except Exception as exc:  # noqa: BLE001 — redacted into `last`
            last = f"{type(exc).__name__}: {exc}"
            if attempt < SR_RETRY_ATTEMPTS:
                time.sleep(SR_RETRY_BACKOFF_SECONDS ** attempt)
            continue
        if response.status_code in (200,):
            return response.text
        if response.status_code in (429, 503):
            wait = retry_after_seconds(response.headers, SR_RETRY_BACKOFF_SECONDS ** attempt)
            last = f"HTTP {response.status_code}"
            if attempt < SR_RETRY_ATTEMPTS:
                time.sleep(wait)
            continue
        raise BasketballReferenceError(
            f"GET {url}: HTTP {response.status_code}. A 403 from this host is "
            "its bot wall, and measurement on 2026-11-11 says it reads the "
            "CLIENT IDENTITY, not only the IP: this same container got 403 "
            "with an honest User-Agent and 200 with a browser-shaped one. So "
            "retrying will not change it, and the thing that would change it "
            "is claiming to be a browser -- an operator decision with "
            f"terms-of-service consequences, taken by setting "
            f"$PROPIQ_BBREF_USER_AGENT, not by this module's default."
        )
    raise BasketballReferenceError(
        f"GET {url} failed after {SR_RETRY_ATTEMPTS} attempts: {last}"
    )


def fetch_sr_season_table(
    season: str,
    kind: str,
    *,
    session: Any | None = None,
    cache_dir: str | Path | None = None,
    refresh: bool | None = None,
) -> SrSeasonTable:
    """
    One season table, from the cache or the site, parsed by the one parser.

    THE CACHE IS THE SCHEDULED JOB'S DE-DUPLICATION — for a season that is
    OVER. A prior season's table is final, so the first fetch writes the
    CSV and every later call reads it and never touches the network — one
    request per table per season, ever. A LIVE season's table is not final:
    it changes after every game night, so its cache is a snapshot of the
    last fetch, not the table. ``refresh=None`` (the default) therefore
    refreshes exactly the seasons that are underway (``is_current_season``)
    and serves the cache for the ones that are over; an explicit True or
    False overrides for one call, which is how a single night's snapshot
    is pinned for reproducing an A/B run exactly.

    The fetched rows go through ``read_sr_season_csv`` unchanged, so every
    guard it carries applies to fetched data too: the League Average
    trailer, the multi-team blocks, the NBA-only refusal, and — because the
    caller must state the season — the leakage check downstream. That
    check still refuses to join a season's aggregate onto its own season,
    current or not: fetching season-to-date data does not make it a
    feature.
    """
    _sr_page(kind)  # refuse unknown kinds before any I/O
    season_start_year(season)

    underway = is_current_season(season)
    if refresh is None:
        refresh = underway

    # A snapshot of a live season and the final table of a dead one are
    # different files; see `sr_cache_path`. A final read never reads a
    # partial, which is what stops a mid-season snapshot being promoted to
    # "the season's table" the day the season ends.
    cache = sr_cache_path(season, kind, cache_dir=cache_dir, partial=underway)
    if cache.exists() and not refresh:
        return read_sr_season_csv(cache, season=season, kind=kind)

    _slug, table_id = SR_PAGES[kind]
    html = _sr_get(sr_page_url(season, kind), session=session)
    table = re.search(
        r'<table[^>]*id="' + re.escape(table_id) + r'"[^>]*>(.*?)</table>',
        html, re.S,
    )
    if table is None:
        raise BasketballReferenceError(
            f"The {season} {kind} page has no table id {table_id!r}. SR "
            "renamed the table, and this fetcher needs the new id."
        )
    body = re.search(r"<tbody[^>]*>(.*?)</tbody>", table.group(1), re.S)
    if body is None:
        raise BasketballReferenceError(
            f"The {season} {kind} table has no tbody; nothing to parse."
        )
    csv_text = _sr_rows_to_csv(body.group(1), kind)

    if cache_dir is not None or SR_CACHE_DIR.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(csv_text, encoding="utf-8")

    table_obj = read_sr_season_csv(io.StringIO(csv_text), season=season, kind=kind)
    # The source name for a fetched table is the page, so a bad row names
    # where it came from.
    table_obj = dataclasses.replace(table_obj, source=sr_page_url(season, kind))
    return table_obj
