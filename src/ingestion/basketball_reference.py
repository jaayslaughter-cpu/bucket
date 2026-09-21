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
"""

from __future__ import annotations

import logging
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
    "play_by_play": "SR_PBP",
    "adjusted_shooting": "SR_ADJ",
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


def _find_header_row(raw: pd.DataFrame) -> int:
    """SR exports may carry one or two header rows; the real one starts 'Rk'."""
    for idx in range(min(len(raw), 5)):
        first = str(raw.iat[idx, 0]).strip()
        if first.lower() == "rk":
            return idx
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
    # Ordered most-specific first. Adjusted shooting is checked before
    # per-game because it also carries FG_PCT.
    ("adjusted_shooting", ("TS_PLUS", "EFG_PLUS")),
    ("play_by_play", ("PG_PCT", "ONCOURT")),
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

    header_idx = _find_header_row(raw)
    group_row = list(raw.iloc[header_idx - 1]) if header_idx > 0 else None
    columns = _flatten_header(group_row, list(raw.iloc[header_idx]))

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
