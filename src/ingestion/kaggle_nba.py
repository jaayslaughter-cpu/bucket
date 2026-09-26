"""
src/ingestion/kaggle_nba.py — NBA player box scores from a Kaggle export.

WHY THIS EXISTS: the orchestrator's panel comes from player_game_logs,
and the only writer for it pulls from stats.nba.com. That endpoint is not
reachable from every environment, so a second path to the same panel is
the difference between a pipeline that runs and one that reports
success_no_data forever.

SCHEMA IS DISCOVERED, NOT ASSUMED. This module was written without network
access to Kaggle, so its column names could not be verified against the
real export. Rather than hardcode a guess, it inspects whatever frame it
is handed, maps the columns it recognises, and REFUSES with a report of
what it found when it cannot identify the required fields. A loader that
guessed would either crash on a rename or, far worse, silently map the
wrong column onto a stat.

Run ``describe_schema`` first on any new export to see the mapping before
committing to it.

SCOPE: NBA only. Any frame carrying NCAA/college markers is refused.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SOURCE_NAME = "kaggle_nba_box_scores"

# The panel contract the feature builder and upsert_player_game_logs expect.
REQUIRED_PANEL_COLS = ("PLAYER_NAME", "GAME_DATE", "PTS")
PANEL_COLS = (
    "PLAYER_ID", "PLAYER_NAME", "GAME_ID", "GAME_DATE", "SEASON",
    "TEAM_ABBREVIATION", "OPPONENT_ABBREVIATION", "IS_HOME",
    "MIN", "PTS", "REB", "AST", "FG3M", "STL", "BLK", "TOV",
    "FGM", "FGA", "FTM", "FTA", "OREB", "DREB",
)

# Candidate source spellings per target column, lowercased and stripped of
# non-alphanumerics before matching. Order matters: the first match wins,
# so the most specific spelling is listed first. Extend this table rather
# than loosening the matcher — a fuzzy match that lands on the wrong column
# is the failure this whole module is shaped to avoid.
# Some exports carry no single name column. The 2025-26 archive splits it
# into firstName + lastName, so PLAYER_NAME is COMPOSED rather than mapped.
COMPOSITE_ALIASES: dict[str, tuple[tuple[str, ...], ...]] = {
    "PLAYER_NAME": (("firstname", "lastname"), ("first", "last")),
}

# Franchise team ids in this archive are all 1610612xxx. All-Star and
# exhibition rosters (Team LeBron, East, West) use 9xxx ids and are not
# franchises, so they never enter the abbreviation crosswalk.
FRANCHISE_ID_PREFIX = "1610612"

# The crosswalk's own spelling for San Antonio is SAN; every other source in
# this project (NBA.com, BigDataBall, the market lines) uses SAS. Mapping it
# is the difference between a panel that joins and one that silently does not.
ARCHIVE_TO_NBA_TEAM: dict[str, str] = {"SAN": "SAS"}

# City spellings the box scores use that TeamHistories does not. Only exact,
# verified equivalences belong here -- this is a rename, not a guess.
CITY_ALIASES: dict[str, str] = {"LA": "Los Angeles"}

_ABBREVIATION_PATTERN = re.compile(r"^[A-Z]{2,4}$")

COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "PLAYER_ID": ("personid", "playerid", "nbaplayerid", "idplayer"),
    "PLAYER_NAME": (
        "playername", "fullname", "player", "namei", "displayname",
        "firstlast", "playernamei",
    ),
    "GAME_ID": ("gameid", "nbagameid", "idgame"),
    "GAME_DATE": ("gamedate", "date", "gamedateest", "gamedatetimeest"),
    "SEASON": ("season", "seasonyear", "seasonid"),
    "TEAM_ABBREVIATION": (
        "teamabbreviation", "teamtricode", "teamabbr", "teamcode",
        "playerteamabbreviation", "playerteamtricode", "playerteamname",
        "playerteamcity", "team",
    ),
    "OPPONENT_ABBREVIATION": (
        "opponentabbreviation", "opponentteamabbreviation", "opponenttricode",
        "opponentteamname", "opponentteamcity", "opponentabbr", "opponent",
        "oppteam", "matchupopponent",
    ),
    "IS_HOME": ("ishome", "homeaway", "hometeamflag", "ishomegame", "home"),
    "MIN": ("minutes", "min", "numminutes", "mp"),
    "PTS": ("points", "pts"),
    "REB": ("reboundstotal", "totalrebounds", "rebounds", "reb", "trb"),
    "AST": ("assists", "ast"),
    "FG3M": (
        "threepointersmade", "fg3m", "threespointersmade", "3pm", "fg3",
        "threesmade",
    ),
    "STL": ("steals", "stl"),
    "BLK": ("blocks", "blk"),
    "TOV": ("turnovers", "tov", "to", "numturnovers"),
    # Shooting volume — see scoring_efficiency. Optional: absent columns
    # simply mean the efficiency layer abstains.
    "FGM": ("fieldgoalsmade", "fgm"),
    "FGA": ("fieldgoalsattempted", "fga"),
    "FTM": ("freethrowsmade", "ftm"),
    "FTA": ("freethrowsattempted", "fta"),
    "OREB": ("reboundsoffensive", "oreb", "orb", "offensiverebounds"),
    "DREB": ("reboundsdefensive", "dreb", "drb", "defensiverebounds"),
}

# A frame carrying any of these is not an NBA player panel.
NCAA_MARKERS = ("ncaa", "college", "cbb")


class KaggleNbaError(RuntimeError):
    """Raised instead of returning a frame built on guessed columns."""


@dataclass
class SchemaReport:
    """What a discovery pass found. Printable, and safe to log."""

    source_columns: list[str] = field(default_factory=list)
    mapped: dict[str, str] = field(default_factory=dict)
    # Targets built from two or more source columns, e.g. PLAYER_NAME
    # from firstName + lastName.
    composed: dict[str, tuple[str, ...]] = field(default_factory=dict)
    missing_required: list[str] = field(default_factory=list)
    missing_optional: list[str] = field(default_factory=list)
    unmapped_source: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return not self.missing_required

    def as_dict(self) -> dict[str, Any]:
        return {
            "usable": self.usable,
            "mapped": self.mapped,
            "missing_required": self.missing_required,
            "missing_optional": self.missing_optional,
            "composed": {k: list(v) for k, v in self.composed.items()},
            "source_column_count": len(self.source_columns),
            "unmapped_source_columns": self.unmapped_source[:40],
        }


def _norm(name: Any) -> str:
    """Lowercase, strip everything but letters and digits."""
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def describe_schema(df: pd.DataFrame) -> SchemaReport:
    """
    Inspect a frame and report how it maps onto the panel contract.

    Read this before trusting any export. It is the whole point of the
    module: it tells you what was recognised instead of letting a silent
    mismapping reach a model.
    """
    report = SchemaReport(source_columns=[str(c) for c in df.columns])
    normalised = {_norm(c): str(c) for c in df.columns}
    claimed: set[str] = set()

    for target, candidates in COLUMN_ALIASES.items():
        for candidate in candidates:
            source = normalised.get(candidate)
            if source is not None and source not in claimed:
                report.mapped[target] = source
                claimed.add(source)
                break

    for target in PANEL_COLS:
        if target in report.mapped:
            continue
        if target in REQUIRED_PANEL_COLS:
            report.missing_required.append(target)
        else:
            report.missing_optional.append(target)

    report.unmapped_source = [
        str(c) for c in df.columns if str(c) not in claimed
    ]
    # A target with no single source column may still be COMPOSABLE. The
    # 2025-26 archive has firstName and lastName but no full name, and
    # refusing the file over that would reject the only complete player
    # history available.
    lookup = {_norm(c): c for c in df.columns}
    for target, candidate_sets in COMPOSITE_ALIASES.items():
        if target in report.mapped:
            continue
        for parts in candidate_sets:
            resolved = [lookup.get(p) for p in parts]
            if all(resolved):
                report.composed[target] = tuple(resolved)
                if target in report.missing_required:
                    report.missing_required.remove(target)
                if target in report.missing_optional:
                    report.missing_optional.remove(target)
                logger.info(
                    "%s composed from %s", target, " + ".join(resolved),
                )
                break

    return report


def _assert_not_ncaa(df: pd.DataFrame) -> None:
    """Refuse a frame that looks like college basketball."""
    haystack = " ".join(_norm(c) for c in df.columns)
    for marker in NCAA_MARKERS:
        if marker in haystack:
            raise KaggleNbaError(
                f"Refusing this frame: column names contain {marker!r}. "
                "NCAA/college is explicitly out of scope for this project."
            )
    for col in ("league", "LEAGUE", "league_name"):
        if col in df.columns:
            values = {str(v).lower() for v in df[col].dropna().unique()[:50]}
            if values and not any("nba" in v for v in values):
                raise KaggleNbaError(
                    f"Refusing this frame: {col} holds {sorted(values)[:5]}, "
                    "which is not NBA."
                )


def _coerce_is_home(series: pd.Series) -> pd.Series:
    """
    Map a home/away column to a boolean, refusing anything ambiguous.

    Datasets encode this as 1/0, True/False, 'home'/'away', or 'H'/'A'.
    A value that matches none of those becomes NA rather than False —
    defaulting it would silently label every away game a home game, which
    is a wrong feature rather than a missing one.
    """
    def one(value: Any) -> Any:
        if pd.isna(value):
            return pd.NA
        text = str(value).strip().lower()
        if text in ("1", "true", "t", "yes", "y", "home", "h"):
            return True
        if text in ("0", "false", "f", "no", "n", "away", "a", "visitor", "v"):
            return False
        return pd.NA

    return series.map(one).astype("boolean")


def load_team_crosswalk(path: str | Path) -> pd.DataFrame:
    """
    Build an era-aware teamId -> abbreviation table from TeamHistories.csv.

    Franchises move and rename: team 1610612737 is TRI (1948-50), then MIL,
    then STL, then ATL. A flat teamId -> abbreviation map would stamp the
    modern code onto every historical row, so the era columns are kept and
    the lookup is by season.

    Two corrections are applied, and both are join-critical rather than
    cosmetic: the file's abbreviations carry trailing whitespace ("ATL  "),
    and its San Antonio code is SAN where every other source here uses SAS.
    """
    frame = pd.read_csv(path)
    required = {"teamId", "teamAbbrev", "seasonFounded", "seasonActiveTill"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KaggleNbaError(
            f"DATA_NOT_AVAILABLE: {path} is missing {missing}; expected the "
            "archive's TeamHistories.csv"
        )

    out = frame.copy()
    out["teamId"] = out["teamId"].astype("string").str.strip()
    out["abbreviation"] = (
        out["teamAbbrev"].astype("string").str.strip().str.upper()
    )
    out["abbreviation"] = out["abbreviation"].replace(ARCHIVE_TO_NBA_TEAM)
    # All-Star and exhibition rosters are not franchises.
    out = out[out["teamId"].str.startswith(FRANCHISE_ID_PREFIX, na=False)]
    if "league" in out.columns:
        out = out[out["league"].astype("string").str.strip().isin(["NBA", "BAA"])]
    out["season_from"] = pd.to_numeric(out["seasonFounded"], errors="coerce")
    out["season_to"] = pd.to_numeric(out["seasonActiveTill"], errors="coerce")

    # City and team name are kept so a row whose teamId is missing can still
    # be resolved. In the 2018-2026 archive 54,547 rows carry no playerteamId
    # while every one of them names its team, and dropping them would discard
    # 31,559 REGULAR-SEASON player-games for a null in a column the row does
    # not actually need.
    for col, target in (("teamCity", "city"), ("teamName", "name")):
        out[target] = (
            out[col].astype("string").str.strip() if col in out.columns
            else pd.Series(pd.NA, index=out.index, dtype="string")
        )
    out["city"] = out["city"].replace({v: k for k, v in CITY_ALIASES.items()})

    logger.info(
        "team crosswalk: %d franchise-era rows across %d teams",
        len(out), out["teamId"].nunique(),
    )
    return out[
        ["teamId", "abbreviation", "season_from", "season_to", "city", "name"]
    ].reset_index(drop=True)


def _season_start_year(season: Any) -> float:
    try:
        return float(str(season).strip()[:4])
    except (TypeError, ValueError):
        return float("nan")


def _resolve_by_era(
    keys: pd.DataFrame,
    years: pd.Series,
    table: pd.DataFrame,
    key_cols: list[str],
) -> pd.Series:
    """
    Join ``keys`` to ``table``'s franchise eras and take the abbreviation in
    use that season.

    Vectorised on purpose. The row-by-row version this replaces took minutes
    on the 305,614-row archive, which is long enough that ingestion stops
    being something you re-run while checking your work.

    A key outside every recorded era falls back to that franchise's most
    recent one, because a franchise always has a current code. A key the
    table does not contain at all stays NA -- see _map_team_names for why
    that matters.
    """
    left = keys.copy()
    left["_row"] = np.arange(len(left))
    left["_year"] = years.to_numpy()

    merged = left.merge(table, on=key_cols, how="left")
    in_era = (
        (merged["season_from"] <= merged["_year"])
        & (merged["_year"] <= merged["season_to"])
    )
    # Prefer an era that actually contains the season; otherwise the latest.
    merged["_rank"] = np.where(in_era, 0, 1)
    merged = merged.sort_values(
        ["_row", "_rank", "season_to"], ascending=[True, True, False]
    ).drop_duplicates("_row", keep="first")

    out = pd.Series(pd.NA, index=keys.index, dtype="string")
    hit = merged["abbreviation"].notna()
    out.iloc[merged.loc[hit, "_row"].to_numpy()] = (
        merged.loc[hit, "abbreviation"].to_numpy()
    )
    return out


def _map_team_ids(
    ids: pd.Series,
    seasons: pd.Series,
    crosswalk: pd.DataFrame,
) -> pd.Series:
    """Resolve each (teamId, season) to the abbreviation in use that year."""
    keys = pd.DataFrame({"teamId": ids.astype("string").str.strip()}, index=ids.index)
    table = crosswalk[["teamId", "abbreviation", "season_from", "season_to"]].copy()
    table["teamId"] = table["teamId"].astype("string").str.strip()
    return _resolve_by_era(keys, seasons.map(_season_start_year), table, ["teamId"])


def _map_team_names(
    cities: pd.Series,
    names: pd.Series,
    seasons: pd.Series,
    crosswalk: pd.DataFrame,
) -> pd.Series:
    """
    Resolve each (city, name, season) to the abbreviation in use that year.

    A fallback for rows whose teamId is missing, NOT a replacement for the id
    path: ids are unambiguous and names are not, so this runs second and only
    fills gaps.

    A pair the crosswalk does not know stays NA. In this archive the unknown
    pairs are Guangzhou Loong-Lions, Hapoel Jerusalem, Melbourne United and
    South East Melbourne Phoenix -- preseason exhibition opponents that are
    not NBA franchises and must never be handed an NBA abbreviation.
    """
    if not {"city", "name"}.issubset(crosswalk.columns):
        return pd.Series(pd.NA, index=cities.index, dtype="string")

    keys = pd.DataFrame(
        {
            "city": cities.astype("string").str.strip().replace(CITY_ALIASES),
            "name": names.astype("string").str.strip(),
        },
        index=cities.index,
    )
    table = crosswalk.dropna(subset=["city", "name"])[
        ["city", "name", "abbreviation", "season_from", "season_to"]
    ].copy()
    table["city"] = table["city"].astype("string").replace(CITY_ALIASES)
    table["name"] = table["name"].astype("string")
    return _resolve_by_era(
        keys, seasons.map(_season_start_year), table, ["city", "name"]
    )


def assert_abbreviations(series: pd.Series, *, column: str) -> None:
    """
    Refuse nicknames in a column the rest of the project joins on.

    ``playerteamName`` holds "Lakers", not "LAL". It matches the alias table
    and maps cleanly, and then every downstream join — market lines, Elo,
    team pace — silently matches nothing. A loud refusal here is the only
    thing that separates that from working code.
    """
    values = series.dropna().astype(str).str.strip()
    if values.empty:
        return
    bad = sorted({v for v in values.unique() if not _ABBREVIATION_PATTERN.match(v)})
    if bad:
        raise KaggleNbaError(
            f"DATA_NOT_AVAILABLE: {column} contains {bad[:5]}, which are names "
            "rather than NBA abbreviations. Everything downstream joins on codes "
            "like LAL and BOS, so these would match nothing while looking "
            "correct. Pass team_crosswalk=load_team_crosswalk('TeamHistories.csv') "
            "to resolve them from the team ids."
        )


def normalize_player_box_scores(
    df: pd.DataFrame,
    report: SchemaReport | None = None,
    *,
    team_crosswalk: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Map a discovered frame onto the panel contract.

    Raises rather than returning a partial frame when a required column
    could not be identified: a panel missing its date or its target stat
    cannot produce leakage-safe features, and a half-built one that looks
    complete is worse than a clear refusal.
    """
    if df.empty:
        raise KaggleNbaError("DATA_NOT_AVAILABLE: the export is empty")

    _assert_not_ncaa(df)
    report = report or describe_schema(df)

    if not report.usable:
        raise KaggleNbaError(
            "DATA_NOT_AVAILABLE: could not identify required column(s) "
            f"{report.missing_required} in this export. Columns present: "
            f"{report.source_columns[:40]}. Add the right spelling to "
            "COLUMN_ALIASES rather than renaming your data — this module "
            "refuses to guess which column holds which stat."
        )

    out = pd.DataFrame(index=df.index)
    for target, source in report.mapped.items():
        out[target] = df[source]

    for target, parts in report.composed.items():
        joined = (
            df[list(parts)]
            .astype("string")
            .fillna("")
            .agg(" ".join, axis=1)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        out[target] = joined.replace({"": pd.NA})

    out["GAME_DATE"] = pd.to_datetime(out["GAME_DATE"], errors="coerce")
    undated = int(out["GAME_DATE"].isna().sum())
    if undated:
        logger.warning("Dropping %d row(s) with an unparseable GAME_DATE", undated)
        out = out.loc[out["GAME_DATE"].notna()]
    if out.empty:
        raise KaggleNbaError("DATA_NOT_AVAILABLE: no rows survived date parsing")

    for numeric in ("MIN", "PTS", "REB", "AST", "FG3M", "STL", "BLK", "TOV",
                    "FGM", "FGA", "FTM", "FTA", "OREB", "DREB"):
        if numeric in out.columns:
            out[numeric] = pd.to_numeric(out[numeric], errors="coerce")

    if "IS_HOME" in out.columns:
        out["IS_HOME"] = _coerce_is_home(out["IS_HOME"])

    for identifier in ("PLAYER_ID", "GAME_ID"):
        if identifier in out.columns:
            out[identifier] = (
                out[identifier].astype("string").str.strip().replace({"": pd.NA})
            )
            # Kaggle exports often carry these as floats (201939.0), which
            # would never join the panel's integer-derived string ids.
            out[identifier] = out[identifier].str.replace(
                r"\.0$", "", regex=True
            )

    if "SEASON" not in out.columns:
        # Seasons run October to June, so a January game belongs to the
        # season that started the previous calendar year.
        out["SEASON"] = out["GAME_DATE"].apply(
            lambda d: f"{d.year}-{str(d.year + 1)[2:]}" if d.month >= 10
            else f"{d.year - 1}-{str(d.year)[2:]}"
        )
        logger.info("SEASON was absent — derived from GAME_DATE")

    # Team codes: resolve from the archive's team ids when a crosswalk is
    # supplied, then REFUSE anything that is still a nickname.
    team_sources = {
        "TEAM_ABBREVIATION": (
            ("playerteamid", "playerteamId"), ("playerteamcity",), ("playerteamname",),
        ),
        "OPPONENT_ABBREVIATION": (
            ("opponentteamid", "opponentteamId"), ("opponentteamcity",),
            ("opponentteamname",),
        ),
    }
    if team_crosswalk is not None and not team_crosswalk.empty:
        # Align the SOURCE frame to the rows that survived. out started as
        # pd.DataFrame(index=df.index) and then dropped unparseable dates
        # WITHOUT resetting the index, so out.index is a subset of df's labels.
        # Reading team ids straight from df while pairing them with
        # out["SEASON"] raised "Length of values (3) does not match length of
        # index (4)" -- the documented date-drop path crashed outright whenever
        # a crosswalk was supplied.
        src = df.loc[out.index]
        lookup = {_norm(c): c for c in src.columns}

        def _find(candidates: tuple[str, ...]) -> str | None:
            return next(
                (lookup.get(_norm(c)) for c in candidates if lookup.get(_norm(c))), None
            )

        for target, (id_cands, city_cands, name_cands) in team_sources.items():
            id_source = _find(id_cands)
            resolved = pd.Series(pd.NA, index=out.index, dtype="string")
            if id_source is not None:
                ids = (
                    src[id_source].astype("string").str.strip()
                    .str.replace(r"\.0$", "", regex=True)
                )
                resolved = _map_team_ids(ids, out["SEASON"], team_crosswalk)
            from_id = int(resolved.notna().sum())

            # Fill the gaps from the team's name. Ids are unambiguous, so they
            # win; a row with no id still knows who it played for.
            city_source, name_source = _find(city_cands), _find(name_cands)
            if city_source is not None and name_source is not None:
                gaps = resolved.isna()
                if gaps.any():
                    by_name = _map_team_names(
                        src.loc[gaps, city_source], src.loc[gaps, name_source],
                        out.loc[gaps, "SEASON"], team_crosswalk,
                    )
                    resolved.loc[gaps] = by_name
            matched = int(resolved.notna().sum())
            logger.info(
                "%s resolved for %d of %d rows (%.1f%%) — %d from the team id, "
                "%d from the team name. The %d still unresolved carry no "
                "abbreviation rather than a guessed one.",
                target, matched, len(out), 100.0 * matched / max(len(out), 1),
                from_id, matched - from_id, len(out) - matched,
            )
            out[target] = resolved

    for target in team_sources:
        if target in out.columns:
            assert_abbreviations(out[target], column=target)

    # Game type and DNP reason are carried, not dropped: preseason must not
    # be pooled into regular-season rolling features, and the comment column
    # is where a did-not-play is recorded.
    passthrough = {"GAME_TYPE": ("gametype",), "DNP_COMMENT": ("comment",)}
    lookup = {_norm(c): c for c in df.columns}
    for target, candidates in passthrough.items():
        source = next((lookup.get(c) for c in candidates if lookup.get(c)), None)
        if source is not None:
            out[target] = df[source]
    if "GAME_TYPE" in out.columns:
        kinds = out["GAME_TYPE"].astype("string").str.strip().str.lower()
        out["IS_REGULAR_SEASON"] = kinds.eq("regular season")
        counts = kinds.value_counts().to_dict()
        if counts:
            logger.info("game types present: %s", counts)
        if not out["IS_REGULAR_SEASON"].all():
            logger.warning(
                "Panel contains %d non-regular-season rows. Filter on "
                "IS_REGULAR_SEASON before building rolling features — preseason "
                "minutes and rotations do not describe the same competition.",
                int((~out["IS_REGULAR_SEASON"]).sum()),
            )

    if "PLAYER_ID" not in out.columns:
        logger.warning(
            "No player id column found. Rows will key on PLAYER_NAME, which "
            "cannot distinguish two players sharing a name."
        )

    before = len(out)
    subset = [c for c in ("PLAYER_ID", "PLAYER_NAME", "GAME_ID", "GAME_DATE")
              if c in out.columns]
    out = out.drop_duplicates(subset=subset, keep="first")
    if len(out) != before:
        logger.warning("Dropped %d duplicate player-game row(s)", before - len(out))

    out["SOURCE"] = SOURCE_NAME
    logger.info(
        "Kaggle NBA panel: %d rows, %d columns mapped (%s missing and left absent)",
        len(out), len(report.mapped), report.missing_optional or "none",
    )
    return out.sort_values(["PLAYER_NAME", "GAME_DATE"]).reset_index(drop=True)


def load_local_export(path: str | Path) -> pd.DataFrame:
    """Read a CSV/Parquet export already on disk. No network required."""
    target = Path(path)
    if not target.exists():
        raise KaggleNbaError(f"DATA_NOT_AVAILABLE: no export at {target}")
    if target.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(target)
    return pd.read_csv(target, low_memory=False)


def load_from_kagglehub(
    dataset: str = "eoinamoore/historical-nba-data-and-player-box-scores",
    file_path: str = "",
) -> pd.DataFrame:
    """
    Load an export via kagglehub.

    kagglehub is an OPTIONAL dependency, imported here rather than at module
    scope so the rest of the pipeline neither requires it nor breaks without
    it. Kaggle also needs credentials (KAGGLE_USERNAME / KAGGLE_KEY, or
    ~/.kaggle/kaggle.json) — the error says so rather than failing opaquely.
    """
    # Argument validation first: a missing file_path is a programming error
    # either way, and reporting it only when kagglehub happens to be
    # installed would make the same call fail two different ways.
    if not file_path:
        raise KaggleNbaError(
            "file_path is required — a Kaggle dataset holds several files and "
            "this module will not pick one for you. List the dataset's files "
            "on its Kaggle page, then pass the player box-score file."
        )

    try:
        import kagglehub
        from kagglehub import KaggleDatasetAdapter
    except ImportError as exc:
        raise KaggleNbaError(
            "kagglehub is not installed. It is an optional dependency: "
            "pip install 'kagglehub[pandas-datasets]'. Alternatively download "
            "the export manually and use load_local_export(), which needs no "
            "network access at all."
        ) from exc

    try:
        return kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS, dataset, file_path
        )
    except Exception as exc:  # noqa: BLE001 — surface the real cause
        raise KaggleNbaError(
            f"Could not load {dataset}:{file_path} — {exc}. Check Kaggle "
            "credentials (KAGGLE_USERNAME/KAGGLE_KEY or ~/.kaggle/kaggle.json) "
            "and that this environment can reach kaggle.com."
        ) from exc
