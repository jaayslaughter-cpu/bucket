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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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


def normalize_player_box_scores(
    df: pd.DataFrame,
    report: SchemaReport | None = None,
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
