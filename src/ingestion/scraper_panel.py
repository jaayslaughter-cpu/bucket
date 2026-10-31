"""
src/ingestion/scraper_panel.py — the multi-source scraper's parquet, as a panel.

WHY THIS EXISTS. ``load_player_panel`` reads ``player_game_logs``, and the one
writer that fills it on the live path is ``src/ingestion/boxscores.py``, which
pulls ``leaguegamelog`` from stats.nba.com. That host does not answer from a
datacenter IP, so the deployment's panel has stayed empty for the whole
2026-27 season to date. The nba-scraper-2026-27 package solves the other half:
run from a residential connection it writes one parquet file per
(game, source) under ``canonical/player_boxscore/``. Nothing read it.

This module is the mapper, and ONLY the mapper. It does no I/O to Postgres and
imports nothing from ``src.db``, so it is testable without a database and the
write stays in the one place that already knows how to do it,
``repository.upsert_player_game_logs``. ``scripts/load_scraper_panel.py`` is
the thin CLI over the two.

FOUR THINGS THE SHAPES DO NOT AGREE ON, each checked against the actual files
rather than assumed, because getting any of them wrong is a silently wrong
panel rather than an error:

  1. THE KEY IS NOT UNIQUE IN THE SOURCE. The scraper writes one row per
     (source, game, player); ``player_game_logs`` is unique on
     (nba_game_id, nba_player_id). With three sources enabled that is three
     rows competing for one key, and the upsert would resolve it by whichever
     happened to be last — a different answer per run. So a precedence is
     declared here and applied to WHOLE ROWS.
  2. PRESEASON. The scraper tags ``season_type`` and keeps "Pre Season" rows;
     ``boxscores.BoxScoreLoadConfig.season_type`` is "Regular Season", so the
     existing panel has none. Letting them in would put October exhibition
     minutes inside every ``_L5`` window.
  3. A DNP IS NOT A ZERO. The scraper marks it ``played=False`` with NaN
     stats, and a genuine zero-minute appearance ``played=True`` with 0.0.
     The panel's columns are nullable precisely so the two stay apart
     (migrations 006 and 009 say why), and a rolling window counts ROWS, so
     including DNPs changes what "last 5 games" measures.
  4. ``is_starter`` IS NOT ``starting_position``. One is a bool, the other is
     G/F/C with a CHECK constraint from migration 008 and a meaning
     ``features/dvp.py`` reads. There is no position in this payload, so the
     column is left unset for ``update_starting_positions`` to fill.

RESEARCH_ONLY. This moves box-score rows. It prices nothing and grades
nothing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.utils.ids import normalise_game_id

logger = logging.getLogger(__name__)

#: Whose row wins when more than one source reported the same player-game.
#: The NBA's own feeds first -- Basketball-Reference is a third-party scrape
#: whose player identity is CROSSWALKED (roster, then fuzzy name, then an
#: overrides CSV), so it is the one source where the join itself can be wrong.
#: This is not a claim that its numbers are wrong: the scraper's own
#: reconciliation found four rebounds off by one and one minutes figure off by
#: two seconds across 578 rows and did NOT establish which side was right.
#: Preferring the league's feed is a tie-break, and the row's ``SOURCE`` column
#: records which one was taken so the choice stays auditable.
SOURCE_PRECEDENCE: tuple[str, ...] = (
    "nba_stats", "nba_live", "basketball_reference",
    # LAST, and deliberately. `ingestion/espn_panel` is the only source
    # a datacenter can reach, which makes it indispensable and not
    # authoritative: it is a public aggregator, while the three above
    # report the NBA's own numbers. Where both have a game, theirs wins.
    # Added 2026-10-31; before that `espn` ranked as an UNKNOWN source,
    # which also sorts last but does so by accident and logs a warning
    # on every load.
    "espn",
)

#: Season types kept by default: what `boxscores.py` already fetches.
DEFAULT_SEASON_TYPES: tuple[str, ...] = ("Regular Season",)

#: The scraper's canonical name -> the panel contract's name. Deliberately
#: NOT every column it writes: `plus_minus`, `game_score`, `oreb`/`dreb`,
#: `fgm`/`ftm` and `fg3a` have no column in `player_game_logs`, and inventing
#: one here would be a schema change disguised as an import.
COLUMN_MAP: dict[str, str] = {
    "game_id": "GAME_ID",
    "player_id_nba": "PLAYER_ID",
    "player_name": "PLAYER_NAME",
    "game_date": "GAME_DATE",
    "season": "SEASON",
    "team_abbr": "TEAM_ABBREVIATION",
    "opponent_abbr": "OPPONENT_ABBREVIATION",
    "pts": "PTS",
    "reb": "REB",
    "ast": "AST",
    "fg3m": "FG3M",
    "fga": "FGA",
    "fta": "FTA",
    "stl": "STL",
    "blk": "BLK",
    "tov": "TOV",
    "pf": "PF",
    "source": "SOURCE",
}

#: Columns that must be present for the mapping to mean anything. A parquet
#: missing one is refused rather than defaulted: `upsert_player_game_logs`
#: uses `.get()` for optional columns, so a typo'd name there would write a
#: column of nulls and report success.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "game_id", "player_id_nba", "player_name", "game_date", "season",
    "home_away", "played", "seconds_played", "source", "season_type",
)


class ScraperPanelError(RuntimeError):
    """The parquet is not there, or is not the shape this mapper maps."""


@dataclass
class LoadReport:
    """What the mapping did, in counts. Printed by the CLI and asserted on."""

    files_read: int = 0
    rows_in: int = 0
    rows_out: int = 0
    dropped_wrong_season_type: int = 0
    dropped_no_player_id: int = 0
    dropped_no_game_id: int = 0
    dropped_dnp: int = 0
    duplicate_rows_resolved: int = 0
    rows_by_source: dict[str, int] = field(default_factory=dict)
    season_types_seen: dict[str, int] = field(default_factory=dict)
    unknown_sources: tuple[str, ...] = ()

    def lines(self) -> list[str]:
        out = [
            f"parquet files read            {self.files_read}",
            f"rows in                       {self.rows_in}",
            f"  wrong season type           -{self.dropped_wrong_season_type}",
            f"  no canonical player id      -{self.dropped_no_player_id}",
            f"  no game id                  -{self.dropped_no_game_id}",
            f"  did not play (DNP)          -{self.dropped_dnp}",
            f"  duplicate player-games      -{self.duplicate_rows_resolved}",
            f"rows out                      {self.rows_out}",
        ]
        if self.season_types_seen:
            out.append("season types seen: " + ", ".join(
                f"{k}={v}" for k, v in sorted(self.season_types_seen.items())))
        if self.rows_by_source:
            out.append("rows kept per source: " + ", ".join(
                f"{k}={v}" for k, v in sorted(self.rows_by_source.items())))
        if self.unknown_sources:
            out.append(
                "SOURCES NOT IN THE PRECEDENCE, ranked last: "
                + ", ".join(self.unknown_sources)
            )
        return out


def resolve_dataset_dir(path: Path) -> Path:
    """
    Find the ``player_boxscore`` parquet directory under whatever was given.

    The scraper's layout is ``{DATA_DIR}/canonical/{dataset}/game_date=.../``,
    so an operator may reasonably point at the data root, the canonical
    directory or the dataset directory itself. Accepting only one of those
    would be a path puzzle; guessing wrong and globbing the parent would read
    `play_by_play` parquet into a box-score mapper.
    """
    path = Path(path)
    if not path.exists():
        raise ScraperPanelError(f"{path} does not exist")
    candidates = [
        path / "canonical" / "player_boxscore",
        path / "player_boxscore",
        path,
    ]
    for candidate in candidates:
        if candidate.is_dir() and candidate.name == "player_boxscore":
            return candidate
    raise ScraperPanelError(
        f"no player_boxscore directory under {path} — looked for "
        f"{[str(c) for c in candidates[:2]]}. Point --canonical-dir at the "
        f"scraper's data directory, its canonical/ directory, or "
        f"canonical/player_boxscore itself."
    )


def read_canonical_boxscores(canonical_dir: Path) -> tuple[pd.DataFrame, int]:
    """Concatenate every parquet file the scraper wrote for this dataset."""
    dataset_dir = resolve_dataset_dir(canonical_dir)
    files = sorted(dataset_dir.rglob("*.parquet"))
    if not files:
        raise ScraperPanelError(
            f"{dataset_dir} holds no parquet files. Run the scraper first: "
            f"`python run_daily.py backfill --start ... --end ...`"
        )
    frames = [pd.read_parquet(f) for f in files]
    frame = pd.concat(frames, ignore_index=True)
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ScraperPanelError(
            f"DATA_NOT_AVAILABLE: {dataset_dir} is missing {missing}. "
            f"Columns present: {sorted(frame.columns)}"
        )
    return frame, len(files)


#: Text that means False. Anything else non-empty means True.
_FALSEY_TEXT = frozenset({"", "false", "f", "0", "no", "n", "none", "nan"})


def _as_bool(values: pd.Series) -> pd.Series:
    """
    A strict boolean read of the ``played`` flag.

    ``Series.astype(bool)`` on the STRING ``"False"`` returns True, because
    every non-empty string is truthy. A DNP that arrived as text would then be
    written to the panel as a player who played and scored nothing — the one
    confusion this module exists to prevent. Parquet carries a real bool, so
    this is defensive rather than a bug anybody has seen; it is here because
    the cost of being wrong is a fabricated zero-point game.
    """
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)

    def one(value: object) -> bool:
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() not in _FALSEY_TEXT
        try:
            if pd.isna(value):
                return False
        except (TypeError, ValueError):
            pass
        return bool(value)

    return values.map(one).astype(bool)


def _precedence_rank(sources: pd.Series, precedence: tuple[str, ...]) -> pd.Series:
    """Rank per the declared order; anything unlisted sorts last, not first."""
    order = {name: i for i, name in enumerate(precedence)}
    return sources.map(lambda s: order.get(s, len(order)))


def to_player_panel(
    frame: pd.DataFrame,
    *,
    include_dnp: bool = False,
    season_types: tuple[str, ...] = DEFAULT_SEASON_TYPES,
    precedence: tuple[str, ...] = SOURCE_PRECEDENCE,
) -> tuple[pd.DataFrame, LoadReport]:
    """
    Map the scraper's canonical box scores onto the panel contract.

    Returns the frame ``repository.upsert_player_game_logs`` expects and a
    report of everything that was dropped on the way, because an import that
    silently discards two thirds of its input looks exactly like one that
    worked.
    """
    report = LoadReport(rows_in=len(frame))
    if frame.empty:
        raise ScraperPanelError("DATA_NOT_AVAILABLE: no rows in the parquet")

    work = frame.copy()
    report.season_types_seen = (
        work["season_type"].fillna("(none)").value_counts().to_dict()
    )

    # [1] Season type. Preseason minutes inside a rolling window are a
    # different game than the one the model is being asked about.
    keep = work["season_type"].isin(season_types)
    report.dropped_wrong_season_type = int((~keep).sum())
    work = work.loc[keep]

    # [2] Identity. A row whose canonical player id did not resolve cannot be
    # keyed, and `player_id_nba` is nullable in the source precisely because
    # the Basketball-Reference crosswalk is allowed to fail rather than guess.
    player_id = work["player_id_nba"].astype("object")
    has_player = player_id.notna() & (
        player_id.map(lambda v: str(v).strip() not in ("", "None", "nan"))
    )
    report.dropped_no_player_id = int((~has_player).sum())
    work = work.loc[has_player]

    game_id = normalise_game_id(work["game_id"]) if not work.empty else work.get("game_id")
    if not work.empty:
        work = work.assign(game_id=game_id)
        has_game = work["game_id"].notna()
        report.dropped_no_game_id = int((~has_game).sum())
        work = work.loc[has_game]

    # [3] DNPs. Excluded by default to match what the panel already holds.
    if not include_dnp and not work.empty:
        played = _as_bool(work["played"])
        report.dropped_dnp = int((~played).sum())
        work = work.loc[played]

    if work.empty:
        raise ScraperPanelError(
            "DATA_NOT_AVAILABLE: no rows survived filtering — "
            + "; ".join(report.lines())
        )

    # [4] One row per player-game, whole rows only. A column-wise merge across
    # sources would produce a line no source ever reported, which is worse
    # than either source's version of it.
    work = work.assign(_rank=_precedence_rank(work["source"], precedence))
    unlisted = sorted(set(work.loc[work["_rank"] == len(precedence), "source"]))
    report.unknown_sources = tuple(unlisted)
    before = len(work)
    work = (
        work.sort_values(["game_id", "player_id_nba", "_rank"], kind="mergesort")
        .drop_duplicates(subset=["game_id", "player_id_nba"], keep="first")
        .drop(columns="_rank")
    )
    report.duplicate_rows_resolved = before - len(work)

    out = work[list(COLUMN_MAP)].rename(columns=COLUMN_MAP).copy()
    out["GAME_DATE"] = pd.to_datetime(work["game_date"], errors="coerce")
    out["IS_HOME"] = work["home_away"].astype(str).str.lower().eq("home")
    # Minutes, from seconds. `upsert_player_game_logs` writes this to a Float
    # column named `minutes`, and `features/minutes_weighted` compares it
    # against prior minute means -- seconds here would be a 60x feature.
    out["MIN"] = pd.to_numeric(work["seconds_played"], errors="coerce") / 60.0

    # NOT zero-filled, on purpose. `pd.to_numeric` keeps NaN as NaN and the
    # upsert casts to Int64, so an unreported stat stays NULL in the table.
    for col in ("PTS", "REB", "AST", "FG3M", "FGA", "FTA", "STL", "BLK", "TOV", "PF"):
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["PLAYER_ID"] = out["PLAYER_ID"].astype(str).str.strip()
    out["SEASON"] = out["SEASON"].astype(str)

    report.rows_out = len(out)
    report.rows_by_source = out["SOURCE"].value_counts().to_dict()
    return out.reset_index(drop=True), report
