"""
src/features/absences.py — the absence feature layer.

WHY A SEPARATE LAYER. ``src.ingestion.inactive_players.attach_absence_features``
needs two frames: the panel and the fetched inactive lists. Every additive layer
in ``build_feature_matrix`` has the signature ``f(df) -> df``, so this wrapper
holds the second one: it loads the cached inactive parquet itself and abstains
when there is none.

WHAT ABSTAINING MEANS HERE. The columns are always added, and they are pd.NA
with ``BBS_INACTIVE_SOURCE = DATA_NOT_AVAILABLE`` until the pull has run. That is
deliberate: ``teammate_cascade`` reads them, and a MISSING column and a column
of nulls send it down different paths — the first says "no absence input exists",
the second says "the input exists and these games were not covered". Both
abstain, and the difference is worth keeping visible.

ORDERING. This must run BEFORE ``teammate_cascade``, which consumes
``BBS_TEAMMATES_OUT``. ``_configured_layers`` in the builder lists it first.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

from src.settlement.boxscore_fetcher import normalize_game_id

logger = logging.getLogger(__name__)

# Where fetch-inactives writes, and the per-season caches beside it.
DEFAULT_CACHE_PATHS = (
    Path("data/external/inactive_players/inactive_players.parquet"),
)
CACHE_GLOB_DIR = Path("data/external/inactive_players")

# Relocating the cache is an environment setting, not a code edit: a pull done
# on one machine and a feature build on another need not share a layout, and a
# test needs a directory that is NOT the developer's data tree. Honoured only
# when no explicit root is passed.
ENV_CACHE_DIR = "PROPIQ_INACTIVE_CACHE_DIR"


def _cache_dir(root: Path | None = None) -> Path:
    if root is not None:
        return Path(root)
    configured = (os.environ.get(ENV_CACHE_DIR) or "").strip()
    return Path(configured) if configured else CACHE_GLOB_DIR


ABSENCE_COLUMNS = (
    "BBS_TEAMMATES_OUT",
    "BBS_VACATED_USAGE",
    "BBS_VACATED_USAGE_UNKNOWN",
)


def load_cached_absences(root: Path | None = None) -> pd.DataFrame | None:
    """Every cached inactive list found on disk, concatenated, or None.

    Both shapes the CLI can leave behind are read: the single
    ``inactive_players.parquet`` the command writes, and the per-season files
    ``save_inactive_players`` produces. Ids are re-cast to text on the way in —
    a padded game id read back as an integer joins to nothing, which is the
    defect this pipeline has already hit twice.
    """
    directory = _cache_dir(root)
    paths = sorted(directory.glob("*.parquet")) if directory.exists() else []
    if not paths:
        return None

    frames: list[pd.DataFrame] = []
    for path in paths:
        try:
            frames.append(pd.read_parquet(path))
        except Exception as exc:  # noqa: BLE001 — one bad file must not lose the rest
            logger.warning("absences: could not read %s: %s", path, exc)
    if not frames:
        return None

    combined = pd.concat(frames, ignore_index=True)
    for column in ("GAME_ID", "PLAYER_ID", "TEAM_ID"):
        if column in combined.columns:
            combined[column] = combined[column].astype("string")
    # Padded BEFORE the dedupe, not after. Parquet written by one writer and a
    # file whose ids were read back as integers give '0021700548' and
    # '21700548' for the same game: distinct here, so both survive the dedupe,
    # and then both normalise to one key downstream and duplicate every panel
    # row for that game in the merge. Same id, same row, once.
    if "GAME_ID" in combined.columns:
        combined["GAME_ID"] = (
            combined["GAME_ID"].astype(str).map(normalize_game_id).astype("string")
        )
    if {"GAME_ID", "PLAYER_ID"}.issubset(combined.columns):
        combined = combined.drop_duplicates(subset=["GAME_ID", "PLAYER_ID"])
    logger.info(
        "absences: loaded %d inactive rows from %d cache file(s)",
        len(combined), len(paths),
    )
    return combined.reset_index(drop=True)


def attach_absence_features_layer(
    df: pd.DataFrame, *, cache_root: Path | None = None
) -> pd.DataFrame:
    """Additive layer: absence counts and vacated usage, or a named abstention."""
    inactives = load_cached_absences(cache_root)
    if inactives is None or inactives.empty:
        out = df.copy()
        for column in ABSENCE_COLUMNS:
            out[column] = pd.NA
        out["BBS_INACTIVE_SOURCE"] = "DATA_NOT_AVAILABLE"
        logger.info(
            "absences: no inactive list cached — columns added as null. Run "
            "`nba_model_cli.py fetch-inactives` where stats.nba.com is reachable."
        )
        return out

    from src.ingestion.inactive_players import (
        InactiveListError,
        attach_absence_features,
    )

    try:
        return attach_absence_features(df, inactives)
    except InactiveListError as exc:
        # A layer that cannot run says so and leaves the frame usable, rather
        # than taking down a feature build that has nothing else wrong with it.
        logger.warning("absences: %s — columns added as null", exc)
        out = df.copy()
        for column in ABSENCE_COLUMNS:
            out[column] = pd.NA
        out["BBS_INACTIVE_SOURCE"] = "DATA_NOT_AVAILABLE"
        return out
