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

# THE AVAILABILITY FALLBACK, and why it is OFF by default.
#
# `src/ingestion/availability_bridge.py` can turn archived official pregame
# availability reports into exactly the frame this layer feeds
# `attach_absence_features`, with (game, team) coverage so a club that did not
# file reads pd.NA rather than 0. That closes the gap where the archive filled
# up nightly while every BBS_ column stayed null.
#
# It is nonetheless default OFF, and the reason is TRAIN/SERVE SKEW rather
# than any doubt about the bridge. The scoring artifacts on the volume were
# fit on a panel whose BBS_TEAMMATES_OUT and BBS_VACATED_USAGE were null on
# every row, because nothing had ever written them. Switching the fallback on
# would start handing a live slate real counts for a column the model only
# ever saw as missing -- a distribution it was never fitted against, at
# serving time, with no measurement. Per AGENTS.md that is an A/B question:
# enable it, rebuild the panel, refit, and let `scripts/feature_ab.py` say
# whether it earns its place.
#
# So the switch exists, the path is tested, and turning it on is a deliberate
# act with a retrain behind it.
ENV_AVAILABILITY_FALLBACK = "PROPIQ_AVAILABILITY_ABSENCES"

#: How far back a report may have been published and still describe the slate
#: being built. Mirrors the archive's own default.
AVAILABILITY_MAX_AGE_HOURS = 6.0


def _availability_fallback_enabled() -> bool:
    raw = (os.environ.get(ENV_AVAILABILITY_FALLBACK) or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def load_availability_absences(panel: pd.DataFrame):
    """Absences and their coverage from the archived availability reports.

    Returns ``(inactives, covered_team_games)`` or ``None`` when the archive
    holds nothing usable. Deliberately quiet about every ordinary reason it
    might find nothing -- no archive directory, no report inside the age
    window, no club resolved -- because this is a fallback and a missing
    optional input is not an error.

    The cutoff is NOW, so only reports published and collected before this
    moment are read; `InputArchive.as_of(mode="system")` enforces both. That
    makes it right for the slate being built and silent on history, which the
    archive does not yet hold.
    """
    try:
        from datetime import datetime

        from src.ingestion.availability_bridge import availability_absences
        from src.ingestion.team_identity import infer_team_abbreviations
        from src.ingestion.model_input_archive import InputArchive
        from src.utils.timezones import UTC
        from src.utils.volume import resolve_state_root
    except ImportError as exc:  # pragma: no cover - optional path
        logger.info("absences: availability fallback unavailable (%s)", exc)
        return None

    root = Path(resolve_state_root()) / "model_inputs"
    if not (root / "snapshots" / "availability").exists():
        return None
    archive = InputArchive(root)
    try:
        snapshots = archive.as_of(
            "availability", datetime.now(UTC),
            mode="system", max_age_hours=AVAILABILITY_MAX_AGE_HOURS,
        )
        # IDENTITY FROM THE WHOLE ARCHIVE, ABSENCES FROM THE WINDOW.
        #
        # The bug this exists to fix: inferring the club map from the as-of
        # window alone means inferring it from a single report, because
        # `as_of` selects the latest full report and the window is six hours.
        # `team_identity` separates two clubs by seeing them against DIFFERENT
        # opponents, so one report resolves 11 of 21 clubs and the next run
        # starts over -- most clubs would abstain forever.
        #
        # Reading every archived report for the map is leakage-safe because an
        # abbreviation is REFERENCE data, not a fact about a game: that
        # "Chicago Bulls" is CHI cannot tell anyone what happens tonight. The
        # ABSENCES still come only from `as_of`, which requires each report to
        # have been published AND collected before the cutoff.
        identity_rows = [
            row
            for snapshot in archive.all_snapshots("availability")
            for row in (snapshot.get("rows") or ())
        ]
    except Exception as exc:  # noqa: BLE001 - a bad archive must not fail the build
        logger.warning("absences: could not read the availability archive: %s", exc)
        return None
    if not snapshots:
        return None

    identity = infer_team_abbreviations(identity_rows) if identity_rows else None
    result = availability_absences(snapshots, panel, identity=identity)
    if not result.usable:
        logger.info(
            "absences: availability archive held no resolvable absence (%s)",
            result.report.get("coverage_counts"),
        )
        return None
    logger.info(
        "absences: availability fallback supplied %d absence(s) over %d covered "
        "(game, team) pair(s)",
        len(result.inactives), len(result.covered_team_games),
    )
    return result.inactives, result.covered_team_games


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
    """Additive layer: absence counts and vacated usage, or a named abstention.

    PRECEDENCE. The cached official INACTIVE LIST wins when it is present,
    because it records who actually did not play and is settled after the
    fact. The archived pregame AVAILABILITY REPORT is the fallback: it is the
    only one of the two that exists before tip-off, which is when a slate is
    built, and it is correspondingly a forecast rather than a record. It is
    read only when the cache has nothing and only when
    PROPIQ_AVAILABILITY_ABSENCES is set -- see the note on train/serve skew
    beside that constant.
    """
    inactives = load_cached_absences(cache_root)
    covered_team_games = None

    if (inactives is None or inactives.empty) and _availability_fallback_enabled():
        fallback = load_availability_absences(df)
        if fallback is not None:
            inactives, covered_team_games = fallback

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
        return attach_absence_features(
            df, inactives, covered_team_games=covered_team_games
        )
    except InactiveListError as exc:
        # A layer that cannot run says so and leaves the frame usable, rather
        # than taking down a feature build that has nothing else wrong with it.
        logger.warning("absences: %s — columns added as null", exc)
        out = df.copy()
        for column in ABSENCE_COLUMNS:
            out[column] = pd.NA
        out["BBS_INACTIVE_SOURCE"] = "DATA_NOT_AVAILABLE"
        return out
