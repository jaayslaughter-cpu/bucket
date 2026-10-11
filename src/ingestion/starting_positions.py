"""
src/ingestion/starting_positions.py — the live writer for STARTING_POSITION.

WHY THIS EXISTS. ``src/features/dvp.py`` is wired into ``build_feature_matrix``
and abstains on every row of a LIVE panel, because it needs to know which
position each player started at and nothing in the live tree produced that.
The Kaggle archive carries it; ``player_game_logs`` did not, so defence-versus-
position was a research column measured on history and inert in production.
That gap, and the measurement that made closing it worth doing, are
docs/fouls_and_dvp.md sections 3a and 5.

THE SOURCE is the NBA's own traditional box score, whose per-player table
carries the position a player STARTED at and leaves it blank for everyone who
came off the bench. Two endpoint versions expose it under different names:

    boxscoretraditionalv3  gameId, teamId, personId, position
    boxscoretraditionalv2  GAME_ID, TEAM_ID, PLAYER_ID, START_POSITION

Both column lists are taken from the installed library's own ``expected_data``
(``nba_api/stats/endpoints/boxscoretraditionalv{2,3}.py``) rather than from
memory. v2 is the one every published example uses and it is the wrong
default: the library's own module docstring says v2 "is deprecated" and its
"Data is no longer being published ... as of the 2025-26 NBA season", which is
a season in this panel. So v3 is preferred and v2 is the fallback.

WHAT IT CANNOT DO HERE, AND THE ONE THING THAT IS THEREFORE UNVERIFIED.
stats.nba.com is denied at this environment's proxy (``CONNECT tunnel failed,
response 403``), the same denial ``src/ingestion/boxscores.py`` and
``src/ingestion/inactive_players.py`` document. So everything below is written
against an injectable ``fetch`` and tested offline, and the live pull has to
run where nba.com is reachable.

That leaves ONE claim this module cannot confirm from here: that v3's
``position`` is the STARTING position (blank for the bench, as v2's
``START_POSITION`` plainly is) and not the player's LISTED position (filled in
for all twelve or thirteen players who dressed). The two are different
quantities. If ``position`` were the listed one, DvP would silently start
bucketing every player who appeared instead of the five who started, the
archive-trained meaning of ``POS_BUCKET`` would change underneath it, and
nothing would look wrong.

So the uncertainty is turned into a REFUSAL rather than an assumption.
``check_starting_position_semantics`` counts filled positions per team-game and
requires five. A listed-position payload yields eleven or more and is rejected
by name, with the count, on the first game pulled. Blank means bench; blank is
not the same as absent, and neither is guessed.

WHY THIS IS NOT THE LEAKAGE CAVEAT ``inactive_players.py`` CARRIES, even
though both read a pregame fact out of a post-game box score. That module's
caveat is real: tonight's inactive list is used for tonight's game, so a late
scratch is information a decision made at line-set time could not have had.
Nothing here is used that way.

  * the bucket a PREDICTED row is given comes from ``assign_position_buckets``,
    the expanding modal of that player's PRIOR starts, shifted. Tonight's
    designation is not an input to tonight's row at all.
  * the bucket a COMPLETED game is counted under (``_aggregation_bucket``) is
    the observed one, and the allowed-against-bucket averages built from it
    are rolled and shifted before they reach a feature, so a game is only ever
    counted into a window that closes before the row reading it.

A late lineup change therefore costs this layer accuracy about a past game's
label, not foresight about a future one. That is a data-quality question, not
a leakage one, and ``docs/fouls_and_dvp.md`` section 2b records the auditor's
own run over the layer.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import pandas as pd

from src.features.dvp import normalise_bucket
from src.settlement.boxscore_fetcher import normalize_game_id

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/external/starting_positions")

# The normalised schema every version is mapped onto. TEAM_ID is carried for
# the five-starters gate, which is a per-team-game count and cannot be done
# without it.
STARTING_POSITION_COLUMNS = ("GAME_ID", "TEAM_ID", "PLAYER_ID", "STARTING_POSITION")

# v3 first: v2 is documented by the library as no longer published from
# 2025-26 onwards.
ENDPOINT_PREFERENCE = ("boxscoretraditionalv3", "boxscoretraditionalv2")

# The data set inside each response that carries per-player rows. Same name in
# both versions, which is why the version is detected from COLUMNS below and
# not from this.
PLAYER_STATS_DATASET = "PlayerStats"

_V3_RENAMES = {
    "gameId": "GAME_ID",
    "teamId": "TEAM_ID",
    "personId": "PLAYER_ID",
    "position": "STARTING_POSITION",
}
_V2_RENAMES = {
    "GAME_ID": "GAME_ID",
    "TEAM_ID": "TEAM_ID",
    "PLAYER_ID": "PLAYER_ID",
    "START_POSITION": "STARTING_POSITION",
}

# Five players start a basketball game. This is the discriminator between a
# starting-position payload and a listed-position one, and it is not a
# heuristic: it is a rule of the sport.
STARTERS_PER_TEAM = 5

DEFAULT_PAUSE_SECONDS = 0.6


class StartingPositionError(RuntimeError):
    """Raised when starting positions cannot be obtained from what was supplied."""


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------


def _frame_from_dataset(dataset: Mapping[str, Any], game_id: str) -> pd.DataFrame:
    """Build a frame from the ``{"headers": [...], "data": [[...]]}`` shape."""
    headers = list(dataset.get("headers") or [])
    rows = list(dataset.get("data") or [])
    if not headers:
        raise StartingPositionError(
            f"DATA_NOT_AVAILABLE: game {game_id} {PLAYER_STATS_DATASET} carries no "
            "headers, so its columns cannot be named. The response shape changed."
        )
    return pd.DataFrame(rows, columns=headers)


def _detect_version(columns: Iterable[str], game_id: str) -> str:
    """Which endpoint version a PlayerStats table came from.

    Raises on anything else, so a changed response is a named refusal rather
    than a silently empty or mis-mapped frame.
    """
    present = set(columns)
    if {"personId", "position"} <= present:
        return "v3"
    if {"PLAYER_ID", "START_POSITION"} <= present:
        return "v2"
    raise StartingPositionError(
        f"DATA_NOT_AVAILABLE: game {game_id} {PLAYER_STATS_DATASET} columns "
        f"{sorted(present)} carry neither v3's (personId, position) nor v2's "
        f"(PLAYER_ID, START_POSITION)."
    )


def parse_starting_positions(
    data_sets: Mapping[str, Any], game_id: str | int
) -> pd.DataFrame:
    """
    Normalise one game's ``PlayerStats`` table, v2 or v3, to four columns.

    ``STARTING_POSITION`` is G, F, C or NA. The three ways a value becomes NA
    are deliberately distinct in what they mean and identical in what they
    produce, because a bucket guessed from any of them would be a fabrication:

      blank      the player did not start. The common case, ~8 of 13 rows.
      unknown    a spelling ``normalise_bucket`` does not recognise. Counted
                 and logged, never mapped to a nearest guess.
      absent     the column is missing entirely -> this raises instead, in
                 ``_detect_version``, because a whole game of bench players is
                 not a thing and would otherwise look like one.
    """
    gid = normalize_game_id(game_id)
    dataset = data_sets.get(PLAYER_STATS_DATASET)
    if dataset is None:
        raise StartingPositionError(
            f"DATA_NOT_AVAILABLE: game {gid} response has no "
            f"{PLAYER_STATS_DATASET} data set; it carries "
            f"{sorted(data_sets)}."
        )

    frame = _frame_from_dataset(dataset, gid)
    version = _detect_version(frame.columns, gid)
    renames = _V3_RENAMES if version == "v3" else _V2_RENAMES
    frame = frame.rename(columns=renames)

    for column in ("GAME_ID", "TEAM_ID", "PLAYER_ID"):
        if column not in frame.columns:
            # GAME_ID is absent from some v3 payloads at the row level because
            # the id is the request parameter. Filling it from the request is
            # safe; inventing a team or player id is not.
            if column == "GAME_ID":
                frame[column] = gid
            else:
                raise StartingPositionError(
                    f"DATA_NOT_AVAILABLE: game {gid} {version} {PLAYER_STATS_DATASET} "
                    f"has no {column}, so its rows cannot be keyed."
                )

    out = pd.DataFrame({
        "GAME_ID": frame["GAME_ID"].map(lambda v: normalize_game_id(v) if pd.notna(v) else pd.NA).astype("string"),
        "TEAM_ID": frame["TEAM_ID"].astype("string"),
        "PLAYER_ID": frame["PLAYER_ID"].astype("string"),
    })

    raw = frame["STARTING_POSITION"]
    out["STARTING_POSITION"] = pd.Series(
        [normalise_bucket(v) for v in raw], index=frame.index, dtype="string"
    )

    # A value that was present and non-blank but did not normalise is the only
    # case worth a warning: the other two NAs are expected.
    unknown = [
        str(v) for v, mapped in zip(raw, out["STARTING_POSITION"])
        if pd.isna(mapped) and str(v).strip() not in {"", "nan", "None", "<NA>"}
    ]
    if unknown:
        logger.warning(
            "starting_positions: game %s had %d unrecognised position value(s) "
            "%s; left null rather than mapped to a nearest bucket.",
            gid, len(unknown), sorted(set(unknown)),
        )

    return out[list(STARTING_POSITION_COLUMNS)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# the gate that settles what the column MEANS
# ---------------------------------------------------------------------------


def check_starting_position_semantics(
    frame: pd.DataFrame, *, expected_starters: int = STARTERS_PER_TEAM
) -> dict[str, Any]:
    """
    Confirm the parsed column is a STARTING position, not a LISTED one.

    Five players start, so every team-game must carry exactly five filled
    positions. A listed-position payload fills one for everyone who dressed
    and lands at eleven or more. Both parse cleanly, both look plausible, and
    only one of them is the quantity ``src/features/dvp.py`` was built and
    measured against.

    Returns a report. Raises ``StartingPositionError`` naming the counts when
    any team-game is off, because the wrong quantity written to
    ``player_game_logs`` would change what POS_BUCKET means for every row
    downstream of it and nothing would look wrong.
    """
    if frame.empty:
        raise StartingPositionError(
            "DATA_NOT_AVAILABLE: no rows to check, so the column's meaning is "
            "unestablished. An empty frame is not a passing gate."
        )
    missing = [c for c in ("GAME_ID", "TEAM_ID", "STARTING_POSITION") if c not in frame.columns]
    if missing:
        raise StartingPositionError(
            f"DATA_NOT_AVAILABLE: cannot check semantics without {missing}."
        )

    filled = frame["STARTING_POSITION"].notna()
    per_team = filled.groupby([frame["GAME_ID"], frame["TEAM_ID"]]).sum().astype(int)
    offenders = per_team[per_team != expected_starters]

    report = {
        "rows": int(len(frame)),
        "team_games": int(len(per_team)),
        "filled": int(filled.sum()),
        "filled_rate": float(filled.mean()),
        "starters_per_team_game_min": int(per_team.min()),
        "starters_per_team_game_max": int(per_team.max()),
        "offending_team_games": int(len(offenders)),
    }

    if len(offenders):
        sample = [
            f"game {g} team {t}: {n}" for (g, t), n in offenders.head(5).items()
        ]
        likely_listed = int(per_team.median()) > expected_starters
        raise StartingPositionError(
            f"DATA_NOT_AVAILABLE: {len(offenders)} of {len(per_team)} team-games "
            f"do not carry exactly {expected_starters} filled positions "
            f"(min {report['starters_per_team_game_min']}, max "
            f"{report['starters_per_team_game_max']}). "
            + (
                "The median is above five, so this payload's position column is "
                "most likely the player's LISTED position rather than the one he "
                "STARTED at — a different quantity from the archive's, and not "
                "what src/features/dvp.py was measured against. "
                if likely_listed else
                "A team-game short of five starters means rows are missing, not "
                "that fewer players started. "
            )
            + "Examples: " + "; ".join(sample)
        )

    logger.info(
        "starting_positions: %d team-game(s), exactly %d filled position(s) each, "
        "%.1f%% of rows filled.",
        report["team_games"], expected_starters, 100 * report["filled_rate"],
    )
    return report


# ---------------------------------------------------------------------------
# fetching
# ---------------------------------------------------------------------------


def _nba_api_fetch(game_id: str, *, endpoint: str, timeout: int = 30) -> dict[str, Any]:
    """Fetch one game's data sets via nba_api. Imported lazily on purpose.

    nba_api lives in the optional ``stats`` extra, so importing it at module
    scope would make this module unimportable — and its tests uncollectable —
    wherever the extra is not installed.
    """
    try:
        if endpoint == "boxscoretraditionalv3":
            from nba_api.stats.endpoints import boxscoretraditionalv3 as module

            call = module.BoxScoreTraditionalV3
        else:
            from nba_api.stats.endpoints import boxscoretraditionalv2 as module  # type: ignore[no-redef]

            call = module.BoxScoreTraditionalV2  # type: ignore[assignment]
    except ImportError as exc:
        raise StartingPositionError(
            "DATA_NOT_AVAILABLE: nba_api is not installed. It is declared in the "
            "optional 'stats' extra — install with `pip install -e '.[stats]'`."
        ) from exc

    box = call(game_id=normalize_game_id(game_id), timeout=timeout)
    return box.nba_response.get_data_sets(endpoint)


def fetch_starting_positions(
    game_id: str | int,
    *,
    fetch: Callable[[str, str], Mapping[str, Any]] | None = None,
    endpoints: Sequence[str] = ENDPOINT_PREFERENCE,
) -> pd.DataFrame:
    """
    One game's starting positions, trying each endpoint version in order.

    ``fetch(game_id, endpoint)`` returns the data-set mapping — the shape
    nba_api's ``get_data_sets`` produces. Injected so tests need neither the
    network nor nba_api.
    """
    gid = normalize_game_id(game_id)
    getter = fetch or (lambda g, e: _nba_api_fetch(g, endpoint=e))
    errors: list[str] = []
    for endpoint in endpoints:
        try:
            return parse_starting_positions(getter(gid, endpoint), gid)
        except StartingPositionError as exc:
            errors.append(f"{endpoint}: {exc}")
        except Exception as exc:  # noqa: BLE001 — recorded, then the next version tried
            errors.append(f"{endpoint}: {type(exc).__name__}: {exc}")
    raise StartingPositionError(
        f"DATA_NOT_AVAILABLE: game {gid} starting positions unavailable from "
        f"{list(endpoints)} — " + " | ".join(errors)
    )


def fetch_many_starting_positions(
    game_ids: Iterable[str | int],
    *,
    fetch: Callable[[str, str], Mapping[str, Any]] | None = None,
    pause_seconds: float = DEFAULT_PAUSE_SECONDS,
    progress_every: int = 100,
    stop_on_error: bool = False,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """
    Starting positions for many games. Returns ``(frame, failures)``.

    Failures are RETURNED, not swallowed: a caller that asked for 1,230 games
    and got 1,180 has to be able to tell, or a partial pull silently becomes a
    season where fifty games had no starters.
    """
    ids = list(dict.fromkeys(normalize_game_id(g) for g in game_ids))
    frames: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []

    for i, gid in enumerate(ids, 1):
        try:
            frames.append(fetch_starting_positions(gid, fetch=fetch))
        except StartingPositionError as exc:
            if stop_on_error:
                raise
            failures.append({"game_id": gid, "error": str(exc)})
        if pause_seconds and i < len(ids):
            time.sleep(pause_seconds)
        if progress_every and i % progress_every == 0:
            logger.info(
                "starting_positions: %d/%d games (%d failed so far)",
                i, len(ids), len(failures),
            )

    if not frames:
        raise StartingPositionError(
            f"DATA_NOT_AVAILABLE: no starting positions could be fetched for any "
            f"of {len(ids)} games. First failure: "
            f"{failures[0]['error'] if failures else 'none recorded'}"
        )

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["GAME_ID", "PLAYER_ID"])
    if failures:
        logger.warning(
            "starting_positions: %d of %d games failed and are absent from the "
            "result", len(failures), len(ids),
        )
    return combined.reset_index(drop=True), failures


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------


def cache_path_for(season: str, root: Path | None = None) -> Path:
    return (root or CACHE_DIR) / f"starting_positions_{season.replace('/', '-')}.parquet"


def load_cached_starting_positions(
    season: str, root: Path | None = None
) -> pd.DataFrame | None:
    """The cached positions for a season, or None when they have not been pulled."""
    path = cache_path_for(season, root)
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    for column in STARTING_POSITION_COLUMNS:
        if column in frame.columns:
            frame[column] = frame[column].astype("string")
    return frame


def save_starting_positions(
    frame: pd.DataFrame, season: str, root: Path | None = None
) -> Path:
    path = cache_path_for(season, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    logger.info("starting_positions: wrote %d rows -> %s", len(frame), path)
    return path
