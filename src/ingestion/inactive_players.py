"""
src/ingestion/inactive_players.py — official pregame inactive lists.

WHY THIS EXISTS. ``src/features/teammate_cascade.py`` is wired into
``build_feature_matrix`` and abstains on every one of the panel's 214,381 rows,
because it needs to know who was OUT and nothing in the tree produces that.
This is the missing input. A star teammate being ruled out is one of the
largest single drivers of another player's minutes and usage, and the layer has
been refusing — correctly — to invent it.

THE SOURCE is the NBA's own box-score summary, whose ``InactivePlayers`` data
set is the real pregame inactive list rather than an inferred proxy. Two
endpoint versions expose it and they disagree about column names and about
which games they cover:

    boxscoresummaryv3  gameId, teamId, personId, firstName, familyName, jerseyNum
    boxscoresummaryv2  PLAYER_ID, FIRST_NAME, LAST_NAME, JERSEY_NUM, TEAM_ID,
                       TEAM_CITY, TEAM_NAME, TEAM_ABBREVIATION

v2 is the one every published example uses, and it is the wrong default: the
library's own wrapper warns that v2 data "may be missing for games on or after
4/10/2025", which covers the whole 2025-26 season in our panel. So v3 is
preferred and v2 is the fallback. ``parse_inactive_players`` accepts either.

WHY IT IS NOT LEAKAGE. The inactive list is announced before tip-off, so using
tonight's list for tonight's game uses information that existed pregame. One
honest caveat: we read it from the box-score summary *after* the game, so a
LATE scratch announced after a book set its line is, for a backtest of a
decision made at line-set time, information the decision could not have had.
That makes this feature legitimate for modelling and slightly optimistic for
replaying a timed decision. Recorded rather than hidden.

WHAT IT CANNOT DO HERE. stats.nba.com is denied at this environment's proxy
(``CONNECT tunnel failed, response 403``), the same denial
``src/ingestion/boxscores.py:16-17`` documents. Everything below is therefore
written against an injectable ``fetch`` so it is unit-testable offline, and the
live pull has to run where nba.com is reachable. The result caches to parquet,
so the ~1,230 calls per season are paid once.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import pandas as pd

from src.settlement.boxscore_fetcher import normalize_game_id

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/external/inactive_players")

# The normalised schema every version is mapped onto.
INACTIVE_COLUMNS = ("GAME_ID", "TEAM_ID", "PLAYER_ID", "PLAYER_NAME", "JERSEY_NUM")

# v3 first: v2 is documented as missing games on or after 2025-04-10.
ENDPOINT_PREFERENCE = ("boxscoresummaryv3", "boxscoresummaryv2")

_V3_RENAMES = {
    "gameId": "GAME_ID",
    "teamId": "TEAM_ID",
    "personId": "PLAYER_ID",
    "jerseyNum": "JERSEY_NUM",
}
_V2_RENAMES = {
    "PLAYER_ID": "PLAYER_ID",
    "TEAM_ID": "TEAM_ID",
    "JERSEY_NUM": "JERSEY_NUM",
    "TEAM_ABBREVIATION": "TEAM_ABBREVIATION",
}

DEFAULT_PAUSE_SECONDS = 0.6


class InactiveListError(RuntimeError):
    """Raised when inactive lists cannot be obtained from what was supplied."""


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------


def _frame_from_dataset(dataset: Mapping[str, Any]) -> pd.DataFrame:
    """Build a frame from the ``{"headers": [...], "data": [[...]]}`` shape."""
    headers = list(dataset.get("headers") or [])
    rows = list(dataset.get("data") or [])
    if not headers:
        raise InactiveListError(
            "DATA_NOT_AVAILABLE: the InactivePlayers data set carries no headers, "
            "so its columns cannot be named. The response shape changed."
        )
    return pd.DataFrame(rows, columns=headers)


def _detect_version(columns: Iterable[str], game_id: str) -> str:
    """Which endpoint version an InactivePlayers table came from.

    Raises on anything else, so a changed response is a named refusal rather
    than a silently empty or mis-mapped frame.
    """
    present = set(columns)
    if {"personId", "firstName", "familyName"} & present:
        return "v3"
    if {"PLAYER_ID", "FIRST_NAME", "LAST_NAME"} & present:
        return "v2"
    raise InactiveListError(
        f"DATA_NOT_AVAILABLE: game {game_id} InactivePlayers columns "
        f"{sorted(present)} match neither the v2 nor the v3 shape."
    )


def parse_inactive_players(
    data_sets: Mapping[str, Any], game_id: str | int
) -> pd.DataFrame:
    """
    Normalise one game's ``InactivePlayers`` data set, v2 or v3.

    An EMPTY list is a real answer — a game where everyone dressed — and comes
    back as an empty frame with the right columns, not an error. A MISSING data
    set is a different thing and raises: it means the response did not contain
    the field, and silently treating that as "nobody was out" would make a
    fetch failure indistinguishable from a healthy roster.
    """
    gid = normalize_game_id(game_id)
    if "InactivePlayers" not in data_sets:
        raise InactiveListError(
            f"DATA_NOT_AVAILABLE: game {gid} response has no InactivePlayers data "
            f"set (saw {sorted(data_sets)[:6]}). An absent data set is not an "
            "empty inactive list."
        )

    frame = _frame_from_dataset(data_sets["InactivePlayers"])

    # The shape is checked BEFORE the empty branch, not inside it. Checked only
    # on the non-empty path, a zero-row table with unrecognised columns became a
    # sentinel — recorded as a verified "nobody out" and turning that game's
    # counts into 0 — while the SAME broken schema carrying one row was refused.
    # A schema break decided by row count is the worst of both: it converts a
    # changed response into false evidence exactly when there is nothing to
    # cross-check it against.
    version = _detect_version(frame.columns, gid)

    if frame.empty:
        # A game where everyone dressed still has to record that it WAS fetched.
        # Returning zero rows loses that: the game vanishes from the concatenated
        # frame and attach_teammate_out_counts can no longer tell it from a game
        # nobody pulled, so a verified "nobody out" came back as
        # DATA_NOT_AVAILABLE. One sentinel row carries the coverage instead, and
        # it travels through concat, parquet and the CLI's resume set — which an
        # out-of-band set of ids would not.
        sentinel = {c: [pd.NA] for c in INACTIVE_COLUMNS}
        sentinel["GAME_ID"] = [gid]
        out = pd.DataFrame(sentinel)
        out["GAME_ID"] = out["GAME_ID"].astype("string")
        out["PLAYER_ID"] = out["PLAYER_ID"].astype("string")
        out["TEAM_ID"] = out["TEAM_ID"].astype("string")
        return out

    if version == "v3":
        work = frame.rename(columns=_V3_RENAMES)
        work["PLAYER_NAME"] = (
            work.get("firstName", "").astype(str).str.strip()
            + " "
            + work.get("familyName", "").astype(str).str.strip()
        ).str.strip()
    else:
        work = frame.rename(columns=_V2_RENAMES)
        work["PLAYER_NAME"] = (
            work.get("FIRST_NAME", "").astype(str).str.strip()
            + " "
            + work.get("LAST_NAME", "").astype(str).str.strip()
        ).str.strip()

    # The row's own gameId is not trusted over the one we asked for: v2 does not
    # carry it at all, and a mismatch would silently file rows under the wrong
    # game.
    work["GAME_ID"] = gid
    for column in INACTIVE_COLUMNS:
        if column not in work.columns:
            work[column] = pd.NA

    keep = list(INACTIVE_COLUMNS)
    if "TEAM_ABBREVIATION" in work.columns:
        keep.append("TEAM_ABBREVIATION")

    out = work[keep].copy()
    # Ids are keys, not numbers. Left as integers they lose the zero padding a
    # join needs, exactly as the parlay log's CSV reader did.
    out["PLAYER_ID"] = out["PLAYER_ID"].astype("string").str.strip()
    out["TEAM_ID"] = out["TEAM_ID"].astype("string").str.strip()
    out["GAME_ID"] = out["GAME_ID"].astype("string")
    return out.reset_index(drop=True)


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
        if endpoint == "boxscoresummaryv3":
            from nba_api.stats.endpoints import boxscoresummaryv3 as module

            call = module.BoxScoreSummaryV3
        else:
            from nba_api.stats.endpoints import boxscoresummaryv2 as module  # type: ignore[no-redef]

            call = module.BoxScoreSummaryV2  # type: ignore[assignment]
    except ImportError as exc:
        raise InactiveListError(
            "DATA_NOT_AVAILABLE: nba_api is not installed. It is declared in the "
            "optional 'stats' extra — install with `pip install -e '.[stats]'`."
        ) from exc

    summary = call(game_id=normalize_game_id(game_id), timeout=timeout)
    return summary.nba_response.get_data_sets(endpoint)


def fetch_inactive_players(
    game_id: str | int,
    *,
    fetch: Callable[[str, str], Mapping[str, Any]] | None = None,
    endpoints: Sequence[str] = ENDPOINT_PREFERENCE,
) -> pd.DataFrame:
    """
    One game's inactive list, trying each endpoint version in order.

    ``fetch(game_id, endpoint)`` returns the data-set mapping — the shape
    nba_api's ``get_data_sets`` produces. Injected so tests need neither the
    network nor nba_api.
    """
    gid = normalize_game_id(game_id)
    getter = fetch or (lambda g, e: _nba_api_fetch(g, endpoint=e))
    errors: list[str] = []
    for endpoint in endpoints:
        try:
            return parse_inactive_players(getter(gid, endpoint), gid)
        except InactiveListError as exc:
            errors.append(f"{endpoint}: {exc}")
        except Exception as exc:  # noqa: BLE001 — recorded, then the next version tried
            errors.append(f"{endpoint}: {type(exc).__name__}: {exc}")
    raise InactiveListError(
        f"DATA_NOT_AVAILABLE: game {gid} inactive list unavailable from "
        f"{list(endpoints)} — " + " | ".join(errors)
    )


def fetch_many_inactive_players(
    game_ids: Iterable[str | int],
    *,
    fetch: Callable[[str, str], Mapping[str, Any]] | None = None,
    pause_seconds: float = DEFAULT_PAUSE_SECONDS,
    progress_every: int = 100,
    stop_on_error: bool = False,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """
    Inactive lists for many games. Returns ``(frame, failures)``.

    Failures are RETURNED, not swallowed: a caller that asked for 1,230 games
    and got 1,180 has to be able to tell, or a partial pull silently becomes a
    season where fifty games had nobody injured.
    """
    ids = list(dict.fromkeys(normalize_game_id(g) for g in game_ids))
    frames: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []

    for i, gid in enumerate(ids, 1):
        try:
            frames.append(fetch_inactive_players(gid, fetch=fetch))
        except InactiveListError as exc:
            if stop_on_error:
                raise
            failures.append({"game_id": gid, "error": str(exc)})
        if pause_seconds and i < len(ids):
            time.sleep(pause_seconds)
        if progress_every and i % progress_every == 0:
            logger.info(
                "inactive_players: %d/%d games (%d failed so far)",
                i, len(ids), len(failures),
            )

    if not frames:
        raise InactiveListError(
            f"DATA_NOT_AVAILABLE: no inactive list could be fetched for any of "
            f"{len(ids)} games. First failure: "
            f"{failures[0]['error'] if failures else 'none recorded'}"
        )

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["GAME_ID", "PLAYER_ID"])
    if failures:
        logger.warning(
            "inactive_players: %d of %d games failed and are absent from the "
            "result", len(failures), len(ids),
        )
    return combined.reset_index(drop=True), failures


def cache_path_for(season: str, root: Path | None = None) -> Path:
    return (root or CACHE_DIR) / f"inactive_players_{season.replace('/', '-')}.parquet"


def load_cached_inactive_players(
    season: str, root: Path | None = None
) -> pd.DataFrame | None:
    """The cached list for a season, or None when it has not been pulled."""
    path = cache_path_for(season, root)
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    for column in ("GAME_ID", "PLAYER_ID", "TEAM_ID"):
        if column in frame.columns:
            frame[column] = frame[column].astype("string")
    return frame


def save_inactive_players(
    frame: pd.DataFrame, season: str, root: Path | None = None
) -> Path:
    path = cache_path_for(season, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    logger.info("inactive_players: wrote %d rows -> %s", len(frame), path)
    return path


# ---------------------------------------------------------------------------
# joining onto the panel
# ---------------------------------------------------------------------------


def team_id_to_abbreviation(
    teams: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, str]:
    """Map NBA team id -> abbreviation. Injectable; falls back to nba_api's static list.

    v3 carries only ``teamId`` while the panel keys on ``TEAM_ABBREVIATION``, so
    a map is required to join them at all. v2 carries the abbreviation directly
    and needs none.
    """
    if teams is None:
        try:
            from nba_api.stats.static import teams as static_teams
        except ImportError as exc:
            raise InactiveListError(
                "DATA_NOT_AVAILABLE: no team map supplied and nba_api is not "
                "installed, so team ids cannot be resolved to abbreviations."
            ) from exc
        teams = static_teams.get_teams()
    return {
        str(t["id"]).strip(): str(t["abbreviation"]).strip()
        for t in teams
        if t.get("id") is not None and t.get("abbreviation")
    }


DEFAULT_USAGE_COLUMN = "USAGE_PROXY_L10"


def _prior_usage_per_absence(
    absences: pd.DataFrame, panel: pd.DataFrame, usage_col: str
) -> pd.DataFrame:
    """Each absent player's usage level as known BEFORE the game he missed.

    An absent player has no row for the game he missed, so his usage has to come
    from his most recent EARLIER appearance. That is an as-of join, and the
    direction is the whole leakage argument: ``allow_exact_matches=False`` with
    ``direction="backward"`` takes a row strictly before this game's timestamp,
    never the game itself and never a later one.

    ``USAGE_PROXY_L10`` is itself built from shift(1) rolling values, so the
    level taken from an earlier game reflects only games before THAT one — two
    steps removed from the game being predicted. Rows with no earlier
    appearance (a rookie, or the first game in the window) come back NaN and are
    COUNTED rather than filled: an invented default would put fabricated usage
    into a feature whose whole point is measuring what is missing.
    """
    timeline = (
        panel[["PLAYER_ID", "GAME_DATE", usage_col]]
        .dropna(subset=["PLAYER_ID", "GAME_DATE", usage_col])
        .copy()
    )
    timeline["PLAYER_ID"] = timeline["PLAYER_ID"].astype("string")
    timeline = timeline.rename(columns={usage_col: "_prior_usage"})
    timeline = timeline.sort_values("GAME_DATE", kind="mergesort")

    left = absences.dropna(subset=["GAME_DATE"]).copy()
    left["PLAYER_ID"] = left["PLAYER_ID"].astype("string")
    left = left.sort_values("GAME_DATE", kind="mergesort")
    if left.empty or timeline.empty:
        left["_prior_usage"] = float("nan")
        return left

    return pd.merge_asof(
        left,
        timeline,
        on="GAME_DATE",
        by="PLAYER_ID",
        direction="backward",
        allow_exact_matches=False,
    )


def attach_absence_features(
    panel: pd.DataFrame,
    inactives: pd.DataFrame,
    *,
    team_map: Mapping[str, str] | None = None,
    usage_col: str = DEFAULT_USAGE_COLUMN,
) -> pd.DataFrame:
    """
    Add the absence features the cascade layer consumes, per (team, game):

    ``BBS_TEAMMATES_OUT``          how many teammates were inactive
    ``BBS_VACATED_USAGE``          the sum of their prior usage share
    ``BBS_VACATED_USAGE_UNKNOWN``  how many of them had no prior usage to sum
    ``BBS_INACTIVE_SOURCE``        whether this game was pulled at all

    WHY BOTH A COUNT AND A SUM. A count cannot tell a team missing 30% of its
    usage from one missing two end-of-bench players, and those are different
    situations for every remaining player's line. The count is kept because it
    is the honest fallback when a prior usage level is unavailable, and because
    the two disagree in a way worth seeing.

    WHY A COUNT AND NOT A PER-ROW FLAG. The panel holds only players who
    APPEARED: median 10 rows per team-game, and not one row with MIN == 0. An
    inactive player therefore has no panel row, so a per-row ``BBS_OUT_FLAG``
    would be 0 on every row and the cascade layer's ``team_outs - flag``
    arithmetic would yield zero teammates out everywhere — the layer would keep
    abstaining with the data in hand. The count has to come from the inactive
    list at team-game level, which is what this does.

    Games with no cached inactive list get ``pd.NA``, NOT 0. "We did not fetch
    this game" and "nobody was out in this game" are different statements and
    only one of them is evidence.
    """
    required = {"GAME_ID", "TEAM_ABBREVIATION"}
    missing = required - set(panel.columns)
    if missing:
        raise InactiveListError(
            f"DATA_NOT_AVAILABLE: panel is missing {sorted(missing)}, so inactive "
            "counts cannot be joined."
        )

    out = panel.copy()
    if inactives.empty:
        out["BBS_TEAMMATES_OUT"] = pd.NA
        out["BBS_VACATED_USAGE"] = pd.NA
        out["BBS_VACATED_USAGE_UNKNOWN"] = pd.NA
        out["BBS_INACTIVE_SOURCE"] = "DATA_NOT_AVAILABLE"
        return out

    work = inactives.copy()

    # Coverage is taken from EVERY row, before any filtering, and includes the
    # sentinel rows that stand for a fetched game with nobody out.
    fetched_games = set(work["GAME_ID"].astype(str).map(normalize_game_id))

    # Sentinels carry no player and must not be counted as an absence.
    real = work["PLAYER_ID"].notna() & (work["PLAYER_ID"].astype("string") != "")
    work = work[real].copy()

    # Per ROW, not all-or-nothing. fetch_many_inactive_players mixes versions:
    # v3 rows carry only teamId while a v2 fallback row carries the abbreviation,
    # so a frame holding both is neither "column absent" nor "all null". Testing
    # it that way skipped the mapping entirely and every v3 game lost its count
    # — reported as DATA_NOT_AVAILABLE, i.e. a game we fetched and where someone
    # WAS out read as unknown.
    if "TEAM_ABBREVIATION" not in work.columns:
        work["TEAM_ABBREVIATION"] = pd.NA
    needs_abbreviation = work["TEAM_ABBREVIATION"].isna()
    if needs_abbreviation.any():
        mapping = dict(team_map) if team_map is not None else team_id_to_abbreviation()
        work.loc[needs_abbreviation, "TEAM_ABBREVIATION"] = (
            work.loc[needs_abbreviation, "TEAM_ID"].astype("string").map(mapping)
        )

    unresolved = int(work["TEAM_ABBREVIATION"].isna().sum())
    if unresolved:
        logger.warning(
            "inactive_players: %d inactive rows have no team abbreviation and are "
            "dropped from the count", unresolved,
        )
        work = work[work["TEAM_ABBREVIATION"].notna()]

    if work.empty:
        counts = pd.DataFrame(columns=[
            "GAME_ID", "TEAM_ABBREVIATION", "BBS_TEAMMATES_OUT",
            "BBS_VACATED_USAGE", "BBS_VACATED_USAGE_UNKNOWN",
        ])
    else:
        # Each absence needs the date of the game it missed before its prior
        # usage can be looked up as-of. The date comes from the panel, keyed on
        # the padded game id like every other join here.
        game_dates = (
            panel.assign(_gid=panel["GAME_ID"].astype(str).map(normalize_game_id))
            .groupby("_gid", sort=False)["GAME_DATE"]
            .min()
            .rename("GAME_DATE")
            .reset_index()
            if "GAME_DATE" in panel.columns
            else pd.DataFrame(columns=["_gid", "GAME_DATE"])
        )
        work["_gid"] = work["GAME_ID"].astype(str).map(normalize_game_id)
        dated = work.merge(game_dates, on="_gid", how="left")

        if usage_col in panel.columns and not game_dates.empty:
            dated = _prior_usage_per_absence(dated, panel, usage_col)
        else:
            logger.warning(
                "inactive_players: %r absent from the panel — vacated usage cannot "
                "be summed and is reported unknown", usage_col,
            )
            dated["_prior_usage"] = float("nan")

        counts = (
            dated.groupby(["GAME_ID", "TEAM_ABBREVIATION"], sort=False)
            .agg(
                BBS_TEAMMATES_OUT=("PLAYER_ID", "nunique"),
                BBS_VACATED_USAGE=("_prior_usage", "sum"),
                BBS_VACATED_USAGE_UNKNOWN=("_prior_usage", lambda s: int(s.isna().sum())),
            )
            .reset_index()
        )

    # Both sides padded to the NBA's 10-character form before joining. The panel
    # stores GAME_ID unpadded ('21700548'); the API returns '0021700548'. Joined
    # as-is, nothing matches at all.
    out["_gid"] = out["GAME_ID"].astype(str).map(normalize_game_id)
    counts["_gid"] = (
        counts["GAME_ID"].astype(str).map(normalize_game_id)
        if not counts.empty
        else pd.Series(dtype="object")
    )

    value_columns = [
        "BBS_TEAMMATES_OUT", "BBS_VACATED_USAGE", "BBS_VACATED_USAGE_UNKNOWN",
    ]
    merged = out.merge(
        counts[["_gid", "TEAM_ABBREVIATION", *value_columns]],
        on=["_gid", "TEAM_ABBREVIATION"],
        how="left",
    )

    # A game present in the pull but with nobody out is a real 0. A game absent
    # from the pull stays NA. fetched_games was taken from every input row above,
    # sentinels included, so an empty inactive list still counts as covered.
    in_pull = merged["_gid"].isin(fetched_games)
    for column in value_columns:
        merged.loc[in_pull, column] = merged.loc[in_pull, column].fillna(0)
    merged["BBS_TEAMMATES_OUT"] = merged["BBS_TEAMMATES_OUT"].astype("Int64")
    merged["BBS_VACATED_USAGE_UNKNOWN"] = merged["BBS_VACATED_USAGE_UNKNOWN"].astype("Int64")
    merged["BBS_VACATED_USAGE"] = pd.to_numeric(
        merged["BBS_VACATED_USAGE"], errors="coerce"
    )
    merged["BBS_INACTIVE_SOURCE"] = pd.Series(
        ["official_inactive_list"] * len(merged), index=merged.index, dtype="object"
    )
    merged.loc[~in_pull, "BBS_INACTIVE_SOURCE"] = "DATA_NOT_AVAILABLE"

    coverage = float(in_pull.mean()) if len(merged) else 0.0
    logger.info(
        "inactive_players: %.1f%% of panel rows covered by the inactive pull "
        "(%d of %d)", coverage * 100.0, int(in_pull.sum()), len(merged),
    )
    return merged.drop(columns=["_gid"])
