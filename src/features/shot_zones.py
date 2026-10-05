"""
src/features/shot_zones.py — shot location and shot type, over the WHOLE panel.

WHY THIS EXISTS WHEN src/features/pbp.py ALREADY DOES SHOT MIX. Coverage —
and the size of the gain was measured, because the first version of this
docstring got it wrong in the generous direction.

That module's docstring says "The logs supplied cover 2025-26 only", and
**that claim is stale**: more event-log parts were supplied since it was
written. Counted on the real panel, PBP_RIM_RATE_L10 is populated for
2021-22 through 2025-26 and absent before, so the PBP_* shot-mix family
covers 126,624 of 214,381 rows (59.1%), not one season. (The stale line in
pbp.py is a separate thing to fix; it is recorded here rather than silently
relied on.)

This layer reads a DIFFERENT source — the NBA's own ``shotchartdetail``
export, one row per shot attempt — available per season from 1996. Measured
over the nine seasons this panel spans:

    PBP shot mix known        126,624 rows   59.1%
    SZ  shot mix known        209,566 rows   97.8%
    newly covered by SZ        83,568 rows   +39.0 points
    SZ missing where PBP has it    626 rows

So the gain is **four additional seasons** (2017-18 through 2020-21) and
39 points of panel coverage, not the eight seasons a reading of pbp.py's
stale line would suggest.

THE SOURCE IS COMPLETE, AND THAT WAS CHECKED THE WAY pbp.py INSISTS ON. A
derived stream is checked against an independent measurement of the same
thing before anything is built on it, because that module records two event
logs that named every game, spanned the right dates, carried no duplicates
and still held 32% and 84% of the events. Counting shots per player-game and
comparing to the PANEL'S OWN FGA, on the 24,895 player-games they share in
2023-24:

    exact match        100.00%
    mean |difference|  0.0000 attempts

Every player-game agrees exactly. That is a stronger result than the event
log's own 99.90%, and ``check_shot_completeness`` is the function that
re-establishes it per season rather than trusting this paragraph.

WHAT THIS GIVES THAT pbp.py CANNOT, besides coverage. The export carries the
NBA's own zone classification, so the zones are not inferred from a distance
threshold:

  SZ_RIM_RATE        Restricted Area
  SZ_PAINT_RATE      In The Paint (Non-RA) — the floater/short-hook band that
                     a distance cut lumps in with either the rim or mid-range
  SZ_MID_RATE        Mid-Range
  SZ_THREE_RATE      SHOT_TYPE == 3PT
  SZ_CORNER3_RATE    Left/Right Corner 3 only. A corner three is the shortest
                     three on the floor and the one most dependent on a
                     teammate's pass; nothing in this project separated it
                     from an above-the-break attempt
  SZ_DIST_AVG        mean SHOT_DISTANCE in feet
  SZ_DUNK_LAYUP_RATE ACTION_TYPE naming a dunk or a layup
  SZ_SELF_CREATED_RATE  pull-up, step-back, fadeaway, turnaround or driving
                     attempts — shots taken off the dribble rather than off a
                     pass. The closest thing to a usage-independent
                     shot-creation signal available without tracking data

WHAT IT CANNOT GIVE, stated so nobody looks for it here. ``PBP_ASSISTED_RATE``
needs the event log's second player, ``PBP_GARBAGE_SHOT_SHARE`` needs the
running score and clock, and ``PBP_GAME_PACE`` needs possessions. This export
has none of the three: it is a shot table, not an event log. Those three
columns remain pbp.py's and remain 2025-26 only.

SZ_FGA IS A DIAGNOSTIC, NOT A FEATURE, for the same reason PBP_FGA is: it is a
count, the panel already carries FGA, FGA_L5 and FGA_L10, and the completeness
check above is computed from it. Shipping it would hand a model one number
twice.

LEAKAGE. A shot happened DURING the game. Every column here is therefore
summarised per player-game and then rolled forward with a shift of one, so a
row carries that player's PREVIOUS games and never his current one.
``summarise_player_games`` deliberately returns same-game values and is NOT
safe to join directly; ``attach_shot_rolling_features`` does the shifting.

OVERLAP WITH PBP_* IS NEAR-TOTAL, AND THAT WAS MEASURED. Where both exist
(125,998 shared rows) they describe the same shots, and they agree:

    SZ_DIST_AVG_L10         vs PBP_SHOT_DIST_AVG_L10    |r| 0.9977
    SZ_THREE_RATE_L10       vs PBP_THREE_RATE_L10       |r| 0.9975
    SZ_DUNK_LAYUP_RATE_L10  vs PBP_DUNK_LAYUP_RATE_L10  |r| 0.9970
    SZ_RIM_RATE_L10         vs PBP_RIM_RATE_L10         |r| 0.9964
    SZ_MID_RATE_L10         vs PBP_MID_RATE_L10         |r| 0.9244

Those are duplicates, well inside the band ``labels._EXCLUDED_AS_REDUNDANT``
was built from, so **shipping both families would be the collinearity mistake
that cost the opponent-defence layer a third of its gain** (see
src/features/defense.py). SZ_* is the one to keep: same numbers, 39 more
points of coverage.

Mid-range is the one pair that disagrees at all, and the disagreement is the
argument for this source: pbp.py infers mid-range from a DISTANCE THRESHOLD
while this reads the NBA's own ``SHOT_ZONE_BASIC``, and the band between the
restricted area and the three-point line is exactly where a threshold and a
zone part company.

PBP_ASSISTED_RATE, PBP_GARBAGE_SHOT_SHARE and PBP_GAME_PACE have no
counterpart here and must stay. The end state is SZ_* replacing the pbp
shot-mix columns and pbp.py keeping those three — but which family a model
should read is still a measurement, so both are built and
``scripts/feature_ab.py --layer shot_zones`` is where it is settled.

RESEARCH ONLY. Nothing here is a betting signal.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

SOURCE_NAME = "shot_zones"

#: Columns this layer reads from a shotchartdetail export.
REQUIRED_SHOT_COLS: tuple[str, ...] = (
    "GAME_ID", "PLAYER_ID", "SHOT_DISTANCE", "SHOT_TYPE",
    "SHOT_ZONE_BASIC", "ACTION_TYPE", "SHOT_MADE_FLAG",
)

#: The NBA's own zone labels. Exact strings, not substrings: "Above the Break
#: 3" and "Left Corner 3" are both threes and only one of them is a corner, so
#: a substring match on "3" would merge them.
ZONE_RIM = ("Restricted Area",)
ZONE_PAINT = ("In The Paint (Non-RA)",)
ZONE_MID = ("Mid-Range",)
ZONE_CORNER3 = ("Left Corner 3", "Right Corner 3")

#: ACTION_TYPE fragments. Matched case-insensitively as substrings because the
#: export spells out compounds ("Driving Finger Roll Layup Shot"), and a shot
#: is a layup whether or not it was also driving.
ACTION_DUNK_LAYUP = ("dunk", "layup", "finger roll")
#: Off-the-dribble attempts. "Driving" is included deliberately: a drive is
#: created by the shooter even when it finishes at the rim.
ACTION_SELF_CREATED = ("pullup", "pull-up", "step back", "stepback",
                       "fadeaway", "turnaround", "driving", "running")

SHORT_WINDOW = 5
ROLL_WINDOW = 10
#: Three prior games before a rate is reported. A rate over one game is that
#: game's shot selection, which is not the player's.
ROLL_MIN_PERIODS = 3

#: Per player-game, before any rolling. NOT safe to join onto a panel.
SUMMARY_COLS: tuple[str, ...] = (
    "SZ_FGA",
    "SZ_DIST_AVG",
    "SZ_RIM_RATE",
    "SZ_PAINT_RATE",
    "SZ_MID_RATE",
    "SZ_THREE_RATE",
    "SZ_CORNER3_RATE",
    "SZ_DUNK_LAYUP_RATE",
    "SZ_SELF_CREATED_RATE",
    "SZ_MADE_PCT",
)

#: SZ_FGA is excluded: a count the panel already carries as FGA. See the
#: docstring.
ROLLED_STATS: tuple[str, ...] = tuple(c for c in SUMMARY_COLS if c != "SZ_FGA")

SHOT_FEATURE_COLS: tuple[str, ...] = tuple(
    f"{stat}_L{window}" for stat in ROLLED_STATS for window in (SHORT_WINDOW, ROLL_WINDOW)
)


class ShotZoneError(ValueError):
    """Raised when the shot export cannot support this layer."""


def _norm_ids(frame: pd.DataFrame, game: str, player: str) -> pd.DataFrame:
    """
    Ids as zero-stripped strings on both sides of every join.

    The export writes GAME_ID as an integer (22300001) and the panel carries
    it as a zero-padded string ("0022300001"). Joining those matches NOTHING
    while looking entirely reasonable, which is the quietest failure available
    here — measured at 1,230 of 1,230 games once stripped.
    """
    out = frame.copy()
    out["_GAME"] = out[game].astype(str).str.strip().str.lstrip("0")
    out["_PLAYER"] = out[player].astype(str).str.strip()
    return out


def prepare_shots(shots: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalise a shotchartdetail export."""
    if shots is None or shots.empty:
        raise ShotZoneError("DATA_NOT_AVAILABLE: the shot export is empty")
    missing = [c for c in REQUIRED_SHOT_COLS if c not in shots.columns]
    if missing:
        raise ShotZoneError(
            f"DATA_NOT_AVAILABLE: shot export missing {missing}. Expected a "
            "shotchartdetail export; found: "
            f"{sorted(shots.columns)[:12]}..."
        )

    out = _norm_ids(shots, "GAME_ID", "PLAYER_ID")
    out["SHOT_DISTANCE"] = pd.to_numeric(out["SHOT_DISTANCE"], errors="coerce")
    out["SHOT_MADE_FLAG"] = pd.to_numeric(out["SHOT_MADE_FLAG"], errors="coerce")

    zone = out["SHOT_ZONE_BASIC"].astype("string")
    action = out["ACTION_TYPE"].astype("string").str.lower()
    out["_IS_RIM"] = zone.isin(ZONE_RIM).astype(float)
    out["_IS_PAINT"] = zone.isin(ZONE_PAINT).astype(float)
    out["_IS_MID"] = zone.isin(ZONE_MID).astype(float)
    out["_IS_CORNER3"] = zone.isin(ZONE_CORNER3).astype(float)
    out["_IS_THREE"] = (
        out["SHOT_TYPE"].astype("string").str.startswith("3PT").fillna(False).astype(float)
    )
    out["_IS_DUNK_LAYUP"] = (
        action.str.contains("|".join(ACTION_DUNK_LAYUP), na=False).astype(float)
    )
    out["_IS_SELF_CREATED"] = (
        action.str.contains("|".join(ACTION_SELF_CREATED), na=False).astype(float)
    )
    return out


def check_shot_completeness(
    shots: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    min_exact_share: float = 0.98,
) -> dict[str, object]:
    """
    Count shots per player-game and compare to the PANEL'S OWN FGA.

    THIS IS THE GATE, not a formality, and the reason is recorded in
    src/features/pbp.py: two event logs supplied to this project named every
    game, spanned the right dates, carried no duplicates, and still held only
    32% and 84% of each game's events — and a partial log produces shot-mix
    rates that look entirely reasonable while being biased by whatever was
    dropped. Nothing downstream can detect it. The panel's FGA is an
    independent measurement of the same quantity, so it is what decides.

    Returns the comparison rather than raising, so a caller can report a bad
    season and continue with the good ones.
    """
    if "FGA" not in panel.columns:
        raise ShotZoneError(
            "DATA_NOT_AVAILABLE: the panel has no FGA, so there is no "
            "independent count to check the shot log against. Refusing to "
            "certify it rather than trusting it."
        )

    counted = (
        prepare_shots(shots)
        .groupby(["_GAME", "_PLAYER"], as_index=False)
        .size()
        .rename(columns={"size": "SZ_FGA"})
    )
    left = _norm_ids(panel, "GAME_ID", "PLAYER_ID")
    left["FGA"] = pd.to_numeric(left["FGA"], errors="coerce")
    merged = left.merge(counted, on=["_GAME", "_PLAYER"], how="inner")
    merged = merged[merged["FGA"].notna()]

    if merged.empty:
        return {
            "status": "DATA_NOT_AVAILABLE",
            "reason": (
                "no player-game is shared by the shot log and the panel — most "
                "likely the id formats do not agree, which matches nothing "
                "while looking correct"
            ),
            "compared": 0,
        }

    # A player with zero FGA takes no shots, so he is absent from the log
    # rather than present with a zero. Those rows cannot disagree and are not
    # evidence either way.
    took_shots = merged[merged["FGA"] > 0]
    diff = (took_shots["SZ_FGA"] - took_shots["FGA"]).abs()
    exact = float((diff == 0).mean()) if len(took_shots) else 0.0
    status = "OK" if exact >= min_exact_share else "DATA_NOT_AVAILABLE"
    report = {
        "status": status,
        "compared": int(len(took_shots)),
        "exact_share": round(exact, 6),
        "mean_abs_diff": round(float(diff.mean()), 6) if len(took_shots) else None,
        "within_one": round(float((diff <= 1).mean()), 6) if len(took_shots) else None,
        "min_exact_share": min_exact_share,
    }
    log = logger.info if status == "OK" else logger.error
    log(
        "Shot log completeness: %d player-games compared, %.4f exact against "
        "the panel's own FGA (floor %.2f), mean |diff| %s. %s",
        report["compared"], exact, min_exact_share, report["mean_abs_diff"],
        "Usable." if status == "OK" else
        "REFUSED: a partial log yields plausible-looking rates biased by "
        "whatever was dropped, and nothing downstream can detect it.",
    )
    return report


def summarise_player_games(shots: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (game, player) of SAME-GAME shot mix.

    NOT SAFE TO JOIN ONTO A PANEL. Every column here describes the game it is
    keyed to, so joining it directly would hand a model the shot selection of
    the game it is predicting. ``attach_shot_rolling_features`` is the only
    intended consumer; this is separate so the completeness check and the
    rolling can be tested apart.
    """
    prepared = prepare_shots(shots)
    grouped = prepared.groupby(["_GAME", "_PLAYER"], as_index=False)
    out = grouped.agg(
        SZ_FGA=("SHOT_DISTANCE", "size"),
        SZ_DIST_AVG=("SHOT_DISTANCE", "mean"),
        SZ_RIM_RATE=("_IS_RIM", "mean"),
        SZ_PAINT_RATE=("_IS_PAINT", "mean"),
        SZ_MID_RATE=("_IS_MID", "mean"),
        SZ_THREE_RATE=("_IS_THREE", "mean"),
        SZ_CORNER3_RATE=("_IS_CORNER3", "mean"),
        SZ_DUNK_LAYUP_RATE=("_IS_DUNK_LAYUP", "mean"),
        SZ_SELF_CREATED_RATE=("_IS_SELF_CREATED", "mean"),
        SZ_MADE_PCT=("SHOT_MADE_FLAG", "mean"),
    )
    logger.info(
        "Shot summary: %d player-games, %d game(s), %d player(s); mean %.2f "
        "attempts, mean distance %.2f ft.",
        len(out), out["_GAME"].nunique(), out["_PLAYER"].nunique(),
        float(out["SZ_FGA"].mean()), float(out["SZ_DIST_AVG"].mean()),
    )
    return out


def attach_shot_rolling_features(
    panel: pd.DataFrame,
    summary: pd.DataFrame | None = None,
    *,
    required: bool = False,
) -> pd.DataFrame:
    """
    Join each player's PRIOR-games shot mix onto his rows.

    ``summary`` is ``summarise_player_games``' output. The join brings the
    same-game values alongside each row and the rolling immediately shifts
    them away again, so no same-game value survives into a feature — the
    shift is inside this function for exactly that reason, and the same-game
    columns are dropped before returning.

    Unmatched rows keep NaN. A league-average fill would read as a measured
    shot profile for a player nobody has shot-charted.
    """
    if panel is None or panel.empty:
        if required:
            raise ShotZoneError("DATA_NOT_AVAILABLE: panel is empty")
        logger.info("Shot-zone layer skipped: empty panel.")
        return panel if panel is not None else pd.DataFrame()

    needed = {"PLAYER_ID", "GAME_ID", "GAME_DATE"}
    absent = needed - set(panel.columns)
    if absent:
        message = f"shot-zone features need {sorted(absent)}"
        if required:
            raise ShotZoneError(f"DATA_NOT_AVAILABLE: {message}")
        logger.info("Shot-zone layer skipped: %s", message)
        return panel

    if summary is None or summary.empty:
        message = (
            "no shot summary supplied. Build one with "
            "scripts/build_shot_panel.py, which checks the log against the "
            "panel's own FGA first."
        )
        if required:
            raise ShotZoneError(f"DATA_NOT_AVAILABLE: {message}")
        logger.info("Shot-zone layer skipped: %s", message)
        return panel

    out = _norm_ids(panel, "GAME_ID", "PLAYER_ID")
    out["GAME_DATE"] = pd.to_datetime(out["GAME_DATE"], errors="coerce")

    cols = [c for c in SUMMARY_COLS if c in summary.columns]
    lookup = summary[["_GAME", "_PLAYER", *cols]].drop_duplicates(
        subset=["_GAME", "_PLAYER"], keep="last"
    )
    before = len(out)
    out = out.merge(lookup, on=["_GAME", "_PLAYER"], how="left")
    if len(out) != before:
        raise ShotZoneError(
            f"Shot join changed the row count ({before} -> {len(out)}). The "
            "summary must hold at most one row per (game, player)."
        )

    order = out.sort_values(["_PLAYER", "GAME_DATE"], kind="mergesort").index
    work = out.loc[order]
    grouped = work.groupby("_PLAYER", sort=False)
    rolled: dict[str, pd.Series] = {}
    for stat in ROLLED_STATS:
        if stat not in work.columns:
            continue
        for window in (SHORT_WINDOW, ROLL_WINDOW):
            rolled[f"{stat}_L{window}"] = grouped[stat].transform(
                lambda s, w=window: (
                    s.shift(1).rolling(w, min_periods=ROLL_MIN_PERIODS).mean()
                )
            )
    for name, series in rolled.items():
        out[name] = series.reindex(out.index)

    # The same-game columns were joined only so the rolling could shift them.
    # Leaving them on the frame would publish the shot selection of the game
    # being predicted, which is the one thing this layer must not do.
    out = out.drop(columns=[c for c in cols if c in out.columns] + ["_GAME", "_PLAYER"])

    first = f"{ROLLED_STATS[0]}_L{ROLL_WINDOW}"
    known = int(out[first].notna().sum()) if first in out.columns else 0
    logger.info(
        "Shot-zone layer: %d of %d rows carry a prior shot profile (%.1f%%). "
        "The rest are a player's first %d shot-charted games, which have no "
        "prior mix and are left null.",
        known, len(out), 100.0 * known / max(len(out), 1), ROLL_MIN_PERIODS,
    )
    return out
