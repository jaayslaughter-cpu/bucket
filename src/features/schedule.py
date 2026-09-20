"""Team schedule context: rest, travel, and rest advantage.

Distinct from ``fatigue_logic``, which measures one player's own game
density. These are team-level facts about the schedule, all of them
published well before tip and therefore safe pregame features.

Three things the player-level view cannot see:

- **Rest advantage.** Our own rest only matters relative to the opponent's.
  A team on two days' rest facing a team on zero is in a different game
  from the same team facing a team on three.
- **The front half of a back-to-back.** Playing again tomorrow is a load
  management risk *today* — the player-level "days since last game" is
  blind to it because it only looks backwards.
- **Travel.** Distance flown is not the same thing as rest. A local
  back-to-back and a coast-to-coast red-eye can both show one day off.

ARENA COORDINATES are approximate venue locations, accurate enough for
distance-in-miles. The relocation table matters more than the precision:
computing a flight to the wrong building is worse than a few miles of
rounding. Neutral-site games are the known blind spot — see
``attach_team_schedule_features``.
"""

from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EARTH_RADIUS_MILES = 3963.2

# Current home venues, (latitude, longitude).
ARENA_COORDS: dict[str, tuple[float, float]] = {
    "ATL": (33.76, -84.40), "BOS": (42.37, -71.06), "BKN": (40.68, -73.98),
    "CHA": (35.23, -80.84), "CHI": (41.88, -87.67), "CLE": (41.50, -81.69),
    "DAL": (32.79, -96.81), "DEN": (39.75, -105.01), "DET": (42.34, -83.05),
    "GSW": (37.77, -122.39), "HOU": (29.75, -95.36), "IND": (39.76, -86.16),
    "LAC": (33.94, -118.34), "LAL": (34.04, -118.27), "MEM": (35.14, -90.05),
    "MIA": (25.78, -80.19), "MIL": (43.04, -87.92), "MIN": (44.98, -93.28),
    "NOP": (29.95, -90.08), "NYK": (40.75, -73.99), "OKC": (35.46, -97.52),
    "ORL": (28.54, -81.38), "PHI": (39.90, -75.17), "PHX": (33.45, -112.07),
    "POR": (45.53, -122.67), "SAC": (38.58, -121.50), "SAS": (29.43, -98.44),
    "TOR": (43.64, -79.38), "UTA": (40.77, -111.90), "WAS": (38.90, -77.02),
}

# Venues a team played in BEFORE a move, keyed by team. Each entry applies
# when the game date is strictly before `until`. Without these, a travel
# feature silently measures the flight to a building that did not exist yet.
ARENA_HISTORY: dict[str, list[tuple[date, tuple[float, float]]]] = {
    "SAC": [(date(2016, 10, 1), (38.64, -121.51))],  # Sleep Train -> Golden 1
    "DET": [(date(2017, 10, 1), (42.70, -83.25))],   # Palace of Auburn Hills -> LCA
    "GSW": [(date(2019, 10, 1), (37.75, -122.20))],  # Oracle (Oakland) -> Chase
    "LAC": [(date(2024, 10, 1), (34.04, -118.27))],  # Crypto.com -> Intuit Dome
    "BKN": [(date(2012, 10, 1), (40.73, -74.07))],   # New Jersey -> Barclays
}

# Whole-league neutral relocations that override the home team's venue.
NEUTRAL_PERIODS: list[tuple[date, date, tuple[float, float], str]] = [
    (date(2020, 7, 30), date(2020, 10, 11), (28.37, -81.55), "Orlando bubble"),
    (date(2020, 12, 22), date(2021, 5, 16), (27.94, -82.45), "Toronto in Tampa"),
]


def haversine_miles(
    lat1: pd.Series | float,
    lon1: pd.Series | float,
    lat2: pd.Series | float,
    lon2: pd.Series | float,
) -> pd.Series | float:
    """Great-circle distance in miles. Vectorized over pandas Series."""
    rlat1, rlon1, rlat2, rlon2 = (np.radians(v) for v in (lat1, lon1, lat2, lon2))
    dlat, dlon = rlat2 - rlat1, rlon2 - rlon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(rlat1) * np.cos(rlat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))) * EARTH_RADIUS_MILES


def venue_for(team: str, game_day: date) -> tuple[float, float] | None:
    """The coordinates a team hosted at on a given date, accounting for moves."""
    if team == "TOR":
        for start, end, coords, _label in NEUTRAL_PERIODS:
            if start <= game_day <= end and coords == (27.94, -82.45):
                return coords
    for cutoff, coords in ARENA_HISTORY.get(team, []):
        if game_day < cutoff:
            return coords
    return ARENA_COORDS.get(team)


def _game_venue(home_team: str, game_day: date) -> tuple[float, float] | None:
    for start, end, coords, _label in NEUTRAL_PERIODS:
        if start <= game_day <= end and coords == (28.37, -81.55):
            return coords  # bubble overrides every home venue
    return venue_for(home_team, game_day)


def attach_team_schedule_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Add team rest, travel, and rest-advantage columns to a player panel.

    Derives a team schedule from the panel itself (one row per team-game),
    so it works on the real panel and the demo panel alike without needing
    a separate schedule feed.

    Rest is partitioned by season. Without that, a team's first game after
    an offseason reads as roughly 150 days of rest, which is true and
    useless — it tells the model nothing about fatigue and swamps the
    within-season variation it should be learning from.

    Known limitation: the venue is taken to be the home team's arena, which
    is wrong for global games. Rows flagged IS_NEUTRAL_SITE get null travel
    rather than a confidently wrong distance.
    """
    required = {"TEAM_ABBREVIATION", "GAME_DATE", "GAME_ID"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"DATA_NOT_AVAILABLE: panel missing {sorted(missing)}")

    out = panel.copy()
    out["GAME_DATE"] = pd.to_datetime(out["GAME_DATE"])
    season_col = "SEASON" if "SEASON" in out.columns else None

    keys = ["TEAM_ABBREVIATION", "GAME_ID", "GAME_DATE"]
    extra = [c for c in ("OPPONENT_ABBREVIATION", "IS_HOME", "IS_NEUTRAL_SITE", "SEASON")
             if c in out.columns]
    schedule = out[keys + extra].drop_duplicates(subset=["TEAM_ABBREVIATION", "GAME_ID"])
    schedule = schedule.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"]).reset_index(drop=True)

    group_cols = ["TEAM_ABBREVIATION", season_col] if season_col else ["TEAM_ABBREVIATION"]
    by_team = schedule.groupby(group_cols, sort=False)["GAME_DATE"]

    # Days between games. 0 means a back-to-back.
    schedule["TEAM_DAYS_REST"] = (schedule["GAME_DATE"] - by_team.shift(1)).dt.days - 1
    schedule["TEAM_DAYS_UNTIL_NEXT"] = (by_team.shift(-1) - schedule["GAME_DATE"]).dt.days - 1
    # Nullable integers rather than booleans: these feed straight into the
    # boosted models, and a null must stay distinguishable from a false.
    schedule["IS_B2B_SECOND"] = (schedule["TEAM_DAYS_REST"] == 0).astype("Int8")
    schedule["IS_B2B_FIRST"] = (schedule["TEAM_DAYS_UNTIL_NEXT"] == 0).astype("Int8")
    schedule.loc[schedule["TEAM_DAYS_REST"].isna(), "IS_B2B_SECOND"] = pd.NA
    schedule.loc[schedule["TEAM_DAYS_UNTIL_NEXT"].isna(), "IS_B2B_FIRST"] = pd.NA
    # A long layoff carries no more information than a medium one.
    schedule["TEAM_DAYS_REST_CAPPED"] = schedule["TEAM_DAYS_REST"].clip(upper=5)

    schedule = _attach_travel(schedule)
    schedule = _attach_rest_advantage(schedule)

    merge_cols = [
        "TEAM_ABBREVIATION", "GAME_ID", "TEAM_DAYS_REST", "TEAM_DAYS_REST_CAPPED",
        "TEAM_DAYS_UNTIL_NEXT", "IS_B2B_SECOND", "IS_B2B_FIRST",
        "TRAVEL_MILES", "OPP_DAYS_REST", "REST_ADVANTAGE",
    ]
    out = out.merge(schedule[merge_cols], on=["TEAM_ABBREVIATION", "GAME_ID"], how="left")

    logger.info(
        "Team schedule features: %d team-games | %d B2B-second, %d B2B-first | "
        "median travel %.0f mi | rest advantage on %d rows",
        len(schedule),
        int(schedule["IS_B2B_SECOND"].fillna(0).sum()),
        int(schedule["IS_B2B_FIRST"].fillna(0).sum()),
        float(schedule["TRAVEL_MILES"].median(skipna=True) or 0.0),
        int(schedule["REST_ADVANTAGE"].notna().sum()),
    )
    return out


def _attach_travel(schedule: pd.DataFrame) -> pd.DataFrame:
    """Miles flown from the previous game's venue to this one."""
    if "IS_HOME" not in schedule.columns or "OPPONENT_ABBREVIATION" not in schedule.columns:
        schedule["TRAVEL_MILES"] = pd.NA
        logger.info("Travel distance skipped: needs IS_HOME and OPPONENT_ABBREVIATION")
        return schedule

    def _venue_row(row: pd.Series) -> tuple[float, float] | None:
        game_day = row["GAME_DATE"].date()
        if bool(row.get("IS_NEUTRAL_SITE", False)):
            return None  # unknown venue; better null than a wrong distance
        host = row["TEAM_ABBREVIATION"] if bool(row.get("IS_HOME", False)) else row["OPPONENT_ABBREVIATION"]
        return _game_venue(str(host), game_day)

    venues = schedule.apply(_venue_row, axis=1)
    schedule["_lat"] = [v[0] if v else np.nan for v in venues]
    schedule["_lon"] = [v[1] if v else np.nan for v in venues]

    group_cols = ["TEAM_ABBREVIATION", "SEASON"] if "SEASON" in schedule.columns else ["TEAM_ABBREVIATION"]
    prev_lat = schedule.groupby(group_cols, sort=False)["_lat"].shift(1)
    prev_lon = schedule.groupby(group_cols, sort=False)["_lon"].shift(1)

    schedule["TRAVEL_MILES"] = haversine_miles(
        prev_lat, prev_lon, schedule["_lat"], schedule["_lon"]
    ).round(1)

    unknown = int(schedule["_lat"].isna().sum())
    if unknown:
        logger.info("%d team-games had no resolvable venue — travel left null", unknown)
    return schedule.drop(columns=["_lat", "_lon"])


def _attach_rest_advantage(schedule: pd.DataFrame) -> pd.DataFrame:
    """Own rest minus the opponent's, for the same game."""
    if "OPPONENT_ABBREVIATION" not in schedule.columns:
        schedule["OPP_DAYS_REST"] = pd.NA
        schedule["REST_ADVANTAGE"] = pd.NA
        return schedule

    opponent_rest = schedule[["GAME_ID", "TEAM_ABBREVIATION", "TEAM_DAYS_REST_CAPPED"]].rename(
        columns={
            "TEAM_ABBREVIATION": "OPPONENT_ABBREVIATION",
            "TEAM_DAYS_REST_CAPPED": "OPP_DAYS_REST",
        }
    )
    merged = schedule.merge(opponent_rest, on=["GAME_ID", "OPPONENT_ABBREVIATION"], how="left")
    merged["REST_ADVANTAGE"] = merged["TEAM_DAYS_REST_CAPPED"] - merged["OPP_DAYS_REST"]
    return merged
