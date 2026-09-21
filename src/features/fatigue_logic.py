"""Schedule-density and travel fatigue adjustment.

Every input here is knowable before tip: the schedule is published in
advance, so "how many games has this player's team played in the last N
days" and "is this an away game in Denver" are both pregame facts.

THE MULTIPLIERS BELOW ARE UNFITTED HEURISTICS. They are documented
starting values chosen to be conservative, not parameters estimated from
data. Do not describe a projection as validated because it passed through
this module. Fitting them against real player game logs is open work —
see docs/DATA_GAPS.md.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Teams whose home arena sits at altitude. An away team travelling here
# gets a small penalty; the home team does not (they are acclimatised).
ALTITUDE_TEAMS = frozenset({"DEN", "UTA"})

# Unfitted, conservative. Applied multiplicatively, worst density wins.
B2B_PENALTY = 0.97
THREE_IN_FOUR_PENALTY = 0.96
FOUR_IN_FIVE_PENALTY = 0.94
ALTITUDE_PENALTY = 0.98


def assess_schedule_density(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag back-to-backs and compressed stretches per player.

    Counts games in the trailing window INCLUDING the current one, which
    is legitimate: the schedule is known in advance. Only prior game
    *dates* are used — never prior game results.
    """
    work = df.copy()
    work["GAME_DATE"] = pd.to_datetime(work["GAME_DATE"])
    work = work.sort_values(["PLAYER_ID", "GAME_DATE"])

    # Partition by season. Grouping on player alone makes the first game
    # after an offseason read as ~150 days of rest — true, but useless: it
    # tells the model nothing about fatigue while swamping the within-season
    # variation the feature exists to capture.
    group_cols = ["PLAYER_ID", "SEASON"] if "SEASON" in work.columns else ["PLAYER_ID"]
    if "SEASON" not in work.columns:
        logger.warning(
            "No SEASON column — rest is computed across season boundaries, so each "
            "player's first game of a season will show an offseason-length gap."
        )

    grouped = work.groupby(group_cols, sort=False)["GAME_DATE"]
    prev_date = grouped.shift(1)
    work["days_rest"] = (work["GAME_DATE"] - prev_date).dt.days

    work["is_back_to_back"] = work["days_rest"] == 1

    def _games_within(dates: pd.Series, window_days: int) -> pd.Series:
        """Games played in the trailing `window_days`, current game included."""
        idx = pd.DatetimeIndex(dates)
        counts = [
            int(((idx > d - pd.Timedelta(days=window_days)) & (idx <= d)).sum())
            for d in idx
        ]
        return pd.Series(counts, index=dates.index, dtype=int)

    work["games_last_4d"] = grouped.transform(lambda s: _games_within(s, 4))
    work["games_last_5d"] = grouped.transform(lambda s: _games_within(s, 5))
    work["is_3_in_4"] = work["games_last_4d"] >= 3
    work["is_4_in_5"] = work["games_last_5d"] >= 4

    return work.loc[df.index] if df.index.equals(work.index) else work


def attach_fatigue_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach ``fatigue_multiplier`` plus the density flags it derives from.

    Precedence when several apply: 4-in-5 > 3-in-4 > back-to-back. They do
    not stack, because a 4-in-5 already contains a back-to-back and
    compounding them would double-count the same tired legs.

    The altitude tax is separate and DOES stack: it fires only on an away
    game against an altitude team that is actually played in that team's
    arena. A neutral-site game is excluded via ``IS_NEUTRAL_SITE``, so a
    team that is not travelling to altitude is not taxed for it.
    """
    df = assess_schedule_density(df)

    multiplier = pd.Series(1.0, index=df.index, dtype=float)
    multiplier = multiplier.mask(df["is_back_to_back"].fillna(False), B2B_PENALTY)
    multiplier = multiplier.mask(df["is_3_in_4"].fillna(False), THREE_IN_FOUR_PENALTY)
    multiplier = multiplier.mask(df["is_4_in_5"].fillna(False), FOUR_IN_FIVE_PENALTY)

    if "OPPONENT_ABBREVIATION" in df.columns and "IS_HOME" in df.columns:
        away = ~df["IS_HOME"].fillna(False).astype(bool)
        at_altitude = df["OPPONENT_ABBREVIATION"].isin(ALTITUDE_TEAMS)
        # A neutral-site game is not played in the opponent's arena, so
        # nobody travels to altitude. This check belongs here, where the tax
        # is applied. It used to be done by rewriting IS_HOME to True for
        # neutral rows in the panel loader, which suppressed the tax but
        # corrupted an active model feature and every home/away report.
        if "IS_NEUTRAL_SITE" in df.columns:
            neutral = df["IS_NEUTRAL_SITE"].fillna(False).astype(bool)
            at_altitude = at_altitude & ~neutral
        else:
            logger.warning(
                "No IS_NEUTRAL_SITE column — the altitude tax cannot tell a neutral-site "
                "game from a trip to Denver, so neutral games will be taxed."
            )
        multiplier = multiplier.where(~(away & at_altitude), multiplier * ALTITUDE_PENALTY)
    else:
        logger.warning(
            "Altitude tax skipped: needs OPPONENT_ABBREVIATION and IS_HOME. "
            "Fatigue reflects schedule density only."
        )

    df["fatigue_multiplier"] = multiplier
    logger.info(
        "Fatigue attached: %d B2B, %d 3-in-4, %d 4-in-5 rows (mean multiplier %.4f)",
        int(df["is_back_to_back"].sum()),
        int(df["is_3_in_4"].sum()),
        int(df["is_4_in_5"].sum()),
        float(df["fatigue_multiplier"].mean()),
    )
    return df
