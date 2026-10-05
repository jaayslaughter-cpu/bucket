"""
src/features/fouls.py — the player's own prior foul history, as pregame features.

WHAT A FOUL COLUMN IS FOR, AND IT IS NOT A FOUL PROP. Nothing here predicts
how many fouls a player will commit. Fouls are in this panel because SIX OF
THEM ENDS A PLAYER'S NIGHT, and a disqualification is the one in-game event
that truncates minutes without any injury, blowout or coaching decision
behind it. A player who averages 1.1 fouls and a player who averages 3.6
project the same points from the same rolling scoring history, and the
second one is carrying a minutes risk the first is not. That risk is what
these columns describe.

WHY A RATE AND A SHARE, NOT ONLY A MEAN. Three separate things are worth
knowing and a single average blurs them:

  PF_L5 / PF_L10 / PF_SEASON   the level. Fouls per game, prior games only.
  PF_PER_MIN_L10               the propensity, independent of playing time.
                               A bench player with 2.0 fouls in 14 minutes
                               is far more foul-prone than a starter with
                               2.0 in 34, and the per-game mean says they
                               are the same. This is a RATIO OF TOTALS
                               (sum of fouls / sum of minutes over the
                               window), not a mean of per-game ratios: the
                               latter lets a two-minute appearance with one
                               foul dominate a ten-game window.
  PF_TROUBLE_RATE_L10          the tail. The share of the prior ten games in
                               which the player reached FIVE fouls — one away
                               from fouling out, which is where coaches start
                               sitting people. The mean cannot express this:
                               2.5 fouls a game is a different player if it
                               is 2-3-2-3-2 than if it is 0-5-0-5-2, and only
                               the second one is at risk of an early exit.

THE SOURCE. ``foulsPersonal`` in the Kaggle NBA archive's
``PlayerStatistics`` export, mapped to ``PF`` by
``src.ingestion.kaggle_nba.COLUMN_ALIASES``. Measured on the pack's own
file: present on 304,395 of 305,614 player-game rows (99.6%), mean 1.51,
min 0, max 6 — a distribution consistent with a real personal-foul count
bounded by disqualification.

AND FROM THE LIVE PULLER TOO, which was nearly missed. An earlier draft of
this module asserted that stats.nba.com's ``leaguegamelog`` does not report
personal fouls, so the live panel could never carry them. The payload's
header list is recorded in ``tests/test_boxscore_ingest.py`` and ``PF`` is
in it, between ``TOV`` and ``PTS``; ``boxscores.COLUMN_MAP`` had simply never
asked for it. It asks now, so this layer runs on a live slate and not only on
a rebuilt archive panel. The claim was about a payload nobody had looked at,
and the thing to look at was already in the repository.

It is NOT the ``pf`` written by ``src.ingestion.bigdataball``: that column is
on ``TeamGameStat`` and is a TEAM total, which is a different measurement
with the same name.

WHY THIS IS NOT IN ``ROLLING_STATS``. Adding "PF" there would have been two
characters of work and would have produced PF_BASELINE, PF_L2, PF_L2_PACE
and a PF member of every downstream form, halflife and minutes-weighted
layer — a dozen columns nothing asked for, applying a fatigue haircut and a
pace multiplier to a foul count as though a foul were production to be
projected. A self-contained layer is both narrower and reversible.

LEAKAGE. Every column is a shift-1 rolling or expanding statistic over the
player's PRIOR games, computed inside the player-season for the two that
are season-scoped. A row never sees its own game's fouls. The layer is
exercised by ``tests/test_fouls.py``, which includes a direct check that a
single enormous foul count never reaches the row it happened in.

NOT WIRED INTO ANY MODEL. These columns are built and exported but are
deliberately absent from ``labels.default_feature_cols``: whether a foul
history improves a points or minutes projection is a measurement, not an
assumption, and ``scripts/feature_ab.py --layer fouls --wire-under-test``
is where it gets made. This is the same path ``minutes_weighted`` is on.

RESEARCH ONLY. Nothing here is a betting signal.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

SOURCE_NAME = "fouls"

# The panel column this layer reads. Absent -> the layer emits nothing.
PF_COLUMN = "PF"

# Five, not six. Six is the disqualification itself, by which point the
# minutes are already lost; five is the state a coach reacts to.
FOUL_TROUBLE_THRESHOLD = 5

# A personal-foul count outside this range is not a personal-foul count.
# Six is the maximum possible in regulation; overtime games can carry a
# disqualified player no further, so the bound holds for them too.
PF_PLAUSIBLE_MAX = 6

ROLL_WINDOW = 10
SHORT_WINDOW = 5
# Deliberately applied to only TWO of the five columns, and the asymmetry is
# the point. A prior-games MEAN is defined from one prior game, so PF_L5,
# PF_L10 and PF_SEASON use min_periods=1 and match how every other {stat}_L5
# in this panel behaves. A RATE and a SHARE are not: one foul in one game is
# a per-minute rate and a trouble rate of either 0.0 or 1.0, both of which
# look like strong statements about a player nobody has seen. Those two wait
# for three games. UNFITTED; nothing has been measured about 3 vs 5.
ROLL_MIN_PERIODS = 3

FOUL_FEATURE_COLS: tuple[str, ...] = (
    "PF_L5",
    "PF_L10",
    "PF_SEASON",
    "PF_PER_MIN_L10",
    "PF_TROUBLE_RATE_L10",
)


class FoulFeatureError(ValueError):
    """Raised when foul features are asked for and PF is not in the panel."""


def _prior_rolling_mean(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    """Mean over the prior ``window`` games. Shift first, then roll."""
    return series.shift(1).rolling(window, min_periods=min_periods).mean()


def _prior_expanding_mean(series: pd.Series) -> pd.Series:
    """Mean over every prior game in the group."""
    return series.shift(1).expanding(min_periods=1).mean()


def _prior_rolling_sum(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    """Sum over the prior ``window`` games. Shift first, then roll."""
    return series.shift(1).rolling(window, min_periods=min_periods).sum()


def attach_foul_features(panel: pd.DataFrame, *, required: bool = False) -> pd.DataFrame:
    """
    Add the player's prior-games foul history to each row.

    Returns the panel UNCHANGED when ``PF`` is absent, which is the normal
    outcome for a panel built before either ingest requested the column. A
    column of zeros would read as a measured foul-free league, so nothing is
    filled.

    ``required=True`` raises instead, for a caller that has established the
    column should be there.
    """
    if panel is None or panel.empty:
        if required:
            raise FoulFeatureError("DATA_NOT_AVAILABLE: panel is empty")
        logger.info("Foul layer skipped: empty panel.")
        return panel if panel is not None else pd.DataFrame()

    if PF_COLUMN not in panel.columns:
        message = (
            f"panel has no {PF_COLUMN} column. Two sources carry it and both "
            "are wired: the archive's foulsPersonal via "
            "src.ingestion.kaggle_nba, and the league game log's PF via "
            "src.ingestion.boxscores. A panel built before either requested "
            "it must be rebuilt to carry it."
        )
        if required:
            raise FoulFeatureError(f"DATA_NOT_AVAILABLE: {message}")
        logger.info("Foul layer skipped: %s", message)
        return panel

    if "PLAYER_ID" not in panel.columns or "GAME_DATE" not in panel.columns:
        message = "foul features need PLAYER_ID and GAME_DATE to order a history"
        if required:
            raise FoulFeatureError(f"DATA_NOT_AVAILABLE: {message}")
        logger.info("Foul layer skipped: %s", message)
        return panel

    out = panel.copy()
    pf = pd.to_numeric(out[PF_COLUMN], errors="coerce")

    # An out-of-range value is dropped to NaN rather than clipped. Clipping
    # would turn a column that is not personal fouls into one that looks
    # like personal fouls; a null says the number was not usable.
    impossible = pf.notna() & ((pf < 0) | (pf > PF_PLAUSIBLE_MAX))
    if impossible.any():
        logger.warning(
            "Foul layer: %d of %d rows carry a PF outside 0-%d and are treated "
            "as unknown, not clipped. A count that cannot be a personal foul "
            "count is more likely a different column than a real outlier.",
            int(impossible.sum()), len(out), PF_PLAUSIBLE_MAX,
        )
        pf = pf.where(~impossible)

    work = pd.DataFrame(
        {
            "_PF": pf,
            "_MIN": pd.to_numeric(out.get("MIN"), errors="coerce"),
            "_TROUBLE": (pf >= FOUL_TROUBLE_THRESHOLD).where(pf.notna()).astype("float"),
            "PLAYER_ID": out["PLAYER_ID"],
            "GAME_DATE": pd.to_datetime(out["GAME_DATE"], errors="coerce"),
        },
        index=out.index,
    )
    if "SEASON" in out.columns:
        work["SEASON"] = out["SEASON"]

    # Sort by (player, date) and remember the original order. The panel
    # arrives sorted this way from build_feature_matrix, but a caller
    # attaching this layer to an arbitrary frame must not get a rolling
    # window computed over rows in file order.
    order = work.sort_values(["PLAYER_ID", "GAME_DATE"], kind="mergesort").index
    work = work.loc[order]

    by_player = work.groupby("PLAYER_ID", sort=False)
    computed = pd.DataFrame(index=work.index)
    computed["PF_L5"] = by_player["_PF"].transform(
        _prior_rolling_mean, window=SHORT_WINDOW, min_periods=1
    )
    computed["PF_L10"] = by_player["_PF"].transform(
        _prior_rolling_mean, window=ROLL_WINDOW, min_periods=1
    )
    computed["PF_TROUBLE_RATE_L10"] = by_player["_TROUBLE"].transform(
        _prior_rolling_mean, window=ROLL_WINDOW, min_periods=ROLL_MIN_PERIODS
    )
    # A RATIO OF TOTALS, not a mean of ratios. The distinction is the whole
    # point of the column: one foul in a two-minute cameo is a per-minute rate
    # of 0.5, eight times any real player's propensity, and a mean of ten such
    # ratios is dominated by whichever game was shortest. Summing both sides
    # first weights each game by the minutes it actually contributed.
    #
    # Both sides are masked to the games where BOTH numbers are known, so the
    # numerator and the denominator always cover the same set of games. Rolling
    # them independently would divide nine games of fouls by ten games of
    # minutes wherever one foul count was missing, and understate the rate
    # without saying so.
    both = work["_PF"].notna() & work["_MIN"].notna()
    work["_PF_PAIRED"] = work["_PF"].where(both)
    work["_MIN_PAIRED"] = work["_MIN"].where(both)
    paired = work.groupby("PLAYER_ID", sort=False)
    fouls_sum = paired["_PF_PAIRED"].transform(
        _prior_rolling_sum, window=ROLL_WINDOW, min_periods=ROLL_MIN_PERIODS
    )
    minutes_sum = paired["_MIN_PAIRED"].transform(
        _prior_rolling_sum, window=ROLL_WINDOW, min_periods=ROLL_MIN_PERIODS
    )
    computed["PF_PER_MIN_L10"] = fouls_sum / minutes_sum.where(minutes_sum > 0)

    # PF_SEASON is scoped to the player-season where a SEASON column exists,
    # for the same reason every other _SEASON column is: a player's October
    # mean must not carry last June's whistle.
    season_keys = ["PLAYER_ID", "SEASON"] if "SEASON" in work.columns else ["PLAYER_ID"]
    computed["PF_SEASON"] = work.groupby(season_keys, sort=False)["_PF"].transform(
        _prior_expanding_mean
    )

    for col in FOUL_FEATURE_COLS:
        out[col] = computed[col].reindex(out.index)

    known = int(out["PF_L10"].notna().sum())
    logger.info(
        "Foul layer: %d of %d rows carry a prior foul history (%.1f%%). The rest "
        "are a player's first game, which has none and is left null.",
        known, len(out), 100.0 * known / max(len(out), 1),
    )
    return out


def attach_foul_features_layer(panel: pd.DataFrame) -> pd.DataFrame:
    """Registry entry point. Same contract as every other additive layer."""
    return attach_foul_features(panel, required=False)
