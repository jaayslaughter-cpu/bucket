"""Return-from-absence minutes ramp (external audit, 2026-11-13).

THE GAP THIS CLOSES, and why it is a prop-pricing gap specifically.

Audit of `mroditall2x/NBA-Model-`, `rot_blend.py`. It carries a `streak`
helper that counts, for a player, how many of his team's games he has missed
in a row, and then projects his minutes as `base * RET_F` -- a baseline
scaled down because a player returning from an absence does not immediately
play his season average.

This project has `src/features/absences.py`, which answers "who is OUT
tonight", and `src/features/teammate_cascade.py`, which answers "how many of
his teammates are out". Neither answers "is HE just back, and from how long
out?" -- and that is the question a rolling feature gets wrong by
construction. Every `{STAT}_L5` and `{STAT}_L10` on this panel is an average
over his last appearances, which are the games BEFORE the absence. On his
first game back those windows describe a healthy player at full minutes,
which is exactly the night he is least likely to play them. The feature set
therefore overstates him on the one night the error is predictable.

HOW THE ABSENCE IS OBSERVED WITHOUT AN INJURY FEED, which is what makes this
computable here at all. `teammate_cascade` records the load-bearing property
of this panel: it "holds only players who APPEARED (median 10 rows per
team-game, no row with MIN == 0)". An absence therefore leaves no row. So a
gap in a player's sequence of appearances, measured against his TEAM's
sequence of games, IS the absence -- no external status source is required,
and nothing is inferred from a status string that may be stale at tip-off.

WHAT IS DELIBERATELY NOT PORTED. The audited repo applies a hardcoded return
factor (`base * RET_F`) and a 20-minute regular threshold. Those are magnitudes
fitted to someone else's rotation data, and hardcoding them here would publish
a minutes haircut this project has never measured. This module emits the
COUNT and the FLAG and lets the model learn the ramp's size, which is both
honest and strictly more general: a 1-game absence and a 15-game absence do
not deserve the same multiplier, and a single constant cannot express that.

LEAKAGE. Every column here is a function of (a) this game's position in his
team's schedule and (b) his PRIOR appearances. The current row's `MIN` is
never read. `RET_BASELINE_MIN_L5` is a grouped rolling mean and is therefore
shifted, per CLAUDE.md: a baseline that included tonight's minutes would be
the single most direct leak available on this panel.

A TRADE IS NOT A RETURN. A player's previous appearance for a different team
says nothing about games missed for this one, so the gap is computed within
(player, team) and a first appearance for a new team reports NaN rather than
a count invented from the two schedules.

NOT IN THE FEATURE CONTRACT. Like `box_rates`, `fouls` and `dvp`, this is
emitted so `scripts/feature_ab.py` can measure whether it earns a place.
Nothing here is wired into the model matrix.

RESEARCH_ONLY. Schedule arithmetic. No wager.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Columns this layer needs. It abstains whole if any is missing, rather than
#: emitting a partial set a caller cannot tell from a complete one.
REQUIRED = ("PLAYER_ID", "TEAM_ABBREVIATION", "GAME_ID", "GAME_DATE", "MIN")

#: Team games missed in a row at or above which a player counts as RETURNING.
#: Two, matching the audited repo's `mstreak < 2` test, and chosen there
#: because a single missed game is routine rest or a coach's decision while
#: two or more is usually an injury with a ramp behind it. It is a LABEL
#: boundary for a flag, not a magnitude: nothing scales by it.
RETURNING_THRESHOLD = 2

#: Window for the shifted baseline minutes. Five appearances, matching the
#: existing `{STAT}_L5` family so the baseline is comparable with the columns
#: it is meant to qualify.
BASELINE_WINDOW = 5

EMITTED = (
    "RET_GAMES_MISSED",
    "RET_IS_RETURNING",
    "RET_FIRST_APPEARANCE_FOR_TEAM",
    "RET_BASELINE_MIN_L5",
)


def _abstain(out: pd.DataFrame, reason: str) -> pd.DataFrame:
    out["RET_GAMES_MISSED"] = np.nan
    out["RET_IS_RETURNING"] = np.nan
    out["RET_FIRST_APPEARANCE_FOR_TEAM"] = np.nan
    out["RET_BASELINE_MIN_L5"] = np.nan
    out["RET_STATUS"] = f"DATA_NOT_AVAILABLE: {reason}"
    logger.info("return_ramp abstained: %s", reason)
    return out


def attach_return_ramp_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the return-from-absence columns.

    ``RET_GAMES_MISSED`` -- his team's games played between his previous
    appearance for that team and this one. 0 means he played his team's last
    game. NaN on his first appearance for the team, where there is no prior
    appearance to measure from.

    ``RET_IS_RETURNING`` -- 1.0 when ``RET_GAMES_MISSED >= 2``, else 0.0; NaN
    wherever the count is NaN. A flag and a count rather than a multiplier,
    for the reason the module docstring gives.

    ``RET_FIRST_APPEARANCE_FOR_TEAM`` -- 1.0 on the row where the count is
    undefined. Emitted so that a NaN count can be told apart from a missing
    input: a model handed only the count cannot distinguish "no prior game"
    from "this layer abstained".

    ``RET_BASELINE_MIN_L5`` -- mean minutes over his previous five
    appearances, SHIFTED, so tonight's minutes are excluded.
    """
    out = df.copy()
    missing = [c for c in REQUIRED if c not in out.columns]
    if missing:
        return _abstain(out, f"need {', '.join(missing)}")

    # An unparseable GAME_DATE is a handled input, not a surprise, so the
    # format-inference warning pandas emits for one is suppressed rather than
    # left to print on every abstention.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        dates = pd.to_datetime(out["GAME_DATE"], errors="coerce")
    if dates.isna().all():
        return _abstain(out, "GAME_DATE parsed to NaT on every row")

    work = pd.DataFrame({
        "_idx": np.arange(len(out)),
        "player": out["PLAYER_ID"].to_numpy(),
        "team": out["TEAM_ABBREVIATION"].astype("string").to_numpy(),
        "game": out["GAME_ID"].astype("string").to_numpy(),
        "date": dates.to_numpy(),
        "min": pd.to_numeric(out["MIN"], errors="coerce").to_numpy(),
    })

    # THE TEAM'S SCHEDULE, built from the panel itself. One row per
    # (team, game), ordered by date, numbered from 0. A team's game that no
    # player row covers does not exist for this panel and cannot be counted as
    # missed by anyone -- which is correct: the evidence for it is absent.
    schedule = (
        work.loc[:, ["team", "game", "date"]]
        .drop_duplicates(subset=["team", "game"])
        .sort_values(["team", "date", "game"], kind="mergesort")
    )
    schedule["team_game_no"] = schedule.groupby("team", sort=False).cumcount()
    work = work.merge(
        schedule.loc[:, ["team", "game", "team_game_no"]],
        on=["team", "game"], how="left", validate="many_to_one",
    )

    # ONE ROW PER APPEARANCE, and the gap computed on THAT, not on the panel
    # rows.
    #
    # THE BUG THIS EXISTS TO PREVENT, found by probing the layer with
    # real-world data shapes rather than the clean synthetic panels the tests
    # were built from. When a player has DUPLICATE (player, team, game) rows --
    # a data defect, but one this panel can carry -- computing the gap directly
    # over panel rows differences a team-game number against ITSELF. The
    # subtraction then yields 0 - 1 = -1, and the layer emitted
    # RET_GAMES_MISSED = -1.0: a negative count of games missed, which is
    # meaningless, does not crash, and would have travelled into the A/B
    # feature matrix as a plausible-looking number. Nineteen passing tests did
    # not see it, because every one of them supplied one row per appearance.
    #
    # Deduplicating first makes the arithmetic correct AND gives both duplicate
    # rows the same honest value, rather than one right answer and one
    # nonsense one. `min` is averaged across duplicates: they are meant to be
    # the same game, so if they disagree the mean is the least-committal
    # reading, and the alternative (taking whichever sorted first) would make
    # the baseline depend on row order.
    appearances = (
        work.loc[:, ["player", "team", "team_game_no", "min"]]
        .groupby(["player", "team", "team_game_no"], sort=False, as_index=False)
        .agg(min=("min", "mean"))
        .sort_values(["player", "team", "team_game_no"], kind="mergesort")
    )

    previous = appearances.groupby(["player", "team"], sort=False)["team_game_no"].shift(1)
    missed = appearances["team_game_no"] - previous - 1.0
    first_for_team = previous.isna()

    # Negative is impossible now: team_game_no strictly increases within a
    # player's distinct appearances for one team.
    #
    # SO THIS BRANCH IS UNREACHABLE, and that is stated rather than implied.
    # Deleting it leaves all 24 tests passing -- it was mutation-tested and
    # confirmed dead -- so it is NOT a guard any test covers. It stays as a
    # runtime net on a path that runs on every slate: if a future change
    # reintroduces a duplicate into this frame, NaN ("cannot tell") is the
    # right output and -1 is not. A reader should not mistake it for a
    # behaviour under test.
    if (missed.dropna() < 0).any():
        logger.warning(
            "return_ramp computed %d negative games-missed values; emitting "
            "NaN for them rather than a meaningless count",
            int((missed.dropna() < 0).sum()),
        )
        missed = missed.where(missed >= 0, np.nan)

    appearances["RET_GAMES_MISSED"] = missed.where(~first_for_team, np.nan)
    appearances["RET_FIRST_APPEARANCE_FOR_TEAM"] = first_for_team.astype(float)
    appearances["RET_IS_RETURNING"] = np.where(
        first_for_team | appearances["RET_GAMES_MISSED"].isna(), np.nan,
        (appearances["RET_GAMES_MISSED"] >= RETURNING_THRESHOLD).astype(float),
    )

    # SHIFTED, per CLAUDE.md rule 1, and shifted INSIDE the group.
    #
    # THE OTHER BUG THIS SPELLING EXISTS TO PREVENT, which the first version of
    # this line had. `groupby(...)["min"].shift(1).rolling(W).mean()` groups
    # only the shift: `.rolling` then runs down the whole Series and its window
    # crosses from one player into the next. On the smoke sample that gave the
    # second player a 30.0 baseline on his FIRST appearance, borrowed from the
    # previous player's rows, where the honest answer is NaN. `transform` with
    # both steps in the callable keeps the window inside the group.
    #
    # Computed on appearances, so a duplicated game counts once in the window
    # instead of twice.
    appearances["RET_BASELINE_MIN_L5"] = (
        appearances.groupby(["player", "team"], sort=False)["min"]
        .transform(
            lambda s: s.shift(1).rolling(BASELINE_WINDOW, min_periods=1).mean()
        )
        .to_numpy()
    )

    work = work.merge(
        appearances.loc[:, ["player", "team", "team_game_no", *EMITTED]],
        on=["player", "team", "team_game_no"], how="left", validate="many_to_one",
    )
    restored = work.sort_values("_idx", kind="mergesort")
    for column in EMITTED:
        out[column] = restored[column].to_numpy()
    out["RET_STATUS"] = "OK"
    return out
