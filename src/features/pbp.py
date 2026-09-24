"""
src/features/pbp.py — player features from play-by-play event logs.

WHAT PLAY-BY-PLAY ADDS. A box score says a player took 14 shots and made 6.
It does not say whether those were rim attempts or long twos, whether he
created them or was set up, whether they came in a tied game or in garbage
time, or how fast his team played while he was on the floor. Those are
different players with the same line, and they project differently.

THE LEAKAGE RULE, WHICH IS THE WHOLE DESIGN. Play-by-play describes what
happened DURING a game. Every number here is therefore computed per
player-game and then rolled forward with a shift of one, exactly like the
box-score features: a row's pbp features describe that player's PREVIOUS
games and never his current one. ``attach_pbp_rolling_features`` does the
shifting; ``summarise_player_games`` deliberately returns same-game values
and is not safe to join directly.

COVERAGE IS A REAL LIMIT. The logs supplied cover 2025-26 only. A feature
that exists in the validation window and nowhere earlier is not a feature,
it is the shape of a leak, and compare_models_on_panel now refuses one.
Anything built here can only be trained and evaluated inside the seasons the
logs actually cover.

THE LOG IS COMPLETE, AND THAT WAS CHECKED. Against the box score on the
1,220 games it shares with the panel:

  - 99.90% of player-games match the box score's field-goal attempts
    exactly, at a mean absolute difference of 0.001 attempts.
  - 1,211 of 1,220 games agree exactly on total attempts; the median
    per-game difference is zero.

An earlier read of this data found it 3% short and concluded the events
were randomly sub-sampled. That was wrong: one part of the event log had
not yet been supplied. With all ten parts the parts are disjoint (zero
duplicates on (gameId, actionNumber)) and the log is whole. The lesson
kept rather than the conclusion: check a derived stream against an
independent measurement of the same thing before building on it.

ON-COURT TIME IS RECONSTRUCTED, AND VALIDATED. The substitution stream
gives seconds on court at a correlation of 0.9970 with the box score's own
minutes, a mean absolute error of 0.31 minutes, and 97.7% of player-games
within two minutes. The starter assumption below therefore holds.

WHAT IS SHIPPED, AND WHY NOT EVERYTHING. PBP_SECONDS_ON_COURT is NOT a
feature: it correlates with the box score's MIN at 0.997, and MIN_L5 and
MIN_SEASON are already among the strongest columns in the panel. Shipping
it would hand the model one number twice -- the same collinearity mistake
that cost a third of the opponent-defence layer's gain. PBP_FGA stays a
diagnostic: it is a count, and it is what the completeness check above is
computed from.

PBP_GAME_PACE IS A GAME CONSTANT, NOT A PLAYER MEASUREMENT. It was once
called PBP_PACE_ON_COURT and documented as "possessions per 48 while a given
player is on the floor". That was false. The player's own seconds cancel out
of the arithmetic exactly -- see the derivation at the computation site --
so every player in a game receives the identical value, verified at a
within-game standard deviation of 0.0. It is shipped as what it actually is:
the pace of the game the player appeared in, which is real information the
box score does not carry per game, and which becomes player-specific only
once it is rolled over each player's own schedule.

It is also NOT independent of the panel's existing pace columns:
PBP_GAME_PACE_L10 correlates 0.82 with PACE_ROLL and 0.78 with
PACE_MULTIPLIER. That is the collinearity this module warns about two
paragraphs up, and it is the reason this column should be measured against
PACE_ROLL rather than assumed additive.

RESEARCH ONLY.
"""

from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SOURCE_NAME = "pbp"

REGULATION_PERIOD_SECONDS = 720
OVERTIME_PERIOD_SECONDS = 300

# Shot zones, in feet. The boundaries are the standard ones: the restricted
# area runs to about 4 ft, the paint to about 14, and the arc sits at 22 in
# the corners and 23.75 above the break.
RIM_MAX_FT = 4.0
MID_MIN_FT = 14.0

SHOT_ACTIONS = ("2pt", "3pt")

# Columns prepare_events adds. Their presence marks a prepared frame.
_DERIVED_COLS = frozenset({"clock_seconds", "elapsed", "margin_abs"})

_CLOCK_RE = re.compile(r"PT(\d+)M([\d.]+)S")

# Shipped as features: rates and pace, none of them a count and none of them
# a restatement of a column the box score already carries. These get rolled
# and joined.
PBP_RATE_COLS: tuple[str, ...] = (
    "PBP_GAME_PACE",
    "PBP_SHOT_DIST_AVG",
    "PBP_RIM_RATE",
    "PBP_MID_RATE",
    "PBP_THREE_RATE",
    "PBP_DUNK_LAYUP_RATE",
    "PBP_ASSISTED_RATE",
    "PBP_CLOSE_SHOT_SHARE",
    "PBP_GARBAGE_SHOT_SHARE",
)

# Computed for validation and diagnostics, never offered as features:
# PBP_FGA is a count, and the on-court columns are a noisier restatement of
# the box score's own MIN.
PBP_DIAGNOSTIC_COLS: tuple[str, ...] = (
    "PBP_FGA",
    # r = 0.997 with the box score's MIN, which is already a feature.
    "PBP_SECONDS_ON_COURT",
    # Seconds times pace; carries nothing the two of them do not.
    "PBP_TEAM_POSS_ON_COURT",
)

PBP_GAME_COLS: tuple[str, ...] = (*PBP_RATE_COLS, *PBP_DIAGNOSTIC_COLS)


class PbpFeatureError(ValueError):
    """Raised when play-by-play features cannot be built from what was given."""


def parse_clock_seconds(value) -> float:
    """``PT12M00.00S`` -> 720.0 seconds remaining. Unparseable -> NaN."""
    if not isinstance(value, str):
        return float("nan")
    m = _CLOCK_RE.fullmatch(value.strip())
    if not m:
        return float("nan")
    return float(m.group(1)) * 60.0 + float(m.group(2))


def elapsed_seconds(period: pd.Series, clock_remaining: pd.Series) -> pd.Series:
    """Seconds since tip-off, across regulation and overtime periods."""
    period = pd.to_numeric(period, errors="coerce")
    # Time in the periods already COMPLETED. Counting from (period - 1)
    # regulation periods works until overtime, then double-counts: the fourth
    # quarter is 720 seconds and the first overtime is 300, so period 5 begins
    # at 2880 rather than at 2460.
    completed = period - 1
    regulation_done = completed.clip(upper=4)
    overtime_done = (completed - 4).clip(lower=0)
    started = (
        regulation_done * REGULATION_PERIOD_SECONDS
        + overtime_done * OVERTIME_PERIOD_SECONDS
    )
    period_length = np.where(
        period <= 4, REGULATION_PERIOD_SECONDS, OVERTIME_PERIOD_SECONDS
    )
    return started + (period_length - clock_remaining)


def prepare_events(pbp: pd.DataFrame) -> pd.DataFrame:
    """Normalise the raw event log: ids as text, a clock in seconds, margin."""
    required = {"gameId", "actionType", "period", "clock"}
    missing = required - set(pbp.columns)
    if missing:
        raise PbpFeatureError(
            f"DATA_NOT_AVAILABLE: event log missing {sorted(missing)}"
        )

    # Idempotent: preparing an already-prepared frame returns it unchanged
    # rather than copying it again. Callers legitimately prepare once and
    # hand the result to several functions that each prepare defensively,
    # and on a full multi-season log each redundant copy is gigabytes.
    if _DERIVED_COLS.issubset(pbp.columns) and not pd.api.types.is_numeric_dtype(
        pbp["gameId"]
    ):
        return pbp

    ev = pbp.copy()
    ev["gameId"] = ev["gameId"].astype(str)
    for col in ("personId", "teamId", "assistPersonId"):
        if col in ev.columns:
            ev[col] = (
                ev[col].astype("string").str.strip()
                .str.replace(r"\.0$", "", regex=True)
            )
    ev["clock_seconds"] = ev["clock"].map(parse_clock_seconds)
    ev["elapsed"] = elapsed_seconds(ev["period"], ev["clock_seconds"])
    for col in ("scoreHome", "scoreAway"):
        if col in ev.columns:
            ev[col] = pd.to_numeric(ev[col], errors="coerce")
    if {"scoreHome", "scoreAway"}.issubset(ev.columns):
        ev["margin_abs"] = (ev["scoreHome"] - ev["scoreAway"]).abs()
    else:
        ev["margin_abs"] = np.nan
    if "orderNumber" in ev.columns:
        ev = ev.sort_values(["gameId", "orderNumber"])
    else:
        ev = ev.sort_values(["gameId", "period", "elapsed"])
    return ev.reset_index(drop=True)


def summarise_shots(events: pd.DataFrame) -> pd.DataFrame:
    """
    Per (game, player) shot profile. SAME-GAME values — roll before joining.
    """
    shots = events[events["actionType"].isin(SHOT_ACTIONS)].copy()
    if shots.empty:
        return pd.DataFrame(columns=["gameId", "personId"])

    dist = pd.to_numeric(shots["shotDistance"], errors="coerce")
    shots["_dist"] = dist
    shots["_is_three"] = (shots["actionType"] == "3pt").astype(float)
    shots["_is_rim"] = (dist <= RIM_MAX_FT).astype(float)
    # Mid-range is a two-pointer from outside the restricted area.
    shots["_is_mid"] = (
        (shots["actionType"] == "2pt") & (dist > RIM_MAX_FT) & (dist >= MID_MIN_FT)
    ).astype(float)
    sub = shots.get("subType", pd.Series(index=shots.index, dtype="object"))
    sub = sub.astype("string").str.lower().fillna("")
    shots["_is_dunk_layup"] = sub.str.contains("layup|dunk", regex=True).astype(float)

    made = shots["shotResult"].astype("string").str.lower().eq("made")
    shots["_made"] = made.astype(float)
    assisted = (
        shots["assistPersonId"].notna() if "assistPersonId" in shots.columns
        else pd.Series(False, index=shots.index)
    )
    shots["_assisted_make"] = (made & assisted).astype(float)

    margin = shots["margin_abs"]
    shots["_close"] = (margin <= 5).astype(float).where(margin.notna())
    shots["_garbage"] = (margin >= 20).astype(float).where(margin.notna())

    grouped = shots.groupby(["gameId", "personId"], dropna=True)
    out = grouped.agg(
        PBP_FGA=("_dist", "size"),
        PBP_SHOT_DIST_AVG=("_dist", "mean"),
        PBP_RIM_RATE=("_is_rim", "mean"),
        PBP_MID_RATE=("_is_mid", "mean"),
        PBP_THREE_RATE=("_is_three", "mean"),
        PBP_DUNK_LAYUP_RATE=("_is_dunk_layup", "mean"),
        PBP_CLOSE_SHOT_SHARE=("_close", "mean"),
        PBP_GARBAGE_SHOT_SHARE=("_garbage", "mean"),
        _makes=("_made", "sum"),
        _assisted=("_assisted_make", "sum"),
    ).reset_index()
    # Share of a player's MAKES that a teammate set up. A self-creator sits
    # low here; a spot-up shooter sits high. Undefined without a make, and
    # left undefined rather than called zero.
    out["PBP_ASSISTED_RATE"] = np.where(
        out["_makes"] > 0, out["_assisted"] / out["_makes"], np.nan
    )
    return out.drop(columns=["_makes", "_assisted"])


def reconstruct_on_court(events: pd.DataFrame) -> pd.DataFrame:
    """
    Seconds on court per (game, player), from substitution events.

    A player is treated as on court from tip unless his first substitution is
    an "in". That is the only assumption available -- the logs carry no
    starting-lineup record per event -- and it is CHECKED rather than
    trusted: validate_on_court_against_minutes compares the result to the box
    score's own minutes.
    """
    subs = events[events["actionType"] == "substitution"].copy()
    if subs.empty or "subType" not in subs.columns:
        return pd.DataFrame(columns=["gameId", "personId", "PBP_SECONDS_ON_COURT"])

    subs["_dir"] = subs["subType"].astype("string").str.lower().str.strip()
    subs = subs[subs["_dir"].isin(["in", "out"])]
    subs = subs.dropna(subset=["personId", "elapsed"])

    game_end = events.groupby("gameId")["elapsed"].max()

    rows: list[dict] = []
    for (game_id, person), block in subs.groupby(["gameId", "personId"], sort=False):
        block = block.sort_values("elapsed")
        end_of_game = float(game_end.get(game_id, np.nan))
        if not np.isfinite(end_of_game):
            continue
        # Started the game if the first substitution involving him is an "out".
        on_since = 0.0 if block["_dir"].iloc[0] == "out" else None
        total = 0.0
        for _, event in block.iterrows():
            when = float(event["elapsed"])
            if event["_dir"] == "in":
                if on_since is None:
                    on_since = when
            elif on_since is not None:
                total += max(0.0, when - on_since)
                on_since = None
        if on_since is not None:
            total += max(0.0, end_of_game - on_since)
        rows.append({
            "gameId": game_id, "personId": person, "PBP_SECONDS_ON_COURT": total,
        })
    return pd.DataFrame(rows)


def validate_on_court_against_minutes(
    on_court: pd.DataFrame, panel: pd.DataFrame
) -> dict[str, float]:
    """
    Compare reconstructed seconds to the box score's own minutes.

    The reconstruction rests on an assumption about who started. This is how
    that assumption is tested rather than believed: the two numbers measure
    the same thing from independent sources and should agree closely.
    """
    if on_court.empty:
        return {"n": 0.0}
    work = panel.copy()
    work["gameId"] = work["GAME_ID"].astype(str)
    work["personId"] = work["PLAYER_ID"].astype(str)
    merged = on_court.merge(
        work[["gameId", "personId", "MIN"]], on=["gameId", "personId"], how="inner"
    ).dropna(subset=["MIN", "PBP_SECONDS_ON_COURT"])
    if merged.empty:
        return {"n": 0.0}
    recon = merged["PBP_SECONDS_ON_COURT"] / 60.0
    box = pd.to_numeric(merged["MIN"], errors="coerce")
    err = (recon - box).abs()
    return {
        "n": float(len(merged)),
        "corr": float(np.corrcoef(recon, box)[0, 1]),
        "mean_abs_error_minutes": float(err.mean()),
        "median_abs_error_minutes": float(err.median()),
        "within_2_minutes": float((err <= 2).mean()),
    }


def team_possessions(events: pd.DataFrame) -> pd.DataFrame:
    """
    Possession changes per game, with the elapsed time of each.

    The ``possession`` column names the team in possession at each event, so a
    possession ends where that value changes. This is the event log's own
    account rather than the box-score estimate.
    """
    if "possession" not in events.columns:
        return pd.DataFrame(columns=["gameId", "elapsed"])
    ev = events.dropna(subset=["elapsed"]).copy()
    ev["possession"] = (
        ev["possession"].astype("string").str.replace(r"\.0$", "", regex=True)
    )
    ev = ev[ev["possession"].notna() & (ev["possession"] != "0")]
    changed = ev["possession"].ne(ev["possession"].shift()) | ev["gameId"].ne(
        ev["gameId"].shift()
    )
    return ev.loc[changed, ["gameId", "elapsed"]].reset_index(drop=True)


def check_log_completeness(
    events: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    min_exact_share: float = 0.80,
    max_median_gap: float = 2.0,
) -> dict[str, float]:
    """
    Compare the event log's shot count to the box score's, per season.

    Two independent counts of the same thing. They should agree; where they
    do not, the log is missing events and every RATE built from it is biased
    by whatever was dropped.

    This is not hypothetical. A 2025-26 log supplied as nine of ten parts
    measured 3% short and looked like random sub-sampling; the tenth part
    made it whole. A 2023-24 log supplied as five of about eleven parts was
    short by a median of 88 attempts in every one of its 1,164 games --
    roughly half of each game -- while still naming every game, so nothing
    about its shape said "partial" except this check.

    Returns one row of numbers per season. Seasons that fail are named in
    ``failing`` so a caller can exclude them rather than average over them.
    """
    if "GAME_ID" not in panel.columns or "FGA" not in panel.columns:
        raise PbpFeatureError(
            "DATA_NOT_AVAILABLE: panel needs GAME_ID and FGA to check the event "
            "log against an independent count"
        )
    # Cast here rather than trusting the caller to have run prepare_events.
    # A raw CSV read gives gameId as int64 when the ids carry no leading zero
    # and as str when they do, so the merge below would fail on dtype for
    # some seasons and succeed for others. This check exists to be run on a
    # log BEFORE anything else touches it; it cannot presuppose preparation.
    shot_games = events.loc[
        events["actionType"].isin(SHOT_ACTIONS), "gameId"
    ].astype(str)
    shots = shot_games.value_counts().rename("pbp_fga")
    work = panel.copy()
    work["gameId"] = work["GAME_ID"].astype(str)
    season = work["SEASON"] if "SEASON" in work.columns else pd.Series("all", index=work.index)
    box = work.groupby(["gameId", season])["FGA"].sum().rename("box_fga").reset_index()
    box.columns = ["gameId", "season", "box_fga"]
    joined = box.merge(shots, left_on="gameId", right_index=True, how="inner")
    if joined.empty:
        raise PbpFeatureError(
            "DATA_NOT_AVAILABLE: no game appears in both the event log and the panel"
        )
    joined["gap"] = joined["box_fga"] - joined["pbp_fga"]

    report: dict[str, dict[str, float]] = {}
    failing: list[str] = []
    for name, block in joined.groupby("season"):
        exact = float((block["gap"] == 0).mean())
        median_gap = float(block["gap"].median())
        report[str(name)] = {
            "games": float(len(block)),
            "exact_share": exact,
            "median_gap": median_gap,
        }
        if exact < min_exact_share or median_gap > max_median_gap:
            failing.append(str(name))
            logger.warning(
                "pbp log for %s is INCOMPLETE: %.1f%% of %d games match the box "
                "score exactly, median shortfall %.0f attempts. Rates built from "
                "it are biased by whatever is missing. Supply the remaining "
                "parts before using this season.",
                name, 100 * exact, len(block), median_gap,
            )
        else:
            logger.info(
                "pbp log for %s: %.1f%% of %d games match the box score exactly, "
                "median gap %.0f.", name, 100 * exact, len(block), median_gap,
            )
    return {"seasons": report, "failing": failing}


def summarise_player_games(
    pbp: pd.DataFrame,
    panel: pd.DataFrame | None = None,
    *,
    require_complete: bool = False,
) -> pd.DataFrame:
    """
    One row per (game, player) of SAME-GAME play-by-play summaries.

    Not safe to join to a panel directly — see this module's docstring. Pass
    the result to attach_pbp_rolling_features, which shifts it.
    """
    events = prepare_events(pbp)
    shots = summarise_shots(events)
    on_court = reconstruct_on_court(events)

    out = shots.merge(on_court, on=["gameId", "personId"], how="outer")

    # team_possessions counts every change of the possessing team, so its
    # per-game total covers BOTH teams. Halve it for the league's convention:
    # possessions per team, which puts pace near 100 rather than near 200.
    poss = team_possessions(events)
    if not poss.empty and "PBP_SECONDS_ON_COURT" in out.columns:
        per_game = (poss.groupby("gameId").size() / 2.0).rename("_team_poss")
        length = events.groupby("gameId")["elapsed"].max().rename("_game_seconds")
        out = out.merge(per_game, left_on="gameId", right_index=True, how="left")
        out = out.merge(length, left_on="gameId", right_index=True, how="left")

        # The player's share of the game's clock. Without lineup intervals per
        # possession this apportions the team's possessions by time on court
        # rather than counting his own, and it is named for what it is.
        share = out["PBP_SECONDS_ON_COURT"] / out["_game_seconds"].where(
            out["_game_seconds"] > 0
        )
        out["PBP_TEAM_POSS_ON_COURT"] = out["_team_poss"] * share

        # PACE IS A GAME CONSTANT. Substituting the line above:
        #
        #   pace = TEAM_POSS_ON_COURT / (player_sec / 2880)
        #        = team_poss * (player_sec / game_sec) * 2880 / player_sec
        #        = team_poss * 2880 / game_sec
        #
        # player_sec cancels. Every player in a game gets the same number, so
        # this is written the short way, which is both cheaper and honest
        # about what it measures. Deriving it from PBP_SECONDS_ON_COURT made
        # it look player-specific when it never was.
        out["PBP_GAME_PACE"] = np.where(
            out["_game_seconds"] > 0,
            out["_team_poss"] * 2880.0 / out["_game_seconds"],
            np.nan,
        )
        out = out.drop(columns=["_team_poss", "_game_seconds"])

    for col in PBP_GAME_COLS:
        if col not in out.columns:
            out[col] = np.nan

    if panel is not None:
        completeness = check_log_completeness(events, panel)
        if completeness["failing"] and require_complete:
            raise PbpFeatureError(
                f"DATA_NOT_AVAILABLE: the event log is incomplete for "
                f"{completeness['failing']}. Every rate built from it would be "
                "biased by the missing events."
            )
        out.attrs["log_completeness"] = completeness
        report = validate_on_court_against_minutes(on_court, panel)
        if report.get("n"):
            logger.info(
                "pbp on-court reconstruction checked against box-score minutes: "
                "n=%d corr=%.4f mean|err|=%.2f min, %.1f%% within 2 minutes",
                int(report["n"]), report["corr"],
                report["mean_abs_error_minutes"], 100 * report["within_2_minutes"],
            )
    logger.info(
        "pbp summaries: %d (game, player) rows across %d games",
        len(out), out["gameId"].nunique(),
    )
    return out


def attach_pbp_rolling_features(
    panel: pd.DataFrame,
    player_games: pd.DataFrame,
    *,
    windows: tuple[int, ...] = (5, 10),
    min_periods: int = 2,
) -> pd.DataFrame:
    """
    Join SHIFTED rolling means of the pbp summaries onto the panel.

    A row receives what the player's previous games looked like. The current
    game never contributes to its own features: the shift happens before the
    rolling window, so the first game of a career is null rather than
    self-describing.
    """
    needed = {"PLAYER_ID", "GAME_ID", "GAME_DATE"}
    missing = needed - set(panel.columns)
    if missing:
        raise PbpFeatureError(f"DATA_NOT_AVAILABLE: panel missing {sorted(missing)}")
    if player_games.empty:
        logger.info("pbp layer skipped: no player-game summaries.")
        return panel

    out = panel.copy()
    out["GAME_ID"] = out["GAME_ID"].astype(str)
    out["_pid"] = out["PLAYER_ID"].astype(str)

    src = player_games.rename(columns={"gameId": "GAME_ID", "personId": "_pid"}).copy()
    src["GAME_ID"] = src["GAME_ID"].astype(str)
    src["_pid"] = src["_pid"].astype(str)

    # Rates only. See PBP_DIAGNOSTIC_COLS for what is deliberately excluded.
    value_cols = [c for c in PBP_RATE_COLS if c in src.columns]
    # One summary row per (game, player), or the left join fans out and every
    # panel row silently becomes several. attach_defense_features guards the
    # same way; a duplicated key is a data fault, not something to average
    # over quietly.
    duplicated = int(src.duplicated(subset=["GAME_ID", "_pid"]).sum())
    if duplicated:
        raise PbpFeatureError(
            f"player_games has {duplicated} duplicate (game, player) row(s). "
            "Joining them would multiply panel rows and misalign every column "
            "assigned positionally afterwards."
        )

    merged = out.merge(
        src[["GAME_ID", "_pid", *value_cols]], on=["GAME_ID", "_pid"], how="left"
    )
    if len(merged) != len(out):
        raise PbpFeatureError(
            f"the pbp join changed the row count ({len(out)} -> {len(merged)})"
        )
    # Remember the caller's order. Rolling needs the rows in player-then-date
    # order, but returning them that way silently reorders the panel: the
    # values stay attached to their own rows, so nothing is mis-joined, and
    # then any caller that assigns a column positionally afterwards corrupts
    # every row. The original order is restored before returning.
    merged.index = out.index
    original_order = merged.index
    merged = merged.sort_values(["_pid", "GAME_DATE"])

    grouped = merged.groupby("_pid", sort=False)
    created: list[str] = []
    for col in value_cols:
        for window in windows:
            name = f"{col}_L{window}"
            merged[name] = grouped[col].transform(
                lambda s, w=window: s.shift(1).rolling(w, min_periods=min_periods).mean()
            )
            created.append(name)

    # The same-game columns are dropped: they describe THIS game and would be
    # postgame information if any feature list ever named one.
    drop = [c for c in (*PBP_GAME_COLS, "_pid") if c in merged.columns]
    merged = merged.drop(columns=drop).reindex(original_order)
    covered = (
        float(merged[created[0]].notna().mean()) if created else 0.0
    )
    logger.info(
        "pbp layer: %d rolling column(s) attached, %.1f%% of rows have a prior-game "
        "value. Same-game pbp columns are dropped, not carried.",
        len(created), 100 * covered,
    )
    return merged
