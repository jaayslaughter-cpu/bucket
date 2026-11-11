"""
src/features/stints.py — lineup stints from the ESPN plays we already store.

WHAT A STINT IS. One (game, team, lineup, period) row: the five players on
the floor for one team, and how many seconds they were out there together.
A box score says a player took 14 shots; a stint says whether those came
with the starters or with the second unit, which is a different player with
the same line and projects differently.

THE CHEAPEST REMAINING PLAYER DATA. The summary payload the panel already
fetches carries the whole play feed, and `espn_game.parse_plays` had already
parsed the participants, the substitutions (type 584, entering-then-leaving)
and the coordinates before the feed was discarded. This module reads the
rows `espn_plays` and `espn_starters` now hold (migration 011) and rebuilds
the floor state from them. No new HTTP, no new dependency, no new source.

THE FLOOR-STATE INVARIANT, WHICH IS THE WHOLE GUARD. A substitution says who
CHANGED; the floor state before it is what the change is applied to. The
invariant is: the LEAVING player must already be on the floor, and the
ENTERING one must not be. If it fails, the feed is not describing a
basketball game -- ESPN's ordering has reversed, or a play has been dropped,
or the starters are wrong -- and a stint built on top of it would have four
or six players and be worse than a stint that was never built. A violation
is counted per game and the game is REFUSED: one bad game must not
contaminate the stints of the other eleven on the night. This is the check
`PlayRow.substitution` names as belonging to whatever tracks floor state,
and this is that thing.

CLOCK IS "M:SS REMAINING", not elapsed. The ESPN feed's `clock.displayValue`
counts DOWN within a period; the seconds of a stint are the difference
between two clock readings, and the period boundary is handled by the
period column, which is why the stint is keyed on (game, team, lineup,
period) and never crosses one.

NOT A MODEL FEATURE, ON PURPOSE. The A/B rule: a new column reaches the
model's matrix by winning a holdout in `scripts/feature_ab.py`, not by
being argued for. This module writes its output to a CSV on the state
root where that measurement can read it, and stops there.

RESEARCH_ONLY. Stored events and derived aggregates. Nothing here prices
or places anything.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.ingestion.espn_game import ESPN_SUBSTITUTION_TYPE_ID
from src.ingestion.espn_panel import espn_player_id

logger = logging.getLogger(__name__)

SOURCE_NAME = "espn_stints"

#: ``PT12M00.00S`` is the event log's form; the ESPN play feed's is
#: ``12:00`` -- EXCEPT under one minute, where it drops the minute entirely
#: and sends ``48.6``, ``13.4`` or ``0.0``. Both forms are counted-down
#: clocks; the period is a separate column.
#:
#: THE SECOND FORM WAS MISSED AND IT COST HALF A GAME. A colon-only pattern
#: left 2 of 58 substitutions in the 2026-11-09 Bulls game unparseable, and an
#: unparseable clock was `continue`d -- so the stint builder skipped the
#: substitution while `_floor_state_invariant_report`, which never reads the
#: clock, applied it. The two then disagreed about who was on the floor: the
#: next real substitution found its `leaving` player already absent, removed
#: nobody, appended the entering one, and produced a SIX-player lineup. The
#: five-player backstop then refused that whole team -- 48 of 476 plays in
#: that game carry the minute-less form, and one of its two teams produced no
#: stints at all while the other validated cleanly, which is how a
#: box-score-minutes check can look healthy on a half-built game.
_CLOCK_RE = re.compile(r"^(?:(\d+):)?([\d.]+)$")

REGULATION_PERIOD_SECONDS = 720
OVERTIME_PERIOD_SECONDS = 300

#: How many players a basketball team puts on the floor at once. Used as a
#: SANITY LIMIT: a lineup that is not five players is not a lineup, and the
#: invariant check exists so this never has to fire. It fires anyway as a
#: backstop, because a stint row with four or six players would look like a
#: feature to anything that joined on it.
PLAYERS_ON_FLOOR = 5


class StintError(ValueError):
    """Raised when stints cannot be built from what was given."""


def lineup_is_valid(lineup: str) -> bool:
    """Five DISTINCT players, which is the only thing a lineup can be.

    A FUNCTION SO THERE IS ONE DEFINITION AND IT CAN BE TESTED. Inline, this
    predicate was shadowed by the floor-state invariant -- the invariant
    rejects every input that would reach it, so no behavioural test could
    show the predicate itself was wrong, and it WAS wrong: it counted the
    semicolon-separated parts rather than the distinct players, so
    ``p1;p1;p3;p4;p5`` passed as a five.
    """
    players = str(lineup).split(";")
    return len(set(players)) == PLAYERS_ON_FLOOR and all(players)


def parse_clock_seconds(value) -> float:
    """``"12:00"`` -> 720.0 and ``"48.6"`` -> 48.6 seconds remaining.

    Both ESPN forms, because the feed uses both and the minute-less one
    appears on every play inside the last minute of a period. Unparseable
    still returns NaN -- see `_build_game_stints` for why that is now a
    REFUSAL rather than a skipped substitution.
    """
    if not isinstance(value, str):
        return float("nan")
    m = _CLOCK_RE.fullmatch(value.strip())
    if not m:
        return float("nan")
    minutes = float(m.group(1)) if m.group(1) is not None else 0.0
    try:
        return minutes * 60.0 + float(m.group(2))
    except ValueError:
        # "12:" and ".": the pattern admits them, float does not.
        return float("nan")


def period_length(period) -> float:
    """A period's own length in seconds; regulation 720, overtime 300."""
    try:
        n = int(period)
    except (TypeError, ValueError):
        return float("nan")
    if n <= 0:
        return float("nan")
    return REGULATION_PERIOD_SECONDS if n <= 4 else OVERTIME_PERIOD_SECONDS


def _floor_state_invariant_report(
    starters: set[str], plays: pd.DataFrame
) -> tuple[int, list[str]]:
    """
    Count the floor-state violations in one game's substitution stream.

    The invariant: at every substitution, the LEAVING player is on the floor
    and the ENTERING one is not. A violation means the stream contradicts
    itself -- a player "entering" who was already out there, or one
    "leaving" who was already on the bench -- and the floor state cannot be
    recovered from it.
    """
    on_floor = set(starters)
    violations = 0
    samples: list[str] = []
    subs_only = plays.loc[
        plays["PLAY_TYPE_ID"].astype(str) == ESPN_SUBSTITUTION_TYPE_ID
    ]
    for _, play in subs_only.iterrows():
        participants = str(play.get("PARTICIPANT_ATHLETE_IDS") or "").split(";")
        if len(participants) != 2:
            continue
        entering, leaving = participants
        if entering == leaving or not entering or not leaving:
            continue
        if leaving not in on_floor or entering in on_floor:
            violations += 1
            if len(samples) < 3:
                samples.append(
                    f"seq {play.get('SEQUENCE_NUMBER')}: entering {entering}, "
                    f"leaving {leaving}"
                )
            # Do not apply an impossible substitution: the state is already
            # unknown, and applying it would spread the corruption.
            continue
        on_floor.discard(leaving)
        on_floor.add(entering)
    return violations, samples


@dataclass
class StintGameResult:
    """What one game's stint reconstruction found, including its refusals."""
    game_id: str
    stints: pd.DataFrame | None
    #: The reconstruction was refused for this game; the reason names why.
    refused_reason: str | None = None
    #: The number of floor-state violations counted before the refusal.
    violations: int = 0
    violation_samples: list[str] | None = None


def _build_game_stints(
    game_id: str,
    team_id: str,
    starters: list[str],
    plays: pd.DataFrame,
) -> StintGameResult:
    """
    One (game, team)'s stints, or a refusal with a reason.

    A stint is one contiguous run of the SAME five players. The lineup is
    the key and the seconds come from the clock: a substitution at 8:12 of
    the first ends the stint that started at 12:00 and begins the next one.
    """
    if len(set(starters)) != PLAYERS_ON_FLOOR:
        return StintGameResult(
            game_id, None,
            refused_reason=(
                f"expected {PLAYERS_ON_FLOOR} starters for team {team_id}, "
                f"got {len(set(starters))}"
            ),
        )
    team_plays = plays.loc[plays["ESPN_TEAM_ID"].astype(str) == str(team_id)]
    if team_plays.empty:
        return StintGameResult(
            game_id, None, refused_reason=f"no plays for team {team_id}",
        )
    on_floor = list(dict.fromkeys(starters))
    violations, samples = _floor_state_invariant_report(
        set(starters), team_plays,
    )
    if violations:
        return StintGameResult(
            game_id, None,
            refused_reason=(
                f"{violations} floor-state violation(s): "
                + "; ".join(samples)
            ),
            violations=violations,
            violation_samples=samples,
        )

    rows: list[dict[str, float | int | str]] = []
    # The clock at which the current lineup took the floor, per period. A
    # period OPENS at its own length -- 720 seconds for regulation, 300 for
    # overtime -- because the first stint of every period is the starters,
    # and a substitution is not what opens it. Missing this made the first
    # stint of every period silently vanish: the opening lineup's seconds
    # were counted from the FIRST SUBSTITUTION rather than from the tip,
    # so every player who never left showed only his bench time.
    opened_at: dict[int, float] = {}
    lineup: tuple[str, ...] = tuple(on_floor)
    # The lineup on the floor AT EACH PERIOD'S END. The closing loop below
    # used the game's FINAL five for every period's closing stint, which
    # credited bench players with minutes they played in no period and
    # robbed the starters of theirs -- a 3.4-minute mean error on the
    # first real game tested. Each period closes with its own five.
    lineup_at_period_end: dict[int, tuple[str, ...]] = {}
    # ONLY SUBSTITUTIONS CHANGE THE FLOOR. The play feed names two
    # participants on far more than substitutions -- a made three lists the
    # shooter and the assister, a steal lists both players -- and the first
    # real game tested read those pairs as substitutions because the
    # participant count was checked and the type id was not. The type id
    # (584) is the discriminator; the participants are only its operands.
    subs_only = team_plays.loc[
        team_plays["PLAY_TYPE_ID"].astype(str) == ESPN_SUBSTITUTION_TYPE_ID
    ]
    # Every period the team played in, from the whole play set: a period
    # with no substitutions still yields its full stint for the starters.
    periods_seen: set[int] = set()
    for value in team_plays.get("PERIOD"):
        try:
            periods_seen.add(int(value))
        except (TypeError, ValueError):
            continue
    for period in sorted(periods_seen):
        opened_at[period] = period_length(period)
    for _, play in subs_only.iterrows():
        period = int(play["PERIOD"]) if pd.notna(play.get("PERIOD")) else None
        if period is None:
            continue
        if period not in opened_at:
            opened_at[period] = period_length(period)
        participants = str(play.get("PARTICIPANT_ATHLETE_IDS") or "").split(";")
        if len(participants) != 2:
            continue
        entering, leaving = participants
        if entering == leaving or not entering or not leaving:
            continue
        clock = parse_clock_seconds(play.get("CLOCK_DISPLAY"))
        if not np.isfinite(clock):
            # REFUSE, DO NOT SKIP. A skipped substitution desynchronises this
            # lineup from the one the invariant check validated, and the next
            # real substitution then appends an entering player without
            # removing anyone -- six on the floor. The backstop catches that
            # and discards the team, so the cost of a `continue` here was
            # half a game's stints reported as a clean build. Without a clock
            # the stint's seconds cannot be computed at all, so there is
            # nothing honest to do but refuse and name it.
            return StintGameResult(
                game_id, None,
                refused_reason=(
                    f"unparseable clock {play.get('CLOCK_DISPLAY')!r} on "
                    f"substitution at sequence {play.get('SEQUENCE_NUMBER')}; "
                    f"the stint's seconds cannot be computed and skipping it "
                    f"would desynchronise the floor state"
                ),
            )
        # Close the stint the previous lineup was on.
        opened = opened_at.get(period)
        if opened is not None and clock < opened:
            seconds = opened - clock
            rows.append({
                "NBA_GAME_ID": game_id,
                "ESPN_TEAM_ID": str(team_id),
                "LINEUP": ";".join(sorted(lineup)),
                "PERIOD": period,
                "STINT_SECONDS": seconds,
            })
        # Apply the substitution and open the next stint.
        lineup = tuple(p for p in lineup if p != leaving) + (entering,)
        opened_at[period] = clock
        lineup_at_period_end[period] = lineup
    # Close the final stint of each period at its end, with the five that
    # was actually on the floor when that period ended.
    for period, opened in opened_at.items():
        length = period_length(period)
        if not np.isfinite(length):
            continue
        closing = lineup_at_period_end.get(period, lineup)
        rows.append({
            "NBA_GAME_ID": game_id,
            "ESPN_TEAM_ID": str(team_id),
            "LINEUP": ";".join(sorted(closing)),
            "PERIOD": period,
            "STINT_SECONDS": max(0.0, opened - 0.0),
        })

    if not rows:
        return StintGameResult(
            game_id, None,
            refused_reason="no substitutions; the starters played the whole game",
        )
    frame = pd.DataFrame(rows)
    # A lineup that is not five DISTINCT players is not a lineup.
    #
    # DISTINCT, NOT JUST FIVE ENTRIES, which the delivered version got wrong.
    # It counted the semicolon-separated parts, so `p1;p1;p3;p4;p5` -- five
    # entries, four players -- passed. That row credits one player twice and
    # another not at all, and anything joining on LINEUP would read it as a
    # real five. It arises from a substitution naming an ENTERING player who
    # is already on the floor: the filter removes the leaver, the append adds
    # a duplicate, and the part count never changes.
    #
    # THIS GUARD AND THE FLOOR-STATE INVARIANT DELIBERATELY OVERLAP, which
    # means neither can be killed by a mutation test on its own -- each
    # catches the other's cases. That is accepted here rather than resolved by
    # deleting one: the invariant is the real check and runs first, and this
    # is what still fires if a later change narrows it. `PLAYERS_ON_FLOOR` is
    # the only number either of them is allowed to know.
    bad = frame.loc[~frame["LINEUP"].apply(lineup_is_valid)]
    if not bad.empty:
        return StintGameResult(
            game_id, None,
            refused_reason=(
                f"{len(bad)} stint(s) with a lineup that is not "
                f"{PLAYERS_ON_FLOOR} players"
            ),
        )
    return StintGameResult(game_id, frame)


def build_stints(plays: pd.DataFrame, starters: pd.DataFrame) -> pd.DataFrame:
    """
    Every (game, team)'s stints from the plays and starters already stored.

    ONE GAME'S REFUSAL IS NOT THE NIGHT'S. A game with no starters, a
    six-player lineup, or a self-contradicting substitution stream is
    refused with a reason and the other games continue -- the same
    failure-isolation rule the panel fetch follows.

    THE COLUMNS ARE THE CONTRACT. ``NBA_GAME_ID`` joins to the panel with no
    translation (both are the ``espn:``-prefixed id), ``LINEUP`` is the five
    player ids sorted and semicolon-joined (stable, so the same five is the
    same key in any order), ``STINT_SECONDS`` is the seconds that lineup was
    on the floor in that period.
    """
    if plays.empty or starters.empty:
        raise StintError(
            "DATA_NOT_AVAILABLE: stints need both espn_plays and espn_starters; "
            "one is empty"
        )
    required_plays = {"NBA_GAME_ID", "SEQUENCE_NUMBER", "ESPN_TEAM_ID",
                      "PLAY_TYPE_ID", "PARTICIPANT_ATHLETE_IDS"}
    missing = required_plays - set(plays.columns)
    if missing:
        raise StintError(f"DATA_NOT_AVAILABLE: plays missing {sorted(missing)}")
    required_starters = {"NBA_GAME_ID", "ESPN_TEAM_ID", "ESPN_ATHLETE_ID"}
    missing = required_starters - set(starters.columns)
    if missing:
        raise StintError(f"DATA_NOT_AVAILABLE: starters missing {sorted(missing)}")

    plays = plays.sort_values(
        ["NBA_GAME_ID", "SEQUENCE_NUMBER"], kind="stable",
    ).reset_index(drop=True)
    frames: list[pd.DataFrame] = []
    refusals: dict[str, str] = {}
    violations: dict[str, int] = {}
    for game_id in plays["NBA_GAME_ID"].astype(str).unique():
        game_plays = plays.loc[plays["NBA_GAME_ID"].astype(str) == game_id]
        game_starters = starters.loc[
            starters["NBA_GAME_ID"].astype(str) == game_id
        ]
        if game_starters.empty:
            refusals[game_id] = "no starters"
            continue
        for team_id in game_starters["ESPN_TEAM_ID"].astype(str).unique():
            team_starters = game_starters.loc[
                game_starters["ESPN_TEAM_ID"].astype(str) == team_id,
                "ESPN_ATHLETE_ID",
            ].astype(str).tolist()
            result = _build_game_stints(game_id, team_id, team_starters, game_plays)
            if result.stints is not None:
                frames.append(result.stints)
            else:
                refusals[f"{game_id}:{team_id}"] = result.refused_reason or "unknown"
                if result.violations:
                    violations[f"{game_id}:{team_id}"] = result.violations

    if refusals:
        logger.warning(
            "STINTS REFUSED %d (game:team) unit(s): %s. A refused game has "
            "no stints rather than wrong ones, which is the honest state.",
            len(refusals), refusals,
        )
    if not frames:
        return pd.DataFrame(columns=[
            "NBA_GAME_ID", "ESPN_TEAM_ID", "LINEUP", "PERIOD", "STINT_SECONDS",
        ])
    return pd.concat(frames, ignore_index=True)


def player_seconds_on_court(stints: pd.DataFrame) -> pd.DataFrame:
    """
    Per (game, player) seconds on court, expanded from the stint rows.

    The stint row carries the lineup as one key; this is the per-player
    view, which is what a validation against the box score's own minutes
    needs (``validate_on_court_against_minutes``).
    """
    if stints.empty:
        return pd.DataFrame(columns=[
            "NBA_GAME_ID", "ESPN_ATHLETE_ID", "SECONDS_ON_COURT",
        ])
    # THE PREFIX IS THE JOIN, AND IT IS NOT OPTIONAL. `LINEUP` carries ESPN's
    # bare athlete ids because that is what the play feed's participants are.
    # The panel's `PLAYER_ID` is the SAME id with this project's `espn:`
    # prefix on it (`src/db/repository.py` builds it from `player_id_nba`,
    # which `espn_panel` writes through `espn_player_id`). Emitting the bare
    # form here made `validate_stints_against_minutes` an inner join that
    # matched zero rows on every real night -- 185 player-games on
    # 2026-11-09, none of them joined -- and a validation that matches
    # nothing reports `n: 0`, which the caller then read as a correlation of
    # zero and alerted on. One writer of the prefix, imported, not re-spelt.
    rows: list[dict[str, Any]] = []
    for _, stint in stints.iterrows():
        for player in str(stint["LINEUP"]).split(";"):
            rows.append({
                "NBA_GAME_ID": stint["NBA_GAME_ID"],
                "ESPN_ATHLETE_ID": espn_player_id(player),
                "SECONDS_ON_COURT": float(stint["STINT_SECONDS"]),
            })
    if not rows:
        return pd.DataFrame(columns=[
            "NBA_GAME_ID", "ESPN_ATHLETE_ID", "SECONDS_ON_COURT",
        ])
    out = pd.DataFrame(rows)
    return out.groupby(
        ["NBA_GAME_ID", "ESPN_ATHLETE_ID"], as_index=False,
    )["SECONDS_ON_COURT"].sum()


def validate_stints_against_minutes(
    on_court: pd.DataFrame, panel: pd.DataFrame,
) -> dict[str, float]:
    """
    Compare the stint-derived seconds to the box score's own minutes.

    THE INDEPENDENT CHECK THE RECONSTRUCTION RESTS ON. The stints are built
    from a starting five and a substitution stream; the box score's minutes
    are an independent count of the same thing. Where the two disagree, the
    reconstruction is wrong, and this says by how much rather than trusting
    the substitution stream.

    The shape mirrors ``pbp.validate_on_court_against_minutes``: a
    correlation, a mean and median absolute error in minutes, and the share
    of player-games within two minutes.
    """
    if on_court.empty:
        return {"n": 0.0}
    work = panel.copy()
    work["GAME_ID"] = work["GAME_ID"].astype(str)
    work["PLAYER_ID"] = work["PLAYER_ID"].astype(str)
    merged = on_court.merge(
        work,
        left_on=["NBA_GAME_ID", "ESPN_ATHLETE_ID"],
        right_on=["GAME_ID", "PLAYER_ID"],
        how="inner",
    ).dropna(subset=["MIN", "SECONDS_ON_COURT"])
    if merged.empty:
        return {"n": 0.0}
    recon = merged["SECONDS_ON_COURT"] / 60.0
    box = pd.to_numeric(merged["MIN"], errors="coerce")
    err = (recon - box).abs()
    return {
        "n": float(len(merged)),
        "corr": float(np.corrcoef(recon, box)[0, 1]),
        "mean_abs_error_minutes": float(err.mean()),
        "median_abs_error_minutes": float(err.median()),
        "within_2_minutes": float((err <= 2).mean()),
    }
