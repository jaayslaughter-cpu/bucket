"""One definition of the NBA season key, and one rule about who may write it.

Three additive feature layers derived their own grouping season when the panel
did not carry one, each with its own copy of this line:

    out["SEASON"] = pd.to_datetime(out["GAME_DATE"], errors="coerce").dt.year

Both halves of it were wrong.

FIRST, THE BOUNDARY. A calendar year splits an NBA season at 1 January, so
every player's group restarted on New Year's Day: the expanding season mean,
the rolling windows and the streak counters all reset mid-season, and nothing
said so. ``src/features/defense.py`` already had the right form — the season's
STARTING year, cutting in August — and that form is now defined here once
instead of three times plus a fourth that disagreed.

SECOND, AND WORSE, the line wrote its guess into the returned frame under the
public name ``SEASON``. ``builder`` applies layers as ``df = attach(df)`` and
halflife runs FIRST, so one layer's private fallback silently became the
grouping key for every layer after it, for ``compare``'s categorical column
list, and for the caller. It also defeated ``minutes_weighted``'s abstention:
that layer refuses to guess a season precisely because a 1 January restart is a
silently wrong answer, but by the time it ran SEASON was never missing. A guess
made by one module was read by five as a fact.

So the key is derived into a PRIVATE column whose user drops it before
returning, and ``SEASON`` stays the panel's to write.

On a panel that carries SEASON — which the production panel does — nothing
here changes anything: the panel's own column is used, exactly as before.

RESEARCH_ONLY.
"""

from __future__ import annotations

import pandas as pd

SEASON_COL = "SEASON"
#: Private grouping key. Never returned to a caller; see ``drop_season_key``.
SEASON_KEY_COL = "_SEASON_KEY"


def season_start_year(dates: pd.Series) -> pd.Series:
    """
    The season's STARTING year. October 2025 and March 2026 are one season.

    August is the cut: the NBA calendar has no games in August, so no real
    game can land on the wrong side of it. A plain ``dt.year`` cuts at 1
    January, in the middle of every season.
    """
    d = pd.to_datetime(dates, errors="coerce")
    return (d.dt.year - (d.dt.month < 8).astype(int)).astype("Int64")


def player_season_keys(
    frame: pd.DataFrame,
    *,
    player_col: str = "PLAYER_ID",
    date_col: str = "GAME_DATE",
) -> tuple[pd.DataFrame, list[str]]:
    """
    Return ``(frame, group_keys)`` for per-player-season grouping.

    Uses the panel's own ``SEASON`` when it has one. Otherwise derives the
    key into ``SEASON_KEY_COL``, which the caller must drop with
    ``drop_season_key`` before returning — a derived season is this layer's
    working assumption, not a column the next layer gets to inherit.
    """
    if SEASON_COL in frame.columns:
        return frame, [player_col, SEASON_COL]
    out = frame.copy()
    out[SEASON_KEY_COL] = season_start_year(out[date_col])
    return out, [player_col, SEASON_KEY_COL]


def drop_season_key(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove the private key. A no-op when the panel carried SEASON."""
    return frame.drop(columns=[SEASON_KEY_COL], errors="ignore")
