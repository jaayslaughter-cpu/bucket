"""Exponential cumulative fatigue load (additive feature layer).

WHAT THIS ADDS, AND WHY IT IS NOT THE EXISTING MULTIPLIER.
``fatigue_logic.attach_fatigue_column`` emits ``fatigue_multiplier`` from four
discrete constants -- B2B 0.97, 3-in-4 0.96, 4-in-5 0.94, altitude 0.98 --
labelled in that module as "Unfitted, conservative", and applied by a
``mask`` cascade so the worst density wins rather than accumulating. That
throws away three things a continuous load keeps:

  - HOW MANY MINUTES those recent games cost. 38 minutes two nights ago and
    12 minutes two nights ago are the same B2B flag and are not the same
    fatigue.
  - HOW RECENT each game was, beyond the flag's fixed window.
  - HOW FAR the team flew and across how many time zones.

This layer emits the load itself, not a haircut. It does NOT modify
``fatigue_multiplier`` or ``{stat}_L2`` -- the intended use is as a model
FEATURE whose coefficient is learned, so nothing here asserts an effect
size. Whether it beats the four constants is an empirical question for
``scripts/feature_ab.py``, not a claim made here.

    L_i(t) = sum over prior games g within k days of
             MIN_g * exp(-lambda * days_ago_g)
                   * (1 + theta * miles_g / 1000 + phi * |tz_shift_g|)

COLUMNS
  FATIGUE_LOAD_L7       the full load above
  FATIGUE_LOAD_MIN_L7   the minutes-decay part only (travel factor == 1)

Both are emitted so the travel contribution is separable: an A/B that moves
with FATIGUE_LOAD_L7 but not FATIGUE_LOAD_MIN_L7 is evidence about travel,
and one that moves with both is not.

LEAKAGE. Every term comes from a STRICTLY PRIOR game of the same player,
via ``shift(g)`` for g >= 1 over date-sorted rows. No same-game minutes,
travel or opponent information is read. A player's first games carry a
partial window rather than an imputed one.

PARAMETERS ARE UNFITTED, exactly as the constants they are meant to replace.
The defaults are midpoints of conventional ranges and are labelled as such in
``DEFAULTS``; fit them with the A/B harness before attributing meaning to a
particular value.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

LOAD_COL = "FATIGUE_LOAD_L7"
LOAD_MIN_COL = "FATIGUE_LOAD_MIN_L7"

# Unfitted. lambda is the midpoint of the 0.15-0.25 range conventionally
# quoted for day-scale fatigue decay; theta and phi are deliberately small so
# travel perturbs the minutes term rather than dominating it. A fitted value
# replaces these; none of the three is a measurement.
DEFAULTS: dict[str, float] = {
    "decay_lambda": 0.20,
    "miles_theta": 0.05,
    "tz_phi": 0.05,
    "window_days": 7.0,
}

# Prior games examined per row. A team plays at most 4 games in 7 days under
# every published NBA schedule format, so 5 covers the window exactly while
# staying vectorised. Rows exceeding it are counted and warned rather than
# silently truncated.
MAX_PRIOR_GAMES = 5

# Standard UTC offsets of current home venues. DST is deliberately ignored:
# this is used only as a DIFFERENCE between two venues, and every team that
# observes DST shifts together, so the difference is unchanged. PHX does not
# observe it, which moves Phoenix's difference by one hour for part of the
# season -- small, one team, and documented rather than hidden.
TEAM_UTC_OFFSET: dict[str, int] = {
    "BOS": -5, "BKN": -5, "NYK": -5, "PHI": -5, "TOR": -5,
    "CHI": -6, "CLE": -5, "DET": -5, "IND": -5, "MIL": -6,
    "ATL": -5, "CHA": -5, "MIA": -5, "ORL": -5, "WAS": -5,
    "DEN": -7, "MIN": -6, "OKC": -6, "POR": -8, "UTA": -7,
    "GSW": -8, "LAC": -8, "LAL": -8, "PHX": -7, "SAC": -8,
    "DAL": -6, "HOU": -6, "MEM": -6, "NOP": -6, "SAS": -6,
}


def _params(cfg: dict[str, Any] | None) -> dict[str, float]:
    block = dict(DEFAULTS)
    if cfg:
        for key in DEFAULTS:
            if key in cfg:
                try:
                    block[key] = float(cfg[key])
                except (TypeError, ValueError):
                    logger.warning(
                        "fatigue_load: %s=%r is not a number — keeping default %s",
                        key, cfg[key], DEFAULTS[key],
                    )
    if block["decay_lambda"] <= 0:
        logger.warning(
            "fatigue_load: decay_lambda must be positive (got %s); a "
            "non-positive value makes older games weigh at least as much as "
            "recent ones. Using the default.",
            block["decay_lambda"],
        )
        block["decay_lambda"] = DEFAULTS["decay_lambda"]
    return block


def _venue_team(df: pd.DataFrame) -> pd.Series | None:
    """The team whose arena hosts each row's game, or None if undeterminable."""
    if not {"IS_HOME", "TEAM_ABBREVIATION"}.issubset(df.columns):
        return None
    if "OPPONENT_ABBREVIATION" not in df.columns:
        return None
    home = df["IS_HOME"].fillna(False).astype(bool)
    return df["TEAM_ABBREVIATION"].where(home, df["OPPONENT_ABBREVIATION"])


def attach_fatigue_load(
    df: pd.DataFrame, cfg: dict[str, Any] | None = None
) -> pd.DataFrame:
    """
    Add ``FATIGUE_LOAD_L7`` and ``FATIGUE_LOAD_MIN_L7``.

    Adds only; never rewrites an existing column, and returns the frame
    unchanged when the columns it needs are absent.
    """
    required = {"PLAYER_ID", "GAME_DATE", "MIN"}
    missing = required - set(df.columns)
    if missing:
        logger.info(
            "fatigue_load skipped: needs %s (missing %s).",
            sorted(required), sorted(missing),
        )
        return df
    if LOAD_COL in df.columns:
        logger.info("fatigue_load: %s already present — leaving it alone.", LOAD_COL)
        return df

    p = _params(cfg)
    out = df.copy()
    order = out.index
    work = out.reset_index(drop=False).rename(columns={"index": "_row"})
    if "_row" not in work.columns:          # a RangeIndex reset names it 'level_0'
        work = work.rename(columns={work.columns[0]: "_row"})
    work["_date"] = pd.to_datetime(work["GAME_DATE"], errors="coerce")
    work["_min"] = pd.to_numeric(work["MIN"], errors="coerce")

    # Travel factor of each row's own game, later shifted into the past.
    factor = pd.Series(1.0, index=work.index, dtype=float)
    if "TRAVEL_MILES" in work.columns:
        miles = pd.to_numeric(work["TRAVEL_MILES"], errors="coerce").fillna(0.0)
        factor = factor + p["miles_theta"] * (miles / 1000.0)
    else:
        logger.info(
            "fatigue_load: no TRAVEL_MILES column, so the distance term is 0. "
            "Run attach_team_schedule_features first to include it."
        )

    venue = _venue_team(work)
    if venue is not None:
        offset = venue.map(TEAM_UTC_OFFSET)
        unmapped = int(offset.isna().sum())
        if unmapped:
            logger.warning(
                "fatigue_load: %d row(s) have a venue outside the offset table "
                "(relocation or a non-NBA code); their time-zone term is 0.",
                unmapped,
            )
        work["_venue_offset"] = offset
        chronological = work.sort_values("_date", kind="mergesort")
        prev_offset = chronological.groupby(
            "TEAM_ABBREVIATION", sort=False
        )["_venue_offset"].shift(1)
        tz_shift = (
            (chronological["_venue_offset"] - prev_offset)
            .abs()
            .reindex(work.index)
            .fillna(0.0)
        )
        factor = factor + p["tz_phi"] * tz_shift
    else:
        logger.info(
            "fatigue_load: need IS_HOME, TEAM_ABBREVIATION and "
            "OPPONENT_ABBREVIATION to locate each venue, so the time-zone "
            "term is 0."
        )
    work["_factor"] = factor

    work = work.sort_values(["PLAYER_ID", "_date"], kind="mergesort")
    grouped = work.groupby("PLAYER_ID", sort=False)

    load = pd.Series(0.0, index=work.index, dtype=float)
    load_min_only = pd.Series(0.0, index=work.index, dtype=float)
    window = float(p["window_days"])

    for lag in range(1, MAX_PRIOR_GAMES + 1):
        prior_min = grouped["_min"].shift(lag)
        prior_date = grouped["_date"].shift(lag)
        prior_factor = grouped["_factor"].shift(lag)
        days_ago = (work["_date"] - prior_date).dt.days.astype("Float64")

        inside = (days_ago >= 1) & (days_ago <= window) & prior_min.notna()
        decay = np.exp(-p["decay_lambda"] * days_ago.astype("float64").to_numpy())
        weighted = pd.Series(
            prior_min.fillna(0.0).to_numpy() * np.nan_to_num(decay),
            index=work.index,
        )
        keep = inside.fillna(False).to_numpy()
        load_min_only = load_min_only + weighted.where(keep, 0.0)
        load = load + (weighted * prior_factor.fillna(1.0)).where(keep, 0.0)

    # The 5-game bound is an assumption about the schedule, so check it.
    sixth = grouped["_date"].shift(MAX_PRIOR_GAMES + 1)
    overflow = ((work["_date"] - sixth).dt.days <= window).fillna(False)
    if bool(overflow.any()):
        logger.warning(
            "fatigue_load: %d row(s) have more than %d prior games within %.0f "
            "days, so their load is truncated at %d. Raise MAX_PRIOR_GAMES.",
            int(overflow.sum()), MAX_PRIOR_GAMES, window, MAX_PRIOR_GAMES,
        )

    work[LOAD_COL] = load
    work[LOAD_MIN_COL] = load_min_only
    restored = work.set_index("_row")
    out[LOAD_COL] = restored[LOAD_COL].reindex(order)
    out[LOAD_MIN_COL] = restored[LOAD_MIN_COL].reindex(order)

    logger.info(
        "fatigue_load attached: mean %s %.2f (min-only %.2f) over %d row(s); "
        "lambda=%.3f theta=%.3f phi=%.3f, all UNFITTED.",
        LOAD_COL, float(out[LOAD_COL].mean()), float(out[LOAD_MIN_COL].mean()),
        len(out), p["decay_lambda"], p["miles_theta"], p["tz_phi"],
    )
    return out


def attach_fatigue_load_layer(df: pd.DataFrame) -> pd.DataFrame:
    """Builder entry point: reads the ``fatigue_load`` config block if present."""
    cfg: dict[str, Any] | None = None
    try:
        from src.models.compare import load_comparison_config

        cfg = (load_comparison_config() or {}).get("fatigue_load")
    except Exception as exc:  # noqa: BLE001 — config is optional here
        logger.debug("fatigue_load: no config block (%s); using defaults.", exc)
    return attach_fatigue_load(df, cfg)
