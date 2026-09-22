"""Pregame-only feature matrix.

THE RULE THIS FILE EXISTS TO ENFORCE: a feature for a game may only use
games that finished BEFORE it. Every rolling statistic here is computed on
a ``.shift(1)`` of the player's own history, so the current game's box
score can never reach its own features.

``LAST_INCLUDED_GAME_DATE`` records the most recent game folded into each
row's features. ``assert_no_lookahead`` checks it is strictly earlier than
``GAME_DATE``, which fails loudly if the shift discipline is ever broken.

Two layers, matching the orchestrator's contract:

    {stat}_BASELINE  layer 1 — blended recent form, nothing situational
    {stat}_L2        layer 2 — BASELINE adjusted for fatigue and pace

``build_feature_matrix`` calls ``attach_fatigue_column`` itself and folds
the multiplier into ``{stat}_L2``. main.py therefore VERIFIES rather than
re-applies; applying it twice would compound the penalty.
"""

from __future__ import annotations

import hashlib
import logging

import pandas as pd

from src.features.fatigue_logic import attach_fatigue_column
from src.features.schedule import attach_team_schedule_features
from src.features.team_strength import attach_elo_features, compute_team_elo

logger = logging.getLogger(__name__)

# Counting stats that get the full rolling treatment.
ROLLING_STATS = ("PTS", "REB", "AST", "PRA", "FG3M", "STL", "BLK", "MIN")

# Layer-1 blend. Recent form dominates, season average stabilises a short
# sample. Unfitted starting weights, not estimated parameters.
BASELINE_WEIGHTS = {"L5": 0.5, "L10": 0.3, "SEASON": 0.2}

FEATURE_SCHEMA_VERSION = "fs_v1_shift1_l2"


def resolved_feature_schema_version(attached_layers: "list[str] | None" = None) -> str:
    """
    The schema version for the feature set actually produced.

    The base version covers the core shift-1 columns. Additive layers can
    be absent (a module not in this repository) or skipped (a layer that
    raised), so a run with fewer layers genuinely has a different feature
    set. Encoding the attached layers as a short suffix keeps two such
    runs from both claiming ``fs_v1_shift1_l2`` and being silently
    interchangeable.
    """
    if not attached_layers:
        return FEATURE_SCHEMA_VERSION
    digest = hashlib.sha256("|".join(sorted(attached_layers)).encode()).hexdigest()[:8]
    return f"{FEATURE_SCHEMA_VERSION}+layers.{digest}"


def _layer_config() -> dict:
    """
    Read the layer settings from config/model_comparison.yaml.

    Config that nothing reads is worse than no config: it states a value
    the system is not using. These layers take keyword arguments, so the
    settings are threaded through here rather than left decorative.

    A missing or unreadable config yields {} and every layer runs on its
    own documented defaults.
    """
    try:
        from src.models.compare import load_comparison_config

        return load_comparison_config() or {}
    except Exception as exc:  # noqa: BLE001 — config is optional, never fatal
        logger.info("Layer config unavailable (%s) — using module defaults.", exc)
        return {}


def _additive_feature_layers() -> list[tuple[str, object]]:
    """
    Optional feature layers, resolved at import time, with config applied.

    Imported individually so a module absent from this repository simply
    does not contribute a layer.
    """
    cfg = _layer_config()
    halflife_cfg = cfg.get("halflife") or {}
    hot_hand_cfg = cfg.get("hot_hand") or {}
    blowout_cfg = cfg.get("blowout") or {}

    def _halflife(df):
        from src.features.halflife import attach_halflife_shrink_features

        return attach_halflife_shrink_features(
            df,
            halflife_games=float(halflife_cfg.get("games", 10.0)),
            shrink_k=float(halflife_cfg.get("shrink_k", 8.0)),
        )

    def _hot_hand(df):
        from src.features.hot_hand import attach_hot_hand_features

        return attach_hot_hand_features(
            df,
            z_threshold=float(hot_hand_cfg.get("z_threshold", 1.0)),
            minutes_stable_ratio=float(hot_hand_cfg.get("minutes_stable_ratio", 0.15)),
        )

    def _blowout(df):
        from src.features.blowout import (
            DEFAULT_SPREAD_THRESHOLD,
            attach_blowout_features,
        )

        return attach_blowout_features(
            df,
            spread_threshold=float(
                blowout_cfg.get("spread_threshold", DEFAULT_SPREAD_THRESHOLD)
            ),
        )

    configured: list[tuple[str, object]] = []
    if _module_has("src.features.halflife", "attach_halflife_shrink_features"):
        configured.append(("halflife.shrink", _halflife))
    if _module_has("src.features.halflife", "attach_pra_component_rollups"):
        from src.features.halflife import attach_pra_component_rollups

        configured.append(("halflife.pra_rollups", attach_pra_component_rollups))
    if _module_has("src.features.hot_hand", "attach_hot_hand_features"):
        configured.append(("hot_hand", _hot_hand))

    # Blowout risk is the one layer gated on config rather than on the
    # module being present, because it is DISABLED BY DEFAULT on measured
    # evidence -- see src/features/blowout.py. Its columns change the
    # feature set, so a run with it on gets a different schema digest and
    # cannot be mistaken for a run with it off.
    if blowout_cfg.get("enabled", False):
        if _module_has("src.features.blowout", "attach_blowout_features"):
            configured.append(("blowout", _blowout))
        else:
            logger.warning(
                "blowout.enabled is true but src.features.blowout is absent — "
                "no blowout columns this run."
            )

    for module_path, func_name, label in (
        ("src.features.teammate_cascade", "attach_teammate_cascade_stub", "teammate_cascade"),
        ("src.features.sports_ev_features", "attach_sports_ev_features", "sports_ev"),
        ("src.features.scoring_efficiency", "attach_box_ts_features", "scoring_efficiency"),
    ):
        if _module_has(module_path, func_name):
            module = __import__(module_path, fromlist=[func_name])
            configured.append((label, getattr(module, func_name)))
        else:
            logger.info("Feature layer %s not present — skipped.", label)
    return configured


def _module_has(module_path: str, func_name: str) -> bool:
    try:
        module = __import__(module_path, fromlist=[func_name])
        return hasattr(module, func_name)
    except ImportError:
        return False


_ADDITIVE_FEATURE_LAYERS = _additive_feature_layers()



def _group_shift_roll(
    df: pd.DataFrame,
    col: str,
    group_keys: list[str],
    *,
    window: int,
    min_periods: int = 1,
) -> pd.Series:
    """Prior-games rolling mean within a group. Shift and window in one call."""
    def _prior(series: pd.Series) -> pd.Series:
        return series.shift(1).rolling(window, min_periods=min_periods).mean()

    return df.groupby(group_keys, sort=False)[col].transform(_prior)


def attach_team_pace(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate team pace from box scores, never from market totals.

    Possessions follow the standard estimate:

        POSS = FGA - OREB + TOV + 0.44 * FTA

    rolled shift-1 within team-season and divided by the league mean for
    that season, giving a multiplier around 1.0.

    THE FALLBACK IS ABSTENTION, NOT 1.0. An earlier version of this file
    wrote PACE_MULTIPLIER = 1.0 whenever the inputs were missing, which
    created a column that looked measured, entered the feature list, and
    told every model that pace was exactly neutral. When the inputs are
    absent the columns are simply not created.
    """
    needed = {"TEAM_ABBREVIATION", "GAME_ID", "FGA", "OREB", "TOV", "FTA"}
    missing = needed - set(df.columns)
    if missing:
        logger.info(
            "Pace estimate skipped: %s absent. PACE_MULTIPLIER is NOT created — "
            "a neutral 1.0 would read as a measurement.", sorted(missing),
        )
        return df

    work = df.copy()
    season_col = "SEASON" if "SEASON" in work.columns else None
    group = ["SEASON", "TEAM_ABBREVIATION", "GAME_ID"] if season_col else [
        "TEAM_ABBREVIATION", "GAME_ID"
    ]

    team = work.groupby(group, as_index=False).agg(
        GAME_DATE=("GAME_DATE", "first"),
        FGA=("FGA", "sum"), OREB=("OREB", "sum"),
        TOV=("TOV", "sum"), FTA=("FTA", "sum"),
    )
    team["POSSESSIONS_EST"] = (
        team["FGA"] - team["OREB"] + team["TOV"] + 0.44 * team["FTA"]
    )
    sort_keys = ["TEAM_ABBREVIATION"] + ([season_col] if season_col else []) + ["GAME_DATE"]
    team = team.sort_values(sort_keys).reset_index(drop=True)

    team_keys = ["TEAM_ABBREVIATION"] + ([season_col] if season_col else [])
    team["PACE_ROLL"] = _group_shift_roll(
        team, "POSSESSIONS_EST", team_keys, window=10, min_periods=3
    )
    # League mean must be as-of-date. A season-wide transform("mean") includes
    # later games' pregame pace rolls and leaks future information into early
    # PACE_MULTIPLIER values.
    if season_col:
        daily = (
            team.groupby([season_col, "GAME_DATE"], as_index=False)["PACE_ROLL"]
            .mean()
            .rename(columns={"PACE_ROLL": "DAY_LEAGUE_PACE"})
            .sort_values([season_col, "GAME_DATE"])
            .reset_index(drop=True)
        )
        daily["LEAGUE_PACE_ASOF"] = daily.groupby(season_col, sort=False)[
            "DAY_LEAGUE_PACE"
        ].transform(lambda s: s.expanding(min_periods=3).mean().shift(1))
        team = team.merge(
            daily[[season_col, "GAME_DATE", "LEAGUE_PACE_ASOF"]],
            on=[season_col, "GAME_DATE"],
            how="left",
        )
        league = team["LEAGUE_PACE_ASOF"]
    else:
        daily = (
            team.groupby("GAME_DATE", as_index=False)["PACE_ROLL"]
            .mean()
            .rename(columns={"PACE_ROLL": "DAY_LEAGUE_PACE"})
            .sort_values("GAME_DATE")
            .reset_index(drop=True)
        )
        daily["LEAGUE_PACE_ASOF"] = (
            daily["DAY_LEAGUE_PACE"].expanding(min_periods=3).mean().shift(1)
        )
        team = team.merge(
            daily[["GAME_DATE", "LEAGUE_PACE_ASOF"]],
            on="GAME_DATE",
            how="left",
        )
        league = team["LEAGUE_PACE_ASOF"]
    # Rows without enough prior games keep NaN rather than a neutral 1.0.
    team["PACE_MULTIPLIER"] = (team["PACE_ROLL"] / league).clip(0.7, 1.3)

    keep = group + ["PACE_ROLL", "PACE_MULTIPLIER"]
    out = work.merge(team[keep], on=group, how="left")
    known = int(out["PACE_MULTIPLIER"].notna().sum())
    logger.info(
        "Pace estimated from box scores for %d of %d rows (rest lack prior games "
        "and stay null).", known, len(out),
    )
    return out


class LookaheadError(AssertionError):
    """Raised when a feature row could see its own game or a later one."""


def _prior_window_mean(series: pd.Series, window: int) -> pd.Series:
    """Mean of the previous `window` games. Never includes the current one.

    Must be used via ``groupby(...).transform``. Calling ``.rolling()`` on
    an already-groupby-shifted Series silently rolls ACROSS players, so one
    player's debut inherits the previous player's last game — applying the
    shift and the window inside the same per-group call is what prevents
    that.
    """
    return series.shift(1).rolling(window, min_periods=1).mean()


def _expanding_prior_mean(series: pd.Series) -> pd.Series:
    """Season-to-date mean of PRIOR games only.

    Deliberately expanding rather than a whole-season average: a
    full-season mean would leak the rest of the season into October rows.
    """
    return series.shift(1).expanding().mean()


def build_feature_matrix(
    player_panel: pd.DataFrame,
    *,
    team_games: pd.DataFrame | None = None,
    market_lines: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build the leakage-safe feature matrix from a raw player game-log panel.

    Expects at minimum PLAYER_ID, GAME_DATE, and one or more of
    ``ROLLING_STATS``. Missing stats are skipped rather than invented.

    ``team_games`` is an optional team-level frame carrying final scores
    (the BigDataBall ``team_game_stats`` output, or anything with game id,
    date, team and points). When supplied, pre-game Elo ratings are joined
    on; when absent, the team-strength columns are simply not created and
    the run is narrower rather than silently filled.

    ``market_lines`` is the matching ``market_lines`` frame. Only its OPENING
    spread and total are read, because those are posted before tip. Closing
    lines are refused outright — see src/features/market_context.py — since
    a number known only at tip is the market's final answer, not a feature.
    """
    if player_panel.empty:
        logger.warning("Empty player panel — returning it unchanged.")
        return player_panel.copy()

    required = {"PLAYER_ID", "GAME_DATE"}
    missing = required - set(player_panel.columns)
    if missing:
        raise ValueError(f"DATA_NOT_AVAILABLE: player panel missing {sorted(missing)}")

    df = player_panel.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)

    # PRA is the exact sum of three real columns, not an estimate, so it is
    # derived here rather than listed as a supported market that cannot in
    # fact be labelled. Deriving it before the rolling loop gives it the same
    # shift-1 treatment as every other stat. A NaN in any component
    # propagates deliberately: a partial sum would read as a real total.
    if {"PTS", "REB", "AST"}.issubset(df.columns) and "PRA" not in df.columns:
        df["PRA"] = (
            pd.to_numeric(df["PTS"], errors="coerce")
            + pd.to_numeric(df["REB"], errors="coerce")
            + pd.to_numeric(df["AST"], errors="coerce")
        )

    by_player = df.groupby("PLAYER_ID", sort=False)

    # The newest game already folded into this row's features.
    df["LAST_INCLUDED_GAME_DATE"] = by_player["GAME_DATE"].shift(1)
    df["CAREER_GAMES_PRIOR"] = by_player.cumcount()

    season_key = ["PLAYER_ID", "SEASON"] if "SEASON" in df.columns else ["PLAYER_ID"]
    by_season = df.groupby(season_key, sort=False)

    present = [s for s in ROLLING_STATS if s in df.columns]
    if not present:
        raise ValueError(
            f"DATA_NOT_AVAILABLE: none of {ROLLING_STATS} present in the panel"
        )
    for stat in present:
        df[stat] = pd.to_numeric(df[stat], errors="coerce")
        for window in (5, 10):
            df[f"{stat}_L{window}"] = by_player[stat].transform(
                _prior_window_mean, window=window
            )
        df[f"{stat}_SEASON"] = by_season[stat].transform(_expanding_prior_mean)

    # Layers that depend on an external frame being supplied, recorded so the
    # schema version reflects what a run actually had.
    market_layers: list[str] = []

    df = attach_team_pace(df)
    df = attach_fatigue_column(df)

    if {"TEAM_ABBREVIATION", "GAME_ID"}.issubset(df.columns):
        df = attach_team_schedule_features(df)
    else:
        logger.info("Team schedule features skipped: needs TEAM_ABBREVIATION and GAME_ID.")

    if team_games is not None and not team_games.empty:
        # Elo is computed over the full team history in date order, then
        # joined by pre-game value only. elo_post never reaches the panel.
        df = attach_elo_features(df, compute_team_elo(team_games))
        market_layers.append("team_elo")

        # Opponent defence, from TEAM totals rather than from sums over the
        # player panel. A panel sum measures roster coverage as much as it
        # measures defence — see src/features/defense.py. A failure here
        # narrows the run rather than stopping it, like the other layers.
        try:
            from src.features.defense import attach_defense_features, build_team_defense

            df = attach_defense_features(df, build_team_defense(team_games))
            market_layers.append("opponent_defense")
        except Exception as exc:  # noqa: BLE001 — enrichment, never fatal
            logger.warning(
                "Opponent-defence layer skipped (%s) — its columns are absent, "
                "not filled with a league average.", exc,
            )
    else:
        logger.info(
            "Team Elo and opponent defence skipped: no team_games frame supplied, "
            "so no opponent-strength or defensive-matchup features. Pass the "
            "BigDataBall team_game_stats frame to enable them."
        )

    if market_lines is not None and not market_lines.empty:
        # OPENING spread and total only. attach_market_context raises on any
        # closing column rather than quietly dropping it.
        from src.features.market_context import attach_market_context

        before = set(df.columns)
        df = attach_market_context(df, market_lines)
        if set(df.columns) - before:
            market_layers.append("market_context")
    else:
        logger.info(
            "Market context skipped: no market_lines frame supplied, so no "
            "implied team totals. Pass the BigDataBall market_lines frame."
        )

    # PACE_MULTIPLIER is NOT materialized when absent. Writing 1.0 into the
    # frame creates a column that looks measured, enters default_feature_cols,
    # and gets trained on as though neutral pace had been observed — the same
    # fabrication the zero-fill in compare.py was removed for. When no pace
    # source is joined the column simply does not exist, resolve_feature_cols
    # drops it with a warning, and the layer-2 adjustment below uses a scalar
    # 1.0 that never reaches the feature matrix.
    # Pace is now MEASURED (attach_team_pace) rather than assumed, which
    # means it is legitimately unknown for a team's first few games of a
    # season — min_periods=3 on the rolling possessions estimate.
    #
    # That unknown must not silently become 1.0, and it must not wipe out
    # L2 either. So layer 2 is published as two columns:
    #
    #   {stat}_L2        BASELINE x fatigue. Claims nothing about pace, and
    #                    is defined for every row, exactly as before.
    #   {stat}_L2_PACE   BASELINE x fatigue x pace. NaN wherever pace was
    #                    never measured, because a pace-adjusted projection
    #                    without a pace estimate does not exist.
    #
    # Collapsing these into one column would force a choice between
    # fabricating a neutral pace and discarding every early-season row.
    has_pace = "PACE_MULTIPLIER" in df.columns
    if has_pace:
        known = int(df["PACE_MULTIPLIER"].notna().sum())
        logger.info(
            "Layer 2: pace measured on %d of %d rows. {stat}_L2_PACE is null "
            "elsewhere; {stat}_L2 carries no pace claim and is always defined.",
            known, len(df),
        )
    else:
        logger.info(
            "PACE_MULTIPLIER absent — no {stat}_L2_PACE column. {stat}_L2 is "
            "unaffected and makes no pace claim."
        )

    for stat in present:
        blended = (
            BASELINE_WEIGHTS["L5"] * df[f"{stat}_L5"]
            + BASELINE_WEIGHTS["L10"] * df[f"{stat}_L10"]
            + BASELINE_WEIGHTS["SEASON"] * df[f"{stat}_SEASON"]
        )
        # Early-season rows have no season mean yet; fall back to what exists
        # rather than dropping the row or inventing a value.
        df[f"{stat}_BASELINE"] = blended.fillna(df[f"{stat}_L5"]).fillna(df[f"{stat}_L10"])
        df[f"{stat}_L2"] = df[f"{stat}_BASELINE"] * df["fatigue_multiplier"]
        if has_pace:
            df[f"{stat}_L2_PACE"] = df[f"{stat}_L2"] * df["PACE_MULTIPLIER"]

    # --- additive feature layers (waves 2, 4b, 5a) ------------------------
    # Each of these ONLY adds columns; none rewrites the core L2/L5/L10/
    # BASELINE set above. They run here, after the season baselines exist,
    # because hot_hand measures recent form against {stat}_SEASON and would
    # otherwise have nothing to compare to.
    #
    # A layer that fails is logged and skipped rather than taking the whole
    # matrix down: these are enrichments, and losing one should narrow the
    # feature set, not stop the pipeline. assert_no_lookahead still runs
    # over whatever they produced.
    attached: list[str] = list(market_layers)
    for layer_name, attach in _ADDITIVE_FEATURE_LAYERS:
        try:
            df = attach(df)
            attached.append(layer_name)
        except Exception as exc:  # noqa: BLE001 — enrichment, never fatal
            logger.warning(
                "Feature layer %s skipped (%s) — its columns are absent, not "
                "filled with a placeholder.", layer_name, exc,
            )

    # The schema version must reflect what was ACTUALLY attached. Layers can
    # be absent or fail, so a fixed string would let an artifact trained
    # with one feature set be scored against another while both claim the
    # same version.
    df["FEATURE_SCHEMA_VERSION"] = resolved_feature_schema_version(attached)

    assert_no_lookahead(df)
    logger.info(
        "Feature matrix: %d rows x %d cols | stats=%s | schema=%s",
        len(df), df.shape[1], present, FEATURE_SCHEMA_VERSION,
    )
    return df


def assert_no_lookahead(features: pd.DataFrame) -> None:
    """
    Fail loudly if any row's features could see its own game or a later one.

    A player's first game has no prior history, so a null
    LAST_INCLUDED_GAME_DATE is correct and is not a violation.
    """
    if "LAST_INCLUDED_GAME_DATE" not in features.columns:
        raise LookaheadError(
            "LAST_INCLUDED_GAME_DATE missing — cannot verify the feature matrix "
            "is pregame-safe. Refusing to certify it."
        )

    game = pd.to_datetime(features["GAME_DATE"])
    last = pd.to_datetime(features["LAST_INCLUDED_GAME_DATE"])
    violations = last.notna() & (last >= game)

    if violations.any():
        sample = features.loc[violations, ["PLAYER_ID", "GAME_DATE", "LAST_INCLUDED_GAME_DATE"]]
        raise LookaheadError(
            f"{int(violations.sum())} rows include a game on or after their own "
            f"GAME_DATE. First offenders:\n{sample.head(5)}"
        )

    logger.debug("Lookahead check passed for %d rows.", len(features))
