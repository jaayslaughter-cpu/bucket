"""Research targets and feature lists for the model comparison.

IMPORTANT — WHAT ``RESEARCH_LINE`` IS AND IS NOT

It is the player's own trailing 10-game average. It is a **stand-in** used
so the comparison machinery can run before real prop lines exist. It is
NOT a sportsbook line, and a probability computed against it is NOT a
betting probability.

Because the projection is built from the same rolling history, scoring
against this line measures roughly "is recent form above medium-term
form". That is a sanity check on plumbing, not evidence of prop skill.
Every export carries ``line_type`` so this can never be misread.

When real timestamped prop lines land, join them in and pass their column
as ``line_col`` — the models take the line as an argument and need no
change.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

LAUNCH_MARKETS = ("PTS", "REB", "AST")
POST_LAUNCH_MARKETS = ("FG3M", "STL", "BLK", "PRA")

RESEARCH_LINE_COL = "RESEARCH_LINE"
RESEARCH_LINE_SOURCE = "L10"
TARGET_COL = "over_hit"


def default_feature_cols(market: str) -> list[str]:
    """
    Numeric pregame features for a market.

    Own-stat history first, then situational context. Categorical columns
    are deliberately absent: CatBoost adds them separately, while XGBoost
    stays numeric-only so its behaviour is unchanged.
    """
    market = market.upper()
    return [
        # Own production history
        f"{market}_L2",
        f"{market}_L5",
        f"{market}_L10",
        f"{market}_SEASON",
        f"{market}_BASELINE",
        "MIN_L5",
        "MIN_L10",
        "MIN_SEASON",
        # Player-level fatigue
        "fatigue_multiplier",
        "days_rest",
        # Continuous cumulative load, beside the multiplier rather than
        # replacing it: the multiplier is four unfitted constants applied as a
        # haircut, the load is minutes-weighted and recency-decayed, and which
        # carries signal is an empirical question (feature_ab --layer
        # fatigue_load). resolve_feature_cols drops either when the panel
        # lacks it, so a panel built without the layer is unaffected.
        "FATIGUE_LOAD_L7",
        "FATIGUE_LOAD_MIN_L7",
        # Team schedule context
        "TEAM_DAYS_REST_CAPPED",
        "REST_ADVANTAGE",
        "IS_B2B_SECOND",
        "IS_B2B_FIRST",
        "TRAVEL_MILES",
        # Opponent strength
        "TEAM_ELO_PRE",
        "OPP_ELO_PRE",
        "ELO_DIFF",
        "ELO_WIN_PROB",
        # Situation. PACE_MULTIPLIER is omitted until a pregame pace source
        # exists; resolve_feature_cols would drop it anyway, with a warning
        # on every market of every run.
        "IS_HOME",
        "CAREER_GAMES_PRIOR",
        # Pregame market context. OPENING lines only -- src/features/
        # market_context.py refuses closing ones, which are known at tip.
        #
        # These were computed and attached for some time before anything
        # read them: build_feature_matrix wrote MKT_IMPLIED_TEAM_TOTAL, the
        # module's own docstring called it "the single most informative
        # pregame number available" for a points prop, and no feature list
        # named it, so no model ever saw it. Listing them here is what makes
        # the market layer reachable.
        #
        # They exist only when a market_lines frame was supplied. Without
        # one, resolve_feature_cols drops them with a warning and the run is
        # narrower -- it does not zero-fill a spread the panel never had.
        "MKT_OPENING_SPREAD",
        "MKT_OPENING_TOTAL",
        "MKT_IMPLIED_TEAM_TOTAL",
        "MKT_IMPLIED_OPP_TOTAL",
        "MKT_IS_FAVORITE",
        # Blowout risk. Off by default and absent from most panels -- see
        # src/features/blowout.py for the measurement that keeps it off.
        "BLOWOUT_FAV_HINGE",
        "BLOWOUT_DOG_HINGE",
        # Opponent defence. Present only when a team_games frame was supplied.
        #
        # DEF_RATING_INDEX_L10 is built and exported but deliberately NOT a
        # feature: within a season it correlates with DEF_RATING_L10 at
        # r = 0.999, so a model receives one number twice and splits its
        # attention between identical candidates. It stays available for
        # reporting, where a league-relative 1.0-centred number is the
        # readable one.
        "DEF_RATING_L10",
        # Pace is published separately from defensive quality on purpose:
        # points allowed per GAME confounds the two. On the real 2025-26
        # panel a team's prior-10 pace predicts tonight's possessions at
        # r = +0.327, so tempo is persistent and worth its own column.
        "DEF_PACE_L10",
        *_DEFENSE_BY_MARKET.get(market, ()),
        # Play-by-play shot mix and game-state context, as prior-game rolling
        # means. Present only for seasons whose event logs were supplied;
        # elsewhere resolve_feature_cols drops them and compare_models_on_panel
        # refuses any that are empty across the training window.
        *_PBP_BY_MARKET.get(market, ()),
        # Recent-form and shot-quality columns. Four feature layers
        # (halflife, hot_hand, sports_ev, scoring_efficiency) ran on every
        # build for months and no market read one of their 71 numeric
        # columns. What goes in here is the subset that is not already
        # carried by a listed feature -- see _FORM_BY_MARKET and
        # _EXCLUDED_AS_REDUNDANT below for the measurement.
        *_FORM_BY_MARKET.get(market, ()),
    ]


# Within-season |r| against this market's already-listed features, measured on
# the real 214,381-row panel across nine seasons, weighted by overlapping rows.
# A column that reproduces a listed feature is not a second opinion: the model
# receives one number twice and splits its attention between identical
# candidates. This is the reasoning already applied to DEF_RATING_INDEX_L10
# (r = 0.999) and it excludes the whole halflife family for the same reason.
#
#   column                  PTS    REB    AST   against
#   {M}_HL                 0.971  0.963  0.970  {M}_SEASON
#   {M}_HL_SHRINK          0.990  0.987  0.990  {M}_SEASON
#   {M}_L2_HL              0.974  0.964  0.970  {M}_L2
#   MIN_HL                 0.975  0.975  0.975  MIN_SEASON
#   MIN_HL_SHRINK          0.990  0.990  0.990  MIN_SEASON
#   USAGE_PROXY_L10        0.969  0.848  0.848  PTS_L10 / MIN_L10
#   SHOT_VOLUME_L5         0.968  0.835  0.835  PTS_BASELINE / MIN_L5
#   SHOT_VOLUME_L10        0.967  0.841  0.841  PTS_L10 / MIN_L10
#   FGA_L5                 0.955  0.839  0.839  PTS_BASELINE / MIN_L5
#   FGA_L10                0.956  0.846  0.846  PTS_L10 / MIN_L10
#   OPP_{M}_ALLOWED_L10    0.836  0.892  0.942  DEF_* (per 100 poss)
#   {M}_MW_L5              0.991  0.989  0.992  {M}_L5
#
# {M}_MW_L5 is the minutes-weighted L5 from features/minutes_weighted.py, which
# the builder DOES produce and this list deliberately omits. Measured the same
# way, 204,529 overlapping rows: 0.976-0.992 against {M}_L5 across PTS, REB,
# AST, FG3M, STL and BLK. A 0.5/1.0/1.5 reweighting of the same five games sits
# inside the halflife band above, so the prediction is the same one -- the model
# receives one number twice. It is registered as a feature_ab layer rather than
# shipped, so `--layer minutes_weighted --wire-under-test` can settle it.
# Full write-up with the per-column table: docs/minutes_weighted.md.
#
# TWO MORE LAYERS ARE IN THE SAME STATE, for the opposite reason: not measured
# redundant. src/features/fouls.py (the player's prior foul history) and
# src/features/dvp.py (opponent defence split by the position it is defending)
# are built on every panel that carries PF and STARTING_POSITION, and no
# market below reads a column from either. Measured the same way as the table
# above, every one of their columns is well clear of the redundancy band --
# the worst is PF_SEASON at 0.631 against MIN_SEASON, and a count accumulated
# over playing time should correlate with playing time; DVP_*_INDEX_L10 tops
# out at 0.389, against the team-level DEF_* column covering the same stat,
# where DEF_RATING_INDEX_L10 sits at 0.999 against DEF_RATING_L10. So they are
# not copies of anything listed here. The halflife and usage_volume results
# below are the reason correlation alone does not settle it: redundancy was
# PREDICTED from |r| and then tested, and the test is what the exclusion rests
# on. Write-up: docs/fouls_and_dvp.md.
#
# FOULS is still unmeasured; `--layer fouls --wire-under-test` is the arm.
#
# DVP HAS NOW BEEN MEASURED, for REB, and it HELPS. 4 chronological folds from
# 2025-01-15 stepping 14 days, 3-season slice (2022-23..2024-25), 76,585 rows,
# ~17,988 distinct validation rows, REB reading the two columns carrying its
# own stat (DVP_REB_ALLOWED_L10, DVP_REB_INDEX_L10):
#
#   ensemble    Brier raw  -0.00070 (sd 0.00034)  4/4 folds better
#   line_aware  Brier raw  -0.00065 (sd 0.00034)  4/4
#   line_aware  Brier cal  -0.00054 (sd 0.00022)  4/4
#   catboost    Brier cal  -0.00099 (sd 0.00077)  4/4
#   xgboost     Brier cal  -0.00061 (sd 0.00052)  4/4
#
# Better on every fold for four of the five models, at 1.3-2.5x the fold
# spread -- the opposite sign and the opposite consistency to the halflife and
# usage_volume tables below. Calibrated ECE did NOT improve (line_aware
# +0.00125, better in only 1 of 4 folds), though every ECE delta is smaller
# than its own fold spread, so that is a thing to watch and not a finding.
#
# PRA WAS THEN RUN IDENTICALLY AND DOES NOT HELP. Its DVP_PRA_* pair is a
# combination market (dvp.DVP_COMBOS), 4 folds, same slice, ~18,179 distinct
# validation rows:
#
#   line_aware  Brier raw  +0.00048 (sd 0.00017)  0/4 folds better
#   line_aware  Brier cal  +0.00044 (sd 0.00011)  0/4
#   catboost    Brier cal  +0.00039 (sd 0.00030)  0/4
#   ensemble    Brier raw  +0.00006 (sd 0.00012)  1/4
#   xgboost     Brier raw  +0.00004 (sd 0.00021)  1/4
#
# line_aware is worse on every fold at 2.8-4x the fold spread; the rest are
# nil. Raw ECE improved (xgboost -0.00195 at 4/4, ensemble -0.00253 at 3/3),
# which is the SAME pattern the halflife and usage_volume note below records
# and the same reading applies: without a Murphy decomposition the honest
# statement is the measurement, and Brier is the metric these calls are made
# on. The measured geometry predicted a weak effect -- PRA's between-bucket
# spread is 1.11 against REB's 2.17, because centre-heavy rebounds and
# guard-heavy assists cancel in the sum -- and the arm came back adverse
# rather than merely weak. Not wired. Write-up: docs/fouls_and_dvp.md
# section 3b.
#
# IT IS STILL NOT WIRED, AND THE REASON IS NOT THE EVIDENCE. The reason used
# to be that STARTING_POSITION had no writer on the live path. It has one now
# -- src/ingestion/starting_positions.py, migration 008 and
# scripts/pull_starting_positions.py -- and what remains is narrower and still
# blocking:
#
#   1. THE PULL HAS NEVER BEEN RUN. stats.nba.com is denied at this
#      environment's proxy, so player_game_logs.starting_position is NULL on
#      every row today and attach_dvp_features still abstains on a live panel.
#   2. THE MEASUREMENT ABOVE WAS MADE ON AN ARCHIVE PANEL, which is not the
#      panel production builds from.
#
# Wiring these columns now would train trees to split on a column that arrives
# empty in production -- the teammate_cascade.py failure mode this project has
# documented once already. The order is: run the pull where nba.com is
# reachable, rebuild the live panel, confirm non-null coverage there, re-run
# this arm on a panel that carries real positions, and only then change the
# tuples below. Numbers, the collinearity table, the slice's cost and the
# writer: docs/fouls_and_dvp.md sections 3a and 5.
#
# OPP_{M}_ALLOWED_L10 is per GAME where the listed DEF_* columns are per 100
# POSSESSIONS. Per-game allowed confounds defensive quality with tempo, which
# is the exact confound DEF_PACE_L10 was published separately to avoid, so the
# per-game column is the worse of two measurements of one thing.
#
# THE PREDICTION WAS THEN TESTED, not left as an argument from correlation.
# scripts/feature_ab.py --layer halflife and --layer usage_volume, both with
# --wire-under-test so nothing had to be shipped to measure it, PTS, 3 folds
# (~13,180 distinct validation rows; see the note below on the printed 65,900):
#
#   halflife      xgboost Brier raw  +0.00022 (sd 0.00006)  0/3 folds better
#                 ensemble Brier raw +0.00025 (sd 0.00012)  0/3
#   usage_volume  catboost Brier cal +0.00057 (sd 0.00006)  0/3
#                 catboost Brier raw +0.00060 (sd 0.00020)  0/3
#
# Adding them makes overall Brier WORSE, on every fold, at 3-10x the fold
# spread, which is consistent with the r = 0.955-0.990 reading: the model
# receives one number twice.
#
# ONE HONEST COMPLICATION. Both redundant sets consistently IMPROVED calibrated
# ECE while leaving Brier flat or worse -- halflife: xgboost -0.00313 and
# ensemble -0.00330, both 3/3; usage_volume: xgboost -0.00383 and ensemble
# -0.00287, both 3/3. The same tension appears in the PTS form result pointing
# the other way.
#
# WHAT THAT DOES AND DOES NOT SHOW. Brier is not a discrimination metric: it
# decomposes into calibration and resolution, so a worse Brier alongside a better
# ECE does NOT establish that these columns cost resolution and bought
# calibration. An earlier version of this comment claimed exactly that. Without a
# Murphy decomposition the honest statement is the measurement itself -- worse
# overall Brier despite lower ECE -- and Brier is the metric these exclusions are
# decided on because it scores the probability as a whole.
_EXCLUDED_AS_REDUNDANT = (
    "{M}_HL", "{M}_HL_SHRINK", "{M}_L2_HL", "MIN_HL", "MIN_HL_SHRINK",
    "USAGE_PROXY_L10", "SHOT_VOLUME_L5", "SHOT_VOLUME_L10", "FGA_L5", "FGA_L10",
    "OPP_{M}_ALLOWED_L10",
)


# The columns that survived that screen: each correlates below 0.45 with every
# feature this market already reads, so each is a genuinely new number rather
# than a re-expression of an old one.
#
#   {M}_HOT_Z            0.26-0.27 vs {M}_L5     recent form vs season baseline
#   {M}_STREAK_ABOVE     0.21-0.24 vs {M}_L5     consecutive prior games over
#   {M}_STREAK_BELOW     0.20-0.22 vs {M}_L5     consecutive prior games under
#   MINUTES_TREND_RATIO  0.158    vs MIN_SEASON  role trending up or down
#   MINUTES_STABLE       0.408    vs MIN_SEASON  is the role settled
#   TS_PCT_L10           0.21-0.27                shooting efficiency, not volume
#   TS_PCT_TREND         0.02-0.11                efficiency direction
#   FT_RATE_L10          0.05-0.25                how the points are earned
#
# TS_PCT_L5 is left out beside TS_PCT_L10 (r = 0.83 with each other); the
# longer window is the less noisy of the pair and TS_PCT_TREND already carries
# the short-run movement.
# MEASURED, not predicted. scripts/feature_ab.py --layer form, 3 chronological
# folds, 103,233 rows across 2022-23..2025-26. Only the markets whose arms
# actually won are populated; the rest are empty for the same reason
# _PBP_BY_MARKET["PTS"] is empty.
#
# ON THE ROW COUNT: the harness prints "validation rows across 3 fold(s): 65900",
# but that sums n_predictions over every model row, counting each held-out
# observation once per model. Five models over three folds means roughly 13,180
# DISTINCT validation rows. The deltas below are per-model and unaffected; the
# count was misread as distinct rows when this was first written up.
_FORM_BY_MARKET: dict[str, tuple[str, ...]] = {
    # PTS: the strongest result in this file. Every tree model improved Brier on
    # 3/3 folds at 4-22x its own fold spread -- xgboost -0.00191 (sd 0.00021),
    # ensemble -0.00194 (sd 0.00014), catboost -0.00177 (sd 0.00034),
    # line_aware -0.00158 (sd 0.00007) -- and xgboost's raw ECE -0.00223 on 3/3.
    # Caveat kept visible: calibrated ECE went the OTHER way for catboost
    # (+0.00837, sd 0.00332, 0/3) and the ensemble (+0.00320, 1/3).
    "PTS": ("PTS_HOT_Z", "PTS_STREAK_ABOVE", "PTS_STREAK_BELOW",
            "MINUTES_TREND_RATIO", "MINUTES_STABLE",
            "TS_PCT_L10", "TS_PCT_TREND", "FT_RATE_L10"),
    # REB: smaller than PTS but consistent. Brier improved on 3/3 folds for
    # line_aware (-0.00086, sd 0.00042), catboost (-0.00073, sd 0.00056) and the
    # ensemble (-0.00036 against a 0.00009 spread, 4x). xgboost's own delta
    # (-0.00017, sd 0.00022) is inside its noise, so xgboost is not what is
    # carrying this.
    "REB": ("REB_HOT_Z", "REB_STREAK_ABOVE", "REB_STREAK_BELOW",
            "MINUTES_TREND_RATIO", "MINUTES_STABLE"),
    # AST GETS NONE, MEASURED -- and this entry was populated before it was
    # measured, on the correlation screen alone. The measurement did not support
    # it. Every Brier delta sat at or below its own fold spread (catboost
    # -0.00031 against sd 0.00069; xgboost -0.00012 against 0.00014; ensemble
    # -0.00017 against 0.00013), line_aware got WORSE (+0.00042, 1/3 folds), and
    # the ensemble's raw ECE was worse on 0/3 folds at 3x its spread (+0.00117,
    # sd 0.00037). A delta smaller than the fold-to-fold spread is not an
    # improvement, so the columns come back out.
    "AST": (),
    # NOT YET MEASURED. These were populated from the correlation screen at the
    # same time AST was, and AST is exactly why an unmeasured entry does not
    # belong here: a column can be genuinely new (r < 0.45 against everything
    # listed) and still not help. Run
    #   python -m scripts.feature_ab --layer form --markets FG3M \
    #     --panel data/external/training_pack/panel.parquet \
    #     --seasons-only 2022-23,2023-24,2024-25,2025-26 --folds 3
    # with --wire-under-test, and populate whichever markets win.
    "FG3M": (),
    "STL": (),
    "BLK": (),
    # PRA has no HOT_Z or STREAK columns to begin with: hot_hand covers the six
    # single stats and _STREAK_STATS covers four, neither of them the combo. Its
    # halflife columns exist and are excluded above with the rest of that family.
    # Unmeasured like the three above.
    "PRA": (),
}


# Which shot-mix signal bears on which market. A rebound prop does not care
# how far out a player shoots; a points prop does, because a rim attempt and
# a long two are worth the same in the box score and not in expectation.
_PBP_BY_MARKET: dict[str, tuple[str, ...]] = {
    # PTS gets NONE, measured. With event logs for three seasons and both arms
    # inside them, the seven-feature PTS set above made every tree model
    # slightly worse: xgboost Brier +0.00061, ensemble +0.00036, catboost
    # +0.00030, each within its own fold spread but all in the same direction,
    # and xgboost's ECE +0.00350. The plausible reading is that a scorer's
    # volume is already carried by PTS_L10 and MIN_L5, and seven weak
    # correlated columns cost more in variance than they return. Whoever wants
    # a smaller PTS subset should measure it rather than restore this one.
    "PTS": (),
    "FG3M": ("PBP_THREE_RATE_L5", "PBP_THREE_RATE_L10", "PBP_SHOT_DIST_AVG_L5"),
    # Where a team shoots from changes where the ball comes off; a rim-heavy
    # diet produces different rebound chances than a three-heavy one. This is
    # where the event log pays: on three seasons, EVERY model improved on all
    # three folds -- catboost Brier -0.00130 against a fold spread of 0.00036,
    # ensemble -0.00084 against 0.00021, line_aware -0.00129 and its ECE
    # -0.00967. Mean deltas at three to four times their own noise.
    "REB": ("PBP_RIM_RATE_L10", "PBP_GARBAGE_SHOT_SHARE_L10",
            "PBP_GAME_PACE_L10"),
    # An assist needs a teammate's make. How much of a player's own scoring is
    # created for him says something about his role in the offence. Smaller
    # than rebounds but just as consistent: ensemble Brier -0.00040 against a
    # 0.00009 spread and xgboost -0.00065 against 0.00019, both 3/3 folds,
    # with xgboost's ECE -0.00300 on all three.
    "AST": ("PBP_ASSISTED_RATE_L10", "PBP_GARBAGE_SHOT_SHARE_L10",
            "PBP_GAME_PACE_L10"),
    "PRA": (
        "PBP_RIM_RATE_L10", "PBP_THREE_RATE_L5", "PBP_ASSISTED_RATE_L10",
    ),
}


# Which defensive rate actually bears on which market. The naive mirror --
# "opponent STL allowed" for a steals prop -- is wrong twice over: steals are
# made BY a defence, not allowed by it, and what drives a player's steal
# count is how loose the OPPONENT is with the ball. Same for blocks, which
# need shot volume to exist at all.
_DEFENSE_BY_MARKET: dict[str, tuple[str, ...]] = {
    # PTS needs nothing extra: DEF_RATING_L10 already IS points allowed per
    # 100 possessions, and is in the universal set above.
    "PTS": (),
    # A rebound needs a miss, so how well the opponent shoots against this
    # defence matters as much as how many boards it concedes.
    "REB": ("DEF_REB_ALLOWED_PER100_L10", "DEF_FG_PCT_ALLOWED_L10"),
    "AST": ("DEF_AST_ALLOWED_PER100_L10",),
    "FG3M": ("DEF_FG3M_ALLOWED_PER100_L10", "DEF_FG_PCT_ALLOWED_L10"),
    # Steals come from opponent turnovers, not from the opponent's own steals.
    "STL": ("DEF_TOV_FORCED_PER100_L10",),
    # Blocks need shots to block.
    "BLK": ("DEF_FGA_ALLOWED_PER100_L10",),
    "PRA": ("DEF_REB_ALLOWED_PER100_L10", "DEF_AST_ALLOWED_PER100_L10"),
}


def attach_research_over_labels(
    df: pd.DataFrame,
    *,
    stat: str = "PTS",
    line_col: str | None = None,
) -> pd.DataFrame:
    """
    Attach ``RESEARCH_LINE`` and the binary ``over_hit`` target.

    ``over_hit`` is 1 when the realised stat finished strictly above the
    line, 0 when strictly below. Exact ties on a whole-number line are a
    PUSH and are dropped from the target (left NaN), because grading a
    push as a loss would bias the label set.

    Pass ``line_col`` to label against real prop lines instead of the
    research stand-in.
    """
    stat = stat.upper()
    work = df.copy()

    if line_col is not None:
        if line_col not in work.columns:
            raise ValueError(f"DATA_NOT_AVAILABLE: line column {line_col!r} not present")
        work[RESEARCH_LINE_COL] = pd.to_numeric(work[line_col], errors="coerce")
        line_type = line_col
    else:
        source_col = f"{stat}_{RESEARCH_LINE_SOURCE}"
        if source_col not in work.columns:
            raise ValueError(
                f"DATA_NOT_AVAILABLE: {source_col} missing — cannot build a research "
                f"line for {stat}. Run build_feature_matrix first."
            )
        work[RESEARCH_LINE_COL] = pd.to_numeric(work[source_col], errors="coerce")
        line_type = f"research_{RESEARCH_LINE_SOURCE.lower()}"

    work["line_type"] = line_type

    if stat not in work.columns:
        raise ValueError(f"DATA_NOT_AVAILABLE: realised {stat} column missing — cannot label")

    actual = pd.to_numeric(work[stat], errors="coerce")
    line = work[RESEARCH_LINE_COL]

    over = actual > line
    under = actual < line
    work[TARGET_COL] = pd.Series(pd.NA, index=work.index, dtype="Float64")
    work.loc[over, TARGET_COL] = 1.0
    work.loc[under, TARGET_COL] = 0.0

    pushes = int((actual == line).sum())
    labelled = int(work[TARGET_COL].notna().sum())
    logger.info(
        "Labelled %s: %d rows (%d pushes dropped, line_type=%s)",
        stat, labelled, pushes, line_type,
    )
    if labelled == 0:
        logger.warning("No labelled rows for %s — every row lacked a line or an outcome.", stat)
    return work


def mask_probabilities_at_unsupported_lines(
    probabilities: pd.Series,
    features: pd.DataFrame,
    requested_line: "float | pd.Series",
    *,
    model_name: str,
    tolerance: float = 1e-6,
) -> tuple[pd.Series, int]:
    """
    Null out classifier probabilities asked for at a line they cannot answer.

    WHY: a binary classifier trained on ``over_hit`` learns P(stat > L) for
    the ONE line L its labels were built from — ``RESEARCH_LINE``. The line
    is not one of its inputs, so asking the same fitted model for a
    probability at a different line returns the identical number. That
    number is not wrong about nothing; it is a confident, precise answer to
    a question nobody asked, published beside the line the caller *did* ask
    about.

    Returning NaN is the honest alternative: the model has no opinion at
    that line. Callers that need arbitrary lines should use the
    distribution path, which derives them from a fitted count distribution
    and is correct at any line.

    Returns ``(masked_probabilities, n_masked)``.
    """
    if RESEARCH_LINE_COL not in features.columns:
        # Nothing to compare against, so validity cannot be established —
        # and an unverifiable probability is not a usable one.
        logger.warning(
            "%s: no %s column, so the scoring line cannot be checked against the "
            "labelled line. Abstaining on all %d rows rather than publishing "
            "probabilities that may belong to a different line.",
            model_name, RESEARCH_LINE_COL, len(features),
        )
        return pd.Series(float("nan"), index=features.index, dtype=float), len(features)

    labelled = pd.to_numeric(features[RESEARCH_LINE_COL], errors="coerce")
    asked = (
        pd.to_numeric(requested_line, errors="coerce")
        if isinstance(requested_line, pd.Series)
        else pd.Series(float(requested_line), index=features.index, dtype=float)
    )

    # A NaN on either side is itself unanswerable.
    mismatched = ~((labelled - asked).abs() <= tolerance)

    out = pd.Series(probabilities, index=features.index, dtype=float).copy()
    n_masked = int(mismatched.sum())
    if n_masked:
        out.loc[mismatched] = float("nan")
        logger.warning(
            "%s: %d of %d rows were scored at a line differing from the labelled "
            "%s. The classifier has no opinion at those lines, so they abstain. "
            "Use the distribution model for arbitrary lines.",
            model_name, n_masked, len(out), RESEARCH_LINE_COL,
        )
    return out, n_masked
