"""
scripts/feature_ab.py — run the model comparison with a feature layer on and
off, and print Brier and ECE for both arms, raw and calibrated.

WHY THIS EXISTS. "Does feature X improve accuracy?" was being answered from
intuition, and intuition was wrong often enough to be worth mechanising. The
harness fits the same models on the same rows with the same chronological
split, changing only whether one additive layer's columns are present, and
reports all four numbers side by side with the delta.

READ THE DELTA, NOT THE LEVEL. A lower Brier in the treated arm is the claim;
the absolute values depend on the market, the window and the panel, and say
nothing on their own. A delta smaller than the fold-to-fold spread is not an
improvement, it is noise wearing a decimal point.

RESEARCH ONLY. This compares model accuracy. It does not price a bet, size a
stake, or recommend a wager.

Usage:
    python -m scripts.feature_ab --layer blowout --markets PTS
    python -m scripts.feature_ab --layer market_context --markets PTS,REB,AST
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger("feature_ab")

# Layers this harness knows how to toggle, each described by the columns it
# owns and, where the panel does not already carry them, how to attach them.
#
# Two arms are built from ONE feature build, never from two. A layer the
# panel already has is removed to make the control; a layer it lacks is
# attached to make the treatment. Building the matrix twice could differ for
# reasons other than the layer under test, which is the one thing this
# harness exists to rule out.
def _attach_blowout(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    from src.features.blowout import DEFAULT_SPREAD_THRESHOLD, attach_blowout_features

    layer_cfg = cfg.get("blowout") or {}
    return attach_blowout_features(
        df,
        spread_threshold=float(
            layer_cfg.get("spread_threshold", DEFAULT_SPREAD_THRESHOLD)
        ),
        required=True,
    )


class Layer:
    """A set of columns that can be present or absent, and how to get them."""

    def __init__(
        self,
        columns: tuple[str, ...],
        attach: Callable[[pd.DataFrame, dict], pd.DataFrame] | None = None,
        note: str = "",
    ) -> None:
        self.columns = columns
        self.attach = attach
        self.note = note

    def arms(
        self, panel: pd.DataFrame, cfg: dict[str, Any]
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
        """Return (control, treatment, columns_under_test)."""
        present = [c for c in self.columns if c in panel.columns]
        if present:
            # Subtractive: the panel has them, so the control is the panel
            # without them. Used for layers that are attached during the
            # feature build and cannot be bolted on afterwards.
            return panel.drop(columns=present), panel, present
        if self.attach is None:
            raise RuntimeError(
                f"none of {list(self.columns)} are on the panel and this layer "
                "cannot attach them — supply the frame the feature build needs"
            )
        treated = self.attach(panel, cfg)
        gained = [c for c in self.columns if c in treated.columns]
        return panel, treated, gained


LAYERS: dict[str, Layer] = {
    "blowout": Layer(
        ("BLOWOUT_FAV_HINGE", "BLOWOUT_DOG_HINGE"),
        attach=_attach_blowout,
        note="Spread hinges. Off by default — see src/features/blowout.py.",
    ),
    "defense": Layer(
        (
            "DEF_RATING_L10",
            "DEF_RATING_INDEX_L10",
            "DEF_PACE_L10",
            "DEF_REB_ALLOWED_PER100_L10",
            "DEF_AST_ALLOWED_PER100_L10",
            "DEF_FG3M_ALLOWED_PER100_L10",
            "DEF_FGA_ALLOWED_PER100_L10",
            "DEF_TOV_FORCED_PER100_L10",
            "DEF_FG_PCT_ALLOWED_L10",
        ),
        note=(
            "Opponent defence per 100 possessions. Needs the team_games frame at "
            "build time, so this arm is subtractive: supply the workbook and the "
            "control drops the columns."
        ),
    ),
    "pbp": Layer(
        (
            "PBP_GAME_PACE_L5", "PBP_GAME_PACE_L10",
            "PBP_SHOT_DIST_AVG_L5", "PBP_SHOT_DIST_AVG_L10",
            "PBP_RIM_RATE_L5", "PBP_RIM_RATE_L10",
            "PBP_MID_RATE_L5", "PBP_MID_RATE_L10",
            "PBP_THREE_RATE_L5", "PBP_THREE_RATE_L10",
            "PBP_DUNK_LAYUP_RATE_L5", "PBP_DUNK_LAYUP_RATE_L10",
            "PBP_ASSISTED_RATE_L5", "PBP_ASSISTED_RATE_L10",
            "PBP_CLOSE_SHOT_SHARE_L5", "PBP_CLOSE_SHOT_SHARE_L10",
            "PBP_GARBAGE_SHOT_SHARE_L5", "PBP_GARBAGE_SHOT_SHARE_L10",
        ),
        note=(
            "Shot mix and game-state context from the event log, as prior-game "
            "rolling means. Only exists for seasons whose logs were supplied, so "
            "both arms must sit inside those seasons."
        ),
    ),
    # Only _L10 is read by any market (labels.py), so _L5 rides along in the
    # panel but changes nothing when toggled. Both are listed so the arm
    # removes the whole column family rather than half of it.
    "pbp_pace": Layer(
        ("PBP_GAME_PACE_L5", "PBP_GAME_PACE_L10"),
        note=(
            "Game pace from the event log, on its own. PBP_GAME_PACE is a GAME "
            "constant -- every player in a game shares the value -- so this asks "
            "whether game pace earns a place beside DEF_PACE_L10, which is the "
            "only other pace column any market reads (r = 0.04 between them; "
            "PACE_ROLL correlates 0.82 but no model reads it)."
        ),
    ),
    "fouls": Layer(
        ("PF_L5", "PF_L10", "PF_SEASON", "PF_PER_MIN_L10", "PF_TROUBLE_RATE_L10"),
        note=(
            "The player's own prior foul history. Six fouls ends a night, so "
            "this asks whether foul propensity carries minutes risk the "
            "minutes columns do not already describe. Needs a panel built "
            "after PF was mapped -- from the archive's foulsPersonal or the "
            "league game log's PF, both of which were carried and neither of "
            "which was requested until now. No market reads these columns, "
            "so run it with --wire-under-test."
        ),
    ),
    "dvp": Layer(
        (
            "DVP_PTS_ALLOWED_L10", "DVP_REB_ALLOWED_L10", "DVP_AST_ALLOWED_L10",
            "DVP_FG3M_ALLOWED_L10", "DVP_STL_ALLOWED_L10", "DVP_BLK_ALLOWED_L10",
            "DVP_PTS_INDEX_L10", "DVP_REB_INDEX_L10", "DVP_AST_INDEX_L10",
            "DVP_FG3M_INDEX_L10", "DVP_STL_INDEX_L10", "DVP_BLK_INDEX_L10",
        ),
        note=(
            "Opponent defence split by the position it is defending -- the "
            "first column in this panel that varies by who the player is as "
            "well as by whom he faces. The question it settles is whether it "
            "adds anything to DEF_RATING_L10, which hands every player in a "
            "game the same number. Covers the 87.2% of rows with an as-of "
            "position bucket; no market reads these columns, so run it with "
            "--wire-under-test."
        ),
    ),
    # The four layers that ran on every build while no market read a column
    # they produce. Both arms are subtractive: the real panel already carries
    # every column, so the control is the panel without them.
    "form": Layer(
        (
            "PTS_HOT_Z", "REB_HOT_Z", "AST_HOT_Z", "FG3M_HOT_Z",
            "STL_HOT_Z", "BLK_HOT_Z",
            "PTS_STREAK_ABOVE", "PTS_STREAK_BELOW",
            "REB_STREAK_ABOVE", "REB_STREAK_BELOW",
            "AST_STREAK_ABOVE", "AST_STREAK_BELOW",
            "FG3M_STREAK_ABOVE", "FG3M_STREAK_BELOW",
            "MINUTES_TREND_RATIO", "MINUTES_STABLE",
            "TS_PCT_L10", "TS_PCT_TREND", "FT_RATE_L10",
        ),
        note=(
            "Recent form and shot quality: hot_hand z-scores, form streaks, "
            "minutes shape, true-shooting. Every column here correlates below "
            "0.45 with each feature its market already reads -- the redundant "
            "members of these layers are listed in labels._EXCLUDED_AS_REDUNDANT "
            "and are NOT under test. These columns ARE wired, so this arm runs "
            "without --wire-under-test."
        ),
    ),
    "halflife": Layer(
        (
            "PTS_HL", "PTS_HL_SHRINK", "PTS_L2_HL",
            "REB_HL", "REB_HL_SHRINK", "REB_L2_HL",
            "AST_HL", "AST_HL_SHRINK", "AST_L2_HL",
            "FG3M_HL", "FG3M_HL_SHRINK", "FG3M_L2_HL",
            "MIN_HL", "MIN_HL_SHRINK",
        ),
        note=(
            "Exponential half-life means and their shrunk forms. Predicted "
            "redundant: r = 0.96-0.99 against {M}_SEASON, {M}_L2 and MIN_SEASON. "
            "Listed so the prediction can be tested rather than asserted -- a "
            "shrunk estimate can still behave better than the raw mean it "
            "mirrors, which correlation alone cannot rule out. NEEDS "
            "--wire-under-test: these columns are deliberately absent from every "
            "feature list, so without it the run refuses by design."
        ),
    ),
    "usage_volume": Layer(
        (
            "USAGE_PROXY_L10", "SHOT_VOLUME_L5", "SHOT_VOLUME_L10",
            "FGA_L5", "FGA_L10",
        ),
        note=(
            "Usage proxy and shot volume. Predicted redundant for PTS "
            "(r = 0.955-0.969 against PTS_L10 and PTS_BASELINE) and moderate "
            "for REB/AST (0.835-0.848 against MIN_L10). NEEDS --wire-under-test."
        ),
    ),
    "opp_allowed_per_game": Layer(
        (
            "OPP_PTS_ALLOWED_L10", "OPP_REB_ALLOWED_L10", "OPP_AST_ALLOWED_L10",
            "OPP_FG3M_ALLOWED_L10", "OPP_STL_ALLOWED_L10", "OPP_BLK_ALLOWED_L10",
        ),
        note=(
            "Opponent allowed per GAME, against the listed DEF_* columns which "
            "are per 100 POSSESSIONS. Per-game allowed confounds defensive "
            "quality with tempo, the confound DEF_PACE_L10 exists to separate. "
            "NEEDS --wire-under-test."
        ),
    ),
    "minutes_weighted": Layer(
        tuple(
            f"{stat}_MW_L5" for stat in ("PTS", "REB", "AST", "STL", "BLK", "FG3M")
        ) + tuple(f"{combo}_MW_L5" for combo in ("PR", "PA", "RA", "PRA")),
        note=(
            "Minutes-weighted L5. Built by the builder but absent from "
            "default_feature_cols on purpose: |r| 0.976-0.992 against {STAT}_L5 "
            "on the 214,381-row panel, inside the band halflife was excluded "
            "for. Run with --wire-under-test to measure before promoting."
        ),
    ),
    "fatigue_load": Layer(
        ("FATIGUE_LOAD_L7", "FATIGUE_LOAD_MIN_L7"),
        note=(
            "Exponential cumulative load: prior minutes decayed by recency, "
            "scaled by flown miles and time-zone change. Tests whether one "
            "continuous quantity beats the four unfitted constants in "
            "fatigue_logic.py (B2B 0.97, 3-in-4 0.96, 4-in-5 0.94, altitude "
            "0.98), which fatigue_multiplier already contributes to the "
            "baseline arm. Run BOTH columns and then FATIGUE_LOAD_MIN_L7 "
            "alone: a gain present in the full column and absent in the "
            "minutes-only one is evidence about travel specifically, and a "
            "gain in both is not. NEEDS --wire-under-test."
        ),
    ),
    "market_context": Layer(
        (
            "MKT_OPENING_SPREAD",
            "MKT_OPENING_TOTAL",
            "MKT_IMPLIED_TEAM_TOTAL",
            "MKT_IMPLIED_OPP_TOTAL",
            "MKT_IS_FAVORITE",
        ),
        note=(
            "The market's own pregame forecast. Needs the market_lines frame at "
            "build time, so this arm is subtractive: supply the workbook and the "
            "control drops the columns."
        ),
    ),
}


# The stat prefixes that make a column market-specific. Used only by
# --wire-under-test, to keep one market's columns out of another's arm.
_STAT_PREFIXES = ("PTS", "REB", "AST", "FG3M", "STL", "BLK", "PRA")

# The two defence columns every market reads, from the universal block of
# labels.default_feature_cols. Everything else DEF_* is assigned per market by
# labels._DEFENSE_BY_MARKET and must not be handed to a market it is not for.
_UNIVERSAL_DEFENSE_COLS = frozenset({"DEF_RATING_L10", "DEF_PACE_L10"})

# The three stats PRA is the sum of, so a PRA arm sees all three matchups.
# Mirrors labels._DEFENSE_BY_MARKET["PRA"].
_PRA_COMPONENTS = frozenset({"PTS", "REB", "AST"})


def column_for_market(column: str, market: str) -> bool:
    """
    Should ``--wire-under-test`` offer this column to ``market``'s arm?

    PER MARKET, not every column to every market. A first version appended all
    of them to all of them, so a PTS run was handed AST_HL and REB_HL_SHRINK
    -- which is not the question being asked, and is not what labels.py does
    either: every dict there is keyed by market. A column carrying another
    market's stat prefix is skipped; MIN_*, MINUTES_*, TS_PCT_*,
    USAGE_PROXY_* and the rest are market-neutral and go to all.

    MODULE LEVEL, and this is the function the run itself calls. It used to be
    a closure inside _run, which meant the only thing a test could reach was
    the helper it delegates to -- so deleting the delegation went unnoticed.
    Both the rule and the wiring are now covered by
    tests/test_dvp.py::test_each_market_is_offered_only_its_own_matchup_columns.
    """
    from src.models.labels import _DEFENSE_BY_MARKET

    market = market.upper()
    # A DEF_* column is not market-neutral just because it carries no stat
    # prefix: labels._DEFENSE_BY_MARKET assigns each one to the markets it
    # bears on, and DEF_REB_ALLOWED_PER100_L10 belongs to REB and PRA, not to
    # PTS. Treating the whole family as neutral meant `--layer defense
    # --wire-under-test --markets PTS` fed a points model the rebound and
    # assist defence columns -- not the question being asked, and the same
    # defect as the stat-prefix one below.
    if column.startswith("DEF_"):
        if column in _UNIVERSAL_DEFENSE_COLS:
            return True
        return column in set(_DEFENSE_BY_MARKET.get(market, ()))
    # DVP_* carries its stat in the MIDDLE, not at the front, so the prefix
    # loop below does not see it: DVP_REB_ALLOWED_L10 starts with neither
    # "REB_" nor "OPP_REB_". Without this branch every market received all
    # twelve, and `--layer dvp --markets PTS` fed a points model the rebound,
    # assist and block matchup columns -- the identical defect the DEF_ branch
    # above exists to fix, reproduced in a launched run before being caught.
    if column.startswith("DVP_"):
        return _dvp_for_market(column, market)
    for stat in _STAT_PREFIXES:
        if stat == market:
            continue
        if column.startswith(f"{stat}_") or column.startswith(f"OPP_{stat}_"):
            return False
    return True


def _dvp_for_market(column: str, market: str) -> bool:
    """
    Does this DVP_* column belong in ``market``'s arm?

    MODULE LEVEL ON PURPOSE, so the routing rule has a test
    (tests/test_dvp.py::test_each_market_is_offered_only_its_own_matchup_columns)
    rather than living only inside the closure that uses it. The rule it
    encodes is not obvious from the column names: DVP_* carries its stat in
    the MIDDLE, so DVP_REB_ALLOWED_L10 starts with neither "REB_" nor
    "OPP_REB_" and the stat-prefix filter in _run does not see it at all.
    """
    if not column.startswith("DVP_"):
        return True
    stat = column[len("DVP_"):].split("_", 1)[0]
    if market.upper() == "PRA":
        return stat in _PRA_COMPONENTS
    return stat == market.upper()


METRICS = (
    ("brier_score", "Brier raw"),
    ("brier_score_calibrated", "Brier cal"),
    ("calibration_error", "ECE raw"),
    ("calibration_error_calibrated", "ECE cal"),
)


def _fold_windows(
    train_end: str, validation_end: str, folds: int, step_days: int
) -> list[tuple[str, str]]:
    """Successive chronological windows, each advanced by ``step_days``.

    Fold 0 is exactly the window the caller asked for; later folds train on
    strictly more history and validate strictly later, so no fold is ever
    scored on rows an earlier fold trained on.
    """
    folds = max(1, int(folds))
    step = pd.Timedelta(days=max(1, int(step_days)))
    t0, v0 = pd.Timestamp(train_end), pd.Timestamp(validation_end)
    if v0 <= t0:
        raise ValueError(
            f"validation_end {validation_end} is not after train_end {train_end}"
        )
    return [
        (str((t0 + i * step).date()), str((v0 + i * step).date()))
        for i in range(folds)
    ]


def _fmt(value: Any) -> str:
    return "     --" if value is None or pd.isna(value) else f"{float(value):7.5f}"


def _delta(on: Any, off: Any) -> str:
    """Signed delta, with the direction spelled out. Lower is better for all
    four metrics here, so a negative delta is the improvement."""
    if on is None or off is None or pd.isna(on) or pd.isna(off):
        return "      --"
    d = float(on) - float(off)
    return f"{d:+8.5f}" + ("  better" if d < 0 else ("  worse" if d > 0 else "  same"))


def main(argv: list[str] | None = None) -> int:
    """Wrapper that guarantees --wire-under-test cannot outlive its own run.

    _run installs a widened default_feature_cols on src.models.compare. main() is
    importable and callable more than once in a process, and a patch left in
    place would silently widen the NEXT layer's arms with this layer's columns --
    so every delta after the first would be measuring something else. Restored
    here whether _run returns or raises.

    NOTHING IS IMPORTED HERE. An earlier version imported src.models.compare at
    the top of this function to capture the original resolver, which pulled the
    whole model stack in before argparse had run -- so `feature_ab --help` needed
    the optional [ml] extras installed. _run hands back what it patched instead,
    and a run that never patched leaves the list empty and nothing to undo.
    """
    patched: list[tuple[Any, Any]] = []
    try:
        return _run(argv, _patched=patched)
    finally:
        for module, original in patched:
            module.default_feature_cols = original


def _run(
    argv: list[str] | None = None,
    *,
    _patched: "list[tuple[Any, Any]] | None" = None,
) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layer", required=True, choices=sorted(LAYERS),
                    help="Which additive feature layer to toggle.")
    ap.add_argument("--markets", default="PTS")
    ap.add_argument("--train-end", default="2025-01-15")
    ap.add_argument("--validation-end", default="2025-02-15")
    ap.add_argument("--folds", type=int, default=1,
                    help="Chronological windows to average over. The delta from "
                         "a single window cannot be told apart from noise, so "
                         "more than one is strongly preferred.")
    ap.add_argument("--step-days", type=int, default=None,
                    help="Days to advance each fold (default: walk_forward.step_days).")
    ap.add_argument("--demo", action="store_true",
                    help="Synthetic panel. Wiring only — the numbers mean nothing.")
    ap.add_argument("--seasons", default=None)
    ap.add_argument("--season-type", default=None)
    ap.add_argument("--panel", default=None,
                    help="A prebuilt feature matrix (parquet) to use instead of "
                         "the loader's own.")
    ap.add_argument("--seasons-only", default=None,
                    help="Restrict BOTH arms to these seasons, comma separated. "
                         "Required when a layer exists for only part of the panel.")
    ap.add_argument("--wire-under-test", action="store_true",
                    help="Add the toggled columns to each market's feature list "
                         "FOR THIS RUN ONLY, so a column that is not yet wired "
                         "can be measured before it is shipped. Without this, a "
                         "layer no market reads cannot be compared at all.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    from scripts.nba_model_cli import _load_real_or_demo
    from src.models.compare import compare_models_on_panel, load_comparison_config

    cfg = load_comparison_config()
    markets = [m.strip().upper() for m in args.markets.split(",") if m.strip()]
    layer = LAYERS[args.layer]

    panel, is_demo = _load_real_or_demo(
        args.demo, args.seasons, args.season_type, args.panel
    )
    if args.seasons_only and "SEASON" in panel.columns:
        keep = [s.strip() for s in args.seasons_only.split(",") if s.strip()]
        before = len(panel)
        panel = panel[panel["SEASON"].isin(keep)].reset_index(drop=True)
        print(f"Restricted to season(s) {keep}: {before:,} -> {len(panel):,} rows. "
              f"Both arms use exactly these rows.\n")
    if is_demo:
        print("DEMO PANEL — synthetic players. These numbers test wiring, not accuracy.\n")

    try:
        control, treatment, under_test = layer.arms(panel, cfg)
    except Exception as exc:  # noqa: BLE001 — a missing input is the answer, not a crash
        print(f"ERROR: cannot build the '{args.layer}' arms: {exc}", file=sys.stderr)
        return 2

    if not under_test:
        print(f"ERROR: the '{args.layer}' layer contributed no columns — "
              "nothing to compare.", file=sys.stderr)
        return 2

    # A layer can be present in the panel and still be read by no model for
    # the markets under test. _PBP_BY_MARKET["PTS"] is deliberately empty on
    # measured evidence, so `--layer pbp` with the default `--markets PTS`
    # produced a table of exact zeros -- a correct answer to a question
    # nobody meant to ask. Say so instead of printing it.
    import src.models.compare as compare_module
    from src.models.labels import default_feature_cols

    if args.wire_under_test:
        # compare.py binds default_feature_cols at import, so the patch has to
        # land on ITS name, not on labels'. Both arms get the widened list: the
        # control frame does not carry these columns, and resolve_feature_cols
        # drops what the frame lacks, so the control trains without them exactly
        # as a subtractive arm does.
        _base = compare_module.default_feature_cols

        def _widened(market, _base=_base, _extra=tuple(under_test)):
            cols = list(_base(market))
            for c in _extra:
                if c not in cols and column_for_market(c, market.upper()):
                    cols.append(c)
            return cols

        # Recorded for main()'s finally, so this patch cannot outlive the run
        # that installed it.
        if _patched is not None:
            _patched.append((compare_module, _base))
        compare_module.default_feature_cols = _widened  # type: ignore[assignment]
        print(f"--wire-under-test: up to {len(under_test)} column(s) added to "
              f"each market's feature list for this run only, routed per "
              f"market by column_for_market -- a column carrying another "
              f"market's stat, in its prefix or inside a DEF_/DVP_ name, is "
              f"skipped. Nothing is written to src/models/labels.py.")

    resolver = (
        compare_module.default_feature_cols if args.wire_under_test
        else default_feature_cols
    )
    read_by = {m: sorted(set(under_test) & set(resolver(m))) for m in markets}
    if not any(read_by.values()):
        print(
            f"ERROR: no market in {markets} reads any '{args.layer}' column. The "
            f"layer is in the panel, but default_feature_cols() selects none of "
            f"{under_test} for these markets, so both arms would train on "
            f"identical features and every delta would be exactly zero. Pick "
            f"markets that read this layer, wire it in src/models/labels.py, or "
            f"pass --wire-under-test to measure it without shipping it.",
            file=sys.stderr,
        )
        return 2
    for market, cols in read_by.items():
        if not cols:
            print(f"  NOTE: {market} reads none of these columns; its arms are "
                  f"identical and its deltas will be exactly zero.")
        else:
            print(f"  {market} reads {len(cols)} of them: {cols}")

    coverage = {c: float(treatment[c].notna().mean()) for c in under_test}
    print(f"Layer '{args.layer}' under test: {under_test}")
    if layer.note:
        print(f"  {layer.note}")
    print("  non-null coverage: "
          + ", ".join(f"{c}={v:.1%}" for c, v in coverage.items()))
    if max(coverage.values()) == 0.0:
        print("  every value is null — the comparison below cannot show a difference.")
    print()

    windows = _fold_windows(
        args.train_end, args.validation_end, args.folds,
        args.step_days or int((cfg.get("walk_forward") or {}).get("step_days", 14)),
    )
    if len(windows) > 1:
        print(f"{len(windows)} chronological folds, "
              f"{windows[0][0]} -> {windows[-1][1]}\n")

    per_fold: list[dict[str, pd.DataFrame]] = []
    for train_end, validation_end in windows:
        fold: dict[str, pd.DataFrame] = {}
        for arm, frame in (("off", control), ("on", treatment)):
            result = compare_models_on_panel(
                frame, markets=markets, train_end=train_end,
                validation_end=validation_end, cfg=cfg,
            )
            fold[arm] = pd.DataFrame(result["summary"])
        if not fold["off"].empty or not fold["on"].empty:
            per_fold.append(fold)

    if not per_fold:
        print("ERROR: no fold produced any scored model.", file=sys.stderr)
        return 2

    arms = {
        arm: pd.concat([f[arm] for f in per_fold], ignore_index=True)
        for arm in ("off", "on")
    }

    n_folds = len(per_fold)
    for market in markets:
        off = arms["off"][arms["off"]["target_market"] == market]
        on = arms["on"][arms["on"]["target_market"] == market]
        if off.empty and on.empty:
            print(f"{market}: no models scored — skipped.\n")
            continue
        print(f"=== {market} ===")
        # SUMMED OVER MODELS, so this is a count of predictions and not of rows:
        # each held-out observation appears once per model. Labelling it
        # "validation rows" got the figure read as distinct rows in three commit
        # messages and two docs headers before anyone noticed it was ~5x too
        # high. Both numbers are printed now.
        arm = off if not off.empty else on
        predictions = int(arm["n_predictions"].sum())
        models = max(int(arm["model_name"].nunique()), 1)
        print(f"    model predictions across {n_folds} fold(s): {predictions} "
              f"({models} model(s), so ~{predictions // models} distinct "
              f"validation rows)")
        if n_folds > 1:
            header = (f"    {'model':<13}{'metric':<12}{'off':>9}{'on':>9}"
                      f"{'delta':>10}{'sd':>9}  folds better")
        else:
            header = f"    {'model':<13}{'metric':<12}{'off':>9}{'on':>9}{'delta':>10}"
        print(header)
        print("    " + "-" * (len(header) - 4))
        for model in sorted(set(off["model_name"]) | set(on["model_name"])):
            a_all = off[off["model_name"] == model]
            b_all = on[on["model_name"] == model]
            for key, label in METRICS:
                a = pd.to_numeric(a_all[key], errors="coerce")
                b = pd.to_numeric(b_all[key], errors="coerce")
                line = (f"    {model:<13}{label:<12}"
                        f"{_fmt(a.mean()):>9}{_fmt(b.mean()):>9}")
                if n_folds > 1 and len(a) == len(b) and len(a):
                    d = (b.to_numpy() - a.to_numpy())
                    finite = np.isfinite(d)
                    if finite.any():
                        d = d[finite]
                        line += (f"{d.mean():+10.5f}{d.std():9.5f}"
                                 f"   {int((d < 0).sum())}/{len(d)}")
                    else:
                        line += f"{'--':>10}{'--':>9}   --"
                else:
                    line += f"  {_delta(b.mean() if len(b) else None, a.mean() if len(a) else None)}"
                print(line)
            print()
    print("Lower is better for every metric above. A mean delta smaller than the "
          "fold-to-fold sd is not evidence;")
    print("neither is a single fold, which is why --folds defaults to a number "
          "you should raise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
