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

    panel, is_demo = _load_real_or_demo(args.demo, args.seasons, args.season_type)
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
        rows = int(off["n_predictions"].sum()) if not off.empty else int(on["n_predictions"].sum())
        print(f"    validation rows across {n_folds} fold(s): {rows}")
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
