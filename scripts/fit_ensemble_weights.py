"""
scripts/fit_ensemble_weights.py — choose ensemble weights from data, not from taste.

The weights in config/model_comparison.yaml have always carried the comment
"starting defaults, not claimed optima". This measures them.

HOW THE WEIGHTS ARE CHOSEN WITHOUT SEEING THE EVALUATION SET. The training
window is split chronologically again: components are fitted on the earlier
part and predict the later part, and the weights are optimised against those
held-out probabilities. The evaluation window the models are finally scored
on is never read here. Choosing weights on it would be fitting on the test
set as surely as training on it.

WHAT IS OPTIMISED. Non-negative weights summing to one, minimising log-loss
on the held-out slice. Log-loss rather than Brier because a blend is a
probability statement and log-loss punishes a confident wrong one properly.

A FITTED WEIGHT IS NOT A PROMISE. The slice is one period of one window; a
weight that wins by less than the spread across markets has not been shown
to be better than the default. Both are printed.

RESEARCH ONLY.

Usage:
    python -m scripts.fit_ensemble_weights --panel <parquet> --markets PTS,REB,AST
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("fit_ensemble_weights")

MIN_HOLDOUT_ROWS = 500


def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def optimise_weights(
    probs: pd.DataFrame, y: np.ndarray, *, seed: int = 42
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Non-negative weights on the simplex minimising log-loss.

    Returns (weights, scores). Components are the frame's columns; rows with
    any missing probability are dropped, because a blend needs every member
    to have answered.
    """
    from scipy.optimize import minimize

    usable = probs.notna().all(axis=1) & np.isfinite(y)
    X = probs.loc[usable].to_numpy(dtype=float)
    yy = y[usable.to_numpy()]
    if len(yy) < MIN_HOLDOUT_ROWS:
        raise ValueError(
            f"DATA_NOT_AVAILABLE: only {len(yy)} rows where every component "
            f"answered; need {MIN_HOLDOUT_ROWS} to fit weights"
        )

    n = X.shape[1]
    start = np.full(n, 1.0 / n)

    def objective(w: np.ndarray) -> float:
        return _log_loss(yy, X @ w)

    result = minimize(
        objective, start, method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"maxiter": 500, "ftol": 1e-10},
    )
    w = np.clip(result.x, 0.0, None)
    w = w / w.sum() if w.sum() > 0 else start
    weights = {c: float(round(v, 4)) for c, v in zip(probs.columns, w)}
    scores = {
        "fitted_log_loss": _log_loss(yy, X @ w),
        "equal_weight_log_loss": _log_loss(yy, X @ start),
        "best_single_log_loss": float(min(_log_loss(yy, X[:, i]) for i in range(n))),
        "n_rows": float(len(yy)),
    }
    return weights, scores


def held_out_probabilities(
    panel: pd.DataFrame, market: str, cfg: dict, *, train_end: str, fit_fraction: float
) -> tuple[pd.DataFrame, np.ndarray]:
    """Fit components on the earlier training rows; predict the later ones."""
    from src.models.compare import (
        build_components,
        prepare_market_panel,
        resolve_feature_cols,
    )
    from src.models.labels import default_feature_cols

    work = prepare_market_panel(panel, market)
    work = work[work["over_hit"].notna()].reset_index(drop=True)
    work = work[work["GAME_DATE"] <= pd.Timestamp(train_end)].reset_index(drop=True)
    if work.empty:
        raise ValueError(f"DATA_NOT_AVAILABLE: no labelled {market} rows before {train_end}")

    xgb_cols, _ = resolve_feature_cols(work, list(default_feature_cols(market)))
    feature_cols = list(xgb_cols)
    for c in ("TEAM_ABBREVIATION", "OPPONENT_ABBREVIATION", "SEASON"):
        if c in work.columns and c not in feature_cols:
            feature_cols.append(c)

    cut = int(len(work) * fit_fraction)
    fit_rows, hold = work.iloc[:cut], work.iloc[cut:]
    # Drop features with no coverage in the rows actually fitted on, exactly
    # as compare_models_on_panel does.
    empty = [c for c in feature_cols if c in fit_rows.columns
             and not fit_rows[c].notna().any()]
    feature_cols = [c for c in feature_cols if c not in empty]
    xgb_cols = [c for c in xgb_cols if c not in empty]
    logger.info(
        "%s: fitting on %d rows (%s..%s), choosing weights on %d rows (%s..%s)",
        market, len(fit_rows), fit_rows["GAME_DATE"].min().date(),
        fit_rows["GAME_DATE"].max().date(), len(hold),
        hold["GAME_DATE"].min().date(), hold["GAME_DATE"].max().date(),
    )

    components = build_components(market, feature_cols, cfg, xgb_feature_cols=xgb_cols)
    out: dict[str, np.ndarray] = {}
    for name, model in components.items():
        try:
            model.fit(fit_rows, hold)
            preds = model.predict_rows(hold, line_col="RESEARCH_LINE")
        except Exception as exc:  # noqa: BLE001 — a component that cannot fit gets no weight
            logger.warning("%s: %s could not be fitted or scored (%s)", market, name, exc)
            continue
        out[name] = np.array(
            [p.probability_over if p.probability_over is not None else np.nan
             for p in preds], dtype=float,
        )
    if len(out) < 2:
        raise ValueError(f"DATA_NOT_AVAILABLE: {market} produced fewer than two components")
    return pd.DataFrame(out, index=hold.index), hold["over_hit"].astype(float).to_numpy()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--panel", default="data/external/training_pack/panel.parquet")
    ap.add_argument("--markets", default="PTS,REB,AST")
    ap.add_argument("--train-end", default="2025-10-01")
    ap.add_argument("--fit-fraction", type=float, default=0.8)
    ap.add_argument("--out", default="outputs/fitted_ensemble_weights.json")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    logger.setLevel(logging.INFO)

    from src.models.compare import load_comparison_config

    cfg = load_comparison_config()
    configured = cfg.get("ensemble_weights") or {}
    panel = pd.read_parquet(args.panel)
    panel["GAME_DATE"] = pd.to_datetime(panel["GAME_DATE"])

    print(f"Weights are chosen on training rows only (<= {args.train_end}). "
          f"The evaluation window is never read.\n")
    print(f"config held at fit time: {configured}\n")

    per_market: dict[str, dict] = {}
    for market in [m.strip().upper() for m in args.markets.split(",") if m.strip()]:
        probs, y = held_out_probabilities(
            panel, market, cfg, train_end=args.train_end,
            fit_fraction=args.fit_fraction,
        )
        weights, scores = optimise_weights(probs, y)
        per_market[market] = {"weights": weights, "scores": scores}
        print(f"=== {market} ===")
        print(f"  held-out rows: {int(scores['n_rows']):,}")
        for name, w in sorted(weights.items(), key=lambda kv: -kv[1]):
            mark = "  (new)" if name not in configured else ""
            print(f"    {name:<14}{w:6.3f}{mark}")
        print(f"  log-loss  fitted {scores['fitted_log_loss']:.5f}"
              f"   equal {scores['equal_weight_log_loss']:.5f}"
              f"   best single {scores['best_single_log_loss']:.5f}")
        # What the configured weights would have scored on the same rows.
        common = [c for c in probs.columns if configured.get(c, 0) > 0]
        if common:
            w = np.array([configured[c] for c in common], dtype=float)
            w = w / w.sum()
            usable = probs[common].notna().all(axis=1)
            got = _log_loss(y[usable.to_numpy()],
                            probs.loc[usable, common].to_numpy() @ w)
            print(f"  configured weights on the same rows: {got:.5f}")
        print()

    names = sorted({n for m in per_market.values() for n in m["weights"]})
    pooled = {
        n: float(round(np.mean([m["weights"].get(n, 0.0) for m in per_market.values()]), 3))
        for n in names
    }
    total = sum(pooled.values())
    pooled = {k: round(v / total, 3) for k, v in pooled.items()} if total else pooled
    print("Averaged across markets (one weight set, as the config holds one):")
    for name, w in sorted(pooled.items(), key=lambda kv: -kv[1]):
        print(f"    {name:<14}{w:6.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # The key used to be "configured", which reads as "what ships". It is not:
    # it is what the config held WHEN THIS FIT RAN. The usual workflow writes
    # the fitted weights back into the config afterwards, at which point the
    # old name made the artifact quietly contradict config/model_comparison.yaml
    # -- it recorded {catboost 0.5, xgboost 0.3, distribution 0.2} long after
    # the config had moved to the fitted set. Name it for what it is, stamp it,
    # and say plainly that it is a historical record.
    out.write_text(json.dumps(
        {
            "_note": (
                "configured_at_fit_time is what config/model_comparison.yaml "
                "held when this file was written, NOT necessarily what ships "
                "now. Read the config for that."
            ),
            "fitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "panel": str(args.panel),
            "train_end": str(args.train_end),
            "configured_at_fit_time": configured,
            "per_market": per_market,
            "pooled": pooled,
        },
        indent=2, default=str,
    ), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
