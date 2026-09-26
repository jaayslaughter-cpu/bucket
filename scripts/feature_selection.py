"""
scripts/feature_selection.py — rank predictors for a market, on training rows only.

SELECTION IS PART OF TRAINING. Choosing features by looking at the validation
window is leakage exactly as much as fitting on it: the chosen set carries
information about rows the model is later scored against, and every metric
afterwards is flattered. Everything here is computed strictly before
``--train-end``, and the evaluation window is never read.

TWO MEASURES, BECAUSE THEY DISAGREE FOR DIFFERENT REASONS.

  GAIN is free -- the booster already has it -- but it is biased toward
  features with many distinct values, which a continuous rolling mean has and
  a binary flag does not. Read it as "what the trees split on", not as "what
  matters".

  PERMUTATION importance shuffles one column in held-out rows and measures how
  much log-loss worsens. It answers the question actually being asked: if this
  feature were noise, how much worse would the model be? It costs a refit's
  worth of predictions and is the one to trust when the two disagree.

STABILITY IS REPORTED, NOT HIDDEN. A feature that ranks first in one fold and
twentieth in the next has not been shown to matter. The ``folds_top10`` column
says how often each feature reached the top ten, and a mean importance smaller
than its own standard deviation is noise with a decimal point.

RESEARCH ONLY. Ranks predictors; places no bets.

Usage:
    python -m scripts.feature_selection --markets PTS,REB,AST
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("feature_selection")


def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _auc(y: np.ndarray, x: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 50:
        return float("nan")
    y, x = y[ok], x[ok]
    pos, neg = y == 1, y == 0
    if not pos.any() or not neg.any():
        return float("nan")
    ranks = pd.Series(x).rank().to_numpy()
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2)
                 / (pos.sum() * neg.sum()))


def rank_features(
    panel: pd.DataFrame,
    market: str,
    *,
    train_end: str,
    n_folds: int = 4,
    permutation_repeats: int = 3,
    permutation_cap: int = 20000,
    seed: int = 42,
) -> pd.DataFrame:
    from xgboost import XGBClassifier

    from src.models.compare import load_comparison_config, resolve_feature_cols
    from src.models.labels import attach_research_over_labels, default_feature_cols
    from src.models.xgboost_pipeline import split_xgboost_config

    work = attach_research_over_labels(panel, stat=market)
    work = work[work["over_hit"].notna()].sort_values("GAME_DATE").reset_index(drop=True)
    # Everything at or before the cutoff. The evaluation window is not read.
    work = work[work["GAME_DATE"] <= pd.Timestamp(train_end)].reset_index(drop=True)
    if work.empty:
        raise ValueError(f"DATA_NOT_AVAILABLE: no labelled {market} rows before {train_end}")

    cols, dropped = resolve_feature_cols(work, list(default_feature_cols(market)))
    if dropped:
        logger.info("%s: %d feature(s) absent from the panel: %s", market, len(dropped), dropped)
    X = work[cols].apply(pd.to_numeric, errors="coerce")
    y = work["over_hit"].astype(float).to_numpy()

    params, tuning = split_xgboost_config((load_comparison_config().get("xgboost") or {}))
    params = {k: v for k, v in params.items() if k != "n_estimators"}
    rng = np.random.default_rng(seed)

    gain_rows: list[dict] = []
    perm_rows: list[dict] = []
    for fold in range(n_folds):
        cut = int(len(work) * (0.50 + 0.1 * fold))
        end = int(len(work) * (0.60 + 0.1 * fold))
        if end - cut < 200 or len(np.unique(y[:cut])) < 2:
            continue
        Xtr, ytr = X.iloc[:cut], y[:cut]
        Xva, yva = X.iloc[cut:end], y[cut:end]

        model = XGBClassifier(
            **params, n_estimators=int(tuning.get("n_estimators_max", 2000)),
            early_stopping_rounds=int(tuning.get("early_stopping_rounds", 40)),
            random_state=seed,
        )
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)

        for name, value in zip(cols, model.feature_importances_):
            gain_rows.append({"feature": name, "fold": fold, "gain": float(value)})

        # Permutation on a capped slice: 20,000 held-out rows is plenty to
        # separate a real feature from noise, and the full slice would make
        # this the slowest part of the run for no extra resolution.
        idx = np.arange(len(Xva))
        if len(idx) > permutation_cap:
            idx = rng.choice(idx, permutation_cap, replace=False)
        Xp, yp = Xva.iloc[idx].reset_index(drop=True), yva[idx]
        base = _log_loss(yp, model.predict_proba(Xp)[:, 1])
        for name in cols:
            deltas = []
            for _ in range(permutation_repeats):
                shuffled = Xp.copy()
                shuffled[name] = rng.permutation(shuffled[name].to_numpy())
                deltas.append(_log_loss(yp, model.predict_proba(shuffled)[:, 1]) - base)
            perm_rows.append({
                "feature": name, "fold": fold, "perm_delta": float(np.mean(deltas)),
            })

    if not perm_rows:
        raise ValueError(f"DATA_NOT_AVAILABLE: no fold produced a fit for {market}")

    gain = pd.DataFrame(gain_rows)
    perm = pd.DataFrame(perm_rows)
    ranks = perm.copy()
    ranks["rank"] = ranks.groupby("fold")["perm_delta"].rank(ascending=False)

    out = (
        perm.groupby("feature")["perm_delta"].agg(["mean", "std"])
        .rename(columns={"mean": "perm_mean", "std": "perm_sd"})
        .join(gain.groupby("feature")["gain"].mean().rename("gain_mean"))
        .join(ranks[ranks["rank"] <= 10].groupby("feature").size().rename("folds_top10"))
    )
    out["folds_top10"] = out["folds_top10"].fillna(0).astype(int)
    out["n_folds"] = perm["fold"].nunique()
    out["coverage"] = [float(X[f].notna().mean()) for f in out.index]
    out["univariate_auc"] = [_auc(y, X[f].to_numpy(dtype=float)) for f in out.index]
    # A mean smaller than its own spread has not been shown to matter.
    out["above_noise"] = out["perm_mean"] > out["perm_sd"].fillna(np.inf)
    return out.sort_values("perm_mean", ascending=False).reset_index()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--panel", default="data/external/training_pack/panel.parquet")
    ap.add_argument("--markets", default="PTS,REB,AST")
    ap.add_argument("--train-end", default="2025-10-01")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--out-dir", default="outputs/feature_selection")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    logger.setLevel(logging.INFO)
    warnings.filterwarnings("ignore")

    panel = pd.read_parquet(args.panel)
    panel["GAME_DATE"] = pd.to_datetime(panel["GAME_DATE"])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Selection window: rows on or before {args.train_end} "
          f"({int((panel['GAME_DATE'] <= pd.Timestamp(args.train_end)).sum()):,} of "
          f"{len(panel):,}). The evaluation window is not read.\n")

    for market in [m.strip().upper() for m in args.markets.split(",") if m.strip()]:
        table = rank_features(panel, market, train_end=args.train_end, n_folds=args.folds)
        path = out_dir / f"feature_selection_{market}.csv"
        table.to_csv(path, index=False)

        n = int(table["n_folds"].iloc[0])
        survivors = int(table["above_noise"].sum())
        print(f"=== {market} — {len(table)} features, {n} folds, "
              f"{survivors} clear their own noise ===")
        head = table.head(args.top)
        print(f"  {'feature':<34}{'perm':>9}{'sd':>9}{'gain':>8}"
              f"{'top10':>7}{'AUC':>7}{'cov':>7}")
        for _, r in head.iterrows():
            flag = " " if r["above_noise"] else "~"
            print(f"{flag} {r['feature']:<34}{r['perm_mean']:9.5f}"
                  f"{(r['perm_sd'] if pd.notna(r['perm_sd']) else float('nan')):9.5f}"
                  f"{r['gain_mean']:8.4f}{int(r['folds_top10']):5d}/{n}"
                  f"{r['univariate_auc']:7.3f}{r['coverage']:7.1%}")
        print(f"  ~ marks a feature whose mean is inside its own fold-to-fold "
              f"spread.\n  wrote {path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
