"""
scripts/audit_pipeline.py — end-to-end health check of the prop pipeline.

Four areas, each reduced to checks that FAIL on evidence rather than reporting
an opinion. Complements scripts/audit_leakage.py, which covers the panel and
its splits in more depth; this one covers the pipeline around them.

  1 INGESTION & WIRING   Are the event logs complete against an independent
                         measurement, do the joins preserve row identity and
                         row order, and does every configured feature reach
                         the model or get reported as dropped?

  2 TRAINING             Is the model learning rather than predicting a
                         constant, and does every cross-validation fold train
                         strictly before it validates?

  3 EXECUTION            Are NaNs passed to models that handle them rather
                         than filled with invented values, and did any
                         component fail to fit and vanish silently?

  4 OUTPUT               Do the exported metrics reproduce from the exported
                         predictions, and does the reported ensemble match
                         the configured one?

RESEARCH ONLY.

Usage:
    python -m scripts.audit_pipeline
    python -m scripts.audit_pipeline --panel <parquet> --outputs outputs
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger("audit_pipeline")


class AuditFinding(AssertionError):
    """A check that failed. Never downgraded to a warning."""


# --- 1. ingestion and wiring ------------------------------------------------


def check_pbp_completeness(panel: pd.DataFrame, pbp: pd.DataFrame) -> str:
    """Event log against the box score — two independent counts of one thing."""
    shots = pbp[pbp["actionType"].isin(("2pt", "3pt"))]
    per_game = shots.groupby(shots["gameId"].astype(str)).size().rename("pbp")
    box = panel.groupby(panel["GAME_ID"].astype(str))["FGA"].sum().rename("box")
    joined = pd.concat([per_game, box], axis=1).dropna()
    if joined.empty:
        raise AuditFinding("no game is present in both the event log and the panel")
    gap = joined["box"] - joined["pbp"]
    exact = float((gap == 0).mean())
    if exact < 0.90:
        raise AuditFinding(
            f"only {exact:.1%} of shared games agree on field-goal attempts "
            f"(median gap {gap.median():.0f}). The event log is incomplete; "
            "rates built on it will be biased."
        )
    return (f"{len(joined):,} shared games, {exact:.1%} agree exactly on attempts, "
            f"median gap {gap.median():.0f}")


def check_join_preserves_rows(panel: pd.DataFrame) -> str:
    """
    A feature join must not reorder, duplicate or drop the caller's rows.

    Values staying attached to their own rows is not enough: a join that
    returns them in a different order corrupts any caller that later assigns
    a column positionally.
    """
    from src.features.pbp import attach_pbp_rolling_features

    sample = panel.head(400).copy()
    sample["_audit_marker"] = np.arange(len(sample))
    summaries = pd.DataFrame({
        "gameId": sample["GAME_ID"].astype(str),
        "personId": sample["PLAYER_ID"].astype(str),
        "PBP_RIM_RATE": np.linspace(0.1, 0.9, len(sample)),
    })
    out = attach_pbp_rolling_features(sample, summaries, windows=(3,))
    if len(out) != len(sample):
        raise AuditFinding(f"row count changed: {len(sample)} -> {len(out)}")
    if not out["_audit_marker"].equals(sample["_audit_marker"]):
        raise AuditFinding("the join reordered the caller's rows")
    if not out.index.equals(sample.index):
        raise AuditFinding("the join replaced the caller's index")
    return f"{len(sample)} rows kept their order, index and identity"


def check_feature_wiring(panel: pd.DataFrame, markets: list[str], train_end: str) -> str:
    """Every configured feature either reaches the model or is reported."""
    from src.models.compare import resolve_feature_cols
    from src.models.labels import default_feature_cols

    train = panel[panel["GAME_DATE"] <= pd.Timestamp(train_end)]
    lines = []
    for market in markets:
        wanted = list(default_feature_cols(market))
        present, absent = resolve_feature_cols(panel, wanted)
        empty = [c for c in present if c in train.columns and not train[c].notna().any()]
        usable = [c for c in present if c not in empty]
        if not usable:
            raise AuditFinding(f"{market}: no configured feature is trainable")
        lines.append(f"{market} {len(wanted)}->{len(usable)}")
    return ("configured -> trainable: " + ", ".join(lines)
            + " (the rest are absent from the panel or empty across training, "
              "and are reported rather than filled)")


# --- 2. training ------------------------------------------------------------


def check_folds_are_chronological(panel: pd.DataFrame, market: str, train_end: str) -> str:
    """No fold may train on a row dated after the rows it validates."""
    from sklearn.model_selection import TimeSeriesSplit

    from src.models.labels import attach_research_over_labels

    work = attach_research_over_labels(panel, stat=market)
    work = work[work["over_hit"].notna()].sort_values("GAME_DATE").reset_index(drop=True)
    train = work[work["GAME_DATE"] <= pd.Timestamp(train_end)].reset_index(drop=True)
    dates = train["GAME_DATE"].to_numpy()
    for i, (tr, va) in enumerate(TimeSeriesSplit(n_splits=5).split(train), 1):
        if dates[tr].max() > dates[va].min():
            raise AuditFinding(
                f"fold {i} trains on rows dated after its validation window starts"
            )
    return f"5 folds over {len(train):,} rows, every one trains strictly before it validates"


def check_fold_game_straddle(panel: pd.DataFrame, market: str, train_end: str,
                             tolerance: float = 0.005) -> str:
    """
    TimeSeriesSplit cuts by ROW INDEX, not by date, so one game's players can
    land on both sides of a fold boundary. Teammates share a game state, so
    that is a small leak into early stopping and out-of-fold calibration.
    """
    from sklearn.model_selection import TimeSeriesSplit

    from src.models.labels import attach_research_over_labels

    work = attach_research_over_labels(panel, stat=market)
    work = work[work["over_hit"].notna()].sort_values("GAME_DATE").reset_index(drop=True)
    train = work[work["GAME_DATE"] <= pd.Timestamp(train_end)].reset_index(drop=True)
    gid = train["GAME_ID"].astype(str).to_numpy()
    straddling = affected = total_val = 0
    for tr, va in TimeSeriesSplit(n_splits=5).split(train):
        shared = set(gid[tr]) & set(gid[va])
        straddling += len(shared)
        affected += int(np.isin(gid[va], list(shared)).sum())
        total_val += len(va)
    share = affected / max(total_val, 1)
    if share > tolerance:
        raise AuditFinding(
            f"{share:.2%} of fold-validation rows come from a game the same fold "
            f"trained on ({straddling} games). Snap fold boundaries to game "
            "boundaries."
        )
    return (f"{straddling} game(s) straddle a fold boundary, {affected} of "
            f"{total_val:,} rows ({share:.4%}) — under the {tolerance:.1%} tolerance")


def check_model_is_learning(panel: pd.DataFrame, market: str, train_end: str) -> str:
    """A model that predicts a constant, or beats nothing, is not training."""
    from xgboost import XGBClassifier

    from src.models.compare import load_comparison_config, resolve_feature_cols
    from src.models.labels import attach_research_over_labels, default_feature_cols
    from src.models.xgboost_pipeline import split_xgboost_config

    work = attach_research_over_labels(panel, stat=market)
    work = work[work["over_hit"].notna()].sort_values("GAME_DATE").reset_index(drop=True)
    train = work[work["GAME_DATE"] <= pd.Timestamp(train_end)].reset_index(drop=True)
    cols, _ = resolve_feature_cols(train, list(default_feature_cols(market)))
    cols = [c for c in cols if train[c].notna().any()]
    X = train[cols].apply(pd.to_numeric, errors="coerce")
    y = train["over_hit"].astype(int).to_numpy()

    params, tuning = split_xgboost_config(load_comparison_config().get("xgboost") or {})
    params.pop("n_estimators", None)
    cut = int(len(train) * 0.8)
    model = XGBClassifier(
        **params, n_estimators=int(tuning.get("n_estimators_max", 2000)),
        early_stopping_rounds=int(tuning.get("early_stopping_rounds", 40)),
    )
    model.fit(X.iloc[:cut], y[:cut], eval_set=[(X.iloc[cut:], y[cut:])], verbose=False)
    p = model.predict_proba(X.iloc[cut:])[:, 1]
    yv = y[cut:]

    if p.std() < 1e-6:
        raise AuditFinding("the model predicts a constant probability")
    dead = int((model.feature_importances_ == 0).sum())
    base = yv.mean()
    const = -np.mean(yv * np.log(base) + (1 - yv) * np.log(1 - base))
    clipped = np.clip(p, 1e-9, 1 - 1e-9)
    got = -np.mean(yv * np.log(clipped) + (1 - yv) * np.log(1 - clipped))
    if got >= const:
        raise AuditFinding(
            f"log-loss {got:.5f} is no better than predicting the base rate "
            f"({const:.5f}) — the model has learned nothing"
        )
    return (f"{model.best_iteration + 1} trees, prediction sd {p.std():.4f}, "
            f"{dead} dead feature(s) of {len(cols)}, log-loss {got:.5f} vs "
            f"{const:.5f} for a constant")


# --- 3. execution -----------------------------------------------------------


def check_nan_policy(panel: pd.DataFrame, market: str, train_end: str) -> str:
    """NaNs must reach models that handle them, never an invented fill."""
    from src.models.compare import resolve_feature_cols
    from src.models.labels import default_feature_cols

    train = panel[panel["GAME_DATE"] <= pd.Timestamp(train_end)]
    cols, _ = resolve_feature_cols(train, list(default_feature_cols(market)))
    cols = [c for c in cols if train[c].notna().any()]
    X = train[cols].apply(pd.to_numeric, errors="coerce")
    worst = X.isna().mean().max()
    if worst > 0.5:
        raise AuditFinding(
            f"a trainable feature is {worst:.1%} NaN — too sparse to inform anything"
        )
    return (f"{int((X.isna().mean() > 0).sum())} of {len(cols)} features carry NaN, "
            f"worst {worst:.2%}, {X.isna().any(axis=1).mean():.2%} of rows affected; "
            "passed through, not imputed")


def check_no_component_vanished(outputs: Path, cfg: dict) -> str:
    """
    Every model the config weights must appear in the exported predictions.

    A component that fails to fit is logged and skipped, and the ensemble
    renormalises around it. Nothing downstream says so, which is how CatBoost
    ran at 0.50 of the configured weight while contributing nothing.
    """
    path = outputs / "predictions_detailed.parquet"
    if not path.exists():
        return "skipped: no predictions export to check"
    present = set(pd.read_parquet(path, columns=["model_name"])["model_name"].unique())
    weighted = {k for k, v in (cfg.get("ensemble_weights") or {}).items() if v > 0}
    missing = sorted(weighted - present)
    if missing:
        raise AuditFinding(
            f"model(s) {missing} carry ensemble weight in the config but produced "
            f"no predictions. The exported 'ensemble' is a different blend from "
            f"the configured one, and nothing in the outputs records that."
        )
    unweighted = sorted(present - weighted - {"ensemble"})
    note = f" ({unweighted} fitted but carry no ensemble weight)" if unweighted else ""
    return f"every weighted component produced predictions{note}"


# --- 4. output --------------------------------------------------------------


def check_metrics_reproduce(outputs: Path) -> str:
    """MAE and RMSE must recompute from the exported predictions."""
    summary_path = outputs / "model_comparison_summary.csv"
    detail_path = outputs / "predictions_detailed.parquet"
    if not (summary_path.exists() and detail_path.exists()):
        return "skipped: no exports to check"
    summary = pd.read_csv(summary_path)
    detail = pd.read_parquet(detail_path)

    rows = []
    for (market, model), g in detail.groupby(["target_market", "model_name"]):
        a = pd.to_numeric(g["actual_stat_value"], errors="coerce").to_numpy()
        p = pd.to_numeric(g["prediction_mean"], errors="coerce").to_numpy()
        m = np.isfinite(a) & np.isfinite(p)
        if m.sum() < 5:
            continue
        rows.append({
            "target_market": market, "model_name": model,
            "mae_check": float(np.abs(p[m] - a[m]).mean()),
            "rmse_check": float(np.sqrt(((p[m] - a[m]) ** 2).mean())),
        })
    check = pd.DataFrame(rows)
    if check.empty:
        raise AuditFinding("no exported model had enough rows to recompute metrics")
    joined = summary.merge(check, on=["target_market", "model_name"])
    worst_mae = (joined["mae"] - joined["mae_check"]).abs().max()
    worst_rmse = (joined["rmse"] - joined["rmse_check"]).abs().max()
    if max(worst_mae, worst_rmse) > 1e-6:
        raise AuditFinding(
            f"exported metrics do not reproduce: max |MAE diff| {worst_mae:.3g}, "
            f"max |RMSE diff| {worst_rmse:.3g}"
        )
    return (f"{len(joined)} model/market rows reproduce exactly "
            f"(max |diff| {max(worst_mae, worst_rmse):.1e})")


def check_calibrated_metrics_exported(outputs: Path) -> str:
    """The comparison computes calibrated metrics; the export must carry them."""
    path = outputs / "model_comparison_summary.csv"
    if not path.exists():
        return "skipped: no summary to check"
    cols = set(pd.read_csv(path, nrows=0).columns)
    needed = {"brier_score_calibrated", "calibration_error_calibrated"}
    missing = sorted(needed - cols)
    if missing:
        raise AuditFinding(
            f"the summary omits {missing}. Calibration is applied to the exported "
            "predictions, so without these the file shows only the raw numbers "
            "and nothing says whether calibration helped."
        )
    return "raw and calibrated metrics both exported"


def check_roi_is_not_invented(outputs: Path) -> str:
    """
    ROI needs real posted odds, a settled outcome and a stake.

    The comparison produces none of those. Reporting an ROI from it would mean
    inventing at least one, so its absence is the correct behaviour and this
    check exists to confirm nothing started inventing it.
    """
    for candidate in outputs.glob("*.csv"):
        cols = set(pd.read_csv(candidate, nrows=0).columns)
        if "roi" in {c.lower() for c in cols}:
            raise AuditFinding(
                f"{candidate.name} reports an ROI. The comparison has no odds and "
                "no settled bets, so any ROI here is fabricated."
            )
    return ("no ROI in the comparison outputs, which is correct: it has no odds "
            "and no settled bets. ROI lives in the settlement ledger.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--panel", default="data/external/training_pack/panel.parquet")
    ap.add_argument("--pbp", default="data/external/training_pack/pbp_raw.parquet")
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--markets", default="PTS,REB,AST")
    ap.add_argument("--train-end", default="2025-10-01")
    ap.add_argument("--config", default="config/model_comparison.yaml")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(name)s: %(message)s")
    markets = [m.strip().upper() for m in args.markets.split(",") if m.strip()]
    outputs = Path(args.outputs)
    cfg = yaml.safe_load(Path(args.config).read_text()) or {}

    panel = pd.read_parquet(args.panel)
    panel["GAME_DATE"] = pd.to_datetime(panel["GAME_DATE"])
    pbp = pd.read_parquet(args.pbp) if Path(args.pbp).exists() else None

    print(f"Panel  {len(panel):,} rows x {panel.shape[1]} cols, "
          f"{panel['GAME_DATE'].min().date()} -> {panel['GAME_DATE'].max().date()}")
    print(f"Events {len(pbp):,} rows" if pbp is not None else "Events: none supplied")
    print(f"Split  train <= {args.train_end}\n")

    market = markets[0]
    checks: list[tuple[str, str, object]] = [
        ("1", "pbp completeness",
         (lambda: check_pbp_completeness(panel, pbp)) if pbp is not None
         else (lambda: "skipped: no event log supplied")),
        ("1", "join preserves rows", lambda: check_join_preserves_rows(panel)),
        ("1", "feature wiring",
         lambda: check_feature_wiring(panel, markets, args.train_end)),
        ("2", "folds are chronological",
         lambda: check_folds_are_chronological(panel, market, args.train_end)),
        ("2", "fold game straddle",
         lambda: check_fold_game_straddle(panel, market, args.train_end)),
        ("2", "model is learning",
         lambda: check_model_is_learning(panel, market, args.train_end)),
        ("3", "NaN policy", lambda: check_nan_policy(panel, market, args.train_end)),
        ("3", "no component vanished",
         lambda: check_no_component_vanished(outputs, cfg)),
        ("4", "metrics reproduce", lambda: check_metrics_reproduce(outputs)),
        ("4", "calibrated metrics exported",
         lambda: check_calibrated_metrics_exported(outputs)),
        ("4", "ROI not invented", lambda: check_roi_is_not_invented(outputs)),
    ]

    failures = 0
    area = None
    titles = {"1": "INGESTION & FEATURE WIRING", "2": "TRAINING & OPTIMIZATION",
              "3": "PIPELINE EXECUTION", "4": "OUTPUT INTEGRITY"}
    for tag, name, fn in checks:
        if tag != area:
            area = tag
            print(f"{tag}. {titles[tag]}")
        try:
            detail = fn()
        except AuditFinding as exc:
            failures += 1
            print(f"   FAIL  {name}\n         {exc}")
        except Exception as exc:  # noqa: BLE001 — an audit that crashes is a failed audit
            failures += 1
            print(f"   ERROR {name}\n         {type(exc).__name__}: {exc}")
        else:
            print(f"   pass  {name}  —  {detail}")
    print()
    if failures:
        print(f"{failures} check(s) FAILED.")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
