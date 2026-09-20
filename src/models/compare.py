"""Walk-forward model comparison and export orchestration."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, mean_squared_error

from src.models.distribution_adapter import DistributionPropModel
from src.models.ensemble import EnsemblePropModel
from src.models.labels import attach_research_over_labels, default_feature_cols
from src.models.prob_calibration import reliability_table
from src.models.walk_forward import fixed_cutoff_split, sort_by_game_date
from src.utils.timezones import format_pacific_iso, to_pacific

logger = logging.getLogger(__name__)


def load_comparison_config(path: Path | str = "config/model_comparison.yaml") -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def _soft_fill(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = 0.0 if c != "RESEARCH_LINE" else np.nan
        elif c in {"BBS_OUT_FLAG", "OPP_DEF_RATING_PROXY"} or c.startswith("OPP_"):
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    return out


def prepare_market_panel(panel: pd.DataFrame, market: str) -> pd.DataFrame:
    labeled = attach_research_over_labels(panel, stat=market)  # type: ignore[arg-type]
    return sort_by_game_date(labeled)


def score_binary(y_true: np.ndarray, p: np.ndarray) -> dict[str, float | None]:
    mask = np.isfinite(p) & np.isfinite(y_true)
    if mask.sum() < 5:
        return {"n": float(mask.sum()), "brier": None, "log_loss": None, "mae": None, "rmse": None, "mean_bias": None}
    yt, pp = y_true[mask].astype(int), np.clip(p[mask], 1e-6, 1 - 1e-6)
    return {
        "n": float(len(yt)),
        "brier": float(brier_score_loss(yt, pp)),
        "log_loss": float(log_loss(yt, pp, labels=[0, 1])),
        "mae": None,
        "rmse": None,
        "mean_bias": None,
    }


def score_mean(y_true: np.ndarray, y_hat: np.ndarray) -> dict[str, float | None]:
    mask = np.isfinite(y_true) & np.isfinite(y_hat)
    if mask.sum() < 5:
        return {"mae": None, "rmse": None, "mean_bias": None, "n": float(mask.sum())}
    yt, yh = y_true[mask], y_hat[mask]
    return {
        "n": float(len(yt)),
        "mae": float(mean_absolute_error(yt, yh)),
        "rmse": float(mean_squared_error(yt, yh) ** 0.5),
        "mean_bias": float((yh - yt).mean()),
    }


def build_components(
    market: str,
    feature_cols: list[str],
    cfg: dict[str, Any],
    *,
    include_catboost: bool = True,
    include_xgboost: bool = True,
    xgb_feature_cols: list[str] | None = None,
) -> dict[str, Any]:
    """
    Build model components.

    XGBoost keeps the original numeric ``default_feature_cols`` list (unchanged
    behavior). CatBoost may add categorical columns separately.
    """
    components: dict[str, Any] = {
        "distribution": DistributionPropModel(target_market=market),
    }
    seed = int(cfg.get("random_seed", 42))
    schema = str(cfg.get("feature_schema_version", "fs_v1_shift1_l2"))
    xgb_cols = list(xgb_feature_cols or feature_cols)
    if include_xgboost:
        try:
            from src.models.xgb_adapter import XGBoostAdapter

            components["xgboost"] = XGBoostAdapter(
                xgb_cols,
                target_market=market,
                feature_schema_version=schema,
                random_state=seed,
            )
        except ImportError as exc:
            logger.warning("XGBoost unavailable: %s", exc)
    if include_catboost:
        try:
            from src.models.catboost_pipeline import CatBoostPropPipeline

            cb_cfg = cfg.get("catboost") or {}
            components["catboost"] = CatBoostPropPipeline(
                feature_cols,
                target_market=market,
                feature_schema_version=schema,
                categorical_features=list(cb_cfg.get("categorical_features") or []),
                hyperparameters={k: v for k, v in cb_cfg.items() if k != "categorical_features"},
                random_seed=seed,
            )
        except ImportError as exc:
            logger.warning("CatBoost unavailable: %s", exc)
    return components


def compare_models_on_panel(
    panel: pd.DataFrame,
    *,
    markets: list[str],
    train_end: str,
    validation_end: str,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Fit on rows with GAME_DATE <= train_end; score (train_end, validation_end].

    Returns summary rows + detailed prediction rows (no fabricated odds).
    """
    cfg = cfg or load_comparison_config()
    weights = cfg.get("ensemble_weights") or {}
    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    calib_rows: list[dict[str, Any]] = []
    winners: dict[str, dict[str, Any]] = {}

    for market in markets:
        work = prepare_market_panel(panel, market)
        xgb_cols = list(default_feature_cols(market))  # type: ignore[arg-type]
        feature_cols = list(xgb_cols)
        # Add categoricals for CatBoost only (XGBoost stays numeric-only)
        for c in ("TEAM_ABBREVIATION", "OPPONENT_ABBREVIATION", "SEASON"):
            if c in work.columns and c not in feature_cols:
                feature_cols = list(feature_cols) + [c]
        work = _soft_fill(work, feature_cols)
        split = fixed_cutoff_split(work, train_end=train_end, validation_end=validation_end)
        train = work.loc[split.train_idx]
        val = work.loc[split.validation_idx]

        components = build_components(market, feature_cols, cfg, xgb_feature_cols=xgb_cols)
        fitted: dict[str, Any] = {}
        for name, model in components.items():
            try:
                model.fit(train, val)
                fitted[name] = model
            except Exception as exc:  # noqa: BLE001
                logger.warning("fit failed market=%s model=%s: %s", market, name, exc)

        if len(fitted) >= 2:
            ens = EnsemblePropModel(
                {k: fitted[k] for k in fitted},
                weights={k: weights.get(k, 0.0) for k in fitted},
                target_market=market,
            )
            fitted["ensemble"] = ens

        y_true = val["over_hit"].astype(float).to_numpy()
        actual = pd.to_numeric(val[market], errors="coerce").to_numpy() if market in val.columns else np.full(len(val), np.nan)
        market_scores: dict[str, dict[str, Any]] = {}

        for name, model in fitted.items():
            try:
                preds = model.predict_rows(val, line_col="RESEARCH_LINE")
            except Exception as exc:  # noqa: BLE001
                logger.warning("predict failed market=%s model=%s: %s", market, name, exc)
                continue
            p_over = np.array([p.probability_over if p.probability_over is not None else np.nan for p in preds], dtype=float)
            means = np.array([p.prediction_mean if p.prediction_mean is not None else np.nan for p in preds], dtype=float)
            bin_s = score_binary(y_true, p_over)
            mean_s = score_mean(actual, means)
            row = {
                "target_market": market,
                "model_name": name,
                "evaluation_start_date": str(split.validation_start.date()),
                "evaluation_end_date": str(split.validation_end.date()),
                "n_predictions": int(bin_s["n"] or 0),
                "mae": mean_s.get("mae"),
                "rmse": mean_s.get("rmse"),
                "mean_bias": mean_s.get("mean_bias"),
                "brier_score": bin_s.get("brier"),
                "log_loss": bin_s.get("log_loss"),
                "calibration_error": None,
                "interval_coverage": None,
                "notes": "RESEARCH_ONLY; RESEARCH_LINE={stat}_L10; not sportsbook",
            }
            # Simple ECE proxy from reliability table
            mask = np.isfinite(p_over) & np.isfinite(y_true)
            if mask.sum() >= 20:
                table = reliability_table(y_true[mask], p_over[mask])
                if table:
                    gaps = [abs(t["calibration_gap"]) * t["n_predictions"] for t in table]
                    ntot = sum(t["n_predictions"] for t in table)
                    row["calibration_error"] = round(sum(gaps) / max(ntot, 1), 4)
                    for t in table:
                        calib_rows.append(
                            {
                                "target_market": market,
                                "model_name": name,
                                **t,
                                "evaluation_start_date": row["evaluation_start_date"],
                                "evaluation_end_date": row["evaluation_end_date"],
                            }
                        )
            summary_rows.append(row)
            market_scores[name] = row

            for i, pred in enumerate(preds):
                r = val.iloc[i]
                # Calendar game_date + cutoffs presented in Pacific (project display TZ).
                gdate = pd.to_datetime(r.get("GAME_DATE"), errors="coerce")
                game_date_pt = None
                if pd.notna(gdate):
                    # Date-only rows: treat as Pacific calendar date (no feed TZ shown).
                    game_date_pt = str(gdate.date())
                detail_rows.append(
                    {
                        "event_id": pred.event_id,
                        "game_date": game_date_pt,
                        "game_start_pt": None,
                        "player_id": pred.player_id,
                        "player_name": pred.player_name,
                        "player_team": r.get("TEAM_ABBREVIATION"),
                        "opponent": r.get("OPPONENT_ABBREVIATION"),
                        "home_away": "home" if bool(r.get("IS_HOME")) else "away",
                        "target_market": market,
                        "prop_line": pred.prop_line,
                        "line_type": "research_l10",
                        "prediction_mean": pred.prediction_mean,
                        "prediction_std_or_dispersion": pred.prediction_std_or_dispersion,
                        "probability_over_raw": pred.probability_over,
                        "probability_under_raw": pred.probability_under,
                        "probability_push_raw": pred.probability_push,
                        "probability_over_calibrated": pred.probability_over_calibrated,
                        "probability_under_calibrated": pred.probability_under_calibrated,
                        "model_name": pred.model_name,
                        "model_version": pred.model_version,
                        "ensemble_weight": (pred.extras or {}).get("component_weights"),
                        "feature_schema_version": pred.feature_schema_version,
                        "data_cutoff_pt": format_pacific_iso(to_pacific(pd.Timestamp(split.train_end).to_pydatetime())),
                        "prediction_timestamp_pt": format_pacific_iso(pred.prediction_timestamp_utc),
                        "actual_stat_value": float(actual[i]) if np.isfinite(actual[i]) else None,
                        "settlement": None,
                        "warnings": "|".join(pred.warnings) if pred.warnings else None,
                    }
                )

            if name == "catboost" and hasattr(fitted[name], "feature_importance_frame"):
                importance_rows.extend(fitted[name].feature_importance_frame().to_dict(orient="records"))

        # Winner by Brier then log_loss then calibration_error
        ranked = [
            (n, s)
            for n, s in market_scores.items()
            if s.get("brier_score") is not None and n != "ensemble"
        ]
        ranked.sort(key=lambda t: (t[1]["brier_score"], t[1].get("log_loss") or 9, t[1].get("calibration_error") or 9))
        if ranked:
            winners[market] = {
                "winner": ranked[0][0],
                "brier_score": ranked[0][1]["brier_score"],
                "log_loss": ranked[0][1].get("log_loss"),
                "calibration_error": ranked[0][1].get("calibration_error"),
                "note": "Lowest Brier on validation window; not a profitability claim",
            }

    return {
        "summary": summary_rows,
        "predictions": detail_rows,
        "feature_importance": importance_rows,
        "calibration": calib_rows,
        "winners": winners,
    }
