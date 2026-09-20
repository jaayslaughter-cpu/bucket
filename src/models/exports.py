"""Write model-comparison CSVs under outputs/ (never secrets).

User-facing timestamps are America/Los_Angeles. Internal storage remains UTC.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.timezones import DISPLAY_TZ_NAME, format_pacific_iso, now_pacific

logger = logging.getLogger(__name__)

SUMMARY_COLS = [
    "target_market", "model_name", "evaluation_start_date", "evaluation_end_date",
    "n_predictions", "mae", "rmse", "mean_bias", "brier_score", "log_loss",
    "calibration_error", "interval_coverage", "notes",
]

DETAIL_COLS = [
    "event_id", "game_date", "game_start_pt", "player_id", "player_name",
    "player_team", "opponent", "home_away", "target_market", "prop_line",
    "line_type", "prediction_mean", "prediction_std_or_dispersion",
    "probability_over_raw", "probability_under_raw", "probability_push_raw",
    "probability_over_calibrated", "probability_under_calibrated",
    "model_name", "model_version", "ensemble_weight", "feature_schema_version",
    "data_cutoff_pt", "prediction_timestamp_pt",
    "actual_stat_value", "settlement", "warnings",
]

IMPORTANCE_COLS = [
    "target_market", "model_version", "feature_name", "importance", "rank", "trained_through_date",
]

CALIBRATION_COLS = [
    "target_market", "model_name", "probability_bin", "n_predictions",
    "mean_predicted_probability", "observed_frequency", "calibration_gap",
    "evaluation_start_date", "evaluation_end_date",
]

QUALITY_COLS = [
    "report_timestamp_pt", "dataset_name", "total_rows", "valid_rows",
    "rejected_rows", "duplicate_rows", "missing_player_id_rows",
    "missing_event_id_rows", "missing_target_rows", "missing_pregame_timestamp_rows",
    "possible_leakage_rows", "notes",
]

LIVE_TEMPLATE_COLS = [
    "game_date", "event_id", "game_start_pt", "player_id", "player_name",
    "player_team", "opponent", "target_market", "sportsbook_or_dfs_source",
    "side", "prop_line", "american_odds", "captured_at_pt", "model_projection",
    "model_probability_over", "model_probability_under",
    "fair_market_probability_if_available", "edge_probability_if_available",
    "warnings", "research_status",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_df(df: pd.DataFrame, path: Path, columns: list[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(df)
    for c in columns:
        if c not in out.columns:
            out[c] = None
    out = out[columns]
    out.to_csv(path, index=False)
    try:
        out.to_parquet(path.with_suffix(".parquet"), index=False)
    except Exception:  # noqa: BLE001
        logger.debug("parquet export skipped for %s", path)
    return len(out)


def write_comparison_exports(
    result: dict[str, Any],
    *,
    output_dir: Path | str = "outputs",
    quality_row: dict[str, Any] | None = None,
    demo: bool = False,
) -> dict[str, Any]:
    root = Path(output_dir)
    if demo:
        root = root / "demo"
        logger.warning("DEMO MODE — writing to %s (not real analysis)", root)
    root.mkdir(parents=True, exist_ok=True)

    files: dict[str, int] = {}
    files["model_comparison_summary.csv"] = _write_df(
        pd.DataFrame(result.get("summary") or []), root / "model_comparison_summary.csv", SUMMARY_COLS
    )
    files["predictions_detailed.csv"] = _write_df(
        pd.DataFrame(result.get("predictions") or []), root / "predictions_detailed.csv", DETAIL_COLS
    )
    files["feature_importance_catboost.csv"] = _write_df(
        pd.DataFrame(result.get("feature_importance") or []),
        root / "feature_importance_catboost.csv",
        IMPORTANCE_COLS,
    )
    files["calibration_report.csv"] = _write_df(
        pd.DataFrame(result.get("calibration") or []), root / "calibration_report.csv", CALIBRATION_COLS
    )
    qdf = pd.DataFrame([quality_row] if quality_row else [])
    files["data_quality_report.csv"] = _write_df(qdf, root / "data_quality_report.csv", QUALITY_COLS)

    live = pd.DataFrame(columns=LIVE_TEMPLATE_COLS)
    files["live_prediction_template.csv"] = _write_df(live, root / "live_prediction_template.csv", LIVE_TEMPLATE_COLS)

    winners_path = root / "winners_by_market.json"
    winners_path.write_text(json.dumps(result.get("winners") or {}, indent=2), encoding="utf-8")

    manifest = {
        "generated_at_pt": format_pacific_iso(now_pacific()),
        "timezone_display": DISPLAY_TZ_NAME,
        "demo_mode": demo,
        "research_only": True,
        "files": {},
        "winners_by_market": result.get("winners") or {},
    }
    for name, nrows in files.items():
        p = root / name
        manifest["files"][name] = {
            "rows": nrows,
            "sha256": _sha256(p) if p.exists() else None,
        }
    (root / "download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Wrote comparison exports to %s (times in %s)", root, DISPLAY_TZ_NAME)
    return manifest
