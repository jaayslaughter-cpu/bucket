#!/usr/bin/env python3
"""PropIQ model comparison CLI (RESEARCH_ONLY).

Examples:
  python -m scripts.nba_model_cli audit-data
  python -m scripts.nba_model_cli compare-models --markets PTS,REB,AST --demo
  python -m scripts.nba_model_cli generate-exports --demo
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, no_args_is_help=True, help="NBA model comparison (research only)")
logger = logging.getLogger("nba_model")


def _setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
    )


def _load_real_or_demo(demo: bool):
    from src.features.builder import build_feature_matrix
    from src.models.data_audit import make_demo_panel

    if demo:
        logger.warning("DEMO MODE — synthetic panel; do not treat metrics as real")
        raw = make_demo_panel()
        return build_feature_matrix(raw), True
    try:
        from src.ingestion.boxscores import load_player_game_logs

        raw = load_player_game_logs()
        if raw is None or raw.empty:
            raise RuntimeError("empty player logs")
        return build_feature_matrix(raw), False
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Real panel unavailable (%s). Re-run with --demo for wiring tests, "
            "or fix boxscore archives. Will not silently fabricate real analysis.",
            exc,
        )
        raise SystemExit(2) from exc


@app.command("ingest-logs")
def ingest_logs(
    seasons: str = typer.Option("2025-26", "--seasons", help="Comma-separated, e.g. 2024-25,2025-26"),
    season_type: str = typer.Option("Regular Season", "--season-type"),
    refresh: bool = typer.Option(False, "--refresh", help="Ignore the cache and re-fetch"),
    verbose: bool = False,
) -> None:
    """Pull player game logs from the NBA stats API and cache them locally.

    Needs network access to stats.nba.com. Run this before any training —
    the BigDataBall workbook is team-level and cannot supply player lines.
    """
    _setup_logging(verbose)
    from src.ingestion.boxscores import BoxScoreFetchError, BoxScoreLoadConfig, load_player_game_logs

    config = BoxScoreLoadConfig(
        seasons=tuple(s.strip() for s in seasons.split(",") if s.strip()),
        season_type=season_type,
        use_cache=not refresh,
    )
    try:
        panel = load_player_game_logs(config)
    except BoxScoreFetchError as exc:
        typer.echo(f"Ingest failed: {exc}", err=True)
        raise SystemExit(2) from exc

    typer.echo(
        json.dumps(
            {
                "rows": int(len(panel)),
                "players": int(panel["PLAYER_ID"].nunique()),
                "games": int(panel["GAME_ID"].nunique()),
                "first_game_date": str(panel["GAME_DATE"].min().date()),
                "last_game_date": str(panel["GAME_DATE"].max().date()),
                "cache_dir": str(config.cache_dir),
            },
            indent=2,
        )
    )


@app.command("audit-data")
def audit_data(
    demo: bool = typer.Option(False, help="Audit a DEMO panel only"),
    verbose: bool = False,
) -> None:
    """Print data readiness field statuses."""
    _setup_logging(verbose)
    from src.models.data_audit import audit_player_panel

    panel, is_demo = _load_real_or_demo(demo)
    report = audit_player_panel(panel, dataset_name="demo_panel" if is_demo else "player_panel")
    typer.echo(json.dumps(report, indent=2, default=str))


@app.command("train-minutes")
def train_minutes(
    start_date: str = typer.Option(..., "--start-date"),
    end_date: str = typer.Option(..., "--end-date"),
    demo: bool = False,
    verbose: bool = False,
) -> None:
    """Train minutes CatBoost on chronological window."""
    _setup_logging(verbose)
    import pandas as pd

    from src.models.minutes_model import MinutesModel

    panel, _ = _load_real_or_demo(demo)
    d = panel.copy()
    d["GAME_DATE"] = pd.to_datetime(d["GAME_DATE"])
    train = d[(d["GAME_DATE"] >= start_date) & (d["GAME_DATE"] <= end_date)]
    model = MinutesModel()
    model.fit(train)
    typer.echo(json.dumps(model.get_model_metadata().model_dump(), indent=2, default=str))


@app.command("train-stats")
def train_stats(
    market: str = typer.Option(..., "--market", help="PTS|REB|AST"),
    start_date: str = typer.Option(..., "--start-date"),
    end_date: str = typer.Option(..., "--end-date"),
    demo: bool = False,
    verbose: bool = False,
) -> None:
    """Fit XGBoost + CatBoost adapters for one market (artifacts under model_runs)."""
    _setup_logging(verbose)
    from src.models.compare import (
        build_components,
        load_comparison_config,
        prepare_market_panel,
        resolve_feature_cols,
    )
    from src.models.labels import default_feature_cols

    market = market.upper()
    if market not in {"PTS", "REB", "AST"}:
        typer.echo("Launch markets are PTS/REB/AST only", err=True)
        raise SystemExit(1)
    panel, is_demo = _load_real_or_demo(demo)
    work = prepare_market_panel(panel, market)
    work = work[(work["GAME_DATE"] >= start_date) & (work["GAME_DATE"] <= end_date)]
    cols, _dropped = resolve_feature_cols(work, list(default_feature_cols(market)))  # type: ignore[arg-type]
    if not cols:
        typer.echo(f"No usable features for {market} in this panel", err=True)
        raise SystemExit(2)
    for c in ("TEAM_ABBREVIATION", "OPPONENT_ABBREVIATION", "SEASON"):
        if c in work.columns and c not in cols:
            cols = list(cols) + [c]
    cfg = load_comparison_config()
    comps = build_components(market, cols, cfg)
    art = Path(cfg.get("artifacts_dir", "data/external/model_runs/comparison"))
    if is_demo:
        art = art / "demo"
    art.mkdir(parents=True, exist_ok=True)
    mid = len(work) * 2 // 3
    train, val = work.iloc[:mid], work.iloc[mid:]
    for name, model in comps.items():
        try:
            model.fit(train, val)
            if hasattr(model, "save"):
                model.save(art / f"{name}_{market}")
            typer.echo(f"fitted {name} {market} -> {art}")
        except Exception as exc:  # noqa: BLE001
            logger.error("%s failed: %s", name, exc)


@app.command("evaluate")
def evaluate(
    market: str = typer.Option("PTS", "--market"),
    start_date: str = typer.Option(..., "--start-date"),
    end_date: str = typer.Option(..., "--end-date"),
    demo: bool = False,
    verbose: bool = False,
) -> None:
    """Evaluate models on a validation window (train ends day before start_date)."""
    _setup_logging(verbose)
    from src.models.compare import compare_models_on_panel, load_comparison_config

    panel, is_demo = _load_real_or_demo(demo)
    # train_end = day before evaluation start
    import pandas as pd

    train_end = str((pd.Timestamp(start_date) - pd.Timedelta(days=1)).date())
    result = compare_models_on_panel(
        panel,
        markets=[market.upper()],
        train_end=train_end,
        validation_end=end_date,
        cfg=load_comparison_config(),
    )
    typer.echo(json.dumps({"summary": result["summary"], "winners": result["winners"]}, indent=2))


@app.command("compare-models")
def compare_models(
    markets: str = typer.Option("PTS,REB,AST", "--markets"),
    train_end: str = typer.Option("2025-01-15", "--train-end"),
    validation_end: str = typer.Option("2025-02-15", "--validation-end"),
    demo: bool = typer.Option(False, help="Use DEMO panel; writes to outputs/demo"),
    verbose: bool = False,
) -> None:
    """Chronological comparison across models; writes outputs/ CSVs."""
    _setup_logging(verbose)
    from src.models.compare import compare_models_on_panel, load_comparison_config
    from src.models.data_audit import audit_player_panel
    from src.models.exports import write_comparison_exports

    panel, is_demo = _load_real_or_demo(demo or False)
    demo = demo or is_demo
    mkt = [m.strip().upper() for m in markets.split(",") if m.strip()]
    cfg = load_comparison_config()
    result = compare_models_on_panel(
        panel, markets=mkt, train_end=train_end, validation_end=validation_end, cfg=cfg
    )
    quality = audit_player_panel(panel, dataset_name="demo_panel" if demo else "player_panel")
    manifest = write_comparison_exports(
        result,
        output_dir=cfg.get("outputs_dir", "outputs"),
        quality_row=quality,
        demo=demo,
    )
    typer.echo(json.dumps({"winners": result.get("winners"), "manifest": manifest}, indent=2, default=str))


@app.command("generate-exports")
def generate_exports(
    start_date: str = typer.Option("2025-01-16", "--start-date"),
    end_date: str = typer.Option("2025-02-15", "--end-date"),
    demo: bool = True,
    verbose: bool = False,
) -> None:
    """Generate downloadable CSVs (defaults to demo for safety)."""
    import pandas as pd

    train_end = str((pd.Timestamp(start_date) - pd.Timedelta(days=1)).date())
    compare_models(
        markets="PTS,REB,AST",
        train_end=train_end,
        validation_end=end_date,
        demo=demo,
        verbose=verbose,
    )


@app.command("predict-slate")
def predict_slate(
    date: str = typer.Option(
        None,
        "--date",
        help="Slate date YYYY-MM-DD interpreted as America/Los_Angeles calendar day",
    ),
    shadow_mode: bool = typer.Option(True, "--shadow-mode/--no-shadow-mode"),
    demo: bool = False,
    verbose: bool = False,
) -> None:
    """Score a slate in shadow mode (no betting actions). Dates are Pacific."""
    _setup_logging(verbose)
    from src.utils.timezones import DISPLAY_TZ_NAME, pacific_calendar_date

    if not shadow_mode:
        typer.echo("Only shadow-mode is supported in RESEARCH_ONLY", err=True)
        raise SystemExit(1)
    slate = date or str(pacific_calendar_date())
    panel, _ = _load_real_or_demo(demo)
    import pandas as pd

    d = panel.copy()
    d["GAME_DATE"] = pd.to_datetime(d["GAME_DATE"])
    # GAME_DATE is a calendar date presented as Pacific slate day (no feed TZ).
    matched = d[d["GAME_DATE"].dt.strftime("%Y-%m-%d") == slate]
    typer.echo(
        json.dumps(
            {
                "date": slate,
                "timezone_display": DISPLAY_TZ_NAME,
                "shadow_mode": True,
                "rows": int(len(matched)),
            },
            indent=2,
        )
    )
    if matched.empty:
        typer.echo("No rows for Pacific slate date (expected off-season or missing logs).")

if __name__ == "__main__":
    app()
