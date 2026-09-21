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
from typing import Optional

import typer

app = typer.Typer(add_completion=False, no_args_is_help=True, help="NBA model comparison (research only)")
logger = logging.getLogger("nba_model")


def _setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
    )


def _load_team_games():
    """Team-level results for Elo, from the licensed workbook when configured.

    Returns None when unavailable — the builder then skips team-strength
    features and says so, rather than inventing ratings.
    """
    import os

    path = os.environ.get("BIGDATABALL_XLSX")
    if not path or not Path(path).exists():
        logger.info(
            "BIGDATABALL_XLSX unset or missing — no team Elo features this run. "
            "Point it at the licensed workbook to enable them."
        )
        return None
    try:
        from src.ingestion.bigdataball import load_bigdataball_workbook

        team_games, _market = load_bigdataball_workbook(path)
        logger.info("Loaded %d team-game rows for Elo from %s", len(team_games), path)
        return team_games
    except Exception as exc:  # noqa: BLE001 — degrade with a reason, never fake it
        logger.warning("Could not load team games (%s) — proceeding without Elo", exc)
        return None


def _load_real_or_demo(
    demo: bool,
    seasons: str | None = None,
    season_type: str | None = None,
):
    """
    Build the feature matrix from real logs, or a synthetic demo panel.

    ``seasons`` and ``season_type`` are threaded through to the loader.
    Without them every command fell back to ``BoxScoreLoadConfig``'s
    default season, so ``ingest-logs --seasons 2024-25`` cached one season
    and the very next command trained on a different one — quietly, and
    with no error to notice.
    """
    from src.features.builder import build_feature_matrix
    from src.models.data_audit import make_demo_panel

    team_games = _load_team_games()

    if demo:
        logger.warning("DEMO MODE — synthetic panel; do not treat metrics as real")
        raw = make_demo_panel()
        # Real team ratings must never be joined onto synthetic players.
        return build_feature_matrix(raw), True
    try:
        from src.ingestion.boxscores import BoxScoreLoadConfig, load_player_game_logs

        config = None
        if seasons or season_type:
            defaults = BoxScoreLoadConfig()
            config = BoxScoreLoadConfig(
                seasons=(
                    tuple(s.strip() for s in seasons.split(",") if s.strip())
                    if seasons else defaults.seasons
                ),
                season_type=season_type or defaults.season_type,
            )
            logger.info(
                "Loading player logs for seasons=%s season_type=%r",
                list(config.seasons), config.season_type,
            )

        raw = load_player_game_logs(config)
        if raw is None or raw.empty:
            raise RuntimeError("empty player logs")
        return build_feature_matrix(raw, team_games=team_games), False
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
    persist: bool = typer.Option(
        False, "--persist",
        help="Also write the logs to Postgres, which is what main.py reads",
    ),
    verbose: bool = False,
) -> None:
    """Pull player game logs from the NBA stats API and cache them locally.

    Needs network access to stats.nba.com. Run this before any training —
    the BigDataBall workbook is team-level and cannot supply player lines.

    The local cache feeds this CLI's own commands. ``main.py`` reads the
    database instead, so a run without ``--persist`` leaves the orchestrator
    with an empty panel no matter how many rows land in the cache.
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

    summary: dict[str, object] = {
        "rows": int(len(panel)),
        "players": int(panel["PLAYER_ID"].nunique()),
        "games": int(panel["GAME_ID"].nunique()),
        "first_game_date": str(panel["GAME_DATE"].min().date()),
        "last_game_date": str(panel["GAME_DATE"].max().date()),
        "cache_dir": str(config.cache_dir),
    }

    if persist:
        from src.db.repository import upsert_player_game_logs

        try:
            summary["rows_written_to_db"] = upsert_player_game_logs(panel)
        except Exception as exc:  # noqa: BLE001 — report, do not claim success
            typer.echo(f"Cached to disk, but the database write failed: {exc}", err=True)
            raise SystemExit(3) from exc
    else:
        summary["rows_written_to_db"] = 0
        summary["note"] = (
            "Cached locally only. main.py reads Postgres, so re-run with "
            "--persist before the orchestrator will see these rows."
        )

    typer.echo(json.dumps(summary, indent=2))


@app.command("ingest-kaggle")
def ingest_kaggle(
    path: str = typer.Option(
        None, "--path",
        help="Local CSV/Parquet export already on disk (no network needed)",
    ),
    dataset: str = typer.Option(
        "eoinamoore/historical-nba-data-and-player-box-scores", "--dataset",
        help="Kaggle dataset ref, used only when --path is omitted",
    ),
    file_path: str = typer.Option(
        None, "--file",
        help="Which file inside the Kaggle dataset to load (required for --dataset)",
    ),
    describe: bool = typer.Option(
        False, "--describe",
        help="Report the discovered column mapping and exit without writing",
    ),
    persist: bool = typer.Option(
        False, "--persist", help="Write the panel to Postgres (what main.py reads)"
    ),
    verbose: bool = False,
) -> None:
    """Load an NBA player box-score export and optionally persist it.

    A second path to the player panel that does not depend on
    stats.nba.com being reachable. Run with --describe first on any new
    export: the column mapping is DISCOVERED, and you should see what was
    recognised before trusting it.
    """
    _setup_logging(verbose)
    from src.ingestion.kaggle_nba import (
        KaggleNbaError,
        describe_schema,
        load_from_kagglehub,
        load_local_export,
        normalize_player_box_scores,
    )

    try:
        raw = (
            load_local_export(path) if path
            else load_from_kagglehub(dataset, file_path or "")
        )
    except KaggleNbaError as exc:
        typer.echo(f"Ingest failed: {exc}", err=True)
        raise SystemExit(2) from exc

    report = describe_schema(raw)
    if describe:
        typer.echo(json.dumps(report.as_dict(), indent=2, default=str))
        # An unusable export is a failure even in describe mode, so a
        # scripted check cannot read "column not recognised" as success.
        raise SystemExit(0 if report.usable else 3)

    try:
        panel = normalize_player_box_scores(raw, report)
    except KaggleNbaError as exc:
        typer.echo(f"Ingest failed: {exc}", err=True)
        raise SystemExit(3) from exc

    summary: dict[str, object] = {
        "rows": int(len(panel)),
        "players": int(panel["PLAYER_NAME"].nunique()),
        "first_game_date": str(panel["GAME_DATE"].min().date()),
        "last_game_date": str(panel["GAME_DATE"].max().date()),
        "mapped_columns": report.mapped,
        "missing_optional": report.missing_optional,
    }

    if persist:
        from src.db.repository import upsert_player_game_logs

        try:
            summary["rows_written_to_db"] = upsert_player_game_logs(panel)
        except Exception as exc:  # noqa: BLE001 — report, never claim success
            typer.echo(f"Parsed the export, but the database write failed: {exc}", err=True)
            raise SystemExit(4) from exc
    else:
        summary["rows_written_to_db"] = 0
        summary["note"] = "Parsed only. Re-run with --persist to write to Postgres."

    typer.echo(json.dumps(summary, indent=2, default=str))


@app.command("ingest-basketball-reference")
def ingest_basketball_reference(
    path: str = typer.Argument(..., help="Basketball-Reference season CSV on disk"),
    season: str = typer.Option(
        ..., "--season",
        help="Season the table covers, e.g. 2024-25. NOT inferred: the CSV "
             "does not carry it, and a wrong season defeats the leakage check.",
    ),
    out: str = typer.Option(
        None, "--out",
        help="Write the parsed season totals to this CSV (one row per player)",
    ),
    splits: bool = typer.Option(
        False, "--splits",
        help="Write per-team rows for traded players instead of season totals",
    ),
    verbose: bool = False,
) -> None:
    """Parse a Basketball-Reference season table (per game, per 100 possessions, play-by-play, adjusted shooting).

    These tables are SEASON AGGREGATES. They are safe as PRIOR-season
    features only; joining one onto its own season leaks the future into
    every game, and src.ingestion.basketball_reference refuses to do it.

    Data from Basketball-Reference.com (Sports Reference LLC). When using SR
    data, please cite them and provide a link and/or a mention.
    """
    _setup_logging(verbose)
    from src.ingestion.basketball_reference import (
        BasketballReferenceError,
        describe_sr_table,
        multi_team_report,
        read_sr_season_csv,
        season_totals,
        team_splits,
    )

    try:
        table = read_sr_season_csv(path, season=season)
        summary = describe_sr_table(table)
    except BasketballReferenceError as exc:
        typer.echo(f"Parse failed: {exc}", err=True)
        raise SystemExit(2) from exc

    mismatches = multi_team_report(table)
    summary["multi_team_blocks"] = int(len(mismatches))
    summary["multi_team_games_mismatched"] = (
        int((~mismatches["MATCHES"]).sum()) if len(mismatches) else 0
    )

    if out:
        frame = team_splits(table) if splits else season_totals(table)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out, index=False)
        summary["written"] = out
        summary["written_rows"] = int(len(frame))
        summary["written_view"] = "team_splits" if splits else "season_totals"

    typer.echo(json.dumps(summary, indent=2, default=str))


@app.command("audit-data")
def audit_data(
    demo: bool = typer.Option(False, help="Audit a DEMO panel only"),
    seasons: str = typer.Option(
        None, "--seasons",
        help="Comma-separated seasons, e.g. 2024-25,2025-26 (default: the loader's)",
    ),
    season_type: str = typer.Option(None, "--season-type", help="e.g. 'Regular Season'"),
    verbose: bool = False,
) -> None:
    """Print data readiness field statuses."""
    _setup_logging(verbose)
    from src.models.data_audit import audit_player_panel

    panel, is_demo = _load_real_or_demo(demo, seasons, season_type)
    report = audit_player_panel(panel, dataset_name="demo_panel" if is_demo else "player_panel")
    typer.echo(json.dumps(report, indent=2, default=str))


@app.command("train-minutes")
def train_minutes(
    start_date: str = typer.Option(..., "--start-date"),
    end_date: str = typer.Option(..., "--end-date"),
    demo: bool = False,
    seasons: str = typer.Option(
        None, "--seasons",
        help="Comma-separated seasons, e.g. 2024-25,2025-26 (default: the loader's)",
    ),
    season_type: str = typer.Option(None, "--season-type", help="e.g. 'Regular Season'"),
    verbose: bool = False,
) -> None:
    """Train minutes CatBoost on chronological window and save the artifact."""
    _setup_logging(verbose)
    import pandas as pd

    from src.models.compare import load_comparison_config
    from src.models.minutes_model import MinutesModel

    panel, is_demo = _load_real_or_demo(demo, seasons, season_type)
    d = panel.copy()
    d["GAME_DATE"] = pd.to_datetime(d["GAME_DATE"])
    train = d[(d["GAME_DATE"] >= start_date) & (d["GAME_DATE"] <= end_date)]
    if train.empty:
        typer.echo(
            f"No rows between {start_date} and {end_date} — nothing to train on.",
            err=True,
        )
        raise SystemExit(2)

    model = MinutesModel()
    model.fit(train)

    cfg = load_comparison_config()
    art = Path(cfg.get("artifacts_dir", "data/external/model_runs/comparison"))
    if is_demo:
        art = art / "demo"
    art.mkdir(parents=True, exist_ok=True)
    # Previously the fitted boosters were reported and then discarded when
    # the process exited, so the command "succeeded" and left nothing behind.
    model.save(art / "minutes")

    payload = model.get_model_metadata().model_dump()
    payload["artifact"] = str(art / "minutes")
    typer.echo(json.dumps(payload, indent=2, default=str))


@app.command("train-stats")
def train_stats(
    market: str = typer.Option(..., "--market", help="PTS|REB|AST"),
    start_date: str = typer.Option(..., "--start-date"),
    end_date: str = typer.Option(..., "--end-date"),
    demo: bool = False,
    seasons: str = typer.Option(
        None, "--seasons",
        help="Comma-separated seasons, e.g. 2024-25,2025-26 (default: the loader's)",
    ),
    season_type: str = typer.Option(None, "--season-type", help="e.g. 'Regular Season'"),
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
    panel, is_demo = _load_real_or_demo(demo, seasons, season_type)
    work = prepare_market_panel(panel, market)
    work = work[(work["GAME_DATE"] >= start_date) & (work["GAME_DATE"] <= end_date)]
    cols, _dropped = resolve_feature_cols(work, list(default_feature_cols(market)))  # type: ignore[arg-type]
    if not cols:
        typer.echo(f"No usable features for {market} in this panel", err=True)
        raise SystemExit(2)
    # CatBoost handles these natively; XGBoost rejects non-numerics outright,
    # so it keeps the numeric list. Passing the combined list to both made
    # every train-stats run fail XGBoost with a DATA_NOT_AVAILABLE and write
    # only two of the three artifacts — quietly, since the loop logs and
    # continues.
    numeric_cols = list(cols)
    for c in ("TEAM_ABBREVIATION", "OPPONENT_ABBREVIATION", "SEASON"):
        if c in work.columns and c not in cols:
            cols = list(cols) + [c]
    cfg = load_comparison_config()
    comps = build_components(market, cols, cfg, xgb_feature_cols=numeric_cols)
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
    seasons: str = typer.Option(
        None, "--seasons",
        help="Comma-separated seasons, e.g. 2024-25,2025-26 (default: the loader's)",
    ),
    season_type: str = typer.Option(None, "--season-type", help="e.g. 'Regular Season'"),
    verbose: bool = False,
) -> None:
    """Evaluate models on a validation window (train ends day before start_date)."""
    _setup_logging(verbose)
    from src.models.compare import compare_models_on_panel, load_comparison_config

    panel, is_demo = _load_real_or_demo(demo, seasons, season_type)
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
    seasons: str = typer.Option(
        None, "--seasons",
        help="Comma-separated seasons, e.g. 2024-25,2025-26 (default: the loader's)",
    ),
    season_type: str = typer.Option(None, "--season-type", help="e.g. 'Regular Season'"),
    verbose: bool = False,
) -> None:
    """Chronological comparison across models; writes outputs/ CSVs."""
    _setup_logging(verbose)
    from src.models.compare import compare_models_on_panel, load_comparison_config
    from src.models.data_audit import audit_player_panel
    from src.models.exports import write_comparison_exports

    panel, is_demo = _load_real_or_demo(demo or False, seasons, season_type)
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
    markets: str = typer.Option("PTS,REB,AST", "--markets", help="Comma-separated"),
    out: str = typer.Option(None, "--out", help="Optional CSV path for the predictions"),
    demo: bool = False,
    seasons: str = typer.Option(
        None, "--seasons",
        help="Comma-separated seasons, e.g. 2024-25,2025-26 (default: the loader's)",
    ),
    season_type: str = typer.Option(None, "--season-type", help="e.g. 'Regular Season'"),
    verbose: bool = False,
) -> None:
    """Score a slate in shadow mode (no betting actions). Dates are Pacific.

    Loads the artifacts written by ``train-stats`` and emits one shadow
    prediction per player-market. The line used is ``RESEARCH_LINE``, a
    trailing 10-game average — NOT a sportsbook line. No EV, no ranking,
    no recommendation.
    """
    _setup_logging(verbose)
    import pandas as pd

    from src.models.compare import (
        build_components,
        load_comparison_config,
        prepare_market_panel,
        resolve_feature_cols,
    )
    from src.models.labels import default_feature_cols
    from src.utils.timezones import DISPLAY_TZ_NAME, pacific_calendar_date

    if not shadow_mode:
        typer.echo("Only shadow-mode is supported in RESEARCH_ONLY", err=True)
        raise SystemExit(1)

    slate = date or str(pacific_calendar_date())
    wanted = [m.strip().upper() for m in markets.split(",") if m.strip()]
    panel, is_demo = _load_real_or_demo(demo, seasons, season_type)

    cfg = load_comparison_config()
    art = Path(cfg.get("artifacts_dir", "data/external/model_runs/comparison"))
    if is_demo:
        art = art / "demo"

    rows: list[dict[str, object]] = []
    per_market: dict[str, object] = {}

    for market in wanted:
        work = prepare_market_panel(panel, market)
        work["GAME_DATE"] = pd.to_datetime(work["GAME_DATE"])
        # GAME_DATE is a calendar date presented as Pacific slate day.
        on_slate = work[work["GAME_DATE"].dt.strftime("%Y-%m-%d") == slate]
        if on_slate.empty:
            per_market[market] = {"status": "NO_ROWS", "rows": 0}
            continue

        cols, _dropped = resolve_feature_cols(on_slate, list(default_feature_cols(market)))
        for c in ("TEAM_ABBREVIATION", "OPPONENT_ABBREVIATION", "SEASON"):
            if c in on_slate.columns and c not in cols:
                cols = list(cols) + [c]

        loaded = {}
        for name, model in build_components(market, cols, cfg).items():
            path = art / f"{name}_{market}"
            if not hasattr(model, "load"):
                continue
            try:
                loaded[name] = model.load(path)
            except Exception as exc:  # noqa: BLE001 — a missing artifact is expected
                logger.info("No usable %s artifact for %s (%s)", name, market, exc)

        if not loaded:
            # Reporting a row count here is what this command used to do, and
            # it read as a scored slate. Say plainly that nothing scored it.
            per_market[market] = {
                "status": "DATA_NOT_AVAILABLE",
                "rows": int(len(on_slate)),
                "reason": f"no trained artifacts under {art} — run train-stats first",
            }
            continue

        scored_here = 0
        for name, model in loaded.items():
            try:
                predictions = model.predict_rows(on_slate, line_col="RESEARCH_LINE")
            except Exception as exc:  # noqa: BLE001
                logger.error("%s could not score %s: %s", name, market, exc)
                continue
            for p in predictions:
                rows.append({
                    "slate_date_pt": slate,
                    "market": market,
                    "model": name,
                    "player_id": p.player_id,
                    "player_name": p.player_name,
                    "event_id": p.event_id,
                    "research_line": p.prop_line,
                    "line_source": "RESEARCH_L10_NOT_A_SPORTSBOOK_LINE",
                    "projection": p.prediction_mean,
                    "probability_over": p.probability_over,
                    "probability_under": p.probability_under,
                    "probability_push": p.probability_push,
                    "warnings": "; ".join(p.warnings) if p.warnings else None,
                })
                scored_here += 1
        per_market[market] = {
            "status": "SCORED" if scored_here else "DATA_NOT_AVAILABLE",
            "rows": int(len(on_slate)),
            "predictions": scored_here,
            "models": sorted(loaded),
        }

    summary: dict[str, object] = {
        "date": slate,
        "timezone_display": DISPLAY_TZ_NAME,
        "shadow_mode": True,
        "demo": is_demo,
        "artifacts_dir": str(art),
        "predictions": len(rows),
        "by_market": per_market,
        "line_note": (
            "RESEARCH_LINE is a trailing 10-game average, not a sportsbook "
            "line. No EV, no ranking, no recommendation."
        ),
    }

    if rows and out:
        target = Path(out)
        target.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(target, index=False)
        summary["csv"] = str(target)

    typer.echo(json.dumps(summary, indent=2, default=str))

    if not rows:
        typer.echo(
            "Nothing scored. Either no games fall on this Pacific slate date, or "
            "train-stats has not written artifacts for these markets.",
            err=True,
        )
        raise SystemExit(4)



# ---------------------------------------------------------------------------
# Paper-research layer (waves 3 / 4 / 5a / 5b)
#
# MANUAL_ONLY throughout. These commands log what YOU decided and grade it
# afterwards; nothing here places a wager, sizes a stake, or ranks a slate
# by edge.
# ---------------------------------------------------------------------------

@app.command("write-run-manifest")
def write_run_manifest_cmd(
    output_dir: str = typer.Option("outputs/demo", "--output-dir"),
    demo: bool = True,
    verbose: bool = False,
) -> None:
    """Write a forward-only run manifest for an existing export directory."""
    _setup_logging(verbose)
    from src.models.artifact_registry import write_run_manifest

    root = Path(output_dir)
    if not root.exists():
        typer.echo(f"DATA_NOT_AVAILABLE: {root} missing — run compare-models first", err=True)
        raise SystemExit(2)
    files = sorted(p for p in root.glob("*") if p.is_file())
    manifest = write_run_manifest(
        root,
        steps=[{"step_name": "cli_write_run_manifest", "notes": "Wave 4 checksum snapshot"}],
        output_files=files,
        meta={"demo_mode": demo, "wave": 4, "source": "nba_model_cli"},
    )
    typer.echo(json.dumps({k: v for k, v in manifest.items() if not str(k).startswith("_")}, indent=2, default=str))


@app.command("research-slate")
def research_slate(
    markets: str = typer.Option("PTS,REB,AST", "--markets"),
    train_end: str = typer.Option("2025-01-15", "--train-end"),
    validation_end: str = typer.Option("2025-02-15", "--validation-end"),
    preferred_model: str = typer.Option("distribution", "--preferred-model"),
    out: Path = typer.Option(Path("outputs/demo/research_slate.csv"), "--out"),
    demo: bool = typer.Option(True, help="DEMO panel for wiring"),
    verbose: bool = False,
) -> None:
    """Build a MANUAL research slate board (no wager placement)."""
    _setup_logging(verbose)
    from src.models.compare import compare_models_on_panel, load_comparison_config
    from src.quant.paper_research import (
        WAVE3_DISCLAIMER,
        research_slate_from_predictions,
        write_slate_csv,
    )
    from src.utils.timezones import pacific_calendar_date

    panel, is_demo = _load_real_or_demo(demo)
    mkt = [m.strip().upper() for m in markets.split(",") if m.strip()]
    result = compare_models_on_panel(
        panel,
        markets=mkt,
        train_end=train_end,
        validation_end=validation_end,
        cfg=load_comparison_config(),
    )
    slate = str(pacific_calendar_date())
    rows = research_slate_from_predictions(
        result.get("predictions") or [],
        slate_date=slate,
        preferred_model=preferred_model or None,
    )
    n = write_slate_csv(rows, out)
    typer.echo(
        json.dumps(
            {
                "placement_mode": "MANUAL_ONLY",
                "disclaimer": WAVE3_DISCLAIMER,
                "slate_date": slate,
                "rows": n,
                "out": str(out),
                "demo": demo or is_demo,
            },
            indent=2,
        )
    )


@app.command("log-manual-bet")
def log_manual_bet_cmd(
    game_id: str = typer.Option(..., "--game-id"),
    prop_stat: str = typer.Option(..., "--prop-stat"),
    line: float = typer.Option(..., "--line"),
    side: str = typer.Option(..., "--side", help="over|under"),
    odds: int = typer.Option(..., "--odds", help="American odds you took"),
    model_prob: float = typer.Option(..., "--model-prob", help="Model P(over)"),
    player_id: Optional[str] = typer.Option(None, "--player-id"),
    player_name: Optional[str] = typer.Option(None, "--player-name"),
    bookmaker: Optional[str] = typer.Option(None, "--bookmaker"),
    unit_stake: float = typer.Option(1.0, "--unit-stake", help="YOUR stake units (not model Kelly)"),
    store_dir: Path = typer.Option(Path("data/external/market_store"), "--store-dir"),
    verbose: bool = False,
) -> None:
    """Log a bet YOU placed manually (paper research). Never places a wager."""
    _setup_logging(verbose)
    from src.quant.historical_store import HistoricalStore, HistoricalStoreConfig
    from src.quant.paper_research import ManualBetInput, log_manual_bet

    side_l = side.lower().strip()
    if side_l not in {"over", "under"}:
        typer.echo("DATA_NOT_AVAILABLE: --side must be over|under", err=True)
        raise SystemExit(2)
    store = HistoricalStore(HistoricalStoreConfig(root=store_dir))
    bet = ManualBetInput(
        game_id=game_id,
        player_id=player_id,
        player_name=player_name,
        prop_stat=prop_stat.upper(),
        line=line,
        bet_side=side_l,  # type: ignore[arg-type]
        taken_odds_american=odds,
        model_prob=model_prob,
        bookmaker=bookmaker,
        unit_stake=unit_stake,
    )
    result = log_manual_bet(store, bet)
    typer.echo(json.dumps(result, indent=2, default=str))


@app.command("paper-report")
def paper_report_cmd(
    store_dir: Path = typer.Option(Path("data/external/market_store"), "--store-dir"),
    verbose: bool = False,
) -> None:
    """Paper-book improvement report (ROI / calibration) — research audit only."""
    _setup_logging(verbose)
    from src.quant.historical_store import HistoricalStore, HistoricalStoreConfig
    from src.quant.paper_research import paper_improvement_report

    store = HistoricalStore(HistoricalStoreConfig(root=store_dir))
    typer.echo(json.dumps(paper_improvement_report(store), indent=2, default=str))


@app.command("pocket-roi")
def pocket_roi_cmd(
    store_dir: Path = typer.Option(Path("data/external/market_store"), "--store-dir"),
    out: Path = typer.Option(Path("outputs/demo/pocket_roi.csv"), "--out"),
    verbose: bool = False,
) -> None:
    """BookieX-style pocket ROI board from manual paper log (no stake sizing)."""
    _setup_logging(verbose)
    from src.quant.historical_store import HistoricalStore, HistoricalStoreConfig
    from src.quant.pocket_roi import write_pocket_roi_csv

    store = HistoricalStore(HistoricalStoreConfig(root=store_dir))
    result = write_pocket_roi_csv(store, out)
    # Drop large pocket list noise in console — keep summary + path
    summary = {k: v for k, v in result.items() if k != "pockets"}
    summary["n_pockets"] = len(result.get("pockets") or [])
    typer.echo(json.dumps(summary, indent=2, default=str))


@app.command("paper-calibration")
def paper_calibration_cmd(
    store_dir: Path = typer.Option(Path("data/external/market_store"), "--store-dir"),
    out: Path = typer.Option(Path("outputs/demo/paper_reliability.csv"), "--out"),
    n_bins: int = typer.Option(10, "--n-bins"),
    verbose: bool = False,
) -> None:
    """Reliability diagram / ECE on settled manual paper bets (research audit)."""
    _setup_logging(verbose)
    from src.quant.historical_store import HistoricalStore, HistoricalStoreConfig
    from src.quant.paper_calibration import write_paper_reliability_csv

    store = HistoricalStore(HistoricalStoreConfig(root=store_dir))
    result = write_paper_reliability_csv(store, out, n_bins=n_bins)
    summary = {
        k: v
        for k, v in result.items()
        if k not in {"reliability_table", "chart_points", "perfect_calibration_line"}
    }
    typer.echo(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    app()


if __name__ == "__main__":
    app()
