#!/usr/bin/env python3
"""The deployed worker: a Pacific-anchored scheduler around the slate pipeline.

RESEARCH_ONLY. This runs research jobs on a clock. It places no wager, contacts
no order API, and sizes no stake — the same rules as every other entry point,
and being unattended is a reason to restate them rather than to relax them.

WHY A WORKER AND NOT A CRON LINE. Railway can run a cron service, but a cron
expression is fixed to a clock and an NBA slate is not: tip-offs move by hours
across a season, and the two jobs here have an ORDER (project before the games,
settle after them) that a pair of independent cron entries cannot express. A
long-lived process also lets the pipeline's own state — a run that is still
going — decide whether the next one may start, which is what ``max_instances``
below is for.

WHY THE TIMES ARE PACIFIC. Every slate cutoff in this repository is a Pacific
calendar day (``utils.timezones.pacific_calendar_date``). A UTC-anchored
schedule would put the late West Coast games of one slate into the next day's
run for part of the year and not the rest, which is the kind of off-by-one that
silently changes what a backtest is measuring.

WHAT IT DOES NOT DO, AND WHY NOT. It does not re-anchor itself to the day's
first tip-off. That would need a schedule feed, every data host is denied from
the environment this was written in, and a scheduler whose timing logic has
never once been exercised against real data is worse than a fixed one: it looks
adaptive and is untested. The fixed anchors below are deliberately early enough
to precede any tip and late enough to follow any finish. Re-anchoring is a
change to make when the schedule feed is reachable and can be tested.

CONCURRENCY. Both jobs run with ``max_instances=1`` and ``coalesce=True``: a run
that overruns its window is not joined by a second copy, and a backlog of missed
firings collapses into one. XGBoost and CatBoost both default to every core, so
two concurrent fits on a shared container contend badly — measured in this
project's own test runs, three concurrent suites turned a 126s run into over
590s. ``PROPIQ_MAX_THREADS`` caps the per-process thread count on top of that.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
from datetime import datetime
from typing import Any

logger = logging.getLogger("propiq.scheduler")

# Pacific, for the reason in the module docstring.
from src.utils.timezones import DISPLAY_TZ, DISPLAY_TZ_NAME  # noqa: E402

ENV_SLATE_HOUR = "PROPIQ_SLATE_HOUR_PT"
ENV_SLATE_MINUTE = "PROPIQ_SLATE_MINUTE_PT"
ENV_SETTLE_HOUR = "PROPIQ_SETTLE_HOUR_PT"
ENV_SETTLE_MINUTE = "PROPIQ_SETTLE_MINUTE_PT"
ENV_MAX_THREADS = "PROPIQ_MAX_THREADS"
ENV_RUN_ON_START = "PROPIQ_RUN_ON_START"
ENV_DISPATCH = "PROPIQ_DISPATCH"
ENV_DISPATCH_ABSTENTIONS = "PROPIQ_DISPATCH_ABSTENTIONS"
ENV_CALIBRATION_REPORT = "PROPIQ_CALIBRATION_REPORT"
ENV_BOARD_CSV = "PROPIQ_BOARD_CSV"
ENV_BOARD_MARKETS = "PROPIQ_BOARD_MARKETS"
ENV_BOARD_TRAIN_END = "PROPIQ_BOARD_TRAIN_END"
ENV_BOARD_VALIDATION_END = "PROPIQ_BOARD_VALIDATION_END"
ENV_MIN_EV = "PROPIQ_MIN_EV"

DEFAULT_CALIBRATION_REPORT = "outputs/calibration.json"
DEFAULT_BOARD_CSV = "outputs/decision_board.csv"

# 09:00 PT: before any NBA tip (the earliest are late morning PT) and after the
# night's box scores have settled into the sources.
DEFAULT_SLATE_HOUR, DEFAULT_SLATE_MINUTE = 9, 0
# 03:30 PT: after even a West Coast overtime game has finished and been posted.
DEFAULT_SETTLE_HOUR, DEFAULT_SETTLE_MINUTE = 3, 30

# A job that misses its window by less than this still runs; past it, the slate
# has moved on and running late would project games that have already tipped.
SLATE_MISFIRE_GRACE_SECONDS = 60 * 60
SETTLE_MISFIRE_GRACE_SECONDS = 6 * 60 * 60


def _int_env(name: str, default: int, *, low: int, high: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("%s=%r is not an integer; using %d", name, raw, default)
        return default
    if not low <= value <= high:
        logger.warning(
            "%s=%d is outside %d..%d; using %d", name, value, low, high, default
        )
        return default
    return value


def cap_thread_counts() -> int | None:
    """
    Cap the numeric libraries' thread pools before anything imports them.

    XGBoost, CatBoost, OpenMP and BLAS each default to every visible core. On a
    shared container that is not parallelism, it is contention: the container's
    CPU share does not grow with the thread count, and the fits slow each other
    down. Returns the cap applied, or None when none was configured.

    Must run before numpy/xgboost are imported — several of these are read once
    at library load and ignored afterwards.
    """
    cap = _int_env(ENV_MAX_THREADS, 0, low=0, high=256)
    if cap <= 0:
        return None
    for variable in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(variable, str(cap))
    logger.info("Thread pools capped at %d per process", cap)
    return cap


def run_slate(argv: list[str] | None = None) -> int:
    """One slate run: ingest, build features, score, gate, persist, record.

    Delegates to ``main.main`` rather than reimplementing the sequence, so the
    scheduled path and the manual one cannot drift. An exception is caught and
    logged: a failed slate must not take the scheduler down, or one bad day
    ends every subsequent day too.
    """
    import main

    try:
        code = main.main(argv or [])
    except Exception:  # noqa: BLE001 — a scheduled job may not kill the worker
        logger.exception("Slate run raised; the scheduler stays up.")
        return 1
    logger.info("Slate run finished with code %s", code)

    # Dispatch is part of the slate job, not a separate one: a card is only
    # worth sending for the slate that was just built, and a second job could
    # fire after a failed build and post yesterday's board.
    if code == 0:
        run_board()
        run_dispatch()
    else:
        logger.warning("Slate returned %s — no board, no card.", code)
    return int(code)


def run_board() -> dict[str, Any]:
    """
    Build the recommendation board CSV that dispatch reads.

    WHY THE SLATE JOB DOES NOT ALREADY PRODUCE THIS. ``main.py`` writes
    projections to Postgres; the BOARD is a separate artifact built by comparing
    models over the panel, and it existed only as a hand-run CLI command. Without
    this step the dispatcher could only ever report that the CSV was missing.

    Shares ``pipeline.slate_board.build_slate_board`` with the CLI so the two
    cannot drift.

    A failure here is logged and returned, not raised: the slate's real output is
    already in Postgres, and a missing board turns into an honest abstention in
    dispatch rather than a dead worker.
    """
    board_csv = os.environ.get(ENV_BOARD_CSV) or DEFAULT_BOARD_CSV
    try:
        from src.pipeline.slate_board import DEFAULT_MARKETS, build_slate_board

        raw_markets = (os.environ.get(ENV_BOARD_MARKETS) or "").strip()
        markets = (
            [m.strip() for m in raw_markets.split(",") if m.strip()]
            if raw_markets else list(DEFAULT_MARKETS)
        )
        train_end = os.environ.get(ENV_BOARD_TRAIN_END) or "2025-01-15"
        validation_end = os.environ.get(ENV_BOARD_VALIDATION_END) or "2025-02-15"
        try:
            min_ev = float(os.environ.get(ENV_MIN_EV) or 0.0)
        except ValueError:
            min_ev = 0.0

        from src.db.repository import load_player_panel

        panel = load_player_panel()
        if panel is None or panel.empty:
            logger.warning("No player panel available — no board built.")
            return {"status": "DATA_NOT_AVAILABLE", "reason": "empty player panel"}

        result = build_slate_board(
            panel,
            out=board_csv,
            markets=markets,
            train_end=train_end,
            validation_end=validation_end,
            min_ev=min_ev,
        )
        logger.info("Board: %s", result.as_dict())
        return {"status": "OK", **result.as_dict()}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Board build raised; the scheduler stays up.")
        return {"status": "FAILED", "error": str(exc), "out": board_csv}


def run_dispatch() -> dict[str, Any]:
    """
    Send the slate's board to Discord, gated on the calibration report.

    WHY THIS EXISTS. Until it did, `main.py` and this worker contained no
    reference to `src/notify` at all: the scheduled run wrote projections and
    PENDING rows and told nobody, and a card only went out if a person ran a
    command by hand. For a pipeline whose point is a daily automated card, that
    was the hole.

    THE GATE IS NOT OPTIONAL HERE. Every board row rests on the model's own
    probability, so the board is gated as MODEL-sourced; with no usable
    calibration report the embed carries the gate's reason instead of the rows.
    While no graded evidence exists that is the only thing this will send, and
    that is correct rather than broken.

    Off unless a webhook is configured — a worker without DISCORD_WEBHOOK_URL
    should not spend a slate trying to post. Set PROPIQ_DISPATCH explicitly to
    override either way, and PROPIQ_DISPATCH_ABSTENTIONS=false to stay silent on
    a withheld or empty board instead of saying so.
    """
    import json
    from pathlib import Path

    webhook_configured = bool((os.environ.get("DISCORD_WEBHOOK_URL") or "").strip())
    if not _flag(ENV_DISPATCH, webhook_configured):
        logger.info(
            "Dispatch off (%s unset and no DISCORD_WEBHOOK_URL). Nothing sent.",
            ENV_DISPATCH,
        )
        return {"status": "SKIPPED", "reason": "dispatch disabled"}

    board_csv = Path(os.environ.get(ENV_BOARD_CSV) or DEFAULT_BOARD_CSV)
    report_path = Path(
        os.environ.get(ENV_CALIBRATION_REPORT) or DEFAULT_CALIBRATION_REPORT
    )

    # Imported BEFORE the try, because the except clause below names
    # DiscordDispatchError: an ImportError inside the try would leave that name
    # unbound and turn a missing dependency into a NameError while handling it.
    from src.notify.discord import DiscordDispatchError

    try:
        import pandas as pd

        from src.notify.discord import (
            DiscordConfig,
            build_abstention_embed,
            build_decision_board_embed,
            send_embeds,
        )
        from src.quant.dfs_payouts import ProbabilitySource
        from src.quant.publication_gate import calibration_gate

        report = None
        if report_path.exists():
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                # An unreadable report is not the same as no report, and the gate
                # would treat None as "none supplied". Say which it was.
                logger.warning("Calibration report %s is not JSON: %s", report_path, exc)
        verdict = calibration_gate(report, probability_source=ProbabilitySource.MODEL)

        if not board_csv.exists():
            if not _flag(ENV_DISPATCH_ABSTENTIONS, True):
                return {"status": "SKIPPED", "reason": f"{board_csv} missing"}
            embeds = [build_abstention_embed(
                f"No board was written for this slate ({board_csv} absent), so "
                "there is nothing to recommend. This is the pipeline reporting "
                "its own state, not a slate with no value in it.",
                title="No recommendations",
            )]
        else:
            frame = pd.read_csv(board_csv)
            rows = [
                type("Row", (), {k: (None if pd.isna(v) else v) for k, v in r.items()})()
                for r in frame.to_dict("records")
            ]
            slate = str(frame["slate_date"].iloc[0]) if "slate_date" in frame else None
            embeds = [build_decision_board_embed(
                rows, slate_date=slate, publication=verdict,
            )]
            if not verdict.allowed and not _flag(ENV_DISPATCH_ABSTENTIONS, True):
                logger.info("Board withheld and abstentions muted — nothing sent.")
                return {
                    "status": "SKIPPED",
                    "reason": "withheld, abstentions muted",
                    "publication": verdict.as_dict(),
                }

        result = send_embeds(embeds, config=DiscordConfig(dry_run=False))
        logger.info("Dispatch: %s", result.status)
        return {"status": result.status, "publication": verdict.as_dict()}
    except DiscordDispatchError as exc:
        logger.error("Dispatch refused: %s", exc)
        return {"status": "REFUSED", "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Dispatch raised; the scheduler stays up.")
        return {"status": "FAILED", "error": str(exc)}


def _flag(name: str, default: bool) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def run_settlement() -> dict[str, Any]:
    """Grade every PENDING prop, then rebuild the calibration report.

    THE ORDER IS THE DEPENDENCY. ``settlement.recorder`` writes the predictions,
    this grades them, and the calibration report is computed from what grading
    produced. Rebuilding it here rather than in the slate job means the morning
    run reads evidence that already includes last night's results, instead of
    evidence one day stale.

    A failed report does not fail settlement: the grading is the durable part and
    the report can be rebuilt from the rows at any time.
    """
    out: dict[str, Any] = {}
    try:
        from src.settlement.runner import settle_pending_props

        report = settle_pending_props()
        out = report.as_dict() if hasattr(report, "as_dict") else {"report": report}
        logger.info("Settlement: %s", out)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Settlement raised; the scheduler stays up.")
        out = {"status": "FAILED", "error": str(exc)}

    out["calibration"] = rebuild_calibration_report()
    return out


def rebuild_calibration_report(path: str | None = None) -> dict[str, Any]:
    """
    Recompute the calibration evidence the publication gate reads.

    Writes JSON to ``PROPIQ_CALIBRATION_REPORT`` (default outputs/calibration.json).
    A DATA_NOT_AVAILABLE report is still written: the gate reads its ``status``
    and refuses, which is the state that should hold until graded rows exist. Not
    writing it at all would leave a stale PASSING report in place after the rows
    behind it aged out.
    """
    import json
    from pathlib import Path

    target = Path(path or os.environ.get(ENV_CALIBRATION_REPORT) or DEFAULT_CALIBRATION_REPORT)
    try:
        from src.settlement.calibration import (
            prop_result_calibration_report,
            summarise,
        )

        report = prop_result_calibration_report()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
        logger.info("Calibration report -> %s: %s", target, summarise(report))
        return {"out": str(target), "status": report.get("status"),
                "n_scored": report.get("n_scored"), "ece": report.get("ece")}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Calibration report failed; the scheduler stays up.")
        return {"status": "FAILED", "error": str(exc), "out": str(target)}


def build_scheduler(scheduler: Any | None = None) -> Any:
    """
    Wire the two jobs onto a scheduler. Separated so it can be tested.

    Returns the scheduler without starting it, so a test can inspect the job
    table — the times, the grace periods and the concurrency limits are the
    part worth asserting, and starting a BlockingScheduler in a test would not
    return.
    """
    if scheduler is None:
        try:
            from apscheduler.schedulers.blocking import BlockingScheduler
        except ImportError as exc:  # pragma: no cover - import guard
            raise SystemExit(
                "APScheduler is not installed. This worker needs the deploy "
                "extra: pip install -e '.[deploy]'"
            ) from exc
        scheduler = BlockingScheduler(timezone=DISPLAY_TZ)

    slate_hour = _int_env(ENV_SLATE_HOUR, DEFAULT_SLATE_HOUR, low=0, high=23)
    slate_minute = _int_env(ENV_SLATE_MINUTE, DEFAULT_SLATE_MINUTE, low=0, high=59)
    settle_hour = _int_env(ENV_SETTLE_HOUR, DEFAULT_SETTLE_HOUR, low=0, high=23)
    settle_minute = _int_env(ENV_SETTLE_MINUTE, DEFAULT_SETTLE_MINUTE, low=0, high=59)

    scheduler.add_job(
        run_slate,
        trigger="cron",
        hour=slate_hour,
        minute=slate_minute,
        timezone=DISPLAY_TZ,
        id="slate",
        name="Daily slate projections (RESEARCH_ONLY)",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=SLATE_MISFIRE_GRACE_SECONDS,
        replace_existing=True,
    )
    scheduler.add_job(
        run_settlement,
        trigger="cron",
        hour=settle_hour,
        minute=settle_minute,
        timezone=DISPLAY_TZ,
        id="settlement",
        name="Grade finished games",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=SETTLE_MISFIRE_GRACE_SECONDS,
        replace_existing=True,
    )
    return scheduler


def describe(scheduler: Any) -> list[dict[str, Any]]:
    """The job table, for a log line at boot and for a test to assert on."""
    return [
        {
            "id": job.id,
            "name": job.name,
            "trigger": str(job.trigger),
            "max_instances": job.max_instances,
            "coalesce": job.coalesce,
            "misfire_grace_time": job.misfire_grace_time,
        }
        for job in scheduler.get_jobs()
    ]


def main() -> int:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
    )
    cap_thread_counts()

    scheduler = build_scheduler()
    for job in describe(scheduler):
        logger.info("scheduled %s", job)
    logger.info(
        "PropIQ worker up at %s (%s). RESEARCH_ONLY — nothing here places or "
        "sizes a wager.",
        datetime.now(DISPLAY_TZ).isoformat(), DISPLAY_TZ_NAME,
    )

    # A container that starts mid-afternoon would otherwise sit idle until the
    # next morning, which on a first deploy looks exactly like a broken worker.
    # Off by default: a redeploy loop would otherwise run the slate repeatedly.
    if (os.environ.get(ENV_RUN_ON_START) or "").strip().lower() in {"1", "true", "yes"}:
        logger.info("%s set — running one slate now.", ENV_RUN_ON_START)
        run_slate()

    def _stop(signum, _frame):
        logger.info("Signal %s — shutting down after the running job.", signum)
        scheduler.shutdown(wait=True)

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Worker stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
