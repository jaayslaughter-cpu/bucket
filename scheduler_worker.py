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
    return int(code)


def run_settlement() -> dict[str, Any]:
    """Grade every PENDING prop whose game has finished.

    The other half of the feedback loop: ``settlement.recorder`` writes the
    predictions, this grades them. Without it the ledger fills with PENDING rows
    and the calibration gate never has evidence to pass.
    """
    try:
        from src.settlement.runner import settle_pending_props

        report = settle_pending_props()
        summary = report.as_dict() if hasattr(report, "as_dict") else {"report": report}
        logger.info("Settlement: %s", summary)
        return summary
    except Exception as exc:  # noqa: BLE001
        logger.exception("Settlement raised; the scheduler stays up.")
        return {"status": "FAILED", "error": str(exc)}


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
