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

THREE JOBS, AND ONE OF THEM IS TIP-ANCHORED. The two daily cron entries are
still fixed Pacific clocks -- deliberately early enough to precede any tip and
late enough to follow any finish -- because the slate build and the settlement
sweep are whole-day operations and nothing is gained by moving them.

What used to be missing is the third kind, and this docstring used to explain
its absence: "it does not re-anchor itself to the day's first tip-off. That
would need a schedule feed, every data host is denied from the environment this
was written in". The feed is reachable now -- ``espn_schedule`` parses
``tipoff_utc`` and the forward slate already depends on it -- so
``schedule_prelock_jobs`` arms ONE one-shot job per game at that game's own
tip-off minus 35 minutes, and ``run_prelock`` re-checks availability for it.

Per GAME rather than per slate, because tip-offs are staggered across an
evening: a single slate-wide re-check is early for the late games and late for
the early ones. And it checks availability ONLY -- it refits nothing and
re-prices nothing, since the one thing that both changes in the last hour and
matters is who is dressing.

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
from datetime import datetime, timezone
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
ENV_RESULTS_CARD = "PROPIQ_RESULTS_CARD"
ENV_PRELOCK = "PROPIQ_PRELOCK"
ENV_PRELOCK_LEAD = "PROPIQ_PRELOCK_LEAD_MINUTES"

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

# MINUTES BEFORE TIP for the availability re-check. 35 is the middle of the
# 30-45 minute window: late enough that ESPN's injury report has settled, early
# enough that a withdrawal is still actionable. Bounded 5..180 -- below 5 the
# fetch and the post may not finish before tip, and past 180 it is not a
# pre-lock check, it is a second morning run.
DEFAULT_PRELOCK_LEAD_MINUTES = 35
PRELOCK_LEAD_MIN, PRELOCK_LEAD_MAX = 5, 180

# A pre-lock check that missed its slot by more than this is useless: the game
# has tipped and the lineup is public. Deliberately SHORTER than the lead time
# so a late fire cannot post after tip.
PRELOCK_MISFIRE_GRACE_SECONDS = 10 * 60

#: Set in main() so the daily planner can add one-shot jobs to the scheduler
#: that is actually running. Tests pass a scheduler explicitly instead.
_SCHEDULER: Any | None = None


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


def check_state_dir(path: str | None = None) -> dict[str, Any]:
    """
    Probe the directory the durable state is written to, at boot.

    WHY THIS IS A SEPARATE CHECK. A Railway volume mounted at /app/data REPLACES
    the directory the Dockerfile created and chowned, with whatever the platform
    provisions — commonly root-owned. This container runs as uid 10001, so the
    first write fails with EACCES, and every writer here catches broadly: the
    failure would surface as "calibration report failed" every night, which
    names the symptom and not the cause.

    Probes by actually creating and deleting a file rather than reading the mode
    bits, because the mode is not the whole answer — an ownership mismatch, a
    read-only mount and a full disk all present differently and all matter.

    Warns, never raises. A worker that will not start is worse than one whose
    first log line says the volume is not writable: the slate still runs and
    still writes to Postgres, which is where the projections go.
    """
    from pathlib import Path

    target = Path(
        path or os.environ.get(ENV_CALIBRATION_REPORT) or DEFAULT_CALIBRATION_REPORT
    ).parent
    probe = target / ".propiq_write_probe"
    try:
        target.mkdir(parents=True, exist_ok=True)
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        logger.error(
            "STATE DIRECTORY %s IS NOT WRITABLE (%s). The calibration report and "
            "the model artifacts live here, so the publication gate will withhold "
            "every card and score_prob_over will find no model. If this is a "
            "mounted volume, it is most likely owned by root while this process "
            "runs as uid %s — chown it to that uid, or run the service as root. "
            "The slate still runs and still writes to Postgres.",
            target, exc, os.getuid() if hasattr(os, "getuid") else "?",
        )
        return {"path": str(target), "writable": False, "error": str(exc)}
    logger.info("State directory %s is writable.", target)
    return {"path": str(target), "writable": True}


#: Below this many training rows an artifact is almost certainly the synthetic
#: demo one. The demo artifacts in this repository were fit on 968 rows; the
#: real panel is 214,381. A deployed worker scoring a live slate with a model
#: fit on a thousand synthetic rows is the worst outcome in this file, because
#: it produces confident-looking numbers rather than abstentions.
MIN_PLAUSIBLE_TRAIN_ROWS = 5_000


def check_parlay_ledger() -> dict[str, Any]:
    """
    Report WHICH parlay ledger this process will write to, at boot.

    WHY A COMMENT WAS NOT ENOUGH. The Dockerfile sets
    ``PROPIQ_PARLAY_LEDGER=postgres`` with a comment saying why, and
    ``open_parlay_log``'s docstring says "SET IT TO postgres ON A CONTAINER".
    Neither is a check. A deploy that does not build from that Dockerfile — a
    buildpack, a plain ``python scheduler_worker.py`` on a VM, or the variable
    overridden in a platform dashboard — gets the CSV ledger on an ephemeral
    filesystem, and every ticket's at-bet-time probability and EV is destroyed
    by the next redeploy. The trace left behind is nothing at all: the CSV
    writes succeed.

    A ticket is the only record of what was predicted BEFORE a game, which is
    the one thing that cannot be recomputed afterwards. So the choice is said
    out loud next to ``check_model_artifact`` — the other thing a redeploy
    breaks silently.

    THE LEVELS ARE THE POINT.

      * postgres, database configured  -> INFO. The deployed shape.
      * postgres, NO database          -> ERROR. The variable is set and the
                                          first write will fail; better said
                                          now than at the first ticket.
      * csv, database configured       -> ERROR. Almost certainly a deployment
                                          that will lose its ledger. Saying
                                          this at WARNING would be read as
                                          acceptable, and it is not.
      * csv, no database               -> INFO. A local run with no Postgres
                                          is legitimately csv, and naming the
                                          consequence is all that is owed.

    Warns, never raises, for the reason ``check_model_artifact`` gives: a
    worker that refuses to start cannot report anything.
    """
    try:
        from src.quant.parlay_log import ENV_LEDGER_BACKEND, resolve_ledger_choice
    except Exception as exc:  # noqa: BLE001 — the probe must not take the boot down
        logger.error("Could not import the ledger resolver (%s).", exc)
        return {"ledger": None, "error": str(exc)}

    try:
        # The SAME resolver open_parlay_log uses, so this cannot report a
        # backend other than the one that will be opened.
        choice = resolve_ledger_choice()
    except Exception as exc:  # noqa: BLE001 — a typo here must not stop settlement
        logger.error(
            "%s is not a usable ledger backend (%s). Every parlay write will "
            "raise until it is corrected.", ENV_LEDGER_BACKEND, exc,
        )
        return {"ledger": None, "error": str(exc)}

    database_configured = True
    try:
        from src.db.session import get_database_url

        get_database_url()
    except Exception:  # noqa: BLE001 — absence is the signal, not an error here
        database_configured = False

    out = {
        "ledger": choice,
        "database_configured": database_configured,
        "env_var": ENV_LEDGER_BACKEND,
    }

    if choice == "postgres" and database_configured:
        logger.info(
            "Parlay ledger: postgres (%s). Tickets survive a redeploy.",
            ENV_LEDGER_BACKEND,
        )
        out["status"] = "OK"
    elif choice == "postgres":
        logger.error(
            "Parlay ledger is set to postgres (%s) but no database is "
            "configured, so the first parlay write will fail. Set DATABASE_URL "
            "or switch the ledger to csv deliberately.", ENV_LEDGER_BACKEND,
        )
        out["status"] = "UNUSABLE"
    elif database_configured:
        logger.error(
            "Parlay ledger is CSV (%s unset or 'csv') while a database IS "
            "configured. On a container that filesystem is ephemeral and "
            "data/** is gitignored, so every ticket's at-bet-time probability "
            "and EV is destroyed by the next redeploy — the one record that "
            "cannot be recomputed. Set %s=postgres.",
            ENV_LEDGER_BACKEND, ENV_LEDGER_BACKEND,
        )
        out["status"] = "EPHEMERAL"
    else:
        logger.info(
            "Parlay ledger: csv, and no database is configured — a local run. "
            "Tickets live under data/external/parlay_log and will not survive "
            "a container redeploy; set %s=postgres where they must.",
            ENV_LEDGER_BACKEND,
        )
        out["status"] = "LOCAL"
    return out


def check_model_artifact() -> dict[str, Any]:
    """
    Resolve the scoring artifact AT BOOT and say what was found.

    WHY THIS IS A BOOT CHECK AND NOT LEFT TO INFERENCE. Without an artifact
    the pipeline does not fail: ``score_prob_over`` returns an all-null Series
    with a reason, every row abstains, the slate job exits 0, and the only
    trace is one INFO line in the middle of a run. A fresh container with an
    unseeded volume therefore ingests the slate, writes projections with no
    probabilities, dispatches nothing useful and looks healthy. That is the
    single most expensive failure mode this worker has, and it is invisible
    until somebody asks why the board has been empty for a week.

    So the question is asked at boot, in the first lines of the log, next to
    ``check_state_dir`` -- the two things a redeploy breaks.

    THREE THINGS ARE REPORTED, not one:

      * nothing resolved      -> ERROR, with every path the resolver tried.
      * resolved but absent   -> ERROR: $PROPIQ_MODEL pointing at a path the
                                 volume does not have is the likeliest
                                 misconfiguration, and it is silent otherwise.
      * resolved and present  -> INFO with the sidecar's own account of itself,
                                 and an ERROR -- not a warning -- when
                                 ``train_row_count`` is small enough that this
                                 is the demo artifact. Scoring a live slate
                                 with a model fit on a thousand synthetic rows
                                 is worse than abstaining, so it is logged at
                                 the level that says so.

    Warns, never raises, for the same reason ``check_state_dir`` does: a worker
    that refuses to start cannot report anything, and the settlement job is
    still useful with no model. The slate still runs; it just abstains, and now
    it says so before it starts rather than after.
    """
    import json
    from pathlib import Path

    try:
        # ENV_MODEL comes from main too, so the name this reports is the name
        # the resolver actually reads rather than a second copy of the string.
        from main import ENV_MODEL, resolve_model_artifact
    except Exception as exc:  # noqa: BLE001 — the probe must not take the boot down
        logger.error("Could not import the model resolver (%s).", exc)
        return {"resolved": False, "error": str(exc)}

    # The RESOLUTION is guarded too, not just the import. An earlier version
    # wrapped only the import, so a resolver that raised -- an unreadable
    # config, a permission error on the artifacts directory -- took the boot
    # down, which contradicts this function's whole policy. Caught by
    # tests/test_scheduler_worker.py::test_the_preflight_never_takes_the_boot_down.
    try:
        path, how = resolve_model_artifact()
    except Exception as exc:  # noqa: BLE001 — the probe must not take the boot down
        logger.error(
            "The model resolver raised (%s), so whether an artifact exists is "
            "UNKNOWN. The slate will abstain if there is none. Treat this as a "
            "missing artifact until the cause is fixed.", exc,
        )
        return {"resolved": False, "error": str(exc)}

    if path is None:
        logger.error(
            "NO SCORING ARTIFACT RESOLVED. %s Every row will ABSTAIN: the slate "
            "job will ingest, build features and write projections with NO "
            "probability, exit 0 and look healthy. Train one and put it where "
            "the resolver looks, or point $%s at it on the mounted volume.",
            how, ENV_MODEL,
        )
        return {"resolved": False, "reason": how}

    target = Path(path)
    if not target.exists():
        logger.error(
            "SCORING ARTIFACT %s DOES NOT EXIST (resolved via %s). Every row "
            "will abstain. If this came from $%s, the path is set but the file "
            "is not on the volume -- which is the likeliest first-deploy "
            "misconfiguration and is otherwise silent.",
            target, how, ENV_MODEL,
        )
        return {"resolved": True, "path": str(target), "exists": False, "how": how}

    meta_path = target.with_suffix(".meta.json")
    info: dict[str, Any] = {
        "resolved": True, "path": str(target), "exists": True, "how": how,
    }
    if not meta_path.exists():
        logger.error(
            "ARTIFACT %s HAS NO .meta.json SIDECAR, so the feature_cols used at "
            "training time are unknown and scoring refuses it. Every row will "
            "abstain.", target,
        )
        info["sidecar"] = False
        return info

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.error("Could not read the sidecar %s (%s).", meta_path, exc)
        info["sidecar"] = False
        info["error"] = str(exc)
        return info

    rows = meta.get("train_row_count")
    info.update(
        sidecar=True,
        market=meta.get("target_market"),
        train_row_count=rows,
        train_start_date=meta.get("train_start_date"),
        train_end_date=meta.get("train_end_date"),
        feature_count=len(meta.get("feature_cols") or []),
        feature_schema_version=meta.get("feature_schema_version"),
        saved_at_utc=meta.get("saved_at_utc"),
    )
    logger.info(
        "Scoring artifact: %s (via %s) | market=%s | %s feature(s) | trained on "
        "%s row(s) from %s to %s | schema=%s | saved %s",
        target.name, how, info["market"], info["feature_count"], rows,
        info["train_start_date"], info["train_end_date"],
        info["feature_schema_version"], info["saved_at_utc"],
    )

    # ONE MARKET PER ARTIFACT. score_prob_over takes a single model_path, so
    # whichever market this was fit for is the only one that can carry a
    # probability this run. Said here because the resolver's glob picks the
    # NEWEST artifact, which is not necessarily the market anyone intended.
    if info["market"]:
        logger.info(
            "This artifact is fit for %s only; every other market abstains this "
            "run, and the recorder skips a row with no probability.",
            info["market"],
        )

    if isinstance(rows, int) and rows < MIN_PLAUSIBLE_TRAIN_ROWS:
        logger.error(
            "ARTIFACT %s WAS TRAINED ON ONLY %d ROWS, below the %d-row floor "
            "that separates a real fit from this repository's synthetic demo "
            "artifacts (fit on 968). If this is the demo model it will produce "
            "confident-looking probabilities for a live slate, which is worse "
            "than abstaining. Verify what is on the volume.",
            target.name, rows, MIN_PLAUSIBLE_TRAIN_ROWS,
        )
        info["implausible_train_rows"] = True

    return info


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


def board_window(today: Any | None = None) -> dict[str, Any]:
    """
    The train/validation window the board is built over, as of TODAY.

    THIS USED TO BE TWO HARDCODED 2025 DATES and that was the other half of the
    most misleading defect in the project. ``run_board`` fell back to
    ``train_end="2025-01-15"`` / ``validation_end="2025-02-15"`` whenever the
    env vars were unset, and ``compare_models_on_panel`` scores the window
    ``(train_end, validation_end]`` -- so every board row described a game in
    early February 2025, the board stamped each one with today's slate date,
    and the window receded one day further into the past on every run. A
    deployed worker would have published the same stale fortnight forever,
    getting less relevant daily, and nothing reported it.

    THE CORRECT WINDOW FOR A LIVE SLATE is anchored on the Pacific calendar
    day, because that is what a slate is (``utils.timezones``):

        validation_end = today      the rows to score. forward_slate writes
                                    one row per (player, scheduled game) for
                                    today, so today IS the slate.
        train_end      = yesterday  fit on everything already played. The
                                    split is `<= train_end` to fit and
                                    `(train_end, validation_end]` to score, so
                                    yesterday/today scores exactly today.

    AN EXPLICIT OVERRIDE STILL WINS, because a backtest board is a legitimate
    thing to build and that is what the env vars are for. What it no longer
    does is happen by accident: an override whose validation_end is in the past
    is reported as a backtest, with the lag named, so a card built from one
    cannot be mistaken for tonight.
    """
    from datetime import timedelta

    from src.utils.timezones import pacific_calendar_date

    anchor = today or pacific_calendar_date()
    default_validation = str(anchor)
    default_train = str(anchor - timedelta(days=1))

    train_end = (os.environ.get(ENV_BOARD_TRAIN_END) or "").strip() or default_train
    validation_end = (
        (os.environ.get(ENV_BOARD_VALIDATION_END) or "").strip() or default_validation
    )
    overridden = sorted(
        name for name, value in (
            (ENV_BOARD_TRAIN_END, os.environ.get(ENV_BOARD_TRAIN_END)),
            (ENV_BOARD_VALIDATION_END, os.environ.get(ENV_BOARD_VALIDATION_END)),
        ) if (value or "").strip()
    )

    lag_days: int | None = None
    try:
        from datetime import date as _date

        parsed = _date.fromisoformat(validation_end)
        lag_days = (anchor - parsed).days
    except (TypeError, ValueError):
        logger.warning(
            "%s=%r is not an ISO date; it is passed through unchanged and "
            "compare_models_on_panel will decide what to do with it.",
            ENV_BOARD_VALIDATION_END, validation_end,
        )

    is_backtest = bool(lag_days is not None and lag_days > 0)
    if is_backtest:
        logger.warning(
            "BOARD IS A BACKTEST, NOT TONIGHT'S SLATE: validation_end=%s is %d "
            "day(s) before today (%s), so every row describes a game already "
            "played. Each row carries its own game_date and the dispatch embed "
            "says so, but nothing here can make a past window current. Unset %s "
            "to score today instead.",
            validation_end, lag_days, anchor,
            " and ".join(overridden) or ENV_BOARD_VALIDATION_END,
        )
    else:
        logger.info(
            "Board window: fit <= %s, score (%s, %s] -- anchored on the Pacific "
            "calendar day%s.",
            train_end, train_end, validation_end,
            f" (overridden by {' and '.join(overridden)})" if overridden else "",
        )

    return {
        "train_end": train_end,
        "validation_end": validation_end,
        "anchor_date": str(anchor),
        "overridden": overridden,
        "lag_days": lag_days,
        "is_backtest": is_backtest,
    }


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
        window = board_window()
        train_end = window["train_end"]
        validation_end = window["validation_end"]
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
        # Arm tonight's tip-anchored re-checks now that a board exists for them
        # to correct. Before the board there is nothing to withdraw.
        #
        # GUARDED SEPARATELY from the board build it sits inside. run_board's
        # own except would catch a failure here and return
        # {"status": "FAILED"} for the BOARD -- reporting a successful board as
        # failed because an unrelated arming step did, which is the kind of
        # misattributed status that sends somebody looking in the wrong place.
        # The board is the durable output; the re-checks are an addition to it.
        try:
            schedule_prelock_jobs()
        except Exception as exc:  # noqa: BLE001 — must not fail the board
            logger.exception(
                "Could not arm tonight's pre-lock re-checks (%s). The board "
                "above STANDS and was written; what is lost is the tip-off "
                "availability re-check, so late scratches will not be "
                "withdrawn tonight.", exc,
            )
        # The window travels with the result so a caller -- and the log line
        # above -- can see WHICH days were scored, not just how many rows came
        # back. A board of 40 rows is the same shape whether it describes
        # tonight or a fortnight in 2025.
        return {"status": "OK", **result.as_dict(), "window": window}
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


def prelock_lead_minutes() -> int:
    """Minutes before tip for the availability re-check, bounded and logged."""
    return _int_env(
        ENV_PRELOCK_LEAD, DEFAULT_PRELOCK_LEAD_MINUTES,
        low=PRELOCK_LEAD_MIN, high=PRELOCK_LEAD_MAX,
    )


def schedule_prelock_jobs(
    scheduler: Any | None = None,
    *,
    slate: Any | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """
    Arm one availability re-check per game, each at that game's own tip-off.

    THIS IS THE JOB THE WORKER DID NOT HAVE. ``build_scheduler`` registers two
    CRON entries, and this module's own opening docstring says why that is not
    enough: "a cron expression is fixed to a clock and an NBA slate is not:
    tip-offs move by hours". It then shipped two fixed clocks anyway, because
    the schedule feed was out of reach when it was written. It is in reach now
    -- ``espn_schedule`` parses ``tipoff_utc`` and the forward slate already
    depends on it -- so the tip-anchored half can exist.

    ONE JOB PER GAME, NOT ONE PER SLATE. Tip-offs are staggered across an
    evening; a single slate-wide re-check is early for the late games and late
    for the early ones. Each job is a one-shot ``date`` trigger at
    ``tipoff - lead``, so every game is checked on its own clock.

    WHAT IS REFUSED, each for a stated reason rather than silently:

      * a game with no ``tipoff_utc``  -> cannot be anchored, so no job. The
        alternative is guessing a tip time, and a check that fires at the
        wrong moment is worse than none.
      * a game that is not PREGAME     -> it has tipped or finished; the
        lineup is already public.
      * a slot already in the past     -> firing immediately for a game that
        tips in ten minutes would post a "pre-lock" card after lock. Reported
        as skipped WITH the slot, because on a mid-day redeploy this is the
        normal and correct outcome for the early games.

    IDEMPOTENT. The job id is derived from the Pacific slate date and the ESPN
    event id, with ``replace_existing=True``, so arming at boot and again
    after the slate job cannot double-post.
    """
    target = scheduler if scheduler is not None else _SCHEDULER
    if target is None:
        logger.warning(
            "No scheduler to arm pre-lock jobs on. They are added by main() at "
            "boot and after each slate; a direct call needs one passed in."
        )
        return {"status": "DATA_NOT_AVAILABLE", "reason": "no scheduler"}

    if not _flag(ENV_PRELOCK, True):
        logger.info("Pre-lock checks off (%s=false). No tip-anchored jobs.", ENV_PRELOCK)
        return {"status": "SKIPPED", "reason": "pre-lock disabled", "armed": 0}

    lead = prelock_lead_minutes()
    moment = now or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)

    try:
        if slate is None:
            from src.ingestion.espn_schedule import load_slate

            slate = load_slate()
    except Exception as exc:  # noqa: BLE001 — a denied feed is a state, not a crash
        logger.warning(
            "Could not read today's schedule (%s), so NO pre-lock check is armed. "
            "The morning board still stands; it simply will not be re-checked "
            "against late scratches tonight.", exc,
        )
        return {"status": "DATA_NOT_AVAILABLE", "reason": str(exc), "armed": 0}

    from datetime import timedelta

    games = list(getattr(slate, "pregame_only", None) or [])
    armed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for game in games:
        event_id = str(getattr(game, "espn_event_id", "") or "")
        home = getattr(game, "home_team", None)
        away = getattr(game, "away_team", None)
        label = " @ ".join([p for p in (away, home) if p]) or event_id or "?"
        tipoff = getattr(game, "tipoff_utc", None)

        if tipoff is None:
            skipped.append({"game": label, "reason": "no tip-off time"})
            continue
        if tipoff.tzinfo is None:
            tipoff = tipoff.replace(tzinfo=timezone.utc)

        run_at = tipoff - timedelta(minutes=lead)
        if run_at <= moment:
            skipped.append({
                "game": label,
                "reason": f"slot {run_at.isoformat()} already passed",
            })
            continue

        slate_day = getattr(game, "slate_date_pt", None) or ""
        job_id = f"prelock:{slate_day}:{event_id or label}"
        target.add_job(
            run_prelock,
            trigger="date",
            run_date=run_at,
            kwargs={
                "event_id": event_id,
                "home": home,
                "away": away,
                "tipoff_iso": tipoff.isoformat(),
            },
            id=job_id,
            name=f"Pre-lock availability re-check — {label}",
            max_instances=1,
            coalesce=True,
            misfire_grace_time=PRELOCK_MISFIRE_GRACE_SECONDS,
            replace_existing=True,
        )
        armed.append({
            "game": label, "run_at": run_at.isoformat(), "tipoff": tipoff.isoformat(),
        })

    logger.info(
        "Pre-lock: armed %d of %d pre-tip game(s) at tip-off minus %d minutes. "
        "Skipped %d: %s",
        len(armed), len(games), lead, len(skipped),
        [s["reason"] for s in skipped[:4]] or "none",
    )
    return {
        "status": "OK", "armed": len(armed), "lead_minutes": lead,
        "jobs": armed, "skipped": skipped,
    }


def run_prelock(
    *,
    event_id: str,
    home: str | None = None,
    away: str | None = None,
    tipoff_iso: str | None = None,
) -> dict[str, Any]:
    """
    Re-check availability for ONE game 30-45 minutes before its tip-off.

    WHY THIS JOB EXISTS. The board is built at 09:00 PT. Tip-offs are six to
    twelve hours later, and the thing that changes in between is who is
    dressing: ESPN's injury report settles in the last hour, and a
    recommendation on a player who will not dress is not a weak
    recommendation, it is a row that should not exist.
    ``src/pipeline/scratches.py`` was written for exactly this and its
    docstring names the gap -- "Nothing dropped a projection when a player was
    ruled out AFTER that projection was written" -- but nothing ever called it
    on a schedule. This is the caller.

    WHAT IT DELIBERATELY DOES NOT DO: refit a model, rebuild the board, or
    re-price anything. Refitting 35 minutes before tip would be absurd, and
    re-pricing needs a live odds feed this project does not have (the board's
    own source precedence lists one feed and this environment cannot reach
    it). The one thing that both changes in this window and matters is
    availability, so that is the only thing checked, and saying so is better
    than implying a full re-evaluation.

    SCOPED TO ONE GAME on purpose. Tip-offs are staggered across an evening,
    so a single slate-wide re-check would be too early for the late games and
    too late for the early ones. One job per game, anchored to that game's own
    tip, is the only version of this that is on time for every game.

    AN UNUSABLE FEED IS REPORTED, NOT TREATED AS A CLEAN SLATE. That is the
    rule the scratch filter turns on and the one an earlier version of
    ``projected_available`` got backwards. A failed check posts a card saying
    the check failed; it does not post silence, because silence reads as
    "nothing changed".

    Never raises. Returns a status dict, like every other job here.
    """
    game = " @ ".join([p for p in (away, home) if p]) or str(event_id)
    if not _flag(ENV_PRELOCK, True):
        logger.info("Pre-lock check off (%s). Nothing done for %s.", ENV_PRELOCK, game)
        return {"status": "SKIPPED", "reason": "pre-lock disabled", "game": game}

    from src.notify.discord import DiscordDispatchError

    try:
        from pathlib import Path

        import pandas as pd

        from src.notify.discord import (
            DiscordConfig,
            build_prelock_correction_embed,
            send_embeds,
        )
        from src.pipeline.scratches import (
            AVAILABILITY_COLUMN,
            DETAIL_COLUMN,
            UNKNOWN,
            WITHHELD,
            apply_scratch_filter,
        )

        board_csv = Path(os.environ.get(ENV_BOARD_CSV) or DEFAULT_BOARD_CSV)
        if not board_csv.exists():
            logger.info(
                "Pre-lock %s: no board at %s, so there is nothing to withdraw.",
                game, board_csv,
            )
            return {"status": "SKIPPED", "reason": "no board", "game": game}

        frame = pd.read_csv(board_csv)
        if frame.empty:
            return {"status": "SKIPPED", "reason": "empty board", "game": game}

        # ONLY THIS GAME'S ROWS, and only the ones that were RECOMMENDED. A
        # row already abstained needs no withdrawal, and the other games have
        # their own jobs at their own tip-offs.
        teams = {t for t in (home, away) if t}
        mine = frame
        if teams and "player_team" in frame.columns:
            in_game = frame["player_team"].isin(teams)
            if "opponent" in frame.columns:
                in_game = in_game | frame["opponent"].isin(teams)
            mine = frame[in_game]
        if "decision_status" in mine.columns:
            mine = mine[mine["decision_status"] == "RECOMMENDED"]
        if mine.empty:
            logger.info(
                "Pre-lock %s: no recommended row on the board for this game.", game
            )
            return {"status": "SKIPPED", "reason": "no recommended rows", "game": game}

        work = mine.copy()
        # apply_scratch_filter reads PLAYER_NAME; the board CSV is lowercase.
        work["PLAYER_NAME"] = work.get("player_name")
        result = apply_scratch_filter(work)

        annotated = result.projections
        rows = [
            type("Row", (), {
                **{k: (None if pd.isna(v) else v) for k, v in r.items()},
                "availability_detail": (
                    None if pd.isna(r.get(DETAIL_COLUMN)) else r.get(DETAIL_COLUMN)
                ),
            })()
            for r in annotated.to_dict("records")
        ]
        withheld = [
            r for r in rows if getattr(r, AVAILABILITY_COLUMN, None) == WITHHELD
        ]
        unknown = [
            r for r in rows if getattr(r, AVAILABILITY_COLUMN, None) == UNKNOWN
        ]

        tipoff_display = _tipoff_display(tipoff_iso)
        logger.info(
            "Pre-lock %s (tip %s): %d recommended row(s) checked, %d withdrawn, "
            "%d unknown, feed verified=%s.",
            game, tipoff_display or "unknown", len(rows), len(withheld),
            len(unknown), result.verified,
        )

        # NOTHING TO SAY IS STILL SAID, unless abstentions are muted: a clear
        # check is the evidence that the check ran, and on a night with a
        # withdrawal the absence of a card on the other games is what tells a
        # reader those were checked too.
        nothing_changed = result.verified and not withheld and not unknown
        if nothing_changed and not _flag(ENV_DISPATCH_ABSTENTIONS, True):
            return {
                "status": "OK", "game": game, "checked": len(rows),
                "withheld": 0, "unknown": 0, "sent": False,
            }

        webhook_configured = bool((os.environ.get("DISCORD_WEBHOOK_URL") or "").strip())
        if not _flag(ENV_DISPATCH, webhook_configured):
            logger.info("Dispatch off — pre-lock result for %s not sent.", game)
            return {
                "status": "OK", "game": game, "checked": len(rows),
                "withheld": len(withheld), "unknown": len(unknown), "sent": False,
            }

        embed = build_prelock_correction_embed(
            game=game,
            tipoff_display=tipoff_display,
            withheld=withheld,
            unknown=unknown,
            verified=result.verified,
            reason=result.reason,
        )
        sent = send_embeds([embed], config=DiscordConfig(dry_run=False))
        return {
            "status": sent.status, "game": game, "checked": len(rows),
            "withheld": len(withheld), "unknown": len(unknown), "sent": True,
        }
    except DiscordDispatchError as exc:
        logger.error("Pre-lock dispatch refused for %s: %s", game, exc)
        return {"status": "REFUSED", "error": str(exc), "game": game}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Pre-lock check raised for %s; the scheduler stays up.", game)
        return {"status": "FAILED", "error": str(exc), "game": game}


def _tipoff_display(tipoff_iso: str | None) -> str | None:
    """Tip-off as a Pacific wall-clock string, or None if it cannot be parsed."""
    if not tipoff_iso:
        return None
    try:
        from src.utils.timezones import to_pacific

        return to_pacific(datetime.fromisoformat(tipoff_iso)).strftime(
            "%-I:%M %p %Z"
        )
    except (TypeError, ValueError) as exc:
        logger.debug("Could not render tip-off %r: %s", tipoff_iso, exc)
        return None


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
    out["results_card"] = run_results_card()
    return out


def run_results_card(slate_date: str | None = None) -> dict[str, Any]:
    """
    Post the day's settled record to Discord.

    WHY THIS IS NOT BEHIND THE CALIBRATION GATE, which every other dispatch
    surface in this project is. That gate exists to stop an UNCALIBRATED
    MODEL PROBABILITY reaching a person. This card carries no probability and
    makes no forecast: it reports props that already reached a final box score
    and how they graded. Withholding settled history for want of calibration
    evidence would withhold the very thing the evidence is built from.

    What the card refuses instead is in ``build_win_loss_embed``: a strike rate
    under the 30-prop minimum is withheld with its reason rather than printed,
    ROI is reported only when a stake was actually recorded, and the CLV note
    travels with the CLV figure.

    Off unless a webhook is configured, like dispatch. A failure is logged and
    returned: the grading is the durable part and a missing card costs nothing.
    """
    webhook_configured = bool((os.environ.get("DISCORD_WEBHOOK_URL") or "").strip())
    if not _flag(ENV_RESULTS_CARD, _flag(ENV_DISPATCH, webhook_configured)):
        logger.info("Results card off. Nothing sent.")
        return {"status": "SKIPPED", "reason": "results card disabled"}

    from src.notify.discord import DiscordDispatchError

    try:
        from datetime import date as _date
        from datetime import timedelta

        from src.notify.discord import (
            DiscordConfig,
            build_win_loss_embed,
            send_embeds,
        )
        from src.settlement.metrics import MIN_SAMPLE_FOR_RATE, get_performance_summary
        from src.utils.timezones import pacific_calendar_date

        # The settlement job runs at 03:30 PT and grades games that finished
        # the previous Pacific calendar day, so that is the day to report. A
        # card dated today would be empty every morning by construction.
        day = slate_date or str(pacific_calendar_date() - timedelta(days=1))
        target = _date.fromisoformat(day)
        summary = get_performance_summary(start_date=target, end_date=target)
        graded = int(getattr(summary.record, "graded_n", 0) or 0)
        if graded == 0 and not _flag(ENV_DISPATCH_ABSTENTIONS, True):
            logger.info("Nothing graded for %s and abstentions muted — nothing sent.", day)
            return {"status": "SKIPPED", "reason": "nothing graded", "slate_date": day}

        embed = build_win_loss_embed(
            summary, slate_date=day, min_sample_for_rate=MIN_SAMPLE_FOR_RATE
        )
        result = send_embeds([embed], config=DiscordConfig(dry_run=False))
        logger.info("Results card for %s: %s (%d graded)", day, result.status, graded)
        return {"status": result.status, "slate_date": day, "graded": graded}
    except DiscordDispatchError as exc:
        logger.error("Results card refused: %s", exc)
        return {"status": "REFUSED", "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Results card raised; the scheduler stays up.")
        return {"status": "FAILED", "error": str(exc)}


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
    check_state_dir()
    # Asked at boot, beside the state directory, because a redeploy breaks the
    # same two things: the volume's permissions and whether anything was ever
    # put on it. See check_model_artifact for why inference is too late.
    check_model_artifact()
    check_parlay_ledger()

    scheduler = build_scheduler()
    # Published so the daily planner can add one-shot jobs to the scheduler
    # that is actually running, rather than to one it was handed once.
    global _SCHEDULER
    _SCHEDULER = scheduler
    # ARMED AT BOOT, not only after the 09:00 slate. A container that starts at
    # 16:00 would otherwise leave every game tonight unchecked, which is
    # exactly when a redeploy is most likely and exactly when the morning board
    # is most stale.
    schedule_prelock_jobs(scheduler)
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
