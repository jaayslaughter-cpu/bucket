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
import re
import signal
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # the annotation only; importing it at runtime would make
    # this module pull in the quant layer at import time, which every deferred
    # import in this file exists to avoid.
    from src.quant.publication_gate import PublicationVerdict


logger = logging.getLogger("propiq.scheduler")

# Two module-level imports, both deliberate despite this file's deferred-import
# policy. `src.notify.discord` reads the webhook variable's NAME, which lives
# there and is never spelled again here; it imports only the standard library at
# module scope, so it pulls in none of the heavy layers those deferrals avoid.
# `src.utils.timezones` is Pacific, for the reason in the module docstring.
from src.notify.discord import ENV_WEBHOOK_URL, webhook_configured  # noqa: E402
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

#: Relative to the DURABLE STATE ROOT, not the working directory. The old value
#: was a bare "outputs/calibration.json", which on Railway is ephemeral: the
#: settlement job writes the report at 03:30 PT, the slate reads it at 09:00 PT,
#: and a redeploy in between left the publication gate with no evidence. Every
#: card was then withheld for "no calibration evidence" -- which is true, and is
#: NOT the real reason, so the operator debugs the gate instead of the volume.
#: `railway_healthcheck` warned about this, and a warning an operator has to act
#: on is worse than a default that is simply right.
CALIBRATION_REPORT_RELPATH = "outputs/calibration.json"
DEFAULT_BOARD_CSV = "outputs/decision_board.csv"


def calibration_report_path(explicit: str | Path | None = None) -> Path:
    """
    The one answer to "where does the publication gate's evidence live".

    FOUR READERS HAD THEIR OWN COPY of ``env or default`` -- the boot probe, the
    dispatcher, the report writer and `railway_healthcheck` -- which is the
    defect this tree keeps finding: the copies agree until one of them is
    changed. The healthcheck checking a different path from the one settlement
    writes would make it certify the wrong file.

    Precedence: an explicit argument, then ``$PROPIQ_CALIBRATION_REPORT``, then
    the state root. An operator who sets the variable still wins.
    """
    if explicit:
        return Path(explicit)
    configured = (os.environ.get(ENV_CALIBRATION_REPORT) or "").strip()
    if configured:
        return Path(configured)
    from src.utils.volume import resolve_state_root

    return resolve_state_root()[0] / CALIBRATION_REPORT_RELPATH

# 09:00 PT: before any NBA tip (the earliest are late morning PT) and after the
# night's box scores have settled into the sources.
DEFAULT_SLATE_HOUR, DEFAULT_SLATE_MINUTE = 9, 0
# 03:30 PT: after even a West Coast overtime game has finished and been posted.
DEFAULT_SETTLE_HOUR, DEFAULT_SETTLE_MINUTE = 3, 30

# A job that misses its window by less than this still runs; past it, the slate
# has moved on and running late would project games that have already tipped.
#: The fallback grace for a job added WITHOUT its own. APScheduler's own
#: default is one second; fifteen minutes is long enough that a brief overrun
#: does not silently drop a job and short enough that nothing stale fires.
#: Every job registered here sets its own, so this only governs the next one.
DEFAULT_MISFIRE_GRACE_SECONDS = 15 * 60

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


#: Dispatch failures to Discord. Defaults to ON when a webhook is configured,
#: the same rule ENV_DISPATCH follows: a worker with no webhook should not
#: spend a boot trying to post.
ENV_ALERT_FAILURES = "PROPIQ_ALERT_FAILURES"
#: At-most-once per stage per Pacific day. Off, every boot of a redeploy loop
#: posts the same unseeded-volume alert again.
ENV_ALERT_DEDUPE = "PROPIQ_ALERT_DEDUPE"
#: Where the "already alerted today" marks live, under the durable state root.
ALERT_STATE_FILENAME = "alerts_sent.json"

#: Patterns scrubbed from an alert before it leaves the process. An alert body
#: is an EXCEPTION STRING going to an outbound channel, and the exceptions this
#: worker raises are database ones: SQLAlchemy puts the connection URL in its
#: message, and that URL carries the password. `notify.discord` already redacts
#: its own webhook from everything it logs; this is the same argument applied
#: to the other credential in the process.
_ALERT_SCRUBBERS: tuple[tuple[str, str], ...] = (
    # postgresql://user:password@host/db  ->  postgresql://user:***@host/db
    (r"(?i)\b([a-z0-9+]+://[^\s:/@]+):[^\s@/]+@", r"\1:***@"),
    # Any remaining key=value that names a secret.
    (r"(?i)\b([A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD|WEBHOOK|CREDENTIAL)"
     r"[A-Z0-9_]*)\s*=\s*\S+",
     r"\1=***"),
)


def scrub_alert(text: Any) -> str:
    """
    Remove credentials from a string about to be sent off the machine.

    THE ALERT BODY IS AN EXCEPTION STRING. `psycopg` and SQLAlchemy both put
    the connection URL into their error messages, and the URL carries the
    password -- so routing failures to Discord without this would publish
    `DATABASE_URL` to a chat channel the first time the database refused a
    connection. That is a worse outcome than the silence P2 set out to fix.

    Webhook redaction is delegated to `notify.discord.redact_webhook`, which
    already owns that pattern; duplicating it here would be a second copy of
    one rule.
    """
    out = str(text)
    for pattern, replacement in _ALERT_SCRUBBERS:
        out = re.sub(pattern, replacement, out)
    try:
        from src.notify.discord import redact_webhook

        out = redact_webhook(out)
    except Exception:  # noqa: BLE001 — scrubbing must not fail the alert
        pass
    return out


def _alert_state_path() -> Any:
    from src.utils.volume import resolve_state_root

    return resolve_state_root()[0] / ALERT_STATE_FILENAME


def _alert_already_sent_today(stage: str) -> bool:
    """
    Has this stage already alerted on today's Pacific date?

    FAILS OPEN -- an unreadable or missing store returns False, meaning SEND.
    The store lives on the mounted volume, and "the volume is not writable" is
    one of the failures this is meant to report: a dedupe that suppressed on
    its own failure would silence exactly the alert it could not record.
    A duplicate alert is a nuisance; a missing one is the problem P2 exists for.
    """
    if not _flag(ENV_ALERT_DEDUPE, True):
        return False
    try:
        import json

        from src.utils.timezones import pacific_calendar_date

        path = _alert_state_path()
        if not path.exists():
            return False
        marks = json.loads(path.read_text(encoding="utf-8"))
        return str(marks.get(stage) or "") == str(pacific_calendar_date())
    except Exception as exc:  # noqa: BLE001 — fail open, and say so
        logger.debug("Alert dedupe store unreadable (%s); sending anyway.", exc)
        return False


def _mark_alert_sent(stage: str) -> None:
    """Best effort. A mark that cannot be written costs a duplicate, not a miss."""
    try:
        import json

        from src.utils.timezones import pacific_calendar_date

        path = _alert_state_path()
        marks: dict[str, str] = {}
        if path.exists():
            try:
                marks = json.loads(path.read_text(encoding="utf-8")) or {}
            except Exception:  # noqa: BLE001 — a corrupt store is replaced
                marks = {}
        marks[stage] = str(pacific_calendar_date())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(marks, indent=2, sort_keys=True), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 — never fail a run over bookkeeping
        logger.debug("Could not record the alert mark for %s (%s).", stage, exc)


def notify_failure(stage: str, detail: str, *, dedupe: bool = True) -> dict[str, Any]:
    """
    Log a failure AND dispatch it to Discord. Never raises.

    WHY THIS EXISTS. Successes dispatched -- the board, the withdrawal card,
    the results card -- and failures did not. A boot ERROR, a FAILED slate, a
    FAILED settlement and a probe FAILURE were log-only, and in Railway's log
    view every line is already red (Python logging writes to stderr, which
    Railway classifies as error severity), so a real failure is
    indistinguishable from an INFO line by eye. The board would go empty for
    days with nobody told. `docs/autonomy_audit_2026-10-10.md` §5.2.

    IT NEVER RAISES AND IT NEVER RECURSES. A dispatch that fails must not turn
    one failure into two, so everything here is swallowed and reported in the
    return value -- and nothing in `run_dispatch`'s own failure path calls
    this, because alerting about a broken webhook through that webhook is a
    loop with no exit.

    THE BODY IS SCRUBBED. See `scrub_alert`: an exception string from this
    worker routinely contains the database URL, password included.
    """
    safe = scrub_alert(detail)
    logger.error("ALERT [%s] %s", stage, safe)

    if not _flag(ENV_ALERT_FAILURES, webhook_configured()):
        return {"status": "SKIPPED", "reason": "alerting off", "stage": stage}

    if dedupe and _alert_already_sent_today(stage):
        return {"status": "DEDUPED", "stage": stage}

    try:
        from src.notify.discord import DiscordConfig, send_embeds

        embed = {
            "title": f"PropIQ: {stage} FAILED",
            # Discord rejects an over-long embed; the first 1,500 characters of
            # an exception carry the cause, and the log has the rest.
            "description": safe[:1500],
            "color": 0xB00020,
            "footer": {
                "text": (
                    "RESEARCH_ONLY — this is a pipeline alert, not a bet "
                    "instruction. Nothing here places or sizes a wager."
                )
            },
        }
        result = send_embeds([embed], config=DiscordConfig(dry_run=False))
        status = getattr(result, "status", "UNKNOWN")
        if status == "SENT":
            _mark_alert_sent(stage)
        return {"status": status, "stage": stage}
    except Exception as exc:  # noqa: BLE001 — one failure must not become two
        logger.warning(
            "Could not dispatch the %s alert (%s). It is in the log above; "
            "that is all that is lost.", stage, scrub_alert(exc),
        )
        return {"status": "FAILED", "stage": stage, "error": scrub_alert(exc)}


def check_database() -> dict[str, Any]:
    """
    Does the database answer? Asked at BOOT, not at 09:00.

    FOUND IN PRODUCTION, 2026-10-11. The container booted at 03:22 PT reporting
    nothing wrong, with `DATABASE_URL` unset. Settlement raised at 03:30, the
    slate refused at 09:00, and the first signal arrived more than five hours
    after a boot that already had every fact needed to say so.

    The three checks beside this one ask about the volume, the artifact and the
    ledger. None asked whether the database answers. The ledger check came
    closest and still only catches it sideways -- it reports that "the first
    parlay write will fail", which names a consequence, and only when the
    ledger is explicitly set to postgres. With the ledger left at its csv
    default and no database configured, the worker boots entirely quietly and
    every job fails for a day.

    NEVER RAISES. A probe that raises at boot takes the worker down before the
    scheduler starts, which is strictly worse than the silence it replaces.
    `session.healthcheck` already returns `(ok, message)` rather than raising;
    the try here is for an import or driver failure underneath it.
    """
    try:
        from src.db.session import healthcheck

        ok, message = healthcheck()
    except Exception as exc:  # noqa: BLE001 — a boot probe may not kill the worker
        logger.error("Database boot check could not run: %s", exc)
        return {"ok": False, "message": f"the database check itself failed: {exc}"}

    out = {"ok": bool(ok), "message": str(message)}
    if ok:
        logger.info("Database reachable at boot.")
    else:
        logger.error(
            "DATABASE UNREACHABLE AT BOOT: %s Every database step will fail "
            "until this is fixed -- settlement at 03:30 PT and the slate at "
            "09:00 PT both need it.", message,
        )
    return out


def notify_boot_problems(
    state: Mapping[str, Any],
    artifact: Mapping[str, Any],
    ledger: Mapping[str, Any],
    database: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """
    Alert on whatever the three boot checks found, reading their return values.

    THE CHECKS STAY PURE. They already return structured dicts and their own
    policy is warn-and-carry-on; threading a dispatch callback through them
    would couple three probes to an outbound channel for no gain. This reads
    what they returned instead, which is also what makes it testable without
    a webhook.

    Deduped per stage per Pacific day, so a redeploy loop does not post the
    same unseeded-volume alert on every boot.
    """
    out: list[dict[str, Any]] = []
    # FIRST, because it is the cause of which the other three are symptoms: an
    # unreachable database makes the ledger unusable and every job fail, and an
    # operator reading one alert should get the one that names the fix.
    #
    # `database` is a REQUIRED argument rather than one defaulting to None. A
    # default would make "nobody passed the database result" indistinguishable
    # from "the database is fine", which is the shape of silent pass this file
    # keeps finding.
    if database.get("ok") is False:
        out.append(notify_failure(
            "boot:database",
            f"The database did not answer at boot: {database.get('message')} "
            f"Nothing that persists will work -- settlement at 03:30 PT and "
            f"the 09:00 PT slate both refuse without it, and the slate's "
            f"preflight exits rather than scoring anything.",
        ))
    if state.get("writable") is False:
        # EVERY unwritable directory, EACH WITH ITS OWN CAUSE. Two findings,
        # both cubic's, both on PR #8:
        #
        #   1. this named only `state["path"]`, so one of two broken
        #      directories got chowned and the next boot produced the same
        #      alert naming the other -- which reads as "the fix did not work",
        #      not "there was a second one";
        #   2. the fix for (1) listed every path under ONE generic cause,
        #      asserting the mounted-volume hypothesis and "the model artifacts
        #      live there" for a calibration report configured outside the state
        #      root -- contradicting `check_state_dir`'s own log line about that
        #      same directory, which says the artifacts live elsewhere.
        #
        # `check_state_dir` already worked out what lives in each directory;
        # this reads that rather than keeping a second copy of the reasoning
        # that can drift from it. A ``state`` dict with no ``failures`` -- a
        # caller predating the key, or a hand-built one -- gets the generic
        # text, which says only what is true of either directory.
        broken = state.get("failures") or [
            {
                "path": state.get("path"),
                "error": state.get("error"),
                "what": "State directory",
                "cause": (
                    "Durable state lives there. A mounted volume is commonly "
                    "root-owned while this process runs as uid 10001."
                ),
            }
        ]
        out.append(notify_failure(
            "boot:state-dir",
            " ".join(
                (
                    f"{f.get('what') or 'State directory'} {f.get('path')} is "
                    f"NOT writable: {f.get('error')}. {f.get('cause') or ''}"
                ).strip()
                for f in broken
            ),
        ))
    if not artifact.get("resolved") or artifact.get("exists") is False:
        out.append(notify_failure(
            "boot:model-artifact",
            f"No usable scoring artifact: {artifact.get('reason') or artifact.get('error') or artifact}. "
            f"EVERY ROW WILL ABSTAIN -- the slate still ingests, builds "
            f"features, writes projections with no probability and exits 0, so "
            f"nothing else will look wrong.",
        ))
    # `implausible_train_rows`, which is the key check_model_artifact actually
    # sets. The first version of this read `demo_suspected`, a key nothing
    # writes, so the branch was dead code that would have reported a healthy
    # deploy and stayed silent on the demo artifact.
    elif artifact.get("implausible_train_rows"):
        out.append(notify_failure(
            "boot:model-artifact-demo",
            f"The resolved artifact looks like the synthetic demo one: "
            f"{artifact}. Scoring a live slate with it is worse than abstaining.",
        ))
    if ledger.get("status") in {"UNUSABLE", "EPHEMERAL"}:
        out.append(notify_failure(
            "boot:parlay-ledger",
            f"Parlay ledger is {ledger.get('status')}: {ledger}. The ledger "
            f"holds each ticket's at-bet-time probability and EV, which is the "
            f"one thing that cannot be recomputed after the game.",
        ))
    return out


def newest_game_date_or_none() -> str | None:
    """
    The newest game date we hold, or None when it cannot be read.

    FAIL-SOFT ON PURPOSE. This feeds the training-recency half of the artifact
    freshness check, and a database hiccup must not be able to withhold a board
    that is otherwise fine. When this returns None that axis is simply NOT
    MEASURED -- `artifact_freshness` omits it from `checks` and says so -- while
    the artifact's own file age is still judged. Guessing a date here, or
    falling back to today, would turn an unreadable database into either a
    false pass or an alarm every offseason.
    """
    try:
        from src.db.repository import newest_team_game_date

        return newest_team_game_date()
    except Exception as exc:  # noqa: BLE001 — absence is the signal, not an error
        logger.warning(
            "Could not read the newest game date (%s), so training recency was "
            "not measured against it. The artifact's own age still was.", exc,
        )
        return None


def artifact_freshness_verdict(
    markets: list[str] | None = None,
) -> PublicationVerdict | None:
    """
    One publication verdict for the age of the artifacts behind tonight's board.

    STRICTEST ACROSS MARKETS, not per row. The board CSV is already written by
    the time dispatch runs and carries no per-row artifact identity, so there is
    no honest way to withhold one market's rows and publish another's from here.
    Withholding the board and naming the stale market is the direction that
    cannot mislead: the abstention embed still goes out, carrying which market
    is stale and by how much, so the operator learns the cause tonight rather
    than from an eventual look at the dates.

    A MARKET WHOSE ARTIFACT DOES NOT RESOLVE IS WITHHELD, not skipped. Nothing
    scored those rows from a model, so a model-sourced board has no business
    publishing numbers for them, and "no artifact" is exactly the unknown age
    this gate refuses to read as a young one.
    """
    from pathlib import Path

    from main import resolve_model_artifact
    from src.models.artifact_freshness import artifact_freshness, artifact_freshness_for
    from src.quant.dfs_payouts import ProbabilitySource
    from src.quant.publication_gate import artifact_freshness_gate, first_refusal

    names = [str(m).upper() for m in (markets if markets is not None else board_markets())]
    if not names:
        return None

    newest = newest_game_date_or_none()
    verdicts = []
    for market in names:
        try:
            path, how = resolve_model_artifact(market=market)
        except Exception as exc:  # noqa: BLE001 — the resolver's failure is the answer
            logger.warning("Could not resolve the %s artifact (%s).", market, exc)
            path, how = None, str(exc)
        if path is None or not Path(path).exists():
            logger.warning(
                "No %s artifact resolved (%s), so its age is unknown.", market, how,
            )
            freshness = artifact_freshness(None, market=market)
        else:
            freshness = artifact_freshness_for(
                path, newest_game_date=newest, market=market
            )
        verdicts.append(
            artifact_freshness_gate(
                freshness, probability_source=ProbabilitySource.MODEL
            )
        )
    return first_refusal(*verdicts)


def board_markets() -> list[str]:
    """
    Which markets this deployment builds a board for.

    SPLIT OUT so the slate and the deploy healthcheck cannot disagree. The
    parsing lived inside ``run_board`` and was unreachable from anywhere else,
    so ``scripts/railway_healthcheck`` would have had to re-implement it --
    and a healthcheck that probes PTS/REB/AST while the worker is configured
    for PTS alone reports a failure that is not one, or misses one that is.
    This repository has now found that defect three times (the parlay ledger
    read twice, PRA defined twice); one reader is the fix.

    An empty or unparseable ``PROPIQ_BOARD_MARKETS`` falls back to the
    pipeline's own defaults rather than to an empty list, which would silently
    build nothing.
    """
    from src.pipeline.slate_board import DEFAULT_MARKETS

    raw = (os.environ.get(ENV_BOARD_MARKETS) or "").strip()
    parsed = [m.strip().upper() for m in raw.split(",") if m.strip()]
    return parsed or list(DEFAULT_MARKETS)


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

    from src.utils.volume import (
        classify_unwritable,
        resolve_state_root,
        unwritable_remedy,
    )

    # BOTH DIRECTORIES THAT MATTER, because they can differ and each breaks
    # something different when it is not writable.
    #
    # THE STATE ROOT was the production defect, found 2026-10-12: a volume was
    # attached at /var/lib/postgresql/data and root-owned,
    # `railway_healthcheck` FAILED on it correctly, and this check -- the one
    # whose whole purpose is catching a root-owned mounted volume -- probed
    # /app/data instead, because that is where the calibration report pointed,
    # and reported "writable". A FALSE ALL-CLEAR, so `notify_boot_problems`
    # alerted nobody about the fault that silently breaks the artifact store.
    # Two readers of one fact, the fifth time in this repository; the resolver
    # in `utils.volume` is the one reader.
    #
    # THE CALIBRATION REPORT'S DIRECTORY stays covered too. Settlement writes
    # it at 03:30 PT and the slate reads it at 09:00 PT, and an operator can
    # point $PROPIQ_CALIBRATION_REPORT outside the state root -- probing only
    # the root would have dropped a check that already existed, which is not a
    # fix, it is a trade.
    #
    # An explicit `path` still wins outright: that is how a caller aims this at
    # one directory it cares about.

    # EACH TARGET CARRIES WHAT LIVES IN IT, because the two directories break
    # different things and have different causes. cubic, PR #8: the single
    # message this check logged was written when there was one target, and it
    # asserted both "the calibration report and the model artifacts live here"
    # and the root-owned-volume hypothesis for whichever directory failed --
    # true of the state root, wrong for a calibration report configured outside
    # it, where no volume is involved and the artifacts are not affected. The
    # same defect this repository has two test files guarding against: text
    # outliving the facts it describes.
    # WHAT LIVES HERE is this function's to say; WHY THE WRITE FAILED and what
    # to do about it belongs to `utils.volume`, once, because there are two
    # probes. Found in a Railway log 2026-10-13: the healthcheck said "NO VOLUME
    # IS MOUNTED THERE ... chown is not the remedy here" and this check said
    # "most likely owned by root ... chown it to that uid", about the same path,
    # three log lines apart. I had taught one of the two the difference. The
    # fault-to-remedy mapping is now `classify_unwritable` + `unwritable_remedy`
    # and neither probe holds a copy of it.
    uid = os.getuid() if hasattr(os, "getuid") else "?"
    state_root_contents = (
        "The model artifacts and, by default, the calibration report live here, "
        "so score_prob_over will find no model and the publication gate will "
        "withhold every card."
    )
    report_contents = (
        "Settlement writes the calibration report here at 03:30 PT and the "
        "09:00 PT slate reads it, so the gate will withhold every model-sourced "
        "card; market-grounded figures are unaffected, and the model artifacts "
        "live elsewhere."
    )

    # The fourth element is (resolved_via, is_a_mount_point): only the state
    # root is somewhere a VOLUME belongs.
    if path is not None:
        targets = [
            (Path(path).parent, "CALIBRATION REPORT DIRECTORY", report_contents,
             (None, False))
        ]
    else:
        root, how = resolve_state_root()
        targets = [(root, "STATE ROOT", state_root_contents, (how, True))]
        report_dir = calibration_report_path().parent
        # Compare resolved paths: "outputs" and "/app/outputs" are the same
        # directory and probing it twice would log a duplicate.
        if report_dir.resolve() != root.resolve():
            targets.append(
                (report_dir, "CALIBRATION REPORT DIRECTORY", report_contents,
                 (None, False))
            )

    failures: list[dict[str, str]] = []
    for candidate, label, contents, (resolved_via, is_mount) in targets:
        # ONE PROBE FILE PER PROCESS. cubic, PR #11: a fixed
        # ``.propiq_write_probe`` meant two probes racing -- the boot
        # healthcheck and this check run seconds apart in the same container --
        # could fail on each other's file, and a stale probe DIRECTORY left by
        # a killed run made `write_text` raise IsADirectoryError, which was
        # then reported as the PARENT being unwritable. The parent is fine in
        # both cases.
        tmp = candidate / f".propiq_write_probe.{os.getpid()}"
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            tmp.write_text("ok", encoding="utf-8")
            tmp.unlink()
        except OSError as exc:
            # AFTER the failure, because which fault it was cannot be known
            # before it. The ERROR is passed in: an EROFS or ENOSPC on a
            # perfectly-owned directory must not be diagnosed as ownership.
            fault = classify_unwritable(candidate, exc)
            failures.append({
                "path": str(candidate),
                "error": str(exc),
                "what": label,
                "fault": fault,
                "cause": f"{contents} " + unwritable_remedy(
                    candidate, fault, uid=uid, how=resolved_via,
                    mount_point=is_mount,
                ),
            })

    if failures:
        # EVERY failure, one line each. cubic, PR #8: this read failures[0] and
        # dropped the rest, so with both directories broken an operator would
        # chown the one named, redeploy, and meet the same alert naming the
        # other. ``path``/``error`` stay on the dict for the callers that read
        # them -- scripts/validate_docker.py and the healthcheck -- and
        # ``failures`` is what carries the whole list.
        for failure in failures:
            logger.error(
                "%s %s IS NOT WRITABLE (%s). %s The slate still runs and still "
                "writes to Postgres.",
                failure["what"], failure["path"], failure["error"], failure["cause"],
            )
        return {
            "path": failures[0]["path"],
            "writable": False,
            "error": failures[0]["error"],
            # WHICH fault, beside path/error, which are also the first
            # failure's. A caller that wants them all reads ``failures``.
            "fault": failures[0]["fault"],
            "failures": failures,
            "checked": [str(t) for t, _, _, _ in targets],
        }
    logger.info(
        "State director%s writable: %s.",
        "y is" if len(targets) == 1 else "ies are",
        ", ".join(str(t) for t, _, _, _ in targets),
    )
    return {
        "path": str(targets[0][0]),
        "writable": True,
        "checked": [str(t) for t, _, _, _ in targets],
    }


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
    """
    One slate run, under the cross-process write lock.

    THE LOCK IS HERE, NOT INSIDE ``main.main``, and the first attempt at A1 had
    it the other way round. Four AST tests across three files walk ``main``'s
    body for the steps it must wire and the order it wires them in; extracting
    that body so it could be indented under a ``with`` moved every one of those
    calls out of the function they assert on. The tests were right and the
    refactor was the mistake. Taking it at the two CALL SITES instead -- here
    and in ``main.py``'s ``__main__`` block, which are ``main``'s only two
    callers -- leaves that body alone and covers MORE than wrapping ``main``
    did: the board write and the dispatch below sit inside this scope, and a
    half-written board is as bad as a half-written upsert.

    ``utils.write_lock`` is reentrant, so nesting stays safe if a third caller
    ever appears.
    """
    from src.utils.write_lock import write_lock

    # ONE SLATE AT A TIME, ACROSS PROCESSES (A1). max_instances=1 above is a
    # property of THIS process; this is the guard against a manual
    # `railway run python main.py` interleaving its writes with ours. The scope
    # deliberately covers the board write and the dispatch below too, not just
    # the scoring: a half-written board is as bad as a half-written upsert.
    with write_lock() as acquired:
        if not acquired:
            logger.warning(
                "Another process holds the slate write lock, so the 09:00 PT "
                "run is SKIPPED. Nothing was scored, no board was built and no "
                "card was sent. A deliberate decline, not a failure."
            )
            return 0
        return _run_slate_locked(argv)


def _run_slate_locked(argv: list[str] | None = None) -> int:
    """
    The slate sequence, with the write lock already held by ``run_slate``.

    Delegates to ``main.main`` rather than reimplementing it, so the scheduled
    path and the manual one cannot drift. An exception is caught and logged: a
    failed slate must not take the scheduler down, or one bad day ends every
    subsequent day too.

    SYSTEMEXIT IS CAUGHT SEPARATELY, AND THAT IS NOT A DETAIL. `main.preflight`
    ends a run it cannot perform with ``raise SystemExit(1)`` (`main.py:264`),
    and argparse exits 2 on bad argv -- both `BaseException`, so `except
    Exception` never saw either. On Railway on 2026-10-11 the 09:00 PT slate
    met an unset `DATABASE_URL`, preflight refused, and the result was a raw
    APScheduler traceback and NO Discord alert -- while an ordinary exception
    on this same line would have alerted. The single most likely failure of a
    misconfigured deploy was the single failure that told nobody.

    `main.main` is a CLI ENTRY POINT being called in-process, so its SystemExit
    means "the CLI would have exited with this code". For this caller that is a
    return value, not a request that the worker exit. `KeyboardInterrupt` is
    deliberately NOT caught: that one IS a shutdown request, and swallowing it
    would make the worker unkillable mid-slate -- a worse defect than this one.
    """
    import main

    try:
        code = main.main(argv or [])
    except SystemExit as exc:
        # `SystemExit(None)` and `SystemExit(0)` are a CLI that SUCCEEDED and
        # must not alert, or the fix turns every clean run into an incident.
        code = 0 if exc.code is None else exc.code
        if isinstance(code, int) and code == 0:
            logger.info("Slate run exited cleanly (SystemExit 0).")
        else:
            logger.error(
                "Slate run exited with SystemExit %r; the scheduler stays up. "
                "This is how main.preflight refuses a run it cannot do -- an "
                "unset DATABASE_URL is the usual cause.", exc.code,
            )
            notify_failure(
                "slate",
                f"The 09:00 PT slate exited early with code {exc.code!r}. "
                f"main.preflight refuses a run it cannot perform -- the usual "
                f"cause is an unset DATABASE_URL, which makes every database "
                f"step raise. Nothing was scored, no board was built and no "
                f"card was sent.",
            )
            return 1 if not isinstance(code, int) else code
    except Exception as exc:  # noqa: BLE001 — a job may not kill the worker
        logger.exception("Slate run raised; the scheduler stays up.")
        notify_failure("slate", f"The 09:00 PT slate raised: {exc}")
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
        # THE FAILURE PATH THAT WENT SILENT, and it is the one that fires.
        # This function's own docstring says it exists because "a boot ERROR,
        # a FAILED slate, a FAILED settlement and a probe FAILURE were
        # log-only" -- yet the two branches above alert on a SystemExit and on
        # a raised exception, and NOTHING alerted when `main.main()` simply
        # RETURNED non-zero. That is not the rare case: main wraps its whole
        # body in `except Exception`, logs "Pipeline FAILED", records the run
        # and returns 1, so the ordinary internal error reaches here as a
        # return code and never as an exception.
        #
        # Measured on the live worker: the 2026-10-20 and 2026-10-21 slates
        # both failed this way -- `MarketFramesUnavailable`, nothing ever
        # ingested -- and the operator got a cheerful "0 graded" results card
        # at 03:30 and silence at 09:00 on the first slate of the season. A
        # board going empty for days with nobody told is the exact failure
        # `notify_failure` was written for.
        #
        # The CAUSE is not in hand here: `main.main()` returns an int, and the
        # exception string lives in its own log line and in the `pipeline_runs`
        # row it writes before returning. So this names the code and where the
        # cause is, rather than inventing a reason it does not know.
        notify_failure(
            "slate",
            f"The 09:00 PT slate exited {code} without building a board, so "
            f"no card was sent and nothing was scored for today. The cause is "
            f"the `Pipeline FAILED:` log line from this run, and the same "
            f"string is stored on the run's `pipeline_runs` row. Common "
            f"causes: nothing ingested yet (no team_game_stats or "
            f"game_market_lines rows, which refuses the feature build), or no "
            f"resolvable model artifact.",
        )
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
        from src.pipeline.slate_board import build_slate_board

        markets = list(board_markets())
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
        # ADVISORY SIZING, attached by this caller rather than by the ranking
        # layer -- see attach_board_sizing for the two guards that require it.
        # getattr, not result.board: the attribute access itself must not be
        # able to fail the board. run_board's own except would turn an
        # AttributeError here into {"status": "FAILED"} for a board that was
        # built and written -- the same misattribution the prelock note below
        # warns about, and it is how the first version of this broke two
        # existing tests whose stub result carries no board.
        sizing = attach_board_sizing(getattr(result, "board", None), board_csv)
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
        return {"status": "OK", **result.as_dict(), "window": window,
                "sizing": sizing}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Board build raised; the scheduler stays up.")
        notify_failure("board", f"The board build raised: {exc}")
        return {"status": "FAILED", "error": str(exc), "out": board_csv}


def attach_board_sizing(board: Any, csv_path: str) -> dict[str, Any]:
    """
    Add an advisory size to each board row, and write it into the board CSV.

    WHY THIS LIVES HERE AND NOT IN src/. Two existing guards say so, and both
    fired when this was first written inside `src/quant/decision_board.py`:

      * `tests/test_advisory_sizing.py::test_no_module_under_src_imports_this_one`
        -- "Nothing under src/ may import it at all. A size is held by a caller
        that already computed it, never fetched by a module that sends or
        places anything."
      * `tests/test_decision_board.py::test_layer_cannot_reach_a_book_or_size_a_stake`
        -- no identifier in that module may contain "kelly".

    So `decision_board` stays a pure ranking layer with no sizing concept, and
    the size is attached by this caller, which is not under `src/` and sends
    nothing. The guards were right and the design is better for it: the
    ranking layer cannot develop an opinion about stake.

    WHAT IT CLOSES. `advisory_sizing` was reachable from exactly one place --
    `scripts/nba_model_cli.py`, behind a flag on a hand-run command -- so the
    nightly board carried no recommended size at all.
    `docs/autonomy_audit_2026-10-10.md` §3.2.

    ADVISORY, AND THAT IS THE WHOLE CONTRACT. `AUTO_PLACED` is written as False
    on every row, nothing in this project places or sizes a wager, and no
    execution path reads these columns. A row the board withheld gets no size.

    POSITIONAL JOIN, asserted. `build_slate_board` wrote the CSV from this same
    ordered list via `board_to_dataframe`, so row i of the frame is candidate i
    -- and a length mismatch means that stopped being true, which would attach
    one row's size to another's price. It refuses rather than guessing.

    NEVER RAISES. The board is the durable output and it is already written; a
    sizing failure must not turn a good board into a failed one.
    """
    out: dict[str, Any] = {"sized": 0, "declined": 0, "status": "OK"}
    try:
        import pandas as pd

        from src.quant.advisory_sizing import advisory_size_for_candidate

        rows = list(board or [])
        if not rows:
            # Also the path for a caller whose result carries no board at all,
            # which is why the caller passes it with getattr: nothing about
            # sizing may decide whether the board succeeded.
            return {"status": "SKIPPED", "reason": "no board rows", "sized": 0}

        sizes = []
        for c in rows:
            try:
                sized = advisory_size_for_candidate(c)
            except Exception as exc:  # noqa: BLE001 — one row must not lose the rest
                logger.warning("Advisory sizing raised for a row (%s).", exc)
                sizes.append(None)
                continue
            sizes.append(sized)

        frame = pd.read_csv(csv_path)
        if len(frame) != len(rows):
            logger.error(
                "Board CSV has %d row(s) and the board has %d; NOT attaching "
                "sizes, because the positional join this relies on is no "
                "longer true and would pair one row's size with another's "
                "price.", len(frame), len(rows),
            )
            return {"status": "FAILED", "reason": "row count mismatch",
                    "csv_rows": len(frame), "board_rows": len(rows)}

        # A declined row gets None, never 0.0: zero would read as "sized, and
        # the answer is nothing" rather than "not sized, and here is why".
        frame["RECOMMENDED_UNITS"] = [
            None if s is None or s.method == "none" else float(s.recommended_units)
            for s in sizes
        ]
        frame["SIZING_METHOD"] = [
            None if s is None or s.method == "none" else s.method for s in sizes
        ]
        frame["KELLY_FRACTION_APPLIED"] = [
            None if s is None or s.method == "none"
            else float(s.kelly_fraction_applied) for s in sizes
        ]
        frame["SIZING_CAPPED"] = [
            None if s is None or s.method == "none" else bool(s.capped) for s in sizes
        ]
        frame["SIZING_NOTE"] = [None if s is None else s.reason for s in sizes]
        # Written on every row, not just sized ones: the claim is about the
        # whole pipeline, not about this number.
        frame["AUTO_PLACED"] = False
        frame.to_csv(csv_path, index=False)

        out["sized"] = int(sum(
            1 for s in sizes if s is not None and s.method != "none"
        ))
        out["declined"] = len(sizes) - out["sized"]
        logger.info(
            "Advisory sizing: %d row(s) sized, %d declined (no valid price, a "
            "withheld row, or push mass with no two sides to condition on). "
            "ADVISORY ONLY -- nothing here places or sizes a wager.",
            out["sized"], out["declined"],
        )
        return out
    except Exception as exc:  # noqa: BLE001 — must not fail an written board
        logger.exception(
            "Could not attach advisory sizing (%s). The board above STANDS and "
            "was written; what is lost is the recommended size column.", exc,
        )
        return {"status": "FAILED", "error": str(exc)}


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

    if not _flag(ENV_DISPATCH, webhook_configured()):
        logger.info(
            "Dispatch off (%s unset and no %s). Nothing sent.",
            ENV_DISPATCH, ENV_WEBHOOK_URL,
        )
        return {"status": "SKIPPED", "reason": "dispatch disabled"}

    board_csv = Path(os.environ.get(ENV_BOARD_CSV) or DEFAULT_BOARD_CSV)
    report_path = calibration_report_path()

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
        from src.quant.publication_gate import calibration_gate, first_refusal

        report = None
        if report_path.exists():
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                # An unreadable report is not the same as no report, and the gate
                # would treat None as "none supplied". Say which it was.
                logger.warning("Calibration report %s is not JSON: %s", report_path, exc)
        verdict = calibration_gate(report, probability_source=ProbabilitySource.MODEL)
        # THE SECOND GATE: the artifact's own age. A fresh calibration report
        # describing a months-old artifact passes the gate above and is evidence
        # about a model nobody is running any more -- see
        # publication_gate.artifact_freshness_gate. first_refusal keeps whichever
        # reason actually fired, because "withheld" with no cause reads in a
        # channel as a broken pipeline rather than as a stale model.
        verdict = first_refusal(verdict, artifact_freshness_verdict())

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
            # THE SIZE IS COMPUTED HERE, by the caller, and carried onto the
            # row. `src/notify/` may not import the sizer -- see
            # `tests/test_advisory_sizing.py::test_no_module_under_src_imports
            # _this_one`, which exists so sizing cannot reach a dispatch or
            # execution path. scheduler_worker is outside src/, holds the size
            # it computed, and hands it over, which is the same shape
            # `build_dfs_entry_embed` already takes its `advisory_size` in.
            #
            # None, never 0.0: `recommended_units_binary` returns 0.0 both for
            # "no edge" and for inputs it refuses, and a board row printing
            # "0u" reads as a stake recommendation rather than as its absence.
            _attach_advisory_units(rows)
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


def _attach_advisory_units(rows: list[Any]) -> None:
    """
    Set ``advisory_units`` on each board row, or leave it None.

    RESEARCH_ONLY. `advisory_sizing` stamps ``AUTO_PLACED: False`` on every
    size it returns and nothing in this project places one; this only puts the
    number where a reader of the card can see it, beside the EV it belongs to.

    Failures are swallowed deliberately: a card that cannot be sent because a
    size could not be computed is strictly worse than a card without the size.
    """
    try:
        from src.quant.advisory_sizing import recommended_units_binary
    except Exception:  # noqa: BLE001 — the card matters more than the size
        logger.warning("Advisory sizing unavailable; board rows carry no size.")
        return
    for row in rows:
        row.advisory_units = None
        prob, odds = getattr(row, "model_prob", None), getattr(row, "american_odds", None)
        if prob is None or odds is None:
            continue
        try:
            american = int(odds)
            decimal = 1.0 + (
                american / 100.0 if american > 0 else 100.0 / abs(american)
            )
            units = float(
                recommended_units_binary(float(prob), decimal).recommended_units
            )
        except Exception:  # noqa: BLE001 — one unsizable row is not a failed card
            continue
        if units > 0:
            row.advisory_units = units


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

    # PROJECTABLE, not merely pre-tip. Found in production 2026-10-12: this
    # armed a tip-off job for every one of five PRESEASON games, one of them
    # against ``LON``, a code with no NBA mapping at all. ``projectable`` is
    # the one reader for "may this fixture be projected" -- a preseason test
    # here and another in the slate pipeline would be the two-readers defect
    # this repository has fixed five times.
    #
    # FAILS OPEN, LOUDLY, for a slate object that predates the property: an
    # arming that silently stops happening is worse than one that warns, and
    # the same policy as ``utils.write_lock``.
    filtered = hasattr(slate, "projectable")
    games = list(getattr(slate, "projectable", None) or [])
    if not filtered:
        logger.warning(
            "This slate object has no .projectable, so the preseason filter "
            "cannot have run on it; falling back to pre-tip games. A preseason "
            "fixture may be armed."
        )
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

    # WHAT THE DENOMINATOR IS. cubic, PR #9: this said "armed N of M pre-tip
    # game(s)" while M had become the PROJECTABLE count, so a slate with two
    # pre-tip games of which one was preseason read "armed 1 of 1 pre-tip
    # game(s)", and an all-excluded slate read "armed 0 of 0" with no mention
    # of the exclusions -- a quiet schedule, again. The three counts are
    # different facts and each is named.
    excluded = list(getattr(slate, "excluded", None) or [])
    pretip = list(getattr(slate, "pregame_only", None) or [])
    # ON THE FALLBACK PATH THESE COUNTS ARE NOT KNOWN. cubic, PR #10: with a
    # slate object that predates `.projectable`, ``games`` is the UNFILTERED
    # pre-tip list, so calling it "projectable" reported excluded fixtures as
    # projectable and the absent `.excluded` as zero exclusions -- two false
    # numbers presented as measurements. The line says which it is.
    if filtered:
        logger.info(
            "Pre-lock: armed %d of %d projectable game(s) (%d pre-tip, %d "
            "excluded) at tip-off minus %d minutes. Skipped %d: %s",
            len(armed), len(games), len(pretip), len(excluded), lead,
            len(skipped), [s["reason"] for s in skipped[:4]] or "none",
        )
    else:
        logger.info(
            "Pre-lock: armed %d of %d pre-tip game(s) at tip-off minus %d "
            "minutes (UNFILTERED: this slate object has no .projectable, so "
            "how many were excluded is not known). Skipped %d: %s",
            len(armed), len(games), lead, len(skipped),
            [s["reason"] for s in skipped[:4]] or "none",
        )
    for game, reason in excluded:
        logger.info(
            "Pre-lock: NOT armed — %s @ %s: %s",
            getattr(game, "away_team", "?"), getattr(game, "home_team", "?"), reason,
        )
    return {
        "status": "OK", "armed": len(armed), "lead_minutes": lead,
        "jobs": armed, "skipped": skipped,
        # Named separately so a caller cannot mistake one for the other, and
        # None rather than a number where the filter did not run: "not
        # measured" must not read as zero.
        "filtered": filtered,
        "projectable": len(games) if filtered else None,
        "pretip": len(pretip) if filtered else None,
        "excluded": [
            f"{getattr(g, 'away_team', '?')} @ {getattr(g, 'home_team', '?')}: {r}"
            for g, r in excluded
        ] if filtered else None,
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

        if not _flag(ENV_DISPATCH, webhook_configured()):
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
        notify_failure("settlement", f"The 03:30 PT settlement raised: {exc}")
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
    if not _flag(ENV_RESULTS_CARD, _flag(ENV_DISPATCH, webhook_configured())):
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

    target = calibration_report_path(path)
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
        # job_defaults, because APScheduler's OWN DEFAULT misfire grace is
        # ONE SECOND. Every add_job below passes its grace explicitly, so the
        # defaults change nothing today -- they exist for the next job someone
        # adds. A job registered without those kwargs would be silently
        # skipped by a two-second delay, which on a BlockingScheduler sharing
        # a thread pool with a running slate is an ordinary occurrence, and
        # "the job did not run" is indistinguishable from "the job ran and
        # found nothing".
        #
        # coalesce=True so a container that was down through several firings
        # runs each job ONCE on return rather than catching up N times --
        # re-running a slate four times would re-dispatch the same card four
        # times.
        scheduler = BlockingScheduler(
            timezone=DISPLAY_TZ,
            job_defaults={
                "max_instances": 1,
                "coalesce": True,
                "misfire_grace_time": DEFAULT_MISFIRE_GRACE_SECONDS,
            },
        )

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
    from src.utils.logstreams import configure_logging

    # Was dead code: `main` is imported above and calls this at module
    # scope, and `basicConfig` is a silent no-op once the root logger has
    # handlers. `configure_logging` is idempotent, so this now states the
    # intent without depending on which module got there first.
    configure_logging()
    cap_thread_counts()
    state_state = check_state_dir()
    # Asked at boot, beside the state directory, because a redeploy breaks the
    # same two things: the volume's permissions and whether anything was ever
    # put on it. See check_model_artifact for why inference is too late.
    artifact_state = check_model_artifact()
    ledger_state = check_parlay_ledger()
    # Asked at boot for the reason in check_database: on 2026-10-11 this was
    # the actual fault and the boot said nothing about it.
    database_state = check_database()
    # ROUTED OUT, not just logged. In Railway's log view every line is already
    # red -- Python logging writes to stderr and Railway classifies stderr as
    # error severity -- so a real boot failure is indistinguishable from an
    # INFO line by eye. Deduped per stage per Pacific day so a redeploy loop
    # does not post the same alert on every boot.
    notify_boot_problems(state_state, artifact_state, ledger_state, database_state)

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
