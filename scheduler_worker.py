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
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timezone
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
ENV_ESPN_INGEST = "PROPIQ_ESPN_INGEST"
ENV_ESPN_HOUR = "PROPIQ_ESPN_HOUR_PT"
ENV_ESPN_MINUTE = "PROPIQ_ESPN_MINUTE_PT"
ENV_ESPN_LOOKBACK = "PROPIQ_ESPN_LOOKBACK_DAYS"
ENV_ESPN_BACKFILL = "PROPIQ_ESPN_BACKFILL_DAYS"
ENV_ESPN_HISTORY_SEASONS = "PROPIQ_ESPN_HISTORY_SEASONS"
ENV_ESPN_HISTORY_CHUNK = "PROPIQ_ESPN_HISTORY_CHUNK_DAYS"
ENV_ESPN_HISTORY_HOUR = "PROPIQ_ESPN_HISTORY_HOUR_PT"
ENV_ESPN_HISTORY_MINUTE = "PROPIQ_ESPN_HISTORY_MINUTE_PT"
ENV_SETTLE_HOUR = "PROPIQ_SETTLE_HOUR_PT"
ENV_SETTLE_MINUTE = "PROPIQ_SETTLE_MINUTE_PT"
ENV_MAX_THREADS = "PROPIQ_MAX_THREADS"
ENV_RUN_ON_START = "PROPIQ_RUN_ON_START"
ENV_DISPATCH = "PROPIQ_DISPATCH"
ENV_DISPATCH_ABSTENTIONS = "PROPIQ_DISPATCH_ABSTENTIONS"
#: Append a RESEARCH PREVIEW embed listing the slate's props -- player,
#: market, line, projection and model lean -- alongside whatever dispatch
#: was already going to send. OFF BY DEFAULT.
#:
#: WHY IT EXISTS. With no model artifact every board row is ABSTAIN, so the
#: recommendation card reads "0 qualifying plays out of N line(s)
#: evaluated" and names nobody; withheld by the publication gate it names
#: nobody either. Both are correct for a card that makes a CLAIM. An
#: operator who wants to see which players and lines the slate is working
#: with then has nowhere to look but the CSV on the volume.
#:
#: IT DOES NOT OPEN THE GATE, and it is not a way round it. The preview is
#: APPENDED, never substituted: the gate's own card still goes out saying
#: what is withheld and why, the preview repeats that verdict inline, and
#: every row keeps its own ABSTAIN. What the gate withholds is the claim;
#: this shows the inputs to a claim nobody is making yet. No probability
#: and no EV appear, because with no artifact there are none.
ENV_RESEARCH_PREVIEW = "PROPIQ_RESEARCH_PREVIEW"
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
#: 02:00 PT. AFTER the last game of the night has finished -- a 19:30 PT tip
#: on the west coast is final well before 23:00 -- and BEFORE settlement at
#: 03:30 and the slate at 09:00, both of which read what this writes.
DEFAULT_ESPN_HOUR, DEFAULT_ESPN_MINUTE = 2, 0
#: How many days back a NIGHTLY run reads. Three, not one: a game finishing
#: after midnight Pacific lands on the next calendar day, and ESPN corrects a
#: box score for a day or two afterwards. The upsert is idempotent, so
#: re-reading a settled night costs a fetch and changes nothing.
DEFAULT_ESPN_LOOKBACK_DAYS = 3
#: How far back the FIRST run reads, when the panel is empty. A three-day
#: window would leave every rolling feature null -- `PTS_L10` needs ten prior
#: games -- so an empty panel is seeded properly rather than left in a state
#: that looks ingested and scores nothing.
DEFAULT_ESPN_BACKFILL_DAYS = 30
#: How many game DATES one history-backfill run reads. Sixty, because a run
#: costs one scoreboard call per date plus one summary call per game on it --
#: roughly 12 games a night, so ~780 calls and on the order of twenty minutes.
#: A whole season in one run would be ~1,400 calls held open against a service
#: that owes this project nothing; sixty dates finishes a season in three
#: nights and leaves a failed chunk cheap to repeat.
DEFAULT_ESPN_HISTORY_CHUNK_DAYS = 60
#: 04:30 PT: after the 02:00 ingest and the 03:30 settlement, and done well
#: before the 09:00 slate. The backfill is the longest-running job here and it
#: must not still be writing to `player_game_logs` when the slate reads it.
DEFAULT_ESPN_HISTORY_HOUR, DEFAULT_ESPN_HISTORY_MINUTE = 4, 30

# A job that misses its window by less than this still runs; past it, the slate
# has moved on and running late would project games that have already tipped.
#: The fallback grace for a job added WITHOUT its own. APScheduler's own
#: default is one second; fifteen minutes is long enough that a brief overrun
#: does not silently drop a job and short enough that nothing stale fires.
#: Every job registered here sets its own, so this only governs the next one.
DEFAULT_MISFIRE_GRACE_SECONDS = 15 * 60

SLATE_MISFIRE_GRACE_SECONDS = 60 * 60
SETTLE_MISFIRE_GRACE_SECONDS = 6 * 60 * 60
#: Six hours, same as settlement: a worker that was down at 02:00 should still
#: fill the panel when it comes back, because the 09:00 slate reads it.
ESPN_MISFIRE_GRACE_SECONDS = 6 * 60 * 60
#: THIRTY MINUTES, deliberately the shortest grace here. Every other job
#: catches up because something downstream reads what it writes today. The
#: history backfill writes seasons-old box scores: a chunk missed at 04:30 is
#: no more urgent at 11:00, and firing it late would have it reading
#: `player_game_logs` while the 09:00 slate does.
ESPN_HISTORY_MISFIRE_GRACE_SECONDS = 30 * 60
#: How long before the slate the history backfill must be scheduled. A chunk
#: of 60 dates is on the order of twenty minutes, so an hour leaves room for a
#: slow night without the two overlapping. Enforced rather than documented:
#: `PROPIQ_ESPN_HISTORY_HOUR_PT=9` against a 09:00 slate would have this
#: writing `player_game_logs` while the slate reads them, and the result is a
#: slate that scores a partially-written panel with nothing saying so. Raised
#: by review on #31.
ESPN_HISTORY_MIN_LEAD_MINUTES = 60

#: Markets to fit, comma separated (e.g. "PTS,REB,AST"). UNSET MEANS OFF, the
#: same shape as the history backfill: the job is registered either way and
#: declines when it is not configured, so turning it on is a variable and not
#: a deploy.
ENV_TRAIN_MARKETS = "PROPIQ_TRAIN_MARKETS"
#: How many of the newest days to hold OUT of the fit. The holdout is the most
#: recent slice on purpose: a model is asked about tomorrow, so the only
#: honest check is on games after everything it learned from.
ENV_TRAIN_HOLDOUT_DAYS = "PROPIQ_TRAIN_HOLDOUT_DAYS"
DEFAULT_TRAIN_HOLDOUT_DAYS = 30
#: How far back the training panel reaches. NOT 0: `load_player_panel`
#: HONOURS this now and computes `lower = upper - days`, so 0 would load a
#: single day. 1200 reaches 2023-07 from a 2026-11 slate, which covers every
#: season the ESPN backfill can fill.
ENV_TRAIN_LOOKBACK_DAYS = "PROPIQ_TRAIN_LOOKBACK_DAYS"
DEFAULT_TRAIN_LOOKBACK_DAYS = 1200
TRAINING_MISFIRE_GRACE_SECONDS = 30 * 60
#: Training holds the slate's write lock, so it must be finished long before
#: 09:00. Measured on the demo panel a market fits in seconds; on the real
#: panel it is minutes. Two hours is not a tuned number, it is a margin.
TRAINING_MIN_LEAD_MINUTES = 120
#: How long after boot the one-off "there is no artifact at all" fit starts.
#: Long enough that the worker is up and its job table is logged first, short
#: enough that an operator who just switched training on sees it happen.
TRAINING_BOOT_DELAY_SECONDS = 30

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


def _board_market_frames() -> dict[str, Any]:
    """
    The team-game and market-line frames for the board, or empty ones.

    THE SAME TWO FRAMES THE SLATE USES. Without `team_games` the matrix has
    no Elo and no `DEF_*`; without `market_lines` no `MKT_*` or `BLOWOUT_*`.
    A board built from a narrower matrix than the slate scored on would be a
    second, quieter answer to the same question.

    NEVER RAISES. `main.resolve_market_frames` raises
    `MarketFramesUnavailable` when BOTH are empty, which is correct for the
    slate -- it has nothing to project. The board is downstream of a slate
    that already decided that, so here an absent frame is a narrower matrix
    and a logged warning, not a dead job.
    """
    empty: dict[str, Any] = {"team_games": None, "market_lines": None}
    try:
        import main as orchestrator

        # The orchestrator's OWN two names for the workbook, not a third
        # copy of the path -- main.py's argparse default is exactly this
        # pair, and a literal here would be a second definition of where
        # the licensed export lives. `persist=False`: the board reads, the
        # slate job ingests.
        xlsx = Path(
            os.environ.get(orchestrator.ENV_BIGDATABALL)
            or orchestrator.DEFAULT_BIGDATABALL_XLSX
        )
        frames = orchestrator.resolve_market_frames(xlsx, persist=False)
        return {
            "team_games": frames.get("team_games"),
            "market_lines": frames.get("market_lines"),
        }
    except Exception as exc:  # noqa: BLE001 — a narrower board beats no board
        logger.warning(
            "Board: no team-game or market-line frames (%s: %s). The matrix "
            "will carry no Elo, DEF_*, MKT_* or BLOWOUT_* columns.",
            type(exc).__name__, exc,
        )
        return empty


def _discard_stale_board(board_csv: Any, slate_date: str) -> bool:
    """
    Delete a board CSV that does not describe ``slate_date``. True if it did.

    A BOARD IS A STATEMENT ABOUT ONE SLATE. `run_dispatch` reads the file
    and sends it without consulting `run_board`'s status, so a file left
    behind by a successful earlier run is indistinguishable, at the
    dispatch surface, from tonight's. Removing it turns "stale
    recommendations" into "no recommendations", and the abstention card
    already says that honestly.

    KEYED ON THE ROWS' OWN `slate_date`, not on the file's mtime: a board
    rebuilt at 09:05 for the same slate is not stale, and a file written
    yesterday at 09:00 is. The column is written by
    `decision_board.board_to_dataframe` for exactly this kind of question.

    A board that cannot be read is removed too. An unparseable file is not
    evidence of anything, and leaving it would have dispatch fail on it
    instead.
    """
    from pathlib import Path as _Path

    target = _Path(str(board_csv))
    if not target.exists():
        return False
    try:
        import pandas as pd

        frame = pd.read_csv(target)
        dates = {str(d) for d in frame.get("slate_date", pd.Series(dtype=str))}
        if dates == {str(slate_date)}:
            return False
        reason = f"describes {sorted(dates) or 'no slate'}"
    except Exception as exc:  # noqa: BLE001 — unreadable is not evidence
        reason = f"could not be read ({type(exc).__name__})"

    try:
        target.unlink()
    except OSError as exc:
        logger.error(
            "Could not remove the stale board %s (%s). DISPATCH MAY SEND IT: "
            "it reads the file without checking which slate it describes.",
            target, exc,
        )
        return False
    logger.warning(
        "Removed the board at %s: it %s, not %s. A board left in place would "
        "be dispatched as tonight's, because dispatch reads the file and not "
        "this job's status.", target, reason, slate_date,
    )
    return True


def _backtest_board(
    board_csv: Any, *, markets: Any, train_end: str, validation_end: str,
    min_ev: float,
) -> tuple[Any, str]:
    """
    The comparison board, for a window whose games have been played.

    KEPT, NOT DELETED. `compare_models_on_panel` is the right tool for a
    backtest: it fits on history and scores a labelled window, which is
    exactly what an override into the past asks for. What it cannot do is
    describe a game with no outcome yet, which is why the live path reads
    `main.py`'s persisted projections instead.

    The feature build stays here because this path needs it -- the research
    line is built from `{MARKET}_L10`, which only `build_feature_matrix`
    adds (#31). The live path needs none of it: those rows were scored at
    09:00 from a matrix `main.py` already built.
    """
    from src.db.repository import load_player_panel
    from src.features.builder import build_feature_matrix
    from src.pipeline.slate_board import build_slate_board

    panel = load_player_panel()
    if panel is None or panel.empty:
        raise ValueError(
            "DATA_NOT_AVAILABLE: empty player panel, so a backtest board has "
            "no history to fit on."
        )
    frames = _board_market_frames()
    features = build_feature_matrix(
        panel,
        team_games=frames["team_games"],
        market_lines=frames["market_lines"],
    )
    logger.info(
        "Backtest board matrix: %d row(s) x %d col(s) from a %d-row panel "
        "(team_games=%d, market_lines=%d).",
        len(features), len(features.columns), len(panel),
        0 if frames["team_games"] is None else len(frames["team_games"]),
        0 if frames["market_lines"] is None else len(frames["market_lines"]),
    )
    result = build_slate_board(
        features,
        out=board_csv,
        markets=markets,
        train_end=train_end,
        validation_end=validation_end,
        min_ev=min_ev,
    )
    return result, "model comparison (backtest)"


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
        markets = list(board_markets())
        window = board_window()
        train_end = window["train_end"]
        validation_end = window["validation_end"]
        try:
            min_ev = float(os.environ.get(ENV_MIN_EV) or 0.0)
        except ValueError:
            min_ev = 0.0

        # A LIVE BOARD AND A BACKTEST BOARD ARE TWO DIFFERENT QUESTIONS, and
        # for the project's whole life this job asked the backtest one and
        # stamped tonight's date on the answer.
        #
        # `build_slate_board` routes through `compare_models_on_panel`, which
        # fits models on labelled history and scores a window of games with
        # KNOWN OUTCOMES -- `compare.py` drops every row whose `over_hit` is
        # null before it splits. A game that has not been played has no
        # outcome, so no label, so it is dropped. That function structurally
        # cannot describe tonight, and three fixes in two days each got one
        # step closer to that wall: #31 supplied the feature matrix it needed,
        # #32 attached the forward slate, and the forward rows were dropped at
        # the label filter. The second of those could not work and was caught
        # in review before it merged.
        #
        # THE NUMBERS FOR TONIGHT ALREADY EXIST. `main.py` scored the slate at
        # 09:00 PT and persisted one row per (player, game, market) with the
        # prop line attached and the EV gate applied -- 460 rows on
        # 2026-11-01, visible only in Postgres because nothing read the table
        # back. Ranking and formatting those IS the board.
        #
        # So: a live slate reads the projections; a backtest window keeps the
        # comparison path, which is the thing it is actually for and which
        # works there, because those games have outcomes.
        if window.get("is_backtest"):
            result, source = _backtest_board(
                board_csv, markets=markets, train_end=train_end,
                validation_end=validation_end, min_ev=min_ev,
            )
        else:
            from src.db.repository import load_projections
            from src.pipeline.slate_board import build_board_from_projections

            projections = load_projections(validation_end)
            if projections.empty:
                logger.warning(
                    "No projections persisted for %s, so there is nothing to "
                    "rank. The slate job writes them at %02d:00 PT; a board "
                    "asked for before it has run, or after a slate that "
                    "refused, has no input. This is the pipeline reporting "
                    "its own state.",
                    validation_end,
                    _int_env(ENV_SLATE_HOUR, DEFAULT_SLATE_HOUR, low=0, high=23),
                )
                # THE REFUSAL MUST TAKE THE OLD BOARD WITH IT. `run_dispatch`
                # reads whatever CSV is on disk and sends it; it does not
                # look at `run_board`'s return value. So a refusal that
                # leaves yesterday's file in place dispatches yesterday's
                # recommendations as tonight's -- and this change is what
                # makes that reachable, because until now no board had ever
                # been written for the file to be stale. Raised by review
                # on #32.
                stale = _discard_stale_board(board_csv, validation_end)
                return {"status": "DATA_NOT_AVAILABLE",
                        "reason": f"no projections persisted for {validation_end}",
                        "discarded_stale_board": stale,
                        "window": window}
            result = build_board_from_projections(
                projections,
                out=board_csv,
                slate_date=validation_end,
                markets=markets,
                min_ev=min_ev,
            )
            source = "projections"
        logger.info("Board source: %s", source)
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
                "source": source, "sizing": sizing}
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
            build_research_preview_embed,
            send_embeds,
        )
        from src.quant.dfs_payouts import ProbabilitySource
        from src.quant.publication_gate import calibration_gate, first_refusal
        from src.utils.timezones import pacific_calendar_date

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
                return {"status": "SKIPPED",
                        "reason": f"{board_csv} missing",
                        "publication": verdict.as_dict()}
            embeds = [build_abstention_embed(
                f"No board was written for this slate ({board_csv} absent), so "
                "there is nothing to recommend. This is the pipeline reporting "
                "its own state, not a slate with no value in it.",
                title="No recommendations",
            )]
        else:
            frame = pd.read_csv(board_csv)
            # THE SECOND HALF OF THE SAME GUARD, and the half that cannot be
            # bypassed by a future code path. `run_board` removes a stale
            # board when IT refuses, but dispatch reads this file without
            # consulting that status at all -- so anything else that leaves a
            # board behind (a crash between write and dispatch, an operator's
            # hand-run CLI, a path that does not exist yet) would send it as
            # tonight's. Checked here, where the number reaches a person.
            #
            # The rows carry their own `slate_date` from
            # `decision_board.board_to_dataframe`; a board for another day is
            # refused with the day it actually describes, because "withheld"
            # without the reason reads as a broken pipeline.
            #
            # A FILE WITH NO `slate_date` COLUMN AT ALL IS LET THROUGH, and
            # that is a deliberate limit rather than an oversight: it cannot
            # be dated, so there is nothing to compare, and every writer in
            # this repository emits the column. Refusing it would only ever
            # block a hand-assembled CSV an operator meant to send.
            board_slates = {
                str(d) for d in frame.get("slate_date", pd.Series(dtype=str))
            }
            today_pt = str(pacific_calendar_date())
            if board_slates and board_slates != {today_pt}:
                logger.error(
                    "BOARD AT %s DESCRIBES %s, NOT %s. Not dispatching it: a "
                    "board is a statement about one slate, and sending "
                    "another day's rows under tonight's heading is the one "
                    "failure worse than sending nothing.",
                    board_csv, sorted(board_slates), today_pt,
                )
                if not _flag(ENV_DISPATCH_ABSTENTIONS, True):
                    return {"status": "SKIPPED",
                            "reason": f"board describes {sorted(board_slates)}, "
                                      f"not {today_pt}",
                            "publication": verdict.as_dict()}
                embeds = [build_abstention_embed(
                    f"The board on disk describes {', '.join(sorted(board_slates))}, "
                    f"not {today_pt}, so it was not sent. A board is a "
                    f"statement about one slate. This is the pipeline "
                    f"reporting its own state.",
                    title="No recommendations",
                )]
                sent = send_embeds(embeds, config=DiscordConfig(dry_run=False))
                # `publication` ON EVERY RETURN, because a caller that reads
                # `out["publication"]` cannot know which branch produced the
                # dict it got. The calibration and freshness verdict is
                # computed above and is true of this slate whether or not a
                # board was dispatched -- withholding it here would make the
                # gate's reason depend on why dispatch declined, which is a
                # different question. Raised by review on #32; the two
                # SKIPPED returns above carried the same gap.
                return {"status": getattr(sent, "status", "OK"),
                        "reason": f"board describes {sorted(board_slates)}, "
                                  f"not {today_pt}",
                        "embeds": len(embeds),
                        "publication": verdict.as_dict()}
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
            # None, never 0.0: the sizer returns 0.0 both for a genuine
            # non-positive edge and for every refusal, and a board row
            # printing "0u" reads as a stake recommendation rather than as
            # its absence. See `_attach_advisory_units`.
            _attach_advisory_units(rows)
            slate = str(frame["slate_date"].iloc[0]) if "slate_date" in frame else None
            embeds = [build_decision_board_embed(
                rows, slate_date=slate, publication=verdict,
            )]
            # APPENDED, NOT SUBSTITUTED. The gate's card goes out either way;
            # this adds the slate's props as research beside it. See
            # ENV_RESEARCH_PREVIEW for why that ordering is the honest one.
            if _flag(ENV_RESEARCH_PREVIEW, False):
                embeds.append(build_research_preview_embed(
                    rows, slate_date=slate, publication=verdict,
                ))
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

    THROUGH `advisory_size_for_candidate`, NOT A SECOND SIZER. The first
    version of this called `recommended_units_binary` on the raw
    ``model_prob``, which is wrong on any whole-number line: binary Kelly's
    ``q`` is P(lose), a push RETURNS the stake, and counting returned money as
    lost UNDERSTATES the size. `advisory_size_for_candidate` already owns that
    reduction -- it conditions on no-push via
    ``model_prob / (model_p_over + model_p_under)`` and refuses outright when
    the line can push and the two sides are not both there -- and its own
    docstring names this exact caller as the gap it was written for:
    "Neither pipeline.slate_board nor scheduler_worker referenced sizing."
    Flagged in review on 2026-10-30; hand-rolling it here was the
    two-definitions defect with money as its subject.

    It also refuses a row the board did not recommend, and a row with no VALID
    book -- both correct here, and neither of which the hand-rolled version
    checked.

    RESEARCH_ONLY. `advisory_sizing` stamps ``AUTO_PLACED: False`` on every
    size it returns and nothing in this project places one; this only puts the
    number where a reader of the card can see it, beside the EV it belongs to.

    Failures are swallowed deliberately: a card that cannot be sent because a
    size could not be computed is strictly worse than a card without the size.
    """
    try:
        from src.quant.advisory_sizing import advisory_size_for_candidate
    except Exception:  # noqa: BLE001 — the card matters more than the size
        logger.warning("Advisory sizing unavailable; board rows carry no size.")
        return
    for row in rows:
        row.advisory_units = None
        try:
            sized = advisory_size_for_candidate(row)
        except Exception:  # noqa: BLE001 — one unsizable row is not a failed card
            continue
        units = float(getattr(sized, "recommended_units", 0.0) or 0.0)
        # None, never 0.0: `advisory_size_for_candidate` returns 0.0 for every
        # refusal AND for a genuine non-positive edge, and a row printing "0u"
        # reads as a recommendation to stake nothing on a real edge rather
        # than as the absence of one. The notifier prints whatever is here, so
        # the distinction has to be made here.
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
    """
    An env flag, DEFAULTING rather than reading as off on an unusable value.

    ONE CONVENTION FOR ONE KIND OF VALUE, in two processes. `main._flag_env`
    recognises the off words as well as the on words and returns the DEFAULT
    for anything it recognises as neither; this function returned
    `raw in {on words}` -- i.e. **False** -- for the same input. So
    `...=maybe` meant "use the default" to the orchestrator and "off" to the
    worker.

    HOW FAR THAT ACTUALLY REACHED, stated honestly because the first version
    of this docstring overstated it: no variable is read by BOTH processes
    today. Review on #32 raised it against `PROPIQ_FORWARD_SLATE`, which the
    board branch it guarded read -- and that branch is gone, replaced by the
    projections path, so the live disagreement went with it.
    `PROPIQ_ESPN_INGEST`, `PROPIQ_DISPATCH` and
    `PROPIQ_DISPATCH_ABSTENTIONS` are this worker's alone; `main.py` never
    reads them. What remains is a trap rather than a bug: two parsers with
    the same job and different answers, one of which the next shared switch
    would be read by. Fixed here, where the next one lands, rather than left
    for it.

    A SECOND IMPLEMENTATION rather than an import, for the same reason
    `_int_env` is: `main.py` is the orchestrator and must not come to need
    APScheduler in order to read a boolean.
    `tests/test_flag_parsers_agree.py` holds the two to the same answer on
    every input, which is what makes two copies safe.
    """
    raw = (os.environ.get(name) or "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


def run_espn_ingest(
    days: Sequence[date] | None = None, *, label: str = "nightly",
    skip_game_ids: "frozenset[str] | set[str]" = frozenset(),
) -> dict[str, Any]:
    """
    Fill the panel from ESPN box scores. The only ingest that needs no laptop.

    ONE WRITE PATH, TWO CALLERS. ``days`` and ``skip_game_ids`` are the
    only things the history backfill needs to vary: every guard below it -- the namespace preflight,
    the two ownership views, the by-key read-back, the post-write namespace
    re-check -- applies identically to a night in 2024 and to last night. A
    backfill with its own copy of that sequence would be a second, quieter
    writer, and the quiet one is the one that would drift. So the backfill
    passes a date list and inherits all of it. ``None`` keeps the nightly
    behaviour exactly: the self-seeding window below.

    WHY THIS JOB EXISTS. The other two paths cannot run here: the BigDataBall
    workbook is a licensed file a human downloads, and the scraper's parquet
    needs an IP nba.com will answer. This container can reach ESPN and cannot
    reach either of those -- measured 2026-10-30 -- so without this job the
    panel can only be filled by hand, which is what kept five consecutive
    slates dead at step [2].

    SELF-SEEDING. An empty panel is read with `PROPIQ_ESPN_BACKFILL_DAYS`
    (30) rather than the nightly `PROPIQ_ESPN_LOOKBACK_DAYS` (3), because
    three days of history leaves every rolling feature null and a panel that
    looks ingested while scoring nothing is worse than an empty one. After
    the first run the short window is enough.

    ONE NAMESPACE PER PANEL, enforced here as in the CLI: ESPN ids are
    `espn:`-prefixed, `features/builder` groups rolling features by
    PLAYER_ID, and a panel holding both namespaces gives any player in both a
    split history and a confident mean over half a season. This REFUSES
    rather than mixing, and says so loudly -- there is no unattended override,
    because the operator who knows the windows do not overlap is the one who
    should pass `--allow-mixed-ids` to the CLI by hand.

    RESEARCH_ONLY. Box-score rows. Nothing here prices or places anything.
    """
    if not _flag(ENV_ESPN_INGEST, True):
        logger.info("ESPN ingest disabled (%s). Nothing fetched.", ENV_ESPN_INGEST)
        return {"status": "SKIPPED", "reason": "disabled"}

    from datetime import timedelta

    from src.utils.timezones import pacific_calendar_date

    try:
        from src.db.repository import (
            ingested_keys_present,
            ingested_row_counts,
            player_id_namespaces,
            team_game_keys,
            team_game_owners_by_date,
            team_game_sources,
            upsert_player_game_logs,
            upsert_team_game_stats,
        )
        from src.ingestion.espn_panel import (
            ESPN_TEAM_SOURCE,
            fetch_player_rows_for_dates,
        )
        from src.ingestion.scraper_panel import to_player_panel
        from src.ingestion.scraper_team_games import aggregate_team_games

        namespaces = player_id_namespaces()
        foreign = {name: n for name, n in namespaces.items() if name != "espn"}
        if foreign:
            detail = ", ".join(f"{k}: {v:,} row(s)" for k, v in sorted(foreign.items()))
            logger.error(
                "ESPN ingest REFUSED: the panel already holds non-ESPN player "
                "ids (%s). Adding ESPN rows would split those players' "
                "histories in builder's groupby(PLAYER_ID). Re-ingest that "
                "source for this window, or run scripts/load_espn_panel with "
                "--allow-mixed-ids by hand if you know they do not overlap.",
                detail,
            )
            notify_failure(
                "espn-ingest",
                f"Refused: the panel already holds non-ESPN player ids ({detail}). "
                f"Mixing them would give any player in both a split rolling "
                f"history, with no error anywhere.",
            )
            return {"status": "REFUSED", "reason": "mixed id namespaces",
                    "namespaces": namespaces}

        empty_panel = not namespaces
        if days is None:
            days_back = _int_env(
                ENV_ESPN_BACKFILL if empty_panel else ENV_ESPN_LOOKBACK,
                DEFAULT_ESPN_BACKFILL_DAYS
                if empty_panel else DEFAULT_ESPN_LOOKBACK_DAYS,
                low=1, high=400,
            )
            end = pacific_calendar_date()
            days = [end - timedelta(days=offset) for offset in range(days_back)]
            label = "empty panel, seeding" if empty_panel else label
        days = list(days)
        if not days:
            logger.info("ESPN ingest (%s): no dates to read.", label)
            return {"status": "SKIPPED", "reason": "no dates requested",
                    "label": label}
        logger.info(
            "ESPN ingest (%s): reading %d day(s), %s -> %s.",
            label, len(days), min(days), max(days),
        )

        frame, report = fetch_player_rows_for_dates(
            days, skip_game_ids=skip_game_ids,
        )
        for line in report.lines():
            logger.info("  %s", line)
        if frame.empty:
            # AN EMPTY FRAME IS NOW TWO DIFFERENT THINGS, and reporting them
            # alike would make the history backfill's best outcome read as
            # its worst. With `skip_game_ids`, a chunk whose every event is
            # already in the panel fetches nothing BY DESIGN and comes back
            # empty -- full coverage, no work to do. Reported as
            # DATA_NOT_AVAILABLE "no finished game in the window" that is
            # simply false: there were games, and they are already ingested.
            # Found in review on #31, in code this PR had just added.
            #
            # `games_seen` counts parsed summary payloads only, so a held
            # event never increments it: nothing fetched AND something held
            # is the no-op. A mixed window -- some held, some fetched and
            # refused -- still reports DATA_NOT_AVAILABLE, because something
            # was wanted and not obtained, with the held count visible so
            # the number is not a mystery.
            if report.skipped_already_held and not report.games_seen:
                logger.info(
                    "ESPN ingest (%s): nothing to fetch -- all %d event(s) "
                    "across %d date(s) are already in the panel.",
                    label, report.skipped_already_held, len(days),
                )
                return {"status": "SKIPPED",
                        "reason": "every event in the window is already in "
                                  "the panel",
                        "label": label,
                        "days": len(days),
                        "already_held": report.skipped_already_held}
            logger.warning(
                "ESPN ingest (%s): no player rows in the window.", label,
            )
            return {"status": "DATA_NOT_AVAILABLE",
                    "reason": "no finished game in the window",
                    "label": label,
                    "days": len(days),
                    "already_held": report.skipped_already_held,
                    "games_seen": report.games_seen}

        panel, _ = to_player_panel(frame)
        # BOTH OWNERSHIP VIEWS. `team_game_sources` catches a row this
        # namespace already holds; `team_game_owners_by_date` catches the
        # SAME GAME under the other namespace, which the id key cannot see.
        teams, team_report = aggregate_team_games(
            frame,
            owned_by=team_game_sources(),
            owned_by_date=team_game_owners_by_date(),
            team_source=ESPN_TEAM_SOURCE,
        )
        # The aggregation's own refusals -- short rosters, one-sided games,
        # stat gaps, rows left to a higher-precedence source. Dropped on the
        # floor until review 2026-10-31, which made incomplete team coverage
        # undiagnosable from the nightly log.
        for line in team_report.lines():
            logger.info("  %s", line)
        before = ingested_row_counts()
        upsert_player_game_logs(panel)
        if not teams.empty:
            upsert_team_game_stats(teams)
        after = ingested_row_counts()

        out = {
            "status": "OK",
            # len(days), not the env number: the backfill passes an explicit
            # list and there is no `days_back` behind it. Two definitions of
            # "how many days did this read" is how a summary starts lying.
            "days": len(days),
            "label": label,
            "games_parsed": report.games_parsed,
            "panel_rows": int(len(panel)),
            "team_games": int(len(teams)),
            "player_game_logs": after.get("player_game_logs"),
            "team_game_stats": after.get("team_game_stats"),
            "added_player_rows": (
                after.get("player_game_logs", 0) - before.get("player_game_logs", 0)
            ),
            "added_team_games": (
                after.get("team_game_stats", 0) - before.get("team_game_stats", 0)
            ),
            "skipped_not_final": len(report.skipped_not_final),
        }
        # THE TEAM WRITE IS VERIFIED BY KEY, not by the table being
        # non-empty: a count passes on rows an earlier run wrote while this
        # one's went nowhere, which is every run after the first. Same check
        # `scripts/ingest_bigdataball` and `load_scraper_panel` make.
        if not teams.empty:
            wanted = team_game_keys(teams)
            found = ingested_keys_present(team_games=wanted).get(
                "team_game_stats", 0
            )
            out["team_keys_present"] = f"{found}/{len(wanted)}"
            if found < len(wanted):
                logger.error(
                    "ESPN ingest: %d of %d team-game key(s) are NOT in the "
                    "table after the write.", len(wanted) - found, len(wanted),
                )
                notify_failure(
                    "espn-ingest",
                    f"{len(wanted) - found} of {len(wanted)} team-game key(s) "
                    f"are missing after the write; it did not survive.",
                )
                out["status"] = "FAILED"
                return out

        # AND THE NAMESPACE AGAIN, AFTER THE WRITE. The check above is a
        # preflight: another ingest can add NBA rows between it and the
        # upsert, so a concurrent run can still produce the mixed panel the
        # guard promises to prevent. `numReplicas` is 1 and `max_instances`
        # is 1, so this job cannot race ITSELF -- the exposure is a manual
        # `scripts/load_espn_panel` or an operator's own ingest landing in
        # the same seconds. Preventing that needs a database-level
        # invariant; this at least turns a silent split history into a loud
        # failure, which is the difference between a board nobody can trust
        # and a board nobody builds. Flagged in review 2026-10-31.
        settled = player_id_namespaces()
        if len([n for n in settled if n != "espn"]) > 0:
            logger.error(
                "ESPN ingest: the panel holds MIXED id namespaces after the "
                "write (%s). Another ingest wrote between the preflight check "
                "and this one. Rolling features for any player in both are "
                "now computed over a split history.", settled,
            )
            notify_failure(
                "espn-ingest",
                f"The panel holds mixed player-id namespaces after this "
                f"write ({settled}). Something else ingested concurrently. "
                f"Rolling features for any player in both are wrong until "
                f"one source is re-ingested for the whole window.",
            )
            out["status"] = "FAILED"
            out["namespaces_after"] = settled
            return out

        # READ BACK, not asserted: an upsert returns rows handed to the
        # statement. See `repository.ingested_row_counts`.
        if not after.get("player_game_logs"):
            logger.error(
                "ESPN ingest wrote %d panel row(s) and player_game_logs reads "
                "EMPTY on a fresh connection.", len(panel),
            )
            notify_failure(
                "espn-ingest",
                f"Wrote {len(panel)} panel row(s); the table reads empty "
                f"afterwards. The write did not survive.",
            )
            out["status"] = "FAILED"
            return out
        logger.info("ESPN ingest: %s", out)
        return out
    except Exception as exc:  # noqa: BLE001 — the scheduler stays up
        logger.exception("ESPN ingest raised; the scheduler stays up.")
        notify_failure("espn-ingest", f"The ESPN ingest raised: {exc}")
        return {"status": "FAILED", "error": str(exc)}


def _history_seasons() -> list[int]:
    """
    Which ESPN season years the backfill is allowed to read, oldest first.

    OFF BY DEFAULT, and that is the design rather than caution. ESPN's season
    calendar is not regular-season-only (`espn_panel.fetch_season_calendar`
    says why), so preseason and playoff dates stay permanently absent from a
    panel that refuses them per event. They would therefore read as "not yet
    ingested" on every run, for ever, and an unconditional nightly backfill
    would spend a few hundred calls re-asking them every night and never
    finish. An operator switches this on, watches it report `remaining: 0`,
    and switches it off; the job says so in its own log when it gets there.

    ESPN'S SEASON YEAR IS THE ENDING ONE: 2026 is the 2025-26 season.
    """
    raw = (os.environ.get(ENV_ESPN_HISTORY_SEASONS) or "").strip()
    if not raw:
        return []
    years: list[int] = []
    for token in re.split(r"[,\s]+", raw):
        if not token:
            continue
        try:
            year = int(token)
        except ValueError:
            logger.warning(
                "%s: %r is not a season year; ignoring it. Use ESPN's ending "
                "year, e.g. 2026 for the 2025-26 season.",
                ENV_ESPN_HISTORY_SEASONS, token,
            )
            continue
        # A bare "26" or a typo'd "20226" would otherwise be fetched and come
        # back as an empty calendar, which reads as "ESPN changed shape".
        if not 1947 <= year <= 2100:
            logger.warning(
                "%s: %d is not a plausible NBA season year; ignoring it.",
                ENV_ESPN_HISTORY_SEASONS, year,
            )
            continue
        if year not in years:
            years.append(year)
    return sorted(years)


def plan_history_dates(
    seasons: Sequence[int],
    held: Mapping[str, set[str]],
    *,
    chunk: int,
    calendar_for: Any,
    window_for: Any,
    not_after: date,
) -> dict[str, Any]:
    """
    The next ``chunk`` game dates to read, absent ones before present ones.

    SEPARATED FROM THE JOB so the plan can be asserted without a network
    call or a database: which dates get picked, in which order, and what is
    left over is the whole behaviour worth testing, and it is unreachable
    through a function that fetches and writes.

    TWO FILTERS, AND THEY EXIST SO THE JOB DOES NOT CHASE DATES IT CANNOT
    FILL. `calendar_for` lists every date with any NBA game in a season, so
    preseason and the playoffs are in it and the per-event guard refuses
    both.

      * ``window_for(year)`` gives the regular season's own first and last
        Pacific date, from ESPN. Outside it is not this job's to fetch.
      * ``not_after`` drops dates not yet played. A game today is not final
        and tomorrow's is not played at all.

    THEN TWO BUCKETS, NOT A FILTER, and this is the part review on #31
    changed twice. A date is NOT skipped because the panel holds a row for
    it: `fetch_player_rows` records a failed event and keeps the others, so
    a night that lost one game to a 404 is in the panel and incomplete. The
    old version skipped it for ever, leaving a silent hole in a rolling
    window.

    So dates with NO rows come first -- they are certainly incomplete --
    and dates that already have rows follow, to be re-examined. A
    re-examined date that is complete costs ONE scoreboard call and no
    summaries, because the caller passes the game ids it already holds and
    `fetch_player_rows` skips them.

    WHY THERE IS NO "COMPLETE" VERDICT HERE ANY MORE. Some dates can never
    produce rows and cannot be distinguished from unread ones without
    recording every attempt: All-Star Sunday is the measured example --
    2026-02-15 carries four events that ESPN labels season type 2, whose
    team codes are WORLD, STARS and STRIPES, which `normalize_team` refuses
    because they are not NBA teams. That date is in the calendar, inside the
    regular-season window, played, and permanently unfillable. A planner
    that promised `remaining: 0` would be lying about it, as two earlier
    versions of this one did. What the caller gets instead is coverage it
    can report honestly.
    """
    seen: set[date] = set()
    absent: list[date] = []
    present: list[date] = []
    season_notes: list[str] = []
    for year in sorted({int(y) for y in seasons}):
        start, end = window_for(year)
        days = list(calendar_for(year))
        candidates = [d for d in days if start <= d <= end and d <= not_after]
        n_absent = 0
        for day in candidates:
            if day in seen:
                continue
            seen.add(day)
            if held.get(day.isoformat()):
                present.append(day)
            else:
                absent.append(day)
                n_absent += 1
        season_notes.append(
            f"{year}: regular season {start} to {end}, {len(days)} game "
            f"date(s) in the calendar, {len(candidates)} played in season, "
            f"{len(candidates) - n_absent} with panel rows, {n_absent} with none"
        )
    absent.sort()
    present.sort()
    ordered = absent + present
    take = ordered[: max(int(chunk), 0)]
    return {
        "dates": take,
        "absent": len(absent),
        "present": len(present),
        "candidates": len(ordered),
        "seasons": season_notes,
    }


def run_history_backfill() -> dict[str, Any]:
    """
    Extend the panel BACKWARDS, one bounded chunk of game dates per run.

    WHY THE PANEL NEEDS THIS AT ALL. The 02:00 ESPN ingest reads three days
    and seeds thirty, which is this season to date -- about 1,830
    player-game rows on 2026-11-01. A per-market over/under model fitted on
    one season of a league that has nine on record is the thinnest version
    of itself, and `models/labels.py` measured the direction: the calibrated
    Brier degrades monotonically as the training window's start date moves
    forward. More history is better, and nothing in this deployment could
    reach any. The licensed workbook is a file a human downloads and
    stats.nba.com does not answer this container, so ESPN is the only path
    to history that runs unattended -- the same argument that put the
    nightly ingest here.

    WHAT IT DOES NOT DO. It does not train, score, or publish anything, and
    it writes nothing a nightly ingest would not write: the same mapper, the
    same ownership guards, the same by-key read-back, through
    `run_espn_ingest`. It is strictly historical, so there is no as-of
    hazard in it -- every date it reads is before every date the model will
    be asked to predict.

    BOUNDED AND RESUMABLE, BY GAME RATHER THAN BY DATE. The resume point is
    `panel_game_ids_by_date`, read from the database rather than remembered
    anywhere, and the ids travel to the fetch so an event already held
    costs nothing. A run that dies half way costs its chunk and nothing
    else.

    IT REPORTS COVERAGE AND NEVER CLAIMS TO BE FINISHED. Two earlier
    versions logged HISTORY BACKFILL COMPLETE, and in both the branch was
    unreachable -- the first counted preseason and playoff dates as
    missing, the second still counted dates that can never produce rows at
    all (All-Star Sunday: four events ESPN labels regular season, played by
    WORLD, STARS and STRIPES, which `normalize_team` refuses). Telling an
    operator to wait for a signal that cannot arrive is worse than telling
    them the number and letting them decide. So this logs how many dates
    have rows and how many do not, says plainly that some of the latter may
    never fill, and asks them to switch the job off once the figure stops
    moving.

    RESEARCH_ONLY. Finished box scores from games already played.
    """
    seasons = _history_seasons()
    if not seasons:
        logger.debug(
            "History backfill not configured (%s unset).", ENV_ESPN_HISTORY_SEASONS,
        )
        return {"status": "SKIPPED", "reason": f"{ENV_ESPN_HISTORY_SEASONS} unset"}
    if not _flag(ENV_ESPN_INGEST, True):
        # The same switch the nightly reads. An operator who turned the ESPN
        # path off has turned OFF the ESPN path, and a backfill that kept
        # reading it would be the one job that ignored them.
        logger.info(
            "History backfill skipped: the ESPN path is disabled (%s).",
            ENV_ESPN_INGEST,
        )
        return {"status": "SKIPPED", "reason": "espn ingest disabled"}

    # THE 60-MINUTE LEAD IS NOT A SUBSTITUTE FOR THE LOCK, and for one
    # deploy it was the only thing standing here. `history_backfill_time`
    # guarantees this job STARTS at least an hour before the slate; nothing
    # made it FINISH. An overrun writes seasons of box scores into
    # `player_game_logs` while the slate reads them, and that function's own
    # docstring already named the cost: "the slate would score a partially
    # written panel, exit 0, and say nothing -- `max_instances` is per job,
    # so it does not help here." A schedule is a prediction; a lock is a
    # guarantee.
    #
    # THE SAME LOCK THE SLATE TAKES (`src/utils/write_lock.py`, A1), so the
    # two cannot interleave whatever either one's duration turns out to be.
    # Taken AFTER the config guards on purpose: a backfill that is switched
    # off should not touch the lock at all, and this job runs every night
    # whether or not it is configured.
    #
    # DECLINING IS THE POINT, and it is the direction that trades a silent
    # wrong answer for a loud missing one. If the slate holds the lock, this
    # run is skipped and tomorrow's picks up the same dates -- the plan is
    # derived from what the panel holds, so nothing is lost by waiting. The
    # reverse (the slate declining) costs one morning's board, which is
    # visible in the channel. Neither is a failure, so neither alerts.
    from src.utils.write_lock import write_lock

    with write_lock() as acquired:
        if not acquired:
            logger.warning(
                "History backfill SKIPPED: another process holds the slate "
                "write lock. Nothing was read and nothing was written. "
                "Tomorrow's run plans from the panel again and picks up the "
                "same dates. A deliberate decline, not a failure."
            )
            return {
                "status": "SKIPPED",
                "reason": "another process holds the slate write lock",
                "seasons": seasons,
            }
        return _run_history_backfill_locked(seasons)


def _run_history_backfill_locked(seasons: list[int]) -> dict[str, Any]:
    """
    The backfill body, with the slate write lock already held.

    Split out for the reason `_run_slate_locked` is: the acquisition stays
    one short, readable block instead of indenting the whole job, and the
    decline path is impossible to miss when reading the caller.
    """
    chunk = _int_env(
        ENV_ESPN_HISTORY_CHUNK, DEFAULT_ESPN_HISTORY_CHUNK_DAYS, low=1, high=400,
    )
    try:
        from datetime import timedelta

        from src.db.repository import panel_game_ids_by_date
        from src.ingestion.espn_panel import (
            ESPN_SOURCE,
            fetch_regular_season_window,
            fetch_season_calendar,
        )
        from src.utils.timezones import pacific_calendar_date

        held = panel_game_ids_by_date(ESPN_SOURCE)
        # YESTERDAY, not today. Tonight's games are not final, and the
        # nightly ingest already re-reads the last three days -- so the
        # backfill stopping here costs nothing and keeps it from queueing a
        # date that cannot be filled yet.
        plan = plan_history_dates(
            seasons, held, chunk=chunk,
            calendar_for=fetch_season_calendar,
            window_for=fetch_regular_season_window,
            not_after=pacific_calendar_date() - timedelta(days=1),
        )
    except Exception as exc:  # noqa: BLE001 — the scheduler stays up
        logger.exception("History backfill could not plan; the scheduler stays up.")
        notify_failure("espn-backfill", f"The history backfill could not plan: {exc}")
        return {"status": "FAILED", "error": str(exc), "seasons": seasons}

    for line in plan["seasons"]:
        logger.info("  season %s", line)

    if not plan["dates"]:
        logger.info(
            "History backfill: no played regular-season date in season(s) %s "
            "to read. Nothing to do.",
            ", ".join(str(y) for y in seasons),
        )
        return {"status": "SKIPPED", "reason": "no candidate dates",
                "seasons": seasons, "absent": plan["absent"],
                "present": plan["present"]}

    first, last = plan["dates"][0], plan["dates"][-1]
    logger.info(
        "History backfill: %d date(s) this run (%s -> %s). Across season(s) "
        "%s: %d candidate date(s), %d with panel rows, %d with none.",
        len(plan["dates"]), first, last,
        ", ".join(str(y) for y in seasons),
        plan["candidates"], plan["present"], plan["absent"],
    )
    # THE GAME IDS TRAVEL WITH THE REQUEST, which is what makes a
    # re-examined date cheap: one scoreboard call, and no summary for an
    # event the panel already holds.
    #
    # EVERY HELD ID, not just this chunk's dates. The first version
    # narrowed the set by date, and that narrowing was unobservable: ESPN
    # event ids are globally unique, so an id belonging to another night
    # can never match an event this chunk fetches. A mutation removing the
    # narrowing changed no test, which is the signal that it was not
    # earning its lines -- so it is gone rather than left looking load-
    # bearing.
    skip_game_ids: set[str] = set()
    for ids in held.values():
        skip_game_ids |= ids
    # THE WRITE IS run_espn_ingest's, not a second copy of it. Its own broad
    # handler owns a failure here, and it alerts with the same `espn-ingest`
    # key the nightly uses -- a backfill chunk failing for the reason a
    # nightly would is not a different incident.
    out = run_espn_ingest(
        days=plan["dates"], label="history backfill",
        skip_game_ids=skip_game_ids,
    )
    out["backfill"] = {
        "seasons": seasons,
        "chunk": chunk,
        "dates_this_run": len(plan["dates"]),
        "first": first.isoformat(),
        "last": last.isoformat(),
        "candidates": plan["candidates"],
        "with_rows": plan["present"],
        "without_rows": plan["absent"],
        "already_held_events": len(skip_game_ids),
    }
    if not plan["absent"]:
        # COVERAGE, NOT COMPLETION. Every candidate date has rows; what is
        # left is whether any of them is short a game, which this run's
        # re-examination answers for its own chunk and not for the rest.
        logger.info(
            "History backfill: every one of the %d candidate date(s) in "
            "season(s) %s has panel rows. The job is now only re-examining "
            "them for gaps, a scoreboard call each. UNSET $%s when you are "
            "satisfied -- it will not announce itself finished, because a "
            "date that can never produce rows (All-Star Sunday, whose teams "
            "are WORLD/STARS/STRIPES) is indistinguishable from one never "
            "read without recording every attempt.",
            plan["candidates"], ", ".join(str(y) for y in seasons),
            ENV_ESPN_HISTORY_SEASONS,
        )
    return out


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


def _training_markets() -> list[str]:
    """The markets to fit, or an empty list when the job is not configured."""
    from src.models.training import MARKETS

    raw = (os.environ.get(ENV_TRAIN_MARKETS) or "").strip()
    if not raw:
        return []
    wanted: list[str] = []
    unknown: list[str] = []
    for part in raw.replace(";", ",").split(","):
        name = part.strip().upper()
        if not name:
            continue
        if name == "ALL":
            wanted = list(MARKETS)
            continue
        if name in MARKETS:
            if name not in wanted:
                wanted.append(name)
        else:
            unknown.append(name)
    if unknown:
        # NAMED AND DROPPED, never silently ignored: "PTS,POINTS" fitting one
        # market while the operator believes two were fitted is the shape of
        # defect this project keeps finding.
        logger.error(
            "%s names %s, which %s not a launch market; fitting %s. The launch "
            "markets are %s, or ALL.",
            ENV_TRAIN_MARKETS, unknown, "is" if len(unknown) == 1 else "are",
            wanted or "nothing", ", ".join(MARKETS),
        )
    return wanted


def training_is_needed_now(markets: list[str]) -> list[str]:
    """
    The configured markets that have NO resolvable artifact yet.

    WHY A BOOT FIT EXISTS AT ALL. The daily job runs at 07:00 PT, so turning
    training on at any other hour left the deployment in its old state --
    every row abstaining -- until the next morning, for no reason other than
    the clock. The first fit is the one that matters most, because it is the
    difference between a board that can score and one that cannot.

    SELF-LIMITING, WHICH IS WHY IT NEEDS NO SWITCH OF ITS OWN. It asks the
    SAME resolver the slate asks (`main.resolve_model_artifact`), so once an
    artifact exists this returns nothing and no boot fit is scheduled. A
    redeploy loop cannot turn into a training loop.

    A path that resolves but does not EXIST counts as missing: `PROPIQ_MODEL`
    pointing at a file that is not there, or the legacy `models/` default,
    both resolve to a path and neither can be scored with.
    """
    if not markets:
        return []
    try:
        import main as propiq_main
    except Exception as exc:  # noqa: BLE001 — a boot fit is never worth a crash
        logger.warning("Could not check for an existing artifact: %s", exc)
        return []
    missing: list[str] = []
    for market in markets:
        try:
            path, _how = propiq_main.resolve_model_artifact(market=market)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Artifact check for %s failed: %s", market, exc)
            continue
        if path is None or not Path(path).exists():
            missing.append(market)
    return missing


def training_time(slate_hour: int, slate_minute: int) -> tuple[int, int]:
    """
    When training runs: a fixed lead before the slate, with no override.

    DELIBERATELY NOT CONFIGURABLE, unlike the backfill's time. Training takes
    the same write lock the slate takes, so a time an operator could move on
    top of the slate would cost the morning's board -- and the backfill
    already needed `history_backfill_time` to refuse exactly that. One fewer
    knob is one fewer way to lose a card.

    It lands after the backfill (which runs at a 60-minute minimum lead) and
    before the slate, so the fit sees the rows the backfill just wrote.
    """
    slate_minutes = slate_hour * 60 + slate_minute
    return divmod((slate_minutes - TRAINING_MIN_LEAD_MINUTES) % (24 * 60), 60)


def run_training() -> dict[str, Any]:
    """
    Fit the scoring artifacts, from the panel in THIS database.

    WHY THE WORKER TRAINS AT ALL. Every row of every board has read
    "DATA_NOT_AVAILABLE: cannot resolve P(over)" since this deployment
    started, for one reason: no artifact exists. `resolve_model_artifact`
    looks in the comparison config's `artifacts_dir`, nothing had ever
    written there, and the only command that writes there --
    `nba_model_cli train-stats` -- needs a feature matrix. The matrix needs
    the panel, the panel is in this database, and the two paths that reach a
    panel from outside both need something this container does not have: a
    licensed workbook a human downloads, or stats.nba.com, which times out
    here. So the remedy was always "run something on your own machine", and
    the artifact could not get back onto the volume if you did.

    TRAINING WHERE THE DATA IS ALSO FIXES THE CONTRACT. `artifacts_dir` is
    relative, so it resolves under the mounted volume and survives a
    redeploy. More importantly the matrix is built here by
    `main.build_features_and_verify_fatigue` -- the SAME function, over the
    same frames, that the slate scores against -- so the artifact's
    feature contract cannot name a column the live matrix lacks. That
    mismatch is not hypothetical: `market_lines` is empty in this
    deployment, so MKT_* and BLOWOUT_* are absent from both, and an
    artifact trained elsewhere with a workbook would abstain on every row
    while looking perfectly healthy.

    THE HOLDOUT IS THE NEWEST SLICE, not a random sample and not the oldest
    two thirds. A model is asked about tomorrow; the only honest check is on
    games played after everything it learned from. `assert_no_lookahead`
    runs inside the matrix build, every rolling feature is already
    `.shift(1)`-ed by the builder, and the split is by date -- so there is
    no path by which a fitted row sees its own game.

    WHAT IT IS NOT. It is not a claim that the result is profitable, or even
    useful. The label is `over_hit` against `{MARKET}_L10`, a research stand-
    in, because historical book lines do not exist for these seasons; the
    `line_aware` component augments across line offsets to make serving at a
    posted line meaningful, and that is a mitigation, not a measurement. The
    publication gate still holds every number back until 100 graded
    predictions clear its calibration bar. Training only lets that counter
    start moving.

    OFF UNLESS CONFIGURED ($PROPIQ_TRAIN_MARKETS), like the backfill.

    RESEARCH_ONLY. Fits a classifier, writes files. Prices nothing.
    """
    markets = _training_markets()
    if not markets:
        logger.debug("Training not configured (%s unset).", ENV_TRAIN_MARKETS)
        return {"status": "SKIPPED", "reason": f"{ENV_TRAIN_MARKETS} unset"}

    # THE LOCK, AFTER THE CONFIG GUARD, for the backfill's reasons exactly: a
    # job that is switched off should not touch the lock, and this one runs
    # every night whether or not it is configured. Training READS the panel
    # the slate reads and WRITES the artifacts the slate loads, so an overlap
    # could hand the slate a half-written model file. Declining is the safe
    # direction -- a day without a new fit costs nothing, because yesterday's
    # artifact is still there and still resolves.
    from src.utils.write_lock import write_lock

    with write_lock() as acquired:
        if not acquired:
            logger.warning(
                "Training SKIPPED: another process holds the slate write lock. "
                "Yesterday's artifact still resolves, so nothing is lost but a "
                "refit; tomorrow's run picks up the same panel plus a day.",
            )
            return {"status": "SKIPPED",
                    "reason": "another process holds the slate write lock",
                    "markets": markets}
        return _run_training_locked(markets)


def _run_training_locked(markets: list[str]) -> dict[str, Any]:
    """The body of `run_training`, with the write lock already held."""
    import pandas as pd

    import main as propiq_main
    from src.db.repository import (
        load_game_market_lines,
        load_player_panel,
        load_team_game_stats,
    )
    from src.models.compare import load_comparison_config
    from src.models.training import TrainingDataError, fit_market_artifacts
    from src.utils.timezones import pacific_calendar_date

    slate_date = str(pacific_calendar_date())
    lookback = _int_env(
        ENV_TRAIN_LOOKBACK_DAYS, DEFAULT_TRAIN_LOOKBACK_DAYS, low=1, high=4000,
    )
    holdout_days = _int_env(
        ENV_TRAIN_HOLDOUT_DAYS, DEFAULT_TRAIN_HOLDOUT_DAYS, low=1, high=365,
    )

    panel = load_player_panel(slate_date=slate_date, lookback_days=lookback)
    if panel is None or panel.empty:
        logger.error(
            "TRAINING FOUND NO PANEL in the %d day(s) before %s. The ESPN "
            "ingest and the history backfill are what fill it; this job only "
            "reads it.", lookback, slate_date,
        )
        return {"status": "FAILED", "reason": "empty panel", "markets": markets}

    team_games = load_team_game_stats(slate_date=slate_date, lookback_days=lookback)
    market_lines = load_game_market_lines(slate_date=slate_date, lookback_days=lookback)
    features = propiq_main.build_features_and_verify_fatigue(
        panel, team_games=team_games, market_lines=market_lines,
    )

    # THE CUTOFF IS COMPUTED FROM THE PANEL, not from today. A panel whose
    # newest game is a week old would otherwise lose a week of its holdout to
    # days with no games in them, and in the off-season the holdout would be
    # empty while the fit set looked fine.
    newest = pd.to_datetime(features["GAME_DATE"]).max()
    train_end = (newest - pd.Timedelta(days=holdout_days)).date().isoformat()
    artifacts_dir = Path(str(
        (load_comparison_config() or {}).get(
            "artifacts_dir", "data/external/model_runs/comparison"
        )
    ))

    reports: dict[str, Any] = {}
    failures: dict[str, str] = {}
    for market in markets:
        try:
            reports[market] = fit_market_artifacts(
                features, market, train_end=train_end, artifacts_dir=artifacts_dir,
            )
        except TrainingDataError as exc:
            # ONE MARKET'S DATA PROBLEM IS NOT THE RUN'S. REB can fit while
            # PTS cannot: PTS needs FGA and FTA for its efficiency features
            # and the other two do not, so a panel missing those columns
            # fails exactly one market -- which is worth seeing by name.
            logger.error("Training %s: %s", market, exc)
            failures[market] = str(exc)
        except Exception as exc:  # noqa: BLE001 — one market, not the run
            logger.exception("Training %s raised", market)
            failures[market] = f"{type(exc).__name__}: {exc}"

    status = "OK" if reports and not failures else ("PARTIAL" if reports else "FAILED")
    out = {
        "status": status,
        "panel_rows": int(len(panel)),
        "feature_rows": int(len(features)),
        "newest_game": str(newest.date()),
        "train_end": train_end,
        "holdout_days": holdout_days,
        "artifacts_dir": str(artifacts_dir),
        "markets": reports,
        "failed": failures,
    }
    logger.info("Training: %s", out)
    if status != "OK":
        logger.error(
            "TRAINING DID NOT FIT EVERY MARKET: %s. Every unfitted market "
            "still scores nothing, so its rows keep abstaining.", failures,
        )
    return out


def history_backfill_time(slate_hour: int, slate_minute: int) -> tuple[int, int]:
    """
    When the history backfill runs, REFUSING a time that races the slate.

    THE OVERRIDE COULD PUT A WRITER ON TOP OF A READER. `_int_env` bounds
    each field to a legal clock value and nothing compared the result with
    the slate's: `PROPIQ_ESPN_HISTORY_HOUR_PT=9` against a 09:00 slate
    scheduled a job that walks seasons of box scores into
    `player_game_logs` at the moment the slate starts reading them. The
    slate would score a partially written panel, exit 0, and say nothing --
    `max_instances` is per job, so it does not help here. Found in review
    on #31, where the only thing standing between an operator and that was
    a comment.

    REFUSED BACK TO THE DEFAULT rather than raised, following `_int_env`:
    a worker that will not boot over a schedule setting is worse than one
    that boots on the documented time and says loudly what it ignored. If
    the DEFAULT is itself unsafe -- an operator moved the slate to 05:00 --
    the backfill is moved to keep its lead, because the slate is the job
    with the deadline.
    """
    minutes = _int_env(
        ENV_ESPN_HISTORY_HOUR, DEFAULT_ESPN_HISTORY_HOUR, low=0, high=23,
    ) * 60 + _int_env(
        ENV_ESPN_HISTORY_MINUTE, DEFAULT_ESPN_HISTORY_MINUTE, low=0, high=59,
    )
    slate_minutes = slate_hour * 60 + slate_minute
    latest = slate_minutes - ESPN_HISTORY_MIN_LEAD_MINUTES
    if minutes <= latest:
        return minutes // 60, minutes % 60

    if latest >= 0:
        # The slate leaves room earlier in the same day; take it.
        fallback = min(
            DEFAULT_ESPN_HISTORY_HOUR * 60 + DEFAULT_ESPN_HISTORY_MINUTE, latest
        )
    else:
        # A slate inside the first hour of the day leaves no room before it,
        # so the backfill goes AFTER it instead -- still the longest gap
        # available, and never overlapping.
        fallback = (slate_minutes + ESPN_HISTORY_MIN_LEAD_MINUTES) % (24 * 60)
    logger.error(
        "HISTORY BACKFILL TIME REFUSED: %02d:%02d PT is not at least %d "
        "minute(s) before the slate at %02d:%02d PT. A backfill chunk takes "
        "about twenty minutes and writes player_game_logs, which the slate "
        "READS -- overlapping them scores a partially written panel and exits "
        "0. Using %02d:%02d PT instead; set $%s / $%s to a time that clears "
        "the slate.",
        minutes // 60, minutes % 60, ESPN_HISTORY_MIN_LEAD_MINUTES,
        slate_hour, slate_minute, fallback // 60, fallback % 60,
        ENV_ESPN_HISTORY_HOUR, ENV_ESPN_HISTORY_MINUTE,
    )
    return fallback // 60, fallback % 60


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
        run_espn_ingest,
        trigger="cron",
        hour=_int_env(ENV_ESPN_HOUR, DEFAULT_ESPN_HOUR, low=0, high=23),
        minute=_int_env(ENV_ESPN_MINUTE, DEFAULT_ESPN_MINUTE, low=0, high=59),
        timezone=DISPLAY_TZ,
        id="espn_ingest",
        name="Fill the panel from ESPN box scores",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=ESPN_MISFIRE_GRACE_SECONDS,
        replace_existing=True,
    )
    history_hour, history_minute = history_backfill_time(slate_hour, slate_minute)
    scheduler.add_job(
        run_history_backfill,
        trigger="cron",
        hour=history_hour,
        minute=history_minute,
        timezone=DISPLAY_TZ,
        id="espn_history_backfill",
        name="Extend the panel backwards from ESPN (off unless configured)",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=ESPN_HISTORY_MISFIRE_GRACE_SECONDS,
        replace_existing=True,
    )
    # THE FIRST FIT DOES NOT WAIT FOR TOMORROW MORNING.
    #
    # Scheduled rather than run inline, and that is the whole point: a fit
    # takes minutes, and doing it before `scheduler.start()` would hold the
    # worker down for that long -- a boot at 08:55 would miss the 09:00
    # slate entirely. As a date job it runs on the scheduler's own pool with
    # the worker already up, and `run_training` takes the slate's write lock,
    # so the two cannot interleave however long the fit turns out to take.
    #
    # It asks the resolver, so it is a no-op on every boot after the first.
    from datetime import timedelta

    boot_markets = training_is_needed_now(_training_markets())
    if boot_markets:
        logger.warning(
            "NO ARTIFACT RESOLVES FOR %s, and training is configured. Fitting "
            "once now rather than waiting for %02d:00 PT: until an artifact "
            "exists every board row abstains and the publication gate's "
            "graded counter cannot move. Later boots find the artifact and "
            "skip this.",
            ", ".join(boot_markets), training_time(slate_hour, slate_minute)[0],
        )
        scheduler.add_job(
            run_training,
            trigger="date",
            run_date=(
                datetime.now(tz=DISPLAY_TZ)
                + timedelta(seconds=TRAINING_BOOT_DELAY_SECONDS)
            ),
            id="training_boot",
            name="Fit the scoring artifacts once, because none exist",
            max_instances=1,
            coalesce=True,
            misfire_grace_time=TRAINING_MISFIRE_GRACE_SECONDS,
            replace_existing=True,
        )
    training_hour, training_minute = training_time(slate_hour, slate_minute)
    scheduler.add_job(
        run_training,
        trigger="cron",
        hour=training_hour,
        minute=training_minute,
        timezone=DISPLAY_TZ,
        id="training",
        name="Fit the scoring artifacts from the panel (off unless configured)",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=TRAINING_MISFIRE_GRACE_SECONDS,
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
