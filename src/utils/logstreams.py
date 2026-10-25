"""
src/utils/logstreams.py — INFO belongs on stdout, not stderr.

WHY THIS EXISTS. `logging.basicConfig()` with no `stream=` attaches a
StreamHandler to **stderr**, and Railway classifies everything a container
writes to stderr as `severity: "error"`. So the deployment's Observability tab
renders lines like

    INFO propiq.scheduler | PropIQ worker up at 2026-10-23T23:13:52
    INFO apscheduler.scheduler | Scheduler started
    INFO src.notify.discord | discord: delivered 1 embed(s)

in red, as errors. On 2026-10-25 an operator reading that tab reported
"containers failing to start, worker crashes" for a deployment that had been
up for 35 hours without a restart, had every healthcheck green except the two
data ones, and had peaked at 0.4 GB of an 8 GB limit. The logs were the only
evidence of a fault, and they were lying about their own severity.

Splitting the streams at WARNING makes the platform's severity mean what it
says: a red line is something that went wrong.

ONE READER OF THIS FACT. There were two `logging.basicConfig` calls, in
`main.py` and `scheduler_worker.py`, and because `basicConfig` is a no-op once
the root logger has handlers, which one took effect depended on import order
(`scheduler_worker` imports `main`, whose call runs at module scope, so the
worker's own call never did anything). That is the defect this file removes as
much as the stream choice.

RESEARCH_ONLY.
"""

from __future__ import annotations

import logging
import os
import sys

#: The format both call sites already used, kept byte-identical so existing
#: log-scraping and the deployment's own docs stay true.
LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s | %(message)s"

#: Records at or above this go to stderr; everything below goes to stdout.
STDERR_FROM = logging.WARNING

#: Marks the root logger as configured by us, so a second call is a no-op
#: rather than a second pair of handlers printing everything twice.
_SENTINEL = "_propiq_streams_configured"


class _BelowWarning(logging.Filter):
    """Let only sub-WARNING records through, so the two handlers partition."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        return record.levelno < STDERR_FROM


def configure_logging(level: str | int | None = None, *, force: bool = False) -> None:
    """
    Send INFO and DEBUG to stdout, WARNING and above to stderr.

    Idempotent: calling it from both entry points is safe, which is the point
    — `basicConfig`'s silent no-op on a configured root is what made the
    second call dead code before.
    """
    root = logging.getLogger()
    if getattr(root, _SENTINEL, False) and not force:
        return

    if level is None:
        level = os.environ.get("LOG_LEVEL", "INFO")
    if isinstance(level, str):
        level = level.strip().upper() or "INFO"
        # An unknown name would make `setLevel` raise and take the process
        # down at import time, which is a bad trade for a typo'd env var.
        level = getattr(logging, level, logging.INFO)
        if not isinstance(level, int):
            level = logging.INFO

    for handler in list(root.handlers):
        root.removeHandler(handler)
        # Only close handlers we are replacing wholesale, and never pytest's
        # or anything else's: `force` is the opt-in for that.
        if force and handler not in (None,):
            try:
                handler.close()
            except Exception:  # noqa: BLE001 — closing a handler must not fail a boot
                pass

    formatter = logging.Formatter(LOG_FORMAT)

    out = logging.StreamHandler(stream=sys.stdout)
    out.setFormatter(formatter)
    out.addFilter(_BelowWarning())

    err = logging.StreamHandler(stream=sys.stderr)
    err.setFormatter(formatter)
    err.setLevel(STDERR_FROM)

    root.addHandler(out)
    root.addHandler(err)
    root.setLevel(level)
    setattr(root, _SENTINEL, True)
