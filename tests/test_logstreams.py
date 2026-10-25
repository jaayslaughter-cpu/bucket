"""
tests/test_logstreams.py — a red log line should mean something went wrong.

WHY THIS EXISTS. Railway classifies everything a container writes to stderr as
`severity: "error"`, and `logging.basicConfig()` attaches its handler to
stderr. So on 2026-10-25 a deployment that had been up for 35 hours, with
migrations applied, the volume writable, the database reachable and Discord
delivering, presented in the Observability tab as an unbroken wall of errors —
"Scheduler started", "discord: delivered 1 embed(s)", "PropIQ worker up" — and
was reported as a crashing container. Nothing was crashing. The logs were
misreporting their own severity.

These tests assert on WHERE A RECORD ACTUALLY LANDS, by capturing the two
streams, rather than on whether a handler was installed: the handler could be
installed on the wrong stream and every assertion about its existence would
still pass.

RESEARCH_ONLY.
"""

from __future__ import annotations

import io
import logging

import pytest

from src.utils.logstreams import (
    LOG_FORMAT,
    STDERR_FROM,
    configure_logging,
)


@pytest.fixture
def streams(monkeypatch):
    """Configure logging against two in-memory streams and return them."""
    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr("sys.stdout", out)
    monkeypatch.setattr("sys.stderr", err)

    root = logging.getLogger()
    saved_handlers, saved_level = list(root.handlers), root.level
    saved_flag = getattr(root, "_propiq_streams_configured", False)
    try:
        configure_logging("DEBUG", force=True)
        yield out, err
    finally:
        for h in list(root.handlers):
            root.removeHandler(h)
        for h in saved_handlers:
            root.addHandler(h)
        root.setLevel(saved_level)
        root._propiq_streams_configured = saved_flag


def test_info_goes_to_stdout_and_not_stderr(streams):
    """THE WHOLE POINT. An INFO line on stderr is a red line in Railway's
    dashboard for something that went right."""
    out, err = streams
    logging.getLogger("propiq.test").info("worker up")

    assert "worker up" in out.getvalue()
    assert "worker up" not in err.getvalue(), (
        "an INFO record reached stderr, which Railway reports as severity "
        "'error' — this is the exact defect the module exists to fix"
    )


def test_debug_also_goes_to_stdout(streams):
    out, err = streams
    logging.getLogger("propiq.test").debug("ingest detail")
    assert "ingest detail" in out.getvalue()
    assert "ingest detail" not in err.getvalue()


@pytest.mark.parametrize("level,call", [
    (logging.WARNING, "warning"),
    (logging.ERROR, "error"),
    (logging.CRITICAL, "critical"),
])
def test_warning_and_above_still_go_to_stderr(streams, level, call):
    """The other half: a real problem must stay on the stream the platform
    treats as an error, or splitting the streams would hide failures."""
    out, err = streams
    getattr(logging.getLogger("propiq.test"), call)("the thing broke")

    assert "the thing broke" in err.getvalue()
    assert "the thing broke" not in out.getvalue(), (
        "a WARNING+ record reached stdout, so a real fault would no longer "
        "be flagged as one"
    )
    assert level >= STDERR_FROM


def test_the_split_is_a_partition_and_never_duplicates(streams):
    """Two handlers over one stream pair is how a log doubles. Every record
    must land exactly once, across both streams."""
    out, err = streams
    log = logging.getLogger("propiq.test")
    log.info("once-info")
    log.error("once-error")

    combined = out.getvalue() + err.getvalue()
    assert combined.count("once-info") == 1
    assert combined.count("once-error") == 1


def test_the_format_is_unchanged(streams):
    """The format string both entry points used is kept byte-identical, so
    anything reading these logs keeps working."""
    assert LOG_FORMAT == "%(asctime)s %(levelname)-8s %(name)s | %(message)s"
    out, _ = streams
    logging.getLogger("propiq.fmt").info("shaped")
    line = out.getvalue().strip()
    assert "INFO" in line
    assert "propiq.fmt" in line
    assert "| shaped" in line


# --- idempotence, which is why the second call site was dead code --------


def test_calling_it_twice_does_not_double_every_line(monkeypatch):
    """
    `basicConfig` is a silent no-op once the root logger has handlers, which
    is why `scheduler_worker`'s call never did anything — `main` is imported
    first and calls it at module scope. The replacement has to be safe to
    call from both, which means the second call must not add a second pair of
    handlers.
    """
    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr("sys.stdout", out)
    monkeypatch.setattr("sys.stderr", err)
    root = logging.getLogger()
    saved, saved_level = list(root.handlers), root.level
    saved_flag = getattr(root, "_propiq_streams_configured", False)
    try:
        configure_logging("INFO", force=True)
        before = len(root.handlers)
        configure_logging("INFO")
        configure_logging("INFO")
        assert len(root.handlers) == before

        logging.getLogger("propiq.test").info("single")
        assert (out.getvalue() + err.getvalue()).count("single") == 1
    finally:
        for h in list(root.handlers):
            root.removeHandler(h)
        for h in saved:
            root.addHandler(h)
        root.setLevel(saved_level)
        root._propiq_streams_configured = saved_flag


def test_a_second_call_leaves_another_library_s_handler_alone():
    """
    THE PROPERTY THE SENTINEL ACTUALLY PROVIDES, and the one my first
    idempotence test missed. Removing the guard did NOT fail that test,
    because `configure_logging` also strips existing handlers on every call,
    so the handler count stayed at two either way and nothing doubled.

    What the guard really prevents is a second call tearing out a handler
    somebody ELSE installed — pytest's `caplog`, or a host's log shipper.
    `main` calls this at module scope, so any later call is a second call,
    and silently dropping caplog's handler would make tests stop seeing
    records while still passing.
    """
    root = logging.getLogger()
    saved, saved_level = list(root.handlers), root.level
    saved_flag = getattr(root, "_propiq_streams_configured", False)
    try:
        configure_logging("INFO", force=True)

        theirs = logging.StreamHandler(stream=io.StringIO())
        root.addHandler(theirs)
        count_with_theirs = len(root.handlers)

        configure_logging("INFO")  # the second call, as main's import makes it

        assert theirs in root.handlers, (
            "the second call removed a handler this module did not install"
        )
        assert len(root.handlers) == count_with_theirs
    finally:
        for h in list(root.handlers):
            root.removeHandler(h)
        for h in saved:
            root.addHandler(h)
        root.setLevel(saved_level)
        root._propiq_streams_configured = saved_flag


# --- a typo'd env var must not take the process down ---------------------


@pytest.mark.parametrize("value", ["", "   ", "VERBOSE", "nonsense", "info"])
def test_an_unusable_log_level_falls_back_instead_of_raising(monkeypatch, value):
    """
    This runs at module import inside the container's boot. `setLevel` raises
    on an unknown level name, which would turn a typo in a Railway variable
    into a container that will not start — a worse outcome than ignoring it.
    """
    monkeypatch.setenv("LOG_LEVEL", value)
    root = logging.getLogger()
    saved, saved_level = list(root.handlers), root.level
    saved_flag = getattr(root, "_propiq_streams_configured", False)
    try:
        configure_logging(force=True)  # must not raise
        assert isinstance(root.level, int)
        assert root.level > 0
    finally:
        for h in list(root.handlers):
            root.removeHandler(h)
        for h in saved:
            root.addHandler(h)
        root.setLevel(saved_level)
        root._propiq_streams_configured = saved_flag


def test_a_lowercase_level_is_honoured_rather_than_ignored(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "warning")
    root = logging.getLogger()
    saved, saved_level = list(root.handlers), root.level
    saved_flag = getattr(root, "_propiq_streams_configured", False)
    try:
        configure_logging(force=True)
        assert root.level == logging.WARNING
    finally:
        for h in list(root.handlers):
            root.removeHandler(h)
        for h in saved:
            root.addHandler(h)
        root.setLevel(saved_level)
        root._propiq_streams_configured = saved_flag


# --- the two entry points really do route through it ---------------------


def test_both_entry_points_call_it_and_neither_calls_basicConfig():
    """
    The fix is only live if the call sites changed. `logging.basicConfig` in
    either file would silently win or lose depending on import order, which
    is the ambiguity this removes.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for name in ("main.py", "scheduler_worker.py"):
        body = (root / name).read_text(encoding="utf-8")
        assert "configure_logging()" in body, f"{name} does not configure logging"
        assert "logging.basicConfig" not in body, (
            f"{name} still calls logging.basicConfig, which writes INFO to "
            f"stderr and makes Railway report it as an error"
        )
