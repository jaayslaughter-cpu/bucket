"""
A1 — one slate run at a time, ACROSS PROCESSES.

WHAT THIS GUARDS, from the candidate-repo evaluation on 2026-10-11.
`scheduler_worker` schedules every job with ``max_instances=1`` and
``coalesce=True``, which is correct and sufficient WITHIN ONE PROCESS and does
nothing about a second one. An operator running ``railway run python main.py``
while the 09:00 PT slate job is firing gets two processes interleaving writes to
the same Postgres rows and the same artifact directory. We run
``numReplicas: 1``, so this is operator concurrency rather than scaling — which
is the kind that happens once and is very hard to diagnose afterwards.

THE IDEA IS CITED, THE CODE IS OURS.
`EdgarParra565/player-performance-forecaster @ nba_model/data/hourly_update.py:74-128`
takes a NON-BLOCKING ``fcntl.flock`` and exits 75 (``EX_TEMPFAIL``) when it is
already held, so an overlapping run is SKIPPED rather than queued. That
repository is unlicensed, so this is the reasoning, not their lines.

DECLINES, NEVER WAITS. Queueing would be worse than skipping here: a slate that
starts an hour late scores a slate that has already tipped, and two runs
serialised back to back both write, just not at once.

REENTRANT IN-PROCESS, and that is not a nicety. ``run_slate`` delegates to
``main.main``; if both took a non-blocking lock naively, the inner acquire would
find the outer one held and the scheduled run would decline ITSELF. A refcount
makes nesting safe.

RESEARCH_ONLY. A lock file. Nothing here reads a market or sizes anything.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from src.utils.write_lock import LOCK_FILENAME, write_lock


def test_a_free_lock_is_acquired(tmp_path):
    with write_lock(root=tmp_path) as acquired:
        assert acquired is True
    assert (tmp_path / LOCK_FILENAME).exists(), "the lock file should persist for reuse"


def test_it_is_reentrant_within_one_process(tmp_path):
    """
    THE TRAP. run_slate -> main.main is a nested acquire of the same lock, and a
    naive non-blocking flock would make the scheduled run decline itself.
    """
    with write_lock(root=tmp_path) as outer:
        assert outer is True
        with write_lock(root=tmp_path) as inner:
            assert inner is True, "nested acquire declined — run_slate would skip itself"
        # still held after the inner scope closes
        with write_lock(root=tmp_path) as again:
            assert again is True


def test_a_second_process_declines_rather_than_waiting(tmp_path):
    """
    The whole point, and it needs a real second process — a thread would share
    the refcount and prove nothing about `flock`.
    """
    holder = textwrap.dedent(f"""
        import sys, time
        sys.path.insert(0, {str(Path.cwd())!r})
        from src.utils.write_lock import write_lock
        with write_lock(root={str(tmp_path)!r}) as got:
            print("HOLDER", got, flush=True)
            time.sleep(8)
    """)
    proc = subprocess.Popen(
        [sys.executable, "-c", holder], stdout=subprocess.PIPE, text=True
    )
    try:
        line = proc.stdout.readline().strip()
        assert line == "HOLDER True", f"the holder never acquired: {line!r}"
        # now, from THIS process, the same lock must decline immediately
        import time

        start = time.monotonic()
        with write_lock(root=tmp_path) as acquired:
            elapsed = time.monotonic() - start
            assert acquired is False, "a second process acquired a held lock"
        assert elapsed < 2.0, f"it waited {elapsed:.1f}s instead of declining"
    finally:
        proc.kill()
        proc.wait(timeout=10)


def test_the_lock_is_released_even_when_the_body_raises(tmp_path):
    with pytest.raises(RuntimeError):
        with write_lock(root=tmp_path) as acquired:
            assert acquired is True
            raise RuntimeError("boom")
    # a fresh acquire must succeed, or one failed slate blocks every later one
    with write_lock(root=tmp_path) as acquired:
        assert acquired is True


def test_an_unlockable_location_proceeds_rather_than_stopping_the_slate(
    tmp_path, monkeypatch, caplog
):
    """
    FAILS OPEN, DELIBERATELY, and says so.

    The status quo before this module was no lock at all. Turning a permission
    problem on the volume into "no slate runs, ever" would be a worse outage
    than the interleaving this prevents, and the same reasoning
    `_alert_already_sent_today` already follows. The warning is the contract.
    """
    import src.utils.write_lock as wl

    def cannot_open(*a, **k):
        raise PermissionError("read-only volume")

    monkeypatch.setattr(wl, "_open_lock_file", cannot_open)
    with caplog.at_level("WARNING"):
        with write_lock(root=tmp_path) as acquired:
            assert acquired is True, "a lock it cannot take must not stop the run"
    assert any("without a lock" in r.message.lower() or "unlocked" in r.message.lower()
               for r in caplog.records), "failing open must be logged"


# --- and it has to actually be wired in -----------------------------------


def test_main_takes_the_lock_around_the_run():
    """
    AST-walked. main.main is the funnel both the scheduled job and a manual
    `railway run python main.py` pass through — it holds every upsert — so the
    lock belongs there and nowhere else, which is also what keeps the nesting
    single-level.
    """
    tree = ast.parse(Path("main.py").read_text(encoding="utf-8"))
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    withs = [
        item
        for node in ast.walk(fn)
        if isinstance(node, ast.With)
        for item in node.items
    ]
    names = []
    for item in withs:
        call = item.context_expr
        if isinstance(call, ast.Call):
            f = call.func
            names.append(f.id if isinstance(f, ast.Name) else getattr(f, "attr", ""))
    assert "write_lock" in names, (
        "main.main does not take the write lock, so a manual run and the "
        "scheduled slate can still interleave their writes"
    )
