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


def test_both_entry_points_take_the_lock():
    """
    AST-walked, at BOTH of `main.main`'s callers.

    THE FIRST VERSION OF THIS TEST ASSERTED THE WRONG DESIGN. It required the
    lock inside `main.main`, so the implementation extracted that function's
    body to indent it under a `with` -- and broke four AST tests across three
    files that walk `main`'s body for the steps it must wire, in order. Those
    tests were right: `main` holding the pipeline is a contract the suite
    depends on. The lock belongs at the two call sites instead, which is also
    wider cover, since `run_slate`'s board write and dispatch fall inside its
    scope where wrapping `main` alone left them out.

    Both sites are asserted because one of them is the hole: locking only the
    scheduled path leaves `railway run python main.py` free to interleave, and
    locking only the manual path leaves the 09:00 job free to do the same.
    """
    import ast

    def takes_lock(tree: ast.AST) -> bool:
        for node in ast.walk(tree):
            if not isinstance(node, ast.With):
                continue
            for item in node.items:
                call = item.context_expr
                if isinstance(call, ast.Call):
                    f = call.func
                    if (isinstance(f, ast.Name) and f.id == "write_lock") or (
                        getattr(f, "attr", "") == "write_lock"
                    ):
                        return True
        return False

    # the MANUAL path: main.py's __main__ block
    main_tree = ast.parse(Path("main.py").read_text(encoding="utf-8"))
    main_guard = [
        n for n in main_tree.body
        if isinstance(n, ast.If)
        and ast.dump(n.test).find("__main__") != -1
    ]
    assert main_guard, "main.py has no __main__ block to lock"
    assert takes_lock(main_guard[0]), (
        "main.py's __main__ block does not take the write lock, so a manual "
        "`railway run python main.py` can interleave with the scheduled slate"
    )

    # the SCHEDULED path: scheduler_worker.run_slate
    worker_tree = ast.parse(Path("scheduler_worker.py").read_text(encoding="utf-8"))
    run_slate = next(
        n for n in ast.walk(worker_tree)
        if isinstance(n, ast.FunctionDef) and n.name == "run_slate"
    )
    assert takes_lock(run_slate), (
        "run_slate does not take the write lock, so the 09:00 PT job can "
        "interleave with a manual run"
    )


def test_main_itself_still_holds_the_pipeline_body():
    """
    THE REGRESSION GUARD for the mistake above, stated as the contract it broke.

    Four tests in test_player_log_refresh, test_market_frames_fallback and
    test_per_market_and_per_row_status walk `main`'s body for the calls it must
    make. If a future change moves that body into a helper again, they all fail
    with "main() does not call: [...]" and the cause is not obvious from any
    one of them. This says it once, here, next to the change that caused it.
    """
    import ast

    tree = ast.parse(Path("main.py").read_text(encoding="utf-8"))
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    called = {
        n.func.id
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    expected = {"preflight", "load_master_guideline", "resolve_market_frames"}
    missing = expected - called
    assert not missing, (
        f"main() no longer calls {sorted(missing)} directly. If its body was "
        "moved into a helper to wrap it in something, four other AST tests are "
        "about to fail too -- take the wrapper at main's CALL SITES instead "
        "(main.py's __main__ block and scheduler_worker.run_slate)."
    )
