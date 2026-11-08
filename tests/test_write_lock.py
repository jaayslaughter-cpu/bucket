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
import threading
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

    # the THIRD writer: the ESPN history backfill. It walks seasons of box
    # scores into `player_game_logs`, and for one deploy the only thing
    # between it and the slate was `history_backfill_time`'s 60-minute lead
    # -- which guarantees when it STARTS, not when it finishes.
    backfill = next(
        n for n in ast.walk(worker_tree)
        if isinstance(n, ast.FunctionDef) and n.name == "run_history_backfill"
    )
    assert takes_lock(backfill), (
        "run_history_backfill does not take the write lock, so an overrunning "
        "backfill can write the panel while the 09:00 slate reads it -- and "
        "the slate would score a partially written panel and exit 0"
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


def test_the_backfill_declines_rather_than_writing_under_a_held_lock(monkeypatch):
    """
    A HELD LOCK MUST STOP IT BEFORE IT READS, not merely before it writes.

    The plan is derived from `panel_game_ids_by_date` -- the same panel the
    slate is writing -- so a backfill that planned under a held lock would
    decide which dates to fetch from a half-written table. Both the read and
    the write are asserted absent.

    Declining costs nothing: tomorrow's run plans from the panel again and
    picks up the same dates.
    """
    import contextlib

    import scheduler_worker as worker
    import src.db.repository as repo
    from src.utils import write_lock as wl

    monkeypatch.setenv(worker.ENV_ESPN_HISTORY_SEASONS, "2025,2026")

    @contextlib.contextmanager
    def _held(*a, **k):
        yield False

    # `run_history_backfill` imports write_lock INSIDE the call, so the module
    # attribute is what it resolves.
    monkeypatch.setattr(wl, "write_lock", _held)
    monkeypatch.setattr(
        worker, "run_espn_ingest",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("the backfill WROTE to the panel under a held lock")
        ),
    )
    monkeypatch.setattr(
        repo, "panel_game_ids_by_date",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("the backfill READ the panel under a held lock")
        ),
    )

    out = worker.run_history_backfill()

    assert out["status"] == "SKIPPED", out
    assert "write lock" in out["reason"], (
        f"the decline does not say why: {out.get('reason')!r}"
    )
    # A decline is not a failure, so it must not look like one.
    assert "error" not in out, out


def test_the_backfill_proceeds_when_the_lock_is_free(monkeypatch):
    """
    THE OTHER HEAD. Without this, a backfill that declined unconditionally
    would satisfy the test above and never ingest anything again -- which is
    the quiet version of the failure the lock exists to prevent.
    """
    import contextlib

    import scheduler_worker as worker
    import src.db.repository as repo
    from src.utils import write_lock as wl

    monkeypatch.setenv(worker.ENV_ESPN_HISTORY_SEASONS, "2025,2026")

    @contextlib.contextmanager
    def _free(*a, **k):
        yield True

    monkeypatch.setattr(wl, "write_lock", _free)

    reached = {}

    def _planned(source, *a, **k):
        reached["panel_read"] = True
        raise RuntimeError("stop here -- getting this far is the assertion")

    monkeypatch.setattr(repo, "panel_game_ids_by_date", _planned)

    out = worker.run_history_backfill()

    assert reached.get("panel_read"), (
        "a FREE lock did not let the backfill reach its planning read, so it "
        f"is declining unconditionally: {out}"
    )


# --- the refcount is per PATH, and that was not enough ---------------------

def _hold_in_a_thread(tmp_path, results, entered, release, key="owner"):
    """Take the lock in a separate thread and hold it until `release` is set."""
    def body():
        try:
            with write_lock(root=tmp_path) as acquired:
                results[key] = acquired
                entered.set()
                release.wait(timeout=10)
        except BaseException as exc:  # noqa: BLE001 — surfaced to the assertion
            results[key] = exc
            entered.set()

    thread = threading.Thread(target=body, daemon=True)
    thread.start()
    assert entered.wait(timeout=10), "the holder thread never acquired the lock"
    return thread


def test_a_second_thread_declines_rather_than_sharing_the_holders_refcount(tmp_path):
    """
    THE DEFECT, and the docstring above walked right past it: "a thread would
    share the refcount and prove nothing about `flock`" treats shared state
    between threads as a nuisance for testing rather than as the bug.

    `_DEPTH` was keyed by lock PATH alone. Reentrancy exists for one call chain
    on one thread -- run_slate -> main.main -- but any OTHER thread that asked
    for the same path found a non-zero depth and took the "already ours" branch,
    so it was handed `True` without ever touching `flock`. Two threads then ran
    the protected region at once.

    That is not hypothetical here: APScheduler's `BlockingScheduler` dispatches
    each job onto its own pool thread, and `max_instances=1` is PER JOB -- which
    `scheduler_worker` itself notes in two places. Settlement, loss analysis,
    the backfill, training and the slate are five different jobs, so two of them
    overlapping is two threads, not two processes.

    `flock` would have refused the second acquire on its own: it conflicts
    across file descriptors within one process. The short-circuit above it is
    the whole defect.
    """
    results, entered, release = {}, threading.Event(), threading.Event()
    thread = _hold_in_a_thread(tmp_path, results, entered, release)
    try:
        assert results["owner"] is True, "the holder thread did not get the lock"
        with write_lock(root=tmp_path) as second:
            assert second is False, (
                "a second THREAD was handed the lock while another thread held "
                "it; both would write the panel at once"
            )
    finally:
        release.set()
        thread.join(timeout=10)


def test_the_holders_own_nesting_still_works_while_another_thread_is_excluded(tmp_path):
    """
    Fixing exclusion must not cost reentrancy. The holder nests; everyone else
    declines; and those are the same lock at the same path.
    """
    results, entered, release = {}, threading.Event(), threading.Event()
    nested: dict[str, object] = {}

    def body():
        with write_lock(root=tmp_path) as outer:
            results["owner"] = outer
            with write_lock(root=tmp_path) as inner:
                nested["inner"] = inner
            entered.set()
            release.wait(timeout=10)

    thread = threading.Thread(target=body, daemon=True)
    thread.start()
    assert entered.wait(timeout=10)
    try:
        assert results["owner"] is True
        assert nested["inner"] is True, "the holder's own nested acquire declined"
        with write_lock(root=tmp_path) as other:
            assert other is False
    finally:
        release.set()
        thread.join(timeout=10)


def test_the_lock_returns_to_the_pool_once_the_holding_thread_leaves(tmp_path):
    """
    A decline must be for the duration of the hold, not for the life of the
    process: the next job's thread has to be able to take it.
    """
    results, entered = {}, threading.Event()
    done = threading.Event()

    # THE FIRST THREAD STAYS ALIVE AFTER RELEASING. Joining it first would let
    # the runtime hand its thread id straight to the next thread, and a holder
    # recorded by id would then compare equal to its own successor -- which is
    # exactly how a mutation that never cleared the owner survived this test.
    def holder():
        with write_lock(root=tmp_path) as acquired:
            results["owner"] = acquired
        entered.set()
        done.wait(timeout=10)

    thread = threading.Thread(target=holder, daemon=True)
    thread.start()
    assert entered.wait(timeout=10)

    after: dict[str, object] = {}
    second_entered = threading.Event()

    def later():
        with write_lock(root=tmp_path) as acquired:
            after["acquired"] = acquired
        second_entered.set()

    later_thread = threading.Thread(target=later, daemon=True)
    later_thread.start()
    assert second_entered.wait(timeout=10)
    later_thread.join(timeout=10)
    done.set()
    thread.join(timeout=10)
    assert after["acquired"] is True, (
        "the lock stayed owned after its holder returned, so every later job "
        "would decline forever"
    )


def test_a_holder_that_raises_does_not_strand_the_lock(tmp_path):
    """
    `_run_slate_locked` catches its own exceptions, but a job that dies inside
    the lock must still hand it back, or one bad night silences every job after.
    """
    entered = threading.Event()

    def body():
        try:
            with write_lock(root=tmp_path):
                entered.set()
                raise RuntimeError("synthetic failure inside the lock")
        except RuntimeError:
            pass

    thread = threading.Thread(target=body, daemon=True)
    thread.start()
    assert entered.wait(timeout=10)
    thread.join(timeout=10)

    with write_lock(root=tmp_path) as acquired:
        assert acquired is True, "a raising holder stranded the lock"
