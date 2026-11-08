"""
One slate run at a time, across processes.

WHY THIS EXISTS (A1, from the candidate-repo evaluation on 2026-10-11).
`scheduler_worker` schedules every job with ``max_instances=1`` and
``coalesce=True``. That is correct, and it is a property of ONE PROCESS. An
operator running ``railway run python main.py`` while the 09:00 PT slate job is
firing gets two processes upserting the same Postgres rows and writing the same
artifact directory, with nothing anywhere preventing it. The deployment runs
``numReplicas: 1``, so this is operator concurrency rather than scaling — the
kind that happens once and is very hard to reconstruct afterwards.

THE IDEA IS CITED; THE CODE IS OURS.
`EdgarParra565/player-performance-forecaster @ nba_model/data/hourly_update.py:74-128`
takes a non-blocking ``fcntl.flock`` and exits 75 (``EX_TEMPFAIL``) when it is
held, so an overlapping run is skipped rather than queued. That repository
carries no licence, so what is taken here is the reasoning.

IT DECLINES, IT NEVER WAITS. Queueing would be worse than skipping: a slate
that starts an hour late scores games that have already tipped, and two runs
serialised back to back both write anyway, just not simultaneously. The caller
decides what declining means — `main` reports it and exits 0, because a
deliberate skip is not a failure.

IT IS REENTRANT WITHIN ONE THREAD, and that is load-bearing rather than tidy.
``run_slate`` delegates to ``main.main``. If the lock were taken at both levels
without a refcount, the inner acquire would find the outer one held and THE
SCHEDULED RUN WOULD DECLINE ITSELF — the one failure mode that would look
exactly like the feature working.

WITHIN ONE THREAD, not within one process, and the distinction is the fix for a
defect this module shipped with. The refcount was keyed by lock PATH alone, so
ANY thread asking for the same path found a non-zero depth and took the nesting
branch — handed ``True`` without ever reaching ``flock``. The reentrancy that
exists for one call chain was being granted to unrelated callers.

That is a live shape, not a hypothetical. APScheduler's ``BlockingScheduler``
dispatches each job onto its own pool thread and ``max_instances`` is PER JOB,
which ``scheduler_worker`` notes in two places. Settlement (03:30), loss
analysis (04:00), the backfill (04:30), training (07:00) and the slate (09:00)
are five jobs on five threads; any overrun puts two of them in the protected
region together. ``flock`` would have refused the second acquire by itself —
it conflicts across file descriptors within one process — so the short-circuit
above it was the entire bug.

A DIFFERENT THREAD DECLINES, exactly as a different process does. The callers
already handle that: ``main`` reports it and exits 0, and the nightly analysis
costs one night's report. Blocking instead would contradict the paragraph
above and queue a slate behind a backfill.

IT FAILS OPEN. If the lock file cannot be created or locked — a read-only
volume, a permissions problem — the run proceeds and says so loudly. Before
this module there was no lock at all, so turning a filesystem problem into "no
slate ever runs" would be a worse outage than the interleaving it prevents.
`scheduler_worker._alert_already_sent_today` already follows that reasoning.

RESEARCH_ONLY. This is a lock file. It reads no market, scores nothing and
sizes nothing.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO

logger = logging.getLogger(__name__)

#: On the durable state root, beside the artifacts the lock protects, so a
#: redeploy that remounts the volume keeps one lock rather than minting a
#: second one somewhere ephemeral.
LOCK_FILENAME = "propiq-write.lock"

#: Depth per lock path, for the run_slate -> main.main nesting above. A plain
#: boolean would make the inner scope release the outer one's lock on exit.
_DEPTH: dict[str, int] = {}
_HANDLES: dict[str, IO[bytes]] = {}

#: WHO HOLDS EACH LOCK, and the only answer to "is it held". Without it the
#: refcount above is shared by every thread in the process, which turns the
#: reentrancy guarantee into a hole: see the module docstring. `_DEPTH` counts
#: one thread's nesting and nothing else.
#:
#: THE THREAD OBJECT, NOT `get_ident()`. An ident is an OS thread id and the
#: runtime RECYCLES it once a thread exits, so a stale entry plus a recycled id
#: would let an unrelated thread nest into a lock it never took. It also made a
#: mutation of the cleanup below survive: the follow-up thread in the test
#: inherited the dead holder's id and compared equal to it. An object cannot be
#: confused with its successor, and the reference lives only while the lock is
#: held.
_OWNERS: dict[str, threading.Thread] = {}

#: Guards the three dicts, and ONLY the bookkeeping -- never held across the
#: ``yield``. Holding it around the body would serialise callers, which is
#: queueing by another name and the opposite of what this module promises.
_STATE = threading.Lock()


def _open_lock_file(path: Path) -> IO[bytes]:
    """Separated so a test can make the filesystem refuse."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # "a+b" creates without truncating: truncating a file another process holds
    # a lock on is legal and pointless, and the byte we never write is why the
    # file's contents are irrelevant to the lock.
    return path.open("a+b")


def _lock_path(root: Path | str | None) -> Path:
    if root is not None:
        return Path(root) / LOCK_FILENAME
    from src.utils.volume import resolve_state_root

    return resolve_state_root()[0] / LOCK_FILENAME


@contextmanager
def write_lock(*, root: Path | str | None = None) -> Iterator[bool]:
    """
    Hold the slate write lock for the duration of the block.

    Yields ``True`` when this caller may write and ``False`` when the lock is
    already held by another process OR by another thread of this one. Nesting
    on the SAME thread re-enters and stays ``True``, which is what
    ``run_slate -> main.main`` relies on. NEVER blocks, NEVER raises on
    contention.
    """
    path = _lock_path(root)
    key = str(path.resolve() if path.parent.exists() else path)

    me = threading.current_thread()
    with _STATE:
        # `_OWNERS` IS THE ONE READER OF "is this held, and by whom". An
        # earlier version asked `_DEPTH.get(key, 0) > 0` here and used
        # `_OWNERS` only to name the thread -- two places encoding one fact,
        # so a stale owner left behind on release was unobservable and the
        # cleanup below could be deleted without a single test noticing.
        held_by = _OWNERS.get(key)
        # Already ours, on THIS thread: nest, do not re-lock.
        mine = held_by == me
        if mine:
            _DEPTH[key] += 1

    if mine:
        try:
            yield True
        finally:
            with _STATE:
                _DEPTH[key] -= 1
        return

    if held_by is not None:
        # Another thread in this process. `flock` below would refuse it anyway,
        # but saying so by name is the difference between an operator looking
        # for a stray `railway run` and looking at the job schedule.
        logger.warning(
            "DECLINED: another thread in this process holds the slate write "
            "lock (%s, held by thread %r). Jobs run on separate scheduler "
            "threads and `max_instances` is per job, so an overrunning job "
            "can still be inside this region. This run is skipped rather "
            "than queued. Nothing was written.", path, held_by.name,
        )
        yield False
        return

    try:
        handle = _open_lock_file(path)
    except OSError as exc:
        logger.warning(
            "Could not open the write lock at %s (%s). PROCEEDING WITHOUT A "
            "LOCK: a concurrent manual run could interleave its writes with "
            "this one. Fix the volume's permissions to restore the guard.",
            path, exc,
        )
        yield True
        return

    try:
        import fcntl

        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            logger.warning(
                "DECLINED: another process holds the slate write lock (%s). "
                "This run is skipped rather than queued -- a slate that starts "
                "late scores games that have already tipped. Nothing was "
                "written.", path,
            )
            yield False
            return
        except OSError as exc:
            handle.close()
            logger.warning(
                "The write lock at %s could not be taken (%s). PROCEEDING "
                "WITHOUT A LOCK.", path, exc,
            )
            yield True
            return
    except ImportError:  # pragma: no cover - POSIX only in this deployment
        logger.warning(
            "fcntl is unavailable, so there is no cross-process write lock on "
            "this platform. PROCEEDING WITHOUT A LOCK."
        )
        handle.close()
        yield True
        return

    with _STATE:
        _DEPTH[key] = 1
        _HANDLES[key] = handle
        _OWNERS[key] = me
    logger.info("Holding the slate write lock (%s, pid %d).", path, os.getpid())
    try:
        yield True
    finally:
        with _STATE:
            _DEPTH[key] = max(0, _DEPTH.get(key, 1) - 1)
            released = _DEPTH.get(key, 0) == 0
            if released:
                # CLEARED BEFORE THE UNLOCK, so no thread can observe an owned
                # key whose file lock has already gone.
                _OWNERS.pop(key, None)
        if released:
            try:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except (ImportError, OSError):
                # Closing the handle releases it anyway; an explicit unlock is
                # belt and braces, and failing to unlock must not mask the
                # exception the body may be propagating.
                pass
            handle.close()
            _HANDLES.pop(key, None)
