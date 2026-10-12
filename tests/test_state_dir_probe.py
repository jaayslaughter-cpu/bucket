"""
The boot probe must test the directory the volume is actually mounted at.

FOUND IN PRODUCTION, 2026-10-12. A Railway volume was attached at
``/var/lib/postgresql/data`` and was root-owned while the process runs as uid
10001. `scripts/railway_healthcheck` caught it:

    [OK  ] state_root: /var/lib/postgresql/data (RAILWAY_VOLUME_MOUNT_PATH)
    [FAIL] state_root_writable: /var/lib/postgresql/data IS NOT WRITABLE

while the worker's own boot check, in the same container seconds later, said:

    State directory /app/data is writable.

A FALSE ALL-CLEAR, and the worst kind: `check_state_dir` is the check that
exists SPECIFICALLY to catch a root-owned mounted volume -- its docstring says
so -- and it probed a different directory and passed. `notify_boot_problems`
therefore saw ``writable: True`` and raised no alert about the one fault that
silently breaks the calibration report and the artifact store.

THE CAUSE: TWO READERS OF ONE FACT. The healthcheck asks
`utils.volume.resolve_state_root`, which honours ``RAILWAY_VOLUME_MOUNT_PATH``.
`check_state_dir` derived its target from the CALIBRATION REPORT's path
instead, whose default does not follow the volume. Same defect shape as
``board_markets``, the PRA rollups, the parlay ledger and the game-id joins --
one reader is the fix, every time.

RESEARCH_ONLY. A write probe on a directory.
"""

from __future__ import annotations

import os

import scheduler_worker as worker
from src.utils.volume import ENV_RAILWAY_VOLUME, ENV_STATE_DIR, resolve_state_root


def test_the_probe_targets_the_resolved_state_root(monkeypatch, tmp_path):
    """
    THE PRODUCTION FAULT, as a test: a volume is attached, and the probe must
    be about THAT directory and not whatever the calibration report defaults to.
    """
    volume = tmp_path / "volume"
    volume.mkdir()
    monkeypatch.delenv(ENV_STATE_DIR, raising=False)
    monkeypatch.setenv(ENV_RAILWAY_VOLUME, str(volume))
    # a calibration report somewhere else entirely, as the deploy had
    monkeypatch.setenv(worker.ENV_CALIBRATION_REPORT, str(tmp_path / "elsewhere/cal.json"))

    assert resolve_state_root()[0] == volume
    out = worker.check_state_dir()
    assert out["path"] == str(volume), (
        f"the probe tested {out['path']!r}, not the resolved state root "
        f"{str(volume)!r} -- an unwritable volume would pass unnoticed"
    )


def test_an_unwritable_volume_is_reported_not_passed(monkeypatch, tmp_path):
    """
    The false all-clear itself. Before the fix this returned writable=True
    because it probed the calibration report's parent, which WAS writable.
    """
    volume = tmp_path / "rootowned"
    volume.mkdir()
    os.chmod(volume, 0o500)  # readable, not writable
    monkeypatch.delenv(ENV_STATE_DIR, raising=False)
    monkeypatch.setenv(ENV_RAILWAY_VOLUME, str(volume))
    monkeypatch.setenv(worker.ENV_CALIBRATION_REPORT, str(tmp_path / "writable/cal.json"))

    out = worker.check_state_dir()
    try:
        if os.getuid() == 0:
            # root bypasses the mode bits, so the probe legitimately succeeds
            # here; the path assertion above is what matters in that case.
            assert out["path"] == str(volume)
            return
    except AttributeError:  # pragma: no cover - POSIX only
        pass
    assert out["writable"] is False, (
        "an unwritable state root was reported as writable -- this is the "
        "false all-clear seen in production on 2026-10-12"
    )


def test_the_alert_fires_on_an_unwritable_state_root():
    """
    The consequence, asserted end to end: `notify_boot_problems` must raise the
    boot:state-dir alert from what the probe returns. A correct probe that
    nothing routes is no better than a wrong one.
    """
    out = worker.notify_boot_problems(
        {"path": "/var/lib/postgresql/data", "writable": False, "error": "EACCES"},
        {"resolved": True, "exists": True},
        {"status": "OK"},
        {"ok": True, "message": "Database reachable."},
    )
    assert "boot:state-dir" in [a["stage"] for a in out]


def test_an_explicit_path_still_wins():
    """The parameter is how tests and callers aim it; the resolver is only the
    default."""
    out = worker.check_state_dir("outputs/calibration.json")
    assert out["path"] == "outputs"


def test_the_probe_and_the_healthcheck_read_the_same_resolver():
    """
    AST-walked: both must call `resolve_state_root`. The production split was
    exactly this -- one of them did and the other inferred the directory from
    an unrelated default.
    """
    import ast
    import inspect
    import pathlib

    src = inspect.getsource(worker.check_state_dir)
    assert "resolve_state_root" in src, (
        "check_state_dir does not consult utils.volume.resolve_state_root, so "
        "it can disagree with the healthcheck about where state lives"
    )
    called = {
        n.func.id
        for n in ast.walk(ast.parse(src.lstrip()))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "resolve_state_root" in called, "it imports the resolver but never calls it"

    hc = pathlib.Path("scripts/railway_healthcheck.py").read_text(encoding="utf-8")
    assert "resolve_state_root" in hc
