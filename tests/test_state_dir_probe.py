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


# --- cubic's two findings on PR #8, both valid -----------------------------


def test_every_failing_directory_is_reported_not_just_the_first(monkeypatch, tmp_path):
    """
    cubic, PR #8: the loop collected every failure into `failures` and then read
    only `failures[0]`. `failures[1:]` was dead data, so when BOTH directories
    were unwritable the second vanished from the log and from the return value —
    an operator would chown the first, redeploy, and meet the same alert again
    naming the other one.

    Simulated by making the probe itself fail, because this container runs as
    root and cannot be denied by mode bits.
    """
    import src.utils.volume as volume

    root = tmp_path / "vol"
    report = tmp_path / "elsewhere"
    root.mkdir()
    report.mkdir()
    monkeypatch.setattr(volume, "resolve_state_root", lambda: (root, "test"))
    monkeypatch.setenv(worker.ENV_CALIBRATION_REPORT, str(report / "cal.json"))

    real_write = type(tmp_path).write_text

    def deny(self, *a, **k):
        if self.name == ".propiq_write_probe":
            raise PermissionError(f"denied: {self.parent}")
        return real_write(self, *a, **k)

    monkeypatch.setattr(type(tmp_path), "write_text", deny)

    out = worker.check_state_dir()
    assert out["writable"] is False
    failed = out.get("failures")
    assert failed is not None, "the return value drops every failure after the first"
    assert len(failed) == 2, f"only {len(failed)} failure(s) reported, expected both"
    assert {f["path"] for f in failed} == {str(root), str(report)}


def test_the_message_names_what_actually_lives_in_the_broken_directory(
    monkeypatch, tmp_path, caplog
):
    """
    cubic, PR #8: the message was written for the single-target check and
    asserted "The calibration report and the model artifacts live here" plus the
    root-owned-volume hypothesis for BOTH cases. When the calibration report's
    own directory is the one that failed, the artifacts live elsewhere and a
    mounted volume is usually not the cause — the text named the wrong contents
    for one of the two cases this very change introduced.

    Documentation outliving its facts, which is the defect this repository
    guards against in two separate test files. I changed the semantics and left
    the message describing the old ones.
    """
    import logging

    import src.utils.volume as volume

    root = tmp_path / "vol"
    report = tmp_path / "elsewhere"
    root.mkdir()
    report.mkdir()
    monkeypatch.setattr(volume, "resolve_state_root", lambda: (root, "test"))
    monkeypatch.setenv(worker.ENV_CALIBRATION_REPORT, str(report / "cal.json"))

    real_write = type(tmp_path).write_text

    def deny_report_only(self, *a, **k):
        if self.name == ".propiq_write_probe" and self.parent == report:
            raise PermissionError("denied")
        return real_write(self, *a, **k)

    monkeypatch.setattr(type(tmp_path), "write_text", deny_report_only)

    with caplog.at_level(logging.ERROR):
        out = worker.check_state_dir()
    assert out["writable"] is False
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "calibration report" in text.lower()
    assert "mounted volume" not in text.lower(), (
        "the mounted-volume hypothesis is named for a calibration-report "
        "failure, where it is usually not the cause"
    )


def test_the_alert_names_every_unwritable_directory(caplog):
    """
    The consequence of cubic's first finding, carried through to the thing an
    operator actually reads. The probe reporting both failures is no use if the
    Discord alert still names one: they chown it, redeploy, and meet the same
    alert — which reads as "the fix did not work", not "there was a second one".
    """
    import logging

    with caplog.at_level(logging.ERROR):
        out = worker.notify_boot_problems(
            {
                "path": "/app/data",
                "writable": False,
                "error": "EACCES",
                "failures": [
                    {"path": "/app/data", "error": "EACCES"},
                    {"path": "/elsewhere", "error": "EROFS"},
                ],
            },
            {"resolved": True, "exists": True},
            {"status": "OK"},
            {"ok": True, "message": "Database reachable."},
        )
    assert "boot:state-dir" in [a["stage"] for a in out]
    # The dispatch result carries no body -- the alert TEXT is what an operator
    # sees, in the log line and in the Discord message built from it.
    body = " ".join(
        r.getMessage() for r in caplog.records if "boot:state-dir" in r.getMessage()
    )
    assert "/app/data" in body
    assert "/elsewhere" in body, (
        "the second unwritable directory is missing from the alert an operator "
        "reads, so they will fix one and see the same message again"
    )


def test_the_alert_carries_each_directory_s_own_cause(caplog):
    """
    cubic, second round on PR #8, and right: the fix for finding 2 gave the LOG
    a per-directory cause and left the Discord ALERT asserting the state-root
    one for whichever directory failed. The two then contradict each other about
    the same failure — the log saying the artifacts live elsewhere and no volume
    is involved, the alert saying a mounted volume is commonly root-owned and
    the artifacts live there. Message outliving its facts, one layer up from
    where this PR fixed it.

    `check_state_dir` already puts a `cause` on every failure. The alert has to
    read it rather than keep a second copy of the reasoning.
    """
    import logging

    with caplog.at_level(logging.ERROR):
        worker.notify_boot_problems(
            {
                "path": "/elsewhere",
                "writable": False,
                "error": "EACCES",
                "failures": [
                    {
                        "path": "/elsewhere",
                        "error": "EACCES",
                        "what": "CALIBRATION REPORT DIRECTORY",
                        "cause": "Settlement writes the calibration report here.",
                    }
                ],
            },
            {"resolved": True, "exists": True},
            {"status": "OK"},
            {"ok": True, "message": "Database reachable."},
        )
    text = " ".join(
        r.getMessage() for r in caplog.records if "boot:state-dir" in r.getMessage()
    )
    assert "Settlement writes the calibration report here." in text, (
        "the alert ignores the cause the probe worked out and substitutes its own"
    )
    assert "mounted volume" not in text.lower(), (
        "the alert asserts the state-root cause for a calibration-report "
        "failure, contradicting the log line about the same directory"
    )
