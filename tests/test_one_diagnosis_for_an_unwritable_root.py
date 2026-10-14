"""
The two write probes must not contradict each other about the same directory.

FOUND IN A RAILWAY LOG, 2026-10-13 (deploy 08c337fb). Three lines apart, about
one path:

    [FAIL] state_root_writable: /var/lib/postgresql/data DOES NOT EXIST AND
    COULD NOT BE CREATED (...), so NO VOLUME IS MOUNTED THERE ... chown is not
    the remedy here: there is nothing to chown.

    ERROR propiq.scheduler | STATE ROOT /var/lib/postgresql/data IS NOT
    WRITABLE (...). ... If this is a mounted volume it is most likely owned by
    root while this process runs as uid 10001 -- chown it to that uid

MINE, and the shape this repository keeps finding: I fixed the diagnosis in
`scripts/railway_healthcheck.py` and left `scheduler_worker.check_state_dir`
holding its own copy. Two readers of one fact; one of them now tells an
operator to chown a path the other has just told them does not exist.

`src/utils/volume.classify_unwritable` is the one reader for WHICH fault a
failed probe hit, and `unwritable_remedy` for what to do about it. Each caller
still says what lives in the directory -- that part really does differ between
the artifact store and the calibration report -- but neither owns the
fault-to-remedy mapping any more.

RESEARCH_ONLY. The text of two warnings.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

import scheduler_worker as worker
from src.utils.volume import (
    ENV_STATE_DIR as ENV_STATE_DIR_NAME,
)
from src.utils.volume import (
    FAULT_MISSING_MOUNT,
    FAULT_NO_SPACE,
    FAULT_NOT_A_DIRECTORY,
    FAULT_NOT_WRITABLE,
    FAULT_READ_ONLY_FS,
    classify_unwritable,
    first_non_directory_ancestor,
    unwritable_remedy,
)

# --- the classifier -------------------------------------------------------


def test_an_absent_path_is_a_missing_mount(tmp_path):
    assert classify_unwritable(tmp_path / "never-mounted") == FAULT_MISSING_MOUNT


def test_a_file_is_not_a_directory(tmp_path):
    as_a_file = tmp_path / "a-file"
    as_a_file.write_text("", encoding="utf-8")
    assert classify_unwritable(as_a_file) == FAULT_NOT_A_DIRECTORY


def test_a_dangling_symlink_is_not_a_directory_either(tmp_path):
    """`exists()` follows the link and is False for a broken one, so a naive
    check calls a stale link a missing mount and sends the operator after a
    volume instead of the link."""
    link = tmp_path / "stale"
    link.symlink_to(tmp_path / "nothing-here")
    assert not link.exists() and link.is_symlink()
    assert classify_unwritable(link) == FAULT_NOT_A_DIRECTORY


def test_an_existing_directory_is_an_ownership_fault(tmp_path):
    assert classify_unwritable(tmp_path) == FAULT_NOT_WRITABLE


# --- the remedy -----------------------------------------------------------


def test_only_the_ownership_fault_says_chown():
    for fault in (FAULT_MISSING_MOUNT, FAULT_NOT_A_DIRECTORY):
        remedy = unwritable_remedy(Path("/x"), fault, uid=10001)
        assert "chown it" not in remedy.lower(), (
            f"{fault} is told to chown a path that cannot be chowned"
        )
    assert "chown it" in unwritable_remedy(
        Path("/x"), FAULT_NOT_WRITABLE, uid=10001
    ).lower()


def test_the_volume_variable_is_not_named_circularly():
    """
    The production line read "Resolved via RAILWAY_VOLUME_MOUNT_PATH: if that
    is RAILWAY_VOLUME_MOUNT_PATH, …" — my own f-string interpolating the
    resolution source into a sentence that then asked whether it was that
    source.
    """
    from src.utils.volume import ENV_RAILWAY_VOLUME

    remedy = unwritable_remedy(
        Path("/var/lib/postgresql/data"), FAULT_MISSING_MOUNT,
        uid=10001, how=ENV_RAILWAY_VOLUME,
    )
    assert f"if that is {ENV_RAILWAY_VOLUME}" not in remedy, (
        "the sentence still asks whether the resolution source is the source "
        "it was just given"
    )
    assert ENV_RAILWAY_VOLUME in remedy, (
        "the variable that produced the path is no longer named at all"
    )
    # And when the root came from somewhere else, the Railway hypothesis is
    # not asserted as though it had.
    other = unwritable_remedy(
        Path("/tmp/x"), FAULT_MISSING_MOUNT, uid=10001,
        how="the repository's own data/ directory (no volume)",
    )
    assert "set by hand" not in other


# --- the two probes agree -------------------------------------------------


@pytest.mark.parametrize("fault_maker, forbidden", [
    ("absent", "chown it"),
    ("file", "chown it"),
])
def test_both_probes_reach_the_same_verdict(fault_maker, forbidden, monkeypatch, tmp_path, caplog):
    """
    THE PRODUCTION CONTRADICTION, as a test. The same directory, both probes,
    and neither may give a remedy the other has ruled out.
    """
    from scripts.railway_healthcheck import run_checks
    from src.utils.volume import ENV_RAILWAY_VOLUME, ENV_STATE_DIR

    if fault_maker == "absent":
        target = tmp_path / "parent-denied" / "data"
    else:
        target = tmp_path / "a-file"
        target.write_text("", encoding="utf-8")

    monkeypatch.delenv(ENV_RAILWAY_VOLUME, raising=False)
    monkeypatch.setenv(ENV_STATE_DIR, str(target))
    monkeypatch.setenv(worker.ENV_CALIBRATION_REPORT, str(target / "calibration.json"))
    monkeypatch.setenv("PROPIQ_BOARD_MARKETS", "PTS")

    if fault_maker == "absent":
        # Deny the mkdir the way an unwritable parent would, since this
        # container runs as root and cannot be denied by mode bits.
        real_mkdir = type(tmp_path).mkdir

        def deny(self, *a, **k):
            if self == target or target in self.parents or self in target.parents:
                raise PermissionError(13, "Permission denied", str(self.parent))
            return real_mkdir(self, *a, **k)

        monkeypatch.setattr(type(tmp_path), "mkdir", deny)

    hc = {c.name: c for c in run_checks(skip_db=True)}["state_root_writable"].detail
    with caplog.at_level(logging.ERROR):
        worker.check_state_dir()
    wk = " ".join(r.getMessage() for r in caplog.records)

    assert forbidden not in hc.lower(), f"healthcheck: {hc}"
    assert forbidden not in wk.lower(), (
        "the worker still gives a remedy the healthcheck has ruled out for the "
        f"same path: {wk}"
    )


# --- cubic's nine findings on PR #11 ---------------------------------------


def test_a_non_directory_ancestor_is_not_a_missing_mount(tmp_path):
    """
    cubic, PR #11: the classifier looked only at the absent leaf. With
    ``PROPIQ_STATE_DIR=/tmp/file/state`` where ``/tmp/file`` is a FILE, the leaf
    is absent so this said "missing mount, attach a volume" — when the remedy
    is to remove the file standing where a directory has to be. `mkdir` raises
    ENOTDIR here, not ENOENT.
    """
    import errno

    obstruction = tmp_path / "file"
    obstruction.write_text("", encoding="utf-8")
    target = obstruction / "state"
    exc = NotADirectoryError(errno.ENOTDIR, "Not a directory", str(target))

    assert classify_unwritable(target, exc) == FAULT_NOT_A_DIRECTORY

    # AND WITHOUT THE ERRNO, which is the case the ancestor walk exists for: a
    # caller that has only the path, or an error carrying some other code. With
    # just the ENOTDIR branch this read as a missing mount.
    assert classify_unwritable(target) == FAULT_NOT_A_DIRECTORY, (
        "the absent leaf is classified without looking at the file standing in "
        "its path"
    )
    assert first_non_directory_ancestor(target) == obstruction

    remedy = unwritable_remedy(target, FAULT_NOT_A_DIRECTORY, uid=10001)
    assert str(obstruction) in remedy, (
        f"the remedy does not name the obstruction the operator must remove: {remedy}"
    )


def test_a_read_only_filesystem_is_not_an_ownership_problem(tmp_path):
    """
    cubic, PR #11: EROFS and ENOSPC on an existing directory were classified as
    FAULT_NOT_WRITABLE, whose remedy is chown. Neither is fixed by chown, and a
    full disk is not fixed by anything an operator does to permissions.
    """
    import errno

    for code, fault in (
        (errno.EROFS, FAULT_READ_ONLY_FS),
        (errno.ENOSPC, FAULT_NO_SPACE),
    ):
        exc = OSError(code, os.strerror(code), str(tmp_path))
        assert classify_unwritable(tmp_path, exc) == fault
        remedy = unwritable_remedy(tmp_path, fault, uid=10001)
        assert "chown it" not in remedy.lower(), f"{fault}: {remedy}"


def test_the_missing_mount_remedy_names_the_path_in_play(tmp_path):
    """
    cubic, PR #11: it always said to mount at /app/data, even when the resolver
    picked an explicit override. Mounting /app/data would leave the path the
    service actually uses absent.
    """
    absent = tmp_path / "srv" / "state"
    remedy = unwritable_remedy(
        absent, FAULT_MISSING_MOUNT, uid=10001, how=ENV_STATE_DIR_NAME
    )
    # THE MOUNT INSTRUCTION must name the path in play. /app/data may still
    # appear as what CLEARING the override would fall back to, which is useful
    # rather than wrong -- so the imperative is what this tests, not the word.
    assert f"mount a volume at {absent}" in remedy, (
        f"the remedy does not tell the operator to mount the path in play: {remedy}"
    )
    assert "mount it at /app/data" not in remedy, (
        f"an explicit override is told to mount the Dockerfile's path: {remedy}"
    )


def test_the_unwritable_remedy_does_not_assert_an_owner_it_cannot_know(tmp_path):
    """
    cubic, PR #11, and the sharpest of the nine: a directory the process owns,
    chmod 0o500, fails the probe on PERMISSIONS. Saying "it is owned by another
    uid" is a specific claim that is false there — the same overclaiming this
    whole helper exists to stop, committed inside it.
    """
    import errno

    exc = PermissionError(errno.EACCES, "Permission denied", str(tmp_path))
    assert classify_unwritable(tmp_path, exc) == FAULT_NOT_WRITABLE
    remedy = unwritable_remedy(tmp_path, FAULT_NOT_WRITABLE, uid=10001,
                               mount_point=False)
    assert "is owned by another uid" not in remedy, (
        f"asserts an owner the probe never established: {remedy}"
    )
    # It must still give the operator both candidate causes and a remedy.
    assert "chown" in remedy.lower()
    assert "mode" in remedy.lower() or "permission" in remedy.lower()


def test_both_probes_report_the_same_fault_not_merely_the_same_ban(
    monkeypatch, tmp_path
):
    """
    cubic, PR #11: the parity test only banned "chown it", so a file
    misclassified as a missing mount would have passed it. The two probes must
    agree on the FAULT.
    """
    import scheduler_worker as worker
    from scripts.railway_healthcheck import run_checks
    from src.utils.volume import ENV_RAILWAY_VOLUME, ENV_STATE_DIR

    as_a_file = tmp_path / "a-file"
    as_a_file.write_text("", encoding="utf-8")
    monkeypatch.delenv(ENV_RAILWAY_VOLUME, raising=False)
    monkeypatch.setenv(ENV_STATE_DIR, str(as_a_file))
    monkeypatch.setenv(worker.ENV_CALIBRATION_REPORT, str(as_a_file / "cal.json"))
    monkeypatch.setenv("PROPIQ_BOARD_MARKETS", "PTS")

    hc = {c.name: c for c in run_checks(skip_db=True)}["state_root_writable"]
    wk = worker.check_state_dir()

    assert hc.data["fault"] == FAULT_NOT_A_DIRECTORY
    assert wk["fault"] == FAULT_NOT_A_DIRECTORY, (
        f"healthcheck says {hc.data['fault']}, worker says {wk['fault']}"
    )


def test_the_probe_file_is_unique_per_process(tmp_path, monkeypatch):
    """
    cubic, PR #11: a fixed ``.propiq_write_probe`` name means two probes racing
    (the boot healthcheck and the worker's own check run seconds apart) can
    fail on each other's file, and a stale probe DIRECTORY left by a killed run
    makes `write_text` raise IsADirectoryError — reported as the parent being
    unwritable, which it is not.
    """
    import scheduler_worker as worker

    seen: list[str] = []
    real_write = type(tmp_path).write_text

    def record(self, *a, **k):
        seen.append(self.name)
        return real_write(self, *a, **k)

    monkeypatch.setattr(type(tmp_path), "write_text", record)
    monkeypatch.setenv(worker.ENV_CALIBRATION_REPORT, str(tmp_path / "cal.json"))
    worker.check_state_dir(str(tmp_path / "cal.json"))

    probes = [n for n in seen if "propiq" in n and "probe" in n]
    assert probes, f"no probe file was written: {seen}"
    assert probes[0] != ".propiq_write_probe", (
        "the probe filename is still fixed, so two concurrent probes collide"
    )
    assert str(os.getpid()) in probes[0], f"not process-unique: {probes[0]}"
