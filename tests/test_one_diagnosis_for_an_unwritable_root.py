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
from pathlib import Path

import pytest

import scheduler_worker as worker
from src.utils.volume import (
    FAULT_MISSING_MOUNT,
    FAULT_NOT_A_DIRECTORY,
    FAULT_NOT_WRITABLE,
    classify_unwritable,
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
