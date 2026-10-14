"""
src/utils/volume.py — where durable state lives, in a container or locally.

WHY THIS EXISTS. Railway mounts a persistent volume and exports its path as
``RAILWAY_VOLUME_MOUNT_PATH``. Nothing in this repository read it: every path
was relative to the working directory, which is correct locally and correct in
the image only because the Dockerfile happens to put the volume at
``/app/data``. A deploy that mounted it anywhere else would write to the
container filesystem instead, succeed, and lose everything at the next
redeploy — the same silent shape as the parlay ledger's CSV default.

THE ORDER IS MOST-EXPLICIT-FIRST, and every step is reported rather than
guessed at, because "which directory is this writing to" is the question a
redeploy makes expensive to get wrong:

  1. ``PROPIQ_STATE_DIR``, so an operator can override everything;
  2. ``RAILWAY_VOLUME_MOUNT_PATH``, the platform's own answer;
  3. ``/app/data`` when it exists, which is the Dockerfile's mount point;
  4. ``<repo>/data``, for a developer machine with no volume at all.

IT DOES NOT MOVE ANYTHING. The artifacts directory stays
``config/model_comparison.yaml``'s ``artifacts_dir``
(``data/external/model_runs/comparison``), which on a container with the volume
at ``/app/data`` already resolves onto the volume. ``artifact_dir_on`` exists to
ANSWER where that is for a given root, so a healthcheck and a deploy document
can agree with the resolver instead of asserting a path of their own.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Operator override, ahead of the platform's own variable.
ENV_STATE_DIR = "PROPIQ_STATE_DIR"
#: Railway exports this for a mounted volume.
ENV_RAILWAY_VOLUME = "RAILWAY_VOLUME_MOUNT_PATH"
#: Where the Dockerfile mounts it, and what docs/deploy_railway.md documents.
CONTAINER_STATE_DIR = Path("/app/data")
#: The repository's own data directory, for a developer machine.
REPO_STATE_DIR = Path(__file__).resolve().parents[2] / "data"

#: artifacts_dir from config/model_comparison.yaml, relative to the state root.
#: Duplicated as a RELATIVE path rather than re-read here so this module stays
#: importable without the config; ``artifact_dir_on`` prefers the config when it
#: can read it, and this is the fallback.
DEFAULT_ARTIFACT_SUBDIR = Path("external/model_runs/comparison")


def resolve_state_root() -> tuple[Path, str]:
    """
    The directory durable state is written to, and HOW it was chosen.

    Returns ``(path, how)``. The second element is the point: a log line saying
    "writing to /app/data via RAILWAY_VOLUME_MOUNT_PATH" and one saying
    "writing to ./data because no volume was found" describe very different
    deployments, and only one of them survives a redeploy.
    """
    override = (os.environ.get(ENV_STATE_DIR) or "").strip()
    if override:
        return Path(override), ENV_STATE_DIR

    railway = (os.environ.get(ENV_RAILWAY_VOLUME) or "").strip()
    if railway:
        return Path(railway), ENV_RAILWAY_VOLUME

    if CONTAINER_STATE_DIR.is_dir():
        return CONTAINER_STATE_DIR, "the container's /app/data mount point"

    return REPO_STATE_DIR, "the repository's own data/ directory (no volume)"


def artifact_dir_on(root: Path | None = None) -> Path:
    """
    Where trained model artifacts live under ``root``.

    Reads ``artifacts_dir`` from the comparison config when it can, so this
    agrees with ``main.resolve_model_artifact`` rather than asserting a second
    path. A relative ``artifacts_dir`` — which is what the config ships — is
    taken relative to the state root's PARENT, because the configured value
    already begins with ``data/``.
    """
    base = root if root is not None else resolve_state_root()[0]
    try:
        from src.models.compare import load_comparison_config

        configured = str((load_comparison_config() or {}).get("artifacts_dir") or "")
    except Exception:  # noqa: BLE001 — a missing config is not fatal here
        configured = ""

    if configured:
        path = Path(configured)
        # An operator who names an absolute path has named it. Redundant with
        # the join below -- pathlib's `/` already discards the left side for an
        # absolute right side -- and kept because relying on that is a trap for
        # the next edit, not a property anyone should have to remember.
        if path.is_absolute():
            return path
        # THE STATE ROOT *IS* THE DATA DIRECTORY, whatever it is called. The
        # configured value ships as "data/external/model_runs/comparison" —
        # relative to the repository root, where `data/` is the state
        # directory — so its leading `data` segment is the root itself and
        # joining it whole would give `<root>/data/external/...`.
        #
        # An earlier version compared that segment against `base.name`, which
        # worked only while the root happened to be named "data": set
        # PROPIQ_STATE_DIR=/mnt/vol and the artifacts resolved to
        # /mnt/external/..., one directory ABOVE the volume — a path that is
        # writable, that nothing reads, and that a redeploy destroys. Caught by
        # tests/test_volume.py::test_the_artifacts_dir_lands_inside_a_state_root_not_named_data.
        parts = path.parts
        if parts and parts[0] == "data":
            return base.joinpath(*parts[1:])
        return base / path
    return base / DEFAULT_ARTIFACT_SUBDIR


# ---------------------------------------------------------------------------
# WHY A FAILED WRITE PROBE FAILED, AND WHAT TO DO ABOUT IT
# ---------------------------------------------------------------------------
# ONE READER, because there are two probes. `scripts/railway_healthcheck.py`
# and `scheduler_worker.check_state_dir` both write a file into the state root
# and both report what happened, and on 2026-10-13 they contradicted each other
# about the same directory three log lines apart: the healthcheck said "NO
# VOLUME IS MOUNTED THERE ... chown is not the remedy here", the worker said
# "most likely owned by root ... chown it to that uid". The healthcheck had been
# taught the difference and the worker had not.
#
# Each caller still says WHAT LIVES in the directory it probed -- the artifact
# store and the calibration report really do break differently -- but the
# fault-to-remedy mapping lives here, once.

#: The path does not exist at all and could not be created.
FAULT_MISSING_MOUNT = "missing_mount"
#: Something IS there, but it is not a directory: a file, or a stale symlink.
FAULT_NOT_A_DIRECTORY = "not_a_directory"
#: A real directory that refuses the write on permissions.
FAULT_NOT_WRITABLE = "not_writable"
#: A real directory on a READ-ONLY filesystem. No chown fixes this.
FAULT_READ_ONLY_FS = "read_only_fs"
#: A real, writable directory on a FULL filesystem. Nothing about permissions
#: is wrong and nothing an operator does to them will help.
FAULT_NO_SPACE = "no_space"


def first_non_directory_ancestor(path: Path) -> Path | None:
    """
    The nearest existing ancestor that is NOT a directory, or None.

    cubic, PR #11: the classifier looked only at the absent leaf, so
    ``PROPIQ_STATE_DIR=/tmp/file/state`` with a FILE at ``/tmp/file`` read as a
    missing mount and the operator was told to attach a volume. ``mkdir``
    raises ENOTDIR there, not ENOENT, and the remedy is to move the file.
    """
    target = Path(path)
    for candidate in (target, *target.parents):
        if os.path.lexists(candidate) and not candidate.is_dir():
            return candidate
    return None


def classify_unwritable(path: Path, exc: BaseException | None = None) -> str:
    """
    Which fault a failed probe hit. ``exc`` is the error the probe raised.

    THE ERRNO DECIDES WHAT THE PATH CANNOT. cubic, PR #11: an existing
    directory on a read-only mount (EROFS) or a full disk (ENOSPC) was
    classified as "not writable", whose remedy is chown -- advice that cannot
    work for either. The errno is the only thing that separates them from a
    permissions failure, so it is read when offered and the path's own shape is
    the fallback.

    ``lexists`` RATHER THAN ``exists``, because `Path.exists()` follows a
    symlink and returns False for a broken one -- so a stale link would be
    classified as a missing mount, and the operator sent looking for a volume
    instead of removing the link.
    """
    import errno as _errno

    code = getattr(exc, "errno", None)
    if code == _errno.EROFS:
        return FAULT_READ_ONLY_FS
    if code == _errno.ENOSPC:
        return FAULT_NO_SPACE
    if code == _errno.ENOTDIR or first_non_directory_ancestor(path) is not None:
        return FAULT_NOT_A_DIRECTORY
    if not os.path.lexists(path):
        return FAULT_MISSING_MOUNT
    if not Path(path).is_dir():
        return FAULT_NOT_A_DIRECTORY
    return FAULT_NOT_WRITABLE


def unwritable_remedy(
    path: Path,
    fault: str,
    *,
    uid: object = "?",
    how: str | None = None,
    mount_point: bool = True,
) -> str:
    """
    What an operator should actually do, for the fault ``classify_unwritable``
    found. Never a remedy the fault rules out.

    ``how`` is ``resolve_state_root``'s second element. The Railway hypothesis
    is named ONLY when the platform's own variable produced the path: an
    earlier version interpolated ``how`` into "Resolved via {how}: if that is
    RAILWAY_VOLUME_MOUNT_PATH ...", which read as a sentence asking whether the
    source was the source it had just been handed.

    ``mount_point`` says whether a VOLUME is supposed to be there. The state
    root is one; the calibration report's directory, which an operator can
    point anywhere, is not -- and telling them to attach a volume because
    ``$PROPIQ_CALIBRATION_REPORT`` names a missing directory would be the
    wrong remedy with confident wording, which is the failure this whole
    helper exists to stop.
    """
    if fault == FAULT_NOT_A_DIRECTORY:
        where = (
            f"point ${ENV_STATE_DIR} at a directory"
            if mount_point
            else "point the variable that names it at a directory"
        )
        # WHICH path is the obstruction: it may be an ANCESTOR, not the target.
        obstruction = first_non_directory_ancestor(path)
        subject = (
            f"{obstruction} EXISTS BUT IS NOT A DIRECTORY, and {path} would "
            "have to live under it"
            if obstruction is not None and obstruction != Path(path)
            else f"{path} EXISTS BUT IS NOT A DIRECTORY"
        )
        return (
            f"{subject} -- a file, or a symlink with nothing at the other end "
            f"-- so nothing can be written under it. Remove or rename that "
            f"path, or {where}. chown cannot help: the problem is what the path "
            "IS, not who owns it."
        )
    if fault == FAULT_READ_ONLY_FS:
        return (
            f"{path} IS ON A READ-ONLY FILESYSTEM. Nothing can be written there "
            "whoever owns it, so permissions are not the problem: the mount "
            "itself is read-only. Remount it read-write, or point the state "
            "root at a writable filesystem. A redeploy will not change this."
        )
    if fault == FAULT_NO_SPACE:
        return (
            f"{path} IS ON A FULL FILESYSTEM. The directory and its permissions "
            "are fine; there is no room to write. Free space on that volume or "
            "give it more -- an artifact and a calibration report are small, so "
            "this usually means something else on the volume has grown."
        )
    if fault == FAULT_MISSING_MOUNT and not mount_point:
        return (
            f"{path} DOES NOT EXIST AND COULD NOT BE CREATED, and its parent is "
            "not writable by this process either. No volume belongs here -- this "
            "directory is wherever it was configured to be -- so create it and "
            f"give it to uid {uid}, or point the variable that names it at a "
            "directory that already exists. chown is not the remedy: there is "
            "nothing there to chown."
        )
    if fault == FAULT_MISSING_MOUNT:
        from_railway = how == ENV_RAILWAY_VOLUME
        source = (
            f"${ENV_RAILWAY_VOLUME} names a path this service has no volume at, "
            "which is the shape when it was set by hand or copied from a service "
            "that does have one -- the database's own "
            "/var/lib/postgresql/data, for instance. "
            if from_railway
            else f"The path came from {how}. " if how else ""
        )
        # WHERE TO MOUNT IT: the path actually in play. cubic, PR #11: this
        # always named /app/data, so an operator who had set $PROPIQ_STATE_DIR
        # to something else would mount the Dockerfile's path and leave the
        # directory this service really uses absent.
        from_override = how == ENV_STATE_DIR
        if from_override:
            fix = (
                f"Either mount a volume at {path} -- the path ${ENV_STATE_DIR} "
                f"names -- or clear ${ENV_STATE_DIR} so the state root falls "
                "back to /app/data, which is where the Dockerfile mounts one."
            )
        else:
            fix = (
                f"Attach a volume to THIS service and mount it at {path}, which "
                f"is the path in play (the Dockerfile documents /app/data); or "
                f"point ${ENV_STATE_DIR} at a directory that exists, accepting "
                "that a container path with no volume behind it is EPHEMERAL "
                "and a redeploy destroys the artifacts and the calibration "
                "evidence."
            )
        return (
            f"{path} DOES NOT EXIST AND COULD NOT BE CREATED, so NO VOLUME IS "
            f"MOUNTED THERE: a mounted volume would already be a directory, and "
            f"this failure is on the path or its parent. {source}{fix} chown is "
            "not the remedy: there is nothing there to chown."
        )
    # WHAT WAS OBSERVED, NOT WHO WE GUESS OWNS IT. cubic, PR #11: this said
    # "it is owned by another uid", a specific claim the probe never
    # established -- a directory this very process owns, mode 0500, fails the
    # same way. Both candidate causes are named; neither is asserted.
    if not mount_point:
        return (
            f"{path} EXISTS AND THE WRITE WAS REFUSED for uid {uid}: either its "
            "owner is a different uid, or its mode denies writing to this one. "
            "chown it to that uid or widen its mode, or point the variable that "
            "names it somewhere this process can write -- under the state root "
            "is the usual answer, because that is the directory the deploy "
            "already has to get right."
        )
    return (
        f"{path} EXISTS AND THE WRITE WAS REFUSED for uid {uid}: either its "
        "owner is a different uid, or its mode denies writing to this one. A "
        "mounted volume shadows the image's chown and is commonly root-owned, "
        f"which is the usual cause here -- chown it to uid {uid}, widen its "
        "mode, or run the service as root. See docs/deploy_railway.md."
    )
