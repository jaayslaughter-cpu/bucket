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
