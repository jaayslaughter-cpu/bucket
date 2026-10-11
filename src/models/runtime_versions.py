"""
src/models/runtime_versions.py — which libraries produced an artifact.

WHY THIS EXISTS. `pyproject.toml` declared `xgboost>=2.0` with no upper bound,
so a rebuild resolved whatever PyPI served that day — and the artifact on the
mounted volume was saved by whatever was current when it was trained. The two
are decoupled by design (the volume survives the redeploy; the image does
not), which means a `docker build` months later can pair a fresh major version
of XGBoost with a booster saved by an older one.

The sidecar recorded `model_version` and `feature_schema_version` and no
library version at all, so that pairing was undetectable: the booster either
loaded with possibly-changed behaviour, or raised something opaque from inside
the C++ layer. A probability that moved because the serving library changed is
indistinguishable, downstream, from a probability that moved because the player
did.

So: record the versions on save, and compare MAJORS on load. A major bump is
reported; a patch or minor difference is not, because warning on every patch
would train everyone to ignore the line that matters.

It never raises and never refuses to load. A warning beside a working booster
is useful; refusing to score tonight's slate over a version string is not.

RESEARCH_ONLY. Metadata about libraries. No odds, no wager.
"""

from __future__ import annotations

import importlib.metadata as _md
import logging
import sys

logger = logging.getLogger(__name__)

#: The libraries whose version can change a saved model's behaviour. Narrow on
#: purpose: recording every installed package would turn the sidecar into a
#: lockfile, and a lockfile does not belong beside a booster.
TRACKED = ("xgboost", "catboost", "scikit-learn", "numpy", "pandas", "scipy")

#: The sidecar key. Absent in every artifact saved before 2026-10-09, which is
#: why `compare` treats a missing block as "unknown" rather than a mismatch.
META_KEY = "runtime_versions"


def collect() -> dict[str, str]:
    """
    The installed versions of `TRACKED`, plus the interpreter.

    A library that is not installed is OMITTED rather than recorded as null:
    CatBoost is an optional extra, and "absent at training time" and "present
    at version null" are different facts.
    """
    out: dict[str, str] = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}"
                  f".{sys.version_info.micro}",
    }
    for name in TRACKED:
        try:
            out[name] = _md.version(name)
        except _md.PackageNotFoundError:
            continue
        except Exception as exc:  # noqa: BLE001 — metadata must not break a save
            logger.debug("Could not read the installed version of %s: %s", name, exc)
    return out


def _major(version: str) -> str:
    return (version or "").split(".", 1)[0].strip()


def compare(recorded: dict[str, str] | None) -> list[str]:
    """
    Majors that differ between `recorded` and what is installed now.

    Returns a list of human-readable differences, empty when there are none.
    An absent or empty `recorded` returns empty: every artifact saved before
    this module existed has no block, and reporting all of those as mismatched
    would be noise about nothing.
    """
    if not recorded:
        return []
    now = collect()
    out: list[str] = []
    for name, was in sorted(recorded.items()):
        is_now = now.get(name)
        if not is_now or not was:
            continue
        if _major(was) != _major(is_now):
            out.append(f"{name} {was} -> {is_now}")
    return out


def warn_on_mismatch(recorded: dict[str, str] | None, artifact: str) -> list[str]:
    """Compare and log. Returns the differences so a caller can report them."""
    diffs = compare(recorded)
    if diffs:
        logger.warning(
            "%s was saved under a DIFFERENT MAJOR VERSION of: %s. The booster "
            "may load and score differently than it did when it was validated, "
            "and a probability that moved for that reason is indistinguishable "
            "downstream from one that moved because the player did. Retrain "
            "under the installed versions, or pin the image back.",
            artifact, ", ".join(diffs),
        )
    return diffs
