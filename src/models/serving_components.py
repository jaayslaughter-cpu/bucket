"""
Which components can serve a probability at a POSTED line, and how to reach them.

WHY THIS EXISTS. `main.resolve_model_artifact` globbed `{component}_{MARKET}.json`
for every component, and `score_prob_over_by_market` chose a scorer with an
`if component == "distribution"` branch. Both encoded, separately, facts that
belong to the model classes: what their artifact is called and how it is
loaded. `line_aware` is what exposed it -- training persists
`line_aware_PTS.manifest.json`, the resolver looked for `line_aware_PTS.json`,
and a component that was fitted, checksummed and verified on every run was
unreachable. The suffix now comes from the class that writes the file.

ONLY LINE-TAKING COMPONENTS ARE LISTED. The entries here answer
`predict_probability_over(features, line)` -- the question the board actually
asks. The XGBoost booster does not: it learns P(stat > RESEARCH_LINE) for the
one line its labels were built from, which is why it has its own scorer and
its own mask. Adding it here would hand the shared scorer a model that returns
the same number at every line.

BEING RESOLVABLE IS NOT BEING SERVED. `$PROPIQ_SCORING_COMPONENT` still selects,
and it still defaults to `distribution`. Listing a component here makes it
*available*; promoting it is a separate decision that needs forward evidence.

RESEARCH_ONLY.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)

#: The suffix used when a class does not declare one.
DEFAULT_ARTIFACT_SUFFIX = ".json"


class LineTakingModel(Protocol):
    """The contract the shared posted-line scorer depends on."""

    target_market: str

    def predict_probability_over(self, features: Any, line: Any) -> Any: ...


def _distribution() -> Any:
    from src.models.distribution_adapter import DistributionPropModel

    return DistributionPropModel


def _line_aware() -> Any:
    from src.models.line_aware import LineAwarePropModel

    return LineAwarePropModel


def _minutes_aware() -> Any:
    from src.models.minutes_aware import MinutesAwarePropModel

    return MinutesAwarePropModel


#: component name -> a thunk returning its class. Imported lazily because
#: `line_aware` pulls in the booster adapters, and the resolver is called on
#: paths where that cost is pure waste.
LINE_TAKING_COMPONENTS: dict[str, Any] = {
    "distribution": _distribution,
    "line_aware": _line_aware,
    # `distribution`'s mean with the minutes variance added. Listed so it can
    # be SHADOW-scored beside whatever serves; see its module docstring on why
    # its phi is not interchangeable with `distribution`'s.
    "minutes_aware": _minutes_aware,
}


def is_line_taking(component: str) -> bool:
    return (component or "").strip().lower() in LINE_TAKING_COMPONENTS


def artifact_suffix(component: str) -> str:
    """
    The filename suffix `{component}_{MARKET}` is published under.

    Read from the class that writes the file, so the resolver cannot look for
    a name training never produces. Unknown components get the default rather
    than raising: the resolver's job is to find a file or say it found none,
    not to adjudicate component names.
    """
    thunk = LINE_TAKING_COMPONENTS.get((component or "").strip().lower())
    if thunk is None:
        return DEFAULT_ARTIFACT_SUFFIX
    try:
        return str(getattr(thunk(), "ARTIFACT_SUFFIX", DEFAULT_ARTIFACT_SUFFIX))
    except Exception as exc:  # noqa: BLE001 — a missing optional dependency
        logger.debug("Could not read %s's artifact suffix (%s)", component, exc)
        return DEFAULT_ARTIFACT_SUFFIX


def load_line_taking(component: str, path: Path | str) -> Any:
    """
    Load a fitted line-taking component from `path`.

    TWO LOAD SHAPES, ONE CALLER. `DistributionPropModel.load` is an instance
    method returning self; `LineAwarePropModel.load` is a classmethod. A
    scorer that hardcoded either would break on the other, so the difference
    is absorbed here instead of at the call site.
    """
    key = (component or "").strip().lower()
    thunk = LINE_TAKING_COMPONENTS.get(key)
    if thunk is None:
        raise ValueError(
            f"{component!r} does not answer at a posted line; "
            f"line-taking components are {sorted(LINE_TAKING_COMPONENTS)}"
        )
    cls = thunk()
    loader = cls.load
    if getattr(loader, "__self__", None) is cls:  # classmethod
        return loader(Path(path))
    return cls().load(Path(path))
