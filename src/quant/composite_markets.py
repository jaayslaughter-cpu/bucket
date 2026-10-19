"""
src/quant/composite_markets.py — one definition of every composite market.

WHY THIS MODULE EXISTS, in the words of the bug it prevents. `features/
builder.attach_pra_from_components` carries this note:

    THIS FUNCTION EXISTS BECAUSE THERE WERE TWO DEFINITIONS AND THE SECOND
    SILENTLY WON. [...] Both looked right, both were plausible, and they
    agreed on every row of the archive panel [...] so the divergence was
    invisible on the only data available.

PRA was ONE composite and it produced that defect. Adding PA, PR, RA and FPTS
by the same hand-rolled route would be four more chances at it, so the
decomposition lives here once and every consumer reads it.

WEIGHTED, NOT JUST SUMMED. PA/PR/RA/PRA are unweighted sums, so a dict of
components would have done. FPTS is not: it is
``1.0*PTS + 1.2*REB + 1.5*AST + 3.0*STL + 3.0*BLK - 1.0*TOV``, including a
NEGATIVE weight on turnovers, which changes the arithmetic in two ways that
matter. The mean is still linear, but the variance is not a plain sum, and the
turnover term REDUCES the composite while being positively correlated with
usage — so a naive sum of variances is wrong in both magnitude and direction.
One weighted representation covers both cases and keeps the FPTS weights where
an operator's terms belong: `config/fantasy_scoring.yaml`, dated and
unverified, exactly as `dfs_payouts.yaml` holds their multiples.

THE MEAN NEEDS NO CORRELATION. THE VARIANCE DOES, AND IT IS NOT OPTIONAL.
E[sum of w_i X_i] = sum of w_i E[X_i] by linearity, whatever the
correlations — so summing component projections is correct for the mean, and
that is why `attach_pra_from_components` is right to do it. Variance is not:

    Var = SUM_i w_i^2 Var_i + 2 * SUM_{i<j} w_i w_j rho_ij sd_i sd_j

Dropping the covariance term understates the spread of PRA badly, because PTS,
REB and AST are all driven by minutes and usage and are positively correlated.
A too-narrow distribution makes every Over and Under look more certain than it
is, which inflates confidence on exactly the markets this module adds. The
covariance term is therefore REQUIRED here, not defaulted: `composite_variance`
refuses a correlation it was not given rather than assuming independence.

RESEARCH_ONLY. This computes projections, not wagers.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

logger = logging.getLogger(__name__)

#: The six box-score components every composite here is built from, in PANEL
#: column names (`espn_game` and the training panel both use these).
COMPONENTS = ("PTS", "REB", "AST", "STL", "BLK", "TOV")

#: Unweighted sums. The key is the market as the board and the ledger name it.
SUM_COMPOSITES: dict[str, tuple[str, ...]] = {
    "PA":  ("PTS", "AST"),
    "PR":  ("PTS", "REB"),
    "RA":  ("REB", "AST"),
    "PRA": ("PTS", "REB", "AST"),
}

#: Where the operator's FPTS weights live. Not a constant in code: see header.
FANTASY_CONFIG = Path("config/fantasy_scoring.yaml")

FPTS = "FPTS"


class CompositeMarketError(ValueError):
    """A composite could not be formed from what was supplied."""


@dataclass(frozen=True)
class Composite:
    """One composite market as a weighted combination of components."""

    market: str
    weights: Mapping[str, float]

    def __post_init__(self) -> None:
        # FROZEN STOPS REBINDING, NOT MUTATION. `spec.weights = {...}` raised
        # and `spec.weights["REB"] = 99.0` did not, so a caller holding the
        # PRA spec could reprice every PRA projection in the process -- and
        # the specs come from module-level tables and a cached config read, so
        # the change would outlive the caller that made it. That is the
        # two-definitions defect again, except the second definition is the
        # first one after someone edited it in place.
        object.__setattr__(self, "weights", MappingProxyType(dict(self.weights)))

    @property
    def components(self) -> tuple[str, ...]:
        return tuple(self.weights)


def fantasy_weights(platform: str, *, path: Path | None = None) -> dict[str, float]:
    """
    The FPTS multipliers for one operator, read from dated config.

    Raises rather than defaulting. A missing or misspelled platform returning
    a silent default would price every FPTS line on the wrong operator's terms
    and look entirely normal doing it.
    """
    import yaml

    config_path = path or FANTASY_CONFIG
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except OSError as exc:
        raise CompositeMarketError(
            f"DATA_NOT_AVAILABLE: cannot read {config_path} for FPTS weights: {exc}"
        ) from exc

    platforms = (raw.get("platforms") or {})
    block = platforms.get(str(platform).strip().lower())
    if not block or not block.get("fpts"):
        raise CompositeMarketError(
            f"DATA_NOT_AVAILABLE: {config_path} has no fpts weights for "
            f"{platform!r}. Known: {sorted(platforms)}"
        )
    matrix = block["fpts"]
    if not isinstance(matrix, Mapping):
        raise CompositeMarketError(
            f"{config_path}: {platform!r} fpts must be a mapping of component "
            f"-> multiplier, got {type(matrix).__name__}"
        )

    weights: dict[str, float] = {}
    for key, value in matrix.items():
        name = str(key).strip().upper()
        try:
            weight = float(str(value).strip())
        except (TypeError, ValueError) as exc:
            raise CompositeMarketError(
                f"{config_path}: {platform!r} FPTS weight for {name} is "
                f"{value!r}, which is not a number"
            ) from exc
        # A NON-FINITE WEIGHT POISONS THE WHOLE COMPOSITE. `inf * PTS` makes
        # every FPTS line infinitely over and `nan` makes every comparison
        # False, which grades as a confident Under. Both look like arithmetic
        # rather than a config error anywhere downstream.
        if not math.isfinite(weight):
            raise CompositeMarketError(
                f"{config_path}: {platform!r} FPTS weight for {name} is "
                f"{value!r}, which is not finite"
            )
        weights[name] = weight

    unknown = sorted(set(weights) - set(COMPONENTS))
    if unknown:
        raise CompositeMarketError(
            f"{config_path} gives {platform!r} FPTS weights for {unknown}, which "
            f"are not box-score components this project reads ({list(COMPONENTS)})"
        )
    # EVERY COMPONENT, OR IT IS NOT A SCORING MATRIX. A missing TOV does not
    # mean turnovers are free -- it means nobody said what they cost, and
    # treating absence as zero scores a six-component product on five and
    # reports a plausible number for it. An operator who genuinely does not
    # score a component writes `0.0`, which is the whole difference between
    # "zero" and "forgotten".
    missing = sorted(set(COMPONENTS) - set(weights))
    if missing:
        raise CompositeMarketError(
            f"DATA_NOT_AVAILABLE: {config_path} gives {platform!r} no FPTS "
            f"weight for {missing}. A component with no stated multiplier is "
            f"not a component worth zero; write 0.0 if that is the term."
        )
    return weights


def composite(market: str, *, platform: str | None = None,
              path: Path | None = None) -> Composite:
    """
    The weighted definition of one composite market.

    FPTS needs a platform because the weights are an operator term; the sums do
    not, because a sum is arithmetic.
    """
    key = str(market).strip().upper()
    if key in SUM_COMPOSITES:
        return Composite(key, {c: 1.0 for c in SUM_COMPOSITES[key]})
    if key == FPTS:
        if not platform:
            raise CompositeMarketError(
                "FPTS needs a platform: the multipliers are an operator's "
                "terms, not a property of basketball. See "
                "config/fantasy_scoring.yaml."
            )
        return Composite(key, fantasy_weights(platform, path=path))
    raise CompositeMarketError(
        f"{market!r} is not a composite market here. Known: "
        f"{sorted(SUM_COMPOSITES) + [FPTS]}"
    )


def _finite(value: Any, market: str, name: str) -> float:
    """
    One reader for "this number must be a real number".

    NaN AND inf ARE NOT PROJECTIONS. `float("nan")` propagated straight
    through the arithmetic and out, and a NaN composite is worse than a
    missing one: it compares False against every line, so the Under grades as
    a confident win and nothing anywhere says the projection was absent.
    """
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise CompositeMarketError(
            f"{market}: {name} is {value!r}, which is not a number"
        ) from exc
    if not math.isfinite(number):
        raise CompositeMarketError(
            f"DATA_NOT_AVAILABLE: {market} was given {name}={value!r}, which is "
            f"not finite. A non-finite component makes the whole composite "
            f"unusable, so this abstains rather than passing it on."
        )
    return number


def composite_mean(spec: Composite, component_means: Mapping[str, Any]) -> float:
    """
    E[composite] = sum of w_i * E[component_i].

    LINEARITY, SO NO CORRELATION TERM. This is exactly why
    `attach_pra_from_components` summing component rollups is correct and why
    the same cannot be said of the variance below.
    """
    total = 0.0
    for name, weight in spec.weights.items():
        if name not in component_means or component_means[name] is None:
            raise CompositeMarketError(
                f"DATA_NOT_AVAILABLE: {spec.market} needs a projection for "
                f"{name}; got {sorted(k for k, v in component_means.items() if v is not None)}"
            )
        total += weight * _finite(component_means[name], spec.market, name)
    # ABSTAIN RATHER THAN RETURN A NUMBER THAT IS NOT ONE. Belt to the braces
    # above: finite inputs can still overflow to inf once weighted and summed,
    # and a NaN composite compares False against every line, which the grader
    # reads as a confident Under rather than as a missing projection.
    if not math.isfinite(total):
        raise CompositeMarketError(
            f"DATA_NOT_AVAILABLE: {spec.market} totals {total} from finite "
            f"components, so the projection is not usable"
        )
    return total


def composite_variance(
    spec: Composite,
    component_sds: Mapping[str, Any],
    correlations: Mapping[tuple[str, str], Any],
) -> float:
    """
    Var = SUM w_i^2 sd_i^2 + 2 * SUM_{i<j} w_i w_j rho_ij sd_i sd_j.

    ``correlations`` IS REQUIRED AND IS NOT DEFAULTED TO ZERO. Independence is
    the one assumption that is certainly false here: PTS, REB and AST are all
    driven by minutes and usage, so omitting the covariance term understates
    the spread and makes every Over and Under on a composite look more certain
    than it is. A missing pair raises instead, so the caller must supply a
    number or abstain -- `quant.leg_correlation.fit_leg_correlations` is where
    fitted ones come from, with its own `usable` flag for thin buckets.

    A NEGATIVE WEIGHT IS NOT A SPECIAL CASE, which is the point of carrying
    weights rather than a component list: ``w_i w_j`` handles the sign, so a
    turnover term that is positively correlated with usage correctly REDUCES
    the composite's variance contribution against the positive terms.
    """
    names = spec.components
    sds: dict[str, float] = {}
    for name in names:
        if name not in component_sds or component_sds[name] is None:
            raise CompositeMarketError(
                f"DATA_NOT_AVAILABLE: {spec.market} variance needs an sd for {name}"
            )
        sd = _finite(component_sds[name], spec.market, f"sd for {name}")
        if sd < 0:
            raise CompositeMarketError(
                f"{spec.market}: sd for {name} is negative, which is not an sd"
            )
        sds[name] = sd

    # THE MATRIX, CHECKED AS A MATRIX. Testing only `total < 0` caught the
    # cases where inconsistency happened to show up in THIS composite's
    # scalar, and passed every other one: {PTS,REB}=+0.9, {PTS,AST}=+0.9,
    # {REB,AST}=-0.9 cannot be realised by any three random variables, yet it
    # yields a comfortably positive variance at equal sds. The projection
    # built on it is then a confident number about a world that cannot exist,
    # and the Over/Under either side of it inherits that confidence.
    #
    # `psd_repair.is_psd` is the front door for this question and already has
    # the eigenvalue reader; a second one here would be the duplication this
    # tree keeps removing.
    _require_consistent(correlations, names, market=spec.market)

    total = 0.0
    for name in names:
        total += (spec.weights[name] ** 2) * (sds[name] ** 2)

    for i, a in enumerate(names):
        for b in names[i + 1:]:
            rho = _correlation(correlations, a, b, market=spec.market)
            total += (
                2.0 * spec.weights[a] * spec.weights[b] * rho * sds[a] * sds[b]
            )

    if total < 0:
        # Belt to the braces: a PSD correlation matrix cannot produce this,
        # but an sd set at the edge of floating point can, and returning a
        # negative variance hands the caller an imaginary sd.
        raise CompositeMarketError(
            f"{spec.market}: the supplied correlations imply a NEGATIVE variance "
            f"({total:.4f}), so they are not a consistent (PSD) matrix. Repair "
            f"them with quant.psd_repair rather than taking a root of this."
        )
    return total


#: How far below zero an eigenvalue may sit and still be called rounding. A
#: 6x6 correlation matrix assembled from separately fitted pairs lands a
#: genuine zero eigenvalue at about -1e-16; -1e-9 is many orders above that
#: and many orders below any real inconsistency.
PSD_TOLERANCE = -1e-9


def _require_consistent(
    correlations: Mapping[tuple[str, str], Any],
    names: tuple[str, ...],
    *,
    market: str,
) -> None:
    """
    Refuse a correlation set no three random variables could produce.

    PAIRS FITTED SEPARATELY DO NOT HAVE TO AGREE WITH EACH OTHER.
    `leg_correlation.fit_leg_correlations` fits one bucket at a time, so
    nothing upstream makes PTSxREB, PTSxAST and REBxAST mutually consistent --
    and for a two-component composite the question does not arise, which is
    why it went unnoticed on PR and PA.
    """
    if len(names) < 3:
        # Any single |rho| <= 1 is realisable by two variables, and
        # `_correlation` already enforces that bound.
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                _correlation(correlations, a, b, market=market)
        return

    import numpy as np

    from src.quant.psd_repair import is_psd

    size = len(names)
    matrix = np.eye(size)
    for i, a in enumerate(names):
        for j in range(i + 1, size):
            rho = _correlation(correlations, a, names[j], market=market)
            matrix[i, j] = matrix[j, i] = rho

    if not is_psd(matrix, eps=PSD_TOLERANCE):
        worst = float(np.linalg.eigvalsh(matrix).min())
        raise CompositeMarketError(
            f"{market}: the supplied correlations are not a consistent (PSD) "
            f"set over {list(names)} — smallest eigenvalue {worst:.6g}. No "
            f"three box-score components can be correlated this way, so the "
            f"variance computed from them would be a confident number about "
            f"an impossible world. Repair them with quant.psd_repair."
        )


def _correlation(
    correlations: Mapping[tuple[str, str], Any], a: str, b: str, *, market: str
) -> float:
    for key in ((a, b), (b, a)):
        if key in correlations and correlations[key] is not None:
            rho = float(correlations[key])
            if not -1.0 <= rho <= 1.0:
                raise CompositeMarketError(
                    f"{market}: correlation for {key} is {rho}, outside [-1, 1]"
                )
            return rho
    raise CompositeMarketError(
        f"DATA_NOT_AVAILABLE: {market} variance needs a correlation for "
        f"({a}, {b}) and none was supplied. Independence is not a safe default "
        f"for box-score components that share minutes and usage."
    )


def realised_composite(
    spec: Composite, component_values: Mapping[str, Any]
) -> float:
    """
    The composite's realised value from a graded box score.

    Same weights as the projection, deliberately: grading a composite with
    different arithmetic from the one that projected it is the two-definitions
    defect this module was written to prevent.
    """
    return composite_mean(spec, component_values)
