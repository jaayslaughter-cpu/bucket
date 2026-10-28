"""
src/models/composite_features.py — rollups for the markets that are
combinations of other markets.

WHAT WAS MISSING, STATED NARROWLY. `src/quant/composite_markets.py` already
owns every fact about a composite market: `SUM_COMPOSITES` decomposes PA, PR,
RA and PRA, `config/fantasy_scoring.yaml` holds the operator's FPTS weights,
and `composite_mean` / `composite_variance` price one. Its only consumer was
`src/settlement/evaluator.py` — so this project could already GRADE a PA or an
FPTS prop and could not PROJECT one, because the feature matrix has no PA_L5
for a model to read. The decomposition existed; the features did not.

THIS MODULE IS NOT A SECOND DEFINITION OF A COMPOSITE. It reads
`SUM_COMPOSITES` and `fantasy_weights` and holds no table of its own. A second
copy of "PA is PTS plus AST" is exactly the defect
`attach_pra_from_components` was written to remove — there were two
definitions of PRA and the second silently won, agreeing on every row of the
archive panel and differing on the live one.

WHY IT IS IN `src/models/` AND NOT IN THE BUILDER, where the rest of the
rollups live. `AGENTS.md` states the layering: `src/features/` may not import
`src/models` or `src/quant`, and names the only two places that break it —
`tests/test_agents_md.py` asserts there are exactly two and fails on a third.
Reading `SUM_COMPOSITES` from inside `builder.py` would be a third. The only
alternative was to copy the decomposition into the features layer, which is
the defect above. So the attachment runs one step later, on the matrix the
builder returns, from a layer that is allowed to see both.

WHY SUMMING ROLLUPS IS THE CORRECT ESTIMATOR, and not a shortcut:
`composite_mean`'s own docstring says it —

    E[composite] = sum of w_i * E[component_i]. LINEARITY, SO NO CORRELATION
    TERM. This is exactly why `attach_pra_from_components` summing component
    rollups is correct and why the same cannot be said of the variance.

So a composite MEAN feature is a weighted sum of component mean features, with
each component keeping its own denominator. The VARIANCE is a different
matter and is NOT computed here: it needs the pairwise correlations that
`composite_variance` requires and refuses to default to zero, because PTS, REB
and AST are all driven by minutes and usage.

NEVER A PART SUM. A suffix whose component columns are not ALL present is
skipped, not summed over what happens to be there. PA from PTS alone is not a
worse PA — it is a number that looks like a PA and is not one. FPTS is the
sharp case: its TOV term is NEGATIVE, so dropping it does not add noise, it
biases every projection upward.

RESEARCH_ONLY. These are features. Nothing here prices, sizes or places
anything.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import pandas as pd

from src.quant.composite_markets import (
    FPTS,
    SUM_COMPOSITES,
    Composite,
    CompositeMarketError,
    fantasy_weights,
)

logger = logging.getLogger(__name__)

#: The rollup suffixes a composite is built at — the ones the builder produces
#: for every stat in `ROLLING_STATS`. `_L2_PACE` is deliberately absent: it is
#: a pace-adjusted derivative the builder computes AFTER PRA, and recomputing
#: it here from composite parts would be a second definition of it.
COMPOSITE_SUFFIXES: tuple[str, ...] = ("L2", "L5", "L10", "SEASON", "BASELINE")

#: Markets this module can attach. The sums come from `composite_markets`;
#: FPTS is weighted and needs an operator.
SUM_MARKETS: tuple[str, ...] = tuple(SUM_COMPOSITES)


@dataclass
class CompositeReport:
    """What was attached, and what was skipped and why."""

    attached: dict[str, list[str]] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)

    def lines(self) -> list[str]:
        out = []
        for market, cols in sorted(self.attached.items()):
            out.append(f"{market}: {len(cols)} column(s) — {', '.join(cols)}")
        for market, why in sorted(self.skipped.items()):
            out.append(f"{market}: SKIPPED — {why}")
        return out


def composite_spec(market: str, *, platform: str | None = None) -> Composite:
    """
    The weighted decomposition of one composite market.

    Delegates to `composite_markets` rather than deciding anything: an
    unweighted sum gets weight 1.0 per component, and FPTS gets the
    operator's. An unknown market raises rather than returning something
    plausible.
    """
    key = market.upper()
    if key in SUM_COMPOSITES:
        return Composite(key, {c: 1.0 for c in SUM_COMPOSITES[key]})
    if key == FPTS:
        # NO GUARD OF ITS OWN HERE, DELIBERATELY. A hand-written "platform is
        # required" check was removed on 2026-10-28 after mutation testing
        # showed it was not load-bearing: `fantasy_weights` already refuses an
        # absent or unknown operator with
        #
        #   DATA_NOT_AVAILABLE: config/fantasy_scoring.yaml has no fpts
        #   weights for 'None'. Known: ['prizepicks', 'underdog']
        #
        # which says more than the guard did. The guard also came with a
        # helper that re-read `fantasy_scoring.yaml` to list the platforms --
        # a second reader of the one fact that file owns, which is the defect
        # this repository keeps removing. One reader, and it is that function.
        return Composite(FPTS, fantasy_weights(str(platform or "")))
    raise CompositeMarketError(
        f"{market!r} is not a composite this project defines. Sums: "
        f"{', '.join(SUM_MARKETS)}; weighted: {FPTS}."
    )


def composite_columns(
    market: str,
    *,
    platform: str | None = None,
    suffixes: Sequence[str] = COMPOSITE_SUFFIXES,
) -> list[str]:
    """The column names `attach_composite_rollups` would produce."""
    composite_spec(market, platform=platform)  # raises on an unknown market
    return [f"{market.upper()}_{s}" for s in suffixes]


def _weighted_sum(
    frame: pd.DataFrame, weights: Mapping[str, float], suffix: str
) -> pd.Series | None:
    """
    One suffix's weighted sum, or None when a component column is absent.

    Returning None rather than a partial sum is the whole contract: see the
    module docstring on FPTS's negative TOV term.
    """
    cols = {name: f"{name}_{suffix}" for name in weights}
    missing = [c for c in cols.values() if c not in frame.columns]
    if missing:
        return None
    total: pd.Series | None = None
    for name, col in cols.items():
        term = pd.to_numeric(frame[col], errors="coerce") * float(weights[name])
        total = term if total is None else total + term
    return total


def attach_composite_rollups(
    df: pd.DataFrame,
    markets: Sequence[str] | None = None,
    *,
    platform: str | None = None,
    suffixes: Sequence[str] = COMPOSITE_SUFFIXES,
    overwrite: bool = False,
) -> tuple[pd.DataFrame, CompositeReport]:
    """
    Attach ``{MARKET}_{SUFFIX}`` for each composite market, from components.

    ``markets`` defaults to the unweighted sums only. FPTS is opt-in because
    it needs a platform, and because it needs TOV rollups which the builder
    only produces when TOV is in ``ROLLING_STATS``.

    PRA is skipped by default when it is already present: ``builder``'s
    ``attach_pra_from_components`` owns it, computes it identically, and two
    writers of one column is the defect this module refuses to become. Pass
    ``overwrite=True`` to recompute anyway.

    Returns the frame and a report. A skipped market is named with its reason
    rather than silently absent — a model contract that asks for PA_L5 and
    finds it missing abstains, and an abstention nobody can explain is only
    half an abstention.
    """
    out = df.copy()
    report = CompositeReport()
    wanted = list(markets) if markets is not None else list(SUM_MARKETS)

    for raw in wanted:
        market = str(raw).upper()
        try:
            spec = composite_spec(market, platform=platform)
        except CompositeMarketError as exc:
            report.skipped[market] = str(exc)
            logger.warning("Composite %s skipped: %s", market, exc)
            continue

        attached: list[str] = []
        blocked: list[str] = []
        for suffix in suffixes:
            column = f"{market}_{suffix}"
            if column in out.columns and not overwrite:
                # Already owned by someone else; do not become a second writer.
                continue
            series = _weighted_sum(out, spec.weights, suffix)
            if series is None:
                blocked.append(suffix)
                continue
            out[column] = series
            attached.append(column)

        if attached:
            report.attached[market] = attached
        if blocked and not attached:
            need = sorted(spec.weights)
            report.skipped[market] = (
                f"no component rollups present for {', '.join(blocked)}; "
                f"{market} = {' + '.join(f'{spec.weights[c]:g}*{c}' for c in need)} "
                f"and a partial sum would be a different quantity, not a "
                f"noisier one"
            )
            logger.warning("Composite %s skipped: %s", market, report.skipped[market])
        elif blocked:
            logger.info(
                "Composite %s attached at %s but not at %s — a component "
                "rollup is missing there.",
                market, ", ".join(c.split("_", 1)[1] for c in attached),
                ", ".join(blocked),
            )

    return out, report
