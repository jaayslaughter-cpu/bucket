"""Read the EV gate's route and take a pick'em row to the payout engine.

RESEARCH_ONLY. Nothing here places a wager, sizes a stake, or contacts an
operator's order API.

WHY THIS MODULE EXISTS. ``contracts.market_ev_gate`` was corrected to return
``route = PICKEM_ENTRY_ROUTE`` on a pick'em row instead of dead-ending it, and
the reason string now names ``dfs_payouts.evaluate_pickem_entry`` as the path
that applies. That correction was only half done: NOTHING IN THIS REPOSITORY
READ ``route``, so every pick'em row still ended at an abstention and the
advertised path was unreachable. A route no caller reads is a dead end with
extra steps. This module is the caller.

It is the same defect this codebase keeps producing in different clothes — a
feature registered in the builder but absent from the feature contract, a config
block with no reader, a table with a grader and no writer — so it is named here
rather than just fixed.

WHAT THE ROUTING ACTUALLY DECIDES. Per leg:

  route == PICKEM_ENTRY_ROUTE   the operator posted a payout multiplier, not a
                                price. Correct for this path. Its probability
                                must come from somewhere other than the row.
  status == GATE_READY          this row IS a two-way market with real American
                                odds on both sides. It does not belong in a
                                pick'em slip, and pricing it against a payout
                                table would be measuring an operator's matrix
                                against a sportsbook's quote. Refused, named.
  anything else                 the gate abstained for its own reason (status
                                not VALID, and so on), which is passed through
                                verbatim rather than restated.

WHERE THE LEG PROBABILITY COMES FROM, IN PRIORITY ORDER, AND WHY.

1. A SHARP TWO-WAY BENCHMARK for the same player, market and line, de-vigged by
   ``dfs_payouts.benchmark_fair_probability``. This is the only source that makes
   the resulting EV a claim about a market disagreement rather than about the
   model.

2. THE MODEL'S OWN PROBABILITY, used only when no benchmark was supplied at all.
   The result is then the model's claim restated in payout terms and inherits its
   calibration error whole.

THE LINE MUST MATCH EXACTLY, AND A MISMATCH ABSTAINS RATHER THAN FALLING BACK.
A benchmark at 25.5 does not price a pick'em leg at 24.5. When a benchmark is
supplied and its line differs, this refuses the leg instead of quietly using the
model number: the mismatch is a data-join fault worth surfacing, and papering
over it with a different probability source is how a confidently wrong edge gets
published. Half a point on a scoring prop is routinely worth several points of
probability.

A ONE-SIDED BENCHMARK IS NOT A BENCHMARK. With only an over price there is
nothing to de-vig against, and the raw implied probability still carries the
book's hold — typically 2 to 5 points on a prop, in the direction that flatters
the entry. Refused.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from src.quant.contracts import (
    GATE_READY,
    PICKEM_ENTRY_ROUTE,
    MarketContext,
    PropMarketSnapshot,
    market_ev_gate,
)
from src.quant.devig_methods import MULTIPLICATIVE
from src.quant.dfs_payouts import (
    PAYOUT_EV_ABSTAIN,
    DfsPayoutError,
    DfsPayoutStructure,
    PayoutEvaluation,
    ProbabilitySource,
    benchmark_fair_probability,
    evaluate_pickem_entry,
)

logger = logging.getLogger(__name__)

LEG_READY = "LEG_READY"
LEG_ABSTAIN = "DATA_NOT_AVAILABLE"

# Lines are quoted in halves and quarters, so equality is exact in practice.
# The tolerance is here for float round-tripping through JSON and YAML, not to
# admit a near-miss: 24.5 and 25.0 are half a point apart and must not match.
LINE_MATCH_TOLERANCE = 1e-6


class DfsEntryError(ValueError):
    """A slip that cannot be assembled at all, as opposed to one that abstains."""


@dataclass(frozen=True)
class PickemLeg:
    """
    One leg of a pick'em slip: the operator's row plus where its probability
    may come from.

    ``market`` is the OPERATOR's row — the thing carrying the payout multiplier.
    ``benchmark_*`` is a different source's two-way quote on the same player,
    market and line. Keeping them in separate fields is deliberate: collapsing
    them into one "odds" field is what lets an operator's multiplier get
    de-vigged as if it were a price.
    """

    leg_id: str
    market: MarketContext | PropMarketSnapshot
    side: str = "over"
    benchmark_over_american: int | None = None
    benchmark_under_american: int | None = None
    benchmark_line: float | None = None
    benchmark_source: str | None = None
    model_probability: float | None = None
    # Which de-vig to apply to the benchmark pair. The default is the only one
    # this pipeline used before alternatives existed, so an unset leg behaves
    # exactly as it did. See src.quant.devig_methods for what the choice is
    # worth: under a third of a point on a normal prop price, one to two points
    # on a heavy favourite.
    devig_method: str = MULTIPLICATIVE


@dataclass(frozen=True)
class LegResolution:
    """What a leg resolved to, and on whose authority."""

    leg_id: str
    status: str
    probability: float | None = None
    probability_source: ProbabilitySource = ProbabilitySource.UNSPECIFIED
    side: str | None = None
    line: float | None = None
    reason: str | None = None
    benchmark_source: str | None = None
    # None on a model-sourced leg: no de-vig happened, and recording a method
    # there would suggest a market price was involved.
    devig_method: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "LEG_ID": self.leg_id,
            "STATUS": self.status,
            "SIDE": self.side,
            "LINE": self.line,
            "PROBABILITY": (
                round(self.probability, 6) if self.probability is not None else None
            ),
            "PROBABILITY_SOURCE": self.probability_source.value,
            "BENCHMARK_SOURCE": self.benchmark_source,
            "DEVIG_METHOD": self.devig_method,
            "REASON": self.reason,
        }


@dataclass
class EntryEvaluation:
    """A routed slip: every leg's resolution plus the entry-level pricing."""

    status: str
    structure_label: str
    legs: list[LegResolution] = field(default_factory=list)
    payout: PayoutEvaluation | None = None
    reason: str | None = None

    @property
    def probability_source(self) -> ProbabilitySource:
        return (
            self.payout.probability_source
            if self.payout is not None
            else ProbabilitySource.UNSPECIFIED
        )

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "STATUS": self.status,
            "STRUCTURE": self.structure_label,
            "REASON": self.reason,
            "LEGS": [leg.as_dict() for leg in self.legs],
        }
        if self.payout is not None:
            out["ENTRY"] = self.payout.as_dict()
        return out


def _as_context(market: MarketContext | PropMarketSnapshot) -> MarketContext:
    return (
        market.to_market_context()
        if isinstance(market, PropMarketSnapshot)
        else market
    )


def _abstain_leg(leg: PickemLeg, reason: str, line: float | None = None) -> LegResolution:
    return LegResolution(
        leg_id=leg.leg_id,
        status=LEG_ABSTAIN,
        side=leg.side,
        line=line,
        reason=reason,
        benchmark_source=leg.benchmark_source,
    )


def resolve_leg_probability(leg: PickemLeg) -> LegResolution:
    """
    Route one leg through the gate, then give it a probability or a named refusal.

    This is where ``market_ev_gate``'s ``route`` field is read. The gate is asked
    even though the answer for a pick'em row is known in advance, because asking
    is what keeps the pick'em definition in one place: a row that stops being
    pick'em, or that never had VALID status, is caught here rather than priced.
    """
    context = _as_context(leg.market)
    side = str(leg.side).strip().lower()
    if side not in {"over", "under"}:
        return _abstain_leg(leg, f"side must be 'over' or 'under', got {leg.side!r}")

    verdict = market_ev_gate(context)
    route = verdict.get("route")
    if route != PICKEM_ENTRY_ROUTE:
        if verdict.get("status") == GATE_READY:
            return _abstain_leg(
                leg,
                "this row is a genuine two-way market, not a pick'em leg: it "
                "cleared the EV gate with prices on both sides. Price it with "
                "ev_engine instead; measuring a sportsbook quote against an "
                "operator's payout table compares two different products.",
                context.line,
            )
        return _abstain_leg(
            leg,
            f"gate did not route this row to the pick'em path: "
            f"{verdict.get('reason')}",
            context.line,
        )

    line = context.line
    if line is None or not math.isfinite(float(line)):
        return _abstain_leg(
            leg,
            "the operator's row carries no finite line, so there is nothing for "
            "a probability to be 'at' and no benchmark line to match against",
        )

    over = leg.benchmark_over_american
    under = leg.benchmark_under_american

    if over is not None and under is not None:
        if leg.benchmark_line is None:
            return _abstain_leg(
                leg,
                "a two-way benchmark was supplied with no line, so it cannot be "
                "shown to price the same contract as the operator's row",
                line,
            )
        if not math.isclose(
            float(leg.benchmark_line), float(line), abs_tol=LINE_MATCH_TOLERANCE
        ):
            return _abstain_leg(
                leg,
                f"benchmark line {float(leg.benchmark_line)} does not match the "
                f"operator's {float(line)}. These are different contracts and "
                "the de-vigged probability of one is not the probability of the "
                "other; falling back to the model here would hide a join fault.",
                line,
            )
        try:
            probability = benchmark_fair_probability(
                int(over), int(under), side=side, method=leg.devig_method,
            )
        except (DfsPayoutError, ValueError, TypeError) as exc:
            return _abstain_leg(leg, f"benchmark could not be de-vigged: {exc}", line)
        return LegResolution(
            leg_id=leg.leg_id,
            status=LEG_READY,
            probability=probability,
            probability_source=ProbabilitySource.SHARP_BENCHMARK,
            side=side,
            line=float(line),
            benchmark_source=leg.benchmark_source,
            devig_method=leg.devig_method,
        )

    if over is not None or under is not None:
        return _abstain_leg(
            leg,
            "only one side of the benchmark was supplied. A single price cannot "
            "be de-vigged and its raw implied probability still carries the "
            "book's hold, which would flatter this entry by however much that "
            "hold is.",
            line,
        )

    model_p = leg.model_probability
    if model_p is None:
        return _abstain_leg(
            leg,
            "no probability source: neither a two-way benchmark nor a model "
            "probability was supplied for this leg",
            line,
        )
    model_p = float(model_p)
    if not (math.isfinite(model_p) and 0.0 < model_p < 1.0):
        return _abstain_leg(
            leg,
            f"model probability {leg.model_probability!r} is not strictly inside "
            "(0, 1)",
            line,
        )
    return LegResolution(
        leg_id=leg.leg_id,
        status=LEG_READY,
        probability=model_p,
        probability_source=ProbabilitySource.MODEL,
        side=side,
        line=float(line),
    )


def combine_probability_sources(
    sources: Sequence[ProbabilitySource],
) -> ProbabilitySource:
    """
    One source for the entry: uniform when every leg agrees, MIXED otherwise.

    MIXED is not a middle ground. An entry is only as market-grounded as its
    weakest leg, so a slip with one model leg among three benchmark ones is not
    "mostly market-grounded" — it carries the model's calibration risk on the
    whole card, because every leg has to land.
    """
    distinct = {ProbabilitySource(s) for s in sources}
    if not distinct:
        return ProbabilitySource.UNSPECIFIED
    if len(distinct) == 1:
        return distinct.pop()
    if distinct <= {
        ProbabilitySource.SHARP_BENCHMARK,
        ProbabilitySource.MODEL,
        ProbabilitySource.MIXED,
    }:
        return ProbabilitySource.MIXED
    return ProbabilitySource.UNSPECIFIED


def route_pickem_entry(
    structure: DfsPayoutStructure,
    legs: Sequence[PickemLeg],
    *,
    correlation: Any | None = None,
    n_sims: int = 200_000,
    seed: int = 20240115,
) -> EntryEvaluation:
    """
    Resolve every leg, then price the slip against the operator's payout matrix.

    ALL-OR-NOTHING ON THE LEGS, deliberately: one unresolved leg abstains the
    whole entry. A slip is a joint product and every leg has to land, so
    dropping the leg that could not be priced and evaluating the rest would
    price a DIFFERENT, shorter slip — and a shorter slip against the same payout
    table reads as a better one.

    ``correlation`` is passed through to ``evaluate_pickem_entry``: absent means
    the exact Poisson-binomial for independent legs, present means the Gaussian
    copula. Teammate and same-game legs are correlated and the sign of the error
    from ignoring that depends on the sign of the correlation, so supply the
    matrix when the legs are related.
    """
    resolutions = [resolve_leg_probability(leg) for leg in legs]
    out = EntryEvaluation(
        status=PAYOUT_EV_ABSTAIN,
        structure_label=structure.label or f"{structure.n_picks}-pick",
        legs=resolutions,
    )

    if not resolutions:
        out.reason = "no legs supplied"
        return out

    if len(resolutions) != structure.n_picks:
        out.reason = (
            f"{len(resolutions)} leg(s) against a {structure.n_picks}-pick "
            "structure; the slip and the payout table describe different products"
        )
        return out

    unresolved = [r for r in resolutions if r.status != LEG_READY]
    if unresolved:
        out.reason = (
            f"{len(unresolved)} of {len(resolutions)} leg(s) could not be "
            "resolved, so the entry is not priced: "
            + "; ".join(f"{r.leg_id}: {r.reason}" for r in unresolved)
        )
        return out

    source = combine_probability_sources([r.probability_source for r in resolutions])
    payout = evaluate_pickem_entry(
        structure,
        [float(r.probability) for r in resolutions],  # type: ignore[arg-type]
        source=source,
        correlation=correlation,
        n_sims=n_sims,
        seed=seed,
    )
    out.payout = payout
    out.status = payout.status
    out.reason = payout.reason
    return out
