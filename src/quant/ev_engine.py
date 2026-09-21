"""
src/quant/ev_engine.py — expected value and closing-line value.

Written for this repository: waves 3-5 import it and no pack ever
contained it. Its interface is taken from the call sites in
``historical_store`` and ``paper_research`` so those modules drop in
unchanged, and its arithmetic is built on ``odds_math`` and
``market_ev_gate`` rather than a second de-vig.

RESEARCH_ONLY. This computes a number; it never recommends a wager,
never sizes a stake and never ranks a slate. A positive EV here is a
statement about the model's probability being right, which is exactly the
thing that has not been established yet.

TWO NUMBERS THAT LOOK ALIKE AND ARE NOT:

- EV uses YOUR model's probability. It is only as good as that model,
  and on an uncalibrated model it is decoration.
- CLV compares the price you took against the closing price. It needs no
  model at all, which is what makes it the more trustworthy of the two
  and the reason it is reported separately rather than summed in.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Literal

from src.quant.contracts import (
    GATE_ABSTAIN,
    GATE_READY,
    MarketContext,
    PropMarketSnapshot,
    market_ev_gate,
)
from src.quant.odds_math import (
    american_to_implied_probability,
    american_to_profit_multiple,
    expected_value_per_unit,
    multiplicative_devig,
)

logger = logging.getLogger(__name__)

MarketType = Literal["player_prop", "moneyline", "spread", "total", "unknown"]

RESEARCH_DISCLAIMER = (
    "RESEARCH_ONLY — expected value is a model output, not a bet "
    "recommendation, a stake size, or a profitability claim."
)


@dataclass(frozen=True)
class EvSide:
    """EV for one side of a two-way market."""

    side: str
    american: int
    model_prob: float
    fair_prob: float
    implied_prob: float
    ev: float
    edge: float
    profit_multiple: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "side": self.side,
            "american": self.american,
            "model_prob": round(self.model_prob, 6),
            "fair_prob": round(self.fair_prob, 6),
            "implied_prob": round(self.implied_prob, 6),
            "ev": round(self.ev, 6),
            "edge": round(self.edge, 6),
            "profit_multiple": round(self.profit_multiple, 6),
        }


@dataclass
class EvEvaluation:
    """Both sides priced, plus which one (if either) cleared the threshold."""

    game_id: str
    status: str = GATE_ABSTAIN
    reason: str | None = None
    sides: list[EvSide] = field(default_factory=list)
    selected_side: str | None = None
    selected_ev: float | None = None
    market_type: MarketType = "unknown"
    market_id: str | None = None
    line: float | None = None
    hold: float | None = None
    ev_threshold: float = 0.0
    disclaimer: str = RESEARCH_DISCLAIMER

    def as_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "status": self.status,
            "reason": self.reason,
            "sides": [s.as_dict() for s in self.sides],
            "selected_side": self.selected_side,
            "selected_ev": (
                round(self.selected_ev, 6) if self.selected_ev is not None else None
            ),
            "market_type": self.market_type,
            "market_id": self.market_id,
            "line": self.line,
            "hold": round(self.hold, 6) if self.hold is not None else None,
            "ev_threshold": self.ev_threshold,
            "disclaimer": self.disclaimer,
        }


@dataclass
class ClvResult:
    """Closing-line value for a taken price."""

    status: str = GATE_ABSTAIN
    reason: str | None = None
    clv: float | None = None
    taken_american: int | None = None
    closing_american: int | None = None
    taken_fair_prob: float | None = None
    closing_fair_prob: float | None = None
    beat_close: bool | None = None
    note: str = (
        "CLV is a market-quality signal, not profit, and is never added to "
        "ROI. Positive CLV does not guarantee future profitability."
    )

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "clv": round(self.clv, 6) if self.clv is not None else None,
            "taken_american": self.taken_american,
            "closing_american": self.closing_american,
            "taken_fair_prob": (
                round(self.taken_fair_prob, 6) if self.taken_fair_prob is not None else None
            ),
            "closing_fair_prob": (
                round(self.closing_fair_prob, 6)
                if self.closing_fair_prob is not None else None
            ),
            "beat_close": self.beat_close,
            "note": self.note,
        }


class EvEngine:
    """
    Prices both sides of a two-way market against a model probability.

    ``ev_threshold`` is the EV per unit a side must clear before it is
    selected. It is a reporting filter, not a betting rule: selecting a
    side here means "this is the side with an edge under this model", and
    nothing about whether to act on it.
    """

    def __init__(self, ev_threshold: float = 0.0) -> None:
        self.ev_threshold = float(ev_threshold)

    def evaluate_two_way(
        self,
        *,
        game_id: str,
        american_a: int,
        american_b: int,
        model_prob_a: float,
        model_prob_b: float | None = None,
        label_a: str = "over",
        label_b: str = "under",
        line: float | None = None,
        market_type: MarketType = "unknown",
        market_id: str | None = None,
    ) -> EvEvaluation:
        """
        EV for both sides. Abstains rather than returning a misleading number.

        The model probabilities are NOT renormalised to sum to one. If they
        do not, that is a fact about the model worth surfacing, and
        silently rescaling would hide a miscalibration behind a tidy
        output.
        """
        out = EvEvaluation(
            game_id=game_id, market_type=market_type, market_id=market_id,
            line=line, ev_threshold=self.ev_threshold,
        )

        if model_prob_b is None:
            # Whole-number lines can carry push mass. Folding that mass into
            # under via 1-P(over) overstates under EV / mis-calibrates.
            whole_line = False
            if line is not None:
                try:
                    lf = float(line)
                    whole_line = math.isfinite(lf) and lf == float(int(lf))
                except (TypeError, ValueError):
                    whole_line = False
            if whole_line:
                out.reason = (
                    "model_prob_b required for whole-number lines (push mass); "
                    "refusing silent 1-P(over) complement"
                )
                return out
            model_prob_b = 1.0 - float(model_prob_a)

        for name, value in (("model_prob_a", model_prob_a), ("model_prob_b", model_prob_b)):
            if value is None or not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                out.reason = f"{name} is not a probability in [0, 1]: {value!r}"
                return out

        try:
            aa, ab = int(american_a), int(american_b)
            if aa == 0 or ab == 0:
                raise ValueError("American odds of 0 are not a price")
            fair = multiplicative_devig(aa, ab)
        except (TypeError, ValueError) as exc:
            out.reason = f"Cannot de-vig this market: {exc}"
            return out

        total = float(model_prob_a) + float(model_prob_b)
        if abs(total - 1.0) > 0.02:
            logger.warning(
                "Model probabilities for %s sum to %.4f, not 1. Reported as "
                "given rather than rescaled — the gap is a model property.",
                game_id, total,
            )

        pairs = (
            (label_a, int(american_a), float(model_prob_a), fair.fair_prob_a, fair.implied_prob_a),
            (label_b, int(american_b), float(model_prob_b), fair.fair_prob_b, fair.implied_prob_b),
        )
        for side, american, model_p, fair_p, implied_p in pairs:
            out.sides.append(EvSide(
                side=side,
                american=american,
                model_prob=model_p,
                fair_prob=fair_p,
                implied_prob=implied_p,
                ev=expected_value_per_unit(model_p, american),
                # Edge against the DE-VIGGED price, not the posted one.
                # Measuring against the posted price counts the book's hold
                # as edge and makes every market look beatable.
                edge=model_p - fair_p,
                profit_multiple=american_to_profit_multiple(american),
            ))

        out.hold = fair.hold
        out.status = "OK"

        best = max(out.sides, key=lambda s: s.ev)
        if best.ev > self.ev_threshold:
            out.selected_side = best.side
            out.selected_ev = best.ev
        else:
            out.reason = (
                f"No side clears the {self.ev_threshold:+.4f} EV threshold "
                f"(best {best.side} at {best.ev:+.4f})"
            )
        return out

    def evaluate_snapshot(
        self,
        market: MarketContext | PropMarketSnapshot,
        model_prob_over: float,
        *,
        market_type: MarketType = "player_prop",
        model_prob_under: float | None = None,
    ) -> EvEvaluation:
        """
        Price a posted market, but only after the gate allows it.

        The gate is the single place that decides whether EV may be
        computed at all — pick'em boards, missing odds, absent lines. Going
        around it here would reintroduce exactly the silent-EV path it
        exists to prevent.

        Pass ``model_prob_under`` for whole-number lines so push mass is
        not silently assigned to under.
        """
        context = (
            market.to_market_context()
            if isinstance(market, PropMarketSnapshot) else market
        )
        verdict = market_ev_gate(context)
        if verdict["status"] != GATE_READY:
            return EvEvaluation(
                game_id=context.game_id,
                status=GATE_ABSTAIN,
                reason=verdict.get("reason"),
                market_type=market_type,
                line=context.line,
                ev_threshold=self.ev_threshold,
            )

        return self.evaluate_two_way(
            game_id=context.game_id,
            american_a=int(context.over_odds_american),
            american_b=int(context.under_odds_american),
            model_prob_a=float(model_prob_over),
            model_prob_b=model_prob_under,
            label_a="over", label_b="under",
            line=context.line,
            market_type=market_type,
            market_id=context.market,
        )


def compute_clv(
    taken_american: int,
    closing_american: int,
    *,
    taken_other_american: int | None = None,
    closing_other_american: int | None = None,
) -> ClvResult:
    """
    Closing-line value as a probability difference.

    Measured on DE-VIGGED probabilities when both sides of each market are
    supplied, and on raw implied probabilities otherwise — the second is
    noisier because it carries each book's hold, so which was used is
    recorded rather than left to be inferred.

    Positive CLV means the price moved toward you after you took it.
    """
    result = ClvResult(taken_american=taken_american, closing_american=closing_american)

    if taken_american is None or closing_american is None:
        result.reason = "Both a taken and a closing price are required"
        return result

    try:
        if taken_other_american is not None and closing_other_american is not None:
            taken_p = multiplicative_devig(taken_american, taken_other_american).fair_prob_a
            closing_p = multiplicative_devig(closing_american, closing_other_american).fair_prob_a
            method = "devigged"
        else:
            taken_p = american_to_implied_probability(int(taken_american))
            closing_p = american_to_implied_probability(int(closing_american))
            method = "raw_implied"
    except (TypeError, ValueError) as exc:
        result.reason = f"Unusable price: {exc}"
        return result

    result.status = "OK"
    result.taken_fair_prob = taken_p
    result.closing_fair_prob = closing_p
    # You beat the close when the market ended up MORE confident in your
    # side than the price you paid implied.
    result.clv = closing_p - taken_p
    result.beat_close = result.clv > 0
    result.note = f"{result.note} Computed on {method} probabilities."
    return result
