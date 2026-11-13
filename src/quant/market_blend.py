"""Market-anchored probability blending (external audit, 2026-11-13).

WHERE THIS CAME FROM, AND WHY IT IS WANTED HERE.

Audit of `omedakbari/nba-edge-model`, `src/models/market_blend.py`. Its own
docstring states the problem it was built for: "The props model alone is ~5x
overconfident against real sportsbook lines, but its disagreement with the
market does carry information." That is a description of THIS project's
measured state, from `prop_result_calibration_report` on 533 graded rows:

    mean stated probability  0.7077
    observed win rate        0.4765   (254-279, 95% CI [43.4%, 51.9%])
    ECE                      0.2311   at bin coverage 1.0 (5/5 bins)
    Brier, served            0.3011
    Brier, constant 0.5      0.2500

A served Brier WORSE than a constant 0.5 means the stated numbers cost more
than saying nothing. Isotonic/sigmoid recalibration (`ledger_calibration`)
is the other half of the answer and cannot be the whole of it: a monotone map
cannot change the ORDERING of rows, so if the model's ranking carries little
signal, recalibration only compresses every row toward the base rate. This
module attacks the other axis -- WHICH quantity gets calibrated. It anchors on
the market's no-vig probability and gives the model's DISAGREEMENT with the
market a small, fitted coefficient:

    m = logit(p_market)
    d = logit(p_model) - m
    z = a + b * m + c * d
    p = expit(z)

WHAT THE PARAMETERS MEAN, which is the reason this form was chosen over an
arithmetic average:

  * c = 0, a = 0, b = 1  ->  exactly the market. The model is ignored.
  * c = 1, a = 0, b = 1  ->  exactly the model. The market is ignored.
  * 0 < c < 1            ->  the market, nudged by the model's disagreement.

So the blend NESTS both endpoints, and `c` is a measurement rather than a
setting: it is the answer to "how much is our disagreement with the market
worth?" If the fit returns c ~ 0, the honest reading is that our model adds
nothing to the closing price, and that is a FINDING to report, not a failure
to hide.

THEIR COEFFICIENTS ARE NOT COPIED. The audited repo reports c ~ 0.15-0.23 for
its own model against its own book sample. Those numbers describe their model,
not ours; importing them would be fabrication with a citation attached. There
are no default coefficients in this module. `fit_market_blend` fits them on
OUR graded rows or refuses.

WHAT IS NOT HERE, DELIBERATELY. Nothing in this module is wired into serving.
Per AGENTS.md, a new component reaches the model matrix only by winning a
holdout in `scripts/feature_ab.py`. This lands as pure, tested math plus a
gated fitter so that the measurement can be run; `blend_probability` is called
by no production path.

THE DATA PREREQUISITE, stated because it bounds the fit today. A no-vig market
probability needs BOTH sides of a price. `prop_results` stores a single `odds`
column (the taken side), so the graded ledger alone cannot produce one.
`projections.odds_over_american` / `.odds_under_american` arrived in migration
010, so the pair is recoverable for rows written from 010 onward and for no row
written before it. The fitter is gated on sample size and distinct slates and
reports INSUFFICIENT_DATA until post-010 history accumulates.

RESEARCH_ONLY. This produces a probability. It does not size, price or place
a wager.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

logger = logging.getLogger(__name__)

RESEARCH_STATUS = "RESEARCH_ONLY"

STATUS_OK = "OK"
STATUS_INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
STATUS_DEGENERATE = "DEGENERATE"

#: Probabilities are clipped to this interval before `logit`. A 0 or a 1 maps
#: to -inf/+inf and poisons every downstream arithmetic; the clip is the only
#: reason this module cannot emit a nan from a valid-looking input.
PROBABILITY_CLIP = (1e-6, 1.0 - 1e-6)

#: Below this many graded rows the fit is not reported as usable. Three free
#: parameters on a few dozen rows fits noise, and noise here flatters the
#: ticket -- the same reasoning, and deliberately the same posture, as
#: `leg_correlation.MIN_PAIRS_PER_BUCKET`.
MIN_ROWS_FOR_FIT = 400

#: And below THIS many distinct slates, independently. Rows from one night
#: share that night's pace, officiating and injury news, so 400 rows drawn
#: from three slates are not 400 independent observations. The independent
#: unit is the SLATE, exactly as `leg_correlation` treats the GAME.
MIN_SLATES_FOR_FIT = 30


class MarketBlendError(RuntimeError):
    """Raised when a blend is asked for on inputs that cannot support one."""


def logit(p: float) -> float:
    """Log-odds, with the clip that keeps 0 and 1 from becoming infinities."""
    value = float(p)
    if not math.isfinite(value):
        raise MarketBlendError(f"probability {p!r} is not finite")
    if not 0.0 <= value <= 1.0:
        raise MarketBlendError(f"probability {p!r} is outside [0, 1]")
    low, high = PROBABILITY_CLIP
    value = min(max(value, low), high)
    return math.log(value / (1.0 - value))


def expit(z: float) -> float:
    """Inverse of `logit`, written to not overflow on a large negative z."""
    value = float(z)
    if not math.isfinite(value):
        raise MarketBlendError(f"log-odds {z!r} is not finite")
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    scaled = math.exp(value)
    return scaled / (1.0 + scaled)


@dataclass(frozen=True)
class MarketBlendCoefficients:
    """One fitted (a, b, c) triple, with the provenance needed to judge it.

    `market` is the prop market the triple was fitted on, or "ALL" for a
    pooled fit. A triple carries its own sample size because a reader deciding
    whether to trust it needs that in the same object -- the alternative is a
    coefficient in one place and its evidence in another, which is how a
    figure fitted on 40 rows comes to be quoted as though it were fitted on
    4,000.
    """

    a: float
    b: float
    c: float
    market: str
    n_rows: int
    n_slates: int
    status: str
    fitted_through: str | None = None
    reason: str | None = None
    research_status: str = RESEARCH_STATUS

    @property
    def usable(self) -> bool:
        return self.status == STATUS_OK

    def as_dict(self) -> dict[str, Any]:
        return {
            "a": self.a, "b": self.b, "c": self.c,
            "market": self.market,
            "n_rows": self.n_rows, "n_slates": self.n_slates,
            "status": self.status, "fitted_through": self.fitted_through,
            "reason": self.reason, "research_status": self.research_status,
        }


def blend_probability(
    p_market: float,
    p_model: float,
    coefficients: MarketBlendCoefficients,
) -> float:
    """
    Blend a market probability with a model probability, in logit space.

    BOTH ARGUMENTS MUST DESCRIBE THE SAME SIDE. This is the one way to use
    this function wrongly and get a plausible number out: pass P(over) from
    the market and P(taken side) from the model and the blend reads the
    model's disagreement with the market as enormous on every under leg,
    because `d` then contains a side flip rather than a disagreement. There is
    no way to detect that from the two floats, so it cannot be guarded here --
    it is the caller's contract, and `tests/test_market_blend.py` states what
    happens when it is broken.

    Refuses on an unusable coefficient set rather than falling back to the
    market or the model: a silent fallback would make a blended board and an
    unblended board indistinguishable in the output, which is precisely the
    confusion an A/B measurement exists to avoid.
    """
    if not coefficients.usable:
        raise MarketBlendError(
            f"coefficients for {coefficients.market} are {coefficients.status}"
            + (f": {coefficients.reason}" if coefficients.reason else "")
        )
    m = logit(p_market)
    d = logit(p_model) - m
    z = coefficients.a + coefficients.b * m + coefficients.c * d
    return expit(z)


def _design_matrix(
    market_probs: Sequence[float], model_probs: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Columns [m, d] where m = logit(p_market), d = logit(p_model) - m."""
    m = np.array([logit(p) for p in market_probs], dtype=float)
    model = np.array([logit(p) for p in model_probs], dtype=float)
    return np.column_stack([m, model - m]), m


def fit_market_blend(
    rows: Sequence[Mapping[str, Any]],
    *,
    market: str = "ALL",
    min_rows: int = MIN_ROWS_FOR_FIT,
    min_slates: int = MIN_SLATES_FOR_FIT,
) -> MarketBlendCoefficients:
    """
    Fit (a, b, c) on graded rows by logistic regression.

    Each row needs `p_market`, `p_model`, `outcome` (1.0 win / 0.0 loss) and
    `game_date`, all describing the SAME side. Rows missing any of the four
    are counted out rather than imputed.

    PURE, and taking rows rather than reading the database, for the reason
    `settlement.calibration.taken_side_pairs` gives: one reader of the ledger
    contract. Recovering `p_market` needs both prices, which live on
    `projections` (migration 010) and not on `prop_results`, so the join is
    the caller's to make and this function's job is only the arithmetic.

    Gated on rows AND on distinct slates. Returns a coefficient set whose
    `status` says which gate it failed; it never returns a usable triple it
    could not support.
    """
    market_probs: list[float] = []
    model_probs: list[float] = []
    outcomes: list[float] = []
    dates: list[Any] = []

    for row in rows:
        p_market, p_model = row.get("p_market"), row.get("p_model")
        outcome, date = row.get("outcome"), row.get("game_date")
        if p_market is None or p_model is None or outcome is None:
            continue
        try:
            pm, pmod, y = float(p_market), float(p_model), float(outcome)
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(pm) and math.isfinite(pmod) and math.isfinite(y)):
            continue
        if not (0.0 <= pm <= 1.0 and 0.0 <= pmod <= 1.0) or y not in (0.0, 1.0):
            continue
        market_probs.append(pm)
        model_probs.append(pmod)
        outcomes.append(y)
        dates.append(date)

    n_rows = len(outcomes)
    n_slates = len({str(d) for d in dates if d is not None})

    def refuse(status: str, reason: str) -> MarketBlendCoefficients:
        return MarketBlendCoefficients(
            a=float("nan"), b=float("nan"), c=float("nan"), market=market,
            n_rows=n_rows, n_slates=n_slates, status=status, reason=reason,
        )

    if n_rows < min_rows:
        return refuse(
            STATUS_INSUFFICIENT_DATA,
            f"{n_rows} usable rows, {min_rows} required; three free "
            "parameters on fewer than that fits noise",
        )
    if n_slates < min_slates:
        return refuse(
            STATUS_INSUFFICIENT_DATA,
            f"{n_rows} rows but only {n_slates} distinct slates, "
            f"{min_slates} required; rows from one night share that night's "
            "pace, officiating and injury news",
        )
    y = np.array(outcomes, dtype=float)
    if len(np.unique(y)) < 2:
        return refuse(
            STATUS_DEGENERATE,
            "every graded row has the same outcome, so no coefficient is "
            "identified",
        )

    design, _ = _design_matrix(market_probs, model_probs)
    if not np.all(np.isfinite(design)):
        return refuse(STATUS_DEGENERATE, "a logit of a supplied probability is not finite")
    # A column of zero variance identifies no coefficient. `d` is exactly zero
    # for every row when the model simply republishes the market, which is a
    # real state worth naming rather than a numerical accident.
    spreads = design.std(axis=0)
    if spreads[1] <= 1e-12:
        return refuse(
            STATUS_DEGENERATE,
            "the model never disagrees with the market on these rows, so c "
            "is not identified",
        )
    if spreads[0] <= 1e-12:
        return refuse(
            STATUS_DEGENERATE,
            "every market probability is identical on these rows, so b is "
            "not identified",
        )

    from sklearn.linear_model import LogisticRegression

    # C is large rather than absent: an unpenalised logistic fit diverges on a
    # separable sample, and a diverged coefficient reads as enormous
    # confidence. The penalty is weak enough not to shrink a real coefficient
    # at these sample sizes and strong enough to keep a separable one finite.
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000)
    model.fit(design, y)
    b, c = (float(v) for v in model.coef_[0])
    a = float(model.intercept_[0])
    if not all(math.isfinite(v) for v in (a, b, c)):
        return refuse(STATUS_DEGENERATE, "the fit did not converge to finite coefficients")

    dated = [str(d) for d in dates if d is not None]
    return MarketBlendCoefficients(
        a=a, b=b, c=c, market=market, n_rows=n_rows, n_slates=n_slates,
        status=STATUS_OK, fitted_through=max(dated) if dated else None,
    )
