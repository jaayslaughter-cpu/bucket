"""Count dispersion estimated from training residuals.

Replaces a hardcoded variance multiplier with a parameter fitted on data,
and picks the distribution family by out-of-sample score rather than
assumption.

WHY NOT JUST USE A NORMAL: points, rebounds and assists are non-negative
integers with a floor at zero and a right tail. A Normal puts mass below
zero and understates the tail, which matters most at exactly the lines
people bet.

WHY NOT JUST USE POISSON: Poisson forces variance == mean. Real NBA
counting stats are overdispersed — minutes vary, role varies, blowouts
happen — so Poisson is systematically overconfident. Negative Binomial
adds one parameter (phi, the variance-to-mean ratio) to absorb that.

The choice is made by comparing mean log-loss on a chronologically held
out slice of the TRAINING data only. Nothing here ever sees validation or
test rows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import nbinom, norm, poisson

logger = logging.getLogger(__name__)

Family = Literal["poisson", "negbin", "zip", "normal"]

# phi is the variance-to-mean ratio. 1.0 is Poisson; below it is
# underdispersion, which for these stats indicates a fitting problem
# rather than a real effect.
MIN_PHI = 1.0
MAX_PHI = 10.0
MIN_ROWS_TO_FIT = 50

# Zero-inflation for the ZIP family: the share of structural zeros on top
# of the Poisson ones. Rebounds and assists have real excess zeros (a
# bench player who plays four minutes), which a plain Poisson cannot
# represent at any mean. Capped below 1 because a fitted value near it
# means the mean is being explained entirely by zeros, which is a fitting
# failure rather than a finding.
MIN_ZERO_INFLATION = 0.0
MAX_ZERO_INFLATION = 0.85

# Normal scale, as a multiple of sqrt(mean). 1.0 reproduces Poisson
# spread; the bounds allow genuine over- and under-dispersion without
# letting the optimiser run away on a small sample.
MIN_SIGMA_SCALE = 0.5
MAX_SIGMA_SCALE = 5.0


@dataclass(frozen=True)
class CountDispersion:
    """A fitted dispersion model. `phi` is meaningless when family is poisson."""

    family: Family
    phi: float
    n_train_rows: int
    selection_scores: dict[str, float]
    fallback_reason: str | None = None
    # Fitted, not assumed. Defaulted so artifacts saved before these
    # families existed still load: an old row reloads as the two-family
    # model it actually was.
    zero_inflation: float = 0.0
    sigma_scale: float = 1.0

    def variance_for(self, mean: float) -> float:
        """Predicted variance at a given mean."""
        mu = max(float(mean), 1e-6)
        if self.family == "poisson":
            return mu
        if self.family == "negbin":
            return mu * self.phi
        if self.family == "zip":
            # E[X] = (1-pi)*lam = mu, so Var = mu * (1 + pi*mu/(1-pi)).
            pi = float(np.clip(self.zero_inflation, 0.0, MAX_ZERO_INFLATION))
            return mu * (1.0 + pi * mu / max(1.0 - pi, 1e-6))
        if self.family == "normal":
            return (self.sigma_scale ** 2) * mu
        return mu

    def zip_lambda(self, mean: float) -> float:
        """Poisson rate inside the ZIP mixture that yields this mean."""
        pi = float(np.clip(self.zero_inflation, 0.0, MAX_ZERO_INFLATION))
        return max(float(mean), 1e-6) / max(1.0 - pi, 1e-6)

    def normal_sd(self, mean: float) -> float:
        """Standard deviation under the normal family."""
        return max(self.sigma_scale * np.sqrt(max(float(mean), 1e-6)), 1e-6)

    def nb_params(self, mean: float) -> tuple[float, float]:
        """(r, p) for scipy's negative binomial at this mean."""
        mu = max(float(mean), 1e-6)
        variance = max(self.variance_for(mu), mu + 1e-6)
        p = mu / variance
        r = (mu * mu) / max(variance - mu, 1e-6)
        return r, p

    def to_dict(self) -> dict[str, object]:
        """Round-trippable form. Unlike ``as_metadata`` this keeps full
        precision — a rounded phi reloads as a different distribution."""
        return {
            "family": self.family,
            "phi": self.phi,
            "n_train_rows": self.n_train_rows,
            "selection_scores": self.selection_scores,
            "fallback_reason": self.fallback_reason,
            "zero_inflation": self.zero_inflation,
            "sigma_scale": self.sigma_scale,
        }

    @classmethod
    def from_dict(cls, payload: dict | None) -> "CountDispersion | None":
        """Rebuild from ``to_dict``. None in, None out — an absent dispersion
        must stay absent rather than defaulting to Poisson, which would look
        like a fitted result."""
        if not payload:
            return None
        return cls(
            family=payload["family"],
            phi=float(payload["phi"]),
            n_train_rows=int(payload["n_train_rows"]),
            selection_scores=dict(payload.get("selection_scores") or {}),
            fallback_reason=payload.get("fallback_reason"),
            # Absent on artifacts written before these families existed.
            zero_inflation=float(payload.get("zero_inflation") or 0.0),
            sigma_scale=float(payload.get("sigma_scale") or 1.0),
        )

    def as_metadata(self) -> dict[str, object]:
        return {
            "dispersion_family": self.family,
            "dispersion_phi": round(self.phi, 4),
            "dispersion_train_rows": self.n_train_rows,
            "dispersion_selection": {k: round(v, 5) for k, v in self.selection_scores.items()},
            "dispersion_fallback_reason": self.fallback_reason,
            "dispersion_zero_inflation": round(self.zero_inflation, 4),
            "dispersion_sigma_scale": round(self.sigma_scale, 4),
        }


def _clean(y_true: np.ndarray, mu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=float)
    m = np.asarray(mu, dtype=float)
    ok = np.isfinite(y) & np.isfinite(m) & (y >= 0) & (m > 0)
    return y[ok], m[ok]


def _neg_log_likelihood(phi: float, y: np.ndarray, mu: np.ndarray) -> float:
    """Negative log-likelihood of the NB fit at a given variance ratio."""
    if phi <= MIN_PHI:
        return float(np.sum(-poisson.logpmf(np.round(y), mu)))
    variance = np.maximum(mu * phi, mu + 1e-6)
    p = mu / variance
    r = (mu * mu) / np.maximum(variance - mu, 1e-6)
    return float(np.sum(-nbinom.logpmf(np.round(y), r, p)))


def _zip_neg_log_likelihood(pi: float, y: np.ndarray, mu: np.ndarray) -> float:
    """Zero-inflated Poisson NLL.

    X is a structural zero with probability pi, otherwise Poisson(lam).
    Holding E[X] = mu fixed gives lam = mu / (1 - pi), so pi buys extra
    zeros WITHOUT moving the mean the regressor already predicted — which
    is what makes it comparable to the other families on the same means.
    """
    pi = float(np.clip(pi, MIN_ZERO_INFLATION, MAX_ZERO_INFLATION))
    lam = mu / max(1.0 - pi, 1e-6)
    k = np.round(y)
    log_p_zero = np.log(np.maximum(pi + (1.0 - pi) * np.exp(-lam), 1e-300))
    log_p_pos = np.log(max(1.0 - pi, 1e-300)) + poisson.logpmf(k, lam)
    return float(np.sum(-np.where(k <= 0, log_p_zero, log_p_pos)))


def _normal_neg_log_likelihood(scale: float, y: np.ndarray, mu: np.ndarray) -> float:
    """Normal NLL scored on the SAME measure as the discrete families.

    A continuous density and a discrete pmf are not comparable — whichever
    is used sets the units, and the family selection would then be decided
    by that choice rather than by fit. So the normal is scored on the
    continuity-corrected probability of each integer,
    P(k - 0.5 < X < k + 0.5), which is a probability like the others.
    """
    scale = float(np.clip(scale, MIN_SIGMA_SCALE, MAX_SIGMA_SCALE))
    sd = np.maximum(scale * np.sqrt(mu), 1e-6)
    k = np.round(y)
    upper = norm.cdf((k + 0.5 - mu) / sd)
    lower = norm.cdf((k - 0.5 - mu) / sd)
    return float(np.sum(-np.log(np.maximum(upper - lower, 1e-300))))


def _mean_nll(
    family: Family,
    y: np.ndarray,
    mu: np.ndarray,
    *,
    phi: float = MIN_PHI,
    zero_inflation: float = 0.0,
    sigma_scale: float = 1.0,
) -> float:
    """Per-row NLL, so families are compared on a per-observation basis."""
    if len(y) == 0:
        return float("inf")
    if family == "zip":
        total = _zip_neg_log_likelihood(zero_inflation, y, mu)
    elif family == "normal":
        total = _normal_neg_log_likelihood(sigma_scale, y, mu)
    else:
        total = _neg_log_likelihood(phi if family == "negbin" else MIN_PHI, y, mu)
    return total / len(y)


def fit_count_dispersion(
    y_true: np.ndarray,
    y_pred_mean: np.ndarray,
    *,
    market: str = "",
    holdout_fraction: float = 0.3,
) -> CountDispersion:
    """
    Fit dispersion on training residuals and pick a family by held-out score.

    ``y_true`` and ``y_pred_mean`` must both come from TRAINING rows, in
    chronological order — the holdout slice is taken from the end.

    Falls back to Poisson with a named reason when there is too little
    data to fit anything, rather than asserting a dispersion nobody
    measured.
    """
    y, mu = _clean(y_true, y_pred_mean)
    n = len(y)

    if n < MIN_ROWS_TO_FIT:
        return CountDispersion(
            family="poisson",
            phi=MIN_PHI,
            n_train_rows=n,
            selection_scores={},
            fallback_reason=(
                f"only {n} usable training rows (need {MIN_ROWS_TO_FIT}); "
                "defaulted to Poisson rather than fitting a dispersion"
            ),
        )

    cut = max(MIN_ROWS_TO_FIT // 2, int(n * (1 - holdout_fraction)))
    cut = min(cut, n - 10)
    y_fit, mu_fit = y[:cut], mu[:cut]
    y_score, mu_score = y[cut:], mu[cut:]

    # Every parameter is fitted on the EARLIER slice only; the family is
    # then chosen on the later slice. Fitting and choosing on the same rows
    # would pick whichever family has the most parameters.
    phi_hat = float(np.clip(
        minimize_scalar(
            _neg_log_likelihood, bounds=(MIN_PHI, MAX_PHI),
            args=(y_fit, mu_fit), method="bounded",
        ).x,
        MIN_PHI, MAX_PHI,
    ))
    pi_hat = float(np.clip(
        minimize_scalar(
            _zip_neg_log_likelihood,
            bounds=(MIN_ZERO_INFLATION, MAX_ZERO_INFLATION),
            args=(y_fit, mu_fit), method="bounded",
        ).x,
        MIN_ZERO_INFLATION, MAX_ZERO_INFLATION,
    ))
    sigma_hat = float(np.clip(
        minimize_scalar(
            _normal_neg_log_likelihood,
            bounds=(MIN_SIGMA_SCALE, MAX_SIGMA_SCALE),
            args=(y_fit, mu_fit), method="bounded",
        ).x,
        MIN_SIGMA_SCALE, MAX_SIGMA_SCALE,
    ))

    scores = {
        "poisson_mean_nll": _mean_nll("poisson", y_score, mu_score),
        "negbin_mean_nll": _mean_nll("negbin", y_score, mu_score, phi=phi_hat),
        "zip_mean_nll": _mean_nll("zip", y_score, mu_score, zero_inflation=pi_hat),
        "normal_mean_nll": _mean_nll("normal", y_score, mu_score, sigma_scale=sigma_hat),
    }
    ranked: list[tuple[str, Family]] = [
        ("poisson_mean_nll", "poisson"),
        ("negbin_mean_nll", "negbin"),
        ("zip_mean_nll", "zip"),
        ("normal_mean_nll", "normal"),
    ]
    family: Family = min(ranked, key=lambda pair: scores[pair[0]])[1]

    scores["fitted_phi"] = phi_hat
    scores["fitted_zero_inflation"] = pi_hat
    scores["fitted_sigma_scale"] = sigma_hat

    logger.info(
        "dispersion %s: family=%s (poisson=%.4f negbin=%.4f[phi=%.3f] "
        "zip=%.4f[pi=%.3f] normal=%.4f[sigma=%.3f], n=%d)",
        market or "?", family,
        scores["poisson_mean_nll"], scores["negbin_mean_nll"], phi_hat,
        scores["zip_mean_nll"], pi_hat, scores["normal_mean_nll"], sigma_hat, n,
    )
    return CountDispersion(
        family=family,
        phi=phi_hat if family == "negbin" else MIN_PHI,
        n_train_rows=n,
        selection_scores=scores,
        zero_inflation=pi_hat if family == "zip" else 0.0,
        sigma_scale=sigma_hat if family == "normal" else 1.0,
    )


def fit_dispersion_out_of_fold(
    train_predict,
    X,
    y: np.ndarray,
    *,
    market: str = "",
    n_folds: int = 3,
) -> CountDispersion:
    """
    Fit dispersion on OUT-OF-FOLD residuals from chronological folds.

    Fitting on in-sample residuals is the trap this exists to avoid: a
    boosted tree fits its own training rows tightly, so in-sample spread
    looks far smaller than reality and the resulting distribution is
    overconfident at exactly the lines people care about. Consistently
    selecting Poisson over Negative Binomial is the tell.

    ``train_predict`` takes (X_train, y_train, X_valid) and returns
    predictions for X_valid. Folds are chronological, so every prediction
    is made by a model that never saw that row or any later one.
    """
    from sklearn.model_selection import TimeSeriesSplit

    y = np.asarray(y, dtype=float)
    n = len(y)

    def _in_sample_fallback(reason: str) -> CountDispersion:
        """In-sample residuals understate spread — say so in the artifact.

        Returned unlabelled, this would be indistinguishable from a real
        out-of-fold fit while being systematically overconfident, which is
        the exact failure this function exists to prevent.
        """
        fitted = fit_count_dispersion(y, np.asarray(train_predict(X, y, X)), market=market)
        return CountDispersion(
            family=fitted.family,
            phi=fitted.phi,
            n_train_rows=fitted.n_train_rows,
            selection_scores=fitted.selection_scores,
            fallback_reason=(
                f"{reason} — dispersion estimated IN-SAMPLE, so the spread is "
                "likely understated and the distribution overconfident"
            ),
        )

    if n < MIN_ROWS_TO_FIT * 2:
        logger.warning(
            "dispersion %s: %d rows is too few for out-of-fold folds; falling back "
            "to an in-sample fit, which understates spread.",
            market or "?", n,
        )
        return _in_sample_fallback(f"only {n} training rows")

    splits = min(n_folds, max(2, n // MIN_ROWS_TO_FIT))
    oof_pred = np.full(n, np.nan)
    for train_idx, valid_idx in TimeSeriesSplit(n_splits=splits).split(np.arange(n)):
        X_tr = X.iloc[train_idx] if hasattr(X, "iloc") else X[train_idx]
        X_va = X.iloc[valid_idx] if hasattr(X, "iloc") else X[valid_idx]
        try:
            oof_pred[valid_idx] = np.asarray(train_predict(X_tr, y[train_idx], X_va), dtype=float)
        except Exception as exc:  # noqa: BLE001 — one bad fold must not lose the rest
            logger.warning("dispersion %s: fold failed (%s); leaving it out", market or "?", exc)

    scored = np.isfinite(oof_pred)
    if scored.sum() < MIN_ROWS_TO_FIT:
        logger.warning(
            "dispersion %s: only %d out-of-fold rows survived; using in-sample instead",
            market or "?", int(scored.sum()),
        )
        return _in_sample_fallback(f"only {int(scored.sum())} out-of-fold rows survived")

    logger.info(
        "dispersion %s: estimating from %d out-of-fold residuals across %d folds",
        market or "?", int(scored.sum()), splits,
    )
    return fit_count_dispersion(y[scored], oof_pred[scored], market=market)


def over_under_push_from_dispersion(
    mean: float,
    line: float,
    dispersion: CountDispersion,
) -> dict[str, object]:
    """
    P(over) / P(under) / P(push) at a line, using the fitted dispersion.

    Whole-number line N: over is stat > N, under is stat < N, and stat == N
    is a push carrying real probability mass. Half-point lines cannot push.
    """
    if mean is None or not np.isfinite(mean) or mean < 0:
        return {
            "status": "DATA_NOT_AVAILABLE",
            "reason": "Non-negative finite projection required",
            "probability_over": None,
            "probability_under": None,
            "probability_push": None,
            "distribution": None,
        }
    if line is None or not np.isfinite(line):
        return {
            "status": "DATA_NOT_AVAILABLE",
            "reason": "Finite prop line required",
            "probability_over": None,
            "probability_under": None,
            "probability_push": None,
            "distribution": None,
        }

    mu = max(float(mean), 1e-6)
    ln = float(line)

    # Each family gets its own pmf/cdf. An `else` that swept every
    # non-negbin family into Poisson would silently price a fitted ZIP or
    # normal with the wrong distribution — the family would be recorded in
    # the metadata and ignored in the arithmetic.
    if dispersion.family == "negbin":
        r, p = dispersion.nb_params(mu)
        pmf = lambda k: float(nbinom.pmf(k, r, p))  # noqa: E731
        cdf = lambda k: float(nbinom.cdf(k, r, p))  # noqa: E731
        name = "NegativeBinomial"
    elif dispersion.family == "zip":
        pi = float(np.clip(dispersion.zero_inflation, 0.0, MAX_ZERO_INFLATION))
        lam = dispersion.zip_lambda(mu)
        # The structural-zero mass sits entirely on k == 0.
        pmf = lambda k: float(  # noqa: E731
            (pi + (1.0 - pi) * poisson.pmf(0, lam)) if k <= 0
            else (1.0 - pi) * poisson.pmf(k, lam)
        )
        cdf = lambda k: float(pi + (1.0 - pi) * poisson.cdf(k, lam)) if k >= 0 else 0.0  # noqa: E731
        name = "ZeroInflatedPoisson"
    elif dispersion.family == "normal":
        sd = dispersion.normal_sd(mu)
        # Continuity correction: the probability of the integer k is the
        # mass between k-0.5 and k+0.5. Without it a whole-number line
        # would carry zero push mass, which is wrong for a count.
        pmf = lambda k: float(  # noqa: E731
            norm.cdf((k + 0.5 - mu) / sd) - norm.cdf((k - 0.5 - mu) / sd)
        )
        cdf = lambda k: float(norm.cdf((k + 0.5 - mu) / sd))  # noqa: E731
        name = "Normal"
    else:
        pmf = lambda k: float(poisson.pmf(k, mu))  # noqa: E731
        cdf = lambda k: float(poisson.cdf(k, mu))  # noqa: E731
        name = "Poisson"

    if float(ln) == float(int(ln)):
        n = int(ln)
        p_push = pmf(n)
        p_under = max(0.0, cdf(n) - p_push)
        p_over = max(0.0, 1.0 - cdf(n))
    else:
        k = int(np.floor(ln))
        p_under = cdf(k)
        p_over = max(0.0, 1.0 - p_under)
        p_push = 0.0

    total = p_over + p_under + p_push
    if total <= 0:
        return {
            "status": "DATA_NOT_AVAILABLE",
            "reason": "Degenerate distribution",
            "probability_over": None,
            "probability_under": None,
            "probability_push": None,
            "distribution": name,
        }

    return {
        "status": "OK",
        "distribution": name,
        "probability_over": round(p_over / total, 6),
        "probability_under": round(p_under / total, 6),
        "probability_push": round(p_push / total, 6),
        "projected_mean": round(mu, 4),
        "standard_deviation": round(float(np.sqrt(dispersion.variance_for(mu))), 4),
        "line": ln,
    }
