"""
src/models/paired_bootstrap.py — is a metric difference real, or is it noise?

WHY THIS EXISTS. `compare.py` picked a market's winner with
``ranked.sort(key=_winner_rank_key)`` and took ``ranked[0]`` -- a decision made
entirely on POINT ESTIMATES. A Brier of 0.2412 beating 0.2418 was promoted as a
win, on a validation window of a few thousand rows, with nothing anywhere
asking whether that gap would survive a different sample. That is how a model
ships for free, and how a retrain gets credited with an improvement it did not
make.

So every difference here is only called ``improved`` when its interval
EXCLUDES ZERO in the direction that counts as better.

PAIRED, because the two models are scored on the same rows. Each resample draws
once and applies the same draw to both, so row-to-row difficulty cancels. An
unpaired bootstrap would inflate the interval with between-row variance both
models already agree about, and make a real improvement look like noise.

CLUSTERED BY GAME, which is where this departs from the implementation it was
read from (Kevocado/NBA_Predictor, surveyed 2026-10-24). That one resamples
ROWS independently:

    idx = rng.integers(0, n, n)

For team-game outcomes that is defensible. For PLAYER PROPS it is not: eight
rows from one game share a game script -- a blowout, a pace collapse, a
starter's early exit move every prop in that game together. Resampling rows
independently treats those eight as eight independent pieces of evidence, which
understates the interval and therefore calls noise significant. The resampling
unit has to be the thing that is independent, and that is the GAME. This
project already established the point in
``scripts/leg_correlation_dependence_check.py``, which measures the clustered
and iid intervals side by side precisely because they differ.

``games=None`` still works and still resamples rows, because a caller with no
game column is better served by a wide-ish interval than by none -- but the
result says ``cluster="iid"`` so nobody can conclude they got the clustered
answer when they did not.

RESEARCH_ONLY. This measures models. It prices nothing and sizes nothing.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import log_loss, roc_auc_score

#: Metric name -> (callable(y, score) -> float, lower_is_better).
#:
#: DIRECTION IS CARRIED WITH THE METRIC, not left to the caller. Lower is
#: better for Brier, log loss and MAE; higher for AUC. Without this a caller
#: reads a worse AUC as a win, which is the one mistake a significance test is
#: supposed to make impossible.
METRICS: dict[str, tuple[Callable[[np.ndarray, np.ndarray], float], bool]] = {
    "brier": (lambda y, p: float(np.mean((p - y) ** 2)), True),
    "log_loss": (lambda y, p: float(log_loss(y, p, labels=[0, 1])), True),
    "auc": (lambda y, p: float(roc_auc_score(y, p)), False),
    "mae": (lambda y, p: float(np.mean(np.abs(p - y))), True),
}

#: Probability metrics need scores strictly inside (0, 1): log loss is infinite
#: at the boundary and a single 0.0 would dominate every resample containing it.
_PROBABILITY_METRICS = frozenset({"brier", "log_loss", "auc"})
_EPS = 1e-12

DEFAULT_RESAMPLES = 2000

#: Fixed so a comparison is reproducible. A seed chosen per run would let
#: somebody re-roll until the interval excluded zero.
DEFAULT_SEED = 20261024

CLUSTER_GAME = "game"
CLUSTER_IID = "iid"


class PairedBootstrapError(ValueError):
    """Raised when the inputs cannot support a comparison at all."""


def _as_float(name: str, values: Sequence[float] | np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=float)
    if out.ndim != 1:
        raise PairedBootstrapError(f"{name} must be one-dimensional, got {out.shape}")
    return out


def _game_index(games: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Unique games, and the row indices belonging to each."""
    unique = np.unique(games)
    return unique, [np.flatnonzero(games == g) for g in unique]


def paired_bootstrap(
    y: Sequence[float] | np.ndarray,
    baseline: Sequence[float] | np.ndarray,
    candidate: Sequence[float] | np.ndarray,
    metric: str,
    *,
    games: Sequence[Any] | np.ndarray | None = None,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Bootstrap ``candidate - baseline`` on the same rows and say whether it is real.

    Returns both point estimates, their difference, the interval of that
    difference, and ``improved`` -- True only when the interval excludes zero in
    the direction that counts as better for ``metric``.

    RESAMPLES WHERE THE METRIC IS UNDEFINED ARE DROPPED, not scored. An AUC draw
    that happens to contain one class only has no value; inventing 0.5 for it
    would drag the interval toward "no difference" for a reason that has nothing
    to do with either model. ``n_resamples_used`` reports how many counted, so a
    comparison that quietly lost most of its draws is visible rather than
    presented as 2000 resamples' worth of evidence.
    """
    if metric not in METRICS:
        raise PairedBootstrapError(
            f"unknown metric {metric!r}; expected one of {sorted(METRICS)}"
        )
    if not 0.0 < alpha < 1.0:
        raise PairedBootstrapError(f"alpha must be in (0, 1), got {alpha}")
    if n_resamples < 1:
        raise PairedBootstrapError(f"n_resamples must be positive, got {n_resamples}")

    score, lower_is_better = METRICS[metric]

    y = _as_float("y", y)
    baseline = _as_float("baseline", baseline)
    candidate = _as_float("candidate", candidate)
    if not (y.shape == baseline.shape == candidate.shape):
        raise PairedBootstrapError(
            "y, baseline and candidate must align: "
            f"{y.shape}, {baseline.shape}, {candidate.shape}"
        )
    n = int(y.size)
    if n == 0:
        raise PairedBootstrapError("nothing to bootstrap")

    if metric in _PROBABILITY_METRICS:
        baseline = np.clip(baseline, _EPS, 1.0 - _EPS)
        candidate = np.clip(candidate, _EPS, 1.0 - _EPS)

    cluster = CLUSTER_IID
    unique_games: np.ndarray | None = None
    rows_for_game: list[np.ndarray] | None = None
    if games is not None:
        games_arr = np.asarray(games)
        if games_arr.shape != y.shape:
            raise PairedBootstrapError(
                f"games must align with y: {games_arr.shape} vs {y.shape}"
            )
        unique_games, rows_for_game = _game_index(games_arr)
        cluster = CLUSTER_GAME

    base_point = score(y, baseline)
    cand_point = score(y, candidate)

    rng = np.random.default_rng(seed)
    diffs: list[float] = []
    for _ in range(n_resamples):
        if cluster == CLUSTER_GAME:
            assert unique_games is not None and rows_for_game is not None
            picked = rng.integers(0, unique_games.size, unique_games.size)
            idx = np.concatenate([rows_for_game[j] for j in picked])
        else:
            # ONE draw, applied to both models: that is what makes it paired.
            idx = rng.integers(0, n, n)
        y_draw = y[idx]
        try:
            # MEASURED, not assumed: a one-class AUC draw does NOT raise.
            # sklearn's roc_auc_score emits UndefinedMetricWarning and returns
            # nan, so `isfinite` below is what actually discards the draw. The
            # warning is expected here and suppressed rather than printed 286
            # times per run; the ValueError guard stays for the metrics that do
            # raise on degenerate input.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UndefinedMetricWarning)
                difference = (
                    score(y_draw, candidate[idx]) - score(y_draw, baseline[idx])
                )
        except ValueError:
            continue
        if np.isfinite(difference):
            diffs.append(float(difference))

    if not diffs:
        raise PairedBootstrapError(
            "no resample produced a defined metric; the comparison is degenerate"
        )

    drawn = np.asarray(diffs)
    low, high = np.percentile(drawn, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    excludes_zero = bool(low > 0.0 or high < 0.0)

    # `improved` IS the one-sided interval test, and nothing else is needed.
    #
    # The first version returned `better and excludes_zero`, written in the
    # belief that the conjunction was the safety gate. It is not: for a
    # lower-is-better metric `better` is `high < 0`, which already implies
    # `excludes_zero`, and for a higher-is-better one `low > 0` does the same.
    # Checked over every sign combination of (low, high) -- the conjunction
    # never differs from `better` alone. Removing a mutation test for it would
    # have been covering dead logic; `excludes_zero` stays as a REPORTED field,
    # because a two-sided answer is worth having in the output, but it does no
    # work in the verdict.
    better = bool(high < 0.0) if lower_is_better else bool(low > 0.0)

    return {
        "metric": metric,
        "lower_is_better": lower_is_better,
        "cluster": cluster,
        "n_rows": n,
        "n_games": int(unique_games.size) if unique_games is not None else None,
        "baseline": base_point,
        "candidate": cand_point,
        "difference": cand_point - base_point,
        "ci_low": float(low),
        "ci_high": float(high),
        "alpha": float(alpha),
        "n_resamples": int(n_resamples),
        "n_resamples_used": int(drawn.size),
        "seed": int(seed),
        "excludes_zero": excludes_zero,
        "improved": better,
        "research_only": True,
    }


def decisive_winner(
    y: Sequence[float] | np.ndarray,
    leader: Sequence[float] | np.ndarray,
    runner_up: Sequence[float] | np.ndarray,
    *,
    metric: str = "brier",
    games: Sequence[Any] | np.ndarray | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Is the point-estimate leader's lead over the runner-up real?

    The leader is the model `compare.py` already ranked first, so this does not
    choose it -- it says whether choosing it was supported. ``decisive`` False
    means the two are tied ON THE EVIDENCE, however far apart their point
    estimates look, and nothing downstream should treat the ranking as a result.
    """
    # The LEADER is the candidate and the runner-up is the baseline, so a
    # negative difference on a lower-is-better metric means the leader won.
    result = paired_bootstrap(
        y, runner_up, leader, metric, games=games, **kwargs
    )
    result["decisive"] = result["improved"]
    result["note"] = (
        "The leader's advantage excludes zero; the ranking is supported."
        if result["improved"]
        else "The leader's advantage is within sampling noise. The two models "
             "are tied on this evidence and the ranking is not a result."
    )
    return result
