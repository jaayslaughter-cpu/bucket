"""
M1 — is a FEATURE-CONDITIONAL spread better than one global phi per market?

THE GAP THIS TESTS. `src/models/residuals.py` predicts variance as
``mean * phi`` with ONE phi per market, so the implied spread scales with the
projection and with nothing else: two players with the same projected points
get the same interval even if one has far more volatile minutes.

THE IDEA IS CITED. `mattdspecht/NBA-Contextual-Modeling` (MIT) trains LightGBM
quantile heads (``objective="quantile", alpha=0.10/0.90``) alongside the mean
model. `docs/external_repo_review_2026-10.md:189` identified it on 2026-10-05
and it has been open since. We already run quantile heads for MINUTES
(`src/models/minutes_model.py:57`), so this is the same technique one layer out,
with CatBoost rather than a new dependency.

WHAT MUST NOT COME WITH IT. Their conversion of the interval into a probability
rebuilds a SYMMETRIC normal (``scale = half_width * 0.78``) and then sets
``prob_under = 1 - prob_over``
(`mattdspecht/NBA-Contextual-Modeling @ nba/modeling/inference.py:393-396`).
That discards the asymmetry the quantile heads exist to capture AND ignores
pushes, which `src/quant/line_diff.py:103` exists to handle. So the quantile
spread feeds the SAME fitted count family here; it does not replace it.

THE GATE. This script decides nothing by itself: it reports held-out negative
log-likelihood for both arms on a leakage-safe chronological split. The
candidate is adopted only if it wins, and a loss is a result worth recording --
`eb8a93d` and `0bee6a5` are two earlier arms this repository measured and did
not wire.

RESEARCH_ONLY. Fits dispersion on historical residuals and prints two numbers.
No market is read, no probability is published, nothing is sized.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logger = logging.getLogger("propiq.quantile_ab")

#: The quantile pair, matching the source and our own minutes heads.
LOW_Q, HIGH_Q = 0.10, 0.90
#: z such that (q90 - q10) / (2z) is the sigma of a normal with those quantiles.
#: 1.2816 is the 90th percentile of the standard normal; the source used 1/0.78
#: ~ 1.28 implicitly and did not say so.
Z_90 = 1.2815515655446004


def _nll_negbin(y: np.ndarray, mu: np.ndarray, var: np.ndarray) -> float:
    """Mean NB negative log-likelihood, the same parameterisation residuals uses."""
    from scipy.stats import nbinom

    mu = np.maximum(mu, 1e-6)
    var = np.maximum(var, mu + 1e-6)
    p = mu / var
    r = (mu * mu) / np.maximum(var - mu, 1e-6)
    return float(-np.mean(nbinom.logpmf(np.round(y), r, p)))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--market", default="PTS")
    ap.add_argument("--train-end", default="2025-06-30")
    ap.add_argument("--validation-end", default="2026-06-30")
    ap.add_argument("--features", default="/tmp/m1_features.parquet",
                    help="prebuilt feature matrix (build_feature_matrix output)")
    ap.add_argument("--iterations", type=int, default=300)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s | %(message)s")

    market = args.market.upper()
    side = Path("data/external/model_runs/comparison")
    meta = json.loads((side / f"xgboost_{market}.meta.json").read_text())
    cols = meta["feature_cols"]

    feats = pd.read_parquet(args.features)
    target = market
    needed = set(cols) | {target, "GAME_DATE"}
    missing = sorted(needed - set(feats.columns))
    if missing:
        logger.error("feature matrix is missing %s", missing[:8])
        return 2

    work = feats.loc[:, sorted(needed)].dropna(subset=[target, "GAME_DATE"]).copy()
    work["GAME_DATE"] = pd.to_datetime(work["GAME_DATE"])

    from src.models.walk_forward import fixed_cutoff_split, sort_by_game_date

    # SORTED AND RE-INDEXED FIRST. `walk_forward.sort_by_game_date` resets the
    # index and its docstring states that the ChronoSplit indices refer to that
    # prepared frame, not to the caller's labels -- applying them to an
    # unsorted frame raises a KeyError listing thousands of ints, which is how
    # this script failed on its first run.
    work = sort_by_game_date(work)
    split = fixed_cutoff_split(
        work, train_end=args.train_end, validation_end=args.validation_end
    )
    tr = work.loc[split.train_idx]
    va = work.loc[split.validation_idx]
    logger.info(
        "split: train %s..%s (%d rows) | test %s..%s (%d rows)",
        split.train_start.date(), split.train_end.date(), len(tr),
        split.validation_start.date(), split.validation_end.date(), len(va),
    )

    # mu from the EXISTING trained mean head, so both arms are compared on the
    # same projection and the only thing that varies is the spread model.
    import xgboost as xgb

    mean_head = xgb.Booster()
    mean_head.load_model(str(side / f"xgboost_{market}.mean.json"))
    mu_tr = mean_head.predict(xgb.DMatrix(tr[cols], feature_names=cols))
    mu_va = mean_head.predict(xgb.DMatrix(va[cols], feature_names=cols))
    y_tr = tr[target].to_numpy(dtype=float)
    y_va = va[target].to_numpy(dtype=float)

    # --- arm A: one global phi, fitted on training residuals (what we ship)
    from src.models.residuals import fit_count_dispersion

    disp = fit_count_dispersion(y_tr, mu_tr, market=market)
    var_a = np.array([disp.variance_for(float(m)) for m in mu_va])
    nll_a = _nll_negbin(y_va, mu_va, var_a)
    logger.info(
        "ARM A (global phi): family=%s phi=%.4f -> test NLL %.5f",
        disp.family, disp.phi, nll_a,
    )

    # --- arm B: per-row sigma from quantile heads, mapped to the same family
    from catboost import CatBoostRegressor

    sigmas = {}
    for q in (LOW_Q, HIGH_Q):
        m = CatBoostRegressor(
            loss_function=f"Quantile:alpha={q}",
            iterations=args.iterations, depth=6, learning_rate=0.05,
            random_seed=42, verbose=False, allow_writing_files=False,
        )
        m.fit(tr[cols], y_tr)
        sigmas[q] = m.predict(va[cols])
        logger.info("  fitted q%.0f head", q * 100)

    half = (sigmas[HIGH_Q] - sigmas[LOW_Q]) / 2.0
    # A crossed or collapsed interval is not a measurement; fall back to the
    # global phi for those rows rather than inventing a spread.
    sigma_row = np.where(half > 0, half / Z_90, np.nan)
    var_b = np.where(
        np.isfinite(sigma_row), sigma_row**2, var_a
    )
    crossed = int(np.sum(~np.isfinite(sigma_row)))
    # The count family still owns the floor: variance below the mean is not a
    # count distribution, and this is where their symmetric-normal shortcut
    # would have silently produced one.
    var_b = np.maximum(var_b, mu_va + 1e-6)
    nll_b = _nll_negbin(y_va, mu_va, var_b)
    logger.info(
        "ARM B (quantile-conditional sigma): %d/%d rows fell back -> test NLL %.5f",
        crossed, len(va), nll_b,
    )

    verdict = "ADOPT" if nll_b < nll_a else "DO NOT ADOPT"
    delta = nll_b - nll_a
    print()
    print(f"market={market}  test rows={len(va)}")
    print(f"  ARM A  global phi ({disp.family}, phi={disp.phi:.4f})  NLL = {nll_a:.5f}")
    print(f"  ARM B  quantile-conditional sigma                      NLL = {nll_b:.5f}")
    print(f"  delta = {delta:+.5f}  ({'better' if delta < 0 else 'worse'})  -> {verdict}")
    print()
    print("  RESEARCH_ONLY. NLL on a held-out chronological slice. Not a")
    print("  profitability claim and not evidence about any future slate.")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "market": market, "test_rows": int(len(va)),
            "train_end": args.train_end, "validation_end": args.validation_end,
            "arm_a_family": disp.family, "arm_a_phi": float(disp.phi),
            "arm_a_nll": nll_a, "arm_b_nll": nll_b, "delta": delta,
            "rows_fell_back": crossed, "verdict": verdict,
            "research_only": True,
        }, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
