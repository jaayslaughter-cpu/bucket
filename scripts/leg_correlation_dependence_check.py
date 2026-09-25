"""Measure the same-game dependence gap, with game-clustered uncertainty.

RESEARCH_ONLY. Reproduces the figures quoted in docs/decision_board.md under
"Fitting leg correlations". Run:

    PYTHONPATH=. python scripts/leg_correlation_dependence_check.py \
        --as-of 2025-01-01 --markets PTS,REB

WHAT IT MEASURES, and why it is two separate things.

The GAP is P(both legs over) minus P(a over) x P(b over) on realised outcomes.
It is a statement about the data: it does not use the fitted rho, so it cannot
be flattered by the fitter. A gap far from zero is why a same-game ticket
cannot be priced as the product of its legs.

The UNCERTAINTY on that gap has to account for clustering. Pairs from one game
share that night's pace, officiating and blowout risk, so treating 165,742
pairs as 165,742 independent observations would overstate precision. This
resamples WHOLE GAMES with replacement, and reports the i.i.d. bootstrap beside
it so the design effect is visible rather than assumed.

Measured on the 2018-2026 panel at --as-of 2025-01-01, same_player PTS x REB:
gap 0.05948, game-clustered SE 0.00061, 95% CI [0.0583, 0.0606], design effect
1.00x. The design effect is ~1 because the gap is a DIFFERENCE between the
joint rate and the product of the marginals -- a game-level shock moves both
terms together and cancels. It would not be ~1 for the joint rate alone.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from src.quant.leg_correlation import realised_leg_outcomes

logger = logging.getLogger(__name__)

DEFAULT_PANEL = "data/external/training_pack/panel.parquet"


def dependence_gap(over_a: np.ndarray, over_b: np.ndarray) -> float:
    """P(both over) - P(a over) * P(b over) on realised outcomes."""
    joint = float(((over_a > 0.5) & (over_b > 0.5)).mean())
    return joint - float(over_a.mean()) * float(over_b.mean())


def bootstrap_gap(
    over_a: np.ndarray,
    over_b: np.ndarray,
    games: np.ndarray,
    *,
    n_resamples: int,
    seed: int,
    cluster: bool,
) -> np.ndarray:
    """Bootstrap the gap, resampling whole games or individual pairs."""
    rng = np.random.default_rng(seed)
    n = len(over_a)
    if not cluster:
        return np.array([
            dependence_gap(*(lambda s: (over_a[s], over_b[s]))(rng.integers(0, n, n)))
            for _ in range(n_resamples)
        ])

    unique_games = np.unique(games)
    index_for = {g: np.flatnonzero(games == g) for g in unique_games}
    out = np.empty(n_resamples)
    for i in range(n_resamples):
        picked = rng.choice(unique_games, size=len(unique_games), replace=True)
        sel = np.concatenate([index_for[g] for g in picked])
        out[i] = dependence_gap(over_a[sel], over_b[sel])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", default=DEFAULT_PANEL)
    parser.add_argument("--as-of", required=True, help="Fit only on games strictly before this")
    parser.add_argument("--markets", default="PTS,REB", help="Exactly two, e.g. PTS,REB")
    parser.add_argument("--resamples", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260925)
    args = parser.parse_args()

    markets = [m.strip().upper() for m in args.markets.split(",") if m.strip()]
    if len(markets) != 2:
        print(f"DATA_NOT_AVAILABLE: --markets needs exactly two, got {markets}")
        return 2

    panel = pd.read_parquet(args.panel)
    outcomes = realised_leg_outcomes(panel, markets=markets, as_of=args.as_of)

    # One row per (game, player) carrying both markets' outcomes. Rows where
    # either market is missing or pushed are already dropped upstream.
    wide = (
        outcomes.pivot_table(
            index=["GAME_ID", "player_key"], columns="market",
            values="outcome", aggfunc="first",
        )
        .dropna()
        .reset_index()
    )
    if wide.empty or not set(markets).issubset(wide.columns):
        print(f"DATA_NOT_AVAILABLE: no rows carrying both {markets}")
        return 2

    a = wide[markets[0]].to_numpy()
    b = wide[markets[1]].to_numpy()
    games = wide["GAME_ID"].to_numpy()

    gap = dependence_gap(a, b)
    joint = float(((a > 0.5) & (b > 0.5)).mean())
    clustered = bootstrap_gap(a, b, games, n_resamples=args.resamples,
                              seed=args.seed, cluster=True)
    iid = bootstrap_gap(a, b, games, n_resamples=args.resamples,
                        seed=args.seed, cluster=False)
    se_clustered = float(clustered.std(ddof=1))
    se_iid = float(iid.std(ddof=1))
    lo, hi = (float(x) for x in np.percentile(clustered, [2.5, 97.5]))

    print(f"markets                  {markets[0]} x {markets[1]}   as-of {args.as_of}")
    print(f"pairs                    {len(a):,} across {len(np.unique(games)):,} games")
    print(f"P({markets[0]} over)              {a.mean():.6f}")
    print(f"P({markets[1]} over)              {b.mean():.6f}")
    print(f"observed joint           {joint:.6f}")
    print(f"independent product      {a.mean() * b.mean():.6f}")
    print(f"GAP                      {gap:.6f}")
    print(f"  game-clustered SE      {se_clustered:.6f}   gap/SE = {gap / se_clustered:.1f}")
    print(f"  i.i.d. SE              {se_iid:.6f}   gap/SE = {gap / se_iid:.1f}")
    print(f"  design effect          {se_clustered / se_iid:.2f}x")
    print(f"  95% clustered CI       [{lo:.6f}, {hi:.6f}]")
    print()
    print("RESEARCH_ONLY. The gap is a property of the outcomes and does not use")
    print("any fitted rho. It says the legs are dependent; it does not say the")
    print("fitted correlation generalises to games outside this window.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
