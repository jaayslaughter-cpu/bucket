# M1 — feature-conditional spread vs one global phi: measured, not adopted

**Date:** 2026-10-11 · **Verdict: DO NOT ADOPT** on all three markets ·
**RESEARCH_ONLY.** Held-out negative log-likelihood on a chronological slice.
Not a profitability claim and not evidence about any future slate.

## What was tested, and why

`src/models/residuals.py` predicts variance as `mean * phi` with **one phi per
market**, so the implied spread scales with the projection and with nothing
else: two players with the same projected points get the same interval even if
one has far more volatile minutes.

`mattdspecht/NBA-Contextual-Modeling` (MIT) trains LightGBM quantile heads
beside the mean model. `docs/external_repo_review_2026-10.md:189` identified
that on 2026-10-05 and it has been open since, with a gate attached: it must
beat the global phi on a leakage-safe split, measured, before being wired. We
already run quantile heads for *minutes* (`src/models/minutes_model.py:57`), so
this was the same technique one layer out, with CatBoost rather than a new
dependency.

`scripts/quantile_dispersion_ab.py` is the arm. Both arms are scored on the
**same `mu`** from the existing trained mean head, so the only thing that varies
is the spread model.

## Result

Chronological split: train ≤ 2025-06-30, test 2025-10-21 … 2026-06-13
(37,460 rows per market). Mean negative log-likelihood, lower is better.

| Market | fitted phi | A: global phi | B: quantile sigma | delta | verdict |
|---|---|---|---|---|---|
| PTS | 7.1758 | **2.78773** | 2.98389 | +0.19616 | DO NOT ADOPT |
| REB | 2.8907 | **2.03227** | 2.07809 | +0.04581 | DO NOT ADOPT |
| AST | 2.1088 | **1.59772** | 1.62777 | +0.03005 | DO NOT ADOPT |

The candidate is worse on **all three**, in the same direction, and **0 of
37,460 rows** fell back to the baseline in any market — it got a fair run on
every row rather than winning or losing on a fallback path. The margin is an
order of magnitude larger on PTS than on REB or AST.

## Why it probably lost, and what that does NOT rule out

The arms were not quite comparing like with like, and this is the honest
limitation of the experiment rather than a defect in it.

The negative binomial variance here must describe the spread of the outcome
around **the mean head's `mu`**. The quantile heads estimate the spread around
**their own conditional median**, which is a different centre. Converting
`(q90 − q10) / 2z` into a sigma and pairing it with another model's `mu` mixes
two models' notions of where the distribution sits, and on a right-skewed count
the median and the mean are not the same number. On PTS, where phi is 7.2 and
the distribution is widest, that mismatch should hurt most — which is the
pattern observed.

So what is ruled out is **this cheap version**: quantile heads bolted onto an
existing mean head. What is *not* ruled out is fitting the mean and quantile
heads jointly, so the interval is anchored on the same centre the variance is
measured against. That is a larger piece of work and would need its own gate.

## What was deliberately not copied

The source converts its interval into a probability by rebuilding a
**symmetric** normal with `scale = half_width * 0.78` and then setting
`prob_under = 1 - prob_over`
(`mattdspecht/NBA-Contextual-Modeling @ nba/modeling/inference.py:393-396`).
That discards the asymmetry the quantile heads exist to capture **and** ignores
pushes, which `src/quant/line_diff.py:103` exists to handle. The arm here feeds
the quantile spread into the same fitted count family instead, and floors the
variance at the mean — because a variance below the mean is not a count
distribution, which is precisely what their shortcut can silently produce.

## Status

The script stays. It is the record of the measurement and the way to re-run it
if the joint-fit version is ever attempted; `scripts/feature_ab.py` is the
precedent for keeping a measurement arm in the tree. Nothing is wired into
production: `residuals.py` is unchanged and the global phi still ships.

Two earlier arms this repository measured and did not wire, for the same
reason: `eb8a93d` (DvP on REB) and `0bee6a5` (the DvP arm for PRA).
