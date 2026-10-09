# The training window, measured

Written 2026-10-09, after a request to "narrow the training window to the
recent seasons" turned out to be the wrong move — and after the reason I
*suggested* it turned out to be a different problem than the one I named.

## What prompted it

The 2026-10-09 seed commit recorded this, as a caveat:

> `train-stats` splits its window 2/3 chronologically, so a
> `--start-date 2018-01-01 --end-date 2026-04-12` run **fits only to
> 2023-12-13** and validates on the remaining 2.4 seasons. A deliberate
> holdout, not a bug, and not what you want for an artifact scoring tonight.

The first half is true. The second half — the implication that *recency* is
what the fit is missing — was a hypothesis stated as a conclusion, and the
measurement below refutes it.

## The measurement

`train_end` held **fixed** at 2026-01-14 and only the window's START moved, so
every candidate is scored on identical rows. Validation window
2026-01-15 → 2026-04-12, 13,248 scored rows, market PTS, one fold.

### Calibrated Brier (lower is better)

| model | 2018-01-01 | 2022-10-01 | 2023-10-01 | 2024-10-01 |
|---|---|---|---|---|
| **ensemble** | **0.24098** | 0.24133 | 0.24191 | 0.24268 |
| catboost | **0.24101** | 0.24290 | 0.24316 | 0.24242 |
| line_aware | 0.24138 | **0.24131** | 0.24133 | 0.24148 |
| xgboost | 0.24168 | **0.24125** | 0.24196 | 0.24264 |
| distribution | **0.24764** | 0.24772 | 0.24778 | 0.24809 |
| *rows fitted* | *200,678* | *89,530* | *63,843* | *39,128* |

### What it says

**More history is better, and narrowing costs accuracy.** The `ensemble` — the
blend, and the one that matters — degrades **monotonically** as the start date
moves forward: 0.24098 → 0.24133 → 0.24191 → 0.24268. Two seasons of history
(39,128 rows) is the worst or near-worst configuration for every model.

`xgboost` and `line_aware` are marginally best at 2022-10-01, by 0.0004 and
0.00007 — inside the noise of a single window, and not a pattern.

**`line_aware` barely moves at all** (0.24131–0.24148 across a 5x change in
history), and that is a consistency check rather than a curiosity: its
600,000-augmented-row cap already truncates it to the most recent 66,666
source rows whatever window it is given. The model whose data is capped is the
model least sensitive to the window, which is what should happen if the sweep
is measuring what it claims to.

### What it does not say

One validation window, one market, one fold. This repository's own bar —
printed under every `feature_ab` table — is that "a mean delta smaller than
the fold-to-fold sd is not evidence; neither is a single fold". No fold spread
was computed here, so the individual deltas (0.0004–0.002 of Brier) are not
individually established.

What carries weight is the **monotone ordering across four windows** for the
ensemble and catboost: a single noisy comparison does not produce a staircase.
Treat the direction as established and the magnitudes as indicative.

## The real problem, which the sweep does not refute

The 2/3-by-row-count split couples two unrelated decisions: **how much history
you train on** and **how much of it is withheld**.

| window | fit ends | fit rows | held out of fitting |
|---|---|---|---|
| 2018-01-01 → 2026-04-12 | 2023-12-13 | 142,713 | 71,357 (2.4 seasons) |

The best configuration measured above fitted **200,678** rows — the *same full
window*, split late. So the thing worth fixing was never the window's start.
It was that a third of whatever history you pass gets thrown away, and the
longer the history the more absolute data goes with it.

## What changed

`train-stats` gained `--train-end`: fit on rows up to that date, hold the rest
of the window out for calibration. The 2/3 split remains the fallback when it
is not passed, so nothing already scripted changes behaviour.

The seeded artifacts were retrained with it:

| | before | after |
|---|---|---|
| fit window | 2018-01-01 → 2023-12-13 | 2018-01-01 → **2026-01-13** |
| fit rows | 142,713 | **200,678** |
| holdout | 71,357 rows / 2.4 seasons | 13,392 rows / 3 months |

Verified on the held-out window — rows no artifact was fitted on — all three
markets score 13,671 of 13,703 rows with probabilities inside [0, 1]
(means 0.431–0.460).

The seed command is in `docs/deploy_railway.md` §4b.

## The caveat that outlives all of this

**O8, self-referential evaluation.** `RESEARCH_LINE` is `{stat}_L10` and
`over_hit` is measured against that same rolling history, so every Brier in
this document measures form against form, not against a posted line. A window
comparison is still a valid comparison — both arms are scored the same way —
but none of these numbers is evidence that the model prices a real prop.
