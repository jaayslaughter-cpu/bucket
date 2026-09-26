# XGBoost: a learned tree count instead of a hardcoded 400

**Status: implemented and on.** `n_estimators` is now chosen per fit from
cross-validated early stopping. The config value is a fallback.

**Research only.** This is about model accuracy, not about wagering.

## What was wrong

Two things, and the second is the substantive one.

**There was no `xgboost` block in `config/model_comparison.yaml` at all.**
CatBoost had one, complete with `early_stopping_rounds: 40`. XGBoost's
hyperparameters lived hardcoded in `src/models/xgboost_pipeline.py` and
`build_components` never passed any, so nothing in the config could reach
them.

**The 400 trees were never tuned against anything.** No early stopping, no
eval set, no search — just a round number applied to every market and every
panel size.

## What it costs

Two tasks, four chronological folds each, everything else held fixed:

| task | arm | Brier | ECE | trees |
|---|---|---|---|---|
| **real 2025-26 team panel** | fixed 400 | 0.28579 | 0.16684 | 400 |
| | fixed 100 | 0.25746 | 0.09244 | 100 |
| | **learned** | **0.24592** | **0.05469** | 6 |
| **planted-signal player panel** | fixed 400 | 0.23600 | 0.12413 | 400 |
| | fixed 100 | 0.22004 | 0.06592 | 100 |
| | **learned** | **0.21844** | **0.05153** | 76 |

Calibration is where the damage was: **ECE fell by 67% and 58%.** An
over-grown boosted tree drives its training probabilities toward 0 and 1, and
a fixed 400 does that regardless of how much signal the panel holds.

Note the two learned counts — 6 and 76. No single constant serves both, which
is the argument against any constant.

## Through the full pipeline

Three chronological folds, planted-signal panel, everything else held fixed:

| model | metric | fixed 400 | learned | delta | sd | folds better |
|---|---|---|---|---|---|---|
| xgboost | Brier raw | 0.23293 | 0.21719 | **−0.01574** | 0.00722 | 3/3 |
| xgboost | Brier cal | 0.23697 | 0.21824 | **−0.01872** | 0.01376 | 3/3 |
| xgboost | ECE raw | 0.09417 | 0.04983 | **−0.04433** | 0.02998 | 3/3 |
| xgboost | ECE cal | 0.09507 | 0.05177 | −0.04330 | 0.04352 | 3/3 |
| line_aware | Brier raw | 0.22958 | 0.22212 | **−0.00746** | 0.00096 | 3/3 |
| line_aware | Brier cal | 0.22789 | 0.22238 | **−0.00551** | 0.00134 | 3/3 |
| line_aware | ECE raw | 0.07317 | 0.05657 | **−0.01660** | 0.00699 | 3/3 |
| line_aware | ECE cal | 0.06653 | 0.04830 | **−0.01823** | 0.00418 | 3/3 |
| ensemble | Brier raw | 0.22421 | 0.22277 | −0.00144 | 0.00211 | 2/3 |
| ensemble | Brier cal | 0.22565 | 0.21982 | **−0.00583** | 0.00194 | 3/3 |
| ensemble | **ECE raw** | 0.04107 | 0.07250 | **+0.03143** | 0.00713 | **0/3** |
| ensemble | ECE cal | 0.05367 | 0.05293 | −0.00073 | 0.00793 | 1/3 |

Unlike the feature layers measured earlier, most of these mean deltas clear
their fold-to-fold spread — `line_aware`'s raw Brier by nearly eight times.
`line_aware` wraps XGBoost, so it inherits the change.

### The one regression, stated plainly

**The ensemble's raw ECE got worse, on all three folds.** Its components each
got better calibrated, and the blend got worse. The likeliest reading is that
its old raw ECE of 0.041 — better than any component's — was partly an
accident: XGBoost's overconfidence was cancelling against another component's
bias, and fixing one half of a cancellation breaks it.

Two things bound the damage. The ensemble's *calibrated* ECE is unchanged
(−0.00073), and the pipeline calibrates before anything reads a probability,
so nothing downstream consumes the raw blend. And its calibrated Brier
improved on all three folds.

It does suggest `ensemble_weights` were fitted — or at least chosen — against
the old component behaviour and are now stale. Refitting them is a separate
change and has not been done.

## How the count is chosen

The `TimeSeriesSplit` loop inside `fit()` already trained five fold models,
computed a log-loss, wrote it to a debug log and discarded them. Those fits
now do the work:

1. Each fold fits with `early_stopping_rounds` against **its own validation
   slice**. That slice comes from inside the training data and, because
   `TimeSeriesSplit` is chronological, lies strictly after the rows the fold
   trains on. The outer validation window is never touched — tuning the tree
   count on the rows the model is later scored against would flatter every
   metric the comparison reports.
2. The final model uses the **median** best iteration, scaled by
   `final rows / mean fold rows`.

No extra fits. The folds were already running.

### Why the median, and why the scaling

Per-fold best iterations on the 4,056-row panel ran **2, 25, 33, 49, 81**,
rising with fold size because `TimeSeriesSplit` gives later folds more data.

- **Median, not minimum.** A fold that stops after two trees is an unlucky
  slice. Using the minimum underfit measurably: Brier 0.22611 against 0.21844.
- **Scaled, because the final fit has more data than any fold.** On the
  signal-bearing panel that took 38 trees to 76, and Brier from 0.21983 to
  0.21844.

A trend-based extrapolation over fold size was not explored; the median with a
linear scale was the best of the arms tested and is simple enough to explain.

## Honest metadata

`get_model_metadata()` reported `dict(model_params)`, which still holds the
configured fallback. After a learned fit that published "400 trees" for a
model that grew 48. Exports now read `effective_params()`, which carries the
count actually used, its provenance (`cross_validated_early_stopping` or
`configured`) and the per-fold best iterations.

## When it does not run

With fewer than 100 rows, or when no fold has both classes, no count can be
learned. The pipeline then uses the configured `n_estimators` and logs that
the number is **not tuned**, rather than implying a search happened.

Setting `early_stopping_rounds: 0` disables the search deliberately and takes
the same path.

## Not done

The mean head (`XGBRegressor`, for the projected stat rather than the over
probability) still uses the configured count. It is a different target with a
different optimal depth of boosting, so it needs its own search rather than
the classifier's number; that is a separate change.
