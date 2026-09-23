# Play-by-play features — built, validated, and not adopted

**Status: implemented; the layer exists and is correct. It did not improve
the baseline, and the reason is more interesting than the result.**

**Research only.** Nothing here prices a bet.

## The data

754,918 events across 1,313 games of 2025-26, from all ten log parts.

**The log is complete, and that was checked** against the box score on the
1,220 games it shares with the panel:

| check | result |
|---|---|
| player-games matching box-score FGA exactly | **99.90%** |
| mean absolute difference in attempts | 0.001 |
| games agreeing on total attempts | 1,211 of 1,220 |

### A correction to an earlier reading

With nine of the ten parts, the log was 3.0% short of the box score, unevenly
across games but proportionally *within* them — both teams, all four periods,
every rotation player each missing about a fifth of their attempts in the
worst game. I concluded the events had been randomly sub-sampled. **That was
wrong.** One part had not yet been supplied. With all ten, the parts are
disjoint (zero duplicates on `(gameId, actionNumber)`) and the log is whole.

The lesson worth keeping is the method, not the conclusion: check a derived
stream against an independent measurement of the same quantity before
building on it.

## What was built

All as **prior-game rolling means**, L5 and L10.

| feature | what it captures |
|---|---|
| `PBP_SHOT_DIST_AVG` | how far out a player shoots |
| `PBP_RIM_RATE` / `PBP_MID_RATE` / `PBP_THREE_RATE` | shot-location mix |
| `PBP_DUNK_LAYUP_RATE` | finishing at the basket |
| `PBP_ASSISTED_RATE` | share of makes a teammate set up — self-creation |
| `PBP_CLOSE_SHOT_SHARE` | attempts with the margin inside 5 |
| `PBP_GARBAGE_SHOT_SHARE` | attempts with the margin 20 or more |
| `PBP_PACE_ON_COURT` | possessions per 48 while this player is on the floor |

### On-court time is reconstructed and validated

Substitutions carry `personId` and an in/out direction but no starting
lineup, so a player is treated as on court from tip unless his first
substitution is an "in". That assumption is **tested, not believed**:

| | correlation with box-score MIN | mean abs error | within 2 min |
|---|---|---|---|
| reconstruction | **0.9999** | **0.20 min** | **100%** |

An overtime bug was caught by a test rather than by the data: elapsed time
counted `(period - 1)` regulation periods even in overtime, so period 5 began
at 2460 seconds instead of 2880 and every overtime event was mis-timed by a
full quarter. Only 4.6% of games reach overtime, which is why the aggregate
still looked fine — fixing it moved the correlation from 0.9970 to 0.9999 and
within-two-minutes from 97.7% to 100%.

### What is deliberately not shipped

- `PBP_SECONDS_ON_COURT` correlates with the box score's `MIN` at **0.997**,
  and `MIN_L5` / `MIN_SEASON` are already among the strongest columns in the
  panel. Shipping it would hand the model one number twice — the same
  collinearity that cost a third of the opponent-defence layer's gain.
- `PBP_FGA` is a count, and counts are what the completeness check is built
  from. It stays a diagnostic.

## The measurement

Both arms inside 2025-26, identical rows, three chronological folds. Selected
cells; the full table is in the commit.

| market | model | metric | off | on | delta | sd | folds better |
|---|---|---|---|---|---|---|---|
| PTS | xgboost | Brier raw | 0.24624 | 0.24644 | +0.00020 | 0.00015 | 1/3 |
| PTS | line_aware | ECE cal | 0.02377 | 0.02953 | +0.00577 | 0.00176 | 0/3 |
| REB | catboost | Brier cal | 0.24323 | 0.24280 | **−0.00043** | 0.00030 | **3/3** |
| REB | xgboost | Brier raw | 0.24222 | 0.24267 | +0.00044 | 0.00066 | 1/3 |
| AST | line_aware | Brier cal | 0.24292 | 0.24233 | **−0.00059** | 0.00034 | **3/3** |
| AST | xgboost | Brier raw | 0.24032 | 0.24076 | +0.00044 | 0.00070 | 1/3 |

**No consistent improvement.** Two cells show a clean small gain (CatBoost on
rebounds, line_aware on assists — both unanimous across folds and both
clearing their own spread). Several show a clean small loss, XGBoost most
consistently. The deltas live in the fourth decimal place and point in
different directions by model and market, which is what noise looks like with
a couple of coincidences in it.

## Why — and this is the finding

**Restricting the training window to the seasons the logs cover costs far
more than the features add.**

The event logs exist for 2025-26 only. A feature present in the validation
window and nowhere earlier is the shape of a leak, so both arms had to sit
inside that season — roughly 12,000 training rows instead of 187,733.

| | training rows | XGBoost Brier (PTS / REB / AST) |
|---|---|---|
| nine-season baseline, validating on 2025-26 | 184,682 | 0.2403 / 0.2378 / 0.2352 |
| 2025-26 only, with pbp features | ~12,000 | 0.2464 / 0.2427 / 0.2408 |

*(different validation windows, so indicative rather than a controlled
comparison — but the gap is ~0.006 Brier, an order of magnitude larger than
any pbp delta at ~0.0005.)*

Losing eight seasons of history costs about ten times what the shot-mix
features return.

## What would change the answer

Event logs for earlier seasons. The layer is built, validated and wired; it
needs no further work to be re-measured. With 2018-2025 logs the comparison
could run on the full panel, where the features would be judged on their own
merit instead of against the cost of the window they force.

```
python -m scripts.feature_ab --layer pbp \
    --panel data/external/training_pack/panel_pbp.parquet \
    --seasons-only <seasons with logs> --markets PTS,REB,AST --folds 5
```
