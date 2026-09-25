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
| `PBP_GAME_PACE` | possessions per 48 **per team**, of the game this player appeared in |

> **`PBP_GAME_PACE` is a game constant, not a player measurement.** It was
> called `PBP_PACE_ON_COURT` and documented as "possessions per 48 while this
> player is on the floor". That was wrong: the player's own seconds cancel out
> of the arithmetic, so every player in a game receives the identical value
> (verified at a within-game standard deviation of 0.0). It is now computed
> and named as what it is. Two consequences worth knowing before using it:
> its total was also un-halved, counting both teams' possessions and putting
> "pace" near 200 instead of the league's ~100; and it only becomes
> player-specific after being rolled over each player's own schedule.
>
> **On collinearity — the precise version.** `PBP_GAME_PACE_L10` correlates
> 0.82 with `PACE_ROLL` and 0.78 with `PACE_MULTIPLIER`, but **no market reads
> either column**, so that is not a modelling concern. Among pace columns the
> models actually read, the only other is `DEF_PACE_L10`, at r = 0.04 —
> effectively independent. An earlier revision of this note called the 0.82 a
> redundancy to remove; it is not.

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

Three seasons of event logs (2023-24, 2024-25, 2025-26), all passing the
completeness check. Both arms on identical rows, 77,546 of them, three
chronological folds: training on the earlier seasons, validating across
2025-26.

**The answer is per-market, and it is no longer ambiguous.**

### Rebounds — the features earn their place

Every model improves, on every fold, by three to four times its own
fold-to-fold spread.

| model | metric | off | on | delta | sd | folds |
|---|---|---|---|---|---|---|
| catboost | Brier raw | 0.24000 | 0.23870 | **−0.00130** | 0.00036 | **3/3** |
| ensemble | Brier raw | 0.24045 | 0.23961 | **−0.00084** | 0.00021 | **3/3** |
| line_aware | Brier raw | 0.24419 | 0.24290 | **−0.00129** | 0.00046 | **3/3** |
| line_aware | ECE raw | 0.03280 | 0.02313 | **−0.00967** | 0.00702 | **3/3** |
| xgboost | Brier raw | 0.24121 | 0.24063 | **−0.00058** | 0.00040 | **3/3** |

This is the one place the mechanism is obvious in advance: a rebound needs a
miss, and where a team shoots from decides where the ball comes off. A
rim-heavy diet produces different rebound chances than a three-heavy one, and
no box-score column says which a team runs.

### Assists — smaller, equally consistent

| model | metric | off | on | delta | sd | folds |
|---|---|---|---|---|---|---|
| ensemble | Brier raw | 0.23632 | 0.23592 | **−0.00040** | 0.00009 | **3/3** |
| ensemble | ECE raw | 0.01443 | 0.01237 | **−0.00207** | 0.00114 | **3/3** |
| xgboost | Brier raw | 0.23707 | 0.23643 | **−0.00065** | 0.00019 | **3/3** |
| xgboost | ECE raw | 0.01943 | 0.01643 | **−0.00300** | 0.00116 | **3/3** |

### Points — no

| model | metric | off | on | delta | sd | folds |
|---|---|---|---|---|---|---|
| xgboost | Brier raw | 0.24174 | 0.24235 | +0.00061 | 0.00073 | 1/3 |
| ensemble | Brier raw | 0.24109 | 0.24145 | +0.00036 | 0.00038 | 1/3 |
| catboost | Brier raw | 0.24082 | 0.24112 | +0.00030 | 0.00034 | 1/3 |
| xgboost | ECE raw | 0.01313 | 0.01663 | +0.00350 | 0.00483 | 1/3 |

Each delta sits inside its own fold spread, so none is individually
conclusive — but all three tree models moved the same way on the same folds,
and only `line_aware` improved. The reading: a scorer's volume is already
carried by `PTS_L10` and `MIN_L5`, and seven weak correlated columns cost
more in variance than they return.

**`_PBP_BY_MARKET["PTS"]` is now empty.** Rebounds and assists keep theirs.
A smaller PTS subset might work; it would need measuring, not restoring.

## What changed the answer

The earlier version of this note said the features could not be judged,
because the logs covered one season and both arms had to sit inside it —
about 12,000 training rows against 184,682, and losing eight seasons of
history cost roughly ten times what the features returned.

Two more seasons of logs removed that constraint. Training is now ~51,000
rows with event-log features present throughout, and the features are judged
on their own merit instead of against the cost of the window they forced.

The prediction in that note was that earlier logs would let the comparison
run fairly. That was right. The expectation that the features would then help
was right for rebounds and assists and wrong for points.
