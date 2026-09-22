# Blowout risk from the opening spread — measured, and not adopted

**Status: implemented, disabled by default.** `config/model_comparison.yaml`
sets `blowout.enabled: false`. This note records why, so the question is
settled by numbers rather than re-argued.

**Research only.** Nothing here prices a bet or sizes a stake.

## The idea

When a game is expected to be lopsided, starters sit in the fourth quarter.
Their counting stats stop accruing while the clock runs, so a points line
fair for 36 minutes is not fair for 29. The opening spread is the market's
pregame estimate of how lopsided the game will be, and it is posted before
tip, so it is admissible as a feature.

## What was built

`src/features/blowout.py` adds two columns from `MKT_OPENING_SPREAD`:

| column | definition (threshold `t`, default 9.0) |
|---|---|
| `BLOWOUT_FAV_HINGE` | `max(0, -spread - t)` — how big a favourite |
| `BLOWOUT_DOG_HINGE` | `max(0, spread - t)` — how big an underdog |

There is deliberately no symmetric `max(0, |spread| - t)`: only one of the
two can be positive, so it is exactly their sum and would be a perfectly
collinear column.

## The data

The BigDataBall workbook in `data/external/bigdataball/`: 1,322 real
2025-26 games, 2,644 team-games, with real opening spreads and real
quarter-by-quarter scores.

Player game logs were **not** available for this measurement (the archive
carries no `PlayerStatistics`, and `stats.nba.com` is blocked by the
session's network policy), so the label below is a **team-level proxy**,
not a player prop. That limitation is real and is restated at the end.

## Result 1 — the spread barely predicts a blowout at all

`P(|final margin| >= 15)` by opening-spread bucket:

| \|opening spread\| | n | P(blowout) | mean \|margin\| |
|---|---|---|---|
| 0–3 | 636 | 0.343 | 12.8 |
| 3–6 | 710 | 0.315 | 11.8 |
| 6–9 | 560 | 0.364 | 12.7 |
| 9–12 | 356 | 0.343 | 13.2 |
| 12+ | 382 | **0.555** | 18.0 |

Flat, and not even monotone, until 12+. Correlation of `|spread|` with
`|final margin|` is r = 0.171.

## Result 2 — the fourth-quarter effect is real, and tiny

Fourth-quarter points as a ratio of the team's own Q1–Q3 average:

| | n | 4Q ratio |
|---|---|---|
| opened 9+ point favourite | 387 | 0.933 |
| opened 9+ point underdog | 387 | 0.991 |

Difference 0.0586, se 0.0174, **t = 3.38**. The effect exists. But one
game's standard deviation on that ratio is 0.249, so the whole effect is
**0.235 sd** — far too small to move a per-game prediction.

Walk-forward RMSE on the continuous ratio, five folds:

| arm | RMSE |
|---|---|
| predict the constant mean | 0.24683 |
| spread + total | 0.24647 |
| spread + total + hinges | 0.24667 |

The hinge arm is worse than the plain spread arm, and both are within
0.0004 of predicting the mean.

## Result 3 — for tree models the hinges are empty by construction

A hinge is a monotone transform of the spread, so it offers a tree no split
point the raw spread does not already offer. Fitted on the real panel:

```
base  importance: SPREAD 0.247  TOTAL 0.256  ITT 0.288  IOT 0.210  FAV 0.000
blow  importance: SPREAD 0.247  TOTAL 0.256  ITT 0.288  IOT 0.210  FAV 0.000
                  FAV_HINGE 0.000   DOG_HINGE 0.000
```

Zero splits in 200 trees. Predictions identical to six decimal places;
Brier and ECE deltas of exactly 0.00000 at thresholds 6, 9 and 12.

Across eight configurations (two proxy labels × four thresholds) the Brier
delta was positive — worse — **every time**.

## What this does not establish

The label is team-level. Team fourth-quarter points barely move even in
blowouts (27.8 → 27.2 from a 0–5 margin to 25+) because substitutes replace
the starters who sat — which is exactly the player-level effect the feature
is aimed at, absorbed at the team level. A player-level A/B on real prop
data could come out differently.

That test needs real player game logs. When they are available:

```
python -m scripts.feature_ab --layer blowout --markets PTS,REB,AST
```

which fits both arms from one feature build and prints Brier and ECE, raw
and calibrated, with deltas.

## Found along the way

`MKT_OPENING_SPREAD`, `MKT_OPENING_TOTAL`, `MKT_IMPLIED_TEAM_TOTAL`,
`MKT_IMPLIED_OPP_TOTAL` and `MKT_IS_FAVORITE` were computed by
`src/features/market_context.py`, attached to every panel by
`build_feature_matrix`, and **read by nothing**. No feature list named them,
so no model ever saw the market's own pregame forecast — including the
implied team total that module's docstring calls "the single most
informative pregame number available" for a points prop.

They are now in `default_feature_cols`. Panels built without a
`market_lines` frame simply do not have them, and `resolve_feature_cols`
drops them with a warning rather than zero-filling a spread that was never
observed. The effect of wiring them in is itself measurable:

```
python -m scripts.feature_ab --layer market_context --markets PTS
```
