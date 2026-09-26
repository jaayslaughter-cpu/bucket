# Opponent-defence features — built, wired, and honestly unproven

**Status: implemented and enabled** whenever a `team_games` frame is supplied.
Absent panels simply do not get the columns.

**Research only.** Nothing here prices a bet or sizes a stake.

## Correcting an earlier claim

My own audit said no defensive-matchup code existed. That was wrong.
`src/features/sports_ev_features.py` has had `attach_opp_allowed_l10` for some
time, producing `OPP_{STAT}_ALLOWED_L10` for six stats. The real problem was
not absence. It was two things: the columns were **read by nothing**, and the
way they were computed had two defects.

## Defect 1 — a panel sum measures roster coverage, not defence

`OPP_{STAT}_ALLOWED_L10` sums the panel's own player rows per team-game. That
is a sum over the players *present in the panel*, not over the team. Dropping
one player per team-game from the demo panel moves "opponent points allowed"
by **-32.7%**. Any filter — eligibility, a season slice, an archive with gaps —
silently rescales it, and two teams whose games happen to carry different
numbers of panel rows look like different defences when they are not.

`src/features/defense.py` reads **team totals** from the same `team_games`
frame that already feeds Elo. A team's points allowed is the other team's
points, which is complete by construction.

## Defect 2 — points per game confounds defence with tempo

A fast team allows more points while defending no worse. On the real 2025-26
panel:

| | correlation with team pace |
|---|---|
| points allowed **per game** | **+0.623** |
| points allowed **per 100 possessions** | **+0.158** |

Everything here is per 100 possessions, and tempo is published separately as
`DEF_PACE_L10` so a model gets two clean signals instead of one blurred one.
Pace earns its own column: a team's prior-10 pace predicts tonight's
possessions at r = +0.327, so it is persistent, not noise.

## What it produces

Keyed to the **defending** team — the player's opponent, not their own team.

| column | meaning |
|---|---|
| `DEF_RATING_L10` | points allowed per 100 possessions |
| `DEF_PACE_L10` | possessions per game |
| `DEF_REB_ALLOWED_PER100_L10` | rebounds conceded per 100 |
| `DEF_AST_ALLOWED_PER100_L10` | assists conceded per 100 |
| `DEF_FG3M_ALLOWED_PER100_L10` | threes conceded per 100 |
| `DEF_FGA_ALLOWED_PER100_L10` | shot volume faced per 100 |
| `DEF_TOV_FORCED_PER100_L10` | opponent turnovers per 100 |
| `DEF_FG_PCT_ALLOWED_L10` | field-goal percentage allowed |
| `DEF_RATING_INDEX_L10` | rating over the as-of league mean; **reporting only** |

On the real 2025-26 panel the spread runs from ORL at 103.1 points allowed
per 100 to MEM at 128.9 — a 26-point matchup range.

### Which rate bears on which market

The naive mirror is wrong twice over. Steals are made *by* a defence, not
allowed by it; what drives a player's steal count is how loose the **opponent**
is with the ball. Blocks need shots to exist at all.

| market | defensive feature |
|---|---|
| PTS | `DEF_RATING_L10` (already points allowed per 100) |
| REB | rebounds allowed **+ FG% allowed** — a rebound needs a miss |
| AST | assists allowed |
| FG3M | threes allowed + FG% allowed |
| STL | **opponent turnovers forced**, not opponent steals |
| BLK | **opponent shot volume** |

## A bug I shipped and then caught

The first version emitted `DEF_PTS_ALLOWED_PER100_L10` *and* `DEF_RATING_L10`.
They are the same quantity: **r = 1.0000**. With `DEF_RATING_INDEX_L10` at
r = 0.999 beside them, the PTS model received one number three times, and the
trees fragmented their splits across identical candidates. Measured on a panel
with a planted defensive signal, Brier over four chronological folds:

| features | Brier | ECE |
|---|---|---|
| no defence features | 0.26749 | 0.13222 |
| **rating alone** | **0.24536** | **0.12382** |
| rating + pace | 0.24622 | 0.12812 |
| index + pace | 0.24565 | 0.12386 |
| rating + index + points-per-100 + pace *(the shipped bug)* | 0.24793 | 0.12779 |

The duplicate cost about a third of the layer's gain. There is now a test that
fails if any two defensive features in one market's list correlate above 0.95.

## What the A/B actually showed, and why it is not the answer

Through the full pipeline, five chronological folds, planted signal, PTS:

| model | metric | off | on | delta | sd | folds better |
|---|---|---|---|---|---|---|
| xgboost | Brier raw | 0.25053 | 0.24812 | −0.00240 | 0.00381 | 4/5 |
| xgboost | ECE cal | 0.05172 | 0.04470 | −0.00702 | 0.00961 | 4/5 |
| ensemble | Brier raw | 0.23473 | 0.23377 | −0.00096 | 0.00192 | 4/5 |
| line_aware | Brier raw | 0.24307 | 0.24239 | −0.00067 | 0.00525 | 2/5 |

Mostly better, and **every mean delta is smaller than its fold-to-fold sd**.
By the harness's own standard that is not evidence.

The synthetic fixture is structurally rigged against the feature. Its teams
differ *only* in defence, so Elo — built from team points — is nearly a perfect
proxy for the planted signal, and the control arm is not defence-blind at all:

| | variance in defensive rating NOT explained by Elo |
|---|---|
| synthetic fixture | 14.2% |
| **real 2025-26 data** | **51.1%** |

On real data the specialised rates are more independent still: threes allowed
correlates with Elo at −0.12, turnovers forced at +0.16. So the fixture
understates the marginal value by roughly 3.6× on the rating alone, and by far
more on the rest.

**That is an argument for expecting a real gain, not a measurement of one.**
The real test needs real player game logs, which this environment does not
have. When they exist:

```
python -m scripts.feature_ab --layer defense --markets PTS,REB,AST --folds 5
```

## Still unfixed

`OPP_{STAT}_ALLOWED_L10` in `sports_ev_features.py` remains, unwired, with the
panel-sum defect described above. It was left alone rather than deleted
because nothing reads it and removing it is a separate change. It should not
be added to a feature list without being rebuilt on team totals first.

Separately, 87 of the 147 numeric columns the builder produces are read by no
model. The market and defence columns were two of those groups; the rest —
half-life, hot-hand, streaks, usage — have not been measured.
