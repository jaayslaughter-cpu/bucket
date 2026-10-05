# Personal fouls and defence-versus-position

RESEARCH_ONLY. Two feature layers, both **built, measurable, and deliberately
not in the feature contract**. This page is the record of what they compute,
what was measured, and the three things measurement caught — two defects in
the DvP layer and one false claim about the live endpoint.

| | module | tests | `feature_ab` arm |
|---|---|---|---|
| Fouls | `src/features/fouls.py` | `tests/test_fouls.py` | `--layer fouls` |
| DvP | `src/features/dvp.py` | `tests/test_dvp.py` | `--layer dvp` |

Both need a panel rebuilt after the ingestion mappings in section 0 — for
fouls either source will do (`scripts/ingest_training_pack.py` from the
archive, or `ingest-logs` from stats.nba.com); for DvP only the archive
carries a position. Neither layer is read by any market, so both arms run
with `--wire-under-test`.

## 0. The finding that made them possible

Both layers were first judged **impossible in this tree**, and the reasoning
was specific and checkable:

> `PlayerGameLog` has no `pf` column and no panel on disk carries `PF`.
> No position column exists anywhere — not in the 196-column panel, not in the
> 214-column pbp panel, not on any DB model.

Every one of those statements is true. The conclusion was wrong. The archive
the panel is *built from* —
`data/external/training_pack/player_boxes/PlayerStatistics_2018_to_2026.csv` —
carries both:

| source column | coverage in that file | mapped to |
|---|---|---|
| `foulsPersonal` | 304,395 / 305,614 rows (99.6%), mean 1.51, range 0–6 | `PF` |
| `startingPosition` | 96,650 / 305,614 rows — G/F/C, starters only | `STARTING_POSITION` |

`COLUMN_ALIASES` in `src/ingestion/kaggle_nba.py` simply never listed them, so
they were dropped at the panel boundary and every derived artifact
truthfully reported their absence. On the filtered 214,381-row regular-season
panel `PF` is present on **100.0%** of rows.

This is the AGENTS.md rule *"inspect, do not assume"* biting in the direction
that is easy to miss: the derived artifact was read, found wanting, and
believed. Read the source file.

## 1. Fouls — what and why

Nothing here predicts fouls. **Six fouls ends a player's night**, and a
disqualification is the one in-game event that truncates minutes with no
injury, blowout or rotation decision behind it. Two players with identical
scoring histories and 1.1 vs 3.6 fouls a game carry different minutes risk,
and the panel had no column that said so.

| column | definition |
|---|---|
| `PF_L5`, `PF_L10` | prior-games mean, shift-1, per player |
| `PF_SEASON` | prior-games expanding mean within player-season |
| `PF_PER_MIN_L10` | **ratio of totals**: Σ fouls / Σ minutes over the prior 10 |
| `PF_TROUBLE_RATE_L10` | share of the prior 10 games reaching **5** fouls |

Three choices worth stating:

**The rate is a ratio of totals, not a mean of ratios.** One foul in a
two-minute cameo is a per-minute rate of 0.5 — eight times any real player's
propensity — and a mean of ten such ratios is dominated by whichever game was
shortest. Summing both sides weights each game by the minutes it contributed.
Numerator and denominator are masked to the games where **both** are known, so
they always cover the same games.

**The trouble threshold is five, not six.** Six is the disqualification
itself, by which point the minutes are already gone. Five is the state a coach
reacts to.

**An out-of-range count is unknown, not clipped.** Clipping 23 to 6 would turn
a column that is *not* personal fouls into one that looks like personal fouls,
and a team total arriving under this name is far likelier than a player
committing 23 fouls. `src/ingestion/bigdataball.py` already writes a `pf`, to
`TeamGameStat`, and it is a team total — the same name and a different
measurement.

### Measured on the real panel (214,381 rows)

| column | coverage | mean |
|---|---|---|
| `PF_L5` / `PF_L10` | 99.4% | 1.86 |
| `PF_SEASON` | 97.7% | 1.88 |
| `PF_PER_MIN_L10` | 98.1% | 0.0888 /min |
| `PF_TROUBLE_RATE_L10` | 98.1% | 0.053 |

Within-season `|r|`, overlap-weighted, against the columns each market
already reads (the methodology of the table in `src/models/labels.py`):

| column | max `\|r\|` vs any listed feature | against |
|---|---|---|
| `PF_SEASON` | 0.631 | `MIN_SEASON` |
| `PF_L10` | 0.607 | `MIN_L10` |
| `PF_L5` | 0.564 | `MIN_L5` |
| `PF_PER_MIN_L10` | 0.402 | `MIN_L10` |
| `PF_TROUBLE_RATE_L10` | 0.275–0.330 | `MIN_L10` / `REB_L10` |

The level columns are **moderately** collinear with minutes, which is exactly
what a count accumulated over playing time should be. That is the argument for
publishing the rate and the share rather than the mean alone, and it holds up:
`PF_L10` vs `PF_PER_MIN_L10` is **0.382** and `PF_L10` vs
`PF_TROUBLE_RATE_L10` is **0.663** — three related columns, not three copies.
Nothing here is near the 0.83–0.99 band `_EXCLUDED_AS_REDUNDANT` was built
from.

## 2. DvP — what and why

`src/features/defense.py` answers "how good is tonight's opponent at
defending?" and hands **every player in a game the identical number**. It
cannot answer "how good is this opponent at defending someone like this
player". This is the first column in the panel that varies by *who the player
is* as well as by whom he faces.

- **Three buckets, G/F/C**, because that is what the archive records. PG/SG/SF
  distinctions it does not carry would be a guess dressed as data.
- **The player's bucket is as-of**: the modal bucket of his own *prior* starts,
  expanding and shifted. His current designation would tell the model he is in
  tonight's starting five, which is a minutes signal about tonight.
- **The aggregation is a mean per player-game**, never a sum. `defense.py`
  refuses to read the player panel because a sum measures roster coverage as
  much as defence — dropping one player per team-game moved its "points
  allowed" by 33%. A position lives only in the player panel, so this layer
  must read it; a mean makes coverage change the sample size, not the level.
- **The window is ten of the DEFENDER's games.** The implementation this was
  adapted from grouped player rows by (season, opponent, position) and called
  `.rolling(10)` on them; a team faces four or five guards a night, so its
  "L10" spanned about two games.
- **The league baseline is as-of**, per (season, bucket): an expanding daily
  mean, shifted. The source used a season-wide median, which is the look-ahead
  `defense._league_relative_index` records being found and fixed once already.

### Position coverage, and the fallback that was rejected

| route | coverage | agreement with the player's own modal start |
|---|---|---|
| as-of modal prior start | **87.2%** of rows | 89.6% of 88,007 starter rows match that night's designation |
| `reference/Players.csv` G/F/C flags | all 1,659 panel players present | 85.1% on pure-flag players; hybrids mostly disagree |

The roster flags were measured and **not used**. 251 of the 1,659 players carry
no flag at all, and where they do the flag and the player's own starts
disagree often: `FLAG=F` yet modal C for 41 players and modal G for 14,
`FLAG=G` yet modal F for 56. The hybrids are worse than ambiguous — `FLAG=GF`
splits 68 F / 43 G and `FLAG=FC` splits 67 C / 25 F, so the flag does not even
indicate which of its two positions the player actually starts at. A
15%-wrong bucket silently mixes two defensive populations under one label. The
27,349 rows with no prior start get **no bucket and no DvP columns**.

### The two defects measurement found

**(a) A per-game minimum sample size annihilated the centre bucket.** A floor
of two player-games per (game, bucket) looked like prudence. The archive names
exactly **one** starting centre per team-game, so the C bucket's per-game
sample is always one, and every centre row came back `NaN` — a third of the
slate silently absent from a feature whose entire purpose is to distinguish
positions. The floor is gone; `ROLL_MIN_PERIODS = 5` over the defender's games
is where a thin sample is refused.
`tests/test_dvp.py::test_every_bucket_including_the_centre_is_populated`.

**(b) Aggregating over observed starters only compared two populations.**
`startingPosition` exists for starters, so an observed-only aggregation
measures what a defence allowed to opposing *starters* while the feature joins
onto every bucketed player, bench included. A completed game is now counted
under its observed designation where there is one and under the player's
as-of bucket otherwise — leakage-safe for the same reason the as-of bucket is,
and it raises the population from five players a side to everyone on the floor
with a bucket.

Before: 58.7% of rows matched, C bucket entirely null.
After: **81.7%** matched, all three buckets populated.

### The split is real

Mean allowed per player-game over the 175,209 matched rows:

| bucket | PTS | REB | AST | BLK | rows |
|---|---|---|---|---|---|
| C | 10.92 | 7.31 | 1.87 | 1.02 | 33,089 |
| F | 11.44 | 4.63 | 2.10 | 0.48 | 69,684 |
| G | 12.89 | 3.36 | 3.65 | 0.30 | 72,436 |

A centre's matchup concedes **2.2×** the rebounds and **3.4×** the blocks a
guard's does.

### Why both `ALLOWED` and `INDEX` are emitted

`DEF_RATING_INDEX_L10` is excluded from the contract as redundant: it and
`DEF_RATING_L10` describe one population — the team — and correlate at
`r = 0.999`. Here the raw column does **not** mean the same thing on every
row, and the measurement says so plainly:

| pair | `\|r\|` |
|---|---|
| `DVP_REB_ALLOWED_L10` vs the player's own `REB_SEASON` | **0.444** |
| `DVP_REB_INDEX_L10` vs the player's own `REB_SEASON` | **0.021** |
| `DVP_AST_ALLOWED_L10` vs the player's own `AST_SEASON` | 0.331 |
| `DVP_AST_INDEX_L10` vs the player's own `AST_SEASON` | 0.013 |

The raw column is partly a readout of **who the player is** rather than of
whom he is facing, because his bucket sets its scale. The index — allowed
divided by what the league allowed *that bucket* as of that date — removes it
almost completely. So the index is the modelling column and the raw value is
kept for reporting, which is the reverse of the defence layer's split, for a
stated reason.

`ALLOWED` and `INDEX` are not interchangeable either: `|r|` between them is
**0.410** on REB and **0.858** on PTS, which follows from the table above —
points separate least across buckets, rebounds most.

### It is not a renamed `DEF_*`

| index column | vs the team-level column covering the same stat | `\|r\|` |
|---|---|---|
| `DVP_REB_INDEX_L10` | `DEF_REB_ALLOWED_PER100_L10` | 0.389 |
| `DVP_AST_INDEX_L10` | `DEF_AST_ALLOWED_PER100_L10` | 0.376 |
| `DVP_PTS_INDEX_L10` | `DEF_RATING_L10` | 0.295 |

Most of what the position split carries is **not** in the team-level defence
columns. Compare `DEF_RATING_INDEX_L10`'s 0.999 against `DEF_RATING_L10`.

### Combination markets: `PRA`, and only `PRA`

`PRA` is a market here (`labels.POST_LAUNCH_MARKETS`) and now gets
`DVP_PRA_ALLOWED_L10` and `DVP_PRA_INDEX_L10`.

**`PR`, `PA` and `RA` were asked for and are deliberately absent.** They are
not markets in this project — `LAUNCH_MARKETS` is `(PTS, REB, AST)` and
`POST_LAUNCH_MARKETS` is `(FG3M, STL, BLK, PRA)` — so nothing could read them
and `feature_ab._dvp_for_market` has no market to route them to. Six columns
computed on every build for nobody is the exact state AGENTS.md records four
feature layers sitting in, with 71 numeric columns no market read. `DVP_COMBOS`
is the one place to turn one on, and
`test_every_combo_is_a_market_this_project_models` fails both ways: adding a
combo without the market, and removing the market while the combo stays.

#### The sum happens per player-game, not over the finished columns

"Allowed means add" is true only while the means share a denominator, and this
layer has two places they can stop doing so. Both aggregations skip nulls **per
column**: the per-game bucket mean is a `groupby(...).mean()`, and the window is
a `.rolling(min_periods=5).mean()`. So a player-game with points but no assists
sits in one mean's denominator and not the other's, and a bucket with eight
games of points and ten of rebounds averages two different sets of games.

Summing inside the player-game — masked to rows where every part is known —
makes one column that then travels the identical path a base stat travels.

**Measured, and on today's panel it changes nothing.** PTS, REB and AST are
null on **0** of the archive panel's 214,381 rows, so the two constructions
agree to 7.1e-15 on all 56,729 rows where both are known. That is floating
point, not disagreement. It is a **guard**, and what it guards against is
reachable: the live panel comes from `player_game_logs`, where `pts`, `reb` and
`ast` are each independently nullable.

It also takes two conditions, which is worth recording because the first
version of the test had only one. Nulling a component is not enough — the
fixture gave every player in a bucket the same line, so dropping one from the
AST denominator left the mean at 4 and the two constructions still agreed
exactly. The divergence needs the bucket to be **heterogeneous** as well. Both
directions are now asserted.

#### The index is re-derived and never summed

`_INDEX` columns are **ratios**. Adding three of them, each divided by a
different league baseline, gives a quantity centred on 3 with no
interpretation. Measured on the real panel:

| | mean | correlation with the correct column |
|---|---|---|
| `DVP_PTS_INDEX + DVP_REB_INDEX + DVP_AST_INDEX` | 3.009 | 0.917 |
| `DVP_PRA_INDEX_L10` (re-derived) | 1.001 | — |

At `r` = 0.917 the summed version is not even a monotone restatement of the
right number. `DVP_PRA_INDEX_L10` is `DVP_PRA_ALLOWED_L10` divided by what the
league allowed **that bucket** in PRA, through the same
`_bucket_relative_index` as every other index column, and it centres on 1.00 in
all three buckets (C 1.006 / F 0.997 / G 1.001).

#### The prior for the combo is weaker than for any of its parts

Stated here so the column is not read as more promising than it is. Combining
averages the position split away, because the component splits point in
opposite directions:

| allowed, per player-game | C | F | G | max/min |
|---|---|---|---|---|
| `REB` | 7.31 | 4.63 | 3.36 | **2.17** |
| `AST` | 1.87 | 2.10 | 3.65 | 1.96 |
| `PTS` | 10.92 | 11.44 | 12.89 | 1.18 |
| `PRA` | 20.09 | 18.16 | 19.91 | **1.11** |

Centres concede rebounds, guards concede assists, and the sum cancels most of
both — `PRA` is the flattest of the four across buckets. The index follows: its
within-bucket dispersion is the smallest of the set (sd 0.115 against REB's
0.133 and AST's 0.198), so there is less matchup left to find.

Coverage is the same 81.7% / 81.4% as the rest of the layer. Neither new column
is in the 0.83–0.99 redundancy band against anything `PRA` already reads — the
top correlate is `MKT_IMPLIED_TEAM_TOTAL` at 0.29, and `DEF_RATING_L10` is
0.20–0.25. The deletion test (check 2 of `scripts/audit_leakage.py`, the one
that catches an as-of mean quietly computed season-wide) moved **0 of 37,361**
surviving rows for both new columns.

Cheap, correct, leakage-clean, and with a smaller expected effect than the REB
arm found. `--layer dvp --wire-under-test --markets PRA` is the arm that would
settle it; it has not been run.

#### What this changed about the PRA arm

`feature_ab._dvp_for_market` used to give `PRA` the three **component**
matchups, because the layer had no combined column. It now routes by the same
equality as every other market, so `PRA` gets `DVP_PRA_*` and not the parts.
Offering both would hand the model one number twice on the `ALLOWED` side,
where `DVP_PRA_ALLOWED_L10` *is* the sum of the three components by
construction — well past the ~0.97 at which `labels._EXCLUDED_AS_REDUNDANT`
excludes whole feature families.

This invalidates nothing: only the REB arm has ever been run, and it reads
`DVP_REB_*` either way. Whether the three component `_INDEX` columns carry
something `DVP_PRA_INDEX_L10` cannot — they are ratios and do not add, so the
combo cannot reconstruct *which* stat a defence concedes — is a separate arm
nobody has run.

## 2b. The repository's own leakage auditor was run

`scripts/audit_leakage.py`'s deletion test rebuilds the feature matrix with
the last 30% of games deleted and requires that **no earlier row move, and no
value flip to or from null**. Two runs:

- `python -m scripts.audit_leakage --deletion-seasons 2025-26 --markets
  PTS,REB` — 6 of 7 checks pass; the deletion test flags only the two
  pre-existing columns named below, and none of the 17 new ones.
- the same deletion method applied directly over 2024-25 **and** 2025-26, to
  widen the sample and report per column:

| | null-flips | values moved | max abs delta |
|---|---|---|---|
| all 5 `PF_*` columns | 0 | 0 | 0 |
| all 12 `DVP_*` columns | 0 | 0 | 0 |

**One pre-existing check still fails, and it is not these layers.** The
deletion test flags `TEAM_DAYS_UNTIL_NEXT` and `IS_B2B_FIRST` (323 null-flips
each) from `src/features/schedule.py:138`, which are built with
`by_team.shift(-1)` — the team's *next* game date. That is a forward read, so
the test is right that it reads deleted games; whether it is leakage is a
separate question, because the NBA schedule is published months ahead, so on
the morning of game T it is genuinely known whether the team plays tomorrow.
The deletion test cannot tell a future *fixture* from a future *result*.
`IS_B2B_FIRST` is in `default_feature_cols`. This is recorded here because the
audit output is not clean and a reader should know which failure belongs to
what — `src/features/schedule.py` is untouched by this work.

## 3. What is still not established

Neither layer is in `src/models/labels.py::default_feature_cols`, and
**correlation is not the measurement that decides it**. The halflife and
usage-volume families were both predicted redundant from `|r|` and then
*tested*, and the test is what the exclusion rests on.

For **fouls** the honest status is unchanged: the columns exist, they are
leakage-safe, they are not copies of anything the contract already carries,
and whether they improve a Brier score is unmeasured.

```
python -m scripts.feature_ab --layer fouls --wire-under-test --markets PTS,REB,AST
```

For **DvP** it has now been run, for REB. Section 3a has the numbers. The
short version: the layer helps, by a small and direction-consistent amount,
and it is still not wired — for a reason that has nothing to do with the
measurement.

## 3a. DvP, measured — REB only

```
python -m scripts.feature_ab --layer dvp --wire-under-test --markets REB --folds 4 \
    --panel <panel>.parquet --seasons-only 2022-23,2023-24,2024-25
```

4 chronological folds, train cutoffs stepping 14 days from 2025-01-15, the
last validation window ending 2025-03-29. 76,585 panel rows; ~17,988 distinct
validation rows. REB reads exactly two of the twelve columns —
`DVP_REB_ALLOWED_L10` and `DVP_REB_INDEX_L10` — because `column_for_market`
routes a `DVP_*` column by the stat in the middle of its name. Coverage on the
slice: 85.1% for `_ALLOWED`, 84.7% for `_INDEX`.

| model | metric | off | on | delta | fold sd | folds better |
|---|---|---|---|---|---|---|
| catboost | Brier raw | 0.23722 | 0.23635 | **-0.00087** | 0.00065 | 4/4 |
| catboost | Brier cal | 0.23675 | 0.23576 | **-0.00099** | 0.00077 | 4/4 |
| ensemble | Brier raw | 0.23670 | 0.23600 | **-0.00070** | 0.00034 | 4/4 |
| ensemble | Brier cal | 0.23633 | 0.23573 | **-0.00061** | 0.00048 | 4/4 |
| line_aware | Brier raw | 0.23928 | 0.23863 | **-0.00065** | 0.00034 | 4/4 |
| line_aware | Brier cal | 0.23896 | 0.23842 | **-0.00054** | 0.00022 | 4/4 |
| xgboost | Brier raw | 0.23694 | 0.23628 | -0.00066 | 0.00053 | 3/4 |
| xgboost | Brier cal | 0.23646 | 0.23585 | **-0.00061** | 0.00052 | 4/4 |
| distribution | Brier raw | 0.24869 | 0.24869 | +0.00000 | 0.00000 | 0/4 |

Lower is better. The bar `feature_ab` prints under every table is that a mean
delta smaller than the fold-to-fold sd is not evidence. By that bar the
accuracy gain clears it for `ensemble` (2.1x sd raw), `line_aware` (2.0x raw,
2.5x calibrated) and `catboost` calibrated (1.3x), and the direction is the
same in all four folds for four of the five models. In absolute terms it is
0.0006–0.0010 of Brier on a base of 0.237 — a 0.3–0.4% relative improvement.
Small, repeatable, and not nothing; nobody should read it as more than that.

**Calibration did not improve, and may have got slightly worse.** Calibrated
ECE: catboost +0.00165 (sd 0.00438, 2/4 folds better), ensemble +0.00097
(sd 0.00472, 2/4), line_aware +0.00125 (sd 0.00300, **1/4**), xgboost -0.00230
(sd 0.00651, 2/3). Every one of those deltas is smaller than its own fold
spread, so none of them is evidence in either direction — but the sign is
adverse in three of four models and `line_aware` was better in only one fold
of four. Recorded as a thing to watch, not as a finding.

**`distribution` is a built-in null arm.** It moved by exactly 0.00000 on
every metric, 0/4 folds, because it fits a parametric distribution over a
fixed feature set and never sees a new column. An A/B harness that leaked a
difference through some shared path — a shared scaler, a reused split, a
cached frame — would move this row too. It did not.

### Why the three-season slice, and what it cost

Full-panel runs were attempted three times and never finished inside this
environment's uptime. The slice is a deliberate narrowing, and it is not free,
but it costs much less than it looks:

| | full panel | 2022-23 → 2024-25 |
|---|---|---|
| rows | 214,381 | 76,585 |
| line_aware source rows before the cutoff | 169,358 | 67,325 |
| line_aware rows kept after the 600,000 augmented-row cap | 66,666 (from 2022-03-08) | 66,666 (from 2022-10-22) |
| rows dropped by the cap | 102,692 | 659 |

`line_aware` trains on **the same 66,666-row budget either way** — the cap
already truncated the full panel to a window starting 2022-03-08, and the
slice starts 2022-10-22. So for the model whose delta was most consistent
across folds, the slice changed which seven months sit at the start of the
window and nothing else. `catboost`, `xgboost` and the `ensemble` over them
are the arms that genuinely lost data: 169,358 source rows down to 67,325,
about 60%. They still improved, 4/4 or 3/4. A fuller panel would plausibly
move the deltas; the direction is what this run establishes, not the size.

### Not a copy of anything wired

Measured on the same 76,585 rows, against REB's own wired numeric features.
The project's exclusion band is 0.83–0.99.

| new column | strongest correlate | next |
|---|---|---|
| `DVP_REB_ALLOWED_L10` | `REB_L10` / `REB_SEASON` 0.437 | `REB_L2` / `REB_BASELINE` 0.434, `REB_L5` 0.414, `DEF_REB_ALLOWED_PER100_L10` 0.154 |
| `DVP_REB_INDEX_L10` | `DEF_REB_ALLOWED_PER100_L10` 0.388 | `DEF_PACE_L10` 0.132, `DEF_RATING_L10` **0.025** |

Nothing is in the band. Two readings worth keeping:

- `DVP_REB_ALLOWED_L10` correlates ~0.44 with the player's *own* rebound
  history. That is the mechanism, not leakage: the number a player faces is
  the allowance to *his* position bucket, so a centre is handed the C-bucket
  figure and centres rebound more. It therefore carries some player-identity
  signal alongside the matchup signal, which is an argument for not shipping
  it alone.
- `DVP_REB_INDEX_L10` sits at **0.025** against `DEF_RATING_L10`. That is the
  question this layer was built to settle — whether splitting opponent defence
  by the position it is defending adds anything to a team-level number that
  hands every player in a game the same value — and the answer is that the
  position-relative index is very nearly orthogonal to it.

The two new columns correlate 0.410 with each other, so they are not two
spellings of one number either.

### It is measured and still not wired

The reason used to be that `STARTING_POSITION` had no writer on the live path.
**Section 5 is that writer.** What remains is narrower and still blocking:

1. the pull has never been run, because stats.nba.com is denied at this
   environment's proxy, so the column is NULL on every row today;
2. the measurement above was made on an **archive** panel, which is not the
   panel production builds from.

A model trained with a feature that is absent in production is worse than one
trained without it — the trees would have learned splits on a column that
arrives empty. So the order is unchanged except that its first step is done:
run the pull where nba.com is reachable, rebuild the live panel, confirm
non-null coverage there, re-run this arm on a panel that has real positions,
and only then touch `labels.py`. Wiring it on the strength of the table above
would ship a dead column, which is the `teammate_cascade.py` failure mode this
project has already documented once.

## 4. Database

`migrations/006_player_game_log_fouls.sql` adds `player_game_logs.pf`,
nullable, `CHECK (pf IS NULL OR pf BETWEEN 0 AND 6)`, `NOT VALID`, **no
backfill**. `NULL` means "this source did not report fouls", which is the
state of every row written before this change: both ingests carry the number
and neither was asked for it. What was never fetched cannot be recovered from
what was stored, so existing rows stay null and re-running either ingest fills
the rows it covers. A default of `0` would be a fabrication — zero fouls is a
specific, clean, low-risk game, and the feature layer can abstain on a null
but not on a zero.

`player_game_logs.pf` is read back by `repository.load_player_panel` as `PF`,
and `src/ingestion/boxscores.py` now fetches `PF` from the live
`leaguegamelog` payload, so `src/features/fouls.py` reaches the **live** path
and not only a panel rebuilt from the archive.

**A claim in the first draft of this page was wrong and is worth recording.**
It said stats.nba.com's league game log does not report personal fouls, so the
live path could never carry them. The endpoint's header list is in this
repository — `tests/test_boxscore_ingest.py` — and `PF` is in it, between
`TOV` and `PTS`. `boxscores.COLUMN_MAP` had simply never asked for it. The
same mistake as section 0, one layer out: a derived artifact was read, found
wanting, and believed, when the source was on disk. It was caught by
`scripts/verify_wiring.py`, whose dangling-reference check flagged that the
docstring cited `src/ingestion/nba_stats.py`, a module that does not exist.

**DvP did not, and section 5 is how that was closed.** The paragraph that
used to sit here said `STARTING_POSITION` reached the panel only from the
archive ingest, that `player_game_logs` had no position column and nothing
writing to it had one to write, and that closing it needed a **writer** first
— "a column nothing fills states a fact the system does not have". It named
two candidates, `src/ingestion/espn_game.py` and `boxscoresummaryv3`. Neither
is the one used; section 5 says why.

It also said "the NBA stats league game log does not report a starting
lineup", and that part is still true and still the reason this is a separate
pass rather than another column in `boxscores.COLUMN_MAP`.

## 5. The position writer

The gap section 3a left open was the writer, and this is it. Four pieces, in
the order a row travels:

| piece | what it does |
|---|---|
| `src/ingestion/starting_positions.py` | pulls the NBA's own traditional box score, one call per game, and normalises it to `(GAME_ID, TEAM_ID, PLAYER_ID, STARTING_POSITION)` |
| `migrations/008_player_game_log_starting_position.sql` | `player_game_logs.starting_position`, `VARCHAR(1)`, nullable, `CHECK IN ('G','F','C')`, `NOT VALID`, no backfill |
| `repository.update_starting_positions` | writes it onto rows that already exist, and never inserts |
| `scripts/pull_starting_positions.py` | the runnable pass: fetch → gate → cache → write |

### Why the traditional box score and not the two candidates named earlier

The paragraph this page used to end on named `src/ingestion/espn_game.py` and
`boxscoresummaryv3`. Neither is what got used.

- **`boxscoresummaryv3`** is the wrong endpoint. Its `InactivePlayers` set is
  who was OUT; its summary sets are game-level. No starting lineup.
- **ESPN** does carry a per-player box score, and the module's own docstring
  records that it "cannot reach the live endpoint to settle which [of two
  documented layouts] the summary returns". Taking a *second* unverifiable
  payload shape as the source for a column whose meaning is already the thing
  in doubt would compound the uncertainty rather than resolve it. ESPN also
  uses its own athlete ids, so it would need the name crosswalk too.
- **`boxscoretraditionalv3`** names the field directly and in the right
  vocabulary. The column list is not from memory: it is in the installed
  library's own `expected_data`, at
  `nba_api/stats/endpoints/boxscoretraditionalv3.py` — `gameId`, `teamId`,
  `personId`, `position`. v2 spells it `START_POSITION` and is the fallback,
  not the default, because the library's own module says v2 "is deprecated"
  and its data "is no longer being published ... as of the 2025-26 NBA
  season" — a season in this panel. The same reasoning as
  `inactive_players.py`'s endpoint preference, for the same reason.

### The one claim the fixtures cannot settle, and what is done about it

v3 renamed v2's `START_POSITION` to `position`, and **stats.nba.com is denied
at this environment's proxy** (`CONNECT tunnel failed, response 403`), so
nothing here can confirm the rename kept the meaning. Two readings:

- *starting* position — blank for everyone who came off the bench, which is
  what v2 plainly was and what the archive's column is;
- *listed* position — filled in for all twelve or thirteen players who
  dressed.

They are **different quantities**. If `position` were the listed one, DvP
would quietly start bucketing everyone who appeared instead of the five who
started; `POS_BUCKET` would stop meaning what section 3a measured it meaning;
every number downstream would still look plausible.

So the uncertainty is a **refusal, not an assumption**.
`check_starting_position_semantics` counts filled positions per team-game and
requires exactly five — not a heuristic, a rule of the sport. A
listed-position payload lands at eleven or more and is rejected by name, with
the count, on the first game pulled, before anything is written.
`scripts/pull_starting_positions.py` exits 3 and writes nothing;
`tests/test_dvp.py::test_a_payload_that_fails_the_semantics_gate_is_never_written`
pins that behaviourally. A team-game *short* of five is a different message,
because too few means rows are missing rather than that fewer players started.

An earlier version of that ordering test compared the two calls' positions in
the file with `body.index(...)`. A mutation that ran the gate on `frame.copy()`
and left a second call in place walked straight past it. The test is now
behavioural: a listed-position payload reaches the database never, whichever
line comes first.

### Three ways a position is NULL, and why they are not distinguished

| | meaning | how often |
|---|---|---|
| blank in the payload | the player did not start | ~8 of 13 rows |
| unrecognised spelling | counted, logged, never guessed | rare |
| no row pulled for this game | the pull has not covered it | all of them, today |

The feature layer treats all three the same way, which is why the column does
not try to tell them apart: `attach_dvp_features` has no bucket either way and
abstains. What it must never do is receive an empty string, because that
*parses* — a bench player would look like a reported position. Hence
`VARCHAR(1)` with a three-value `CHECK`, and normalisation through
`dvp.normalise_bucket` in the ingest **and again** in
`repository.update_starting_positions`: the second is not belt-and-braces, it
is the only guard on a caller that did not come through the ingest.

### It is not the leakage caveat `inactive_players.py` carries

Both read a pregame fact out of a post-game box score, so the resemblance is
worth being explicit about. That module's caveat is real — tonight's inactive
list is used for tonight's game, so a late scratch is information a decision
made at line-set time could not have had. Nothing here works that way:

- the bucket a **predicted** row is given comes from `assign_position_buckets`,
  the expanding modal of that player's **prior** starts, shifted. Tonight's
  designation is not an input to tonight's row at all.
- the bucket a **completed** game is counted under (`_aggregation_bucket`) is
  the observed one, and the allowed-against-bucket averages built from it are
  rolled and shifted before they reach a feature, so a game is only counted
  into a window that closes before the row reading it.

A late lineup change therefore costs this layer accuracy about a past game's
label, not foresight about a future one. Data quality, not leakage. Section 2b
is the auditor's own run over the layer.

### It updates and never inserts

A position with no game log behind it would be a row with a bucket and no
statistics — enough to shift an opponent's allowed-against-that-bucket average
while contributing nothing to it. Those pairs are **counted and reported** as
`unmatched`, not created and not dropped in silence, because a large unmatched
count means the two endpoints disagree about ids and that is worth seeing.
Clearing a position to NULL *is* permitted and counted separately: a corrected
payload that moves a player from starter to bench has to be able to say so.

### A defect found on the way in

`upsert_player_game_logs` carried `pf` in its insert payload and **not** in
its `ON CONFLICT DO UPDATE` set. Its own docstring says "re-ingesting a season
corrects rows rather than duplicating them", and for `pf` alone that was
false: a season ingested before `boxscores.COLUMN_MAP` asked for fouls kept
`pf` NULL forever, and re-ingesting could not fix it. Found only because
`starting_position` was about to go in by the same route and would have
inherited the same silence. Both are in the conflict set now, and
`test_the_upsert_carries_starting_position_and_updates_it_on_conflict` asserts
both halves for both columns.

### What is still not done

The pull has never been run. Running it needs a machine where nba.com is
reachable:

```
# where nba.com is reachable
python -m scripts.pull_starting_positions --season 2025-26 --limit 5 --dry-run
python -m scripts.pull_starting_positions --season 2025-26 --no-write

# where the database is
python -m scripts.pull_starting_positions --season 2025-26 --from-cache
```

`--limit 5 --dry-run` first, deliberately: five games is enough for the
five-starters gate to settle what `position` means, and costs five calls
rather than 1,230 if the answer is the wrong one.

Then the live panel is rebuilt, non-null coverage is confirmed on it, and the
section 3a arm is re-run on a panel carrying real positions. Only after that
does `labels.py` change. Until then DvP remains measured on history and inert
in production, and `AGENTS.md` section 7 says so.

No position column was added to any model. `STARTING_POSITION` is a panel
column read at feature-build time; `POS_BUCKET` is derived from it and written
to the feature matrix, and the *observed* designation is used internally by the
aggregation and never written anywhere —
`tests/test_dvp.py::test_the_observed_designation_is_never_written_to_the_panel`.
