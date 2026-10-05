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
*tested*, and the test is what the exclusion rests on. The honest status of
these two layers is: the columns exist, they are leakage-safe, they are not
copies of anything the contract already carries, and whether they improve a
Brier score is unmeasured.

```
python -m scripts.feature_ab --layer fouls --wire-under-test --markets PTS,REB,AST
python -m scripts.feature_ab --layer dvp   --wire-under-test --markets PTS,REB,AST
```

Record the result here and in `labels.py` when it is run. Until then neither
is a feature, and nothing downstream reads either.

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

**DvP does not, and that is a known gap rather than a surprise.**
`STARTING_POSITION` reaches the panel only from the archive ingest. The live
path builds its panel from `player_game_logs`, which has no position column,
and nothing writing to that table has one to write: the NBA stats league game
log does not report a starting lineup. On a live slate the layer logs that it
is skipping and adds no columns. Closing it needs a **writer** first — a
column nothing fills states a fact the system does not have — and the
candidates are `src/ingestion/espn_game.py` (`BoxScoreRow` is player-level)
or `boxscoresummaryv3` where `stats.nba.com` is reachable. Until then DvP is a
research column measured on history.

No position column was added to any model. `STARTING_POSITION` is a panel
column read at feature-build time; `POS_BUCKET` is derived from it and written
to the feature matrix, and the *observed* designation is used internally by the
aggregation and never written anywhere —
`tests/test_dvp.py::test_the_observed_designation_is_never_written_to_the_panel`.
