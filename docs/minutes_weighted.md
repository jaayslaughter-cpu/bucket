# Minutes-weighted recent averages (`{STAT}_MW_L5`)

RESEARCH_ONLY. Built, measurable, and **deliberately not in the feature
contract**. This page is the measurement that decision rests on.

Module: `src/features/minutes_weighted.py`. Tests: `tests/test_minutes_weighted.py`.

## What it computes

Last-`w` prior games per player-season, each game weighted by its minutes
against that player's prior season-to-date minutes mean:

| prior-game minutes vs prior `MIN_SEASON` | weight |
|---|---|
| `< 0.70 ×` | 0.5 |
| `> 0.85 ×` | 1.5 |
| otherwise | 1.0 |

Emitted: `{PTS,REB,AST,STL,BLK,FG3M}_MW_L{w}` plus the combo aliases
`PR_`, `PA_`, `RA_`, `PRA_MW_L{w}`, named from `w` rather than hardcoded.

Leakage: the stat and the minutes are both `.shift(1)` within player-season,
and the baseline is the panel's `MIN_SEASON`, which
`builder._expanding_prior_mean` computes as `shift(1).expanding().mean()`. That
was read rather than assumed, because the module's whole value depends on it.
`tests/test_minutes_weighted.py::test_the_current_game_never_enters_its_own_average`
pins it by tampering with row *i*'s own stat and requiring row *i*'s value not
to move.

## Why it is not in `default_feature_cols`

Within-season `|r|` against columns the contract already carries, on the real
214,381-row panel (204,529 overlapping rows, overlap-weighted — the same
methodology as the table in `src/models/labels.py`):

| column | vs `{STAT}_L5` | vs `{STAT}_BASELINE` | vs `{STAT}_L10` |
|---|---|---|---|
| `AST_MW_L5` | **0.992** | 0.984 | 0.953 |
| `PTS_MW_L5` | **0.991** | 0.984 | 0.952 |
| `REB_MW_L5` | **0.989** | 0.980 | 0.943 |
| `FG3M_MW_L5` | 0.988 | 0.974 | 0.918 |
| `BLK_MW_L5` | 0.984 | 0.965 | 0.885 |
| `STL_MW_L5` | 0.976 | 0.949 | 0.830 |

0.976–0.992 against `{STAT}_L5` is **inside** the band this repository already
excluded the whole halflife family for (0.971–0.990), where the A/B measured
Brier getting worse on every fold. A 0.5/1.0/1.5 reweighting of the same five
games is a second copy of one number, not a second opinion.

So: `build_feature_matrix` **produces** the columns, `scripts/feature_ab.py`
can **measure** them, and `labels.default_feature_cols` **omits** them. The
exclusion is recorded beside the others in `src/models/labels.py`
(`_EXCLUDED_AS_REDUNDANT`) so it reads as evidence rather than an oversight.

Promote only after a measurement, never by an edit:

```bash
python -m scripts.feature_ab --layer minutes_weighted --wire-under-test
```

## Three defects wiring it surfaced

Each has a test that fails without the fix.

1. **Combo aliases were hardcoded to `_L5`** while the window was a parameter,
   so `window=10` produced `PTS_MW_L10` beside a `PR_MW_L5` holding ten-game
   data — a column whose name contradicted its contents. Both paths now name
   from `window` via `_emitted_columns`.
2. **A missing `SEASON` fell back to `GAME_DATE.dt.year`**, which splits an NBA
   season at 1 January and restarts every player's window on New Year's Day.
   It now abstains and names the missing column. See
   `docs/season_key.md` — that same fallback existed in three other layers,
   where it also wrote its guess into the frame.
3. **The layer returned a re-sorted frame.** `builder` applies layers as
   `df = attach(df)`, so it silently reordered the whole feature matrix for
   every later layer and for the caller. The caller's row order and index are
   now restored before returning.

## What it is not

It is not a bet, a stake, or a recommendation. It is six columns and four
aliases that are currently excluded from the model's inputs on measured
grounds.
