# The season key: one definition, and nobody else's column to write

Date: 2026-10-02. RESEARCH_ONLY. Module: `src/features/season.py`.
Test: `tests/test_season_key.py` (20 tests).

Found while reviewing the uploaded go-live pack (`docs/go_live_pack_review.md`),
by asking a narrower question: does `minutes_weighted`'s documented abstention
on a missing `SEASON` actually fire through `build_feature_matrix`? It did not.

## What was wrong

Three additive feature layers each carried their own copy of this line:

```python
if "SEASON" not in out.columns:
    out["SEASON"] = pd.to_datetime(out["GAME_DATE"], errors="coerce").dt.year
```

`src/features/halflife.py`, `src/features/hot_hand.py`,
`src/features/sports_ev_features.py`. Both halves of it were wrong.

### 1. The boundary

A calendar year splits an NBA season at **1 January**. Grouping by it restarts
every player's group on New Year's Day: the expanding season mean, the rolling
windows and the streak counters all reset mid-season, and nothing said so.

`src/features/defense.py` already had the correct form — the season's
*starting* year, cutting in August, where the NBA calendar has no games:

```python
(d.dt.year - (d.dt.month < 8).astype(int)).astype("Int64")
```

Four implementations, one of them right. It is now defined once in
`src/features/season.py::season_start_year`, and `defense.py` delegates to it.

### 2. Who owned the column — the worse half

The line wrote its guess into the returned frame under the **public** name
`SEASON`. `builder` applies layers as `df = attach(df)` and **halflife runs
first**, so one layer's private fallback became:

- the grouping key for every additive layer after it,
- a categorical feature column for `src/models/compare.py`
  (`DEFAULT_CATEGORICAL_COLS` includes `SEASON`),
- a column on the matrix the caller gets back.

And it silently defeated `minutes_weighted`'s abstention. That layer refuses to
guess a season precisely because a 1 January restart is a silently wrong
answer — but by the time it ran, `SEASON` was never missing. Measured, not
assumed: renaming `SEASON` out of the frame immediately before the layer loop
left `minutes_weighted_status == "OK"` anyway, because halflife had already put
one back.

A guess made by one module was read by five as a fact.

## The fix

`player_season_keys(frame)` returns `(frame, group_keys)`. When the panel
carries `SEASON` it is used and the frame is handed back untouched — so on the
production panel **nothing changes at all**. When it does not, the key is
derived into the private `_SEASON_KEY`, which the layer drops before returning
via `drop_season_key`.

A derived season is a layer's working assumption, not a column the next layer
inherits.

## What it changes, honestly

| | before | after |
|---|---|---|
| panel carries `SEASON` (production, the 214k panel) | panel's own column | unchanged, identical output |
| panel lacks `SEASON` | fabricated `SEASON` column, 1 Jan group restart, inherited by every later layer | correct Aug-cut key, private to each layer |
| `minutes_weighted` on a `SEASON`-less builder run | ran on a fabricated key, reported `OK` | abstains and names `SEASON`, as its own tests always claimed |
| `compare` categoricals on a `SEASON`-less panel | silently fed a wrong season label | column absent, so it is simply not used |

The last row is a behaviour change, and it is the point: a categorical
populated with a wrong season label is worse than no categorical.

## Mutation check

Reverting halflife alone fails 4 tests; reverting `hot_hand` and
`form_streaks` fails 5. The value-level test
(`test_the_derived_key_groups_exactly_as_an_explicit_season_would`) fails for
halflife and `form_streaks` — their numbers genuinely move — and does **not**
fail for `hot_hand` on that fixture, where only the contamination test
discriminates. Stated rather than smoothed over.

A separate mutation (grouping `minutes_weighted` by `SEASON` alone, so values
bleed across players) passes every layer-level test in
`tests/test_minutes_weighted.py` and is caught only by the builder-path
alignment test added with it.

## Still derived elsewhere, by design

`src/features/fatigue_logic.py` has no season column either, and warns about
it rather than inventing one:

> No SEASON column — rest is computed across season boundaries, so each
> player's first game of a season will show an offseason-length gap.

That is the right pattern and was left alone.
