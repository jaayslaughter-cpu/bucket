# Basketball-Reference season tables

## Attribution (required)

> Data from [Basketball-Reference.com](https://www.basketball-reference.com/)
> (Sports Reference LLC). When using SR data, please cite us and provide a
> link and/or a mention.

Sports Reference asks for that citation, so anything built from these files —
a report, a chart, a Discord message — has to carry it. The parser keeps the
string in `SR_ATTRIBUTION` and stamps it on `frame.attrs["attribution"]` so it
travels with the data rather than living only here.

## What goes in this directory

The CSVs exported from a season page, one file per table:

| Table | Basketball-Reference page | Parsed `kind` | Column prefix |
| --- | --- | --- | --- |
| Per Game | `.../leagues/NBA_2026_per_game.html` | `per_game` | `SR_PG_` |
| Play-by-Play | `.../leagues/NBA_2026_play-by-play.html` | `play_by_play` | `SR_PBP_` |
| Adjusted Shooting | `.../leagues/NBA_2026_adj_shooting.html` | `adjusted_shooting` | `SR_ADJ_` |

The `.csv` files themselves are **gitignored**. They are licensed
third-party data; keep them local, the same way the BigDataBall workbook is
kept local. Only this README and `.gitkeep` are tracked.

Name them so the season is unambiguous, e.g.
`NBA_2024-25_per_game.csv`.

## The one rule that matters

**These tables are season aggregates, and a season aggregate must never be
joined onto its own season.**

A row here summarises a player's entire season. Attached to a November game
as a feature, it tells the model the November game's own outcome — and
December's, and April's. A season TS% is computed *from* the game being
predicted. The `Awards` column is the same problem at its most obvious:
award shares are voted after the season ends, so "MVP-4" on a November row
is the season's ending handed to the model at its start.

`src/ingestion/basketball_reference.py` therefore refuses that join.
`prior_season_features` and `attach_prior_season_features` raise
`SeasonAggregateLeakageError` unless the table's season strictly precedes
the target season, and there is no flag to turn it off. For within-season
form, use the shift-1 rolling features in `src/features/builder.py`, which
are leakage-safe by construction.

Legitimate uses:

- **Prior-season priors.** A rookie-year TS%, a prior-season usage profile,
  last year's position mix — known before the target season tips off.
- **Role/archetype context.** The play-by-play table's position estimates
  (`PG%`…`C%`) are a far better description of how a player is actually used
  than the single `Pos` label, and they come from the prior season.
- **League baselines.** `league_average_row()` returns the trailer row, which
  is a fair prior-season normaliser.

## Usage

```bash
# Inspect a file: kind, rows, columns, multi-team blocks, what was dropped
PYTHONPATH=. python scripts/nba_model_cli.py ingest-basketball-reference \
    data/external/basketball_reference/NBA_2024-25_per_game.csv \
    --season 2024-25

# Write the de-duplicated season totals (one row per player)
PYTHONPATH=. python scripts/nba_model_cli.py ingest-basketball-reference \
    data/external/basketball_reference/NBA_2024-25_per_game.csv \
    --season 2024-25 --out data/cache/sr_per_game_2024-25.csv
```

`--season` is required and is never inferred: the CSV does not carry the
season, and a wrong one silently defeats every leakage check.

## Quirks the parser handles

1. **Traded players appear twice over.** A player dealt mid-season gets a
   `2TM`/`3TM`/`4TM` row (his season TOTAL) *plus* one row per team, all
   sharing the same `Rk`. Summing the raw column double-counts him — on the
   2025-26 play-by-play sample, James Harden's minutes are counted twice, a
   2,438-minute error. Use `season_totals()` or `team_splits()`, never the
   raw frame; rows are tagged `IS_MULTI_TEAM_TOTAL`.
2. **A `League Average` trailer row** with `-9999` in the id column. Dropped
   from the player rows, available via `league_average_row()`.
3. **Blank percentage cells mean "no attempts", not zero.** A player who took
   no threes has an empty `3P%`. Writing `0.0` there would tell a model he is
   a 0% shooter rather than a non-shooter. Blanks become NaN; a written
   `.000` stays `0.0`.
4. **Repeated column names under different group headers.** The play-by-play
   table's two-row header has `Shoot` and `Off.` twice — once under "Fouls
   Committed", once under "Fouls Drawn". A naive read yields `Shoot` and
   `Shoot.1` and invites mapping fouls drawn onto fouls committed. The group
   row is used to disambiguate: `FOULS_COMMITTED_SHOOT`, `FOULS_DRAWN_SHOOT`.

## Joining to the panel

The trailing `-additional` column holds the stable Basketball-Reference
player id (`doncilu01`), exposed as `BBREF_PLAYER_ID`. It is **not** the
NBA.com person id the panel keys on, and this repository has no crosswalk
between the two yet. Until one exists, `attach_prior_season_features` joins
on normalised player names (accents stripped, so `Nikola Jokić` matches
`Nikola Jokic`) and returns a `PriorSeasonJoinReport`. Read it before
training on these columns: a low match rate means the name join failed, not
that the league is full of rookies. Unmatched players keep NaN — they are
never filled with a league average, which would assert that a rookie is
exactly average.

Basketball-Reference also spells three franchises differently from NBA.com
(`BRK`, `CHO`, `PHO`); both are kept, as `TEAM` and `NBA_TEAM`.
