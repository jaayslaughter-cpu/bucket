# NBA historical archive (Kaggle export)

The player-game history a prop model is allowed to learn from. The files
themselves are **gitignored** — only this README is tracked.

## What goes here

| File | Size | Why it matters |
|---|---|---|
| `PlayerStatistics.csv` | 390 MB | One row per player per game. **This is the file that unblocks the panel.** |
| `TeamHistories.csv` | small | `teamId` → abbreviation crosswalk. **Required**, see below |
| `Games.csv` | 11 MB | 73,279 games, 1946→2026. Opponent, venue, game type |
| `LeagueSchedule25_26.csv` | small | Tip times (Eastern in the file; store UTC, display Pacific) |
| `Players.csv` | small | Person master: draft year, position flags |
| `PlayerStatisticsExtended.csv` | 453 MB | Usage, true shooting, pace, shot mix |
| `PlayByPlay.parquet` | 933 MB | Not loaded today |

## Loading it

```bash
PYTHONPATH=. python scripts/nba_model_cli.py ingest-kaggle \
    --path data/external/nba_archive/archive_small/PlayerStatistics.csv \
    --team-crosswalk data/external/nba_archive/archive_small/TeamHistories.csv \
    --describe          # inspect the mapping first

# then, to write it to Postgres
PYTHONPATH=. python scripts/nba_model_cli.py ingest-kaggle \
    --path ... --team-crosswalk ... --persist
```

## Three things about this export that broke the loader

The loader was written without access to these files. The real schema
differs in three ways, and only one of them failed loudly.

1. **There is no full-name column.** The file has `firstName` and
   `lastName`, so `PLAYER_NAME` came back missing and the whole export was
   refused. It is now **composed** from the two.

2. **There is no abbreviation column, and the nickname looks like one.**
   The file has `playerteamCity` ("Los Angeles") and `playerteamName`
   ("Lakers") — no "LAL" anywhere. `playerteamName` matched the alias table
   and mapped cleanly into `TEAM_ABBREVIATION`, and then every downstream
   join — market lines, Elo, team pace — silently matched *nothing* while
   the panel looked correct. Codes are now resolved from `playerteamId` via
   `TeamHistories.csv`, and a nickname that reaches `TEAM_ABBREVIATION` is
   **refused** rather than passed through. **This is why `--team-crosswalk`
   is required.**

   The crosswalk needs three corrections of its own, all join-critical:
   - codes carry trailing whitespace (`"ATL  "`)
   - San Antonio is `SAN` where every other source here uses `SAS`
   - All-Star rosters (`Team LeBron`, `East`) are not franchises; they use
     `9xxx` ids rather than `1610612xxx`
   - franchises move: team `1610612737` is `TRI` before 1950 and `ATL`
     after 1968, so the lookup is by season rather than flat

3. **There is no season column**, and preseason is mixed in. `SEASON` is
   derived from `gameDate` (October starts the next season), and `gameType`
   is carried as `GAME_TYPE` with an `IS_REGULAR_SEASON` flag. Filter on it
   before building rolling features: preseason minutes and rotations do not
   describe the same competition.

`comment` is carried as `DNP_COMMENT` — a did-not-play is what voids a prop
leg, so the reason has to reach the log.

## What this archive is not

It carries no spread, no total and no player prop. It cannot make a line
real. That check stays on PropLine.
