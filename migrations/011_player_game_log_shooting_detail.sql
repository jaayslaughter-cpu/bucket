-- ============================================================================
-- PropIQ Analytics — the made shots and the rebound split behind the counts
-- Migration 011: player_game_logs.fgm, .fg3a, .ftm, .oreb, .dreb
-- Target: PostgreSQL 14+ / Supabase
-- Scope: NBA only. RESEARCH_ONLY — box-score counts, not a wager.
-- ============================================================================
--
-- THIS IS THE MIGRATION 009 SAID TO WRITE.
--
-- 009 added `fga` and `fta` and deliberately stopped there:
--
--     WHY ONLY TWO COLUMNS, AND NOT fgm/ftm
--     `scoring_efficiency` reads FGA and FTA and nothing else [...] Both live
--     sources carry made shots too, and they are deliberately left out. A
--     column nothing reads is a column that rots [...] If a feature later
--     needs FG%, that is its own migration with its own reason.
--
-- That reason now exists, and it is named below. This migration is NOT a
-- speculative widening of the table: it lands with its reader in the same
-- commit, and `tests/test_box_rates.py` fails if the reader stops reading.
--
-- THE READER: src/features/box_rates.py
--
-- Four rates that the panel could not express before, each on a SHIFTED
-- prior window so a game never sees itself:
--
--     EFG_PCT_L10   = (FGM + 0.5 * FG3M) / FGA     needs fgm
--     FT_PCT_L10    = FTM / FTA                    needs ftm
--     FG3A_RATE_L10 = FG3A / FGA                   needs fg3a
--     OREB_SHARE_L10 = OREB / (OREB + DREB)        needs oreb, dreb
--
-- WHAT THE LAST ONE IS NOT. It is NOT offensive rebound percentage. True
-- OREB% is OREB / (OREB + opponent DREB), which needs the opponent's
-- rebounds and therefore a team-level join this table cannot provide from a
-- player row. Calling it OREB% would be a fabrication with a respectable
-- name. It is the player's own split between offensive and defensive
-- rebounds, which is a usage signal rather than an opportunity-share one,
-- and the column name says `SHARE` so nobody reads it as the other thing.
--
-- WHERE THE DATA ALREADY WAS, AND WHY IT WAS DISCARDED
--
-- The ESPN ingest has fetched all five every night since the panel was built
-- on that source: `espn_panel.STAT_MAP` maps PTS/FGM/FGA/FG3M/FG3A/FTM/FTA/
-- OREB/DREB/REB/AST/STL/BLK/TOV/PF, and `espn_game` has already split
-- "11-18" into FGM/FGA before that runs. They were then dropped one step
-- later, at `scraper_panel.COLUMN_MAP`, which says why in its own comment:
--
--     Deliberately NOT every column it writes: [...] `oreb`/`dreb`,
--     `fgm`/`ftm` and `fg3a` have no column in `player_game_logs`, and
--     inventing one here would be a schema change disguised as an import.
--
-- So the mapper pointed at the schema, the schema pointed at 009, and 009
-- pointed at a future reason. Nothing was wrong; the chain was just open at
-- the far end. This closes it, and that comment is corrected in the same
-- commit so it no longer describes a column that exists.
--
-- WHY NULLABLE, WITH NO DEFAULT AND NO BACKFILL — the same reasoning as 009,
-- and it binds harder here. NULL means "this source did not report it",
-- which is the honest state of every row written before this migration. A
-- zero default would be a fabrication with teeth in all four rates: FT% is
-- FTM/FTA, so a backfilled zero makes every historical free throw a miss and
-- reads as a 0% shooter rather than an unknown one; EFG% and FG3A_RATE
-- divide by FGA and would publish an efficiency for a game nobody shot in;
-- OREB_SHARE divides by (OREB + DREB), so two zeros are a division by zero.
-- The layer abstains on a null. It cannot abstain on a zero.
--
-- Re-running the ESPN ingest fills the rows it covers. `raw_json` on this
-- table holds the payload for rows already written, so a backfill is
-- possible later and is deliberately not attempted here: it is a separate
-- operation over 56,000+ rows with its own failure modes, and the rates
-- abstain honestly until it runs.
--
-- RANGES, and what each one is actually for
--
-- Bounds mirror 009's: generous, and chosen to separate a PLAYER row from a
-- TEAM total where that is possible at all.
--
--   * fgm 0-40. Player FGM has never exceeded 36 in a game (Wilt, 1962);
--     team FGM runs roughly 25-70, so the bound separates them only at the
--     top of the team range. A sanity guard more than a namespace guard.
--   * fg3a 0-40. Player FG3A tops out around 24; team FG3A runs 15-70 and
--     overlaps entirely. SANITY ONLY — it cannot tell a player from a team,
--     and saying otherwise would be false comfort, exactly as 009 said of
--     fta.
--   * ftm 0-40. Same overlap as fta, same conclusion: sanity only.
--   * oreb 0-30 and dreb 0-40. Player OREB tops out near 18 and DREB near
--     30; team OREB runs 2-25 and team DREB 20-50, so these overlap too.
--
-- NOT VALID so an existing table is not rewritten; the checks apply to every
-- new and updated row from here on. Same choice as 006, 008 and 009.

ALTER TABLE player_game_logs
    ADD COLUMN IF NOT EXISTS fgm INTEGER;
ALTER TABLE player_game_logs
    ADD COLUMN IF NOT EXISTS fg3a INTEGER;
ALTER TABLE player_game_logs
    ADD COLUMN IF NOT EXISTS ftm INTEGER;
ALTER TABLE player_game_logs
    ADD COLUMN IF NOT EXISTS oreb INTEGER;
ALTER TABLE player_game_logs
    ADD COLUMN IF NOT EXISTS dreb INTEGER;

ALTER TABLE player_game_logs
    DROP CONSTRAINT IF EXISTS ck_player_game_log_fgm_range;
ALTER TABLE player_game_logs
    ADD CONSTRAINT ck_player_game_log_fgm_range
    CHECK (fgm IS NULL OR (fgm >= 0 AND fgm <= 40))
    NOT VALID;

ALTER TABLE player_game_logs
    DROP CONSTRAINT IF EXISTS ck_player_game_log_fg3a_range;
ALTER TABLE player_game_logs
    ADD CONSTRAINT ck_player_game_log_fg3a_range
    CHECK (fg3a IS NULL OR (fg3a >= 0 AND fg3a <= 40))
    NOT VALID;

ALTER TABLE player_game_logs
    DROP CONSTRAINT IF EXISTS ck_player_game_log_ftm_range;
ALTER TABLE player_game_logs
    ADD CONSTRAINT ck_player_game_log_ftm_range
    CHECK (ftm IS NULL OR (ftm >= 0 AND ftm <= 40))
    NOT VALID;

ALTER TABLE player_game_logs
    DROP CONSTRAINT IF EXISTS ck_player_game_log_oreb_range;
ALTER TABLE player_game_logs
    ADD CONSTRAINT ck_player_game_log_oreb_range
    CHECK (oreb IS NULL OR (oreb >= 0 AND oreb <= 30))
    NOT VALID;

ALTER TABLE player_game_logs
    DROP CONSTRAINT IF EXISTS ck_player_game_log_dreb_range;
ALTER TABLE player_game_logs
    ADD CONSTRAINT ck_player_game_log_dreb_range
    CHECK (dreb IS NULL OR (dreb >= 0 AND dreb <= 40))
    NOT VALID;

COMMENT ON COLUMN player_game_logs.fgm IS
    'The PLAYER''s made field goals, 0-40. NULL where the source did not '
    'report them — never 0, which would make every historical game a '
    'scoreless shooting night rather than an unknown one. Read by '
    'features/box_rates.py for EFG_PCT.';

COMMENT ON COLUMN player_game_logs.fg3a IS
    'The PLAYER''s three-point attempts, 0-40. NULL where unreported. The '
    'bound is a sanity guard only: team FG3A runs 15-70 and overlaps this '
    'range entirely. Read by features/box_rates.py for FG3A_RATE.';

COMMENT ON COLUMN player_game_logs.ftm IS
    'The PLAYER''s made free throws, 0-40. NULL where unreported — never 0, '
    'which FT% would read as a 0% shooter. Read by features/box_rates.py.';

COMMENT ON COLUMN player_game_logs.oreb IS
    'The PLAYER''s offensive rebounds, 0-30. NULL where unreported. Paired '
    'with dreb for OREB_SHARE, which is NOT offensive rebound percentage: '
    'true OREB%% needs the opponent''s defensive rebounds and no player row '
    'carries them.';

COMMENT ON COLUMN player_game_logs.dreb IS
    'The PLAYER''s defensive rebounds, 0-40. NULL where unreported. See the '
    'note on oreb about what OREB_SHARE is and is not.';
