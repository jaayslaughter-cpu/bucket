-- ============================================================================
-- PropIQ Analytics — the shot volume behind a player's points
-- Migration 009: player_game_logs.fga, player_game_logs.fta
-- Target: PostgreSQL 14+ / Supabase
-- Scope: NBA only. RESEARCH_ONLY — box-score counts, not a wager.
-- ============================================================================
--
-- WHY THESE COLUMNS EXIST
--
-- The PTS model could not score a single row from the database, and this is
-- why. Its feature contract names TS_PCT_L10, TS_PCT_TREND and FT_RATE_L10;
-- src/features/scoring_efficiency.py declares REQUIRED_COLS = ("PTS", "FGA",
-- "FTA") and abstains without them; and `player_game_logs` stored neither
-- attempt column. So the layer attached nothing, three contract features
-- were absent, and main.py's train/serve check skipped P(Over) for every
-- player, on every slate, permanently.
--
-- MEASURED, 2026-10-18, against a 2026-04-10 slate built from this table:
--
--     REB  scored 325/325
--     AST  scored 325/325
--     PTS  MISSING ['TS_PCT_L10', 'TS_PCT_TREND', 'FT_RATE_L10']
--
-- This was NOT a data-freshness problem and no ingest could fix it. The
-- artifacts were trained on a panel built from full box scores, so the
-- features existed at fit time and could not exist at serve time: a
-- train/serve gap that the contract check caught and reported honestly every
-- run, while nobody read the line.
--
-- WHY ONLY TWO COLUMNS, AND NOT fgm/ftm
--
-- `scoring_efficiency` reads FGA and FTA and nothing else:
--     TS% = PTS / (2 * (FGA + 0.44 * FTA)),  FT_RATE = FTA / FGA
-- Both live sources carry made shots too, and they are deliberately left out.
-- A column nothing reads is a column that rots -- this repository has deleted
-- a source precedence with no client (6d9a130) and found a document
-- describing seven modules that never existed. If a feature later needs FG%,
-- that is its own migration with its own reason.
--
-- WHY NULLABLE, WITH NO DEFAULT AND NO BACKFILL
--
-- NULL means "this source did not report attempts", which is the honest state
-- of every row written before this migration. A default of 0 would be a
-- fabrication with teeth: TS% divides by (2 * (FGA + 0.44 * FTA)), so a
-- zero-attempt row is a division by zero, and a backfilled zero would either
-- produce an infinity or be silently coerced into an efficiency figure for a
-- game nobody shot in. The layer abstains on a null; it cannot abstain on a
-- zero. Re-running either ingest fills the rows it covers.
--
-- RANGES, and what each one is actually for
--
-- Measured on 214,381 regular-season player-games (2018-01-01..2026-04-12):
-- player FGA runs 0-47 (mean 8.4) and player FTA runs 0-43 (mean 2.1).
--
--   * fga 0-60 DOES guard against a team total arriving under this name. Team
--     FGA in team_game_stats runs 66-122, so the bound separates the two
--     cleanly with 13 of headroom over the observed player maximum.
--   * fta 0-60 DOES NOT guard against that, and saying otherwise would be
--     false comfort: team FTA runs 2-59 and overlaps the player range
--     entirely. No range check can tell a player's free-throw attempts from a
--     team's. This bound is a sanity guard against a garbled value only.
--
-- NOT VALID so an existing table is not rewritten; the checks apply to every
-- new and updated row from here on.

ALTER TABLE player_game_logs
    ADD COLUMN IF NOT EXISTS fga INTEGER;
ALTER TABLE player_game_logs
    ADD COLUMN IF NOT EXISTS fta INTEGER;

ALTER TABLE player_game_logs
    DROP CONSTRAINT IF EXISTS ck_player_game_log_fga_range;
ALTER TABLE player_game_logs
    ADD CONSTRAINT ck_player_game_log_fga_range
    CHECK (fga IS NULL OR (fga >= 0 AND fga <= 60))
    NOT VALID;

ALTER TABLE player_game_logs
    DROP CONSTRAINT IF EXISTS ck_player_game_log_fta_range;
ALTER TABLE player_game_logs
    ADD CONSTRAINT ck_player_game_log_fta_range
    CHECK (fta IS NULL OR (fta >= 0 AND fta <= 60))
    NOT VALID;

COMMENT ON COLUMN player_game_logs.fga IS
    'The PLAYER''s field-goal attempts, 0-60. NULL where the source did not '
    'report them — never 0, which TS% would divide by. Not a team total: '
    'team FGA runs 66-122 and the bound separates them.';

COMMENT ON COLUMN player_game_logs.fta IS
    'The PLAYER''s free-throw attempts, 0-60. NULL where the source did not '
    'report them — never 0. The bound is a sanity guard only: team FTA runs '
    '2-59 and overlaps this range, so it cannot distinguish a team total.';
