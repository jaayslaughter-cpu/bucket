-- ============================================================================
-- PropIQ Analytics — the player's own personal fouls
-- Migration 006: player_game_logs.pf
-- Target: PostgreSQL 14+ / Supabase
-- Scope: NBA only. RESEARCH_ONLY — a box-score count, not a wager.
-- ============================================================================
--
-- WHY THIS COLUMN EXISTS
--
-- Six personal fouls ends a player's night. It is the one in-game event that
-- truncates minutes with no injury, blowout or rotation decision behind it,
-- and `player_game_logs` had no record of it: a foul-prone player and a clean
-- one were indistinguishable in this table. src/features/fouls.py builds the
-- prior-games foul history (PF_L5/L10/SEASON, a per-minute rate and the share
-- of recent games reaching five fouls) and had no database column to read
-- from or write to.
--
-- IT IS NOT team_game_stats.pf
--
-- That column already exists, is written by src/ingestion/bigdataball, and is
-- a TEAM total. The two are different measurements sharing a name. Reading
-- the team's figure as a player's would put five players' fouls on one player
-- and turn every starter into a disqualification risk.
--
-- WHY NULLABLE, WITH NO DEFAULT AND NO BACKFILL
--
-- NULL here means "this source did not report fouls", and that is the honest
-- state of every row written before this migration. Both live sources DO
-- carry the number and neither was asked for it:
--
--   * stats.nba.com's leaguegamelog has had PF in its header list all along
--     (between TOV and PTS; the recorded list is in
--     tests/test_boxscore_ingest.py). src/ingestion/boxscores.py COLUMN_MAP
--     simply did not request it, and now does.
--   * the Kaggle archive's PlayerStatistics export carries foulsPersonal on
--     304,395 of its 305,614 player-game rows (99.6%).
--
-- An earlier draft of this comment asserted that the league game log does not
-- report fouls. That was wrong, and it was wrong in the direction that
-- matters: it would have left the live path permanently foul-blind on a
-- claim about a payload nobody had looked at. The header list was in this
-- repository the whole time.
--
-- Existing rows stay NULL because what was not fetched cannot be recovered
-- from what was stored. Re-running either ingest fills the rows it covers.
--
-- A DEFAULT OF 0 WOULD BE A FABRICATION, and a consequential one: zero fouls
-- is a specific, clean, low-risk game, and backfilling it would teach the
-- minutes-risk feature that the entire history was foul-free. The feature
-- abstains on a null and reports nothing; it cannot abstain on a zero.
--
-- RANGE
--
-- Bounded to 0-6 because that is what a personal-foul count can be: six is
-- disqualification and there is no seventh. The bound is a guard against a
-- different column arriving under this name — a team total, or a count of
-- fouls drawn — rather than against a real outlier, and the feature layer
-- treats an out-of-range value as unknown rather than clipping it for the
-- same reason. NOT VALID so an existing table is not rewritten; the check
-- applies to every new and updated row from here on.

ALTER TABLE player_game_logs
    ADD COLUMN IF NOT EXISTS pf INTEGER;

ALTER TABLE player_game_logs
    DROP CONSTRAINT IF EXISTS ck_player_game_log_pf_range;
ALTER TABLE player_game_logs
    ADD CONSTRAINT ck_player_game_log_pf_range
    CHECK (pf IS NULL OR (pf >= 0 AND pf <= 6))
    NOT VALID;

COMMENT ON COLUMN player_game_logs.pf IS
    'The PLAYER''s personal fouls, 0-6. NULL where the source did not report '
    'them — never 0. Not team_game_stats.pf, which is a team total.';
