-- ============================================================================
-- PropIQ Analytics — the position a player STARTED at
-- Migration 008: player_game_logs.starting_position
-- Target: PostgreSQL 14+ / Supabase
-- Scope: NBA only. RESEARCH_ONLY — a box-score designation, not a wager.
-- ============================================================================
--
-- WHY THIS COLUMN EXISTS
--
-- src/features/dvp.py splits opponent defence by the position it is defending
-- — the only feature in this project that varies by WHO THE PLAYER IS as well
-- as by whom he faces. It needs a starting position, the Kaggle archive has
-- one, and `player_game_logs` did not. So the layer was built, leakage-audited
-- and measured on history while abstaining on every row of a LIVE panel:
-- `attach_dvp_features` found no STARTING_POSITION column, logged that it was
-- skipping, and added nothing.
--
-- The measurement is why this is worth a column rather than a note. REB, four
-- chronological folds, `--layer dvp --wire-under-test`: better on every fold
-- for four of five models, at 1.3-2.5x the fold spread, and DVP_REB_INDEX_L10
-- sits at 0.025 against DEF_RATING_L10 — very nearly orthogonal to the
-- team-level number that hands every player in a game the same value. Numbers
-- and caveats: docs/fouls_and_dvp.md section 3a.
--
-- WHAT GOES IN IT
--
-- One of 'G', 'F', 'C', or NULL. Exactly the vocabulary and exactly the
-- distribution the archive uses: of 214,381 archive panel rows, 35,594 are G,
-- 35,593 F, 17,799 C and 125,395 NULL — five starters per team-game in a
-- 2:2:1 mix, and a NULL for everyone who came off the bench.
--
-- NULL THEREFORE MEANS TWO THINGS, and they are not distinguished here on
-- purpose, because the feature layer treats them the same way:
--
--   * the player did not start. The common case.
--   * the source did not report a position for this row.
--
-- WHY NULLABLE, WITH NO DEFAULT AND NO BACKFILL
--
-- Existing rows stay NULL because what was never fetched cannot be recovered
-- from what was stored. The writer is src/ingestion/starting_positions.py,
-- which reads the NBA's own traditional box score — `position` on
-- boxscoretraditionalv3, `START_POSITION` on the deprecated v2 — one call per
-- game, cached to parquet, and then
-- repository.update_starting_positions writes it onto rows that already
-- exist. It is a SEPARATE endpoint from the league game log that fills the
-- rest of this table, which is why it is a separate pass and not another
-- column in the main upsert's payload.
--
-- A DEFAULT WOULD BE A FABRICATION. There is no neutral position: writing 'G'
-- would make every unfetched row a guard, and writing '' would make every
-- player in history a bench player — the second is worse, because DvP cannot
-- abstain on a value that parses.
--
-- RANGE
--
-- Bounded to the three buckets src/features/dvp.py recognises. The archive and
-- both endpoint versions emit exactly these, so the check is a guard against a
-- DIFFERENT column arriving under this name — a listed position like 'PG' or
-- 'F-C', which is a different quantity from a starting designation and would
-- change what POS_BUCKET means for every row downstream. The ingest normalises
-- through dvp.normalise_bucket before it gets here and refuses a payload whose
-- filled-position count per team-game is not five, which is the real gate;
-- this constraint is the second line. NOT VALID so an existing table is not
-- rewritten; the check applies to every new and updated row from here on.

ALTER TABLE player_game_logs
    ADD COLUMN IF NOT EXISTS starting_position VARCHAR(1);

ALTER TABLE player_game_logs
    DROP CONSTRAINT IF EXISTS ck_player_game_log_starting_position;
ALTER TABLE player_game_logs
    ADD CONSTRAINT ck_player_game_log_starting_position
    CHECK (starting_position IS NULL OR starting_position IN ('G', 'F', 'C'))
    NOT VALID;

COMMENT ON COLUMN player_game_logs.starting_position IS
    'The position the player STARTED at: G, F, C, or NULL for a bench '
    'appearance or an unreported row. Never '''' — an empty string parses and '
    'would make a bench player look like a reported position. Written by '
    'src/ingestion/starting_positions.py from boxscoretraditionalv3.position '
    '(v2: START_POSITION), normalised through dvp.normalise_bucket.';
