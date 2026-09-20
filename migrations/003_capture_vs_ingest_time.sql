-- ============================================================================
-- PropIQ Analytics — separate observation time from ingest time
-- Migration 003: prop_line_snapshots + game_market_lines
-- Target: PostgreSQL 14+ / Supabase
-- Scope: NBA only.
-- ============================================================================
--
-- THE PROBLEM
--
-- captured_at_utc defaulted to now(), so every row claimed it was observed
-- at the moment it was written. For a workbook of last season's lines
-- loaded today, that is false by months.
--
-- It is not a cosmetic error. Line movement and CLV are both measured
-- against this timestamp, and a fabricated one does not read as missing
-- data — it reads as a line that never moved. A backfilled season would
-- have produced a clean, entirely artificial "no movement" signal.
--
-- THE SPLIT
--
--   captured_at_utc  — when the SOURCE observed the line. Nullable, no
--                      default. NULL means the source did not report it,
--                      which is a fact we can act on.
--   ingested_at_utc  — when WE wrote the row. Always known, and never a
--                      claim about the book.
--
-- EXISTING ROWS
--
-- Their captured_at_utc holds an ingest time wearing an observation
-- label. This migration moves that value to ingested_at_utc, where it is
-- true, and nulls captured_at_utc, which is honest: for those rows the
-- observation time was never recorded. Analyses that need an observation
-- time will now correctly abstain on them rather than trusting a
-- fabricated one.
--
-- This is safe to re-run.
-- ============================================================================

BEGIN;

-- --- prop_line_snapshots ----------------------------------------------------

ALTER TABLE prop_line_snapshots
    ADD COLUMN IF NOT EXISTS ingested_at_utc TIMESTAMPTZ;

-- Preserve what the old column actually held: an ingest time.
UPDATE prop_line_snapshots
   SET ingested_at_utc = captured_at_utc
 WHERE ingested_at_utc IS NULL
   AND captured_at_utc IS NOT NULL;

UPDATE prop_line_snapshots
   SET ingested_at_utc = now()
 WHERE ingested_at_utc IS NULL;

ALTER TABLE prop_line_snapshots
    ALTER COLUMN ingested_at_utc SET NOT NULL,
    ALTER COLUMN ingested_at_utc SET DEFAULT now();

-- Stop the old column asserting an observation time it never had.
ALTER TABLE prop_line_snapshots
    ALTER COLUMN captured_at_utc DROP DEFAULT,
    ALTER COLUMN captured_at_utc DROP NOT NULL;

UPDATE prop_line_snapshots
   SET captured_at_utc = NULL
 WHERE captured_at_utc = ingested_at_utc;

CREATE INDEX IF NOT EXISTS ix_prop_line_snapshots_ingested_at_utc
    ON prop_line_snapshots (ingested_at_utc);

-- --- game_market_lines ------------------------------------------------------

ALTER TABLE game_market_lines
    ADD COLUMN IF NOT EXISTS ingested_at_utc TIMESTAMPTZ;

UPDATE game_market_lines
   SET ingested_at_utc = captured_at_utc
 WHERE ingested_at_utc IS NULL
   AND captured_at_utc IS NOT NULL;

UPDATE game_market_lines
   SET ingested_at_utc = now()
 WHERE ingested_at_utc IS NULL;

ALTER TABLE game_market_lines
    ALTER COLUMN ingested_at_utc SET NOT NULL,
    ALTER COLUMN ingested_at_utc SET DEFAULT now();

ALTER TABLE game_market_lines
    ALTER COLUMN captured_at_utc DROP DEFAULT,
    ALTER COLUMN captured_at_utc DROP NOT NULL;

UPDATE game_market_lines
   SET captured_at_utc = NULL
 WHERE captured_at_utc = ingested_at_utc;

COMMIT;

COMMENT ON COLUMN prop_line_snapshots.captured_at_utc IS
    'When the SOURCE observed this line. NULL = not reported. Never defaulted.';
COMMENT ON COLUMN prop_line_snapshots.ingested_at_utc IS
    'When this row was written by PropIQ. Not an observation time.';
COMMENT ON COLUMN game_market_lines.captured_at_utc IS
    'When the SOURCE observed this line. NULL = not reported. Never defaulted.';
COMMENT ON COLUMN game_market_lines.ingested_at_utc IS
    'When this row was written by PropIQ. Not an observation time.';
