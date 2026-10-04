-- ============================================================================
-- PropIQ Analytics — the under and the push on a projection
-- Migration 005: projections.prob_under + projections.prob_push
-- Target: PostgreSQL 14+ / Supabase
-- Scope: NBA only. RESEARCH_ONLY — these are predicted probabilities, not
--        wagers, and no stake is recorded anywhere in this table.
-- ============================================================================
--
-- WHY THESE COLUMNS EXIST
--
-- `projections` stored only `prob_over`. On a HALF line that loses nothing:
-- a push is impossible, so the under is exactly 1 - P(over) and can be
-- recomputed at any time.
--
-- On a WHOLE line it is not. Push has real mass there, so 1 - P(over) is the
-- probability of "under OR push" and not the probability of under. A reader
-- holding only `prob_over` cannot tell which kind of line a row describes, so
-- every whole-line under computed after the fact was wrong, and the push mass
-- was unrecoverable the moment the row was written.
--
-- `src/models/residuals.py::over_under_push_from_dispersion` has computed all
-- three legs all along, and five call sites across four model modules use it.
-- The number existed; the column to put it in did not.
--
-- WHAT IS AND IS NOT WRITTEN
--
-- NULL is a real answer here and the common one. `main.assemble_projections`
-- routes every row through `paper_research.resolve_two_way_model_probs`, which:
--
--   * half or unknown line  -> push = 0, under = 1 - over          (both set)
--   * whole line, push mass supplied -> the supplied figures       (both set)
--   * whole line, no push mass       -> BOTH NULL plus a reason    (refused)
--
-- The third case is the point. The binary classifier behind `prob_over` does
-- not model a push, so on a whole line there is nothing honest to write, and
-- the refusal reason lands in `projections.notes`.
--
-- Backfill is deliberately NOT attempted. Existing rows do not record whether
-- their line was whole, and deriving an under for the ones that were would
-- write the exact error these columns exist to prevent.
-- ============================================================================

ALTER TABLE projections
    ADD COLUMN IF NOT EXISTS prob_under DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS prob_push  DOUBLE PRECISION;

-- A probability is in [0, 1] or absent. NOT VALID so the constraint applies to
-- new and updated rows without a full table scan on a large existing table;
-- run `VALIDATE CONSTRAINT` separately once the backlog is known good.
ALTER TABLE projections
    DROP CONSTRAINT IF EXISTS ck_projection_prob_under_range;
ALTER TABLE projections
    ADD CONSTRAINT ck_projection_prob_under_range
    CHECK (prob_under IS NULL OR (prob_under >= 0.0 AND prob_under <= 1.0))
    NOT VALID;

ALTER TABLE projections
    DROP CONSTRAINT IF EXISTS ck_projection_prob_push_range;
ALTER TABLE projections
    ADD CONSTRAINT ck_projection_prob_push_range
    CHECK (prob_push IS NULL OR (prob_push >= 0.0 AND prob_push <= 1.0))
    NOT VALID;

-- The three legs are one distribution: when all three are present they sum to
-- 1. A tolerance of 1e-6 absorbs float round-trips without admitting a row
-- whose legs disagree. Rows with a NULL leg are exempt, which is the refusal
-- path above rather than a gap in the check.
ALTER TABLE projections
    DROP CONSTRAINT IF EXISTS ck_projection_probs_sum_to_one;
ALTER TABLE projections
    ADD CONSTRAINT ck_projection_probs_sum_to_one
    CHECK (
        prob_over IS NULL
        OR prob_under IS NULL
        OR prob_push IS NULL
        OR abs((prob_over + prob_under + prob_push) - 1.0) <= 1e-6
    )
    NOT VALID;

COMMENT ON COLUMN projections.prob_under IS
    'P(under). NULL on a whole line with no push mass — never 1 - prob_over.';
COMMENT ON COLUMN projections.prob_push IS
    'P(push). 0 on a half line; NULL on a whole line the model cannot split.';
