-- ============================================================================
-- PropIQ Analytics — why a row's market status is what it is
-- Migration 007: projections.market_status_reason
-- Target: PostgreSQL 14+ / Supabase
-- Scope: NBA only. RESEARCH_ONLY — a gate verdict, not a wager.
-- ============================================================================
--
-- WHAT CHANGED ABOUT market_status ITSELF
--
-- It is now PER ROW. `main.evaluate_ev_gate` loops every captured prop line,
-- asks `quant.contracts.market_ev_gate` about each one, and returned COUNTS
-- plus a single collapsed status. `assemble_projections` wrote that single
-- status onto every row, and this table stores it per row. A slate with one
-- priced prop and two hundred and ninety-nine unpriced ones therefore
-- labelled all three hundred READY_FOR_EVALUATION — a claim about a different
-- row, in the only form this column is ever read. The gate reads ONE market
-- context, so a per-row answer is the only kind it has; the aggregate is a
-- summary for a log line.
--
-- No migration is needed for that part: the column already exists and only
-- its contents were wrong. This migration adds the half that was missing.
--
-- WHY THE REASON NEEDS ITS OWN COLUMN
--
-- DATA_NOT_AVAILABLE has two causes and they are not interchangeable:
--
--   * NO LINE reached this row at all — nobody posted a price for this
--     player and market, so there was nothing for the gate to evaluate.
--   * A LINE WAS POSTED and the gate refused it — one side of the price was
--     missing, the line was not finite, the status was not VALID, or it is a
--     pick'em multiplier, which is not a two-way price and is ROUTED to
--     `quant.dfs_payouts` rather than de-vigged here.
--
-- "Nobody priced this player" and "a price was posted and could not be
-- de-vigged" lead a reader to opposite conclusions about whether the market
-- exists. Collapsing them into one bare status is how somebody concludes a
-- market is unavailable when it is merely unpriceable, or the reverse.
--
-- WHY NOT `notes`
--
-- `notes` already carries the under/push refusal from
-- `paper_research.resolve_two_way_model_probs`. Two unrelated reasons in one
-- free-text column cannot be read apart by anything downstream, and appending
-- would make both harder to parse than either is alone.
--
-- NULLABLE, NO DEFAULT, NO BACKFILL
--
-- NULL means "this row predates the per-row verdict". The old rows' statuses
-- were a slate aggregate, so there is no per-row reason to recover for them —
-- inventing one would attach a specific explanation to a value that never had
-- one. Rows written from here on carry both.

ALTER TABLE projections
    ADD COLUMN IF NOT EXISTS market_status_reason TEXT;

COMMENT ON COLUMN projections.market_status_reason IS
    'Why market_status is what it is, per row. Distinguishes "no line reached '
    'this row" from "a line was posted and the gate refused it" — different '
    'facts with opposite implications. NULL on rows written before the '
    'per-row verdict existed.';

COMMENT ON COLUMN projections.market_status IS
    'This ROW''s EV-gate verdict, from this row''s own prop line. Was a '
    'slate-wide aggregate written onto every row until migration 007.';
