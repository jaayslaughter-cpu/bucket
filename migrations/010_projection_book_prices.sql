-- ============================================================================
-- PropIQ Analytics — the price a projection was made against
-- Migration 010: projections.odds_over_american / odds_under_american / line_source
-- Target: PostgreSQL 14+ / Supabase
-- Scope: NBA only. RESEARCH_ONLY — a captured price, not a wager.
-- ============================================================================
--
-- WHAT WAS BROKEN, WITH THE NUMBERS FROM 2026-11-09
--
--   PropLine: 19340 row(s) across 25 book(s) from 106 event(s).
--   EV gate: 17210 ready, 2130 abstained.
--   Board:   'n_consider_with_book_ev': 0, 'sources_used': []
--
-- Seventeen thousand two hundred and ten quotes passed the EV gate with
-- two-way American odds, and not one reached the board. The gate ran, the
-- de-vigging ran, and then the price was dropped on the floor: `projections`
-- stopped at `line, prob_over, prob_under, prob_push`, so every board row was
-- `basis: model_lean, ev: None` BY CONSTRUCTION rather than because no market
-- existed.
--
-- The effect was that "no positive-EV prop tonight" and "no price ever
-- reached the ranker" were indistinguishable in the output, and the second
-- was always what had happened.
--
-- WHY ON `projections` RATHER THAN JOINED AT BOARD TIME
--
-- The price and the projection are ONE FACT AT ONE MOMENT. `main.
-- assemble_projections` already resolves the board's spelling to the panel's
-- through `id_crosswalk.resolve_board_names` and attaches `line` from the
-- match; the odds are in that same matched row. Re-joining at board time
-- would mean a SECOND name matcher against a snapshot table that stores the
-- book's spelling -- the exact duplication `posted_line_for_market`'s
-- docstring refuses -- and it would attach whatever price existed when the
-- board was built rather than the one the projection was made against.
--
-- `prop_results` already stores `odds` and `source` per recorded prediction
-- for this reason. This makes `projections` consistent with it.
--
-- NULLABLE, NO DEFAULT, NO BACKFILL
--
-- NULL means "no two-way price was matched to this row", which is the honest
-- state for every row written before this migration and for every row where
-- the book posted one side or none. A zero or a 0.0 multiplier would read as
-- a price of even money, which is a number somebody could rank on.
--
-- A PICK'EM MULTIPLIER IS NOT STORED HERE. `is_pickem` rows are routed to
-- `quant.dfs_payouts`, not de-vigged, and `market_ev_gate` already refuses
-- them. Writing a multiplier into an American-odds column would make the two
-- indistinguishable downstream.

ALTER TABLE projections
    ADD COLUMN IF NOT EXISTS odds_over_american INTEGER;

ALTER TABLE projections
    ADD COLUMN IF NOT EXISTS odds_under_american INTEGER;

ALTER TABLE projections
    ADD COLUMN IF NOT EXISTS line_source TEXT;

COMMENT ON COLUMN projections.odds_over_american IS
    'American odds for the OVER on the matched prop line, as captured when '
    'this projection was assembled. NULL when no two-way price was matched. '
    'Never a pick''em multiplier — those are routed to quant.dfs_payouts.';

COMMENT ON COLUMN projections.odds_under_american IS
    'American odds for the UNDER on the matched prop line. NULL when no '
    'two-way price was matched.';

COMMENT ON COLUMN projections.line_source IS
    'Which book supplied the matched line and odds. NULL when no line was '
    'matched. Without it the board cannot say which feed priced a row, and '
    'sources_used reads as empty on a slate that was fully priced.';
