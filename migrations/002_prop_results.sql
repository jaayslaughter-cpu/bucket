-- ============================================================================
-- PropIQ Analytics — Win/Loss/Push settlement tracker
-- Migration 002: prop_results + performance views
-- Target: PostgreSQL 14+ / Supabase
-- Scope: NBA only.
-- ============================================================================
--
-- DESIGN NOTES
--
-- 1. outcome_status is a CHECK-constrained text column, not a Postgres ENUM.
--    ENUMs require ALTER TYPE to extend and lock the table; a CHECK is
--    trivially alterable and Supabase's REST layer serialises it more
--    predictably. 'VOID' is included alongside the four you specified —
--    a player who is a late scratch produces a voided prop, which is
--    genuinely distinct from a PUSH and must not be counted in the
--    win-rate denominator.
--
-- 2. PUSH IS ONLY POSSIBLE ON WHOLE-NUMBER LINES. A 25.5 line cannot tie.
--    is_whole_number_line is generated so the settlement audit can verify
--    that no PUSH was ever recorded against a half-point line — that
--    would indicate a float-comparison bug.
--
-- 3. odds is nullable ON PURPOSE. Pick'em boards (PrizePicks/Underdog/
--    Sleeper) publish a payout multiplier, not two-way American odds.
--    ROI is undefined without a price, so those rows keep odds NULL and
--    are excluded from ROI (but still counted in W-L-P).
--
-- 4. Money is NUMERIC, never FLOAT — binary floats silently drift on
--    accumulation, which is unacceptable for a P/L ledger.
-- ============================================================================

BEGIN;

CREATE TABLE IF NOT EXISTS prop_results (
    id                      BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    -- Linkage back to what was predicted
    projection_id           INTEGER REFERENCES projections(id) ON DELETE SET NULL,
    run_id                  VARCHAR(64),

    -- Identity (denormalised so results survive a projections purge)
    nba_game_id             VARCHAR(32)  NOT NULL,
    nba_player_id           VARCHAR(32),
    player_name             VARCHAR(128) NOT NULL,
    game_date               DATE         NOT NULL,
    market                  VARCHAR(32)  NOT NULL,   -- PTS / REB / AST / PRA / FG3M ...

    -- What we predicted
    predicted_line          NUMERIC(8,2) NOT NULL,
    predicted_side          VARCHAR(8)   NOT NULL,   -- 'OVER' | 'UNDER'
    model_projection        NUMERIC(8,2),
    prob_over               NUMERIC(6,5),

    -- Price. NULL for pick'em (multiplier, not two-way American odds).
    odds                    INTEGER,
    payout_multiplier       NUMERIC(6,3),
    source                  VARCHAR(32),
    is_pickem               BOOLEAN NOT NULL DEFAULT FALSE,

    -- What actually happened
    actual_result           NUMERIC(8,2),
    minutes_played          NUMERIC(6,2),
    did_not_play            BOOLEAN NOT NULL DEFAULT FALSE,

    outcome_status          VARCHAR(16) NOT NULL DEFAULT 'PENDING',

    -- Settlement ledger. NULL when odds are NULL (ROI undefined).
    stake_units             NUMERIC(10,4),
    profit_units            NUMERIC(10,4),

    -- Closing line value — a market-quality signal, NOT profit.
    closing_line            NUMERIC(8,2),
    closing_odds            INTEGER,
    clv_line_points         NUMERIC(8,2),
    clv_prob_points         NUMERIC(8,5),

    -- Provenance
    result_source           VARCHAR(32),
    settled_at_utc          TIMESTAMPTZ,
    raw_boxscore_json       JSONB,
    settlement_note         TEXT,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT ck_outcome_status CHECK (
        outcome_status IN ('WIN', 'LOSS', 'PUSH', 'PENDING', 'VOID')
    ),
    CONSTRAINT ck_predicted_side CHECK (predicted_side IN ('OVER', 'UNDER')),

    -- A settled row must carry a result; a PENDING row must not.
    CONSTRAINT ck_settled_has_result CHECK (
        (outcome_status = 'PENDING' AND actual_result IS NULL)
        OR (outcome_status IN ('VOID') )
        OR (outcome_status IN ('WIN','LOSS','PUSH') AND actual_result IS NOT NULL)
    ),

    -- A PUSH is arithmetically impossible unless the line is a whole number.
    CONSTRAINT ck_push_requires_whole_line CHECK (
        outcome_status <> 'PUSH' OR predicted_line = ROUND(predicted_line)
    ),

    CONSTRAINT uq_prop_result UNIQUE (nba_game_id, player_name, market, predicted_line, predicted_side, source)
);

CREATE INDEX IF NOT EXISTS ix_prop_results_pending
    ON prop_results (outcome_status, game_date)
    WHERE outcome_status = 'PENDING';

CREATE INDEX IF NOT EXISTS ix_prop_results_game    ON prop_results (nba_game_id);
CREATE INDEX IF NOT EXISTS ix_prop_results_player  ON prop_results (player_name, market);
CREATE INDEX IF NOT EXISTS ix_prop_results_date    ON prop_results (game_date);
CREATE INDEX IF NOT EXISTS ix_prop_results_settled ON prop_results (settled_at_utc);

-- keep updated_at honest
CREATE OR REPLACE FUNCTION touch_prop_results_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_prop_results_updated_at ON prop_results;
CREATE TRIGGER trg_prop_results_updated_at
    BEFORE UPDATE ON prop_results
    FOR EACH ROW EXECUTE FUNCTION touch_prop_results_updated_at();

-- ============================================================================
-- PERFORMANCE VIEWS
--
-- Two separate denominators, deliberately:
--   graded_n  = WIN + LOSS + PUSH   (everything that settled)
--   decided_n = WIN + LOSS          (pushes/voids excluded)
-- Strike rate uses decided_n. Reporting a win% that includes pushes in the
-- denominator understates performance; reporting one that hides the push
-- count entirely overstates certainty. Both numbers are exposed.
-- ============================================================================

CREATE OR REPLACE VIEW v_prop_performance_summary AS
SELECT
    COUNT(*) FILTER (WHERE outcome_status IN ('WIN','LOSS','PUSH'))        AS graded_n,
    COUNT(*) FILTER (WHERE outcome_status = 'WIN')                          AS wins,
    COUNT(*) FILTER (WHERE outcome_status = 'LOSS')                         AS losses,
    COUNT(*) FILTER (WHERE outcome_status = 'PUSH')                         AS pushes,
    COUNT(*) FILTER (WHERE outcome_status = 'VOID')                         AS voids,
    COUNT(*) FILTER (WHERE outcome_status = 'PENDING')                      AS pending,
    COUNT(*) FILTER (WHERE outcome_status IN ('WIN','LOSS'))                AS decided_n,

    -- Strike rate over DECIDED props only (pushes/voids excluded).
    CASE WHEN COUNT(*) FILTER (WHERE outcome_status IN ('WIN','LOSS')) > 0
         THEN ROUND(
             COUNT(*) FILTER (WHERE outcome_status = 'WIN')::NUMERIC
             / COUNT(*) FILTER (WHERE outcome_status IN ('WIN','LOSS')) * 100, 2)
    END AS strike_rate_pct,

    -- ROI over PRICED props only. Pick'em rows (odds IS NULL) are excluded
    -- because ROI is undefined without a payout price.
    COUNT(*) FILTER (WHERE odds IS NOT NULL
                       AND outcome_status IN ('WIN','LOSS','PUSH'))        AS priced_n,
    ROUND(SUM(stake_units)  FILTER (WHERE odds IS NOT NULL), 4)             AS staked_units,
    ROUND(SUM(profit_units) FILTER (WHERE odds IS NOT NULL), 4)             AS profit_units,
    CASE WHEN COALESCE(SUM(stake_units) FILTER (WHERE odds IS NOT NULL), 0) > 0
         THEN ROUND(
             SUM(profit_units) FILTER (WHERE odds IS NOT NULL)
             / SUM(stake_units) FILTER (WHERE odds IS NOT NULL) * 100, 2)
    END AS roi_pct,

    -- Unpriced (pick'em) rows: counted for W-L-P, excluded from ROI.
    COUNT(*) FILTER (WHERE odds IS NULL
                       AND outcome_status IN ('WIN','LOSS','PUSH'))        AS unpriced_graded_n,

    -- CLV is a market-quality signal, NOT profit. Reported separately.
    ROUND(AVG(clv_line_points) FILTER (WHERE clv_line_points IS NOT NULL), 4) AS avg_clv_line_points,
    ROUND(AVG(clv_prob_points) FILTER (WHERE clv_prob_points IS NOT NULL), 5) AS avg_clv_prob_points,
    CASE WHEN COUNT(*) FILTER (WHERE clv_prob_points IS NOT NULL) > 0
         THEN ROUND(
             COUNT(*) FILTER (WHERE clv_prob_points > 0)::NUMERIC
             / COUNT(*) FILTER (WHERE clv_prob_points IS NOT NULL) * 100, 2)
    END AS pct_positive_clv
FROM prop_results;

CREATE OR REPLACE VIEW v_prop_performance_by_market AS
SELECT
    market,
    COUNT(*) FILTER (WHERE outcome_status IN ('WIN','LOSS'))  AS decided_n,
    COUNT(*) FILTER (WHERE outcome_status = 'WIN')             AS wins,
    COUNT(*) FILTER (WHERE outcome_status = 'LOSS')            AS losses,
    COUNT(*) FILTER (WHERE outcome_status = 'PUSH')            AS pushes,
    CASE WHEN COUNT(*) FILTER (WHERE outcome_status IN ('WIN','LOSS')) > 0
         THEN ROUND(
             COUNT(*) FILTER (WHERE outcome_status = 'WIN')::NUMERIC
             / COUNT(*) FILTER (WHERE outcome_status IN ('WIN','LOSS')) * 100, 2)
    END AS strike_rate_pct,
    CASE WHEN COALESCE(SUM(stake_units) FILTER (WHERE odds IS NOT NULL), 0) > 0
         THEN ROUND(
             SUM(profit_units) FILTER (WHERE odds IS NOT NULL)
             / SUM(stake_units) FILTER (WHERE odds IS NOT NULL) * 100, 2)
    END AS roi_pct
FROM prop_results
GROUP BY market
ORDER BY decided_n DESC;

-- Audit view: surfaces settlement anomalies rather than hiding them.
CREATE OR REPLACE VIEW v_settlement_audit AS
SELECT
    'push_on_half_line'   AS issue,
    COUNT(*)              AS n
FROM prop_results
WHERE outcome_status = 'PUSH' AND predicted_line <> ROUND(predicted_line)
UNION ALL
SELECT 'settled_without_result', COUNT(*)
FROM prop_results
WHERE outcome_status IN ('WIN','LOSS','PUSH') AND actual_result IS NULL
UNION ALL
SELECT 'priced_without_profit', COUNT(*)
FROM prop_results
WHERE odds IS NOT NULL AND outcome_status IN ('WIN','LOSS') AND profit_units IS NULL
UNION ALL
SELECT 'stale_pending_over_48h', COUNT(*)
FROM prop_results
WHERE outcome_status = 'PENDING' AND game_date < (CURRENT_DATE - INTERVAL '2 days');

COMMIT;
