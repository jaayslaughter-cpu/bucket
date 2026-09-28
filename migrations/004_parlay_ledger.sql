-- ============================================================================
-- PropIQ Analytics — parlay ledger
-- Migration 004: parlay_tickets + parlay_legs
-- Target: PostgreSQL 14+ / Supabase
-- Scope: NBA only. RESEARCH_ONLY — this records what was predicted, never a
--        wager placed by this system.
-- ============================================================================
--
-- WHY THESE TABLES EXIST
--
-- The ledger lived in data/external/parlay_log/*.csv. `data/**` is gitignored
-- and a container filesystem is ephemeral, so every tracked ticket — its
-- at-bet-time joint probability and EV — was destroyed by the next redeploy.
-- Those two numbers are the only record of what was believed BEFORE a game,
-- and they are the one thing that cannot be recomputed afterwards.
--
-- DESIGN NOTES
--
-- 1. `record` is JSONB and holds the whole ParlayTicketRecord /
--    ParlayLegRecord as written. The schema lives in
--    src/quant/parlay_log.py and changes; mirroring forty columns here would
--    mean a field added there and forgotten here is dropped on every write,
--    silently. Storing the record whole makes a new field durable the day it
--    is added, and `schema_version` records the shape it was written in.
--
-- 2. The scalar columns are lifted OUT of the record, not instead of it, so
--    the ledger can be queried and joined without JSON path expressions.
--    They are a read convenience; `record` is the truth. A row whose record
--    lacks a promoted key simply carries NULL in that column.
--
-- 3. There is no DELETE path anywhere in the application for these tables.
--    Settlement UPDATEs a row in place; nothing removes one.
--
-- 4. The at-bet-time freeze — refusing to let a settlement rewrite the
--    probability or EV a ticket was logged with — is enforced in
--    ParlayLogStore, not here. A CHECK constraint cannot tell a settlement
--    write from a rewrite, since both arrive as an UPDATE of the same row.
-- ============================================================================

CREATE TABLE IF NOT EXISTS parlay_tickets (
    ticket_id       VARCHAR(64) PRIMARY KEY,
    slate_date      VARCHAR(16),
    created_at_utc  TIMESTAMPTZ,
    ticket_result   VARCHAR(16),
    n_legs          INTEGER,
    schema_version  VARCHAR(16),
    record          JSONB NOT NULL,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_parlay_tickets_slate_date  ON parlay_tickets (slate_date);
CREATE INDEX IF NOT EXISTS ix_parlay_tickets_created     ON parlay_tickets (created_at_utc);
CREATE INDEX IF NOT EXISTS ix_parlay_tickets_result      ON parlay_tickets (ticket_result);

CREATE TABLE IF NOT EXISTS parlay_legs (
    ticket_id       VARCHAR(64) NOT NULL,
    leg_id          VARCHAR(64) NOT NULL,
    slate_date      VARCHAR(16),
    game_id         VARCHAR(32),
    player_name     VARCHAR(128),
    market          VARCHAR(32),
    leg_result      VARCHAR(16),
    schema_version  VARCHAR(16),
    record          JSONB NOT NULL,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (ticket_id, leg_id)
);

CREATE INDEX IF NOT EXISTS ix_parlay_legs_ticket      ON parlay_legs (ticket_id);
CREATE INDEX IF NOT EXISTS ix_parlay_legs_slate_date  ON parlay_legs (slate_date);
CREATE INDEX IF NOT EXISTS ix_parlay_legs_game        ON parlay_legs (game_id);
CREATE INDEX IF NOT EXISTS ix_parlay_legs_player      ON parlay_legs (player_name);
CREATE INDEX IF NOT EXISTS ix_parlay_legs_market      ON parlay_legs (market);
CREATE INDEX IF NOT EXISTS ix_parlay_legs_result      ON parlay_legs (leg_result);
