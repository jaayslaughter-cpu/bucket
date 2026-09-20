"""
src/db/models.py — PostgreSQL / Supabase schema for PropIQ Analytics.

NBA ONLY. No NCAA/CBB tables, columns, or enums — scope is deliberately
narrow per the current project decision.

Design notes:
- All timestamps are timezone-aware UTC (``DateTime(timezone=True)``).
  Display conversion to America/Los_Angeles happens at the UI layer only.
- ``market_snapshots`` carries a ``status`` column mirroring
  ``src.quant.contracts.MarketContext.status`` so the RESEARCH_ONLY
  abstention gate survives the round-trip through the database — a row
  with status ``DATA_NOT_AVAILABLE`` must never be silently treated as a
  usable price.
- ``raw_json`` columns use JSONB so nothing from a source is discarded.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Team(Base):
    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    nba_team_id: Mapped[str | None] = mapped_column(String(16), unique=True, index=True)
    abbreviation: Mapped[str] = mapped_column(String(8), index=True)
    full_name: Mapped[str] = mapped_column(String(64))
    conference: Mapped[str | None] = mapped_column(String(8))
    division: Mapped[str | None] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Player(Base):
    __tablename__ = "players"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    nba_player_id: Mapped[str | None] = mapped_column(String(32), unique=True, index=True)
    full_name: Mapped[str] = mapped_column(String(128), index=True)
    team_abbreviation: Mapped[str | None] = mapped_column(String(8), index=True)
    position: Mapped[str | None] = mapped_column(String(16))
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class Game(Base):
    __tablename__ = "games"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    nba_game_id: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    game_date: Mapped[datetime] = mapped_column(Date, index=True)
    season: Mapped[str] = mapped_column(String(16), index=True)  # e.g. "2025-26"
    home_team_abbr: Mapped[str] = mapped_column(String(8), index=True)
    away_team_abbr: Mapped[str] = mapped_column(String(8), index=True)
    tipoff_utc: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(32), default="scheduled")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class TeamGameStat(Base):
    """
    One row per team per game. Populated from BigDataBall exports
    (licensed) and/or PBP-derived box scores.
    """

    __tablename__ = "team_game_stats"
    __table_args__ = (UniqueConstraint("nba_game_id", "team_abbr", name="uq_team_game"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    nba_game_id: Mapped[str] = mapped_column(String(32), index=True)
    game_date: Mapped[datetime] = mapped_column(Date, index=True)
    team_abbr: Mapped[str] = mapped_column(String(8), index=True)
    opponent_abbr: Mapped[str | None] = mapped_column(String(8), index=True)
    is_home: Mapped[bool] = mapped_column(Boolean, default=False)
    # Neutral-site games (NBA Cup final, global games) have is_home=False
    # for BOTH teams. This flag distinguishes "neutral" from "away" so
    # home-court and fatigue logic don't treat one as the other.
    is_neutral_site: Mapped[bool] = mapped_column(Boolean, default=False)

    points: Mapped[int | None] = mapped_column(Integer)
    fg: Mapped[int | None] = mapped_column(Integer)
    fga: Mapped[int | None] = mapped_column(Integer)
    fg3: Mapped[int | None] = mapped_column(Integer)
    fg3a: Mapped[int | None] = mapped_column(Integer)
    ft: Mapped[int | None] = mapped_column(Integer)
    fta: Mapped[int | None] = mapped_column(Integer)
    oreb: Mapped[int | None] = mapped_column(Integer)
    dreb: Mapped[int | None] = mapped_column(Integer)
    reb: Mapped[int | None] = mapped_column(Integer)
    ast: Mapped[int | None] = mapped_column(Integer)
    stl: Mapped[int | None] = mapped_column(Integer)
    blk: Mapped[int | None] = mapped_column(Integer)
    tov: Mapped[int | None] = mapped_column(Integer)
    pf: Mapped[int | None] = mapped_column(Integer)

    poss: Mapped[float | None] = mapped_column(Float)
    pace: Mapped[float | None] = mapped_column(Float)
    off_eff: Mapped[float | None] = mapped_column(Float)
    def_eff: Mapped[float | None] = mapped_column(Float)
    rest_days: Mapped[str | None] = mapped_column(String(8))

    source: Mapped[str] = mapped_column(String(32), default="bigdataball")
    raw_json: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class GameMarketLine(Base):
    """
    Opening/closing spread, total, moneyline per team-game.

    This is the table that unlocks EV/CLV: PropIQ's quant layer abstains
    (MarketContext.status = DATA_NOT_AVAILABLE) until verified,
    timestamped market data exists. BigDataBall exports are a licensed,
    timestamped source — rows here carry provenance so the gate can
    distinguish them from an invented number.
    """

    __tablename__ = "game_market_lines"
    __table_args__ = (UniqueConstraint("nba_game_id", "team_abbr", "source", name="uq_game_market"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    nba_game_id: Mapped[str] = mapped_column(String(32), index=True)
    game_date: Mapped[datetime] = mapped_column(Date, index=True)
    team_abbr: Mapped[str] = mapped_column(String(8), index=True)

    opening_spread: Mapped[float | None] = mapped_column(Float)
    opening_total: Mapped[float | None] = mapped_column(Float)
    opening_odds_raw: Mapped[str | None] = mapped_column(String(64))
    closing_spread: Mapped[float | None] = mapped_column(Float)
    closing_total: Mapped[float | None] = mapped_column(Float)
    closing_odds_raw: Mapped[str | None] = mapped_column(String(64))
    moneyline: Mapped[str | None] = mapped_column(String(32))
    halftime_raw: Mapped[str | None] = mapped_column(String(64))

    line_movement_1: Mapped[str | None] = mapped_column(String(64))
    line_movement_2: Mapped[str | None] = mapped_column(String(64))
    line_movement_3: Mapped[str | None] = mapped_column(String(64))

    source: Mapped[str] = mapped_column(String(32), default="bigdataball")
    status: Mapped[str] = mapped_column(String(32), default="VALID")
    raw_json: Mapped[dict | None] = mapped_column(JSONB)

    # Same split as prop_line_snapshots: observation time is the source's to
    # report, ingest time is ours. A workbook loaded months later must not
    # claim its lines were seen today.
    captured_at_utc: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    ingested_at_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )


class PlayerGameLog(Base):
    """Player box-score panel — the input to the feature builder."""

    __tablename__ = "player_game_logs"
    __table_args__ = (UniqueConstraint("nba_game_id", "nba_player_id", name="uq_player_game"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    nba_game_id: Mapped[str] = mapped_column(String(32), index=True)
    nba_player_id: Mapped[str] = mapped_column(String(32), index=True)
    player_name: Mapped[str] = mapped_column(String(128), index=True)
    game_date: Mapped[datetime] = mapped_column(Date, index=True)
    season: Mapped[str] = mapped_column(String(16), index=True)
    team_abbr: Mapped[str | None] = mapped_column(String(8), index=True)
    opponent_abbr: Mapped[str | None] = mapped_column(String(8), index=True)
    is_home: Mapped[bool] = mapped_column(Boolean, default=False)
    # Must be propagated from team_game_stats: fatigue_logic applies the
    # altitude tax on `IS_HOME == False`, which would wrongly tax BOTH
    # teams at a neutral-site game in Denver/Utah. See
    # repository.load_player_panel, which neutralises IS_HOME for these.
    is_neutral_site: Mapped[bool] = mapped_column(Boolean, default=False)

    minutes: Mapped[float | None] = mapped_column(Float)
    pts: Mapped[int | None] = mapped_column(Integer)
    reb: Mapped[int | None] = mapped_column(Integer)
    ast: Mapped[int | None] = mapped_column(Integer)
    fg3m: Mapped[int | None] = mapped_column(Integer)
    stl: Mapped[int | None] = mapped_column(Integer)
    blk: Mapped[int | None] = mapped_column(Integer)
    tov: Mapped[int | None] = mapped_column(Integer)

    source: Mapped[str] = mapped_column(String(32))
    raw_json: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class PropLineSnapshot(Base):
    """
    Timestamped pick'em / sportsbook prop lines (research capture only).

    ``status`` mirrors MarketContext: VALID vs DATA_NOT_AVAILABLE. A
    pick'em multiplier is NOT a sportsbook two-way price — ``is_pickem``
    keeps that distinction explicit so downstream EV code can refuse to
    treat them interchangeably.
    """

    __tablename__ = "prop_line_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source: Mapped[str] = mapped_column(String(32), index=True)
    is_pickem: Mapped[bool] = mapped_column(Boolean, default=True)

    # When the line was OBSERVED AT THE SOURCE. Nullable with no default on
    # purpose: defaulting it to now() stamped every backfilled row with its
    # ingest time and called that an observation time. Line movement and CLV
    # are both measured against this field, so a fabricated value would not
    # look like missing data — it would look like a line that never moved.
    # NULL means the source did not report it.
    captured_at_utc: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), index=True, nullable=True
    )
    # When THIS ROW was written. Always known, never a claim about the book.
    ingested_at_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), index=True, default=utcnow
    )

    player_name: Mapped[str] = mapped_column(String(128), index=True)
    nba_player_id: Mapped[str | None] = mapped_column(String(32), index=True)
    market: Mapped[str] = mapped_column(String(32), index=True)  # PTS / REB / AST / PRA / FG3M ...
    line: Mapped[float | None] = mapped_column(Float)
    over_odds_american: Mapped[int | None] = mapped_column(Integer)
    under_odds_american: Mapped[int | None] = mapped_column(Integer)
    payout_multiplier: Mapped[float | None] = mapped_column(Float)

    game_date: Mapped[datetime | None] = mapped_column(Date, index=True)
    nba_game_id: Mapped[str | None] = mapped_column(String(32), index=True)
    status: Mapped[str] = mapped_column(String(32), default="VALID")
    raw_json: Mapped[dict | None] = mapped_column(JSONB)


class Projection(Base):
    """Model output per player/market/game — the pipeline's deliverable."""

    __tablename__ = "projections"
    # Keyed on the projection's natural identity, NOT on run_id. run_id is a
    # fresh UUID per execution, so including it meant a re-run of the same
    # slate never conflicted and simply accumulated a second set of rows.
    # run_id is still stored, as provenance for which run last wrote each row;
    # the per-run audit trail lives in pipeline_runs.
    __table_args__ = (
        UniqueConstraint("nba_game_id", "player_name", "market", name="uq_projection"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    nba_player_id: Mapped[str | None] = mapped_column(String(32), index=True)
    player_name: Mapped[str] = mapped_column(String(128), index=True)
    nba_game_id: Mapped[str | None] = mapped_column(String(32), index=True)
    game_date: Mapped[datetime | None] = mapped_column(Date, index=True)
    market: Mapped[str] = mapped_column(String(32), index=True)

    baseline_projection: Mapped[float | None] = mapped_column(Float)
    fatigue_multiplier: Mapped[float | None] = mapped_column(Float)
    fatigue_notes: Mapped[str | None] = mapped_column(String(128))
    final_projection: Mapped[float | None] = mapped_column(Float)

    line: Mapped[float | None] = mapped_column(Float)
    prob_over: Mapped[float | None] = mapped_column(Float)
    model_version: Mapped[str | None] = mapped_column(String(64))

    market_status: Mapped[str] = mapped_column(String(32), default="DATA_NOT_AVAILABLE")
    ev_per_dollar: Mapped[float | None] = mapped_column(Float)
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class PipelineRun(Base):
    """Audit trail — one row per main.py execution."""

    __tablename__ = "pipeline_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    started_at_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at_utc: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(32), default="running")
    stage_summary: Mapped[dict | None] = mapped_column(JSONB)
    error_summary: Mapped[str | None] = mapped_column(Text)


class PropResult(Base):
    """
    Settlement ledger — one row per graded prop.

    Mirrors migrations/002_prop_results.sql. Key design points:

    - `outcome_status` is CHECK-constrained to WIN/LOSS/PUSH/PENDING/VOID.
      VOID exists because a DNP/scratch is not a losing bet; counting it
      as a LOSS would understate the model's real strike rate.
    - `odds` is nullable ON PURPOSE. Pick'em boards publish a payout
      multiplier, not two-way American odds, so ROI is undefined for
      those rows — they still count toward W-L-P.
    - Money columns are Numeric, never Float: binary floats drift on
      accumulation, which is unacceptable in a P/L ledger (verified:
      1,000 summed -110 payouts diverge from exact by ~2e-12 and grow).
    - CLV is stored but NEVER summed into profit — it is a market-quality
      signal, not money.
    """

    __tablename__ = "prop_results"
    __table_args__ = (
        UniqueConstraint(
            "nba_game_id", "player_name", "market", "predicted_line",
            "predicted_side", "source", name="uq_prop_result",
        ),
        CheckConstraint(
            "outcome_status IN ('WIN','LOSS','PUSH','PENDING','VOID')",
            name="ck_outcome_status",
        ),
        CheckConstraint("predicted_side IN ('OVER','UNDER')", name="ck_predicted_side"),
        # Mirrors ck_settled_has_result in migrations/002_prop_results.sql.
        # Without it here, a table created by Base.metadata.create_all()
        # accepts graded rows carrying no actual_result, which the metrics
        # views then count as settled.
        CheckConstraint(
            "(outcome_status = 'PENDING' AND actual_result IS NULL)"
            " OR (outcome_status = 'VOID')"
            " OR (outcome_status IN ('WIN','LOSS','PUSH') AND actual_result IS NOT NULL)",
            name="ck_settled_has_result",
        ),
        CheckConstraint(
            "outcome_status <> 'PUSH' OR predicted_line = ROUND(predicted_line)",
            name="ck_push_requires_whole_line",
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    projection_id: Mapped[int | None] = mapped_column(ForeignKey("projections.id", ondelete="SET NULL"))
    run_id: Mapped[str | None] = mapped_column(String(64), index=True)

    nba_game_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    nba_player_id: Mapped[str | None] = mapped_column(String(32), index=True)
    player_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    game_date: Mapped[datetime] = mapped_column(Date, nullable=False, index=True)
    market: Mapped[str] = mapped_column(String(32), nullable=False, index=True)

    predicted_line: Mapped[Decimal] = mapped_column(Numeric(8, 2), nullable=False)
    predicted_side: Mapped[str] = mapped_column(String(8), nullable=False)
    model_projection: Mapped[Decimal | None] = mapped_column(Numeric(8, 2))
    prob_over: Mapped[Decimal | None] = mapped_column(Numeric(6, 5))

    odds: Mapped[int | None] = mapped_column(Integer)
    payout_multiplier: Mapped[Decimal | None] = mapped_column(Numeric(6, 3))
    source: Mapped[str | None] = mapped_column(String(32), index=True)
    is_pickem: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    actual_result: Mapped[Decimal | None] = mapped_column(Numeric(8, 2))
    minutes_played: Mapped[Decimal | None] = mapped_column(Numeric(6, 2))
    did_not_play: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    outcome_status: Mapped[str] = mapped_column(String(16), nullable=False, default="PENDING", index=True)

    stake_units: Mapped[Decimal | None] = mapped_column(Numeric(10, 4))
    profit_units: Mapped[Decimal | None] = mapped_column(Numeric(10, 4))

    closing_line: Mapped[Decimal | None] = mapped_column(Numeric(8, 2))
    closing_odds: Mapped[int | None] = mapped_column(Integer)
    clv_line_points: Mapped[Decimal | None] = mapped_column(Numeric(8, 2))
    clv_prob_points: Mapped[Decimal | None] = mapped_column(Numeric(8, 5))

    result_source: Mapped[str | None] = mapped_column(String(32))
    settled_at_utc: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    raw_boxscore_json: Mapped[dict | None] = mapped_column(JSONB)
    settlement_note: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
