"""Historical bet lifecycle store for EV/CLV research retraining.

Persists paper/research decisions only — no live wager placement.
Default: Parquet under ``data/external/market_store/`` + optional SQLite.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, get_args
from uuid import uuid4

import pandas as pd
from pydantic import BaseModel, Field

from src.quant.ev_engine import ClvResult, EvEvaluation

BetResult = Literal["WIN", "LOSS", "PUSH", "PENDING", "VOID"]
MarketType = Literal["player_prop", "moneyline", "spread", "total", "unknown"]


class BetLifecycleRecord(BaseModel):
    """Full research bet lifecycle row for ML retraining / post-mortems."""

    bet_id: str = Field(default_factory=lambda: uuid4().hex[:16])
    created_at_utc: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    settled_at_utc: datetime | None = None
    game_id: str
    player_id: str | None = None
    player_name: str | None = None
    market_type: MarketType = "player_prop"
    market_id: str | None = None
    prop_stat: str | None = None
    line: float | None = None
    bet_side: str
    bookmaker: str | None = None
    opening_odds_american: int | None = None
    taken_odds_american: int
    closing_odds_american: int | None = None
    closing_odds_other_american: int | None = None
    model_prob: float
    fair_prob_at_bet: float | None = None
    ev_at_bet_time: float | None = None
    clv: float | None = None
    actual_outcome: float | None = None
    stat_result: float | None = None
    bet_result: BetResult = "PENDING"
    profit_loss: float | None = None
    unit_stake: float = 1.0
    notes: str = "RESEARCH_ONLY"
    season: str | None = None
    source: str = "oddspapi"
    # Wave 5a research tags for pocket ROI (optional; never used for Kelly)
    confidence_tier: str | None = None
    edge_letter_grade: str | None = None
    model_name: str | None = None


class HistoricalStoreConfig(BaseModel):
    root: Path = Field(
        default_factory=lambda: Path("data/external/market_store")
    )
    parquet_name: str = "bet_lifecycle.parquet"
    csv_name: str = "bet_lifecycle.csv"
    sqlite_name: str = "bet_lifecycle.sqlite"
    use_sqlite: bool = True
    prefer_parquet: bool = True


class HistoricalStore:
    """Append-only research store with post-game auto-grading."""

    def __init__(self, config: HistoricalStoreConfig | None = None) -> None:
        self.config = config or HistoricalStoreConfig()
        self.config.root.mkdir(parents=True, exist_ok=True)
        self._parquet_ok = False
        if self.config.prefer_parquet:
            try:
                import pyarrow  # noqa: F401

                self._parquet_ok = True
            except ImportError:
                self._parquet_ok = False
        if self.config.use_sqlite:
            self._init_sqlite()

    @property
    def parquet_path(self) -> Path:
        return self.config.root / self.config.parquet_name

    @property
    def csv_path(self) -> Path:
        return self.config.root / self.config.csv_name

    @property
    def sqlite_path(self) -> Path:
        return self.config.root / self.config.sqlite_name

    def _init_sqlite(self) -> None:
        with sqlite3.connect(self.sqlite_path) as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS bet_lifecycle (
                    bet_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    bet_result TEXT,
                    created_at_utc TEXT,
                    settled_at_utc TEXT
                )
                """
            )

    def _table_path_write(self, df: pd.DataFrame) -> None:
        if self._parquet_ok:
            df.to_parquet(self.parquet_path, index=False)
        else:
            df.to_csv(self.csv_path, index=False)

    @staticmethod
    def _string_columns() -> set[str]:
        """Field names the record declares as text.

        Derived from the model rather than listed, so a new string field
        cannot quietly miss the dtype pin below.
        """
        out: set[str] = set()
        for name, field in BetLifecycleRecord.model_fields.items():
            annotation = field.annotation
            if annotation is str:
                out.add(name)
                continue
            # Optional[str], Literal["WIN", ...] and similar.
            args = get_args(annotation)
            if args and all(a is str or isinstance(a, str) or a is type(None) for a in args):
                out.add(name)
        return out

    def load_frame(self) -> pd.DataFrame:
        if self._parquet_ok and self.parquet_path.exists():
            return pd.read_parquet(self.parquet_path)
        if self.csv_path.exists():
            # Identifiers MUST be read back as text. bet_id is uuid4().hex[:16],
            # which is all digits about once in 1,100 ids; CSV has no types, so
            # pandas infers such an id as an int64. That breaks more than
            # validation: `df["bet_id"] == record.bet_id` compares an int to a
            # str and silently matches nothing, so settling or updating that
            # bet looks like it succeeded and changes no row.
            header = pd.read_csv(self.csv_path, nrows=0).columns
            dtypes = {c: str for c in self._string_columns() if c in header}
            return pd.read_csv(self.csv_path, dtype=dtypes)
        return pd.DataFrame()

    def pending_identity_key(self, record: BetLifecycleRecord) -> tuple[Any, ...]:
        """Dedup key: player/game/stat/side/line (Wave 4 lifecycle)."""
        return (
            str(record.game_id),
            str(record.player_id or record.player_name or ""),
            str(record.prop_stat or ""),
            str(record.bet_side).lower(),
            round(float(record.line), 3) if record.line is not None else None,
        )

    def list_pending(self) -> list[BetLifecycleRecord]:
        df = self.load_frame()
        if df.empty or "bet_result" not in df.columns:
            return []
        pending = df[df["bet_result"] == "PENDING"]
        return [self._row_to_record(r) for r in pending.to_dict(orient="records")]

    @staticmethod
    def _row_to_record(row: dict[str, Any]) -> BetLifecycleRecord:
        cleaned: dict[str, Any] = {}
        for k, v in row.items():
            if v is None:
                cleaned[k] = None
            elif isinstance(v, float) and v != v:  # NaN
                cleaned[k] = None
            else:
                cleaned[k] = v
        return BetLifecycleRecord.model_validate(cleaned)

    def grade_pending_props(
        self,
        actuals: dict[str, float],
        *,
        key_fields: tuple[str, ...] = ("game_id", "player_id"),
    ) -> list[BetLifecycleRecord]:
        """
        Grade PENDING prop rows when ``actuals`` provides a match.

        ``actuals`` keys are ``f"{game_id}|{player_id}"`` (or game_id alone when
        player_id is empty) → actual_stat.
        """
        graded: list[BetLifecycleRecord] = []
        for rec in self.list_pending():
            if rec.market_type not in {"player_prop", "unknown"} and rec.prop_stat is None:
                continue
            pid = rec.player_id or ""
            key = f"{rec.game_id}|{pid}"
            alt = str(rec.game_id)
            actual = actuals.get(key, actuals.get(alt))
            if actual is None:
                continue
            graded.append(self.grade_prop(rec, actual_stat=float(actual)))
        return graded

    def append(
        self,
        record: BetLifecycleRecord,
        *,
        allow_duplicate_pending: bool = False,
    ) -> BetLifecycleRecord:
        """
        Append one research row.

        Wave 4: rejects a second PENDING with the same identity key unless
        ``allow_duplicate_pending=True``. Prefer ``append_after_grading``.
        """
        if record.bet_result == "PENDING" and not allow_duplicate_pending:
            key = self.pending_identity_key(record)
            for existing in self.list_pending():
                if self.pending_identity_key(existing) == key:
                    return existing
        return self._write_append(record)

    def _write_append(self, record: BetLifecycleRecord) -> BetLifecycleRecord:
        row = record.model_dump(mode="json")
        df_new = pd.DataFrame([row])
        prev = self.load_frame()
        df = pd.concat([prev, df_new], ignore_index=True) if not prev.empty else df_new
        self._table_path_write(df)
        if self.config.use_sqlite:
            with sqlite3.connect(self.sqlite_path) as con:
                con.execute(
                    """
                    INSERT OR REPLACE INTO bet_lifecycle
                    (bet_id, payload_json, bet_result, created_at_utc, settled_at_utc)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        record.bet_id,
                        json.dumps(row),
                        record.bet_result,
                        record.created_at_utc.isoformat()
                        if isinstance(record.created_at_utc, datetime)
                        else str(record.created_at_utc),
                        record.settled_at_utc.isoformat() if record.settled_at_utc else None,
                    ),
                )
        return record

    def append_after_grading(
        self,
        records: list[BetLifecycleRecord],
        *,
        actuals: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """
        Wave 4 lifecycle: grade PENDING first, then append new picks.

        Mirrors NBA-NCAA "update old picks FIRST before adding new ones".
        """
        graded = self.grade_pending_props(actuals or {})
        appended: list[BetLifecycleRecord] = []
        skipped_dup = 0
        for rec in records:
            if rec.bet_result == "PENDING":
                key = self.pending_identity_key(rec)
                if any(self.pending_identity_key(p) == key for p in self.list_pending()):
                    skipped_dup += 1
                    continue
            appended.append(self._write_append(rec))
        return {
            "graded_pending": len(graded),
            "appended": len(appended),
            "skipped_duplicate_pending": skipped_dup,
            "graded_ids": [g.bet_id for g in graded],
            "appended_ids": [a.bet_id for a in appended],
            "note": "RESEARCH_ONLY grade-before-append; no live wager placement",
        }

    def update(self, record: BetLifecycleRecord) -> BetLifecycleRecord:
        df = self.load_frame()
        if df.empty:
            return self._write_append(record)
        mask = df["bet_id"] == record.bet_id
        if not mask.any():
            return self._write_append(record)
        row = record.model_dump(mode="json")
        # Replace entire row to avoid CSV float64/NaN dtype lock on optional fields
        keep = df.loc[~mask]
        df = pd.concat([keep, pd.DataFrame([row])], ignore_index=True)
        self._table_path_write(df)
        if self.config.use_sqlite:
            with sqlite3.connect(self.sqlite_path) as con:
                con.execute(
                    """
                    INSERT OR REPLACE INTO bet_lifecycle
                    (bet_id, payload_json, bet_result, created_at_utc, settled_at_utc)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        record.bet_id,
                        json.dumps(row),
                        record.bet_result,
                        record.created_at_utc.isoformat()
                        if isinstance(record.created_at_utc, datetime)
                        else str(record.created_at_utc),
                        record.settled_at_utc.isoformat() if record.settled_at_utc else None,
                    ),
                )
        return record

    def get(self, bet_id: str) -> BetLifecycleRecord | None:
        df = self.load_frame()
        if df.empty or "bet_id" not in df.columns:
            return None
        hit = df[df["bet_id"] == bet_id]
        if hit.empty:
            return None
        return self._row_to_record(hit.iloc[0].to_dict())

    def record_from_evaluation(
        self,
        evaluation: EvEvaluation,
        *,
        taken_american: int,
        model_prob: float,
        bet_side: str | None = None,
        player_id: str | None = None,
        player_name: str | None = None,
        prop_stat: str | None = None,
        bookmaker: str | None = None,
        opening_odds_american: int | None = None,
        closing_odds_american: int | None = None,
        closing_odds_other_american: int | None = None,
        clv: ClvResult | None = None,
        fair_prob_at_bet: float | None = None,
        season: str | None = None,
        unit_stake: float = 1.0,
    ) -> BetLifecycleRecord:
        side = bet_side or evaluation.selected_side
        if not side:
            raise ValueError("DATA_NOT_AVAILABLE: no bet_side / selected_side")
        raw_type = evaluation.market_type or "unknown"
        allowed = {"player_prop", "moneyline", "spread", "total", "unknown"}
        mtype: MarketType = raw_type if raw_type in allowed else "unknown"  # type: ignore[assignment]
        return BetLifecycleRecord(
            game_id=evaluation.game_id,
            player_id=player_id,
            player_name=player_name,
            market_type=mtype,
            market_id=evaluation.market_id,
            prop_stat=prop_stat,
            line=evaluation.line,
            bet_side=side,
            bookmaker=bookmaker,
            opening_odds_american=opening_odds_american,
            taken_odds_american=taken_american,
            closing_odds_american=closing_odds_american,
            closing_odds_other_american=closing_odds_other_american,
            model_prob=model_prob,
            fair_prob_at_bet=fair_prob_at_bet,
            ev_at_bet_time=evaluation.selected_ev,
            clv=clv.clv if clv and clv.status == "OK" else None,
            unit_stake=unit_stake,
            season=season,
        )

    def grade_prop(
        self,
        record: BetLifecycleRecord,
        *,
        actual_stat: float,
        settled_at_utc: datetime | None = None,
    ) -> BetLifecycleRecord:
        """Auto-grade Over/Under (or home/away numeric) against ``line``."""
        if record.line is None:
            record.bet_result = "VOID"
            record.notes = "DATA_NOT_AVAILABLE: missing line for grading"
            record.settled_at_utc = settled_at_utc or datetime.now(timezone.utc)
            return self.update(record)

        actual = float(actual_stat)
        line = float(record.line)
        side = record.bet_side.lower()
        record.stat_result = actual
        record.actual_outcome = actual

        if side in {"over", "o"}:
            if actual > line:
                result: BetResult = "WIN"
            elif actual < line:
                result = "LOSS"
            else:
                result = "PUSH"
        elif side in {"under", "u"}:
            if actual < line:
                result = "WIN"
            elif actual > line:
                result = "LOSS"
            else:
                result = "PUSH"
        else:
            # Non-prop sides need explicit winner flags — mark VOID if misused
            result = "VOID"
            record.notes = f"DATA_NOT_AVAILABLE: cannot auto-grade side={record.bet_side}"

        record.bet_result = result
        record.settled_at_utc = settled_at_utc or datetime.now(timezone.utc)
        record.profit_loss = _pnl(
            result, record.taken_odds_american, unit_stake=record.unit_stake
        )
        return self.update(record)

    def grade_moneyline(
        self,
        record: BetLifecycleRecord,
        *,
        home_won: bool,
        settled_at_utc: datetime | None = None,
    ) -> BetLifecycleRecord:
        side = record.bet_side.lower()
        won = (side == "home" and home_won) or (side == "away" and not home_won)
        record.bet_result = "WIN" if won else "LOSS"
        record.actual_outcome = 1.0 if home_won else 0.0
        record.settled_at_utc = settled_at_utc or datetime.now(timezone.utc)
        record.profit_loss = _pnl(
            record.bet_result, record.taken_odds_american, unit_stake=record.unit_stake
        )
        return self.update(record)

    def roi_summary(self) -> dict[str, Any]:
        df = self.load_frame()
        if df.empty:
            return {"status": "DATA_NOT_AVAILABLE", "n": 0}
        settled = df[df["bet_result"].isin(["WIN", "LOSS", "PUSH"])]
        if settled.empty:
            return {"status": "DATA_NOT_AVAILABLE", "n": 0, "reason": "no settled rows"}
        stake = float(settled["unit_stake"].fillna(1.0).sum())
        pnl = float(settled["profit_loss"].fillna(0.0).sum())
        return {
            "status": "OK",
            "n": int(len(settled)),
            "wins": int((settled["bet_result"] == "WIN").sum()),
            "losses": int((settled["bet_result"] == "LOSS").sum()),
            "pushes": int((settled["bet_result"] == "PUSH").sum()),
            "stake_units": round(stake, 4),
            "profit_loss": round(pnl, 4),
            "roi": round(pnl / stake, 6) if stake else None,
            "mean_clv": (
                round(float(settled["clv"].dropna().mean()), 6)
                if settled["clv"].notna().any()
                else None
            ),
        }


def _pnl(result: BetResult, american: int, *, unit_stake: float) -> float:
    if result == "PUSH" or result == "VOID" or result == "PENDING":
        return 0.0
    stake = float(unit_stake)
    if result == "LOSS":
        return -stake
    # WIN — American 0 is not a price (would divide by zero on favorite formula).
    if american == 0:
        raise ValueError("American odds of 0 are not a price")
    if american > 0:
        return stake * (american / 100.0)
    return stake * (100.0 / abs(american))
