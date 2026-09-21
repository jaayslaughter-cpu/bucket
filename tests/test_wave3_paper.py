"""Wave 3 manual paper-research layer tests."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.quant.contracts import PropMarketSnapshot
from src.quant.historical_store import HistoricalStore, HistoricalStoreConfig
from src.quant.paper_research import (
    PLACEMENT_MODE,
    ManualBetInput,
    enrich_row_with_book,
    log_manual_bet,
    paper_improvement_report,
    research_slate_from_predictions,
    write_slate_csv,
)


def test_placement_mode_is_manual_only():
    assert PLACEMENT_MODE == "MANUAL_ONLY"


def test_research_slate_from_predictions(tmp_path: Path):
    details = [
        {
            "event_id": "g1",
            "player_id": "p1",
            "player_name": "Demo",
            "player_team": "LAL",
            "opponent": "BOS",
            "target_market": "PTS",
            "prop_line": 24.5,
            "prediction_mean": 26.0,
            "prediction_std_or_dispersion": 4.0,
            "probability_over_raw": 0.58,
            "probability_under_raw": 0.42,
            "model_name": "distribution",
            "hot_hand_status": "NEUTRAL",
        },
        {
            "event_id": "g1",
            "player_id": "p1",
            "player_name": "Demo",
            "player_team": "LAL",
            "opponent": "BOS",
            "target_market": "PTS",
            "prop_line": 24.5,
            "prediction_mean": 25.0,
            "prediction_std_or_dispersion": None,
            "probability_over_raw": 0.60,
            "probability_under_raw": 0.40,
            "model_name": "xgboost",
            "hot_hand_status": "NEUTRAL",
        },
    ]
    rows = research_slate_from_predictions(details, slate_date="2025-01-16", preferred_model="distribution")
    assert len(rows) == 1
    assert rows[0].placement_mode == "MANUAL_ONLY"
    assert rows[0].model_p_over == pytest.approx(0.58)
    assert rows[0].model_p_under == pytest.approx(0.42)
    assert rows[0].model_p_push == pytest.approx(0.0)
    assert rows[0].preferred_side in {"over", "under"}
    assert rows[0].edge_letter_grade_under is not None
    assert rows[0].confidence_tier in {"HIGH", "MODERATE", "LOW", "DISAGREE", "ABSTAIN"}
    out = tmp_path / "slate.csv"
    assert write_slate_csv(rows, out) == 1
    assert out.exists()


def test_enrich_book_valid_computes_ev():
    from src.quant.paper_research import ResearchSlateRow

    row = ResearchSlateRow(
        slate_date="2025-01-16",
        event_id="g1",
        player_id="p1",
        target_market="PTS",
        model_p_over=0.60,
    )
    market = PropMarketSnapshot(
        game_id="g1",
        bookmaker="demo",
        market_id="m1",
        line=24.5,
        over_odds_american=-110,
        under_odds_american=-110,
        status="VALID",
        captured_at_utc=datetime.now(timezone.utc),
    )
    enriched = enrich_row_with_book(row, market)
    assert enriched.book_status == "VALID"
    assert enriched.book_ev_over is not None
    assert enriched.book_ev_under is not None
    assert enriched.model_p_under == pytest.approx(0.40)
    assert enriched.preferred_side in {"over", "under"}
    assert enriched.book_ev_over > 0


def test_enrich_book_prices_under_side_explicitly():
    from src.quant.paper_research import ResearchSlateRow

    row = ResearchSlateRow(
        slate_date="2025-01-16",
        event_id="g1",
        player_id="p1",
        target_market="PTS",
        research_line=27.0,
        model_p_over=0.35,
        model_p_under=0.55,
        model_p_push=0.10,
    )
    market = PropMarketSnapshot(
        game_id="g1",
        bookmaker="demo",
        market_id="m1",
        line=27.0,
        over_odds_american=-110,
        under_odds_american=-110,
        status="VALID",
        captured_at_utc=datetime.now(timezone.utc),
    )
    enriched = enrich_row_with_book(row, market)
    assert enriched.book_status == "VALID"
    assert enriched.book_ev_under is not None
    assert enriched.preferred_side == "under"


def test_log_manual_bet_and_report(tmp_path: Path):
    store = HistoricalStore(HistoricalStoreConfig(root=tmp_path, use_sqlite=False, prefer_parquet=False))
    bet = ManualBetInput(
        game_id="g1",
        player_id="p1",
        player_name="Demo",
        prop_stat="PTS",
        line=24.5,
        bet_side="over",
        taken_odds_american=-110,
        model_prob=0.58,
        unit_stake=1.0,
    )
    result = log_manual_bet(store, bet)
    assert result["placement_mode"] == "MANUAL_ONLY"
    assert result["appended"] == 1

    # Grade via actuals on second call
    bet2 = ManualBetInput(
        game_id="g2",
        player_id="p2",
        prop_stat="REB",
        line=8.5,
        bet_side="under",
        taken_odds_american=-110,
        model_prob=0.45,
    )
    result2 = log_manual_bet(store, bet2, actuals={"g1|p1": 30.0})
    assert result2["graded_pending"] == 1

    report = paper_improvement_report(store)
    assert report["status"] == "OK"
    assert report["placement_mode"] == "MANUAL_ONLY"
    assert report["n_settled"] >= 1
