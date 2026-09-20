"""Wave 5a: Sports-EV features + pocket ROI board."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.features.builder import assert_no_lookahead, build_feature_matrix
from src.features.sports_ev_features import (
    attach_form_streaks,
    attach_opp_allowed_l10,
    attach_usage_proxy,
)
from src.models.data_audit import make_demo_panel
from src.quant.historical_store import HistoricalStore, HistoricalStoreConfig
from src.quant.paper_research import ManualBetInput, log_manual_bet
from src.quant.pocket_roi import build_pocket_roi_board, write_pocket_roi_csv


def test_usage_proxy_shift_safe_when_box_present():
    raw = make_demo_panel(n_players=2, n_games=20)
    # Add box attempts for usage
    rng = np.random.default_rng(0)
    raw = raw.copy()
    raw["FGA"] = rng.integers(8, 20, size=len(raw)).astype(float)
    raw["FTA"] = rng.integers(0, 8, size=len(raw)).astype(float)
    raw["TOV"] = rng.integers(0, 4, size=len(raw)).astype(float)
    raw["OREB"] = rng.integers(0, 3, size=len(raw)).astype(float)
    out = attach_usage_proxy(raw)
    assert out.attrs.get("usage_proxy_status") == "OK"
    assert "USAGE_PROXY" in out.columns
    assert "USAGE_PROXY_L10" in out.columns
    # First game per player: shifted usage must be NaN
    first = out.sort_values(["PLAYER_ID", "GAME_DATE"]).groupby("PLAYER_ID").head(1)
    assert first["USAGE_PROXY"].isna().all()


def test_usage_proxy_missing_box_is_dna():
    raw = make_demo_panel(n_players=1, n_games=5)
    out = attach_usage_proxy(raw)
    assert out.attrs.get("usage_proxy_status") == "DATA_NOT_AVAILABLE"
    assert out["USAGE_PROXY"].isna().all()


def test_streaks_and_opp_allowed_in_builder():
    raw = make_demo_panel(n_players=3, n_games=25)
    feats = build_feature_matrix(raw)
    assert "PTS_STREAK_ABOVE" in feats.columns
    assert "PTS_STREAK_BELOW" in feats.columns
    assert "OPP_PTS_ALLOWED_L10" in feats.columns
    assert_no_lookahead(feats)


def test_form_streak_increases_on_hot_run():
    df = pd.DataFrame(
        {
            "PLAYER_ID": ["P1"] * 15,
            "SEASON": ["2024-25"] * 15,
            "GAME_DATE": pd.date_range("2024-11-01", periods=15, freq="D"),
            "GAME_ID": [f"g{i}" for i in range(15)],
            "PTS": [10] * 8 + [25] * 7,
            "PTS_SEASON": [12.0] * 15,
        }
    )
    out = attach_form_streaks(df)
    # Late elevated games should show above-streak > 0
    assert out["PTS_STREAK_ABOVE"].iloc[-1] >= 1


def test_opp_allowed_requires_opponent():
    df = pd.DataFrame(
        {
            "PLAYER_ID": ["A"],
            "TEAM_ABBREVIATION": ["LAL"],
            "GAME_ID": ["g1"],
            "SEASON": ["2024-25"],
            "GAME_DATE": [pd.Timestamp("2024-11-01")],
            "PTS": [20],
        }
    )
    out = attach_opp_allowed_l10(df)
    assert out.attrs.get("opp_allowed_status") == "DATA_NOT_AVAILABLE"


def test_pocket_roi_board(tmp_path: Path):
    store = HistoricalStore(HistoricalStoreConfig(root=tmp_path, use_sqlite=False, prefer_parquet=False))
    r1 = log_manual_bet(
        store,
        ManualBetInput(
            game_id="g1",
            player_id="p1",
            prop_stat="PTS",
            line=24.5,
            bet_side="over",
            taken_odds_american=-110,
            model_prob=0.58,
            confidence_tier="HIGH",
            edge_letter_grade="B",
            model_name="distribution",
            unit_stake=1.0,
        ),
    )
    r2 = log_manual_bet(
        store,
        ManualBetInput(
            game_id="g2",
            player_id="p2",
            prop_stat="REB",
            line=8.5,
            bet_side="under",
            taken_odds_american=-110,
            model_prob=0.45,
            confidence_tier="LOW",
            edge_letter_grade="D",
            model_name="xgboost",
            unit_stake=1.0,
        ),
    )
    for bet_id, actual in ((r1["bet_id"], 30.0), (r2["bet_id"], 12.0)):
        rec = store.get(bet_id)
        assert rec is not None
        store.grade_prop(rec, actual_stat=actual)

    board = build_pocket_roi_board(store)
    assert board["status"] == "OK"
    assert board["overall"] is not None
    assert board["n_settled"] == 2
    assert any(p["pocket"] == "prop_stat" for p in board["pockets"])
    assert any(p["pocket"] == "confidence_tier" and p["key"] == "HIGH" for p in board["pockets"])

    out = write_pocket_roi_csv(store, tmp_path / "pocket.csv")
    assert Path(out["out"]).exists()
    assert out["rows_written"] >= 1
