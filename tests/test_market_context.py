"""
Tests for src/features/market_context.py and its wiring into the builder.

The licensed workbook is gitignored, so these build their own frames in its
shape. The property under test is the one that decides whether market data
helps or quietly ruins a backtest: opening lines are pregame, closing lines
are not, and the code must not be able to confuse them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.builder import build_feature_matrix
from src.features.market_context import (
    MARKET_FEATURE_COLS,
    ClosingLineLeakageError,
    MarketContextError,
    assert_no_closing_lines,
    attach_market_context,
    build_market_context,
    closing_line_value,
    implied_team_total,
)


def _market_lines() -> pd.DataFrame:
    """Two games, four team rows, in the workbook's column shape."""
    return pd.DataFrame([
        {"nba_game_id": "0022500001", "game_date": "2025-10-21", "team_abbr": "HOU",
         "opening_spread": 6.5, "opening_total": 225.5,
         "closing_spread": 6.5, "closing_total": 225.5, "moneyline": "+200"},
        {"nba_game_id": "0022500001", "game_date": "2025-10-21", "team_abbr": "OKC",
         "opening_spread": -6.5, "opening_total": 225.5,
         "closing_spread": -6.5, "closing_total": 225.5, "moneyline": "-245"},
        {"nba_game_id": "0022500002", "game_date": "2025-10-21", "team_abbr": "GSW",
         "opening_spread": 3.5, "opening_total": 224.5,
         "closing_spread": -2.5, "closing_total": 227.5, "moneyline": "-148"},
        {"nba_game_id": "0022500002", "game_date": "2025-10-21", "team_abbr": "LAL",
         "opening_spread": -3.5, "opening_total": 224.5,
         "closing_spread": 2.5, "closing_total": 227.5, "moneyline": "+125"},
    ])


def _panel() -> pd.DataFrame:
    rows = []
    for game, teams in (("0022500001", ("HOU", "OKC")), ("0022500002", ("GSW", "LAL"))):
        for team in teams:
            for p in range(4):
                rows.append({
                    "PLAYER_ID": f"{team}{p}", "PLAYER_NAME": f"{team} P{p}",
                    "GAME_ID": game, "GAME_DATE": pd.Timestamp("2025-10-21"),
                    "SEASON": "2025-26", "TEAM_ABBREVIATION": team,
                    "OPPONENT_ABBREVIATION": teams[1 - teams.index(team)],
                    "IS_HOME": team in ("OKC", "LAL"), "MIN": 30.0,
                    "PTS": 20.0, "REB": 5.0, "AST": 4.0,
                    "FGA": 15.0, "FTA": 4.0, "OREB": 1.0, "TOV": 2.0,
                })
    return pd.DataFrame(rows)


# --- the market's own forecast ------------------------------------------


def test_implied_team_total_matches_the_posted_numbers():
    """OKC at -6.5 in a 225.5 game implies 116.0; the dog implies 109.5."""
    assert implied_team_total(225.5, -6.5) == pytest.approx(116.0)
    assert implied_team_total(225.5, 6.5) == pytest.approx(109.5)
    assert implied_team_total(None, -6.5) is None
    assert implied_team_total(225.5, float("nan")) is None


def test_the_two_implied_totals_sum_back_to_the_posted_total():
    context = build_market_context(_market_lines())
    residual = (
        context["MKT_IMPLIED_TEAM_TOTAL"] + context["MKT_IMPLIED_OPP_TOTAL"]
        - context["MKT_OPENING_TOTAL"]
    ).abs().max()
    assert residual < 1e-9


def test_a_missing_spread_is_unknown_not_a_pick_em():
    lines = _market_lines()
    lines.loc[0, "opening_spread"] = None
    context = build_market_context(lines)
    assert pd.isna(context.loc[0, "MKT_IS_FAVORITE"])
    assert pd.isna(context.loc[0, "MKT_IMPLIED_TEAM_TOTAL"])


# --- the line that must never become a feature ---------------------------


def test_closing_columns_are_refused_as_features():
    """
    A closing number is known only at tip. Joining it to a projection made
    hours earlier hands the model the market's final answer.
    """
    with pytest.raises(ClosingLineLeakageError, match="known only at tip"):
        assert_no_closing_lines(["PTS_L5", "closing_spread"])

    leaky = _panel().assign(closing_total=227.5)
    with pytest.raises(ClosingLineLeakageError):
        attach_market_context(leaky, _market_lines())


def test_no_closing_column_survives_the_join():
    merged = attach_market_context(_panel(), _market_lines())
    assert not [c for c in merged.columns if "closing" in str(c).lower()]
    for col in MARKET_FEATURE_COLS:
        assert col in merged.columns


# --- joining -------------------------------------------------------------


def test_unmatched_rows_keep_nan_rather_than_a_league_average():
    panel = _panel()
    panel.loc[panel["TEAM_ABBREVIATION"] == "HOU", "GAME_ID"] = "9999999999"
    merged = attach_market_context(panel, _market_lines())
    unmatched = merged[merged["GAME_ID"] == "9999999999"]
    assert len(unmatched) == 4
    assert unmatched["MKT_OPENING_TOTAL"].isna().all()
    assert merged["MKT_OPENING_TOTAL"].notna().sum() == 12


def test_duplicate_team_game_rows_do_not_fan_out_the_join():
    doubled = pd.concat([_market_lines(), _market_lines().head(1)], ignore_index=True)
    merged = attach_market_context(_panel(), doubled)
    assert len(merged) == len(_panel())      # a many_to_one join, not a cross


def test_missing_market_lines_narrows_the_run_instead_of_failing():
    unchanged = attach_market_context(_panel(), None)
    assert list(unchanged.columns) == list(_panel().columns)

    with pytest.raises(MarketContextError, match="missing"):
        build_market_context(pd.DataFrame({"nba_game_id": ["1"]}))


# --- the closing line's one legitimate use -------------------------------


def test_closing_line_value_measures_the_move_and_is_zero_sum():
    clv = closing_line_value(_market_lines())
    graded = clv[clv["status"] == "OK"]
    assert len(graded) == 4

    # GSW opened +3.5 and closed -2.5: six points toward whoever took the open.
    gsw = graded[graded["team_abbr"] == "GSW"].iloc[0]
    assert gsw["clv_line_points"] == pytest.approx(6.0)
    lal = graded[graded["team_abbr"] == "LAL"].iloc[0]
    assert lal["clv_line_points"] == pytest.approx(-6.0)

    # Across both sides of every game the move nets to zero.
    assert graded["clv_line_points"].sum() == pytest.approx(0.0)


def test_clv_abstains_when_either_end_is_unpriced():
    lines = _market_lines()
    lines.loc[0, "closing_spread"] = None
    clv = closing_line_value(lines)
    assert clv.loc[0, "status"] == "DATA_NOT_AVAILABLE"
    assert pd.isna(clv.loc[0, "clv_line_points"])


# --- the wiring ----------------------------------------------------------


def test_market_lines_reach_the_feature_matrix_and_move_the_schema_version():
    without = build_feature_matrix(_panel())
    with_market = build_feature_matrix(_panel(), market_lines=_market_lines())

    assert not [c for c in without.columns if c.startswith("MKT_")]
    for col in MARKET_FEATURE_COLS:
        assert col in with_market.columns
    assert with_market["MKT_IMPLIED_TEAM_TOTAL"].notna().all()

    # A run with market context genuinely has a different feature set, and
    # must not claim the same schema version as one without.
    assert (
        without["FEATURE_SCHEMA_VERSION"].iloc[0]
        != with_market["FEATURE_SCHEMA_VERSION"].iloc[0]
    )


def test_the_builder_still_runs_its_lookahead_assertion_with_market_columns():
    from src.features.builder import assert_no_lookahead

    built = build_feature_matrix(_panel(), market_lines=_market_lines())
    assert_no_lookahead(built)     # raises if anything postgame slipped in
    assert np.isfinite(built["MKT_OPENING_TOTAL"]).all()
