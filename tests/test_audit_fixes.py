"""
Regression tests for the external audit's findings (2026-09-20).

Each test here failed against the code as it stood before the audit patch.
They are kept separate from the feature suites so the provenance of the fix
stays visible: these are defects found by review, not by a failing feature.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.quant.contracts import MarketContext, market_ev_gate


def _league_speeds_up_panel() -> pd.DataFrame:
    """A season where league pace rises steadily from day 1 to day 40."""
    rows = []
    for day in range(1, 41):
        for team in ("AAA", "BBB", "CCC", "DDD"):
            rows.append({
                "PLAYER_NAME": f"{team}_p", "TEAM_ABBREVIATION": team,
                "GAME_ID": f"{day:03d}{team}",
                "GAME_DATE": pd.Timestamp("2025-01-01") + pd.Timedelta(days=day),
                "SEASON": "2024-25",
                "FGA": 90 + day * 0.5, "OREB": 10, "TOV": 14, "FTA": 20, "PTS": 20,
            })
    return pd.DataFrame(rows)


def test_league_pace_baseline_is_as_of_date_not_season_wide():
    """
    PACE_MULTIPLIER divided each team's rolling pace by a season-wide league
    mean, which includes games that have not happened yet. In a season that
    speeds up, every early row was measured against a faster league than
    existed at the time — a systematic bias a backtest can exploit and a
    live run cannot.
    """
    from src.features.builder import attach_team_pace

    panel = _league_speeds_up_panel()
    cutoff = pd.Timestamp("2025-01-21")

    full = attach_team_pace(panel)
    truncated = attach_team_pace(panel[panel["GAME_DATE"] <= cutoff].copy())

    # The definition of no-lookahead: deleting every game AFTER the cutoff
    # must not change a single value at or before it. Under the season-wide
    # mean it changed all of them, because the divisor was computed from
    # games the row could not have seen.
    key = ["GAME_ID", "TEAM_ABBREVIATION"]
    before = full[full["GAME_DATE"] <= cutoff].set_index(key)["PACE_MULTIPLIER"]
    after = truncated.set_index(key)["PACE_MULTIPLIER"]
    aligned = before.align(after, join="inner")
    assert len(aligned[0]) > 0

    pd.testing.assert_series_equal(
        aligned[0].sort_index(), aligned[1].sort_index(),
        check_names=False, rtol=1e-9,
    )

    # And the early rows are no longer systematically pushed below 1 by a
    # league mean that includes later, faster games.
    early = full[full["GAME_DATE"] < "2025-01-15"]["PACE_MULTIPLIER"].dropna()
    assert len(early)
    assert early.mean() == pytest.approx(1.0, abs=0.05)


def test_market_context_is_accepted_without_a_total_attribute():
    """
    Both helpers advertise ``MarketContext | PropMarketSnapshot`` but read
    ``market.total``, which MarketContext does not have. Every MarketContext
    caller raised AttributeError.
    """
    from src.quant.line_diff import pickem_vs_book_line_diff
    from src.quant.paper_research import ResearchSlateRow, enrich_row_with_book

    ctx = MarketContext(
        game_id="g1", status="VALID", line=24.5,
        over_odds_american=-110, under_odds_american=-110,
    )
    assert not hasattr(ctx, "total")

    diff = pickem_vs_book_line_diff(24.0, ctx)
    assert diff.status in {"OK", "DATA_NOT_AVAILABLE"}

    row = ResearchSlateRow(
        slate_date="2026-01-16", event_id="g1", player_id="p1",
        target_market="PTS", model_p_over=0.60, model_p_under=0.40,
    )
    enriched = enrich_row_with_book(row, ctx)
    assert enriched.book_line == pytest.approx(24.5)
    assert enriched.book_status == "VALID"


def test_gate_names_missing_odds_instead_of_leaking_a_cast_error():
    """NaN odds reached the de-vig and surfaced as 'cannot convert float NaN'."""
    for bad in (float("nan"), 0):
        verdict = market_ev_gate(MarketContext(
            game_id="g1", status="VALID", line=24.5,
            over_odds_american=bad, under_odds_american=-110,
        ))
        assert verdict["status"] == "DATA_NOT_AVAILABLE"
        assert "missing" in verdict["reason"]


def test_feature_select_keeps_nan_instead_of_inventing_zero():
    """
    fill_value defaulted to 0.0, so a player with no prior games was handed
    to the model as a genuine 0.0-point average rather than as unknown.
    """
    from src.models.feature_spec import FeatureSpec

    spec = FeatureSpec(market="PTS", features=("PTS_L5",), categorical=())
    frame = pd.DataFrame({"PTS_L5": [12.0, None]})
    assert spec.select(frame)["PTS_L5"].isna().sum() == 1
    assert spec.select(frame, fill_value=0.0)["PTS_L5"].isna().sum() == 0


def test_zero_american_odds_are_rejected_in_pnl():
    """American 0 is not a price; the favourite formula divided by zero."""
    from src.quant.historical_store import _pnl

    with pytest.raises(ValueError, match="not a price"):
        _pnl("WIN", 0, unit_stake=1.0)
    assert _pnl("WIN", -110, unit_stake=1.0) == pytest.approx(0.909, abs=1e-3)


def test_edge_grade_is_signed_for_the_requested_side():
    """
    The grade used |edge|, so a side the model actively disliked graded the
    same as one it loved. A losing under could come back A+.
    """
    from src.models.edge_grades import research_edge_letter_grade

    kwargs = dict(prediction_mean=30.0, prop_line=24.0, prediction_std=4.0)
    over = research_edge_letter_grade(**kwargs, side="over")
    under = research_edge_letter_grade(**kwargs, side="under")

    assert over["edge_letter_grade"] != "F"
    assert under["edge_letter_grade"] == "F"
    assert over["side_lean"] == "over"


def test_postgame_efficiency_columns_do_not_survive_onto_the_feature_frame():
    """TS_PCT / SHOT_VOLUME / FT_RATE describe the game being predicted."""
    from src.features.scoring_efficiency import attach_box_ts_features

    panel = pd.DataFrame({
        "PLAYER_ID": ["1"] * 6,
        "PLAYER_NAME": ["A"] * 6,
        "GAME_DATE": pd.date_range("2025-01-01", periods=6, freq="D"),
        "PTS": [20, 22, 18, 25, 30, 15],
        "FGA": [15, 16, 14, 18, 20, 12],
        "FTA": [4, 5, 3, 6, 8, 2],
    })
    out = attach_box_ts_features(panel)
    for postgame in ("TS_PCT", "SHOT_VOLUME", "FT_RATE"):
        assert postgame not in out.columns
    assert "TS_PCT_L5" in out.columns
