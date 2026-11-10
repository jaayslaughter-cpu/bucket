"""
The captured two-way price reaching the ranker.

THE DEFECT, WITH THE NUMBERS FROM 2026-11-09:

    PropLine: 19340 row(s) across 25 book(s) from 106 event(s).
    EV gate: 17210 ready, 2130 abstained.
    Board:   'n_consider_with_book_ev': 0, 'sources_used': []

17,210 quotes passed the EV gate with two-way American odds and not one
reached the board. `projections` stopped at `line, prob_over, prob_under,
prob_push`, so every board row was `model_lean` with `ev: None` BY
CONSTRUCTION -- "no positive-EV prop tonight" and "no price ever reached the
ranker" produced identical output, and it was always the second.

`enrich_row_with_book` had existed the whole time. Nothing ever had a price
to hand it.

RESEARCH_ONLY. Synthetic rows; no EV here is a real edge.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.pipeline.slate_board import projection_detail_rows
from src.quant.decision_board import build_decision_board, decision_board_summary
from src.quant.paper_research import research_slate_from_predictions

MODEL = "projection"


def _projection(**over):
    row = {
        "nba_game_id": "0022600321", "nba_player_id": "1",
        "player_name": "DEMO_A", "market": "PTS", "game_date": "2026-11-10",
        "final_projection": 27.0, "line": 24.5,
        "prob_over": 0.62, "prob_under": 0.38, "prob_push": 0.0,
        "odds_over_american": -115, "odds_under_american": -105,
        "line_source": "propline",
    }
    row.update(over)
    return pd.DataFrame([row])


def _board(frame):
    rows = research_slate_from_predictions(
        projection_detail_rows(frame), slate_date="2026-11-10",
        preferred_model=MODEL,
    )
    return rows, build_decision_board(rows)


# --- the price completes the journey ---------------------------------------

def test_a_priced_projection_produces_book_ev_not_a_bare_lean():
    rows, board = _board(_projection())
    summary = decision_board_summary(board)

    assert rows[0].book_status == "VALID"
    assert summary["n_consider_with_book_ev"] >= 1, (
        "the board still establishes no book EV on a row that carries a "
        "two-way price"
    )
    assert summary["n_consider_model_lean_only"] == 0
    assert {c.decision_basis for c in board} == {"book_ev"}


def test_the_board_names_the_feed_that_priced_it():
    """
    `decision_board_summary` builds `sources_used` from `book_source`, and
    `enrich_row_with_book` never set it -- so a fully priced board reported
    that nobody had quoted it.
    """
    rows, board = _board(_projection())
    assert rows[0].book_source == "propline"
    assert decision_board_summary(board)["sources_used"] == ["propline"]


def test_both_sides_are_priced_from_one_de_vigged_quote():
    rows, _board_rows = _board(_projection())
    row = rows[0]
    assert row.book_ev_over is not None and row.book_ev_under is not None
    # A two-way price cannot be +EV on both sides once the vig is removed.
    assert not (row.book_ev_over > 0 and row.book_ev_under > 0), (
        "both sides came out positive, so the overround was never removed"
    )


def test_the_odds_reach_the_detail_row_at_all():
    detail = projection_detail_rows(_projection())[0]
    assert detail["odds_over_american"] == -115
    assert detail["odds_under_american"] == -105
    assert detail["line_source"] == "propline"


# --- what must NOT happen --------------------------------------------------

def test_a_row_with_no_price_still_ranks_on_lean_rather_than_raising():
    """
    Every row written before migration 010 has no price, and most live rows
    never get one. They must still reach the board.
    """
    rows, board = _board(_projection(
        odds_over_american=None, odds_under_american=None, line_source=None,
    ))
    assert rows[0].book_status == "DATA_NOT_AVAILABLE"
    assert board, "an unpriced row fell out of the board entirely"
    assert decision_board_summary(board)["n_consider_with_book_ev"] == 0


def test_an_unpriced_row_is_not_warned_about_on_every_line():
    """
    `enrich_row_with_book(row, None)` appends "No priced market attached".
    On a board where most rows are legitimately unpriced that is a warning on
    nearly every row, which trains a reader to ignore the warning column.
    """
    rows, _ = _board(_projection(odds_over_american=None, odds_under_american=None))
    assert not any("No priced market attached" in w for w in rows[0].warnings)


def test_one_sided_odds_are_not_a_two_way_price():
    rows, board = _board(_projection(odds_under_american=None))
    assert rows[0].book_status == "DATA_NOT_AVAILABLE"
    assert decision_board_summary(board)["n_consider_with_book_ev"] == 0


@pytest.mark.parametrize("bad", [0, "", "not-a-number"])
def test_an_unusable_odds_value_is_refused_rather_than_priced(bad):
    """
    Zero is not a price, and neither is a string the book never sent.

    AND IT IS REFUSED QUIETLY. `market_ev_gate` would also refuse these, so
    the first version of this test passed with the local guard mutated away
    -- it was asserting the gate's behaviour, not this module's. What the
    local guard adds is that an unusable value never becomes a "priced
    market" that then fails the gate and appends a warning to the row. The
    warning is the observable difference.
    """
    rows, _ = _board(_projection(odds_over_american=bad))
    assert rows[0].book_status == "DATA_NOT_AVAILABLE"
    assert rows[0].book_source is None
    # THE OBSERVABLE THE LOCAL GUARD ADDS, measured rather than assumed:
    # without it the junk value becomes a MarketContext, the gate refuses it,
    # and the row carries "Two-way American odds required; one or both sides
    # are missing" -- a warning on a row that simply had no usable quote. On
    # a board where most rows are unpriced that trains a reader to ignore the
    # warning column. Two earlier versions of this assertion tested the
    # GATE's behaviour instead and passed with the guard mutated away.
    assert rows[0].warnings == [], (
        f"{bad!r} reached the gate and warned: {rows[0].warnings}"
    )


def test_a_pickem_multiplier_never_becomes_american_odds():
    """
    A pick'em payout multiplier is not a two-way price. `market_ev_gate`
    refuses them and `quant.dfs_payouts` is where they belong; writing one
    into these columns would make the two indistinguishable downstream.
    """
    import main as propiq_main

    projections = pd.DataFrame([{
        "PLAYER_NAME": "DEMO_A", "MARKET": "PTS",
    }])
    board = pd.DataFrame([{
        "player_name": "DEMO_A", "market": "PTS", "line": 24.5,
        "over_odds_american": -115, "under_odds_american": -105,
        "is_pickem": True, "source": "prizepicks",
    }])
    priced = propiq_main._attach_prop_line_prices(projections, board)
    assert priced["ODDS_OVER"].iloc[0] is None
    assert priced["ODDS_UNDER"].iloc[0] is None
    assert priced["LINE_SOURCE"].iloc[0] is None


def test_the_price_and_the_line_come_from_one_resolution():
    """
    ONE CROSSWALK. The line and the odds must be read from the same resolved
    board row; two calls to `resolve_board_names` would be two chances for a
    row's price to come from a different player than its line.
    """
    import ast
    import inspect

    import main as propiq_main

    src = inspect.getsource(propiq_main._attach_prop_line_prices)
    tree = ast.parse(inspect.cleandoc(src))
    called = {
        n.func.id for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "_resolved_board_lookup" in called
    assert "resolve_board_names" not in called, (
        "the price attach resolves names itself instead of sharing the "
        "lookup the line join uses"
    )
