"""
Tests for src/quant/decision_board.py — the MANUAL_ONLY decision layer.

Three things are load-bearing and each has a test that fails without it:

1. PropLine is PRIMARY and OddsPapi the FALLBACK, by precedence rather than
   by arrival order, with every skip explained.
2. P(under) is never the complement of P(over) when a push is possible.
3. A model lean never outranks a priced edge, and nothing in this layer can
   place, price or size a wager.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from src.quant.contracts import PropMarketSnapshot
from src.quant.decision_board import (
    BOARD_DISCLAIMER,
    PLACEMENT_MODE,
    SOURCE_PRECEDENCE,
    BettingDecisionCandidate,
    DecisionBoardError,
    build_decision_board,
    candidate_to_manual_bet_fields,
    decision_board_summary,
    enrich_row_with_resolved_market,
    expand_row_to_candidates,
    line_can_push,
    propline_row_to_snapshot,
    resolve_market,
    write_decision_board_csv,
)
from src.quant.paper_research import (
    ResearchSlateRow,
    enrich_row_with_book,
    model_prob_for_side,
    resolve_two_way_model_probs,
)


def _snap(source: str, **kw) -> PropMarketSnapshot:
    base = dict(
        game_id="g1", market="PTS", player_name="Demo", line=24.5,
        over_odds_american=-110, under_odds_american=-110,
        bookmaker=kw.pop("bookmaker", "somebook"), source=source, status="VALID",
        captured_at_utc=datetime.now(timezone.utc),
    )
    base.update(kw)
    return PropMarketSnapshot(**base)


PROPLINE = _snap("propline", over_odds_american=-115, under_odds_american=-105)
ODDSPAPI = _snap("oddspapi")
PROPLINE_PICKEM = _snap(
    "propline", over_odds_american=None, under_odds_american=None,
    payout_multiplier=3.0, is_pickem=True, bookmaker="prizepicks",
)


def _row(**kw) -> ResearchSlateRow:
    base = dict(
        slate_date="2026-01-16", event_id="g1", player_id="p1",
        player_name="Demo", target_market="PTS", research_line=24.5,
        model_p_over=0.62, model_p_under=0.38,
    )
    base.update(kw)
    return ResearchSlateRow(**base)


# --- the push rule -------------------------------------------------------


def test_resolve_half_line_complements_under():
    po, pu, pp, warn = resolve_two_way_model_probs(p_over=0.58, line=24.5)
    assert warn is None
    assert po == pytest.approx(0.58)
    assert pu == pytest.approx(0.42)
    assert pp == pytest.approx(0.0)


def test_resolve_whole_line_refuses_silent_complement():
    """1 - P(over) is P(under) + P(push); using it books every push as a win."""
    po, pu, pp, warn = resolve_two_way_model_probs(p_over=0.55, line=27.0)
    assert po == pytest.approx(0.55)
    assert pu is None and pp is None
    assert warn is not None and "DATA_NOT_AVAILABLE" in warn


def test_resolve_unknown_line_refuses_too():
    """A line we cannot see cannot be shown to be a half-line."""
    assert line_can_push(None) is True
    po, pu, pp, warn = resolve_two_way_model_probs(p_over=0.55, line=None)
    assert pu is None and pp is None
    assert warn is not None and "DATA_NOT_AVAILABLE" in warn


def test_resolve_refuses_an_inconsistent_triple_rather_than_clamping():
    po, pu, pp, warn = resolve_two_way_model_probs(p_over=0.80, p_push=0.40, line=27.0)
    assert pu is None
    assert warn is not None and "negative residual" in warn


def test_model_prob_for_side_never_invents_an_under():
    assert model_prob_for_side(bet_side="under", p_over=0.55, line=24.5) == pytest.approx(0.45)
    assert model_prob_for_side(bet_side="under", p_over=0.55, line=27.0) is None
    assert model_prob_for_side(
        bet_side="under", p_over=0.55, p_under=0.40, p_push=0.05, line=27.0
    ) == pytest.approx(0.40)


def test_enrich_row_with_book_refuses_the_complement_on_a_whole_line():
    whole = _snap("propline", line=8.0, market="REB")
    out = enrich_row_with_book(
        _row(target_market="REB", research_line=8.0, model_p_over=0.58, model_p_under=None),
        whole,
    )
    assert out.book_ev_under is None
    assert any("DATA_NOT_AVAILABLE" in w for w in out.warnings)

    priced = enrich_row_with_book(
        _row(target_market="REB", research_line=8.0, model_p_over=0.58,
             model_p_under=0.34, model_p_push=0.08),
        whole,
    )
    assert priced.book_ev_under is not None


# --- source precedence: PropLine primary, OddsPapi fallback --------------


def test_propline_is_preferred_over_oddspapi():
    assert SOURCE_PRECEDENCE == ("propline", "oddspapi")
    assert resolve_market([ODDSPAPI, PROPLINE]).source == "propline"
    assert resolve_market([ODDSPAPI, PROPLINE]).fallback_used is False


def test_precedence_not_arrival_order():
    assert resolve_market([ODDSPAPI, PROPLINE]).source == "propline"
    assert resolve_market([PROPLINE, ODDSPAPI]).source == "propline"


def test_oddspapi_is_used_when_propline_cannot_be_priced():
    """
    A PropLine pick'em row posts a payout multiplier, not a price. The
    fallback fires, is flagged, and the reason names the pick'em board.
    """
    resolution = resolve_market([PROPLINE_PICKEM, ODDSPAPI])
    assert resolution.source == "oddspapi"
    assert resolution.fallback_used is True
    assert "propline" in resolution.skipped_summary
    assert "multiplier" in resolution.skipped_summary
    assert resolution.pickem_snapshot is PROPLINE_PICKEM


def test_a_lone_fallback_source_is_not_reported_as_a_fallback():
    assert resolve_market([ODDSPAPI]).fallback_used is False


def test_no_usable_source_abstains_with_a_reason():
    assert resolve_market([PROPLINE_PICKEM]).snapshot is None
    assert "multiplier" in (resolve_market([PROPLINE_PICKEM]).reason or "")
    assert resolve_market([]).reason


def test_resolved_market_stamps_the_source_on_the_row():
    row, resolution = enrich_row_with_resolved_market(_row(), [PROPLINE_PICKEM, ODDSPAPI])
    assert row.book_status == "VALID"
    assert row.book_source == "oddspapi"
    assert row.book_fallback_used is True
    assert "propline" in row.book_sources_skipped
    # The pick'em line survives for line research even though it cannot price.
    assert row.pickem_line == pytest.approx(24.5)
    assert resolution.source == "oddspapi"

    board = build_decision_board([row])
    assert {c.book_source for c in board} == {"oddspapi"}
    assert all(c.book_fallback_used for c in board)


def test_propline_rows_bridge_into_snapshots():
    class Row:
        source = "draftkings"
        player_name = "Demo"
        market = "PTS"
        line = 24.5
        status = "VALID"
        over_odds_american = -110
        under_odds_american = -110
        nba_game_id = "0022500123"
        is_pickem = False
        payout_multiplier = None

    snap = propline_row_to_snapshot(Row())
    assert snap.source == "propline"        # the FEED, which precedence reads
    assert snap.bookmaker == "draftkings"   # the BOOK that posted it
    assert resolve_market([snap]).source == "propline"

    with pytest.raises(DecisionBoardError, match="Not a PropLine row"):
        propline_row_to_snapshot(object())


# --- board semantics -----------------------------------------------------


def test_decision_board_expands_both_sides_and_ranks_plus_ev():
    enriched = enrich_row_with_book(_row(), PROPLINE)
    assert enriched.book_status == "VALID"
    assert enriched.book_ev_over is not None
    assert enriched.book_ev_under is not None
    assert enriched.preferred_side in {"over", "under"}

    board = build_decision_board([enriched], min_ev=0.0, consider_only=False)
    assert len(board) == 2
    assert {c.side for c in board} == {"over", "under"}
    assert board[0].placement_mode == "MANUAL_ONLY"
    assert PLACEMENT_MODE == "MANUAL_ONLY"
    assert "MANUAL_ONLY" in BOARD_DISCLAIMER

    consider = [c for c in board if c.decision_status == "CONSIDER"]
    assert consider and consider[0].decision_basis == "book_ev"
    assert board[0].rank == 1


def test_require_valid_book_abstains_without_odds():
    cands = expand_row_to_candidates(_row(model_p_over=0.70, model_p_under=0.30),
                                     require_valid_book=True)
    assert all(c.decision_status == "ABSTAIN" for c in cands)
    assert all("require-valid-book" in c.why for c in cands)


def test_a_model_lean_never_outranks_a_priced_edge():
    """
    The lean is far larger in magnitude (0.49 past even vs an EV of ~0.08)
    and still ranks below, because the bands never interleave.
    """
    lean = _row(player_name="Leaner", model_p_over=0.99, model_p_under=0.01)
    priced = enrich_row_with_book(_row(player_name="Priced", model_p_over=0.58,
                                       model_p_under=0.42), PROPLINE)
    board = build_decision_board([lean, priced], consider_only=True)

    assert board[0].decision_basis == "book_ev"
    ev_ranks = [c.rank for c in board if c.decision_basis == "book_ev"]
    lean_ranks = [c.rank for c in board if c.decision_basis == "model_lean"]
    assert max(ev_ranks) < min(lean_ranks)
    assert max(c.rank_score for c in board if c.decision_basis == "model_lean") > 0.4


def test_a_pickem_line_alone_can_never_be_considered():
    row = _row(model_p_over=None, model_p_under=None,
               pickem_line=24.5, pickem_source="prizepicks")
    cands = expand_row_to_candidates(row)
    assert {c.decision_basis for c in cands} == {"pickem_line_only"}
    assert all(c.decision_status == "ABSTAIN" for c in cands)
    assert all(c.book_ev is None for c in cands)


def test_min_ev_and_top_n_and_consider_only():
    enriched = enrich_row_with_book(_row(), PROPLINE)
    assert any(c.decision_status == "CONSIDER"
               for c in build_decision_board([enriched], min_ev=0.0))
    assert not any(c.decision_status == "CONSIDER"
                   for c in build_decision_board([enriched], min_ev=0.50))
    assert len(build_decision_board([enriched], top_n=1)) == 1
    assert len(build_decision_board([enriched], consider_only=True)) == 1


def test_the_board_never_claims_an_outcome():
    from src.quant.decision_board import FORBIDDEN_CLAIM_WORDS, _assert_no_claims

    for word in ("lock", "guaranteed", "best bet"):
        with pytest.raises(DecisionBoardError):
            _assert_no_claims(f"this is a {word}")

    board = build_decision_board([enrich_row_with_book(_row(model_p_over=0.99,
                                                           model_p_under=0.01), PROPLINE)])
    for c in board:
        assert not any(w in c.why.lower() for w in FORBIDDEN_CLAIM_WORDS)


def test_layer_cannot_reach_a_book_or_size_a_stake():
    """
    MANUAL_ONLY is a property of the code, not just of the docstring.

    Checked against the parsed AST rather than the raw text, so prose that
    says "never computes Kelly" does not trip the very guard it describes.
    """
    import ast
    import inspect

    from src.quant import decision_board

    tree = ast.parse(inspect.getsource(decision_board))

    imported: set[str] = set()
    identifiers: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
        elif isinstance(node, ast.Name):
            identifiers.add(node.id.lower())
        elif isinstance(node, ast.Attribute):
            identifiers.add(node.attr.lower())
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            identifiers.add(node.name.lower())

    for network in ("requests", "http", "httpx", "urllib", "socket", "aiohttp"):
        assert network not in imported, f"decision_board must not import {network!r}"

    for banned in ("kelly", "stake_size", "place_bet", "submit_order"):
        offenders = [i for i in identifiers if banned in i]
        assert not offenders, f"decision_board must not call {offenders!r}"

    fields = BettingDecisionCandidate.model_fields
    assert not [f for f in fields if "stake" in f or "bankroll" in f]


# --- handoff to the manual log ------------------------------------------


def test_candidate_maps_to_manual_log_fields():
    row = _row(
        target_market="AST", research_line=7.5, model_p_over=0.40, model_p_under=0.60,
        book_status="VALID", book_line=7.5, book_over_american=-115,
        book_under_american=-105, book_ev_over=-0.02, book_ev_under=0.05,
        preferred_side="under",
    )
    board = build_decision_board([row], min_ev=0.0, consider_only=True)
    under = next(c for c in board if c.side == "under")
    fields = candidate_to_manual_bet_fields(under)
    assert fields["status"] == "READY_TO_LOG_AFTER_YOU_BET"
    assert fields["side"] == "under"
    # model_prob is P(THE SIDE TAKEN) — 0.60, not P(over).
    assert fields["model_prob"] == pytest.approx(0.60)
    assert "kelly" in fields["note"].lower()
    assert not [k for k in fields if "stake" in k]


def test_abstained_candidates_do_not_hand_over_log_fields():
    cands = expand_row_to_candidates(_row(model_p_over=0.51, model_p_under=0.49))
    weak = next(c for c in cands if c.decision_status == "ABSTAIN")
    assert candidate_to_manual_bet_fields(weak)["status"] == "ABSTAIN"


# --- end to end ----------------------------------------------------------


def test_board_summary_and_csv(tmp_path):
    row, _ = enrich_row_with_resolved_market(_row(), [ODDSPAPI, PROPLINE])
    board = build_decision_board([row])

    summary = decision_board_summary(board)
    assert summary["placement_mode"] == PLACEMENT_MODE
    assert summary["sources_used"] == ["propline"]
    assert summary["n_candidates"] == 2

    out = tmp_path / "decision_board.csv"
    assert write_decision_board_csv(board, out) == 2
    frame = pd.read_csv(out)
    for column in ("side", "decision_status", "decision_basis", "book_ev",
                   "preferred_side", "why", "rank", "book_source"):
        assert column in frame.columns
