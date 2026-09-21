"""
Tests for src/quant/decision_board.py — the MANUAL_ONLY decision layer.

Three things are load-bearing here and each has a test that fails without it:

1. PropLine is PRIMARY and OddsPapi is the FALLBACK, by precedence rather
   than by arrival order, with every skip explained.
2. P(under) is never the complement of P(over) when a push is possible.
3. Nothing in this layer places, prices or sizes a wager.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.quant.contracts import PropMarketSnapshot
from src.quant.decision_board import (
    DECISION_DISCLAIMER,
    FORBIDDEN_CLAIM_WORDS,
    PLACEMENT_MODE,
    SOURCE_PRECEDENCE,
    DecisionBoardError,
    board_summary,
    decision_board_from_slate,
    decision_log_fields,
    expand_sides,
    line_can_push,
    propline_row_to_snapshot,
    rank_decisions,
    resolve_market,
    side_probabilities,
    write_decision_board_csv,
)


def _snap(source: str, **kw) -> PropMarketSnapshot:
    base = dict(
        game_id="G1", market="PTS", player_name="A", line=24.5,
        over_odds_american=-110, under_odds_american=-110,
        bookmaker=kw.pop("bookmaker", "somebook"), source=source, status="VALID",
    )
    base.update(kw)
    return PropMarketSnapshot(**base)


PROPLINE = _snap("propline", over_odds_american=-115, under_odds_american=-105)
ODDSPAPI = _snap("oddspapi")
PROPLINE_PICKEM = _snap(
    "propline", over_odds_american=None, under_odds_american=None,
    payout_multiplier=3.0, is_pickem=True, bookmaker="prizepicks",
)


# --- source precedence: PropLine primary, OddsPapi fallback --------------


def test_propline_is_preferred_over_oddspapi():
    assert SOURCE_PRECEDENCE == ("propline", "oddspapi")
    resolution = resolve_market([ODDSPAPI, PROPLINE])
    assert resolution.source == "propline"
    assert resolution.fallback_used is False


def test_precedence_not_arrival_order():
    """Listing OddsPapi first must not make it primary."""
    assert resolve_market([ODDSPAPI, PROPLINE]).source == "propline"
    assert resolve_market([PROPLINE, ODDSPAPI]).source == "propline"


def test_oddspapi_is_used_when_propline_cannot_be_priced():
    """
    A PropLine pick'em row posts a payout multiplier, not a price. The
    fallback fires, is flagged, and the reason names the pick'em board — a
    fallback nobody can explain is indistinguishable from a bug.
    """
    resolution = resolve_market([PROPLINE_PICKEM, ODDSPAPI])
    assert resolution.source == "oddspapi"
    assert resolution.fallback_used is True
    assert "propline" in resolution.skipped_summary
    assert "multiplier" in resolution.skipped_summary
    # The pick'em line survives for line research even though it cannot price.
    assert resolution.pickem_snapshot is PROPLINE_PICKEM


def test_a_lone_fallback_source_is_not_reported_as_a_fallback():
    resolution = resolve_market([ODDSPAPI])
    assert resolution.source == "oddspapi"
    assert resolution.fallback_used is False


def test_no_usable_source_abstains_with_a_reason():
    resolution = resolve_market([PROPLINE_PICKEM])
    assert resolution.snapshot is None
    assert resolution.is_priced is False
    assert "multiplier" in (resolution.reason or "")

    empty = resolve_market([])
    assert empty.snapshot is None
    assert empty.reason


def test_propline_rows_bridge_into_snapshots():
    class Row:
        source = "draftkings"
        player_name = "A"
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
    assert snap.game_id == "0022500123"
    assert resolve_market([snap]).source == "propline"

    with pytest.raises(DecisionBoardError, match="Not a PropLine row"):
        propline_row_to_snapshot(object())


# --- the push rule -------------------------------------------------------


def test_line_can_push_only_on_whole_numbers():
    assert line_can_push(24.0) is True
    assert line_can_push(24.5) is False
    # An unknown line cannot be SHOWN to be a half-line, so it refuses.
    assert line_can_push(None) is True
    assert line_can_push(float("nan")) is True


def test_half_line_under_is_the_exact_complement():
    probs = side_probabilities(p_over=0.55, line=24.5)
    assert probs["under"][0] == pytest.approx(0.45)
    assert probs["under"][1] is None


def test_whole_line_refuses_the_complement():
    """
    1 - P(over) is P(under) + P(push). On a whole line that books every push
    as an under win and inflates the under's EV by exactly the push mass.
    """
    probs = side_probabilities(p_over=0.55, line=24.0)
    assert probs["under"][0] is None
    assert "push" in probs["under"][1].lower()


def test_unknown_line_refuses_the_complement_too():
    probs = side_probabilities(p_over=0.55, line=None)
    assert probs["under"][0] is None
    assert "half-line" in probs["under"][1]


def test_explicit_under_probability_is_used_as_given():
    probs = side_probabilities(p_over=0.58, p_under=0.34, p_push=0.08, line=24.0)
    assert probs["under"][0] == pytest.approx(0.34)
    assert probs["under"][1] is None


def test_a_refused_under_does_not_cost_the_over_its_ev():
    """
    The de-vig needs both PRICES, but each side's EV needs only its own
    probability. A whole line with no P(under) still yields a real over EV.
    """
    whole = _snap("propline", line=8.0)
    rows = {r.side: r for r in expand_sides(
        event_id="G1", target_market="REB", model_p_over=0.58,
        research_line=8.0, resolution=resolve_market([whole]),
    )}
    assert rows["over"].decision_basis == "book_ev"
    assert rows["over"].book_ev is not None and rows["over"].book_ev > 0
    assert rows["under"].decision_basis == "unavailable"
    assert rows["under"].book_ev is None
    # Not comparable, so no side is preferred.
    assert rows["over"].preferred_side is None


def test_push_mass_does_not_change_the_over_ev():
    """Whether the under is priced must not move the over's number."""
    whole = _snap("propline", line=8.0)
    without = {r.side: r for r in expand_sides(
        target_market="REB", model_p_over=0.58, research_line=8.0,
        resolution=resolve_market([whole]),
    )}
    with_push = {r.side: r for r in expand_sides(
        target_market="REB", model_p_over=0.58, model_p_under=0.34,
        model_p_push=0.08, research_line=8.0, resolution=resolve_market([whole]),
    )}
    assert without["over"].book_ev == pytest.approx(with_push["over"].book_ev)
    assert with_push["under"].book_ev is not None
    assert with_push["over"].preferred_side == "over"


# --- board semantics -----------------------------------------------------


def test_both_sides_are_always_expanded():
    rows = expand_sides(target_market="PTS", model_p_over=0.58, research_line=24.5)
    assert [r.side for r in rows] == ["over", "under"]


def test_preferred_side_only_when_both_sides_are_priced():
    priced = {r.side: r for r in expand_sides(
        target_market="PTS", model_p_over=0.58, research_line=24.5,
        resolution=resolve_market([PROPLINE]),
    )}
    assert priced["over"].preferred_side == "over"
    assert priced["under"].preferred_side == "over"

    unpriced = expand_sides(target_market="PTS", model_p_over=0.58, research_line=24.5)
    assert all(r.preferred_side is None for r in unpriced)


def test_pickem_never_produces_book_ev():
    rows = expand_sides(
        target_market="PTS", model_p_over=0.58, research_line=24.5,
        resolution=resolve_market([PROPLINE_PICKEM]),
    )
    assert {r.decision_basis for r in rows} == {"pickem_line_only"}
    assert all(r.book_ev is None for r in rows)
    assert all(r.decision_status == "ABSTAIN" for r in rows)
    assert all(r.pickem_line == pytest.approx(24.5) for r in rows)


def test_min_ev_controls_consider():
    kwargs = dict(
        target_market="PTS", model_p_over=0.58, research_line=24.5,
        resolution=resolve_market([PROPLINE]),
    )
    loose = {r.side: r for r in expand_sides(min_ev=0.0, **kwargs)}
    assert loose["over"].decision_status == "CONSIDER"

    strict = {r.side: r for r in expand_sides(min_ev=0.50, **kwargs)}
    assert strict["over"].decision_status == "ABSTAIN"
    assert strict["over"].book_ev is not None     # the EV is still reported


def test_a_model_lean_never_outranks_a_priced_edge():
    """
    Bands sit between status and score. Sorting a lean and a de-vigged EV on
    one numeric scale would imply the two numbers are comparable.
    """
    lean = expand_sides(
        player_name="Leaner", target_market="PTS", model_p_over=0.99,
        research_line=24.5,
    )
    priced = expand_sides(
        player_name="Priced", target_market="PTS", model_p_over=0.58,
        research_line=24.5, resolution=resolve_market([PROPLINE]),
    )
    ranked = rank_decisions([*lean, *priced], consider_only=True)

    # The lean is far larger in magnitude (0.49 past even vs an EV of 0.08)
    # and still ranks below, because the bands never interleave.
    assert ranked[0].decision_basis == "book_ev"
    assert ranked[0].rank == 1
    ev_ranks = [r.rank for r in ranked if r.decision_basis == "book_ev"]
    lean_ranks = [r.rank for r in ranked if r.decision_basis == "model_lean"]
    assert max(ev_ranks) < min(lean_ranks)
    assert max(r.lean_score for r in ranked if r.decision_basis == "model_lean") > 0.4


def test_require_valid_book_demotes_unpriced_rows():
    rows = expand_sides(target_market="PTS", model_p_over=0.80, research_line=24.5)
    assert any(r.decision_status == "CONSIDER" for r in rows)

    gated = rank_decisions(rows, require_valid_book=True)
    assert all(r.decision_status == "ABSTAIN" for r in gated)
    assert "require-valid-book" in gated[0].why


def test_consider_only_and_top_n():
    rows = expand_sides(
        target_market="PTS", model_p_over=0.58, research_line=24.5,
        resolution=resolve_market([PROPLINE]),
    )
    assert len(rank_decisions(rows, consider_only=True)) == 1
    assert len(rank_decisions(rows, top_n=1)) == 1


def test_the_board_never_claims_an_outcome():
    from src.quant.decision_board import _assert_no_claims

    for word in ("lock", "guaranteed", "best bet"):
        with pytest.raises(DecisionBoardError):
            _assert_no_claims(f"this is a {word}")

    rows = expand_sides(
        target_market="PTS", model_p_over=0.99, research_line=24.5,
        resolution=resolve_market([PROPLINE]),
    )
    for row in rows:
        lowered = row.why.lower()
        assert not any(w in lowered for w in FORBIDDEN_CLAIM_WORDS)


def test_layer_cannot_reach_a_book_or_size_a_stake():
    """MANUAL_ONLY is a property of the code, not just of the docstring."""
    import inspect

    from src.quant import decision_board

    source = inspect.getsource(decision_board)
    for banned in ("import requests", "http://", "urllib", "kelly", "stake_size"):
        assert banned not in source.lower(), f"decision_board must not contain {banned!r}"

    rows = expand_sides(target_market="PTS", model_p_over=0.58, research_line=24.5)
    assert not [f for f in rows[0].model_dump() if "stake" in f or "bankroll" in f]


# --- handoff to the manual log ------------------------------------------


def test_log_fields_for_an_over_carry_p_over():
    rows = {r.side: r for r in expand_sides(
        event_id="G1", target_market="PTS", model_p_over=0.58, research_line=24.5,
        resolution=resolve_market([PROPLINE]),
    )}
    fields = decision_log_fields(rows["over"])
    assert fields["side"] == "over"
    assert fields["model_prob"] == pytest.approx(0.58)
    assert fields["model_prob_side"] == pytest.approx(0.58)
    assert fields["odds"] == -115
    assert "stake" not in " ".join(k for k in fields if k != "note")


def test_log_fields_invert_an_under_only_on_a_half_line():
    half = {r.side: r for r in expand_sides(
        target_market="PTS", model_p_over=0.58, research_line=24.5,
        resolution=resolve_market([PROPLINE]),
    )}
    fields = decision_log_fields(half["under"])
    assert fields["model_prob_side"] == pytest.approx(0.42)
    assert fields["model_prob"] == pytest.approx(0.58)   # back to P(over)

    whole = {r.side: r for r in expand_sides(
        target_market="REB", model_p_over=0.58, model_p_under=0.34,
        model_p_push=0.08, research_line=8.0,
        resolution=resolve_market([_snap("propline", line=8.0)]),
    )}
    whole_fields = decision_log_fields(whole["under"])
    assert whole_fields["model_prob_side"] == pytest.approx(0.34)
    # 1 - 0.34 = 0.66 would be P(over) + P(push), not P(over). Refuse it.
    assert whole_fields["model_prob"] is None


# --- end to end ----------------------------------------------------------


def test_board_from_slate_rows_and_csv(tmp_path):
    slate = [
        {
            "slate_date": "2026-01-02", "event_id": "G1", "player_id": "p1",
            "player_name": "A", "target_market": "PTS", "research_line": 24.5,
            "model_p_over": 0.58,
        },
        {
            "slate_date": "2026-01-02", "event_id": "G2", "player_id": "p2",
            "player_name": "B", "target_market": "REB", "research_line": 8.0,
            "model_p_over": 0.55,
        },
    ]
    markets = {("G1", "p1", "PTS"): [ODDSPAPI, PROPLINE]}
    board = decision_board_from_slate(slate, markets=markets)

    assert len(board) == 4                      # both sides of both rows
    assert [r.rank for r in board] == [1, 2, 3, 4]
    priced = [r for r in board if r.decision_basis == "book_ev"]
    assert {r.book_source for r in priced} == {"propline"}

    # G2 is a whole line with no P(under): over leans, under abstains.
    g2 = {r.side: r for r in board if r.event_id == "G2"}
    assert g2["over"].decision_basis == "model_lean"
    assert g2["under"].decision_basis == "unavailable"

    summary = board_summary(board)
    assert summary["placement_mode"] == PLACEMENT_MODE
    assert summary["sources_used"] == ["propline"]
    assert DECISION_DISCLAIMER in summary["disclaimer"]

    out = tmp_path / "decision_board.csv"
    assert write_decision_board_csv(board, out) == 4
    frame = pd.read_csv(out)
    for column in ("side", "decision_status", "decision_basis", "book_ev",
                   "preferred_side", "why", "rank"):
        assert column in frame.columns


# --- the push rule where it was actually wrong ---------------------------


def test_enrich_row_with_book_refuses_the_complement_on_a_whole_line():
    """
    ``enrich_row_with_book`` used to price the under with 1 - P(over). On a
    whole line that is P(under) + P(push), so the under's EV was inflated by
    the push mass on every integer line.
    """
    from src.quant.paper_research import ResearchSlateRow, enrich_row_with_book

    whole = _snap("propline", line=8.0, market="REB")
    row = ResearchSlateRow(
        slate_date="2026-01-02", event_id="G1", player_id="p1",
        target_market="REB", model_p_over=0.58,
    )
    out = enrich_row_with_book(row, whole)
    assert out.book_status == "VALID"
    assert out.book_ev_over is not None          # the over is still priced
    assert out.book_ev_under is None             # the under is not invented
    assert any("push" in w.lower() for w in out.warnings)

    # With a real P(under) supplied, the under prices normally.
    with_under = enrich_row_with_book(
        row.model_copy(update={"model_p_under": 0.34, "model_p_push": 0.08}), whole
    )
    assert with_under.book_ev_under is not None


def test_enrich_row_with_book_still_uses_the_complement_on_a_half_line():
    from src.quant.paper_research import ResearchSlateRow, enrich_row_with_book

    row = ResearchSlateRow(
        slate_date="2026-01-02", event_id="G1", player_id="p1",
        target_market="PTS", model_p_over=0.58,
    )
    out = enrich_row_with_book(row, PROPLINE)
    assert out.book_ev_over is not None
    assert out.book_ev_under is not None
    assert not any("push" in w.lower() for w in out.warnings)


def _settled(store, *, side: str, line: float, won: bool, p_over: float,
             p_side: float | None):
    from src.quant.historical_store import BetLifecycleRecord

    record = BetLifecycleRecord(
        game_id=f"G{line}{side}{won}{p_over}", prop_stat="PTS", line=line,
        bet_side=side, taken_odds_american=-110, model_prob=p_over,
        model_prob_side=p_side, bet_result="WIN" if won else "LOSS",
        profit_loss=0.909 if won else -1.0,
    )
    return store.append(record, allow_duplicate_pending=True)


def test_paper_report_prefers_the_recorded_side_probability(tmp_path):
    """
    An under bet's calibration must use P(under) as recorded, not
    1 - P(over), which on a whole line is too high by the push mass.
    """
    from src.quant.historical_store import HistoricalStore, HistoricalStoreConfig
    from src.quant.paper_research import paper_improvement_report

    store = HistoricalStore(HistoricalStoreConfig(root=tmp_path, use_sqlite=False))
    for i in range(12):
        _settled(store, side="under", line=8.0 + i * 0.01 * 0, won=i % 2 == 0,
                 p_over=0.58 + i * 0.001, p_side=0.34)
    report = paper_improvement_report(store)
    calib = report["probability_calibration"]
    assert calib["n"] == 12
    # 0.34 as recorded, not 1 - 0.58 = 0.42.
    assert calib["mean_model_prob"] == pytest.approx(0.34, abs=1e-3)


def test_paper_report_skips_whole_line_unders_with_no_side_probability(tmp_path):
    from src.quant.historical_store import HistoricalStore, HistoricalStoreConfig
    from src.quant.paper_research import paper_improvement_report

    store = HistoricalStore(HistoricalStoreConfig(root=tmp_path, use_sqlite=False))
    for i in range(12):
        _settled(store, side="under", line=8.0, won=i % 2 == 0,
                 p_over=0.58 + i * 0.001, p_side=None)
    calib = paper_improvement_report(store)["probability_calibration"]
    assert calib["status"] == "DATA_NOT_AVAILABLE"
    assert calib["skipped_push_ambiguous"] == 12
    assert "push mass" in calib["reason"]
