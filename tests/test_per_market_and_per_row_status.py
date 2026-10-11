"""B5 and B6 from the pre-flight audit: one market per run, one status per slate.

Two defects that both made a per-row column say something about a different
row, or about no row at all.

B5 — ONE MARKET PER RUN. ``score_prob_over`` takes a single ``model_path`` and
``assemble_projections`` writes its output only onto the market the artifact's
sidecar names. That part is right: copying a points model's P(Over) onto the
rebound frame would publish one market's probability as another's. The
consequence was that a single run could never produce a probability for more
than ONE market — AST, REB and FG3M came out null, and
``settlement.recorder`` skips a row with no probability, so three of four
markets were unrecordable no matter how many models had been trained.
``resolve_model_artifact`` already took a ``market`` and globbed
``xgboost_{MARKET}.json``; nothing called it that way.

B6 — ONE STATUS PER SLATE. ``evaluate_ev_gate`` asks the gate about each
captured prop line and then collapsed the answers into one status.
``assemble_projections`` wrote that onto every row and
``repository.persist_projections`` stores it per row. One priced prop out of
three hundred labelled all three hundred READY_FOR_EVALUATION.

AND THE GATE WAS BEING ASKED AN IMPOSSIBLE QUESTION. The loop built a
``MarketContext`` without ``line``, and ``market_ev_gate`` refuses a context
with no finite line — "EV is a claim about a probability at a specific
number". So every row abstained for a reason that was this function's own
doing, and a perfect two-way price would have abstained too. ``market``,
``player_name``, ``is_pickem`` and ``payout_multiplier`` were dropped on the
same floor, so pick'em rows were never routed either.
"""

from __future__ import annotations

import pandas as pd
import pytest

from main import (
    DEFAULT_STATS,
    GATE_NO_LINE_REASON,
    assemble_projections,
    evaluate_ev_gate,
)

MARKETS = ("PTS", "REB", "AST")


def features(n: int = 4) -> pd.DataFrame:
    frame = pd.DataFrame({
        "PLAYER_ID": [f"20000{i}" for i in range(n)],
        "PLAYER_NAME": [f"P{i}" for i in range(n)],
        "GAME_ID": ["0022500001"] * n,
        "GAME_DATE": pd.to_datetime(["2026-10-05"] * n),
        "fatigue_multiplier": [1.0] * n,
    })
    for stat in DEFAULT_STATS:
        frame[f"{stat}_BASELINE"] = 20.0
        frame[f"{stat}_L2"] = 20.0
    return frame


def prop_lines(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _line(player, market, *, line=24.5, over=-110, under=-110,
          status="VALID", is_pickem=False, payout=None):
    return {
        "source": "propline", "player_name": player, "market": market,
        "line": line, "over_odds_american": over, "under_odds_american": under,
        "status": status, "is_pickem": is_pickem, "payout_multiplier": payout,
        "nba_game_id": "0022500001", "captured_at_utc": None,
    }


# --- B6: the gate is asked a question it can answer --------------------

def test_a_real_two_way_price_is_ready_now_that_the_line_is_passed():
    """
    THE DEFECT: MarketContext.line defaulted to None and the loop never set
    it, so market_ev_gate refused every row for a reason the caller created.
    A perfect price abstained.
    """
    verdict = evaluate_ev_gate(
        prop_lines([_line("P0", "PTS")]), pd.DataFrame()
    )
    assert verdict["ready"] == 1
    assert verdict["abstained"] == 0
    assert verdict["by_key"][("P0", "PTS")]["status"] == "READY_FOR_EVALUATION"


def test_a_line_that_is_not_finite_is_still_refused():
    verdict = evaluate_ev_gate(
        prop_lines([_line("P0", "PTS", line=None)]), pd.DataFrame()
    )
    assert verdict["ready"] == 0
    assert "line" in verdict["by_key"][("P0", "PTS")]["reason"].lower()


def test_a_one_sided_price_is_refused():
    verdict = evaluate_ev_gate(
        prop_lines([_line("P0", "PTS", under=None)]), pd.DataFrame()
    )
    assert verdict["ready"] == 0
    assert "two-way" in verdict["by_key"][("P0", "PTS")]["reason"].lower()


def test_a_pickem_row_is_routed_rather_than_dead_ended():
    """
    is_pickem and payout_multiplier were dropped too, so a pick'em row looked
    like an ordinary one. A payout multiplier is not a two-way price.
    """
    from src.quant.contracts import PICKEM_ENTRY_ROUTE

    verdict = evaluate_ev_gate(
        prop_lines([_line("P0", "PTS", is_pickem=True, payout=3.0)]),
        pd.DataFrame(),
    )
    entry = verdict["by_key"][("P0", "PTS")]
    assert entry["status"] == "DATA_NOT_AVAILABLE"
    assert entry["route"] == PICKEM_ENTRY_ROUTE
    assert "dfs_payouts" in entry["reason"]


def test_the_pickem_FLAG_alone_routes_without_a_multiplier():
    """
    market_ev_gate fires on `is_pickem OR payout_multiplier is not None`, so a
    fixture carrying BOTH cannot tell which one is wired — the first version of
    the test above passed with `is_pickem=` deleted from the context. A pick'em
    row with the flag and no multiplier isolates the flag, and is a real shape:
    an operator can mark a board pick'em without publishing a multiple.
    """
    from src.quant.contracts import PICKEM_ENTRY_ROUTE

    verdict = evaluate_ev_gate(
        prop_lines([_line("P0", "PTS", is_pickem=True, payout=None)]),
        pd.DataFrame(),
    )
    entry = verdict["by_key"][("P0", "PTS")]
    assert entry["route"] == PICKEM_ENTRY_ROUTE, (
        "the is_pickem flag is not reaching the gate"
    )


def test_a_multiplier_alone_routes_without_the_flag():
    """The other half, so neither input can be dropped unnoticed."""
    from src.quant.contracts import PICKEM_ENTRY_ROUTE

    verdict = evaluate_ev_gate(
        prop_lines([_line("P0", "PTS", is_pickem=False, payout=3.0)]),
        pd.DataFrame(),
    )
    assert verdict["by_key"][("P0", "PTS")]["route"] == PICKEM_ENTRY_ROUTE


# --- B6: the status lands per row --------------------------------------

def test_one_priced_prop_does_not_label_the_whole_slate_ready():
    """
    THE DEFECT, exactly. One ready prop out of many wrote
    READY_FOR_EVALUATION onto every row of every market.
    """
    feats = features(3)
    lines = prop_lines([_line("P0", "PTS")])          # one priced prop
    verdict = evaluate_ev_gate(lines, pd.DataFrame())
    out = assemble_projections(
        feats, {}, verdict, prop_lines=lines, stats=MARKETS
    )
    ready = out[out["MARKET_STATUS"] == "READY_FOR_EVALUATION"]
    assert len(ready) == 1, out[["PLAYER_NAME", "MARKET", "MARKET_STATUS"]]
    assert ready.iloc[0]["PLAYER_NAME"] == "P0"
    assert ready.iloc[0]["MARKET"] == "PTS"
    # Everything else abstains, and the slate aggregate is NOT what landed.
    assert verdict["status"] == "READY_FOR_EVALUATION"
    assert (out["MARKET_STATUS"] == "DATA_NOT_AVAILABLE").sum() == len(out) - 1


def test_each_row_gets_its_own_verdict_not_the_first_one():
    feats = features(2)
    lines = prop_lines([
        _line("P0", "PTS"),                       # ready
        _line("P1", "PTS", under=None),           # refused: one-sided
    ])
    verdict = evaluate_ev_gate(lines, pd.DataFrame())
    out = assemble_projections(
        feats, {}, verdict, prop_lines=lines, stats=("PTS",)
    )
    by_player = out.set_index("PLAYER_NAME")["MARKET_STATUS"].to_dict()
    assert by_player["P0"] == "READY_FOR_EVALUATION"
    assert by_player["P1"] == "DATA_NOT_AVAILABLE"


def test_no_line_and_a_refused_line_are_different_facts():
    """
    Both are DATA_NOT_AVAILABLE and they lead a reader to opposite
    conclusions about whether the market exists at all.
    """
    feats = features(2)
    lines = prop_lines([_line("P0", "PTS", under=None)])   # P1 has no line
    verdict = evaluate_ev_gate(lines, pd.DataFrame())
    out = assemble_projections(
        feats, {}, verdict, prop_lines=lines, stats=("PTS",)
    )
    reasons = out.set_index("PLAYER_NAME")["MARKET_STATUS_REASON"].to_dict()
    assert reasons["P1"] == GATE_NO_LINE_REASON
    assert reasons["P0"] != GATE_NO_LINE_REASON
    assert "two-way" in reasons["P0"].lower()


def test_an_empty_gate_verdict_leaves_every_row_unavailable_with_a_reason():
    feats = features(2)
    out = assemble_projections(
        feats, {}, {"status": "DATA_NOT_AVAILABLE", "by_key": {}},
        prop_lines=None, stats=("PTS",),
    )
    assert (out["MARKET_STATUS"] == "DATA_NOT_AVAILABLE").all()
    assert (out["MARKET_STATUS_REASON"] == GATE_NO_LINE_REASON).all()


def test_the_reason_is_persisted_rather_than_computed_and_dropped():
    """A column main.py computes and nothing stores is the defect this
    repository keeps finding; see migration 007."""
    import inspect

    from src.db import repository
    from src.db.models import Projection

    assert "market_status_reason" in Projection.__table__.columns
    source = inspect.getsource(repository.persist_projections)
    assert '"market_status_reason": r.get("MARKET_STATUS_REASON")' in source
    # And it is refreshed on a re-run, not only inserted.
    assert '"market_status_reason",' in source


# --- B5: every market can carry a probability --------------------------

def _probs(feats: pd.DataFrame, market: str, value: float = 0.56) -> pd.Series:
    series = pd.Series([value] * len(feats), index=feats.index)
    series.attrs["target_market"] = market
    return series


def test_more_than_one_market_can_carry_a_probability_in_one_run():
    """
    THE DEFECT: a single model_path meant exactly one market was ever scored,
    so AST, REB and FG3M rows were null and the recorder skipped them.
    """
    feats = features(2)
    scored = {
        "PTS": _probs(feats, "PTS", 0.56),
        "REB": _probs(feats, "REB", 0.48),
        "AST": _probs(feats, "AST", 0.52),
    }
    out = assemble_projections(
        feats, scored, {"by_key": {}}, prop_lines=None, stats=MARKETS
    )
    got = out.dropna(subset=["PROB_OVER"]).groupby("MARKET")["PROB_OVER"].first()
    assert set(got.index) == set(MARKETS)
    assert got["PTS"] == pytest.approx(0.56)
    assert got["REB"] == pytest.approx(0.48)
    assert got["AST"] == pytest.approx(0.52)


def test_a_market_with_no_artifact_stays_null_rather_than_borrowing_one():
    """The original guard, and it must survive the multi-market change: one
    market's probability must never appear under another's name."""
    feats = features(2)
    out = assemble_projections(
        feats, {"PTS": _probs(feats, "PTS")}, {"by_key": {}},
        prop_lines=None, stats=MARKETS,
    )
    scored = out.dropna(subset=["PROB_OVER"])["MARKET"].unique()
    assert list(scored) == ["PTS"]
    assert out[out["MARKET"] == "REB"]["PROB_OVER"].isna().all()
    assert out[out["MARKET"] == "AST"]["PROB_OVER"].isna().all()


def test_a_bare_series_still_works_and_still_scopes_to_its_own_market():
    """verify_wiring and several tests pass a Series, and an explicit --model
    is genuinely one artifact for one market."""
    feats = features(2)
    out = assemble_projections(
        feats, _probs(feats, "REB", 0.61), {"by_key": {}},
        prop_lines=None, stats=MARKETS,
    )
    scored = out.dropna(subset=["PROB_OVER"])
    assert list(scored["MARKET"].unique()) == ["REB"]
    assert scored["PROB_OVER"].iloc[0] == pytest.approx(0.61)


def test_a_series_with_no_market_attributes_to_nothing():
    feats = features(2)
    naked = pd.Series([0.5] * len(feats), index=feats.index)
    out = assemble_projections(
        feats, naked, {"by_key": {}}, prop_lines=None, stats=MARKETS
    )
    assert out["PROB_OVER"].isna().all()


def test_the_market_keys_are_case_insensitive():
    feats = features(2)
    out = assemble_projections(
        feats, {"pts": _probs(feats, "PTS")}, {"by_key": {}},
        prop_lines=None, stats=("PTS",),
    )
    assert out["PROB_OVER"].notna().all()


def test_the_scorer_resolves_one_artifact_per_market(monkeypatch):
    """
    resolve_model_artifact already took a market and globbed
    xgboost_{MARKET}.json. Nothing called it that way; this asserts it is
    called that way now.
    """
    import main

    asked: list[str | None] = []

    def _resolve(explicit=None, *, market=None):
        asked.append(market)
        return (f"/fake/xgboost_{market}.json", "glob") if market else (None, "none")

    def _score(feats, lines, path):
        market = str(path).split("xgboost_")[1].split(".")[0]
        series = pd.Series([0.5] * len(feats), index=feats.index)
        series.attrs["target_market"] = market
        return series

    monkeypatch.setattr(main, "resolve_model_artifact", _resolve)
    monkeypatch.setattr(main, "score_prob_over", _score)

    out = main.score_prob_over_by_market(
        features(2), prop_lines([_line("P0", "PTS")]), markets=MARKETS
    )
    assert asked == list(MARKETS)
    assert set(out) == set(MARKETS)


def test_an_explicit_artifact_is_scored_once_not_once_per_market(monkeypatch):
    """
    Looping an explicit path over every market would score the same booster
    four times and hand three results to markets it was not fit for — the
    exact mistake the sidecar check exists to prevent.
    """
    import main

    calls: list[object] = []

    def _score(feats, lines, path):
        calls.append(path)
        series = pd.Series([0.5] * len(feats), index=feats.index)
        series.attrs["target_market"] = "PTS"
        return series

    monkeypatch.setattr(main, "score_prob_over", _score)
    out = main.score_prob_over_by_market(
        features(2), prop_lines([_line("P0", "PTS")]),
        explicit="/fake/xgboost_PTS.json", markets=MARKETS,
    )
    assert len(calls) == 1
    assert set(out) == {"PTS"}


def test_a_mislabelled_sidecar_is_skipped_not_misattributed(monkeypatch):
    import main

    def _resolve(explicit=None, *, market=None):
        return (f"/fake/xgboost_{market}.json", "glob")

    def _score(feats, lines, path):
        # Every artifact claims PTS, whatever it was resolved for.
        series = pd.Series([0.5] * len(feats), index=feats.index)
        series.attrs["target_market"] = "PTS"
        return series

    monkeypatch.setattr(main, "resolve_model_artifact", _resolve)
    monkeypatch.setattr(main, "score_prob_over", _score)
    out = main.score_prob_over_by_market(
        features(2), prop_lines([_line("P0", "PTS")]), markets=MARKETS
    )
    assert set(out) == {"PTS"}, "a sidecar claiming another market must be skipped"


def test_a_market_with_no_resolvable_artifact_is_reported(monkeypatch, caplog):
    import logging

    import main

    monkeypatch.setattr(
        main, "resolve_model_artifact",
        lambda explicit=None, *, market=None: (None, "nothing found"),
    )
    with caplog.at_level(logging.INFO):
        out = main.score_prob_over_by_market(
            features(2), prop_lines([_line("P0", "PTS")]), markets=MARKETS
        )
    assert out == {}
    assert "one file per market by design" in caplog.text


def test_main_scores_per_market_and_keeps_by_key_out_of_the_run_summary():
    """
    by_key is one entry per priced (player, market). The run summary is
    persisted as JSON, and putting hundreds of duplicate keys in it would
    bloat every row of pipeline_runs.
    """
    import inspect

    import main

    source = inspect.getsource(main.main)
    assert "score_prob_over_by_market(" in source
    assert 'k != "by_key"' in source
