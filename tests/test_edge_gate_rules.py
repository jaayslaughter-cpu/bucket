"""
Two refusals borrowed from `Kevocado/NBA_Predictor`'s `odds/edge_gate.py`.

Read 2026-10-18 while auditing four candidate repositories. That module's
reasoning transferred; its code did not (it predicts game outcomes --
169 references to `spread`, 149 to `home_win`). Three rules, of which this
project already satisfied one:

  1. ONE FLAGGED PICK PER EVENT. "Two flagged singles on one event is two
     opinions where only one can be right." For player props the unit is
     (player, market): books post alternate ladders and
     `normalize_event_props` pairs on the point deliberately, so Over 24.5
     and Under 27.5 on one player could both reach the board as independent
     findings. A reader taking both is hedged into the hold.

  2. A STALE SLATE YIELDS ZERO PICKS, NOT A FILTERED SUBSET. "A stale row
     that still clears the bar is the dangerous case: it looks actionable and
     is not." Filtering to the rows that look fresh is worse than refusing,
     because it leaves a short, confident board built on a market nobody can
     vouch for.

  3. Same book, same point, both sides -- ALREADY SATISFIED here, asserted
     below so it stays that way.

Both additions can only shrink the board. RESEARCH_ONLY: this selects rows
and flags none for wagering.
"""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from src.quant import decision_board as db


def _cand(player, market, side, line, status="RECOMMENDED", score=1.0):
    return db.BettingDecisionCandidate(
        slate_date="2026-10-20", event_id="0022600001", player_id=player,
        player_name=player, target_market=market, side=side, line=line,
        decision_status=status, rank_score=score,
    )


# --- rule 1: one recommendation per player and market -----------------------


def test_two_recommended_lines_on_one_player_market_become_one():
    """The alternate-ladder case: Over 24.5 and Under 27.5 on one player."""
    cands = [
        _cand("A Player", "PTS", "over", 24.5, score=2.0),
        _cand("A Player", "PTS", "under", 27.5, score=1.0),
    ]
    out, abstained = db.one_recommendation_per_player_market(cands)
    assert abstained == 1
    flagged = [c for c in out if c.decision_status == "RECOMMENDED"]
    assert len(flagged) == 1
    assert flagged[0].line == 24.5, "the better-ranked line lost the flag"


def test_the_loser_abstains_rather_than_disappearing():
    """Its numbers are still the honest result for that line; what it loses
    is the flag."""
    cands = [
        _cand("A Player", "PTS", "over", 24.5, score=2.0),
        _cand("A Player", "PTS", "under", 27.5, score=1.0),
    ]
    out, _ = db.one_recommendation_per_player_market(cands)
    assert len(out) == 2
    loser = next(c for c in out if c.line == 27.5)
    assert loser.decision_status == "ABSTAIN"


def test_the_loser_says_why():
    cands = [
        _cand("A Player", "PTS", "over", 24.5, score=2.0),
        _cand("A Player", "PTS", "under", 27.5, score=1.0),
    ]
    out, _ = db.one_recommendation_per_player_market(cands)
    loser = next(c for c in out if c.line == 27.5)
    assert any("already" in w and "recommended" in w for w in loser.warnings)


def test_the_status_stays_inside_its_own_type():
    """
    Found while writing this: the first version assigned "CONSIDER", which is
    not in ``DecisionStatus = Literal["RECOMMENDED", "ABSTAIN"]``. Pydantic
    does not catch an out-of-type assignment without validate_assignment, so
    it would have ridden along and surprised every consumer branching on the
    two legal values.
    """
    import typing

    allowed = set(typing.get_args(db.DecisionStatus))
    cands = [
        _cand("A Player", "PTS", "over", 24.5, score=2.0),
        _cand("A Player", "PTS", "under", 27.5, score=1.0),
    ]
    out, _ = db.one_recommendation_per_player_market(cands)
    for cand in out:
        assert cand.decision_status in allowed


def test_different_markets_on_one_player_are_not_the_same_opinion():
    """PTS and REB on one player are two findings, not two opinions about
    one. Only (player, market) collides."""
    cands = [
        _cand("A Player", "PTS", "over", 24.5, score=2.0),
        _cand("A Player", "REB", "over", 8.5, score=1.5),
        _cand("A Player", "AST", "under", 5.5, score=1.0),
    ]
    out, abstained = db.one_recommendation_per_player_market(cands)
    assert abstained == 0
    assert len([c for c in out if c.decision_status == "RECOMMENDED"]) == 3


def test_different_players_in_one_market_are_not_the_same_opinion():
    cands = [
        _cand("A Player", "PTS", "over", 24.5, score=2.0),
        _cand("B Player", "PTS", "over", 19.5, score=1.5),
    ]
    _, abstained = db.one_recommendation_per_player_market(cands)
    assert abstained == 0


def test_an_already_abstaining_duplicate_is_left_alone():
    """The rule thins FLAGGED picks. An abstention is not a competing
    opinion and must not be counted as one."""
    cands = [
        _cand("A Player", "PTS", "over", 24.5, score=2.0),
        _cand("A Player", "PTS", "under", 27.5, status="ABSTAIN", score=1.0),
    ]
    out, abstained = db.one_recommendation_per_player_market(cands)
    assert abstained == 0
    assert len([c for c in out if c.decision_status == "RECOMMENDED"]) == 1


def test_the_rule_runs_after_the_sort_in_the_board():
    """
    Order matters: "the first of a pair" must mean "the better one by the
    board's own ranking", not "whichever row arrived first".
    """
    import inspect

    src = inspect.getsource(db.build_decision_board)
    # The CALL, not the docstring mention of the same name -- the first
    # version of this test compared against the docstring and "passed"
    # whichever order the code was in.
    call = "one_recommendation_per_player_market(cands)"
    assert call in src
    assert src.index("cands.sort(") < src.index(call)


# --- rule 2: a stale slate yields zero, not a subset ------------------------


#: A FIXED instant, for the pure `market_age_minutes` assertions only.
NOW = pd.Timestamp("2026-10-20T18:00:00Z")


def _minutes_ago(minutes: int) -> pd.Timestamp:
    """Relative to the REAL clock, because the gate reads it.

    The first version of these tests offset a hardcoded 2026-10-20, which is
    in the future relative to the machine clock -- so the age came out
    NEGATIVE, the gate never fired, and the tests failed for a reason that had
    nothing to do with the code.
    """
    return pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=minutes)


def _row(**kw):
    """A row that DOES produce candidates, so a zero result means the gate."""
    from src.quant.paper_research import ResearchSlateRow

    base = dict(
        slate_date="2026-10-20", event_id="g1", player_id="p1",
        player_name="Demo", target_market="PTS", research_line=24.5,
        model_p_over=0.62, model_p_under=0.38,
    )
    base.update(kw)
    return ResearchSlateRow(**base)


def test_rows_that_would_produce_candidates_actually_do():
    """
    The control for the test below. Without it, "zero candidates" proves
    nothing: the first version of that test passed an EMPTY row list, so
    deleting the `return []` entirely still produced [] and the mutation
    survived.
    """
    assert db.build_decision_board([_row()]) != []


def test_a_stale_slate_yields_zero_candidates(caplog):
    rows = [_row()]
    assert db.build_decision_board(rows) != [], "the control row stopped working"
    with caplog.at_level(logging.ERROR):
        out = db.build_decision_board(
            rows, market_as_of=_minutes_ago(90), max_market_age_minutes=60,
        )
    assert out == [], "a stale slate still produced candidates"
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "STALE SLATE" in text
    assert "ZERO CANDIDATES" in text


def test_the_refusal_says_it_is_not_a_filtered_subset(caplog):
    """The wording matters because the tempting alternative -- keep the rows
    that look fresh -- is the more dangerous behaviour."""
    with caplog.at_level(logging.ERROR):
        db.build_decision_board(
            [], market_as_of=_minutes_ago(300), max_market_age_minutes=60,
        )
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "rather than the subset" in text


def test_a_fresh_slate_is_not_gated():
    out = db.build_decision_board(
        [_row()], market_as_of=_minutes_ago(5), max_market_age_minutes=60,
    )
    assert out != [], "a 5-minute-old market was gated as stale"


def test_no_limit_means_no_gate():
    """A caller that does not set a limit gets the previous behaviour. The
    gate cannot invent a policy it was not given."""
    out = db.build_decision_board([_row()], market_as_of=_minutes_ago(43200))
    assert out != [], "a 30-day-old market was gated with no limit set"


def test_a_limit_with_no_observation_time_warns_and_does_not_pass_silently(caplog):
    """
    The dangerous middle: a limit is set, the timestamp is missing, and
    nothing is checked. Guessing `now()` would make every slate look fresh,
    so the board is built unguarded AND says so.
    """
    with caplog.at_level(logging.WARNING):
        db.build_decision_board([], market_as_of=None, max_market_age_minutes=60)
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "NOT CHECKED" in text
    assert "This is not a pass." in text


@pytest.mark.parametrize("bad", ["not a date", "", object()])
def test_an_unparseable_observation_time_is_treated_as_unknown(bad, caplog):
    with caplog.at_level(logging.WARNING):
        db.build_decision_board([], market_as_of=bad, max_market_age_minutes=60)
    assert "NOT CHECKED" in " ".join(r.getMessage() for r in caplog.records)


def test_market_age_handles_naive_and_aware_timestamps():
    """A naive observation time against an aware now (or the reverse) must
    not raise -- this runs on the publication path."""
    assert db.market_age_minutes("2026-10-20T17:00:00", NOW) == pytest.approx(60.0)
    assert db.market_age_minutes(NOW - pd.Timedelta(minutes=30), NOW) == pytest.approx(30.0)
    assert db.market_age_minutes(None) is None


def test_the_default_limit_is_an_hour():
    """Following the rule this came from: odds no older than an hour."""
    assert db.DEFAULT_MAX_MARKET_AGE_MINUTES == 60
    assert db.ENV_MAX_MARKET_AGE == "PROPIQ_MAX_MARKET_AGE_MINUTES"


# --- rule 3: already satisfied, and it stays that way -----------------------


def test_pairing_is_same_book_same_point_both_sides():
    """
    "An edge between one book's total and another's is a difference between
    two shops' prices, not a disagreement with the market." This project
    already pairs on (book, market, player, point); the test is here so a
    future change to that key is a deliberate one.
    """
    import inspect

    from src.ingestion import propline

    src = inspect.getsource(propline.normalize_event_props)
    assert "(book_key, stat, player, point_value)" in src
