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


def test_two_different_players_sharing_one_id_are_not_one_player():
    """
    THE COLLAPSE THIS RULE CAN CAUSE, which is worse than the contradiction
    it prevents. The key was `player_id or player_name`, so it trusted the id
    completely: two different players carrying one id -- a fuzzy
    name-to-id resolution mapping two Williamses together, or a shared
    placeholder -- became one key and the second player lost its flag.

    It stayed invisible because the demoted candidate remains on the board
    wearing an ABSTAIN, so only a `consider_only` caller would ever have
    noticed a whole player missing. `tests/test_decision_board`'s
    `test_a_model_lean_never_outranks_a_priced_edge` was that caller, and it
    had been silently losing its entire model-lean side.
    """
    first = _cand("shared-id", "PTS", "over", 24.5, score=2.0)
    second = _cand("shared-id", "PTS", "over", 24.5, score=1.0)
    first.player_name = "First Player"
    second.player_name = "Second Player"

    out, abstained = db.one_recommendation_per_player_market([first, second])
    # `len` FIRST: `all()` over an empty list is true, so without this the
    # test passes against a helper that drops every candidate it is given.
    assert len(out) == 2
    assert abstained == 0, "two different players were treated as one"
    assert all(c.decision_status == "RECOMMENDED" for c in out)
    assert {c.player_name for c in out} == {"First Player", "Second Player"}


def test_two_players_sharing_a_name_are_not_one_player():
    """
    The id half of the key, and why it cannot be dropped in favour of the
    name alone: the NBA has carried exact duplicate names, and the id is the
    authoritative identity field. A mutation removing it survived until this
    test existed.
    """
    first = _cand("p1", "PTS", "over", 24.5, score=2.0)
    second = _cand("p2", "PTS", "over", 24.5, score=1.0)
    first.player_name = second.player_name = "Mike Dunleavy"

    out, abstained = db.one_recommendation_per_player_market([first, second])
    assert len(out) == 2
    assert abstained == 0, "two players with one name were treated as one"
    assert all(c.decision_status == "RECOMMENDED" for c in out)
    assert {c.player_id for c in out} == {"p1", "p2"}


def test_one_player_under_two_spellings_is_the_accepted_cost():
    """
    The other side of that trade, stated so it is a decision and not a
    surprise: the same player reaching the board twice under two spellings
    does NOT dedupe. It takes a duplicate slate row for one (player, market)
    to get there, which is an upstream defect this gate is the wrong place to
    paper over -- and the alternative is suppressing real findings silently.
    """
    first = _cand("p1", "PTS", "over", 24.5, score=2.0)
    second = _cand("p1", "PTS", "under", 27.5, score=1.0)
    first.player_name = "PJ Washington"
    second.player_name = "P.J. Washington"

    out, abstained = db.one_recommendation_per_player_market([first, second])
    assert len(out) == 2
    assert abstained == 0
    assert all(c.decision_status == "RECOMMENDED" for c in out)


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


def test_an_unset_limit_falls_back_to_the_env_default():
    """
    The limit HAS a default (an hour); only the observation time has none.
    A caller that supplies `market_as_of` and no limit therefore gets the
    gate -- which is the point, so a future caller does not have to know to
    ask for it.
    """
    out = db.build_decision_board([_row()], market_as_of=_minutes_ago(43200))
    assert out == [], "a 30-day-old market was published under the default limit"


def test_the_gate_can_be_disabled_deliberately(monkeypatch):
    monkeypatch.setenv(db.ENV_MAX_MARKET_AGE, "off")
    out = db.build_decision_board([_row()], market_as_of=_minutes_ago(43200))
    assert out != [], "'off' did not disable the gate"


def test_an_explicit_limit_with_no_observation_time_warns(caplog):
    """
    The dangerous middle: the caller ASKED for a check and gave nothing to
    check against. Guessing `now()` would make every slate look fresh, so the
    board is built unguarded AND says so.
    """
    with caplog.at_level(logging.WARNING):
        out = db.build_decision_board(
            [_row()], market_as_of=None, max_market_age_minutes=60,
        )
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "NOT CHECKED" in text
    assert "This is not a pass." in text
    # The log line is not the behaviour. Without this a refusal could hide
    # behind the warning: it would say "built unguarded" and return nothing.
    assert out != [], "the board was refused, not built unguarded"


def test_the_env_default_with_no_observation_time_is_silent(caplog):
    """
    Not the same situation: nobody asked for a check here, the limit merely
    has a default. Warning on it would fire on every call from every caller
    that has not been wired yet, which is all of them.
    """
    with caplog.at_level(logging.WARNING):
        db.build_decision_board([_row()])
    assert "NOT CHECKED" not in " ".join(r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("bad", ["not a date", "", object()])
def test_an_unparseable_observation_time_is_treated_as_unknown(bad, caplog):
    with caplog.at_level(logging.WARNING):
        out = db.build_decision_board(
            [_row()], market_as_of=bad, max_market_age_minutes=60,
        )
    assert "NOT CHECKED" in " ".join(r.getMessage() for r in caplog.records)
    # `_row()`, not `[]`: an empty slate returns an empty board whether the
    # gate refused or not, so the old version of this test would have passed
    # against an implementation that treated an unreadable timestamp as a
    # hard refusal -- the opposite of what it claims to assert.
    assert out != [], "an unreadable timestamp became a refusal"


def test_market_age_handles_naive_and_aware_timestamps():
    """A naive observation time against an aware now (or the reverse) must
    not raise -- this runs on the publication path."""
    # naive observation, aware now
    assert db.market_age_minutes("2026-10-20T17:00:00", NOW) == pytest.approx(60.0)
    # aware observation, NAIVE now -- the other branch, which nothing reached
    assert db.market_age_minutes(
        NOW - pd.Timedelta(minutes=45), NOW.tz_localize(None),
    ) == pytest.approx(45.0)
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
    from src.ingestion import propline

    # One player, one market, four outcomes: two books x two points, with
    # each side split across the pair. If the key were player-alone, or
    # dropped the book or the point, these would splice together.
    payload = {
        "id": "0022600001",
        "commence_time": "2026-10-20T23:00:00Z",
        "bookmakers": [
            {
                "key": book,
                "markets": [{
                    "key": "player_points",
                    "outcomes": [
                        {"name": "Over", "description": "A Player",
                         "point": point, "price": -110},
                        {"name": "Under", "description": "A Player",
                         "point": point, "price": -110},
                    ],
                }],
            }
            for book in ("draftkings", "fanduel")
            for point in (24.5, 27.5)
        ],
    }

    rows = propline.normalize_event_props(payload)
    keys = {(r.source, r.market, r.player_name, r.line) for r in rows}
    assert len(rows) == 4, f"rows collapsed or multiplied: {len(rows)}"
    assert keys == {
        ("draftkings", "PTS", "A Player", 24.5),
        ("draftkings", "PTS", "A Player", 27.5),
        ("fanduel", "PTS", "A Player", 24.5),
        ("fanduel", "PTS", "A Player", 27.5),
    }
    # Both sides landed on ONE row per key -- that is what "paired" means.
    for row in rows:
        assert (
            row.over_odds_american is not None
            and row.under_odds_american is not None
        ), f"{row.source} {row.line} did not pair both sides"


def test_the_env_var_is_actually_read(monkeypatch):
    """
    `tests/test_scheduler_worker` caught this: the limit's env var existed
    only as a name in a log line, so an operator could set it, see no change,
    and conclude the gate was broken. A documented knob that does nothing is
    worse than no knob.
    """
    monkeypatch.delenv(db.ENV_MAX_MARKET_AGE, raising=False)
    assert db.market_age_limit_minutes() == db.DEFAULT_MAX_MARKET_AGE_MINUTES
    monkeypatch.setenv(db.ENV_MAX_MARKET_AGE, "90")
    assert db.market_age_limit_minutes() == 90
    assert db.market_age_limit_minutes(15) == 15, "an explicit limit must win"
    for off in ("0", "off", "none", "disabled"):
        monkeypatch.setenv(db.ENV_MAX_MARKET_AGE, off)
        assert db.market_age_limit_minutes() is None, f"{off!r} did not disable"


@pytest.mark.parametrize("bad", ["soon", "inf", "-inf", "nan", "60 minutes", "-60"])
def test_an_unusable_limit_falls_back_rather_than_guessing(bad, caplog, monkeypatch):
    """
    EVERY UNUSABLE VALUE LANDS IN THE SAME PLACE: a warning and the default.

    `inf` is the one that mattered. `int(float("inf"))` raises OverflowError
    where `nan` raises ValueError, so catching ValueError alone let an
    operator's `=inf` crash board construction rather than be refused. And
    `-60` used to return None, switching the staleness gate off silently --
    a typo must never resolve in the direction of "stop checking".
    """
    monkeypatch.setenv(db.ENV_MAX_MARKET_AGE, bad)
    with caplog.at_level(logging.WARNING):
        assert db.market_age_limit_minutes() == db.DEFAULT_MAX_MARKET_AGE_MINUTES
    assert "not a number of minutes" in " ".join(
        r.getMessage() for r in caplog.records
    )


def test_an_explicit_zero_disables_the_gate_rather_than_emptying_the_board():
    """
    A LIMIT OF ZERO IS NOT A LIMIT OF ZERO MINUTES. `explicit` used to return
    straight out, so `max_market_age_minutes=0` -- documented as the way to
    turn the gate off -- set the limit to zero instead and every market on
    earth read as stale. The caller asked for no gate and got an empty board.
    """
    assert db.market_age_limit_minutes(0) is None
    out = db.build_decision_board(
        [_row()], market_as_of=_minutes_ago(43200), max_market_age_minutes=0,
    )
    assert out != [], "an explicit 0 emptied the board instead of disabling the gate"


def test_an_explicit_negative_limit_warns_and_takes_the_default(caplog):
    """Same rule as the env var, because both go through one reader."""
    with caplog.at_level(logging.WARNING):
        assert db.market_age_limit_minutes(-60) == db.DEFAULT_MAX_MARKET_AGE_MINUTES
    assert "not a number of minutes" in " ".join(
        r.getMessage() for r in caplog.records
    )


def test_a_demoted_candidate_does_not_keep_its_place(caplog):
    """
    DEMOTION INVALIDATES THE SORT. The abstain pass runs after `cands.sort()`
    and after the `consider_only` filter, so a candidate demoted to ABSTAIN
    kept the high position it held while it was still RECOMMENDED. `top_n=1`
    then spent its only slot on the abstention and cut the genuine
    recommendation ranked below it, and `consider_only=True` -- which means
    "recommendations only" -- handed back an ABSTAIN.
    """
    rows = [
        # Two lines on ONE player: the second is demoted by the abstain pass.
        _row(research_line=24.5, model_p_over=0.70, model_p_under=0.30),
        _row(research_line=27.5, model_p_over=0.62, model_p_under=0.38),
        # A genuine second recommendation, ranked BELOW the one that gets
        # demoted (lean 0.08 against 0.12) -- so a stale sort puts the
        # abstention above it and `top_n=2` cuts this one instead.
        _row(player_id="p2", player_name="B Player",
             research_line=20.5, model_p_over=0.58, model_p_under=0.42),
    ]

    unfiltered = db.build_decision_board(rows)
    assert [c.decision_status for c in unfiltered].count("ABSTAIN") >= 1, (
        "fixture demoted nothing, so this test proves nothing"
    )

    out = db.build_decision_board(rows, consider_only=True, top_n=2)
    assert len(out) == 2, f"expected two recommendations, got {len(out)}"
    assert all(c.decision_status == "RECOMMENDED" for c in out), (
        "consider_only returned an ABSTAIN: "
        f"{[(c.player_name, c.line, c.decision_status) for c in out]}"
    )
    assert {c.player_name for c in out} == {"Demo", "B Player"}, (
        "the demoted line kept its place and cut a real recommendation: "
        f"{[(c.player_name, c.line) for c in out]}"
    )
    assert [c.rank for c in out] == [1, 2]


def test_a_caller_passing_no_observation_time_is_unaffected():
    """Every caller today passes none, so this change is inert until one is
    wired — stated as a test so that stays true by accident of nobody, not by
    luck."""
    assert db.build_decision_board([_row()]) != []
