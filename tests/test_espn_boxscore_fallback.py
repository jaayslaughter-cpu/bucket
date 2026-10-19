"""
The box score we can actually reach, and the fallback that reaches for it.

Measured 2026-10-19, two days before the opener: cdn.nba.com answers 403 in
0.47s with the real headers, stats.nba.com never completes a connection, and
a control host through the same proxy answered — so the refusal is NBA's. With
no box score the settlement loop cannot close, the ledger never fills, and the
publication gate withholds every card for want of evidence that grading is the
only source of.

WHAT IS TESTED HERE IS THE ADAPTER, not an ESPN parser. `src/ingestion/
espn_game` already parses both documented summary shapes, zips header names to
values with strict=True, and splits made-attempted pairs; the first version of
the module under test duplicated all of it, worse. What remains, and what these
tests cover, is deciding WHICH ESPN event is the NBA game being graded.

RESEARCH_ONLY. Every assertion is about reading results that happened.
"""

from __future__ import annotations

import pytest

from src.ingestion.espn_game import BoxScoreRow
from src.settlement import boxscore_fetcher as cdn
from src.settlement import espn_boxscore as espn


class _Summary:
    """A GameSummary stand-in: the adapter reads `box_score` only."""

    def __init__(self, rows):
        self.box_score = list(rows)


def _row(name, *, dnp=False, athlete_id="1", **stats):
    #: Panel stat names, as `espn_game` emits them.
    base = {"MIN": 25.0, "PTS": 8.0, "REB": 6.0, "AST": 0.0, "STL": 0.0,
            "BLK": 0.0, "TOV": 1.0, "FG3M": 1.0, "FGM": 3.0, "FGA": 5.0,
            "FTM": 1.0, "FTA": 3.0}
    base.update(stats)
    return BoxScoreRow(
        espn_event_id="401811026", espn_athlete_id=athlete_id, player_name=name,
        espn_team_id="8", did_not_play=dnp, stats=base, raw={},
    )


# --- mapping a parsed row onto settlement's contract ------------------------


def test_panel_names_become_the_cdn_field_names_settlement_reads():
    """
    Settlement reads the CDN's field names verbatim so no renaming layer can
    drift, which means the translation has to happen in exactly one place.
    """
    out = espn.settlement_rows(_Summary([_row("Tobias Harris")]))
    h = out["Tobias Harris"]
    assert (h["points"], h["reboundsTotal"], h["assists"]) == (8, 6, 0)
    assert (h["threePointersMade"], h["steals"], h["blocks"], h["turnovers"]) == (1, 0, 0, 1)
    assert (h["fieldGoalsMade"], h["fieldGoalsAttempted"]) == (3, 5)
    assert (h["freeThrowsMade"], h["freeThrowsAttempted"]) == (1, 3)
    assert h["minutes_played"] == 25.0
    assert h["did_not_play"] is False
    assert h["source"] == "espn"


def test_an_espn_athlete_id_never_poses_as_an_nba_person_id():
    """
    ESPN numbers athletes itself. Putting that in `nba_player_id` hands every
    downstream reader an ESPN id in a field whose name promises an NBA one — a
    join against player_game_logs would match nothing, or the wrong person.
    """
    h = espn.settlement_rows(_Summary([_row("Tobias Harris", athlete_id="6440")]))["Tobias Harris"]
    assert h["nba_player_id"] is None
    assert h["espn_athlete_id"] == "6440"


def test_a_dnp_is_voided_not_zeroed():
    """Zeroing a DNP grades every Over as a LOSS — inventing losses."""
    out = espn.settlement_rows(_Summary([
        _row("Bench Guy", dnp=True, MIN=0.0, PTS=0.0, REB=0.0, AST=0.0),
    ]))
    assert out["Bench Guy"]["did_not_play"] is True
    assert out["Bench Guy"]["minutes_status"] == "DNP_DECLARED"


def test_unreadable_minutes_with_real_stats_is_not_a_dnp():
    """
    The CDN path learned this the hard way: folding a failed minutes parse into
    DNP voided every player whose minutes format changed, against a box score
    showing they had played, and those rows left W-L entirely.
    """
    h = espn.settlement_rows(_Summary([_row("Tobias Harris", MIN=None)]))["Tobias Harris"]
    assert h["minutes_status"] == "UNAVAILABLE"
    assert h["did_not_play"] is False, "a player with 8 points was voided"


def test_duplicate_names_are_marked_ambiguous():
    """
    This path has NO personId to settle by, so detection is the only
    protection left: the matcher must refuse rather than pick.
    """
    out = espn.settlement_rows(_Summary([
        _row("Same Name", athlete_id="1"), _row("Same Name", athlete_id="2"),
    ]))
    assert out["Same Name"]["ambiguous_name"] is True


def test_a_stat_espn_omitted_stays_absent_rather_than_becoming_zero():
    """
    `PTS=None` is a PRESENT key with a null value, which is not the case this
    claims to cover: a regression that mishandles a MISSING key would pass.
    The key is removed.
    """
    row = _row("Partial")
    del row.stats["PTS"]
    h = espn.settlement_rows(_Summary([row]))["Partial"]
    assert h["points"] is None, "an omitted stat was invented as a zero"


@pytest.mark.parametrize("raw,expected", [
    (8.0, 8), (0.0, 0), (24.0, 24),
    (3.5, None), (0.5, None), (float("inf"), None), (float("nan"), None),
    ("x", None), (None, None),
    # Negative counting stats are impossible, not merely unusual. Every field
    # this feeds is a count; PLUS_MINUS is the one signed figure ESPN sends
    # and it is deliberately absent from PANEL_TO_CDN, so nothing legitimately
    # negative reaches here. Settling an Under on a -3 would be a real-looking
    # result built on impossible input.
    (-1.0, None), (-3.0, None), (-0.0, 0),
])
def test_a_fractional_counting_stat_is_refused_not_rounded(raw, expected):
    """
    THE ROUNDING MODE DECIDED A W/L. This was `int(round(...))`, so a
    malformed 3.5 points became a confident 4 — and on a 3.5 line that is the
    difference between WIN and LOSS. A player cannot score half a point, so a
    fractional counting stat is corrupt input, and the only honest reading of
    corrupt input is none.
    """
    assert espn._int_or_none(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    (25.0, 25.0), (0.0, 0.0),
    (-5.0, None), (float("inf"), None), (float("nan"), None), ("??", None),
])
def test_minutes_must_be_finite_and_non_negative(raw, expected):
    """
    A bare `float()` let -5 and inf through as "parsed", so malformed
    availability data read as a row that played. Anything unusable takes the
    unavailable-minutes path, which is already written for this uncertainty.
    """
    assert espn._minutes_or_none(raw) == expected


def test_negative_minutes_do_not_read_as_a_played_row():
    """The whole point of the guard above, at the level that matters."""
    h = espn.settlement_rows(_Summary([_row("Odd", MIN=-5.0)]))["Odd"]
    assert h["minutes_played"] is None
    assert h["minutes_status"] == "UNAVAILABLE"


def test_names_differing_only_by_punctuation_are_still_ambiguous():
    """
    THE GUARD WAS BLIND TO THE COLLISION IT EXISTS TO CATCH. Keyed on the raw
    name, "P.J. Washington" and "PJ Washington" were two entries — and
    `_match_player` normalises, so it would have matched whichever it reached
    first and graded one player against the other's line.
    """
    out = espn.settlement_rows(_Summary([
        _row("P.J. Washington", athlete_id="1"),
        _row("PJ Washington", athlete_id="2"),
    ]))
    assert any(v["ambiguous_name"] for v in out.values()), (
        "a normalised-identical pair was not flagged"
    )
    assert all(v["ambiguous_name"] for v in out.values()), (
        "only one side of the collision was flagged, so the matcher can still pick"
    )


def test_a_diacritic_only_difference_matches():
    """
    `Nikola Jokić` in an ESPN box score must reach a pending `Nikola Jokic`
    prop. The settlement normaliser re-exports `id_crosswalk`'s rule, which
    strips combining marks; a fourth hand-rolled rule did not, and this is the
    case it exists for.
    """
    cands = _cands({"A": ["Nikola Jokić"], "B": ["Someone Else"]})
    assert espn.select_event_by_players(cands, {"Nikola Jokic"}) == "A"


# --- identifying which ESPN event is our game -------------------------------


def _cands(mapping):
    return {eid: {name: {} for name in names} for eid, names in mapping.items()}


def test_the_event_containing_every_graded_player_is_chosen():
    cands = _cands({"A": ["Alpha One", "Alpha Two"], "B": ["Beta One", "Beta Two"]})
    assert espn.select_event_by_players(cands, {"Beta One", "Beta Two"}) == "B"


def test_a_unique_PARTIAL_overlap_is_refused():
    """
    THE CATASTROPHE THIS FORECLOSES, and the first version allowed it. If the
    game we want is not final yet and some other game that night contains one
    same-named player, a partial rule picks that event and grades a whole
    game's props against a different game's box score. Overlap of 1, unique,
    and completely wrong.
    """
    cands = _cands({"A": ["Alpha One", "Alpha Two"], "B": ["Beta One", "Beta Two"]})
    with pytest.raises(espn.EspnBoxScoreError, match="split across events"):
        espn.select_event_by_players(cands, {"Alpha One", "Beta One"})


def test_a_player_on_no_box_score_anywhere_does_not_block_the_game():
    """
    A scratch who did not travel reaches no box score at all. Requiring them
    would make the whole game ungradeable forever, so they are excluded from
    the requirement and REPORTED — they simply never settle.
    """
    cands = _cands({"A": ["Alpha One", "Alpha Two"]})
    assert espn.select_event_by_players(
        cands, {"Alpha One", "Alpha Two", "Never Travelled"},
    ) == "A"


def test_matching_uses_the_same_normaliser_the_matcher_settles_with():
    """
    A weaker rule here than in `_match_player` means the SELECTOR can fail to
    recognise a player the MATCHER would have matched, so a punctuation-only
    difference leaves a night PENDING for a reason nothing logged.
    """
    cands = _cands({"A": ["P.J. Washington"], "B": ["Someone Else"]})
    assert espn.select_event_by_players(cands, {"PJ Washington"}) == "A"


def test_two_events_both_containing_everyone_is_refused():
    cands = _cands({"A": ["Shared One"], "B": ["Shared One"]})
    with pytest.raises(espn.EspnBoxScoreError, match="Refusing to pick"):
        espn.select_event_by_players(cands, {"Shared One"})


def test_no_overlap_at_all_is_refused():
    with pytest.raises(espn.EspnBoxScoreError, match="appear in any"):
        espn.select_event_by_players(_cands({"A": ["Alpha"]}), {"Gamma"})


@pytest.mark.parametrize("players", [set(), {""}, {"   "}])
def test_no_usable_expected_players_is_refused(players):
    """The identification method's one input. Without it there is no method."""
    with pytest.raises(espn.EspnBoxScoreError):
        espn.select_event_by_players(_cands({"A": ["Alpha"]}), players)


# --- the slate is fetched once per run, not once per game -------------------


def test_the_slate_is_fetched_once_for_the_whole_run(monkeypatch):
    """
    NOT AN OPTIMISATION, A RATE-LIMIT GUARD. Without the cache a twelve-game
    CDN outage re-fetched the whole slate per game — about 156 requests against
    a host whose 429 would then strand the grading this exists to rescue.
    """
    calls = {"scoreboard": 0, "summary": 0}

    class _Game:
        espn_event_id = "A"
        home_team = "CHA"

    def _scoreboard(d, session=None):
        calls["scoreboard"] += 1
        return [_Game()]

    def _summary(eid, session=None):
        calls["summary"] += 1
        return _Summary([_row("Alpha One"), _row("Alpha Two")])

    monkeypatch.setattr(espn, "final_events_for_date", _scoreboard)
    monkeypatch.setattr(espn, "fetch_summary", _summary)

    cache: dict = {}
    for _ in range(5):
        rows, _meta = espn.fetch_espn_player_stats_for_game(
            "2026-04-10", {"Alpha One", "Alpha Two"}, cache=cache,
        )
        assert "Alpha One" in rows
    assert calls == {"scoreboard": 1, "summary": 1}, (
        f"the slate was re-fetched per game: {calls}"
    )


def test_no_final_events_reads_as_not_final(monkeypatch):
    monkeypatch.setattr(espn, "final_events_for_date", lambda d, session=None: [])
    with pytest.raises(espn.EspnGameNotFinalError):
        espn.fetch_espn_player_stats_for_game("2026-04-10", {"X"}, cache={})


# --- the fallback's precedence ----------------------------------------------


def test_the_cdn_wins_when_it_answers(monkeypatch):
    """The CDN is exact and carries the personId. It is only unreachable."""
    monkeypatch.setattr(cdn, "fetch_final_boxscore", lambda *a, **k: {"game": {}})
    monkeypatch.setattr(cdn, "extract_player_stats", lambda p: {"From": {"source": "cdn"}})

    def _never(*a, **k):
        raise AssertionError("ESPN was called while the CDN was answering")

    monkeypatch.setattr(espn, "fetch_espn_player_stats_for_game", _never)
    stats, _ = cdn.fetch_player_stats_for_game(
        "0022600001", game_date="2026-10-21", expected_players={"From"},
    )
    assert stats["From"]["source"] == "cdn"


def test_not_final_is_authoritative_and_does_not_try_espn(monkeypatch):
    """
    The game genuinely has not finished. Asking a second source for a verdict
    we already have would only turn up a disagreement to act on.
    """
    def _not_final(*a, **k):
        raise cdn.GameNotFinalError("still playing")

    monkeypatch.setattr(cdn, "fetch_final_boxscore", _not_final)

    def _never(*a, **k):
        raise AssertionError("ESPN was called for a game that is not final")

    monkeypatch.setattr(espn, "fetch_espn_player_stats_for_game", _never)
    with pytest.raises(cdn.GameNotFinalError):
        cdn.fetch_player_stats_for_game(
            "0022600001", game_date="2026-10-21", expected_players={"X"},
        )


def test_a_403_falls_back_to_espn(monkeypatch):
    """The actual production case: cdn.nba.com 403."""
    def _forbidden(*a, **k):
        raise cdn.BoxScoreError("403 from NBA CDN — bot detection")

    monkeypatch.setattr(cdn, "fetch_final_boxscore", _forbidden)
    monkeypatch.setattr(
        espn, "fetch_espn_player_stats_for_game",
        lambda d, p, session=None, cache=None: ({"Harris": {"source": "espn"}}, {}),
    )
    stats, _ = cdn.fetch_player_stats_for_game(
        "0022600001", game_date="2026-04-10", expected_players={"Harris"},
    )
    assert stats["Harris"]["source"] == "espn"


@pytest.mark.parametrize("date_arg,players", [
    (None, {"X"}), ("2026-10-21", set()), (None, set()),
])
def test_without_its_inputs_the_fallback_is_not_attempted(monkeypatch, date_arg, players):
    """
    It refuses to guess which game it is reading, the same way the stale-market
    gate refuses to invent an observation time. The CDN's own error surfaces,
    so the cause is not replaced by a vaguer one.
    """
    def _forbidden(*a, **k):
        raise cdn.BoxScoreError("403 from NBA CDN")

    monkeypatch.setattr(cdn, "fetch_final_boxscore", _forbidden)

    def _never(*a, **k):
        raise AssertionError("ESPN was attempted without the inputs it needs")

    monkeypatch.setattr(espn, "fetch_espn_player_stats_for_game", _never)
    with pytest.raises(cdn.BoxScoreError, match="403 from NBA CDN"):
        cdn.fetch_player_stats_for_game(
            "0022600001", game_date=date_arg, expected_players=players,
        )


@pytest.mark.parametrize("espn_reason,expected", [
    ("None of the 3 player(s) being graded appear in any final event", cdn.GameNotFinalError),
    ("The player set is split across events, so settling", cdn.GameNotFinalError),
    ("ESPN request failed after 3 attempts", cdn.BoxScoreError),
])
def test_an_unfinished_game_is_not_reported_as_a_fetch_failure(
    monkeypatch, espn_reason, expected,
):
    """
    A GAME STILL BEING PLAYED MUST NOT READ AS AN OUTAGE. The selector cannot
    find our event while it is in progress, and on a night where other games
    are already final that decline arrives as an ordinary error. Counted as a
    failure it inflates games_failed and fills report.errors with games that
    are merely mid-flight — which is how a healthy night looks like an
    incident. The props stay PENDING either way; the label is what an
    operator reads.
    """
    def _forbidden(*a, **k):
        raise cdn.BoxScoreError("403 from NBA CDN")

    def _espn(*a, **k):
        raise espn.EspnBoxScoreError(espn_reason)

    monkeypatch.setattr(cdn, "fetch_final_boxscore", _forbidden)
    monkeypatch.setattr(espn, "fetch_espn_player_stats_for_game", _espn)
    with pytest.raises(expected):
        cdn.fetch_player_stats_for_game(
            "0022600001", game_date="2026-04-10", expected_players={"X"},
        )


def test_when_both_sources_genuinely_fail_both_causes_are_named(monkeypatch):
    """
    "Box score fetch failed" with one cause hides which half of the fallback
    is broken, and an operator then debugs the wrong host.
    """
    monkeypatch.setattr(
        cdn, "fetch_final_boxscore",
        lambda *a, **k: (_ for _ in ()).throw(cdn.BoxScoreError("403 from NBA CDN")),
    )
    monkeypatch.setattr(
        espn, "fetch_espn_player_stats_for_game",
        lambda *a, **k: (_ for _ in ()).throw(
            espn.EspnBoxScoreError("ESPN request failed after 3 attempts")),
    )
    with pytest.raises(cdn.BoxScoreError) as caught:
        cdn.fetch_player_stats_for_game(
            "0022600001", game_date="2026-04-10", expected_players={"X"},
        )
    message = str(caught.value)
    assert "403 from NBA CDN" in message and "ESPN request failed" in message


# --- provenance -------------------------------------------------------------


def test_an_espn_graded_row_does_not_claim_the_nba_cdn():
    """
    PROVENANCE IS NOT DECORATION. `result_source` was hardcoded to
    `nba_cdn_liveData`, so every ESPN-graded row claimed a feed it never
    touched. The settlement audit's whole value is being able to ask where a
    number came from and get a true answer.
    """
    from src.settlement.runner import (
        CDN_RESULT_SOURCE,
        ESPN_RESULT_SOURCE,
        _result_source,
    )

    assert _result_source({"source": "espn"}) == ESPN_RESULT_SOURCE
    assert _result_source({}) == CDN_RESULT_SOURCE
    assert ESPN_RESULT_SOURCE != CDN_RESULT_SOURCE
