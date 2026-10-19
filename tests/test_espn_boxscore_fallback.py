"""
The box score we can actually reach, and the fallback that reaches for it.

Measured 2026-10-19, two days before the opener: `cdn.nba.com` answers 403
in 0.47s with the real headers, `stats.nba.com` never completes a connection,
and a control host through the same proxy answered — so the refusal is NBA's,
not local. With no box score, settlement cannot close, the ledger never fills,
and the publication gate withholds every card for want of calibration evidence
that grading is the only source of.

RESEARCH_ONLY. Every assertion here is about reading results that happened.
"""

from __future__ import annotations

import pytest

from src.settlement import boxscore_fetcher as cdn
from src.settlement import espn_boxscore as espn

#: The real label order ESPN served for event 401811026 on 2026-04-10, and
#: Tobias Harris's real line beside it. Copied from the live payload rather
#: than invented, so the fixture cannot drift into a shape ESPN never sends.
LABELS = ["MIN", "PTS", "FG", "3PT", "FT", "REB", "AST",
          "TO", "STL", "BLK", "OREB", "DREB", "PF", "+/-"]
HARRIS = ["25", "8", "3-5", "1-1", "1-3", "6", "0", "1", "0", "0", "1", "5", "2", "+2"]


def _summary(*, labels=None, athletes=None, status="STATUS_FINAL",
             home="CHA", team="DET", event_id="401811026"):
    return {
        "header": {
            "id": event_id,
            "competitions": [{
                "status": {"type": {"name": status}},
                "competitors": [
                    {"homeAway": "home", "team": {"abbreviation": home}},
                    {"homeAway": "away", "team": {"abbreviation": team}},
                ],
            }],
        },
        "boxscore": {
            "players": [{
                "team": {"abbreviation": team},
                "statistics": [{
                    "labels": list(labels if labels is not None else LABELS),
                    "athletes": list(athletes if athletes is not None else [
                        {"athlete": {"displayName": "Tobias Harris", "id": "6440"},
                         "didNotPlay": False, "starter": True, "stats": list(HARRIS)},
                    ]),
                }],
            }],
        },
    }


# --- reading the stat line ---------------------------------------------------


def test_the_real_line_parses_field_for_field():
    out = espn.extract_espn_player_stats(_summary())
    h = out["Tobias Harris"]
    assert (h["points"], h["reboundsTotal"], h["assists"]) == (8, 6, 0)
    assert (h["threePointersMade"], h["threePointersAttempted"]) == (1, 1)
    assert (h["fieldGoalsMade"], h["fieldGoalsAttempted"]) == (3, 5)
    assert (h["freeThrowsMade"], h["freeThrowsAttempted"]) == (1, 3)
    assert (h["steals"], h["blocks"], h["turnovers"]) == (0, 0, 1)
    assert h["minutes_played"] == 25.0
    assert h["minutes_status"] == "PARSED"
    assert h["did_not_play"] is False
    assert h["is_home"] is False, "DET were away; the home flag read the wrong side"


def test_stats_are_read_by_label_not_by_position():
    """
    THE DEFECT THIS FORECLOSES. ESPN ships a positional `stats` array beside a
    `labels` array. Indexing by a hardcoded offset reads rebounds as assists
    the day ESPN inserts a column — silently, and in the direction of grading
    every prop against the wrong number.

    Same line, columns shuffled: the answer must not move.
    """
    order = [4, 0, 6, 2, 1, 3, 5, 7, 8, 9, 10, 11, 12, 13]
    shuffled_labels = [LABELS[i] for i in order]
    shuffled_stats = [HARRIS[i] for i in order]
    out = espn.extract_espn_player_stats(_summary(
        labels=shuffled_labels,
        athletes=[{"athlete": {"displayName": "Tobias Harris", "id": "6440"},
                   "didNotPlay": False, "stats": shuffled_stats}],
    ))
    h = out["Tobias Harris"]
    assert (h["points"], h["reboundsTotal"], h["assists"]) == (8, 6, 0)
    assert (h["fieldGoalsMade"], h["fieldGoalsAttempted"]) == (3, 5)


@pytest.mark.parametrize("raw,expected", [
    ("3-5", (3, 5)), ("0-0", (0, 0)), ("12-20", (12, 20)),
    ("--", (None, None)), ("", (None, None)), (None, (None, None)),
    ("3", (None, None)), ("3-5-7", (None, None)), ("x-y", (None, None)),
])
def test_made_attempted_splits_or_declines(raw, expected):
    """A malformed pair yields None, never a half-read number."""
    assert espn._split_made_attempted(raw) == expected


def test_a_non_final_event_is_refused():
    """Grading a live game settles props against a partial stat line."""
    with pytest.raises(espn.EspnGameNotFinalError):
        espn.extract_espn_player_stats(_summary(status="STATUS_IN_PROGRESS"))


def test_a_dnp_is_voided_not_zeroed():
    """
    A DNP must VOID, not read as zeros. Zeroing it grades every Over as a
    LOSS — inventing losses out of players who never took the floor.
    """
    out = espn.extract_espn_player_stats(_summary(athletes=[
        {"athlete": {"displayName": "Bench Guy", "id": "1"},
         "didNotPlay": True, "reason": "DNP-Coach's Decision",
         "stats": ["0", "0", "0-0", "0-0", "0-0", "0", "0", "0", "0", "0", "0", "0", "0", "0"]},
    ]))
    g = out["Bench Guy"]
    assert g["did_not_play"] is True
    assert g["minutes_status"] == "DNP_DECLARED"
    assert g["not_playing_reason"] == "DNP-Coach's Decision"


def test_unparseable_minutes_with_real_stats_is_not_a_dnp():
    """
    Mirrors the CDN path's hard-won reasoning: folding a failed minutes parse
    into DNP voided every player whose minutes format changed, against a box
    score showing they had played. Those rows left W-L entirely, so a parser
    break read as a quiet sample shrink rather than an error.
    """
    stats = list(HARRIS)
    stats[0] = "??"
    out = espn.extract_espn_player_stats(_summary(athletes=[
        {"athlete": {"displayName": "Tobias Harris", "id": "6440"},
         "didNotPlay": False, "stats": stats},
    ]))
    h = out["Tobias Harris"]
    assert h["minutes_status"] == "UNAVAILABLE"
    assert h["did_not_play"] is False, "a player with 8 points was voided"


def test_an_espn_athlete_id_never_poses_as_an_nba_person_id():
    """
    ESPN numbers athletes itself (Harris is 6440). Putting that in
    `nba_player_id` would hand every downstream reader an ESPN id in a field
    whose name promises an NBA one — a join against player_game_logs would
    then match nothing, or worse, the wrong person.
    """
    h = espn.extract_espn_player_stats(_summary())["Tobias Harris"]
    assert h["nba_player_id"] is None
    assert h["espn_athlete_id"] == "6440"
    assert h["source"] == "espn"


def test_duplicate_names_are_marked_ambiguous():
    """
    Two players can share a full name, and this path has NO personId to settle
    by, so detection is the only protection left: the matcher must refuse
    rather than grade one player against the other's line.
    """
    out = espn.extract_espn_player_stats(_summary(athletes=[
        {"athlete": {"displayName": "Same Name", "id": "1"}, "stats": list(HARRIS)},
        {"athlete": {"displayName": "Same Name", "id": "2"}, "stats": list(HARRIS)},
    ]))
    assert out["Same Name"]["ambiguous_name"] is True


# --- identifying which ESPN event is our game -------------------------------


def _cands(mapping):
    return {eid: {name: {} for name in names} for eid, names in mapping.items()}


def test_the_event_is_chosen_by_the_players_being_graded():
    cands = _cands({"A": ["Alpha One", "Alpha Two"], "B": ["Beta One", "Beta Two"]})
    assert espn.select_event_by_players(cands, {"Beta One", "Beta Two"}) == "B"


def test_matching_is_case_and_space_insensitive():
    cands = _cands({"A": ["Tobias Harris"], "B": ["Someone Else"]})
    assert espn.select_event_by_players(cands, {"  tobias harris "}) == "A"


def test_a_tie_is_refused_rather_than_broken():
    """
    Picking the wrong event grades a whole game's props against a different
    game's box score. Settlement has a six-hour misfire grace precisely so
    that waiting for the next run is cheap; guessing is not.
    """
    cands = _cands({"A": ["Shared Name", "Alpha"], "B": ["Shared Name", "Beta"]})
    with pytest.raises(espn.EspnBoxScoreError, match="equally well"):
        espn.select_event_by_players(cands, {"Shared Name"})


def test_no_overlap_is_refused():
    cands = _cands({"A": ["Alpha"], "B": ["Beta"]})
    with pytest.raises(espn.EspnBoxScoreError, match="do not|does not|None of"):
        espn.select_event_by_players(cands, {"Gamma"})


def test_no_expected_players_is_refused():
    """The identification method's one input. Without it there is no method."""
    with pytest.raises(espn.EspnBoxScoreError, match="expected players"):
        espn.select_event_by_players(_cands({"A": ["Alpha"]}), set())


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
        lambda d, p, session=None: ({"Tobias Harris": {"source": "espn"}}, {"id": "x"}),
    )
    stats, _ = cdn.fetch_player_stats_for_game(
        "0022600001", game_date="2026-04-10", expected_players={"Tobias Harris"},
    )
    assert stats["Tobias Harris"]["source"] == "espn"


@pytest.mark.parametrize("date_arg,players", [
    (None, {"X"}), ("2026-10-21", set()), (None, set()),
])
def test_without_its_inputs_the_fallback_is_not_attempted(monkeypatch, date_arg, players):
    """
    It refuses to guess which game it is reading, the same way the stale-market
    gate refuses to invent an observation time. The CDN's own error is what
    surfaces, so the cause is not replaced by a vaguer one.
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


def test_when_both_sources_fail_both_causes_are_named(monkeypatch):
    """
    "Box score fetch failed" with one cause hides which half of the fallback
    is broken, and an operator then debugs the wrong host.
    """
    def _forbidden(*a, **k):
        raise cdn.BoxScoreError("403 from NBA CDN")

    def _espn_down(*a, **k):
        raise espn.EspnBoxScoreError("ESPN request failed after 3 attempts")

    monkeypatch.setattr(cdn, "fetch_final_boxscore", _forbidden)
    monkeypatch.setattr(espn, "fetch_espn_player_stats_for_game", _espn_down)
    with pytest.raises(cdn.BoxScoreError) as caught:
        cdn.fetch_player_stats_for_game(
            "0022600001", game_date="2026-04-10", expected_players={"X"},
        )
    message = str(caught.value)
    assert "403 from NBA CDN" in message and "ESPN request failed" in message


def test_espn_having_no_final_game_reads_as_not_final(monkeypatch):
    """
    Not a failure: the props stay PENDING and are graded on the next run,
    which is what the six-hour misfire grace is for.
    """
    def _forbidden(*a, **k):
        raise cdn.BoxScoreError("403 from NBA CDN")

    def _nothing_final(*a, **k):
        raise espn.EspnGameNotFinalError("no FINAL events yet")

    monkeypatch.setattr(cdn, "fetch_final_boxscore", _forbidden)
    monkeypatch.setattr(espn, "fetch_espn_player_stats_for_game", _nothing_final)
    with pytest.raises(cdn.GameNotFinalError):
        cdn.fetch_player_stats_for_game(
            "0022600001", game_date="2026-04-10", expected_players={"X"},
        )
