"""ESPN game summary -> box score and plays.

FIXTURE PROVENANCE. Payloads are derived from the shapes documented in
github.com/pseudo-r/public-espn-api (docs/response_schemas.md, "Game Summary").
They are NOT captured live: this environment denies outbound CONNECT to
site.api.espn.com. Nothing here is presented as observed ESPN output.
"""

from __future__ import annotations

import pytest
import requests

from src.ingestion.espn_client import EspnConfig, EspnError
from src.ingestion.espn_game import (
    _split_made_attempted,
    fetch_summary,
    parse_box_score,
    parse_plays,
    parse_summary,
)

NAMES = ["MIN", "FG", "3PT", "FT", "OREB", "DREB", "REB", "AST", "STL", "BLK", "TO", "PF", "+/-", "PTS"]
CURRY = ["36", "12-24", "4-10", "4-4", "0", "5", "5", "7", "1", "0", "2", "2", "+8", "32"]


def _summary(names=None, athletes=None, plays=None) -> dict:
    return {
        "boxscore": {"teams": [{
            "team": {"id": "9"},
            "players": [{
                "team": {"id": "9"},
                "statistics": [{
                    "names": names if names is not None else NAMES,
                    "athletes": athletes if athletes is not None else [{
                        "athlete": {"id": "3136776", "displayName": "Stephen Curry"},
                        "didNotPlay": False,
                        "stats": CURRY,
                    }],
                }],
            }],
        }]},
        "plays": plays if plays is not None else [{
            "id": "4017654340001",
            "sequenceNumber": "1",
            "text": "S. Curry makes 2-pt jump shot from 14 ft",
            "clock": {"displayValue": "11:42"},
            "period": {"number": 1},
            "team": {"id": "9"},
            "scoreValue": 2,
            "scoringPlay": True,
        }],
    }


# --- the stat-alignment hazard ------------------------------------------

def test_stats_are_zipped_by_espn_s_own_header_names():
    rows, notes = parse_box_score(_summary(), "401765432")
    assert notes == []
    stats = rows[0].stats
    assert stats["MIN"] == 36.0
    assert stats["PTS"] == 32.0
    assert stats["AST"] == 7.0
    assert stats["REB"] == 5.0
    assert stats["TOV"] == 2.0          # ESPN calls it "TO"
    assert stats["PLUS_MINUS"] == 8.0   # "+8" parses signed


def test_reordering_espn_s_names_moves_the_values_with_them():
    """The whole point of zipping: position must never be assumed.

    Here PTS and AST are swapped in BOTH arrays. A parser indexing by position
    would read 32 assists and 7 points and look perfectly healthy.
    """
    swapped_names = list(NAMES)
    swapped_stats = list(CURRY)
    i, j = NAMES.index("AST"), NAMES.index("PTS")
    swapped_names[i], swapped_names[j] = swapped_names[j], swapped_names[i]
    swapped_stats[i], swapped_stats[j] = swapped_stats[j], swapped_stats[i]

    rows, _ = parse_box_score(
        _summary(names=swapped_names, athletes=[{
            "athlete": {"id": "1", "displayName": "Stephen Curry"},
            "didNotPlay": False, "stats": swapped_stats,
        }]),
        "e1",
    )
    assert rows[0].stats["PTS"] == 32.0
    assert rows[0].stats["AST"] == 7.0


def test_a_new_stat_column_does_not_shift_the_others():
    names = ["MIN", "REB_CHANCES", "PTS", "AST"]
    stats = ["30", "9", "25", "6"]
    rows, _ = parse_box_score(
        _summary(names=names, athletes=[{
            "athlete": {"id": "1", "displayName": "X"}, "didNotPlay": False, "stats": stats,
        }]),
        "e1",
    )
    assert rows[0].stats["PTS"] == 25.0 and rows[0].stats["AST"] == 6.0
    assert "REB_CHANCES" not in rows[0].stats   # unmapped label is not invented


def test_length_mismatch_drops_stats_rather_than_aligning_by_position():
    rows, notes = parse_box_score(
        _summary(names=NAMES, athletes=[{
            "athlete": {"id": "1", "displayName": "Truncated"},
            "didNotPlay": False, "stats": ["36", "12-24"],
        }]),
        "e1",
    )
    assert rows[0].stats == {}
    assert any("mismatched" in n for n in notes)
    assert rows[0].player_name == "Truncated"   # the row is kept, so it is visible


def test_made_attempted_pairs_split_into_two_columns():
    rows, _ = parse_box_score(_summary(), "e1")
    s = rows[0].stats
    assert (s["FGM"], s["FGA"]) == (12.0, 24.0)
    assert (s["FG3M"], s["FG3A"]) == (4.0, 10.0)
    assert (s["FTM"], s["FTA"]) == (4.0, 4.0)


def test_absent_stat_cells_are_none_not_zero():
    rows, _ = parse_box_score(
        _summary(names=["MIN", "PTS"], athletes=[{
            "athlete": {"id": "1", "displayName": "Bench"},
            "didNotPlay": False, "stats": ["--", "--"],
        }]),
        "e1",
    )
    assert rows[0].stats["MIN"] is None and rows[0].stats["PTS"] is None


# --- DNP is the scratch signal ------------------------------------------

def test_did_not_play_is_separated_from_players_who_appeared():
    summary = parse_summary(
        _summary(athletes=[
            {"athlete": {"id": "1", "displayName": "Played"}, "didNotPlay": False, "stats": CURRY},
            {"athlete": {"id": "2", "displayName": "Scratched"}, "didNotPlay": True, "stats": []},
        ]),
        "e1",
    )
    assert [r.player_name for r in summary.played] == ["Played"]
    assert summary.inactive_names == ["Scratched"]


# --- plays ---------------------------------------------------------------

def test_plays_parse_the_documented_fields():
    plays = parse_plays(_summary(), "e1")
    assert len(plays) == 1
    play = plays[0]
    assert play.period == 1
    assert play.clock_display == "11:42"
    assert play.sequence_number == 1
    assert play.scoring_play is True
    assert play.score_value == 2
    assert "jump shot" in play.text


def test_plays_carry_none_of_the_nba_native_column_names():
    """Nobody may point `scripts/build_pbp_panel.py` at this.

    THIS TEST WAS THE TRIPWIRE AND IT FIRED. Its previous name and docstring
    said ESPN plays carry "no shot distance or action type", and the
    instruction was that if ESPN ever grew those, the failure is the signal to
    revisit rather than to delete the assertion. ESPN HAD ALREADY GROWN ONE:
    every play carries a structured `type` ({"id", "text"}). The assertion
    never fired because it checks for NBA's *key names*, which ESPN still does
    not use -- so the claim in the name went stale without the test noticing.

    What is asserted here is the narrow thing that is still true and still
    worth guarding: the NBA-native names the PBP panel reads are absent, so
    this payload cannot be fed to that script. What ESPN has INSTEAD is
    asserted by the tests below, under its own names.
    """
    row = parse_plays(_summary(), "e1")[0].as_dict()
    for nba_only in ("shotDistance", "shotResult", "actionType", "personId", "actionNumber"):
        assert nba_only not in row, (
            f"{nba_only} appeared under NBA's own name; build_pbp_panel.py "
            "would now silently accept this feed"
        )
    # And no shot DISTANCE under any name: it exists only inside `text` and is
    # deliberately not parsed out of English prose.
    assert not any("DIST" in k.upper() for k in row), sorted(row)


# --- the fields this parser used to drop --------------------------------
#
# All three were already arriving and were thrown away, which cost this project
# the whole play-by-play layer: the module docstring recorded their absence as
# a property of ESPN's FEED rather than of this parser. Each is measured
# against the live endpoint in the docstring; these tests pin the parsing.


SUB_PLAY = {
    "id": "9", "sequenceNumber": "9", "period": {"number": 2},
    "clock": {"displayValue": "7:01"}, "team": {"id": "9"},
    "text": "Gary Payton II enters the game for Stephen Curry",
    "type": {"id": "584", "text": "Substitution"},
    "participants": [
        {"athlete": {"id": "3907387"}},
        {"athlete": {"id": "3136776"}},
    ],
}


def test_the_structured_action_type_is_kept():
    row = parse_plays(_summary(plays=[SUB_PLAY]), "e1")[0]
    assert row.play_type_id == "584"
    assert row.play_type_text == "Substitution"
    assert row.as_dict()["PLAY_TYPE_TEXT"] == "Substitution"


def test_a_play_with_no_type_block_keeps_none_rather_than_a_guess():
    row = parse_plays({"plays": [{"id": "1"}]}, "e1")[0]
    assert row.play_type_id is None and row.play_type_text is None


def test_participant_athlete_ids_are_kept_in_espns_order():
    """
    ORDER IS LOAD-BEARING, so this asserts the exact sequence rather than the
    set. `substitution()` reads position 0 as the entering player; sorting
    these for tidiness would swap who came on for who went off, and nothing
    downstream could tell.
    """
    row = parse_plays(_summary(plays=[SUB_PLAY]), "e1")[0]
    assert row.participant_athlete_ids == ("3907387", "3136776")
    assert row.participant_athlete_ids != tuple(sorted(row.participant_athlete_ids)), (
        "the fixture no longer distinguishes ESPN's order from sorted order"
    )
    assert row.as_dict()["PARTICIPANT_ATHLETE_IDS"] == "3907387;3136776"


def test_a_participant_with_no_athlete_id_is_skipped_not_blanked():
    play = {**SUB_PLAY, "participants": [
        {"athlete": {"id": "3907387"}}, {"athlete": {}}, {"athlete": {"id": ""}},
    ]}
    row = parse_plays({"plays": [play]}, "e1")[0]
    assert row.participant_athlete_ids == ("3907387",)


def test_the_participants_default_is_not_shared_between_rows():
    """A mutable default is how one game's participants reach another's row."""
    a, b = parse_plays({"plays": [{"id": "1"}, {"id": "2"}]}, "e1")
    assert a.participant_athlete_ids == () and b.participant_athlete_ids == ()
    assert isinstance(a.participant_athlete_ids, tuple)


# --- substitutions, which is what the stint layer needs -----------------


def test_a_substitution_resolves_to_entering_and_leaving():
    row = parse_plays(_summary(plays=[SUB_PLAY]), "e1")[0]
    assert row.substitution() == ("3907387", "3136776")


def test_a_substitution_is_matched_on_the_type_id_not_the_english_text():
    """
    The type id is the whole reason `type` is kept. A play whose prose says
    "enters the game" but whose type is something else is NOT a substitution,
    and one with the right id and no such wording IS -- otherwise this parser
    is back to depending on English.
    """
    prose_only = {**SUB_PLAY, "type": {"id": "99", "text": "Jump Shot"}}
    assert parse_plays({"plays": [prose_only]}, "e1")[0].substitution() is None

    id_only = {**SUB_PLAY, "text": "lineup change"}
    assert parse_plays({"plays": [id_only]}, "e1")[0].substitution() == (
        "3907387", "3136776"
    )


@pytest.mark.parametrize("participants,why", [
    ([{"athlete": {"id": "1"}}], "one participant"),
    ([{"athlete": {"id": "1"}}, {"athlete": {"id": "2"}}, {"athlete": {"id": "3"}}],
     "three participants"),
    ([], "none"),
    ([{"athlete": {"id": "1"}}, {"athlete": {"id": "1"}}], "the same player twice"),
])
def test_a_substitution_that_cannot_be_resolved_returns_none(participants, why):
    """
    HALF A SUBSTITUTION IS WORSE THAN NONE. Applied to floor state it leaves
    four players on one side, and every stint after it is wrong. Dropped, not
    guessed.
    """
    play = {**SUB_PLAY, "participants": participants}
    assert parse_plays({"plays": [play]}, "e1")[0].substitution() is None, why


def test_a_non_substitution_play_is_not_one():
    row = parse_plays(_summary(), "e1")[0]
    assert row.substitution() is None


# --- coordinates, which is what a shot-zone layer needs -----------------


def test_a_real_coordinate_is_kept():
    play = {"id": "1", "coordinate": {"x": 26, "y": 6}}
    row = parse_plays({"plays": [play]}, "e1")[0]
    assert (row.coord_x, row.coord_y) == (26, 6)
    assert row.as_dict()["COORD_X"] == 26


def test_the_espn_sentinel_coordinate_is_refused():
    """
    ESPN emits (-214748340, -214748365) where a play has no position -- 135
    times in one game, on 48 of 237 SHOOTING plays. Read naively it puts a shot
    two hundred million units off the floor, and a zone layer averaging that in
    would produce numbers no one could trace back.
    """
    from src.ingestion.espn_game import ESPN_COORDINATE_SENTINEL

    x, y = ESPN_COORDINATE_SENTINEL
    row = parse_plays({"plays": [{"id": "1", "coordinate": {"x": x, "y": y}}]}, "e1")[0]
    assert row.coord_x is None and row.coord_y is None


def test_the_sentinel_is_refused_on_its_own_account_not_by_the_magnitude_bound(
    monkeypatch,
):
    """
    MUTATION THAT SURVIVED THE FIRST ATTEMPT: deleting the sentinel check. It
    survived because the sentinel is also far outside COORDINATE_ABS_LIMIT, so
    the second net caught it and the test above passed either way -- two checks
    where one subsumed the other, and no way for a reader to tell which
    rejected a value.

    Both are kept because they say different things ("ESPN states no position"
    against "this is not on a court"), so this test raises the bound out of the
    way and proves the sentinel check does its own work. Without that, the
    precise and documented reason could be deleted silently and only the
    coarse one would remain.
    """
    from src.ingestion import espn_game

    monkeypatch.setattr(espn_game, "COORDINATE_ABS_LIMIT", 10**9)
    x, y = espn_game.ESPN_COORDINATE_SENTINEL
    assert abs(x) < 10**9, "the bound no longer clears the sentinel"

    row = espn_game.parse_plays(
        {"plays": [{"id": "1", "coordinate": {"x": x, "y": y}}]}, "e1",
    )[0]
    assert row.coord_x is None and row.coord_y is None

    # ... and with the bound raised, a merely-large coordinate now gets
    # through, so the monkeypatch demonstrably took effect.
    through = espn_game.parse_plays(
        {"plays": [{"id": "1", "coordinate": {"x": 5000, "y": 3}}]}, "e1",
    )[0]
    assert through.coord_x == 5000, "the bound was not actually raised"


def test_a_coordinate_beyond_any_court_is_refused_even_if_it_is_not_the_sentinel():
    """The second net, for a different overflow value than the one observed."""
    row = parse_plays(
        {"plays": [{"id": "1", "coordinate": {"x": 5000, "y": 3}}]}, "e1",
    )[0]
    assert row.coord_x is None and row.coord_y is None


def test_a_negative_coordinate_is_legitimate_and_kept():
    """
    y reached -4 in the observed data: behind the baseline. A guard written as
    "reject negatives" would throw away real positions, so this pins that the
    bound is on MAGNITUDE.
    """
    row = parse_plays(
        {"plays": [{"id": "1", "coordinate": {"x": -12, "y": -4}}]}, "e1",
    )[0]
    assert (row.coord_x, row.coord_y) == (-12, -4)


@pytest.mark.parametrize("coord", [
    {"x": 26},
    {"y": 6},
    {"x": 26, "y": None},
    {"x": "left", "y": 6},
    {"x": 26, "y": -214748365},
])
def test_a_half_usable_coordinate_yields_neither_half(coord):
    """
    BOTH OR NEITHER. Keeping the half that parsed would place the play on the
    baseline at that x -- a coordinate that looks plausible and is not where
    anything happened, which is worse than no coordinate at all.
    """
    row = parse_plays({"plays": [{"id": "1", "coordinate": coord}]}, "e1")[0]
    assert row.coord_x is None and row.coord_y is None, coord


def test_no_coordinate_block_is_not_an_error():
    row = parse_plays({"plays": [{"id": "1"}]}, "e1")[0]
    assert row.coord_x is None and row.coord_y is None


# --- the starting five, which is the other half of the stint input ------


def test_the_starter_flag_is_kept():
    """
    Substitutions say who CHANGED, never who began, so a period's opening five
    cannot be reconstructed without this. It was being dropped.
    """
    athletes = [
        {"athlete": {"id": "1", "displayName": "A Starter"}, "didNotPlay": False,
         "starter": True, "stats": CURRY},
        {"athlete": {"id": "2", "displayName": "A Reserve"}, "didNotPlay": False,
         "starter": False, "stats": CURRY},
    ]
    rows, notes = parse_box_score(_summary(athletes=athletes), "e1")
    assert notes == []
    assert [r.starter for r in rows] == [True, False]
    assert rows[0].as_dict()["STARTER"] is True


def test_a_hand_built_box_score_row_is_not_a_starter_by_default():
    """
    MUTATION THAT SURVIVED THE FIRST ATTEMPT: flipping the dataclass default to
    True. `parse_box_score` always passes `starter` explicitly, so no test that
    goes through the parser can see the default at all -- it is reachable only
    by constructing a row directly, which other code and fixtures do. A row
    nobody stated anything about must not claim a place in the starting five.
    """
    from src.ingestion.espn_game import BoxScoreRow

    row = BoxScoreRow(
        espn_event_id="e1", espn_athlete_id="1", player_name="P",
        espn_team_id="9", did_not_play=False,
    )
    assert row.starter is False
    assert row.as_dict()["STARTER"] is False


def test_a_payload_with_no_starter_flag_yields_no_starters_rather_than_five():
    """
    Defaulting True, or inferring the first five of the list, would hand the
    stint layer a lineup nobody stated. False is the honest default: no
    starters, so no stints, so a visible gap instead of a quiet fiction.
    """
    athletes = [
        {"athlete": {"id": str(i), "displayName": f"P{i}"}, "didNotPlay": False,
         "stats": CURRY} for i in range(6)
    ]
    rows, _ = parse_box_score(_summary(athletes=athletes), "e1")
    assert not any(r.starter for r in rows)


def test_non_integer_sequence_number_becomes_none():
    plays = parse_plays({"plays": [{"id": "1", "sequenceNumber": "abc"}]}, "e1")
    assert plays[0].sequence_number is None


# --- degradation --------------------------------------------------------

def test_missing_both_sections_abstains():
    out = parse_summary({}, "e1")
    assert out.status == "DATA_NOT_AVAILABLE"
    assert any("neither" in n for n in out.notes)


def test_pregame_payload_with_no_box_score_still_returns_plays():
    out = parse_summary({"plays": [{"id": "1", "text": "tip"}]}, "e1")
    assert out.status == "OK"
    assert any("no box score" in n for n in out.notes)


def test_non_dict_payload_abstains():
    assert parse_summary(["nope"], "e1").status == "DATA_NOT_AVAILABLE"


# --- fetch --------------------------------------------------------------

class _Response:
    def __init__(self, status=200, body=None):
        self.status_code = status
        self._body = body or {}

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 500:
            raise requests.exceptions.HTTPError(str(self.status_code))


class _Session:
    def __init__(self, *responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def get(self, url, params=None, timeout=None, headers=None):
        self.calls.append({"url": url, "params": params})
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_the_event_id_is_sent_as_the_event_param():
    session = _Session(_Response(200, _summary()))
    fetch_summary("401765432", session=session)
    assert session.calls[0]["params"] == {"event": "401765432"}
    assert session.calls[0]["url"].endswith("/summary")


def test_a_4xx_is_not_retried():
    session = _Session(_Response(404))
    with pytest.raises(EspnError, match="retrying will not help"):
        fetch_summary("1", session=session)
    assert len(session.calls) == 1


def test_transport_failure_raises_rather_than_returning_an_empty_summary():
    cfg = EspnConfig(retry_attempts=2, retry_backoff=1.0)
    session = _Session(
        requests.exceptions.ConnectionError("reset"),
        requests.exceptions.ConnectionError("reset"),
    )
    with pytest.raises(EspnError, match="unreachable after 2 attempts"):
        fetch_summary("1", config=cfg, session=session)


def _summary_players_at_top_level() -> dict:
    """The CDN-gamepackage layout: players as a SIBLING of teams."""
    return {
        "boxscore": {
            "teams": [{"team": {"id": "9"}, "statistics": []}],
            "players": [{
                "team": {"id": "9"},
                "statistics": [{
                    "names": NAMES,
                    "athletes": [{
                        "athlete": {"id": "3136776", "displayName": "Stephen Curry"},
                        "didNotPlay": False,
                        "stats": CURRY,
                    }],
                }],
            }],
        },
        "plays": [],
    }


def test_both_documented_box_score_layouts_are_parsed():
    """The reference documents two shapes and this cannot reach the endpoint.

    docs/response_schemas.md shows boxscore.teams[].players[] under "Game
    Summary" and boxscore: {teams, players} under "CDN Game Package". Picking
    one and being wrong yields zero rows on every live response, silently.
    """
    nested, _ = parse_box_score(_summary(), "e1")
    top_level, _ = parse_box_score(_summary_players_at_top_level(), "e1")

    for rows in (nested, top_level):
        assert len(rows) == 1
        assert rows[0].player_name == "Stephen Curry"
        assert rows[0].stats["PTS"] == 32.0
        assert rows[0].espn_team_id == "9"


def test_an_athlete_reached_by_both_layouts_is_not_duplicated():
    payload = _summary()
    payload["boxscore"]["players"] = payload["boxscore"]["teams"][0]["players"]
    rows, _ = parse_box_score(payload, "e1")
    assert len(rows) == 1, f"the same athlete was emitted {len(rows)} times"


def test_empty_stats_on_a_player_who_appeared_is_flagged():
    """A DNP legitimately has no stats. Someone who played does not."""
    rows, notes = parse_box_score(
        _summary(athletes=[{
            "athlete": {"id": "1", "displayName": "Played But Blank"},
            "didNotPlay": False,
            "stats": [],
        }]),
        "e1",
    )
    assert any("mismatched" in n for n in notes), (
        "a played row with no stats passed silently and would be counted "
        "among those who appeared"
    )
    assert rows[0].stats == {}


def test_a_dnp_with_no_stats_is_still_not_flagged():
    rows, notes = parse_box_score(
        _summary(athletes=[{
            "athlete": {"id": "2", "displayName": "Scratched"},
            "didNotPlay": True,
            "stats": [],
        }]),
        "e1",
    )
    assert notes == [], f"an empty DNP row was wrongly flagged: {notes}"
    assert rows[0].did_not_play is True


@pytest.mark.parametrize("raw", ["3-x", "x-5", "3-", "-5", "12-abc"])
def test_a_one_sided_made_attempted_pair_yields_neither_half(raw):
    """
    BOTH HALVES OR NEITHER. This docstring said "never a guess" while
    returning ``(3.0, None)`` for ``"3-x"``: a made count with no attempts
    beside it. A shooting split with a numerator and no denominator is not a
    partial reading, it is a wrong one — FG% from it is infinite, and a prop
    graded on attempts would be graded against a missing number. Found by
    cubic on PR #16; the settlement fallback reads these columns now, so the
    half-read would have reached grading.
    """
    assert _split_made_attempted(raw) == (None, None)


def test_a_well_formed_pair_still_splits():
    """The control: the fix must not have made the parser refuse everything."""
    assert _split_made_attempted("12-24") == (12.0, 24.0)
    assert _split_made_attempted("0-0") == (0.0, 0.0)


@pytest.mark.parametrize("raw", ["inf-5", "5-inf", "nan-5", "5-nan", "-inf-5"])
def test_a_non_finite_half_is_refused(raw):
    """
    `_number` returns inf for "inf", which survived the both-halves-present
    check and then raised OverflowError the moment a caller converted it to an
    int — taking the whole ESPN event down and leaving every prop on that game
    PENDING. Refusing the pair costs one stat; raising costs a game.
    """
    assert _split_made_attempted(raw) == (None, None)
