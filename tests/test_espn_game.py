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


def test_plays_carry_no_shot_distance_or_action_type():
    """Guards the docstring's claim, so nobody wires this into the PBP layer.

    scripts/build_pbp_panel.py requires actionType, personId, shotDistance and
    shotResult. If ESPN plays ever grew those, this test failing is the signal
    to revisit — not a reason to delete the assertion.
    """
    row = parse_plays(_summary(), "e1")[0].as_dict()
    for nba_only in ("shotDistance", "shotResult", "actionType", "personId", "actionNumber"):
        assert nba_only not in row


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
