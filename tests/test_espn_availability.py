"""ESPN injuries and rosters -> availability.

FIXTURE PROVENANCE. Payloads are derived from the shapes documented in
github.com/pseudo-r/public-espn-api (docs/response_schemas.md, "League-wide
Injuries" and "Team Roster"). They are NOT captured live — this environment
denies outbound CONNECT to site.api.espn.com. Nothing here is presented as
observed ESPN output.
"""

from __future__ import annotations

import pytest

from src.ingestion.espn_availability import (
    UNAVAILABLE,
    RosterPlayer,
    normalize_status,
    parse_injuries,
    parse_roster,
    projected_available,
)


def _injuries(*entries) -> dict:
    return {
        "timestamp": "2025-03-23T12:00:00Z",
        "status": "success",
        "injuries": [{
            "team": {"id": "9", "displayName": "Golden State Warriors", "abbreviation": "GSW"},
            "injuries": list(entries),
        }],
    }


def _entry(name: str, status: str = "Day-To-Day", athlete_id: str = "1") -> dict:
    return {
        "id": "12345",
        "athlete": {"id": athlete_id, "displayName": name, "position": {"abbreviation": "PG"}},
        "type": {"name": "knee"},
        "status": status,
        "date": "2025-03-20T00:00Z",
    }


# --- status bucketing ---------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("Out", "OUT"),
    ("Out (Knee)", "OUT"),
    ("Doubtful", "DOUBTFUL"),
    ("Questionable", "QUESTIONABLE"),
    ("Game Time Decision", "QUESTIONABLE"),
    ("GTD", "QUESTIONABLE"),
    ("Probable", "PROBABLE"),
    ("Day-To-Day", "DAY_TO_DAY"),
    ("Active", "AVAILABLE"),
])
def test_known_status_text_buckets_correctly(raw, expected):
    assert normalize_status(raw) == expected


@pytest.mark.parametrize("raw", [None, "", "  ", "Reconditioning", "¿qué?"])
def test_unknown_status_is_never_treated_as_available(raw):
    """Silence and novelty both mean unknown. Unknown is not healthy."""
    assert normalize_status(raw) == "DATA_NOT_AVAILABLE"


def test_a_dict_shaped_status_is_read_not_stringified():
    report = parse_injuries(_injuries({
        "athlete": {"id": "1", "displayName": "X"},
        "status": {"name": "Out", "type": "out"},
    }))
    assert report.injuries[0].status == "OUT"


def test_unrecognised_status_is_reported_rather_than_swallowed():
    report = parse_injuries(_injuries(_entry("Novel Case", status="Reconditioning")))
    assert report.injuries[0].status == "DATA_NOT_AVAILABLE"
    assert report.injuries[0].status_raw == "Reconditioning"
    assert any("not guessed" in n for n in report.notes)


# --- parsing ------------------------------------------------------------

def test_parses_the_documented_injury_shape():
    report = parse_injuries(_injuries(_entry("Stephen Curry")))
    assert report.status == "OK"
    row = report.injuries[0]
    assert row.player_name == "Stephen Curry"
    assert row.team_abbreviation == "GSW"
    assert row.status == "DAY_TO_DAY"
    assert row.detail == "knee"


def test_empty_injuries_abstains_and_says_it_is_ambiguous():
    report = parse_injuries({"injuries": []})
    assert report.status == "DATA_NOT_AVAILABLE"
    assert any("identical" in n for n in report.notes)


def test_entry_without_a_name_is_skipped():
    report = parse_injuries(_injuries({"athlete": {"id": "1"}, "status": "Out"}))
    assert report.injuries == []


def test_non_dict_payload_abstains():
    assert parse_injuries("nope").status == "DATA_NOT_AVAILABLE"


def test_conflicting_duplicate_statuses_keep_the_later_and_warn(caplog):
    import logging

    report = parse_injuries(_injuries(
        _entry("Same Player", status="Out", athlete_id="1"),
        _entry("Same Player", status="Probable", athlete_id="1"),
    ))
    with caplog.at_level(logging.WARNING):
        lookup = report.by_name()
    assert lookup["same player"].status == "PROBABLE"
    assert "two different statuses" in caplog.text


# --- rosters ------------------------------------------------------------

def test_roster_flattens_espn_s_position_groups():
    payload = {"athletes": [{"position": "G", "items": [
        {"id": "3136776", "displayName": "Stephen Curry", "jersey": "30",
         "position": {"abbreviation": "SG"}, "status": {"name": "Active"}},
    ]}]}
    players = parse_roster(payload)
    assert len(players) == 1
    assert players[0].player_name == "Stephen Curry"
    assert players[0].jersey == "30"
    assert players[0].position == "SG"
    assert players[0].status == "Active"


def test_roster_also_handles_a_flat_athletes_array():
    """Some leagues return athletes[] ungrouped; both shapes must work."""
    flat = {"athletes": [{"id": "1", "displayName": "Flat Player"}]}
    assert [p.player_name for p in parse_roster(flat)] == ["Flat Player"]


# --- the inference, named as one ----------------------------------------

def test_out_and_doubtful_are_withheld_from_a_projection():
    roster = [
        RosterPlayer("Out Guy", "1", "1", "PG", "Active"),
        RosterPlayer("Doubtful Guy", "2", "2", "SG", "Active"),
        RosterPlayer("Questionable Guy", "3", "3", "SF", "Active"),
        RosterPlayer("Unmentioned Guy", "4", "4", "C", "Active"),
    ]
    report = parse_injuries(_injuries(
        _entry("Out Guy", status="Out", athlete_id="1"),
        _entry("Doubtful Guy", status="Doubtful", athlete_id="2"),
        _entry("Questionable Guy", status="Questionable", athlete_id="3"),
    ))
    available, withheld = projected_available(roster, report)
    assert sorted(p.player_name for p in withheld) == ["Doubtful Guy", "Out Guy"]
    assert sorted(p.player_name for p in available) == ["Questionable Guy", "Unmentioned Guy"]


def test_unmentioned_players_stay_in_the_projection():
    """Most of a roster is healthy and absent from an injury feed."""
    roster = [RosterPlayer("Nobody Mentioned", "9", "9", "PF", "Active")]
    available, withheld = projected_available(roster, parse_injuries(_injuries()))
    assert [p.player_name for p in available] == ["Nobody Mentioned"]
    assert withheld == []


def test_unavailable_set_is_explicit_and_excludes_questionable():
    assert UNAVAILABLE == frozenset({"OUT", "DOUBTFUL"})


def test_unavailable_names_lists_only_the_withheld_statuses():
    report = parse_injuries(_injuries(
        _entry("A", status="Out", athlete_id="1"),
        _entry("B", status="Questionable", athlete_id="2"),
    ))
    assert report.unavailable_names == ["A"]


def test_no_availability_multiplier_is_invented():
    """Ingestion reports status; an effect size is a modelling choice."""
    import inspect

    from src.ingestion import espn_availability

    source = inspect.getsource(espn_availability)
    for forbidden in ("AVAILABILITY_MULT", "avail_mult", "0.35", "0.70", "0.92"):
        assert forbidden not in source, (
            f"{forbidden!r} would bury an effect size in an ingestion layer"
        )
