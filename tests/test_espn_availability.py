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
    out = projected_available(roster, report)
    assert out.status == "OK"
    assert sorted(p.player_name for p in out.withheld) == ["Doubtful Guy", "Out Guy"]
    assert sorted(p.player_name for p in out.available) == [
        "Questionable Guy", "Unmentioned Guy"
    ]


def test_unmentioned_players_stay_in_the_projection():
    """Most of a roster is healthy and absent from an injury feed."""
    roster = [RosterPlayer("Nobody Mentioned", "9", "9", "PF", "Active")]
    report = parse_injuries(_injuries(_entry("Someone Else", status="Out")))
    out = projected_available(roster, report)
    assert out.status == "OK"
    assert [p.player_name for p in out.available] == ["Nobody Mentioned"]
    assert out.withheld == [] and out.unknown == []


def test_unavailable_set_is_explicit_and_excludes_questionable():
    assert UNAVAILABLE == frozenset({"OUT", "DOUBTFUL"})


def test_unavailable_names_lists_only_the_withheld_statuses():
    report = parse_injuries(_injuries(
        _entry("A", status="Out", athlete_id="1"),
        _entry("B", status="Questionable", athlete_id="2"),
    ))
    assert report.unavailable_names == ["A"]


def test_no_availability_multiplier_reaches_a_caller():
    """Ingestion reports status; an effect size is a modelling choice.

    The earlier version grepped the module source for the literals "0.35",
    "0.70" and "0.92". That asserted incidental text: any of those could
    legitimately appear as a timeout, a threshold or a number in prose, and an
    unrelated edit would red-CI it. This checks the surface a caller actually
    sees instead.
    """
    report = parse_injuries(_injuries(
        _entry("Q Guy", status="Questionable", athlete_id="1"),
        _entry("Out Guy", status="Out", athlete_id="2"),
    ))

    for row in report.injuries:
        emitted = row.as_dict()
        numeric = {
            k: v for k, v in emitted.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        assert not numeric, (
            f"ingestion emitted a numeric field {numeric} — a dampener or "
            "effect size belongs in the model layer, not here"
        )
        assert not any(
            "MULT" in k.upper() or "WEIGHT" in k.upper() or "FACTOR" in k.upper()
            for k in emitted
        ), f"an effect-size-shaped key reached a caller: {sorted(emitted)}"

    # and the public surface offers no such helper to reach for
    import src.ingestion.espn_availability as mod

    exported = [n for n in dir(mod) if not n.startswith("_")]
    assert not [
        n for n in exported
        if "mult" in n.lower() or "dampen" in n.lower()
    ], f"module exports an availability multiplier: {exported}"


def test_a_failed_report_abstains_instead_of_declaring_everyone_healthy():
    """The inversion this guards: a fetch failure must not mean "all fit".

    projected_available previously folded every uncertainty into `available`,
    so an empty or failed report returned the whole roster as likely to play.
    """
    roster = [
        RosterPlayer("A", "1", "1", "PG", "Active"),
        RosterPlayer("B", "2", "2", "SG", "Active"),
    ]
    failed = parse_injuries({"injuries": []})
    assert failed.status == "DATA_NOT_AVAILABLE"

    out = projected_available(roster, failed)
    assert out.status == "DATA_NOT_AVAILABLE"
    assert out.available == [], "a failed report produced available players"
    assert [p.player_name for p in out.unknown] == ["A", "B"]
    assert any("unknown" in n for n in out.notes)


def test_an_unbucketable_status_lands_in_unknown_not_available():
    roster = [RosterPlayer("Novel", "1", "1", "PG", "Active")]
    report = parse_injuries(_injuries(
        _entry("Novel", status="Reconditioning", athlete_id="1"),
    ))
    out = projected_available(roster, report)
    assert [p.player_name for p in out.unknown] == ["Novel"]
    assert out.available == []


def test_matching_prefers_the_athlete_id_over_the_name():
    """Names collide and get reformatted; ids do not.

    The roster player and the injury row share an id but spell the name
    differently, so a name-only match would miss the OUT entirely.
    """
    roster = [RosterPlayer("C.J. McCollum", "3033", "3", "SG", "Active")]
    report = parse_injuries(_injuries(
        _entry("CJ McCollum", status="Out", athlete_id="3033"),
    ))
    out = projected_available(roster, report)
    assert [p.player_name for p in out.withheld] == ["C.J. McCollum"]
    assert out.available == []


def test_a_name_match_is_recorded_when_no_id_is_available():
    roster = [RosterPlayer("No Id Player", None, "9", "C", "Active")]
    report = parse_injuries(_injuries(
        {"athlete": {"displayName": "No Id Player"}, "status": "Out"},
    ))
    out = projected_available(roster, report)
    assert [p.player_name for p in out.withheld] == ["No Id Player"]
    assert any("matched by name" in n for n in out.notes)
