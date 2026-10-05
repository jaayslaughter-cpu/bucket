"""The name crosswalk — and why it is not a fuzzy matcher.

Three modules told the reader to route name variance "through
`ingestion/id_crosswalk.py`", a path that did not exist. The consequence was
measured: one diacritic skipped a row for want of a line source, so a
systematic format difference recorded ZERO gradeable rows while the run
reported success.

The reference implementation used rapidfuzz `token_sort_ratio` above
`score_cutoff=85`. The scores below are why that cannot work, and the first
test is the measurement rather than an argument.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.ingestion.id_crosswalk import (
    AMBIGUOUS,
    MATCHED,
    UNMATCHED,
    PlayerIdCrosswalk,
    PlayerRecord,
    apply_name_map,
    crosswalk_from_frame,
    normalise_player_name,
    resolve_board_names,
)

#: (must match, panel spelling, board spelling)
MUST_MATCH = [
    ("Nikola Jokić", "Nikola Jokic"),
    ("Luka Dončić", "Luka Doncic"),
    ("Kristaps Porziņģis", "Kristaps Porzingis"),
    ("Nikola Vučević", "Nikola Vucevic"),
    ("Shai Gilgeous-Alexander", "Shai Gilgeous Alexander"),
    ("P.J. Washington", "PJ Washington"),
    ("T.J. McConnell", "TJ McConnell"),
    ("De'Aaron Fox", "DeAaron Fox"),
    ("Jaren Jackson Jr.", "Jaren Jackson Jr"),
]
#: Different people, or a father and a son. Must never match.
MUST_REFUSE = [
    ("Jalen Williams", "Jaylen Williams"),
    ("Jalen Johnson", "Jaylen Johnson"),
    ("Jaylen Brown", "Jalen Brown"),
    ("Marcus Morris", "Markieff Morris"),
    ("Kevin Porter Jr.", "Michael Porter Jr."),
    ("Gary Payton II", "Gary Payton"),
    ("Jabari Smith Jr.", "Jabari Smith"),
]


def test_no_fuzzy_cutoff_can_separate_the_two_classes():
    """
    THE MEASUREMENT THAT SETS THE DESIGN. If a score threshold could do this
    job, the simpler implementation would be correct and this module would be
    over-engineered. It cannot: the most dangerous pair in the league scores
    higher than every pair we must catch.
    """
    from rapidfuzz import fuzz

    worst_refuse = max(
        fuzz.token_sort_ratio(a, b) for a, b in MUST_REFUSE
    )
    best_needed = min(
        max(fuzz.token_sort_ratio(a, b), fuzz.ratio(a, b)) for a, b in MUST_MATCH
    )
    assert worst_refuse > best_needed, (
        f"a pair that must be refused scores {worst_refuse:.1f} while the "
        f"hardest pair we must catch scores {best_needed:.1f} — if this ever "
        "inverts, a score cutoff becomes viable and this module can simplify"
    )
    # and specifically, the reference's cutoff of 85 gets both wrong
    assert fuzz.token_sort_ratio("Jalen Williams", "Jaylen Williams") > 85
    assert fuzz.token_sort_ratio("Luka Doncic", "Luka Dončić") < 85


@pytest.mark.parametrize("panel_name,board_name", MUST_MATCH)
def test_the_normaliser_matches_one_player_spelled_two_ways(panel_name, board_name):
    assert normalise_player_name(panel_name) == normalise_player_name(board_name)


@pytest.mark.parametrize("a,b", MUST_REFUSE)
def test_the_normaliser_keeps_different_people_apart(a, b):
    assert normalise_player_name(a) != normalise_player_name(b)


def test_suffixes_are_kept_because_they_distinguish_a_father_from_a_son():
    assert normalise_player_name("Gary Payton II") != normalise_player_name("Gary Payton")
    assert normalise_player_name("Jabari Smith Jr.") != normalise_player_name("Jabari Smith")


def test_an_empty_or_unusable_name_is_a_miss_not_a_key():
    for value in ("", "   ", None, float("nan")):
        assert normalise_player_name(value) == "" or normalise_player_name(value) == "nan"
    crosswalk = PlayerIdCrosswalk([PlayerRecord(name="A B", player_id="1")])
    assert crosswalk.match_one("").status == UNMATCHED
    assert crosswalk.match_one(None).status == UNMATCHED


# --- resolution ------------------------------------------------------------

def test_a_resolved_name_carries_the_panel_s_id():
    crosswalk = PlayerIdCrosswalk([
        PlayerRecord(name="Nikola Jokić", player_id="203999", team="DEN"),
    ])
    hit = crosswalk.match_one("Nikola Jokic")
    assert hit.status == MATCHED
    assert hit.matched_name == "Nikola Jokić"
    assert hit.player_id == "203999"


def test_a_name_no_panel_player_has_is_reported_not_guessed():
    crosswalk = PlayerIdCrosswalk([PlayerRecord(name="Nikola Jokić", player_id="1")])
    miss = crosswalk.match_one("Someone Else")
    assert miss.status == UNMATCHED
    assert "fuzzy" in (miss.reason or ""), "the reason should say why it did not guess"


def test_two_players_with_one_normalised_name_abstain():
    """
    The case a fuzzy matcher resolves by picking one. Matching the wrong player
    onto a price is a wrong record, not a near miss.
    """
    crosswalk = PlayerIdCrosswalk([
        PlayerRecord(name="John Smith", player_id="1", team="LAL"),
        PlayerRecord(name="John Smith", player_id="2", team="BOS"),
    ])
    out = crosswalk.match_one("John Smith")
    assert out.status == AMBIGUOUS
    assert out.player_id is None
    assert "LAL" in (out.reason or "") and "BOS" in (out.reason or "")


def test_a_team_breaks_the_tie_when_the_board_supplies_one():
    crosswalk = PlayerIdCrosswalk([
        PlayerRecord(name="John Smith", player_id="1", team="LAL"),
        PlayerRecord(name="John Smith", player_id="2", team="BOS"),
    ])
    hit = crosswalk.match_one("John Smith", team="BOS")
    assert hit.status == MATCHED and hit.player_id == "2"
    assert crosswalk.match_one("John Smith", team="MIA").status == AMBIGUOUS


def test_a_team_is_a_tiebreak_and_not_a_requirement():
    """A board that publishes no team must still resolve its unambiguous rows."""
    crosswalk = PlayerIdCrosswalk([PlayerRecord(name="Nikola Jokić", player_id="1", team="DEN")])
    assert crosswalk.match_one("Nikola Jokic", team=None).status == MATCHED
    assert crosswalk.match_one("Nikola Jokic", team="LAL").status == MATCHED


def test_the_same_player_repeated_in_a_panel_is_one_player():
    """A panel has ~400 rows per player; that is not 400 collisions."""
    frame = pd.DataFrame({
        "PLAYER_NAME": ["Nikola Jokić"] * 50,
        "PLAYER_ID": ["203999"] * 50,
        "TEAM_ABBREVIATION": ["DEN"] * 50,
    })
    crosswalk = crosswalk_from_frame(frame)
    assert len(crosswalk) == 1
    assert crosswalk.match_one("Nikola Jokic").status == MATCHED


def test_match_many_reports_every_miss_by_name():
    crosswalk = PlayerIdCrosswalk([PlayerRecord(name="Nikola Jokić", player_id="1")])
    report = crosswalk.match_many(["Nikola Jokic", "Nobody Here", "Also Missing"])
    assert report.n_queried == 3 and report.n_matched == 1
    assert sorted(report.unmatched) == ["Also Missing", "Nobody Here"]
    assert report.name_map == {"Nikola Jokic": "Nikola Jokić"}


def test_parallel_name_and_team_sequences_are_required_to_match():
    crosswalk = PlayerIdCrosswalk([PlayerRecord(name="A B", player_id="1")])
    with pytest.raises(ValueError, match="parallel"):
        crosswalk.match_many(["A B", "C D"], teams=["LAL"])


# --- the three call sites --------------------------------------------------

def test_the_prop_line_join_now_survives_a_diacritic():
    """
    THE DEFECT. `main._attach_prop_lines` joined on an exact name, so a board
    spelling `Jokic` against a panel spelling `Jokić` matched nothing.
    """
    from main import _attach_prop_lines

    projections = pd.DataFrame({
        "PLAYER_NAME": ["Nikola Jokić", "Luka Dončić"],
        "MARKET": ["PTS", "PTS"],
    })
    board = pd.DataFrame({
        "player_name": ["Nikola Jokic", "Luka Doncic"],
        "market": ["PTS", "PTS"],
        "line": [28.5, 31.5],
    })
    lines = _attach_prop_lines(projections, board)
    assert list(lines) == [28.5, 31.5]


def test_the_prop_line_join_still_refuses_a_different_player():
    from main import _attach_prop_lines

    projections = pd.DataFrame({"PLAYER_NAME": ["Jalen Williams"], "MARKET": ["PTS"]})
    board = pd.DataFrame({
        "player_name": ["Jaylen Williams"], "market": ["PTS"], "line": [19.5],
    })
    assert _attach_prop_lines(projections, board).isna().all()


def test_the_recorder_records_a_row_whose_board_spelling_differs():
    from src.settlement.recorder import pending_prop_result_rows

    projections = pd.DataFrame({
        "PLAYER_NAME": ["Nikola Jokić"], "GAME_ID": ["0022500001"],
        "GAME_DATE": [pd.Timestamp("2025-11-01")], "MARKET": ["PTS"],
        "LINE": [28.5], "PROB_OVER": [0.58], "AVAILABILITY": ["AVAILABLE"],
    })
    board = pd.DataFrame({
        "player_name": ["Nikola Jokic"], "market": ["PTS"], "line": [28.5],
        "source": ["propline"], "nba_game_id": ["0022500001"],
    })
    out = pending_prop_result_rows(projections, board, run_id="t")
    assert len(out.rows) == 1, (
        f"the board's spelling still skipped the row: "
        f"{[s['reason'] for s in out.skipped]}"
    )


def test_the_scratch_filter_withholds_a_player_espn_spells_differently():
    """
    THE SAFETY BUG. `scratches._normalise` was lowercase and whitespace only,
    so ESPN's `Nikola Jokić` did not match the panel's `Nikola Jokic` and a
    player reported OUT was labelled AVAILABLE. Demonstrated before the fix.
    """
    from src.ingestion.espn_availability import AvailabilityReport, InjuryRow
    from src.pipeline.scratches import AVAILABILITY_COLUMN, WITHHELD, apply_scratch_filter

    row = InjuryRow(
        player_name="Nikola Jokić", espn_athlete_id="1", espn_team_id="7",
        team_abbreviation="DEN", status="OUT", status_raw="Out",
        detail="rest", reported_date="2026-10-05",
    )
    out = apply_scratch_filter(
        pd.DataFrame({"PLAYER_NAME": ["Nikola Jokic"], "MARKET": ["PTS"]}),
        AvailabilityReport(status="OK", injuries=[row]),
    )
    assert out.projections[AVAILABILITY_COLUMN].iloc[0] == WITHHELD
    assert out.withheld == ["Nikola Jokic"]


def test_the_scratch_filter_does_not_withhold_a_similarly_named_player():
    from src.ingestion.espn_availability import AvailabilityReport, InjuryRow
    from src.pipeline.scratches import AVAILABILITY_COLUMN, AVAILABLE, apply_scratch_filter

    row = InjuryRow(
        player_name="Jaylen Williams", espn_athlete_id="1", espn_team_id="25",
        team_abbreviation="OKC", status="OUT", status_raw="Out",
        detail=None, reported_date=None,
    )
    out = apply_scratch_filter(
        pd.DataFrame({"PLAYER_NAME": ["Jalen Williams"], "MARKET": ["PTS"]}),
        AvailabilityReport(status="OK", injuries=[row]),
    )
    assert out.projections[AVAILABILITY_COLUMN].iloc[0] == AVAILABLE


def test_the_three_modules_no_longer_name_a_path_that_does_not_exist():
    from pathlib import Path

    root = Path(__file__).parent.parent
    assert (root / "src" / "ingestion" / "id_crosswalk.py").is_file()


def test_an_unresolved_name_is_left_alone_rather_than_dropped():
    board = pd.DataFrame({"player_name": ["Nobody Here"], "market": ["PTS"]})
    panel = pd.DataFrame({"PLAYER_NAME": ["Nikola Jokić"], "PLAYER_ID": ["1"]})
    name_map, report = resolve_board_names(board, panel)
    assert name_map == {}
    assert report.unmatched == ["Nobody Here"]
    assert list(apply_name_map(board, name_map)["player_name"]) == ["Nobody Here"]
