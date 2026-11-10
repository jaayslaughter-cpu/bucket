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


# --- the explicit alias table: book spellings the normaliser cannot reach ---


def _panel(names_and_teams):
    import pandas as pd

    return pd.DataFrame(
        [{"PLAYER_NAME": n, "TEAM_ABBREVIATION": t} for n, t in names_and_teams]
    )


def _board(names):
    import pandas as pd

    return pd.DataFrame([{"player_name": n} for n in names])


# VERIFIED AGAINST THE PANEL'S OWN NAME SOURCE, not guessed. The panel stores
# ESPN's `athlete.displayName` (src/ingestion/espn_game.py:387), and these are
# the spellings ESPN's own NBA rosters carried on 2026-11-05:
#
#   board "Alexandre Sarr"  -> ESPN "Alex Sarr"        (WSH, sole Sarr)
#   board "Joshua Giddey"   -> ESPN "Josh Giddey"      (CHI, sole Giddey)
#   board "Jaime Jaquez"    -> ESPN "Jaime Jaquez Jr." (MIL, sole Jaquez)
#
# All three were logged unmatched by the 2026-11-04 slate.
# The 2026-11-05 slate (ORL at PHX) added two more, checked the same way and
# mismatching in OPPOSITE directions: one book drops the suffix, the other
# adds one the panel does not carry.
#
#   board "Wendell Carter"      -> ESPN "Wendell Carter Jr." (ORL)
#   board "Jase Richardson Jr." -> ESPN "Jase Richardson"    (ORL)
VERIFIED_ALIAS_CASES = [
    ("Alexandre Sarr", "Alex Sarr", "WSH"),
    ("Joshua Giddey", "Josh Giddey", "CHI"),
    ("Jaime Jaquez", "Jaime Jaquez Jr.", "MIL"),
    ("Wendell Carter", "Wendell Carter Jr.", "ORL"),
    ("Jase Richardson Jr.", "Jase Richardson", "ORL"),
]


@pytest.mark.parametrize("board_name,panel_name,team", VERIFIED_ALIAS_CASES)
def test_a_known_book_spelling_resolves_to_the_panel_name(board_name, panel_name, team):
    """
    THE NORMALISER CANNOT REACH THESE, and it should not try.

    "Alexandre" -> "Alex" is not a diacritic, a dot or a hyphen; it is a
    different given name for the same person. No deterministic rule derives
    one from the other, and the fuzzy score that would is the same score that
    accepts Jalen Williams as Jaylen Williams. So they are named, one by one,
    from the panel's own source.
    """
    name_map, report = resolve_board_names(
        _board([board_name]), _panel([(panel_name, team)])
    )
    assert name_map.get(board_name) == panel_name, (
        f"{board_name!r} did not resolve to {panel_name!r}: {report.as_dict()}"
    )
    assert not report.unmatched


def test_an_alias_never_overrides_a_real_panel_player_of_that_name():
    """
    EXACT FIRST, ALWAYS. If the panel ever carries the board's own spelling --
    a second Sarr, or ESPN changing its mind -- that player wins, because the
    alias exists to reach a name the panel has, not to rewrite one it has.
    """
    name_map, report = resolve_board_names(
        _board(["Alexandre Sarr"]),
        _panel([("Alexandre Sarr", "WSH"), ("Alex Sarr", "CHA")]),
    )
    assert name_map.get("Alexandre Sarr") == "Alexandre Sarr"
    assert not report.unmatched


def test_an_alias_that_lands_on_two_players_is_still_refused():
    """
    The alias is a spelling, not a licence to guess. Landing on two people is
    the same refusal as any other ambiguous key.
    """
    name_map, report = resolve_board_names(
        _board(["Joshua Giddey"]),
        _panel([("Josh Giddey", "CHI"), ("Josh Giddey", "OKC")]),
    )
    assert "Joshua Giddey" not in name_map
    assert report.ambiguous, report.as_dict()


def test_the_alias_table_is_normalised_and_does_not_chain():
    """
    AN INVARIANT, not a behaviour: every key and value is already in canonical
    form, and no value is itself a key. A chain would make resolution depend
    on dict order, which is how a crosswalk starts guessing.
    """
    from src.ingestion.id_crosswalk import BOARD_NAME_ALIASES

    for key, value in BOARD_NAME_ALIASES.items():
        assert key == normalise_player_name(key), f"{key!r} is not normalised"
        assert value == normalise_player_name(value), f"{value!r} is not normalised"
        assert key != value, f"{key!r} aliases itself"
        assert value not in BOARD_NAME_ALIASES, (
            f"{key!r} -> {value!r} -> {BOARD_NAME_ALIASES[value]!r} is a chain"
        )


def test_suffix_stripping_is_still_not_a_rule(caplog):
    """
    THE FATHER/SON GUARD IS UNCHANGED, and this is the test that would catch
    an alias table quietly growing into one. "Jaime Jaquez" is aliased
    BY NAME; "Gary Payton" is not, so it must not reach "Gary Payton II".
    """
    name_map, report = resolve_board_names(
        _board(["Gary Payton"]), _panel([("Gary Payton II", "GS")])
    )
    assert "Gary Payton" not in name_map
    assert report.unmatched == ["Gary Payton"]


def test_every_unmatched_name_is_logged_and_says_which_kind_of_miss_it_is(caplog):
    """
    THE WARNING IS THE ONLY DIAGNOSIS THERE IS, so it has to be complete.

    On 2026-11-04 production logged "23 board name(s) match no panel player"
    and then showed five of them, two of which (Kon Knueppel, Ivica Zubac)
    the panel did not carry under ANY spelling. So the sample read as a
    spelling problem when half of it was a coverage problem, and the other
    eighteen names were never visible at all.

    A miss beside a same-surname candidate is an alias to add. A miss beside
    nothing is a player the slate's panel does not hold. The warning must
    name every miss and tell the two apart.
    """
    import logging

    board = _board([
        "Alexandre Sarr",   # aliased, so NOT a miss at all
        "Joshua Giddey",    # ditto
        "Jaylen Nowell",    # panel has a different Nowell -> spelling candidate
        "Kon Knueppel",     # panel holds no Knueppel -> coverage, not spelling
        "Ivica Zubac",      # ditto
        "Not A Player",     # ditto
    ])
    panel = _panel([
        ("Alex Sarr", "WSH"),
        ("Josh Giddey", "CHI"),
        ("Jalen Nowell", "MEM"),
    ])

    with caplog.at_level(logging.WARNING, logger="src.ingestion.id_crosswalk"):
        _name_map, report = resolve_board_names(board, panel)

    assert len(report.unmatched) == 4, report.unmatched
    blob = caplog.text

    # EVERY miss by name, not a sample.
    for missed in ("Jaylen Nowell", "Kon Knueppel", "Ivica Zubac", "Not A Player"):
        assert missed in blob, f"{missed!r} was not named in the warning: {blob}"

    # AND THE TWO KINDS ARE DISTINGUISHABLE.
    assert "Jaylen Nowell -> ['Jalen Nowell']" in blob, (
        "a miss with a same-surname candidate must show it, because that is "
        f"the one that an alias entry fixes: {blob}"
    )
    assert "Kon Knueppel -> no such surname in panel" in blob, (
        f"a coverage miss must say so rather than looking like a typo: {blob}"
    )

    # THE ALIASED NAMES ARE NOT IN THE MISS LIST.
    assert "Alexandre Sarr" not in blob and "Joshua Giddey" not in blob


def test_the_surname_hint_ignores_a_generational_suffix():
    """
    THE DIAGNOSTIC'S OWN BUG, found by the first slate that ran with it.

    `names_sharing_surname` compared the LAST TOKEN, and for a suffixed name
    the last token is "jr", not the surname. So the 2026-11-05 slate logged:

        Jase Richardson Jr. -> ['Tim Hardaway Jr.', 'Wendell Carter Jr.']
        Wendell Carter      -> ['Jevon Carter']

    Both are wrong in the way that matters. The first offers two unrelated
    players because all three names end in "Jr."; the second omits the one
    name that is actually the answer -- the panel's "Wendell Carter Jr." --
    because ITS last token is "jr" too. The hint pointed away from the fix
    on the very row the fix was for.

    Suffixes still belong to the matching key (see
    `test_suffix_stripping_is_still_not_a_rule`). This is the hint only.
    """
    crosswalk = PlayerIdCrosswalk([
        PlayerRecord(name="Wendell Carter Jr.", team="ORL"),
        PlayerRecord(name="Jevon Carter", team="ORL"),
        PlayerRecord(name="Tim Hardaway Jr.", team="DET"),
    ])

    # The board drops the suffix: the panel's suffixed Carter is the answer,
    # and the unrelated Jevon Carter is a legitimate same-surname neighbour.
    assert crosswalk.names_sharing_surname("Wendell Carter") == [
        "Jevon Carter", "Wendell Carter Jr.",
    ]

    # The board adds one: no panel Richardson at all, and emphatically not
    # every player whose name happens to end in "Jr.".
    assert crosswalk.names_sharing_surname("Jase Richardson Jr.") == []


def test_a_suffix_only_name_does_not_crash_the_hint():
    """A degenerate name must not raise inside a warning path."""
    crosswalk = PlayerIdCrosswalk([PlayerRecord(name="Jr.", team="XXX")])
    assert crosswalk.names_sharing_surname("Jr.") == ["Jr."]
    assert crosswalk.names_sharing_surname("") == []
