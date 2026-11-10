"""A prop with no box-score line is VOID, and nothing invents a stake.

Both defects came from the same 2026-11-09 slate. Its Discord results card
read "288 graded, 24 still pending" and "ROI -3.83% on 288.00 unit(s) staked"
-- on a night the publication gate WITHHELD the board, so not one of those 288
predictions was ever published as a pick.

Every guard here was mutation-tested: the fix reverted, the named test watched
go red, the fix put back.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as _date
from decimal import Decimal

import pytest

from src.settlement.evaluator import Outcome, SettlementError, settle_prop
from src.settlement.runner import MATCH_ABSENT, MATCH_AMBIGUOUS, _match_player

# --- the matcher stops making the caller guess ---------------------------


def test_the_matcher_distinguishes_absent_from_ambiguous():
    """
    THE CALLER USED TO GUESS AND GUESSED WRONG. `runner.py` logged "Likely a
    name-format mismatch; route through id_crosswalk" for EVERY failure,
    unconditionally -- including for the seven players `espn_boxscore` had
    already named in the line above as scratches who did not travel. Two
    readers of one fact, and the less informed one reached the log. Acting on
    it would have added crosswalk aliases for correctly spelled names.
    """
    found = {"Stephen Curry": {"player_name": "Stephen Curry", "points": 30}}
    assert _match_player(found, "Stephen Curry") == (found["Stephen Curry"], None)

    assert _match_player(found, "Zion Williamson") == (None, MATCH_ABSENT)

    shared = {"Chris Johnson": {"player_name": "Chris Johnson", "ambiguous_name": True}}
    assert _match_player(shared, "Chris Johnson") == (None, MATCH_AMBIGUOUS)


def test_a_normalised_match_still_resolves():
    """The hyphen/case retry must survive the signature change."""
    stats = {"Collin Murray-Boyles": {"player_name": "Collin Murray-Boyles", "points": 9}}
    got, why = _match_player(stats, "collin murray boyles")
    assert why is None and got is stats["Collin Murray-Boyles"]


# --- a player with no line is voided, not left pending -------------------


def test_a_prop_with_no_box_score_line_is_voided():
    """
    VOID IS THE CORRECT TERMINAL STATE WHICHEVER CAUSE IT WAS. The game is
    final and its box score was fetched whole, so a player with no line in it
    produced no stat and the prop can never grade to WIN/LOSS/PUSH. PENDING
    was a state it would sit in until the 14-day age cutoff dropped it
    silently -- 24 rows did exactly that on 2026-11-09.
    """
    void = settle_prop(
        market="PTS", predicted_line=Decimal("18.5"), predicted_side="OVER",
        player_stats=None,
    )
    assert void.outcome is Outcome.VOID
    assert void.actual_result is None
    assert void.stake_units is None and void.profit_units is None


def test_a_void_carries_no_stake_even_when_one_is_offered():
    """A voided prop is excluded from ROI, so it must not carry a stake."""
    void = settle_prop(
        market="PTS", predicted_line=Decimal("18.5"), predicted_side="OVER",
        player_stats=None, odds=-110, stake_units=Decimal("1"),
    )
    assert void.outcome is Outcome.VOID
    assert void.stake_units is None and void.profit_units is None


@dataclass
class _Prop:
    """A `prop_results` row stand-in — the runner only sets attributes."""

    nba_game_id: str = "0022600001"
    game_date: _date = _date(2026, 11, 9)
    player_name: str = "De'Aaron Fox"
    market: str = "PTS"
    predicted_line: float = 13.5
    predicted_side: str = "OVER"
    odds: int = -165
    source: str = "bovada"
    stake_units: object = None
    closing_line: object = None
    closing_odds: object = None
    outcome_status: str = "PENDING"
    actual_result: object = None
    profit_units: object = None
    minutes_played: object = None
    did_not_play: object = None
    clv_line_points: object = None
    clv_prob_points: object = None
    settlement_note: object = None
    result_source: object = None
    settled_at_utc: object = None
    raw_boxscore_json: object = None


FOX = {
    "De'Aaron Fox": {
        "points": 20, "reboundsTotal": 4, "assists": 6,
        "minutes_played": 33.0, "did_not_play": False,
    },
}


# THESE RUN THE REAL `settle_pending_props`. The first version of this section
# asserted on runner.py's SOURCE TEXT instead -- which conftest's own
# `run_settlement` docstring warns against in as many words -- and four
# mutations survived it: defaulting the stake at the call site, leaving an
# absent player PENDING, dropping an ambiguous name into the void branch, and
# writing did_not_play=True on a void. One of those tests was also a tautology
# (`assert A or B` where B was always true). The behaviour is tested now.


def test_the_runner_voids_a_player_with_no_box_score_line(monkeypatch, run_settlement):
    """
    PENDING was a state the row sat in until the 14-day age cutoff dropped it
    silently. 24 rows did exactly that on 2026-11-09.
    """
    absent = _Prop(player_name="Zion Williamson")
    report = run_settlement(monkeypatch, [absent], player_stats=FOX)

    assert absent.outcome_status == "VOID", absent.outcome_status
    assert report.voids == 1
    assert report.props_unmatched == 1, "the crosswalk signal was lost"
    assert absent.settled_at_utc is not None, "a voided row was never settled"
    assert absent.actual_result is None


def test_the_voided_row_carries_no_stake_and_no_profit(monkeypatch, run_settlement):
    absent = _Prop(player_name="Zion Williamson")
    run_settlement(monkeypatch, [absent], player_stats=FOX)
    assert absent.stake_units is None and absent.profit_units is None


def test_the_void_does_not_assert_that_the_player_did_not_play(
    monkeypatch, run_settlement,
):
    """
    THE CAUSE IS NOT KNOWABLE FROM THE RUNNER and must not be written to a
    column other code reads. A scratch who did not dress and a spelling this
    matcher cannot resolve are the SAME observation: absent from a complete box
    score. `did_not_play=True` would reinstate exactly the over-confidence this
    change removes, and `raw_boxscore_json` would be a fabricated payload.
    """
    absent = _Prop(player_name="Zion Williamson")
    run_settlement(monkeypatch, [absent], player_stats=FOX)

    assert absent.did_not_play is None, "the void asserted a scratch"
    assert absent.raw_boxscore_json is None, "the void invented a box score"
    assert absent.result_source is None, "the void claimed a provenance"
    note = str(absent.settlement_note or "")
    assert "scratch" in note and "could not resolve" in note, (
        f"the note names only one cause: {note!r}"
    )


def test_an_ambiguous_name_stays_pending_and_is_not_voided(
    monkeypatch, run_settlement,
):
    """
    THE ASYMMETRY IS THE POINT. An ambiguous name is RECOVERABLE -- the player
    is in the box score, twice, and a crosswalk entry says which one -- so
    voiding it would discard a row that can still be graded.
    """
    shared = _Prop(player_name="Chris Johnson")
    stats = {
        "Chris Johnson": {
            "player_name": "Chris Johnson", "ambiguous_name": True,
            "points": 11, "minutes_played": 20.0, "did_not_play": False,
        },
    }
    report = run_settlement(monkeypatch, [shared], player_stats=stats)

    assert shared.outcome_status == "PENDING", shared.outcome_status
    assert report.voids == 0, "a recoverable row was voided"
    assert report.props_unmatched == 1


def test_a_matched_row_grades_without_acquiring_a_stake(monkeypatch, run_settlement):
    """
    THE CALL-SITE HALF OF THE STAKE FIX. `stake_or_default` is gone from the
    module, but the call site could just as easily re-default inline -- and a
    unit test on `settle_prop` would not notice, because it never runs the
    loop that passes the row's own NULL through.
    """
    fox = _Prop()
    report = run_settlement(monkeypatch, [fox], player_stats=FOX)

    assert fox.outcome_status == "WIN", fox.outcome_status
    assert report.props_graded == 1
    assert fox.stake_units is None, "the runner invented a stake"
    assert fox.profit_units is None, "the runner booked a profit on no stake"


def test_a_dry_run_voids_nothing(monkeypatch, run_settlement):
    """`dry_run` must write nothing — including the new VOID branch."""
    absent = _Prop(player_name="Zion Williamson")
    report = run_settlement(monkeypatch, [absent], player_stats=FOX, dry_run=True)

    assert absent.outcome_status == "PENDING", "a dry run persisted a void"
    assert absent.settled_at_utc is None
    assert report.voids == 1, "the report should still say what it would do"


def test_stake_or_default_is_gone_and_does_not_come_back():
    """
    It turned the recorder's deliberate NULL into one unit at grading time.
    Named here because reintroducing a "sensible default" is the obvious
    well-meaning change, and it is the one that produced a betting record
    nobody decided to keep.
    """
    import src.settlement.runner as runner

    assert not hasattr(runner, "stake_or_default")


# --- nothing invents a stake --------------------------------------------


@pytest.mark.parametrize("odds", [-110, 150, None])
def test_a_graded_row_without_a_stake_reports_no_roi(odds):
    result = settle_prop(
        market="PTS", predicted_line=Decimal("25.5"), predicted_side="OVER",
        player_stats={"points": 30}, odds=odds,
    )
    assert result.outcome is Outcome.WIN, "the row must still grade W/L"
    assert result.stake_units is None
    assert result.profit_units is None


def test_the_default_stake_is_none_not_one_unit():
    """
    MUTATION BAIT. Restoring `stake_units: Any = Decimal(1)` is a one-token
    change and every other test would still pass, because they pass a stake
    explicitly or assert only the outcome.
    """
    import inspect

    sig = inspect.signature(settle_prop)
    assert sig.parameters["stake_units"].default is None


def test_an_explicit_stake_is_still_honoured():
    """Removing the default must not remove the parameter: a backtest stakes."""
    result = settle_prop(
        market="PTS", predicted_line=Decimal("25.5"), predicted_side="OVER",
        player_stats={"points": 30}, odds=100, stake_units=Decimal("2"),
    )
    assert result.stake_units == Decimal("2")
    assert result.profit_units == Decimal("2")


def test_a_zero_stake_is_refused_rather_than_read_as_absent():
    """
    "Staked nothing" and "says nothing about a stake" are different
    statements. Collapsing them would let a zero-stake row grade silently
    instead of raising on a nonsense wager.
    """
    with pytest.raises(SettlementError):
        settle_prop(
            market="PTS", predicted_line=Decimal("25.5"), predicted_side="OVER",
            player_stats={"points": 30}, odds=-110, stake_units=Decimal("0"),
        )


# --- the card explains why ROI is absent --------------------------------


def _notes(priced_n: int, unpriced_n: int, staked):
    """Drive the REAL note builder. Earlier this helper re-implemented the
    if/elif itself, which made the assertions below unfalsifiable -- they would
    have passed with `metrics` arbitrarily wrong. `roi_exclusion_notes` was
    extracted so there is one definition and the test can reach it."""
    from src.settlement.metrics import RoiBlock, roi_exclusion_notes

    roi = RoiBlock(
        priced_n=priced_n, unpriced_graded_n=unpriced_n, staked_units=staked,
    )
    return roi_exclusion_notes(roi)


def test_priced_props_with_no_stake_get_a_named_reason_not_a_blank():
    """
    A card showing no ROI and no reason reads as a broken pipeline. 288 priced
    props settled unstaked is the DESIGNED state and has to say so.
    """
    only = _notes(288, 0, None)
    assert len(only) == 1
    assert "none carries a stake" in only[0]
    assert "places no wager" in only[0]
    assert "288" in only[0], "the reason does not say how many it covers"


def test_no_priced_props_is_a_different_reason_from_no_stake():
    notes = _notes(0, 0, None)
    assert len(notes) == 1 and "No priced props settled" in notes[0]


def test_a_real_stake_produces_no_exclusion_note():
    assert _notes(288, 0, 288.0) == []


def test_both_exclusions_are_reported_when_both_hold():
    """
    MUTATION THE if/elif CHAIN HID. A slate can carry pick'em rows AND no
    stakes; the chain reported only the first, so the second exclusion went
    unmentioned on the card entirely.
    """
    notes = _notes(288, 12, None)
    assert len(notes) == 2, notes
    assert any("pick'em" in n for n in notes)
    assert any("none carries a stake" in n for n in notes)


def test_the_summary_joins_every_note_rather_than_the_first():
    """`get_performance_summary` must put ALL the reasons on the card."""
    import pathlib

    import src.settlement.metrics as metrics

    source = pathlib.Path(metrics.__file__).read_text(encoding="utf-8")
    assert "roi_exclusion_notes(roi)" in source, (
        "the summary no longer calls the note builder"
    )
    assert "join(roi_exclusion_notes(roi))" in source, (
        "the summary takes one note instead of joining them"
    )
