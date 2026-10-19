"""
Closing line value: the validator that needs no history.

CLV compares the line YOU took against THAT SAME GAME's close, so it reads on
the first game of a season — no warm-up, no prior-season data, no calibration
report. Win/loss needs roughly three thousand settled props before a
two-point edge separates from noise; CLV is continuous and starts paying
information immediately.

The machinery was complete and inert: `compute_clv` existed, the columns
existed, the Discord card already showed the averages, and nothing populated
`closing_line`, so every row returned {None, None}.

RESEARCH_ONLY. CLV measures market agreement, not profit.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as _date

import pytest

from src.settlement import closing_lines as cl


@dataclass
class _Row:
    """A PropLineRow stand-in: the adapter reads these five fields."""

    source: str
    market: str
    player_name: str
    line: float
    over_odds_american: int | None = None
    under_odds_american: int | None = None


def _ladder(player="De'Aaron Fox", book="bovada", market="PTS"):
    """
    A real-shaped ladder: mid-rungs balanced, outer rungs lopsided.

    NAMES ARE CLEAN HERE BECAUSE THE PAYLOAD'S ARE. The closing payload goes
    through `normalize_event_props`, which strips PropLine's "(SAS)" team
    suffix at the ingestion boundary, so both sides of this lookup carry the
    panel's canonical form. An earlier version of this fixture fed the raw
    suffixed name and tested a shape that no longer reaches here.
    """
    return [
        _Row(book, market, player, 11.5, -275, 200),
        _Row(book, market, player, 13.5, -165, 130),
        _Row(book, market, player, 14.5, -115, -115),   # the main line
        _Row(book, market, player, 15.5, 130, -165),
        _Row(book, market, player, 17.5, 240, -320),
    ]


# --- the approved rule: most balanced two-way price is the main line --------


def test_the_main_line_is_the_most_balanced_two_way_price():
    """
    THE LADDER MAKES THIS A CHOICE, NOT A LOOKUP. Measured on one real event,
    3 to 10 points per (book, market, player). Liquidity would be the natural
    tiebreak and PropLine reports it on 258 of 1022 outcomes, so it cannot be
    the rule. The point the book prices closest to a coin flip is the one it
    is treating as the main number.
    """
    lookup = cl.build_closing_lookup(_ladder())
    quote = lookup[("bovada", "PTS", "deaaron fox")]
    assert quote.main_line == 14.5
    assert (quote.main_over_odds, quote.main_under_odds) == (-115, -115)
    assert quote.n_points == 5


def test_imbalance_is_none_for_a_one_way_quote():
    """
    Treating a missing side as perfectly balanced would make every half-quoted
    rung look like the main line — the deepest, most lopsided rungs included.
    """
    assert cl.price_imbalance(-115, None) is None
    assert cl.price_imbalance(None, -115) is None
    assert cl.price_imbalance(-115, -115) == pytest.approx(0.0)
    assert cl.price_imbalance(-275, 200) > cl.price_imbalance(-120, -110)


def test_a_ladder_with_no_two_way_rung_has_no_main_line():
    """
    No main line to name, but the taken point's own price may still be there,
    so the probability measure can work where the line measure cannot.
    """
    rows = [_Row("bovada", "PTS", "Someone", 20.5, -110, None)]
    quote = cl.build_closing_lookup(rows)[("bovada", "PTS", "someone")]
    assert quote.main_line is None
    assert quote.by_point == {20.5: (-110, None)}


# --- the error this refuses to make ----------------------------------------


def test_closing_odds_come_from_the_point_that_was_taken():
    """
    `compute_clv` keeps the line move and the price move separate because
    "reporting one as the other is the most common CLV error". So the closing
    PRICE must be the same point's price — using the main line's price would
    let a line move leak into clv_prob_points and read as a price edge.
    """
    lookup = cl.build_closing_lookup(_ladder())
    line, odds = cl.closing_for_prop(
        lookup, source="bovada", market="PTS",
        player_name="De'Aaron Fox", taken_line=13.5, taken_side="OVER",
    )
    assert line == 14.5, "the line measure uses the MAIN close"
    assert odds == -165, "the price measure uses the SAME POINT we took"


def test_the_side_decides_which_price_is_returned():
    lookup = cl.build_closing_lookup(_ladder())
    _, over = cl.closing_for_prop(
        lookup, source="bovada", market="PTS",
        player_name="De'Aaron Fox", taken_line=15.5, taken_side="OVER")
    _, under = cl.closing_for_prop(
        lookup, source="bovada", market="PTS",
        player_name="De'Aaron Fox", taken_line=15.5, taken_side="UNDER")
    assert (over, under) == (130, -165)


def test_a_point_no_longer_offered_abstains_on_price_but_not_on_line():
    """
    The line still moved and that is measurable; the price comparison has
    nothing to compare against, so it declines rather than substituting.
    """
    lookup = cl.build_closing_lookup(_ladder())
    line, odds = cl.closing_for_prop(
        lookup, source="bovada", market="PTS",
        player_name="De'Aaron Fox", taken_line=99.5, taken_side="OVER")
    assert line == 14.5
    assert odds is None


def test_the_book_is_part_of_the_key():
    """
    Comparing a line taken at DraftKings against Bovada's close measures two
    shops disagreeing, not the market moving.
    """
    lookup = cl.build_closing_lookup(_ladder(book="bovada"))
    line, odds = cl.closing_for_prop(
        lookup, source="draftkings", market="PTS",
        player_name="De'Aaron Fox", taken_line=14.5, taken_side="OVER")
    assert (line, odds) == (None, None)


def test_names_are_compared_through_the_shared_normaliser():
    """
    The ledger stores the panel's canonical name; the closing payload carries
    PropLine's. Both go through `settlement.names`, so a diacritic or a dot
    cannot lose a quote.
    """
    rows = [_Row("bovada", "PTS", "Nikola Jokić", 28.5, -115, -115)]
    lookup = cl.build_closing_lookup(rows)
    line, _ = cl.closing_for_prop(
        lookup, source="bovada", market="PTS",
        player_name="Nikola Jokic", taken_line=28.5, taken_side="OVER")
    assert line == 28.5


def test_a_row_without_a_point_is_skipped():
    """
    The closing payload carries `line_type: "milestone"` outcomes ("To Score
    10+ Points") whose point is null. They are not over/under props.
    """
    rows = [_Row("bovada", "PTS", "Someone", None, 140, None)]  # type: ignore[arg-type]
    assert cl.build_closing_lookup(rows) == {}


def test_an_unknown_prop_returns_nothing_rather_than_a_guess():
    assert cl.closing_for_prop(
        {}, source="bovada", market="PTS",
        player_name="Nobody", taken_line=10.5, taken_side="OVER") == (None, None)


# --- and the measure it feeds ----------------------------------------------


def test_clv_signs_favour_the_bettor_on_both_sides():
    """
    An Over taken at 13.5 that closes 14.5 is +1.0; the same move on an Under
    is -1.0. `compute_clv` owns that sign and this confirms the wiring hands
    it the right numbers.
    """
    from src.settlement.evaluator import compute_clv

    lookup = cl.build_closing_lookup(_ladder())
    # EACH SIDE'S OWN TAKE PRICE. The 13.5 rung is -165/+130, so handing the
    # Under the Over's -165 would manufacture a price move out of nothing --
    # which an earlier version of this test did, and the code correctly
    # reported as a -0.19 move.
    for side, taken_odds, expected in (("OVER", -165, 1.0), ("UNDER", 130, -1.0)):
        line, odds = cl.closing_for_prop(
            lookup, source="bovada", market="PTS",
            player_name="De'Aaron Fox", taken_line=13.5, taken_side=side)
        clv = compute_clv(predicted_line=13.5, predicted_side=side,
                          closing_line=line, bet_odds=taken_odds, closing_odds=odds)
        assert float(clv["clv_line_points"]) == pytest.approx(expected)
        # Same point, unchanged price: the price measure must read exactly
        # zero, which is the proof that no line move leaked into it.
        assert float(clv["clv_prob_points"]) == pytest.approx(0.0)


# --- the wiring, exercised rather than grepped ----------------------------
#
# THESE THREE TESTS USED TO READ `runner.py` AS TEXT. One walked the AST for an
# `if` whose test mentioned `closing_line` and `None` -- which an INVERTED
# guard satisfies just as well, so a regression that overwrote every recorded
# close would have passed it. One asserted the substring
# "taken_line=prop.predicted_line" appears somewhere in the file, which a
# comment satisfies. One asserted "except Exception" appears in a slice of the
# file, which any unrelated handler in that slice satisfies. All three
# confirmed that the source still LOOKS right; none ran it. cubic was correct
# on every count, and the harness below runs the real `settle_pending_props`
# against fake rows instead -- which immediately found two live bugs the text
# checks could not see.


@dataclass
class _Prop:
    """A `prop_results` row stand-in: the runner only sets attributes."""

    nba_game_id: str = "0022600001"
    game_date: _date = _date(2026, 10, 21)
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


def _fox_lookup():
    """A close whose MAIN line is 14.5 and whose taken 13.5 rung is priced."""
    return cl.build_closing_lookup(_ladder())


def test_the_runner_adds_a_close_and_does_not_overwrite_one(monkeypatch, run_settlement):
    """
    A ROW THAT ALREADY CARRIES A CLOSE KEEPS IT. Settlement retries — the
    six-hour misfire grace exists so a game can be graded on a later run — so
    an unconditional write would replace the number that was true at tip-off
    with whatever the endpoint says days later, silently rewriting history in
    the one table CLV is measured from.

    Run with a prefilled row AND an empty one, in one call, so an inverted
    guard fails on one of them whichever way round it is inverted.
    """
    empty = _Prop()
    prefilled = _Prop(player_name="De'Aaron Fox", closing_line=99.5,
                      closing_odds=-500)
    report = run_settlement(monkeypatch, [empty, prefilled],
                             lookup=_fox_lookup())

    assert report.props_graded == 2
    # The empty row was filled from the capture: 14.5 is the balanced rung.
    assert float(empty.closing_line) == 14.5
    # The prefilled row is untouched, both fields.
    assert float(prefilled.closing_line) == 99.5
    assert int(prefilled.closing_odds) == -500


def test_a_row_with_a_close_but_no_price_still_gets_its_same_point_price(
    monkeypatch, run_settlement,
):
    """
    THE TWO FIELDS FILL INDEPENDENTLY, so gating both on the line left rows
    permanently without a price. A row can carry a main close and no price for
    the point actually taken — they come from different rungs of the ladder —
    and `clv_prob_points` is exactly the half that abstains without it.
    """
    row = _Prop(closing_line=14.5, closing_odds=None)
    run_settlement(monkeypatch, [row], lookup=_fox_lookup())

    assert float(row.closing_line) == 14.5, "the recorded line must survive"
    assert int(row.closing_odds) == -165, (
        "the price of the TAKEN point (13.5 over) was never looked up"
    )
    assert row.clv_prob_points is not None, "the price measure stayed blind"


def test_the_close_is_priced_at_the_point_taken_not_at_the_main_line(
    monkeypatch, run_settlement,
):
    """
    The two outputs feed different measures and must not be crossed. The
    ladder's main line is 14.5 at -115/-115 while the point TAKEN is 13.5 at
    -165/+130; pricing the close off the main line would read a 1.0-point
    line move as a price edge of about 19 probability points.

    Distinct main and taken points, so the numbers themselves say which one
    was used — no comment or substring can satisfy this.
    """
    row = _Prop(predicted_line=13.5, predicted_side="OVER", odds=-165)
    run_settlement(monkeypatch, [row], lookup=_fox_lookup())

    assert float(row.closing_line) == 14.5, "the main line is the LINE measure"
    assert int(row.closing_odds) == -165, (
        "closing_odds must be the price of 13.5 (the point taken), not the "
        "price of the 14.5 main line"
    )
    # Same point, unchanged price: the price measure reads exactly zero, which
    # is the proof that the 1.0-point line move did not leak into it.
    assert float(row.clv_prob_points) == pytest.approx(0.0)
    assert float(row.clv_line_points) == pytest.approx(1.0)


def test_a_capture_failure_does_not_stop_grading(monkeypatch, run_settlement):
    """
    A closing line is a market-quality measure, not a settlement input. A prop
    with no quote still grades to WIN/LOSS/PUSH; trading a complete grading
    run for a complete CLV run would be the wrong way round.

    The failure is INJECTED here. The old version asserted that the string
    "except Exception" appeared in a slice of runner.py, which proves nothing
    about what that handler catches or whether grading continues past it.
    """
    row = _Prop()
    report = run_settlement(
        monkeypatch, [row],
        capture_raises=RuntimeError("PropLine 503 on every event"),
    )

    assert report.props_graded == 1, "a CLV outage stranded a night's grading"
    assert row.outcome_status == "WIN"       # 20 points over 13.5
    assert row.closing_line is None          # no close, so no CLV
    assert row.clv_line_points is None


def test_a_dry_run_writes_nothing_at_all(monkeypatch, run_settlement):
    """
    `dry_run` IS A REHEARSAL, and `session_scope()` commits on a clean exit —
    it does not care that the caller asked for one. The close was resolved
    straight onto the row ABOVE the only `if not dry_run` guard, so a dry run
    persisted every closing line and price it captured while correctly
    leaving the grade alone. Half-written rows are worse than either.
    """
    row = _Prop()
    report = run_settlement(monkeypatch, [row], lookup=_fox_lookup(),
                             dry_run=True)

    assert report.props_graded == 1, "a dry run still reports what it would do"
    assert row.outcome_status == "PENDING"
    assert row.closing_line is None, "a dry run persisted a closing line"
    assert row.closing_odds is None, "a dry run persisted a closing price"


def test_the_closing_capture_uses_the_runs_own_http_session(monkeypatch):
    """
    ONE SESSION PER RUN, threaded through. `capture_for_dates` took a
    `session` and dropped it, so the closing-line calls built their own pool
    and ignored every adapter, proxy and retry policy configured on the
    runner's — and in a test they would reach the real host instead of the
    injected transport.
    """
    seen = {}

    class _Client:
        def __init__(self, session=None, **_kw):
            seen["session"] = session

        def fetch_closing_odds(self, *_a, **_k):
            return {}

    import src.ingestion.propline as pl
    monkeypatch.setattr(pl, "PropLineClient", _Client)
    monkeypatch.setattr(
        "src.db.repository.propline_event_ids_for_dates", lambda _d: ["129636"]
    )

    sentinel = object()
    cl.capture_for_dates({_date(2026, 10, 21)}, session=sentinel)
    assert seen["session"] is sentinel, (
        "the closing-line client built its own session instead of using the "
        "one the settlement run already owns"
    )


def test_equally_balanced_rungs_are_not_decided_by_payload_order():
    """
    TWO RUNGS CAN TIE EXACTLY. -115/-115 and -200/-200 both sit at zero
    imbalance, and `min` on imbalance alone returned whichever the provider
    happened to serialise first — so the same close resolved to a different
    main line depending on upstream dict ordering, giving an irreproducible
    CLV baseline.

    The tiebreak is the OVERROUND, which is a reason rather than a
    convenience: at equal balance the tighter price is the one the book is
    making a market on, and -200/-200 is about 19 points of vig on a rung it
    does not want traded.
    """
    tight = _Row("bovada", "PTS", "De'Aaron Fox", 14.5, -115, -115)
    wide = _Row("bovada", "PTS", "De'Aaron Fox", 16.5, -200, -200)

    for order in ([tight, wide], [wide, tight]):
        quote = cl.build_closing_lookup(order)[("bovada", "PTS", "deaaron fox")]
        assert quote.main_line == 14.5, (
            "the main line moved when the payload was re-ordered"
        )


def test_a_three_way_tie_still_resolves_the_same_way_every_time():
    """
    Equal imbalance AND equal overround, so only the last two keys are left.
    Nearest the middle of the posted ladder wins, then the lower point — the
    answer has to be total or ordering decides it again.
    """
    rungs = [
        _Row("bovada", "PTS", "Fox", 12.5, -110, -110),
        _Row("bovada", "PTS", "Fox", 14.5, -110, -110),
        _Row("bovada", "PTS", "Fox", 16.5, -110, -110),
    ]
    import itertools
    for order in itertools.permutations(rungs):
        quote = cl.build_closing_lookup(list(order))[("bovada", "PTS", "fox")]
        assert quote.main_line == 14.5, "the middle of the ladder is the main line"
