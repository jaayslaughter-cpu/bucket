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


# --- the wiring: a close is added, never overwritten -----------------------


def test_the_runner_fills_the_close_only_when_it_is_missing():
    """
    A ROW THAT ALREADY CARRIES A CLOSE KEEPS IT. Settlement retries — the
    six-hour misfire grace exists so a game can be graded on a later run — so
    an unconditional write would replace the number that was true at tip-off
    with whatever the endpoint says days later, silently rewriting history in
    the one table CLV is measured from.
    """
    import ast
    from pathlib import Path as _Path

    source = _Path("src/settlement/runner.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    guarded = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = ast.unparse(node.test)
        if "closing_line" in test and "None" in test:
            body = ast.unparse(node)
            if "closing_for_prop" in body:
                guarded = True
    assert guarded, (
        "closing_for_prop is no longer behind an `if prop.closing_line is None` "
        "guard, so a re-run can overwrite a recorded close"
    )


def test_the_runner_asks_for_the_same_point_not_the_main_line_price():
    """
    The two outputs feed different measures and must not be crossed. If the
    call ever passed the main line as `taken_line`, clv_prob_points would
    absorb the line move and report it as a price edge.
    """
    from pathlib import Path as _Path

    source = _Path("src/settlement/runner.py").read_text(encoding="utf-8")
    assert "taken_line=prop.predicted_line" in source, (
        "the closing price must be looked up at the point that was TAKEN"
    )
    assert "taken_side=prop.predicted_side" in source


def test_a_capture_failure_does_not_stop_grading():
    """
    A closing line is a market-quality measure, not a settlement input. A prop
    with no quote still grades to WIN/LOSS/PUSH; trading a complete grading
    run for a complete CLV run would be the wrong way round.
    """
    from pathlib import Path as _Path

    source = _Path("src/settlement/runner.py").read_text(encoding="utf-8")
    block = source.split("capture_for_dates(")[1].split("by_game.items()")[0]
    assert "except Exception" in block, (
        "closing-line capture is no longer wrapped, so one bad event can "
        "strand a whole night's grading"
    )
