"""
tests/test_odds_plausibility.py — one bad price must not cost the night.

THE OUTAGE THIS EXISTS TO PREVENT, measured rather than imagined.

The 2026-11-11 09:00 PT slate exited 1:

    Pipeline FAILED: (psycopg.errors.NumericValueOutOfRange) integer out of range
    [SQL: INSERT INTO projections (... odds_over_american ... ) VALUES ...]

`projections` has exactly two INTEGER columns, `odds_over_american` and
`odds_under_american`, and `ODDS_OVER`/`ODDS_UNDER` are set only from
`main._int_or_none_american`. A price larger than a 32-bit integer arrived
from the provider, passed every check between the API and the database, and
Postgres rejected the whole multi-row INSERT -- so no projections were
written, no card was sent and nothing was scored for the night, over one
malformed field.

AND THE DATABASE WAS THE ONLY THING THAT CAUGHT IT.
`contracts.american_to_implied_probability` refused 0 and nothing else, so
odds of 3,000,000,000 returned an implied probability of 3.3e-8 -- a figure
that de-vigs without complaint and reads as a near-certainty on the other
side. A wider column would have turned a crash into a silent fiction, which
is the worse failure.

The alert text also misdiagnosed it, offering "nothing ingested yet" or "no
resolvable model artifact" as the likely causes. Both were false; the cause
was in the log line it told the reader to go and find.

RESEARCH_ONLY. This is about whether a number is a price. No wager.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.quant.contracts import (
    AMERICAN_ODDS_LIMIT,
    american_to_implied_probability,
    devig_two_way,
)

#: Larger than int4, which is what actually reached Postgres.
BEYOND_INT4 = 3_000_000_000

#: Beyond this project's bound but comfortably inside int4 -- so a test using
#: it proves the bound is semantic and not merely an overflow guard.
BEYOND_LIMIT_WITHIN_INT4 = AMERICAN_ODDS_LIMIT + 1


def test_the_limit_is_far_below_int4_so_the_bound_is_semantic() -> None:
    """
    The column's width is not the reason for the bound.

    If the limit were chosen to fit int4, a schema change to BIGINT would
    read as licence to remove it -- and the silent-probability failure above
    would be what remained.
    """
    assert AMERICAN_ODDS_LIMIT == 100_000
    assert AMERICAN_ODDS_LIMIT < 2_147_483_647 / 1000, (
        "the limit is close enough to int4 to look like an overflow guard"
    )


@pytest.mark.parametrize("odds", [BEYOND_INT4, -BEYOND_INT4,
                                  BEYOND_LIMIT_WITHIN_INT4,
                                  -BEYOND_LIMIT_WITHIN_INT4])
def test_an_implausible_price_is_refused_not_converted(odds: int) -> None:
    """
    THE SILENT FAILURE, which the crash was hiding.

    Without the bound this returns a tiny probability for a huge price and a
    near-1.0 for a huge negative one. Both de-vig cleanly and neither is
    detectable downstream.
    """
    with pytest.raises(ValueError, match=str(AMERICAN_ODDS_LIMIT)):
        american_to_implied_probability(odds)


@pytest.mark.parametrize("odds", [-110, 110, 100, -100, 2500, -2500,
                                  AMERICAN_ODDS_LIMIT, -AMERICAN_ODDS_LIMIT])
def test_a_real_price_still_converts(odds: int) -> None:
    """The bound must not reject prices a book actually posts."""
    p = american_to_implied_probability(odds)
    assert 0.0 < p < 1.0, (odds, p)


def test_zero_is_still_refused_for_its_own_reason() -> None:
    """The pre-existing refusal, kept distinct from the new one."""
    with pytest.raises(ValueError, match="not a price"):
        american_to_implied_probability(0)


def test_devig_refuses_a_market_built_on_an_implausible_side() -> None:
    """
    The refusal must survive the call the slate actually makes.

    `devig_two_way` is where a price becomes a fair probability, and a bound
    that only guards the single-price helper would be bypassed by every
    caller that de-vigs a pair.
    """
    with pytest.raises(ValueError):
        devig_two_way(BEYOND_INT4, -110)
    with pytest.raises(ValueError):
        devig_two_way(-110, BEYOND_INT4)
    # And a normal pair is unaffected.
    fair = devig_two_way(-110, -110)
    assert 0.0 < fair["fair_probability_over"] < 1.0


# ---------------------------------------------------------------------------
# The gate between the provider and the INTEGER column.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        (-110, -110),
        (250, 250),
        ("-115", -115),
        (0, None),              # pre-existing: 0 is not a price
        (None, None),
        ("", None),
        ("not a number", None),
        (BEYOND_INT4, None),    # the value that failed the slate
        (-BEYOND_INT4, None),
        (BEYOND_LIMIT_WITHIN_INT4, None),
        (float("inf"), None),   # int(float("inf")) raises OverflowError
        (float("-inf"), None),
        (float("nan"), None),
    ],
)
def test_the_db_funnel_admits_only_plausible_prices(value, expected) -> None:
    """
    `_int_or_none_american` is the only writer of ODDS_OVER/ODDS_UNDER.

    `inf` is in this table because `int(float("inf"))` raises `OverflowError`,
    which the original `except (TypeError, ValueError)` did not catch -- so an
    infinite price would have propagated an exception out of the price
    matcher rather than returning None.
    """
    import main

    assert main._int_or_none_american(value) == expected


def test_an_implausible_price_costs_its_row_and_not_the_slate(caplog) -> None:
    """
    THE WHOLE POINT, driven through the real price matcher.

    One malformed price must leave its own row unpriced and leave every other
    row priced. Before the bound it failed the INSERT and cost all of them.
    """
    import logging

    import main

    projections = pd.DataFrame({
        "PLAYER_NAME": ["Good Player", "Bad Price"],
        "MARKET": ["PTS", "PTS"],
    })
    prop_lines = pd.DataFrame({
        "player_name": ["Good Player", "Bad Price"],
        "market": ["PTS", "PTS"],
        "line": [20.5, 18.5],
        "over_odds_american": [-110, BEYOND_INT4],
        "under_odds_american": [-110, -110],
        "source": ["book", "book"],
    })

    with caplog.at_level(logging.ERROR):
        out = main._attach_prop_line_prices(projections, prop_lines)

    # `pd.isna`, not `is None`: a column holding one number and one null is
    # float dtype, so the null arrives as NaN. Asserting `is None` here failed
    # on `[-110.0, nan]` -- the code was right and the assertion was not.
    assert out["ODDS_OVER"].iloc[0] == -110
    assert pd.isna(out["ODDS_OVER"].iloc[1]), out["ODDS_OVER"].tolist()
    assert out["ODDS_UNDER"].iloc[0] == -110
    assert "REFUSED AS IMPLAUSIBLE" in caplog.text, (
        "a refused price was indistinguishable from a row the book never "
        "priced"
    )
    assert str(BEYOND_INT4) in caplog.text, "the offending value was not named"


def test_a_board_with_no_bad_prices_logs_no_refusal(caplog) -> None:
    """So the error above means something when it appears."""
    import logging

    import main

    projections = pd.DataFrame({"PLAYER_NAME": ["A B"], "MARKET": ["PTS"]})
    prop_lines = pd.DataFrame({
        "player_name": ["A B"], "market": ["PTS"], "line": [20.5],
        "over_odds_american": [-110], "under_odds_american": [-105],
        "source": ["book"],
    })

    with caplog.at_level(logging.ERROR):
        main._attach_prop_line_prices(projections, prop_lines)

    assert "REFUSED AS IMPLAUSIBLE" not in caplog.text


def test_an_unpriced_row_is_not_reported_as_a_refusal(caplog) -> None:
    """
    A missing price is normal; a refused one is a provider fault.

    Conflating them would make the error fire on every ordinary board that
    has a one-sided market, and an error that fires constantly is read as
    noise.
    """
    import logging

    import main

    projections = pd.DataFrame({"PLAYER_NAME": ["A B"], "MARKET": ["PTS"]})
    prop_lines = pd.DataFrame({
        "player_name": ["A B"], "market": ["PTS"], "line": [20.5],
        "over_odds_american": [None], "under_odds_american": [-105],
        "source": ["book"],
    })

    with caplog.at_level(logging.ERROR):
        out = main._attach_prop_line_prices(projections, prop_lines)

    assert pd.isna(out["ODDS_OVER"].iloc[0])
    assert "REFUSED AS IMPLAUSIBLE" not in caplog.text
