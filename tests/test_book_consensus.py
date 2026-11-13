"""
tests/test_book_consensus.py -- the board currently picks a book by accident.

WHAT THIS REPLACES, established by reading the code rather than assumed.

`pull_nba_prop_lines` keys props `(book_key, stat, player, point_value)` and
sets `PropLineRow.source` to the BOOK, so several books' prices for one prop
are already fetched and retained. `decision_board.resolve_market` then ranks
candidates with `_source_rank` against `SOURCE_PRECEDENCE`, which holds the
single entry "propline". No book key matches it, so every book gets
`len(precedence)`, they all tie, the sort is stable, and the board takes
whichever book arrived first and cleared the EV gate.

That makes the reported price a function of provider response order. The
consensus is order-independent by construction, and
`test_the_consensus_does_not_depend_on_the_order_books_arrive_in` is the test
that says so.

RESEARCH_ONLY. Probabilities. No wager.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import pytest

from src.quant.book_consensus import (
    DEFAULT_MIN_BOOKS,
    STATUS_INSUFFICIENT_BOOKS,
    STATUS_OK,
    consensus_fair_probability,
)
from src.quant.contracts import AMERICAN_ODDS_LIMIT, devig_two_way


@dataclass
class Snap:
    """Duck-typed stand-in: the module reads by attribute, not by import."""

    bookmaker: str
    line: float
    over_odds_american: int | None
    under_odds_american: int | None


@dataclass
class SourceOnly:
    """A candidate naming its book in `source` rather than `bookmaker`."""

    source: str
    line: float
    over_odds_american: int | None
    under_odds_american: int | None


# ---------------------------------------------------------------------------
# THE CORRECTNESS PROPERTY: de-vig first, then average
# ---------------------------------------------------------------------------


def test_it_averages_fair_probabilities_and_not_the_raw_prices() -> None:
    """
    THE HALF THAT IS EASY TO GET WRONG.

    The vig belongs to each book's own pair, so averaging prices carries each
    book's margin into the result and the figure is no longer a fair
    probability. This pins the arithmetic to the de-vig-then-average order by
    computing the expected value from `devig_two_way` directly.
    """
    books = [Snap("dk", 24.5, -110, -110), Snap("fd", 24.5, 150, -170)]
    result = consensus_fair_probability(books)

    expected = (
        devig_two_way(-110, -110)["fair_probability_over"]
        + devig_two_way(150, -170)["fair_probability_over"]
    ) / 2
    assert result.fair_probability_over == pytest.approx(expected, abs=1e-12)

    # And it is NOT the average of the implied probabilities, which still
    # carry the hold -- the two differ here by more than rounding.
    from src.quant.contracts import american_to_implied_probability
    vigged = (
        american_to_implied_probability(-110) + american_to_implied_probability(150)
    ) / 2
    assert abs(result.fair_probability_over - vigged) > 1e-3


def test_a_single_books_consensus_is_that_books_own_fair_probability() -> None:
    """The arithmetic must reduce correctly at n=1, even though min_books
    normally forbids reporting it."""
    result = consensus_fair_probability([Snap("dk", 24.5, -110, -130)], min_books=1)
    assert result.status == STATUS_OK
    assert result.fair_probability_over == pytest.approx(
        devig_two_way(-110, -130)["fair_probability_over"], abs=1e-12
    )
    assert result.dispersion == 0.0


# ---------------------------------------------------------------------------
# ORDER INDEPENDENCE -- the defect being replaced
# ---------------------------------------------------------------------------


def test_the_consensus_does_not_depend_on_the_order_books_arrive_in() -> None:
    """
    THE WHOLE POINT.

    `resolve_market` returns a different book's price depending on which
    arrived first, so the board's number moves with provider response order
    while the market has not moved at all. Every permutation here must give
    one answer.
    """
    books = [
        Snap("dk", 24.5, -110, -110),
        Snap("fd", 24.5, -115, -105),
        Snap("mgm", 24.5, -120, 100),
        Snap("czr", 24.5, 105, -125),
    ]
    baseline = consensus_fair_probability(books).fair_probability_over
    assert baseline is not None

    rng = random.Random(17)
    for _ in range(12):
        shuffled = books[:]
        rng.shuffle(shuffled)
        again = consensus_fair_probability(shuffled)
        assert again.fair_probability_over == pytest.approx(baseline, abs=1e-12)
        assert again.n_books == 4
        # The book LIST may be ordered differently, but the set cannot differ.
        assert set(again.books) == {"dk", "fd", "mgm", "czr"}


def test_a_tie_on_the_modal_line_breaks_deterministically() -> None:
    """
    Two lines with equal support must not resolve by input order, or the
    order-dependence this module removes would come back through the tie.
    """
    books = [Snap("dk", 24.5, -110, -110), Snap("fd", 25.5, -110, -110)]
    first = consensus_fair_probability(books, min_books=1)
    second = consensus_fair_probability(list(reversed(books)), min_books=1)
    assert first.line == second.line == 24.5, (first.line, second.line)


# ---------------------------------------------------------------------------
# exclusions, always by name
# ---------------------------------------------------------------------------


def test_a_one_sided_book_is_excluded_and_named() -> None:
    """
    A one-sided quote has no opposing price to normalise against, so its vig
    cannot be removed. Folding in its raw implied probability would shift the
    consensus by that book's whole margin.
    """
    result = consensus_fair_probability([
        Snap("dk", 24.5, -110, -110),
        Snap("fd", 24.5, -115, None),
        Snap("mgm", 24.5, -110, -110),
    ])
    assert result.n_books == 2
    assert set(result.books) == {"dk", "mgm"}
    assert any("fd" in item and "under" in item for item in result.excluded), result.excluded


def test_an_implausible_price_is_excluded_and_named() -> None:
    """
    A price beyond AMERICAN_ODDS_LIMIT is refused rather than de-vigged into a
    near-certainty -- the bound the 2026-11-11 outage installed.

    WHAT THIS DOES AND DOES NOT PIN, stated because a mutation showed the
    difference. Deleting this module's `plausible_american_odds` call leaves
    this test GREEN, because `devig_two_way` reaches
    `contracts.american_to_implied_probability`, which enforces the same bound
    and raises -- and the except branch excludes the book anyway. So the bound
    is enforced twice over, and this asserts the OUTCOME (the book is dropped
    and named) rather than which layer caught it. The explicit check is kept
    for its message: "no usable over price" names the problem, where the
    exception path reports whatever the de-vigger said.
    """
    result = consensus_fair_probability([
        Snap("dk", 24.5, -110, -110),
        Snap("bad", 24.5, AMERICAN_ODDS_LIMIT + 1, -110),
        Snap("mgm", 24.5, -105, -115),
    ])
    assert result.n_books == 2
    assert "bad" not in result.books
    assert any("bad" in item for item in result.excluded), result.excluded


def test_books_on_another_line_are_excluded_and_named() -> None:
    """
    Two books quoting 24.5 and 25.5 price different questions. The modal line
    forms the consensus and the rest are counted out visibly -- a consensus
    resting on two of five books while reporting five is the same class of
    error as a coefficient fitted on 40 rows quoted as 4,000.
    """
    result = consensus_fair_probability([
        Snap("dk", 24.5, -110, -110),
        Snap("fd", 24.5, -112, -108),
        Snap("odd", 27.5, -110, -110),
    ])
    assert result.line == 24.5
    assert result.n_books == 2
    assert any("odd" in item and "27.5" in item for item in result.excluded), result.excluded


def test_every_excluded_book_appears_in_the_report() -> None:
    """An exclusion that is not reported is indistinguishable from a book that
    was never there."""
    result = consensus_fair_probability([
        Snap("ok1", 24.5, -110, -110),
        Snap("ok2", 24.5, -110, -110),
        Snap("nosides", 24.5, None, None),
        Snap("noline", float("nan"), -110, -110),
        Snap("otherline", 30.5, -110, -110),
    ])
    assert result.n_books == 2
    joined = " | ".join(result.excluded)
    for name in ("nosides", "noline", "otherline"):
        assert name in joined, (name, result.excluded)


# ---------------------------------------------------------------------------
# refusal
# ---------------------------------------------------------------------------


def test_one_book_is_not_a_consensus_by_default() -> None:
    """A single de-vigged pair is already available without this module;
    calling it a consensus would overstate it."""
    assert DEFAULT_MIN_BOOKS == 2
    result = consensus_fair_probability([Snap("dk", 24.5, -110, -110)])
    assert result.status == STATUS_INSUFFICIENT_BOOKS
    assert result.fair_probability_over is None
    assert not result.usable
    assert "1 book(s)" in (result.reason or "")


def test_no_two_sided_book_refuses_rather_than_returning_a_number() -> None:
    result = consensus_fair_probability([
        Snap("a", 24.5, -110, None), Snap("b", 24.5, None, -110),
    ])
    assert result.status == STATUS_INSUFFICIENT_BOOKS
    assert result.fair_probability_over is None
    assert result.n_books == 0


def test_an_empty_candidate_list_refuses_without_raising() -> None:
    for empty in ([], None):
        result = consensus_fair_probability(empty)  # type: ignore[arg-type]
        assert result.status == STATUS_INSUFFICIENT_BOOKS
        assert result.fair_probability_over is None


# ---------------------------------------------------------------------------
# dispersion -- the figure the current board cannot report at all
# ---------------------------------------------------------------------------


def test_books_in_exact_agreement_have_zero_dispersion() -> None:
    result = consensus_fair_probability([
        Snap("a", 24.5, -110, -110), Snap("b", 24.5, -110, -110),
        Snap("c", 24.5, -110, -110),
    ])
    assert result.dispersion == pytest.approx(0.0, abs=1e-12)
    assert result.fair_probability_over == pytest.approx(0.5, abs=1e-12)


def test_disagreeing_books_report_a_larger_dispersion_than_agreeing_ones() -> None:
    """
    Dispersion is what tells a reader whether a consensus means anything. The
    single-book path cannot produce it, which is part of why an arbitrary pick
    reads as confidently as a unanimous board.
    """
    tight = consensus_fair_probability([
        Snap("a", 24.5, -110, -110), Snap("b", 24.5, -112, -108),
    ])
    wide = consensus_fair_probability([
        Snap("a", 24.5, -250, 200), Snap("b", 24.5, 200, -250),
    ])
    assert wide.dispersion > tight.dispersion * 5, (tight.dispersion, wide.dispersion)


# ---------------------------------------------------------------------------
# duck typing and provenance
# ---------------------------------------------------------------------------


def test_a_candidate_naming_its_book_in_source_is_read_too() -> None:
    """`propline_row_to_snapshot` reads by attribute so the quant layer takes
    no dependency on ingestion; this follows that convention."""
    result = consensus_fair_probability([
        SourceOnly("dk", 24.5, -110, -110), SourceOnly("fd", 24.5, -110, -110),
    ])
    assert result.status == STATUS_OK
    assert set(result.books) == {"dk", "fd"}


def test_the_result_carries_its_own_provenance() -> None:
    result = consensus_fair_probability([
        Snap("dk", 24.5, -110, -110), Snap("fd", 24.5, -110, -110),
    ])
    payload = result.as_dict()
    assert payload["n_books"] == 2
    assert payload["line"] == 24.5
    assert payload["method"] == "multiplicative"
    assert payload["research_status"] == "RESEARCH_ONLY"
    assert payload["mean_hold"] == pytest.approx(0.047619, abs=1e-5)


def test_the_method_is_selectable_and_reported() -> None:
    """Four de-vig methods exist in this repo; a consensus that silently used
    multiplicative while labelled otherwise would misreport itself."""
    result = consensus_fair_probability(
        [Snap("dk", 24.5, -250, 200), Snap("fd", 24.5, -240, 195)], method="shin",
    )
    assert result.status == STATUS_OK
    assert result.method == "shin"
    multiplicative = consensus_fair_probability(
        [Snap("dk", 24.5, -250, 200), Snap("fd", 24.5, -240, 195)],
    )
    assert result.fair_probability_over != pytest.approx(
        multiplicative.fair_probability_over, abs=1e-9
    ), "shin and multiplicative agreed exactly on a lopsided pair"
