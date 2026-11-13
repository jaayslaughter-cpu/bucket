"""No-vig consensus across the books that priced one prop.

WHAT THIS IS FOR, and the defect it exists to replace.

Audit of `kevon306/lockit-labs`, `review.devig_pair`: de-vig EACH bookmaker's
pair separately, then average the resulting fair probabilities across every
book that posted BOTH sides.

That turns out to be worth more here than a noise-reduction argument, because
of what the board does today. `pull_nba_prop_lines` fetches props keyed
`(book_key, stat, player, point_value)` and sets `PropLineRow.source` to the
BOOK, so several books' prices for one prop are already fetched and already
retained -- nothing new has to be ingested for this.

`decision_board.resolve_market` then sorts the candidates by
`_source_rank(...)` against `SOURCE_PRECEDENCE`, which holds the single entry
"propline". A book key such as "draftkings" matches no entry, so
`_source_rank` returns `len(precedence)` for EVERY book: they all tie, the
sort is stable, and the board takes whichever book happened to arrive first
and clear the EV gate. That is not a choice of the best price or of a
consensus -- it is arbitrary selection by insertion order, and it changes with
the order the provider returns events in.

So this module's value is not merely a tighter estimate. It replaces an
arbitrary draw with a figure computed from every book that priced the prop,
and it reports how far apart those books were, which the current path cannot
say at all.

WHY AVERAGE THE PROBABILITIES AND NOT THE PRICES. The vig is a property of
each book's own pair. Averaging American odds -- or decimal odds -- across
books mixes in whatever margin each one charges, so the result is not a fair
probability of anything. De-vig first, then average. Both halves matter.

WHY ONLY BOOKS THAT POSTED BOTH SIDES. A one-sided quote cannot have its vig
removed: there is no opposing price to normalise against. Including its raw
implied probability would push the consensus by exactly that book's margin,
in the direction of whichever side it happened to post.

WHY ONE LINE AT A TIME. Two books quoting 24.5 and 25.5 are pricing different
questions, and averaging their P(over) answers neither. Books on the modal
line form the consensus; the rest are counted out BY NAME rather than folded
in. `line_diff` is the tool for transporting a probability from one line to
another, and it belongs to the caller, not here.

NOT WIRED INTO SERVING. Changing which price the board reports is a serving
change, so per AGENTS.md it reaches production by winning a holdout in
`scripts/feature_ab.py`, not by being imported here. `resolve_market` is
untouched.

RESEARCH_ONLY. This produces a probability. It does not size or place a wager.
"""

from __future__ import annotations

import logging
import math
import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Sequence

from src.quant.contracts import plausible_american_odds
from src.quant.devig_methods import MULTIPLICATIVE, DevigMethodError, devig_two_way

logger = logging.getLogger(__name__)

RESEARCH_STATUS = "RESEARCH_ONLY"

STATUS_OK = "OK"
STATUS_INSUFFICIENT_BOOKS = "INSUFFICIENT_BOOKS"

#: Two books is the minimum that makes the word "consensus" mean anything. A
#: single book's de-vigged pair is already available without this module, and
#: calling it a consensus would overstate it.
DEFAULT_MIN_BOOKS = 2


@dataclass(frozen=True)
class ConsensusResult:
    """A consensus probability plus everything needed to judge it."""

    fair_probability_over: float | None
    line: float | None
    n_books: int
    books: tuple[str, ...]
    method: str
    status: str
    dispersion: float | None = None
    mean_hold: float | None = None
    excluded: tuple[str, ...] = field(default_factory=tuple)
    reason: str | None = None
    research_status: str = RESEARCH_STATUS

    @property
    def usable(self) -> bool:
        return self.status == STATUS_OK

    def as_dict(self) -> dict[str, Any]:
        return {
            "fair_probability_over": self.fair_probability_over,
            "line": self.line,
            "n_books": self.n_books,
            "books": list(self.books),
            "method": self.method,
            "status": self.status,
            "dispersion": self.dispersion,
            "mean_hold": self.mean_hold,
            "excluded": list(self.excluded),
            "reason": self.reason,
            "research_status": self.research_status,
        }


def _book_name(snapshot: Any, index: int) -> str:
    """The book's name, read by attribute like `propline_row_to_snapshot`."""
    for attribute in ("bookmaker", "source"):
        raw = getattr(snapshot, attribute, None)
        if raw is not None and str(raw).strip():
            return str(raw).strip().lower()
    return f"<unnamed book {index}>"


def _finite_line(snapshot: Any) -> float | None:
    raw = getattr(snapshot, "line", None)
    try:
        value = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def consensus_fair_probability(
    snapshots: Sequence[Any],
    *,
    method: str = MULTIPLICATIVE,
    min_books: int = DEFAULT_MIN_BOOKS,
) -> ConsensusResult:
    """
    Average the de-vigged P(over) across every book that priced one prop.

    `snapshots` are candidates for ONE player and ONE market -- the same thing
    `decision_board.resolve_market` receives. Each needs `line`,
    `over_odds_american` and `under_odds_american`, and a `bookmaker` or
    `source` to be named by.

    Books are grouped by line and the MODAL line wins; books on another line
    are named in `excluded` rather than averaged in. A book missing a side, or
    carrying a price `plausible_american_odds` refuses, is excluded by name
    too -- an excluded book is always visible, because a consensus that
    quietly rests on two of five books while reporting five would be the same
    class of error as a figure fitted on 40 rows quoted as 4,000.

    `dispersion` is the population standard deviation of the per-book fair
    probabilities: 0.0 when the books agree exactly, and the figure a caller
    should look at before trusting a consensus at all. The current board
    cannot report it, because it only ever looks at one book.
    """
    excluded: list[str] = []
    priced: list[tuple[str, float, float, float]] = []  # book, line, p_over, hold

    for index, snapshot in enumerate(snapshots or ()):
        name = _book_name(snapshot, index)
        line = _finite_line(snapshot)
        if line is None:
            excluded.append(f"{name}: no usable line")
            continue
        over = plausible_american_odds(getattr(snapshot, "over_odds_american", None))
        under = plausible_american_odds(getattr(snapshot, "under_odds_american", None))
        if over is None or under is None:
            missing = "over" if over is None else "under"
            if over is None and under is None:
                missing = "both sides"
            excluded.append(f"{name}: no usable {missing} price")
            continue
        try:
            fair = devig_two_way(over, under, method=method)
        except (DevigMethodError, ValueError) as exc:
            excluded.append(f"{name}: {exc}")
            continue
        priced.append((name, line, float(fair.fair_prob_a), float(fair.hold)))

    chosen = str(method).strip().lower()

    if not priced:
        return ConsensusResult(
            fair_probability_over=None, line=None, n_books=0, books=(),
            method=chosen, status=STATUS_INSUFFICIENT_BOOKS,
            excluded=tuple(excluded),
            reason="no book posted a two-sided price this module could de-vig",
        )

    # THE MODAL LINE. Ties break on the lower line, deterministically: an
    # arbitrary tie-break here would reintroduce exactly the order-dependence
    # this module exists to remove.
    counts = Counter(line for _, line, _, _ in priced)
    best = max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0]
    on_line = [row for row in priced if row[1] == best]
    for name, line, _, _ in priced:
        if line != best:
            excluded.append(f"{name}: priced {line} not the modal {best}")

    probabilities = [p for _, _, p, _ in on_line]
    books = tuple(name for name, _, _, _ in on_line)

    if len(on_line) < max(int(min_books), 1):
        return ConsensusResult(
            fair_probability_over=None, line=best, n_books=len(on_line),
            books=books, method=chosen, status=STATUS_INSUFFICIENT_BOOKS,
            excluded=tuple(excluded),
            reason=(
                f"{len(on_line)} book(s) priced the modal line {best}, "
                f"{min_books} required for a consensus"
            ),
        )

    return ConsensusResult(
        fair_probability_over=float(statistics.fmean(probabilities)),
        line=best,
        n_books=len(on_line),
        books=books,
        method=chosen,
        status=STATUS_OK,
        dispersion=float(statistics.pstdev(probabilities)) if len(probabilities) > 1 else 0.0,
        mean_hold=float(statistics.fmean([h for _, _, _, h in on_line])),
        excluded=tuple(excluded),
    )
