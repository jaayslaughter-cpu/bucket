"""
src/ingestion/dfs_pricing.py — is a DFS row a two-way price, or a pick'em row?

ONE RULE, ONE READER, and this module exists because there were briefly two.
`ingestion/propline.normalize_event_props` has carried a careful, specific
answer for months:

  * PrizePicks quotes synthetic prices whose real payout comes from parlay
    correct-count, so it is a pick'em board HOWEVER NUMERIC ITS PRICES LOOK.
  * **Underdog at payout_multiplier 1.0 is a genuine two-way price** and is
    deliberately NOT flagged, because `quant.contracts.market_ev_gate`
    abstains when `is_pickem` is set OR `payout_multiplier` is not None --
    so flagging a standard Underdog pick throws away a real de-viggable
    quote.
  * A multiplier that is not 1.0 is a boost or a discount: the quoted price
    does not carry the full payout and the row is not comparable at all.
  * PrizePicks `goblin` and `demon` are different products from its
    `standard` market line.

On 2026-10-18 the new pick'em capture clients were written with an adapter
that hardcoded ``is_pickem=True``, no odds at all, and the multiplier passed
straight through. Every one of those four contradicted the above. The Apify
Underdog actor returns ``higher_american_odds: "-120"`` and
``lower_american_odds: "-114"`` at ``payout_multiplier: "1.0"`` -- precisely
the two-way price PropLine preserves on purpose -- and that adapter would
have dropped both prices and then set two flags that make the EV gate
abstain. Two readers of one fact, and the new one was wrong.

So the rule moved here and both callers read it. Same shape as
`utils.http_retry`, and for the same reason.

RESEARCH_ONLY. This classifies a posted price. It sizes nothing and places
nothing; what it decides is whether a row may be DE-VIGGED at all, which is
upstream of every EV question and is not itself an EV claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: Books whose quoted price is not a two-way price however numeric it looks.
#: PrizePicks posts synthetic even money and takes its margin from the
#: correct-count payout table instead.
#:
#: UNDERDOG AND SLEEPER ARE DELIBERATELY ABSENT. Underdog publishes real
#: per-side American odds; Sleeper prices per-leg dynamic odds, which is
#: exactly why `config/dfs_payouts.yaml` has no tiered matrix for it and its
#: legs go through the two-way path instead.
PICKEM_BOOKS = frozenset({"prizepicks"})

#: The only DFS odds type that is the book's market line. PrizePicks'
#: ``goblin`` (easier line, lower payout) and ``demon`` (harder line, higher
#: payout) sit beside it and are different products.
STANDARD_DFS_ODDS_TYPE = "standard"

#: How far from 1.0 a payout multiplier may sit and still mean "no boost".
MULTIPLIER_TOLERANCE = 1e-9


@dataclass(frozen=True)
class DfsPricing:
    """What may be done with one DFS row's price."""

    #: Whether the row can be compared to sportsbook consensus at all.
    comparable: bool
    #: Why not, when it cannot. None when it can.
    reason: str | None
    #: Whether the EV gate must refuse to de-vig this row.
    is_pickem: bool
    #: What to STORE as the row's payout multiplier. None for a standard
    #: 1.0 pick, on purpose: the gate abstains whenever this is set, so
    #: recording "present and equal to 1.0" would abstain on every ordinary
    #: Underdog row.
    stored_payout_multiplier: float | None


def is_pickem_book(book_key: Any) -> bool:
    """Whether this book's prices are pick'em rather than two-way."""
    return str(book_key or "").strip().lower() in PICKEM_BOOKS


def classify(
    book_key: Any,
    *,
    dfs_odds_type: Any = None,
    payout_multiplier: Any = None,
) -> DfsPricing:
    """
    The one answer, for one row.

    Takes the two fields that decide it rather than a provider payload, so
    both a PropLine outcome and an Apify actor record can ask the same
    question without either shape leaking in here.
    """
    pickem = is_pickem_book(book_key)

    if dfs_odds_type is not None and str(dfs_odds_type).strip().lower() != (
        STANDARD_DFS_ODDS_TYPE
    ):
        return DfsPricing(
            comparable=False,
            reason=f"PrizePicks {dfs_odds_type} line, not the standard market line",
            is_pickem=pickem,
            stored_payout_multiplier=None,
        )

    if payout_multiplier is None:
        return DfsPricing(True, None, pickem, None)

    try:
        value = float(payout_multiplier)
    except (TypeError, ValueError):
        return DfsPricing(
            comparable=False,
            reason=f"unreadable payout_multiplier {payout_multiplier!r}",
            is_pickem=pickem,
            stored_payout_multiplier=None,
        )

    if abs(value - 1.0) > MULTIPLIER_TOLERANCE:
        return DfsPricing(
            comparable=False,
            reason=f"payout_multiplier {value} — boosted/discounted pick",
            is_pickem=pickem,
            # Kept, because a non-comparable row that IS still stored
            # somewhere must carry the factor that made it so.
            stored_payout_multiplier=value,
        )

    # A standard 1.0 pick. Comparable, and the multiplier is NOT recorded.
    return DfsPricing(True, None, pickem, None)
