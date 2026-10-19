"""
src/utils/http_retry.py — how long to wait when a server says "later".

ONE RULE, ONE READER. Until now there were three implementations of
``Retry-After`` in this tree and they did not agree:

    src/settlement/boxscore_fetcher.py  hardened (math.isfinite, capped)
    src/ingestion/espn_client.py        capped, but returns nan for `nan`
    src/ingestion/propline.py           neither capped nor hardened

Measured, not assumed. Given ``Retry-After: nan`` the ESPN client returned
``nan`` and ``time.sleep(nan)`` raises ValueError("Invalid value NaN") -- a
rate limit became a crashed fetch. Given ``Retry-After: inf`` PropLine
returned ``inf`` and ``time.sleep(inf)`` raises OverflowError. Given
``Retry-After: 86400`` PropLine returned 86400.0 and would have slept the
slate job for twenty-four hours, which is worse than crashing because
nothing above it has a timeout. Given ``-1`` or ``nan`` PropLine returned
0.0 and retried instantly -- backing off from a rate limiter by not backing
off at all.

Same defect shape as `board_markets`, the PRA rollups, the parlay ledger,
the game-id joins and `check_state_dir`: two (here three) readers of one
fact, drifting apart until the weakest one decides what happens in
production. One reader is the fix, every time.

THE RULE. Only the plain integer-seconds form is honoured. The HTTP-date
form (RFC 7231 permits it) is IGNORED rather than parsed approximately,
because a wrong date parse sleeps for either no time or a very long one.
Anything non-finite, zero or negative falls back to the caller's own
backoff. Whatever survives is capped: a server asking a slate job to wait an
hour is not something to obey inside a window that closes at tip-off -- cap
it and let the caller exhaust its attempts and report, rather than hang.

RESEARCH_ONLY project. This is a sleep duration.
"""

from __future__ import annotations

import math
from typing import Any

#: Longest wait any client will honour from a server. A slate job has a
#: deadline that a remote host knows nothing about.
MAX_RETRY_AFTER_SECONDS = 60.0

HEADER = "Retry-After"


def _headers_of(source: Any) -> Any:
    """Accept either a headers mapping or an object carrying one.

    The three call sites were split between the two conventions -- ESPN
    passed the response, the other two passed ``response.headers`` -- and
    making the shared rule take both is what let all three adopt it without
    a flag day.
    """
    if hasattr(source, "headers"):
        return getattr(source, "headers", None) or {}
    return source if source is not None else {}


def has_retry_after(source: Any) -> bool:
    """Whether the server actually sent the header, for logging only."""
    headers = _headers_of(source)
    try:
        return HEADER in headers
    except TypeError:
        return False


def valid_seconds(value: Any) -> float | None:
    """
    THE RULE ITSELF: a finite, positive, capped number of seconds, or None
    for "this is not a usable wait".

    Separated from `sanitise_seconds` so a caller with a SECOND source can
    tell "the server gave me nothing usable" from "the server said wait
    ``fallback``". PropLine has exactly that: a hint in the 429 body and a
    ``Retry-After`` header. Collapsing both answers into ``fallback`` made an
    unusable body hint shadow a perfectly good header -- see
    `PropLineClient._retry_after_seconds`.
    """
    if value is None:
        return None
    try:
        seconds = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if not math.isfinite(seconds) or seconds <= 0:
        return None
    return min(seconds, MAX_RETRY_AFTER_SECONDS)


def sanitise_seconds(value: Any, fallback: float) -> float:
    """
    One wait value, validated and capped, or ``fallback``.

    Extracted so a hint that did NOT arrive in the header gets the identical
    guard. PropLine signals its burst-limit backoff in the 429 BODY as
    ``detail.retry_after_seconds`` -- its own SDK reads it there -- and a
    second code path applying its own looser rule to that number is how the
    three-readers defect this module was created to end would come straight
    back.
    """
    seconds = valid_seconds(value)
    return float(fallback) if seconds is None else seconds


def retry_after_seconds(source: Any, fallback: float) -> float:
    """The server's ``Retry-After`` in seconds, else ``fallback``, capped."""
    headers = _headers_of(source)
    try:
        raw = headers.get(HEADER)
    except (AttributeError, TypeError):
        return float(fallback)
    return sanitise_seconds(raw, fallback)
