"""
src/ingestion/propline.py — PropLine odds API client (NBA ONLY).

RESEARCH_ONLY. This module reads posted prop lines for analysis. It places
no wagers, recommends none, and ranks nothing by edge.

WHY THIS MATTERS HERE: until now the only "line" in this project was
RESEARCH_LINE, a trailing 10-game average. Labelling against it makes the
evaluation target self-referential — the models were scored on "is recent
form above medium-term form", not on prop skill. A real posted line with a
real capture timestamp is what makes CLV, de-vigged fair probabilities and
honest calibration possible at all.

SCOPE: NBA only, by explicit instruction. The provider also serves MLB,
NHL, soccer, tennis, NCAAB and more; every one of those is refused here
rather than filtered downstream, so a typo cannot silently spend quota on
a sport this project does not model. NCAA is excluded outright.

CREDENTIALS: the API key is read from the PROPLINE_API_KEY environment
variable and is never written to disk, logged, or placed in a URL. See
_auth_headers for why the header form is used rather than the documented
query-parameter form.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import requests

logger = logging.getLogger(__name__)

API_BASE_URL = "https://api.prop-line.com/v1"

# The ONLY sport this module will request. Not a default — a hard limit.
NBA_SPORT_KEY = "basketball_nba"

# Refused by name so the failure is loud and specific rather than a 404 or,
# worse, a successful pull of data this project must not carry.
FORBIDDEN_SPORT_SUBSTRINGS = ("ncaa", "college")

ENV_API_KEY = "PROPLINE_API_KEY"

# PropLine market key -> PropIQ stat column. Kept explicit: an unmapped key
# is logged and skipped, never guessed into a stat. Guessing here would
# attach a real posted line to the wrong statistic, which is worse than not
# ingesting it at all.
DEFAULT_MARKET_MAP: dict[str, str] = {
    "player_points": "PTS",
    "player_rebounds": "REB",
    "player_assists": "AST",
    "player_threes": "FG3M",
    "player_steals": "STL",
    "player_blocks": "BLK",
    "player_points_rebounds_assists": "PRA",
}

# Books whose "price" is not a two-way sportsbook price. Their rows are
# still ingested (they are real posted numbers) but flagged so the EV gate
# abstains on them instead of de-vigging a payout multiplier.
# /odds/history downsampling buckets. Anything else is rejected by the API,
# so it is rejected here rather than spent as a request.
HISTORY_INTERVALS: frozenset[str] = frozenset({"30s", "1m", "5m", "15m", "30m", "1h"})

# Event-age cap by plan, in days. Documented limits, not guesses — an event
# older than the cap comes back redacted rather than as an error.
TIER_EVENT_AGE_DAYS: dict[str, int | None] = {
    "hobby": 30,
    "pro": 90,
    "streaming_lite": 180,
    "streaming": 365,
    "enterprise": None,
}

# PropLine began recording in April 2026. Anything before that does not exist
# at any tier, so asking for it wastes quota against a guaranteed miss.
ARCHIVE_START = date(2026, 4, 1)


DFS_BOOKS = frozenset({"prizepicks", "underdog"})


class PropLineError(RuntimeError):
    """Base class. Never raised in a way that yields silent empty data."""


class PropLineAuthError(PropLineError):
    """Missing or rejected API key."""


class PropLineRateLimited(PropLineError):
    """429/503 that survived the retry budget."""


class PropLineUnavailable(PropLineError):
    """The provider could not serve this request."""


class PropLineTierLimited(PropLineError):
    """
    The event is older than the plan's event-age cap.

    PropLine answers an over-cap request with a REDACTED body rather than an
    error: market structure, ``redacted: true`` and an ``upgrade_url``. Read
    naively that is an event with no odds, which is indistinguishable from a
    game nobody priced. It is raised here so the two can never be confused.
    """


@dataclass(frozen=True)
class PropLineConfig:
    """
    How to talk to the provider.

    ``max_attempts`` bounds total tries per request, not retries after the
    first. Concurrency is deliberately absent: this client is sequential,
    because the provider's own guidance is that the bulk endpoints already
    return a whole slate per call and only per-event endpoints need fanning
    out at all.
    """

    base_url: str = API_BASE_URL
    timeout_seconds: float = 20.0
    max_attempts: int = 4
    backoff_seconds: float = 2.0
    # Below this many daily requests remaining, refuse to start new work so
    # a long run cannot exhaust the quota mid-slate and leave a half-built
    # picture that looks complete.
    min_daily_remaining: int = 5


@dataclass
class PropLineQuota:
    """Live quota state, parsed from response headers."""

    daily_limit: int | None = None
    daily_used: int | None = None
    daily_remaining: int | None = None
    daily_reset_utc: datetime | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "daily_limit": self.daily_limit,
            "daily_used": self.daily_used,
            "daily_remaining": self.daily_remaining,
            "daily_reset_utc": (
                self.daily_reset_utc.isoformat() if self.daily_reset_utc else None
            ),
        }


@dataclass
class PropLineRow:
    """One posted line for one player/market, both sides paired."""

    source: str
    player_name: str
    market: str
    line: float | None
    over_odds_american: int | None
    under_odds_american: int | None
    nba_player_id: str | None = None
    nba_game_id: str | None = None
    game_date: date | None = None
    captured_at_utc: datetime | None = None
    is_pickem: bool = False
    payout_multiplier: float | None = None
    status: str = "VALID"
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class PropLineSnapshotResult:
    """A pull from one source, matching the pipeline's snapshot contract."""

    source: str
    status: str
    captured_at_utc: datetime | None = None
    lines: list[PropLineRow] = field(default_factory=list)
    message: str | None = None
    quota: dict[str, Any] | None = None


def _auth_headers(api_key: str) -> dict[str, str]:
    """
    Authenticate via header, never the documented query-parameter form.

    The provider accepts ``?apiKey=...``, but a key in a URL leaks into
    places a header does not: web-server access logs, proxy logs, browser
    and CLI history, and any Referer sent onward. Same credential, strictly
    smaller blast radius.
    """
    return {
        "X-API-Key": api_key,
        "Accept": "application/json",
        "User-Agent": "PropIQ-Analytics/1.0 (research; NBA only)",
    }


def load_api_key(explicit: str | None = None) -> str:
    """
    Resolve the API key from the environment.

    Raises rather than returning None: a caller that silently proceeds
    without a key produces an empty result indistinguishable from "no games
    today", and that is exactly the confusion this project keeps trying to
    design out.
    """
    key = (explicit or os.environ.get(ENV_API_KEY) or "").strip()
    if not key:
        raise PropLineAuthError(
            f"DATA_NOT_AVAILABLE: no PropLine API key. Set {ENV_API_KEY} in your "
            ".env (never commit it). Without a key this module refuses to run "
            "rather than return an empty slate that looks like a quiet night."
        )
    return key


def _assert_nba_only(sport_key: str) -> str:
    """Refuse any sport but NBA, and refuse NCAA by name."""
    key = (sport_key or "").strip().lower()
    if any(bad in key for bad in FORBIDDEN_SPORT_SUBSTRINGS):
        raise PropLineError(
            f"Refusing sport {sport_key!r}: NCAA/college is explicitly out of scope "
            "for this project."
        )
    if key != NBA_SPORT_KEY:
        raise PropLineError(
            f"Refusing sport {sport_key!r}: this module is NBA-only "
            f"({NBA_SPORT_KEY}). Requesting another sport would spend quota on "
            "data this pipeline does not model."
        )
    return key


def _parse_epoch(value: Any) -> datetime | None:
    """Unix seconds -> aware UTC datetime. None on anything unparseable."""
    from src.utils.timezones import UTC

    try:
        return datetime.fromtimestamp(int(value), tz=UTC)
    except (TypeError, ValueError, OSError, OverflowError):
        return None


def _parse_iso(value: Any) -> datetime | None:
    """ISO-8601 -> aware UTC datetime. None on anything unparseable.

    Never falls back to "now": a fabricated observation time is the exact
    failure mode captured_at_utc was split out to prevent.
    """
    if not value or not isinstance(value, str):
        return None
    try:
        from src.utils.timezones import to_utc

        return to_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
    except (TypeError, ValueError):
        return None


class PropLineClient:
    """
    Sequential NBA-only client for the PropLine odds API.

    Deliberately not concurrent. The provider allows 20 in-flight requests
    per key, but its own guidance is that the bulk endpoints return a whole
    slate per call — a full NBA night is one request, not one per game. The
    only fan-out this project needs is per-event props, and at NBA slate
    sizes sequential calls stay far inside the burst budget while making
    quota use trivially predictable.
    """

    def __init__(
        self,
        api_key: str | None = None,
        config: PropLineConfig | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.config = config or PropLineConfig()
        self._api_key = load_api_key(api_key)
        self._session = session or requests.Session()
        self.quota = PropLineQuota()

    # -- internals ---------------------------------------------------------

    def _read_quota(self, response: requests.Response) -> None:
        """Track the live quota from response headers."""
        headers = response.headers

        def _int(name: str) -> int | None:
            try:
                return int(headers[name])
            except (KeyError, TypeError, ValueError):
                return None

        self.quota.daily_limit = _int("X-Daily-Limit") or self.quota.daily_limit
        used = _int("X-Daily-Used")
        remaining = _int("X-Daily-Remaining")
        if used is not None:
            self.quota.daily_used = used
        if remaining is not None:
            self.quota.daily_remaining = remaining
        reset = headers.get("X-Daily-Reset")
        if reset:
            self.quota.daily_reset_utc = _parse_epoch(reset)

        # RFC 8594: the provider promises a 12-month window on anything
        # deprecated. Surfacing it in the log is how that promise reaches us
        # without anyone polling a changelog page.
        if headers.get("Deprecation"):
            logger.warning(
                "PropLine reports this endpoint DEPRECATED (sunset %s). "
                "Plan a migration before that date.",
                headers.get("Sunset", "unspecified"),
            )

    @staticmethod
    def _retry_after_seconds(response: requests.Response, fallback: float) -> float:
        """Honour Retry-After when the provider sends it."""
        raw = response.headers.get("Retry-After")
        try:
            return max(0.0, float(raw))
        except (TypeError, ValueError):
            return fallback

    def _request(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        method: str = "GET",
        json_body: Any = None,
    ) -> Any:
        """
        Call one path with bounded retries.

        Retries only what is genuinely transient: 429 (both the daily cap
        and the burst limiter), 503 (concurrency), and connection errors.
        A 401/403 is a credential problem and retrying it just burns the
        remaining quota against the same rejection.
        """
        url = f"{self.config.base_url.rstrip('/')}/{path.lstrip('/')}"
        delay = self.config.backoff_seconds
        last_error: Exception | None = None

        for attempt in range(1, self.config.max_attempts + 1):
            try:
                # GET stays on session.get. Every caller and every injected
                # test double implements it; switching the common path to
                # session.request would silently widen what a "session" must
                # provide, and only POST actually needs the general form.
                if method.upper() == "GET":
                    response = self._session.get(
                        url,
                        headers=_auth_headers(self._api_key),
                        params=params or {},
                        timeout=self.config.timeout_seconds,
                    )
                else:
                    response = self._session.request(
                        method,
                        url,
                        headers=_auth_headers(self._api_key),
                        params=params or {},
                        json=json_body,
                        timeout=self.config.timeout_seconds,
                    )
            except requests.RequestException as exc:
                last_error = exc
                if attempt == self.config.max_attempts:
                    break
                logger.warning(
                    "PropLine request failed (%s), retry %d/%d in %.1fs",
                    exc, attempt, self.config.max_attempts, delay,
                )
                time.sleep(delay)
                delay *= 2
                continue

            self._read_quota(response)

            if response.status_code in (401, 403):
                # Never echo the key or the full URL into the log.
                raise PropLineAuthError(
                    f"PropLine rejected the API key (HTTP {response.status_code}). "
                    f"Check {ENV_API_KEY}; rotate it if it may have been exposed."
                )

            if response.status_code == 404:
                raise PropLineUnavailable(f"PropLine 404 for {path}")

            if response.status_code in (429, 503):
                wait = self._retry_after_seconds(response, delay)
                if attempt == self.config.max_attempts:
                    raise PropLineRateLimited(
                        f"PropLine still limiting after {attempt} attempts "
                        f"(HTTP {response.status_code}). Quota: {self.quota.as_dict()}"
                    )
                logger.warning(
                    "PropLine %s — honouring Retry-After %.1fs (attempt %d/%d)",
                    response.status_code, wait, attempt, self.config.max_attempts,
                )
                time.sleep(wait)
                delay *= 2
                continue

            if response.status_code >= 500:
                last_error = PropLineUnavailable(f"HTTP {response.status_code}")
                if attempt == self.config.max_attempts:
                    break
                time.sleep(delay)
                delay *= 2
                continue

            response.raise_for_status()
            try:
                return response.json()
            except ValueError as exc:
                raise PropLineUnavailable(
                    f"PropLine returned non-JSON for {path}"
                ) from exc

        raise PropLineUnavailable(
            f"PropLine unreachable for {path} after {self.config.max_attempts} "
            f"attempts: {last_error}"
        )

    def _check_quota_headroom(self) -> None:
        """Refuse to start work that would run the quota to zero mid-slate."""
        remaining = self.quota.daily_remaining
        if remaining is not None and remaining <= self.config.min_daily_remaining:
            raise PropLineRateLimited(
                f"Only {remaining} PropLine requests left today (reset "
                f"{self.quota.daily_reset_utc}). Stopping rather than "
                "building a partial slate that looks complete."
            )

    # -- public surface ----------------------------------------------------

    def fetch_events(self, sport_key: str = NBA_SPORT_KEY) -> list[dict[str, Any]]:
        """Upcoming NBA events. No odds — cheap discovery."""
        sport = _assert_nba_only(sport_key)
        payload = self._request(f"sports/{sport}/events")
        events = payload if isinstance(payload, list) else []
        logger.info("PropLine: %d NBA events", len(events))
        return events

    def fetch_event_markets(self, event_id: str) -> list[str]:
        """
        Market keys actually available on one event.

        Used before a props pull so the market list is discovered rather
        than assumed. A key this project has no mapping for is reported,
        not silently dropped — an unmapped market is a gap to close, and
        guessing which stat it means would attach a real posted line to the
        wrong statistic.
        """
        payload = self._request(f"sports/{NBA_SPORT_KEY}/events/{event_id}/markets")
        if isinstance(payload, dict):
            keys = payload.get("markets") or payload.get("market_keys") or []
        else:
            keys = payload or []
        return [str(k) for k in keys if k]

    def fetch_event_props(
        self,
        event_id: str,
        markets: list[str],
        bookmakers: list[str] | None = None,
    ) -> dict[str, Any]:
        """Player props for one NBA event, with each book's own ids."""
        if not markets:
            raise PropLineError("No market keys requested — refusing an unbounded pull")
        self._check_quota_headroom()
        params: dict[str, Any] = {
            "markets": ",".join(markets),
            # Book ids make the join exact instead of name-matched.
            "includeBookIds": "true",
        }
        if bookmakers:
            params["bookmakers"] = ",".join(bookmakers)
        payload = self._request(
            f"sports/{NBA_SPORT_KEY}/events/{event_id}/odds", params
        )
        return payload if isinstance(payload, dict) else {}


    # ------------------------------------------------------------------
    # historical endpoints
    # ------------------------------------------------------------------

    @staticmethod
    def _assert_not_redacted(payload: Any, *, what: str) -> Any:
        """
        Turn a tier-redacted body into a named refusal.

        An over-cap event returns market structure with ``redacted: true``
        and an ``upgrade_url`` — a 200 that reads like "this game had no
        odds". Left alone it would enter the archive as a genuine absence.
        """
        if isinstance(payload, dict) and payload.get("redacted"):
            raise PropLineTierLimited(
                f"{what} is older than this plan's event-age cap, so PropLine "
                "returned the redacted shape rather than the data. This is NOT "
                "an event without odds. Caps: hobby 30d, pro 90d, "
                "streaming_lite 180d, streaming 365d. "
                f"Upgrade path: {payload.get('upgrade_url') or 'see your plan page'}"
            )
        return payload

    def fetch_closing_odds(
        self,
        event_id: str,
        markets: list[str],
    ) -> dict[str, Any]:
        """
        Opening and closing price for every outcome, in one call.

        Closing is the last snapshot at or before ``commence_time`` — the
        number CLV is measured against. Opening is the first snapshot in the
        same 14-day pre-tip window.

        READ ``opening_age_seconds`` BEFORE TRUSTING AN OPEN. PropLine's
        archive starts April 2026, so for any book they began polling after a
        line was posted, ``opening_*`` is first-observed-by-them rather than
        the book's true open. Minutes rather than hours is the tell.
        """
        if not markets:
            raise PropLineError("No market keys requested — refusing an unbounded pull")
        self._check_quota_headroom()
        payload = self._request(
            f"sports/{NBA_SPORT_KEY}/events/{event_id}/odds/closing",
            {"markets": ",".join(markets)},
        )
        payload = self._assert_not_redacted(payload, what=f"event {event_id}")
        return payload if isinstance(payload, dict) else {}

    def fetch_odds_history(
        self,
        event_id: str,
        markets: list[str],
        *,
        from_utc: datetime | None = None,
        to_utc: datetime | None = None,
        relative_from: str | None = None,
        relative_to: str | None = None,
        interval: str | None = None,
        changes_only: bool = True,
    ) -> dict[str, Any]:
        """
        Full snapshot history per outcome, scoped and downsampled server-side.

        ``relative_*`` are offsets from tip (``-3h``, ``-30m``, ``0``) and are
        mutually exclusive with their absolute counterparts — sending both is
        rejected here rather than spent as a request against a 4xx.

        ``changes_only`` defaults True: a tick that repeats the previous
        (price, point, liquidity) carries no information and a season of them
        is most of the payload.
        """
        if not markets:
            raise PropLineError("No market keys requested — refusing an unbounded pull")
        if from_utc is not None and relative_from is not None:
            raise PropLineError("from_utc and relative_from are mutually exclusive")
        if to_utc is not None and relative_to is not None:
            raise PropLineError("to_utc and relative_to are mutually exclusive")
        if interval is not None and interval not in HISTORY_INTERVALS:
            raise PropLineError(
                f"interval {interval!r} is not one of {sorted(HISTORY_INTERVALS)}"
            )

        params: dict[str, Any] = {"markets": ",".join(markets)}
        if from_utc is not None:
            params["from"] = from_utc.isoformat()
        if to_utc is not None:
            params["to"] = to_utc.isoformat()
        if relative_from is not None:
            params["relative_from"] = relative_from
        if relative_to is not None:
            params["relative_to"] = relative_to
        if interval is not None:
            params["interval"] = interval
        if changes_only:
            params["changes_only"] = "true"

        self._check_quota_headroom()
        payload = self._request(
            f"sports/{NBA_SPORT_KEY}/events/{event_id}/odds/history", params
        )
        payload = self._assert_not_redacted(payload, what=f"event {event_id}")
        return payload if isinstance(payload, dict) else {}

    def fetch_event_results(self, event_id: str) -> Any:
        """
        Resolved prop outcomes for one event.

        The response shape is NOT in the documentation this client was built
        from, so the payload is returned as received. Normalise it with
        ``describe_resolution_payload`` first, which reports the fields it
        found instead of assuming which one holds the graded result.
        """
        self._check_quota_headroom()
        payload = self._request(f"sports/{NBA_SPORT_KEY}/events/{event_id}/results")
        return self._assert_not_redacted(payload, what=f"event {event_id}")

    def fetch_player_history(self, player_name: str) -> Any:
        """
        One player's prop history with resolution.

        Same caveat as ``fetch_event_results``: the response shape is not in
        the documented set, so nothing about it is assumed here.
        """
        if not str(player_name).strip():
            raise PropLineError("player_name is required")
        self._check_quota_headroom()
        payload = self._request(
            f"sports/{NBA_SPORT_KEY}/players/{str(player_name).strip()}/history"
        )
        return self._assert_not_redacted(payload, what=f"player {player_name}")

    def grade_clv(self, bets: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Grade placed bets against the close. Stateless — nothing is stored.

        Each bet needs ``ref, sport_key, event_id, market, bookmaker,
        selection, side, point, price`` and optionally ``stake``.

        TWO CLV NUMBERS COME BACK AND THEY ARE NOT INTERCHANGEABLE.
        ``clv_pct`` is price against price: quotable, and vig-blind, so it
        flatters a bet taken on the juicy side of a wide market.
        ``ev_vs_close_pct`` scores the price against the DE-VIGGED close and
        is the honest one — the same rule this project already applies to
        edge. Prefer it, and never sum either into a return.
        """
        if not bets:
            raise PropLineError("No bets supplied to grade")
        missing = [
            b.get("ref") or f"#{i}"
            for i, b in enumerate(bets)
            if not {"event_id", "market", "selection", "side", "price"} <= set(b)
        ]
        if missing:
            raise PropLineError(
                f"Bets {missing} are missing required fields (event_id, market, "
                "selection, side, price)"
            )
        self._check_quota_headroom()
        payload = self._request("clv/grade", method="POST", json_body=bets)
        return payload if isinstance(payload, dict) else {}

    def export_resolved_props(self, params: dict[str, Any] | None = None) -> Any:
        """
        Bulk CSV of resolved props plus closing lines (Pro tier).

        The single call that backfills an archive rather than accumulating
        one slate at a time. Its columns are not in the documented set, so
        the raw text comes back and the caller inspects it.
        """
        self._check_quota_headroom()
        return self._request("exports/resolved-props", params or {})


def _strip_player_namespace(player_id: Any) -> str | None:
    """
    ``nba:201939`` -> ``201939`` so it joins the panel's PLAYER_ID directly.

    The provider namespaces its ids by league. The bare numeric part is the
    league's own permanent id, which is exactly what the box-score panel
    carries, so stripping the prefix is what makes the join work without
    name matching. A non-NBA namespace returns None rather than a number
    that would join onto the wrong player.
    """
    if player_id is None:
        return None
    text = str(player_id).strip()
    if not text:
        return None
    if ":" not in text:
        return text
    namespace, _, bare = text.partition(":")
    if namespace.lower() != "nba":
        logger.debug("Ignoring non-NBA player id %r", text)
        return None
    return bare or None


def _american(price: Any) -> int | None:
    """American odds as an int, or None. 0 is not a price."""
    try:
        value = int(price)
    except (TypeError, ValueError):
        return None
    return value or None


def _is_outcome_live(outcome: dict[str, Any], market_last_update: datetime | None) -> bool:
    """
    Apply the provider's own staleness rule.

    Their docs are explicit: an outcome whose ``last_seen_at`` predates its
    market's ``last_update`` missed the latest delivery, which means the
    book has stopped sending that selection while still sending the market
    — a withdrawal in progress. Treat it as unavailable. Without this check
    a pulled selection keeps reading as a live, bettable price for the
    couple of minutes before it drops out of the feed entirely.
    """
    if market_last_update is None:
        return True
    last_seen = _parse_iso(outcome.get("last_seen_at"))
    if last_seen is None:
        return True  # book publishes no per-outcome sighting — nothing to test
    return last_seen >= market_last_update


def _dfs_row_is_comparable(book_key: str, outcome: dict[str, Any]) -> tuple[bool, str | None]:
    """
    Decide whether a DFS row is comparable to sportsbook consensus.

    Two separate traps, both of which make a row look like a mispriced edge
    when it is simply a different product:

    - PrizePicks posts ``goblin`` (easier line, lower payout) and ``demon``
      (harder line, higher payout) alongside its ``standard`` market line.
      Only ``standard`` is the market line.
    - Underdog scales payouts with ``payout_multiplier``. Anything but 1.0
      is a boost or discount, so the quoted price does not carry the full
      payout and cannot be compared like-for-like.
    """
    dfs_type = outcome.get("dfs_odds_type")
    if dfs_type is not None and str(dfs_type).lower() != "standard":
        return False, f"PrizePicks {dfs_type} line, not the standard market line"

    multiplier = outcome.get("payout_multiplier")
    if multiplier is not None:
        try:
            value = float(multiplier)
        except (TypeError, ValueError):
            return False, f"unreadable payout_multiplier {multiplier!r}"
        if abs(value - 1.0) > 1e-9:
            return False, f"payout_multiplier {value} — boosted/discounted pick"
    return True, None


def normalize_event_props(
    payload: dict[str, Any],
    market_map: dict[str, str] | None = None,
) -> list[PropLineRow]:
    """
    Flatten one event's odds payload into paired Over/Under rows.

    Pairing is keyed on (book, market, player, point) because books post
    alternate ladders — the same player can carry Over 24.5 and Over 27.5
    on one board, and pairing on player alone would splice two different
    lines into one row.

    Unknown fields are ignored by construction: the provider adds books,
    markets and fields inside v1 continuously, so this reads only what it
    understands and leaves the rest in ``raw``.
    """
    mapping = market_map or DEFAULT_MARKET_MAP
    event_id = str(payload.get("id") or "") or None
    commence = _parse_iso(payload.get("commence_time"))
    from src.utils.timezones import pacific_calendar_date

    # Pacific slate day, not UTC date — a 7pm PT tip is still that day's slate.
    game_date = pacific_calendar_date(commence) if commence else None

    # (book, market, player, point) -> partially built row
    pending: dict[tuple, dict[str, Any]] = {}
    unmapped: set[str] = set()
    skipped_stale = 0
    skipped_dfs = 0

    for book in payload.get("bookmakers") or []:
        book_key = str(book.get("key") or "").strip().lower()
        if not book_key:
            continue

        for market in book.get("markets") or []:
            raw_key = str(market.get("key") or "").strip().lower()
            stat = mapping.get(raw_key)
            if stat is None:
                if raw_key:
                    unmapped.add(raw_key)
                continue

            # A market the book has pulled pregame is not a live price.
            if market.get("suspended_at"):
                continue

            market_last_update = _parse_iso(market.get("last_update"))

            for outcome in market.get("outcomes") or []:
                if not _is_outcome_live(outcome, market_last_update):
                    skipped_stale += 1
                    continue

                comparable, why = _dfs_row_is_comparable(book_key, outcome)
                if not comparable:
                    skipped_dfs += 1
                    logger.debug("Skipping %s row: %s", book_key, why)
                    continue

                player = (outcome.get("description") or "").strip()
                if not player:
                    continue
                side = (outcome.get("name") or "").strip().lower()
                if side not in ("over", "under"):
                    continue

                point = outcome.get("point")
                try:
                    point_value = float(point) if point is not None else None
                except (TypeError, ValueError):
                    point_value = None

                key = (book_key, stat, player, point_value)
                row = pending.setdefault(key, {
                    "over": None,
                    "under": None,
                    "player_id": None,
                    "captured_at": None,
                    "captured_source": None,
                    "raw": {},
                })

                row[side] = _american(outcome.get("price"))
                row["player_id"] = row["player_id"] or _strip_player_namespace(
                    outcome.get("player_id")
                )

                # Observation time, in order of directness. book_updated_at
                # is the book's own publish time but only a few books send
                # it; last_change_at is the provider's observation of when
                # the price last moved and is populated for every book.
                # recorded_at is deliberately NOT used: it is the provider's
                # scrape time, which is an ingest time, not an observation
                # of the line.
                observed = _parse_iso(outcome.get("book_updated_at"))
                observed_source = "book_updated_at"
                if observed is None:
                    observed = _parse_iso(outcome.get("last_change_at"))
                    observed_source = "last_change_at"
                if observed is not None:
                    current = row["captured_at"]
                    if current is None or observed > current:
                        row["captured_at"] = observed
                        row["captured_source"] = observed_source

                row["raw"][side] = {
                    "price": outcome.get("price"),
                    "point": point,
                    "book_outcome_id": outcome.get("book_outcome_id"),
                    "outcome_id": outcome.get("outcome_id"),
                    "player_id": outcome.get("player_id"),
                    "liquidity": outcome.get("liquidity"),
                    "liquidity_updated_at": outcome.get("liquidity_updated_at"),
                    "payout_multiplier": outcome.get("payout_multiplier"),
                    "dfs_odds_type": outcome.get("dfs_odds_type"),
                    "last_change_at": outcome.get("last_change_at"),
                    "last_seen_at": outcome.get("last_seen_at"),
                    "book_updated_at": outcome.get("book_updated_at"),
                }

    rows: list[PropLineRow] = []
    for (book_key, stat, player, point_value), data in pending.items():
        raw = dict(data["raw"])
        raw["captured_at_source"] = data["captured_source"]
        raw["propline_market_map"] = {"stat": stat}

        rows.append(PropLineRow(
            source=book_key,
            player_name=player,
            market=stat,
            line=point_value,
            over_odds_american=data["over"],
            under_odds_american=data["under"],
            nba_player_id=data["player_id"],
            nba_game_id=event_id,
            game_date=game_date,
            captured_at_utc=data["captured_at"],
            # PrizePicks quotes synthetic +100/+100 whose real payout comes
            # from parlay correct-count, so it is a pick'em board however
            # numeric its prices look. Underdog at multiplier 1.0 is a
            # genuine two-way price and is NOT flagged — see below.
            is_pickem=(book_key == "prizepicks"),
            # Only a real scaling factor lands here. A standard 1.0 pick is
            # left None on purpose: market_ev_gate abstains whenever this is
            # set, and treating "multiplier present and equal to 1.0" as a
            # pick'em would throw away Underdog's real de-viggable prices.
            payout_multiplier=None,
            status="VALID",
            raw=raw,
        ))

    if unmapped:
        logger.warning(
            "PropLine served %d market key(s) this project has no mapping for: %s. "
            "They are skipped, not guessed — add them to DEFAULT_MARKET_MAP once "
            "you have confirmed which stat each one means.",
            len(unmapped), sorted(unmapped),
        )
    if skipped_stale:
        logger.info(
            "Skipped %d outcome(s) whose last_seen_at predates the market's "
            "last_update — the book is withdrawing them, so they are not live.",
            skipped_stale,
        )
    if skipped_dfs:
        logger.info(
            "Skipped %d DFS row(s) that are not comparable to sportsbook "
            "consensus (goblin/demon lines, or boosted payouts).", skipped_dfs,
        )
    return rows


def pull_nba_prop_lines(
    api_key: str | None = None,
    config: PropLineConfig | None = None,
    market_map: dict[str, str] | None = None,
    bookmakers: list[str] | None = None,
    max_events: int | None = None,
    session: requests.Session | None = None,
) -> list[PropLineSnapshotResult]:
    """
    Pull today's NBA player props, one snapshot per book.

    Returns snapshots grouped by source so the pipeline can report per-book
    status. A source that could not be read returns a DATA_NOT_AVAILABLE
    snapshot carrying the reason, never an empty VALID one — "no lines" and
    "we could not look" must not be the same signal downstream.
    """
    mapping = market_map or DEFAULT_MARKET_MAP

    try:
        client = PropLineClient(api_key=api_key, config=config, session=session)
    except PropLineAuthError as exc:
        return [PropLineSnapshotResult(
            source="propline", status="DATA_NOT_AVAILABLE", message=str(exc)
        )]

    try:
        events = client.fetch_events()
    except PropLineError as exc:
        return [PropLineSnapshotResult(
            source="propline", status="DATA_NOT_AVAILABLE",
            message=f"Could not list NBA events: {exc}",
            quota=client.quota.as_dict(),
        )]

    if not events:
        return [PropLineSnapshotResult(
            source="propline", status="DATA_NOT_AVAILABLE",
            message="No upcoming NBA events — expected off-season or between slates.",
            quota=client.quota.as_dict(),
        )]

    if max_events is not None:
        events = events[:max_events]

    wanted_markets = sorted(mapping)
    by_source: dict[str, list[PropLineRow]] = {}
    failures: list[str] = []

    for event in events:
        event_id = str(event.get("id") or "").strip()
        if not event_id:
            continue
        try:
            payload = client.fetch_event_props(event_id, wanted_markets, bookmakers)
        except PropLineUnavailable as exc:
            # A 404 here means the event id was merged or retired. Their docs
            # are explicit that it will never start resolving again, so this
            # is "re-discover the event", not "retry forever".
            failures.append(f"event {event_id}: {exc} (id may have been merged)")
            continue
        except PropLineRateLimited as exc:
            failures.append(f"stopped early: {exc}")
            break
        except PropLineError as exc:
            failures.append(f"event {event_id}: {exc}")
            continue

        for row in normalize_event_props(payload, mapping):
            by_source.setdefault(row.source, []).append(row)

    snapshots: list[PropLineSnapshotResult] = []
    for source, rows in sorted(by_source.items()):
        # The snapshot's captured_at is the newest observation it contains,
        # not the moment we pulled it. Those differ, and conflating them is
        # what makes a backfilled board look like it never moved.
        observed = [r.captured_at_utc for r in rows if r.captured_at_utc]
        snapshots.append(PropLineSnapshotResult(
            source=source,
            status="VALID",
            captured_at_utc=max(observed) if observed else None,
            lines=rows,
            quota=client.quota.as_dict(),
        ))

    if not snapshots:
        snapshots.append(PropLineSnapshotResult(
            source="propline", status="DATA_NOT_AVAILABLE",
            message=(
                "No mapped NBA prop lines across "
                f"{len(events)} event(s). " + ("; ".join(failures) if failures else "")
            ).strip(),
            quota=client.quota.as_dict(),
        ))
    elif failures:
        logger.warning("PropLine partial pull — %d problem(s): %s", len(failures), failures[:5])

    logger.info(
        "PropLine: %d row(s) across %d book(s) from %d event(s). Quota: %s",
        sum(len(s.lines) for s in snapshots), len(by_source), len(events),
        client.quota.as_dict(),
    )
    return snapshots
