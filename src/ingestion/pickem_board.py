"""
src/ingestion/pickem_board.py — read one pick'em board, from the wire or a file.

ONE READER FOR BOTH SOURCES. Underdog and Sleeper differ in exactly one
respect: where the bytes come from. Underdog publishes a board endpoint;
Sleeper, per the logged-in check recorded in `docs/pickem_props.md` on
2026-09-15, publishes no readable board at all, so its capture path is
operator-supplied JSON. Everything after the bytes -- the mapping, the NBA
filter, the status vocabulary, the timestamps -- is identical, and writing it
twice is how the two would drift apart. `underdog_props` and `sleeper_props`
are source-specific wrappers over this.

WHAT THIS WILL NOT DO. A 401 or 403 is reported and NOT retried. A board that
answers an ordinary client with a refusal is not a transient failure, and
retrying it is the first step of the thing `docs/pickem_props.md` bans: no
TLS impersonation, no fingerprint spoofing, no proxy rotation, no captcha
solving. The snapshot says the board declined to serve us and stops there.

NO CREDENTIALS. A public board needs none. If a board requires a token, a
cookie or a signed header, it is not public, capturing it is not what was
approved on 2026-09-15, and this module has nowhere to put one.

RESEARCH_ONLY. Fetches posted lines and timestamps them.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

from src.ingestion.pickem_schema import (
    REQUIRED_FIELDS,
    STATUS_EMPTY,
    STATUS_NOT_CONFIGURED,
    STATUS_UNAVAILABLE,
    STATUS_VALID,
    PickemSnapshot,
    PickemSourceConfig,
    rows_from_payload,
)
from src.utils.http_retry import retry_after_seconds

logger = logging.getLogger(__name__)

TIMEOUT_SECONDS = 20.0
MAX_ATTEMPTS = 3
BACKOFF_BASE = 2.0

#: Honest and unremarkable. This identifies the project and does NOT
#: impersonate a browser: a board that serves only what looks like Chrome is
#: declining automated clients, and dressing up as one to get past that is
#: the evasion `docs/pickem_props.md` bans.
HEADERS = {
    "Accept": "application/json",
    "User-Agent": "PropIQ-Analytics/1.0 (NBA research; pick'em line capture)",
}

#: Statuses that mean "the board refused an ordinary client". Never retried.
REFUSED_STATUSES = (401, 403, 451)


def _now() -> datetime:
    """When WE read the board. Never used as a line's `captured_at_utc`."""
    return datetime.now(UTC)


def fetch_board_payload(
    url: str, *, session: requests.Session | None = None
) -> tuple[Any, int | None, str | None]:
    """
    GET one board. Returns ``(payload, http_status, error)``.

    Never raises: a capture that throws takes out whatever called it, and the
    whole point of a snapshot status is that a dark board is data.
    """
    http = session or requests
    delay = BACKOFF_BASE

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = http.get(url, headers=HEADERS, timeout=TIMEOUT_SECONDS)
        except requests.RequestException as exc:
            if attempt == MAX_ATTEMPTS:
                return None, None, f"{type(exc).__name__}: {exc}"
            time.sleep(delay)
            delay *= 2
            continue

        status = response.status_code

        if status in REFUSED_STATUSES:
            # NOT RETRIED, and that is the point. See the module docstring.
            return None, status, (
                f"the board answered HTTP {status} to an ordinary client. This "
                f"is not retried and no fingerprint is spoofed to get past it "
                f"— see docs/pickem_props.md. If this board is not publicly "
                f"readable, the sanctioned path is operator-supplied JSON."
            )

        if status == 404:
            return None, status, (
                f"HTTP 404 for {url} — the board path has moved. Re-record it "
                f"and update config/pickem_sources.yaml."
            )

        if status == 429 or status >= 500:
            if attempt == MAX_ATTEMPTS:
                return None, status, f"HTTP {status} after {attempt} attempt(s)"
            wait = retry_after_seconds(response.headers, delay)
            logger.warning(
                "pick'em board %s — HTTP %s, waiting %.1fs (attempt %d/%d)",
                url, status, wait, attempt, MAX_ATTEMPTS,
            )
            time.sleep(wait)
            delay *= 2
            continue

        if status >= 400:
            return None, status, f"HTTP {status}"

        try:
            return response.json(), status, None
        except ValueError as exc:
            return None, status, f"the board returned non-JSON: {exc}"

    return None, None, f"unreachable after {MAX_ATTEMPTS} attempts"


def newest_local_payload(directory: str | Path) -> tuple[Any, str | None, str | None]:
    """
    The newest operator-supplied board response in a directory.

    Returns ``(payload, path, error)``. This is Sleeper's whole capture path
    and Underdog's fallback, so "the directory is empty" is a NAMED state
    rather than a zero-row success.
    """
    root = Path(directory)
    if not root.is_dir():
        return None, None, f"no directory at {root}"
    candidates = sorted(
        (p for p in root.glob("*.json") if p.is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None, None, (
            f"no *.json in {root}. This source is read from operator-supplied "
            f"board responses; drop one in and re-run."
        )
    newest = candidates[0]
    try:
        return json.loads(newest.read_text(encoding="utf-8")), str(newest), None
    except (OSError, ValueError) as exc:
        return None, str(newest), f"{newest} is not readable JSON: {exc}"



# ---------------------------------------------------------------------------
# the Apify transport: a PAID Actor run
# ---------------------------------------------------------------------------

#: Standard Apify REST. The `owner~name` slug form is the one the Actor's own
#: listing uses (`GET /v2/acts/zen-studio~underdog-player-props/builds/
#: default`); the run-sync path is Apify's documented synchronous run. Kept in
#: config as `apify.endpoint` so a change at their end is an edit, not a
#: release.
APIFY_RUN_SYNC_TEMPLATE = (
    "https://api.apify.com/v2/acts/{actor}/run-sync-get-dataset-items"
)
ENV_APIFY_TOKEN = "APIFY_API_TOKEN"

#: An Actor run is not a page fetch. These boards take tens of seconds.
APIFY_TIMEOUT_SECONDS = 180.0


def apify_actor_slug(actor: str) -> str:
    """``owner/name`` as Apify's URL form ``owner~name``."""
    return str(actor).strip().replace("/", "~")


def fetch_apify_dataset(
    spec: dict[str, Any],
    *,
    session: requests.Session | None = None,
) -> tuple[Any, int | None, str | None]:
    """
    Run one Apify Actor synchronously and return its dataset items.

    THIS CALL COSTS MONEY -- these Actors bill per line captured -- so it is
    reached only from a source whose `enabled` is true, and nothing here
    retries a refusal or a payment problem. Never raises, and never logs the
    token.
    """
    import os

    actor = str(spec.get("actor") or "").strip()
    if not actor:
        return None, None, "the apify block names no `actor`"

    token_env = str(spec.get("token_env") or ENV_APIFY_TOKEN)
    token = (os.environ.get(token_env) or "").strip()
    if not token:
        return None, None, (
            f"${token_env} is not set, so no Apify run was attempted. Get a "
            f"token from https://console.apify.com/account/integrations. These "
            f"Actors bill per line: see the cost note in "
            f"config/pickem_sources.yaml before the first paid run."
        )

    template = str(spec.get("endpoint") or APIFY_RUN_SYNC_TEMPLATE)
    url = template.format(actor=apify_actor_slug(actor))
    run_input = dict(spec.get("input") or {})
    timeout = float(spec.get("timeout_seconds") or APIFY_TIMEOUT_SECONDS)

    http = session or requests
    delay = BACKOFF_BASE

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = http.post(
                url,
                json=run_input,
                headers={
                    # The token goes in a header, never the query string: a
                    # key in a URL lands in access logs, proxy logs and any
                    # Referer sent onward. Same rule as `propline._auth_headers`.
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                timeout=timeout,
            )
        except requests.RequestException as exc:
            if attempt == MAX_ATTEMPTS:
                return None, None, f"{type(exc).__name__}: {exc}"
            time.sleep(delay)
            delay *= 2
            continue

        status = response.status_code

        if status in (401, 403):
            # A credential problem. Retrying spends nothing but tells us
            # nothing either, and the token must not appear in the message.
            return None, status, (
                f"Apify rejected the token (HTTP {status}). Check ${token_env}; "
                f"rotate it if it may have been exposed."
            )
        if status == 402:
            return None, status, (
                "Apify reports a payment or plan limit (HTTP 402). These "
                "Actors bill per line; nothing was captured."
            )
        if status == 404:
            return None, status, (
                f"Apify has no Actor {actor!r} (HTTP 404), or the run-sync "
                f"path has moved. Check `apify.actor` and `apify.endpoint` in "
                f"config/pickem_sources.yaml."
            )
        if status == 429 or status >= 500:
            if attempt == MAX_ATTEMPTS:
                return None, status, f"Apify HTTP {status} after {attempt} attempt(s)"
            wait = retry_after_seconds(response.headers, delay)
            logger.warning(
                "Apify %s — HTTP %s, waiting %.1fs (attempt %d/%d)",
                actor, status, wait, attempt, MAX_ATTEMPTS,
            )
            time.sleep(wait)
            delay *= 2
            continue
        if status >= 400:
            return None, status, f"Apify HTTP {status}"

        try:
            items = response.json()
        except ValueError as exc:
            return None, status, f"Apify returned non-JSON: {exc}"

        # run-sync-get-dataset-items answers with the ITEM ARRAY itself. The
        # mapping's `rows` path is read against whatever shape arrives, so a
        # bare list is wrapped under a known key rather than special-cased
        # in the parser.
        if isinstance(items, list):
            return {"items": items}, status, None
        return items, status, None

    return None, None, f"Apify unreachable after {MAX_ATTEMPTS} attempts"


def capture(
    config: PickemSourceConfig,
    *,
    session: requests.Session | None = None,
    from_file: str | Path | None = None,
) -> PickemSnapshot:
    """
    One snapshot of one board, under that source's declared mapping.

    `from_file` reads a recorded response instead of the network, which is how
    a mapping is checked before anyone trusts it on the wire.
    """
    ingested = _now()

    if not config.configured:
        # BEFORE any request. An unconfigured source must not spend a call,
        # and must not report an empty board -- "we are not set up" and "the
        # board posted nothing" send an operator to different places.
        return PickemSnapshot(
            source=config.name,
            status=STATUS_NOT_CONFIGURED,
            ingested_at_utc=ingested,
            message=config.why_not_configured(),
        )

    http_status: int | None = None
    if from_file is not None:
        try:
            payload = json.loads(Path(from_file).read_text(encoding="utf-8"))
            error = None
        except (OSError, ValueError) as exc:
            payload, error = None, f"{from_file} is not readable JSON: {exc}"
    elif config.transport == "apify":
        payload, http_status, error = fetch_apify_dataset(
            config.apify, session=session
        )
    elif config.board_url:
        payload, http_status, error = fetch_board_payload(
            config.board_url, session=session
        )
    elif config.local_dir:
        payload, where, error = newest_local_payload(config.local_dir)
        if where:
            logger.info("%s: read operator-supplied board %s", config.name, where)
    else:
        payload, error = None, (
            f"{config.name} has neither a board_url nor a local_dir in "
            f"config/pickem_sources.yaml"
        )

    if error is not None:
        return PickemSnapshot(
            source=config.name,
            status=STATUS_UNAVAILABLE,
            ingested_at_utc=ingested,
            message=error,
            http_status=http_status,
        )

    lines, unresolved, dropped = rows_from_payload(payload, config)

    if not lines:
        # READ, BUT NOTHING USABLE IN IT -- and which of those two matters.
        # A board that posted nothing leaves the mapping resolving cleanly; a
        # restructured one does not. Reporting both as EMPTY is how a stale
        # mapping hides as the off-season for as long as nobody checks.
        #
        # The test is STRUCTURAL ONLY: the row array itself not resolving, or
        # a required field missing from every row. A null value in one row of
        # a hundred is data about that row, not evidence about the mapping,
        # and treating it as the latter would cry stale on a board that is
        # merely incomplete.
        structural = "rows" in unresolved or any(
            name in unresolved for name in REQUIRED_FIELDS
        )
        if structural:
            return PickemSnapshot(
                source=config.name,
                status=STATUS_UNAVAILABLE,
                ingested_at_utc=ingested,
                http_status=http_status,
                unresolved=unresolved,
                message=(
                    f"the board was read and produced NO rows, with "
                    f"{unresolved} unresolved. Either the paths have moved or "
                    f"the board posted nothing readable — re-record a response "
                    f"and run `--describe` over it to tell which."
                ),
            )
        return PickemSnapshot(
            source=config.name,
            status=STATUS_EMPTY,
            ingested_at_utc=ingested,
            http_status=http_status,
            message=(
                f"the board was read and posted no NBA lines this project "
                f"maps{': ' + str(dropped) if dropped else ''}"
            ),
        )

    return PickemSnapshot(
        source=config.name,
        status=STATUS_VALID,
        ingested_at_utc=ingested,
        http_status=http_status,
        lines=lines,
        unresolved=unresolved,
        message=(f"dropped {dropped}" if dropped else None),
    )
