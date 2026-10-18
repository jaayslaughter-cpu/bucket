"""Shared HTTP for the ESPN public JSON endpoints (RESEARCH_ONLY, NBA only).

One place for the retry policy so the schedule, game and injury modules cannot
drift apart on it -- the same reason ``nba_playbyplay`` deliberately mirrors
``boxscores``.

NO CREDENTIALS. These endpoints are public and take none. If a future endpoint
needs a key it belongs in an env var like ``PROPLINE_API_KEY``, never here.

POLICY, and why each part is deliberate:
  - a 4xx is NOT retried, WITH ONE EXEMPTION: the request itself is wrong, so a
    second identical one wastes a call and hides the cause behind a
    timeout-shaped error. 429 is the exemption and IS retried, because there
    the request is fine and there have merely been too many of them; an
    integer Retry-After is honoured and capped at MAX_RETRY_AFTER_SECONDS,
    whose rule lives in ``utils.http_retry``.
  - a failed fetch RAISES. It never returns an empty dict, because a caller
    must be able to tell "no games tonight" from "the fetch failed".
  - the base host is overridable by env var so a test or mirror needs no code
    edit, and no URL is hardcoded at a call site.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import requests

# THE RETRY-AFTER RULE IS NOT THIS MODULE'S. It was, and this module's copy
# returned `nan` for ``Retry-After: nan`` -- `min(nan, 60.0)` is nan -- and
# `time.sleep(nan)` raises ValueError, so a garbled header crashed the fetch it
# was meant to pace. Three clients parsed that header and disagreed; one reader
# is the fix. The cap is re-exported under its old name because this module's
# docstring and `tests/test_espn_schedule.py` both refer to it here.
from src.utils.http_retry import MAX_RETRY_AFTER_SECONDS as MAX_RETRY_AFTER_SECONDS
from src.utils.http_retry import has_retry_after as _has_retry_after
from src.utils.http_retry import retry_after_seconds as _retry_after_seconds

logger = logging.getLogger(__name__)

SITE_API_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
CORE_API_BASE = "https://sports.core.api.espn.com"
WEB_API_BASE = "https://site.web.api.espn.com/apis/common/v3/sports/basketball/nba"

# The SITE API ROOT, with no endpoint path. Every ESPN module appends its own
# path to this (/scoreboard, /summary, /injuries, /teams/{id}/roster), so a
# mirror or a test override is set once and works for all of them. An earlier
# revision had espn_schedule read the same variable as a COMPLETE scoreboard
# URL, so overriding it sent summary and injury requests to paths under that
# scoreboard endpoint — silently wrong rather than failing.
ENV_SITE_BASE = "PROPIQ_ESPN_SITE_BASE"


def site_url(path: str, config: "EspnConfig | None" = None) -> str:
    """Join an endpoint path onto the configured site root."""
    base = (config or EspnConfig()).site_base.rstrip("/")
    return f"{base}/{path.lstrip('/')}"


TOO_MANY_REQUESTS = 429


class EspnError(RuntimeError):
    """An ESPN fetch failed. Never raised in place of legitimately empty data."""


@dataclass(frozen=True)
class EspnConfig:
    timeout: int = 30
    retry_attempts: int = 3
    retry_backoff: float = 2.0
    user_agent: str = "PropIQ-Analytics/research (public ESPN JSON)"
    site_base: str = field(
        default_factory=lambda: os.environ.get(ENV_SITE_BASE, "").strip() or SITE_API_BASE
    )


def get_json(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    config: EspnConfig | None = None,
    session: requests.Session | None = None,
) -> Any:
    """GET and decode JSON, with retry on transport faults only."""
    cfg = config or EspnConfig()
    sess = session or requests.Session()
    last: Exception | None = None

    for attempt in range(1, cfg.retry_attempts + 1):
        try:
            response = sess.get(
                url,
                params=params or None,
                timeout=cfg.timeout,
                headers={"Accept": "application/json", "User-Agent": cfg.user_agent},
            )
            status = getattr(response, "status_code", 200)
            if status == TOO_MANY_REQUESTS:
                # 429 is the one 4xx that IS transient: the request is fine,
                # there have just been too many of them. Falling through to the
                # generic 4xx raise would abandon a slate over rate limiting.
                wait = _retry_after_seconds(response, cfg.retry_backoff ** attempt)
                if attempt < cfg.retry_attempts:
                    logger.warning(
                        "espn: 429 for %s, attempt %d/%d; waiting %.1fs%s",
                        url, attempt, cfg.retry_attempts, wait,
                        " (Retry-After)" if _has_retry_after(response) else "",
                    )
                    time.sleep(wait)
                    continue
                raise EspnError(
                    f"ESPN rate-limited {url} on all {cfg.retry_attempts} attempts"
                )
            if 400 <= status < 500:
                raise EspnError(
                    f"ESPN returned {status} for {url} params={params}; "
                    "the request is wrong, so retrying will not help."
                )
            response.raise_for_status()
            return response.json()
        except EspnError:
            raise
        except (requests.exceptions.RequestException, ValueError) as exc:
            last = exc
            if attempt < cfg.retry_attempts:
                wait = cfg.retry_backoff ** attempt
                logger.warning(
                    "espn: attempt %d/%d for %s failed (%s); retrying in %.1fs",
                    attempt, cfg.retry_attempts, url, exc, wait,
                )
                time.sleep(wait)

    raise EspnError(f"ESPN unreachable after {cfg.retry_attempts} attempts: {url}: {last}")


def as_dict(value: Any) -> dict[str, Any]:
    """``value`` if it is a dict, else {} — for walking payloads defensively."""
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    """``value`` if it is a list, else [] — for walking payloads defensively."""
    return value if isinstance(value, list) else []
