"""
Both NBA CDN clients must back off the same way, and both must see a 429.

FOUND BY AUDIT, 2026-10-14. `docs/autonomy_audit_2026-10-10.md` P4 recorded
exactly one client on a daily path with neither exponential backoff nor a 429
branch, while `ingestion/espn_client` and `ingestion/propline` had both. The
fix landed in `src/settlement/boxscore_fetcher.py` — whose constant now carries
the finding in its own comment:

    #: EXPONENTIAL, as the base of `RETRY_BACKOFF ** attempt`. This was
    #: multiplied by the attempt number instead -- 2s, 4s, 6s -- which
    #: docs/autonomy_audit_2026-10-10.md P4 recorded as ...

and `src/ingestion/nba_playbyplay.py`, which IMPORTS that very constant, was
left using it as a multiplier: ``RETRY_BACKOFF * attempt``. So the one module
documenting the defect is the one that fixed it, and the module reading its
constant still has it. It also has 403 and 404 branches but no 429, so a rate
limit falls through to `raise_for_status` and is retried on the linear schedule
with the server's own Retry-After ignored — against cdn.nba.com, which is the
host this project is most careful about.

RESEARCH_ONLY. Retry arithmetic on a read-only public endpoint.
"""

from __future__ import annotations

import inspect

import pytest
import requests

import src.ingestion.nba_playbyplay as pbp
import src.settlement.boxscore_fetcher as fetcher


class _Resp:
    def __init__(self, status: int, headers: dict | None = None, payload: dict | None = None):
        self.status_code = status
        self.headers = headers or {}
        self._payload = payload or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"{self.status_code}")


class _Session:
    """Fails with `statuses` in order, recording every sleep the client asks for."""

    def __init__(self, statuses):
        self.statuses = list(statuses)
        self.calls = 0

    def get(self, url, **kwargs):
        self.calls += 1
        status = self.statuses[min(self.calls - 1, len(self.statuses) - 1)]
        if isinstance(status, Exception):
            raise status
        return _Resp(status, headers={"Retry-After": "7"} if status == 429 else {})


@pytest.fixture
def sleeps(monkeypatch):
    recorded: list[float] = []
    monkeypatch.setattr(pbp.time, "sleep", recorded.append)
    monkeypatch.setattr(fetcher.time, "sleep", recorded.append)
    return recorded


# --- exponential, in both ---------------------------------------------------


def test_play_by_play_backs_off_exponentially(sleeps, monkeypatch):
    """
    FOUR ATTEMPTS, NOT THREE, because at three the bug is invisible: the client
    sleeps only after attempts 1 and 2, and ``2 * attempt`` and ``2 ** attempt``
    are both 2s then 4s. The schedules diverge for the first time on the third
    sleep -- 6s linear against 8s exponential -- so a test at the shipped
    RETRY_ATTEMPTS passes either way and proves nothing.

    The defect is therefore LATENT at today's configuration and becomes real
    the moment anyone raises the attempt count. It is still wrong arithmetic
    against the constant's documented meaning, and this is what pins it.
    """
    monkeypatch.setattr(pbp, "RETRY_ATTEMPTS", 4)
    session = _Session([requests.exceptions.ConnectionError("boom")] * 4)
    with pytest.raises(pbp.PlayByPlayError):
        pbp.fetch_playbyplay("0022500001", session=session)

    assert session.calls == 4
    base = fetcher.RETRY_BACKOFF
    assert sleeps == [base ** 1, base ** 2, base ** 3], (
        f"waited {sleeps}, expected {[base ** 1, base ** 2, base ** 3]}: the "
        "constant is documented as the BASE of RETRY_BACKOFF ** attempt, and a "
        "multiplier gives 2s, 4s, 6s"
    )


def test_the_box_score_fetcher_already_does_and_still_does(sleeps, monkeypatch):
    monkeypatch.setattr(fetcher, "RETRY_ATTEMPTS", 4)
    session = _Session([requests.exceptions.ConnectionError("boom")] * 4)
    with pytest.raises(fetcher.BoxScoreError):
        fetcher.fetch_boxscore("0022500001", session=session)
    base = fetcher.RETRY_BACKOFF
    assert sleeps == [base ** 1, base ** 2, base ** 3]


# --- 429, in both -----------------------------------------------------------


def test_play_by_play_honours_retry_after_on_429(sleeps):
    """
    A 429 is not a wrong request: the server is asking for a pause and saying
    how long. Ignoring it against cdn.nba.com is how access gets revoked.
    """
    session = _Session([429, 429, 429])
    with pytest.raises(pbp.PlayByPlayError) as excinfo:
        pbp.fetch_playbyplay("0022500001", session=session)

    assert "rate-limit" in str(excinfo.value).lower() or "429" in str(excinfo.value)
    assert sleeps == [7.0, 7.0], (
        f"waited {sleeps}; the server sent Retry-After: 7 and it was ignored"
    )


def test_a_404_is_still_not_retried(sleeps):
    """The counterpart: a wrong request must not be repeated."""
    session = _Session([404])
    with pytest.raises(pbp.PlayByPlayError):
        pbp.fetch_playbyplay("0022500001", session=session)
    assert session.calls == 1
    assert sleeps == []


# --- the two header sets must not drift -------------------------------------


def test_the_shared_browser_headers_are_identical_in_both_clients():
    """
    The stats.nba.com set carries two ``x-nba-stats-*`` headers the CDN does
    not need, and that difference is correct. The BROWSER part is duplicated
    verbatim in two files, so a User-Agent bumped in one and not the other
    leaves one host 403ing while the other works -- the failure both header
    sets exist to prevent.
    """
    import src.ingestion.boxscores as stats

    shared = ("User-Agent", "Accept", "Accept-Language", "Referer", "Origin")
    for key in shared:
        assert stats.HEADERS.get(key) == fetcher.HEADERS.get(key), (
            f"{key} differs between stats.nba.com and cdn.nba.com headers: "
            f"{stats.HEADERS.get(key)!r} vs {fetcher.HEADERS.get(key)!r}"
        )
    # And the stats-only extras stay where they belong.
    assert stats.HEADERS["x-nba-stats-origin"] == "stats"
    assert "x-nba-stats-origin" not in fetcher.HEADERS


def test_both_cdn_clients_send_the_browser_headers():
    """AST-free: the call sites must actually pass HEADERS, not just define it."""
    for module, func in ((pbp, "fetch_playbyplay"), (fetcher, "fetch_boxscore")):
        source = inspect.getsource(getattr(module, func))
        assert "headers=HEADERS" in source, (
            f"{module.__name__}.{func} does not send the browser headers, so "
            "cdn.nba.com will bot-detect it"
        )


# --- cubic's follow-ups on PR #11 ------------------------------------------


@pytest.mark.parametrize("raw, expected", [
    ("7", 7.0),                 # the plain integer-seconds form
    ("-1", None),               # negative: nonsense, fall back
    ("nan", None),              # ValueError("Invalid value NaN")
    ("inf", None),              # OverflowError, not a hang -- measured
    ("0", None),                # zero is not a wait
    ("Wed, 21 Oct 2026 07:28:00 GMT", None),   # HTTP-date form, not parsed
    ("", None),
    (None, None),
])
def test_retry_after_is_validated_before_it_reaches_sleep(raw, expected):
    """
    cubic, PR #11: ``float(retry_after)`` accepted ``-1``, ``nan`` and ``inf``
    and passed them to `time.sleep`, which refuses all three — ValueError,
    ValueError and OverflowError respectively, measured rather than assumed, so
    the failure is a CRASHED fetch in every case and not a hang. The HTTP-date
    form fell through to the local schedule silently.

    `ingestion/espn_client._retry_after_seconds` already did all of this
    correctly; this is the same rule, in one place, for both CDN clients.
    """
    from src.settlement.boxscore_fetcher import retry_after_seconds

    got = retry_after_seconds({"Retry-After": raw} if raw is not None else {}, 99.0)
    assert got == (expected if expected is not None else 99.0)


def test_retry_after_is_capped():
    """A server asking for an hour must not hold a slate job for an hour."""
    from src.settlement.boxscore_fetcher import MAX_RETRY_AFTER_SECONDS, retry_after_seconds

    assert retry_after_seconds({"Retry-After": "3600"}, 2.0) == MAX_RETRY_AFTER_SECONDS


def test_both_cdn_clients_share_one_retry_policy():
    """
    cubic, PR #11: `fetch_playbyplay` carried a second full copy of the 429 and
    backoff policy already in `fetch_boxscore`, which is how the first copy
    came to be fixed while the other drifted -- the defect this PR is itself
    repairing. One helper, called by both.
    """
    import ast
    import inspect
    import textwrap

    for module, func in ((pbp, "fetch_playbyplay"), (fetcher, "fetch_boxscore")):
        source = inspect.getsource(getattr(module, func))
        called = {
            node.func.id
            for node in ast.walk(ast.parse(textwrap.dedent(source)))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "retry_after_seconds" in called, (
            f"{module.__name__}.{func} parses Retry-After itself instead of "
            "calling the shared helper"
        )
