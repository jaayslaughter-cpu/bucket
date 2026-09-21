"""
PropLine ingest — NBA only, research only.

These tests never touch the network. Every payload below mirrors the
shapes documented by the provider, so the parsing rules (staleness, DFS
comparability, alt-ladder pairing, timestamp provenance) are exercised
without spending quota or depending on a live slate.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.propline import (
    DEFAULT_MARKET_MAP,
    ENV_API_KEY,
    NBA_SPORT_KEY,
    PropLineAuthError,
    PropLineClient,
    PropLineConfig,
    PropLineError,
    PropLineRateLimited,
    PropLineUnavailable,
    _assert_nba_only,
    _auth_headers,
    _strip_player_namespace,
    load_api_key,
    normalize_event_props,
)

# --------------------------------------------------------------------------
# Credentials
# --------------------------------------------------------------------------

def test_key_travels_in_a_header_never_in_the_url():
    """A key in a query string leaks into logs a header never reaches.

    The provider accepts ?apiKey=, but URLs land in web-server access logs,
    proxy logs, shell history and onward Referer headers.
    """
    headers = _auth_headers("secret-value")
    assert headers["X-API-Key"] == "secret-value"
    assert not any("secret-value" in k for k in headers)


def test_missing_key_raises_rather_than_returning_an_empty_slate(monkeypatch):
    """An empty result would be indistinguishable from 'no games today'."""
    monkeypatch.delenv(ENV_API_KEY, raising=False)
    with pytest.raises(PropLineAuthError, match="DATA_NOT_AVAILABLE"):
        load_api_key()

    monkeypatch.setenv(ENV_API_KEY, "   ")
    with pytest.raises(PropLineAuthError):
        load_api_key()

    monkeypatch.setenv(ENV_API_KEY, "real-key")
    assert load_api_key() == "real-key"


def test_rejected_key_error_does_not_echo_the_key(monkeypatch):
    """An auth failure must not put the credential in the log."""
    monkeypatch.setenv(ENV_API_KEY, "super-secret-key")

    class _Resp:
        status_code = 401
        headers: dict = {}

    class _Session:
        def get(self, *a, **kw):
            return _Resp()

    client = PropLineClient(session=_Session())
    with pytest.raises(PropLineAuthError) as excinfo:
        client.fetch_events()
    assert "super-secret-key" not in str(excinfo.value)


# --------------------------------------------------------------------------
# Scope: NBA only, never NCAA
# --------------------------------------------------------------------------

def test_only_nba_is_allowed_and_ncaa_is_refused_by_name():
    assert _assert_nba_only(NBA_SPORT_KEY) == NBA_SPORT_KEY

    for forbidden in ("basketball_ncaab", "americanfootball_ncaaf", "NCAAB"):
        with pytest.raises(PropLineError, match="NCAA"):
            _assert_nba_only(forbidden)

    for other in ("baseball_mlb", "icehockey_nhl", "tennis"):
        with pytest.raises(PropLineError, match="NBA-only"):
            _assert_nba_only(other)


# --------------------------------------------------------------------------
# Player identity
# --------------------------------------------------------------------------

def test_player_id_namespace_is_stripped_only_for_nba():
    """The bare numeric id is what the box-score panel carries."""
    assert _strip_player_namespace("nba:201939") == "201939"
    assert _strip_player_namespace("201939") == "201939"
    # A non-NBA namespace must NOT yield a bare number that would join onto
    # an unrelated NBA player with the same integer.
    assert _strip_player_namespace("mlb:592450") is None
    assert _strip_player_namespace("espn:8439") is None
    assert _strip_player_namespace(None) is None
    assert _strip_player_namespace("") is None


# --------------------------------------------------------------------------
# Parsing rules
# --------------------------------------------------------------------------

def _payload(markets):
    return {
        "id": "nba-1",
        "sport_key": NBA_SPORT_KEY,
        "commence_time": "2026-01-15T03:10:00Z",
        "bookmakers": [{"key": "draftkings", "title": "DraftKings", "markets": markets}],
    }


def _ou(player, point, over, under, **extra):
    base = {"description": player, "point": point}
    return [
        {**base, "name": "Over", "price": over, **extra},
        {**base, "name": "Under", "price": under, **extra},
    ]


def test_over_and_under_pair_into_one_row():
    rows = normalize_event_props(_payload([{
        "key": "player_points",
        "last_update": "2026-01-15T02:00:00Z",
        "outcomes": _ou("Nikola Jokic", 27.5, -115, -105,
                        player_id="nba:203999",
                        last_change_at="2026-01-15T01:59:00Z"),
    }]))

    assert len(rows) == 1
    row = rows[0]
    assert row.market == "PTS"
    assert row.player_name == "Nikola Jokic"
    assert row.line == 27.5
    assert row.over_odds_american == -115
    assert row.under_odds_american == -105
    assert row.nba_player_id == "203999"
    assert row.nba_game_id == "nba-1"
    assert row.game_date.isoformat() == "2026-01-15"


def test_alternate_ladders_stay_separate_rows():
    """Pairing on player alone would splice two different lines together."""
    rows = normalize_event_props(_payload([{
        "key": "player_points",
        "last_update": "2026-01-15T02:00:00Z",
        "outcomes": (
            _ou("Nikola Jokic", 24.5, -140, 115)
            + _ou("Nikola Jokic", 27.5, -115, -105)
            + _ou("Nikola Jokic", 30.5, 130, -160)
        ),
    }]))

    assert len(rows) == 3
    by_line = {r.line: r for r in rows}
    assert set(by_line) == {24.5, 27.5, 30.5}
    assert by_line[24.5].over_odds_american == -140
    assert by_line[30.5].under_odds_american == -160


def test_withdrawn_selection_is_dropped():
    """last_seen_at older than the market's last_update = being withdrawn.

    The provider's own rule. Without it a pulled selection keeps reading as
    a live, bettable price for the couple of minutes before it vanishes.
    """
    rows = normalize_event_props(_payload([{
        "key": "player_points",
        "last_update": "2026-01-15T02:00:00Z",
        "outcomes": (
            _ou("Live Player", 20.5, -110, -110,
                last_seen_at="2026-01-15T02:00:00Z")
            + _ou("Withdrawn Player", 18.5, -110, -110,
                  last_seen_at="2026-01-15T01:55:00Z")
        ),
    }]))

    assert [r.player_name for r in rows] == ["Live Player"]


def test_suspended_market_is_dropped():
    """A market the book pulled pregame is a last quote, not a live price."""
    rows = normalize_event_props(_payload([{
        "key": "player_points",
        "last_update": "2026-01-15T02:00:00Z",
        "suspended_at": "2026-01-15T02:05:00Z",
        "outcomes": _ou("Scratched Player", 20.5, -110, -110),
    }]))
    assert rows == []


def test_unmapped_market_is_skipped_never_guessed(caplog):
    """Guessing would attach a real posted line to the wrong statistic."""
    import logging

    with caplog.at_level(logging.WARNING):
        rows = normalize_event_props(_payload([{
            "key": "player_turnovers_invented",
            "last_update": "2026-01-15T02:00:00Z",
            "outcomes": _ou("Some Player", 2.5, -110, -110),
        }]))

    assert rows == []
    assert "no mapping" in caplog.text
    assert "player_turnovers_invented" in caplog.text


# --------------------------------------------------------------------------
# Timestamp provenance
# --------------------------------------------------------------------------

def test_observation_time_prefers_the_books_own_publish_time():
    rows = normalize_event_props(_payload([{
        "key": "player_points",
        "last_update": "2026-01-15T02:00:00Z",
        "outcomes": _ou("Nikola Jokic", 27.5, -115, -105,
                        book_updated_at="2026-01-15T01:58:00Z",
                        last_change_at="2026-01-15T01:30:00Z"),
    }]))
    assert rows[0].captured_at_utc == datetime(2026, 1, 15, 1, 58, tzinfo=timezone.utc)
    assert rows[0].raw["captured_at_source"] == "book_updated_at"


def test_observation_time_falls_back_to_last_change_at():
    """Populated for every book, unlike book_updated_at."""
    rows = normalize_event_props(_payload([{
        "key": "player_points",
        "last_update": "2026-01-15T02:00:00Z",
        "outcomes": _ou("Nikola Jokic", 27.5, -115, -105,
                        book_updated_at=None,
                        last_change_at="2026-01-15T01:30:00Z"),
    }]))
    assert rows[0].captured_at_utc == datetime(2026, 1, 15, 1, 30, tzinfo=timezone.utc)
    assert rows[0].raw["captured_at_source"] == "last_change_at"


def test_observation_time_is_null_when_the_source_reports_none():
    """Never now(). A fabricated capture time reads as a line that never moved."""
    rows = normalize_event_props(_payload([{
        "key": "player_points",
        "last_update": "2026-01-15T02:00:00Z",
        "outcomes": _ou("Nikola Jokic", 27.5, -115, -105),
    }]))
    assert rows[0].captured_at_utc is None


# --------------------------------------------------------------------------
# DFS comparability
# --------------------------------------------------------------------------

def _dfs_payload(book, outcomes):
    return {
        "id": "nba-1",
        "commence_time": "2026-01-15T03:10:00Z",
        "bookmakers": [{"key": book, "markets": [{
            "key": "player_points",
            "last_update": "2026-01-15T02:00:00Z",
            "outcomes": outcomes,
        }]}],
    }


def test_prizepicks_goblin_and_demon_lines_are_excluded():
    """They are different products, not a mispriced market line."""
    outcomes = (
        _ou("Player A", 25.5, 100, 100, dfs_odds_type="standard")
        + _ou("Player A", 21.5, 100, 100, dfs_odds_type="goblin")
        + _ou("Player A", 29.5, 100, 100, dfs_odds_type="demon")
    )
    rows = normalize_event_props(_dfs_payload("prizepicks", outcomes))

    assert len(rows) == 1
    assert rows[0].line == 25.5
    # Synthetic +100/+100 whose payout comes from parlay correct-count.
    assert rows[0].is_pickem is True


def test_underdog_boosted_picks_are_excluded_but_standard_ones_are_kept():
    """A 1.5x boost is not comparable to sportsbook consensus; 1.0 is."""
    outcomes = (
        _ou("Player A", 25.5, -113, 101, payout_multiplier=1.0)
        + _ou("Player B", 6.5, -120, 105, payout_multiplier=1.5)
    )
    rows = normalize_event_props(_dfs_payload("underdog", outcomes))

    assert [r.player_name for r in rows] == ["Player A"]
    # A standard Underdog price IS a real two-way price. Leaving the column
    # None is what lets the EV gate de-vig it instead of abstaining.
    assert rows[0].payout_multiplier is None
    assert rows[0].is_pickem is False


# --------------------------------------------------------------------------
# The point of all of this: real odds must clear the EV gate
# --------------------------------------------------------------------------

def test_a_real_two_way_price_clears_the_ev_gate():
    """main.py used to hardcode is_pickem=True on every prop row.

    That made market_ev_gate abstain unconditionally — the gate refused the
    very data it exists to evaluate. This is the end-to-end proof that a
    genuine sportsbook price now gets through.
    """
    from src.quant.contracts import MarketContext, market_ev_gate

    rows = normalize_event_props(_payload([{
        "key": "player_points",
        "last_update": "2026-01-15T02:00:00Z",
        "outcomes": _ou("Nikola Jokic", 27.5, -110, -110,
                        last_change_at="2026-01-15T01:59:00Z"),
    }]))
    row = rows[0]

    verdict = market_ev_gate(MarketContext(
        game_id=row.nba_game_id,
        status=row.status,
        market=row.market,
        player_name=row.player_name,
        line=row.line,
        over_odds_american=row.over_odds_american,
        under_odds_american=row.under_odds_american,
        payout_multiplier=row.payout_multiplier,
        is_pickem=row.is_pickem,
        source=row.source,
        captured_at_utc=row.captured_at_utc,
    ))

    assert verdict["status"] == "READY_FOR_EVALUATION", verdict["reason"]
    assert verdict["fair_probability_over"] == pytest.approx(0.5, abs=1e-9)
    assert verdict["hold"] > 0


def test_a_prizepicks_row_still_abstains_at_the_gate():
    """A payout multiplier is not a price, however numeric it looks."""
    from src.quant.contracts import MarketContext, market_ev_gate

    rows = normalize_event_props(_dfs_payload(
        "prizepicks", _ou("Player A", 25.5, 100, 100, dfs_odds_type="standard")
    ))
    verdict = market_ev_gate(MarketContext(
        game_id="nba-1", status="VALID", market=rows[0].market,
        line=rows[0].line,
        over_odds_american=rows[0].over_odds_american,
        under_odds_american=rows[0].under_odds_american,
        is_pickem=rows[0].is_pickem,
    ))
    assert verdict["status"] == "DATA_NOT_AVAILABLE"
    assert "Pick'em" in verdict["reason"]


# --------------------------------------------------------------------------
# HTTP behaviour
# --------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, status_code=200, json_body=None, headers=None):
        self.status_code = status_code
        self._json = json_body if json_body is not None else []
        self.headers = headers or {}

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise AssertionError("should not be reached in these tests")


def test_rate_limit_honours_retry_after(monkeypatch):
    monkeypatch.setenv(ENV_API_KEY, "k")
    slept: list[float] = []
    monkeypatch.setattr("src.ingestion.propline.time.sleep", slept.append)

    calls = {"n": 0}

    class _Session:
        def get(self, *a, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                return _FakeResponse(429, headers={"Retry-After": "7"})
            return _FakeResponse(200, [{"id": "1"}])

    client = PropLineClient(session=_Session(), config=PropLineConfig(max_attempts=3))
    assert client.fetch_events() == [{"id": "1"}]
    assert slept == [7.0], "Retry-After was not honoured"


def test_persistent_rate_limit_raises_rather_than_returning_empty(monkeypatch):
    monkeypatch.setenv(ENV_API_KEY, "k")
    monkeypatch.setattr("src.ingestion.propline.time.sleep", lambda *_: None)

    class _Session:
        def get(self, *a, **kw):
            return _FakeResponse(429, headers={"Retry-After": "1"})

    client = PropLineClient(session=_Session(), config=PropLineConfig(max_attempts=2))
    with pytest.raises(PropLineRateLimited):
        client.fetch_events()


def test_quota_headers_are_tracked(monkeypatch):
    monkeypatch.setenv(ENV_API_KEY, "k")

    class _Session:
        def get(self, *a, **kw):
            return _FakeResponse(200, [], headers={
                "X-Daily-Limit": "1000",
                "X-Daily-Used": "563",
                "X-Daily-Remaining": "437",
                "X-Daily-Reset": "1785542400",
            })

    client = PropLineClient(session=_Session())
    client.fetch_events()
    assert client.quota.daily_limit == 1000
    assert client.quota.daily_remaining == 437
    assert client.quota.daily_reset_utc is not None


def test_refuses_to_burn_the_last_of_the_daily_quota(monkeypatch):
    """A run that exhausts quota mid-slate leaves a partial picture."""
    monkeypatch.setenv(ENV_API_KEY, "k")

    class _Session:
        def get(self, *a, **kw):
            return _FakeResponse(200, {}, headers={"X-Daily-Remaining": "2"})

    client = PropLineClient(session=_Session(), config=PropLineConfig(min_daily_remaining=5))
    client.fetch_events()  # populates quota
    with pytest.raises(PropLineRateLimited, match="Stopping"):
        client.fetch_event_props("nba-1", ["player_points"])


def test_merged_event_id_surfaces_as_unavailable(monkeypatch):
    """A 404 means re-discover the event, not retry forever."""
    monkeypatch.setenv(ENV_API_KEY, "k")

    class _Session:
        def get(self, *a, **kw):
            return _FakeResponse(404)

    client = PropLineClient(session=_Session())
    with pytest.raises(PropLineUnavailable):
        client.fetch_event_props("stale-id", ["player_points"])


def test_unbounded_market_pull_is_refused(monkeypatch):
    monkeypatch.setenv(ENV_API_KEY, "k")
    client = PropLineClient(session=object())
    with pytest.raises(PropLineError, match="unbounded"):
        client.fetch_event_props("nba-1", [])


def test_market_map_covers_the_projects_stats():
    """Every mapped target must be a stat the feature builder produces."""
    from src.features.builder import ROLLING_STATS

    for stat in DEFAULT_MARKET_MAP.values():
        assert stat in ROLLING_STATS, f"{stat} is not a modelled stat"
