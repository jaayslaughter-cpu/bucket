"""
Tests for the PropLine historical endpoints and their normalisers.

The client was live-only. These endpoints are what make calibration and CLV
possible, and the properties worth pinning are: a tier-redacted body must
never read as "this game had no odds", a payload whose shape is undocumented
must not be guessed at, and the two CLV numbers must stay distinguishable.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd
import pytest

from src.ingestion.propline import (
    ARCHIVE_START,
    ENV_API_KEY,
    HISTORY_INTERVALS,
    TIER_EVENT_AGE_DAYS,
    PropLineClient,
    PropLineError,
    PropLineTierLimited,
)
from src.ingestion.propline_history import (
    PropLineHistoryError,
    bets_to_grade_payload,
    clv_from_closing,
    describe_resolution_payload,
    normalize_closing_odds,
    normalize_clv_grade,
    plan_history_window,
)

TODAY = date(2026, 9, 22)

CLOSING_PAYLOAD = {
    "id": "5885", "sport_key": "basketball_nba",
    "commence_time": "2026-04-19T20:10:00Z",
    "bookmakers": [{
        "key": "draftkings", "title": "DraftKings",
        "markets": [{"key": "player_points", "outcomes": [
            {"name": "Over", "description": "LeBron James", "player_id": "nba:2544",
             "price": 116, "point": 24.5, "closing_at": "2026-04-19T20:08:14Z",
             "closing_age_seconds": 106, "is_stale": False,
             "opening_price": 104, "opening_point": 24.5,
             "opening_at": "2026-04-17T13:02:51Z", "opening_age_seconds": 198429},
            {"name": "Under", "description": "LeBron James", "player_id": "nba:2544",
             "price": -148, "point": 24.5, "closing_at": "2026-04-19T20:08:14Z",
             "opening_price": -128, "opening_point": 24.5,
             "opening_at": "2026-04-17T13:02:51Z", "opening_age_seconds": 198429},
        ]}],
    }],
}


def _client(monkeypatch, payload, status=200):
    monkeypatch.setenv(ENV_API_KEY, "test-key")

    class _Resp:
        status_code = status
        headers: dict = {}
        text = ""

        def json(self):
            return payload

        def raise_for_status(self):
            return None

    class _Session:
        def __init__(self):
            self.calls = []

        def get(self, url, **kw):
            self.calls.append({"method": "GET", "url": url, **kw})
            return _Resp()

        def request(self, method, url, **kw):
            self.calls.append({"method": method, "url": url, **kw})
            return _Resp()

    session = _Session()
    return PropLineClient(session=session), session


# --- what a plan can actually reach --------------------------------------


def test_the_archive_start_and_the_tier_cap_both_bind():
    """
    Two limits stack. On the lower plans the cap bites first, and a 30-day
    window measured from an off-season day contains no basketball whatever
    the archive holds.
    """
    assert ARCHIVE_START == date(2026, 4, 1)
    assert TIER_EVENT_AGE_DAYS["hobby"] == 30
    assert TIER_EVENT_AGE_DAYS["enterprise"] is None

    hobby = plan_history_window("hobby", as_of=TODAY)
    assert hobby.earliest == date(2026, 8, 23)
    assert "event-age cap" in hobby.reason

    streaming_lite = plan_history_window("streaming_lite", as_of=TODAY)
    assert streaming_lite.earliest == ARCHIVE_START     # the archive now binds
    assert "archive start" in streaming_lite.reason

    unlimited = plan_history_window("enterprise", as_of=TODAY)
    assert unlimited.earliest == ARCHIVE_START


def test_a_window_landing_in_the_offseason_is_reported_unusable():
    """The 2025-26 NBA season ended 2026-06-13."""
    season_end = date(2026, 6, 13)
    for tier in ("hobby", "pro"):
        window = plan_history_window(tier, as_of=TODAY, season_start=season_end)
        assert window.usable is False
        assert "off-season" in window.reason

    assert plan_history_window(
        "streaming_lite", as_of=TODAY, season_start=season_end
    ).usable is True


def test_an_unknown_tier_is_refused_by_name():
    with pytest.raises(PropLineHistoryError, match="Unknown tier"):
        plan_history_window("platinum", as_of=TODAY)


# --- the redacted body ---------------------------------------------------


def test_a_tier_redacted_body_is_not_an_event_without_odds(monkeypatch):
    """
    Over-cap requests answer 200 with redacted: true. Read naively that is a
    game nobody priced, which is a different fact entirely.
    """
    redacted = {"id": "1", "redacted": True, "upgrade_url": "https://example/upgrade"}
    client, _ = _client(monkeypatch, redacted)

    with pytest.raises(PropLineTierLimited) as closing:
        client.fetch_closing_odds("1", ["player_points"])
    assert "event-age cap" in str(closing.value)
    assert "https://example/upgrade" in str(closing.value)

    with pytest.raises(PropLineTierLimited):
        client.fetch_odds_history("1", ["player_points"])
    with pytest.raises(PropLineTierLimited):
        client.fetch_event_results("1")


# --- request construction ------------------------------------------------


def test_history_parameters_are_validated_before_spending_a_request(monkeypatch):
    client, session = _client(monkeypatch, CLOSING_PAYLOAD)

    with pytest.raises(PropLineError, match="mutually exclusive"):
        client.fetch_odds_history(
            "1", ["player_points"],
            from_utc=datetime(2026, 4, 19, tzinfo=timezone.utc), relative_from="-3h",
        )
    with pytest.raises(PropLineError, match="interval"):
        client.fetch_odds_history("1", ["player_points"], interval="7m")
    with pytest.raises(PropLineError, match="unbounded"):
        client.fetch_odds_history("1", [])
    assert session.calls == []           # nothing was sent

    client.fetch_odds_history(
        "1", ["player_points"], relative_from="-3h", relative_to="0", interval="1m",
    )
    params = session.calls[-1]["params"]
    assert params["relative_from"] == "-3h"
    assert params["interval"] == "1m"
    assert params["changes_only"] == "true"      # repeats carry no information
    assert "1m" in HISTORY_INTERVALS


def test_clv_grade_posts_and_validates_its_bets(monkeypatch):
    client, session = _client(monkeypatch, {"summary": {}, "bets": []})

    with pytest.raises(PropLineError, match="No bets"):
        client.grade_clv([])
    with pytest.raises(PropLineError, match="missing required fields"):
        client.grade_clv([{"ref": "b1", "event_id": 1}])

    client.grade_clv([{
        "ref": "b1", "event_id": 1, "market": "player_points",
        "selection": "LeBron James", "side": "Over", "price": -110,
    }])
    call = session.calls[-1]
    assert call["method"] == "POST"
    assert call["url"].endswith("/clv/grade")
    assert call["json"][0]["ref"] == "b1"


# --- /odds/closing, documented shape -------------------------------------


def test_closing_odds_flatten_with_the_player_id_namespace_stripped():
    frame = normalize_closing_odds(CLOSING_PAYLOAD)
    assert len(frame) == 2
    assert set(frame["side"]) == {"over", "under"}
    # nba:2544 joins the panel's personId directly.
    assert set(frame["player_id"]) == {"2544"}
    assert frame["player_name"].iloc[0] == "LeBron James"
    assert frame["opening_price"].iloc[0] == 104
    assert frame["closing_price"].iloc[0] == 116

    assert normalize_closing_odds({}).empty


def test_clv_is_devigged_when_both_sides_exist_and_is_zero_sum():
    """
    De-vigged CLV nets to zero across the two sides of one market. A non-zero
    sum would mean the hold was left in.
    """
    clv = clv_from_closing(normalize_closing_odds(CLOSING_PAYLOAD))
    graded = clv[clv["status"] == "OK"]
    assert len(graded) == 2
    assert set(graded["clv_method"]) == {"devigged"}
    assert graded["clv"].sum() == pytest.approx(0.0, abs=1e-9)
    assert set(graded["beat_close"]) == {True, False}


def test_a_one_sided_market_falls_back_to_raw_implied_and_says_so():
    one_sided = {
        "id": "2",
        "bookmakers": [{"key": "dk", "markets": [{"key": "player_points", "outcomes": [
            {"name": "Over", "description": "B", "price": 116, "point": 24.5,
             "opening_price": 104, "opening_age_seconds": 99999},
        ]}]}],
    }
    clv = clv_from_closing(normalize_closing_odds(one_sided))
    assert clv["clv_method"].iloc[0] == "raw_implied"
    # Raw-implied carries the book's hold, so it differs from the devigged value.
    devigged = clv_from_closing(normalize_closing_odds(CLOSING_PAYLOAD))
    over = devigged[devigged["side"] == "over"]["clv"].iloc[0]
    assert clv["clv"].iloc[0] != pytest.approx(over)


def test_an_open_first_seen_minutes_before_tip_is_not_an_open():
    late = {
        "id": "3",
        "bookmakers": [{"key": "dk", "markets": [{"key": "player_points", "outcomes": [
            {"name": "Over", "description": "Late", "price": -110, "point": 10.5,
             "opening_price": -108, "opening_age_seconds": 600},
        ]}]}],
    }
    clv = clv_from_closing(normalize_closing_odds(late))
    assert clv["status"].iloc[0] == "DATA_NOT_AVAILABLE"
    assert "10 minutes before tip" in clv["reason"].iloc[0]
    assert pd.isna(clv["clv"].iloc[0])


# --- POST /clv/grade, documented shape -----------------------------------


def test_the_devigged_clv_is_ordered_ahead_of_the_quotable_one():
    """
    clv_pct is price-vs-price and vig-blind, so it flatters a bet taken on
    the juicy side of a wide market. Both are kept; the honest one leads.
    """
    graded = normalize_clv_grade({
        "summary": {"bets": 1, "matched": 1, "avg_clv_pct": 6.52,
                    "avg_ev_vs_close_pct": 0.08, "profit_units": -1.0},
        "bets": [{"ref": "b1", "matched": True, "clv_pct": 6.52,
                  "ev_vs_close_pct": 0.08, "beat_close": True,
                  "resolution": "lost", "actual_value": 1.0,
                  "closing_fair_prob": 0.4085, "fair_source": "pinnacle"}],
    })
    columns = list(graded.bets.columns)
    assert columns.index("ev_vs_close_pct") < columns.index("clv_pct")
    assert graded.honest_clv_column == "ev_vs_close_pct"
    # A bet can beat the close and still lose; both facts survive.
    assert bool(graded.bets["beat_close"].iloc[0]) is True
    assert graded.bets["resolution"].iloc[0] == "lost"
    assert graded.summary["profit_units"] == -1.0


def test_logged_bets_convert_into_a_grade_request():
    payload = bets_to_grade_payload([{
        "bet_id": "abc123", "game_id": "0022500001", "prop_stat": "PTS",
        "player_name": "LeBron James", "bet_side": "under", "line": 24.5,
        "taken_odds_american": -115, "bookmaker": "draftkings", "unit_stake": 2.0,
    }])
    assert payload[0]["ref"] == "abc123"
    assert payload[0]["side"] == "Under"
    assert payload[0]["price"] == -115
    assert payload[0]["sport_key"] == "basketball_nba"

    with pytest.raises(PropLineHistoryError, match="missing"):
        bets_to_grade_payload([{"bet_id": "x", "game_id": "1"}])


# --- undocumented shapes are discovered, never guessed -------------------


def test_a_single_resolution_column_is_usable():
    report = describe_resolution_payload({"results": [
        {"player": "A", "market": "PTS", "line": 24.5, "resolution": "won"},
    ]})
    assert report.rows == 1
    assert report.resolution_candidates == ["resolution"]
    assert report.usable


def test_several_resolution_candidates_are_reported_not_chosen(caplog):
    """
    Two columns could both plausibly hold the grade and mean different
    things. Picking one silently is how a backtest grades the wrong field.
    """
    report = describe_resolution_payload([
        {"player": "A", "resolution": "won", "actual_value": 27.0, "status": "final"},
    ])
    assert len(report.resolution_candidates) > 1
    assert report.usable is False
    assert "pick one deliberately" in caplog.text


def test_no_resolution_column_is_reported_rather_than_invented(caplog):
    report = describe_resolution_payload([{"player": "A", "line": 24.5}])
    assert report.resolution_candidates == []
    assert report.usable is False
    assert "No resolution-looking column" in caplog.text


def test_an_uninspectable_payload_is_refused():
    with pytest.raises(PropLineHistoryError, match="Cannot inspect"):
        describe_resolution_payload(42)
