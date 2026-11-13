"""Red-first tests for the research-only, point-in-time input archive.

Synthetic examples test contracts, not claims about historical player status.
"""

from datetime import datetime, timedelta

import pytest

from src.ingestion.model_input_archive import (
    InputArchive,
    InputError,
    build_absence_export,
    build_odds_export,
    discover_report_urls,
    normalize_odds,
    parse_report_pages,
    report_time,
)
from src.utils.timezones import UTC

REPORT = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_2026-11-08_03_00PM.pdf"
AT = datetime(2026, 11, 8, 20, 0, tzinfo=UTC)


def entry(**updates):
    row = {
        "game_date": "2026-11-08", "matchup": "LAL@DET",
        "team": "Los Angeles Lakers", "player_name": "Test Player",
        "status": "OUT", "reason": "Injury/Illness - Knee",
    }
    return row | updates


def event(**updates):
    return {
        "id": "provider-event", "sport_key": "basketball_nba",
        "commence_time": "2026-11-09T00:00:00Z",
        "bookmakers": [{
            "key": "draftkings", "markets": [{
                "key": "player_points", "last_update": "2026-11-08T19:55:00Z",
                "outcomes": [
                    {"name": "Over", "description": "Test Player", "point": 20.5, "price": -110},
                    {"name": "Under", "description": "Test Player", "point": 20.5, "price": -115},
                ],
            }],
        }],
    } | updates


def save(archive, rows, observed=AT, collected=AT, status="OK", kind="availability"):
    return archive.write(
        kind=kind, source="nba_official" if kind == "availability" else "propline",
        source_url=REPORT if kind == "availability" else "https://api.prop-line.com/v1",
        observed_at=observed, collected_at=collected, status=status, rows=rows,
    )


def test_report_time_is_eastern_converted_to_utc():
    assert report_time(REPORT) == AT
    assert report_time(REPORT.replace("11-08", "07-08")).hour == 19


@pytest.mark.parametrize("url", [
    REPORT.replace("https:", "http:"), REPORT.replace("ak-static.cms.nba.com", "example.org"),
    REPORT + "?api_key=not-a-secret", REPORT.replace("_03_00PM", "_25_00PM"),
])
def test_only_exact_official_report_urls_are_accepted(url):
    with pytest.raises(InputError):
        report_time(url)


def test_discovery_uses_published_links_not_fabricated_dates():
    html = f'<a href="{REPORT}">report</a><a href="{REPORT}">duplicate</a>'
    html += '<a href="https://example.org/report.pdf">untrusted</a>'
    assert discover_report_urls(html, before=AT) == [REPORT]
    assert discover_report_urls(html, before=AT - timedelta(seconds=1)) == []


def test_append_only_archive_roundtrips_and_dedupes_exact_replays(tmp_path):
    store = InputArchive(tmp_path)
    first = save(store, [entry()])
    assert save(store, [entry()]) == first
    assert len(list(tmp_path.glob("snapshots/*/*.json"))) == 1
    save(store, [entry(status="AVAILABLE")], observed=AT + timedelta(hours=1),
         collected=AT + timedelta(hours=1))
    assert len(list(tmp_path.glob("snapshots/*/*.json"))) == 2
    assert store.as_of("availability", AT)[0]["rows"][0]["status"] == "OUT"


def test_later_report_does_not_resurrect_an_older_out(tmp_path):
    store = InputArchive(tmp_path)
    save(store, [entry()])
    save(store, [entry(player_name=None, status="NOT_YET_SUBMITTED")],
         observed=AT + timedelta(minutes=15), collected=AT + timedelta(minutes=15))
    result = store.as_of("availability", AT + timedelta(minutes=20))
    assert len(result) == 1
    assert result[0]["rows"][0]["status"] == "NOT_YET_SUBMITTED"


def test_replay_and_actual_system_knowledge_are_different(tmp_path):
    store = InputArchive(tmp_path)
    save(store, [entry()], collected=AT + timedelta(days=3))
    assert store.as_of("availability", AT, mode="system") == []
    assert len(store.as_of("availability", AT, mode="published")) == 1


def test_stale_snapshot_is_unknown_not_last_known_healthy(tmp_path):
    store = InputArchive(tmp_path)
    save(store, [entry(status="AVAILABLE")])
    assert store.as_of("availability", AT + timedelta(hours=7), max_age_hours=6) == []


def test_a_failed_collection_blocks_system_mode_fallback(tmp_path):
    store = InputArchive(tmp_path)
    save(store, [entry()])
    save(store, [], observed=AT + timedelta(minutes=15),
         collected=AT + timedelta(minutes=15), status="FAILED")
    selected = store.as_of("availability", AT + timedelta(minutes=20))
    assert selected[0]["status"] == "FAILED"
    assert selected[0]["rows"] == []


def test_naive_timestamps_are_refused_not_assumed(tmp_path):
    with pytest.raises(InputError, match="timezone"):
        save(InputArchive(tmp_path), [entry()], observed=AT.replace(tzinfo=None))


def test_future_publication_is_refused(tmp_path):
    with pytest.raises(InputError):
        save(InputArchive(tmp_path), [entry()], observed=AT + timedelta(minutes=1))


def test_credentials_anywhere_in_payload_are_refused(tmp_path):
    rows = [entry()] * 25 + [entry(reason={"authorization": "test-value"})]
    with pytest.raises(InputError, match="credential"):
        save(InputArchive(tmp_path), rows)
    assert not list(tmp_path.rglob("*.json"))


def test_secret_in_source_url_is_refused(tmp_path):
    with pytest.raises(InputError, match="URL"):
        InputArchive(tmp_path).write(
            kind="odds", source="propline", source_url="https://api.prop-line.com?apiKey=test",
            observed_at=AT, collected_at=AT, rows=[], status="FAILED",
        )


def test_odds_both_sides_same_book_market_player_line_and_time():
    rows, rejected = normalize_odds(event(), collected_at=AT)
    assert not rejected
    assert len(rows) == 1
    assert rows[0]["over_price"] == -110
    assert rows[0]["under_price"] == -115
    assert rows[0]["provider_event_id"] == "provider-event"
    assert rows[0]["market"] == "player_points"
    assert rows[0]["observed_at_utc"] == "2026-11-08T19:55:00+00:00"


def test_different_lines_are_not_paired():
    payload = event()
    payload["bookmakers"][0]["markets"][0]["outcomes"][1]["point"] = 21.5
    rows, rejected = normalize_odds(payload, collected_at=AT)
    assert rows == []
    assert rejected


@pytest.mark.parametrize("price", [0, 99, float("nan"), float("inf"), -110.5])
def test_bad_prices_never_enter_paired_training_rows(price):
    payload = event()
    payload["bookmakers"][0]["markets"][0]["outcomes"][0]["price"] = price
    rows, rejected = normalize_odds(payload, collected_at=AT)
    assert not rows and rejected


def test_duplicate_side_does_not_win_by_insertion_order():
    payload = event()
    outcomes = payload["bookmakers"][0]["markets"][0]["outcomes"]
    outcomes.append(outcomes[0] | {"price": -120})
    assert normalize_odds(payload, collected_at=AT)[0] == []
    outcomes.reverse()
    assert normalize_odds(payload, collected_at=AT)[0] == []


def test_withdrawn_and_undated_quotes_abstain():
    payload = event()
    market = payload["bookmakers"][0]["markets"][0]
    market["outcomes"][0]["last_seen_at"] = "2026-11-08T19:00:00Z"
    assert normalize_odds(payload, collected_at=AT)[0] == []
    market["outcomes"][0].pop("last_seen_at")
    market.pop("last_update")
    assert normalize_odds(payload, collected_at=AT)[0] == []


def test_synthetic_pickem_is_not_a_two_way_sportsbook():
    payload = event()
    payload["bookmakers"][0]["key"] = "prizepicks"
    assert normalize_odds(payload, collected_at=AT)[0] == []


def test_game_spread_and_total_are_retained_without_calling_them_opening():
    payload = event()
    payload["bookmakers"][0]["markets"] = [{
        "key": "spreads", "last_update": "2026-11-08T19:55:00Z",
        "outcomes": [
            {"name": "Los Angeles Lakers", "point": -4.5, "price": -110},
            {"name": "Detroit Pistons", "point": 4.5, "price": -110},
        ],
    }]
    payload["home_team"] = "Detroit Pistons"
    payload["away_team"] = "Los Angeles Lakers"
    rows, _ = normalize_odds(payload, collected_at=AT)
    assert len(rows) == 1
    assert rows[0]["home_line"] == 4.5
    assert rows[0]["away_line"] == -4.5
    assert "opening_spread" not in rows[0]


def test_nba_only():
    with pytest.raises(InputError, match="NBA"):
        normalize_odds(event(sport_key="americanfootball_nfl"), collected_at=AT)


def test_quote_does_not_leak_past_its_own_observation_time(tmp_path):
    store = InputArchive(tmp_path)
    rows, _ = normalize_odds(event(), collected_at=AT)
    store.write(kind="odds", source="propline", source_url="https://api.prop-line.com/v1",
                observed_at=AT - timedelta(hours=1), collected_at=AT, rows=rows, status="OK")
    assert store.as_of("odds", AT - timedelta(minutes=10), mode="published") == []


def test_export_requires_explicit_game_and_player_crosswalks():
    snapshot = {"status": "OK", "rows": [entry()], "observed_at_utc": AT.isoformat()}
    games = [{
        "game_date": "2026-11-08", "matchup": "LAL@DET", "team": "Los Angeles Lakers",
        "game_id": "espn:401909952", "team_abbreviation": "LAL",
    }]
    players = [{
        "team": "Los Angeles Lakers", "player_name": "Test Player",
        "player_id": "espn:123",
    }]
    rows, unresolved = build_absence_export([snapshot], games, players)
    assert not unresolved and rows[0]["PLAYER_ID"] == "espn:123"
    assert rows[0]["GAME_ID"] == "espn:401909952"
    rows, unresolved = build_absence_export([snapshot], games, [])
    assert rows == [] and unresolved


def test_unknown_or_ambiguous_player_does_not_become_zero_absences():
    snapshot = {"status": "OK", "rows": [entry()], "observed_at_utc": AT.isoformat()}
    games = [{
        "game_date": "2026-11-08", "matchup": "LAL@DET", "team": "Los Angeles Lakers",
        "game_id": "espn:401909952", "team_abbreviation": "LAL",
    }]
    players = [
        {"team": "Los Angeles Lakers", "player_name": "Test Player", "player_id": "espn:1"},
        {"team": "Los Angeles Lakers", "player_name": "Test Player", "player_id": "espn:2"},
    ]
    rows, unresolved = build_absence_export([snapshot], games, players)
    assert not rows and unresolved


def test_geometry_parser_carries_context_across_pages_without_counting_not_submitted():
    # Synthetic geometry, mirroring the seven observed PDF columns. Parser
    # gets words, not English line splitting that mixes reason continuations.
    def word(text, x, y):
        return {"text": text, "x0": x, "x1": x + 10, "top": y, "bottom": y + 10}
    headers = [word(t, 10 + i * 100, 100) for i, t in enumerate(
        ["GameDate", "GameTime", "Matchup", "Team", "PlayerName", "CurrentStatus", "Reason"]
    )]
    first = headers + [
        word("11/08/2026", 11, 130), word("07:00(ET)", 111, 130),
        word("LAL@DET", 211, 130), word("Los Angeles Lakers", 311, 130),
        word("Player, Test", 411, 130), word("Out", 511, 130),
        word("Injury/Illness - Knee", 611, 123),
        word("Sprain", 611, 137),
    ]
    second = [
        word("Other, Test", 411, 90), word("Questionable", 511, 90),
        word("Injury/Illness - Ankle", 611, 90),
        word("Detroit Pistons", 311, 120), word("NOT YET SUBMITTED", 611, 120),
    ]
    rows = parse_report_pages([first, second])
    assert len(rows) == 3
    assert rows[0]["player_name"] == "Test Player"
    assert rows[0]["reason"] == "Injury/Illness - Knee Sprain"
    assert rows[1]["game_date"] == "2026-11-08"
    assert rows[1]["team"] == "Los Angeles Lakers"
    assert rows[2]["status"] == "NOT_YET_SUBMITTED"
    assert rows[2]["player_name"] is None


def test_empty_pdf_shape_is_refused():
    with pytest.raises(InputError):
        parse_report_pages([[]])


def test_split_header_words_and_page_footer_are_not_player_data():
    def word(text, x, y):
        return {"text": text, "x0": x, "x1": x + 5, "top": y, "bottom": y + 10}
    page = [
        word("Game", 10, 100), word("Date", 40, 100),
        word("Game", 110, 100), word("Time", 140, 100),
        word("Matchup", 210, 100), word("Team", 310, 100),
        word("Player", 410, 100), word("Name", 445, 100),
        word("Current", 510, 100), word("Status", 550, 100), word("Reason", 610, 100),
        word("11/08/2026", 11, 130), word("07:00(ET)", 111, 130),
        word("LAL@DET", 211, 130), word("Los Angeles Lakers", 311, 130),
        word("Player, Test", 411, 130), word("Out", 511, 130),
        word("Knee", 611, 130), word("Page", 399, 538), word("1 of 1", 422, 538),
    ]
    assert parse_report_pages([page])[0]["team"] == "Los Angeles Lakers"


def test_suspended_market_never_enters_export():
    payload = event()
    payload["bookmakers"][0]["markets"][0]["suspended_at"] = AT.isoformat()
    rows, rejected = normalize_odds(payload, collected_at=AT)
    assert not rows and rejected


def test_historical_quote_import_keeps_observation_and_collection_separate():
    rows, _ = normalize_odds(event(), collected_at=AT + timedelta(days=2), max_quote_age_minutes=None)
    assert len(rows) == 1
    assert rows[0]["observed_at_utc"] != rows[0]["collected_at_utc"]


def test_live_capture_rejects_an_old_but_present_market():
    rows, rejected = normalize_odds(event(), collected_at=AT + timedelta(days=2))
    assert not rows and rejected


def test_unmatched_outcome_player_ids_cannot_pair():
    payload = event()
    outcomes = payload["bookmakers"][0]["markets"][0]["outcomes"]
    outcomes[0]["player_id"] = "nba:1"
    outcomes[1]["player_id"] = "nba:2"
    assert normalize_odds(payload, collected_at=AT)[0] == []


def test_different_change_times_are_not_different_market_snapshots():
    payload = event()
    outcomes = payload["bookmakers"][0]["markets"][0]["outcomes"]
    outcomes[0]["last_change_at"] = "2026-11-08T19:00:00Z"
    outcomes[1]["last_change_at"] = "2026-11-08T19:30:00Z"
    assert len(normalize_odds(payload, collected_at=AT)[0]) == 1


def test_odds_export_uses_explicit_event_and_panel_identity_crosswalk():
    rows, _ = normalize_odds(event(), collected_at=AT)
    snapshots = [{"status": "OK", "rows": rows}]
    games = [{
        "provider_event_id": "provider-event", "game_id": "espn:1",
        "team": "Los Angeles Lakers",
    }]
    players = [{"team": "Los Angeles Lakers", "player_name": "Test Player", "player_id": "espn:2"}]
    resolved, unresolved = build_odds_export(snapshots, games, players)
    assert not unresolved
    assert resolved[0]["GAME_ID"] == "espn:1"
    assert resolved[0]["PLAYER_ID"] == "espn:2"
    assert resolved[0]["provider_event_id"] == "provider-event"
    assert build_odds_export(snapshots, [], players)[0] == []


def test_odds_crosswalk_refuses_conflicting_game_ids():
    rows, _ = normalize_odds(event(), collected_at=AT)
    games = [
        {"provider_event_id": "provider-event", "game_id": "espn:1", "team": "Lakers"},
        {"provider_event_id": "provider-event", "game_id": "espn:2", "team": "Lakers"},
    ]
    assert build_odds_export([{"status": "OK", "rows": rows}], games, [])[0] == []


def test_pdf_name_suffix_is_preserved_when_nba_inserts_two_commas():
    from src.ingestion.model_input_archive import report_player_name
    assert report_player_name("Udeh, Jr., Ernest") == "Ernest Udeh Jr."
    assert report_player_name("Smith Jr., Jabari") == "Jabari Smith Jr."


def test_post_tip_quotes_are_not_pregame_training_data():
    payload = event(commence_time="2026-11-08T19:00:00Z")
    assert normalize_odds(payload, collected_at=AT)[0] == []


def test_a_pre_tip_quote_is_not_exported_for_a_post_tip_prediction_cutoff(tmp_path):
    store = InputArchive(tmp_path)
    rows, _ = normalize_odds(event(), collected_at=AT)
    save(store, rows, kind="odds")
    assert store.as_of("odds", AT + timedelta(hours=5)) == []


def test_pdf_internal_timestamp_must_match_the_source_url():
    from src.ingestion.model_input_archive import validate_report_title
    validate_report_title("Injury Report: 11/08/26 03:00 PM", AT)
    with pytest.raises(InputError):
        validate_report_title("Injury Report: 11/09/26 03:00 PM", AT)


def test_duplicate_market_blocks_cannot_create_two_conflicting_training_pairs():
    import copy
    payload = event()
    market = copy.deepcopy(payload["bookmakers"][0]["markets"][0])
    market["outcomes"][0]["price"] = -120
    payload["bookmakers"][0]["markets"].append(market)
    rows, rejected = normalize_odds(payload, collected_at=AT)
    assert rows == [] and rejected


def test_archive_corruption_is_loud(tmp_path):
    archive = InputArchive(tmp_path)
    path = save(archive, [entry()])
    # Intentional corruption of a fixture, not an archive writer operation.
    path.write_text("{}")
    with pytest.raises(InputError, match="checksum"):
        archive.as_of("availability", AT)
