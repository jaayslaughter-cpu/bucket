"""ESPN scoreboard -> slate.

FIXTURE PROVENANCE. The payloads below are derived from the response shape
documented in github.com/pseudo-r/public-espn-api (docs/response_schemas.md,
"Scoreboard"). They are NOT captured live: this environment denies outbound
CONNECT to site.api.espn.com, so the parser is tested against the documented
contract and the first real run is the real test. Nothing here is presented as
observed ESPN output.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest
import requests

from src.ingestion.espn_schedule import (
    STATE_FINAL,
    STATE_PREGAME,
    EspnScheduleConfig,
    EspnScheduleError,
    fetch_scoreboard,
    load_slate,
    normalize_team,
    parse_scoreboard,
    slate_join_keys,
)


def _event(
    event_id: str = "401765432",
    iso: str = "2025-03-15T02:30Z",
    home: str = "GSW",
    away: str = "BOS",
    state: str = "pre",
) -> dict:
    return {
        "id": event_id,
        "date": iso,
        "shortName": f"{away} @ {home}",
        "status": {"type": {"state": state, "shortDetail": "7:30 PM PT"}},
        "competitions": [{
            "id": event_id,
            "venue": {
                "fullName": "Chase Center",
                "address": {"city": "San Francisco", "state": "CA"},
            },
            "competitors": [
                {"homeAway": "home", "team": {"id": "9", "abbreviation": home}},
                {"homeAway": "away", "team": {"id": "2", "abbreviation": away}},
            ],
        }],
    }


def _payload(*events: dict) -> dict:
    return {"leagues": [{"abbreviation": "NBA"}], "events": list(events)}


# --- parsing the documented shape ---------------------------------------

def test_parses_a_documented_event():
    result = parse_scoreboard(_payload(_event()))
    assert result.status == "OK"
    assert len(result.games) == 1
    game = result.games[0]
    assert game.espn_event_id == "401765432"
    assert game.home_team == "GSW"
    assert game.away_team == "BOS"
    assert game.state == STATE_PREGAME
    assert game.venue == "Chase Center"
    assert game.venue_city == "San Francisco"


def test_tipoff_parses_espn_s_trailing_z_format():
    """2025-03-15T02:30Z is not a format fromisoformat accepts unmodified."""
    result = parse_scoreboard(_payload(_event(iso="2025-03-15T02:30Z")))
    tip = result.games[0].tipoff_utc
    assert tip == datetime(2025, 3, 15, 2, 30, tzinfo=timezone.utc)


def test_slate_date_is_the_pacific_day_not_the_utc_day():
    """A 02:30 UTC tip is the PREVIOUS evening in Pacific — the slate day."""
    result = parse_scoreboard(_payload(_event(iso="2025-03-15T02:30Z")))
    assert result.games[0].slate_date_pt == date(2025, 3, 14)


def test_unparseable_date_leaves_tipoff_null_rather_than_guessing():
    result = parse_scoreboard(_payload(_event(iso="not a date")))
    assert result.games[0].tipoff_utc is None
    assert result.games[0].slate_date_pt is None


# --- the two integration traps ------------------------------------------

def test_event_id_is_never_presented_as_a_game_id():
    """ESPN ids and NBA game ids are different namespaces."""
    row = parse_scoreboard(_payload(_event())).games[0].as_dict()
    assert "ESPN_EVENT_ID" in row
    assert "GAME_ID" not in row, (
        "an ESPN event id must not be exposed under a name that invites a "
        "join against the panel's NBA game ids"
    )


def test_join_keys_are_date_and_teams():
    result = parse_scoreboard(_payload(_event()))
    assert slate_join_keys(result) == [(date(2025, 3, 14), "GSW", "BOS")]


@pytest.mark.parametrize(
    "espn_code,expected",
    [("GS", "GSW"), ("NO", "NOP"), ("NY", "NYK"), ("SA", "SAS"),
     ("UTAH", "UTA"), ("WSH", "WAS"), ("PHO", "PHX")],
)
def test_known_divergent_codes_map_to_nba_codes(espn_code, expected):
    mapped, recognised = normalize_team(espn_code)
    assert (mapped, recognised) == (expected, True)


def test_codes_already_matching_pass_straight_through():
    assert normalize_team("BOS") == ("BOS", True)
    assert normalize_team(" gsw ") == ("GSW", True)


def test_an_unknown_code_is_reported_not_guessed():
    """The failure mode to avoid is mapping an unknown code to a near match."""
    mapped, recognised = normalize_team("ZZZ")
    assert mapped == "ZZZ" and recognised is False

    result = parse_scoreboard(_payload(_event(home="ZZZ")))
    assert result.unmapped_teams == ["ZZZ"]
    assert any("not guessed" in n for n in result.notes)
    # and the game is still returned, so the gap is visible rather than silent
    assert result.games[0].home_team == "ZZZ"


def test_neutral_site_is_unknown_not_false():
    """The scoreboard carries no neutral-site flag; None says so honestly."""
    assert parse_scoreboard(_payload(_event())).games[0].is_neutral_site is None


# --- degradation --------------------------------------------------------

def test_empty_events_abstains_and_says_why_it_is_ambiguous():
    result = parse_scoreboard({"events": []})
    assert result.status == "DATA_NOT_AVAILABLE"
    assert any("empty slate" in n for n in result.notes)


def test_non_dict_payload_abstains():
    assert parse_scoreboard(["not", "a", "dict"]).status == "DATA_NOT_AVAILABLE"


def test_one_malformed_event_does_not_cost_the_rest_of_the_slate():
    result = parse_scoreboard(_payload(_event("1"), {"no": "id"}, _event("3")))
    assert [g.espn_event_id for g in result.games] == ["1", "3"]
    assert any("skipped" in n for n in result.notes)


def test_missing_competitions_still_yields_the_event():
    bare = {"id": "9", "date": "2025-03-15T02:30Z", "status": {"type": {"state": "pre"}}}
    result = parse_scoreboard(_payload(bare))
    assert result.games[0].home_team is None
    assert result.games[0].away_team is None


def test_pregame_only_filters_out_started_games():
    result = parse_scoreboard(_payload(
        _event("1", state=STATE_PREGAME),
        _event("2", state=STATE_FINAL),
        _event("3", state="in"),
    ))
    assert [g.espn_event_id for g in result.pregame_only] == ["1"]


# --- fetch behaviour ----------------------------------------------------

class _Response:
    def __init__(self, status: int = 200, body: dict | None = None):
        self.status_code = status
        self._body = body if body is not None else {}

    def json(self) -> dict:
        return self._body

    def raise_for_status(self) -> None:
        if self.status_code >= 500:
            raise requests.exceptions.HTTPError(f"{self.status_code}")


class _Session:
    def __init__(self, *responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def get(self, url, params=None, timeout=None, headers=None):
        self.calls.append({"url": url, "params": params})
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_the_date_is_sent_as_yyyymmdd():
    session = _Session(_Response(200, _payload(_event())))
    fetch_scoreboard(date(2025, 3, 14), session=session)
    assert session.calls[0]["params"] == {"dates": "20250314"}


def test_a_4xx_is_not_retried():
    session = _Session(_Response(404))
    with pytest.raises(EspnScheduleError, match="retrying will not help"):
        fetch_scoreboard(date(2025, 3, 14), session=session)
    assert len(session.calls) == 1


def test_a_transport_error_retries_then_raises_rather_than_returning_empty():
    cfg = EspnScheduleConfig(retry_attempts=2, retry_backoff=1.0)
    session = _Session(
        requests.exceptions.ConnectionError("reset"),
        requests.exceptions.ConnectionError("reset"),
    )
    with pytest.raises(EspnScheduleError, match="unreachable after 2 attempts"):
        fetch_scoreboard(date(2025, 3, 14), config=cfg, session=session)
    assert len(session.calls) == 2


def test_a_retry_that_succeeds_returns_the_payload():
    cfg = EspnScheduleConfig(retry_attempts=3, retry_backoff=1.0)
    session = _Session(
        requests.exceptions.ConnectionError("reset"),
        _Response(200, _payload(_event())),
    )
    out = load_slate(date(2025, 3, 14), config=cfg, session=session)
    assert out.status == "OK" and len(out.games) == 1


def test_base_url_is_overridable_by_env_not_hardcoded(monkeypatch):
    monkeypatch.setenv("PROPIQ_ESPN_BASE_URL", "https://mirror.invalid/scoreboard")
    assert EspnScheduleConfig().base_url == "https://mirror.invalid/scoreboard"


def test_no_api_key_is_read_or_required():
    """The endpoint is public; a credential here would be a mistake to catch."""
    import inspect

    from src.ingestion import espn_schedule

    source = inspect.getsource(espn_schedule)
    for forbidden in ("API_KEY", "api_key", "SECRET", "token"):
        assert forbidden not in source, f"{forbidden!r} has no business here"
