"""Tests for the NBA CDN play-by-play fetcher."""

from __future__ import annotations

import json

import pandas as pd
import pytest
import requests

from src.ingestion.nba_playbyplay import (
    CDN_PLAYBYPLAY_URL,
    PlayByPlayError,
    fetch_many_playbyplay,
    fetch_playbyplay,
    fetch_playbyplay_frame,
    game_ids_from_panel,
    parse_playbyplay_payload,
)


def _action(n: int, **over):
    base = {
        "actionNumber": n, "orderNumber": n * 10, "clock": "PT11M58.00S",
        "period": 1, "periodType": "REGULAR", "actionType": "2pt",
        "subType": "Jump Shot", "personId": 201939, "teamId": 1610612744,
        "teamTricode": "GSW", "possession": 1610612744,
        "scoreHome": "2", "scoreAway": "0", "shotDistance": 24.0,
        "shotResult": "Made", "shotValue": 2, "isFieldGoal": 1,
        "assistPersonId": None, "x": 10.0, "y": 20.0,
        "description": "S. Curry 24' jump shot",
    }
    base.update(over)
    return base


def _payload(game_id="0022500001", n=3):
    return {
        "meta": {"version": 1, "code": 200},
        "game": {"gameId": game_id, "actions": [_action(i) for i in range(1, n + 1)]},
    }


class _Resp:
    def __init__(self, status=200, payload=None):
        self.status_code = status
        self._payload = payload if payload is not None else _payload()

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"{self.status_code}")


class _Session:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[str] = []

    def get(self, url, **kw):
        self.calls.append(url)
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


# --- parsing ----------------------------------------------------------------


def test_actions_flatten_with_the_game_id_stamped_on_every_row():
    """The action dicts carry no gameId, and every downstream key needs one."""
    frame = parse_playbyplay_payload(_payload(n=4))
    assert len(frame) == 4
    assert (frame["gameId"] == "0022500001").all()
    assert frame.columns[0] == "gameId"


def test_the_parsed_frame_carries_what_the_feature_layer_reads():
    from src.features.pbp import prepare_events, summarise_shots

    frame = parse_playbyplay_payload(_payload(n=5))
    events = prepare_events(frame)
    shots = summarise_shots(events)
    assert len(shots) == 1
    assert shots["PBP_FGA"].iloc[0] == 5
    assert shots["PBP_THREE_RATE"].iloc[0] == 0.0


def test_an_empty_action_list_raises_rather_than_returning_an_empty_frame():
    """An empty frame is indistinguishable from a game in which nothing
    happened, and would pass through the feature builder as a player with no
    shots."""
    for payload in ({"game": {"gameId": "1", "actions": []}}, {"game": {"gameId": "1"}}, {}):
        with pytest.raises(PlayByPlayError, match="DATA_NOT_AVAILABLE"):
            parse_playbyplay_payload(payload)


def test_a_payload_without_a_game_id_is_refused():
    with pytest.raises(PlayByPlayError, match="gameId"):
        parse_playbyplay_payload({"game": {"actions": [_action(1)]}})


def test_a_changed_schema_is_caught_not_absorbed():
    payload = _payload()
    for action in payload["game"]["actions"]:
        action.pop("clock")
    with pytest.raises(PlayByPlayError, match="missing"):
        parse_playbyplay_payload(payload)


def test_optional_fields_absent_from_a_game_become_null_columns():
    """A game with no three-pointers carries no shotValue; the column must
    still exist so a season concatenates to a stable shape."""
    payload = _payload()
    for action in payload["game"]["actions"]:
        action.pop("shotDistance")
        action.pop("area", None)
    frame = parse_playbyplay_payload(payload)
    assert "shotDistance" in frame.columns
    assert frame["shotDistance"].isna().all()


# --- fetching ---------------------------------------------------------------


def test_the_url_and_headers_match_the_box_score_fetcher():
    """Same host, same bot detection — the two must send the same headers."""
    from src.settlement.boxscore_fetcher import CDN_BOXSCORE_URL, HEADERS

    session = _Session([_Resp()])
    fetch_playbyplay("0022500001", session=session)
    assert session.calls[0] == CDN_PLAYBYPLAY_URL.format(game_id="0022500001")
    assert CDN_PLAYBYPLAY_URL.rsplit("/", 2)[0] == CDN_BOXSCORE_URL.rsplit("/", 2)[0]
    assert "Referer" in HEADERS and "User-Agent" in HEADERS


def test_a_short_game_id_is_zero_padded():
    session = _Session([_Resp()])
    fetch_playbyplay(22500001, session=session)
    assert "playbyplay_0022500001.json" in session.calls[0]


@pytest.mark.parametrize("status,match", [(403, "bot detection"), (404, "does not exist")])
def test_403_and_404_say_what_actually_happened(status, match):
    session = _Session([_Resp(status=status)])
    with pytest.raises(PlayByPlayError, match=match):
        fetch_playbyplay("0022500001", session=session)


def test_a_transient_error_is_retried(monkeypatch):
    monkeypatch.setattr("src.ingestion.nba_playbyplay.time.sleep", lambda *_: None)
    session = _Session([
        requests.exceptions.ConnectionError("reset"), _Resp(),
    ])
    frame = fetch_playbyplay_frame("0022500001", session=session)
    assert len(frame) == 3
    assert len(session.calls) == 2


def test_exhausted_retries_raise_rather_than_returning_nothing(monkeypatch):
    monkeypatch.setattr("src.ingestion.nba_playbyplay.time.sleep", lambda *_: None)
    session = _Session([requests.exceptions.ConnectionError("x")] * 3)
    with pytest.raises(PlayByPlayError, match="after 3 attempts"):
        fetch_playbyplay("0022500001", session=session)


# --- multi-game -------------------------------------------------------------


def test_failures_are_returned_not_swallowed():
    """A caller that gets 2 games back must be able to tell it asked for 3."""
    session = _Session([
        _Resp(payload=_payload("0022500001")),
        _Resp(status=404),
        _Resp(payload=_payload("0022500003")),
    ])
    events, failures = fetch_many_playbyplay(
        ["0022500001", "0022500002", "0022500003"], session=session, pause_seconds=0
    )
    assert events["gameId"].nunique() == 2
    assert len(failures) == 1
    assert failures[0]["game_id"] == "0022500002"


def test_overlapping_pulls_do_not_double_count_actions():
    session = _Session([
        _Resp(payload=_payload("0022500001")),
        _Resp(payload=_payload("0022500001")),
    ])
    events, _ = fetch_many_playbyplay(
        ["0022500001", "0022500001"], session=session, pause_seconds=0
    )
    assert len(events) == 3


def test_every_game_failing_raises():
    session = _Session([_Resp(status=404), _Resp(status=404)])
    with pytest.raises(PlayByPlayError, match="DATA_NOT_AVAILABLE"):
        fetch_many_playbyplay(["1", "2"], session=session, pause_seconds=0)


def test_stop_on_error_propagates():
    session = _Session([_Resp(status=404)])
    with pytest.raises(PlayByPlayError):
        fetch_many_playbyplay(["1"], session=session, pause_seconds=0, stop_on_error=True)


# --- panel driving ----------------------------------------------------------


def test_game_ids_come_back_in_date_order_deduplicated():
    """An interrupted pull should cover a contiguous span, not a scatter."""
    panel = pd.DataFrame({
        "GAME_ID": ["3", "1", "2", "1"],
        "GAME_DATE": pd.to_datetime(["2025-10-25", "2025-10-21", "2025-10-23", "2025-10-21"]),
        "SEASON": ["2025-26"] * 4,
    })
    assert game_ids_from_panel(panel) == ["0000000001", "0000000002", "0000000003"]


def test_seasons_filter_applies():
    panel = pd.DataFrame({
        "GAME_ID": ["1", "2"],
        "GAME_DATE": pd.to_datetime(["2024-10-21", "2025-10-21"]),
        "SEASON": ["2024-25", "2025-26"],
    })
    assert game_ids_from_panel(panel, seasons=["2025-26"]) == ["0000000002"]


def test_a_panel_without_game_ids_is_refused():
    with pytest.raises(PlayByPlayError, match="DATA_NOT_AVAILABLE"):
        game_ids_from_panel(pd.DataFrame({"PLAYER_ID": ["p"]}))


# --- the real-data round trip ----------------------------------------------


def test_a_real_game_round_trips_through_the_parser(tmp_path):
    """The strongest check available without network access: take a real game
    out of the uploaded CSVs, rebuild the CDN payload shape from it, and
    confirm the parser reproduces the columns the pipeline reads."""
    import glob

    files = sorted(glob.glob("data/external/training_pack/pbp/*.csv"))
    if not files:
        pytest.skip("no event-log CSVs available in this checkout")
    src = pd.read_csv(files[0], low_memory=False, nrows=4000)
    src["gameId"] = src["gameId"].astype(str)
    gid = src["gameId"].value_counts().index[0]
    real = src[src["gameId"] == gid]

    actions = json.loads(real.drop(columns=["gameId"]).to_json(orient="records"))
    parsed = parse_playbyplay_payload(
        {"meta": {}, "game": {"gameId": gid, "actions": actions}}
    )
    assert len(parsed) == len(real)
    for col in ("actionNumber", "clock", "period", "actionType", "shotResult"):
        left = real.sort_values("actionNumber")[col].astype(str).reset_index(drop=True)
        right = parsed.sort_values("actionNumber")[col].astype(str).reset_index(drop=True)
        assert left.equals(right), col
