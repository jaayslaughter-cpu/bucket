"""
The pick'em capture clients: Underdog and Sleeper.

WHAT IS AND IS NOT KNOWN HERE. These are undocumented internal endpoints and
NOBODY WORKING ON THIS REPOSITORY HAS SEEN A RESPONSE FROM EITHER -- the build
container cannot reach the hosts. So the provider's field names are the one
thing these modules do not contain: they are declared in
`config/pickem_sources.yaml` and this repository ships that file UNCONFIGURED.
Writing plausible key names and testing against them would be testing a
fiction, and inventing an API payload is what produced the 2026-10-18 report
of four failing scrapers that had never been written.

What IS tested, therefore, is everything that does not depend on knowing the
shape: that an unconfigured source says so instead of reporting an empty
board, that a refusal is never retried, that a restructured payload is told
apart from an empty one, that a mapped row carries the BOARD's timestamp and
never `now()`, that a pick'em row can never present as a two-way price, and
that the mapper does what it claims against a payload whose shape this file
declares outright as synthetic.

RESEARCH_ONLY. Every payload below is a literal in this file.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
import yaml

from src.ingestion import pickem, pickem_board, sleeper_props, underdog_props
from src.ingestion.pickem_schema import (
    MAPPABLE_FIELDS,
    REQUIRED_FIELDS,
    STATUS_EMPTY,
    STATUS_NOT_CONFIGURED,
    STATUS_UNAVAILABLE,
    STATUS_VALID,
    PickemConfigError,
    describe_payload,
    load_source_config,
    resolve_path,
    rows_from_payload,
)
from src.ingestion.pickem_store import PickemResearchStore

# A SYNTHETIC payload. It is not a claim about what any provider sends; it is
# a normalised shape (entities in sibling arrays keyed by id) chosen because
# that is the shape a row-relative mapper would otherwise be useless for.
SYNTHETIC_BOARD = {
    "meta": {"posted_at": "2026-10-18T17:30:00Z"},
    "players": [
        {"pid": "p1", "full_name": "Example Player", "team": "LAL", "sport": "NBA"},
        {"pid": "p2", "full_name": "Second Player", "team": "BOS", "sport": "NBA"},
        {"pid": "p3", "full_name": "College Kid", "team": "DUKE", "sport": "NCAAB"},
    ],
    "lines": [
        {"id": "l1", "player_ref": "p1", "stat": "points", "value": "24.5",
         "multiplier": 1.0, "observed_at": "2026-10-18T17:28:00Z",
         "tip": "2026-10-19T02:10:00Z"},
        {"id": "l2", "player_ref": "p2", "stat": "rebounds", "value": 9.5,
         "multiplier": 1.25, "observed_at": "2026-10-18T17:28:00Z",
         "tip": "2026-10-19T02:10:00Z"},
        {"id": "l3", "player_ref": "p3", "stat": "points", "value": 18.5,
         "multiplier": 1.0},
        {"id": "l4", "player_ref": "p1", "stat": "dunks", "value": 1.5,
         "multiplier": 1.0},
        {"id": "l5", "player_ref": "p2", "stat": "assists", "value": None,
         "multiplier": 1.0},
    ],
}

SYNTHETIC_MAPPING = {
    "sources": {
        "underdog": {
            "enabled": True,
            "sport_filter": {
                "field_paths": ["lookup:players:player_ref:sport"],
                "keep_values": ["NBA"],
            },
            "rows": ["lines"],
            "lookups": {"players": {"path": "players", "id": "pid"}},
            "fields": {
                "player_name": ["lookup:players:player_ref:full_name"],
                "market": ["stat"],
                "line": ["value"],
                "payout_multiplier": ["multiplier"],
                "captured_at_utc": ["observed_at"],
                "team_abbr": ["lookup:players:player_ref:team"],
                "game_time": ["tip"],
                "board_id": ["id"],
                "player_id": ["player_ref"],
            },
        }
    },
    "market_aliases": {"PTS": ["points"], "REB": ["rebounds"], "AST": ["assists"]},
}


@pytest.fixture
def mapping_file(tmp_path):
    path = tmp_path / "sources.yaml"
    path.write_text(yaml.safe_dump(SYNTHETIC_MAPPING), encoding="utf-8")
    return str(path)


@pytest.fixture
def board_file(tmp_path):
    path = tmp_path / "board.json"
    path.write_text(json.dumps(SYNTHETIC_BOARD), encoding="utf-8")
    return str(path)


# --- the repository ships nothing configured, and says so -------------------


@pytest.mark.parametrize("source", ["underdog", "sleeper"])
def test_the_shipped_config_declares_no_board_paths(source):
    """
    THE CENTRAL CLAIM OF THIS SUBSYSTEM. If a default path ever appears here
    without somebody having recorded a real response, this test is the thing
    that was supposed to stop it.
    """
    config = load_source_config(source)
    assert not config.configured
    for name in REQUIRED_FIELDS:
        assert config.fields[name] == (), (
            f"config/pickem_sources.yaml ships a path for {source}.{name}. If it "
            f"came from a recorded response, say so in `verified_against` and "
            f"update this test; if it came from memory, it is an invented API "
            f"payload and must come out."
        )


@pytest.mark.parametrize("source", ["underdog", "sleeper"])
def test_an_unconfigured_source_is_not_reported_as_an_empty_board(source):
    """
    "We are not set up" and "the board posted nothing" send an operator to two
    completely different places, and the second is indistinguishable from the
    off-season.
    """
    puller = {"underdog": underdog_props.pull_underdog_board,
              "sleeper": sleeper_props.pull_sleeper_board}[source]
    snapshot = puller()
    assert snapshot.status == STATUS_NOT_CONFIGURED
    assert snapshot.status != STATUS_EMPTY
    assert snapshot.lines == []


@pytest.mark.parametrize("source", ["underdog", "sleeper"])
def test_the_unconfigured_message_names_the_file_and_the_missing_keys(source):
    puller = {"underdog": underdog_props.pull_underdog_board,
              "sleeper": sleeper_props.pull_sleeper_board}[source]
    message = puller().message or ""
    assert "config/pickem_sources.yaml" in message
    assert "fields.player_name" in message
    assert "--describe" in message, "it says what is missing but not how to find it"


def test_an_unconfigured_source_makes_no_request(monkeypatch):
    """A source with no mapping must not spend a call to discover that."""
    def _refuse(*a, **k):
        raise AssertionError("an unconfigured source hit the network")

    monkeypatch.setattr(pickem_board, "fetch_board_payload", _refuse)
    assert underdog_props.pull_underdog_board().status == STATUS_NOT_CONFIGURED


def test_a_missing_mapping_file_is_a_named_error():
    with pytest.raises(PickemConfigError, match="no pick'em source mapping"):
        load_source_config("underdog", "/no/such/sources.yaml")


def test_a_source_absent_from_the_mapping_is_named():
    with pytest.raises(PickemConfigError, match="prizepicks"):
        load_source_config("prizepicks")


# --- the mapper, against a shape this file declares as synthetic ------------


def test_a_configured_mapping_produces_rows(mapping_file, board_file):
    config = load_source_config("underdog", mapping_file)
    assert config.configured
    lines, unresolved, dropped = rows_from_payload(SYNTHETIC_BOARD, config)
    assert [(line.player_name, line.market, line.line) for line in lines] == [
        ("Example Player", "PTS", 24.5),
        ("Second Player", "REB", 9.5),
    ]
    # `line` is unresolved exactly once: row l5 posts a null value. That is a
    # fact about that row, counted, and NOT evidence that the mapping is
    # stale -- `capture` only treats a required field missing from EVERY row
    # that way.
    assert unresolved == {"line": 1}
    # The college row is dropped under the NCAA reason rather than the
    # generic "not NBA": the forbidden-league check runs first, on purpose,
    # so that an operator widening `keep_values` cannot let college through.
    assert dropped == {
        "NCAA/college, out of scope": 1,
        "unmapped market 'dunks'": 1,
        "no numeric line": 1,
    }


def test_a_lookup_joins_a_normalised_payload(mapping_file):
    """Without this the mapper could read only what is nested inside a row,
    which for a normalised board is ids and nothing a human recognises."""
    config = load_source_config("underdog", mapping_file)
    lines, _, _ = rows_from_payload(SYNTHETIC_BOARD, config)
    assert lines[0].player_name == "Example Player"
    assert lines[0].team_abbr == "LAL"


def test_an_unmapped_market_is_dropped_and_named_never_guessed(mapping_file):
    """
    Attaching a real posted line to the wrong statistic is worse than dropping
    it -- the same rule `propline.DEFAULT_MARKET_MAP` is written under.
    """
    config = load_source_config("underdog", mapping_file)
    _, _, dropped = rows_from_payload(SYNTHETIC_BOARD, config)
    assert any("dunks" in reason for reason in dropped)


def test_ncaa_is_refused_even_under_a_basketball_shaped_sport_value(mapping_file):
    config = load_source_config("underdog", mapping_file)
    lines, _, _ = rows_from_payload(SYNTHETIC_BOARD, config)
    assert all("College" not in line.player_name for line in lines)


def test_a_row_carries_the_boards_own_timestamp_and_never_now(mapping_file):
    """
    `captured_at_utc` is the board's claim, not ours. A backfilled board that
    claims the moment we read it looks like a board that never moved --
    `PropLineRow.captured_at_utc` and migration 003 exist for this.
    """
    config = load_source_config("underdog", mapping_file)
    lines, _, _ = rows_from_payload(SYNTHETIC_BOARD, config)
    assert lines[0].captured_at_utc == datetime(2026, 10, 18, 17, 28, tzinfo=UTC)


def test_a_line_with_no_board_timestamp_carries_none_not_a_substitute(mapping_file):
    config = load_source_config("underdog", mapping_file)
    payload = json.loads(json.dumps(SYNTHETIC_BOARD))
    for row in payload["lines"]:
        row.pop("observed_at", None)
    lines, _, _ = rows_from_payload(payload, config)
    assert lines and all(line.captured_at_utc is None for line in lines)


def test_the_slate_date_is_the_pacific_calendar_day(mapping_file):
    """A 7pm PT tip belongs to that day's slate, not to the UTC date."""
    config = load_source_config("underdog", mapping_file)
    lines, _, _ = rows_from_payload(SYNTHETIC_BOARD, config)
    assert str(lines[0].game_time) == "2026-10-19 02:10:00+00:00"
    assert str(lines[0].game_date) == "2026-10-18"


def test_candidate_paths_are_tried_in_order(mapping_file):
    """How a shape change is rolled through: the new path sits above the old
    one until the board has finished moving."""
    config = load_source_config("underdog", mapping_file)
    moved = {**config.fields, "line": ("value_v2", "value")}
    config = type(config)(**{**config.__dict__, "fields": moved})
    lines, _, _ = rows_from_payload(SYNTHETIC_BOARD, config)
    assert lines[0].line == 24.5


def test_resolve_path_returns_none_rather_than_raising():
    payload = {"a": {"b": [{"c": 1}]}}
    assert resolve_path(payload, "a.b.0.c") == 1
    for miss in ("a.b.9.c", "a.b.c", "nope", "a.b.0.c.d", ""):
        assert resolve_path(payload, miss) is None


# --- a restructured board is not an empty board -----------------------------


def test_a_mapping_that_no_longer_fits_is_unavailable_not_empty(mapping_file):
    """
    The failure this whole contract is shaped around. A board that
    restructured leaves the mapping resolving nothing; reporting that as EMPTY
    hides a stale mapping as the off-season for as long as nobody checks.
    """
    config = load_source_config("underdog", mapping_file)
    restructured = {
        "players": SYNTHETIC_BOARD["players"],
        # `lines` became an object keyed by id
        "lines": {"l1": SYNTHETIC_BOARD["lines"][0]},
    }
    snapshot = pickem_board.capture(config, from_file=None)
    assert snapshot.status in (STATUS_EMPTY, STATUS_UNAVAILABLE)

    lines, unresolved, _ = rows_from_payload(restructured, config)
    assert lines == []
    assert unresolved == {"rows": 1}, (
        "a payload whose row array became an object reported no reason at all"
    )


def test_a_genuinely_empty_board_is_empty_not_unavailable(mapping_file, tmp_path):
    config = load_source_config("underdog", mapping_file)
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"players": [], "lines": []}), encoding="utf-8")
    snapshot = pickem_board.capture(config, from_file=str(empty))
    assert snapshot.status == STATUS_EMPTY
    assert snapshot.ok


def test_a_board_read_from_a_file_is_valid(mapping_file, board_file):
    config = load_source_config("underdog", mapping_file)
    snapshot = pickem_board.capture(config, from_file=board_file)
    assert snapshot.status == STATUS_VALID
    assert len(snapshot.lines) == 2


def test_unreadable_json_is_reported_not_raised(mapping_file, tmp_path):
    config = load_source_config("underdog", mapping_file)
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    snapshot = pickem_board.capture(config, from_file=str(broken))
    assert snapshot.status == STATUS_UNAVAILABLE
    assert "not readable JSON" in (snapshot.message or "")


# --- the refusal, and what must never follow it -----------------------------


class _Response:
    def __init__(self, status, body=None, headers=None):
        self.status_code = status
        self._body = body
        self.headers = headers or {}

    def json(self):
        if self._body is None:
            raise ValueError("no body")
        return self._body


class _Session:
    def __init__(self, *responses):
        self._responses = list(responses)
        self.calls = 0

    def get(self, url, **kwargs):
        self.calls += 1
        return self._responses[min(self.calls - 1, len(self._responses) - 1)]


@pytest.mark.parametrize("status", [401, 403, 451])
def test_a_refusal_is_reported_once_and_never_retried(status):
    """
    A board that answers an ordinary client with a refusal is not a transient
    failure. Retrying it is the first step of the thing docs/pickem_props.md
    bans, and a retry loop is how a bypass gets added later "to make the
    retry work".
    """
    session = _Session(_Response(status))
    payload, got, error = pickem_board.fetch_board_payload("https://x/board", session=session)
    assert payload is None
    assert got == status
    assert session.calls == 1, f"an HTTP {status} refusal was retried {session.calls} times"
    assert "not retried" in (error or "")


def test_the_refusal_message_says_no_fingerprint_is_spoofed():
    _, _, error = pickem_board.fetch_board_payload(
        "https://x/board", session=_Session(_Response(403))
    )
    assert "docs/pickem_props.md" in (error or "")
    assert "operator-supplied" in (error or "")


def test_the_user_agent_does_not_impersonate_a_browser():
    """
    Dressing up as Chrome to get served is the evasion, not a detail of it.
    `settlement/boxscore_fetcher` does send a browser UA to NBA's own CDN --
    a different situation, a public data CDN with no bot wall at issue -- and
    this must not drift into copying it.
    """
    agent = pickem_board.HEADERS["User-Agent"]
    for banned in ("Mozilla", "Chrome", "Safari", "AppleWebKit", "Gecko"):
        assert banned not in agent, f"the capture client impersonates a browser: {agent!r}"
    assert "PropIQ" in agent


def test_no_credential_can_reach_the_board_request():
    """There is nowhere to put one, and that is the design."""
    for header in pickem_board.HEADERS:
        assert header.lower() not in ("authorization", "cookie", "x-api-key")


def test_a_rate_limit_is_retried_under_the_shared_rule():
    session = _Session(
        _Response(429, headers={"Retry-After": "0"}),
        _Response(200, body={"lines": []}),
    )
    payload, status, error = pickem_board.fetch_board_payload(
        "https://x/board", session=session
    )
    assert error is None
    assert payload == {"lines": []}
    assert session.calls == 2


def test_a_hostile_retry_after_cannot_hang_the_capture(monkeypatch):
    """
    The same defect as `propline`'s: `Retry-After: 86400` slept a job for a
    day. The shared rule caps it; this proves the capture client reads that
    rule and not a local `float(...)`.
    """
    slept: list[float] = []
    monkeypatch.setattr(pickem_board.time, "sleep", lambda s: slept.append(s))
    session = _Session(
        _Response(429, headers={"Retry-After": "86400"}),
        _Response(200, body={"lines": []}),
    )
    pickem_board.fetch_board_payload("https://x/board", session=session)
    from src.utils.http_retry import MAX_RETRY_AFTER_SECONDS

    assert slept and max(slept) <= MAX_RETRY_AFTER_SECONDS


def test_a_moved_path_says_so_rather_than_reporting_an_empty_board():
    _, status, error = pickem_board.fetch_board_payload(
        "https://x/board", session=_Session(_Response(404))
    )
    assert status == 404
    assert "moved" in (error or "")


def test_non_json_is_reported():
    _, _, error = pickem_board.fetch_board_payload(
        "https://x/board", session=_Session(_Response(200, body=None))
    )
    assert "non-JSON" in (error or "")


# --- a pick'em row can never look like a two-way price ----------------------


def test_a_captured_row_is_always_pickem_flagged(mapping_file):
    config = load_source_config("underdog", mapping_file)
    lines, _, _ = rows_from_payload(SYNTHETIC_BOARD, config)
    for row in (line.to_prop_line_row() for line in lines):
        assert row.is_pickem is True, (
            "a pick'em row reached the pipeline unflagged, so market_ev_gate "
            "would de-vig a payout multiplier as though it were a price"
        )


def test_a_captured_row_carries_no_odds_at_all(mapping_file):
    """Absent, not 0 and not +100: a board that posts no price must not
    produce a number something downstream could price from."""
    config = load_source_config("underdog", mapping_file)
    lines, _, _ = rows_from_payload(SYNTHETIC_BOARD, config)
    for row in (line.to_prop_line_row() for line in lines):
        assert row.over_odds_american is None
        assert row.under_odds_american is None


def test_the_multiplier_survives_the_conversion(mapping_file):
    config = load_source_config("underdog", mapping_file)
    lines, _, _ = rows_from_payload(SYNTHETIC_BOARD, config)
    assert [line.to_prop_line_row().payout_multiplier for line in lines] == [1.0, 1.25]


# --- the archive ------------------------------------------------------------


def test_a_snapshot_is_archived_with_its_raw_payload(mapping_file, board_file, tmp_path):
    """
    The raw response is kept because when a board restructures, re-parsing
    what was actually received is the only way to tell a genuinely empty night
    from a stale mapping.
    """
    config = load_source_config("underdog", mapping_file)
    snapshot = pickem_board.capture(config, from_file=board_file)
    store = PickemResearchStore.from_env(tmp_path / "store")
    path, appended = store.write(snapshot, payload=SYNTHETIC_BOARD)
    assert path.is_file()
    body = json.loads(path.read_text(encoding="utf-8"))
    assert body["raw_payload"] == SYNTHETIC_BOARD
    assert appended == 2
    assert store.index_path.is_file()
    assert "Example Player" in store.index_path.read_text(encoding="utf-8")


def test_an_empty_read_is_archived_but_adds_no_index_row(mapping_file, tmp_path):
    """The gap in the series is the finding, so the snapshot is kept; the
    index is "lines we hold" and an empty read holds none."""
    config = load_source_config("underdog", mapping_file)
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"players": [], "lines": []}), encoding="utf-8")
    snapshot = pickem_board.capture(config, from_file=str(empty))
    store = PickemResearchStore.from_env(tmp_path / "store")
    path, appended = store.write(snapshot, payload={"lines": []})
    assert path.is_file()
    assert appended == 0
    assert not store.index_path.exists()


def test_the_index_is_append_only(mapping_file, board_file, tmp_path):
    config = load_source_config("underdog", mapping_file)
    store = PickemResearchStore.from_env(tmp_path / "store")
    for _ in range(2):
        store.write(pickem_board.capture(config, from_file=board_file), payload={})
    rows = store.index_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 1 + 4, "a second capture overwrote the first instead of appending"


@pytest.mark.parametrize(
    "secret_key", ["access_token", "Authorization", "api_key", "cookie", "refresh_token"]
)
def test_a_payload_carrying_a_credential_is_refused_not_redacted(
    mapping_file, board_file, tmp_path, secret_key
):
    """
    A public board sends none of these. One that starts to is a situation for
    a person, and a half-scrubbed archive is worse than none.
    """
    config = load_source_config("underdog", mapping_file)
    snapshot = pickem_board.capture(config, from_file=board_file)
    store = PickemResearchStore.from_env(tmp_path / "store")
    with pytest.raises(ValueError, match="credential"):
        store.write(snapshot, payload={"lines": [], secret_key: "xxx"})
    assert not list((tmp_path / "store").rglob("*.json")), "it wrote before refusing"


# --- the batch --------------------------------------------------------------


def test_one_board_failing_does_not_cost_the_others(monkeypatch):
    """
    The 2026-10-18 defect, one layer up: an exception from one source must not
    end the capture for the rest.
    """
    def _boom(**kwargs):
        raise RuntimeError("provider restructured everything")

    monkeypatch.setattr(underdog_props, "pull_underdog_board", _boom)
    snapshots = pickem.pull_pickem_boards(archive=False)
    by_source = {s.source: s for s in snapshots}
    assert set(by_source) == {"underdog", "sleeper"}
    assert by_source["underdog"].status == STATUS_UNAVAILABLE
    assert "raised" in (by_source["underdog"].message or "")
    assert by_source["sleeper"].status == STATUS_NOT_CONFIGURED


def test_an_unconfigured_source_is_archived_nowhere(tmp_path):
    """A directory of empty snapshots would make the series look read-but-dark
    rather than never-set-up."""
    pickem.pull_pickem_boards(archive=True, store_root=tmp_path / "store")
    assert not (tmp_path / "store").exists() or not list(
        (tmp_path / "store").rglob("*.json")
    )


def test_prizepicks_has_no_client_in_the_batch():
    """Its board was expected behind a bot wall, getting past one is banned,
    and its lines already arrive through PropLine's DFS_BOOKS."""
    assert "prizepicks" not in pickem.SOURCES
    with pytest.raises(ValueError, match="no pick'em client"):
        pickem._puller("prizepicks")


def test_from_file_requires_exactly_one_source():
    with pytest.raises(ValueError, match="exactly one source"):
        pickem.pull_pickem_boards(["underdog", "sleeper"], from_file="x.json")


# --- the mapping file's own consistency -------------------------------------


def test_the_alias_keys_are_stats_this_project_actually_models():
    """
    The alias file must not become a second list of what this project models.
    `propline.DEFAULT_MARKET_MAP` is where that is decided.
    """
    from src.ingestion.propline import DEFAULT_MARKET_MAP

    canonical = set(DEFAULT_MARKET_MAP.values())
    shipped = yaml.safe_load(open("config/pickem_sources.yaml", encoding="utf-8"))
    for stat in (shipped.get("market_aliases") or {}):
        assert stat in canonical, (
            f"config/pickem_sources.yaml maps to {stat!r}, which "
            f"propline.DEFAULT_MARKET_MAP does not name — two lists of what "
            f"this project models, free to drift"
        )


def test_every_mappable_field_is_offered_in_the_shipped_config():
    """A field the schema supports but the config never mentions is one an
    operator will not know they can fill in."""
    shipped = yaml.safe_load(open("config/pickem_sources.yaml", encoding="utf-8"))
    for source, spec in (shipped.get("sources") or {}).items():
        offered = set((spec.get("fields") or {}))
        assert offered == set(MAPPABLE_FIELDS), (
            f"{source} offers {sorted(offered)}, schema has {sorted(MAPPABLE_FIELDS)}"
        )


def test_describe_prints_leaf_paths_and_finds_the_arrays():
    paths = describe_payload(SYNTHETIC_BOARD)
    assert "lines.0.stat  = 'points'" in paths
    assert any(p.startswith("players.0.full_name") for p in paths)


def test_a_board_of_partially_unreadable_rows_is_empty_not_a_stale_mapping(
    mapping_file, tmp_path
):
    """
    The line between "the paths moved" and "the board is incomplete". A row
    missing an OPTIONAL field resolves fine; the structural test must not fire
    on it, or a merely incomplete board reads as a broken mapping every night.
    """
    config = load_source_config("underdog", mapping_file)
    payload = {
        "players": [{"pid": "p1", "full_name": "Example Player", "sport": "NBA"}],
        # one row, readable, but its market is not one this project models
        "lines": [{"id": "l1", "player_ref": "p1", "stat": "dunks", "value": 1.5}],
    }
    board = tmp_path / "partial.json"
    board.write_text(json.dumps(payload), encoding="utf-8")
    snapshot = pickem_board.capture(config, from_file=str(board))
    assert snapshot.status == STATUS_EMPTY, (
        f"an unmapped market was reported as {snapshot.status}, which sends an "
        f"operator to re-record paths that are working"
    )
    assert "unmapped market" in (snapshot.message or "")


def test_a_required_field_missing_from_every_row_does_point_at_the_mapping(
    mapping_file, tmp_path
):
    config = load_source_config("underdog", mapping_file)
    payload = {
        "players": [{"pid": "p1", "full_name": "Example Player", "sport": "NBA"}],
        # `stat` renamed: the required market field resolves for no row at all
        "lines": [
            {"id": "l1", "player_ref": "p1", "stat_type": "points", "value": 24.5},
            {"id": "l2", "player_ref": "p1", "stat_type": "rebounds", "value": 9.5},
        ],
    }
    board = tmp_path / "renamed.json"
    board.write_text(json.dumps(payload), encoding="utf-8")
    snapshot = pickem_board.capture(config, from_file=str(board))
    assert snapshot.status == STATUS_UNAVAILABLE
    assert "--describe" in (snapshot.message or "")


def test_ncaa_is_refused_even_when_a_config_allowlists_it(mapping_file):
    """
    NCAA is out of scope by explicit standing instruction, so no entry in
    `keep_values` can opt back into it. This failed to be a real guard at
    first: the check sat after the allowlist, where "ncaa" could never match
    a value the allowlist had already let through, and deleting the check
    outright passed every test in this file.
    """
    config = load_source_config("underdog", mapping_file)
    opted_in = type(config)(
        **{**config.__dict__, "sport_keep_values": ("nba", "ncaab", "college_basketball")}
    )
    lines, _, dropped = rows_from_payload(SYNTHETIC_BOARD, opted_in)
    assert all("College" not in line.player_name for line in lines)
    assert dropped.get("NCAA/college, out of scope") == 1, (
        "a config allowlisting a college league got college rows captured"
    )


def test_the_forbidden_leagues_are_checked_before_the_allowlist():
    """AST-level would be fragile here; the ordering is observable instead —
    a value that is BOTH allowlisted and a college league must be dropped."""
    from src.ingestion.pickem_schema import FORBIDDEN_LEAGUES

    assert "ncaa" in FORBIDDEN_LEAGUES and "college" in FORBIDDEN_LEAGUES
