"""
Tests for src/ingestion/starting_positions.py — the live writer for
STARTING_POSITION.

This is the input src/features/dvp.py has been abstaining for want of on every
live row, so the load-bearing properties are the ones that decide whether the
layer comes on with the SAME quantity it was measured against:

  * a blank position means BENCH and must not become a bucket. Eight of
    thirteen rows in a real box score are blank;
  * the five-starters gate separates a STARTING position from a LISTED one.
    Both parse, both look plausible, and only one is what POS_BUCKET means;
  * the join actually matches — the panel stores GAME_ID unpadded and the NBA
    returns it zero-padded to ten, so a naive merge matches nothing at all.

Everything runs offline: stats.nba.com is denied at this environment's proxy,
so the fetch is injected rather than performed. The column names in the
payload fixtures below are copied from the installed library's own
``expected_data`` (nba_api/stats/endpoints/boxscoretraditionalv{2,3}.py), not
from memory, because this project has already shipped one wrong claim about an
endpoint's header list.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.ingestion.starting_positions import (
    STARTERS_PER_TEAM,
    STARTING_POSITION_COLUMNS,
    StartingPositionError,
    cache_path_for,
    check_starting_position_semantics,
    fetch_many_starting_positions,
    fetch_starting_positions,
    load_cached_starting_positions,
    parse_starting_positions,
    save_starting_positions,
)

GAME = "0022500123"
HOME, AWAY = "1610612747", "1610612738"

# Five starters per team in the 2:2:1 mix the archive uses, then bench rows
# whose position is the empty string the endpoint actually sends.
_STARTERS = [("G", "F"), ("G", "F"), ("F", "G"), ("F", "G"), ("C", "C")]


def _v3_rows(game_id=GAME, bench_per_team=3, bench_value=""):
    rows = []
    pid = 1000
    for team, index in ((HOME, 0), (AWAY, 1)):
        for pos_pair in _STARTERS:
            pid += 1
            rows.append([game_id, int(team), pid, pos_pair[index]])
        for _ in range(bench_per_team):
            pid += 1
            rows.append([game_id, int(team), pid, bench_value])
    return rows


def _v3_payload(game_id=GAME, rows=None, headers=None, **kw):
    """The v3 data-set shape nba_api's get_data_sets produces."""
    return {
        "PlayerStats": {
            "headers": headers or ["gameId", "teamId", "personId", "position"],
            "data": _v3_rows(game_id, **kw) if rows is None else rows,
        },
        "TeamStats": {"headers": ["gameId"], "data": [[game_id]]},
    }


def _v2_payload(game_id=GAME, rows=None):
    if rows is None:
        rows = [
            [r[0], r[1], r[2], f"Player {r[2]}", r[3]]
            for r in _v3_rows(game_id)
        ]
    return {
        "PlayerStats": {
            "headers": [
                "GAME_ID", "TEAM_ID", "PLAYER_ID", "PLAYER_NAME", "START_POSITION",
            ],
            "data": rows,
        },
    }


# --- it parses both versions onto one schema -------------------------------

def test_the_v3_payload_parses_to_the_normalised_schema():
    out = parse_starting_positions(_v3_payload(), GAME)
    assert list(out.columns) == list(STARTING_POSITION_COLUMNS)
    assert len(out) == 16  # 5 starters + 3 bench, two teams
    assert out["STARTING_POSITION"].notna().sum() == 10


def test_the_deprecated_v2_payload_parses_to_the_same_schema():
    """
    v2 names the column START_POSITION and is the one every published example
    uses. It is the FALLBACK, not the default, because the library's own
    module says its data is no longer published from 2025-26 — a season in
    this panel.
    """
    out = parse_starting_positions(_v2_payload(), GAME)
    assert list(out.columns) == list(STARTING_POSITION_COLUMNS)
    assert out["STARTING_POSITION"].notna().sum() == 10
    assert set(out.loc[out["STARTING_POSITION"].notna(), "STARTING_POSITION"]) == {
        "G", "F", "C"
    }


def test_both_versions_produce_identical_rows():
    """Two names for one quantity must not become two quantities."""
    v3 = parse_starting_positions(_v3_payload(), GAME)
    v2 = parse_starting_positions(_v2_payload(), GAME)
    pd.testing.assert_frame_equal(v3, v2)


def test_the_mix_matches_the_archive_two_guards_two_forwards_one_centre():
    """
    The archive's 214,381 rows are 35,594 G / 35,593 F / 17,799 C. A writer
    that produced a different mix would be writing a different thing.
    """
    out = parse_starting_positions(_v3_payload(), GAME)
    counts = out["STARTING_POSITION"].value_counts().to_dict()
    assert counts == {"G": 4, "F": 4, "C": 2}


# --- what a blank means ----------------------------------------------------

def test_a_blank_position_is_bench_and_does_not_become_a_bucket():
    """
    The single most consequential line here. Eight of thirteen rows in a real
    box score are blank, and a blank mapped to any bucket would put every
    bench player into an opponent's allowed-against-that-position average.
    """
    out = parse_starting_positions(_v3_payload(), GAME)
    assert out["STARTING_POSITION"].isna().sum() == 6
    assert "" not in set(out["STARTING_POSITION"].dropna())


@pytest.mark.parametrize("blank", ["", "   ", None])
def test_every_spelling_of_blank_is_null(blank):
    out = parse_starting_positions(_v3_payload(bench_value=blank), GAME)
    assert out["STARTING_POSITION"].isna().sum() == 6


def test_an_unrecognised_position_is_left_null_rather_than_guessed(caplog):
    """
    A bucket guessed from an unknown spelling puts the player in the wrong
    defensive population and still produces a plausible number.
    """
    rows = _v3_rows()
    rows[0][3] = "WING"
    with caplog.at_level("WARNING"):
        out = parse_starting_positions(_v3_payload(rows=rows), GAME)
    assert out["STARTING_POSITION"].isna().sum() == 7
    assert "unrecognised" in caplog.text and "WING" in caplog.text


# --- THE GATE: a starting position is not a listed position ----------------

def test_the_gate_passes_a_real_starting_position_payload():
    report = check_starting_position_semantics(
        parse_starting_positions(_v3_payload(), GAME)
    )
    assert report["team_games"] == 2
    assert report["starters_per_team_game_min"] == STARTERS_PER_TEAM
    assert report["starters_per_team_game_max"] == STARTERS_PER_TEAM


def test_the_gate_refuses_a_listed_position_payload_and_says_which_it_is():
    """
    THE DEFECT THIS EXISTS TO CATCH. v3 renamed v2's START_POSITION to
    `position`, and this environment cannot reach the endpoint to confirm the
    rename kept the meaning. If `position` were the player's LISTED position
    it would be filled for all thirteen who dressed, parse cleanly, and
    quietly change POS_BUCKET from "the five who started" to "everyone who
    played" — after dvp.py had been measured against the former.

    So the uncertainty is a refusal, not an assumption.
    """
    rows = _v3_rows(bench_value="G")  # everyone who dressed has a position
    frame = parse_starting_positions(_v3_payload(rows=rows), GAME)
    with pytest.raises(StartingPositionError) as exc:
        check_starting_position_semantics(frame)
    assert "LISTED position" in str(exc.value)
    assert "8" in str(exc.value)  # 5 starters + 3 bench, per team


def test_the_gate_refuses_a_team_game_short_of_five_and_says_so_differently():
    """Too few is missing rows, not fewer starters, and reads differently."""
    rows = [r for r in _v3_rows() if not (r[1] == int(HOME) and r[3] == "C")]
    frame = parse_starting_positions(_v3_payload(rows=rows), GAME)
    with pytest.raises(StartingPositionError) as exc:
        check_starting_position_semantics(frame)
    assert "rows are missing" in str(exc.value)
    assert "LISTED position" not in str(exc.value)


def test_an_empty_frame_is_not_a_passing_gate():
    """
    A gate that accepts nothing accepts everything. A pull that returned zero
    rows would otherwise sail through and write nothing while reporting
    success.
    """
    with pytest.raises(StartingPositionError, match="not a passing gate"):
        check_starting_position_semantics(
            pd.DataFrame(columns=list(STARTING_POSITION_COLUMNS))
        )


# --- refusals rather than silence ------------------------------------------

def test_a_response_with_no_player_table_is_named_not_empty():
    with pytest.raises(StartingPositionError, match="no PlayerStats data set"):
        parse_starting_positions({"TeamStats": {"headers": ["gameId"], "data": []}}, GAME)


def test_a_table_with_no_headers_is_named():
    with pytest.raises(StartingPositionError, match="carries no headers"):
        parse_starting_positions({"PlayerStats": {"headers": [], "data": []}}, GAME)


def test_a_table_matching_neither_version_is_refused_by_name():
    payload = {"PlayerStats": {"headers": ["gameId", "teamId", "personId"], "data": []}}
    with pytest.raises(StartingPositionError, match="neither v3's"):
        parse_starting_positions(payload, GAME)


def test_a_row_with_no_player_id_cannot_be_keyed_and_is_refused():
    payload = {
        "PlayerStats": {
            "headers": ["gameId", "teamId", "position"],
            "data": [[GAME, int(HOME), "G"]],
        }
    }
    with pytest.raises(StartingPositionError, match="neither v3's"):
        parse_starting_positions(payload, GAME)


# --- game ids -------------------------------------------------------------

def test_the_game_id_is_zero_padded_so_the_join_can_match():
    """
    The panel stores GAME_ID unpadded and the NBA returns it padded to ten.
    inactive_players.py records the same hazard: a naive merge matches
    nothing at all, silently.
    """
    out = parse_starting_positions(_v3_payload(game_id="22500123"), "22500123")
    assert set(out["GAME_ID"]) == {"0022500123"}


def test_a_payload_without_a_row_level_game_id_takes_it_from_the_request():
    """Filling the id from the request is safe; inventing a player id is not."""
    rows = [[int(HOME), 1001, "G"]]
    payload = {
        "PlayerStats": {
            "headers": ["teamId", "personId", "position"],
            "data": rows,
        }
    }
    out = parse_starting_positions(payload, GAME)
    assert set(out["GAME_ID"]) == {GAME}


# --- fetching -------------------------------------------------------------

def test_v3_is_tried_before_the_deprecated_v2():
    seen = []

    def fetch(game_id, endpoint):
        seen.append(endpoint)
        return _v3_payload(game_id)

    fetch_starting_positions(GAME, fetch=fetch)
    assert seen == ["boxscoretraditionalv3"]


def test_v2_is_the_fallback_when_v3_has_nothing():
    seen = []

    def fetch(game_id, endpoint):
        seen.append(endpoint)
        if endpoint == "boxscoretraditionalv3":
            raise RuntimeError("404")
        return _v2_payload(game_id)

    out = fetch_starting_positions(GAME, fetch=fetch)
    assert seen == ["boxscoretraditionalv3", "boxscoretraditionalv2"]
    assert out["STARTING_POSITION"].notna().sum() == 10


def test_both_versions_failing_names_both():
    def fetch(game_id, endpoint):
        raise RuntimeError("denied")

    with pytest.raises(StartingPositionError, match="boxscoretraditionalv3.*boxscoretraditionalv2"):
        fetch_starting_positions(GAME, fetch=fetch)


def test_partial_failures_are_returned_not_swallowed():
    """
    A caller that asked for 1,230 games and got 1,180 has to be able to tell,
    or a partial pull silently becomes a season where fifty games had no
    starters.
    """
    def fetch(game_id, endpoint):
        if game_id.endswith("002"):
            raise RuntimeError("gone")
        return _v3_payload(game_id)

    frame, failures = fetch_many_starting_positions(
        ["0022500001", "0022500002", "0022500003"], fetch=fetch, pause_seconds=0
    )
    assert len(failures) == 1 and failures[0]["game_id"] == "0022500002"
    assert set(frame["GAME_ID"]) == {"0022500001", "0022500003"}


def test_every_game_failing_raises_rather_than_returning_an_empty_frame():
    def fetch(game_id, endpoint):
        raise RuntimeError("denied")

    with pytest.raises(StartingPositionError, match="no starting positions could be fetched"):
        fetch_many_starting_positions(["0022500001"], fetch=fetch, pause_seconds=0)


# --- cache ----------------------------------------------------------------

def test_the_cache_round_trips_and_keeps_ids_as_strings(tmp_path):
    """
    An id read back as an int loses its zero padding, and the join it was
    padded for stops matching.
    """
    frame = parse_starting_positions(_v3_payload(), GAME)
    save_starting_positions(frame, "2025-26", root=tmp_path)
    back = load_cached_starting_positions("2025-26", root=tmp_path)
    assert back is not None
    assert back["GAME_ID"].dtype == "string"
    assert set(back["GAME_ID"]) == {GAME}
    assert back["STARTING_POSITION"].notna().sum() == 10


def test_an_unpulled_season_reads_as_none_not_as_an_empty_frame(tmp_path):
    """None means "not pulled"; an empty frame would mean "nobody started"."""
    assert load_cached_starting_positions("1999-00", root=tmp_path) is None
    assert not cache_path_for("1999-00", root=tmp_path).exists()


# --- the database writer ---------------------------------------------------
#
# `update_starting_positions` is exercised against a stub session rather than
# source-inspected. The parts worth testing ARE the matching, the counting and
# the refusal to insert, and `inspect.getsource(...) in` proves none of them.
# A stub works because the function uses plain `select` and ORM attribute
# assignment, with no Postgres-only `pg_insert` in its path.


class _StubSession:
    def __init__(self, rows):
        self._rows = rows
        self.statements = []

    def execute(self, statement):
        self.statements.append(statement)
        rows = self._rows

        class _Result:
            def scalars(self):
                class _S:
                    def all(self_inner):
                        return rows
                return _S()
        return _Result()


def _stub_scope(rows):
    import contextlib

    @contextlib.contextmanager
    def scope():
        yield _StubSession(rows)

    return scope


def _log(game, player, position=None):
    from src.db.models import PlayerGameLog

    row = PlayerGameLog()
    row.nba_game_id = game
    row.nba_player_id = player
    row.starting_position = position
    return row


def _run_writer(monkeypatch, rows, frame):
    from src.db import repository

    monkeypatch.setattr(repository, "session_scope", _stub_scope(rows))
    return repository.update_starting_positions(frame)


def _frame(pairs):
    return pd.DataFrame(
        [{"GAME_ID": g, "PLAYER_ID": p, "STARTING_POSITION": s} for g, p, s in pairs]
    )


def test_the_writer_fills_a_position_onto_an_existing_row(monkeypatch):
    rows = [_log(GAME, "1001"), _log(GAME, "1002")]
    out = _run_writer(monkeypatch, rows, _frame([(GAME, "1001", "C")]))
    assert out == {"matched": 1, "updated": 1, "unmatched": 0, "cleared": 0}
    assert rows[0].starting_position == "C"
    assert rows[1].starting_position is None


def test_the_writer_never_inserts_a_position_with_no_box_score_behind_it(monkeypatch):
    """
    THE DESIGN DECISION THIS PINS. A position with no game log is a row with a
    bucket and no statistics — enough to move an opponent's
    allowed-against-that-bucket average while contributing nothing to it. It
    is counted, not created, and not dropped silently either.
    """
    rows = [_log(GAME, "1001")]
    out = _run_writer(
        monkeypatch, rows, _frame([(GAME, "1001", "G"), (GAME, "9999", "C")])
    )
    assert out["matched"] == 1
    assert out["unmatched"] == 1
    assert len(rows) == 1, "a row was inserted for a player with no game log"


def test_a_row_already_carrying_the_right_position_is_not_counted_as_updated(monkeypatch):
    """Re-running the pull must not report work it did not do."""
    rows = [_log(GAME, "1001", "F")]
    out = _run_writer(monkeypatch, rows, _frame([(GAME, "1001", "F")]))
    assert out == {"matched": 1, "updated": 0, "unmatched": 0, "cleared": 0}


def test_a_correction_to_null_is_permitted_and_counted_separately(monkeypatch):
    """
    A corrected payload that moves a player from starter to bench has to be
    able to say so. A writer that could only ever fill would make that
    uncorrectable, and `cleared` is reported separately because clearing a
    position is not the same event as setting one.
    """
    rows = [_log(GAME, "1001", "G")]
    out = _run_writer(monkeypatch, rows, _frame([(GAME, "1001", None)]))
    assert out["cleared"] == 1 and out["updated"] == 1
    assert rows[0].starting_position is None


def test_the_writer_normalises_rather_than_trusting_its_caller(monkeypatch):
    """
    migration 008's CHECK admits only G, F and C. A listed position reaching
    this column would change what POS_BUCKET means downstream, so the
    normaliser runs here too and not only in the ingest.
    """
    rows = [_log(GAME, "1001"), _log(GAME, "1002")]
    out = _run_writer(
        monkeypatch, rows, _frame([(GAME, "1001", "PG"), (GAME, "1002", "WING")])
    )
    assert rows[0].starting_position == "G", "a guard spelt PG was rejected outright"
    assert rows[1].starting_position is None, "an unknown spelling was guessed"
    assert out["updated"] == 1


def test_an_empty_frame_is_refused_rather_than_reported_as_a_successful_write(monkeypatch):
    out = _run_writer(monkeypatch, [_log(GAME, "1001")], pd.DataFrame())
    assert out == {"matched": 0, "updated": 0, "unmatched": 0, "cleared": 0}


def test_a_frame_missing_a_required_column_is_named(monkeypatch):
    from src.db import repository

    monkeypatch.setattr(repository, "session_scope", _stub_scope([]))
    with pytest.raises(ValueError, match="STARTING_POSITION"):
        repository.update_starting_positions(
            pd.DataFrame([{"GAME_ID": GAME, "PLAYER_ID": "1001"}])
        )


# --- the column reaches the panel the live path builds from ----------------

def test_the_upsert_carries_starting_position_and_updates_it_on_conflict():
    """
    BOTH HALVES, because only one of them was true for `pf`. It was in the
    insert payload and NOT in the ON CONFLICT set, so a season ingested before
    the column existed kept it NULL forever and re-ingesting could not fix it.
    The docstring said "re-ingesting a season corrects rows". Found while
    adding this column by the same route.
    """
    import inspect

    from src.db import repository

    source = inspect.getsource(repository.upsert_player_game_logs)
    assert '"starting_position": _normalised_start_positions(' in source
    conflict = source.split("on_conflict_do_update")[1]
    assert '"starting_position"' in conflict
    assert '"pf"' in conflict, "pf is written but still never updated on conflict"


def test_the_loaded_panel_exposes_starting_position_so_dvp_reaches_the_live_path():
    """
    A layer that only runs on a rebuilt archive panel is a research column.
    repository.load_player_panel is what the live path builds features from,
    so the position has to come back out of the table it was written to.
    """
    import inspect

    from src.db import repository

    source = inspect.getsource(repository.load_player_panel)
    assert '"STARTING_POSITION": r.starting_position' in source


def test_the_model_declares_the_column_the_migration_adds():
    from src.db.models import PlayerGameLog

    column = PlayerGameLog.__table__.columns["starting_position"]
    assert column.nullable, "a bench appearance has no position to record"
    assert column.type.length == 1, "a one-letter bucket; 'PG' must not fit"


def test_the_migration_admits_only_the_three_buckets():
    from pathlib import Path

    sql = Path("migrations/008_player_game_log_starting_position.sql").read_text()
    assert "starting_position VARCHAR(1)" in sql
    assert "IN ('G', 'F', 'C')" in sql
    assert "NOT VALID" in sql, "an existing table must not be rewritten"
    assert "DEFAULT" not in sql.split("ALTER TABLE")[1], "a default would be a fabrication"
