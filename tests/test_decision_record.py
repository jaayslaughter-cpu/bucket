"""
The chained record of every slate decision, abstentions included.

WHAT THIS EXISTS TO PREVENT. `prop_results` holds only the rows that cleared
the recorder. On 2026-11-08 production recorded 0 of 796 projections, so that
table says nothing happened on a day the pipeline scored a full slate. Any
strike rate or calibration figure built on it has a denominator nobody outside
the process can see, and nothing stops the rows being edited afterwards.

These tests pin three properties: every projection the recorder saw appears,
skipped ones carry the recorder's OWN reason, and the digest is derived from
content so an edited record cannot present the hash it used to have.

RESEARCH_ONLY. Fixtures are synthetic; no outcome here is a real result.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.settlement.decision_record import (
    artifact_provenance,
    build_record,
    canonical_bytes,
    decision_rows,
    latest_sha256,
    record_path,
    sha256_of,
    utc_stamp,
    write_record,
)
from src.settlement.recorder import pending_prop_result_rows

AS_OF = "2026-11-09T09:05:00-08:00"
SLATE = "2026-11-09"


def _projection(**overrides) -> dict:
    row = {
        "PLAYER_NAME": "DEMO_A", "PLAYER_ID": "203999", "GAME_ID": "0022500123",
        "GAME_DATE": pd.Timestamp("2026-11-09"), "MARKET": "PTS",
        "LINE": 25.5, "PROB_OVER": 0.62, "FINAL_PROJECTION": 27.1,
    }
    row.update(overrides)
    return row


def _lines(names=("DEMO_A",)) -> pd.DataFrame:
    return pd.DataFrame([{
        "player_name": n, "market": "PTS", "line": 25.5, "source": "propline",
        "over_odds_american": -115, "under_odds_american": -105,
        "is_pickem": False, "payout_multiplier": None,
        "nba_game_id": "0022500123", "nba_player_id": "203999",
    } for n in names])


def _record(rows, prev=None, **kw):
    return build_record(slate_date=SLATE, as_of_ts=AS_OF, rows=rows,
                        prev_sha256=prev, **kw)


# --- the denominator ------------------------------------------------------

def test_a_slate_that_recorded_nothing_still_records_every_decision():
    """
    THE 2026-11-08 CASE. The recorder took 0 rows and skipped every one; the
    ledger was silent about a full slate. A record that only carried `rows`
    would reproduce exactly that silence.
    """
    frame = pd.DataFrame([
        _projection(PLAYER_NAME="DEMO_A", PROB_OVER=None),
        _projection(PLAYER_NAME="DEMO_B", LINE=None),
    ])
    report = pending_prop_result_rows(frame, _lines(("DEMO_A", "DEMO_B")), run_id="r")
    assert not report.rows, "fixture should skip both, or it tests nothing"

    body, _ = _record(decision_rows(report))

    assert body["counts"]["total"] == 2, (
        "a projection the recorder saw is missing from the record, so the "
        "denominator behind any later figure is unverifiable"
    )
    assert body["counts"]["by_decision"] == {"SKIPPED": 2}
    assert set(body["counts"]["skipped_by_reason"]) == {
        "no posted line", "no model probability",
    }


def test_the_skip_reason_is_the_recorders_own_words():
    """
    ONE TAXONOMY. Re-bucketing the recorder's reasons here would be a second
    classification of one fact, and the two would drift the first time either
    changed. The string is carried verbatim.
    """
    frame = pd.DataFrame([_projection(PROB_OVER=None)])
    report = pending_prop_result_rows(frame, _lines(), run_id="r")
    verbatim = report.skipped[0]["reason"]

    rows = decision_rows(report)
    assert rows[0]["reason"] == verbatim
    assert rows[0]["player_name"] == "DEMO_A", (
        "the skip entry lost the player; the record would name no one"
    )
    assert rows[0]["market"] == "PTS"


def test_a_recorded_row_carries_the_prediction_it_made():
    frame = pd.DataFrame([_projection()])
    report = pending_prop_result_rows(frame, _lines(), run_id="r")
    assert report.rows, "fixture should record one row"

    row = decision_rows(report)[0]
    assert row["decision"] == "RECORDED"
    assert row["predicted_side"] == "OVER"
    assert row["predicted_line"] == pytest.approx(25.5)
    assert row["prob_over"] == pytest.approx(0.62)
    # NOT A WAGER. A stake has no place in this record and the recorder cannot
    # write one; asserting its absence keeps that true.
    assert "stake_units" not in row and "stake" not in row


# --- the chain ------------------------------------------------------------

def test_the_digest_does_not_depend_on_key_order():
    """
    A chain whose hash moved with dict insertion order would break on a Python
    upgrade rather than on a tampered record -- the one thing it must not do.
    """
    a = {"decision": "RECORDED", "player_name": "X", "market": "PTS"}
    b = {"market": "PTS", "player_name": "X", "decision": "RECORDED"}
    assert sha256_of(_record([a])[0]) == sha256_of(_record([b])[0])


def test_changing_one_probability_changes_the_digest():
    base = [{"decision": "RECORDED", "player_name": "X", "market": "PTS",
             "prob_over": 0.62}]
    edited = [{**base[0], "prob_over": 0.63}]
    assert _record(base)[1] != _record(edited)[1]


def test_a_record_edited_on_disk_no_longer_matches_what_followed_it(tmp_path):
    """
    THE WHOLE POINT. Record B is chained to A's digest. Rewriting A afterwards
    -- the move a ledger owner would make to improve a historic result -- must
    leave A's new digest disagreeing with the `prev_sha256` B already carries.
    """
    body_a, sha_a = _record([{"decision": "RECORDED", "player_name": "X",
                              "market": "PTS", "prob_over": 0.62}])
    write_record(tmp_path, body_a)

    body_b, _ = build_record(
        slate_date="2026-11-10", as_of_ts="2026-11-10T09:05:00-08:00",
        rows=[{"decision": "RECORDED", "player_name": "Y", "market": "REB"}],
        prev_sha256=latest_sha256(tmp_path),
    )
    write_record(tmp_path, body_b)
    assert body_b["prev_sha256"] == sha_a

    path_a = record_path(tmp_path, slate_date=SLATE, as_of_ts=AS_OF)
    tampered = json.loads(path_a.read_text())
    tampered["rows"][0]["prob_over"] = 0.99
    path_a.write_bytes(canonical_bytes(tampered))

    assert sha256_of(tampered) != body_b["prev_sha256"], (
        "a rewritten record still matches the digest the next one recorded, "
        "so the chain proves nothing"
    )


def test_the_chain_head_is_rehashed_from_content_not_a_stored_field(tmp_path):
    """
    `latest_sha256` must DERIVE the digest. Trusting a value stored inside the
    file would let an edited record hand back the hash it used to have.
    """
    body, sha = _record([{"decision": "RECORDED", "player_name": "X", "market": "PTS"}])
    path = write_record(tmp_path, body)
    assert latest_sha256(tmp_path) == sha

    edited = json.loads(path.read_text())
    edited["rows"][0]["market"] = "REB"
    path.write_bytes(canonical_bytes(edited))

    assert latest_sha256(tmp_path) != sha
    assert latest_sha256(tmp_path) == sha256_of(edited)


def test_an_empty_chain_starts_rather_than_failing(tmp_path):
    assert latest_sha256(tmp_path) is None
    body, _ = _record([], prev=latest_sha256(tmp_path))
    assert body["prev_sha256"] is None


def test_writing_is_atomic_and_leaves_no_partial_file(tmp_path):
    body, _ = _record([{"decision": "RECORDED", "player_name": "X", "market": "PTS"}])
    write_record(tmp_path, body)
    assert not list(tmp_path.rglob("*.tmp")), "a temp file survived the write"
    assert len(list(tmp_path.rglob("*.json"))) == 1


# --- provenance -----------------------------------------------------------

def test_provenance_identifies_the_artifact_without_publishing_its_path(tmp_path):
    """
    These records are meant to be publishable. The digest identifies the
    generation; the directory it sat in identifies the container and buys
    nothing.
    """
    art = tmp_path / "deep" / "nested" / "distribution_PTS.json"
    art.parent.mkdir(parents=True)
    art.write_bytes(b'{"dispersion": 1.0}')

    prov = artifact_provenance({"PTS": str(art)})
    assert prov["PTS"]["artifact"] == "distribution_PTS.json"
    # The digest of the FILE'S BYTES, computed independently here. `len(...) == 64`
    # would pass for any hex string and prove nothing about what was hashed.
    assert prov["PTS"]["sha256"] == hashlib.sha256(b'{"dispersion": 1.0}').hexdigest()
    assert prov["PTS"]["bytes"] == 19
    assert "deep" not in json.dumps(prov), "the record publishes the container layout"
    assert "nested" not in json.dumps(prov)


def test_an_unreadable_artifact_is_reported_rather_than_dropped(tmp_path):
    """
    A market silently missing from provenance reads as "not scored", which is
    a different and far more comfortable claim than "scored by something we
    could not identify".
    """
    prov = artifact_provenance({"PTS": str(tmp_path / "gone.json"), "REB": None})
    assert prov["PTS"]["sha256"] is None and prov["PTS"]["error"]
    assert prov["REB"]["error"] == "no artifact recorded"


def test_two_generations_of_the_same_filename_do_not_look_identical(tmp_path):
    art = tmp_path / "distribution_PTS.json"
    art.write_bytes(b'{"dispersion": 1.0}')
    first = artifact_provenance({"PTS": str(art)})["PTS"]["sha256"]
    art.write_bytes(b'{"dispersion": 2.0}')
    second = artifact_provenance({"PTS": str(art)})["PTS"]["sha256"]
    assert first != second, (
        "a refit under the same filename is indistinguishable, so calibration "
        "still cannot be bound to the generation that produced it"
    )


# --- it does not become a tip sheet ---------------------------------------

def test_the_record_states_what_it_is_not():
    body, _ = _record([], gate={"status": "PUBLISH_WITHHELD"})
    note = body["note"].lower()
    assert "not wagers" in note and "stake" in note
    assert "not evidence of profit" in note
    assert body["gate"] == {"status": "PUBLISH_WITHHELD"}


def test_the_report_object_may_be_empty_without_raising():
    body, digest = _record(decision_rows(SimpleNamespace(rows=[], skipped=[])))
    assert body["counts"]["total"] == 0 and len(digest) == 64


def test_the_chain_head_is_the_newest_record_not_the_first(tmp_path):
    """
    A MUTATION SURVIVED THE FIRST VERSION of these tests: taking `files[0]`
    instead of `files[-1]` passed, because the chain test only ever had one
    record on disk when it asked for the head. With three, grafting onto the
    oldest would silently fork the chain at its root.
    """
    digests = []
    for day, hour in (("2026-11-09", "09"), ("2026-11-10", "09"), ("2026-11-11", "09")):
        body, sha = build_record(
            slate_date=day, as_of_ts=f"{day}T{hour}:05:00-08:00",
            rows=[{"decision": "RECORDED", "player_name": "X", "market": "PTS"}],
            prev_sha256=latest_sha256(tmp_path),
        )
        write_record(tmp_path, body)
        digests.append(sha)

    assert len(list(tmp_path.rglob("*.json"))) == 3
    assert latest_sha256(tmp_path) == digests[-1], (
        "the chain head is not the newest record, so the next slate would "
        "graft onto an earlier one and fork the chain"
    )


def test_the_filename_sorts_by_time_across_a_daylight_saving_change():
    """
    Pacific local stamps do not sort chronologically on the November change:
    01:30-07:00 happens BEFORE 01:15-08:00 but sorts after it. The chain finds
    its head by sorting filenames, so the name is UTC.
    """
    assert utc_stamp("2026-11-01T01:30:00-07:00") == "20261101T083000Z"

    # THROUGH `record_path`, not `utc_stamp` directly: asserting the helper
    # alone let a mutation that stopped CALLING it pass untouched. What the
    # chain sorts is the path.
    root = Path("/state")
    earlier = record_path(root, slate_date="2026-11-01", as_of_ts="2026-11-01T01:30:00-07:00")
    later = record_path(root, slate_date="2026-11-01", as_of_ts="2026-11-01T01:15:00-08:00")
    assert str(earlier) < str(later), (
        "the record written second sorts first, so the chain would graft onto "
        "the wrong parent for one hour a year"
    )


def test_the_slate_actually_writes_a_record():
    """
    AST-walked inside `main.main`, because a pure module nothing calls is a
    module that records nothing. The 2026-11-08 silence this exists to fix was
    not a bug in a writer -- there was no writer.

    Asserted as a call INSIDE `main`, not as an import at the top of the file:
    an import survives the call site being deleted, which is exactly the shape
    of regression that would restore the silence.
    """
    import ast

    tree = ast.parse(Path("main.py").read_text(encoding="utf-8"))
    fn = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.FunctionDef) and n.name == "main"), None,
    )
    assert fn is not None, "main.main has been renamed; this guard needs updating"

    called = {
        node.func.id for node in ast.walk(fn)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    missing = {"build_record", "write_record", "decision_rows"} - called
    assert not missing, (
        f"main() no longer calls {sorted(missing)}, so a slate's decisions "
        f"leave no chained record"
    )


def test_the_record_is_chained_to_whatever_came_before():
    """
    `prev_sha256=latest_sha256(...)` is what makes it a chain rather than a
    pile of files. A record written with `prev_sha256=None` every time is
    individually signable and collectively worthless -- any one of them could
    be deleted without a trace.
    """
    import ast

    tree = ast.parse(Path("main.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "id", "") == "build_record"):
            continue
        prev = next((k for k in node.keywords if k.arg == "prev_sha256"), None)
        assert prev is not None, "build_record is called without prev_sha256"
        assert isinstance(prev.value, ast.Call) and (
            getattr(prev.value.func, "id", "") == "latest_sha256"
        ), "prev_sha256 is not read from the chain head, so the records do not chain"
        return
    raise AssertionError("no build_record call found in main.py")
