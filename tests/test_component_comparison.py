"""
Scoring the serving component against its shadows on the same graded rows.

WHY THIS CAN BE GOT WRONG QUIETLY. The join is between two stores that spell
a date differently, the outcome has to be derived without leaning on which
side the SERVED component took, and a component that abstains more will look
better unless the comparison is paired. Each of those failures produces a
confident number rather than an error.

RESEARCH_ONLY. Every row here is synthetic.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from src.settlement.component_comparison import (
    MIN_ROWS_FOR_A_VERDICT,
    SERVED_COMPONENT,
    compare_components,
    over_hit,
    row_key,
)


def _graded(n, *, start=0, actual=None, line=20.5, market="PTS", **over):
    rows = []
    for i in range(start, start + n):
        row = {
            "outcome_status": "WIN", "market": market,
            "game_date": dt.date(2026, 11, 9), "player_name": f"DEMO_{i}",
            "nba_player_id": str(i), "predicted_side": "OVER",
            "predicted_line": line,
            "actual_result": (line + 1.0) if actual is None else actual,
            "did_not_play": False,
        }
        row.update(over)
        rows.append(row)
    return rows


def _records(n, *, start=0, served=0.6, shadow=0.6, market="PTS", line=20.5,
             shadow_name="line_aware", shadow_from=0):
    rows = []
    for i in range(start, start + n):
        row = {
            "decision": "RECORDED", "player_name": f"DEMO_{i}", "market": market,
            # The shape JSON gives a pandas Timestamp, which is NOT the shape
            # the ledger's `date` gives. The join has to survive that.
            "game_date": "2026-11-09 00:00:00",
            "predicted_line": line, "prob_over": served,
        }
        if i >= shadow_from:
            row["shadow"] = {shadow_name: shadow}
        rows.append(row)
    return rows


# --- the outcome -----------------------------------------------------------

@pytest.mark.parametrize("actual,line,expected", [
    (25.0, 20.5, 1.0),
    (18.0, 20.5, 0.0),
    (20.0, 20.0, None),      # push
    (None, 20.5, None),
    (25.0, None, None),
    (float("nan"), 20.5, None),
])
def test_over_hit_is_derived_from_the_stat_and_the_line(actual, line, expected):
    """
    NOT from predicted_side and outcome_status. Those encode the same fact
    through the recorder's side choice, so reading the outcome back out of
    them would make a shadow's score depend on which side the SERVED
    component happened to take.
    """
    assert over_hit(actual, line) == expected


def test_a_push_is_excluded_rather_than_scored_either_way():
    report = compare_components(
        _graded(4, actual=20.0, line=20.0), _records(4), as_of="t",
    )
    assert report["status"] == "DATA_NOT_AVAILABLE"
    assert report["excluded_by_reason"]["push or missing result/line"] == 4


def test_a_scratch_is_not_a_forecast_error():
    report = compare_components(
        _graded(4, did_not_play=True), _records(4), as_of="t",
    )
    assert "did not play (a scratch is not a forecast error)" in report["excluded_by_reason"]


# --- the join --------------------------------------------------------------

def test_the_key_survives_two_spellings_of_the_same_date():
    """
    A record carries whatever JSON made of a Timestamp; the ledger carries a
    `date`. Normalising in two places is how a join silently matches nothing
    and reports it as "no graded rows yet".
    """
    assert row_key("DEMO_0", "pts", "2026-11-09 00:00:00") == \
           row_key("DEMO_0", "PTS", dt.date(2026, 11, 9))


def test_a_graded_row_with_no_record_is_counted_not_dropped_silently():
    report = compare_components(_graded(3), [], as_of="t")
    assert report["status"] == "DATA_NOT_AVAILABLE"
    assert report["excluded_by_reason"]["no decision record for this graded row"] == 3


def test_both_components_are_scored_on_the_joined_rows():
    report = compare_components(_graded(10), _records(10), as_of="t")
    assert report["status"] == "OK"
    assert report["matched_rows"] == 10
    assert set(report["components"]) == {SERVED_COMPONENT, "line_aware"}
    assert report["per_component"][SERVED_COMPONENT]["n"] == 10
    assert report["per_component"]["line_aware"]["n"] == 10


# --- the paired set --------------------------------------------------------

def test_a_component_that_abstains_more_does_not_get_an_easier_row_set():
    """
    THE TRAP THIS REPORT EXISTS TO AVOID. Scoring A on 10 rows and B on 4 and
    comparing the two numbers rewards abstention. The head-to-head is the
    subset every component answered; the per-component totals sit beside it so
    the abstention is visible rather than hidden.
    """
    report = compare_components(
        _graded(10), _records(10, shadow_from=6), as_of="t",
    )
    assert report["per_component"][SERVED_COMPONENT]["answered"] == 10
    assert report["per_component"]["line_aware"]["answered"] == 4
    assert report["per_component"]["line_aware"]["abstained"] == 6
    assert report["paired_rows"] == 4
    assert report["head_to_head"][SERVED_COMPONENT]["n"] == 4
    assert report["head_to_head"]["line_aware"]["n"] == 4


# --- what it will and will not say -----------------------------------------

def test_below_the_evidence_floor_no_component_is_called_better():
    report = compare_components(_graded(10), _records(10, shadow=0.9), as_of="t")
    assert report["paired_rows"] < MIN_ROWS_FOR_A_VERDICT
    joined = " ".join(report["findings"])
    assert "below the" in joined and "no component is called better" in joined


def test_above_the_floor_the_lower_brier_is_named():
    n = MIN_ROWS_FOR_A_VERDICT + 10
    # Every over hits, so the component saying 0.95 is the honest one.
    report = compare_components(
        _graded(n), _records(n, served=0.55, shadow=0.95), as_of="t",
    )
    assert report["paired_rows"] == n
    h2h = report["head_to_head"]
    assert h2h["line_aware"]["brier"] < h2h[SERVED_COMPONENT]["brier"]
    assert "line_aware has the lower Brier score" in " ".join(report["findings"])


def test_a_gap_too_small_to_act_on_is_called_a_tie():
    n = MIN_ROWS_FOR_A_VERDICT + 10
    report = compare_components(
        _graded(n), _records(n, served=0.80, shadow=0.8005), as_of="t",
    )
    assert "Treat these as tied" in " ".join(report["findings"])


def test_one_component_alone_is_not_a_comparison():
    rows = _records(5)
    for row in rows:
        row.pop("shadow", None)
    report = compare_components(_graded(5), rows, as_of="t")
    assert "nothing to compare" in " ".join(report["findings"])


def test_it_never_claims_profit():
    n = MIN_ROWS_FOR_A_VERDICT + 10
    report = compare_components(
        _graded(n), _records(n, served=0.55, shadow=0.95), as_of="t",
    )
    joined = (" ".join(report["findings"]) + report["note"]).lower()
    assert "not a profit figure" in joined
    assert "not the decision" in joined


# --- the domain ------------------------------------------------------------

def test_an_unlikely_forecast_is_scored_not_excluded():
    """
    These are P(over) over the FULL [0, 1] range, not the taken-side
    probabilities `settlement.calibration` bins over (0.5, 1.0]. Reusing that
    domain here would silently drop every row a component thought unlikely --
    exactly the rows where the two disagree most.
    """
    report = compare_components(
        _graded(20), _records(20, served=0.2, shadow=0.8), as_of="t",
    )
    assert report["per_component"][SERVED_COMPONENT]["n"] == 20, (
        "a sub-0.5 probability was excluded; the comparison is scoring only "
        "the rows the model was confident about"
    )
    assert report["per_component"][SERVED_COMPONENT]["mean_predicted"] == pytest.approx(0.2)


# --- reading the records ---------------------------------------------------

def test_records_are_read_from_the_layout_that_wrote_them(tmp_path):
    from src.settlement.decision_record import (
        build_record,
        read_records,
        recorded_rows_from,
        write_record,
    )

    for day in ("2026-11-09", "2026-11-10"):
        body, _ = build_record(
            slate_date=day, as_of_ts=f"{day}T09:05:00-08:00",
            rows=[{"decision": "RECORDED", "player_name": "X", "market": "PTS"}],
            prev_sha256=None,
        )
        write_record(tmp_path, body)

    assert len(read_records(tmp_path)) == 2
    assert len(recorded_rows_from(read_records(tmp_path))) == 2
    assert len(read_records(tmp_path, since="2026-11-10")) == 1


def test_one_corrupt_record_does_not_hide_the_others(tmp_path):
    from src.settlement.decision_record import build_record, read_records, write_record

    body, _ = build_record(
        slate_date="2026-11-09", as_of_ts="2026-11-09T09:05:00-08:00",
        rows=[{"decision": "RECORDED", "player_name": "X", "market": "PTS"}],
        prev_sha256=None,
    )
    write_record(tmp_path, body)
    broken = tmp_path / "decision_records" / "2026" / "11" / "10"
    broken.mkdir(parents=True)
    (broken / "20261110T170500Z.json").write_text("{not json", encoding="utf-8")

    read = read_records(tmp_path)
    assert len(read) == 1, "a corrupt file hid a night of evidence"
    assert json.dumps(read[0])  # still a usable record


def test_no_records_at_all_is_not_a_crash(tmp_path):
    from src.settlement.decision_record import read_records

    assert read_records(tmp_path) == []


# --- the script ------------------------------------------------------------

def test_an_unreadable_ledger_exits_two_not_zero(monkeypatch):
    """
    A non-zero exit must mean "this did not run". An empty ledger reports
    itself and exits 0; a database fault must not look the same.
    """
    from scripts import compare_components as script

    def boom(**kwargs):
        raise RuntimeError("no database")

    monkeypatch.setattr("src.db.repository.load_graded_prop_residual_rows", boom)
    assert script.main(["--days", "1"]) == script.EXIT_LEDGER_UNREADABLE


def test_an_empty_ledger_reports_itself_and_exits_zero(monkeypatch, tmp_path, capsys):
    from scripts import compare_components as script

    monkeypatch.setattr(
        "src.db.repository.load_graded_prop_residual_rows", lambda **k: []
    )
    monkeypatch.setattr(
        "src.utils.volume.resolve_state_root", lambda: (tmp_path, "test")
    )
    assert script.main(["--days", "1"]) == script.EXIT_OK
    out = capsys.readouterr().out
    assert "DATA_NOT_AVAILABLE" in out
    assert "RESEARCH_ONLY" in out
