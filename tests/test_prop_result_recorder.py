"""The prop_results writer: predictions recorded for forward grading.

prop_results had a table, a grader that selects PENDING rows, and a metrics
layer aggregating W/L/PUSH, stake, profit and both CLV columns — and no writer
anywhere, so every figure it could produce was an aggregate over zero rows.

Two properties carry these tests. Nothing that could produce a wrong ledger
entry is guessed, and nothing here writes a stake.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.settlement.recorder import pending_prop_result_rows


def projections(**overrides) -> pd.DataFrame:
    row = {
        "PLAYER_NAME": "DEMO_A",
        "PLAYER_ID": "203999",
        "GAME_ID": "0022500123",
        "GAME_DATE": pd.Timestamp("2026-01-15"),
        "MARKET": "PTS",
        "LINE": 25.5,
        "PROB_OVER": 0.62,
        "FINAL_PROJECTION": 27.1,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def lines(**overrides) -> pd.DataFrame:
    row = {
        "player_name": "DEMO_A",
        "market": "PTS",
        "line": 25.5,
        "source": "propline",
        "over_odds_american": -115,
        "under_odds_american": -105,
        "is_pickem": False,
        "payout_multiplier": None,
        "nba_game_id": "0022500123",
        "nba_player_id": "203999",
    }
    row.update(overrides)
    return pd.DataFrame([row])


# --- the row it builds ---------------------------------------------------

def test_a_complete_projection_becomes_one_pending_row():
    report = pending_prop_result_rows(projections(), lines(), run_id="run-1")
    assert len(report.rows) == 1
    row = report.rows[0]
    assert row["outcome_status"] == "PENDING"
    assert row["predicted_side"] == "OVER"
    assert row["predicted_line"] == pytest.approx(25.5)
    assert row["odds"] == -115
    assert row["source"] == "propline"
    assert row["run_id"] == "run-1"
    assert row["actual_result"] is None


def test_the_under_side_takes_the_under_price():
    report = pending_prop_result_rows(projections(PROB_OVER=0.38), lines())
    row = report.rows[0]
    assert row["predicted_side"] == "UNDER"
    assert row["odds"] == -105, "took the over price for an under prediction"


def test_prob_over_stays_prob_over_on_an_under_row():
    """
    The COLUMN means P(over). Storing the taken side's probability would flip
    its meaning for every UNDER row, and the calibration layer reads it.
    """
    report = pending_prop_result_rows(projections(PROB_OVER=0.38), lines())
    assert report.rows[0]["prob_over"] == pytest.approx(0.38)


def test_no_stake_is_ever_written():
    """These are predictions. PropIQ does not place or size wagers."""
    report = pending_prop_result_rows(projections(), lines())
    assert "stake_units" not in report.rows[0]
    assert "profit_units" not in report.rows[0]


def test_a_pickem_row_keeps_its_multiplier_and_has_no_odds():
    report = pending_prop_result_rows(
        projections(),
        lines(is_pickem=True, payout_multiplier=3.0,
              over_odds_american=None, under_odds_american=None),
    )
    row = report.rows[0]
    assert row["is_pickem"] is True
    assert row["payout_multiplier"] == pytest.approx(3.0)
    assert row["odds"] is None, "a payout multiplier is not a price"


# --- what it refuses to guess -------------------------------------------

def test_a_projection_with_no_line_is_skipped():
    report = pending_prop_result_rows(projections(LINE=None), lines())
    assert report.rows == []
    assert report.skipped_by_reason == {"no posted line": 1}


def test_a_projection_with_no_probability_is_skipped():
    """PROB_OVER is written only for the market the model was trained for."""
    report = pending_prop_result_rows(projections(PROB_OVER=None), lines())
    assert report.rows == []
    assert "no model probability" in report.skipped_by_reason


def test_an_exact_coin_flip_predicts_no_side():
    """Rounding 0.50 to OVER would put a coin flip in the ledger as a call."""
    report = pending_prop_result_rows(projections(PROB_OVER=0.5), lines())
    assert report.rows == []
    assert any("exactly 0.50" in r for r in report.skipped_by_reason)


def test_a_row_with_no_source_is_skipped_because_the_unique_key_needs_one():
    """
    Postgres treats NULLs in a unique index as distinct, so a source-less row
    conflicts with nothing and every re-run inserts it again. An unbounded
    duplicate is worse than a missing row.
    """
    report = pending_prop_result_rows(projections(), lines(source=None))
    assert report.rows == []
    assert any("unique key" in r for r in report.skipped_by_reason)


def test_a_row_with_no_game_id_is_skipped():
    report = pending_prop_result_rows(
        projections(GAME_ID=None), lines(nba_game_id=None),
    )
    assert report.rows == []
    assert any("grader could never match" in r for r in report.skipped_by_reason)


def test_a_projection_with_no_matching_line_row_is_skipped_not_invented():
    """An unmatched player has no source, so there is nothing to record."""
    report = pending_prop_result_rows(projections(), lines(player_name="SOMEONE_ELSE"))
    assert report.rows == []


def test_the_join_is_exact_and_never_fuzzy():
    """
    A near-miss name must not attach another player's price. In a settlement
    ledger that is not a near miss, it is a wrong record.
    """
    report = pending_prop_result_rows(projections(), lines(player_name="demo_a"))
    assert report.rows == []


@pytest.mark.parametrize("bad", [1.4, -0.2])
def test_a_probability_outside_zero_one_is_skipped(bad):
    report = pending_prop_result_rows(projections(PROB_OVER=bad), lines())
    assert report.rows == []
    assert any("outside [0, 1]" in r for r in report.skipped_by_reason)


def test_every_skip_is_counted_and_reported():
    """A recorder that silently wrote fewer rows would repeat the defect."""
    frame = pd.concat([
        projections(),
        projections(PLAYER_NAME="DEMO_B", LINE=None),
        projections(PLAYER_NAME="DEMO_C", PROB_OVER=None),
    ], ignore_index=True)
    board = pd.concat([
        lines(),
        lines(player_name="DEMO_B"),
        lines(player_name="DEMO_C"),
    ], ignore_index=True)

    report = pending_prop_result_rows(frame, board)
    assert len(report.rows) == 1
    assert len(report.skipped) == 2
    payload = report.as_dict()
    assert payload["rows_built"] == 1
    assert payload["rows_skipped"] == 2
    assert "not wagers" in payload["note"]


def test_an_empty_frame_is_not_an_error():
    assert pending_prop_result_rows(pd.DataFrame()).rows == []
    assert pending_prop_result_rows(projections(), None).rows == []


def test_a_board_missing_its_join_keys_records_nothing():
    board = pd.DataFrame([{"line": 25.5, "source": "propline"}])
    report = pending_prop_result_rows(projections(), board)
    assert report.rows == []


def test_the_most_recent_capture_of_a_line_wins():
    board = pd.concat([
        lines(over_odds_american=-150),
        lines(over_odds_american=-115),
    ], ignore_index=True)
    report = pending_prop_result_rows(projections(), board)
    assert report.rows[0]["odds"] == -115


# --- the upsert guard ----------------------------------------------------

def test_the_upsert_never_un_settles_a_graded_row():
    """
    ON CONFLICT DO UPDATE without a WHERE would overwrite a graded row's
    pre-settlement fields on the next run of the same slate, rewriting history
    in the one table whose job is to record what was predicted beforehand.
    """
    from sqlalchemy.dialects import postgresql

    from src.db.repository import PROP_RESULT_REFRESHABLE, pending_prop_result_statement

    report = pending_prop_result_rows(projections(), lines(), run_id="run-1")
    sql = str(
        pending_prop_result_statement(report.rows).compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": True},
        )
    )
    assert "ON CONFLICT ON CONSTRAINT uq_prop_result DO UPDATE" in sql
    assert "WHERE prop_results.outcome_status = 'PENDING'" in sql

    # The grader's own output columns must never be in the refresh set.
    for graded in (
        "outcome_status", "actual_result", "profit_units", "stake_units",
        "settled_at_utc", "clv_prob_points", "did_not_play",
    ):
        assert graded not in PROP_RESULT_REFRESHABLE


def test_the_writer_sends_nothing_when_there_is_nothing_to_send():
    from src.db.repository import record_pending_prop_results

    assert record_pending_prop_results([]) == 0


# --- the seam between the assembler and this reader ----------------------
#
# The defect test_projection_roundtrip.py exists for, one module along: the
# assembler writes a DataFrame and this reads it by string key, so a renamed
# column produces silent nulls rather than an error. Same guard, new reader.

def _keys_read_from(function_name: str, module_path: str, variable: str) -> set[str]:
    """Every ``<variable>.get("KEY")`` inside the named function."""
    import ast
    from pathlib import Path

    source = (Path(__file__).parent.parent / module_path).read_text()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            keys = set()
            for sub in ast.walk(node):
                if (
                    isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Attribute)
                    and sub.func.attr == "get"
                    and isinstance(sub.func.value, ast.Name)
                    and sub.func.value.id == variable
                    and sub.args
                    and isinstance(sub.args[0], ast.Constant)
                ):
                    keys.add(sub.args[0].value)
            return keys
    raise AssertionError(f"{function_name} not found in {module_path}")


def test_every_projection_key_the_recorder_reads_is_one_the_assembler_writes():
    from test_projection_roundtrip import _keys_written_by_assemble_projections

    read = _keys_read_from(
        "pending_prop_result_rows", "src/settlement/recorder.py", "row",
    )
    written = _keys_written_by_assemble_projections()
    missing = read - written
    assert not missing, (
        f"recorder reads {sorted(missing)}, which assemble_projections never "
        "writes — those would be silent nulls, so every row would be skipped "
        "for a reason that is not the real one"
    )


def test_every_line_field_the_recorder_wants_is_one_the_ingester_writes():
    """
    LINE_FIELDS is what the recorder pulls off the captured board. `source` in
    particular is not optional: without it every row is skipped.
    """
    import ast
    from pathlib import Path

    from src.settlement.recorder import LINE_FIELDS, LINE_JOIN_KEYS

    source = (Path(__file__).parent.parent / "main.py").read_text()
    written: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == "ingest_prop_lines":
            for sub in ast.walk(node):
                if isinstance(sub, ast.Dict):
                    written |= {
                        k.value for k in sub.keys
                        if isinstance(k, ast.Constant) and isinstance(k.value, str)
                    }
    assert written, "ingest_prop_lines builds no dict literal any more"
    missing = (set(LINE_FIELDS) | set(LINE_JOIN_KEYS)) - written
    assert not missing, f"recorder wants {sorted(missing)} from the prop board"
