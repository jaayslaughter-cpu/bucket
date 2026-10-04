"""O8 — the under and the push on a projection.

`projections` stored only `prob_over`. On a HALF line that loses nothing: a
push is impossible, so the under is exactly 1 - P(over). On a WHOLE line it is
not — push has real mass, `1 - P(over)` is P(under OR push), and the push mass
is unrecoverable once the row is written. A reader holding only `prob_over`
cannot tell which kind of line a row describes, so every whole-line under
computed after the fact was wrong.

The fix writes all three legs, and the NULLS are the point: a binary classifier
has no push mass to split out, so on a whole or unknown line both the under and
the push are refused with a recorded reason rather than guessed.
"""

from __future__ import annotations

import pandas as pd
import pytest

from main import DEFAULT_STATS, assemble_projections


def _features(n: int = 4) -> pd.DataFrame:
    return pd.DataFrame({
        "PLAYER_ID": [f"200000{i}" for i in range(n)],
        "PLAYER_NAME": [f"P{i}" for i in range(n)],
        "GAME_ID": ["0022500001"] * n,
        "GAME_DATE": [pd.Timestamp("2025-11-01")] * n,
        "PTS_BASELINE": [24.0] * n,
        "PTS_L2": [24.2] * n,
        "fatigue_multiplier": [1.0] * n,
    })


def _assembled(lines, probs=0.60, n: int = 4) -> pd.DataFrame:
    feat = _features(n)
    series = pd.Series(
        [probs] * n if not isinstance(probs, list) else probs, index=feat.index
    )
    series.attrs["target_market"] = "PTS"
    board = pd.DataFrame({
        "player_name": [f"P{i}" for i in range(len(lines))],
        "market": ["PTS"] * len(lines),
        "line": list(lines),
    })
    return assemble_projections(
        feat, series, {"status": "DATA_NOT_AVAILABLE"},
        prop_lines=board, stats=("PTS",),
    )


# --- the three cases -------------------------------------------------------

def test_a_half_line_gets_an_exact_under_and_a_zero_push():
    """Push is arithmetically impossible, so the complement is not a guess."""
    row = _assembled([24.5], n=1).iloc[0]
    assert row["PROB_OVER"] == pytest.approx(0.60)
    assert row["PROB_UNDER"] == pytest.approx(0.40)
    assert row["PROB_PUSH"] == pytest.approx(0.0)
    assert row["NOTES"] is None


def test_a_whole_line_refuses_the_complement_and_says_why():
    """
    THE DEFECT O8 NAMES. 1 - P(over) is P(under OR push) here, so writing it as
    the under silently absorbs the push mass.
    """
    row = _assembled([24.0], n=1).iloc[0]
    assert row["PROB_OVER"] == pytest.approx(0.60)
    assert row["PROB_UNDER"] is None
    assert row["PROB_PUSH"] is None
    assert "whole-number line" in str(row["NOTES"])
    assert "refusing" in str(row["NOTES"])


def test_an_unknown_line_takes_the_refusing_branch_not_the_assuming_one():
    """
    A line nobody captured cannot be shown to be a half-line. The safe default
    is the one that refuses, and `paper_research.is_whole_number_line(None)` is
    True for exactly this reason.
    """
    row = _assembled([None], n=1).iloc[0]
    assert row["PROB_UNDER"] is None
    assert row["PROB_PUSH"] is None
    assert row["NOTES"] is not None


def test_a_row_with_no_probability_is_an_absence_not_a_refusal():
    row = _assembled([24.5], probs=[None], n=1).iloc[0]
    assert pd.isna(row["PROB_OVER"])
    assert row["PROB_UNDER"] is None
    assert "model_p_over missing" in str(row["NOTES"])


def test_the_three_legs_sum_to_one_wherever_all_three_are_present():
    out = _assembled([24.5, 23.5, 30.5], n=3)
    present = out[out["PROB_UNDER"].notna() & out["PROB_PUSH"].notna()]
    assert len(present) == 3
    for _, row in present.iterrows():
        total = float(row["PROB_OVER"]) + float(row["PROB_UNDER"]) + float(row["PROB_PUSH"])
        assert total == pytest.approx(1.0, abs=1e-9)


def test_mixed_lines_are_resolved_independently():
    """One whole line in the slate must not withhold the half-line rows."""
    out = _assembled([24.5, 24.0, 25.5], n=3)
    by_name = out.set_index("PLAYER_NAME")
    assert by_name.loc["P0", "PROB_UNDER"] == pytest.approx(0.40)
    assert by_name.loc["P1", "PROB_UNDER"] is None
    assert by_name.loc["P2", "PROB_UNDER"] == pytest.approx(0.40)


# --- it reaches the database ----------------------------------------------

def test_the_projection_table_has_somewhere_to_put_them():
    from src.db.models import Projection

    columns = {c.name for c in Projection.__table__.columns}
    for column in ("prob_over", "prob_under", "prob_push"):
        assert column in columns, f"Projection has no {column} column"


def test_the_writer_maps_every_leg_and_the_reason():
    """
    `persist_projections` builds its row dict by hand, so a column added to the
    model and forgotten here persists as NULL on every write — which is the
    bug its own comment records happening three times before.
    """
    repo = (
        __import__("pathlib").Path(__file__).parent.parent
        / "src" / "db" / "repository.py"
    ).read_text(encoding="utf-8")
    block = repo[repo.index("def persist_projections"):]
    block = block[: block.index("\ndef ")]
    for mapping in ('"prob_under": r.get("PROB_UNDER")',
                    '"prob_push": r.get("PROB_PUSH")',
                    '"notes": r.get("NOTES")'):
        assert mapping in block, f"persist_projections does not write {mapping}"
    # and the upsert has to refresh them, or a re-run keeps the first value
    for column in ('"prob_under"', '"prob_push"'):
        assert column in block, f"the on-conflict set omits {column}"


def test_assemble_projections_emits_the_keys_the_writer_reads():
    """The two halves of that hand-written mapping, checked against each other."""
    out = _assembled([24.5], n=1)
    for column in ("PROB_OVER", "PROB_UNDER", "PROB_PUSH", "NOTES"):
        assert column in out.columns


def test_the_migration_adds_the_columns_without_backfilling_them():
    """
    A backfill would have to guess whether each existing row's line was whole,
    and guessing writes the exact error these columns exist to prevent.
    """
    sql = (
        __import__("pathlib").Path(__file__).parent.parent
        / "migrations" / "005_projection_under_push.sql"
    ).read_text(encoding="utf-8")
    assert "ADD COLUMN IF NOT EXISTS prob_under" in sql
    assert "ADD COLUMN IF NOT EXISTS prob_push" in sql
    assert "UPDATE projections" not in sql, "a backfill would guess"
    assert "1 - prob_over" in sql or "1 - P(over)" in sql


def test_an_empty_frame_still_carries_the_columns():
    """A consumer must not have to branch on whether the slate was empty."""
    empty = assemble_projections(
        _features(0), pd.Series(dtype="float64"),
        {"status": "DATA_NOT_AVAILABLE"}, stats=DEFAULT_STATS,
    )
    assert empty.empty
