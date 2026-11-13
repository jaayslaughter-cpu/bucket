"""
tests/test_projection_nan_is_null.py — NaN is not NULL, and it cost two slates.

THE OUTAGE, AND IT WAS NOT WHAT I FIRST SAID IT WAS.

The 2026-11-11 and 2026-11-12 slates both exited 1 with

    Pipeline FAILED: (psycopg.errors.NumericValueOutOfRange) integer out of range
    [SQL: INSERT INTO projections (... odds_over_american ...)]

My first diagnosis was a provider price larger than a 32-bit integer, and I
shipped a magnitude bound for it. The bound is reasonable hardening and it
was NOT the cause: the value is not large, it is NaN.

    commit 03db157, 2026-11-10  migration 010 gives `projections` its first
                                two INTEGER columns and
                                `_attach_prop_line_prices` starts filling them
    2026-11-11  541 of 1140 rows priced  -> FAILED
    2026-11-12  232 of 444  rows priced  -> FAILED

A board is only ever PARTLY priced, so the column pandas builds is float64
with NaN in the unpriced rows. Postgres casting a float8 NaN to INTEGER
raises exactly "integer out of range". It failed every night from the day the
columns landed, and the magnitude bound could not have helped -- it makes
MORE NaNs, one per refused price.

AND THE REPOSITORY ALREADY KNEW. `repository._records` exists for this and
says so in its own docstring:

    the driver receives a float NaN where an integer column expects NULL and
    the insert fails

`persist_projections` hand-built its rows with `r.get(...)` and never went
through it.

THE SECOND HALF OF THE TRAP, measured because the first fix changed nothing:
converting the frame and then iterating with `iterrows` does not work.

    converted.iloc[1]["ODDS_OVER"]             -> None
    converted.to_dict("records")[1]["ODDS_OVER"] -> None
    list(converted.iterrows())[1][1].get(...)  -> nan

`iterrows` rebuilds each row as a Series and reintroduces the NaN. That is
why `_records` ends in `to_dict(orient="records")`.

RESEARCH_ONLY. Null handling on a write path. No wager.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def _partially_priced_frame() -> pd.DataFrame:
    """
    A slate frame shaped like the live one: some rows priced, some not.

    This is the ONLY shape that triggers the failure. All rows priced gives a
    clean float column; no rows priced gives object dtype holding None, which
    inserts as NULL. The mixture is what pandas turns into float64 with NaN,
    and every real board is a mixture.
    """
    return pd.DataFrame({
        "PLAYER_ID": ["1", "2"],
        "PLAYER_NAME": ["Priced Player", "Unpriced Player"],
        "GAME_ID": ["g1", "g1"],
        "GAME_DATE": [pd.Timestamp("2026-11-12")] * 2,
        "MARKET": ["PTS", "PTS"],
        "BASELINE": [20.0, 18.0],
        "FATIGUE_MULTIPLIER": [1.0, 1.0],
        "FATIGUE_NOTES": ["", ""],
        "FINAL_PROJECTION": [20.5, np.nan],
        "LINE": [20.5, np.nan],
        "PROB_OVER": [0.55, np.nan],
        "PROB_UNDER": [0.45, np.nan],
        "PROB_PUSH": [np.nan, np.nan],
        "ODDS_OVER": [-110.0, np.nan],
        "ODDS_UNDER": [-110.0, np.nan],
        "LINE_SOURCE": ["book", None],
        "MARKET_STATUS": ["OK", "DATA_NOT_AVAILABLE"],
        "MARKET_STATUS_REASON": [None, "no price"],
        "NOTES": [None, None],
    })


def _captured_rows(frame: pd.DataFrame, monkeypatch) -> list[dict]:
    """Run `persist_projections` with the session stubbed, and keep its rows."""
    import src.db.repository as repo

    captured: list[dict] = []

    class _Session:
        def execute(self, stmt, *a, **k):
            # The values are on the statement for a pg_insert(...).values(...)
            compiled = getattr(stmt, "_values", None) or {}
            if compiled:
                captured.append({k2: v for k2, v in compiled.items()})
            return None

    class _Scope:
        def __enter__(self):
            return _Session()

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(repo, "session_scope", lambda: _Scope())
    monkeypatch.setattr(repo, "batched", lambda rows: [rows])

    def _capture(rows_in):
        captured.clear()
        captured.extend(rows_in)
        return [rows_in]

    monkeypatch.setattr(repo, "batched", _capture)
    repo.persist_projections(frame, run_id="test-run")
    return captured


# ---------------------------------------------------------------------------
# The outage.
# ---------------------------------------------------------------------------


def test_an_unpriced_row_sends_null_and_not_nan(monkeypatch) -> None:
    """
    THE ONE THAT MATTERS. A float NaN in an INTEGER column is the failure.

    Asserted on the rows handed to the driver, because that is where the
    value either is or is not NaN -- a DataFrame cell being None proved
    nothing, since `iterrows` undid it.
    """
    rows = _captured_rows(_partially_priced_frame(), monkeypatch)
    assert len(rows) == 2, rows

    priced, unpriced = rows
    assert priced["odds_over_american"] == -110.0

    for column in ("odds_over_american", "odds_under_american"):
        value = unpriced[column]
        assert value is None, (
            f"{column} was sent as {value!r} ({type(value).__name__}). A "
            f"float NaN in an INTEGER column is what failed the 2026-11-11 "
            f"and 2026-11-12 slates."
        )
        assert not (isinstance(value, float) and np.isnan(value))


def test_no_numeric_column_sends_nan_either(monkeypatch) -> None:
    """
    The NUMERIC columns did not error -- which is worse, not better.

    Postgres accepts 'NaN' as a numeric VALUE, so a missing `line` or
    `prob_over` was being STORED as NaN rather than NULL. This repository
    treats a null as "unknown" everywhere else, and NaN-as-a-value is a
    fabrication that compares false against itself and silently poisons any
    aggregate over the column.
    """
    rows = _captured_rows(_partially_priced_frame(), monkeypatch)
    unpriced = rows[1]

    for column in ("final_projection", "line", "prob_over", "prob_under",
                   "prob_push"):
        value = unpriced[column]
        assert value is None or not (
            isinstance(value, float) and np.isnan(value)
        ), f"{column} was sent as NaN rather than NULL"


def test_a_fully_priced_frame_still_sends_its_prices(monkeypatch) -> None:
    """The conversion must not null out real values."""
    frame = _partially_priced_frame()
    frame.loc[1, ["ODDS_OVER", "ODDS_UNDER", "LINE", "PROB_OVER"]] = [
        -105.0, -115.0, 18.5, 0.52,
    ]
    rows = _captured_rows(frame, monkeypatch)

    assert rows[1]["odds_over_american"] == -105.0
    assert rows[1]["odds_under_american"] == -115.0
    assert rows[1]["line"] == 18.5
    assert rows[1]["prob_over"] == 0.52


# ---------------------------------------------------------------------------
# The mechanism, pinned so the fix cannot be undone by a plausible edit.
# ---------------------------------------------------------------------------


def test_persist_projections_does_not_iterate_the_frame_directly() -> None:
    """
    `iterrows` REINTRODUCES THE NaN, so using it is the bug.

    Structural rather than behavioural on purpose: the tests above would also
    pass if someone converted the frame and then iterated it with `iterrows`
    on a frame that happened to be all-object. This pins the mechanism that
    makes the conversion survive.
    """
    source = (REPO / "src" / "db" / "repository.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    target = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "persist_projections"
    )
    calls = {
        node.func.attr for node in ast.walk(target)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "iterrows" not in calls, (
        "persist_projections iterates the frame with iterrows, which rebuilds "
        "each row as a Series and turns None back into float NaN -- the exact "
        "failure this module documents"
    )
    names = {
        node.func.id for node in ast.walk(target)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "_records" in names, (
        "persist_projections no longer goes through `_records`, which is the "
        "one implementation of NaN -> None in this module"
    )


def test_iterrows_really_does_undo_the_conversion() -> None:
    """
    The measurement the fix rests on, kept executable rather than asserted in
    a comment.

    If a future pandas makes `iterrows` preserve None, this fails and the
    structural test above can be relaxed deliberately rather than by someone
    guessing.
    """
    df = pd.DataFrame({"ODDS_OVER": [-110.0, np.nan], "NAME": ["a", "b"]})
    converted = df.astype(object).where(pd.notna(df), None)

    assert converted.iloc[1]["ODDS_OVER"] is None
    assert converted.to_dict(orient="records")[1]["ODDS_OVER"] is None

    via_iterrows = list(converted.iterrows())[1][1].get("ODDS_OVER")
    assert via_iterrows is not None and np.isnan(via_iterrows), (
        "iterrows no longer reintroduces NaN -- the hazard this module "
        "documents has changed and the comments should be revisited"
    )


# ---------------------------------------------------------------------------
# My own false-positive log.
# ---------------------------------------------------------------------------


def test_an_unpriced_side_is_not_logged_as_an_implausible_price() -> None:
    """
    MY BUG, AND THE TEST I WROTE FOR IT PASSED ANYWAY.

    `test_an_unpriced_row_is_not_reported_as_a_refusal` put a literal None in
    the fixture frame. The real lookup comes from a DataFrame and yields float
    NaN, for which `nan not in (None, "")` is True and `str(nan) != "0"` is
    also True -- so every ordinary unpriced side was reported as a REFUSED
    IMPLAUSIBLE price. On 2026-11-12's board that would have been 212 of 444
    rows. The test was right about the rule and wrong about the value.
    """
    import io
    import logging

    import main

    # NOT caplog. `configure_logging` leaves `propiq.main` non-propagating, so
    # caplog captures nothing from it and the first version of this test
    # asserted against an empty string -- which would have passed any claim
    # about absence. A handler on the logger itself sees what is logged
    # regardless of propagation. The same trap makes
    # `test_a_real_sized_artifact_reports_itself_and_raises_no_alarm` pass
    # only when earlier tests in its file have run.
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    logger = logging.getLogger("propiq.main")
    previous_handlers, previous_level = logger.handlers, logger.level
    logger.handlers = [handler]
    logger.setLevel(logging.ERROR)

    projections = pd.DataFrame({
        "PLAYER_NAME": ["A B", "C D", "E F"], "MARKET": ["PTS"] * 3,
    })
    lines = pd.DataFrame({
        "player_name": ["A B", "C D", "E F"],
        "market": ["PTS"] * 3,
        "line": [20.5, 18.5, 15.5],
        # priced / unpriced / genuinely implausible
        "over_odds_american": [-110, None, 3_000_000_000],
        "under_odds_american": [-110, -110, -110],
        "source": ["book"] * 3,
    })

    try:
        main._attach_prop_line_prices(projections, lines)
    finally:
        logger.handlers, logger.level = previous_handlers, previous_level
    text = stream.getvalue()

    assert text, "nothing was logged, so this test asserts nothing"
    assert "REFUSED AS IMPLAUSIBLE" in text
    assert "1 POSTED PRICE" in text, (
        f"the unpriced row was counted as a refusal: {text}"
    )
    assert "nan" not in text.lower(), (
        f"a NaN was named as an implausible price; it is an absent one: {text}"
    )
    assert "3000000000" in text
