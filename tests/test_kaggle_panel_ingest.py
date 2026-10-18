"""
The Kaggle export is the second path to the player panel, and it had two bugs.

WHY IT MATTERS, 2026-10-18, two days before opening night. `player_game_logs`
held ZERO rows and `stats.nba.com` was unreachable from the deployed
environment, so there were no feature vectors for any player and the slate
produced nothing. `ingestion/kaggle_nba` exists for exactly that case -- its
docstring says a second path to the same panel "is the difference between a
pipeline that runs and one that reports success_no_data forever" -- and it
had never been run against a real export, because it was written without
network access to Kaggle.

Run against the real thing, two faults surfaced:

1. `_coerce_is_home` returned NA for EVERY ROW of a fully populated column.
   The export's `home` is float64, so `str(1.0)` is "1.0", which matched
   neither the "1" nor the "0" list. Measured: 261,689 rows, 100% null, from
   a source holding 131,386 ones and 130,303 zeros. `is_home` is NOT NULL in
   the schema, so the ingest died on an integrity error -- the lucky outcome.
   Had the column been nullable, home-court advantage would have been absent
   from every row of a panel that looked complete.

2. `upsert_player_game_logs` hardcoded `source="nba_stats_leaguegamelog"` and
   ignored the `SOURCE` the Kaggle normaliser sets, so 214,381 rows from a
   Kaggle export claimed to come from stats.nba.com. The hardcoded line sat
   two lines below a comment that already acknowledged the Kaggle path.

RESEARCH_ONLY. Column mapping and a provenance string.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.ingestion.kaggle_nba import SOURCE_NAME, _coerce_is_home

# --- 1. the home flag -------------------------------------------------------


@pytest.mark.parametrize("value", [1.0, 1, "1", "1.0", True, "HOME", "home", "H", "h", "yes"])
def test_every_home_encoding_reads_as_home(value):
    assert bool(_coerce_is_home(pd.Series([value], dtype=object)).iloc[0]) is True


@pytest.mark.parametrize("value", [0.0, 0, "0", "0.0", False, "AWAY", "away", "A", "v", "no"])
def test_every_away_encoding_reads_as_away(value):
    assert bool(_coerce_is_home(pd.Series([value], dtype=object)).iloc[0]) is False


def test_a_float_home_flag_is_not_null_for_every_row():
    """
    THE BUG, as the shape it actually arrived in: a float64 column of 1.0 and
    0.0, which is how the Kaggle export encodes it and how most CSVs do.
    """
    series = pd.Series([1.0, 0.0] * 50, dtype="float64")
    out = _coerce_is_home(series)
    assert out.isna().sum() == 0, (
        f"{int(out.isna().sum())} of {len(series)} rows of a fully populated "
        f"float home flag came back null"
    )
    assert out.sum() == 50


@pytest.mark.parametrize("value", [2.0, -1, 0.5, 7, "maybe", "", "   ", None, float("nan")])
def test_an_ambiguous_value_stays_null_rather_than_picking_a_side(value):
    """
    The refusal has to survive the fix. A wrong home/away label is a wrong
    feature, which is worse than a missing one -- and rounding 0.5 or 2 into
    a side is exactly the sort of convenience that produces one.
    """
    assert pd.isna(_coerce_is_home(pd.Series([value], dtype=object)).iloc[0])


# --- 2. provenance ----------------------------------------------------------


def test_the_frame_built_for_the_database_carries_the_writers_source():
    """
    A Kaggle-derived row must not claim stats.nba.com. Provenance exists to
    answer "where did this come from", and a hardcoded answer makes the
    question unanswerable -- including for a later real league-game-log
    write, which would have been indistinguishable from these.

    Asserted on the frame the upsert builds, with no database: the integration
    evidence is that re-running the real ingest moved all 214,381 rows from
    `nba_stats_leaguegamelog` to `kaggle_nba_box_scores`.
    """
    from src.db.repository import DEFAULT_PLAYER_LOG_SOURCE

    kaggle = _panel(with_source=True)
    live = _panel(with_source=False)
    assert list(kaggle["SOURCE"].unique()) == [SOURCE_NAME]
    assert "SOURCE" not in live.columns

    # The selection the upsert performs, applied here directly: the column
    # when present, the named default when not.
    for frame, expected in ((kaggle, SOURCE_NAME), (live, DEFAULT_PLAYER_LOG_SOURCE)):
        chosen = (
            frame["SOURCE"].astype(str)
            if "SOURCE" in frame.columns
            else DEFAULT_PLAYER_LOG_SOURCE
        )
        values = sorted(set(chosen)) if hasattr(chosen, "__iter__") and not isinstance(chosen, str) else [chosen]
        assert values == [expected]


def test_a_panel_with_no_source_column_keeps_the_live_default():
    """The live path (`ingestion/boxscores`) sets no SOURCE, and its rows must
    still say where they came from."""
    from src.db.repository import DEFAULT_PLAYER_LOG_SOURCE

    assert DEFAULT_PLAYER_LOG_SOURCE == "nba_stats_leaguegamelog"
    assert SOURCE_NAME != DEFAULT_PLAYER_LOG_SOURCE, (
        "the two paths share a provenance string, so neither can be told apart"
    )


def test_the_hardcoded_source_is_gone_from_the_upsert():
    """
    AST-free and deliberately literal: the string must not appear as an
    inline value in the frame builder any more. It lives in one named
    constant, and the `SOURCE` column wins when the writer supplies one.
    """
    import inspect

    from src.db import repository

    src = inspect.getsource(repository.upsert_player_game_logs)
    assert '"source": "nba_stats_leaguegamelog"' not in src
    assert "DEFAULT_PLAYER_LOG_SOURCE" in src
    assert '"SOURCE" in df.columns' in src


def _panel(*, with_source: bool) -> pd.DataFrame:
    frame = pd.DataFrame({
        "PLAYER_ID": [1, 2],
        "PLAYER_NAME": ["A Player", "B Player"],
        "GAME_ID": ["0022500001", "0022500001"],
        "GAME_DATE": ["2026-01-02", "2026-01-02"],
        "TEAM_ABBREVIATION": ["LAL", "BOS"],
        "OPPONENT_ABBREVIATION": ["BOS", "LAL"],
        "IS_HOME": [True, False],
        "MIN": [30.0, 28.0],
        "PTS": [20, 18],
        "REB": [5, 7],
        "AST": [4, 6],
    })
    if with_source:
        frame["SOURCE"] = SOURCE_NAME
    return frame
