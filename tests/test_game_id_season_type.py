"""
An NBA game id says what KIND of game it is, and that is the only signal we have.

WHY THIS MATTERS, found 2026-10-18. `team_game_stats` has no season_type column
and the BigDataBall workbook publishes none, so the artifact-freshness gate had
no way to compare like with like: it measured an artifact's `train_end_date`
against the newest game of ANY kind. Training drops non-regular-season rows
(`scripts/ingest_training_pack` decision 2), so train_end can never pass the end
of the regular season, while the newest row in October is a June Finals game.
The gate read 151 days over a 21-day limit and NO RETRAIN COULD CLEAR IT.

The id's third character carries the answer. Measured against the 2025-26
workbook rather than taken from documentation:

    002  regular season   2025-10-21 .. 2026-04-12   2,460 rows
    005  play-in          2026-04-14 .. 2026-04-17       12
    004  playoffs         2026-04-18 .. 2026-06-13      170
    006  NBA Cup final    2025-12-16                      2

The ranges do not overlap and they fall in real calendar order, which is what
makes the mapping evidence rather than a guess about a prefix.

RESEARCH_ONLY. String parsing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.ids import (
    REGULAR_SEASON_ID_DIGIT,
    SEASON_TYPE_ALL_STAR,
    SEASON_TYPE_CUP_FINAL,
    SEASON_TYPE_PLAY_IN,
    SEASON_TYPE_PLAYOFFS,
    SEASON_TYPE_PRESEASON,
    SEASON_TYPE_REGULAR,
    SEASON_TYPE_SQL_POSITION,
    SEASON_TYPE_UNKNOWN,
    is_regular_season_game_id,
    season_type_from_game_id,
)

WORKBOOK = "data/external/bigdataball/2025-2026_NBA_Box_Score_Team-Stats.xlsx"


@pytest.mark.parametrize(
    "game_id,expected",
    [
        ("0012500001", SEASON_TYPE_PRESEASON),
        ("0022500001", SEASON_TYPE_REGULAR),
        ("0032500001", SEASON_TYPE_ALL_STAR),
        ("0042500406", SEASON_TYPE_PLAYOFFS),
        ("0052500101", SEASON_TYPE_PLAY_IN),
        ("0062500001", SEASON_TYPE_CUP_FINAL),
    ],
)
def test_each_kind_is_read_from_the_third_character(game_id, expected):
    assert season_type_from_game_id(game_id) == expected


@pytest.mark.parametrize(
    "value",
    ["", "   ", None, "abc", "not-an-id", 12, "00", "0", [], {}, "0072500001"],
)
def test_an_id_the_parser_cannot_read_is_unknown_not_regular(value):
    """
    THE MUTATION THAT SURVIVED THE FIRST PASS. Defaulting an unreadable id to
    REGULAR changed nothing visible, because every caller then filtered in SQL
    on the digit and length — so the parser's own default was never exercised.
    It still has to be right: a caller that reaches for the parser instead
    (any new one will) must get UNKNOWN, because admitting one unreadable row
    to a regular-season-only comparison reinstates the whole fault.
    """
    assert season_type_from_game_id(value) == SEASON_TYPE_UNKNOWN
    assert is_regular_season_game_id(value) is False


def test_only_the_regular_season_kind_is_regular():
    for game_id in ("0012500001", "0032500001", "0042500406", "0052500101", "0062500001"):
        assert is_regular_season_game_id(game_id) is False
    assert is_regular_season_game_id("0022500001") is True


def test_the_sql_constants_agree_with_the_parser():
    """
    The SQL filter in `db.repository` reads these two rather than restating
    the mapping. If they disagreed with the parser, a row would be regular
    season to one reader and not the other — which is the defect shape this
    repository keeps finding, in the fix for it.
    """
    probe = "00" + REGULAR_SEASON_ID_DIGIT + "2500001"
    assert len(probe) == 10
    assert probe[SEASON_TYPE_SQL_POSITION - 1] == REGULAR_SEASON_ID_DIGIT
    assert season_type_from_game_id(probe) == SEASON_TYPE_REGULAR


def test_the_mapping_holds_against_the_real_workbook():
    """
    The evidence in the module docstring, asserted. If a future export numbers
    its games differently this fails here rather than silently moving a gate.
    """
    import pandas as pd

    if not Path(WORKBOOK).is_file():  # pragma: no cover - licensed export
        pytest.skip("licensed workbook not present")

    from src.ingestion.bigdataball import load_bigdataball_workbook

    stats, _ = load_bigdataball_workbook(WORKBOOK)
    dates = pd.to_datetime(stats["game_date"])
    kinds = stats["nba_game_id"].map(season_type_from_game_id)

    assert SEASON_TYPE_UNKNOWN not in set(kinds), (
        "this workbook carries game ids the parser cannot read, so the "
        "season-type filter would silently exclude real regular-season rows"
    )
    regular = dates[kinds == SEASON_TYPE_REGULAR]
    playoffs = dates[kinds == SEASON_TYPE_PLAYOFFS]
    assert len(regular) > len(playoffs) > 0

    # The ordering is what makes the mapping believable: the regular season
    # ends before the play-in, which ends before the playoffs.
    assert regular.max() < dates[kinds == SEASON_TYPE_PLAY_IN].min()
    assert dates[kinds == SEASON_TYPE_PLAY_IN].max() < playoffs.min()
    assert str(regular.max().date()) == "2026-04-12"
    assert str(dates.max().date()) == "2026-06-13"
