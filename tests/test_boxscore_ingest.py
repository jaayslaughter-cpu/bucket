"""Parser tests for player game-log ingestion.

The payload below is a STRUCTURAL fixture: it mirrors the endpoint's
headers/rowSet shape so the parsing rules can be tested offline. The stat
values are placeholders and are not real NBA results — nothing here is
used as data, only as structure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.boxscores import BoxScoreFetchError, _parse_matchup, parse_league_game_log

HEADERS = [
    "SEASON_ID", "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION",
    "GAME_ID", "GAME_DATE", "MATCHUP", "WL", "MIN", "FGM", "FGA", "FG3M",
    "FG3A", "FTM", "FTA", "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV",
    "PF", "PTS", "PLUS_MINUS",
]


def _row(game_id, matchup, game_date="2025-10-21"):
    return [
        "22025", 201939, "Test Player", 1610612744, "GSW",
        game_id, game_date, matchup, "W", 34.0, 9, 18, 3,
        7, 4, 4, 1, 4, 5, 6, 1, 0, 2,
        2, 25, 8,
    ]


def _payload(rows):
    return {"resultSets": [{"name": "LeagueGameLog", "headers": HEADERS, "rowSet": rows}]}


def test_game_ids_keep_leading_zeros():
    """NBA game ids are zero-padded 10-char strings; JSON may unpad them."""
    frame = parse_league_game_log(_payload([_row(22500001, "GSW @ LAL")]), season="2025-26")
    assert frame["GAME_ID"].iloc[0] == "0022500001"
    assert frame["GAME_ID"].map(len).eq(10).all()


def test_matchup_parsing_gives_opponent_and_home_flag():
    assert _parse_matchup("GSW @ LAL") == ("LAL", False)
    assert _parse_matchup("GSW vs. LAL") == ("LAL", True)
    assert _parse_matchup(None) == (None, None)
    assert _parse_matchup("nonsense") == (None, None)


def test_away_and_home_rows_map_correctly():
    frame = parse_league_game_log(
        _payload([_row("0022500001", "GSW @ LAL"), _row("0022500002", "GSW vs. LAL")]),
        season="2025-26",
    )
    assert frame["OPPONENT_ABBREVIATION"].tolist() == ["LAL", "LAL"]
    assert frame["IS_HOME"].tolist() == [False, True]


def test_empty_rowset_raises_rather_than_returning_nothing():
    """An empty frame would read as 'a season with no games'."""
    with pytest.raises(BoxScoreFetchError, match="zero player-game rows"):
        parse_league_game_log(_payload([]), season="2025-26")


def test_missing_expected_columns_raises():
    bad = {"resultSets": [{"headers": ["PLAYER_ID", "PTS"], "rowSet": [[1, 2]]}]}
    with pytest.raises(BoxScoreFetchError, match="missing expected columns"):
        parse_league_game_log(bad, season="2025-26")


def test_parsed_frame_matches_the_feature_builder_contract():
    """The panel must slot straight into build_feature_matrix."""
    from src.features.builder import build_feature_matrix

    # Distinct dates: a player never has two games on one calendar day, and
    # the builder's lookahead guard rightly rejects a same-day prior game.
    rows = [
        _row(f"00225000{i:02d}", "GSW @ LAL", game_date=f"2025-10-{20 + i:02d}")
        for i in range(1, 6)
    ]
    frame = parse_league_game_log(_payload(rows), season="2025-26")
    for required in ("PLAYER_ID", "GAME_DATE", "PTS", "REB", "AST", "MIN"):
        assert required in frame.columns

    feats = build_feature_matrix(frame)
    assert "PTS_L2" in feats.columns
    assert "fatigue_multiplier" in feats.columns
