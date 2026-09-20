"""
tests/test_bigdataball_loader.py — runs against the REAL licensed
workbook, not synthetic fixtures.

Catches the two bugs that a synthetic fixture would have missed:
  1. NBA game ids losing their leading zeros (pandas int coercion)
  2. Team abbreviations coming from BigDataBall's own initials column
     ('Gol', 'Bro') instead of NBA.com's ('GSW', 'BKN')
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from src.ingestion.bigdataball import load_bigdataball_workbook, load_team_map

def _find_workbook() -> Path | None:
    """The download's filename suffix varies ('_1', '__1_', none), so glob."""
    root = Path(__file__).parent.parent / "data" / "external" / "bigdataball"
    return next(iter(sorted(root.glob("*NBA_Box_Score_Team-Stats*.xlsx"))), None)


WORKBOOK = _find_workbook()

pytestmark = pytest.mark.skipif(
    WORKBOOK is None,
    reason="Licensed BigDataBall workbook not present (not committed to git).",
)


@pytest.fixture(scope="module")
def frames():
    return load_bigdataball_workbook(WORKBOOK)


def test_loads_expected_row_count(frames):
    stats, market = frames
    # Two rows per game (home + away)
    assert len(stats) == len(market)
    assert len(stats) % 2 == 0
    assert stats["nba_game_id"].nunique() == len(stats) // 2


def test_game_ids_keep_leading_zeros(frames):
    """NBA game ids are zero-padded 10-char strings — '0022500001', not 22500001."""
    stats, _ = frames
    sample = stats["nba_game_id"].iloc[0]
    assert isinstance(sample, str)
    assert len(sample) == 10
    assert sample.startswith("00")
    assert all(len(g) == 10 for g in stats["nba_game_id"])


def test_team_abbreviations_are_nba_com_format(frames):
    """
    Must be NBA.com initials (GSW/BKN/PHX), NOT BigDataBall initials
    (Gol/Bro/Pho) — everything else in PropIQ keys on the NBA.com form.
    """
    stats, _ = frames
    teams = set(stats["team_abbr"].unique())
    assert len(teams) == 30, f"Expected 30 teams, got {len(teams)}"
    assert all(t.isupper() for t in teams), f"Found non-uppercase abbreviations: {teams}"
    assert "GSW" in teams and "Gol" not in teams
    assert "BKN" in teams and "Bro" not in teams


def test_home_away_pairing_is_consistent(frames):
    """
    Exactly one home team per game — EXCEPT neutral-site games, where
    neither team is home. The 2025-26 workbook has 6 such games (NBA Cup
    final, global games), so this must not be asserted as universal.
    """
    stats, _ = frames
    neutral_games = set(stats.loc[stats["is_neutral_site"], "nba_game_id"])
    regular = stats[~stats["nba_game_id"].isin(neutral_games)]

    per_game = regular.groupby("nba_game_id")["is_home"].sum()
    assert (per_game == 1).all(), "Non-neutral games must have exactly one home team"

    # Neutral games: zero home teams on both rows
    neutral = stats[stats["nba_game_id"].isin(neutral_games)]
    assert not neutral["is_home"].any(), "Neutral-site games must not mark either team home"


def test_neutral_site_games_are_flagged(frames):
    """
    Neutral sites are real and must be distinguishable from away games,
    so home-court/fatigue logic doesn't treat one as the other.
    """
    stats, _ = frames
    neutral_rows = stats[stats["is_neutral_site"]]
    assert len(neutral_rows) > 0, "Expected neutral-site rows in the 2025-26 workbook"
    assert len(neutral_rows) % 2 == 0, "Neutral rows should come in team pairs"
    # Every neutral row is also not-home
    assert not neutral_rows["is_home"].any()


def test_opponent_is_the_other_team(frames):
    stats, _ = frames
    first_game = stats["nba_game_id"].iloc[0]
    pair = stats[stats["nba_game_id"] == first_game]
    a, b = pair.iloc[0], pair.iloc[1]
    assert a["opponent_abbr"] == b["team_abbr"]
    assert b["opponent_abbr"] == a["team_abbr"]


def test_market_status_reflects_real_data(frames):
    """
    Rows with a closing spread or total are VALID; rows without stay
    DATA_NOT_AVAILABLE. Never silently filled with a guess.
    """
    _, market = frames
    valid = market[market["status"] == "VALID"]
    for _, row in valid.head(50).iterrows():
        assert row["closing_spread"] is not None or row["closing_total"] is not None

    invalid = market[market["status"] == "DATA_NOT_AVAILABLE"]
    for _, row in invalid.head(50).iterrows():
        assert row["closing_spread"] is None and row["closing_total"] is None


def test_spreads_are_symmetric_within_a_game(frames):
    """Home and away closing spreads should be opposite signs."""
    _, market = frames
    checked = 0
    for game_id, pair in market.groupby("nba_game_id"):
        if len(pair) != 2:
            continue
        a, b = pair.iloc[0]["closing_spread"], pair.iloc[1]["closing_spread"]
        if a is None or b is None:
            continue
        assert a == pytest.approx(-b), f"Asymmetric spread in {game_id}: {a} vs {b}"
        checked += 1
        if checked >= 100:
            break
    assert checked > 0, "No spread pairs were actually checked"


def test_team_map_prefers_nba_com_column():
    mapping = load_team_map(WORKBOOK)
    assert mapping.get("Golden State") == "GSW"
    assert mapping.get("Brooklyn") == "BKN"
