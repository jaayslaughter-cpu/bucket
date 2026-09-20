"""Elo and schedule-context features.

The properties asserted here are the ones that catch a wrong join or a
leaked postgame field — the failures that would otherwise look like a
slightly worse model rather than a bug.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.schedule import (
    ARENA_COORDS,
    attach_team_schedule_features,
    haversine_miles,
    venue_for,
)
from src.features.team_strength import (
    EloConfig,
    attach_elo_features,
    compute_team_elo,
    expected_score,
    margin_multiplier,
)


def _two_team_season(n_games: int = 40, home_wins: bool = True) -> pd.DataFrame:
    """Alternating home/away games between two teams, one always winning."""
    rows = []
    for g in range(n_games):
        date = pd.Timestamp("2025-10-21") + pd.Timedelta(days=2 * g)
        a_home = g % 2 == 0
        a_pts, b_pts = (110, 100) if home_wins else (100, 110)
        for team, opp, is_home, pts in (
            ("AAA", "BBB", a_home, a_pts),
            ("BBB", "AAA", not a_home, b_pts),
        ):
            rows.append({
                "nba_game_id": f"00225{g:05d}", "game_date": date, "team_abbr": team,
                "opponent_abbr": opp, "is_home": is_home, "points": pts,
                "season": "2025-26", "is_neutral_site": False,
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Elo mechanics
# --------------------------------------------------------------------------

def test_expected_score_is_symmetric_and_centred():
    assert expected_score(1500, 1500) == pytest.approx(0.5)
    assert expected_score(1600, 1500) + expected_score(1500, 1600) == pytest.approx(1.0)
    # A 400-point edge is 10:1 by construction.
    assert expected_score(1900, 1500) == pytest.approx(10 / 11, abs=1e-6)


def test_home_advantage_raises_the_home_side():
    assert expected_score(1500, 1500, home_advantage=100) > 0.5


def test_margin_multiplier_damps_the_bigger_favourite():
    """Same blowout, larger pre-game edge -> smaller rating move."""
    underdog_blowout = margin_multiplier(20, winner_rating_edge=-200)
    favourite_blowout = margin_multiplier(20, winner_rating_edge=400)
    assert favourite_blowout < underdog_blowout


def test_margin_multiplier_grows_with_margin():
    edge = 50
    assert margin_multiplier(30, edge) > margin_multiplier(5, edge)


def test_ratings_are_zero_sum():
    """Elo moves points between teams; it cannot create them."""
    elo = compute_team_elo(_two_team_season())
    final = elo.sort_values("game_date").groupby("team").tail(1)
    assert final["elo_post"].mean() == pytest.approx(1500.0, abs=1e-6)


def test_winning_team_gains_and_loser_loses():
    elo = compute_team_elo(_two_team_season(home_wins=True))
    final = elo.sort_values("game_date").groupby("team").tail(1).set_index("team")
    # AAA wins at home; over alternating games the stronger side pulls ahead.
    assert final.loc["AAA", "elo_post"] != pytest.approx(1500.0)
    assert (final.loc["AAA", "elo_post"] - 1500.0) == pytest.approx(
        -(final.loc["BBB", "elo_post"] - 1500.0), abs=1e-6
    )


def test_elo_pre_never_reflects_the_current_game():
    """elo_pre must equal the previous game's elo_post for that team."""
    elo = compute_team_elo(_two_team_season(n_games=10)).sort_values("game_date")
    one = elo[elo["team"] == "AAA"].reset_index(drop=True)
    for i in range(1, len(one)):
        assert one.loc[i, "elo_pre"] == pytest.approx(one.loc[i - 1, "elo_post"], abs=1e-6)


def test_first_game_starts_at_the_base_rating():
    elo = compute_team_elo(_two_team_season(n_games=3)).sort_values("game_date")
    assert elo.iloc[0]["elo_pre"] == pytest.approx(1500.0)


def test_win_probability_accounts_for_both_sides_home_advantage():
    """Each game's two probabilities must sum to 1.

    Applying only the scored team's home advantage leaves both rows above
    their true value, so away probabilities come out systematically high.
    """
    elo = compute_team_elo(_two_team_season(n_games=6))
    per_game = elo.groupby("game_id")["elo_win_probability"].sum()
    assert per_game.apply(lambda s: s == pytest.approx(1.0, abs=1e-6)).all()


def test_neutral_site_gives_neither_team_home_advantage():
    games = _two_team_season(n_games=2)
    games["is_neutral_site"] = True
    elo = compute_team_elo(games)
    first = elo[elo["game_id"] == elo["game_id"].min()]
    assert set(first["elo_win_probability"].round(6)) == {0.5}


def test_offseason_regression_pulls_toward_the_mean():
    season_one = _two_team_season(n_games=30, home_wins=True)
    season_two = _two_team_season(n_games=2, home_wins=True)
    season_two["game_date"] = season_two["game_date"] + pd.Timedelta(days=365)
    season_two["season"] = "2026-27"
    season_two["nba_game_id"] = season_two["nba_game_id"] + "B"

    config = EloConfig(offseason_regression=0.5, league_mean=1500.0)
    elo = compute_team_elo(pd.concat([season_one, season_two]), config).sort_values("game_date")

    last_old = elo[elo["season"] == "2025-26"].groupby("team").tail(1).set_index("team")
    first_new = elo[elo["season"] == "2026-27"].groupby("team").head(1).set_index("team")
    for team in ("AAA", "BBB"):
        assert abs(first_new.loc[team, "elo_pre"] - 1500.0) < abs(last_old.loc[team, "elo_post"] - 1500.0)


def test_unpaired_game_is_skipped_not_guessed():
    games = _two_team_season(n_games=2)
    orphan = games.iloc[[0]].copy()
    orphan["nba_game_id"] = "00225ORPHAN"
    elo = compute_team_elo(pd.concat([games, orphan]))
    assert "00225ORPHAN" not in set(elo["game_id"])


def test_missing_score_does_not_update_ratings():
    games = _two_team_season(n_games=3)
    games.loc[games["nba_game_id"] == "0022500001", "points"] = np.nan
    elo = compute_team_elo(games)
    assert "0022500001" not in set(elo["game_id"])


def test_attach_elo_leaves_unmatched_rows_null():
    elo = compute_team_elo(_two_team_season(n_games=4))
    panel = pd.DataFrame({
        "GAME_ID": ["0022500000", "NOT_A_GAME"],
        "TEAM_ABBREVIATION": ["AAA", "AAA"],
        "PLAYER_ID": ["p1", "p1"],
    })
    out = attach_elo_features(panel, elo)
    assert out.loc[0, "TEAM_ELO_PRE"] == pytest.approx(1500.0)
    assert pd.isna(out.loc[1, "TEAM_ELO_PRE"])


def test_attach_elo_exposes_no_postgame_column():
    """elo_post must never reach the feature matrix."""
    elo = compute_team_elo(_two_team_season(n_games=4))
    panel = pd.DataFrame({
        "GAME_ID": ["0022500000"], "TEAM_ABBREVIATION": ["AAA"], "PLAYER_ID": ["p1"],
    })
    out = attach_elo_features(panel, elo)
    assert not [c for c in out.columns if "POST" in c.upper()]


# --------------------------------------------------------------------------
# Schedule context
# --------------------------------------------------------------------------

def test_haversine_matches_a_known_distance():
    bos, lal = ARENA_COORDS["BOS"], ARENA_COORDS["LAL"]
    miles = haversine_miles(bos[0], bos[1], lal[0], lal[1])
    assert 2550 < miles < 2650  # Boston to Los Angeles


def test_haversine_of_a_point_with_itself_is_zero():
    lat, lon = ARENA_COORDS["CHI"]
    assert haversine_miles(lat, lon, lat, lon) == pytest.approx(0.0, abs=1e-9)


def test_relocated_venue_resolves_by_date():
    """Golden State played in Oakland before 2019-20, San Francisco after."""
    from datetime import date

    before = venue_for("GSW", date(2018, 12, 1))
    after = venue_for("GSW", date(2022, 12, 1))
    assert before != after
    assert after == ARENA_COORDS["GSW"]


def test_bubble_overrides_the_home_venue():
    from datetime import date

    from src.features.schedule import _game_venue

    assert _game_venue("LAL", date(2020, 8, 15)) == (28.37, -81.55)
    assert _game_venue("LAL", date(2019, 12, 15)) == ARENA_COORDS["LAL"]


def _panel_from(team_games: pd.DataFrame) -> pd.DataFrame:
    return team_games.rename(columns={
        "nba_game_id": "GAME_ID", "game_date": "GAME_DATE",
        "team_abbr": "TEAM_ABBREVIATION", "opponent_abbr": "OPPONENT_ABBREVIATION",
        "is_home": "IS_HOME", "is_neutral_site": "IS_NEUTRAL_SITE", "season": "SEASON",
    })


def test_rest_advantage_is_antisymmetric():
    """One team's advantage is exactly the other's disadvantage."""
    out = attach_team_schedule_features(_panel_from(_two_team_season(n_games=12)))
    per_game = out.groupby("GAME_ID")["REST_ADVANTAGE"].sum(min_count=2).dropna()
    assert len(per_game) > 0
    assert per_game.abs().max() == pytest.approx(0.0, abs=1e-9)


def test_back_to_back_halves_are_equal_in_number():
    """Every back-to-back has exactly one first game and one second."""
    out = attach_team_schedule_features(_panel_from(_two_team_season(n_games=12)))
    schedule = out.drop_duplicates(subset=["TEAM_ABBREVIATION", "GAME_ID"])
    assert int(schedule["IS_B2B_SECOND"].fillna(0).sum()) == int(
        schedule["IS_B2B_FIRST"].fillna(0).sum()
    )


def test_first_game_of_a_season_has_no_rest_value():
    """A debut has no prior game; NaN is correct and 7 is fabrication."""
    out = attach_team_schedule_features(_panel_from(_two_team_season(n_games=5)))
    first = out.sort_values("GAME_DATE").groupby("TEAM_ABBREVIATION").head(1)
    assert first["TEAM_DAYS_REST"].isna().all()
    assert first["TRAVEL_MILES"].isna().all()


def test_rest_is_partitioned_by_season():
    """The first game after an offseason must not read as ~150 days rest."""
    one = _two_team_season(n_games=4)
    two = _two_team_season(n_games=4)
    two["game_date"] = two["game_date"] + pd.Timedelta(days=300)
    two["season"] = "2026-27"
    two["nba_game_id"] = two["nba_game_id"] + "B"

    out = attach_team_schedule_features(_panel_from(pd.concat([one, two])))
    new_season = out[out["SEASON"] == "2026-27"].sort_values("GAME_DATE")
    opener = new_season.groupby("TEAM_ABBREVIATION").head(1)
    assert opener["TEAM_DAYS_REST"].isna().all()


def test_neutral_site_travel_is_null_not_wrong():
    games = _two_team_season(n_games=4)
    games["is_neutral_site"] = True
    out = attach_team_schedule_features(_panel_from(games))
    assert out["TRAVEL_MILES"].isna().all()


def test_missing_columns_raise_rather_than_silently_skip():
    with pytest.raises(ValueError, match="DATA_NOT_AVAILABLE"):
        attach_team_schedule_features(pd.DataFrame({"TEAM_ABBREVIATION": ["AAA"]}))
