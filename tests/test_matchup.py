"""Opponent defence from real assignments — src/features/matchup.py.

THE RECORD THIS LAYER SETTLES. docs/external_repo_review_2026-09.md's
2026-09-26 addendum says of the leaguedashptdefend endpoint: "Supersedes P1.4.
Gives POSITION directly *and* a better defender feature than position-bucketed
DvP." Position-bucketed DvP was built anyway (src/features/dvp.py). This is
the layer it was a proxy for: the defender is named rather than inferred from
a G/F/C bucket.

What the tests pin, in order:

  1. the defending team is DERIVED, because `team_id` on a matchup row is the
     OFFENSIVE player's team — reading it as the defender would describe each
     team's own offence as its defence and still fill the column;
  2. the shift, over the defending team's GAMES rather than its ~94 matchup
     rows per game, which is the window-span defect dvp.py documents finding
     in its own source;
  3. ratios of totals, so a defender who spent eleven seconds on a star does
     not weigh the same as one who guarded him for twenty minutes;
  4. the clock-string minutes parse, because pd.to_numeric would null the
     column and the layer would abstain on every row while looking like it had
     simply found no data;
  5. the as-of league baseline, never season-wide;
  6. that only the measured-useful half ships.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.matchup import (
    MATCHUP_FEATURE_COLS,
    MATCHUP_SHIPPABLE_COLS,
    ROLL_MIN_PERIODS,
    MatchupFeatureError,
    attach_matchup_features,
    build_defender_allowed,
    parse_matchup_minutes,
)

HOME, AWAY = 1610612737, 1610612738
CODES = {HOME: "AAA", AWAY: "BBB"}


def rows_for_game(game, *, home_allows=1.0, away_allows=1.0, defenders=3):
    """
    One game, both directions. ``*_allows`` is points per matchup minute
    conceded BY that team, so a row for an offensive player on AWAY is defence
    by HOME.
    """
    out = []
    for off_team, allowed in ((AWAY, home_allows), (HOME, away_allows)):
        for d in range(defenders):
            out.append({
                "game_id": game, "team_id": off_team,
                "home_team_id": HOME, "away_team_id": AWAY,
                "team_tricode": CODES[off_team],
                "matchups_person_id": 9000 + d,
                "matchup_minutes": "10:00",
                "player_points": 10.0 * allowed,
                "matchup_field_goals_made": 4.0 * allowed,
                "matchup_field_goals_attempted": 10.0,
            })
    return out


def matchups(n_games=12, **kw):
    rows = []
    for i in range(n_games):
        rows.extend(rows_for_game(f"002250{i:04d}", **kw))
    return pd.DataFrame(rows)


def dates(n_games=12):
    return pd.DataFrame({
        "nba_game_id": [f"002250{i:04d}" for i in range(n_games)],
        "game_date": pd.date_range("2025-10-21", periods=n_games, freq="2D"),
    })


def panel(n_games=12):
    return pd.DataFrame({
        "PLAYER_ID": ["P"] * n_games,
        "GAME_ID": [f"002250{i:04d}" for i in range(n_games)],
        "GAME_DATE": pd.date_range("2025-10-21", periods=n_games, freq="2D"),
        "TEAM_ABBREVIATION": ["BBB"] * n_games,
        "OPPONENT_ABBREVIATION": ["AAA"] * n_games,
    })


# --- 1. the defending team is derived ----------------------------------

def test_the_defending_team_is_the_other_side_not_team_id():
    """
    `team_id` is the OFFENSIVE player's team. Reading it as the defender would
    attribute every team's own offence to its defence — a full column of
    plausible numbers describing the wrong thing.
    """
    mu = matchups(12, home_allows=2.0, away_allows=0.5)
    allowed = build_defender_allowed(mu, dates(12))
    by_team = allowed.groupby("_DEF_TEAM")["MU_PTS_PER_MIN_L10"].mean()
    # AAA (home) conceded 2.0/min; BBB (away) conceded 0.5/min.
    assert by_team["AAA"] == pytest.approx(2.0, abs=0.01)
    assert by_team["BBB"] == pytest.approx(0.5, abs=0.01)


def test_the_join_gives_a_player_his_opponents_defence_not_his_own_teams():
    panel_df = panel(12)        # player on BBB, facing AAA
    mu = matchups(12, home_allows=2.0, away_allows=0.5)
    allowed = build_defender_allowed(mu, dates(12))
    out = attach_matchup_features(panel_df, allowed, required=True)
    known = out["MU_PTS_PER_MIN_L10"].dropna()
    assert not known.empty
    assert known.iloc[-1] == pytest.approx(2.0, abs=0.01), "got BBB's own defence"


# --- 2. the shift, over GAMES ------------------------------------------

def test_tonights_concession_is_never_in_tonights_number():
    mu = matchups(12)
    spiked = mu.copy()
    target = spiked["game_id"] == "0022500008"
    spiked.loc[target, "player_points"] = 99.0

    base = build_defender_allowed(mu, dates(12)).set_index(["_DEF_TEAM", "_GAME"])
    after = build_defender_allowed(spiked, dates(12)).set_index(["_DEF_TEAM", "_GAME"])
    key = ("AAA", "22500008")
    assert base.loc[key, "MU_PTS_PER_MIN_L10"] == pytest.approx(
        after.loc[key, "MU_PTS_PER_MIN_L10"], nan_ok=True
    )
    # And it DOES reach a later game, or the layer would be inert.
    later = ("AAA", "22500009")
    assert after.loc[later, "MU_PTS_PER_MIN_L10"] > base.loc[later, "MU_PTS_PER_MIN_L10"]


def test_a_teams_first_games_of_a_season_have_no_prior_form():
    allowed = build_defender_allowed(matchups(12), dates(12))
    aaa = allowed[allowed["_DEF_TEAM"] == "AAA"].sort_values("game_date")
    assert aaa["MU_PTS_PER_MIN_L10"].head(ROLL_MIN_PERIODS).isna().all()


def test_the_window_counts_games_not_matchup_rows():
    """
    ~94 matchup rows per team-game in the real export, so a window over ROWS
    would span about one game. dvp.py documents finding exactly that defect in
    its own source. With 3 defenders per side here, a row-window of 10 would
    cover ~3 games; a game-window covers 10.
    """
    allowed = build_defender_allowed(matchups(20), dates(20))
    aaa = allowed[allowed["_DEF_TEAM"] == "AAA"]
    # One row per GAME, not per matchup row.
    assert len(aaa) == 20


# --- 3. ratios of totals -----------------------------------------------

def test_a_brief_assignment_does_not_weigh_the_same_as_a_long_one():
    """
    Eleven seconds on a star and twenty minutes on him are not equal evidence.
    A mean of per-matchup ratios would treat them as such.
    """
    rows = []
    for i in range(12):
        g = f"002250{i:04d}"
        base = {
            "game_id": g, "team_id": AWAY, "home_team_id": HOME,
            "away_team_id": AWAY, "team_tricode": "BBB",
            "matchup_field_goals_attempted": 10.0,
            "matchup_field_goals_made": 4.0,
        }
        # 20 minutes conceding 1.0/min, and 11 seconds conceding 10.0/min.
        rows.append({**base, "matchups_person_id": 1,
                     "matchup_minutes": "20:00", "player_points": 20.0})
        rows.append({**base, "matchups_person_id": 2,
                     "matchup_minutes": "0:11", "player_points": 1.833})
        rows.extend(rows_for_game(g)[3:])   # the other side
    allowed = build_defender_allowed(pd.DataFrame(rows), dates(12))
    aaa = allowed[allowed["_DEF_TEAM"] == "AAA"]["MU_PTS_PER_MIN_L10"].dropna()
    assert not aaa.empty
    # Ratio of totals: 21.833 pts / 20.183 min = 1.08. A mean of ratios would
    # be (1.0 + 10.0) / 2 = 5.5.
    assert aaa.iloc[-1] == pytest.approx(1.08, abs=0.05)
    assert aaa.iloc[-1] < 2.0, "a mean of per-matchup ratios would be ~5.5"


# --- 4. the clock-string parse -----------------------------------------

@pytest.mark.parametrize(("raw", "expected"), [
    ("0:00", 0.0), ("0:30", 0.5), ("1:00", 1.0), ("10:00", 10.0),
    ("2:57", 2.95), ("12.5", 12.5),
])
def test_matchup_minutes_parse_from_the_clock_string(raw, expected):
    assert parse_matchup_minutes(raw) == pytest.approx(expected, abs=0.01)


@pytest.mark.parametrize("raw", [None, np.nan, "", "nan", "none", "abc", ":"])
def test_an_unparseable_clock_is_nan_rather_than_zero(raw):
    """
    Zero minutes is a real value meaning "assigned but never on court
    together"; a null means "unknown". A pd.to_numeric would null the WHOLE
    column and the layer would abstain on every row while looking as though it
    had simply found no data.
    """
    assert np.isnan(parse_matchup_minutes(raw))


# --- 5. the as-of league baseline --------------------------------------

def test_the_index_baseline_cannot_see_the_rest_of_the_season():
    """A season-wide mean folds games that have not been played into an
    October index — the look-ahead defense.py records fixing."""
    drift = []
    for i in range(18):
        drift.extend(rows_for_game(f"002250{i:04d}", home_allows=1.0 + 0.1 * i))
    mu = pd.DataFrame(drift)
    full = build_defender_allowed(mu, dates(18))
    early = build_defender_allowed(
        mu[mu["game_id"] <= "0022500011"], dates(12)
    )
    keys = ["_DEF_TEAM", "_GAME"]
    merged = full.merge(early, on=keys, suffixes=("_f", "_e"))
    both = (merged["MU_PTS_PER_MIN_INDEX_L10_f"].notna()
            & merged["MU_PTS_PER_MIN_INDEX_L10_e"].notna())
    assert both.sum() > 0
    pd.testing.assert_series_equal(
        merged.loc[both, "MU_PTS_PER_MIN_INDEX_L10_f"],
        merged.loc[both, "MU_PTS_PER_MIN_INDEX_L10_e"],
        check_names=False,
    )


# --- 6. only the measured-useful half ships ----------------------------

def test_only_the_points_per_minute_pair_is_shippable():
    """
    Measured on the full panel with nine seasons attached: the FG% pair
    correlates 0.91-0.96 with DEF_FG_PCT_ALLOWED_L10, inside the band
    labels._EXCLUDED_AS_REDUNDANT was built from. The PTS_PER_MIN pair sits at
    0.74-0.77 against DEF_RATING_L10 and is outside it.
    """
    assert set(MATCHUP_SHIPPABLE_COLS) == {
        "MU_PTS_PER_MIN_L10", "MU_PTS_PER_MIN_INDEX_L10"
    }
    assert not any("FG_PCT" in c for c in MATCHUP_SHIPPABLE_COLS)
    # The FG% columns are still COMPUTED, so the measurement stays reproducible.
    assert any("FG_PCT" in c for c in MATCHUP_FEATURE_COLS)


def test_the_feature_ab_arm_tests_only_what_would_ship():
    import importlib.util

    spec = importlib.util.spec_from_file_location("_fab", "scripts/feature_ab.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert set(module.LAYERS["matchup"].columns) == set(MATCHUP_SHIPPABLE_COLS)


def test_no_market_reads_a_matchup_column_yet():
    from src.models.labels import default_feature_cols

    for market in ("PTS", "REB", "AST", "PRA", "FG3M", "STL", "BLK", "MIN"):
        assert not set(default_feature_cols(market)) & set(MATCHUP_FEATURE_COLS), market


# --- shape and refusal -------------------------------------------------

def test_the_join_never_changes_the_row_count():
    p = panel(12)
    out = attach_matchup_features(p, build_defender_allowed(matchups(12), dates(12)),
                                  required=True)
    assert len(out) == len(p)


def test_an_export_missing_a_required_column_is_refused_by_name():
    bad = matchups(4).drop(columns=["player_points"])
    with pytest.raises(MatchupFeatureError, match="player_points"):
        build_defender_allowed(bad, dates(4))


def test_a_date_frame_sharing_no_game_is_refused_with_the_reason():
    other = dates(4).assign(nba_game_id=["0099900000", "0099900001",
                                         "0099900002", "0099900003"])
    with pytest.raises(MatchupFeatureError, match="no matchup game id"):
        build_defender_allowed(matchups(4), other)


def test_no_table_means_no_columns_rather_than_zeros():
    p = panel(4)
    out = attach_matchup_features(p, None)
    assert not set(MATCHUP_FEATURE_COLS) & set(out.columns)
    pd.testing.assert_frame_equal(out, p)


def test_an_empty_panel_is_not_a_crash():
    assert attach_matchup_features(pd.DataFrame(), None).empty


def test_a_duplicated_lookup_row_is_refused_rather_than_multiplying_rows():
    allowed = build_defender_allowed(matchups(12), dates(12))
    doubled = pd.concat([allowed, allowed.head(1)], ignore_index=True)
    with pytest.raises(MatchupFeatureError, match="duplicate"):
        attach_matchup_features(panel(12), doubled, required=True)
