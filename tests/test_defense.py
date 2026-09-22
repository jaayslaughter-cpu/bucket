"""Tests for the opponent-defence layer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.defense import (
    DEFENSE_FEATURE_COLS,
    DefenseFeatureError,
    attach_defense_features,
    build_team_defense,
)


def _team_games(n_games: int = 40, teams=("AAA", "BBB", "CCC", "DDD")) -> pd.DataFrame:
    """Two rows per game. BBB is a deliberately stingy defence, DDD a leaky one.

    Shooting allowed is given its OWN per-team level, in a different order
    from points allowed. Deriving fg from points would make FG%-allowed an
    exact multiple of the defensive rating, and a collinearity guard would
    then fail on the fixture rather than on the code. On the real 2025-26
    panel those two correlate at 0.830 — related, not identical.
    """
    rng = np.random.default_rng(11)
    allowed_by = {"AAA": 112.0, "BBB": 100.0, "CCC": 114.0, "DDD": 126.0}
    fg_pct_by = {"AAA": 0.49, "BBB": 0.46, "CCC": 0.43, "DDD": 0.47}
    # Every allowed stat gets its own per-team level, each in a different
    # order. A constant (reb = 44.0 for all teams) makes rebounds-per-100 a
    # pure function of pace, and the guard then fails on the fixture.
    reb_by = {"AAA": 46.0, "BBB": 41.0, "CCC": 48.0, "DDD": 43.0}
    ast_by = {"AAA": 23.0, "BBB": 27.0, "CCC": 22.0, "DDD": 28.0}
    tov_by = {"AAA": 15.0, "BBB": 12.0, "CCC": 16.0, "DDD": 13.0}
    fg3_by = {"AAA": 12.0, "BBB": 15.0, "CCC": 11.0, "DDD": 14.0}
    rows = []
    start = pd.Timestamp("2025-10-20")
    for g in range(n_games):
        home, away = teams[g % 4], teams[(g + 1 + g // 4) % 4]
        if home == away:
            away = teams[(g + 2) % 4]
        poss = float(rng.uniform(95, 106))
        for team, opp in ((home, away), (away, home)):
            # This team SCORES what its opponent allows, per 100, times poss.
            pts = allowed_by[opp] / 100.0 * poss + rng.normal(0, 3)
            rows.append({
                "nba_game_id": f"00{g:06d}", "game_date": start + pd.Timedelta(days=g),
                "team_abbr": team, "opponent_abbr": opp,
                "points": pts, "poss": poss,
                "fga": poss * 0.85,
                "fg": poss * 0.85 * (fg_pct_by[opp] + rng.normal(0, 0.01)),
                "fg3": fg3_by[opp] + rng.normal(0, 1.0),
                "reb": reb_by[opp] + rng.normal(0, 2.0),
                "ast": ast_by[opp] + rng.normal(0, 1.5),
                "tov": tov_by[opp] + rng.normal(0, 1.5),
            })
    return pd.DataFrame(rows)


def test_defensive_rating_recovers_the_true_ordering():
    d = build_team_defense(_team_games())
    last = d.dropna(subset=["DEF_RATING_L10"]).sort_values("game_date").groupby(
        "team_abbr").tail(1).set_index("team_abbr")["DEF_RATING_L10"]
    assert last["BBB"] < last["AAA"] < last["CCC"] < last["DDD"]


def test_rates_are_per_possession_not_per_game():
    """A team that plays fast must not look like a bad defence for it."""
    tg = _team_games()
    fast = tg.copy()
    fast["poss"] = fast["poss"] * 1.25
    # Same defensive quality per 100, so the same points allowed per 100.
    for col in ("points", "fg", "fg3", "fga", "reb", "ast", "tov"):
        fast[col] = fast[col] * 1.25

    slow_rating = build_team_defense(tg)["DEF_RATING_L10"]
    fast_rating = build_team_defense(fast)["DEF_RATING_L10"]
    assert np.allclose(
        slow_rating.to_numpy(), fast_rating.to_numpy(), equal_nan=True, rtol=1e-9
    )
    # ...while the pace column DOES move, which is the point of splitting them.
    assert build_team_defense(fast)["DEF_PACE_L10"].mean() > (
        build_team_defense(tg)["DEF_PACE_L10"].mean() * 1.2
    )


def test_no_row_sees_its_own_game():
    d = build_team_defense(_team_games())
    merged = d.merge(
        _team_games()[["nba_game_id", "team_abbr", "points"]],
        on=["nba_game_id", "team_abbr"], how="left",
    )
    # First ROLL_MIN_PERIODS games of a team have no prior form at all.
    assert merged["DEF_RATING_L10"].isna().sum() > 0


def test_deleting_later_games_never_moves_an_earlier_value():
    """The league baseline must be as-of, not season-wide. A season-wide mean
    folds games that have not been played into an early-season index."""
    tg = _team_games(n_games=60)
    full = build_team_defense(tg)
    cut = tg["game_date"].quantile(0.6)
    trunc = build_team_defense(tg[tg["game_date"] < cut].copy())

    key = ["nba_game_id", "team_abbr"]
    cols = ["DEF_RATING_L10", "DEF_RATING_INDEX_L10", "DEF_PACE_L10"]
    j = full.merge(trunc[key + cols], on=key, suffixes=("_f", "_t"))
    assert len(j) > 0
    for c in cols:
        delta = (j[f"{c}_f"] - j[f"{c}_t"]).abs()
        assert (delta.fillna(0.0) < 1e-9).all(), f"{c} depends on future games"


def test_index_is_centred_on_the_league():
    d = build_team_defense(_team_games(n_games=80))
    idx = d["DEF_RATING_INDEX_L10"].dropna()
    assert 0.9 < idx.mean() < 1.1
    assert idx.min() < 1.0 < idx.max()


def test_possessions_are_never_invented():
    tg = _team_games().drop(columns=["poss", "fga", "tov"])
    with pytest.raises(DefenseFeatureError, match="DATA_NOT_AVAILABLE"):
        build_team_defense(tg)


def test_possessions_fall_back_to_the_standard_estimate():
    tg = _team_games().drop(columns=["poss"])
    tg["oreb"] = 10.0
    tg["fta"] = 22.0
    out = build_team_defense(tg)
    assert out["DEF_PACE_L10"].notna().any()


def test_empty_or_incomplete_team_games_is_refused():
    with pytest.raises(DefenseFeatureError, match="empty"):
        build_team_defense(pd.DataFrame())
    with pytest.raises(DefenseFeatureError, match="missing"):
        build_team_defense(pd.DataFrame({"nba_game_id": ["1"], "team_abbr": ["AAA"]}))


def test_a_game_with_only_one_team_has_no_opponent():
    tg = _team_games()
    solo = tg.groupby("nba_game_id").head(1)
    with pytest.raises(DefenseFeatureError, match="two distinct teams"):
        build_team_defense(solo)


# --- the join ---------------------------------------------------------------


def _panel(tg: pd.DataFrame | None = None):
    """One player per side of a real game from the fixture, so the join keys
    always match what build_team_defense actually produced."""
    tg = _team_games() if tg is None else tg
    game = tg[tg["nba_game_id"] == tg["nba_game_id"].iloc[0]]
    return pd.DataFrame({
        "PLAYER_ID": [f"p{i}" for i in range(len(game))],
        "GAME_ID": game["nba_game_id"].tolist(),
        "TEAM_ABBREVIATION": game["team_abbr"].tolist(),
        "OPPONENT_ABBREVIATION": game["opponent_abbr"].tolist(),
    })


def test_the_join_uses_the_opponent_not_the_player_s_own_team():
    """Joining on TEAM_ABBREVIATION would hand the model its own team's
    defence and still fill every row with plausible numbers."""
    tg = _team_games()
    panel = _panel(tg)
    home, away = panel["TEAM_ABBREVIATION"].iloc[0], panel["TEAM_ABBREVIATION"].iloc[1]

    d = build_team_defense(tg).copy()
    # Stamp each team's defence with an identifiable value.
    d["DEF_RATING_L10"] = np.where(d["team_abbr"] == home, 100.0, 130.0)

    out = attach_defense_features(panel, d)
    got = out.set_index("TEAM_ABBREVIATION")["DEF_RATING_L10"]
    # The home player faces the away defence (130), not their own (100).
    assert got[home] == 130.0
    assert got[away] == 100.0


def test_unmatched_rows_stay_null_rather_than_league_average():
    d = build_team_defense(_team_games())
    panel = _panel().assign(OPPONENT_ABBREVIATION=["ZZZ", "ZZZ"])
    out = attach_defense_features(panel, d)
    assert out["DEF_RATING_L10"].isna().all()


def test_a_duplicated_team_game_is_refused_rather_than_multiplying_rows():
    d = build_team_defense(_team_games())
    dupe = pd.concat([d, d.head(len(d))], ignore_index=True)
    with pytest.raises(DefenseFeatureError, match="row count"):
        attach_defense_features(_panel(), dupe)


def test_missing_opponent_column_skips_or_raises():
    d = build_team_defense(_team_games())
    bare = pd.DataFrame({"PLAYER_ID": ["p1"], "GAME_ID": ["1"]})
    assert "DEF_RATING_L10" not in attach_defense_features(bare, d).columns
    with pytest.raises(DefenseFeatureError, match="DATA_NOT_AVAILABLE"):
        attach_defense_features(bare, d, required=True)


# --- reachability -----------------------------------------------------------


def test_defensive_columns_reach_the_feature_list():
    from src.models.labels import default_feature_cols

    for market in ("PTS", "REB", "AST", "FG3M", "STL", "BLK", "PRA"):
        cols = set(default_feature_cols(market))
        assert {"DEF_RATING_L10", "DEF_PACE_L10"} <= cols


def test_each_market_gets_the_rate_that_bears_on_it():
    from src.models.labels import default_feature_cols

    # Points allowed per 100 IS the defensive rating; emitting both shipped
    # the same number twice and cost a third of the layer's measured gain.
    assert "DEF_RATING_L10" in default_feature_cols("PTS")
    assert "DEF_PTS_ALLOWED_PER100_L10" not in default_feature_cols("PTS")
    assert "DEF_FG3M_ALLOWED_PER100_L10" in default_feature_cols("FG3M")
    # Steals come from opponent turnovers, not the opponent's own steals.
    assert "DEF_TOV_FORCED_PER100_L10" in default_feature_cols("STL")
    assert not [c for c in default_feature_cols("STL") if "STL_ALLOWED" in c]
    # Blocks need shot volume.
    assert "DEF_FGA_ALLOWED_PER100_L10" in default_feature_cols("BLK")


def test_feature_list_never_names_a_postgame_column():
    from src.models.compare import POSTGAME_ONLY_COLS
    from src.models.labels import default_feature_cols

    for market in ("PTS", "REB", "AST", "FG3M", "STL", "BLK", "PRA"):
        assert not set(default_feature_cols(market)) & POSTGAME_ONLY_COLS


def test_ab_harness_knows_the_defense_layer():
    from scripts.feature_ab import LAYERS

    assert set(LAYERS["defense"].columns) <= set(DEFENSE_FEATURE_COLS)
    with pytest.raises(RuntimeError, match="cannot attach"):
        LAYERS["defense"].arms(pd.DataFrame({"PLAYER_ID": ["a"]}), {})


def test_no_two_defensive_features_are_the_same_number(): 
    """Shipping points-allowed-per-100 beside the defensive rating handed the
    PTS model one number three times. Measured cost on a planted signal:
    Brier 0.24536 with the rating alone, 0.24793 with all four."""
    d = build_team_defense(_team_games(n_games=120))
    from src.models.labels import default_feature_cols

    for market in ("PTS", "REB", "AST", "FG3M", "STL", "BLK", "PRA"):
        cols = [c for c in default_feature_cols(market) if c in d.columns]
        sub = d[cols].dropna()
        if len(sub) < 20 or len(cols) < 2:
            continue
        corr = sub.corr().abs()
        pairs = corr.where(~np.eye(len(cols), dtype=bool)).stack()
        worst_pair = pairs.idxmax()
        assert pairs.max() < 0.95, (
            f"{market}: {worst_pair[0]} and {worst_pair[1]} correlate at "
            f"{pairs.max():.4f} — the model receives one number twice"
        )


def test_the_league_index_is_still_produced_for_reporting():
    """It is not a feature (r = 0.999 with the rating within a season), but a
    1.0-centred league-relative number is the readable one for a report."""
    d = build_team_defense(_team_games(n_games=80))
    assert "DEF_RATING_INDEX_L10" in d.columns
    assert d["DEF_RATING_INDEX_L10"].notna().any()
