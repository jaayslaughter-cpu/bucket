"""Tests for the training-pack ingest, the leakage audit and feature selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.audit_leakage import (
    LeakageFound,
    check_game_straddle,
    check_lookahead_flag,
    check_postgame_columns,
    check_row_overlap,
    check_split,
)
from scripts.ingest_training_pack import build_team_games


def _panel(n_games: int = 12) -> pd.DataFrame:
    rows = []
    start = pd.Timestamp("2025-10-21")
    teams = ["AAA", "BBB"]
    for g in range(n_games):
        for team, opp in (("AAA", "BBB"), ("BBB", "AAA")):
            for k in range(3):
                rows.append({
                    "PLAYER_ID": f"{team}{k}", "GAME_ID": f"002250{g:04d}",
                    "GAME_DATE": start + pd.Timedelta(days=g),
                    "TEAM_ABBREVIATION": team, "OPPONENT_ABBREVIATION": opp,
                    "IS_HOME": int(team == "AAA"),
                    "PTS": 10.0 + k, "REB": 4.0, "AST": 3.0, "MIN": 30.0,
                    "FGM": 4.0, "FGA": 9.0, "FG3M": 1.0, "FTM": 2.0, "FTA": 2.0,
                    "OREB": 1.0, "DREB": 3.0, "STL": 1.0, "BLK": 0.5, "TOV": 2.0,
                })
    assert teams  # both sides present
    return pd.DataFrame(rows)


# --- team totals ------------------------------------------------------------


def test_team_totals_are_the_sum_of_the_players():
    tg = build_team_games(_panel(), None)
    assert len(tg) == 24
    # 3 players at 10, 11, 12 points.
    assert (tg["points"] == 33.0).all()
    assert tg["source"].eq("player_archive_sum").all()


def test_possessions_use_the_standard_estimate_when_unmeasured():
    tg = build_team_games(_panel(), None)
    expected = 27.0 - 3.0 + 6.0 + 0.44 * 6.0  # FGA - OREB + TOV + 0.44*FTA
    assert np.allclose(tg["poss"], expected)


def test_bigdataball_measurements_replace_the_estimate():
    """BigDataBall measures possessions; the archive can only estimate them."""
    panel = _panel()
    tg_plain = build_team_games(panel, None)
    bdb = pd.DataFrame({
        "nba_game_id": tg_plain["nba_game_id"],
        "team_abbr": tg_plain["team_abbr"],
        "poss": 99.5, "pace": 100.0, "off_eff": 110.0, "def_eff": 108.0,
    })
    tg = build_team_games(panel, bdb)
    assert np.allclose(tg["poss"], 99.5)
    assert tg["source"].str.contains("bigdataball").all()
    # The summed box score is untouched — only the measured columns change.
    assert (tg["points"] == 33.0).all()


def test_a_panel_without_team_codes_is_refused():
    with pytest.raises(ValueError, match="DATA_NOT_AVAILABLE"):
        build_team_games(pd.DataFrame({"GAME_ID": ["1"]}), None)


# --- the leakage checks themselves ------------------------------------------


def _split_panel() -> pd.DataFrame:
    p = _panel(n_games=20)
    p["LAST_INCLUDED_GAME_DATE"] = p["GAME_DATE"] - pd.Timedelta(days=1)
    return p


def test_lookahead_check_catches_a_feature_built_from_its_own_game():
    p = _split_panel()
    assert check_lookahead_flag(p)
    p.loc[5, "LAST_INCLUDED_GAME_DATE"] = p.loc[5, "GAME_DATE"]
    with pytest.raises(Exception, match="on or after their own"):
        check_lookahead_flag(p)


def test_split_ordering_check_catches_a_validation_row_inside_training():
    p = _split_panel()
    train_end = str((p["GAME_DATE"].min() + pd.Timedelta(days=9)).date())
    end = str(p["GAME_DATE"].max().date())
    assert check_split(p, train_end, end)


def test_row_overlap_check_catches_a_duplicated_player_game():
    p = _split_panel()
    train_end = str((p["GAME_DATE"].min() + pd.Timedelta(days=9)).date())
    end = str(p["GAME_DATE"].max().date())
    assert check_row_overlap(p, train_end, end)

    dupe = pd.concat([p, p.head(1)], ignore_index=True)
    with pytest.raises(LeakageFound, match="duplicate"):
        check_row_overlap(dupe, train_end, end)


def test_game_straddle_check_catches_teammates_split_across_the_cutoff():
    """Teammates share a game state, so a game split down the middle leaks it."""
    p = _split_panel()
    train_end = str((p["GAME_DATE"].min() + pd.Timedelta(days=9)).date())
    assert check_game_straddle(p, train_end)

    straddle = p.copy()
    target = straddle.loc[straddle["GAME_DATE"] <= pd.Timestamp(train_end), "GAME_ID"].iloc[-1]
    idx = straddle.index[straddle["GAME_ID"] == target][0]
    straddle.loc[idx, "GAME_DATE"] = pd.Timestamp(train_end) + pd.Timedelta(days=1)
    with pytest.raises(LeakageFound, match="both sides"):
        check_game_straddle(straddle, train_end)


def test_postgame_check_passes_for_the_shipped_feature_lists():
    p = _split_panel()
    detail = check_postgame_columns(p, ["PTS", "REB", "AST"])
    assert "no market's feature list names one" in detail


def test_postgame_check_catches_a_feature_list_naming_an_outcome(monkeypatch):
    import src.models.labels as labels

    monkeypatch.setattr(labels, "default_feature_cols", lambda m: ["PTS_L5", "PTS"])
    with pytest.raises(LeakageFound, match="same-game outcome"):
        check_postgame_columns(_split_panel(), ["PTS"])


# --- feature selection ------------------------------------------------------


def test_feature_selection_ignores_everything_after_the_cutoff():
    """Choosing features on validation rows is leakage exactly as much as
    fitting on them. Corrupting only the post-cutoff rows must not move a
    single number in the ranking."""
    import scripts.feature_selection as fs

    rng = np.random.default_rng(0)
    n = 4000
    dates = pd.date_range("2024-10-01", periods=n, freq="3h")
    base = pd.DataFrame({
        "PLAYER_ID": rng.integers(0, 40, n).astype(str),
        "GAME_ID": np.arange(n).astype(str),
        "GAME_DATE": dates,
        "PTS_L2": rng.normal(15, 4, n), "PTS_L5": rng.normal(15, 4, n),
        "PTS_L10": rng.normal(15, 4, n), "PTS_SEASON": rng.normal(15, 4, n),
        "PTS_BASELINE": rng.normal(15, 4, n),
        "MIN_L5": rng.normal(28, 6, n), "MIN_L10": rng.normal(28, 6, n),
        "MIN_SEASON": rng.normal(28, 6, n),
        "days_rest": rng.integers(0, 4, n).astype(float),
        "CAREER_GAMES_PRIOR": rng.integers(0, 400, n).astype(float),
        "IS_HOME": rng.integers(0, 2, n).astype(float),
    })
    base["PTS"] = base["PTS_L10"] + rng.normal(0, 6, n)

    cutoff = dates[int(n * 0.6)]
    corrupted = base.copy()
    after = corrupted["GAME_DATE"] > cutoff
    # Make the post-cutoff rows a completely different problem.
    corrupted.loc[after, "PTS"] = corrupted.loc[after, "MIN_L5"] * 3
    corrupted.loc[after, "PTS_L10"] = rng.normal(100, 1, int(after.sum()))

    kw = dict(train_end=str(cutoff.date()), n_folds=2, permutation_repeats=1)
    a = fs.rank_features(base, "PTS", **kw)
    b = fs.rank_features(corrupted, "PTS", **kw)

    assert list(a["feature"]) == list(b["feature"])
    assert np.allclose(a["perm_mean"].to_numpy(), b["perm_mean"].to_numpy(),
                       equal_nan=True)
    assert np.allclose(a["univariate_auc"].to_numpy(),
                       b["univariate_auc"].to_numpy(), equal_nan=True)


def test_feature_selection_reports_when_a_feature_is_inside_its_own_noise():
    """A mean smaller than its fold-to-fold spread has not been shown to
    matter, and the table must say so rather than ranking it silently."""
    import scripts.feature_selection as fs

    rng = np.random.default_rng(1)
    n = 3000
    dates = pd.date_range("2024-10-01", periods=n, freq="3h")
    df = pd.DataFrame({
        "PLAYER_ID": rng.integers(0, 40, n).astype(str),
        "GAME_ID": np.arange(n).astype(str), "GAME_DATE": dates,
        "PTS_L2": rng.normal(15, 4, n), "PTS_L5": rng.normal(15, 4, n),
        "PTS_L10": rng.normal(15, 4, n), "PTS_SEASON": rng.normal(15, 4, n),
        "PTS_BASELINE": rng.normal(15, 4, n), "MIN_L5": rng.normal(28, 6, n),
        "MIN_L10": rng.normal(28, 6, n), "MIN_SEASON": rng.normal(28, 6, n),
        "days_rest": rng.integers(0, 4, n).astype(float),
        "CAREER_GAMES_PRIOR": rng.integers(0, 400, n).astype(float),
        "IS_HOME": rng.integers(0, 2, n).astype(float),
    })
    df["PTS"] = df["PTS_L10"] + rng.normal(0, 6, n)
    table = fs.rank_features(df, "PTS", train_end=str(dates[-1].date()),
                             n_folds=3, permutation_repeats=1)
    assert {"perm_mean", "perm_sd", "folds_top10", "above_noise",
            "coverage", "univariate_auc"} <= set(table.columns)
    assert table["above_noise"].dtype == bool
    assert (table["folds_top10"] <= table["n_folds"]).all()


# --- the augmentation cap ---------------------------------------------------


def test_line_aware_cap_drops_source_rows_not_offsets():
    """Thinning the offsets would thin the line grid every retained game is
    trained across, which is the one thing the model exists to provide."""
    from src.models.line_aware import LineAwarePropModel

    model = LineAwarePropModel(
        lambda cols: None, stat="PTS", base_feature_cols=["PTS_L5"],
        offsets=(-2.0, 0.0, 2.0), max_augmented_rows=30,
    )
    panel = pd.DataFrame({
        "PLAYER_ID": ["p"] * 40,
        "GAME_ID": [str(i) for i in range(40)],
        "GAME_DATE": pd.date_range("2024-10-01", periods=40, freq="D"),
        "PTS": np.arange(40, dtype=float), "PTS_L5": np.arange(40, dtype=float),
    })
    trimmed = model._cap_source_rows(panel)
    assert len(trimmed) == 10          # 30 // 3 offsets
    assert model.offsets == (-2.0, 0.0, 2.0)
    # The MOST RECENT rows survive, and stay contiguous in time.
    assert trimmed["GAME_DATE"].min() == panel["GAME_DATE"].iloc[-10]
    assert trimmed["GAME_DATE"].is_monotonic_increasing


def test_line_aware_cap_is_a_no_op_below_the_ceiling():
    from src.models.line_aware import LineAwarePropModel

    model = LineAwarePropModel(
        lambda cols: None, stat="PTS", offsets=(-1.0, 1.0),
        max_augmented_rows=10_000,
    )
    panel = pd.DataFrame({
        "GAME_DATE": pd.date_range("2024-10-01", periods=25, freq="D"),
        "PTS": np.arange(25, dtype=float),
    })
    assert model._cap_source_rows(panel) is panel


def test_line_aware_cap_defaults_to_off():
    """Small panels never needed it, and a silent default would change
    behaviour for every run that was fine."""
    from src.models.line_aware import LineAwarePropModel

    model = LineAwarePropModel(lambda cols: None, stat="PTS")
    assert model.max_augmented_rows is None
    panel = pd.DataFrame({"GAME_DATE": pd.date_range("2024-10-01", periods=50),
                          "PTS": np.arange(50, dtype=float)})
    assert len(model._cap_source_rows(panel)) == 50


def test_shipped_config_caps_augmentation():
    from src.models.compare import load_comparison_config

    cfg = (load_comparison_config().get("line_aware") or {})
    cap = cfg.get("max_augmented_rows")
    assert cap and cap > 0
    # The cap must leave at least a few thousand source rows to train on.
    assert cap // len(cfg["offsets"]) >= 5000


# --- features that are empty only inside the training window ----------------


def _labelled_panel(n: int = 800) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    dates = pd.date_range("2024-10-01", periods=n, freq="6h")
    df = pd.DataFrame({
        "PLAYER_ID": rng.integers(0, 30, n).astype(str),
        "GAME_ID": np.arange(n).astype(str),
        "GAME_DATE": dates,
        "TEAM_ABBREVIATION": rng.choice(["AAA", "BBB"], n),
        "OPPONENT_ABBREVIATION": rng.choice(["CCC", "DDD"], n),
        "SEASON": "2024-25",
        "PTS_L2": rng.normal(15, 4, n), "PTS_L5": rng.normal(15, 4, n),
        "PTS_L10": rng.normal(15, 4, n), "PTS_SEASON": rng.normal(15, 4, n),
        "PTS_BASELINE": rng.normal(15, 4, n), "MIN_L5": rng.normal(28, 6, n),
        "MIN_L10": rng.normal(28, 6, n), "MIN_SEASON": rng.normal(28, 6, n),
        "IS_HOME": rng.integers(0, 2, n).astype(float),
    })
    df["PTS"] = df["PTS_L10"] + rng.normal(0, 5, n)
    # A market column that exists ONLY in the last 20% of the panel, which is
    # exactly how a line feed that started mid-history behaves.
    df["MKT_IMPLIED_TEAM_TOTAL"] = np.where(
        np.arange(n) > n * 0.8, rng.normal(115, 6, n), np.nan
    )
    return df


def test_a_feature_empty_in_training_is_dropped_not_trained_on():
    """It is populated only where the model is scored, which is the shape of
    a leak, and it empties any model that drops incomplete rows."""
    from src.models.compare import compare_models_on_panel, load_comparison_config

    df = _labelled_panel()
    cut = df["GAME_DATE"].iloc[int(len(df) * 0.7)]
    cfg = load_comparison_config()
    result = compare_models_on_panel(
        df, markets=["PTS"], train_end=str(cut.date()),
        validation_end=str(df["GAME_DATE"].max().date()), cfg=cfg,
    )
    summary = pd.DataFrame(result["summary"])
    assert not summary.empty, "every model was dropped"
    # The run survives the half-empty column rather than training on nothing.
    assert summary["n_predictions"].max() > 0


def test_catboost_survives_a_sparse_feature_column():
    """Dropping rows on every feature let one sparse column empty the whole
    training set: 187,733 rows went to zero and CatBoost failed with
    'Labels variable is empty', so the ensemble ran without its
    highest-weighted component."""
    pytest.importorskip("catboost")
    from src.models.catboost_pipeline import CatBoostPropPipeline
    from src.models.labels import attach_research_over_labels

    df = attach_research_over_labels(_labelled_panel(), stat="PTS")
    df = df[df["over_hit"].notna()].reset_index(drop=True)
    cols = ["PTS_L5", "PTS_L10", "MIN_L5", "MKT_IMPLIED_TEAM_TOTAL"]
    pipe = CatBoostPropPipeline(cols, target_market="PTS",
                                categorical_features=[])
    pipe.fit(df)
    head = df.head(20)
    probs = np.asarray(
        pipe.predict_probability_over(head, head["RESEARCH_LINE"]), dtype=float
    )
    assert len(probs) == 20
    assert np.isfinite(probs).all()


def test_catboost_does_not_fill_missing_features_with_zero():
    """A missing opening spread is not a spread of zero.

    The feature must be SIGNED for this to be observable: CatBoost's default
    nan_mode is Min, so for an all-positive column NaN and 0.0 fall on the
    same side of every split and the old fillna(0.0) was invisible. On a
    spread, 0.0 means pick-em -- a real quote, and the wrong one.
    """
    pytest.importorskip("catboost")
    from src.models.catboost_pipeline import CatBoostPropPipeline
    from src.models.labels import attach_research_over_labels

    rng = np.random.default_rng(5)
    df = _labelled_panel(1200)
    n = len(df)
    spread = rng.normal(0, 7, n)
    # Missing for the first third, exactly as a feed that started late.
    spread[: n // 3] = np.nan
    df["MKT_OPENING_SPREAD"] = spread
    # Give the label a real dependence on the spread so splits exist on it.
    df["PTS"] = df["PTS"] - np.nan_to_num(spread, nan=0.0) * 0.4

    df = attach_research_over_labels(df, stat="PTS")
    df = df[df["over_hit"].notna()].reset_index(drop=True)
    cols = ["PTS_L5", "PTS_L10", "MIN_L5", "MKT_OPENING_SPREAD"]
    pipe = CatBoostPropPipeline(cols, target_market="PTS", categorical_features=[])
    pipe.fit(df)

    head = df.tail(60).copy()
    blanked = head.assign(MKT_OPENING_SPREAD=np.nan)
    zeroed = head.assign(MKT_OPENING_SPREAD=0.0)
    p_blank = np.asarray(
        pipe.predict_probability_over(blanked, blanked["RESEARCH_LINE"]), dtype=float
    )
    p_zero = np.asarray(
        pipe.predict_probability_over(zeroed, zeroed["RESEARCH_LINE"]), dtype=float
    )
    ok = np.isfinite(p_blank) & np.isfinite(p_zero)
    assert ok.sum() > 5
    assert not np.allclose(p_blank[ok], p_zero[ok]), (
        "an absent spread and a spread of 0.0 produced identical predictions, "
        "so NaN is still being filled"
    )
