"""The player's own prior foul history — src/features/fouls.py.

WHY THE LAYER EXISTS AT ALL is a data finding rather than a design idea: the
panel's source carries ``foulsPersonal`` on 304,395 of its 305,614
player-game rows, and nothing in this project read it. An earlier reading of
this tree concluded a foul feature was impossible here, on the grounds that
``PlayerGameLog`` has no ``pf`` column and no panel on disk carries ``PF``.
Both facts were true and the conclusion was wrong: the archive has the
column, the panel contract simply never mapped it.

What the tests below pin, in order:

  1. the shift — a row never sees its own game's fouls, which is the one
     thing that would make every other number here meaningless;
  2. the per-minute rate is a RATIO OF TOTALS, so a two-minute cameo with one
     foul does not dominate a ten-game window;
  3. foul trouble is a SHARE of recent games at five fouls, which is the tail
     the mean cannot express — 0-5-0-5-2 and 2-3-2-3-2 average alike;
  4. PF_SEASON stays inside the player-season;
  5. row order in, row order out, because the builder applies layers as
     ``df = attach(df)``;
  6. an absent or unusable PF produces NO columns rather than zeros.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.fouls import (
    FOUL_FEATURE_COLS,
    FOUL_TROUBLE_THRESHOLD,
    PF_PLAUSIBLE_MAX,
    FoulFeatureError,
    attach_foul_features,
    attach_foul_features_layer,
)


def panel(fouls, minutes=None, *, player="A", season="2025-26"):
    n = len(fouls)
    return pd.DataFrame({
        "PLAYER_ID": [player] * n,
        "SEASON": [season] * n,
        "GAME_DATE": pd.date_range("2025-10-21", periods=n, freq="2D"),
        "GAME_ID": [f"002250{i:04d}" for i in range(n)],
        "PF": list(fouls),
        "MIN": list(minutes) if minutes is not None else [30.0] * n,
        "PTS": [20.0] * n,
    })


# --- 1. the shift -------------------------------------------------------

def test_a_rows_own_fouls_never_reach_its_own_features():
    """
    The whole layer rests on this. A single enormous foul game is planted in
    the middle; every column on THAT row must be computed from the games
    before it and must not move when the planted value changes.
    """
    low = panel([1, 1, 1, 1, 1, 1])
    spiked = low.copy()
    spiked.loc[3, "PF"] = 6

    out_low = attach_foul_features(low, required=True)
    out_spiked = attach_foul_features(spiked, required=True)

    for col in FOUL_FEATURE_COLS:
        assert out_low.loc[3, col] == pytest.approx(
            out_spiked.loc[3, col], nan_ok=True
        ), f"{col} on row 3 moved when row 3's own PF changed"
    # And it DOES reach the next row, or the feature would be inert.
    assert out_spiked.loc[4, "PF_L5"] > out_low.loc[4, "PF_L5"]


def test_the_first_game_has_no_prior_history():
    out = attach_foul_features(panel([3, 2, 1]), required=True)
    assert pd.isna(out.loc[0, "PF_L5"])
    assert pd.isna(out.loc[0, "PF_L10"])
    assert pd.isna(out.loc[0, "PF_SEASON"])


def test_the_mean_is_over_prior_games_only():
    out = attach_foul_features(panel([0, 2, 4, 6]), required=True)
    assert out.loc[1, "PF_L5"] == pytest.approx(0.0)
    assert out.loc[2, "PF_L5"] == pytest.approx(1.0)
    assert out.loc[3, "PF_L5"] == pytest.approx(2.0)


# --- 2. the per-minute rate is a ratio of totals ------------------------

def test_the_per_minute_rate_is_a_ratio_of_totals_not_a_mean_of_ratios():
    """
    A two-minute cameo with one foul is a per-minute rate of 0.5, eight times
    any real player's propensity. A MEAN OF RATIOS over four games lets that
    one cameo carry a quarter of the weight; a ratio of totals weights each
    game by the minutes it contributed.

    Three 30-minute games at 1 foul, then a 2-minute game with 1 foul. The
    fifth row's window covers all four.
    """
    out = attach_foul_features(
        panel([1, 1, 1, 1, 1], minutes=[30, 30, 30, 2, 30]), required=True
    )
    ratio_of_totals = 4 / 92
    mean_of_ratios = np.mean([1 / 30, 1 / 30, 1 / 30, 1 / 2])
    assert out.loc[4, "PF_PER_MIN_L10"] == pytest.approx(ratio_of_totals)
    assert out.loc[4, "PF_PER_MIN_L10"] != pytest.approx(mean_of_ratios, abs=1e-3)


def test_the_rate_separates_two_players_the_per_game_mean_calls_identical():
    """
    A starter with 2 fouls in 34 minutes and a reserve with 2 in 14 have the
    same PF_L10 and very different propensities. If PF_PER_MIN_L10 did not
    separate them the column would add nothing to PF_L10.
    """
    starter = attach_foul_features(
        panel([2] * 6, minutes=[34] * 6, player="S"), required=True
    )
    reserve = attach_foul_features(
        panel([2] * 6, minutes=[14] * 6, player="R"), required=True
    )
    assert starter.loc[5, "PF_L10"] == pytest.approx(reserve.loc[5, "PF_L10"])
    assert reserve.loc[5, "PF_PER_MIN_L10"] > starter.loc[5, "PF_PER_MIN_L10"] * 2


def test_zero_prior_minutes_leaves_the_rate_unknown_rather_than_infinite():
    out = attach_foul_features(
        panel([0, 0, 0, 1], minutes=[0, 0, 0, 20]), required=True
    )
    assert pd.isna(out.loc[3, "PF_PER_MIN_L10"])


# --- 3. foul trouble is a share, not a mean -----------------------------

def test_foul_trouble_is_the_share_of_recent_games_at_five_fouls():
    out = attach_foul_features(panel([5, 1, 5, 1, 1, 1]), required=True)
    # Prior to row 3: 5, 1, 5 -> two of three.
    assert out.loc[3, "PF_TROUBLE_RATE_L10"] == pytest.approx(2 / 3)


def test_the_share_distinguishes_two_histories_with_the_same_mean():
    """
    0-5-0-5-2 and 2-3-2-3-2 both average 2.4 fouls. Only the first is a player
    a coach has had to sit, and only PF_TROUBLE_RATE_L10 says so.
    """
    spiky = attach_foul_features(panel([0, 5, 0, 5, 2, 1], player="X"), required=True)
    steady = attach_foul_features(panel([2, 3, 2, 3, 2, 1], player="Y"), required=True)
    assert spiky.loc[5, "PF_L5"] == pytest.approx(steady.loc[5, "PF_L5"])
    assert spiky.loc[5, "PF_TROUBLE_RATE_L10"] == pytest.approx(0.4)
    assert steady.loc[5, "PF_TROUBLE_RATE_L10"] == pytest.approx(0.0)


def test_the_threshold_is_five_not_six():
    """
    Six is the disqualification itself, by which point the minutes are gone.
    Five is the state a coach reacts to, so a history of fives must register.
    """
    assert FOUL_TROUBLE_THRESHOLD == 5
    out = attach_foul_features(panel([5, 5, 5, 1]), required=True)
    assert out.loc[3, "PF_TROUBLE_RATE_L10"] == pytest.approx(1.0)


# --- 4. the season scope ------------------------------------------------

def test_pf_season_does_not_carry_last_seasons_whistle():
    first = panel([6, 6, 6], season="2024-25")
    second = panel([0, 0, 0], season="2025-26")
    second["GAME_DATE"] = pd.date_range("2025-10-21", periods=3, freq="2D")
    first["GAME_DATE"] = pd.date_range("2024-10-21", periods=3, freq="2D")
    out = attach_foul_features(
        pd.concat([first, second], ignore_index=True), required=True
    )
    # Row 4 is the second game of the new season: one prior game, 0 fouls.
    assert out.loc[4, "PF_SEASON"] == pytest.approx(0.0)
    # PF_L10 is player-scoped by design and DOES see across the boundary.
    assert out.loc[4, "PF_L10"] > 0


def test_a_panel_without_season_still_produces_pf_season():
    """Player-scoped rather than refused: a panel with no SEASON is narrower,
    not unusable."""
    out = attach_foul_features(panel([1, 2, 3]).drop(columns=["SEASON"]), required=True)
    assert out.loc[2, "PF_SEASON"] == pytest.approx(1.5)


# --- 5. order in, order out ---------------------------------------------

def test_the_frame_comes_back_in_the_order_it_went_in():
    """
    The builder applies layers as ``df = attach(df)``, so a layer that
    returned a re-sorted frame would silently reorder the whole feature
    matrix for every later layer. minutes_weighted shipped exactly that bug.
    """
    frame = panel([1, 2, 3, 4, 5, 6])
    shuffled = frame.iloc[[4, 0, 3, 1, 5, 2]].reset_index(drop=True)
    out = attach_foul_features(shuffled, required=True)
    assert out["GAME_ID"].tolist() == shuffled["GAME_ID"].tolist()
    assert out["PF"].tolist() == shuffled["PF"].tolist()


def test_a_shuffled_frame_gets_the_same_numbers_as_a_sorted_one():
    """The rolling window must follow the dates, not the file order."""
    frame = panel([0, 1, 2, 3, 4, 5])
    shuffled = frame.iloc[[3, 1, 5, 0, 4, 2]].reset_index(drop=True)
    sorted_out = attach_foul_features(frame, required=True).set_index("GAME_ID")
    shuffled_out = attach_foul_features(shuffled, required=True).set_index("GAME_ID")
    for col in FOUL_FEATURE_COLS:
        pd.testing.assert_series_equal(
            sorted_out[col].sort_index(), shuffled_out[col].sort_index()
        )


def test_two_players_histories_do_not_bleed_into_each_other():
    a = panel([6, 6, 6], player="A")
    b = panel([0, 0, 0], player="B")
    out = attach_foul_features(pd.concat([a, b], ignore_index=True), required=True)
    assert out.loc[4, "PF_L5"] == pytest.approx(0.0)


# --- 6. absence and implausibility are not zero -------------------------

def test_an_absent_pf_column_adds_nothing_rather_than_zeros():
    """
    A column of zeros would read as a measured foul-free league, and the
    minutes-risk signal is built out of exactly this number.
    """
    frame = panel([1, 2, 3]).drop(columns=["PF"])
    out = attach_foul_features_layer(frame)
    assert not set(FOUL_FEATURE_COLS) & set(out.columns)
    pd.testing.assert_frame_equal(out, frame)


def test_required_turns_an_absent_pf_into_a_refusal():
    with pytest.raises(FoulFeatureError, match="DATA_NOT_AVAILABLE"):
        attach_foul_features(panel([1, 2, 3]).drop(columns=["PF"]), required=True)


def test_an_absent_player_id_is_refused_rather_than_guessed():
    with pytest.raises(FoulFeatureError, match="PLAYER_ID"):
        attach_foul_features(panel([1, 2, 3]).drop(columns=["PLAYER_ID"]), required=True)


@pytest.mark.parametrize("impossible", [-1, 7, 23, 100])
def test_a_count_that_cannot_be_a_personal_foul_is_unknown_not_clipped(impossible):
    """
    Clipping 23 to 6 would turn a column that is not personal fouls into one
    that looks like personal fouls. A team total arriving under this name is
    far likelier than a player committing 23 fouls.
    """
    frame = panel([1, 1, 1, 1, 1])
    frame.loc[1, "PF"] = impossible
    out = attach_foul_features(frame, required=True)
    # Row 2's window covers rows 0 and 1; the bad value is excluded, so the
    # mean is row 0's alone rather than a clipped PF_PLAUSIBLE_MAX.
    assert out.loc[2, "PF_L5"] == pytest.approx(1.0)
    assert out.loc[2, "PF_L5"] != pytest.approx((1 + PF_PLAUSIBLE_MAX) / 2)


def test_six_fouls_is_kept_because_six_is_possible():
    out = attach_foul_features(panel([6, 1, 1]), required=True)
    assert out.loc[1, "PF_L5"] == pytest.approx(6.0)


def test_an_empty_panel_is_not_a_crash():
    out = attach_foul_features_layer(pd.DataFrame())
    assert out.empty


def test_an_all_null_pf_leaves_every_feature_null_rather_than_zero():
    frame = panel([1, 2, 3, 4])
    frame["PF"] = np.nan
    out = attach_foul_features(frame, required=True)
    for col in FOUL_FEATURE_COLS:
        assert out[col].isna().all(), col


# --- the layer is built but deliberately not wired ----------------------

def test_no_market_reads_a_foul_column_yet():
    """
    Whether a foul history improves a projection is a measurement, not an
    assumption. scripts/feature_ab.py --layer fouls --wire-under-test is
    where it gets made; until it has been, these columns stay out of the
    contract. If this test starts failing because a measurement was made and
    the column was wired, delete it and record the numbers in labels.py.
    """
    from src.models.labels import default_feature_cols

    for market in ("PTS", "REB", "AST", "PRA", "FG3M", "STL", "BLK", "MIN"):
        listed = set(default_feature_cols(market))
        assert not listed & set(FOUL_FEATURE_COLS), market


def test_the_layer_is_registered_with_feature_ab_so_it_can_be_measured():
    """A layer with no arm cannot be measured, and an unmeasurable feature
    layer is how minutes_weighted sat unread for months."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("_fab", "scripts/feature_ab.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert "fouls" in module.LAYERS
    assert set(module.LAYERS["fouls"].columns) == set(FOUL_FEATURE_COLS)


# --- the database column ------------------------------------------------

def test_the_migration_adds_the_column_without_backfilling_a_zero():
    """
    Zero fouls is a specific, clean, low-risk game. Backfilling it would teach
    the minutes-risk feature that the whole history was foul-free, and the
    feature can abstain on a null but not on a zero. Rows written before
    boxscores.COLUMN_MAP began requesting PF have nothing to be filled FROM —
    what was never fetched is not recoverable from what was stored — so they
    stay null and a re-ingest fills what it covers.
    """
    from pathlib import Path

    sql = (
        Path(__file__).parent.parent / "migrations" / "006_player_game_log_fouls.sql"
    ).read_text(encoding="utf-8")
    # The comments explain why there is no default; the STATEMENTS are what
    # must not contain one, so they are checked on their own.
    statements = "\n".join(
        line for line in sql.splitlines() if not line.lstrip().startswith("--")
    )
    assert "ADD COLUMN IF NOT EXISTS pf" in statements
    assert "UPDATE player_game_logs" not in statements, "a backfill would fabricate"
    assert "DEFAULT" not in statements.upper()
    assert "NOT NULL" not in statements.upper()
    # The bound is what a personal-foul count can be, six being the
    # disqualification itself.
    assert "pf >= 0 AND pf <= 6" in statements
    assert "NOT VALID" in statements


def test_the_upsert_writes_pf_and_leaves_it_null_when_the_panel_has_none():
    """
    A source that does not report fouls must leave the column unknown. The
    assertion is on the mapping, because exercising the write needs Postgres.
    """
    import inspect

    from src.db import repository

    source = inspect.getsource(repository.upsert_player_game_logs)
    assert '"pf": pd.to_numeric(df.get("PF"), errors="coerce")' in source
    assert '"pf"' in source.split("astype(\"Int64\")")[0]


def test_the_loaded_panel_exposes_pf_so_the_layer_reaches_the_live_path():
    """
    A layer that only runs on a rebuilt archive panel is a research column.
    repository.load_player_panel is what the live path builds features from,
    so PF has to come back out of the table it was written to.
    """
    import inspect

    from src.db import repository

    source = inspect.getsource(repository.load_player_panel)
    assert '"PF": r.pf' in source
