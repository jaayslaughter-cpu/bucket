"""Shot location and type over the whole panel — src/features/shot_zones.py.

WHY THIS LAYER EXISTS is a coverage finding, and the first version of its
docstring overstated it. src/features/pbp.py says "The logs supplied cover
2025-26 only", and that line is STALE: counted on the real panel, the PBP_*
shot-mix family is populated for 2021-22 through 2025-26, 126,624 of 214,381
rows (59.1%). This layer reads the NBA's own shotchartdetail export instead
and reaches 209,566 rows (97.8%) — four more seasons and 39 points, not the
eight seasons the stale line implies.

What the tests below pin:

  1. the shift — a row never sees its own game's shot selection, which is the
     one thing that would make every other number here meaningless;
  2. the completeness gate, which can STOP a build, because pbp.py records two
     event logs that looked complete at 32% and 84% coverage;
  3. the id normalisation, because the export writes GAME_ID unpadded and the
     panel pads it, and that mismatch matches nothing while looking correct;
  4. zones read from the NBA's own labels rather than inferred from a distance
     threshold, and corner threes separated from above-the-break;
  5. the same-game columns do not escape onto the panel.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.features.shot_zones import (
    ROLL_MIN_PERIODS,
    SHOT_FEATURE_COLS,
    SUMMARY_COLS,
    ShotZoneError,
    attach_shot_rolling_features,
    check_shot_completeness,
    prepare_shots,
    summarise_player_games,
)


def shot(game, player, *, zone="Mid-Range", three=False, action="Jump Shot",
         dist=15, made=1):
    return {
        "GAME_ID": game, "PLAYER_ID": player, "SHOT_DISTANCE": dist,
        "SHOT_TYPE": "3PT Field Goal" if three else "2PT Field Goal",
        "SHOT_ZONE_BASIC": zone, "ACTION_TYPE": action, "SHOT_MADE_FLAG": made,
    }


def shots(rows):
    return pd.DataFrame(rows)


def panel(n_games=8, player="2000001", game_prefix="002250"):
    return pd.DataFrame({
        "PLAYER_ID": [player] * n_games,
        "GAME_ID": [f"{game_prefix}{i:04d}" for i in range(n_games)],
        "GAME_DATE": pd.date_range("2025-10-21", periods=n_games, freq="2D"),
        "FGA": [4] * n_games,
    })


def _even_shots(n_games=8, player="2000001", game_prefix="002250", per_game=4):
    rows = []
    for i in range(n_games):
        for _ in range(per_game):
            rows.append(shot(f"{game_prefix}{i:04d}", player))
    return shots(rows)


# --- 1. the shift -------------------------------------------------------

def test_a_rows_own_shot_selection_never_reaches_its_own_features():
    """
    The whole layer rests on this. One game's shot mix is replaced wholesale;
    every column on THAT row must not move.
    """
    base = _even_shots(n_games=8)
    spiked = base.copy()
    target = spiked["GAME_ID"] == "0022500004"
    spiked.loc[target, "SHOT_ZONE_BASIC"] = "Restricted Area"
    spiked.loc[target, "SHOT_DISTANCE"] = 1

    p = panel(8)
    a = attach_shot_rolling_features(p, summarise_player_games(base), required=True)
    b = attach_shot_rolling_features(p, summarise_player_games(spiked), required=True)
    row = p.index[p["GAME_ID"] == "0022500004"][0]
    for col in SHOT_FEATURE_COLS:
        assert a.loc[row, col] == pytest.approx(b.loc[row, col], nan_ok=True), col
    # And it DOES reach the next row, or the feature would be inert.
    later = p.index[p["GAME_ID"] == "0022500005"][0]
    assert b.loc[later, "SZ_RIM_RATE_L5"] > a.loc[later, "SZ_RIM_RATE_L5"]


def test_the_first_games_have_no_prior_mix():
    out = attach_shot_rolling_features(
        panel(8), summarise_player_games(_even_shots(8)), required=True
    )
    # min_periods is 3 prior games, so the first three rows cannot have one.
    assert out["SZ_RIM_RATE_L10"].head(ROLL_MIN_PERIODS).isna().all()
    assert out["SZ_RIM_RATE_L10"].iloc[ROLL_MIN_PERIODS:].notna().all()


def test_the_same_game_columns_do_not_escape_onto_the_panel():
    """
    They are joined only so the rolling can shift them. Leaving them behind
    would publish the shot selection of the game being predicted.
    """
    out = attach_shot_rolling_features(
        panel(8), summarise_player_games(_even_shots(8)), required=True
    )
    leaked = [c for c in SUMMARY_COLS if c in out.columns]
    assert not leaked, leaked
    assert "_GAME" not in out.columns
    assert "_PLAYER" not in out.columns


# --- 2. the completeness gate ------------------------------------------

def test_a_complete_log_passes_against_the_panels_own_fga():
    report = check_shot_completeness(_even_shots(8, per_game=4), panel(8))
    assert report["status"] == "OK"
    assert report["exact_share"] == pytest.approx(1.0)
    assert report["mean_abs_diff"] == pytest.approx(0.0)


def test_a_partial_log_is_refused_rather_than_averaged():
    """
    pbp.py records two logs that named every game, spanned the right dates,
    carried no duplicates and still held 32% and 84% of the events. Rates from
    a partial log look reasonable and are biased by whatever was dropped.
    """
    thin = _even_shots(8, per_game=4)
    # Drop one shot from most games: the count no longer matches FGA.
    thin = thin.groupby("GAME_ID", group_keys=False).head(3)
    report = check_shot_completeness(thin, panel(8))
    assert report["status"] == "DATA_NOT_AVAILABLE"
    assert report["exact_share"] < 0.98


def test_a_panel_with_no_fga_is_refused_not_trusted():
    with pytest.raises(ShotZoneError, match="no FGA"):
        check_shot_completeness(_even_shots(4), panel(4).drop(columns=["FGA"]))


def test_a_player_who_took_no_shots_is_not_evidence_of_a_gap():
    """He is absent from the log rather than present with a zero, so his row
    cannot disagree and must not count against the log."""
    p = panel(8)
    p.loc[p.index[:4], "FGA"] = 0
    report = check_shot_completeness(_even_shots(8, per_game=4), p)
    assert report["status"] == "OK"
    assert report["compared"] == 4


def test_ids_that_look_different_are_still_matched():
    """
    The export writes GAME_ID as an integer and the panel zero-pads it.
    Joining those matches nothing while looking entirely reasonable.
    """
    log = _even_shots(4, per_game=4)
    log["GAME_ID"] = log["GAME_ID"].str.lstrip("0").astype(int)  # 22500000
    report = check_shot_completeness(log, panel(4))
    assert report["compared"] == 4
    assert report["exact_share"] == pytest.approx(1.0)


def test_a_log_sharing_no_game_with_the_panel_says_so():
    log = _even_shots(4, game_prefix="009990")
    report = check_shot_completeness(log, panel(4))
    assert report["status"] == "DATA_NOT_AVAILABLE"
    assert "id formats" in report["reason"]


# --- 3. zones come from the NBA's labels, not a distance cut -----------

def test_a_corner_three_is_separated_from_above_the_break():
    """
    A corner three is the shortest three on the floor and the one most
    dependent on a teammate's pass. Nothing in this project separated it, and
    a substring match on "3" would merge the two.
    """
    rows = [
        shot("1", "P", zone="Left Corner 3", three=True, dist=22),
        shot("1", "P", zone="Above the Break 3", three=True, dist=26),
    ]
    s = summarise_player_games(shots(rows))
    assert s["SZ_THREE_RATE"].iloc[0] == pytest.approx(1.0)
    assert s["SZ_CORNER3_RATE"].iloc[0] == pytest.approx(0.5)


def test_the_paint_band_is_its_own_zone_not_folded_into_the_rim():
    rows = [
        shot("1", "P", zone="Restricted Area", dist=1),
        shot("1", "P", zone="In The Paint (Non-RA)", dist=8),
    ]
    s = summarise_player_games(shots(rows))
    assert s["SZ_RIM_RATE"].iloc[0] == pytest.approx(0.5)
    assert s["SZ_PAINT_RATE"].iloc[0] == pytest.approx(0.5)
    assert s["SZ_MID_RATE"].iloc[0] == pytest.approx(0.0)


@pytest.mark.parametrize("action", [
    "Driving Layup Shot", "Running Finger Roll Layup Shot", "Slam Dunk Shot",
    "Cutting Dunk Shot",
])
def test_compound_action_names_still_count_as_dunk_or_layup(action):
    """The export spells out compounds; a shot is a layup whether or not it
    was also driving."""
    s = summarise_player_games(shots([shot("1", "P", action=action)]))
    assert s["SZ_DUNK_LAYUP_RATE"].iloc[0] == pytest.approx(1.0)


@pytest.mark.parametrize("action", [
    "Pullup Jump shot", "Step Back Jump shot", "Fadeaway Jump Shot",
    "Turnaround Hook Shot", "Driving Floating Jump Shot",
])
def test_off_the_dribble_attempts_count_as_self_created(action):
    s = summarise_player_games(shots([shot("1", "P", action=action)]))
    assert s["SZ_SELF_CREATED_RATE"].iloc[0] == pytest.approx(1.0)


def test_a_catch_and_shoot_jumper_is_not_self_created():
    s = summarise_player_games(shots([shot("1", "P", action="Jump Shot")]))
    assert s["SZ_SELF_CREATED_RATE"].iloc[0] == pytest.approx(0.0)


# --- 4. shape and refusal ---------------------------------------------

def test_sz_fga_is_a_diagnostic_and_is_never_rolled_into_a_feature():
    """A count the panel already carries as FGA, FGA_L5 and FGA_L10. Shipping
    it would hand a model one number twice."""
    assert "SZ_FGA" in SUMMARY_COLS
    assert not any(c.startswith("SZ_FGA") for c in SHOT_FEATURE_COLS)


def test_the_join_never_changes_the_row_count():
    p = panel(8)
    out = attach_shot_rolling_features(p, summarise_player_games(_even_shots(8)),
                                       required=True)
    assert len(out) == len(p)


def test_the_frame_comes_back_in_the_order_it_went_in():
    p = panel(8)
    shuffled = p.sample(frac=1.0, random_state=5).reset_index(drop=True)
    out = attach_shot_rolling_features(
        shuffled, summarise_player_games(_even_shots(8)), required=True
    )
    assert out["GAME_ID"].tolist() == shuffled["GAME_ID"].tolist()


def test_two_players_profiles_do_not_bleed_into_each_other():
    rows = []
    for i in range(6):
        rows.append(shot(f"002250{i:04d}", "A", zone="Restricted Area", dist=1))
        rows.append(shot(f"002250{i:04d}", "B", zone="Above the Break 3",
                         three=True, dist=26))
    p = pd.concat([panel(6, player="A"), panel(6, player="B")], ignore_index=True)
    p["FGA"] = 1
    out = attach_shot_rolling_features(p, summarise_player_games(shots(rows)),
                                       required=True)
    a = out[out["PLAYER_ID"] == "A"]["SZ_RIM_RATE_L10"].dropna()
    b = out[out["PLAYER_ID"] == "B"]["SZ_RIM_RATE_L10"].dropna()
    # Series == approx collapses to one bool, so compare elementwise.
    assert not a.empty and not b.empty
    assert (a - 1.0).abs().max() < 1e-9, a.tolist()
    assert b.abs().max() < 1e-9, b.tolist()


def test_an_export_missing_a_required_column_is_refused_by_name():
    bad = _even_shots(2).drop(columns=["SHOT_ZONE_BASIC"])
    with pytest.raises(ShotZoneError, match="SHOT_ZONE_BASIC"):
        prepare_shots(bad)


def test_no_summary_means_no_columns_rather_than_zeros():
    p = panel(4)
    out = attach_shot_rolling_features(p, None)
    assert not set(SHOT_FEATURE_COLS) & set(out.columns)
    pd.testing.assert_frame_equal(out, p)


def test_required_turns_a_missing_summary_into_a_refusal():
    with pytest.raises(ShotZoneError, match="DATA_NOT_AVAILABLE"):
        attach_shot_rolling_features(panel(4), None, required=True)


def test_an_empty_panel_is_not_a_crash():
    assert attach_shot_rolling_features(pd.DataFrame(), None).empty


def test_a_duplicated_summary_row_cannot_multiply_player_rows():
    summary = summarise_player_games(_even_shots(8))
    doubled = pd.concat([summary, summary.head(1)], ignore_index=True)
    out = attach_shot_rolling_features(panel(8), doubled, required=True)
    assert len(out) == 8


# --- built but deliberately not wired ---------------------------------

def test_no_market_reads_a_shot_zone_column_yet():
    from src.models.labels import default_feature_cols

    for market in ("PTS", "REB", "AST", "PRA", "FG3M", "STL", "BLK", "MIN"):
        assert not set(default_feature_cols(market)) & set(SHOT_FEATURE_COLS), market


def test_the_layer_is_registered_with_feature_ab_so_it_can_be_measured():
    import importlib.util

    spec = importlib.util.spec_from_file_location("_fab", "scripts/feature_ab.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert "shot_zones" in module.LAYERS
    assert set(module.LAYERS["shot_zones"].columns) == set(SHOT_FEATURE_COLS)
