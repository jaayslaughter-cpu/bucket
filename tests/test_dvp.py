"""Defence versus position — src/features/dvp.py.

WHY THIS LAYER EXISTS is also a data finding. An earlier reading of this tree
concluded a defence-versus-position feature was impossible here because "no
position column exists anywhere — not in the 196-column panel, not in the
214-column pbp panel, not on any DB model". Every one of those statements was
true, and the conclusion was wrong: the archive's own
``PlayerStatistics`` export carries ``startingPosition``, the panel contract
simply never mapped it. The honest lesson is the one AGENTS.md already
states — inspect the file, not the derived artifact.

THE SOURCE IMPLEMENTATION THIS WAS ADAPTED FROM HAD THREE DEFECTS, and the
tests named after them below fail if any is reintroduced:

  1. it rolled over PLAYER ROWS grouped by (season, opponent, position), and a
     team faces four or five guards a night, so its "L10" spanned about two
     games;
  2. it stored only the LATEST value per (season, opponent, position), which
     as a training feature means every October row carries June's defence;
  3. its league baseline was a season-wide median, the exact look-ahead
     ``defense._league_relative_index`` records being found and fixed.

AND ONE DEFECT WAS MINE, found by running the layer on the real panel rather
than on this fixture: a per-game minimum sample size of two player-games
nulled the C bucket for every centre in the league, because the archive names
exactly one starting centre per team-game.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.dvp import (
    ALLOWED_TEMPLATE,
    DVP_FEATURE_COLS,
    INDEX_TEMPLATE,
    POS_BUCKET_COLUMN,
    POSITION_BUCKETS,
    ROLL_WINDOW,
    STARTING_POSITION_COLUMN,
    DvpFeatureError,
    assign_position_buckets,
    attach_dvp_features,
    attach_dvp_features_layer,
    build_opponent_allowed,
    normalise_bucket,
)

TEAMS = ["AAA", "BBB", "CCC", "DDD"]
# Two guards, two forwards, one centre started; three more come off the bench.
LINEUP = [
    ("G1", "G"), ("G2", "G"), ("F1", "F"), ("F2", "F"), ("C1", "C"),
    ("G3", None), ("F3", None), ("C2", None),
]


def league(
    rounds: int = 14,
    *,
    concede: dict[str, dict[str, float]] | None = None,
    drift: float = 0.0,
    no_centre_for: set[tuple[str, int]] | None = None,
) -> pd.DataFrame:
    """
    A four-team round robin with a fixed eight-man rotation per team.

    ``concede`` plants a per-defender, per-bucket points level: everything a
    bucket scores against that team is set to it. Without it every player
    scores 10, so any variation in the output is the layer inventing signal.

    ``drift`` adds ``drift * round`` points to everyone, so the league's level
    CHANGES over the season. A uniform league cannot tell an as-of baseline
    from a season-wide one — both equal 10 — and the first version of
    ``test_the_index_baseline_cannot_see_the_rest_of_the_season`` therefore
    passed with the look-ahead reinstated.

    ``no_centre_for`` is a set of (team, round) pairs in which that team
    fields no centre at all: its centre starts at forward and its reserve
    centre does not dress. Its OPPONENT then has a game with no C bucket
    observed against it, which is what the full (game, bucket) grid exists
    for. Without such a game the grid is indistinguishable from the observed
    pairs alone.
    """
    rows = []
    date = pd.Timestamp("2025-10-21")
    game = 0
    no_centre_for = no_centre_for or set()
    for rnd in range(rounds):
        pairs = [(TEAMS[0], TEAMS[1]), (TEAMS[2], TEAMS[3])] if rnd % 2 == 0 else [
            (TEAMS[0], TEAMS[2]), (TEAMS[1], TEAMS[3])
        ]
        for home, away in pairs:
            game += 1
            gid = f"00225{game:05d}"
            for team, opp in ((home, away), (away, home)):
                small_ball = (team, rnd) in no_centre_for
                for suffix, started in LINEUP:
                    if small_ball and suffix == "C2":
                        continue
                    if small_ball and suffix == "C1":
                        started = "F"
                    bucket = started or ("G" if "G" in suffix else "F" if "F" in suffix else "C")
                    pts = 10.0 + drift * rnd
                    if concede and opp in concede:
                        pts = concede[opp].get(bucket, pts) + drift * rnd
                    rows.append({
                        "PLAYER_ID": f"{team}-{suffix}",
                        "GAME_ID": gid,
                        "GAME_DATE": date + pd.Timedelta(days=2 * rnd),
                        "TEAM_ABBREVIATION": team,
                        "OPPONENT_ABBREVIATION": opp,
                        STARTING_POSITION_COLUMN: started,
                        "PTS": pts,
                        "REB": 5.0,
                        "AST": 4.0,
                        "FG3M": 2.0,
                        "STL": 1.0,
                        "BLK": 0.5,
                        "MIN": 30.0,
                    })
    return pd.DataFrame(rows).sort_values(
        ["PLAYER_ID", "GAME_DATE"], kind="mergesort"
    ).reset_index(drop=True)


# --- normalise_bucket ---------------------------------------------------

@pytest.mark.parametrize(("raw", "expected"), [
    ("G", "G"), ("PG", "G"), ("SG", "G"), ("guard", "G"),
    ("F", "F"), ("SF", "F"), ("forward", "F"),
    ("C", "C"), ("Center", "C"), ("CENTRE", "C"),
    ("F-C", "F"), ("G/F", "G"), ("  c  ", "C"),
])
def test_known_position_spellings_collapse_onto_three_buckets(raw, expected):
    assert normalise_bucket(raw) == expected


def test_pf_as_a_position_string_is_a_forward_not_a_foul_count():
    """
    "PF" is power forward in a position string and PERSONAL FOULS as a panel
    column (src/features/fouls.py). The two share a name and nothing else.
    """
    assert normalise_bucket("PF") == "F"


@pytest.mark.parametrize("raw", [None, np.nan, "", "nan", "DH", "goalkeeper", 7])
def test_an_unrecognised_position_is_none_rather_than_a_guess(raw):
    """
    A bucket guessed from an unknown spelling puts the player in the wrong
    defensive population and still produces a plausible number.
    """
    assert normalise_bucket(raw) is None


# --- the as-of bucket ---------------------------------------------------

def test_a_players_first_start_does_not_bucket_its_own_row():
    """
    The current game's designation says the player is in tonight's starting
    five, which is minutes information about tonight.
    """
    panel = league(rounds=4)
    buckets = assign_position_buckets(panel)
    first = panel.groupby("PLAYER_ID", sort=False).head(1).index
    assert buckets.loc[first].isna().all()


def test_the_bucket_is_the_modal_prior_start():
    panel = league(rounds=6)
    buckets = assign_position_buckets(panel)
    rows = panel[panel["PLAYER_ID"] == "AAA-C1"]
    assert buckets.loc[rows.index[1:]].eq("C").all()


def test_a_player_who_never_started_gets_no_bucket():
    panel = league(rounds=6)
    buckets = assign_position_buckets(panel)
    bench = panel[panel["PLAYER_ID"] == "AAA-G3"]
    assert buckets.loc[bench.index].isna().all()


def test_a_changed_designation_moves_the_bucket_with_a_lag():
    panel = league(rounds=8)
    mask = (panel["PLAYER_ID"] == "AAA-C1") & (
        panel["GAME_DATE"] >= pd.Timestamp("2025-10-27")
    )
    panel.loc[mask, STARTING_POSITION_COLUMN] = "F"
    buckets = assign_position_buckets(panel)
    rows = panel[panel["PLAYER_ID"] == "AAA-C1"].sort_values("GAME_DATE")
    # Still a centre on the first F game, because the modal prior start is C.
    assert buckets.loc[rows.index[3]] == "C"
    # Enough F starts later and the bucket follows.
    assert buckets.loc[rows.index[-1]] == "F"


# --- the observed designation must not escape ---------------------------

def test_the_observed_designation_is_never_written_to_the_panel():
    """
    A column saying "this player started tonight" is a minutes signal about
    the current game. The aggregation uses it internally, on COMPLETED games
    only, and must not leave it behind.
    """
    out = attach_dvp_features(league(rounds=12), required=True)
    assert "_BUCKET" not in out.columns
    assert "_OBS" not in out.columns
    leaked = [c for c in out.columns if c.startswith("_")]
    assert not leaked, leaked
    # POS_BUCKET is the AS-OF estimate, not tonight's designation.
    panel = league(rounds=12)
    first = panel.groupby("PLAYER_ID", sort=False).head(1).index
    assert out.loc[first, POS_BUCKET_COLUMN].isna().all()


# --- the join is on the player's own bucket -----------------------------

def test_two_players_in_the_same_game_get_different_numbers_by_position():
    """
    The entire point of the layer. A join that dropped the bucket would
    reproduce DEF_RATING_L10 under a new name.
    """
    panel = league(rounds=14, concede={"BBB": {"G": 30.0, "F": 10.0, "C": 4.0}})
    out = attach_dvp_features(panel, required=True)
    col = ALLOWED_TEMPLATE.format(stat="PTS")
    versus_bbb = out[
        (out["OPPONENT_ABBREVIATION"] == "BBB") & out[col].notna()
    ]
    by_bucket = versus_bbb.groupby(POS_BUCKET_COLUMN)[col].mean()
    assert set(by_bucket.index) == set(POSITION_BUCKETS)
    assert by_bucket["G"] > by_bucket["F"] > by_bucket["C"]
    assert by_bucket["G"] == pytest.approx(30.0, abs=0.5)
    assert by_bucket["C"] == pytest.approx(4.0, abs=0.5)


def test_every_bucket_including_the_centre_is_populated():
    """
    A per-game minimum sample size of two nulled the C bucket for every
    centre in the league: the archive names exactly ONE starting centre per
    team-game, so the per-game sample is always one. Found on the real
    214,381-row panel, not on this fixture — which is why the fixture now
    carries a single starting centre per side too.
    """
    out = attach_dvp_features(league(rounds=14), required=True)
    col = ALLOWED_TEMPLATE.format(stat="REB")
    covered = out.loc[out[col].notna(), POS_BUCKET_COLUMN].unique()
    assert set(covered) == set(POSITION_BUCKETS)


def test_a_row_with_no_bucket_gets_no_matchup_rather_than_an_average():
    panel = league(rounds=14)
    out = attach_dvp_features(panel, required=True)
    col = ALLOWED_TEMPLATE.format(stat="PTS")
    unbucketed = out[out[POS_BUCKET_COLUMN].isna()]
    assert len(unbucketed) > 0
    assert out.loc[unbucketed.index, col].isna().all()


# --- the defender is the opponent ---------------------------------------

def test_the_rate_describes_the_opponent_and_not_the_players_own_team():
    """
    Getting this backwards hands every model a full column of plausible
    numbers describing the wrong defence. BBB concedes 30 to guards; a guard
    playing FOR BBB must not receive 30.
    """
    panel = league(rounds=14, concede={"BBB": {"G": 30.0, "F": 10.0, "C": 10.0}})
    out = attach_dvp_features(panel, required=True)
    col = ALLOWED_TEMPLATE.format(stat="PTS")
    guards = out[(out[POS_BUCKET_COLUMN] == "G") & out[col].notna()]
    against = guards[guards["OPPONENT_ABBREVIATION"] == "BBB"][col].mean()
    playing_for = guards[guards["TEAM_ABBREVIATION"] == "BBB"][col].mean()
    assert against == pytest.approx(30.0, abs=0.5)
    assert playing_for < 15.0


# --- the shift ----------------------------------------------------------

def test_tonights_concession_is_never_in_tonights_number():
    panel = league(rounds=14)
    spiked = panel.copy()
    target = spiked[
        (spiked["OPPONENT_ABBREVIATION"] == "BBB")
        & (spiked["GAME_DATE"] == pd.Timestamp("2025-11-06"))
    ].index
    assert len(target) > 0
    spiked.loc[target, "PTS"] = 99.0

    col = ALLOWED_TEMPLATE.format(stat="PTS")
    base = attach_dvp_features(panel, required=True).set_index(
        ["PLAYER_ID", "GAME_ID"]
    )[col]
    after = attach_dvp_features(spiked, required=True).set_index(
        ["PLAYER_ID", "GAME_ID"]
    )[col]
    rows = panel.loc[target].set_index(["PLAYER_ID", "GAME_ID"]).index
    pd.testing.assert_series_equal(base.loc[rows], after.loc[rows])
    # And it DOES reach a later game, or the layer would be inert.
    assert (after > base + 1e-9).any()


def test_a_defenders_first_games_of_a_season_have_no_prior_form():
    panel = league(rounds=14)
    out = attach_dvp_features(panel, required=True)
    col = ALLOWED_TEMPLATE.format(stat="PTS")
    opening = out[out["GAME_DATE"] == panel["GAME_DATE"].min()]
    assert out.loc[opening.index, col].isna().all()


# --- the window is over the defender's games ----------------------------

def test_the_window_counts_the_defenders_games_not_the_player_rows():
    """
    The source implementation grouped player rows and called .rolling(10) on
    them. A team faces five guards a night, so its "L10" spanned two games.
    Here a defender's number must change the moment its eleventh game pushes
    its first out of the window — which happens after ten GAMES, not after
    ten guard-rows.

    Each team plays every other round, so game N of the season is round N for
    that team. Planting a value in the defender's FIRST game and reading a row
    played against it in rounds ROLL_MIN_PERIODS..ROLL_WINDOW shows the window
    holding it; a row in round ROLL_WINDOW + 2 shows it gone.
    """
    panel = league(rounds=ROLL_WINDOW + 4)
    spiked = panel.copy()
    first_date = panel["GAME_DATE"].min()
    plant = spiked[
        (spiked["OPPONENT_ABBREVIATION"] == "AAA")
        & (spiked["GAME_DATE"] == first_date)
    ].index
    spiked.loc[plant, "PTS"] = 50.0

    col = ALLOWED_TEMPLATE.format(stat="PTS")
    base = attach_dvp_features(panel, required=True)
    after = attach_dvp_features(spiked, required=True)
    moved = (after[col] - base[col]).abs() > 1e-9
    rounds_moved = sorted(
        ((after.loc[moved, "GAME_DATE"] - first_date).dt.days // 2).unique()
    )
    # AAA plays once per round, so its first game is round 0 and sits inside
    # the window for rounds 1..ROLL_WINDOW.
    assert min(rounds_moved) >= 1
    assert max(rounds_moved) <= ROLL_WINDOW
    # If the window were over player rows it would be exhausted in two or
    # three rounds; it must still be live near the end of it.
    assert max(rounds_moved) >= ROLL_WINDOW - 1


def test_a_game_with_no_player_of_a_bucket_still_occupies_the_window():
    """
    The grid is the full cross product of a defender's games and all three
    buckets. If an unobserved (game, bucket) pair were skipped, a ten-game
    window would quietly span twelve.

    BBB goes small in rounds 2, 4 and 6 — no centre on the floor at all — so
    whoever BBB plays in those rounds has a game with no C bucket observed
    against it. Those games must still be rows in the table, carrying null,
    rather than being absent from the C window.
    """
    small = {("BBB", rnd) for rnd in (2, 4, 6)}
    panel = league(rounds=14, no_centre_for=small)
    table = build_opponent_allowed(panel)

    games = (
        panel[["OPPONENT_ABBREVIATION", "GAME_ID"]]
        .drop_duplicates()
        .groupby("OPPONENT_ABBREVIATION")
        .size()
    )
    per_defender = table.groupby(["team_abbr", "_BUCKET"]).size()
    assert len(per_defender) == len(games) * len(POSITION_BUCKETS)
    for (team, _bucket), rows in per_defender.items():
        assert rows == games[team], (team, _bucket)

    # And the specific unobserved pairs are present and null, not missing.
    bbb_games = set(
        panel.loc[panel["TEAM_ABBREVIATION"] == "BBB", "GAME_ID"]
    ) & set(
        panel.loc[
            panel["GAME_DATE"].isin(
                panel["GAME_DATE"].min() + pd.Timedelta(days=2) * np.array([2, 4, 6])
            ),
            "GAME_ID",
        ]
    )
    assert bbb_games
    gap = table[
        table["nba_game_id"].isin(bbb_games) & (table["_BUCKET"] == "C")
    ]
    opponents_of_bbb = gap[gap["team_abbr"] != "BBB"]
    assert len(opponents_of_bbb) == len(bbb_games)


# --- the league baseline is as-of ---------------------------------------

def test_the_index_baseline_cannot_see_the_rest_of_the_season():
    """
    The source implementation divided by a season-wide median. Truncating the
    season must then change an early row's index, and must not here.
    """
    # The league's level must DRIFT, or a season-wide mean and an as-of mean
    # are the same number and the look-ahead is invisible.
    panel = league(rounds=ROLL_WINDOW + 6, drift=2.0)
    cut = panel["GAME_DATE"].min() + pd.Timedelta(days=2 * (ROLL_WINDOW + 1))
    col = INDEX_TEMPLATE.format(stat="PTS")

    full = attach_dvp_features(panel, required=True)
    early = attach_dvp_features(
        panel[panel["GAME_DATE"] <= cut].copy(), required=True
    )
    keys = ["PLAYER_ID", "GAME_ID"]
    merged = full.merge(early, on=keys, suffixes=("_full", "_early"))
    both = merged[f"{col}_full"].notna() & merged[f"{col}_early"].notna()
    assert both.sum() > 0
    pd.testing.assert_series_equal(
        merged.loc[both, f"{col}_full"], merged.loc[both, f"{col}_early"],
        check_names=False,
    )


def test_the_index_is_scoped_to_the_bucket_not_pooled_across_them():
    """
    Dividing a centre's rebounds-allowed by a league mean that included
    guards would report every centre matchup as favourable. With a uniform
    league every bucket's index must sit at 1.0.
    """
    out = attach_dvp_features(league(rounds=ROLL_WINDOW + 6), required=True)
    col = INDEX_TEMPLATE.format(stat="REB")
    by_bucket = out.groupby(POS_BUCKET_COLUMN)[col].mean()
    assert set(by_bucket.index) == set(POSITION_BUCKETS)
    for bucket in POSITION_BUCKETS:
        assert by_bucket[bucket] == pytest.approx(1.0, abs=0.02), bucket


# --- a mean, not a sum --------------------------------------------------

def test_dropping_a_player_from_every_game_barely_moves_the_level():
    """
    The reason defense.py refuses to sum the player panel: a sum measures
    roster coverage as much as defence, and dropping one player per team-game
    moved its "points allowed" by 33%. A MEAN over the bucket's player-games
    changes the sample size, not the level.
    """
    panel = league(rounds=14)
    thinned = panel[panel["PLAYER_ID"].str.split("-").str[1] != "G2"].copy()
    col = ALLOWED_TEMPLATE.format(stat="PTS")
    full = attach_dvp_features(panel, required=True)
    short = attach_dvp_features(thinned, required=True)
    guards_full = full.loc[full[POS_BUCKET_COLUMN] == "G", col].mean()
    guards_short = short.loc[short[POS_BUCKET_COLUMN] == "G", col].mean()
    assert guards_short == pytest.approx(guards_full, rel=0.02)


# --- shape and refusal --------------------------------------------------

def test_the_join_never_changes_the_row_count():
    panel = league(rounds=14)
    out = attach_dvp_features(panel, required=True)
    assert len(out) == len(panel)


def test_the_frame_comes_back_in_the_order_it_went_in():
    panel = league(rounds=8)
    shuffled = panel.sample(frac=1.0, random_state=7).reset_index(drop=True)
    out = attach_dvp_features(shuffled, required=True)
    assert out["PLAYER_ID"].tolist() == shuffled["PLAYER_ID"].tolist()
    assert out["GAME_ID"].tolist() == shuffled["GAME_ID"].tolist()


def test_a_shuffled_frame_gets_the_same_numbers_as_a_sorted_one():
    panel = league(rounds=12)
    shuffled = panel.sample(frac=1.0, random_state=11).reset_index(drop=True)
    keys = ["PLAYER_ID", "GAME_ID"]
    a = attach_dvp_features(panel, required=True).set_index(keys).sort_index()
    b = attach_dvp_features(shuffled, required=True).set_index(keys).sort_index()
    for col in DVP_FEATURE_COLS:
        pd.testing.assert_series_equal(a[col], b[col], check_names=False)


def test_an_absent_position_column_adds_nothing_rather_than_a_guess():
    panel = league(rounds=8).drop(columns=[STARTING_POSITION_COLUMN])
    out = attach_dvp_features_layer(panel)
    assert not set(DVP_FEATURE_COLS) & set(out.columns)
    pd.testing.assert_frame_equal(out, panel)


def test_an_all_null_position_column_is_refused_rather_than_derived():
    """
    A bucket derived from rebounds and assists would mix two defensive
    populations under one label. There is no position in a box score.
    """
    panel = league(rounds=8)
    panel[STARTING_POSITION_COLUMN] = None
    with pytest.raises(DvpFeatureError, match="DATA_NOT_AVAILABLE"):
        build_opponent_allowed(panel)
    out = attach_dvp_features_layer(panel)
    allowed = ALLOWED_TEMPLATE.format(stat="PTS")
    assert allowed not in out.columns


def test_a_missing_opponent_is_refused():
    panel = league(rounds=8).drop(columns=["OPPONENT_ABBREVIATION"])
    with pytest.raises(DvpFeatureError, match="OPPONENT_ABBREVIATION"):
        build_opponent_allowed(panel)


def test_a_duplicated_lookup_row_is_refused_rather_than_multiplying_rows():
    panel = league(rounds=12)
    table = build_opponent_allowed(panel)
    doubled = pd.concat([table, table.head(1)], ignore_index=True)
    with pytest.raises(DvpFeatureError, match="duplicate"):
        attach_dvp_features(panel, doubled, required=True)


def test_an_empty_panel_is_not_a_crash():
    assert attach_dvp_features_layer(pd.DataFrame()).empty


# --- built but deliberately not wired -----------------------------------

def test_no_market_reads_a_dvp_column_yet():
    """
    THE MEASUREMENT HAS NOW BEEN MADE, and the reason this stays unwired
    changed with it. `--layer dvp --wire-under-test --markets REB --folds 4`
    came back better on every fold for four of five models, at 1.3-2.5x the
    fold spread (docs/fouls_and_dvp.md section 3a). So "unmeasured" is no
    longer the answer, and a docstring that still said so would be this
    project's most-repeated defect.

    What blocks it is the WRITER, pinned by the companion test below: the live
    panel is built from `player_game_logs`, which has no position column, so
    both DVP_REB_* columns are null on every live row. Wiring them would train
    trees to split on a column that arrives empty in production.
    """
    from src.models.labels import default_feature_cols

    for market in ("PTS", "REB", "AST", "PRA", "FG3M", "STL", "BLK", "MIN"):
        listed = set(default_feature_cols(market))
        assert not listed & set(DVP_FEATURE_COLS), market


def test_the_live_table_still_has_no_position_column_which_is_what_blocks_wiring():
    """
    The tripwire for the test above. `attach_dvp_features` needs
    STARTING_POSITION; only the archive ingest supplies it. If a position
    writer ever reaches `player_game_logs`, this goes red — which is the
    signal to re-run the REB arm on a live-shaped panel and reopen the wiring
    decision, not to delete the assertion.
    """
    from src.db.models import PlayerGameLog
    from src.features.dvp import STARTING_POSITION_COLUMN

    columns = {c.name for c in PlayerGameLog.__table__.columns}
    assert STARTING_POSITION_COLUMN.lower() not in columns
    assert not [c for c in columns if "position" in c or c in {"pos", "start_pos"}], (
        "a position column reached player_game_logs, so DvP may no longer be "
        "null on the live path; re-measure and revisit "
        "test_no_market_reads_a_dvp_column_yet"
    )


def test_the_layer_is_registered_with_feature_ab_so_it_can_be_measured():
    import importlib.util

    spec = importlib.util.spec_from_file_location("_fab", "scripts/feature_ab.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert "dvp" in module.LAYERS
    # POS_BUCKET is a string label, not a numeric feature, so it is not an arm.
    assert POS_BUCKET_COLUMN not in module.LAYERS["dvp"].columns
    assert set(module.LAYERS["dvp"].columns) == {
        c for c in DVP_FEATURE_COLS if c != POS_BUCKET_COLUMN
    }


# --- the builder actually runs them -------------------------------------

def test_both_new_layers_are_registered_with_the_builder():
    """
    A layer the builder does not call produces nothing on a real panel while
    its own tests keep passing. That is how two of the four form layers ran
    for months with no market reading a column.
    """
    from src.features.builder import _ADDITIVE_FEATURE_LAYERS

    labels = [label for label, _ in _ADDITIVE_FEATURE_LAYERS]
    assert "fouls" in labels
    assert "dvp" in labels


def test_the_builder_emits_both_layers_columns_and_a_distinct_schema_version():
    """
    End to end through build_feature_matrix, because the registry entry and
    the layer contract are two different things and only this exercises both.
    """
    from src.features.builder import FEATURE_SCHEMA_VERSION, build_feature_matrix
    from src.features.fouls import FOUL_FEATURE_COLS

    panel = league(rounds=16)
    panel["PF"] = np.resize([0, 1, 2, 3, 4, 5], len(panel))
    panel["SEASON"] = "2025-26"
    for col, value in (("TOV", 1.0), ("FGM", 4.0), ("FGA", 9.0), ("FTM", 2.0),
                       ("FTA", 3.0), ("OREB", 1.0), ("DREB", 4.0)):
        panel[col] = value
    panel["IS_HOME"] = True

    out = build_feature_matrix(panel)
    assert len(out) == len(panel)
    for col in (*FOUL_FEATURE_COLS, *DVP_FEATURE_COLS):
        assert col in out.columns, col
    assert out["DVP_PTS_ALLOWED_L10"].notna().any()
    assert out["PF_L10"].notna().any()
    # A run with these layers must not be mistakable for one without them.
    version = out["FEATURE_SCHEMA_VERSION"].iloc[0]
    assert version != FEATURE_SCHEMA_VERSION
    assert version.startswith(f"{FEATURE_SCHEMA_VERSION}+layers.")


def test_the_join_survives_either_dtype_of_team_code():
    """
    A REGRESSION GUARD, and it is worth saying what it does not establish.
    The real panel's team codes come through kaggle_nba as pandas "string" and
    a hand-built frame's are plain object; this pins that both join and that
    the panel's own dtype comes back unchanged. It does NOT prove the
    alignment in attach_dvp_features is load-bearing — reverting that line
    leaves this test passing, because pandas currently matches across the two
    dtypes. The comment at that line says the same thing. What makes the
    property worth pinning is the failure mode if pandas ever stops: a merge
    that matches nothing returns a full column of nulls that reads exactly
    like a league with no prior form.
    """
    col = ALLOWED_TEMPLATE.format(stat="PTS")
    for dtype in ("object", "string"):
        panel = league(rounds=14)
        panel["OPPONENT_ABBREVIATION"] = panel["OPPONENT_ABBREVIATION"].astype(dtype)
        panel["TEAM_ABBREVIATION"] = panel["TEAM_ABBREVIATION"].astype(dtype)
        out = attach_dvp_features(panel, required=True)
        assert out[col].notna().sum() > 0, dtype
        # And the panel's own dtype is handed back unchanged.
        assert out["OPPONENT_ABBREVIATION"].dtype == panel["OPPONENT_ABBREVIATION"].dtype


# --- the A/B arm must ask the right question ----------------------------

def test_each_market_is_offered_only_its_own_matchup_columns():
    """
    DVP_* carries its stat in the MIDDLE — DVP_REB_ALLOWED_L10 starts with
    neither "REB_" nor "OPP_REB_" — so feature_ab's stat-prefix filter does
    not see it, and without an explicit branch every market received all
    twelve columns. A points model handed the rebound, assist and block
    matchup columns is not measuring "does defence-versus-position help
    points"; it is the identical defect the DEF_ branch in that filter was
    written to fix, and it was reproduced in a launched run before being
    caught.

    PRA takes the three it is the sum of, which is how
    labels._DEFENSE_BY_MARKET already treats it.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("_fab2", "scripts/feature_ab.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # column_for_market is the function the run itself calls. An earlier
    # version of this test exercised only the _dvp_for_market helper it
    # delegates to, and deleting the delegation left the test passing.
    under_test = module.LAYERS["dvp"].columns
    for market, expected in (
        ("PTS", {"DVP_PTS_ALLOWED_L10", "DVP_PTS_INDEX_L10"}),
        ("REB", {"DVP_REB_ALLOWED_L10", "DVP_REB_INDEX_L10"}),
        ("AST", {"DVP_AST_ALLOWED_L10", "DVP_AST_INDEX_L10"}),
        ("FG3M", {"DVP_FG3M_ALLOWED_L10", "DVP_FG3M_INDEX_L10"}),
        ("STL", {"DVP_STL_ALLOWED_L10", "DVP_STL_INDEX_L10"}),
        ("BLK", {"DVP_BLK_ALLOWED_L10", "DVP_BLK_INDEX_L10"}),
    ):
        got = {c for c in under_test if module.column_for_market(c, market)}
        assert got == expected, (market, sorted(got))

    pra = {c for c in under_test if module.column_for_market(c, "PRA")}
    assert pra == {
        "DVP_PTS_ALLOWED_L10", "DVP_PTS_INDEX_L10",
        "DVP_REB_ALLOWED_L10", "DVP_REB_INDEX_L10",
        "DVP_AST_ALLOWED_L10", "DVP_AST_INDEX_L10",
    }, sorted(pra)

    # MIN reads no matchup column: there is no DVP_MIN_*, and a market with
    # no member of the family must get nothing rather than everything.
    assert not {c for c in under_test if module.column_for_market(c, "MIN")}

    # The routing does not disturb the families already handled: DEF_* still
    # goes by labels._DEFENSE_BY_MARKET, and the market-neutral PF_* columns
    # still reach every market, because foul propensity bears on minutes
    # rather than on any one stat.
    assert module.column_for_market("DEF_RATING_L10", "PTS") is True
    assert module.column_for_market("DEF_REB_ALLOWED_PER100_L10", "PTS") is False
    assert module.column_for_market("DEF_REB_ALLOWED_PER100_L10", "REB") is True
    assert all(
        module.column_for_market(c, m)
        for c in module.LAYERS["fouls"].columns
        for m in ("PTS", "REB", "AST")
    )

    # AND THE CALL SITE, which is the half that decides what the models
    # actually train on. This assertion is STRUCTURAL and says so: the
    # widening closure lives inside _run and only a full A/B run reaches it,
    # so what is pinned here is that the closure consults the routing
    # function at all. Removing the filter from it is otherwise invisible to
    # every test in this file -- which was true until this was added.
    import inspect

    source = inspect.getsource(module._run)
    assert "column_for_market(c, market.upper())" in source, (
        "_widened no longer routes per market; every market would receive "
        "every column under test and every delta would answer a different "
        "question than the one asked"
    )
