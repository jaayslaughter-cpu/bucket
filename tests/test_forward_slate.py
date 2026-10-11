"""O1 — rows for a slate that has not been played.

`load_player_panel` reads COMPLETED box scores and `_filter_to_slate` keeps
only rows dated on the slate, so a 09:00 PT run found an empty intersection and
exited 0 with `success_no_data`. Every day. It did not look broken.

The tests that matter here are the leakage ones: a forward row carries no
box-score stat, so the rolling features must read that player's own prior REAL
games and the forward row must have nothing of its own to leak. That is checked
by VALUE, not by inspection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from src.features.builder import assert_no_lookahead, build_feature_matrix
from src.pipeline.forward_slate import (
    BOX_SCORE_COLUMNS,
    FORWARD_FLAG,
    attach_forward_slate,
    build_forward_rows,
)


@dataclass
class Game:
    game_id: str
    home_team: str
    away_team: str
    is_neutral_site: bool = False


def panel(
    teams=(("LAL", "BOS"), ("BOS", "LAL")),
    *,
    players: int = 3,
    games: int = 8,
) -> pd.DataFrame:
    rows = []
    for team, opp in teams:
        for p in range(players):
            for i in range(games):
                rows.append({
                    "PLAYER_ID": f"{team}{p}", "PLAYER_NAME": f"{team} P{p}",
                    "GAME_ID": f"00225{team}{i:03d}",
                    "GAME_DATE": pd.Timestamp("2025-10-21") + pd.Timedelta(days=2 * i),
                    "SEASON": "2025-26", "TEAM_ABBREVIATION": team,
                    "OPPONENT_ABBREVIATION": opp, "IS_HOME": i % 2 == 0,
                    "IS_NEUTRAL_SITE": False,
                    "MIN": 30.0, "PTS": 18.0 + i + p, "REB": 5.0, "AST": 4.0,
                    "FG3M": 2.0, "STL": 1.0, "BLK": 0.5, "TOV": 2.0,
                    "FGA": 15.0, "FTA": 4.0,
                })
    return pd.DataFrame(rows)


def forward_only(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame[FORWARD_FLAG].fillna(False).astype(bool)]


# --- the rows exist at all -------------------------------------------------

def test_a_scheduled_game_produces_a_row_per_recent_player_on_both_teams():
    out = build_forward_rows(
        panel(), [Game("401700001", "LAL", "BOS")], slate_date="2025-11-15"
    )
    assert out.status == "OK"
    assert out.n_forward_rows == 6          # 3 players x 2 teams
    rows = forward_only(out.panel)
    assert set(rows["TEAM_ABBREVIATION"]) == {"LAL", "BOS"}
    assert set(rows["GAME_ID"]) == {"401700001"}
    assert (pd.to_datetime(rows["GAME_DATE"]) == pd.Timestamp("2025-11-15")).all()


def test_home_and_away_are_set_from_the_schedule_not_guessed():
    out = build_forward_rows(
        panel(), [Game("401700001", "LAL", "BOS")], slate_date="2025-11-15"
    )
    rows = forward_only(out.panel).set_index("PLAYER_NAME")
    assert bool(rows.loc["LAL P0", "IS_HOME"]) is True
    assert bool(rows.loc["BOS P0", "IS_HOME"]) is False
    assert rows.loc["LAL P0", "OPPONENT_ABBREVIATION"] == "BOS"
    assert rows.loc["BOS P0", "OPPONENT_ABBREVIATION"] == "LAL"


def test_the_history_is_kept_and_only_the_slate_rows_are_added():
    base = panel()
    out = build_forward_rows(
        base, [Game("401700001", "LAL", "BOS")], slate_date="2025-11-15"
    )
    assert len(out.panel) == len(base) + 6
    assert not out.panel.loc[~out.panel[FORWARD_FLAG].astype(bool)].empty


# --- leakage. the point of the whole design. ------------------------------

def test_a_forward_row_carries_no_box_score_stat():
    out = build_forward_rows(
        panel(), [Game("401700001", "LAL", "BOS")], slate_date="2025-11-15"
    )
    rows = forward_only(out.panel)
    present = [c for c in BOX_SCORE_COLUMNS if c in rows.columns]
    assert present, "the fixture has no box-score columns to blank"
    for column in present:
        assert rows[column].isna().all(), f"{column} was written on a forward row"


def test_the_forward_row_s_features_are_its_own_prior_real_games():
    """
    MEASURED, not inspected. Player LAL P0 scores 18..25 over eight games; the
    forward row's PTS_L5 must be the mean of the LAST FIVE of those, 21..25.
    """
    out = build_forward_rows(
        panel(), [Game("401700001", "LAL", "BOS")], slate_date="2025-11-15"
    )
    feat = build_feature_matrix(out.panel)
    row = feat.loc[
        feat[FORWARD_FLAG].astype(bool) & (feat["PLAYER_NAME"] == "LAL P0")
    ].iloc[0]
    assert row["PTS_L5"] == pytest.approx(np.mean([21, 22, 23, 24, 25]))
    assert row["PTS_L10"] == pytest.approx(np.mean([18, 19, 20, 21, 22, 23, 24, 25]))
    assert not pd.isna(row["PTS_L2"]), "the forward row got no layer-2 projection"


def test_a_forward_row_cannot_leak_into_a_real_row_s_features():
    """Adding the forward rows must not change any historical row's features."""
    base = panel()
    before = build_feature_matrix(base).set_index(["PLAYER_ID", "GAME_ID"])["PTS_L5"]
    out = build_forward_rows(
        base, [Game("401700001", "LAL", "BOS")], slate_date="2025-11-15"
    )
    after = build_feature_matrix(out.panel)
    after = after.loc[~after[FORWARD_FLAG].astype(bool)].set_index(
        ["PLAYER_ID", "GAME_ID"]
    )["PTS_L5"]
    pd.testing.assert_series_equal(
        before.sort_index(), after.sort_index(), check_names=False
    )


def test_the_lookahead_assertion_still_passes_with_forward_rows():
    out = build_forward_rows(
        panel(), [Game("401700001", "LAL", "BOS")], slate_date="2025-11-15"
    )
    assert_no_lookahead(build_feature_matrix(out.panel))


# --- who counts as playing -------------------------------------------------

def test_a_player_who_did_not_appear_recently_is_not_projected():
    """
    A row with MIN of 0 is a player on the bench. Projecting one puts a name on
    the board the box score already said did not play.
    """
    base = panel()
    base.loc[base["PLAYER_NAME"] == "LAL P2", "MIN"] = 0.0
    out = build_forward_rows(
        base, [Game("401700001", "LAL", "BOS")], slate_date="2025-11-15"
    )
    assert "LAL P2" not in set(forward_only(out.panel)["PLAYER_NAME"])
    assert out.n_forward_rows == 5


def test_a_player_whose_minutes_are_unknown_is_not_projected():
    base = panel()
    base.loc[base["PLAYER_NAME"] == "LAL P1", "MIN"] = np.nan
    out = build_forward_rows(
        base, [Game("401700001", "LAL", "BOS")], slate_date="2025-11-15"
    )
    assert "LAL P1" not in set(forward_only(out.panel)["PLAYER_NAME"])


def test_only_the_lookback_window_counts():
    """A player who stopped appearing before the window drops out of it."""
    base = panel()
    cutoff = pd.Timestamp("2025-10-21") + pd.Timedelta(days=2 * 3)
    base.loc[
        (base["PLAYER_NAME"] == "LAL P0") & (base["GAME_DATE"] > cutoff), "MIN"
    ] = 0.0
    out = build_forward_rows(
        base, [Game("401700001", "LAL", "BOS")],
        slate_date="2025-11-15", lookback_games=3,
    )
    assert "LAL P0" not in set(forward_only(out.panel)["PLAYER_NAME"])


# --- abstention, every path ------------------------------------------------

def test_a_team_with_no_history_in_the_panel_is_named_not_invented():
    out = build_forward_rows(
        panel(), [Game("401700001", "MIA", "DEN")], slate_date="2025-11-15"
    )
    assert out.status == "DATA_NOT_AVAILABLE"
    assert out.n_forward_rows == 0
    assert out.teams_without_history == ["DEN", "MIA"]
    assert out.games_skipped and "recent history" in out.games_skipped[0]["reason"]


def test_one_unknown_team_still_projects_the_other():
    out = build_forward_rows(
        panel(), [Game("401700001", "LAL", "MIA")], slate_date="2025-11-15"
    )
    assert out.status == "OK"
    assert set(forward_only(out.panel)["TEAM_ABBREVIATION"]) == {"LAL"}
    assert out.teams_without_history == ["MIA"]


def test_an_empty_schedule_is_a_valid_off_day():
    out = build_forward_rows(panel(), [], slate_date="2025-11-15")
    assert out.status == "EMPTY"
    assert out.n_forward_rows == 0
    assert out.panel is not None


def test_an_empty_panel_cannot_be_projected_from():
    out = build_forward_rows(
        pd.DataFrame(), [Game("401700001", "LAL", "BOS")], slate_date="2025-11-15"
    )
    assert out.status == "DATA_NOT_AVAILABLE"
    assert "empty panel" in " ".join(out.notes)


def test_a_game_already_in_the_panel_is_not_duplicated():
    """The real row wins — a synthetic duplicate would double every join."""
    base = panel()
    existing = str(base["GAME_ID"].iloc[0])
    out = build_forward_rows(
        base, [Game(existing, "LAL", "BOS")], slate_date="2025-11-15"
    )
    assert out.n_forward_rows == 0
    assert "already in the panel" in out.games_skipped[0]["reason"]


def test_a_game_missing_a_team_code_is_skipped_with_a_reason():
    out = build_forward_rows(
        panel(), [Game("401700001", "LAL", "")], slate_date="2025-11-15"
    )
    assert out.games_skipped
    assert "missing" in out.games_skipped[0]["reason"]


def test_an_unreachable_schedule_leaves_the_panel_untouched():
    """
    The network is optional. A denied endpoint must degrade to the old
    behaviour with a named reason, not fail the slate.
    """
    import src.ingestion.espn_schedule as sched

    base = panel()
    original = sched.load_slate
    sched.load_slate = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("403 denied"))
    try:
        out = attach_forward_slate(base, slate_date="2025-11-15")
    finally:
        sched.load_slate = original
    assert out.status == "DATA_NOT_AVAILABLE"
    assert "403 denied" in " ".join(out.notes)
    assert len(out.panel) == len(base)


def test_only_pre_tip_games_are_projected():
    """
    A game under way or finished is not something to project: its box score is
    the answer. `SlateResult.pregame_only` is what gets used.
    """
    import src.ingestion.espn_schedule as sched

    class Slate:
        status = "OK"
        notes: list[str] = []
        unmapped_teams: list[str] = []
        games = [Game("401700001", "LAL", "BOS"), Game("401700002", "LAL", "BOS")]
        pregame_only = [Game("401700001", "LAL", "BOS")]

    original = sched.load_slate
    sched.load_slate = lambda *a, **k: Slate()
    try:
        out = attach_forward_slate(panel(), slate_date="2025-11-15")
    finally:
        sched.load_slate = original
    assert set(forward_only(out.panel)["GAME_ID"]) == {"401700001"}
    assert "1 pre-tip game(s) of 2" in " ".join(out.notes)


# --- the slate run uses it -------------------------------------------------

def test_main_attaches_the_forward_slate_and_can_be_turned_off():
    import main

    body = (
        __import__("pathlib").Path(main.__file__)
    ).read_text(encoding="utf-8")
    assert "attach_forward_slate" in body
    assert main.ENV_FORWARD_SLATE == "PROPIQ_FORWARD_SLATE"
    assert main._flag_env(main.ENV_FORWARD_SLATE, True) is True


def test_the_flag_defaults_on_and_respects_an_explicit_off(monkeypatch):
    import main

    monkeypatch.setenv(main.ENV_FORWARD_SLATE, "false")
    assert main._flag_env(main.ENV_FORWARD_SLATE, True) is False
    monkeypatch.setenv(main.ENV_FORWARD_SLATE, "nonsense")
    assert main._flag_env(main.ENV_FORWARD_SLATE, True) is True
