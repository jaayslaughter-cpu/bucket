"""
main.py wired its own inputs away.

``ingest_market_lines`` read the BigDataBall workbook, parsed BOTH frames it
contains, persisted both, and returned only the market one — while
``build_features_and_verify_fatigue`` called ``build_feature_matrix`` with
neither. Since the builder omits a layer whose input is absent rather than
inventing it, the orchestrator silently produced a narrower feature set than
scripts/nba_model_cli.py builds from the same workbook: no team Elo, no market
context, no opponent defence, no blowout hinges.

Nothing failed, because nothing asserted the frames arrived. These tests do.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import main as orchestrator


def _matrix(panel: pd.DataFrame) -> pd.DataFrame:
    """The minimum a feature matrix must carry to pass the fatigue checks."""
    out = panel.copy()
    out["fatigue_multiplier"] = 1.0
    out["is_back_to_back"] = False
    out["is_3_in_4"] = False
    out["is_4_in_5"] = False
    out["PTS_L2"] = 10.0
    return out


def _panel() -> pd.DataFrame:
    return pd.DataFrame({
        "PLAYER_ID": ["1", "1"],
        "GAME_ID": ["0021700001", "0021700002"],
        "GAME_DATE": pd.to_datetime(["2018-01-01", "2018-01-03"]),
        "SEASON": ["2017-18", "2017-18"],
        "TEAM_ABBREVIATION": ["LAL", "LAL"],
        "PTS": [10.0, 12.0],
    })


def test_the_orchestrator_passes_both_workbook_frames_to_the_feature_build(
    monkeypatch,
):
    """The frames reach build_feature_matrix, by name."""
    import src.features.builder as builder

    seen: dict[str, object] = {}

    def _spy(panel, *, team_games=None, market_lines=None):
        seen["team_games"] = team_games
        seen["market_lines"] = market_lines
        return _matrix(panel)

    monkeypatch.setattr(builder, "build_feature_matrix", _spy)
    monkeypatch.setattr(builder, "assert_no_lookahead", lambda df: None)

    team_games = pd.DataFrame({"nba_game_id": ["0021700001"], "PTS": [100]})
    market_lines = pd.DataFrame({"nba_game_id": ["0021700001"], "spread": [-3.5]})

    orchestrator.build_features_and_verify_fatigue(
        _panel(), team_games=team_games, market_lines=market_lines
    )

    assert seen["team_games"] is team_games, "team_games never reached the builder"
    assert seen["market_lines"] is market_lines, "market_lines never reached the builder"


def test_omitting_the_frames_is_reported_rather_than_silent(monkeypatch, caplog):
    """A narrower matrix is allowed — the workbook is licensed and may be
    absent — but it must say so. The defect this replaces was silent."""
    import src.features.builder as builder

    monkeypatch.setattr(
        builder, "build_feature_matrix",
        lambda panel, *, team_games=None, market_lines=None: _matrix(panel),
    )
    monkeypatch.setattr(builder, "assert_no_lookahead", lambda df: None)

    with caplog.at_level("WARNING"):
        orchestrator.build_features_and_verify_fatigue(_panel())
    assert any("ABSENT from this matrix" in r.message for r in caplog.records), (
        "a narrower feature set was built without saying so"
    )


@pytest.mark.parametrize("which", ["team_games", "market_lines", "both"])
def test_an_empty_frame_is_reported_the_same_as_a_missing_one(
    monkeypatch, caplog, which
):
    """EMPTY is absent, because that is how the builder treats it.

    build_feature_matrix gates both layers on ``is not None and not .empty``, so
    a workbook that parsed to zero rows skips Elo and market context exactly as
    a missing frame does. Warning only on None left that case silent — the very
    thing this function was changed to prevent.
    """
    import src.features.builder as builder

    monkeypatch.setattr(
        builder, "build_feature_matrix",
        lambda panel, *, team_games=None, market_lines=None: _matrix(panel),
    )
    monkeypatch.setattr(builder, "assert_no_lookahead", lambda df: None)

    full = pd.DataFrame({"nba_game_id": ["0021700001"], "PTS": [100]})
    empty = pd.DataFrame(columns=["nba_game_id", "PTS"])
    frames = {
        "team_games": (empty, full),
        "market_lines": (full, empty),
        "both": (empty, empty),
    }[which]

    with caplog.at_level("WARNING"):
        orchestrator.build_features_and_verify_fatigue(
            _panel(), team_games=frames[0], market_lines=frames[1]
        )
    assert any("ABSENT from this matrix" in r.message for r in caplog.records), (
        f"an empty {which} frame built a narrower matrix without saying so"
    )


def test_the_training_pointer_names_the_flags_the_command_actually_requires():
    """train-stats has three required options and writes somewhere the scorer
    does not read. A pointer naming only --market fails when followed, and a
    pointer omitting the artifact path trains successfully and still skips
    scoring. Both halves are asserted against the command's real signature."""
    import inspect

    from scripts.nba_model_cli import train_stats

    required = {
        name for name, param in inspect.signature(train_stats).parameters.items()
        if getattr(param.default, "default", None) is ...
    }
    assert {"market", "start_date", "end_date"} <= required, required

    source = Path(orchestrator.__file__).read_text(encoding="utf-8")
    for flag in ("--start-date", "--end-date"):
        assert flag in source, f"{flag} is required by train-stats and unmentioned"
    assert "model_runs/comparison" in source, (
        "the artifact directory train-stats writes to is not named, so a reader "
        "cannot point --model at it"
    )


def test_ingest_market_lines_returns_both_frames(monkeypatch):
    """Both, not just the market one: the team frame is an input to the feature
    build and was being parsed and then dropped on the floor."""
    import src.ingestion.bigdataball as bdb

    team_games = pd.DataFrame({"nba_game_id": ["0021700001"], "PTS": [100]})
    market_lines = pd.DataFrame({
        "nba_game_id": ["0021700001"], "status": ["VALID"], "spread": [-3.5],
    })
    monkeypatch.setattr(
        bdb, "load_bigdataball_workbook", lambda path: (team_games, market_lines)
    )

    returned = orchestrator.ingest_market_lines(Path("unused.xlsx"), persist=False)
    assert isinstance(returned, tuple) and len(returned) == 2
    assert returned[0] is team_games
    assert returned[1] is market_lines


def test_the_stale_training_pointer_is_gone():
    """main.py told the reader to run scripts/train_model.py, which does not
    exist. The real command is nba_model_cli.py train-stats."""
    source = Path(orchestrator.__file__).read_text(encoding="utf-8")
    assert "scripts/train_model.py" not in source
    assert "train-stats" in source
    assert not Path("scripts/train_model.py").exists(), (
        "the script now exists, so this test's premise is stale"
    )


@pytest.mark.parametrize("missing", ["fatigue_multiplier", "PTS_L2"])
def test_the_fatigue_guards_still_refuse_a_matrix_without_them(monkeypatch, missing):
    """The pass-through must not have loosened the checks that were already
    there."""
    import src.features.builder as builder

    def _short(panel, *, team_games=None, market_lines=None):
        return _matrix(panel).drop(columns=[missing])

    monkeypatch.setattr(builder, "build_feature_matrix", _short)
    monkeypatch.setattr(builder, "assert_no_lookahead", lambda df: None)

    with pytest.raises(RuntimeError):
        orchestrator.build_features_and_verify_fatigue(_panel())
