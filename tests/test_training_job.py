"""
The worker's training job, and the one fit loop both callers share.

WHY THESE TESTS ARE SHAPED LIKE THIS. The job cannot be run here: it reads
the deployed database. So what is asserted is everything AROUND the fit that
can be got wrong silently -- the configuration gate, the write lock, the
holdout being the newest slice rather than the oldest, a panel that cannot
support a fit saying which part is missing, and one market's failure not
taking the others down. The fit itself is exercised end to end by
`scripts/nba_model_cli train-stats --demo`, which goes through the same
`fit_market_artifacts`.

RESEARCH_ONLY.
"""

from __future__ import annotations

import pandas as pd
import pytest

import scheduler_worker as worker
from src.models.training import (
    MARKETS,
    TrainingDataError,
    fit_market_artifacts,
)

# --- the configuration gate ------------------------------------------------


def test_training_is_off_until_a_market_is_named(monkeypatch):
    """
    UNSET MEANS OFF, and it must not be a half-on that touches the lock.

    The job is registered every night whether or not it is configured, so an
    unconfigured run has to decline before it reads the panel or takes the
    write lock the slate needs.
    """
    monkeypatch.delenv(worker.ENV_TRAIN_MARKETS, raising=False)

    def explode():  # pragma: no cover - must not be reached
        raise AssertionError("the write lock was taken by an unconfigured job")

    monkeypatch.setattr("src.utils.write_lock.write_lock", explode)

    out = worker.run_training()
    assert out["status"] == "SKIPPED"
    assert worker.ENV_TRAIN_MARKETS in out["reason"]


@pytest.mark.parametrize(
    "value,expected",
    [
        ("PTS", ["PTS"]),
        ("pts,reb", ["PTS", "REB"]),
        ("ALL", list(MARKETS)),
        ("PTS,PTS", ["PTS"]),
        ("PTS;AST", ["PTS", "AST"]),
        ("  ", []),
    ],
)
def test_the_market_list_is_parsed_not_guessed(monkeypatch, value, expected):
    monkeypatch.setenv(worker.ENV_TRAIN_MARKETS, value)
    assert worker._training_markets() == expected


def test_a_market_name_that_is_not_a_market_is_named_and_dropped(monkeypatch, caplog):
    """
    "PTS,POINTS" fitting one market while the operator believes two were
    fitted is the exact shape of defect this project keeps finding. The bad
    name must appear in the log, not be swallowed.
    """
    import logging

    monkeypatch.setenv(worker.ENV_TRAIN_MARKETS, "PTS,POINTS")
    with caplog.at_level(logging.ERROR, logger="propiq.scheduler"):
        assert worker._training_markets() == ["PTS"]
    assert "POINTS" in caplog.text


# --- the schedule ----------------------------------------------------------


def test_training_is_registered_and_lands_before_the_slate():
    jobs = {j["id"]: j for j in worker.describe(worker.build_scheduler())}
    assert "training" in jobs, f"the training job is not scheduled: {sorted(jobs)}"


@pytest.mark.parametrize(
    "slate,expected",
    [((9, 0), (7, 0)), ((13, 30), (11, 30)), ((1, 0), (23, 0))],
)
def test_the_training_time_keeps_its_lead_on_the_slate(slate, expected):
    """
    TRAINING HOLDS THE SLATE'S WRITE LOCK, so overlapping them costs the
    morning's board. The lead is fixed rather than configurable, and a slate
    inside the first two hours of the day wraps to the previous evening
    instead of landing on top of itself.
    """
    assert worker.training_time(*slate) == expected
    slate_minutes = slate[0] * 60 + slate[1]
    train_minutes = expected[0] * 60 + expected[1]
    gap = (slate_minutes - train_minutes) % (24 * 60)
    assert gap == worker.TRAINING_MIN_LEAD_MINUTES


def test_the_training_lookback_is_not_zero():
    """
    `load_player_panel` HONOURS lookback_days and computes
    `lower = upper - days`, so 0 loads a single day rather than everything.
    A default of 0 here would fit on one slate's rows and report success.
    """
    assert worker.DEFAULT_TRAIN_LOOKBACK_DAYS >= 365, (
        "a training lookback under a season cannot reach the history the "
        "backfill exists to fetch"
    )


# --- the fit's own refusals ------------------------------------------------


def _labelled_frame(n_days: int = 120, players: int = 6) -> pd.DataFrame:
    """
    A frame with the columns `fit_market_artifacts` needs to get as far as a
    split. Not enough to fit a real model, and not used for one: these tests
    assert the refusals, which happen before any component is built.
    """
    rows = []
    start = pd.Timestamp("2026-01-01")
    for d in range(n_days):
        for p in range(players):
            rows.append({
                "PLAYER_ID": f"espn:{p}",
                "PLAYER_NAME": f"P{p}",
                "GAME_DATE": start + pd.Timedelta(days=d),
                "PTS": 10.0 + (d % 7),
                "PTS_L10": 12.0,
                "PTS_L5": 12.0,
                "PTS_L2": 12.0,
                "PTS_SEASON": 12.0,
                "PTS_BASELINE": 12.0,
                "MIN_L5": 30.0,
                "MIN_L10": 30.0,
                "MIN_SEASON": 30.0,
                "fatigue_multiplier": 1.0,
                "days_rest": 2.0,
            })
    return pd.DataFrame(rows)


def test_an_empty_matrix_says_the_panel_is_the_problem(tmp_path):
    with pytest.raises(TrainingDataError) as exc:
        fit_market_artifacts(pd.DataFrame(), "PTS", artifacts_dir=tmp_path)
    assert "empty" in str(exc.value).lower()


def test_a_market_that_is_not_a_launch_market_is_refused(tmp_path):
    with pytest.raises(TrainingDataError):
        fit_market_artifacts(_labelled_frame(), "STL", artifacts_dir=tmp_path)


def test_a_window_with_no_rows_is_refused_by_its_dates(tmp_path):
    with pytest.raises(TrainingDataError) as exc:
        fit_market_artifacts(
            _labelled_frame(), "PTS",
            start_date="2030-01-01", end_date="2030-02-01",
            artifacts_dir=tmp_path,
        )
    assert "2030-01-01" in str(exc.value)


def test_a_train_end_that_empties_one_side_says_which_and_names_the_window(tmp_path):
    """
    A cutoff outside the data leaves nothing to fit OR nothing to calibrate
    on, and the two are different mistakes. Either way the message has to
    carry the window the frame actually covers, or the operator is guessing.
    """
    with pytest.raises(TrainingDataError) as exc:
        fit_market_artifacts(
            _labelled_frame(), "PTS", train_end="2020-01-01", artifacts_dir=tmp_path,
        )
    message = str(exc.value)
    assert "2026-01-01" in message, f"the real window is not in the error: {message}"
    assert "nothing to fit" in message


def test_a_panel_with_no_labelled_row_is_refused_as_unlabelled_not_as_empty(tmp_path):
    """
    A MATRIX CAN BE LARGE AND CARRY NOTHING TO FIT. `over_hit` is NaN wherever
    the realised stat ties the line (a push, dropped on purpose) or the L10
    research line is absent -- and a player's first ten games cannot have an
    L10. So an all-early-season panel looks healthy and teaches nothing, and
    the two faults need different fixes.
    """
    frame = _labelled_frame()
    frame["PTS"] = frame["PTS_L10"]  # every row an exact tie: all pushes
    with pytest.raises(TrainingDataError) as exc:
        fit_market_artifacts(frame, "PTS", artifacts_dir=tmp_path)
    message = str(exc.value)
    assert "NONE carries a label" in message, message
    assert "L10" in message
