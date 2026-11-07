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


# --- the first fit, which must not wait for tomorrow morning ---------------


def test_no_boot_fit_when_training_is_switched_off(monkeypatch):
    monkeypatch.delenv(worker.ENV_TRAIN_MARKETS, raising=False)
    jobs = {j["id"] for j in worker.describe(worker.build_scheduler())}
    assert "training_boot" not in jobs
    assert worker.training_is_needed_now([]) == []


def test_a_boot_fit_is_scheduled_when_no_artifact_resolves(monkeypatch, tmp_path):
    """
    TURNING TRAINING ON AT ANY HOUR BUT 07:00 PT used to leave the deployment
    in its old state -- every row abstaining -- until the next morning, for
    no reason but the clock. The first fit is the one that matters, because
    it is the difference between a board that can score and one that cannot.
    """
    monkeypatch.setenv(worker.ENV_TRAIN_MARKETS, "PTS")
    monkeypatch.setattr(
        "main.resolve_model_artifact",
        lambda *a, **k: (tmp_path / "absent.json", "test"),
    )
    jobs = {j["id"] for j in worker.describe(worker.build_scheduler())}
    assert "training_boot" in jobs, sorted(jobs)
    # AND THE DAILY JOB IS STILL THERE: the boot fit is an addition, not a
    # replacement, or training would happen exactly once per deploy.
    assert "training" in jobs


def test_the_boot_fit_stops_once_an_artifact_exists(monkeypatch, tmp_path):
    """
    SELF-LIMITING, which is why it needs no switch of its own: it asks the
    same resolver the slate asks. A redeploy loop must not become a training
    loop.
    """
    artifact = tmp_path / "xgboost_PTS.json"
    artifact.write_text("{}")
    monkeypatch.setenv(worker.ENV_TRAIN_MARKETS, "PTS")
    monkeypatch.setattr("main.resolve_model_artifact", lambda *a, **k: (artifact, "test"))
    # THE FRESHNESS HALF IS STUBBED ALLOWED, because this test is about the
    # resolver half only. Left real, it refuses this fixture for a reason that
    # is correct and not the point: a bare .json with no `.meta.json` beside
    # it has no readable age, and an artifact of unknown age is treated as
    # stale. The two halves have their own tests below.
    allowed = type("V", (), {"allowed": True, "reason": None})()
    monkeypatch.setattr(worker, "artifact_freshness_verdict", lambda *a, **k: allowed)

    assert worker.training_is_needed_now(["PTS"]) == []
    jobs = {j["id"] for j in worker.describe(worker.build_scheduler())}
    assert "training_boot" not in jobs, (
        "a boot fit was scheduled although the artifact resolves and exists"
    )


def test_a_resolver_path_that_does_not_exist_counts_as_missing(monkeypatch, tmp_path):
    """
    `PROPIQ_MODEL` pointing at a file that is not there, and the legacy
    `models/xgb_prop_over.json` default, both RESOLVE to a path. Neither can
    be scored with, so neither may suppress the first fit.
    """
    monkeypatch.setattr(
        "main.resolve_model_artifact",
        lambda *a, **k: (tmp_path / "models" / "xgb_prop_over.json", "test"),
    )
    assert worker.training_is_needed_now(["PTS", "REB"]) == ["PTS", "REB"]


def test_the_boot_fit_is_a_scheduled_job_and_not_an_inline_call(monkeypatch):
    """
    A fit takes minutes. Running it before `scheduler.start()` would hold the
    worker down for that long, and a boot at 08:55 would miss the 09:00
    slate entirely. It has to be a job on the scheduler's own pool.
    """
    import ast
    import inspect
    import textwrap

    source = inspect.getsource(worker.build_scheduler)
    assert 'id="training_boot"' in source
    assert 'trigger="date"' in source, (
        "the boot fit is not a date job, so it may be running inline and "
        "delaying the worker's start"
    )

    # AST, NOT A STRING SEARCH. The first version of this test looked for
    # "scheduler.start()" in the source and found it in this function's own
    # COMMENT about not calling it -- prose, matched as if it were code. What
    # it has to rule out is a CALL: `run_training()` or `scheduler.start()`
    # executed while building the table, either of which holds the worker
    # down for the length of a fit.
    tree = ast.parse(textwrap.dedent(source))
    called: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                called.add(func.id)
            elif isinstance(func, ast.Attribute):
                called.add(func.attr)
    assert "run_training" not in called, (
        "build_scheduler CALLS run_training; a fit takes minutes and this "
        "runs before the worker is up, so a boot near 09:00 loses the slate"
    )
    assert "start" not in called, (
        "build_scheduler starts the scheduler; the caller does that, and the "
        "boot fit must be queued rather than run here"
    )


# --- the cutoff: counted in game dates, because the calendar lied ----------


def _production_shaped_dates() -> list[pd.Timestamp]:
    """
    Two complete past seasons and the current one, with the NBA off-season
    gaps between them -- the exact shape the first real fit ran against.
    """
    return (
        list(pd.date_range("2024-10-22", "2025-04-13", freq="D"))
        + list(pd.date_range("2025-10-21", "2026-04-12", freq="D"))
        + list(pd.date_range("2026-10-20", "2026-11-05", freq="D"))
    )


def test_the_cutoff_does_not_land_in_the_off_season():
    """
    THE DEFECT PRODUCTION CAUGHT, IN ONE ASSERTION.

    A 30-day holdout measured as a calendar subtraction from the newest game
    (2026-11-05) puts the cutoff at 2026-10-06, which is in the off-season.
    The last game at or before that is 2026-04-12, so the fit ended there and
    the model saw NONE of the current season. The deploy probe said it
    outright: "Training stopped at 2026-04-12 and we hold games through
    2026-11-05 -- 207 day(s) of games the model never saw, over the 21-day
    limit".

    Counting game dates instead, and shrinking the holdout until the last
    FITTED game is inside the limit, keeps the cutoff on a real game date.
    """
    dates = _production_shaped_dates()
    cutoff, held, lag = worker.choose_train_cutoff(dates, 30, 21)

    assert lag <= 21, (
        f"cutoff {cutoff.date()} leaves a {lag}-day training lag, which the "
        f"freshness check refuses -- this is the 207-day bug"
    )
    assert cutoff >= pd.Timestamp("2026-10-20"), (
        f"the cutoff fell back to {cutoff.date()}, before the current season "
        f"started, so the fit sees none of it"
    )
    assert 1 <= held < len(dates)


@pytest.mark.parametrize("holdout", [7, 14, 21, 30, 60, 120])
def test_the_lag_limit_is_respected_whatever_the_holdout_asks_for(holdout):
    """
    The holdout is a REQUEST; the limit is a constraint. Any holdout that
    would strand the fit behind the limit is shrunk, never honoured.
    """
    dates = _production_shaped_dates()
    _cutoff, held, lag = worker.choose_train_cutoff(dates, holdout, 21)
    assert lag <= 21, f"holdout={holdout} produced a {lag}-day lag"
    assert held >= 1, "nothing was held out, so there is nothing to calibrate on"


def test_the_default_holdout_stays_under_the_freshness_limit(monkeypatch):
    """
    A 30-day default against a 21-day limit is what shipped first, and every
    artifact it wrote was refused on arrival. The default is derived from the
    limit now, so the two cannot drift apart.
    """
    from src.models.artifact_freshness import max_training_lag_days

    assert worker.default_train_holdout_days() < max_training_lag_days()

    monkeypatch.setenv("PROPIQ_MAX_TRAINING_LAG_DAYS", "40")
    assert worker.default_train_holdout_days() < 40
    assert worker.default_train_holdout_days() >= 7


def test_a_panel_too_short_to_satisfy_the_limit_reports_the_lag_rather_than_hiding_it():
    """
    One game date inside the limit means holding out anything at all strands
    the fit behind it. That is a panel problem, so the lag comes back as it
    is and the caller says so -- rather than a split that pretends to fix it.
    """
    dates = [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-11-05")]
    _cutoff, held, lag = worker.choose_train_cutoff(dates, 14, 21)
    assert held == 1
    assert lag > 21, "the honest answer here is a lag over the limit"


def test_a_stale_artifact_counts_as_needing_a_fit(monkeypatch, tmp_path):
    """
    STALE RESOLVES AND EXISTS, so a missing-only check left the deployment
    carrying artifacts the gate refuses until the next 07:00 PT -- for a
    reason a refit fixes in minutes.
    """
    artifact = tmp_path / "xgboost_PTS.json"
    artifact.write_text("{}")
    monkeypatch.setattr("main.resolve_model_artifact", lambda *a, **k: (artifact, "test"))
    refused = type("V", (), {"allowed": False, "reason": "207 days of games unseen"})()
    monkeypatch.setattr(worker, "artifact_freshness_verdict", lambda *a, **k: refused)

    assert worker.training_is_needed_now(["PTS"]) == ["PTS"]


def test_a_fresh_artifact_does_not_trigger_a_fit(monkeypatch, tmp_path):
    """The other half: self-limiting still holds, or every boot refits."""
    artifact = tmp_path / "xgboost_PTS.json"
    artifact.write_text("{}")
    monkeypatch.setattr("main.resolve_model_artifact", lambda *a, **k: (artifact, "test"))
    allowed = type("V", (), {"allowed": True, "reason": None})()
    monkeypatch.setattr(worker, "artifact_freshness_verdict", lambda *a, **k: allowed)

    assert worker.training_is_needed_now(["PTS"]) == []
