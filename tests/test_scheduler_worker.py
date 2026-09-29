"""The deployed worker's schedule, concurrency limits and failure policy.

Nothing here starts a scheduler — a BlockingScheduler's start() does not return.
``build_scheduler`` is separated from ``main`` precisely so the job table can be
inspected, because the times, the grace periods and the concurrency limits are
the part worth asserting.
"""

from __future__ import annotations

import pytest

import scheduler_worker as worker
from src.utils.timezones import DISPLAY_TZ_NAME


class FakeScheduler:
    """Records jobs instead of running them."""

    def __init__(self) -> None:
        self.jobs: list[dict] = []

    def add_job(self, func, **kwargs):
        self.jobs.append({"func": func, **kwargs})
        return kwargs.get("id")

    def get_jobs(self):
        return [
            type("Job", (), {
                "id": j["id"], "name": j["name"],
                "trigger": f"cron[hour={j['hour']},minute={j['minute']}]",
                "max_instances": j["max_instances"], "coalesce": j["coalesce"],
                "misfire_grace_time": j["misfire_grace_time"],
            })()
            for j in self.jobs
        ]


def _jobs() -> dict[str, dict]:
    scheduler = FakeScheduler()
    worker.build_scheduler(scheduler)
    return {j["id"]: j for j in scheduler.jobs}


# --- the schedule --------------------------------------------------------

def test_both_halves_of_the_loop_are_scheduled():
    """Projecting without grading fills the ledger with PENDING rows for ever."""
    jobs = _jobs()
    assert set(jobs) == {"slate", "settlement"}
    assert jobs["slate"]["func"] is worker.run_slate
    assert jobs["settlement"]["func"] is worker.run_settlement


def test_the_schedule_is_anchored_to_pacific_not_utc():
    """
    Every slate cutoff in this repository is a Pacific calendar day. A
    UTC-anchored schedule would move the late West Coast games of one slate into
    the next day's run for part of the year and not the rest.
    """
    for job in _jobs().values():
        assert str(job["timezone"]) == DISPLAY_TZ_NAME


def test_settlement_runs_after_the_games_and_the_slate_before_them():
    jobs = _jobs()
    assert jobs["slate"]["hour"] == 9, "9am PT is before the earliest NBA tip"
    assert jobs["settlement"]["hour"] == 3, (
        "3am PT is after a West Coast overtime game has finished and posted"
    )


def test_the_times_are_configurable(monkeypatch):
    monkeypatch.setenv(worker.ENV_SLATE_HOUR, "11")
    monkeypatch.setenv(worker.ENV_SLATE_MINUTE, "45")
    jobs = _jobs()
    assert (jobs["slate"]["hour"], jobs["slate"]["minute"]) == (11, 45)


@pytest.mark.parametrize("bad", ["25", "-1", "half past nine", ""])
def test_an_unusable_hour_falls_back_to_the_default_rather_than_crashing(
    monkeypatch, bad
):
    """An unattended worker that will not start is worse than one on defaults."""
    monkeypatch.setenv(worker.ENV_SLATE_HOUR, bad)
    assert _jobs()["slate"]["hour"] == worker.DEFAULT_SLATE_HOUR


# --- concurrency ---------------------------------------------------------

def test_no_job_can_run_twice_at_once():
    """
    XGBoost and CatBoost default to every core, so a second concurrent fit is
    contention rather than parallelism — measured in this project's own runs,
    three concurrent test suites turned 126s into over 590s.
    """
    for job in _jobs().values():
        assert job["max_instances"] == 1
        assert job["coalesce"] is True, "a backlog of misfires would pile up"


def test_a_slate_missed_by_hours_is_not_run_late():
    """Running late would project games that have already tipped."""
    jobs = _jobs()
    assert jobs["slate"]["misfire_grace_time"] == 60 * 60
    # Settlement has no such hurry: a finished game stays finished.
    assert jobs["settlement"]["misfire_grace_time"] > jobs["slate"]["misfire_grace_time"]


# --- thread caps ---------------------------------------------------------

def test_the_thread_cap_is_applied_to_every_pool(monkeypatch):
    for variable in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    ):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv(worker.ENV_MAX_THREADS, "2")

    assert worker.cap_thread_counts() == 2
    import os

    assert os.environ["OMP_NUM_THREADS"] == "2"
    assert os.environ["MKL_NUM_THREADS"] == "2"


def test_an_existing_thread_setting_is_not_overridden(monkeypatch):
    """The deployer's own value beats ours."""
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.setenv(worker.ENV_MAX_THREADS, "2")
    worker.cap_thread_counts()
    import os

    assert os.environ["OMP_NUM_THREADS"] == "8"


def test_no_cap_is_applied_when_none_is_configured(monkeypatch):
    monkeypatch.delenv(worker.ENV_MAX_THREADS, raising=False)
    assert worker.cap_thread_counts() is None


# --- failure policy ------------------------------------------------------

def test_a_failing_slate_does_not_take_the_worker_down(monkeypatch):
    """One bad day must not end every subsequent day too."""
    import main

    def explode(argv=None):
        raise RuntimeError("ingestion blew up")

    monkeypatch.setattr(main, "main", explode)
    assert worker.run_slate([]) == 1


def test_a_failing_settlement_reports_rather_than_raising(monkeypatch):
    import src.settlement.runner as runner

    def explode(*args, **kwargs):
        raise RuntimeError("box score host is down")

    monkeypatch.setattr(runner, "settle_pending_props", explode)
    out = worker.run_settlement()
    assert out["status"] == "FAILED"
    assert "box score host is down" in out["error"]


def test_the_slate_job_delegates_rather_than_reimplementing_the_pipeline():
    """
    A second copy of the sequence would drift from main.py, and the scheduled
    path is the one nobody watches.
    """
    import inspect

    source = inspect.getsource(worker.run_slate)
    assert "main.main" in source


# --- what it says about itself -------------------------------------------

def test_the_worker_states_that_it_places_nothing():
    assert "RESEARCH_ONLY" in worker.__doc__
    assert "places no wager" in worker.__doc__


# --- dispatch, which the worker did not do at all before -----------------

def test_the_slate_job_now_builds_a_board_and_dispatches(monkeypatch):
    """
    Before this, main.py and this worker contained no reference to src/notify:
    the scheduled run wrote projections and told nobody.
    """
    import main

    calls = []
    monkeypatch.setattr(main, "main", lambda argv=None: 0)
    monkeypatch.setattr(worker, "run_board", lambda: calls.append("board") or {})
    monkeypatch.setattr(worker, "run_dispatch", lambda: calls.append("dispatch") or {})

    assert worker.run_slate([]) == 0
    assert calls == ["board", "dispatch"], "board must be built before it is sent"


def test_a_failed_slate_dispatches_nothing(monkeypatch):
    """Otherwise a broken build posts yesterday's board as today's card."""
    import main

    calls = []
    monkeypatch.setattr(main, "main", lambda argv=None: 1)
    monkeypatch.setattr(worker, "run_board", lambda: calls.append("board") or {})
    monkeypatch.setattr(worker, "run_dispatch", lambda: calls.append("dispatch") or {})

    assert worker.run_slate([]) == 1
    assert calls == []


def test_dispatch_is_off_without_a_webhook(monkeypatch):
    """A worker with no DISCORD_WEBHOOK_URL should not spend a slate trying."""
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    monkeypatch.delenv(worker.ENV_DISPATCH, raising=False)
    out = worker.run_dispatch()
    assert out["status"] == "SKIPPED"
    assert "disabled" in out["reason"]


def test_dispatch_can_be_forced_off_even_with_a_webhook(monkeypatch):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/1/x")
    monkeypatch.setenv(worker.ENV_DISPATCH, "false")
    assert worker.run_dispatch()["status"] == "SKIPPED"


def test_a_missing_board_dispatches_an_honest_abstention(monkeypatch, tmp_path):
    """
    The state today: nothing recommended. That must read as the pipeline
    reporting itself, not as a slate that was examined and found empty.
    """
    sent = {}

    def fake_send(embeds, config=None, **kw):
        sent["embeds"] = list(embeds)
        return type("R", (), {"status": "OK"})()

    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/1/x")
    monkeypatch.setenv(worker.ENV_DISPATCH, "true")
    monkeypatch.setenv(worker.ENV_BOARD_CSV, str(tmp_path / "absent.csv"))
    monkeypatch.setenv(worker.ENV_CALIBRATION_REPORT, str(tmp_path / "absent.json"))

    import src.notify.discord as discord_mod

    monkeypatch.setattr(discord_mod, "send_embeds", fake_send)

    out = worker.run_dispatch()
    assert out["status"] == "OK"
    body = sent["embeds"][0]["description"]
    assert "No board was written" in body
    assert "not a slate with no value" in body


def test_abstentions_can_be_muted(monkeypatch, tmp_path):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/1/x")
    monkeypatch.setenv(worker.ENV_DISPATCH, "true")
    monkeypatch.setenv(worker.ENV_DISPATCH_ABSTENTIONS, "false")
    monkeypatch.setenv(worker.ENV_BOARD_CSV, str(tmp_path / "absent.csv"))
    out = worker.run_dispatch()
    assert out["status"] == "SKIPPED"


def test_a_board_with_no_calibration_report_is_dispatched_withheld(
    monkeypatch, tmp_path
):
    """The gate is applied in the worker, not only in the CLI."""
    import pandas as pd

    sent = {}

    def fake_send(embeds, config=None, **kw):
        sent["embeds"] = list(embeds)
        return type("R", (), {"status": "OK"})()

    board = tmp_path / "board.csv"
    pd.DataFrame([{
        "slate_date": "2026-09-29", "decision_status": "RECOMMENDED",
        "decision_basis": "model_lean", "player_name": "DEMO_A",
        "target_market": "PTS", "side": "over", "line": 24.5,
        "model_prob": 0.71, "book_ev": None, "american_odds": None,
        "book_source": None,
    }]).to_csv(board, index=False)

    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/1/x")
    monkeypatch.setenv(worker.ENV_DISPATCH, "true")
    monkeypatch.setenv(worker.ENV_BOARD_CSV, str(board))
    monkeypatch.setenv(worker.ENV_CALIBRATION_REPORT, str(tmp_path / "absent.json"))

    import src.notify.discord as discord_mod

    monkeypatch.setattr(discord_mod, "send_embeds", fake_send)

    out = worker.run_dispatch()
    assert out["status"] == "OK"
    assert out["publication"]["PUBLICATION_STATUS"] == "PUBLISH_WITHHELD"
    body = sent["embeds"][0]["description"]
    assert "withheld from publication" in body
    assert "DEMO_A" not in str(sent["embeds"][0])


def test_settlement_rebuilds_the_calibration_report(monkeypatch, tmp_path):
    """
    The order is the dependency: grade, then recompute the evidence, so the
    morning run reads evidence that includes last night's results.
    """
    import src.settlement.runner as runner

    monkeypatch.setattr(
        runner, "settle_pending_props",
        lambda *a, **k: type("R", (), {"as_dict": lambda self: {"props_graded": 0}})(),
    )
    target = tmp_path / "calibration.json"
    monkeypatch.setenv(worker.ENV_CALIBRATION_REPORT, str(target))

    import src.settlement.calibration as cal

    monkeypatch.setattr(
        cal, "prop_result_calibration_report",
        lambda **kw: {"status": "DATA_NOT_AVAILABLE", "reason": "nothing settled",
                      "n_scored": 0},
    )

    out = worker.run_settlement()
    assert out["calibration"]["status"] == "DATA_NOT_AVAILABLE"
    assert target.exists(), "an abstaining report must still be written"


def test_an_abstaining_report_is_written_so_a_stale_pass_cannot_linger(
    monkeypatch, tmp_path
):
    """
    Not writing it would leave yesterday's PASSING report in place after the
    rows behind it aged out of the window.
    """
    import json

    import src.settlement.calibration as cal

    target = tmp_path / "calibration.json"
    target.write_text(json.dumps({"status": "OK", "ece": 0.01, "n_scored": 500}))
    monkeypatch.setattr(
        cal, "prop_result_calibration_report",
        lambda **kw: {"status": "DATA_NOT_AVAILABLE", "reason": "aged out"},
    )

    worker.rebuild_calibration_report(str(target))
    assert json.loads(target.read_text())["status"] == "DATA_NOT_AVAILABLE"
