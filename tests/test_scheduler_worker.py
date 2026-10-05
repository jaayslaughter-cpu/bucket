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


# --- the state directory probe ------------------------------------------

def test_a_writable_state_dir_reports_writable(tmp_path, monkeypatch):
    monkeypatch.setenv(worker.ENV_CALIBRATION_REPORT, str(tmp_path / "calibration.json"))
    out = worker.check_state_dir()
    assert out["writable"] is True
    assert not list(tmp_path.glob(".propiq_write_probe")), "the probe file was left behind"


@pytest.mark.skipif(
    hasattr(__import__("os"), "getuid") and __import__("os").getuid() == 0,
    reason=(
        "root bypasses directory permission bits, so an unwritable directory "
        "cannot be simulated. Worth noting as a deployment fact: running the "
        "Railway service as root is one of the two fixes for a root-owned "
        "volume, and it makes this failure mode impossible."
    ),
)
def test_an_unwritable_state_dir_is_named_rather_than_surfacing_later(
    tmp_path, monkeypatch, caplog
):
    """
    A Railway volume mounted at /app/data REPLACES the directory the Dockerfile
    chowned, commonly with a root-owned one, while this process runs as uid
    10001. Every writer here catches broadly, so without this probe the failure
    appears as "calibration report failed" every night — the symptom, not the
    cause.
    """
    import logging

    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)  # readable, not writable
    try:
        monkeypatch.setenv(
            worker.ENV_CALIBRATION_REPORT, str(locked / "calibration.json")
        )
        with caplog.at_level(logging.ERROR):
            out = worker.check_state_dir()
        assert out["writable"] is False
        assert "NOT WRITABLE" in caplog.text
        assert "owned by root" in caplog.text
        assert "still writes to Postgres" in caplog.text, (
            "the operator should know what still works"
        )
    finally:
        locked.chmod(0o700)


def test_the_probe_never_raises_and_still_names_the_cause(monkeypatch, caplog):
    """
    A worker that will not start is worse than one that logs the problem.

    This also carries the message assertions under root, where the
    permission-bits test above is skipped: /proc is unwritable for everyone, so
    the error branch and its wording are exercised either way.
    """
    import logging

    monkeypatch.setenv(worker.ENV_CALIBRATION_REPORT, "/proc/nonexistent/x.json")
    with caplog.at_level(logging.ERROR):
        out = worker.check_state_dir()
    assert out["writable"] is False
    assert out["error"]
    assert "NOT WRITABLE" in caplog.text
    assert "owned by root" in caplog.text
    assert "still writes to Postgres" in caplog.text, (
        "the operator should know what still works"
    )


def test_the_probe_runs_at_boot():
    import inspect

    assert "check_state_dir()" in inspect.getsource(worker.main)


# --- deployment manifests ------------------------------------------------

def test_every_env_var_the_code_reads_is_documented():
    """
    An undocumented variable is one nobody sets on purpose. 16 of 28 were
    missing from .env.example when the Railway audit ran, including the two
    that decide whether a deployed run keeps anything.
    """
    import pathlib
    import re

    root = pathlib.Path(__file__).parent.parent
    sources = (
        list((root / "src").rglob("*.py"))
        + list((root / "scripts").glob("*.py"))
        + [root / "main.py", root / "scheduler_worker.py"]
    )
    pattern = re.compile(
        r'os\.environ(?:\.get)?\(\s*["\']([A-Z_][A-Z0-9_]*)["\']'
        r'|os\.getenv\(\s*["\']([A-Z_][A-Z0-9_]*)["\']'
        r'|ENV_[A-Z_]+\s*=\s*["\']([A-Z_][A-Z0-9_]*)["\']'
    )
    read: set[str] = set()
    for path in sources:
        for m in pattern.finditer(path.read_text()):
            read.add(m.group(1) or m.group(2) or m.group(3))

    documented = (root / ".env.example").read_text()
    missing = sorted(n for n in read if n not in documented)
    assert not missing, f"{len(missing)} env var(s) absent from .env.example: {missing}"


def test_the_worker_scheduler_is_installable_from_requirements_txt():
    """
    Railway's Nixpacks builder reads requirements.txt, NOT pyproject's extras.
    APScheduler was only in the extra, so a Nixpacks build produced a container
    whose worker hit its import guard and exited in a restart loop.
    """
    import pathlib

    req = (pathlib.Path(__file__).parent.parent / "requirements.txt").read_text()
    assert "APScheduler" in req


def test_the_image_pins_a_timezone_and_a_volume_backed_report_path():
    """
    TZ so the container's default is a decision rather than the host's, and the
    calibration report on the mounted volume because the settlement job writes
    it at 03:30 PT and the slate job reads it at 09:00 PT.
    """
    import pathlib

    dockerfile = (pathlib.Path(__file__).parent.parent / "Dockerfile").read_text()
    assert "ENV TZ=" in dockerfile
    assert "PROPIQ_CALIBRATION_REPORT=/app/data/" in dockerfile


# ---------------------------------------------------------------------------
# B3 — the board window. Two hardcoded 2025 dates used to be the fallback.
# ---------------------------------------------------------------------------

def test_the_default_window_is_anchored_on_today_not_on_two_2025_dates(monkeypatch):
    """
    THE DEFECT: run_board fell back to train_end="2025-01-15" /
    validation_end="2025-02-15". compare_models_on_panel scores
    (train_end, validation_end], so every board row described a game in early
    February 2025, the board stamped each one with today's slate date, and the
    window receded one day further into the past on every run.
    """
    from datetime import date

    monkeypatch.delenv("PROPIQ_BOARD_TRAIN_END", raising=False)
    monkeypatch.delenv("PROPIQ_BOARD_VALIDATION_END", raising=False)

    window = worker.board_window(today=date(2026, 10, 5))
    assert window["validation_end"] == "2026-10-05"
    assert window["train_end"] == "2026-10-04"
    assert window["is_backtest"] is False
    assert window["lag_days"] == 0
    assert "2025-01-15" not in window.values()
    assert "2025-02-15" not in window.values()


def test_the_window_scores_exactly_today(monkeypatch):
    """
    fixed_cutoff_split fits on `<= train_end` and scores
    `(train_end, validation_end]`, so yesterday/today scores today's rows and
    nothing else. forward_slate writes one row per scheduled game for today.
    """
    from datetime import date, timedelta

    monkeypatch.delenv("PROPIQ_BOARD_TRAIN_END", raising=False)
    monkeypatch.delenv("PROPIQ_BOARD_VALIDATION_END", raising=False)

    today = date(2026, 1, 20)
    window = worker.board_window(today=today)
    train = date.fromisoformat(window["train_end"])
    validation = date.fromisoformat(window["validation_end"])
    assert validation == today
    assert validation - train == timedelta(days=1)


def test_the_window_moves_with_the_day(monkeypatch):
    """A fixed fallback does not. That was the whole bug."""
    from datetime import date

    monkeypatch.delenv("PROPIQ_BOARD_TRAIN_END", raising=False)
    monkeypatch.delenv("PROPIQ_BOARD_VALIDATION_END", raising=False)

    first = worker.board_window(today=date(2026, 3, 1))
    later = worker.board_window(today=date(2026, 3, 2))
    assert first["validation_end"] != later["validation_end"]
    assert later["validation_end"] == "2026-03-02"


def test_an_explicit_override_still_wins(monkeypatch):
    """A backtest board is a legitimate thing to build; that is what these are
    for. What it must no longer do is happen by accident."""
    from datetime import date

    monkeypatch.setenv("PROPIQ_BOARD_TRAIN_END", "2025-01-15")
    monkeypatch.setenv("PROPIQ_BOARD_VALIDATION_END", "2025-02-15")
    window = worker.board_window(today=date(2026, 10, 5))
    assert window["train_end"] == "2025-01-15"
    assert window["validation_end"] == "2025-02-15"
    assert set(window["overridden"]) == {
        "PROPIQ_BOARD_TRAIN_END", "PROPIQ_BOARD_VALIDATION_END"
    }


def test_a_past_window_is_reported_as_a_backtest_with_its_lag(monkeypatch, caplog):
    """
    The old fallback was silent. A board whose rows describe games already
    played must say so and name how stale it is.
    """
    import logging
    from datetime import date

    monkeypatch.setenv("PROPIQ_BOARD_VALIDATION_END", "2025-02-15")
    monkeypatch.delenv("PROPIQ_BOARD_TRAIN_END", raising=False)
    with caplog.at_level(logging.WARNING):
        window = worker.board_window(today=date(2026, 10, 5))
    assert window["is_backtest"] is True
    assert window["lag_days"] == 597
    blob = caplog.text
    assert "BACKTEST" in blob
    assert "597" in blob


def test_a_window_ending_today_is_not_called_a_backtest(monkeypatch):
    from datetime import date

    monkeypatch.setenv("PROPIQ_BOARD_VALIDATION_END", "2026-10-05")
    monkeypatch.delenv("PROPIQ_BOARD_TRAIN_END", raising=False)
    window = worker.board_window(today=date(2026, 10, 5))
    assert window["is_backtest"] is False


def test_an_unparseable_override_is_passed_through_rather_than_crashing(
    monkeypatch, caplog
):
    """A worker that will not start cannot report anything."""
    import logging
    from datetime import date

    monkeypatch.setenv("PROPIQ_BOARD_VALIDATION_END", "not-a-date")
    monkeypatch.delenv("PROPIQ_BOARD_TRAIN_END", raising=False)
    with caplog.at_level(logging.WARNING):
        window = worker.board_window(today=date(2026, 10, 5))
    assert window["validation_end"] == "not-a-date"
    assert window["lag_days"] is None
    assert window["is_backtest"] is False
    assert "not an ISO date" in caplog.text


# ---------------------------------------------------------------------------
# B1 — the artifact preflight. Its absence used to surface only at inference.
# ---------------------------------------------------------------------------

def test_no_resolvable_artifact_is_an_error_at_boot_naming_every_path(
    monkeypatch, caplog
):
    """
    THE DEFECT: without an artifact the pipeline does not fail. score_prob_over
    returns an all-null Series, every row abstains, the slate job exits 0, and
    a fresh container with an unseeded volume looks healthy while publishing
    nothing. The question has to be asked at boot.
    """
    import logging

    monkeypatch.delenv("PROPIQ_MODEL", raising=False)
    monkeypatch.setattr(
        "main.resolve_model_artifact",
        lambda *a, **k: (None, "no artifact found: tried A, B and C"),
    )
    with caplog.at_level(logging.ERROR):
        info = worker.check_model_artifact()
    assert info["resolved"] is False
    assert "tried A, B and C" in info["reason"]
    assert "NO SCORING ARTIFACT RESOLVED" in caplog.text
    assert "ABSTAIN" in caplog.text


def test_a_path_that_is_set_but_absent_is_an_error_not_a_silence(
    monkeypatch, caplog, tmp_path
):
    """The likeliest first-deploy misconfiguration: PROPIQ_MODEL set, volume
    unseeded."""
    import logging

    missing = tmp_path / "not_there" / "xgboost_PTS.json"
    monkeypatch.setattr(
        "main.resolve_model_artifact", lambda *a, **k: (missing, "PROPIQ_MODEL")
    )
    with caplog.at_level(logging.ERROR):
        info = worker.check_model_artifact()
    assert info["exists"] is False
    assert "DOES NOT EXIST" in caplog.text


def test_an_artifact_with_no_sidecar_is_an_error(monkeypatch, caplog, tmp_path):
    """Scoring refuses it anyway — the feature_cols are unknown — so saying so
    at boot is the difference between a named cause and a silent abstention."""
    import logging

    artifact = tmp_path / "xgboost_PTS.json"
    artifact.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "main.resolve_model_artifact", lambda *a, **k: (artifact, "glob")
    )
    with caplog.at_level(logging.ERROR):
        info = worker.check_model_artifact()
    assert info["sidecar"] is False
    assert "NO .meta.json SIDECAR" in caplog.text


def _artifact(tmp_path, rows: int):
    import json

    artifact = tmp_path / "xgboost_PTS.json"
    artifact.write_text("{}", encoding="utf-8")
    (tmp_path / "xgboost_PTS.meta.json").write_text(
        json.dumps({
            "feature_cols": ["a", "b", "c"],
            "target_market": "PTS",
            "train_row_count": rows,
            "train_start_date": "2018-01-01",
            "train_end_date": "2026-04-12",
            "feature_schema_version": "fs_v1_shift1_l2",
            "saved_at_utc": "2026-10-05T00:00:00+00:00",
        }),
        encoding="utf-8",
    )
    return artifact


def test_a_demo_sized_artifact_is_flagged_rather_than_scored_quietly(
    monkeypatch, caplog, tmp_path
):
    """
    The worst outcome in this file: a model fit on ~1,000 synthetic rows
    scoring a live slate produces confident-looking numbers rather than
    abstentions. This repository's demo artifacts were fit on 968 rows.
    """
    import logging

    artifact = _artifact(tmp_path, rows=968)
    monkeypatch.setattr(
        "main.resolve_model_artifact", lambda *a, **k: (artifact, "glob")
    )
    with caplog.at_level(logging.ERROR):
        info = worker.check_model_artifact()
    assert info["implausible_train_rows"] is True
    assert "968" in caplog.text
    assert str(worker.MIN_PLAUSIBLE_TRAIN_ROWS) in caplog.text


def test_a_real_sized_artifact_reports_itself_and_raises_no_alarm(
    monkeypatch, caplog, tmp_path
):
    import logging

    artifact = _artifact(tmp_path, rows=214_381)
    monkeypatch.setattr(
        "main.resolve_model_artifact", lambda *a, **k: (artifact, "glob")
    )
    with caplog.at_level(logging.INFO):
        info = worker.check_model_artifact()
    assert info["resolved"] and info["exists"] and info["sidecar"]
    assert "implausible_train_rows" not in info
    assert info["train_row_count"] == 214_381
    assert info["market"] == "PTS"
    assert info["feature_count"] == 3
    # One artifact scores one market, and the log says which.
    assert "fit for PTS only" in caplog.text
    assert "ERROR" not in caplog.text.upper().replace("ERRORS", "")


def test_the_preflight_never_takes_the_boot_down(monkeypatch):
    """Same policy as check_state_dir: a worker that refuses to start cannot
    report anything, and the settlement job is still useful with no model."""
    def _boom(*a, **k):
        raise RuntimeError("resolver exploded")

    monkeypatch.setattr("main.resolve_model_artifact", _boom)
    info = worker.check_model_artifact()
    assert info["resolved"] is False


def test_both_boot_probes_run_before_the_scheduler_is_built():
    """A redeploy breaks the same two things: the volume's permissions and
    whether anything was ever put on it."""
    import inspect

    source = inspect.getsource(worker.main)
    assert "check_state_dir()" in source
    assert "check_model_artifact()" in source
    assert source.index("check_model_artifact()") < source.index("build_scheduler()")


def test_run_board_actually_uses_the_rolling_window(monkeypatch):
    """
    THE HELPER IS NOT THE WIRING. board_window can be correct while run_board
    ignores it, and every test above would still pass — the first version of
    this file had exactly that gap, and replacing run_board's call with the two
    hardcoded 2025 dates left it green.

    So this captures what build_slate_board is actually handed.
    """
    from datetime import date, timedelta

    import src.pipeline.slate_board as slate_board_module
    from src.utils import timezones

    monkeypatch.delenv("PROPIQ_BOARD_TRAIN_END", raising=False)
    monkeypatch.delenv("PROPIQ_BOARD_VALIDATION_END", raising=False)

    today = date(2026, 10, 5)
    monkeypatch.setattr(timezones, "pacific_calendar_date", lambda *a, **k: today)

    seen: dict[str, object] = {}

    class _Result:
        def as_dict(self):
            return {"written_rows": 0}

    def _fake_build(panel, **kwargs):
        seen.update(kwargs)
        return _Result()

    monkeypatch.setattr(slate_board_module, "build_slate_board", _fake_build)

    import pandas as pd

    import src.db.repository as repo
    monkeypatch.setattr(
        repo, "load_player_panel",
        lambda *a, **k: pd.DataFrame({"PLAYER_ID": ["1"], "GAME_DATE": [today]}),
    )

    out = worker.run_board()
    assert out["status"] == "OK"
    assert seen["validation_end"] == str(today)
    assert seen["train_end"] == str(today - timedelta(days=1))
    assert seen["train_end"] != "2025-01-15"
    assert seen["validation_end"] != "2025-02-15"
    # And the window travels with the result, so a log line says WHICH days
    # were scored rather than only how many rows came back.
    assert out["window"]["validation_end"] == str(today)
    assert out["window"]["is_backtest"] is False


# ---------------------------------------------------------------------------
# B4 — the tip-anchored pre-lock re-check. The worker had no such job at all.
# ---------------------------------------------------------------------------

from datetime import datetime, timedelta, timezone  # noqa: E402


class _Game:
    """One SlateGame, in the shape schedule_prelock_jobs reads."""

    def __init__(self, event_id, tipoff, *, home="BOS", away="LAL",
                 slate_date_pt="2026-10-05"):
        self.espn_event_id = event_id
        self.tipoff_utc = tipoff
        self.home_team = home
        self.away_team = away
        self.slate_date_pt = slate_date_pt


class _Slate:
    def __init__(self, games):
        self._games = games

    @property
    def pregame_only(self):
        return self._games


NOW = datetime(2026, 10, 5, 18, 0, tzinfo=timezone.utc)


def _tip(hours: float) -> datetime:
    return NOW + timedelta(hours=hours)


def test_one_job_is_armed_per_game_at_that_games_own_tipoff(monkeypatch):
    """
    NOT one slate-wide check. Tip-offs are staggered across an evening, so a
    single re-check is early for the late games and late for the early ones.
    This module's own docstring says a cron expression cannot express a slate;
    it then shipped two fixed clocks because the schedule feed was out of
    reach. It is in reach now.
    """
    monkeypatch.delenv("PROPIQ_PRELOCK", raising=False)
    monkeypatch.delenv("PROPIQ_PRELOCK_LEAD_MINUTES", raising=False)
    sched = FakeScheduler()
    slate = _Slate([
        _Game("401", _tip(2)),
        _Game("402", _tip(5), home="MIA", away="DEN"),
    ])
    out = worker.schedule_prelock_jobs(sched, slate=slate, now=NOW)

    assert out["status"] == "OK"
    assert out["armed"] == 2
    assert len(sched.jobs) == 2
    # Each job's run_date is ITS OWN tip minus the lead, so the two differ.
    runs = sorted(j["run_date"] for j in sched.jobs)
    assert runs[0] == _tip(2) - timedelta(minutes=35)
    assert runs[1] == _tip(5) - timedelta(minutes=35)
    assert runs[0] != runs[1]


def test_the_lead_is_inside_the_thirty_to_forty_five_minute_window():
    assert 30 <= worker.DEFAULT_PRELOCK_LEAD_MINUTES <= 45


def test_the_lead_is_configurable(monkeypatch):
    monkeypatch.setenv("PROPIQ_PRELOCK_LEAD_MINUTES", "45")
    sched = FakeScheduler()
    out = worker.schedule_prelock_jobs(
        sched, slate=_Slate([_Game("401", _tip(3))]), now=NOW
    )
    assert out["lead_minutes"] == 45
    assert sched.jobs[0]["run_date"] == _tip(3) - timedelta(minutes=45)


def test_an_unusable_lead_falls_back_rather_than_refusing(monkeypatch):
    monkeypatch.setenv("PROPIQ_PRELOCK_LEAD_MINUTES", "9999")
    out = worker.schedule_prelock_jobs(
        FakeScheduler(), slate=_Slate([_Game("401", _tip(3))]), now=NOW
    )
    assert out["lead_minutes"] == worker.DEFAULT_PRELOCK_LEAD_MINUTES


def test_a_game_with_no_tipoff_is_skipped_with_its_reason(monkeypatch):
    """Guessing a tip time would fire the check at the wrong moment, which is
    worse than not firing it."""
    monkeypatch.delenv("PROPIQ_PRELOCK", raising=False)
    sched = FakeScheduler()
    out = worker.schedule_prelock_jobs(
        sched, slate=_Slate([_Game("401", None)]), now=NOW
    )
    assert out["armed"] == 0
    assert not sched.jobs
    assert "no tip-off time" in out["skipped"][0]["reason"]


def test_a_slot_already_past_is_skipped_not_fired_immediately(monkeypatch):
    """
    On a mid-day redeploy this is the normal outcome for the early games. A
    "pre-lock" card posted after lock is worse than no card.
    """
    monkeypatch.delenv("PROPIQ_PRELOCK", raising=False)
    sched = FakeScheduler()
    out = worker.schedule_prelock_jobs(
        sched,
        slate=_Slate([_Game("401", _tip(0.25)), _Game("402", _tip(4))]),
        now=NOW,
    )
    assert out["armed"] == 1
    assert len(sched.jobs) == 1
    assert "already passed" in out["skipped"][0]["reason"]


def test_a_naive_tipoff_is_treated_as_utc_rather_than_crashing(monkeypatch):
    monkeypatch.delenv("PROPIQ_PRELOCK", raising=False)
    # Deliberately naive: the point of the test is that a tz-less tip-off
    # from a feed is read as UTC rather than raising on the subtraction.
    naive = datetime(2026, 10, 5, 23, 0)  # noqa: DTZ001
    out = worker.schedule_prelock_jobs(
        FakeScheduler(), slate=_Slate([_Game("401", naive)]), now=NOW
    )
    assert out["armed"] == 1


def test_arming_twice_replaces_rather_than_duplicates(monkeypatch):
    """main() arms at boot and run_board arms again after the slate. Without a
    stable id that would post two cards per game."""
    monkeypatch.delenv("PROPIQ_PRELOCK", raising=False)
    sched = FakeScheduler()
    slate = _Slate([_Game("401", _tip(3))])
    worker.schedule_prelock_jobs(sched, slate=slate, now=NOW)
    worker.schedule_prelock_jobs(sched, slate=slate, now=NOW)
    ids = [j["id"] for j in sched.jobs]
    assert len(set(ids)) == 1
    assert all(j["replace_existing"] for j in sched.jobs)
    assert ids[0] == "prelock:2026-10-05:401"


def test_the_jobs_cannot_overlap_and_will_not_fire_long_after_tip(monkeypatch):
    monkeypatch.delenv("PROPIQ_PRELOCK", raising=False)
    sched = FakeScheduler()
    worker.schedule_prelock_jobs(
        sched, slate=_Slate([_Game("401", _tip(3))]), now=NOW
    )
    job = sched.jobs[0]
    assert job["max_instances"] == 1
    # Shorter than the lead, so a late fire cannot land after tip-off.
    assert job["misfire_grace_time"] < worker.DEFAULT_PRELOCK_LEAD_MINUTES * 60


def test_the_game_identity_reaches_the_job(monkeypatch):
    monkeypatch.delenv("PROPIQ_PRELOCK", raising=False)
    sched = FakeScheduler()
    worker.schedule_prelock_jobs(
        sched, slate=_Slate([_Game("401", _tip(3), home="BOS", away="LAL")]), now=NOW
    )
    kwargs = sched.jobs[0]["kwargs"]
    assert kwargs["event_id"] == "401"
    assert kwargs["home"] == "BOS"
    assert kwargs["away"] == "LAL"
    assert kwargs["tipoff_iso"].startswith("2026-10-05")
    assert sched.jobs[0]["func"] is worker.run_prelock


def test_a_denied_schedule_arms_nothing_and_says_so(monkeypatch, caplog):
    """The morning board still stands; it simply will not be re-checked."""
    import logging

    monkeypatch.delenv("PROPIQ_PRELOCK", raising=False)

    def _boom(*a, **k):
        raise RuntimeError("espn denied")

    monkeypatch.setattr("src.ingestion.espn_schedule.load_slate", _boom)
    sched = FakeScheduler()
    with caplog.at_level(logging.WARNING):
        out = worker.schedule_prelock_jobs(sched, now=NOW)
    assert out["status"] == "DATA_NOT_AVAILABLE"
    assert out["armed"] == 0
    assert not sched.jobs
    assert "espn denied" in caplog.text


def test_the_checks_can_be_turned_off(monkeypatch):
    monkeypatch.setenv("PROPIQ_PRELOCK", "false")
    sched = FakeScheduler()
    out = worker.schedule_prelock_jobs(
        sched, slate=_Slate([_Game("401", _tip(3))]), now=NOW
    )
    assert out["status"] == "SKIPPED"
    assert not sched.jobs


def test_the_planner_is_armed_at_boot_and_after_the_board(monkeypatch):
    """
    A container starting at 16:00 would otherwise leave every game tonight
    unchecked — exactly when a redeploy is likeliest and the morning board is
    most stale. And arming only at boot would miss a schedule change.
    """
    import inspect

    main_src = inspect.getsource(worker.main)
    assert "schedule_prelock_jobs(scheduler)" in main_src
    assert main_src.index("_SCHEDULER = scheduler") < main_src.index(
        "schedule_prelock_jobs(scheduler)"
    )
    board_src = inspect.getsource(worker.run_board)
    assert "schedule_prelock_jobs()" in board_src


def test_a_direct_call_with_no_scheduler_reports_rather_than_crashing(monkeypatch):
    monkeypatch.setattr(worker, "_SCHEDULER", None)
    out = worker.schedule_prelock_jobs(slate=_Slate([_Game("401", _tip(3))]), now=NOW)
    assert out["status"] == "DATA_NOT_AVAILABLE"


# --- run_prelock: the job itself ---------------------------------------

def _board(tmp_path, players=(("A. Player", "LAL", "BOS", "RECOMMENDED"),)):
    import pandas as pd

    frame = pd.DataFrame([
        {
            "slate_date": "2026-10-05", "game_date": "2026-10-05",
            "event_id": "0022500001", "player_id": f"200000{i}",
            "player_name": name, "player_team": team, "opponent": opp,
            "target_market": "PTS", "side": "over", "line": 24.5,
            "decision_status": status, "model_prob": 0.56,
        }
        for i, (name, team, opp, status) in enumerate(players)
    ])
    path = tmp_path / "decision_board.csv"
    frame.to_csv(path, index=False)
    return path


class _Report:
    """An espn_availability.AvailabilityReport, in the shape the filter reads."""

    def __init__(self, injuries, status="OK"):
        self.status = status
        self.injuries = injuries
        self.notes: list[str] = []


class _Injury:
    def __init__(self, player_name, status="OUT", detail=None):
        self.player_name = player_name
        self.status = status
        self.status_raw = status
        self.detail = detail


def test_a_player_ruled_out_after_the_board_is_withdrawn(monkeypatch, tmp_path):
    """
    THE WHOLE POINT. scratches.py was written for this and its docstring names
    the gap — "Nothing dropped a projection when a player was ruled out AFTER
    that projection was written" — and nothing ever called it on a schedule.
    """
    monkeypatch.setenv("PROPIQ_BOARD_CSV", str(_board(tmp_path)))
    monkeypatch.delenv("PROPIQ_PRELOCK", raising=False)
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    monkeypatch.setattr(
        "src.ingestion.espn_availability.fetch_injuries",
        lambda **k: _Report([_Injury("A. Player", "OUT", "Out (knee)")]),
    )
    out = worker.run_prelock(event_id="401", home="BOS", away="LAL")
    assert out["status"] == "OK"
    assert out["checked"] == 1
    assert out["withheld"] == 1


def test_a_healthy_slate_withdraws_nobody(monkeypatch, tmp_path):
    monkeypatch.setenv("PROPIQ_BOARD_CSV", str(_board(tmp_path)))
    monkeypatch.delenv("PROPIQ_PRELOCK", raising=False)
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    monkeypatch.setattr(
        "src.ingestion.espn_availability.fetch_injuries", lambda **k: _Report([])
    )
    out = worker.run_prelock(event_id="401", home="BOS", away="LAL")
    assert out["withheld"] == 0


def test_a_diacritic_difference_still_withdraws_the_right_player(
    monkeypatch, tmp_path
):
    """
    ESPN publishes "Nikola Jokić" and the panel carries "Nikola Jokic". Before
    the crosswalk those did not compare equal and the filter withheld NOBODY —
    a live safety bug, not a cosmetic one. The match is exact on a normalised
    name, never fuzzy.
    """
    path = _board(tmp_path, players=(("Nikola Jokic", "DEN", "LAL", "RECOMMENDED"),))
    monkeypatch.setenv("PROPIQ_BOARD_CSV", str(path))
    monkeypatch.delenv("PROPIQ_PRELOCK", raising=False)
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    monkeypatch.setattr(
        "src.ingestion.espn_availability.fetch_injuries",
        lambda **k: _Report([_Injury("Nikola Jokić", "OUT", "Out")]),
    )
    out = worker.run_prelock(event_id="401", home="LAL", away="DEN")
    assert out["withheld"] == 1


def test_only_this_games_rows_are_checked(monkeypatch, tmp_path):
    """The other games have their own jobs at their own tip-offs."""
    path = _board(tmp_path, players=(
        ("A. Player", "LAL", "BOS", "RECOMMENDED"),
        ("B. Player", "MIA", "DEN", "RECOMMENDED"),
    ))
    monkeypatch.setenv("PROPIQ_BOARD_CSV", str(path))
    monkeypatch.delenv("PROPIQ_PRELOCK", raising=False)
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    monkeypatch.setattr(
        "src.ingestion.espn_availability.fetch_injuries",
        lambda **k: _Report([_Injury("B. Player", "OUT")]),
    )
    out = worker.run_prelock(event_id="401", home="BOS", away="LAL")
    assert out["checked"] == 1       # only the LAL/BOS row
    assert out["withheld"] == 0      # B. Player is another game's problem


def test_an_already_abstained_row_needs_no_withdrawal(monkeypatch, tmp_path):
    path = _board(tmp_path, players=(("A. Player", "LAL", "BOS", "ABSTAIN"),))
    monkeypatch.setenv("PROPIQ_BOARD_CSV", str(path))
    monkeypatch.delenv("PROPIQ_PRELOCK", raising=False)
    out = worker.run_prelock(event_id="401", home="BOS", away="LAL")
    assert out["status"] == "SKIPPED"
    assert out["reason"] == "no recommended rows"


def test_an_unusable_feed_is_reported_and_clears_nobody(monkeypatch, tmp_path):
    """
    The rule the whole scratch filter turns on. A failed check must not read
    as a healthy slate, and must not be silent either — silence reads as
    "nothing changed".
    """
    monkeypatch.setenv("PROPIQ_BOARD_CSV", str(_board(tmp_path)))
    monkeypatch.delenv("PROPIQ_PRELOCK", raising=False)
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)

    def _denied(**k):
        raise RuntimeError("espn denied")

    monkeypatch.setattr(
        "src.ingestion.espn_availability.fetch_injuries", _denied
    )
    out = worker.run_prelock(event_id="401", home="BOS", away="LAL")
    assert out["status"] == "OK"
    assert out["withheld"] == 0      # nothing is withdrawn
    # ...and nothing is cleared: the embed for this case says the check FAILED.
    from src.notify.discord import build_prelock_correction_embed

    blob = repr(build_prelock_correction_embed(
        game="LAL @ BOS", tipoff_display=None, verified=False,
        reason="injury feed unreachable",
    ))
    assert "FAILED" in blob
    assert "not a healthy slate" in blob


def test_no_board_means_nothing_to_withdraw(monkeypatch, tmp_path):
    monkeypatch.setenv("PROPIQ_BOARD_CSV", str(tmp_path / "absent.csv"))
    monkeypatch.delenv("PROPIQ_PRELOCK", raising=False)
    out = worker.run_prelock(event_id="401")
    assert out["status"] == "SKIPPED"
    assert out["reason"] == "no board"


def test_the_job_can_be_turned_off(monkeypatch, tmp_path):
    monkeypatch.setenv("PROPIQ_BOARD_CSV", str(_board(tmp_path)))
    monkeypatch.setenv("PROPIQ_PRELOCK", "false")
    out = worker.run_prelock(event_id="401")
    assert out["status"] == "SKIPPED"
    assert out["reason"] == "pre-lock disabled"


def test_the_job_never_takes_the_worker_down(monkeypatch, tmp_path):
    monkeypatch.setenv("PROPIQ_BOARD_CSV", str(_board(tmp_path)))
    monkeypatch.delenv("PROPIQ_PRELOCK", raising=False)

    def _boom(*a, **k):
        raise RuntimeError("filter exploded")

    monkeypatch.setattr("src.pipeline.scratches.apply_scratch_filter", _boom)
    out = worker.run_prelock(event_id="401", home="BOS", away="LAL")
    assert out["status"] == "FAILED"


def test_it_refits_nothing_and_reprices_nothing(monkeypatch):
    """
    Refitting 35 minutes before tip would be absurd, and re-pricing needs a
    live odds feed this project cannot reach. The job must stay an
    availability check, so this pins what it does NOT touch.
    """
    import inspect

    source = inspect.getsource(worker.run_prelock)
    for forbidden in (
        "compare_models_on_panel", "build_slate_board", "build_feature_matrix",
        "resolve_market", "score_prob_over", "board_window",
    ):
        assert forbidden not in source, f"run_prelock reaches for {forbidden}"


def test_a_failure_arming_prelock_does_not_report_the_board_as_failed(
    monkeypatch, tmp_path, caplog
):
    """
    run_board's own broad except would catch an arming failure and return
    {"status": "FAILED"} for the BOARD — reporting a written board as failed
    because an unrelated step did. The board is the durable output.
    """
    import logging

    import pandas as pd

    import src.db.repository as repo
    import src.pipeline.slate_board as slate_board_module

    monkeypatch.delenv("PROPIQ_BOARD_TRAIN_END", raising=False)
    monkeypatch.delenv("PROPIQ_BOARD_VALIDATION_END", raising=False)

    class _Result:
        def as_dict(self):
            return {"written_rows": 3}

    monkeypatch.setattr(
        slate_board_module, "build_slate_board", lambda panel, **k: _Result()
    )
    monkeypatch.setattr(
        repo, "load_player_panel",
        lambda *a, **k: pd.DataFrame({"PLAYER_ID": ["1"], "GAME_DATE": ["2026-10-05"]}),
    )

    def _boom(*a, **k):
        raise RuntimeError("scheduler rejected the job")

    monkeypatch.setattr(worker, "schedule_prelock_jobs", _boom)

    with caplog.at_level(logging.ERROR):
        out = worker.run_board()
    assert out["status"] == "OK", "a written board must not be reported as failed"
    assert out["written_rows"] == 3
    assert "board above STANDS" in caplog.text
