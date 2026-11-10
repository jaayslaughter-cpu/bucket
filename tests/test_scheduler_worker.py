"""The deployed worker's schedule, concurrency limits and failure policy.

Nothing here starts a scheduler — a BlockingScheduler's start() does not return.
``build_scheduler`` is separated from ``main`` precisely so the job table can be
inspected, because the times, the grace periods and the concurrency limits are
the part worth asserting.
"""

from __future__ import annotations

from pathlib import Path

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


def board_panel(rows: int = 12, start: str = "2026-10-01"):
    """
    A panel ``build_feature_matrix`` accepts, for the board tests.

    ONE COPY, because three tests need it. `run_board` builds the feature
    matrix before handing anything to `build_slate_board` (the research line
    needs `{MARKET}_L10`, which only the builder adds), so a stub panel of
    `PLAYER_ID` and `GAME_DATE` alone now fails the board at the builder
    rather than travelling through it. These are the columns the builder
    reads, with enough rows for a ten-game window to exist.
    """
    import pandas as pd

    return pd.DataFrame({
        "PLAYER_ID": ["1"] * rows,
        "PLAYER_NAME": ["HARNESS ONE"] * rows,
        "GAME_ID": [f"espn:{i}" for i in range(rows)],
        "GAME_DATE": pd.date_range(start, periods=rows),
        "SEASON": ["2026-27"] * rows,
        "TEAM_ABBREVIATION": ["LAL"] * rows,
        "OPPONENT_ABBREVIATION": ["GSW"] * rows,
        "IS_HOME": [True] * rows,
        "PTS": [20.0 + i for i in range(rows)],
        "REB": [5.0] * rows, "AST": [4.0] * rows, "FG3M": [1.0] * rows,
        "STL": [1.0] * rows, "BLK": [0.5] * rows, "MIN": [30.0] * rows,
        "FGA": [15.0] * rows, "FTA": [4.0] * rows, "TOV": [2.0] * rows,
        "PF": [2.0] * rows,
    })



def _frozen_pacific_day(monkeypatch, day="2026-11-01"):
    """
    Pin the Pacific day for a test that goes through the real `run_dispatch`.

    FROZEN, NOT READ TWICE. `run_dispatch` reads the Pacific day itself to
    refuse a board describing another slate, so a fixture calling
    `pacific_calendar_date()` separately races it across Pacific midnight and
    fails in the staleness guard rather than on its own subject. Raised by
    review on #32 against two other files; the same defect was in these.

    Returns the frozen `date`; `str()` it for a CSV cell.
    """
    from datetime import date as _date

    from src.utils import timezones

    frozen = _date.fromisoformat(day)
    monkeypatch.setattr(timezones, "pacific_calendar_date", lambda *a, **k: frozen)
    return frozen


def board_projections(slate_date="2026-11-01", rows=3, with_probability=False):
    """
    Persisted projection rows, in `repository.load_projections`' shape.

    ONE COPY, because the board tests need it and it is the board's real
    input now. `prob_over` is null by default, which is production's state
    with no model artifact: the board still writes a row and says so per
    row, rather than writing nothing.
    """
    import pandas as pd

    return pd.DataFrame([
        {
            "run_id": "run_harness",
            "nba_player_id": f"espn:{i}",
            "player_name": f"HARNESS {i}",
            "nba_game_id": "espn:401909903",
            "game_date": pd.Timestamp(slate_date).date(),
            "market": ["PTS", "REB", "AST"][i % 3],
            "final_projection": 20.0 + i,
            "baseline_projection": 19.0 + i,
            "line": 19.5 + i,
            "prob_over": 0.55 if with_probability else None,
            "prob_under": 0.45 if with_probability else None,
            "prob_push": None,
            "market_status": (
                "READY_FOR_EVALUATION" if with_probability else "DATA_NOT_AVAILABLE"
            ),
            "market_status_reason": None,
            "ev_per_dollar": None,
            "model_version": None,
        }
        for i in range(rows)
    ])


def _jobs() -> dict[str, dict]:
    scheduler = FakeScheduler()
    worker.build_scheduler(scheduler)
    return {j["id"]: j for j in scheduler.jobs}


# --- the schedule --------------------------------------------------------

def test_both_halves_of_the_loop_are_scheduled():
    """
    Projecting without grading fills the ledger with PENDING rows for ever.

    ASSERTED AS "NEITHER IS MISSING", not as "there are exactly two". The
    equality held until `espn_ingest` was added on 2026-10-31 and then failed
    for a job that was supposed to be there -- a test that breaks on every
    addition teaches people to edit the test rather than read it. What must
    never happen is one half of the loop going absent, and that is what this
    now says.
    """
    jobs = _jobs()
    assert {"slate", "settlement"} <= set(jobs)
    assert jobs["slate"]["func"] is worker.run_slate
    assert jobs["settlement"]["func"] is worker.run_settlement


def test_the_panel_is_filled_before_either_half_reads_it():
    """
    The third job, and its whole point is WHEN it runs. Settlement grades
    against the panel and the slate builds features from it, so an ingest
    scheduled after either one fills the panel for tomorrow while today reads
    yesterday's.
    """
    jobs = _jobs()
    assert "espn_ingest" in jobs
    assert jobs["espn_ingest"]["func"] is worker.run_espn_ingest
    # (hour, minute), not hour alone: the minutes are independently
    # configurable, so an hour-only comparison rejects a perfectly good
    # 02:00 / 02:30 pair and accepts nothing it should. Flagged in review
    # 2026-10-31.
    def at(job_id: str) -> tuple[int, int]:
        return (jobs[job_id]["hour"], jobs[job_id]["minute"])

    assert at("espn_ingest") < at("settlement"), (
        "the ingest must finish before the jobs that read the panel"
    )
    assert at("settlement") < at("slate")


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


def test_a_preflight_exit_is_caught_alerted_and_returned(monkeypatch):
    """
    FOUND IN PRODUCTION, 2026-10-11. `main.preflight` ends a run it cannot do
    with `raise SystemExit(1)` (main.py:264) — and `SystemExit` derives from
    BaseException, so the `except Exception` above it never saw it. On Railway
    the 09:00 PT slate hit an unset DATABASE_URL and produced a raw APScheduler
    traceback with NO Discord alert, while an ordinary exception on the very
    same line alerted correctly. The one failure a misconfigured deploy is
    guaranteed to hit was the one failure that told nobody.

    `main.main` is a CLI entry point called in-process here, so its SystemExit
    is "the CLI would have exited with this code" — a return value for this
    caller, not a request that the worker exit.
    """
    import main

    alerts = []
    monkeypatch.setattr(
        worker, "notify_failure",
        lambda stage, detail, **kw: alerts.append((stage, detail)) or {},
    )

    def preflight_refuses(argv=None):
        raise SystemExit(1)

    monkeypatch.setattr(main, "main", preflight_refuses)

    assert worker.run_slate([]) == 1
    assert [stage for stage, _ in alerts] == ["slate"]
    assert "exit" in alerts[0][1].lower()


def test_an_argparse_exit_is_also_caught(monkeypatch):
    """argparse exits 2 on bad argv. Same boundary, same reasoning."""
    import main

    monkeypatch.setattr(worker, "notify_failure", lambda *a, **k: {})
    monkeypatch.setattr(
        main, "main", lambda argv=None: (_ for _ in ()).throw(SystemExit(2))
    )
    assert worker.run_slate([]) == 2


def test_a_zero_system_exit_is_not_reported_as_a_failure(monkeypatch):
    """
    `SystemExit(None)` and `SystemExit(0)` are a CLI succeeding, which must not
    alert — or the fix above turns every clean run into an incident.
    """
    import main

    alerts = []
    monkeypatch.setattr(
        worker, "notify_failure", lambda *a, **k: alerts.append(a) or {},
    )
    monkeypatch.setattr(worker, "run_board", lambda: {})
    monkeypatch.setattr(worker, "run_dispatch", lambda: {})
    monkeypatch.setattr(
        main, "main", lambda argv=None: (_ for _ in ()).throw(SystemExit(None))
    )
    assert worker.run_slate([]) == 0
    assert alerts == []


def test_a_nonzero_return_is_alerted_not_only_logged(monkeypatch):
    """
    FOUND IN PRODUCTION, 2026-10-22, and the SECOND TIME this exact gap has
    been found in this function. `test_a_preflight_exit_is_caught_alerted_and
    _returned` above records the first: SystemExit slipped past `except
    Exception` and the one failure a misconfigured deploy is guaranteed to hit
    told nobody. This is the same sentence with a different branch.

    `main.main` wraps its ENTIRE body in `except Exception`, logs "Pipeline
    FAILED", records the run and RETURNS 1 (main.py:1842-1851). So the
    ordinary internal error never reaches `run_slate` as an exception at all
    — it arrives as a return code, and that branch only called
    `logger.warning`. Every line in Railway's log view is already red because
    Python logging writes to stderr, so a silent real failure is
    indistinguishable from an INFO line by eye.

    Measured on the live worker: the 2026-10-20 and 2026-10-21 slates both
    failed this way (`MarketFramesUnavailable`, nothing ever ingested). The
    operator received a "0 graded" results card at 03:30 and NOTHING at 09:00
    on the first slate of the season.
    """
    import main

    alerts = []
    monkeypatch.setattr(
        worker, "notify_failure",
        lambda stage, detail, **kw: alerts.append((stage, detail)) or {},
    )
    monkeypatch.setattr(worker, "run_board", lambda: {})
    monkeypatch.setattr(worker, "run_dispatch", lambda: {})
    # Returns, does not raise -- exactly what main.main does on any internal
    # failure, which is the whole point of this test.
    monkeypatch.setattr(main, "main", lambda argv=None: 1)

    assert worker.run_slate([]) == 1
    assert [stage for stage, _ in alerts] == ["slate"], (
        "a slate that failed by RETURNING non-zero sent no alert"
    )
    # The alert must say where the cause is, since the code alone is not one.
    detail = alerts[0][1]
    assert "pipeline_runs" in detail and "Pipeline FAILED" in detail
    assert "no card was sent" in detail


def test_a_successful_slate_still_sends_no_failure_alert(monkeypatch):
    """
    The guard against the obvious over-correction: alerting on every return
    would turn each clean run into an incident, which is how alerting gets
    muted and then the real failure is missed anyway.
    """
    import main

    alerts = []
    monkeypatch.setattr(
        worker, "notify_failure", lambda *a, **k: alerts.append(a) or {},
    )
    boarded = []
    monkeypatch.setattr(worker, "run_board", lambda: boarded.append("board") or {})
    monkeypatch.setattr(worker, "run_dispatch", lambda: boarded.append("card") or {})
    monkeypatch.setattr(main, "main", lambda argv=None: 0)

    assert worker.run_slate([]) == 0
    assert alerts == []
    assert boarded == ["board", "card"], "a clean slate must still build and send"


def test_a_failing_slate_alerts_exactly_once(monkeypatch):
    """
    NOT TWICE. The SystemExit branch alerts and returns early; this checks it
    does not then fall through to the new one. A duplicate alert per failure
    trains the reader to ignore the channel.
    """
    import main

    alerts = []
    monkeypatch.setattr(
        worker, "notify_failure",
        lambda stage, detail, **kw: alerts.append((stage, detail)) or {},
    )
    monkeypatch.setattr(worker, "run_board", lambda: {})
    monkeypatch.setattr(worker, "run_dispatch", lambda: {})

    for failure in (
        lambda argv=None: (_ for _ in ()).throw(SystemExit(1)),
        lambda argv=None: (_ for _ in ()).throw(RuntimeError("boom")),
        lambda argv=None: 1,
    ):
        alerts.clear()
        monkeypatch.setattr(main, "main", failure)
        assert worker.run_slate([]) == 1
        assert len(alerts) == 1, f"{len(alerts)} alerts for one failure"


def test_a_keyboard_interrupt_still_propagates(monkeypatch):
    """
    The fix catches SystemExit, NOT BaseException. A Ctrl-C or a SIGINT during
    a slate is a shutdown request, and swallowing it would make the worker
    unkillable mid-run — a worse defect than the one being fixed.
    """
    import main

    monkeypatch.setattr(
        main, "main", lambda argv=None: (_ for _ in ()).throw(KeyboardInterrupt())
    )
    with pytest.raises(KeyboardInterrupt):
        worker.run_slate([])


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
    # The frozen Pacific day: the subject here is the calibration gate, and
    # dispatch refuses a board describing another day before it ever reaches
    # that gate (see
    # `test_a_board_for_another_day_is_refused_rather_than_dispatched`).
    slate = _frozen_pacific_day(monkeypatch)
    pd.DataFrame([{
        "slate_date": str(slate),
        "decision_status": "RECOMMENDED",
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


def test_a_board_for_another_day_is_refused_rather_than_dispatched(
    monkeypatch, tmp_path
):
    """
    The 9:03 failure on 2026-11-01 left no board on disk, which is why nothing
    went out. The worse shape is the one where a board IS on disk and it
    describes yesterday: dispatch reads the CSV without consulting the status
    `run_board` returned, so a crash between write and dispatch, or an
    operator's hand-run CLI, would publish last night's lines under tonight's
    heading. The row carries its own `slate_date`; dispatch compares it to the
    Pacific calendar day and names the day it actually describes.
    """
    import pandas as pd

    sent = {}

    def fake_send(embeds, config=None, **kw):
        sent["embeds"] = list(embeds)
        return type("R", (), {"status": "OK"})()

    today = _frozen_pacific_day(monkeypatch)
    yesterday = str(today - timedelta(days=1))
    board = tmp_path / "board.csv"
    pd.DataFrame([{
        "slate_date": yesterday, "decision_status": "RECOMMENDED",
        "decision_basis": "model_lean", "player_name": "DEMO_STALE",
        "target_market": "PTS", "side": "over", "line": 24.5,
        "model_prob": 0.71, "book_ev": None, "american_odds": None,
        "book_source": None,
    }]).to_csv(board, index=False)

    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/1/x")
    monkeypatch.setenv(worker.ENV_DISPATCH, "true")
    monkeypatch.setenv(worker.ENV_BOARD_CSV, str(board))

    import src.notify.discord as discord_mod

    monkeypatch.setattr(discord_mod, "send_embeds", fake_send)

    out = worker.run_dispatch()

    # An abstention went out, and it names the day the file describes rather
    # than reading as a broken pipeline.
    assert sent["embeds"], "dispatch sent nothing at all"
    blob = str(sent["embeds"][0])
    assert yesterday in blob, f"the abstention does not name {yesterday}: {blob}"
    assert str(today) in blob
    # And the stale row never reached a person.
    assert "DEMO_STALE" not in blob
    assert yesterday in out["reason"]
    # AND THE GATE'S VERDICT TRAVELS. A caller reading out["publication"]
    # cannot know which branch produced its dict, so every return carries one:
    # the calibration/freshness verdict is true of this slate whether or not a
    # board went out. Raised by review on #32.
    assert "publication" in out, (
        f"the stale-board return dropped the publication verdict: {sorted(out)}"
    )
    assert "PUBLICATION_STATUS" in out["publication"]


def test_a_board_for_today_is_not_refused_as_stale(monkeypatch, tmp_path):
    """
    The other head of the same guard: the comparison is to TODAY, not to a
    constant, so a board written for tonight passes through to the rest of
    dispatch. Without this, refusing every board would satisfy the test above.
    """
    import pandas as pd

    sent = {}

    def fake_send(embeds, config=None, **kw):
        sent["embeds"] = list(embeds)
        return type("R", (), {"status": "OK"})()

    today = _frozen_pacific_day(monkeypatch)
    board = tmp_path / "board.csv"
    pd.DataFrame([{
        "slate_date": str(today),
        "decision_status": "RECOMMENDED",
        "decision_basis": "model_lean", "player_name": "DEMO_TODAY",
        "target_market": "PTS", "side": "over", "line": 24.5,
        "model_prob": 0.71, "book_ev": None, "american_odds": None,
        "book_source": None,
    }]).to_csv(board, index=False)

    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/1/x")
    monkeypatch.setenv(worker.ENV_DISPATCH, "true")
    monkeypatch.setenv(worker.ENV_BOARD_CSV, str(board))

    import src.notify.discord as discord_mod

    monkeypatch.setattr(discord_mod, "send_embeds", fake_send)

    out = worker.run_dispatch()

    blob = str(sent.get("embeds"))
    assert "not " + str(today) not in blob
    assert "describes" not in str(out.get("reason", ""))


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
    assert not list(tmp_path.glob(".propiq_write_probe*")), "the probe file was left behind"


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
        _assert_refused_write_was_diagnosed(out, caplog.text)
    finally:
        locked.chmod(0o700)


def _assert_refused_write_was_diagnosed(out: dict, text: str) -> None:
    """
    What a refused write must produce, WITHOUT pinning the prose.

    THIS HELPER EXISTS BECAUSE THE TEST ABOVE ONLY RUNS IN CI. It is skipped
    under root -- root ignores mode bits -- so every local run skips it, and
    twice now a rewording of the shared remedy passed here and failed there:
    once on "owned by root" and once on "owned by another uid". A CI-only test
    that asserts a sentence fragment is a tripwire for whoever edits the
    sentence, so this asserts the FAULT CODE and the ACTIONS offered, and the
    root-independent test below exercises the same branch locally.
    """
    assert out["writable"] is False
    assert out["fault"] == "not_writable", out["fault"]
    assert "NOT WRITABLE" in text
    assert "WRITE WAS REFUSED" in text
    # Both candidate causes and both remedies, since the probe establishes
    # neither ownership nor mode.
    assert "chown" in text.lower()
    assert "mode" in text.lower()
    assert "still writes to Postgres" in text, (
        "the operator should know what still works"
    )


def test_a_refused_write_on_the_report_directory_is_diagnosed(
    monkeypatch, tmp_path, caplog
):
    """
    THE SAME BRANCH AS THE ROOT-SKIPPED TEST ABOVE, root-independent: the write
    is denied directly rather than by mode bits, so this runs under any uid and
    the assertions above are exercised locally instead of only in CI.
    """
    import logging

    import src.utils.volume as volume

    root = tmp_path / "vol"
    report = tmp_path / "elsewhere"
    root.mkdir()
    report.mkdir()
    monkeypatch.setattr(volume, "resolve_state_root", lambda: (root, "test"))
    monkeypatch.setenv(worker.ENV_CALIBRATION_REPORT, str(report / "calibration.json"))

    real_write = type(tmp_path).write_text

    def deny_report_only(self, *a, **k):
        if self.name.startswith(".propiq_write_probe") and self.parent == report:
            raise PermissionError(13, "Permission denied", str(self))
        return real_write(self, *a, **k)

    monkeypatch.setattr(type(tmp_path), "write_text", deny_report_only)
    with caplog.at_level(logging.ERROR):
        out = worker.check_state_dir()

    _assert_refused_write_was_diagnosed(out, caplog.text)
    assert "CALIBRATION REPORT DIRECTORY" in caplog.text


def test_the_state_root_variant_still_names_the_root_owned_volume(
    monkeypatch, tmp_path, caplog
):
    """
    THE OTHER HALF, and ROOT-INDEPENDENT so it runs everywhere.

    The test above is skipped under root -- mode bits do not apply -- which
    means a local full-suite run never exercised the ownership branch at all,
    and a rewording of it passed here and failed in CI. This one denies the
    write directly, so the same behaviour is covered whatever uid runs it.
    """
    import logging

    import src.utils.volume as volume

    root = tmp_path / "vol"
    root.mkdir()
    monkeypatch.setattr(volume, "resolve_state_root", lambda: (root, "test"))
    monkeypatch.setenv(worker.ENV_CALIBRATION_REPORT, str(root / "calibration.json"))

    real_write = type(tmp_path).write_text

    def deny(self, *a, **k):
        if self.name.startswith(".propiq_write_probe"):
            raise PermissionError(13, "Permission denied", str(self))
        return real_write(self, *a, **k)

    monkeypatch.setattr(type(tmp_path), "write_text", deny)
    with caplog.at_level(logging.ERROR):
        out = worker.check_state_dir()

    assert out["writable"] is False
    assert out["fault"] == "not_writable"
    text = caplog.text
    assert "STATE ROOT" in text
    assert "root-owned" in text, (
        "the state root's ownership branch no longer names the root-owned "
        "volume, which is the documented Railway failure"
    )
    # The remedy, in the wording the shared helper now uses: it offers chown
    # AND the mode, because the probe establishes neither.
    assert "chown it to uid" in text
    assert "widen its mode" in text
    assert "docs/deploy_railway.md" in text


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
    # THE CAUSE, AND THE RIGHT ONE. This used to require "owned by root" for a
    # path that DOES NOT EXIST -- the conflation `utils.volume.classify_
    # unwritable` was added to end, asserted. /proc/nonexistent cannot be
    # chowned because there is nothing there, so the probe must name the fault
    # it found rather than the hypothesis that fits the other one.
    assert out["fault"] == "missing_mount", out["fault"]
    assert "DOES NOT EXIST" in caplog.text
    assert "owned by root" not in caplog.text, (
        "an absent path is still diagnosed as an ownership problem"
    )
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

    So this captures what the board path is actually handed. It reads the
    PROJECTIONS path now: a live slate no longer goes through
    `compare_models_on_panel`, because that function scores a window of
    games with known outcomes and tonight's have none. The window still
    decides which day is asked for, which is what this asserts.
    """
    from datetime import date, timedelta

    import src.db.repository as repo
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

    def _fake_build(projections, **kwargs):
        seen.update(kwargs)
        seen["rows"] = len(projections)
        return _Result()

    monkeypatch.setattr(
        slate_board_module, "build_board_from_projections", _fake_build,
    )
    monkeypatch.setattr(
        repo, "load_projections", lambda slate, **k: board_projections(slate),
    )

    # The comparison path must not be reached for a live slate at all.
    def _must_not_compare(*a, **k):
        raise AssertionError("a live board went through compare_models_on_panel")

    monkeypatch.setattr(slate_board_module, "build_slate_board", _must_not_compare)

    out = worker.run_board()
    assert out["status"] == "OK"
    assert seen["slate_date"] == str(today), (
        f"the board asked for the wrong day: {seen.get('slate_date')}"
    )
    assert seen["slate_date"] != "2025-02-15"
    assert out["source"] == "projections"
    # And the window travels with the result, so a log line says WHICH days
    # were scored rather than only how many rows came back.
    assert out["window"]["validation_end"] == str(today)
    assert out["window"]["train_end"] == str(today - timedelta(days=1))
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

    import src.db.repository as repo
    import src.pipeline.slate_board as slate_board_module

    monkeypatch.delenv("PROPIQ_BOARD_TRAIN_END", raising=False)
    monkeypatch.delenv("PROPIQ_BOARD_VALIDATION_END", raising=False)

    class _Result:
        def as_dict(self):
            return {"written_rows": 3}

    monkeypatch.setattr(
        slate_board_module, "build_board_from_projections",
        lambda projections, **k: _Result(),
    )
    monkeypatch.setattr(
        repo, "load_projections", lambda slate, **k: board_projections(slate),
    )

    def _boom(*a, **k):
        raise RuntimeError("scheduler rejected the job")

    monkeypatch.setattr(worker, "schedule_prelock_jobs", _boom)

    with caplog.at_level(logging.ERROR):
        out = worker.run_board()
    assert out["status"] == "OK", "a written board must not be reported as failed"
    assert out["written_rows"] == 3
    assert "board above STANDS" in caplog.text


# --- the publication gate's evidence lives on the durable volume ------------


def test_the_calibration_report_defaults_under_the_durable_state_root(monkeypatch):
    """
    THE DEFAULT WAS EPHEMERAL, AND THE SYMPTOM BLAMED THE WRONG THING.
    `outputs/calibration.json` is relative to the working directory, so on
    Railway the settlement job wrote the report at 03:30 PT, a redeploy before
    09:00 PT took it with it, and the slate then withheld every card for "no
    calibration evidence" — true, and not the real reason, so an operator
    debugs the gate instead of the volume.

    `railway_healthcheck` warned about it, but a warning somebody has to
    notice and act on is worse than a default that is simply right.
    """
    from src.utils.volume import resolve_state_root

    monkeypatch.delenv(worker.ENV_CALIBRATION_REPORT, raising=False)
    root = resolve_state_root()[0].resolve()
    assert worker.calibration_report_path().resolve().is_relative_to(root)


def test_an_operator_who_names_a_path_still_wins(monkeypatch):
    """The default moved; it did not become a policy."""
    monkeypatch.setenv(worker.ENV_CALIBRATION_REPORT, "/tmp/operator/choice.json")
    assert worker.calibration_report_path() == Path("/tmp/operator/choice.json")
    # An explicit argument outranks even the variable.
    assert worker.calibration_report_path("/tmp/explicit.json") == Path(
        "/tmp/explicit.json"
    )


def test_the_healthcheck_certifies_the_file_settlement_actually_writes():
    """
    ONE RESOLVER, OR THE HEALTHCHECK IS WORSE THAN NOTHING. Four readers each
    had their own `env or default`: the boot probe, the dispatcher, the report
    writer and `railway_healthcheck`. Copies agree until one is changed, and a
    healthcheck that certifies a different path from the one settlement writes
    certifies nothing.
    """
    import ast

    source = (Path(__file__).resolve().parents[1]
              / "scripts" / "railway_healthcheck.py").read_text(encoding="utf-8")

    # THE CALL, NOT THE SUBSTRING. Grepping the whole file for
    # "calibration_report_path()" passed on a mention in a comment or a
    # docstring, so the check survived the implementation regressing — which
    # is the same assertion-that-cannot-fail shape this suite keeps finding.
    tree = ast.parse(source)
    function = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.FunctionDef) and n.name == "_check_calibration_report"),
        None,
    )
    assert function is not None, "_check_calibration_report was renamed or removed"
    assert any(
        isinstance(n, ast.Call) and getattr(n.func, "id", None) == "calibration_report_path"
        for n in ast.walk(function)
    ), "the healthcheck stopped CALLING the shared resolver"
    assert "DEFAULT_CALIBRATION_REPORT" not in source, (
        "the healthcheck grew its own copy of the default back"
    )


# --- the board job must hand over a FEATURE MATRIX, not the raw panel ----


def test_the_backtest_board_builds_features_before_handing_the_panel_over(
    monkeypatch, tmp_path,
):
    """
    THE BUG AN EMPTY DATABASE HID FOR THE WHOLE PROJECT'S LIFE.

    MOVED TO THE BACKTEST WINDOW, where the assertion still holds and the
    path still exists. A live board no longer goes through
    `compare_models_on_panel` at all -- it reads `main.py`'s persisted
    projections, which were scored from a matrix main already built. The
    comparison path remains for an override into the past, and it still
    needs the feature build in front of it for exactly the reason below.

    `run_board` loaded `load_player_panel()` and passed it straight to
    `build_slate_board`, which hands it to `compare_models_on_panel` ->
    `attach_research_over_labels`, and that needs `{MARKET}_L10` to build a
    research line. The raw panel carries PTS; only `build_feature_matrix`
    adds PTS_L10.

    It never surfaced because the guard above it -- "no player panel
    available" -- returned early on every single run while the panel was
    empty. The first night real rows existed (2026-10-31, 1,830 of them from
    the ESPN ingest), the guard passed and the job raised
    `DATA_NOT_AVAILABLE: PTS_L10 missing`.

    `scripts/nba_model_cli.py` gets this right -- its `_load_real_or_demo`
    builds the matrix -- and its comment says the two must not drift because
    "the drifting one would be the unattended one". They shared
    `build_slate_board` and not the feature build in front of it, and the
    unattended one is exactly the one that drifted.
    """
    import src.db.repository as repo
    import src.pipeline.slate_board as slate_board_module

    monkeypatch.setattr(
        repo, "load_player_panel", lambda *a, **k: board_panel(),
    )
    # A WINDOW IN THE PAST, so the comparison path is the one taken.
    monkeypatch.setenv("PROPIQ_BOARD_TRAIN_END", "2026-10-25")
    monkeypatch.setenv("PROPIQ_BOARD_VALIDATION_END", "2026-10-28")

    handed: dict = {}

    class _Result:
        written_rows = 0
        out = None
        slate_date = "2026-10-31"
        summary: dict = {}

        def as_dict(self):
            return {}

    def _capture(frame, **kwargs):
        handed["cols"] = list(frame.columns)
        return _Result()

    monkeypatch.setattr(slate_board_module, "build_slate_board", _capture)
    monkeypatch.setenv(worker.ENV_BOARD_CSV, str(tmp_path / "board.csv"))
    # STUBBED, so this test asserts ONE thing. Left real, it reaches
    # `main.resolve_market_frames` and its result depends on whether the
    # machine happens to have the licensed workbook or a reachable database
    # -- a test that reads a different matrix on CI than on a developer's
    # laptop. The helper's own contract is asserted separately, in
    # `test_board_market_frames_calls_the_orchestrator_that_actually_exists`.
    # Raised by review on #31.
    monkeypatch.setattr(
        worker, "_board_market_frames",
        lambda: {"team_games": None, "market_lines": None},
    )

    worker.run_board()

    assert handed, "build_slate_board was never called"
    # The exact column `attach_research_over_labels` raised on.
    assert "PTS_L10" in handed["cols"], (
        "run_board handed over the raw panel; the research line needs the "
        f"rolling features. Got: {sorted(handed['cols'])[:12]}"
    )
    for col in ("REB_L10", "AST_L10"):
        assert col in handed["cols"], f"{col} missing from the board's matrix"


def test_board_market_frames_calls_the_orchestrator_that_actually_exists(
    monkeypatch, tmp_path,
):
    """
    A HELPER THAT ALWAYS TAKES ITS EXCEPT BRANCH IS NOT A FALLBACK.

    `_board_market_frames` wraps `main.resolve_market_frames` in a broad
    `except Exception` so a board can still be built from a narrower matrix
    when there are no team-game or market-line rows. That handler also
    swallows an `AttributeError` for a name on `main` that does not exist --
    the first version called `main.bigdataball_path()`, which main.py has
    never defined. Every board would have logged "no team-game or
    market-line frames" and built without Elo, `DEF_*`, `MKT_*` or
    `BLOWOUT_*`, and the warning would have read like an empty database.

    So this asserts the happy path: the orchestrator is really called, with
    the workbook path the orchestrator itself names, and its frames come
    back to the caller rather than the empty pair.
    """
    import pandas as pd

    import main as orchestrator

    team_games = pd.DataFrame({"game_date": ["2026-10-30"], "team": ["LAL"]})
    market_lines = pd.DataFrame({"game_date": ["2026-10-30"], "total": [223.5]})
    calls: list[dict] = []

    def _frames(xlsx_path, **kwargs):
        calls.append({"path": xlsx_path, **kwargs})
        return {
            "team_games": team_games,
            "market_lines": market_lines,
            "source": "database",
            "note": "harness",
        }

    monkeypatch.setattr(orchestrator, "resolve_market_frames", _frames)

    workbook = tmp_path / "harness_team_stats.xlsx"
    monkeypatch.setenv(orchestrator.ENV_BIGDATABALL, str(workbook))

    got = worker._board_market_frames()

    assert calls, (
        "the orchestrator was never called -- _board_market_frames took its "
        "except branch, which means the name it asks main for does not exist"
    )
    assert str(calls[0]["path"]) == str(workbook), (
        "the board read a different workbook path than $"
        f"{orchestrator.ENV_BIGDATABALL} names: {calls[0]['path']!r}"
    )
    # The board reads. The slate job is what ingests.
    assert calls[0].get("persist") is False, "the board must not persist"
    assert got["team_games"] is team_games
    assert got["market_lines"] is market_lines


# --- the history backfill: a bounded, resumable walk backwards ------------


def test_the_backfill_job_is_on_the_schedule_after_settlement():
    """
    A job nothing registers cannot deepen a panel. And the TIME matters: the
    backfill is the longest-running job here, and firing it inside the 09:00
    slate's window would have it writing `player_game_logs` while the slate
    reads them.
    """
    jobs = _jobs()
    assert "espn_history_backfill" in jobs, (
        "the history backfill is not registered; the panel can only ever "
        "hold the nightly ingest's 30-day seed"
    )
    job = jobs["espn_history_backfill"]
    assert job["func"] is worker.run_history_backfill
    assert job["max_instances"] == 1, "two concurrent backfills would interleave writes"
    assert (job["hour"], job["minute"]) == (4, 30), (
        "the backfill must sit after settlement (03:30) and clear of the "
        f"slate (09:00); it is at {job['hour']}:{job['minute']:02d}"
    )
    # Thirty minutes, not the six hours the other ingests get: a chunk of
    # seasons-old box scores is no more urgent late, and a late fire would
    # land in the slate's window.
    assert job["misfire_grace_time"] == 30 * 60


def test_the_backfill_is_off_unless_a_season_is_named(monkeypatch):
    """
    OFF BY DEFAULT IS THE DESIGN, not timidity. ESPN's season calendar
    includes preseason and playoff dates, which the per-event guard refuses,
    so those dates are permanently absent from the panel and read as "not
    yet ingested" on every run. An unconditional nightly backfill would
    re-ask them for ever and never finish.
    """
    monkeypatch.delenv(worker.ENV_ESPN_HISTORY_SEASONS, raising=False)
    out = worker.run_history_backfill()
    assert out["status"] == "SKIPPED"
    assert worker.ENV_ESPN_HISTORY_SEASONS in out["reason"]


def test_the_backfill_obeys_the_same_switch_as_the_nightly_ingest(monkeypatch):
    """One ESPN path, one off switch. An operator who disabled the ESPN
    ingest disabled the ESPN ingest; a backfill that kept fetching would be
    the one job that ignored them."""
    monkeypatch.setenv(worker.ENV_ESPN_HISTORY_SEASONS, "2026")
    monkeypatch.setenv(worker.ENV_ESPN_INGEST, "false")
    out = worker.run_history_backfill()
    assert out["status"] == "SKIPPED"
    assert "disabled" in out["reason"]


def test_history_seasons_parses_years_and_refuses_nonsense(monkeypatch):
    """A typo must not reach `fetch_season_calendar`, where it comes back as
    an empty calendar and reads as "ESPN changed shape"."""
    monkeypatch.setenv(worker.ENV_ESPN_HISTORY_SEASONS, "2026, 2025 2026,, 26, twenty")
    assert worker._history_seasons() == [2025, 2026], "oldest first, deduplicated"

    monkeypatch.setenv(worker.ENV_ESPN_HISTORY_SEASONS, "20226")
    assert worker._history_seasons() == [], "an implausible year is not a season"

    monkeypatch.setenv(worker.ENV_ESPN_HISTORY_SEASONS, "   ")
    assert worker._history_seasons() == []


def test_the_plan_takes_the_oldest_missing_dates_and_reports_the_rest():
    """
    OLDEST FIRST, and this is the assertion that pins it.

    The rolling features need a player's history to sit BEFORE the games
    being predicted. Filling 2024-10 before 2025-01 means every intermediate
    state of the panel is one whose newest rows have real windows behind
    them. Newest-first would spend the whole backfill with the null windows
    on the rows that get scored.
    """
    from datetime import date as _d

    calendar = {
        2025: [_d(2024, 10, 22), _d(2024, 10, 23), _d(2024, 10, 24)],
        2026: [_d(2025, 10, 21), _d(2025, 10, 22)],
    }
    windows = {
        2025: (_d(2024, 10, 22), _d(2025, 4, 14)),
        2026: (_d(2025, 10, 21), _d(2026, 4, 13)),
    }
    held = {"2024-10-23": {"espn:1"}}  # a night the panel already holds

    plan = worker.plan_history_dates(
        [2026, 2025], held, chunk=2,
        calendar_for=calendar.__getitem__,
        window_for=windows.__getitem__,
        not_after=_d(2026, 10, 31),
    )
    assert plan["dates"] == [_d(2024, 10, 22), _d(2024, 10, 24)], (
        "dates with NO rows come first, oldest first; the night with rows "
        f"waits behind them. Got {plan['dates']}"
    )
    assert plan["candidates"] == 5
    assert plan["absent"] == 4
    assert plan["present"] == 1
    assert any("2025: regular season 2024-10-22 to 2025-04-14" in n
               for n in plan["seasons"])


def test_a_date_with_rows_is_re_examined_rather_than_skipped_for_ever():
    """
    THE HOLE A DATE-LEVEL RESUME MARKER LEAVES, raised by review on #31.

    `fetch_player_rows` records a failed event and keeps the others -- that
    isolation is deliberate, so one 404 does not cost the night. The
    consequence is that a night which lost one game to a 404 IS WRITTEN and
    is incomplete. Two earlier versions treated "the panel has a row for
    this date" as "this date is done", so that night was skipped for ever:
    a silent hole in a rolling window, which is the one thing the panel
    must not have.

    Dates with rows are therefore re-examined, after the ones with none.
    The caller passes the game ids it already holds, so a date that turns
    out to be complete costs one scoreboard call and no summaries.
    """
    from datetime import date as _d

    calendar = {2026: [_d(2025, 10, 21), _d(2025, 10, 22), _d(2025, 10, 23)]}
    windows = {2026: (_d(2025, 10, 21), _d(2026, 4, 13))}
    held = {
        "2025-10-21": {"espn:1", "espn:2"},
        "2025-10-22": {"espn:3"},
        "2025-10-23": {"espn:4"},
    }

    plan = worker.plan_history_dates(
        [2026], held, chunk=10,
        calendar_for=calendar.__getitem__,
        window_for=windows.__getitem__,
        not_after=_d(2026, 10, 31),
    )
    assert plan["dates"] == calendar[2026], (
        "every date with rows must still be offered for re-examination; a "
        f"partial night is otherwise never retried. Got {plan['dates']}"
    )
    assert plan["absent"] == 0
    assert plan["present"] == 3


def test_the_plan_excludes_preseason_playoffs_and_games_not_yet_played():
    """
    THE REASON THE FIRST VERSION OF THIS JOB COULD NEVER FINISH.

    `fetch_season_calendar` lists every date with any NBA game in a season,
    so preseason in early October and the playoffs through June are in it.
    The per-event guard refuses both, so those dates never get panel rows --
    and a work list built from the calendar alone reported them missing on
    EVERY run. `remaining` never reached 0, the COMPLETE branch was
    unreachable, and the job's own docstring told operators to wait for it.
    Raised by review on #31.

    Dates not yet played are the same defect for the current season: a game
    tonight is not final and tomorrow's has not happened, so both would stay
    permanently missing.
    """
    from datetime import date as _d

    calendar = {
        2026: [
            _d(2025, 10, 6),    # preseason — before the regular season opens
            _d(2025, 10, 21),   # opening night
            _d(2025, 12, 25),
            _d(2026, 4, 20),    # playoffs — after the regular season ends
        ],
        # THE CURRENT SEASON, which is where "not yet played" lives. The
        # first version of this test put its unplayed date in the 2026
        # season, where the regular-season window already excluded it — so
        # dropping the played filter altogether left the test green. A date
        # has to be IN SEASON and unplayed for the filter to be the only
        # thing holding it out.
        2027: [
            _d(2026, 10, 21),   # played
            _d(2026, 10, 31),   # played, the cutoff itself
            _d(2026, 11, 1),    # TODAY — tonight's games are not final
            _d(2026, 11, 5),    # not played at all
        ],
    }
    windows = {
        2026: (_d(2025, 10, 21), _d(2026, 4, 13)),
        2027: (_d(2026, 10, 20), _d(2027, 4, 12)),
    }

    plan = worker.plan_history_dates(
        [2026, 2027], {}, chunk=50,
        calendar_for=calendar.__getitem__,
        window_for=windows.__getitem__,
        not_after=_d(2026, 10, 31),
    )
    assert plan["dates"] == [
        _d(2025, 10, 21), _d(2025, 12, 25), _d(2026, 10, 21), _d(2026, 10, 31),
    ], (
        "only played regular-season dates are this job's to fetch; got "
        f"{plan['dates']}"
    )
    assert plan["candidates"] == 4
    assert plan["absent"] == 4


def test_the_backfill_reports_coverage_and_never_claims_to_be_finished(
    monkeypatch, caplog,
):
    """
    NO COMPLETION CLAIM, and that is the third version of this decision.

    Two earlier ones logged HISTORY BACKFILL COMPLETE and in both the
    branch was unreachable: the first counted preseason and playoff dates
    as missing, the second still counted dates that can never produce rows.
    All-Star Sunday is the measured example -- 2026-02-15 carries four
    events ESPN labels season type 2, played by WORLD, STARS and STRIPES,
    which `normalize_team` refuses because they are not NBA teams. That
    date is in the calendar, inside the regular-season window, played, and
    permanently unfillable, and it cannot be told apart from a date never
    read without recording every attempt.

    Telling an operator to wait for a signal that cannot arrive is worse
    than telling them the number. So the job reports coverage, says some
    dates may never fill, and asks them to decide.
    """
    import logging
    from datetime import date as _d

    import src.db.repository as repo
    import src.ingestion.espn_panel as espn

    monkeypatch.setenv(worker.ENV_ESPN_HISTORY_SEASONS, "2026")
    monkeypatch.setenv(worker.ENV_ESPN_INGEST, "true")
    monkeypatch.setattr(
        espn, "fetch_season_calendar", lambda year, **k: [_d(2025, 10, 21)],
    )
    monkeypatch.setattr(
        espn, "fetch_regular_season_window",
        lambda year, **k: (_d(2025, 10, 21), _d(2026, 4, 13)),
    )
    monkeypatch.setattr(
        repo, "panel_game_ids_by_date",
        lambda source: {"2025-10-21": {"espn:401", "espn:402"}},
    )

    seen: dict = {}

    def _capture(days=None, **kwargs):
        seen["days"] = list(days or [])
        seen["skip"] = set(kwargs.get("skip_game_ids") or ())
        return {"status": "OK"}

    monkeypatch.setattr(worker, "run_espn_ingest", _capture)

    with caplog.at_level(logging.INFO):
        out = worker.run_history_backfill()

    assert out["status"] != "COMPLETE", (
        "the job must not claim to be finished; a permanently unfillable "
        "date cannot be told apart from one never read"
    )
    assert out["backfill"]["without_rows"] == 0
    assert out["backfill"]["with_rows"] == 1
    # THE DATE IS STILL RE-EXAMINED, and the ids it already holds travel
    # with the request so it costs one scoreboard call and no summaries.
    assert seen["days"] == [_d(2025, 10, 21)]
    assert seen["skip"] == {"espn:401", "espn:402"}, (
        f"the held game ids did not reach the fetch: {seen['skip']}"
    )
    # EVERY held id travels, not just this chunk's dates: ESPN event ids are
    # globally unique, so an id from another night can never match an event
    # this chunk fetches, and narrowing by date bought nothing.
    assert out["backfill"]["already_held_events"] == 2
    assert "UNSET $" in caplog.text, "nothing told the operator to switch it off"


def test_the_backfill_writes_through_the_nightly_ingest_not_its_own_copy(
    monkeypatch,
):
    """
    ONE WRITE PATH. Every guard that makes the nightly ingest safe -- the
    id-namespace preflight, the two ownership views, the by-key read-back,
    the post-write namespace re-check -- applies identically to a night in
    2024. A backfill with its own copy of that sequence would be a second,
    quieter writer, and the quiet one is the one that drifts.

    So this asserts the backfill calls `run_espn_ingest` with its planned
    dates, rather than reaching for the mapper itself.
    """
    from datetime import date as _d

    import src.db.repository as repo
    import src.ingestion.espn_panel as espn

    monkeypatch.setenv(worker.ENV_ESPN_HISTORY_SEASONS, "2026")
    monkeypatch.setenv(worker.ENV_ESPN_INGEST, "true")
    monkeypatch.setenv(worker.ENV_ESPN_HISTORY_CHUNK, "2")

    season = [_d(2025, 10, 21), _d(2025, 10, 22), _d(2025, 10, 23)]
    monkeypatch.setattr(espn, "fetch_season_calendar", lambda year, **k: season)
    monkeypatch.setattr(
        espn, "fetch_regular_season_window",
        lambda year, **k: (_d(2025, 10, 21), _d(2026, 4, 13)),
    )
    monkeypatch.setattr(repo, "panel_game_ids_by_date", lambda source: {})

    seen: dict = {}

    def _capture(days=None, **kwargs):
        seen["days"] = list(days or [])
        seen["kwargs"] = kwargs
        return {"status": "OK", "panel_rows": 11}

    monkeypatch.setattr(worker, "run_espn_ingest", _capture)

    out = worker.run_history_backfill()

    assert seen.get("days") == season[:2], (
        f"the chunk bound was not honoured or the order was wrong: {seen.get('days')}"
    )
    assert out["status"] == "OK"
    assert out["backfill"]["candidates"] == 3, "the third date waits its turn"
    assert out["backfill"]["dates_this_run"] == 2
    assert out["backfill"]["first"] == "2025-10-21"
    assert out["backfill"]["last"] == "2025-10-22"


def test_the_nightly_window_is_unchanged_when_no_dates_are_passed(monkeypatch):
    """
    The parameter must not move the nightly. `run_espn_ingest()` with no
    argument still reads `PROPIQ_ESPN_LOOKBACK_DAYS` back from today, and a
    regression here would quietly stop the panel tracking last night.
    """
    from datetime import date as _d

    import src.db.repository as repo
    import src.ingestion.espn_panel as espn
    from src.utils import timezones

    monkeypatch.setattr(timezones, "pacific_calendar_date", lambda *a, **k: _d(2026, 11, 1))
    monkeypatch.setenv(worker.ENV_ESPN_INGEST, "true")
    monkeypatch.setenv(worker.ENV_ESPN_LOOKBACK, "3")
    # A non-empty panel, so the nightly window is used rather than the seed.
    monkeypatch.setattr(repo, "player_id_namespaces", lambda: {"espn": 1830})

    asked: dict = {}

    def _fetch(days, **kwargs):
        import pandas as pd

        # THE REAL REPORT, not a hand-rolled stand-in. The stub version
        # carried only the three fields this test happened to reach, so when
        # `skipped_already_held` was added the production code read an
        # attribute the stub did not have and this test failed for a reason
        # that had nothing to do with its subject. A fake that has to be
        # kept in step with the thing it fakes is a second definition of it.
        from src.ingestion.espn_panel import EspnPanelReport

        asked["days"] = list(days)
        return pd.DataFrame(), EspnPanelReport()

    monkeypatch.setattr(espn, "fetch_player_rows_for_dates", _fetch)

    worker.run_espn_ingest()

    assert asked.get("days") == [_d(2026, 11, 1), _d(2026, 10, 31), _d(2026, 10, 30)], (
        f"the nightly window changed: {asked.get('days')}"
    )


def test_explicit_days_reach_the_fetch_instead_of_the_nightly_window(monkeypatch):
    """
    THE MUTATION THAT SURVIVED THE FIRST ROUND OF THESE TESTS.

    `run_espn_ingest` computes the nightly window only `if days is None`.
    Change that to an unconditional branch and the backfill's planned dates
    are discarded and replaced with the last three days -- every chunk
    re-reads this week, `remaining` never falls, and the job reports OK for
    ever while the panel never deepens by a single night.

    Nothing caught it, because the test that asserts the plan reaches
    `run_espn_ingest` stubs `run_espn_ingest` out, and the test that pins the
    nightly window passes no days, so both branches agree. This one calls the
    REAL function with explicit dates and watches what the fetch is handed.
    """
    from datetime import date as _d

    import src.db.repository as repo
    import src.ingestion.espn_panel as espn

    monkeypatch.setenv(worker.ENV_ESPN_INGEST, "true")
    monkeypatch.setattr(repo, "player_id_namespaces", lambda: {"espn": 1830})

    # FOUR dates, not three. The first version of this test used three and
    # asserted `out["days"] == 3`, which a mutation hardcoding the nightly
    # lookback (also 3) satisfied exactly. The length has to differ from
    # every default in play for the assertion to be able to fail.
    wanted = [_d(2024, 10, 22), _d(2024, 10, 24), _d(2024, 11, 3), _d(2024, 11, 9)]
    asked: dict = {}

    def _fetch(days, **kwargs):
        import pandas as pd

        # THE REAL REPORT, not a hand-rolled stand-in. The stub version
        # carried only the three fields this test happened to reach, so when
        # `skipped_already_held` was added the production code read an
        # attribute the stub did not have and this test failed for a reason
        # that had nothing to do with its subject. A fake that has to be
        # kept in step with the thing it fakes is a second definition of it.
        from src.ingestion.espn_panel import EspnPanelReport

        asked["days"] = list(days)
        return pd.DataFrame(), EspnPanelReport()

    monkeypatch.setattr(espn, "fetch_player_rows_for_dates", _fetch)

    out = worker.run_espn_ingest(days=wanted, label="history backfill")

    assert asked.get("days") == wanted, (
        "the dates the caller asked for did not reach the fetch; the nightly "
        f"window replaced them. Got {asked.get('days')}"
    )
    # The fetch returns an empty frame, so this lands on the
    # DATA_NOT_AVAILABLE return -- NAMED, because that is the branch whose
    # day count is asserted here. It counts what was read rather than the
    # env lookback, which is what makes a backfill chunk's summary legible.
    # The OK branch's count is the same expression and is NOT covered by
    # this test: reaching it needs a database write.
    assert out["status"] == "DATA_NOT_AVAILABLE"
    assert out["days"] == len(wanted)
    assert out["label"] == "history backfill"


def test_an_empty_date_list_is_not_silently_turned_into_the_nightly_window(
    monkeypatch,
):
    """`days=[]` means "nothing to read", which is what a finished backfill
    hands over. Falling through to the nightly window there would make the
    completion case re-read this week every night."""
    import src.db.repository as repo
    import src.ingestion.espn_panel as espn

    monkeypatch.setenv(worker.ENV_ESPN_INGEST, "true")
    monkeypatch.setattr(repo, "player_id_namespaces", lambda: {"espn": 1830})

    def _must_not_fetch(*a, **k):
        raise AssertionError("fetched ESPN for an empty date list")

    monkeypatch.setattr(espn, "fetch_player_rows_for_dates", _must_not_fetch)

    out = worker.run_espn_ingest(days=[], label="history backfill")
    assert out["status"] == "SKIPPED"
    assert out["reason"] == "no dates requested"


def test_the_backfill_stops_at_yesterday_not_today(monkeypatch):
    """
    TONIGHT'S GAMES ARE NOT FINAL. `espn_panel` refuses a game that is not
    STATUS_FINAL, so today's date can never be filled by a backfill chunk
    run this morning -- it would be queued, fetched, refused per event, and
    queued again tomorrow. The nightly ingest re-reads the last three days
    and owns that window.

    Asserted at the JOB, because the cutoff is the job's to choose: the
    planner takes `not_after` as a parameter, so a planner test cannot see
    whether the caller passed today or yesterday.
    """
    from datetime import date as _d

    import src.db.repository as repo
    import src.ingestion.espn_panel as espn
    from src.utils import timezones

    monkeypatch.setenv(worker.ENV_ESPN_HISTORY_SEASONS, "2027")
    monkeypatch.setenv(worker.ENV_ESPN_INGEST, "true")
    monkeypatch.setenv(worker.ENV_ESPN_HISTORY_CHUNK, "50")
    monkeypatch.setattr(
        timezones, "pacific_calendar_date", lambda *a, **k: _d(2026, 11, 1),
    )
    monkeypatch.setattr(
        espn, "fetch_season_calendar",
        lambda year, **k: [_d(2026, 10, 30), _d(2026, 10, 31), _d(2026, 11, 1)],
    )
    monkeypatch.setattr(
        espn, "fetch_regular_season_window",
        lambda year, **k: (_d(2026, 10, 20), _d(2027, 4, 12)),
    )
    monkeypatch.setattr(repo, "panel_game_ids_by_date", lambda source: {})

    seen: dict = {}

    def _capture(days=None, **kwargs):
        seen["days"] = list(days or [])
        return {"status": "OK"}

    monkeypatch.setattr(worker, "run_espn_ingest", _capture)
    worker.run_history_backfill()

    assert seen["days"] == [_d(2026, 10, 30), _d(2026, 10, 31)], (
        f"today reached the backfill's work list: {seen['days']}"
    )


def test_the_backfill_is_refused_a_time_that_races_the_slate(monkeypatch):
    """
    AN OVERRIDE COULD PUT A WRITER ON TOP OF A READER.

    `_int_env` bounds each field to a legal clock value and nothing compared
    the result with the slate's, so `PROPIQ_ESPN_HISTORY_HOUR_PT=9` against a
    09:00 slate scheduled a job that walks seasons of box scores into
    `player_game_logs` at the moment the slate starts reading them. The slate
    would score a partially written panel and exit 0. `max_instances` is per
    job and does not help. Raised by review on #31, where a comment was the
    only thing standing in the way.
    """
    monkeypatch.delenv(worker.ENV_SLATE_HOUR, raising=False)
    monkeypatch.delenv(worker.ENV_SLATE_MINUTE, raising=False)

    # Safe: the documented default, comfortably clear of a 09:00 slate.
    monkeypatch.delenv(worker.ENV_ESPN_HISTORY_HOUR, raising=False)
    monkeypatch.delenv(worker.ENV_ESPN_HISTORY_MINUTE, raising=False)
    assert worker.history_backfill_time(9, 0) == (4, 30)

    # An operator-chosen time that still clears the slate is honoured.
    monkeypatch.setenv(worker.ENV_ESPN_HISTORY_HOUR, "6")
    monkeypatch.setenv(worker.ENV_ESPN_HISTORY_MINUTE, "15")
    assert worker.history_backfill_time(9, 0) == (6, 15)

    # ON TOP OF THE SLATE -> refused.
    monkeypatch.setenv(worker.ENV_ESPN_HISTORY_HOUR, "9")
    monkeypatch.setenv(worker.ENV_ESPN_HISTORY_MINUTE, "0")
    assert worker.history_backfill_time(9, 0) == (4, 30), "a collision stood"

    # INSIDE THE LEAD -> refused too. 08:30 does not collide but a chunk
    # takes about twenty minutes, so it can still be writing at 09:00.
    monkeypatch.setenv(worker.ENV_ESPN_HISTORY_HOUR, "8")
    monkeypatch.setenv(worker.ENV_ESPN_HISTORY_MINUTE, "30")
    assert worker.history_backfill_time(9, 0) == (4, 30)


def test_an_early_slate_moves_the_backfill_rather_than_overlapping_it(
    monkeypatch,
):
    """
    THE DEFAULT ITSELF CAN BE UNSAFE. An operator who moves the slate to
    05:00 leaves 04:30 inside the lead, and the slate is the job with the
    deadline — so the backfill moves, not the slate.
    """
    monkeypatch.delenv(worker.ENV_ESPN_HISTORY_HOUR, raising=False)
    monkeypatch.delenv(worker.ENV_ESPN_HISTORY_MINUTE, raising=False)

    hour, minute = worker.history_backfill_time(5, 0)
    assert (hour, minute) == (4, 0), (
        f"the backfill stayed at {hour}:{minute:02d} against a 05:00 slate"
    )

    # A slate inside the first hour of the day leaves no room before it, so
    # the backfill goes after it instead. Never overlapping is the invariant;
    # being earlier is only how that is usually achieved.
    hour, minute = worker.history_backfill_time(0, 30)
    assert (hour, minute) == (1, 30)


def test_the_registered_backfill_job_honours_the_refusal(monkeypatch):
    """The resolver is not the wiring. `build_scheduler` must use it, or the
    guard is a function nobody calls — the gap this repo keeps finding."""
    monkeypatch.setenv(worker.ENV_SLATE_HOUR, "9")
    monkeypatch.setenv(worker.ENV_SLATE_MINUTE, "0")
    monkeypatch.setenv(worker.ENV_ESPN_HISTORY_HOUR, "9")
    monkeypatch.setenv(worker.ENV_ESPN_HISTORY_MINUTE, "0")

    jobs = _jobs()
    backfill = jobs["espn_history_backfill"]
    assert (backfill["hour"], backfill["minute"]) != (9, 0), (
        "the backfill was registered on top of the slate"
    )
    assert (backfill["hour"], backfill["minute"]) == (4, 30)


def test_a_chunk_already_in_the_panel_is_a_no_op_not_missing_data(monkeypatch):
    """
    THE BACKFILL'S BEST OUTCOME MUST NOT READ AS ITS WORST.

    With `skip_game_ids`, a chunk whose every event is already in the panel
    fetches nothing BY DESIGN and comes back with an empty frame — full
    coverage, no work to do. `run_espn_ingest` reported that as
    `DATA_NOT_AVAILABLE: no finished game in the window`, which is simply
    false: there were games, and they are already ingested. An operator
    watching coverage rise would see the completion state reported as
    missing data. Found in review on #31, in code this PR had just added.

    `games_seen` counts parsed summary payloads only, so a held event never
    increments it — nothing fetched AND something held is the no-op.
    """
    from datetime import date as _d

    import src.db.repository as repo
    import src.ingestion.espn_panel as espn

    monkeypatch.setenv(worker.ENV_ESPN_INGEST, "true")
    monkeypatch.setattr(repo, "player_id_namespaces", lambda: {"espn": 1830})

    def _fetch(days, **kwargs):
        import pandas as pd
        held = set(kwargs.get("skip_game_ids") or ())
        report = espn.EspnPanelReport()
        report.skipped_already_held = len(held)
        return pd.DataFrame(), report

    monkeypatch.setattr(espn, "fetch_player_rows_for_dates", _fetch)

    out = worker.run_espn_ingest(
        days=[_d(2024, 10, 22)], label="history backfill",
        skip_game_ids={"espn:1", "espn:2"},
    )
    assert out["status"] == "SKIPPED", (
        f"a fully-covered chunk was reported as {out['status']}: "
        f"{out.get('reason')}"
    )
    assert "already in the panel" in out["reason"]
    assert out["already_held"] == 2


def test_a_window_with_nothing_held_and_nothing_fetched_is_still_unavailable(
    monkeypatch,
):
    """
    THE OTHER SIDE OF THAT BRANCH. A window ESPN has no finished game for is
    genuinely DATA_NOT_AVAILABLE, and softening it would hide a night the
    ingest truly failed to read — which is what the nightly path depends on
    this status for.
    """
    from datetime import date as _d

    import src.db.repository as repo
    import src.ingestion.espn_panel as espn

    monkeypatch.setenv(worker.ENV_ESPN_INGEST, "true")
    monkeypatch.setattr(repo, "player_id_namespaces", lambda: {"espn": 1830})

    def _fetch(days, **kwargs):
        import pandas as pd
        return pd.DataFrame(), espn.EspnPanelReport()

    monkeypatch.setattr(espn, "fetch_player_rows_for_dates", _fetch)

    out = worker.run_espn_ingest(days=[_d(2024, 10, 22)], label="nightly")
    assert out["status"] == "DATA_NOT_AVAILABLE"
    assert out["already_held"] == 0


def test_a_mixed_window_is_unavailable_rather_than_a_clean_no_op(monkeypatch):
    """
    SOME HELD, SOME FETCHED AND REFUSED. Something was wanted and not
    obtained, so this is not full coverage — but the held count travels with
    the status so the number is not a mystery.
    """
    from datetime import date as _d

    import src.db.repository as repo
    import src.ingestion.espn_panel as espn

    monkeypatch.setenv(worker.ENV_ESPN_INGEST, "true")
    monkeypatch.setattr(repo, "player_id_namespaces", lambda: {"espn": 1830})

    def _fetch(days, **kwargs):
        import pandas as pd
        report = espn.EspnPanelReport()
        report.skipped_already_held = 3
        report.games_seen = 1               # one was fetched...
        report.skipped_not_final.append("401909999")   # ...and refused
        return pd.DataFrame(), report

    monkeypatch.setattr(espn, "fetch_player_rows_for_dates", _fetch)

    out = worker.run_espn_ingest(
        days=[_d(2024, 10, 22)], label="history backfill",
        skip_game_ids={"espn:1", "espn:2", "espn:3"},
    )
    assert out["status"] == "DATA_NOT_AVAILABLE", (
        "a window with a refused game is not full coverage"
    )
    assert out["already_held"] == 3
    assert out["games_seen"] == 1


# --- a live board reads what the slate already projected -------------------


def test_a_live_board_reads_projections_and_never_compares_models(
    monkeypatch, tmp_path,
):
    """
    THE WALL THREE FIXES WALKED INTO.

    `compare_models_on_panel` drops every row whose `over_hit` is null
    before it splits (`compare.py`), so a game with no outcome yet is
    dropped and the function cannot describe tonight. #31 gave it the
    feature matrix it needed; #32 attached the forward slate and the
    forward rows were dropped at that same label filter — a fix that could
    not work, caught in review.

    The numbers already existed: `main.py` persists one row per (player,
    game, market) with the line attached and the EV gate applied, and
    nothing had ever read that table back. So a live board reads those and
    does not compare models at all.
    """
    from datetime import date as _d

    import src.db.repository as repo
    import src.pipeline.slate_board as slate_board_module
    from src.utils import timezones

    today = _d(2026, 11, 1)
    monkeypatch.setattr(timezones, "pacific_calendar_date", lambda *a, **k: today)
    monkeypatch.delenv("PROPIQ_BOARD_TRAIN_END", raising=False)
    monkeypatch.delenv("PROPIQ_BOARD_VALIDATION_END", raising=False)
    monkeypatch.setenv(worker.ENV_BOARD_CSV, str(tmp_path / "board.csv"))

    asked: dict = {}

    def _load(slate=None, **kwargs):
        asked["slate"] = slate
        return board_projections(str(today))

    monkeypatch.setattr(repo, "load_projections", _load)

    def _must_not_compare(*a, **k):
        raise AssertionError(
            "a live board reached compare_models_on_panel, which drops "
            "unlabelled rows and so cannot describe an unplayed game"
        )

    monkeypatch.setattr(slate_board_module, "build_slate_board", _must_not_compare)
    # The panel loader belongs to the backtest path; a live board must not
    # need it at all.
    monkeypatch.setattr(
        repo, "load_player_panel",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("a live board loaded the player panel")
        ),
    )

    out = worker.run_board()

    assert out["status"] == "OK", out
    assert out["source"] == "projections"
    assert asked["slate"] == str(today), (
        f"the board read the wrong slate date: {asked}"
    )
    # THE CSV IS THE DELIVERABLE, so read it. That the file EXISTS says
    # nothing: an empty frame with a header satisfies that, and so would a
    # board built from some other source. What matters is that the persisted
    # projections arrived in it unchanged.
    import pandas as pd

    assert (tmp_path / "board.csv").exists(), "no board CSV was written"
    written = pd.read_csv(tmp_path / "board.csv")
    assert not written.empty, "the board CSV is a header with no rows"

    # Every projected player reached the board, under the market it was
    # projected for.
    pairs = set(zip(written["player_name"], written["target_market"]))
    assert pairs == {
        ("HARNESS 0", "PTS"), ("HARNESS 1", "REB"), ("HARNESS 2", "AST"),
    }, f"the projections did not survive into the board: {sorted(pairs)}"

    # The numbers are CARRIED, not recomputed. `final_projection` 20.0 and
    # `line` 19.5 came from the projections row for HARNESS 0; a board that
    # had quietly re-derived them would not match.
    first = written.loc[written["player_name"] == "HARNESS 0"].iloc[0]
    assert float(first["prediction_mean"]) == 20.0, (
        f"the projection was altered on the way through: {first['prediction_mean']}"
    )
    assert float(first["line"]) == 19.5
    assert str(first["slate_date"]) == str(today)
    assert str(first["game_date"]) == str(today), (
        "the board lost game_date, which paper_research warns about per row"
    )
    assert str(first["event_id"]) == "espn:401909903"

    # AND THE NULL PROBABILITY SURVIVES AS NULL. This is production's state
    # with no model artifact: the row is written and says per row that it
    # cannot resolve P(over), rather than a fabricated 0.5 that would read as
    # a coin-flip opinion the model never formed.
    assert written["model_prob"].isna().all(), (
        "a probability appeared on a board built from projections that had "
        f"none: {written['model_prob'].tolist()}"
    )
    assert set(written["decision_status"]) == {"ABSTAIN"}
    assert (written["book_status"] == "DATA_NOT_AVAILABLE").all()
    assert written["why"].str.contains("cannot resolve P").all(), (
        f"no row says why it abstained: {written['why'].tolist()}"
    )


def test_a_backtest_window_still_uses_the_comparison_path(monkeypatch, tmp_path):
    """
    KEPT, NOT DELETED. An override into the past asks exactly what
    `compare_models_on_panel` is for — fit on history, score a labelled
    window — and those games have outcomes, so it works there. Replacing it
    with the projections path would mean a backtest could only ever
    describe days `main.py` happened to have run.
    """
    from datetime import date as _d

    import src.db.repository as repo
    import src.pipeline.slate_board as slate_board_module
    from src.utils import timezones

    monkeypatch.setattr(
        timezones, "pacific_calendar_date", lambda *a, **k: _d(2026, 11, 1),
    )
    monkeypatch.setenv("PROPIQ_BOARD_TRAIN_END", "2026-10-25")
    monkeypatch.setenv("PROPIQ_BOARD_VALIDATION_END", "2026-10-28")
    monkeypatch.setenv(worker.ENV_BOARD_CSV, str(tmp_path / "board.csv"))

    monkeypatch.setattr(repo, "load_player_panel", lambda *a, **k: board_panel())
    monkeypatch.setattr(
        worker, "_board_market_frames",
        lambda: {"team_games": None, "market_lines": None},
    )

    def _must_not_read_projections(*a, **k):
        raise AssertionError("a backtest board read tonight's projections")

    monkeypatch.setattr(repo, "load_projections", _must_not_read_projections)

    seen: dict = {}

    class _Result:
        def as_dict(self):
            return {"written_rows": 0}

    def _capture(frame, **kwargs):
        seen["train_end"] = kwargs.get("train_end")
        seen["validation_end"] = kwargs.get("validation_end")
        return _Result()

    monkeypatch.setattr(slate_board_module, "build_slate_board", _capture)

    out = worker.run_board()
    assert out["window"]["is_backtest"] is True
    assert out["source"] == "model comparison (backtest)"
    assert seen == {"train_end": "2026-10-25", "validation_end": "2026-10-28"}


def test_no_projections_for_the_slate_is_reported_not_crashed(monkeypatch, tmp_path):
    """
    A board asked for before the slate has run, or after a slate that
    refused, has no input. That is the pipeline reporting its own state —
    and it must name the day, because "no projections" with no date sends
    an operator looking at the wrong one.
    """
    from datetime import date as _d

    import pandas as pd

    import src.db.repository as repo
    from src.utils import timezones

    monkeypatch.setattr(
        timezones, "pacific_calendar_date", lambda *a, **k: _d(2026, 11, 1),
    )
    monkeypatch.delenv("PROPIQ_BOARD_TRAIN_END", raising=False)
    monkeypatch.delenv("PROPIQ_BOARD_VALIDATION_END", raising=False)
    monkeypatch.setenv(worker.ENV_BOARD_CSV, str(tmp_path / "board.csv"))
    monkeypatch.setattr(
        repo, "load_projections",
        lambda slate, **k: pd.DataFrame(columns=list(repo.PROJECTION_BOARD_COLS)),
    )

    out = worker.run_board()
    assert out["status"] == "DATA_NOT_AVAILABLE"
    assert "2026-11-01" in out["reason"], (
        f"the refusal does not name the day it asked for: {out['reason']}"
    )
    assert not (tmp_path / "board.csv").exists(), (
        "an empty board CSV was written, which dispatch would read as a slate "
        "with no value in it"
    )


def test_a_refused_board_removes_yesterdays_file_rather_than_leaving_it(
    monkeypatch, tmp_path
):
    """
    The version of the 9:03 failure that would have been worse. `run_board`
    refused and wrote nothing — but a board from a PREVIOUS successful run is
    still on disk, and `run_dispatch` reads that file without consulting the
    status `run_board` returned. Left in place, yesterday's lines go out under
    tonight's heading. The refusal removes it, so the operator sees "no
    recommendations" instead of wrong ones.
    """
    from datetime import date as _d

    import pandas as pd

    import src.db.repository as repo
    from src.utils import timezones

    board = tmp_path / "board.csv"
    pd.DataFrame([{
        "slate_date": "2026-10-31", "player_name": "YESTERDAY_A",
        "target_market": "PTS", "side": "over", "line": 24.5,
        "decision_status": "RECOMMENDED",
    }]).to_csv(board, index=False)

    monkeypatch.setattr(
        timezones, "pacific_calendar_date", lambda *a, **k: _d(2026, 11, 1),
    )
    monkeypatch.delenv("PROPIQ_BOARD_TRAIN_END", raising=False)
    monkeypatch.delenv("PROPIQ_BOARD_VALIDATION_END", raising=False)
    monkeypatch.setenv(worker.ENV_BOARD_CSV, str(board))
    monkeypatch.setattr(
        repo, "load_projections",
        lambda slate, **k: pd.DataFrame(columns=list(repo.PROJECTION_BOARD_COLS)),
    )

    out = worker.run_board()

    assert out["status"] == "DATA_NOT_AVAILABLE"
    assert out["discarded_stale_board"] is True, (
        "the refusal did not report removing the stale file, so nothing "
        "tells an operator why dispatch went quiet"
    )
    assert not board.exists(), (
        "yesterday's board survived a refusal; dispatch reads this file and "
        "would have published 2026-10-31 as tonight's slate"
    )


def test_a_board_already_written_for_this_slate_is_not_discarded(tmp_path):
    """
    The other head: the key is the rows' own `slate_date`, not the file's
    age. A board rebuilt at 09:05 for tonight is not stale, and deleting it
    would throw away the run that succeeded.
    """
    import pandas as pd

    board = tmp_path / "board.csv"
    pd.DataFrame([{
        "slate_date": "2026-11-01", "player_name": "TONIGHT_A",
        "target_market": "PTS", "side": "over", "line": 24.5,
    }]).to_csv(board, index=False)

    assert worker._discard_stale_board(board, "2026-11-01") is False
    assert board.exists(), "tonight's own board was deleted"


def test_an_unreadable_board_is_removed_rather_than_left_for_dispatch(tmp_path):
    """
    An unparseable file is not evidence of anything. Leaving it would move
    the failure to `run_dispatch`, which reads the CSV with no guard of its
    own against a parse error.
    """
    board = tmp_path / "board.csv"
    board.write_bytes(b"\x00\x01 not,a,csv\n\"unterminated")

    assert worker._discard_stale_board(board, "2026-11-01") is True
    assert not board.exists()


# --- the research preview: props visible without opening the gate ---------


def _withheld_board(tmp_path, slate):
    """A board in production's real state: rows, projections, NO probability."""
    import pandas as pd

    board = tmp_path / "board.csv"
    # UNDER FIRST, DELIBERATELY. Both sides carry the same |projection -
    # line|, so a preview that merely de-duplicated on (player, market)
    # would keep whichever row it saw first and print UNDER against a
    # projection ABOVE the line. Row order is the only thing that makes the
    # leaning-side filter load-bearing, and with `over` first a mutation
    # removing that filter survived. See
    # `test_the_preview_shows_the_props_without_claiming_them`.
    rows = [
        {"slate_date": slate, "game_date": slate, "event_id": "espn:401",
         "player_id": "espn:1", "player_name": "HARNESS JOKIC",
         "target_market": "PTS", "side": side, "line": 27.5,
         "prediction_mean": 29.1, "model_prob": None, "book_ev": None,
         "american_odds": None, "book_source": None,
         "decision_status": "ABSTAIN", "decision_basis": "unavailable"}
        for side in ("under", "over")
    ]
    pd.DataFrame(rows).to_csv(board, index=False)
    return board


def test_the_preview_is_off_by_default_so_no_prop_reaches_the_card(
    monkeypatch, tmp_path
):
    """
    THE DEFAULT IS UNCHANGED. The publication gate strips every name from a
    withheld board, and turning that off is an operator's decision, never a
    side effect of this code existing.
    """
    sent = {}

    def fake_send(embeds, config=None, **kw):
        sent["embeds"] = list(embeds)
        return type("R", (), {"status": "OK"})()

    slate = str(_frozen_pacific_day(monkeypatch))
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/1/x")
    monkeypatch.setenv(worker.ENV_DISPATCH, "true")
    monkeypatch.setenv(worker.ENV_BOARD_CSV, str(_withheld_board(tmp_path, slate)))
    monkeypatch.setenv(worker.ENV_CALIBRATION_REPORT, str(tmp_path / "absent.json"))
    monkeypatch.delenv(worker.ENV_RESEARCH_PREVIEW, raising=False)
    # ONE SUBJECT PER TEST. The slips are a separate section with its own
    # default (ON) and its own tests in tests/test_slip_dispatch.py; counting
    # total embeds here was a proxy for "no preview", and the proxy broke the
    # day a second section existed. Switched off so this asserts the preview.
    monkeypatch.setenv(worker.ENV_DISPATCH_SLIPS, "false")

    import src.notify.discord as discord_mod

    monkeypatch.setattr(discord_mod, "send_embeds", fake_send)

    out = worker.run_dispatch()

    assert out["publication"]["PUBLICATION_STATUS"] == "PUBLISH_WITHHELD"
    assert len(sent["embeds"]) == 1, "the preview was appended without being asked for"
    assert "HARNESS JOKIC" not in str(sent["embeds"]), (
        "a player name reached the card with the preview OFF"
    )


def test_the_preview_shows_the_props_without_claiming_them(monkeypatch, tmp_path):
    """
    WITH THE FLAG ON, the props are visible AND the refusal still goes out.

    The operator asked to see the slate; they did not ask to be told an
    uncalibrated lean is a recommendation. So this asserts BOTH: the player
    reaches the card, and the withheld verdict is still on it beside him.
    """
    sent = {}

    def fake_send(embeds, config=None, **kw):
        sent["embeds"] = list(embeds)
        return type("R", (), {"status": "OK"})()

    slate = str(_frozen_pacific_day(monkeypatch))
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/1/x")
    monkeypatch.setenv(worker.ENV_DISPATCH, "true")
    monkeypatch.setenv(worker.ENV_BOARD_CSV, str(_withheld_board(tmp_path, slate)))
    monkeypatch.setenv(worker.ENV_CALIBRATION_REPORT, str(tmp_path / "absent.json"))
    monkeypatch.setenv(worker.ENV_RESEARCH_PREVIEW, "true")
    # See the note in the test above: the slips are their own section.
    monkeypatch.setenv(worker.ENV_DISPATCH_SLIPS, "false")

    import src.notify.discord as discord_mod

    monkeypatch.setattr(discord_mod, "send_embeds", fake_send)

    out = worker.run_dispatch()
    blob = str(sent["embeds"])

    # THE GATE IS NOT OPENED. Its verdict is unchanged and its card still went.
    assert out["publication"]["PUBLICATION_STATUS"] == "PUBLISH_WITHHELD"
    assert len(sent["embeds"]) == 2, (
        f"expected the gate's card AND the preview, got {len(sent['embeds'])}"
    )
    assert "withheld from publication" in blob

    # AND THE PROP IS VISIBLE -- player, market, side, line and projection.
    assert "HARNESS JOKIC" in blob, "the preview did not name the player"
    assert "27.5" in blob and "29.1" in blob, (
        "the preview showed the player without the line and projection, which "
        "is the only content it exists to carry"
    )
    # ONE ROW PER (player, market): the board carries over AND under.
    assert blob.count("HARNESS JOKIC") == 1, (
        "both sides of the same prop were listed; the preview is meant to show "
        "the leaning side only"
    )
    # THE LEANING SIDE, and the fixture lists UNDER first so this cannot pass
    # by accident: a projection of 29.1 against a line of 27.5 leans OVER.
    assert "OVER" in blob, "proj 29.1 above line 27.5 leans OVER, not UNDER"
    assert "UNDER" not in blob, (
        "the preview printed UNDER for a projection above the line -- it is "
        "de-duplicating on (player, market) rather than picking the lean"
    )
    # NO FABRICATED NUMBER. There is no probability, so none may appear.
    assert "ABSTAIN" in blob
    assert "not recommendations" in blob.lower()
