"""The invariants that catch a job returning OK while doing nothing.

`notify_failure` alerted only on a RAISED exception. Every defect found in the
week to 2026-11-11 was in the other class, and every one was found by a person
reading Railway logs by hand. These checks watch the pipeline's OUTPUTS and
ask whether they are consistent with each other.

Every guard was mutation-tested: fix reverted, named test watched go red, fix
put back.
"""

from __future__ import annotations

from datetime import date

from src.pipeline import self_check as sc

TODAY = date(2026, 11, 11)


def _rec(slate, *, recorded=250, parent="abc", rows=None, shadow_prob=0.44,
         served_prob=0.61, game_date=None):
    """A decision record of the shape `decision_record.build_record` writes."""
    n = recorded if rows is None else len(rows)
    body_rows = rows if rows is not None else [
        {
            "decision": "RECORDED", "player_name": f"P{i}", "market": "PTS",
            "game_date": game_date or slate, "prob_over": served_prob,
            "shadow": {"distribution": shadow_prob},
        }
        for i in range(n)
    ]
    return {
        "slate_date": slate,
        "as_of_ts": f"{slate}T17:00:00Z",
        "prev_sha256": parent,
        "rows": body_rows,
        "counts": {
            "total": len(body_rows),
            "by_decision": {
                "RECORDED": sum(
                    1 for r in body_rows if r.get("decision") == "RECORDED"
                ),
            },
        },
    }


# --- no evidence is never a green OK ------------------------------------


def test_nothing_reports_ok_without_evidence():
    """
    THE WHOLE POINT OF THE THIRD STATUS. A monitor that answers OK when it has
    no data converts absence of evidence into a statement that everything is
    fine -- which is precisely the failure mode it exists to catch.
    """
    results = sc.run_all([], [], as_of=TODAY)
    assert len(results) == 5
    assert all(r.status == sc.CHECK_CANNOT_TELL for r in results), [
        (r.name, r.status) for r in results
    ]
    assert not any(r.alerting for r in results)


def test_every_check_is_in_the_sweep():
    """A check nobody calls is the defect this module is about."""
    names = {r.name for r in sc.run_all([], [], as_of=TODAY)}
    assert names == {
        "a_slate_was_recorded_recently",
        "recorded_volume_has_not_collapsed",
        "every_component_answered",
        "aged_slates_are_graded",
        "records_are_chained",
    }


def test_one_broken_check_does_not_hide_the_others(monkeypatch):
    """
    A monitor that dies on its first surprise reports nothing about the rest,
    so a raising check becomes its OWN alert and the sweep continues.
    """
    def boom(*_a, **_kw):
        raise RuntimeError("the check itself is broken")

    monkeypatch.setattr(sc, "records_are_chained", boom)
    results = sc.run_all([_rec("2026-11-09")], [], as_of=TODAY)

    assert len(results) == 5, "the sweep stopped at the broken check"
    broken = next(r for r in results if r.name == "records_are_chained")
    assert broken.alerting
    assert "RuntimeError" in broken.reason
    assert "silent monitor" in broken.reason


# --- the slate is still recording ---------------------------------------


def test_a_stale_newest_record_alerts():
    """
    THE SOURCE OF THE "graded 0 for three nights" CLASS. With nothing being
    recorded, nothing can ever be graded, and the 04:00 job reports an honest
    zero every night while the real failure sits twelve hours upstream.
    """
    r = sc.a_slate_was_recorded_recently([_rec("2026-11-05")], as_of=TODAY)
    assert r.alerting
    assert "2026-11-05" in r.reason and "6 day(s)" in r.reason


def test_a_fresh_record_is_ok():
    r = sc.a_slate_was_recorded_recently([_rec("2026-11-10")], as_of=TODAY)
    assert r.status == sc.CHECK_OK


def test_an_unreadable_slate_date_cannot_tell_rather_than_passing():
    r = sc.a_slate_was_recorded_recently(
        [_rec("not-a-date"), _rec("")], as_of=TODAY,
    )
    assert r.status == sc.CHECK_CANNOT_TELL


# --- the chain ----------------------------------------------------------


def test_a_record_with_no_parent_breaks_the_chain():
    records = [_rec("2026-11-09", parent=None), _rec("2026-11-10", parent=None)]
    r = sc.records_are_chained(records)
    assert r.alerting and "names no parent" in r.reason


def test_two_records_claiming_one_parent_is_a_fork():
    """Two writers produced records from the same state, so one of them lost."""
    records = [
        _rec("2026-11-08", parent=None),
        _rec("2026-11-09", parent="same"),
        _rec("2026-11-10", parent="same"),
    ]
    r = sc.records_are_chained(records)
    assert r.alerting and "more than one record" in r.reason


def test_a_valid_chain_is_ok():
    records = [
        _rec("2026-11-08", parent=None),
        _rec("2026-11-09", parent="a"),
        _rec("2026-11-10", parent="b"),
    ]
    assert sc.records_are_chained(records).status == sc.CHECK_OK


def test_one_record_cannot_tell():
    assert sc.records_are_chained([_rec("2026-11-10")]).status == sc.CHECK_CANNOT_TELL


# --- aged slates are graded ---------------------------------------------


def test_an_aged_slate_with_nothing_graded_alerts():
    """A settlement that did nothing leaves the slate at 100% ungraded."""
    records = [_rec("2026-11-08", recorded=300)]
    r = sc.aged_slates_are_graded(records, [], as_of=TODAY)
    assert r.alerting
    assert "300 recorded row(s) never graded (100%)" in r.reason
    assert "should be VOID, not PENDING" in r.reason


def test_an_aged_slate_that_graded_is_ok():
    records = [_rec("2026-11-08", recorded=300)]
    graded = ["2026-11-08"] * 290
    assert sc.aged_slates_are_graded(records, graded, as_of=TODAY).status == sc.CHECK_OK


def test_a_recent_slate_is_not_judged_for_being_ungraded():
    """Settlement has not run against it yet, so ungraded is expected."""
    records = [_rec("2026-11-11", recorded=300)]
    r = sc.aged_slates_are_graded(records, [], as_of=TODAY)
    assert r.status == sc.CHECK_CANNOT_TELL


def test_the_game_date_is_what_joins_not_the_slate_date():
    """
    A slate date and a game date differ across midnight, and the ledger keys
    on the game. Matching on the slate date would read a correctly graded
    slate as entirely ungraded.
    """
    records = [_rec("2026-11-08", recorded=100, game_date="2026-11-09")]
    assert sc.aged_slates_are_graded(
        records, ["2026-11-09"] * 100, as_of=TODAY,
    ).status == sc.CHECK_OK
    assert sc.aged_slates_are_graded(
        records, ["2026-11-08"] * 100, as_of=TODAY,
    ).alerting


def test_a_slate_that_recorded_nothing_is_not_counted_as_ungraded():
    records = [_rec("2026-11-08", rows=[])]
    r = sc.aged_slates_are_graded(records, [], as_of=TODAY)
    assert r.status == sc.CHECK_CANNOT_TELL


# --- every component answered -------------------------------------------


def test_a_shadow_that_scored_nothing_alerts():
    """
    THE DEFECT THAT HID LONGEST. A fitted `line_aware` artifact was persisted,
    checksummed and verified on every run while no resolver could find it, so
    a component the pipeline believed it was measuring answered nothing.
    """
    r = sc.every_component_answered([_rec("2026-11-10", shadow_prob=None)])
    assert r.alerting
    assert "distribution" in r.reason and "scored 0 of" in r.reason


def test_a_serving_component_that_scored_nothing_alerts():
    r = sc.every_component_answered([_rec("2026-11-10", served_prob=None)])
    assert r.alerting
    assert "SERVING component scored 0" in r.reason


def test_both_scoring_is_ok():
    r = sc.every_component_answered([_rec("2026-11-10")])
    assert r.status == sc.CHECK_OK
    assert "distribution=250" in r.reason


def test_no_shadow_at_all_cannot_tell_rather_than_passing():
    """
    `PROPIQ_SHADOW_COMPONENTS=none` is a legitimate setting, and with no
    shadow there is nothing to compare the server against -- which is not the
    same as the comparison being healthy.
    """
    rows = [
        {"decision": "RECORDED", "player_name": "P", "market": "PTS",
         "game_date": "2026-11-10", "prob_over": 0.6}
    ]
    r = sc.every_component_answered([_rec("2026-11-10", rows=rows)])
    assert r.status == sc.CHECK_CANNOT_TELL


def test_only_the_newest_record_is_judged():
    """
    A component added today scored nothing on last week's records, by
    definition. Judging the whole history would alert for ever on every new
    component.
    """
    old = _rec("2026-11-08", shadow_prob=None)
    new = _rec("2026-11-10")
    assert sc.every_component_answered([old, new]).status == sc.CHECK_OK


# --- volume ------------------------------------------------------------


def test_a_collapsed_slate_alerts():
    records = [_rec(f"2026-11-0{d}", recorded=n)
               for d, n in ((6, 300), (7, 280), (8, 310), (9, 12))]
    r = sc.recorded_volume_has_not_collapsed(records)
    assert r.alerting
    assert "12 prediction(s)" in r.reason and "median of 300" in r.reason


def test_ordinary_variation_does_not_alert():
    """
    312 on 2026-11-09 against 257 on 11-10 is an honest difference -- the
    second night had fewer posted lines. A day-on-day comparison would cry
    wolf every other night, which is why this is a median over history.
    """
    records = [_rec(f"2026-11-0{d}", recorded=n)
               for d, n in ((6, 300), (7, 312), (8, 280), (9, 257))]
    assert sc.recorded_volume_has_not_collapsed(records).status == sc.CHECK_OK


def test_too_little_history_cannot_tell():
    records = [_rec("2026-11-09", recorded=300), _rec("2026-11-10", recorded=5)]
    assert sc.recorded_volume_has_not_collapsed(records).status == sc.CHECK_CANNOT_TELL


def test_an_all_zero_history_cannot_tell_rather_than_dividing_by_it():
    records = [_rec(f"2026-11-0{d}", rows=[]) for d in (6, 7, 8, 9)]
    r = sc.recorded_volume_has_not_collapsed(records)
    assert r.status == sc.CHECK_CANNOT_TELL
    assert "no normal volume" in r.reason


# --- the summary a person reads -----------------------------------------


def test_the_summary_carries_every_broken_reason():
    results = [
        sc.CheckResult("a", sc.CHECK_ALERT, "first thing broke"),
        sc.CheckResult("b", sc.CHECK_OK, "fine"),
        sc.CheckResult("c", sc.CHECK_ALERT, "second thing broke"),
        sc.CheckResult("d", sc.CHECK_CANNOT_TELL, "no data"),
    ]
    s = sc.summarise(results)
    assert s["checks"] == 4
    assert s["by_status"] == {sc.CHECK_ALERT: 2, sc.CHECK_OK: 1, sc.CHECK_CANNOT_TELL: 1}
    assert s["alerting"] == ["a", "c"]
    assert s["reasons"] == ["a: first thing broke", "c: second thing broke"], (
        "an alert body that names the check but not the reason sends a reader "
        "back to the logs, which is what this exists to replace"
    )
    assert len(s["results"]) == 4


def test_cannot_tell_never_counts_as_alerting():
    assert not sc.CheckResult("x", sc.CHECK_CANNOT_TELL, "no data").alerting
    assert not sc.CheckResult("x", sc.CHECK_OK, "fine").alerting
    assert sc.CheckResult("x", sc.CHECK_ALERT, "broke").alerting


def test_it_reads_nothing_and_writes_nothing():
    """
    Pure, like `settlement.component_comparison`: the caller does the reads
    and the alerting. A monitor that opened its own database connection would
    have a second way to fail and would fail it silently.
    """
    import ast
    from pathlib import Path

    source = Path("src/pipeline/self_check.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    # DOCSTRINGS AND COMMENTS ARE STRIPPED, because this module's docstring
    # NAMES `notify_failure` to explain what it does not do -- and a blunt
    # substring scan failed on its own explanation. The check is about code.
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            node.value = ast.Constant(value="")
    code = ast.unparse(tree)

    for forbidden in ("session_scope", "read_records", "open(", "requests",
                      "datetime.now", "date.today", "notify_failure"):
        assert forbidden not in code, f"{forbidden!r} makes this impure"
    # And no clock of its own: `as_of` is a parameter on every check that
    # needs one, so the same records always produce the same verdict.
    assert "pacific_calendar_date" not in code
    for fn in ("a_slate_was_recorded_recently", "aged_slates_are_graded", "run_all"):
        assert f"{fn}(" in code


# --- the wiring: a broken invariant must actually reach Discord ---------


def test_the_sweep_routes_alerts_to_notify_failure(monkeypatch):
    """
    THE HALF THAT MATTERS. A monitor whose findings go only to a log is the
    thing it was built to replace -- every defect in the week to 2026-11-11
    was already in the logs, and was found by a person reading them.
    """
    import scheduler_worker as worker

    sent = []
    monkeypatch.setattr(
        worker, "notify_failure",
        lambda stage, detail, **kw: sent.append((stage, detail)) or {"status": "SENT"},
    )
    # One stale record: the newest slate is six days old.
    summary = worker._run_self_check([_rec("2026-11-05")], [])

    assert len(sent) == 1, f"the alert did not reach notify_failure: {sent}"
    stage, detail = sent[0]
    assert stage == "self_check", "a per-check stage would be deduped away"
    assert "a_slate_was_recorded_recently" in detail
    assert "2026-11-05" in detail, "the body does not name what broke"
    assert summary["alerting"], summary


def test_a_healthy_sweep_sends_nothing(monkeypatch):
    """Silence is the correct output when every invariant holds."""
    import scheduler_worker as worker

    sent = []
    monkeypatch.setattr(
        worker, "notify_failure",
        lambda stage, detail, **kw: sent.append((stage, detail)),
    )
    records = [
        _rec("2026-11-08", parent=None, recorded=300),
        _rec("2026-11-09", parent="a", recorded=290),
        _rec("2026-11-10", parent="b", recorded=280),
        _rec("2026-11-11", parent="c", recorded=270),
    ]
    graded = [{"game_date": "2026-11-08"}] * 295 + [{"game_date": "2026-11-09"}] * 285
    summary = worker._run_self_check(records, graded)

    assert sent == [], f"a healthy pipeline raised an alert: {sent}"
    assert not summary["alerting"], summary["results"]


def test_one_alert_carries_every_broken_invariant(monkeypatch):
    """
    `notify_failure` dedupes by stage per day, so five stages would send five
    messages tonight and none of them again tomorrow. One stage, every reason.
    """
    import scheduler_worker as worker

    sent = []
    monkeypatch.setattr(
        worker, "notify_failure",
        lambda stage, detail, **kw: sent.append((stage, detail)),
    )
    # Stale AND unchained AND an aged slate with nothing graded.
    records = [
        _rec("2026-11-01", parent=None, recorded=300),
        _rec("2026-11-02", parent=None, recorded=300),
    ]
    worker._run_self_check(records, [])

    assert len(sent) == 1
    _stage, detail = sent[0]
    assert "a_slate_was_recorded_recently" in detail
    assert "aged_slates_are_graded" in detail
    assert "records_are_chained" in detail
    assert detail.count("|") >= 2, f"the reasons were not joined: {detail}"


def test_the_monitor_never_takes_the_nights_report_down(monkeypatch):
    """
    It runs inside the section that produces the residuals. A monitor that
    raises would cost the report it was watching.
    """
    import scheduler_worker as worker

    monkeypatch.setattr(
        worker, "notify_failure",
        lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("discord is down")),
    )
    out = worker._run_self_check([_rec("2026-11-05")], [])
    assert out["status"] == "FAILED"
    assert "discord is down" in out["error"]


def test_the_two_jobs_that_never_alerted_now_do():
    """
    `run_loss_analysis` and `run_training` had no try/except at all, so an
    exception propagated into APScheduler, which logs it and moves on -- and
    in Railway every Python line is already red, so a real failure was
    indistinguishable from an INFO line by eye.

    `run_dispatch` is still deliberately absent: alerting about a broken
    webhook through that webhook is a loop with no exit.
    """
    import inspect

    import scheduler_worker as worker

    for fn in (worker.run_loss_analysis, worker.run_training):
        assert "notify_failure" in inspect.getsource(fn), fn.__name__
    assert "notify_failure" not in inspect.getsource(worker.run_dispatch)


def test_the_0400_job_actually_runs_the_sweep(monkeypatch, tmp_path):
    """
    MUTATION THAT SURVIVED THE FIRST ATTEMPT: deleting the
    `_run_self_check(records, rows)` call from `_run_loss_analysis_locked`.
    Every wiring test above calls `_run_self_check` DIRECTLY, so removing the
    one line that invokes it from the night's job changed nothing they could
    see -- a monitor nobody calls, which is the exact defect class this module
    was written for, reproduced in its own tests.

    This runs the real job body with its two reads stubbed.
    """
    import scheduler_worker as worker
    import src.db.repository as repo
    import src.settlement.decision_record as dr

    graded = [
        {"game_date": "2026-11-08", "model_projection": 20.0,
         "actual_result": 21.0, "nba_player_id": "p1", "did_not_play": False},
    ]
    # One record, dated far enough back that `a_slate_was_recorded_recently`
    # must alert -- so a sweep that ran is distinguishable from one that did not.
    records = [_rec("2026-11-01", parent=None, recorded=300)]

    monkeypatch.setattr(repo, "load_graded_prop_residual_rows", lambda **_k: graded)
    monkeypatch.setattr(repo, "load_player_panel", lambda **_k: __import__(
        "pandas").DataFrame())
    monkeypatch.setattr(dr, "read_records", lambda *_a, **_k: records)
    monkeypatch.setattr(
        "src.utils.volume.resolve_state_root", lambda: (tmp_path, "test"),
    )
    monkeypatch.setattr(worker, "loss_analysis_report_path", lambda: tmp_path / "la.json")

    sent = []
    monkeypatch.setattr(
        worker, "notify_failure",
        lambda stage, detail, **kw: sent.append((stage, detail)),
    )

    report = worker._run_loss_analysis_locked()

    assert "self_check" in report, (
        "the night's report carries no self_check section, so the sweep did "
        "not run inside the job"
    )
    assert report["self_check_alerting"], report["self_check"]
    assert "a_slate_was_recorded_recently" in report["self_check_alerting"]
    assert sent and sent[0][0] == "self_check", (
        f"the sweep ran but its alert never left the job: {sent}"
    )

    # AND IT REACHED THE FILE AN OPERATOR READS, not only the return dict.
    import json

    written = json.loads((tmp_path / "la.json").read_text(encoding="utf-8"))
    assert "self_check" in written
    assert written["self_check"]["alerting"] == report["self_check_alerting"]
