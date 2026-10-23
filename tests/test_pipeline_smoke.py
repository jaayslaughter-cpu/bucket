"""
The pipeline smoke wrapper: stage selection, the first-wall verdict, and the
exit-code contract.

WHAT IS WORTH TESTING HERE, given the script's whole point is that it owns no
stage logic. Not the stages — each is another tool's business and shelling out
to all five in a unit test would be slow, networked and a second copy of what
those tools already assert. What IS this script's own: which stages are
selected, whether a non-blocking failure is forgiven, which failure is named
the wall, and the exit code. That last one is a contract: somebody will write
`pipeline_smoke && deploy`, and a script that exits 0 while the pipeline is
dead is worse than no script.

`run_stage` is monkeypatched throughout, so nothing here starts a subprocess,
touches the network or needs a database.

RESEARCH_ONLY.
"""

from __future__ import annotations

import pytest

from scripts import pipeline_smoke as smoke


def _fake_runner(outcomes: dict[str, int]):
    """Return a run_stage that reports the exit code named per stage."""
    def run(stage):
        code = outcomes.get(stage.name, 0)
        return smoke.Result(
            stage,
            smoke.PASS if code == 0 else smoke.FAIL,
            code,
            0.0,
            [f"fake exit {code}"],
        )
    return run


# --- the stage list -------------------------------------------------------


def test_the_default_pass_does_not_include_the_slow_networked_stage():
    """
    MEASURED AT 99 SECONDS on a first run: `audit-data` fetches player logs
    from stats.nba.com and retries three times with backoff, and that host is
    denied from this project's own build container. A smoke test run under
    pressure must not spend a minute and a half on a stage whose failure is
    forgiven anyway.
    """
    names = [s.name for s in smoke.stages(send=False)]
    assert "panel_audit" not in names
    assert names == [
        "scrape_features", "model_artifacts", "ev_contract", "discord_payload",
    ], "the default pass changed shape"

    with_panel = [s.name for s in smoke.stages(send=False, with_panel=True)]
    assert "panel_audit" in with_panel
    # In pipeline order, not appended at the end.
    assert with_panel.index("panel_audit") < with_panel.index("model_artifacts")


def test_the_panel_stage_is_non_blocking_and_time_bounded():
    panel = next(s for s in smoke.stages(send=False, with_panel=True)
                 if s.name == "panel_audit")
    assert panel.blocking is False, "a networked optional audit must not be a wall"
    assert panel.timeout <= 300


def test_nothing_is_sent_unless_send_is_passed():
    """
    The default must build the embed and post nothing. `notify-discord` is
    dry-run unless `--send`, and this script must not be the thing that
    quietly starts posting to a channel.
    """
    dry = next(s for s in smoke.stages(send=False) if s.name == "discord_payload")
    assert "--send" not in dry.argv

    live = next(s for s in smoke.stages(send=True) if s.name == "discord_payload")
    assert "--send" in live.argv


def test_every_stage_runs_on_this_interpreter():
    """
    `sys.executable`, never "python". This tree has pip for 3.13 on PATH beside
    a 3.11 interpreter, and the wrong one installed every dependency somewhere
    the project could not import it — exit 0, nothing importable. A subprocess
    that picks its own interpreter can diagnose a tree it is not running on.
    """
    import sys

    for stage in smoke.stages(send=False, with_panel=True):
        assert stage.argv[0] == sys.executable, stage.name


# --- the verdict and the exit code ---------------------------------------


def test_a_clean_pass_exits_zero(monkeypatch, capsys):
    monkeypatch.setattr(smoke, "run_stage", _fake_runner({}))
    assert smoke.main([]) == 0
    assert "every blocking stage passed" in capsys.readouterr().out


def test_a_blocking_failure_exits_nonzero_and_names_the_first_wall(
    monkeypatch, capsys,
):
    """
    THE CONTRACT. Somebody will write `pipeline_smoke && deploy`. Exiting 0
    while the pipeline is dead is worse than having no script at all.
    """
    monkeypatch.setattr(
        smoke, "run_stage",
        _fake_runner({"scrape_features": 1, "model_artifacts": 1}),
    )
    assert smoke.main([]) == 1
    out = capsys.readouterr().out
    assert "FIRST WALL is `scrape_features`" in out
    # The ones behind it are listed, because on a fresh deploy there are
    # usually several and stopping at the first hides them.
    assert "model_artifacts" in out


def test_a_non_blocking_failure_is_reported_but_forgiven(monkeypatch, capsys):
    """
    The panel can be absent while the live path is fine, and vice versa. A
    forgiven failure must still be visible — reported, not swallowed.
    """
    monkeypatch.setattr(smoke, "run_stage", _fake_runner({"panel_audit": 2}))
    assert smoke.main(["--with-panel"]) == 0
    out = capsys.readouterr().out
    assert "panel_audit" in out and "non-blocking" in out
    assert "every blocking stage passed" in out


def test_every_stage_runs_even_after_a_failure(monkeypatch):
    """
    A run that stops at the first wall hides the walls behind it. All five run
    unless --stop-on-fail.
    """
    ran: list[str] = []

    def record(stage):
        ran.append(stage.name)
        code = 1 if stage.name == "scrape_features" else 0
        return smoke.Result(stage, smoke.PASS if code == 0 else smoke.FAIL,
                            code, 0.0, [])

    monkeypatch.setattr(smoke, "run_stage", record)
    smoke.main([])
    assert ran == [
        "scrape_features", "model_artifacts", "ev_contract", "discord_payload",
    ]


def test_stop_on_fail_skips_the_rest_and_says_so(monkeypatch, capsys):
    monkeypatch.setattr(smoke, "run_stage", _fake_runner({"scrape_features": 1}))
    assert smoke.main(["--stop-on-fail"]) == 1
    out = capsys.readouterr().out
    assert "Not run:" in out
    assert "SKIPPED" in out


@pytest.mark.parametrize("argv,expected", [
    (["--stage", "ev_contract"], ["ev_contract"]),
    (["--stage", "ev_contract", "--stage", "discord_payload"],
     ["ev_contract", "discord_payload"]),
])
def test_stage_selection_runs_only_what_was_asked(monkeypatch, argv, expected):
    ran: list[str] = []

    def record(stage):
        ran.append(stage.name)
        return smoke.Result(stage, smoke.PASS, 0, 0.0, [])

    monkeypatch.setattr(smoke, "run_stage", record)
    assert smoke.main(argv) == 0
    assert ran == expected


def test_an_unknown_stage_is_refused_rather_than_silently_running_everything(
    monkeypatch, capsys,
):
    """A typo'd --stage that fell back to "all" would post to Discord on a
    run the operator thought was scoped to one stage."""
    monkeypatch.setattr(smoke, "run_stage", _fake_runner({}))
    assert smoke.main(["--stage", "not_a_stage"]) == 2
    assert "Unknown stage" in capsys.readouterr().err


def test_the_json_form_carries_the_same_verdict(monkeypatch, capsys):
    import json

    monkeypatch.setattr(smoke, "run_stage", _fake_runner({"model_artifacts": 1}))
    assert smoke.main(["--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["first_wall"] == "model_artifacts"
    assert payload["blocking_failures"] == ["model_artifacts"]
    assert payload["research_only"] is True
    assert payload["discord_sent"] is False
    assert len(payload["stages"]) == 4


# --- the output must not carry credentials -------------------------------


def test_stage_output_goes_through_the_projects_own_scrubber(monkeypatch):
    """
    A stage's output is an exception string, and `psycopg` and SQLAlchemy both
    put the connection URL — password included — into their error messages.
    This must route through `scheduler_worker.scrub_alert`, which already owns
    that rule, rather than a third copy of a redaction pattern.
    """
    leaky = "boom: postgresql://user:hunter2@db.example.com:5432/propiq"
    scrubbed = smoke._scrub(leaky)
    assert "hunter2" not in scrubbed


def test_output_is_withheld_entirely_if_the_scrubber_cannot_be_imported(
    monkeypatch,
):
    """
    This script reports on a tree that may be mid-breakage, so it must still
    produce a verdict when `scheduler_worker` will not import. Printing
    nothing is the safe direction: unscrubbed output could carry DATABASE_URL.
    """
    import builtins

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "scheduler_worker":
            raise ImportError("simulated breakage")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    out = smoke._scrub("postgresql://user:hunter2@host/db")
    assert "hunter2" not in out
    assert "withheld" in out
