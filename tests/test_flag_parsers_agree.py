"""
Two boolean env parsers, one meaning.

`main._flag_env` and `scheduler_worker._flag` do the same job in two
processes: turn a Railway variable into a boolean. They gave different
answers for the same input -- `maybe` was the configured default to one and
`off` to the other.

NO VARIABLE IS READ BY BOTH TODAY, and this file does not pretend
otherwise. Review on #32 raised it against `PROPIQ_FORWARD_SLATE`, read by
the forward-slate board branch; that branch was replaced by the projections
path and took the live disagreement with it. `PROPIQ_ESPN_INGEST`,
`PROPIQ_DISPATCH` and `PROPIQ_DISPATCH_ABSTENTIONS` belong to the worker
alone. So this guards a trap and not a bug: the next switch both processes
need would be read through two parsers that disagree, and nothing would log
it.

They stay two small implementations rather than one import, for the reason
`_int_env` does: `main.py` is the orchestrator and must not come to require
APScheduler in order to read a boolean. This file is the price of that --
it holds both to the same answer on every input, including the ones nobody
means to type.
"""

from __future__ import annotations

import pytest

import main as orchestrator
import scheduler_worker as worker

#: Every input shape a human or a Railway variable can produce. The last
#: group is the one that mattered: values that are neither on nor off.
INPUTS = [
    "1", "true", "TRUE", "True", "yes", "on", " on ", "\tTrue\n",
    "0", "false", "FALSE", "no", "off", " off ",
    "", "   ", "maybe", "2", "-1", "null", "None", "y", "n", "t", "f",
    "enabled", "disabled", "ON!", "true ish",
]


@pytest.mark.parametrize("raw", INPUTS)
@pytest.mark.parametrize("default", [True, False])
def test_the_two_parsers_return_the_same_answer(monkeypatch, raw, default):
    monkeypatch.setenv("PROPIQ_FLAG_PARSER_HARNESS", raw)
    mine = worker._flag("PROPIQ_FLAG_PARSER_HARNESS", default)
    theirs = orchestrator._flag_env("PROPIQ_FLAG_PARSER_HARNESS", default)
    assert mine is theirs, (
        f"{raw!r} with default {default}: worker._flag says {mine}, "
        f"main._flag_env says {theirs} -- the slate job and the board job "
        f"would disagree about the same switch"
    )


@pytest.mark.parametrize("default", [True, False])
def test_an_unset_variable_takes_the_default_in_both(monkeypatch, default):
    monkeypatch.delenv("PROPIQ_FLAG_PARSER_HARNESS", raising=False)
    assert worker._flag("PROPIQ_FLAG_PARSER_HARNESS", default) is default
    assert (
        orchestrator._flag_env("PROPIQ_FLAG_PARSER_HARNESS", default) is default
    )


def test_an_unusable_value_defaults_rather_than_reading_as_off(monkeypatch):
    """
    THE CASE THAT WAS WRONG, named on its own so a regression says which.
    `maybe` is not `off`. A variable set to something nobody can parse means
    the operator's intent is unknown, and the honest answer is the
    configured default -- not the quietest one.
    """
    monkeypatch.setenv("PROPIQ_FLAG_PARSER_HARNESS", "maybe")
    assert worker._flag("PROPIQ_FLAG_PARSER_HARNESS", True) is True
    assert worker._flag("PROPIQ_FLAG_PARSER_HARNESS", False) is False


def test_no_boolean_switch_is_read_by_both_processes_through_two_parsers():
    """
    THE CLAIM ABOVE, ASSERTED. This file says no variable is currently read
    by both; if one appears, the parsers agreeing (above) is what keeps it
    safe -- and this test is the thing that notices, so the docstrings stop
    being true the loud way rather than the quiet way.

    Reads both modules' source for `PROPIQ_*` literals rather than their
    constants, because a shared switch could arrive as an inline string.
    """
    import re
    from pathlib import Path

    def names(path: str) -> set[str]:
        return set(re.findall(r"PROPIQ_[A-Z0-9_]+", Path(path).read_text()))

    shared = names("scheduler_worker.py") & names("main.py")
    # PROPIQ_MODEL is a PATH, not a flag, and PROPIQ_FORWARD_SLATE survives
    # in the worker only inside prose explaining why it no longer reads it.
    booleans = {
        n for n in shared
        if n not in {"PROPIQ_MODEL", "PROPIQ_FORWARD_SLATE"}
    }
    assert not booleans, (
        f"{sorted(booleans)} is now read by both processes. That is fine -- "
        f"the parsers agree -- but the docstrings here and on "
        f"`scheduler_worker._flag` say no such switch exists, and they are "
        f"now wrong. Update them."
    )
