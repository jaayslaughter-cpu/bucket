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


def _env_names_referenced(path: str) -> set[str]:
    """
    `PROPIQ_*` names a module actually REFERENCES, excluding prose.

    PARSED, NOT GREPPED, and the first version of this grepped raw source.
    That counted the name inside `scheduler_worker._flag`'s own docstring --
    which explains why the worker no longer reads `PROPIQ_FORWARD_SLATE` --
    so the name looked shared and had to be exempted, and the exemption made
    the test blind to the thing it exists to catch. Raised by review on #32.

    An AST walk drops comments for free and drops the leading docstring of
    every module, class and function explicitly. What is left is a string the
    module evaluates: a constant, an argument, a dict key. That is a
    reference.
    """
    import ast
    import re
    from pathlib import Path

    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            docstrings.add(id(first.value))

    found: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
        ):
            found |= set(re.findall(r"PROPIQ_[A-Z0-9_]+", node.value))
    return found


def test_the_docstring_prose_is_not_what_this_test_reads():
    """
    THE SCANNER'S OWN BLIND SPOT, closed and then asserted closed.

    `scheduler_worker._flag`'s docstring names `PROPIQ_FORWARD_SLATE` in
    prose. If the scanner ever counts that as a reference again, the test
    below needs an exemption for it, and an exemption is exactly what made
    the previous version unable to see the name becoming shared.
    """
    worker_refs = _env_names_referenced("scheduler_worker.py")
    assert "PROPIQ_FORWARD_SLATE" not in worker_refs, (
        "the scanner is counting prose as a reference again; the test below "
        "is now measuring the docstrings rather than the code"
    )
    # And it has not gone blind in the other direction: it still sees the
    # switches the worker really does read.
    assert {"PROPIQ_DISPATCH", "PROPIQ_ESPN_INGEST"} <= worker_refs, (
        f"the scanner stopped seeing real references: {sorted(worker_refs)}"
    )


def test_no_boolean_switch_is_read_by_both_processes_through_two_parsers():
    """
    THE CLAIM ABOVE, ASSERTED. This file says no variable is currently read
    by both; if one appears, the parsers agreeing (above) is what keeps it
    safe -- and this test is the thing that notices, so the docstrings stop
    being true the loud way rather than the quiet way.
    """
    shared = (
        _env_names_referenced("scheduler_worker.py")
        & _env_names_referenced("main.py")
    )
    # ONE EXEMPTION, and it is not a flag: PROPIQ_MODEL names an artifact
    # PATH. Both processes resolve it and neither parses it as a boolean, so
    # it says nothing about the two parsers agreeing.
    booleans = shared - {"PROPIQ_MODEL"}
    assert not booleans, (
        f"{sorted(booleans)} is now read by both processes. That is fine -- "
        f"the parsers agree -- but the docstrings here and on "
        f"`scheduler_worker._flag` say no such switch exists, and they are "
        f"now wrong. Update them."
    )
