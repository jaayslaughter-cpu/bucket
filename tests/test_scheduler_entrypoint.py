"""
tests/test_scheduler_entrypoint.py — nothing may be defined after the guard.

THE OUTAGE THIS EXISTS TO PREVENT, measured rather than imagined.

Deploy eee9dbe9 (2026-11-11 16:05 UTC) passed every health check, printed
`VERDICT: OK`, and then died:

    File "/app/scheduler_worker.py", line 4556, in main
      scheduler = build_scheduler()
    File "/app/scheduler_worker.py", line 4411, in build_scheduler
      run_espn_plays_backfill,
    NameError: name 'run_espn_plays_backfill' is not defined

`run_espn_plays_backfill` and `run_lineup_stints` were defined at lines 4671
and 4759 -- BELOW `if __name__ == "__main__": sys.exit(main())` at 4645.

WHY 3894 PASSING TESTS SAW NOTHING. Every test imports the module. Under
import `__name__` is `"scheduler_worker"`, the guard is false, the module body
runs to the end, and all four functions are defined before any test touches
them. Under `python scheduler_worker.py` -- which is what `scripts/start.sh`
runs in the container -- the guard is true and fires the moment execution
reaches it, so `main()` is called while the lines below it have never
executed.

An import-based suite cannot see this, however large it is. The check has to
be on the module's STRUCTURE, which is what this file does: parse the source,
find the entrypoint guard, and assert nothing executable is defined after it.

RESEARCH_ONLY. Static analysis of this repository's own entrypoint.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

#: Every module in this repository that a container or a person runs directly.
#: A module with an entrypoint guard is one whose top-to-bottom execution order
#: matters; one without is only ever imported and is not at risk.
ENTRYPOINT_MODULES = ("scheduler_worker.py", "main.py")

#: Node types whose placement after the guard is the defect. A function or
#: class defined below the guard does not exist when the guard's body runs.
DEFINING_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _main_guard_index(module: ast.Module) -> int | None:
    """The position of `if __name__ == "__main__":` in the module body."""
    for position, node in enumerate(module.body):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == "__name__"
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value == "__main__"
        ):
            return position
    return None


@pytest.mark.parametrize("filename", ENTRYPOINT_MODULES)
def test_nothing_is_defined_after_the_entrypoint_guard(filename: str) -> None:
    """
    The guard runs the program; anything below it does not exist yet.

    This is the assertion deploy eee9dbe9 needed and did not have. It is
    deliberately structural rather than behavioural: the failure only appears
    when the module is executed as `__main__`, and a test that imports it --
    which is every other test in this repository -- has already defined
    everything by the time it looks.
    """
    source = (REPO / filename).read_text(encoding="utf-8")
    module = ast.parse(source)
    guard = _main_guard_index(module)
    if guard is None:
        pytest.skip(f"{filename} has no __main__ guard, so execution order "
                    f"cannot strand a definition")

    stranded = [
        f"{type(node).__name__} {getattr(node, 'name', '?')} "
        f"at {filename}:{node.lineno}"
        for node in module.body[guard + 1:]
        if isinstance(node, DEFINING_NODES)
    ]
    assert not stranded, (
        "defined BELOW the `if __name__ == \"__main__\"` guard, so it does not "
        "exist when that guard calls into the program:\n  "
        + "\n  ".join(stranded)
        + "\n\nThis is invisible to every test that imports the module. Move "
          "the definition(s) above the guard."
    )


@pytest.mark.parametrize("filename", ENTRYPOINT_MODULES)
def test_the_guard_is_the_last_executable_statement(filename: str) -> None:
    """
    Also catches a module-level CALL or ASSIGNMENT placed after the guard.

    A function stranded below the guard is the defect that happened; a
    module-level assignment down there is the same mistake with a different
    symptom -- the name is missing rather than uncallable. Docstrings, imports
    and `__all__`-style constants are not at issue, so only statements that
    execute work are refused.
    """
    source = (REPO / filename).read_text(encoding="utf-8")
    module = ast.parse(source)
    guard = _main_guard_index(module)
    if guard is None:
        pytest.skip(f"{filename} has no __main__ guard")

    offenders: list[str] = []
    for node in module.body[guard + 1:]:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue  # a trailing string literal executes nothing
        offenders.append(
            f"{type(node).__name__} at {filename}:{node.lineno}"
        )
    assert not offenders, (
        "statement(s) after the `__main__` guard, which never run when the "
        "module is executed directly:\n  " + "\n  ".join(offenders)
    )


def test_the_detector_finds_the_arrangement_that_crashed_production() -> None:
    """
    MAKE IT FAIL FIRST, on the real shape rather than on a toy.

    A guard-placement checker that cannot flag the exact arrangement of deploy
    eee9dbe9 is not a check. This reconstructs it: a module whose scheduler
    builder references a job defined below the guard.
    """
    crashed = (
        "import sys\n"
        "def build_scheduler():\n"
        "    return run_espn_plays_backfill\n"
        "def main():\n"
        "    return build_scheduler()\n"
        'if __name__ == "__main__":\n'
        "    sys.exit(main())\n"
        "def run_espn_plays_backfill():\n"
        "    return {}\n"
    )
    module = ast.parse(crashed)
    guard = _main_guard_index(module)

    assert guard is not None, "the guard itself was not detected"
    stranded = [
        node for node in module.body[guard + 1:]
        if isinstance(node, DEFINING_NODES)
    ]
    assert len(stranded) == 1
    assert stranded[0].name == "run_espn_plays_backfill"

    # And the fixed arrangement is clean, so the detector is not simply
    # flagging every module it is shown.
    fixed = ast.parse(
        "import sys\n"
        "def run_espn_plays_backfill():\n"
        "    return {}\n"
        "def build_scheduler():\n"
        "    return run_espn_plays_backfill\n"
        "def main():\n"
        "    return build_scheduler()\n"
        'if __name__ == "__main__":\n'
        "    sys.exit(main())\n"
    )
    fixed_guard = _main_guard_index(fixed)
    assert fixed_guard is not None
    assert not [
        node for node in fixed.body[fixed_guard + 1:]
        if isinstance(node, DEFINING_NODES)
    ]


def test_the_jobs_the_scheduler_registers_are_defined_before_the_guard() -> None:
    """
    The specific names that crashed, pinned by name.

    The structural check above is the general guard. This is the regression:
    `build_scheduler` references these four, and all four were below the line.
    """
    source = (REPO / "scheduler_worker.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    guard = _main_guard_index(module)
    assert guard is not None

    before = {
        node.name for node in module.body[:guard]
        if isinstance(node, DEFINING_NODES)
    }
    for name in (
        "run_espn_plays_backfill",
        "run_lineup_stints",
        "_espn_season_start",
        "_season_days_between",
    ):
        assert name in before, (
            f"{name} is not defined above the __main__ guard; "
            f"`build_scheduler` references it and the worker will NameError "
            f"at boot"
        )
