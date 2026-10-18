"""
A document that names a module must say whether that module exists.

FOUND 2026-10-18, from a bug report. `docs/pickem_props.md` described a
pick'em capture subsystem in the present tense -- seven modules, a CLI, a
storage layout, and a status table reading "Endpoint works" for Underdog and
"May also 403 Cloudflare" for PrizePicks. Checked against the tree, and then
against every commit in the repository with `git log --all --diff-filter=A`:
NONE OF THEM HAS EVER EXISTED. The document was a design written as a
description.

The cost was not theoretical. It produced a request to debug four failing
scrapers, three of which are not there -- a root-cause analysis of code that
was never written. The repository already fixed this exact defect once by
deleting a source precedence that had no client (commit `6d9a130`, "delete
OddsPapi"), and `tests/test_agents_md.py` guards AGENTS.md the same way.

THE RULE. Every repository path a document names must either be in the tree,
or be marked planned on the line that names it. Not "documented somewhere as
planned" -- marked where a reader's eye lands.

RESEARCH_ONLY. This test reads Markdown and calls `Path.exists`.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Documents under this rule. Deliberately a list rather than a glob: adding a
#: document here is a decision, and a glob would make every new note a
#: tripwire for the person writing it.
GUARDED_DOCS = ("docs/pickem_props.md",)

#: A backticked path into this repository: `src/...py`, `scripts/...py`,
#: `tests/...py`, `config/....yaml`, or a directory under `data/`.
PATH_IN_BACKTICKS = re.compile(
    r"`((?:src|scripts|tests|config|migrations|docs|data)/[A-Za-z0-9_./-]+)`"
)

#: What marks a path as not-yet-written. Checked case-insensitively on the
#: same LINE as the path, because that is where a reader sees it.
PLANNED_MARKERS = ("planned", "not built", "does not exist", "none of these exist")


def _cited_paths(doc: Path) -> list[tuple[int, str, str]]:
    out: list[tuple[int, str, str]] = []
    for number, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1):
        for match in PATH_IN_BACKTICKS.finditer(line):
            out.append((number, match.group(1), line))
    return out


@pytest.mark.parametrize("doc_path", GUARDED_DOCS)
def test_every_path_a_doc_names_exists_or_is_marked_planned(doc_path):
    doc = REPO_ROOT / doc_path
    assert doc.is_file(), f"{doc_path} is itself missing"

    unmarked: list[str] = []
    for number, cited, line in _cited_paths(doc):
        if (REPO_ROOT / cited).exists():
            continue
        if any(marker in line.lower() for marker in PLANNED_MARKERS):
            continue
        unmarked.append(f"  {doc_path}:{number} names {cited}, which is not in the tree")

    assert not unmarked, (
        "a document describes code that does not exist, in the present tense:\n"
        + "\n".join(unmarked)
        + "\n\nEither write the module or mark the line planned. A reader who "
        "believes this document will report bugs against files that were "
        "never written -- which is exactly what happened on 2026-10-18."
    )


def test_the_guard_actually_catches_an_unmarked_path(tmp_path):
    """
    The guard above passes trivially if the regex matches nothing. This proves
    it fires -- and that the planned marker is what silences it, not the
    absence of a match.
    """
    doc = tmp_path / "sample.md"
    doc.write_text(
        "We ingest from `src/ingestion/nowhere_props.py` nightly.\n",
        encoding="utf-8",
    )
    cited = _cited_paths(doc)
    assert cited, "the regex did not match a backticked repository path"
    number, path, line = cited[0]
    assert path == "src/ingestion/nowhere_props.py"
    assert not (REPO_ROOT / path).exists()
    assert not any(m in line.lower() for m in PLANNED_MARKERS)

    doc.write_text(
        "We will ingest from `src/ingestion/nowhere_props.py` (planned).\n",
        encoding="utf-8",
    )
    _, _, marked = _cited_paths(doc)[0]
    assert any(m in marked.lower() for m in PLANNED_MARKERS)


def test_the_pickem_capture_modules_that_are_still_absent_stay_documented_as_such():
    """
    A canary, not a wish, and it has already fired once: on 2026-10-18 the
    Underdog and Sleeper clients were built, this test failed listing all six
    new modules, and `docs/pickem_props.md` was rewritten from NOT BUILT to
    BUILT-BUT-UNCONFIGURED. That is the direction this defect needs catching
    in. Delete the entry here when you write the module, not the test.

    What remains unwritten, and why:
      prizepicks_props  -- its board was expected behind a bot wall and
                           getting past one is banned; its lines arrive via
                           PropLine's DFS_BOOKS instead.
      sleeper (metadata) -- player metadata and injuries. Injuries come from
                           ESPN today (`src/ingestion/espn_availability.py`),
                           so this has no caller waiting on it.
    """
    planned = [
        "src/ingestion/prizepicks_props.py",
        "src/ingestion/sleeper.py",
    ]
    built = [p for p in planned if (REPO_ROOT / p).exists()]
    assert not built, (
        f"{built} now exist(s). Good -- update docs/pickem_props.md to describe "
        f"what it really does and remove it from this list."
    )


def test_the_capture_subsystem_that_was_built_is_all_present():
    """
    The other direction, which the original canary could not express: these
    were written together and a half-deleted subsystem would leave the
    document describing a client whose batch or store had gone.
    """
    built = [
        "src/ingestion/pickem_schema.py",
        "src/ingestion/pickem_board.py",
        "src/ingestion/pickem_store.py",
        "src/ingestion/pickem.py",
        "src/ingestion/underdog_props.py",
        "src/ingestion/sleeper_props.py",
        "scripts/pull_pickem_props.py",
        "config/pickem_sources.yaml",
    ]
    missing = [p for p in built if not (REPO_ROOT / p).exists()]
    assert not missing, f"the pick'em capture subsystem is missing {missing}"


def test_the_consumer_side_does_exist_and_has_nothing_to_read():
    """
    The asymmetry that made the document believable: the thing that CONSUMES
    pick'em rows was built, so the subsystem looked half-present rather than
    absent.
    """
    assert (REPO_ROOT / "src/quant/line_diff.py").exists()
    source = (REPO_ROOT / "src/quant/line_diff.py").read_text(encoding="utf-8")
    assert "pickem_vs_book_line_diff" in source
