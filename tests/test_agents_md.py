"""AGENTS.md has to stay true, or it is worse than nothing.

A working agreement that cites a file which does not exist, or a line that has
moved, teaches the wrong map — which is exactly why the uploaded pack's version
of this file was not adopted (docs/go_live_pack_review.md section 4). So every
path it names is resolved here, every `file:line` citation is checked against
the line it points at, and the one architectural exception it documents is
counted, so a third instance makes this test fail rather than making the file
quietly wrong.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
AGENTS = ROOT / "AGENTS.md"
TEXT = AGENTS.read_text(encoding="utf-8")

#: (path, line number, text that must be on that line). Each was read off the
#: file when AGENTS.md was written; this is what keeps it honest as code moves.
CITATIONS = [
    ("src/quant/advisory_sizing.py", 114, '"AUTO_PLACED": False'),
    ("src/quant/contracts.py", 187, 'context.status != "VALID"'),
    ("src/quant/publication_gate.py", 150, "def calibration_gate"),
    ("src/features/builder.py", 665, "def assert_no_lookahead"),
    ("src/features/builder.py", 121, "from src.models.compare import load_comparison_config"),
    ("src/features/fatigue_load.py", 284, "from src.models.compare import load_comparison_config"),
    ("tests/test_publication_gate.py", 66, "for forbidden in"),
    ("main.py", 4, "NBA ONLY"),
    ("main.py", 241, "def load_master_guideline"),
    ("src/ingestion/propline.py", 15, "refused here"),
    ("src/utils/timezones.py", 11, 'DISPLAY_TZ_NAME = "America/Los_Angeles"'),
    ("src/utils/timezones.py", 12, 'STORAGE_TZ_NAME = "UTC"'),
]


def test_the_file_is_there_at_all():
    assert AGENTS.is_file()
    assert "RESEARCH_ONLY" in TEXT


@pytest.mark.parametrize("path,line,needle", CITATIONS)
def test_every_line_citation_still_points_at_what_it_claims(path, line, needle):
    target = ROOT / path
    assert target.is_file(), f"AGENTS.md cites {path}, which is not in the tree"
    lines = target.read_text(encoding="utf-8").splitlines()
    assert line <= len(lines), f"{path} has {len(lines)} lines; AGENTS.md cites {line}"
    assert needle in lines[line - 1], (
        f"{path}:{line} now reads {lines[line - 1].strip()!r}; "
        f"AGENTS.md says it contains {needle!r}"
    )


def _cited_in_prose(path: str, line: int) -> bool:
    """A citation may be written `file.py:97` or as a range `file.py:11-12`."""
    stem = path.split("/")[-1]
    for name in (path, stem):
        if f"{name}:{line}" in TEXT:
            return True
        for lo, hi in re.findall(rf"{re.escape(name)}:(\d+)-(\d+)", TEXT):
            if int(lo) <= line <= int(hi):
                return True
    return False


@pytest.mark.parametrize("path,line,_needle", CITATIONS)
def test_every_cited_line_number_appears_in_the_file(path, line, _needle):
    """Guards the other direction: a citation dropped from the prose but left
    in this table would otherwise go on passing."""
    assert _cited_in_prose(path, line), (
        f"{path}:{line} is in the citation table but no longer in AGENTS.md"
    )


#: Paths AGENTS.md names BECAUSE they are absent. Each must be described as
#: absent in the prose, which the test below checks rather than trusts.
DOCUMENTED_AS_ABSENT = {"config/master_guideline_props.yaml"}


def test_every_repository_path_it_names_exists():
    """
    Catches the failure that kept the pack's version out: six docs and a module
    that were not here.
    """
    candidates = set(
        re.findall(r"\b((?:src|tests|scripts|docs|config|migrations)/[\w./*-]+)", TEXT)
    )
    missing = sorted(
        p.rstrip(".,)") for p in candidates
        if "*" not in p
        and p.rstrip(".,)") not in DOCUMENTED_AS_ABSENT
        and not (ROOT / p.rstrip(".,)")).exists()
    )
    assert not missing, f"AGENTS.md names paths that do not exist: {missing}"


@pytest.mark.parametrize("path", sorted(DOCUMENTED_AS_ABSENT))
def test_a_path_exempted_as_absent_is_said_to_be_absent(path):
    """
    The exemption list is the one way a nonexistent path can appear in
    AGENTS.md, so it may not be a quiet escape hatch: the prose has to say the
    file is not here, and if the file ever lands the exemption must go.
    """
    assert path in TEXT
    assert not (ROOT / path).exists(), (
        f"{path} now exists — drop it from DOCUMENTED_AS_ABSENT and from the "
        f"'not in the repo' wording in AGENTS.md"
    )
    where = TEXT.index(path)
    nearby = TEXT[where : where + 220]
    assert "not in the repo" in nearby, (
        f"AGENTS.md names {path} without saying it is absent"
    )


def test_the_root_entrypoints_it_names_exist():
    for name in ("main.py", "scheduler_worker.py", "AGENTS.md", ".env.example",
                 ".dockerignore", "Dockerfile", "pyproject.toml"):
        assert name in TEXT, f"AGENTS.md no longer mentions {name}"
        assert (ROOT / name).exists(), f"AGENTS.md names {name}, which is absent"


# --- the claims, checked against the code rather than restated --------------

def test_no_code_path_marks_a_size_as_placed():
    """The first hard line in the file. If this ever flips, the file is lying."""
    body = (ROOT / "src" / "quant" / "advisory_sizing.py").read_text(encoding="utf-8")
    assert '"AUTO_PLACED": False' in body
    assert '"AUTO_PLACED": True' not in body


def test_the_claim_words_are_really_banned_somewhere_that_fails():
    body = (ROOT / "tests" / "test_publication_gate.py").read_text(encoding="utf-8")
    for word in ("guaranteed", "profitable", "lock", "certain"):
        assert f'"{word}"' in body, f"AGENTS.md says {word!r} is banned by a test"


def test_the_layering_exception_is_exactly_the_two_sites_it_documents():
    """
    AGENTS.md states the dependency direction and then names the only two
    places that break it. A third would make the file wrong, so it fails here
    instead — and if one is removed, the file should stop claiming it.
    """
    sites = []
    for path in sorted((ROOT / "src" / "features").rglob("*.py")):
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"(from|import)\s+src\.(models|quant)\b", line):
                sites.append((path.relative_to(ROOT).as_posix(), i))
    assert sites == [
        ("src/features/builder.py", 121),
        ("src/features/fatigue_load.py", 284),
    ], f"the features layer's imports of models/quant have changed: {sites}"


def test_the_two_exceptions_are_lazy_imports_of_a_config_reader():
    """The exception is narrow on purpose: a config loader, imported inside a
    function so it cannot create an import cycle, and never a fitter."""
    for path, line in (("src/features/builder.py", 121),
                       ("src/features/fatigue_load.py", 284)):
        raw = (ROOT / path).read_text(encoding="utf-8").splitlines()[line - 1]
        assert raw.startswith("        ") or raw.startswith("    "), (
            f"{path}:{line} is a module-level import of src.models"
        )
        assert "load_comparison_config" in raw


def test_it_does_not_invite_a_sport_the_project_refuses():
    """Scope is NBA only, so this file must not read as an NCAA roadmap."""
    lowered = TEXT.lower()
    assert "ncaa" in lowered, "the exclusion should be stated, not silent"
    for invitation in ("add ncaa", "ncaa support", "widen to ncaa", "implement ncaa"):
        assert invitation not in lowered


def test_it_states_what_is_not_true_rather_than_only_what_works():
    """
    The section that stops an ambition being read as a fact. Its absence is how
    a working agreement turns into marketing.
    """
    assert "What is still not true" in TEXT
    for claim in (
        "No model is profitable",
        "never been built",
        "do not survive a redeploy",
        "ProbabilitySource.MODEL",
    ):
        assert claim in TEXT, f"the honesty section no longer says: {claim}"


# --- the wiring: the file has to be reachable by what should read it -------

def test_claude_md_points_at_the_agreement_rather_than_copying_it():
    """
    Claude Code loads CLAUDE.md by name, so that file is how AGENTS.md gets
    read at all. It must POINT, not restate: two copies of a working agreement
    drift, and then neither is trustworthy.
    """
    claude = ROOT / "CLAUDE.md"
    assert claude.is_file(), "AGENTS.md is not wired to anything that loads it"
    body = claude.read_text(encoding="utf-8")
    assert "AGENTS.md" in body
    assert len(body.splitlines()) < 40, (
        "CLAUDE.md is growing into a second agreement; put the rule in "
        "AGENTS.md and cite its enforcement point"
    )


def test_claude_md_does_not_restate_the_hard_lines():
    """A duplicated table is the drift. The pointer may summarise habits, not
    re-specify the rules or their enforcement points."""
    body = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    for owned_by_agents in ("AUTO_PLACED", "calibration_gate", "MarketContext",
                            "PROPLINE_API_KEY", "DATA_NOT_AVAILABLE"):
        assert owned_by_agents not in body, (
            f"CLAUDE.md restates {owned_by_agents}, which AGENTS.md owns"
        )


def test_the_settlement_readme_no_longer_reads_as_an_ncaa_roadmap():
    """
    It carried a section describing how to add an NCAA box-score fetcher, which
    is a plan for something this project refuses to do.
    """
    body = (ROOT / "src" / "settlement" / "README.md").read_text(encoding="utf-8")
    assert "ncaa_boxscore_fetcher" not in body
    assert "Out of scope" in body


def test_it_does_not_claim_a_web_ui():
    assert "not a Streamlit app" in TEXT
    assert "binds no port" in TEXT
