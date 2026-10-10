"""
tests/test_deploy_audit_citations.py — the audits must keep resolving.

WHY THIS EXISTS. The recurring defect in this repository is not wrong code, it
is **documentation outliving its facts**: `dvp.py`'s docstring described a
behaviour it no longer had, the builder claimed "layers only add", the deploy
page's migration list named 002-005 and went stale when 006 landed, and my own
Railway audit asserted the Dockerfile did not set `PROPIQ_PARLAY_LEDGER` when
line 73 did. Each was true when written.

`tests/test_agents_md.py` solved this for `AGENTS.md` by checking that every
citation resolves. The two audits written on 2026-10-09 cite thirty-odd exact
line numbers, which is the most perishable kind of claim there is — any edit
above a cited line moves it. So the same guard applies here:

  * every `` `path:line` `` and `` `path:lo-hi` `` citation names a file that
    exists and a range inside it;
  * and for the findings whose POINT is what sits at that line, an anchor
    string must still be there.

The anchor table is deliberately short. Checking every citation's content
would mean restating the audits in Python, and a test that duplicates the
document it guards drifts from it the same way.

RESEARCH_ONLY project. This test reads files.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

AUDITS = (
    "docs/automation_audit_2026-10-09.md",
    "docs/configuration_audit_2026-10-09.md",
    "docs/autonomy_audit_2026-10-10.md",
)

#: `path:12` or `path:12-34` inside backticks. The path must look like a real
#: one (a slash or a known root file) so prose such as `09:00` is not matched.
CITATION = re.compile(
    r"`((?:[A-Za-z0-9_./-]+/)?[A-Za-z0-9_.-]+\.(?:py|md|toml|txt|sql|yaml|yml)"
    r"|Dockerfile|Procfile):(\d+)(?:-(\d+))?`"
)

#: The citations whose CONTENT is the finding. (path, line, substring).
ANCHORS = [
    ("Dockerfile", 39, "python:3.11-slim"),
    ("Dockerfile", 54, "TZ=Etc/UTC"),
    ("docs/external_feature_harvest.md", 90, "ODDS_API_KEY"),
    ("docs/pickem_props.md", 61, "ODDS_API_KEY"),
    ("src/models/xgb_adapter.py", 390, "_VERSIONS_KEY"),
    ("src/models/xgb_adapter.py", 417, "warn_on_mismatch"),
]


def _citations() -> list[tuple[str, str, int, int]]:
    out: list[tuple[str, str, int, int]] = []
    for audit in AUDITS:
        body = (REPO / audit).read_text(encoding="utf-8")
        for m in CITATION.finditer(body):
            lo = int(m.group(2))
            hi = int(m.group(3)) if m.group(3) else lo
            out.append((audit, m.group(1), lo, hi))
    return out


def test_the_audits_cite_something():
    """A regex that matches nothing would make every test below vacuous."""
    found = _citations()
    assert len(found) >= 10, f"only {len(found)} citations parsed: {found}"


@pytest.mark.parametrize("audit,path,lo,hi", _citations())
def test_every_cited_line_exists(audit: str, path: str, lo: int, hi: int):
    target = REPO / path
    assert target.is_file(), f"{audit} cites {path}, which is not in the tree"
    count = len(target.read_text(encoding="utf-8").splitlines())
    assert 1 <= lo <= hi <= count, (
        f"{audit} cites {path}:{lo}-{hi} and the file has {count} lines"
    )


@pytest.mark.parametrize("path,line,needle", ANCHORS)
def test_the_findings_still_sit_where_they_are_cited(path: str, line: int, needle: str):
    body = (REPO / path).read_text(encoding="utf-8").splitlines()
    assert needle in body[line - 1], (
        f"an audit cites {path}:{line} for {needle!r}; that line now reads "
        f"{body[line - 1].strip()!r}. Either the finding moved or it is gone — "
        f"update the audit rather than this test."
    )


def test_every_file_the_audits_name_is_in_the_tree():
    """
    A path that no longer exists is the same defect as a stale line number and
    easier to introduce — the audits name three dozen modules and scripts, and
    a rename moves them all at once.
    """
    import re

    # A NAMED FILE has a directory, or is one of the known root files. A bare
    # suffix discussed in prose -- `.meta.json`, `.mean.json` -- is neither,
    # and flagging those sent this test looking for a file nobody claimed
    # existed. The citation regex above already makes this distinction; this
    # one did not.
    pat = re.compile(
        r"`((?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+"
        r"\.(?:py|md|yaml|yml|toml|txt|sh|json|sql)"
        r"|Dockerfile|Procfile|railway\.json|requirements\.txt|pyproject\.toml)`"
    )
    for audit in AUDITS:
        named = set(pat.findall((REPO / audit).read_text(encoding="utf-8")))
        assert named, f"{audit} names no files, so this test proves nothing"
        missing = sorted(n for n in named if not (REPO / n).exists())
        assert not missing, f"{audit} names paths that do not exist: {missing}"


def test_the_audits_do_not_claim_a_banned_source_is_required():
    """
    A standing constraint, asserted on the deliverables that are most likely to
    break it: these documents enumerate environment variables, and the Odds API
    is a BANNED sportsbook source here. Listing it as required would wire an
    operator's expectations to a source this project has decided against.
    """
    for audit in AUDITS:
        body = (REPO / audit).read_text(encoding="utf-8")
        for banned in ("SPORTSDATA_API_KEY", "THE_ODDS_API_KEY"):
            if banned in body:
                # It may only appear while being REFUSED.
                for para in body.split("\n"):
                    if banned in para:
                        assert re.search(
                            r"not added|not here|banned|appears nowhere|Verdict",
                            para, re.I,
                        ), f"{audit} mentions {banned} without refusing it: {para}"
