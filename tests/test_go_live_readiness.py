"""`docs/go_live_readiness.md` has to keep agreeing with the tree.

That page said scheduling was ABSENT and `prop_results` had no writer. Both
were closed by `8ed808c`, dated the SAME DAY as the audit — so it was overtaken
within hours and sat wrong for a week, and an external reviewer following it
produced a list of P0s that had already been built. A readiness page that
disagrees with the repository is worse than no page: it is work misdirected
with a citation attached.

So the page is pinned here, in both directions:

  - what it claims is FIXED must still be fixed;
  - what it lists as OPEN must still be open — and when one is fixed, THIS
    TEST FAILS ON PURPOSE, with a message saying to update the page. That
    failure is the mechanism, not an inconvenience.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "go_live_readiness.md"
TEXT = DOC.read_text(encoding="utf-8")


def test_the_page_exists_and_says_when_it_was_reconciled():
    assert DOC.is_file()
    assert "reconciled 2026-10-04" in TEXT
    assert "RESEARCH_ONLY" in TEXT


def test_it_tells_the_reader_to_run_the_checker_rather_than_trust_it():
    """The page's own staleness is the reason the checker exists."""
    assert "scripts/verify_wiring" in TEXT


def test_every_doc_it_points_at_exists():
    cited = set(re.findall(r"`(docs/[\w./-]+\.md)`", TEXT))
    missing = sorted(p for p in cited if not (ROOT / p).exists())
    assert not missing, f"the page points at pages that do not exist: {missing}"


# --- the commit attributions, which a first draft got wrong -----------------

def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=60
    ).stdout.strip()


def _git_available() -> bool:
    try:
        return bool(_git("rev-parse", "--git-dir"))
    except (OSError, subprocess.SubprocessError):
        return False


def _claimed_attributions() -> list[tuple[str, str]]:
    """
    PARSE THE PAGE, do not keep a second copy of it.

    A first version of this test held its own hardcoded (sha, path) list and
    only checked that the sha appeared *somewhere* in the page — so swapping
    one attribution for another sha that is also cited elsewhere passed. It
    measured nothing. These pairs are now read out of the sentences that make
    the claim: "`<sha>` added `<path>`".
    """
    return [
        (sha, path)
        for sha, path in re.findall(r"`([0-9a-f]{7})` added `([\w./-]+)`", TEXT)
    ]


@pytest.mark.skipif(not _git_available(), reason="no git in this environment")
def test_the_page_makes_checkable_attribution_claims():
    claims = _claimed_attributions()
    assert len(claims) >= 5, (
        f"only {len(claims)} '<sha> added <path>' claims found; the commit table "
        "must stay in that form or this check stops looking"
    )


@pytest.mark.skipif(not _git_available(), reason="no git in this environment")
@pytest.mark.parametrize("sha,path", _claimed_attributions())
def test_each_cited_commit_really_added_what_the_page_credits_it_with(sha, path):
    assert (ROOT / path).exists(), f"the page credits a commit with adding {path}, which is gone"
    creators = _git("log", "--diff-filter=A", "--format=%h", "--", path).split()
    assert creators, f"{path} has no creating commit — was it renamed?"
    assert any(c.startswith(sha) or sha.startswith(c) for c in creators), (
        f"the page says {sha} added {path}, but git says {creators[-1]} did"
    )


@pytest.mark.skipif(not _git_available(), reason="no git in this environment")
def test_every_hash_on_the_page_resolves():
    for sha in set(re.findall(r"`([0-9a-f]{7})`", TEXT)):
        assert _git("cat-file", "-t", sha) == "commit", (
            f"the page cites {sha}, which is not a commit in this repository"
        )


# --- what it claims is fixed -----------------------------------------------

def test_32_prop_results_really_has_a_writer():
    """The original 'top blocker'. The page now calls it PASS."""
    repo = (ROOT / "src" / "db" / "repository.py").read_text(encoding="utf-8")
    assert "pg_insert(PropResult)" in repo
    assert (ROOT / "src" / "settlement" / "recorder.py").is_file()
    assert "pending_prop_result_rows" in (ROOT / "main.py").read_text(encoding="utf-8")


def test_32_parlays_really_reach_postgres():
    models = (ROOT / "src" / "db" / "models.py").read_text(encoding="utf-8")
    assert "class ParlayTicketRow" in models
    assert "class ParlayLegRow" in models
    assert (ROOT / "migrations" / "004_parlay_ledger.sql").is_file()


def test_41_scheduling_is_not_absent():
    for path in ("Dockerfile", ".dockerignore", "scheduler_worker.py"):
        assert (ROOT / path).exists(), f"the page calls 4.1 PASS but {path} is gone"
    for deps in ("requirements.txt", "pyproject.toml"):
        body = (ROOT / deps).read_text(encoding="utf-8")
        assert "APScheduler" in body, f"APScheduler is missing from {deps}"


def test_11_the_scratch_filter_is_wired_into_the_slate():
    assert "apply_scratch_filter" in (ROOT / "main.py").read_text(encoding="utf-8")


def test_13_the_feature_contract_is_checked_at_serve_time():
    assert "verify_feature_contract" in (ROOT / "main.py").read_text(encoding="utf-8")


def test_43_the_pooling_claims_are_still_true():
    body = (ROOT / "src" / "db" / "session.py").read_text(encoding="utf-8")
    for claim in ("pool_pre_ping=True", "pool_size=5", "max_overflow=5",
                  "expire_on_commit=False", "sslmode=require"):
        assert claim in body, f"the page calls 4.3 PASS but {claim} is gone"


def test_the_two_corrections_are_kept_rather_than_quietly_dropped():
    assert "wrong when written" in TEXT
    assert "_matrix" in TEXT, "the FeatureSpec order/dtype correction is gone"
    assert "SOURCE_PRECEDENCE" in TEXT, "the OddsPapi correction is gone"


# --- what it lists as open. FAILING HERE MEANS UPDATE THE PAGE. ------------

def test_o8_projection_still_stores_only_prob_over():
    models = (ROOT / "src" / "db" / "models.py").read_text(encoding="utf-8")
    block = models[models.index("class Projection"):]
    block = block[: block.index("\nclass ")]
    assert "prob_over" in block
    for closed in ("prob_under", "prob_push"):
        assert closed not in block, (
            f"O8 is fixed — Projection now stores {closed}. Update "
            "docs/go_live_readiness.md: move O8 out of the open list and into "
            "the verdict table."
        )


def test_o9_there_is_still_no_daily_win_loss_embed():
    body = (ROOT / "src" / "notify" / "discord.py").read_text(encoding="utf-8")
    builders = re.findall(r"^def (build_\w+_embed)", body, re.M)
    assert builders, "no embed builders found at all"
    wl = [b for b in builders if any(k in b for k in ("win_loss", "wl_", "record"))]
    assert not wl, (
        f"O9 is fixed — {wl} exists. Update docs/go_live_readiness.md."
    )
    assert len(builders) == 4, (
        f"the page says four builders; there are now {len(builders)}: {builders}. "
        "Update the 4.2 row."
    )


def test_o10_nothing_populates_the_tipoff_column():
    assert "tipoff_utc" in (ROOT / "src" / "db" / "models.py").read_text(encoding="utf-8")
    writers = []
    for path in list((ROOT / "src").rglob("*.py")) + [ROOT / "main.py"]:
        if "__pycache__" in str(path) or path.name in {"models.py", "espn_schedule.py"}:
            continue
        if re.search(r"tipoff_utc\s*=|\[.tipoff_utc.\]\s*=", path.read_text(encoding="utf-8")):
            writers.append(path.relative_to(ROOT).as_posix())
    assert not writers, (
        f"O10 is fixed — {writers} writes tipoff_utc. Update "
        "docs/go_live_readiness.md."
    )


def test_o11_the_schedule_and_roster_sources_are_still_unwired():
    assert (ROOT / "src" / "ingestion" / "espn_schedule.py").is_file()
    assert "def fetch_roster" in (
        ROOT / "src" / "ingestion" / "espn_availability.py"
    ).read_text(encoding="utf-8")
    for entry in ("main.py", "scheduler_worker.py"):
        body = (ROOT / entry).read_text(encoding="utf-8")
        assert "espn_schedule" not in body, (
            f"O11 is fixed — {entry} imports espn_schedule. Update "
            "docs/go_live_readiness.md and docs/integration_audit.md."
        )


def test_o2_the_model_path_mismatch_is_still_there():
    """
    The worker passes no --model and nothing overrides the path, so a scheduled
    run reads main.MODEL_ARTIFACT_DEFAULT whatever was trained.
    """
    worker = (ROOT / "scheduler_worker.py").read_text(encoding="utf-8")
    assert "--model" not in worker, (
        "O2 is fixed — the worker now passes --model. Update the page."
    )
    assert "PROPIQ_MODEL" not in worker, (
        "O2 is fixed — an env override exists now. Update the page."
    )


def test_the_open_list_is_numbered_without_gaps():
    found = sorted(int(n) for n in set(re.findall(r"\*\*O(\d+)\*\*", TEXT)))
    assert found == list(range(1, len(found) + 1)), f"O-numbers have gaps: {found}"
    assert len(found) >= 10
