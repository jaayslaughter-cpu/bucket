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

# These two were open items on this page until 2026-10-04. The guard above
# fired when they were fixed, which is what it is for; they are now pinned the
# other way round, as closures the page claims and the tree must keep.


def test_the_projection_table_carries_all_three_legs():
    """
    Was O8. The page's "Closed since this page was reconciled" table claims
    these columns exist and that the writer fills them.
    """
    models = (ROOT / "src" / "db" / "models.py").read_text(encoding="utf-8")
    block = models[models.index("class Projection"):]
    block = block[: block.index("\nclass ")]
    for column in ("prob_over", "prob_under", "prob_push"):
        assert column in block, f"the page says {column} exists on Projection"
    assert "migrations/005_projection_under_push.sql" in TEXT
    assert (ROOT / "migrations" / "005_projection_under_push.sql").is_file()


def test_the_results_card_exists_and_the_page_says_how_it_is_sent():
    """
    Was O9. Both halves matter: a builder nothing calls on a schedule is the
    shape the gap was in to begin with.
    """
    body = (ROOT / "src" / "notify" / "discord.py").read_text(encoding="utf-8")
    builders = re.findall(r"^def (build_\w+_embed)", body, re.M)
    assert "build_win_loss_embed" in builders, "the page claims a fifth builder"
    assert len(builders) == 5, (
        f"the page says five builders; there are {len(builders)}: {builders}"
    )
    worker = (ROOT / "scheduler_worker.py").read_text(encoding="utf-8")
    settle = worker[worker.index("def run_settlement"):]
    settle = settle[: settle.index("\ndef ")]
    assert "run_results_card()" in settle, (
        "the page says the settlement job sends it, and nothing calls it"
    )


def test_the_page_records_both_closures_rather_than_deleting_the_items():
    assert "Closed since this page was reconciled" in TEXT
    assert "prob_under" in TEXT and "build_win_loss_embed" in TEXT


def test_o8_nothing_populates_the_tipoff_column():
    assert "tipoff_utc" in (ROOT / "src" / "db" / "models.py").read_text(encoding="utf-8")
    writers = []
    for path in list((ROOT / "src").rglob("*.py")) + [ROOT / "main.py"]:
        if "__pycache__" in str(path) or path.name in {"models.py", "espn_schedule.py"}:
            continue
        if re.search(r"tipoff_utc\s*=|\[.tipoff_utc.\]\s*=", path.read_text(encoding="utf-8")):
            writers.append(path.relative_to(ROOT).as_posix())
    assert not writers, (
        f"O8 is fixed — {writers} writes tipoff_utc. Update "
        "docs/go_live_readiness.md."
    )


def test_the_forward_slate_is_wired_into_the_run():
    """
    Was the first open item on this page. The panel is completed box scores, so
    without this a 09:00 PT run has no rows for tonight.
    """
    assert (ROOT / "src" / "pipeline" / "forward_slate.py").is_file()
    body = (ROOT / "main.py").read_text(encoding="utf-8")
    assert "attach_forward_slate" in body, (
        "the page says the forward slate is wired, and main.py does not use it"
    )
    assert "PROPIQ_FORWARD_SLATE" in body


def test_fetch_roster_is_still_uncalled_and_the_page_says_why():
    """
    The page claims this is now DELIBERATE: a roster fetch needs the ESPN-name
    crosswalk that does not exist. If something starts calling it, the page's
    reasoning needs revisiting rather than silently going stale.
    """
    # A CALL or an IMPORT, not the word: forward_slate.py names it in prose to
    # explain why it does NOT use it, and an earlier version of this check
    # counted that explanation as a caller.
    callers = []
    for path in list((ROOT / "src").rglob("*.py")) + [ROOT / "main.py"]:
        if "__pycache__" in str(path) or path.name == "espn_availability.py":
            continue
        body = path.read_text(encoding="utf-8")
        if re.search(r"fetch_roster\s*\(|import[^\n]*\bfetch_roster\b", body):
            callers.append(path.relative_to(ROOT).as_posix())
    assert not callers, (
        f"{callers} now calls fetch_roster — update the crosswalk reasoning in "
        "docs/go_live_readiness.md"
    )
    assert "fetch_roster" in TEXT


def test_the_model_resolver_reaches_the_scheduled_worker():
    """
    Was the second open item. The worker calls main.main([]) with no arguments,
    so the environment is the only channel — which is why the fix is a resolver
    reading PROPIQ_MODEL rather than argv plumbing.
    """
    import main

    assert hasattr(main, "resolve_model_artifact")
    assert main.ENV_MODEL == "PROPIQ_MODEL"
    assert "PROPIQ_MODEL" in TEXT
    path, reason = main.resolve_model_artifact()
    assert path is not None or "PROPIQ_MODEL" in reason, (
        "with nothing resolvable, the reason must name the paths it tried"
    )


def test_the_open_list_is_numbered_without_gaps():
    """
    Renumbering on a closure is easy to get wrong, and a gap or a duplicate
    makes every cross-reference to an O-number ambiguous.
    """
    found = sorted(int(n) for n in set(re.findall(r"\*\*O(\d+)\*\*", TEXT)))
    assert found == list(range(1, len(found) + 1)), f"O-numbers have gaps: {found}"
    rows = len(re.findall(r"^\| \*\*O\d+\*\* \|", TEXT, re.M))
    assert rows == len(found), (
        f"{rows} table row(s) but {len(found)} distinct O-number(s) — a "
        "reference in the prose points at an item that is not in the table"
    )
    assert found, "the open list is empty; if that is true, say so in prose"
