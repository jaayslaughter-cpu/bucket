"""The Docker validator's own checks, made to fail.

`scripts/validate_docker.py` is the only thing standing between a reasoned
Dockerfile and a deploy, and a check that cannot fail is worse than no check:
it reports PASS either way. Every check it makes is driven here against a
synthetic tree built to break exactly one of them.

The dockerignore matcher gets its own tests because a wrong answer there is
specifically a wrong answer about whether a credential ships in an image layer.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from scripts.validate_docker import (
    EXPECTED_UID,
    Report,
    _ignore_rules,
    _is_ignored,
    daemon_state,
    preflight,
)

ROOT = Path(__file__).resolve().parents[1]

GOOD_DOCKERFILE = """FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1
ENV TZ=Etc/UTC
WORKDIR /app
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir -e ".[ml,db,deploy]"
RUN python -c "import xgboost, catboost, sklearn; print('ml extra OK')"
COPY . .
RUN useradd --create-home --uid 10001 propiq && chown -R propiq:propiq /app
USER propiq
CMD ["python", "scheduler_worker.py"]
"""

GOOD_IGNORE = """# comment
.env
.env.*
!.env.example
*.pem
*.key
secrets/
data/
outputs/
catboost_info/
*.zip
"""

GOOD_PYPROJECT = """[project]
name = "x"
[project.optional-dependencies]
ml = ["xgboost"]
db = ["sqlalchemy"]
deploy = ["apscheduler"]
"""


def tree(tmp_path: Path, *, dockerfile=GOOD_DOCKERFILE, ignore=GOOD_IGNORE,
         pyproject=GOOD_PYPROJECT, entrypoint="scheduler_worker.py") -> Path:
    (tmp_path / "Dockerfile").write_text(dockerfile, encoding="utf-8")
    (tmp_path / ".dockerignore").write_text(ignore, encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(pyproject, encoding="utf-8")
    if entrypoint:
        (tmp_path / entrypoint).write_text("", encoding="utf-8")
    return tmp_path


def run(root: Path) -> Report:
    report = Report()
    preflight(report, root)
    return report


def failures(report: Report) -> list[str]:
    return [name for name, _ in report.failed]


# --- the control: a sound tree passes, so the failures below mean something --

#: The one line that now guarantees the ML extra resolved and OpenMP loads.
IMPORT_CHECK_LINE = (
    'RUN python -c "import xgboost, catboost, sklearn; print(\'ml extra OK\')"'
)


def test_a_dockerfile_with_no_build_time_import_check_fails(tmp_path):
    """
    THE GUARANTEE THAT REPLACED THE apt LAYER. The Dockerfile used to
    apt-install libgomp1, justified by "without it the image builds and then
    fails at import". The first real build (2026-10-10) disproved that -- the
    xgboost wheel vendors its own libgomp -- and the layer was removed, because
    it installed a library nothing loaded and was the only thing in the build
    needing the Debian package index.

    What took its place is that import layer, and it is now the ONLY thing
    asserting at build time that the ML extra resolved and OpenMP loads. A
    guarantee resting on one line nobody checks is one edit from being gone.
    """
    assert IMPORT_CHECK_LINE in GOOD_DOCKERFILE, "the fixture no longer has it"
    root = tree(tmp_path, dockerfile=GOOD_DOCKERFILE.replace(
        IMPORT_CHECK_LINE + "\n", ""))
    assert "the ML extra is import-checked at build time" in failures(run(root))


def test_a_comment_naming_the_import_does_not_satisfy_the_check(tmp_path):
    """
    The trap this repository has hit five times: a source assertion satisfied
    by prose ABOUT the code. The check matches RUN instructions only, so the
    comment explaining why the layer exists cannot stand in for the layer.
    """
    root = tree(tmp_path, dockerfile=GOOD_DOCKERFILE.replace(
        IMPORT_CHECK_LINE, "# " + IMPORT_CHECK_LINE))
    assert "the ML extra is import-checked at build time" in failures(run(root))


def test_importing_only_one_of_the_two_ml_libraries_fails(tmp_path):
    """Both are installed by the `ml` extra and both are scored with, so
    checking one would leave the other to fail at the first inference."""
    root = tree(tmp_path, dockerfile=GOOD_DOCKERFILE.replace(
        "import xgboost, catboost, sklearn", "import xgboost"))
    assert "the ML extra is import-checked at build time" in failures(run(root))


def test_a_sound_tree_passes_every_check(tmp_path):
    report = run(tree(tmp_path))
    assert not report.failed, report.failed
    # EIGHT, not seven, since 2026-10-10: the build-time ML import check was
    # added when the libgomp1 apt layer was removed. The count is pinned so a
    # check that stops running cannot hide behind the ones that still do --
    # the names are asserted too, so a change here says WHICH check moved
    # rather than only that the arithmetic did.
    assert len(report.passed) == 8, sorted(report.passed)
    assert "the ML extra is import-checked at build time (['catboost', " \
           "'sklearn', 'xgboost'])" in report.passed


def test_this_repository_s_own_tree_passes(tmp_path):
    """The checker is pointed at the real Dockerfile, not only at fixtures."""
    report = run(ROOT)
    assert not report.failed, report.failed


# --- each check, made to fail -------------------------------------------------

def test_the_expected_uid_is_the_one_the_deploy_page_tells_you_to_chown_to():
    """
    docs/deploy_railway.md hands the operator a literal
    `chown -R 10001:10001 /app/data`. If the image's uid, the checker's
    constant and that instruction ever disagree, the instruction silently
    stops fixing the thing it is there to fix.
    """
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    page = (ROOT / "docs" / "deploy_railway.md").read_text(encoding="utf-8")
    assert f"--uid {EXPECTED_UID} " in dockerfile
    assert f"{EXPECTED_UID}:{EXPECTED_UID}" in page
    assert GOOD_DOCKERFILE.count(f"--uid {EXPECTED_UID} ") == 1, "fixture drifted"


def test_a_missing_dockerfile_is_not_a_pass(tmp_path):
    report = run(tmp_path)
    assert failures(report) == ["Dockerfile exists"]


def test_a_cmd_naming_a_script_that_is_not_there_fails(tmp_path):
    root = tree(tmp_path, entrypoint="")
    assert "CMD target exists" in failures(run(root))


def test_a_cmd_with_no_script_at_all_fails(tmp_path):
    root = tree(tmp_path, dockerfile=GOOD_DOCKERFILE.replace(
        'CMD ["python", "scheduler_worker.py"]', 'CMD ["bash"]'))
    assert "CMD names a script" in failures(run(root))


@pytest.mark.parametrize("line", [
    "ENV PROPLINE_API_KEY=sk-live-not-a-real-key",
    "ENV DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/1/2",
    "ENV DATABASE_URL=postgresql://u:p@h/db",
    "ARG GITHUB_TOKEN=ghp_x",
    "ENV ADMIN_PASSWORD=hunter2",
])
def test_a_credential_defaulted_in_the_image_fails(tmp_path, line):
    """
    A key in a layer survives every later layer that deletes it, and the image
    is pushed to a registry. This is the check that must not be decorative.
    """
    root = tree(tmp_path, dockerfile=GOOD_DOCKERFILE + line + "\n")
    assert "no credential defaulted in the image" in failures(run(root))


def test_a_credential_on_an_env_continuation_line_is_found(tmp_path):
    """
    `ENV A=1 \\` puts the next assignment on a line that does not start with
    ENV. Scanning line by line misses it, and this repo's own Dockerfile uses
    continuations, so that is the shape a real leak would take.
    """
    root = tree(tmp_path, dockerfile=GOOD_DOCKERFILE + (
        "ENV PYTHONUNBUFFERED=1 \\\n"
        "    PROPLINE_API_KEY=sk-live-not-a-real-key\n"
    ))
    assert "no credential defaulted in the image" in failures(run(root))


def test_the_project_s_own_non_secret_env_vars_are_not_flagged(tmp_path):
    """
    The real Dockerfile sets PROPIQ_CALIBRATION_REPORT, PROPIQ_PARLAY_LEDGER,
    PROPIQ_MAX_THREADS and TZ with values. A check that flagged those would be
    turned off within a week.
    """
    root = tree(tmp_path, dockerfile=GOOD_DOCKERFILE + (
        "ENV PROPIQ_CALIBRATION_REPORT=/app/data/calibration.json\n"
        "ENV PROPIQ_PARLAY_LEDGER=postgres\n"
        "ENV PROPIQ_MAX_THREADS=2\n"
        "ENV PIP_DISABLE_PIP_VERSION_CHECK=1\n"
    ))
    report = run(root)
    assert not report.failed, report.failed


def test_an_empty_env_default_is_not_a_credential(tmp_path):
    """`ENV DISCORD_WEBHOOK_URL=` declares the variable without a value."""
    root = tree(tmp_path, dockerfile=GOOD_DOCKERFILE + "ENV DISCORD_WEBHOOK_URL=\n")
    assert "no credential defaulted in the image" not in failures(run(root))


def test_installing_an_extra_pyproject_does_not_declare_fails(tmp_path):
    root = tree(tmp_path, pyproject=GOOD_PYPROJECT.replace('deploy = ["apscheduler"]', ""))
    assert "every installed extra is declared" in failures(run(root))


def test_installing_no_extra_fails(tmp_path):
    """The ML extra is installed on purpose; a minimal image fails at inference."""
    root = tree(tmp_path, dockerfile=GOOD_DOCKERFILE.replace(
        'RUN pip install --no-cache-dir -e ".[ml,db,deploy]"',
        "RUN pip install --no-cache-dir -r requirements.txt"))
    assert "Dockerfile installs an extra" in failures(run(root))


def test_dropping_the_ml_extra_fails(tmp_path):
    root = tree(tmp_path, dockerfile=GOOD_DOCKERFILE.replace('.[ml,db,deploy]', '.[db,deploy]'))
    assert "the ML and deploy extras are installed" in failures(run(root))


def test_a_user_line_without_a_pinned_uid_fails(tmp_path):
    root = tree(tmp_path, dockerfile=GOOD_DOCKERFILE.replace("--uid 10001 ", ""))
    assert "the non-root uid is pinned" in failures(run(root))


def test_a_uid_the_docs_do_not_name_fails(tmp_path):
    """
    docs/deploy_railway.md tells the operator to `chown -R 10001:10001` the
    volume. If the image's uid moves, that instruction silently stops working.
    """
    root = tree(tmp_path, dockerfile=GOOD_DOCKERFILE.replace("--uid 10001", "--uid 1000"))
    assert "the non-root uid matches the docs" in failures(run(root))


def test_no_user_line_is_reported_as_root_not_as_a_pass(tmp_path):
    root = tree(tmp_path, dockerfile=GOOD_DOCKERFILE.replace("USER propiq\n", ""))
    report = run(root)
    assert "runs as a non-root uid" in [n for n, _ in report.skipped]
    assert not any("root" in n for n in report.passed)


def test_a_missing_dockerignore_fails(tmp_path):
    root = tree(tmp_path)
    (root / ".dockerignore").unlink()
    assert ".dockerignore exists" in failures(run(root))


@pytest.mark.parametrize("dropped", [".env", "data/", "*.pem", "*.key", "secrets/"])
def test_an_ignore_file_that_lets_a_secret_through_fails(tmp_path, dropped):
    root = tree(tmp_path, ignore=GOOD_IGNORE.replace(dropped + "\n", ""))
    assert "secrets and bulk data are excluded" in failures(run(root))


def test_an_ignore_file_that_excludes_the_app_fails(tmp_path):
    root = tree(tmp_path, ignore=GOOD_IGNORE + "scheduler_worker.py\n")
    assert "the application is not excluded" in failures(run(root))


def test_an_ignore_pattern_the_matcher_cannot_evaluate_is_a_failure(tmp_path):
    """
    A guess about whether .env ships is worse than an admission. The matcher
    implements a subset of Docker's syntax and refuses to answer outside it.
    """
    root = tree(tmp_path, ignore=GOOD_IGNORE + "**/weird pattern\n")
    report = run(root)
    assert ".dockerignore uses only patterns this checker implements" in failures(report)
    assert "secrets and bulk data are excluded" not in report.passed


# --- the matcher --------------------------------------------------------------

def test_the_real_ignore_file_hides_env_and_keeps_the_example():
    rules = _ignore_rules((ROOT / ".dockerignore").read_text(encoding="utf-8"))
    assert _is_ignored(".env", rules)
    assert _is_ignored(".env.local", rules)
    assert not _is_ignored(".env.example", rules), "negation must win, as Docker does it"


def test_the_real_ignore_file_hides_bulk_data_and_keeps_the_code():
    rules = _ignore_rules((ROOT / ".dockerignore").read_text(encoding="utf-8"))
    assert _is_ignored("data/external/panel.parquet", rules)
    assert _is_ignored("outputs/decision_board.csv", rules)
    assert not _is_ignored("scheduler_worker.py", rules)
    assert not _is_ignored("src/quant/publication_gate.py", rules)


def test_last_matching_rule_wins():
    rules = _ignore_rules("*.md\n!README.md\n")
    assert _is_ignored("CHANGELOG.md", rules)
    assert not _is_ignored("README.md", rules)


def test_comments_and_blank_lines_are_not_patterns():
    rules = _ignore_rules("# .env\n\n   \n")
    assert rules == []
    assert not _is_ignored(".env", rules)


def test_a_directory_pattern_covers_what_is_under_it():
    rules = _ignore_rules("data/\n")
    assert _is_ignored("data", rules)
    assert _is_ignored("data/a/b/c.parquet", rules)
    assert not _is_ignored("database.py", rules)


# --- the daemon gate ----------------------------------------------------------

def test_a_missing_daemon_is_reported_rather_than_crashed():
    """
    This environment has the docker client and no daemon, which is exactly the
    case the script has to survive: it must say so and keep the preflight.
    """
    up, detail = daemon_state()
    assert isinstance(up, bool)
    assert detail, "a verdict without a reason is not useful in a log"


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory mode bits")
def test_the_unwritable_probe_fixture_is_actually_unwritable(tmp_path):
    """
    The smoke phase mounts a mode-0555 directory to prove check_state_dir
    detects it. If the fixture were writable the check would pass vacuously —
    which is how a permissions test quietly stops testing permissions.
    """
    d = tmp_path / "root_owned"
    d.mkdir()
    os.chmod(d, 0o555)
    assert not stat.S_IMODE(d.stat().st_mode) & stat.S_IWUSR
    with pytest.raises(OSError):
        (d / "probe").write_text("x", encoding="utf-8")
