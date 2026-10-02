"""Build and smoke-test the worker image — Railway roadmap step 2.

RESEARCH_ONLY. Builds and runs a container. Places no wager, contacts no
operator, needs no credential: every smoke check below runs with an empty
environment on purpose, because that is the state a fresh container starts in.

    python -m scripts.validate_docker              # preflight, build, smoke
    python -m scripts.validate_docker --preflight  # preflight only, no daemon
    python -m scripts.validate_docker --skip-build # smoke an image already built

Runs on Windows and Linux alike: it drives the `docker` CLI through
subprocess rather than a shell, so there is no .sh/.ps1 split to keep in sync.

WHY THIS EXISTS. `Dockerfile` says of itself:

    NOT BUILT OR RUN ANYWHERE YET. The environment this was written in has the
    docker client but no daemon [...] Every instruction here is reasoned from
    the repository's own dependency metadata, not from a green build.

and `docs/deploy_railway.md` makes a claim that has never been executed: that
`scheduler_worker.check_state_dir()` catches a root-owned volume mounted over
`/app/data`. Phase C below runs that exact scenario, so the claim is tested
rather than documented.

PHASES
  A. preflight — no daemon required, and the part this repository's own CI
     environment can verify. Reads the Dockerfile, the ignore file and
     pyproject and checks they agree with each other and with the tree.
  B. build — `docker build`.
  C. smoke — six `docker run` checks against the built image, including the
     two volume-permission cases and a check that no `.env` reached a layer.

Exit code is the number of failed checks, so it is usable from CI.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TAG = "propiq-worker:smoke"

#: Paths that must never reach a build layer. A credential in an image layer
#: survives every later layer that deletes it, and the image is pushed.
MUST_BE_IGNORED = (
    ".env",
    ".env.local",
    ".env.production",
    "secrets/token.json",
    "server.pem",
    "id_rsa.key",
    "data/external/panel.parquet",
    "outputs/decision_board.csv",
    "catboost_info/learn_error.tsv",
    "PropIQ_Historical_Training_Pack.zip",
)
#: Paths the image cannot work without.
MUST_BE_KEPT = (
    "scheduler_worker.py",
    "main.py",
    "pyproject.toml",
    "src/quant/publication_gate.py",
    ".env.example",
)
#: Extras the Dockerfile installs. Each must exist in pyproject.
EXPECTED_EXTRAS = ("ml", "db", "deploy")
EXPECTED_UID = 10001
EXPECTED_JOB_IDS = {"slate", "settlement"}


@dataclass
class Report:
    passed: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)

    def ok(self, name: str) -> None:
        self.passed.append(name)
        print(f"  PASS  {name}")

    def bad(self, name: str, why: str) -> None:
        self.failed.append((name, why))
        print(f"  FAIL  {name}\n          {why}")

    def skip(self, name: str, why: str) -> None:
        self.skipped.append((name, why))
        print(f"  SKIP  {name} — {why}")


# --- dockerignore ------------------------------------------------------------

#: Pattern forms this matcher implements. A pattern outside the subset is a
#: failed check, not a silent miss: a wrong answer about whether .env ships is
#: worse than no answer.
_SUPPORTED = re.compile(r"^!?[A-Za-z0-9_./*?\[\]-]+/?$")


def _ignore_rules(text: str) -> list[tuple[str, bool]]:
    rules: list[tuple[str, bool]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        negate = line.startswith("!")
        pattern = line[1:] if negate else line
        rules.append((pattern.rstrip("/"), negate))
    return rules


def _is_ignored(path: str, rules: list[tuple[str, bool]]) -> bool:
    """Last matching rule wins, as Docker does it."""
    ignored = False
    for pattern, negate in rules:
        if fnmatch.fnmatch(path, pattern) or path.startswith(pattern + "/"):
            ignored = not negate
    return ignored


def preflight(report: Report, root: Path = ROOT) -> None:
    """``root`` is a parameter so the checks can be exercised against a
    synthetic tree; see tests/test_validate_docker.py."""
    print("\nA. preflight (no daemon required)")

    dockerfile = root / "Dockerfile"
    if not dockerfile.is_file():
        report.bad("Dockerfile exists", f"{dockerfile} not found")
        return
    body = dockerfile.read_text(encoding="utf-8")
    report.ok("Dockerfile exists")

    # 1. the CMD target is a real file
    cmd = re.search(r'^CMD \[(.+)\]', body, re.M)
    if not cmd:
        report.bad("CMD is present", "no CMD instruction")
    else:
        parts = [p.strip().strip('"') for p in cmd.group(1).split(",")]
        target = next((p for p in parts if p.endswith(".py")), None)
        if target is None:
            report.bad("CMD names a script", f"CMD {parts} names no .py entrypoint")
        elif not (root / target).is_file():
            report.bad("CMD target exists", f"CMD runs {target}, which is not in the tree")
        else:
            report.ok(f"CMD target exists ({target})")

    # 2. no secret baked in. Scanned on the JOINED body, because an ENV with
    #    line continuations puts its later assignments on lines that do not
    #    start with ENV — a secret hiding there is the easiest one to miss.
    #    The keyword may sit anywhere in the variable name: DISCORD_WEBHOOK_URL
    #    is a webhook, and an earlier version of this check did not see it.
    joined = re.sub(r"\\\s*\n\s*", " ", body)
    leaks = [
        m.group(0).strip()
        for m in re.finditer(
            r"(?m)^\s*(?:ENV|ARG)\s[^\n]*?"
            r"\b[A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD|WEBHOOK|CREDENTIAL|DSN"
            r"|DATABASE_URL)[A-Z0-9_]*\s*=\s*\S+",
            joined,
        )
    ]
    if leaks:
        report.bad("no credential defaulted in the image", "; ".join(leaks))
    else:
        report.ok("no credential defaulted in the image")

    # 3. the extras it installs exist
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    installed = re.findall(r'pip install[^\n]*\.\[([^\]]+)\]', body)
    declared = set(re.findall(r"^(\w[\w-]*)\s*=\s*\[", pyproject, re.M))
    if not installed:
        report.bad("Dockerfile installs an extra", "no `pip install -e .[...]` found")
    else:
        asked = {e.strip() for group in installed for e in group.split(",")}
        missing = sorted(asked - declared)
        if missing:
            report.bad(
                "every installed extra is declared",
                f"Dockerfile installs {sorted(asked)}; pyproject has no {missing}",
            )
        elif not set(EXPECTED_EXTRAS) <= asked:
            report.bad(
                "the ML and deploy extras are installed",
                f"expected {EXPECTED_EXTRAS}, Dockerfile installs {sorted(asked)}",
            )
        else:
            report.ok(f"every installed extra is declared ({', '.join(sorted(asked))})")

    # 4. the USER line, which every permission check below depends on
    user = re.search(r"^USER\s+(\S+)", body, re.M)
    uid = re.search(r"--uid\s+(\d+)", body)
    if not user:
        report.skip("runs as a non-root uid", "no USER instruction — container runs as root")
    elif not uid:
        report.bad("the non-root uid is pinned", f"USER {user.group(1)} but no --uid in useradd")
    elif int(uid.group(1)) != EXPECTED_UID:
        report.bad("the non-root uid matches the docs", f"uid {uid.group(1)}, docs say {EXPECTED_UID}")
    else:
        report.ok(f"runs as non-root uid {uid.group(1)}")

    # 5. the ignore file, matched rather than grepped
    ignore_path = root / ".dockerignore"
    if not ignore_path.is_file():
        report.bad(".dockerignore exists", "absent: `COPY . .` would copy .env and data/")
        return
    text = ignore_path.read_text(encoding="utf-8")
    unsupported = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
        and not _SUPPORTED.match(line.strip())
    ]
    if unsupported:
        report.bad(
            ".dockerignore uses only patterns this checker implements",
            f"cannot evaluate {unsupported} — the result below would be a guess",
        )
        return
    rules = _ignore_rules(text)

    leaked = [p for p in MUST_BE_IGNORED if not _is_ignored(p, rules)]
    if leaked:
        report.bad("secrets and bulk data are excluded", f"would be COPYed into a layer: {leaked}")
    else:
        report.ok("secrets and bulk data are excluded")

    dropped = [p for p in MUST_BE_KEPT if _is_ignored(p, rules)]
    if dropped:
        report.bad("the application is not excluded", f"excluded by .dockerignore: {dropped}")
    else:
        report.ok("the application is not excluded")


# --- docker ------------------------------------------------------------------

def _docker(*args: str, env: dict[str, str] | None = None, timeout: int = 1800):
    return subprocess.run(
        ["docker", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, **(env or {})},
    )


def daemon_state() -> tuple[bool, str]:
    if shutil.which("docker") is None:
        return False, "no docker client on PATH"
    try:
        proc = _docker("info", "--format", "{{.ServerVersion}}", timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"docker info failed: {exc}"
    if proc.returncode != 0:
        first = (proc.stderr or proc.stdout).strip().splitlines()
        return False, first[0] if first else "docker info returned non-zero"
    return True, proc.stdout.strip()


def build(report: Report, tag: str) -> bool:
    print(f"\nB. build ({tag})")
    proc = _docker("build", "-t", tag, ".")
    if proc.returncode != 0:
        tail = "\n          ".join((proc.stderr or proc.stdout).strip().splitlines()[-12:])
        report.bad("docker build", tail)
        return False
    report.ok("docker build")
    return True


_PY_IMPORTS = (
    "import apscheduler, sqlalchemy, pandas, numpy, pydantic; "
    "import xgboost, catboost; "
    "print('ok')"
)
_PY_JOBS = (
    "import json, scheduler_worker as w; "
    "print(json.dumps([j['id'] for j in w.describe(w.build_scheduler())]))"
)
_PY_PROBE = "import json, scheduler_worker as w; print(json.dumps(w.check_state_dir()))"


def _run(report: Report, name: str, tag: str, script: str, *, mounts=(), expect_rc=0):
    args = ["run", "--rm", "--network", "none"]
    for host, container, *mode in mounts:
        args += ["-v", f"{host}:{container}" + (f":{mode[0]}" if mode else "")]
    proc = _docker(*args, tag, "python", "-c", script, timeout=300)
    if proc.returncode != expect_rc:
        tail = "\n          ".join((proc.stderr or proc.stdout).strip().splitlines()[-8:])
        report.bad(name, f"exit {proc.returncode}: {tail}")
        return None
    return proc.stdout.strip()


def smoke(report: Report, tag: str) -> None:
    print(f"\nC. in-image smoke ({tag})")
    import tempfile

    # 1. the extras actually landed. A minimal image starts cleanly and then
    #    fails at the first inference, unattended, after the slate is ingested.
    if _run(report, "the ML and deploy extras import", tag, _PY_IMPORTS):
        report.ok("the ML and deploy extras import")

    # 2. the entrypoint imports with an empty environment — no DATABASE_URL,
    #    no PROPLINE_API_KEY. A fresh container has none of them.
    out = _run(report, "the scheduler builds its job table", tag, _PY_JOBS)
    if out is not None:
        try:
            ids = set(json.loads(out.splitlines()[-1]))
        except (ValueError, IndexError):
            report.bad("the scheduler builds its job table", f"unparseable output: {out!r}")
        else:
            if ids == EXPECTED_JOB_IDS:
                report.ok(f"the scheduler builds its job table ({', '.join(sorted(ids))})")
            else:
                report.bad(
                    "the scheduler builds its job table",
                    f"expected {sorted(EXPECTED_JOB_IDS)}, got {sorted(ids)}",
                )

    # 3. the USER line took effect
    out = _run(report, "the process runs as the non-root uid", tag, "import os; print(os.getuid())")
    if out is not None:
        if out.splitlines()[-1].strip() == str(EXPECTED_UID):
            report.ok(f"the process runs as uid {EXPECTED_UID}")
        else:
            report.bad("the process runs as the non-root uid", f"uid {out!r}, expected {EXPECTED_UID}")

    # 4. no .env reached a layer. The ignore file says so; this asks the image.
    out = _run(
        report, "no .env in the image", tag,
        "import pathlib,sys; "
        "hits=[str(p) for p in pathlib.Path('/app').rglob('.env*') "
        "if p.name != '.env.example']; print(hits)",
    )
    if out is not None:
        if out.splitlines()[-1].strip() == "[]":
            report.ok("no .env in the image")
        else:
            report.bad("no .env in the image", f"found {out}")

    # 5 and 6. THE DOCUMENTED TRAP, EXECUTED. docs/deploy_railway.md says a
    # volume mounted at /app/data replaces the chowned directory with a
    # root-owned one, and that check_state_dir() names it at boot. Both halves
    # are asserted here: unwritable is DETECTED, writable is not false-alarmed.
    with tempfile.TemporaryDirectory() as tmp:
        root_owned = Path(tmp) / "root_owned"
        root_owned.mkdir()
        os.chmod(root_owned, 0o555)
        out = _run(
            report, "an unwritable volume is detected", tag, _PY_PROBE,
            mounts=[(str(root_owned), "/app/data")],
        )
        if out is not None:
            try:
                verdict = json.loads(out.splitlines()[-1])
            except (ValueError, IndexError):
                report.bad("an unwritable volume is detected", f"unparseable: {out!r}")
            else:
                if verdict.get("writable") is False:
                    report.ok("an unwritable volume is detected, not silently tolerated")
                else:
                    report.bad(
                        "an unwritable volume is detected",
                        "check_state_dir reported writable on a mode-0555 mount — the "
                        "documented boot warning would never fire, and the real "
                        "symptom is a nightly 'calibration report failed'",
                    )

        writable = Path(tmp) / "writable"
        writable.mkdir()
        os.chmod(writable, 0o777)
        out = _run(
            report, "a writable volume is not false-alarmed", tag, _PY_PROBE,
            mounts=[(str(writable), "/app/data")],
        )
        if out is not None:
            try:
                verdict = json.loads(out.splitlines()[-1])
            except (ValueError, IndexError):
                report.bad("a writable volume is not false-alarmed", f"unparseable: {out!r}")
            else:
                if verdict.get("writable") is True:
                    report.ok("a writable volume is not false-alarmed")
                else:
                    report.bad(
                        "a writable volume is not false-alarmed",
                        f"reported unwritable on a 0777 mount: {verdict}",
                    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default=DEFAULT_TAG)
    ap.add_argument("--preflight", action="store_true", help="phase A only")
    ap.add_argument("--skip-build", action="store_true", help="smoke an existing image")
    args = ap.parse_args(argv)

    report = Report()
    preflight(report)

    if args.preflight:
        print("\n--preflight: phases B and C not attempted.")
    else:
        up, detail = daemon_state()
        if not up:
            print(f"\nB. build — SKIPPED: {detail}")
            print("C. smoke — SKIPPED: needs a built image")
            report.skip("docker build", detail)
            report.skip("in-image smoke", "no daemon")
            print(
                "\nThe preflight above is the whole of what can be checked without a\n"
                "daemon. Steps 2 and the volume-permission claim in\n"
                "docs/deploy_railway.md stay unverified until this runs where Docker\n"
                "is available."
            )
        else:
            print(f"  docker daemon {detail}")
            built = True if args.skip_build else build(report, args.tag)
            if built:
                smoke(report, args.tag)
            else:
                report.skip("in-image smoke", "build failed")

    print(
        f"\n{len(report.passed)} passed, {len(report.failed)} failed, "
        f"{len(report.skipped)} skipped"
    )
    for name, why in report.failed:
        print(f"  FAILED: {name} — {why}")
    return len(report.failed)


if __name__ == "__main__":
    sys.exit(main())
