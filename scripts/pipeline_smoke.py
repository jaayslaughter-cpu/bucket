"""
scripts/pipeline_smoke.py — one stage-by-stage verdict over the whole path.

WHAT THIS IS, AND WHAT IT DELIBERATELY IS NOT. Every stage below already has a
tool that owns it. This CALLS those tools as subprocesses and reports which one
first refuses; it reimplements none of their logic, holds no copy of their
rules, and knows nothing about markets, odds, features or webhooks. The moment
it starts parsing a stage's output for meaning rather than its exit code, it
has become a second reader of that stage's facts -- the defect this repository
keeps removing (the parlay ledger read twice, PRA defined twice, the webhook
variable spelled five times). So it reads EXIT CODES and prints tails.

WHY IT EXISTS AT ALL, given that. Five commands in sequence, each with its own
flags and its own idea of what "fine" looks like, is a thing people run wrong
or run partially. One command that runs all five and says WHICH STAGE IS THE
WALL is worth the ~200 lines, and nothing else in the tree answers that in one
pass: `railway_healthcheck` probes static readiness, `verify_wiring` probes
wiring, and neither runs the pipeline.

THE STAGES, and who owns each:

    scrape_features   main.py --no-db                 ingest + feature build
    model_artifacts   scripts.railway_healthcheck     artifacts, volume, config
    ev_contract       scripts.verify_wiring           EV path and dispatch wiring
    discord_payload   nba_model_cli notify-discord    the embed, built not sent

and, only with --with-panel because it is slow and leaves the machine:

    panel_audit       nba_model_cli audit-data        the panel's own audit

EVERY STAGE RUNS, even after one fails, unless --stop-on-fail. A run that stops
at the first wall hides the three walls behind it, and on a fresh deploy there
are usually several. The summary names the FIRST blocking stage because that is
the one to fix, and lists the rest because they are next.

IT SENDS NOTHING BY DEFAULT. `notify-discord` is dry-run unless `--send`, so the
default pass builds the embed and prints it. Pass --send to actually post, which
is the only way to prove the webhook end to end.

NO SECRETS REACH THE OUTPUT. Stage output is scrubbed through
`scheduler_worker.scrub_alert`, which already owns that rule and delegates
webhook redaction to `notify.discord.redact_webhook`. A third copy of a
redaction pattern is how one of them drifts and leaks.

COSTS NOTHING IN API QUOTA while step [2] is the wall: PropLine is step [3] and
the pipeline refuses before it. Once [2] passes, a full pass does spend one
slate's worth of PropLine quota -- which is the point of a smoke test, but
worth knowing before running it in a loop.

RESEARCH_ONLY. This runs diagnostics. It places nothing, sizes nothing and
grades nothing.

    PYTHONPATH=. python -m scripts.pipeline_smoke
    PYTHONPATH=. python -m scripts.pipeline_smoke --stage discord_payload --send
    PYTHONPATH=. python -m scripts.pipeline_smoke --json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

#: How many lines of a stage's own output to show. Enough to carry the reason
#: it refused -- these tools put the cause in their last lines on purpose --
#: without reprinting a whole pipeline run.
TAIL_LINES = 12

PASS, FAIL, TIMEOUT, SKIPPED = "PASS", "FAIL", "TIMEOUT", "SKIPPED"


@dataclass
class Stage:
    """One stage, the command that owns it, and why it is in the list."""

    name: str
    argv: list[str]
    owns: str
    #: False for a stage whose failure is informative but not a wall: the
    #: pipeline can still be diagnosed past it.
    blocking: bool = True
    timeout: int = 900


@dataclass
class Result:
    stage: Stage
    status: str
    exit_code: int | None
    seconds: float
    tail: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.name,
            "owns": self.stage.owns,
            "status": self.status,
            "exit_code": self.exit_code,
            "seconds": round(self.seconds, 1),
            "blocking": self.stage.blocking,
            "tail": self.tail,
        }


def stages(*, send: bool, with_panel: bool = False) -> list[Stage]:
    """
    The five stages in pipeline order.

    `sys.executable`, NOT "python". This container has pip for 3.13 on PATH
    beside a 3.11 interpreter, and installing with the wrong one put every
    dependency somewhere the project could not import it -- exit 0, nothing
    importable. A subprocess that picks its own interpreter can diagnose a
    tree it is not actually running on.
    """
    discord = [
        sys.executable, "-m", "scripts.nba_model_cli", "notify-discord",
        "--source", "abstention",
        "--message", "pipeline_smoke: dispatch reachability check (RESEARCH_ONLY)",
    ]
    if send:
        discord.append("--send")

    out = [
        Stage(
            "scrape_features",
            [sys.executable, "main.py", "--no-db"],
            "ingest + feature build (main.py, dry run, no persistence)",
        ),
        Stage(
            "model_artifacts",
            [sys.executable, "-m", "scripts.railway_healthcheck"],
            "artifacts, state root, calibration report, dispatch config",
        ),
        Stage(
            "ev_contract",
            [sys.executable, "-m", "scripts.verify_wiring"],
            "EV path and dispatch wiring",
        ),
        Stage(
            "discord_payload",
            discord,
            "the Discord embed (built; sent only with --send)",
        ),
    ]

    # OPT-IN, BECAUSE IT IS SLOW AND IT LEAVES THE MACHINE. Measured at 99s
    # on a first run: `audit-data` fetches player logs from stats.nba.com and
    # retries three times with backoff, and that host is denied or timed out
    # from plenty of networks (it is denied from this project's own build
    # container). A smoke test somebody runs under pressure must not spend a
    # minute and a half on a stage whose failure is forgiven anyway, so the
    # panel's health is a separate question asked with --with-panel.
    if with_panel:
        out.insert(1, Stage(
            "panel_audit",
            [sys.executable, "-m", "scripts.nba_model_cli", "audit-data"],
            "the training panel's own audit (hits stats.nba.com)",
            blocking=False,
            timeout=300,
        ))
    return out


def _scrub(text: str) -> str:
    """
    Through the one scrubber this project already has.

    Imported lazily and defensively: this script's job is to report on a tree
    that may be mid-breakage, and it must still produce a verdict if
    `scheduler_worker` cannot be imported at all. Falling back to printing
    NOTHING is the safe direction -- an unscrubbed tail could carry
    DATABASE_URL, password included.
    """
    try:
        from scheduler_worker import scrub_alert

        return scrub_alert(text)
    except Exception:  # noqa: BLE001 — see above
        return (
            "[output withheld: scheduler_worker.scrub_alert could not be "
            "imported, so this output could not be checked for credentials]"
        )


def run_stage(stage: Stage) -> Result:
    """Run one stage. Never raises: a crashed stage is a result, not an abort."""
    started = time.monotonic()
    try:
        proc = subprocess.run(
            stage.argv,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=stage.timeout,
            # PYTHONPATH=. is how every other entry point in this tree is
            # invoked; without it `import src...` fails from a subprocess.
            env={**_env(), "PYTHONPATH": "."},
        )
    except subprocess.TimeoutExpired:
        return Result(stage, TIMEOUT, None, time.monotonic() - started, [
            f"no output within {stage.timeout}s",
        ])
    except (OSError, ValueError) as exc:
        return Result(stage, FAIL, None, time.monotonic() - started, [
            f"could not start: {exc}",
        ])

    combined = (proc.stdout or "") + (proc.stderr or "")
    tail = _scrub(combined).strip().splitlines()[-TAIL_LINES:]
    status = PASS if proc.returncode == 0 else FAIL
    return Result(stage, status, proc.returncode, time.monotonic() - started, tail)


def _env() -> dict[str, str]:
    import os

    return dict(os.environ)


def _print_human(results: list[Result], *, sent: bool) -> None:
    width = max(len(r.stage.name) for r in results)
    print()
    print("PropIQ pipeline smoke — RESEARCH_ONLY, places nothing")
    print("=" * 72)
    for r in results:
        flag = "" if r.stage.blocking else "  (non-blocking)"
        print(f"  [{r.status:<7}] {r.stage.name:<{width}}  {r.seconds:>6.1f}s"
              f"  exit={r.exit_code}{flag}")
        print(f"             owns: {r.stage.owns}")
        if r.status != PASS:
            for line in r.tail:
                print(f"             | {line}")
        print()

    walls = [r for r in results if r.status != PASS and r.stage.blocking]
    print("=" * 72)
    if not walls:
        print("VERDICT: every blocking stage passed.")
    else:
        first = walls[0]
        print(f"VERDICT: FIRST WALL is `{first.stage.name}` — {first.stage.owns}")
        if len(walls) > 1:
            rest = ", ".join(w.stage.name for w in walls[1:])
            print(f"         Behind it, also failing: {rest}")
        print("         Fix the first one; the rest may resolve with it.")
    skipped = [r for r in results if r.status == SKIPPED]
    if skipped:
        print(f"         Not run: {', '.join(r.stage.name for r in skipped)}")
    print(f"         Discord: {'POSTED' if sent else 'dry run, nothing sent'}")
    print()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Run every pipeline stage's own tool and say which one is "
                    "the wall. RESEARCH_ONLY."
    )
    ap.add_argument("--stage", action="append", metavar="NAME",
                    help="Run only this stage. Repeatable.")
    ap.add_argument("--send", action="store_true",
                    help="Actually POST the Discord embed. Off by default.")
    ap.add_argument("--stop-on-fail", action="store_true",
                    help="Stop at the first blocking failure instead of "
                         "running every stage.")
    ap.add_argument("--json", action="store_true", dest="as_json",
                    help="Emit one JSON object instead of the table.")
    ap.add_argument("--with-panel", action="store_true", dest="with_panel",
                    help="Also audit the training panel. Slow (~100s) and it "
                         "fetches from stats.nba.com; non-blocking either way.")
    ap.add_argument("--list", action="store_true",
                    help="Print the stages and exit.")
    args = ap.parse_args(argv)

    all_stages = stages(send=args.send, with_panel=args.with_panel)
    known = {s.name for s in all_stages}

    if args.list:
        for s in all_stages:
            print(f"{s.name:<18} {s.owns}")
        return 0

    if args.stage:
        unknown = sorted(set(args.stage) - known)
        if unknown:
            print(f"Unknown stage(s): {unknown}. Known: {sorted(known)}",
                  file=sys.stderr)
            return 2
        selected = [s for s in all_stages if s.name in set(args.stage)]
    else:
        selected = all_stages

    results: list[Result] = []
    stopped = False
    for stage in selected:
        if stopped:
            results.append(Result(stage, SKIPPED, None, 0.0,
                                  ["not run: an earlier blocking stage failed"]))
            continue
        result = run_stage(stage)
        results.append(result)
        if args.stop_on_fail and result.status != PASS and stage.blocking:
            stopped = True

    if args.as_json:
        walls = [r.stage.name for r in results
                 if r.status != PASS and r.stage.blocking]
        print(json.dumps({
            "research_only": True,
            "discord_sent": bool(args.send),
            "first_wall": walls[0] if walls else None,
            "blocking_failures": walls,
            "stages": [r.as_dict() for r in results],
        }, indent=2))
    else:
        _print_human(results, sent=bool(args.send))

    # Non-zero when a BLOCKING stage did not pass, so this is usable in a
    # shell `&&` chain. A non-blocking failure is reported and forgiven.
    return 1 if any(r.status != PASS and r.stage.blocking for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
