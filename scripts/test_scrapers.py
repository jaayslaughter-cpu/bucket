"""
scripts/test_scrapers.py — hit each prop source independently and say what
came back.

WHAT THIS IS FOR. When somebody reports "the scrapers are broken", the first
question is which of them exist, and the second is what each one actually
returns. Answering both took a day of reading on 2026-10-18 because a
document described a subsystem that had never been written. This script
answers both in about ten seconds, and it answers them from the tree and the
wire rather than from any document.

WHAT IT WILL NOT DO. It does not invent a client for a platform that has
none, and it does not report a plausible-looking failure for an endpoint it
never called. A source with no client prints ABSENT and says what building
one would mean. Printing "403 Cloudflare" for a module that does not exist is
exactly how the 2026-10-18 report happened.

NO CREDENTIALS ARE PRINTED. PropLine needs $PROPLINE_API_KEY; this reports
whether one is set and never what it is.

RESEARCH_ONLY. Read-only diagnostics. It fetches posted lines and prints a
sample; it prices nothing, recommends nothing and places nothing.

Usage:
    python -m scripts.test_scrapers              # every source
    python -m scripts.test_scrapers --source propline
    python -m scripts.test_scrapers --sample 10  # rows to print per source
    python -m scripts.test_scrapers --offline    # wiring only, no network

Exit codes:
    0  every source with a client answered (an empty board is an answer)
    1  at least one source with a client could not be read
    2  bad arguments
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logger = logging.getLogger("test_scrapers")

EXIT_OK = 0
EXIT_SOURCE_FAILED = 1

#: Platforms this project is allowed to read, and where each one's client is.
#: A module path of None means NO CLIENT EXISTS -- stated here once, checked
#: against the filesystem at runtime, never asserted from memory.
SOURCES: dict[str, dict[str, Any]] = {
    "propline": {
        "module": "src/ingestion/propline.py",
        "what": "licensed prop-line API; also carries underdog and prizepicks board lines",
    },
    "underdog": {
        "module": None,
        "what": "pick'em board capture",
        "note": (
            "No client, and none has ever been committed. Underdog lines DO "
            "reach this pipeline today -- as a book key inside PropLine's "
            "payload (DFS_BOOKS, src/ingestion/propline.py), which is the "
            "sanctioned path. A direct client is designed but not built: "
            "docs/pickem_props.md."
        ),
    },
    "prizepicks": {
        "module": None,
        "what": "pick'em board capture",
        "note": (
            "No client, and none has ever been committed. PrizePicks lines "
            "reach this pipeline the same way Underdog's do, through "
            "PropLine's DFS_BOOKS. A direct client is designed but not "
            "built, and docs/pickem_props.md forbids getting past a bot wall "
            "to build one."
        ),
    },
    "sleeper": {
        "module": None,
        "what": "pick'em board capture",
        "note": (
            "No client, and none has ever been committed. Sleeper is not a "
            "PropLine book either, so its lines reach this pipeline by no "
            "route at all. config/dfs_payouts.yaml records why it is absent "
            "from the payout matrix: it prices per-leg dynamic odds rather "
            "than a fixed tiered one."
        ),
    },
}


@dataclass
class SourceReport:
    name: str
    state: str  # OK | EMPTY | FAILED | ABSENT
    detail: str = ""
    http_status: int | None = None
    rows: int = 0
    samples: list[str] = field(default_factory=list)

    @property
    def counts_against_us(self) -> bool:
        """ABSENT is not a failure. Nothing is broken; nothing is there."""
        return self.state == "FAILED"


def _check_propline(sample: int, offline: bool) -> SourceReport:
    from src.ingestion.propline import (
        ENV_API_KEY,
        PropLineAuthError,
        PropLineError,
        pull_nba_prop_lines,
    )

    if not (os.environ.get(ENV_API_KEY) or "").strip():
        return SourceReport(
            "propline",
            "FAILED",
            f"${ENV_API_KEY} is not set in this environment, so the client "
            f"refuses to run rather than return an empty board that reads "
            f"like a quiet night.",
        )
    if offline:
        return SourceReport("propline", "OK", "key present; --offline, no request made")

    try:
        snapshots = pull_nba_prop_lines()
    except PropLineAuthError as exc:
        return SourceReport("propline", "FAILED", str(exc), http_status=401)
    except PropLineError as exc:
        return SourceReport("propline", "FAILED", str(exc))
    except Exception as exc:  # noqa: BLE001 — a diagnostic reports, never raises
        return SourceReport(
            "propline", "FAILED", f"{type(exc).__name__}: {exc}"
        )

    rows = [line for snap in snapshots for line in snap.lines]
    statuses = {snap.source: snap.status for snap in snapshots}
    messages = [snap.message for snap in snapshots if snap.message]

    if not rows:
        return SourceReport(
            "propline",
            "EMPTY",
            "; ".join(messages) or f"no lines; per-book status {statuses}",
        )

    samples = [
        f"{r.source:12} {r.player_name:24} {r.market:4} {r.line:>6} "
        f"over={r.over_odds_american} under={r.under_odds_american} "
        f"captured={r.captured_at_utc}"
        for r in rows[:sample]
    ]
    return SourceReport("propline", "OK", f"per-book status {statuses}",
                        rows=len(rows), samples=samples)


def _check_absent(name: str) -> SourceReport:
    return SourceReport(name, "ABSENT", SOURCES[name].get("note", ""))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Hit each prop source independently and report what came back.",
    )
    parser.add_argument("--source", choices=sorted(SOURCES), default=None)
    parser.add_argument("--sample", type=int, default=5,
                        help="rows to print per source (default 5)")
    parser.add_argument("--offline", action="store_true",
                        help="check wiring only; make no network request")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    wanted = [args.source] if args.source else sorted(SOURCES)
    reports: list[SourceReport] = []

    repo_root = Path(__file__).resolve().parents[1]
    for name in wanted:
        module = SOURCES[name]["module"]
        # CHECKED, not trusted: the table above says where a client lives and
        # the filesystem says whether it does. A table that drifts from the
        # tree is the defect that produced this script.
        if module is None or not (repo_root / module).is_file():
            if module is not None:
                reports.append(SourceReport(
                    name, "FAILED",
                    f"this script expected a client at {module} and it is not there",
                ))
            else:
                reports.append(_check_absent(name))
            continue
        if name == "propline":
            reports.append(_check_propline(args.sample, args.offline))
        else:  # pragma: no cover - no other source has a client yet
            reports.append(SourceReport(name, "FAILED", "no probe implemented"))

    width = max(len(r.name) for r in reports)
    print()
    print(f"{'SOURCE'.ljust(width)}  STATE    ROWS  DETAIL")
    print("-" * 100)
    for r in reports:
        status = f"[{r.http_status}] " if r.http_status else ""
        print(f"{r.name.ljust(width)}  {r.state:<7}  {r.rows:>4}  {status}{r.detail}")
    for r in reports:
        if r.samples:
            print(f"\n--- {r.name}: first {len(r.samples)} parsed line(s)")
            for line in r.samples:
                print(f"    {line}")

    failed = [r.name for r in reports if r.counts_against_us]
    absent = [r.name for r in reports if r.state == "ABSENT"]
    print()
    if absent:
        print(f"ABSENT (nothing to fix — no client was ever written): {', '.join(absent)}")
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        return EXIT_SOURCE_FAILED
    print("Every source with a client answered.")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
