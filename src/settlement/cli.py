"""
src/settlement/cli.py — command line entry point for settlement.

The grading engine (``runner``) and the reporting queries (``metrics``)
both existed, but nothing wired them to a command, so the documented
``python -m src.settlement.cli settle`` did not run. This module is that
wiring and nothing more: it parses arguments, calls the existing
functions, and prints their output as JSON.

Exit codes are meaningful, because a scheduled run is graded by them:
    0  the command completed
    1  the command failed (bad arguments, database or fetch error)

A settle run that grades nothing is NOT a failure — an empty slate and a
broken pipeline must not return the same code, or a cron alert can never
tell them apart.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import timedelta
from typing import Any

from src.settlement.metrics import get_performance_by_market, get_performance_summary
from src.settlement.runner import settle_pending_props
from src.utils.timezones import pacific_calendar_date

logger = logging.getLogger(__name__)


def _print(payload: dict[str, Any] | list[Any]) -> None:
    """Emit JSON on stdout so the output can be piped or stored."""
    print(json.dumps(payload, indent=2, default=str))


def _cmd_settle(args: argparse.Namespace) -> int:
    report = settle_pending_props(
        max_game_age_days=args.max_age_days,
        dry_run=args.dry_run,
    )
    payload = report.as_dict()
    payload["dry_run"] = args.dry_run
    if args.dry_run:
        payload["note"] = "DRY RUN — nothing was written to the database."
    _print(payload)
    return 0


def _cmd_summary(args: argparse.Namespace) -> int:
    # Pacific calendar day: an NBA slate running past midnight UTC is still
    # the same game day, and a UTC-dated window would drop last night.
    end = pacific_calendar_date()
    start = end - timedelta(days=args.days)
    summary = get_performance_summary(
        start_date=start,
        end_date=end,
        market=args.market,
        include_pickem=not args.exclude_pickem,
    )
    payload = summary.as_dict()
    if args.by_market:
        payload["by_market"] = get_performance_by_market(start_date=start, end_date=end)
    _print(payload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.settlement.cli",
        description=(
            "Grade settled props and report performance (NBA only). "
            "RESEARCH ONLY — this reports what already happened and places "
            "no wagers."
        ),
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    settle = sub.add_parser("settle", help="Grade PENDING props against final box scores")
    settle.add_argument(
        "--max-age-days", type=int, default=14,
        help="Lookback bound, so a permanently stuck row is not retried forever",
    )
    settle.add_argument(
        "--dry-run", action="store_true",
        help="Fetch and grade, but write nothing to the database",
    )
    settle.set_defaults(func=_cmd_settle)

    summary = sub.add_parser("summary", help="W-L-P, strike rate, ROI and CLV")
    summary.add_argument("--days", type=int, default=30, help="Lookback window in days")
    summary.add_argument("--market", type=str, default=None, help="Filter to one market")
    summary.add_argument(
        "--by-market", action="store_true", help="Include the per-market breakdown"
    )
    summary.add_argument(
        "--exclude-pickem", action="store_true",
        help="Drop rows carrying a payout multiplier rather than two-way odds",
    )
    summary.set_defaults(func=_cmd_summary)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if getattr(args, "days", 1) is not None and getattr(args, "days", 1) <= 0:
        parser.error("--days must be positive")
    if getattr(args, "max_age_days", 1) is not None and getattr(args, "max_age_days", 1) <= 0:
        parser.error("--max-age-days must be positive")

    try:
        return int(args.func(args))
    except Exception as exc:  # noqa: BLE001 — the boundary; nothing above catches
        logger.error("Settlement command failed: %s", exc, exc_info=True)
        _print({"status": "FAILED", "command": args.command, "error": str(exc)})
        return 1


if __name__ == "__main__":
    sys.exit(main())
