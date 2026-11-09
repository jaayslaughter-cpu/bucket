"""
scripts/compare_components.py — which scoring component was more accurate.

WHAT IT IS FOR. `config/model_comparison.yaml` ranks the components, but that
ranking was fitted against RESEARCH-LINE labels: it says which is best at
"over the trailing average". Serving asks "over a POSTED line", and nothing in
this project has measured that. Every slate now scores the serving component
AND every other line-taking one on the same rows at the same lines
(`main.shadow_probabilities`), writing the challengers into the decision
record. This reads them back once those rows have graded.

THE SHAPE IS `analyze_losses`'s. The arithmetic is pure in
`src/settlement/component_comparison.py`; this file is the database read, the
record read and the printing. Keeping the join and the scoring out of a script
is what lets them be tested without a database.

EXIT STATUS IS ABOUT THE COMPARISON, NOT ABOUT THE MODELS. 0 when the report
was produced, INCLUDING when it honestly reports that nothing has graded yet
-- the expected state until the first settlement after a slate. 2 when the
ledger could not be read, 3 when the decision records could not be. A
non-zero exit must mean "this did not run", or a scheduled copy becomes an
alarm nobody reads.

IT PROMOTES NOTHING. A lower Brier is evidence for a decision about
`$PROPIQ_SCORING_COMPONENT`, not the decision, and not a profit figure.
Switching what serves also resets the graded evidence accumulated under the
current component, which is a cost this report cannot see.

Usage:
    python -m scripts.compare_components
    python -m scripts.compare_components --days 60 --market PTS
    python -m scripts.compare_components --json /app/data/component_comparison.json

RESEARCH_ONLY. Reads graded research predictions and recorded probabilities
and reports arithmetic on them. It prices nothing, sizes nothing, places
nothing and changes no configuration.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger("compare_components")

EXIT_OK = 0
EXIT_LEDGER_UNREADABLE = 2
EXIT_RECORDS_UNREADABLE = 3


def _row(label: str, scored: dict[str, Any]) -> str:
    def fmt(key: str, width: int = 9) -> str:
        value = scored.get(key)
        return f"{value:>{width}.6f}" if isinstance(value, (int, float)) else f"{'--':>{width}}"

    return (
        f"    {label:<14}"
        f" n={scored.get('n', 0):<6}"
        f" brier={fmt('brier')}"
        f" ece={fmt('ece')}"
        f" mean_p={fmt('mean_predicted')}"
        f" observed={fmt('observed_over_rate')}"
    )


def render(report: dict[str, Any]) -> str:
    """The report as a person reads it."""
    lines: list[str] = []
    add = lines.append
    add("")
    add("=" * 78)
    add("  COMPONENT COMPARISON — P(over) against whether the over hit")
    add("  RESEARCH_ONLY. Not a profit figure and not a promotion.")
    add("=" * 78)
    add(f"  as of           : {report.get('as_of')}")
    add(f"  graded rows read: {report.get('graded_rows_read')}")
    add(f"  record rows read: {report.get('record_rows_read')}")
    add(f"  joined          : {report.get('matched_rows')}")

    if report.get("status") != "OK":
        add("")
        add(f"  STATUS: {report.get('status')}")
        add(f"  {report.get('reason', '')}")
        for reason, count in sorted((report.get("excluded_by_reason") or {}).items()):
            add(f"    excluded {count:>5}  {reason}")
        add("=" * 78)
        return "\n".join(lines)

    add("")
    add("  EVERY ROW EACH COMPONENT ANSWERED")
    add("  (a component that abstains more can look better by picking easier rows)")
    for name, scored in sorted((report.get("per_component") or {}).items()):
        add(_row(name, scored))
        add(f"    {'':<14} answered={scored.get('answered')} abstained={scored.get('abstained')}")

    add("")
    add(f"  HEAD TO HEAD — the {report.get('paired_rows')} row(s) EVERY component answered")
    for name, scored in sorted((report.get("head_to_head") or {}).items()):
        add(_row(name, scored))

    excluded = report.get("excluded_by_reason") or {}
    if excluded:
        add("")
        add("  NOT JOINED")
        for reason, count in sorted(excluded.items()):
            add(f"    {count:>6}  {reason}")

    add("")
    add("  FINDINGS")
    for finding in report.get("findings") or []:
        add(f"    - {finding}")

    add("")
    add("  Nothing above is a recommendation to act, a size, or a claim of")
    add("  profitability. A Brier score says how honest the probabilities were.")
    add("=" * 78)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m scripts.compare_components",
        description="Brier and ECE per scoring component over the graded window.",
    )
    ap.add_argument("--days", type=int, default=30,
                    help="How far back to read graded rows (default 30).")
    ap.add_argument("--market", default=None, help="PTS|REB|AST; default all.")
    ap.add_argument("--since", default=None,
                    help="Only read decision records from this slate date (YYYY-MM-DD).")
    ap.add_argument("--bins", type=int, default=10, help="Reliability bins (default 10).")
    ap.add_argument("--json", dest="json_out", default=None,
                    help="Also write the report as JSON to this path.")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    from src.settlement.component_comparison import compare_components
    from src.utils.timezones import format_pacific_iso, now_pacific

    try:
        from src.db.repository import load_graded_prop_residual_rows

        graded = load_graded_prop_residual_rows(
            lookback_days=args.days, market=args.market
        )
    except Exception as exc:  # noqa: BLE001 — the ledger read is a hard dependency
        logger.error(
            "Could not read the graded ledger: %s. This is a database or "
            "configuration fault, not an empty ledger -- an empty one reports "
            "itself and exits 0.", exc,
        )
        return EXIT_LEDGER_UNREADABLE

    try:
        from src.settlement.decision_record import read_records, recorded_rows_from
        from src.utils.volume import resolve_state_root

        state_root, how = resolve_state_root()
        logger.info("Reading decision records under %s (via %s).", state_root, how)
        records = read_records(state_root, since=args.since)
        record_rows = recorded_rows_from(records)
    except Exception as exc:  # noqa: BLE001 — so is the record read
        logger.error(
            "Could not read the decision records: %s. Without them there are "
            "no challenger probabilities to compare against.", exc,
        )
        return EXIT_RECORDS_UNREADABLE

    if args.market:
        wanted = str(args.market).upper()
        record_rows = [r for r in record_rows if str(r.get("market") or "").upper() == wanted]

    report = compare_components(
        graded, record_rows, n_bins=args.bins,
        as_of=format_pacific_iso(now_pacific()),
    )
    report["filters"] = {"days": args.days, "market": args.market, "since": args.since}
    report["records_read"] = len(records)

    print(render(report))

    if args.json_out:
        target = Path(args.json_out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        logger.info("Report written to %s", target)
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
