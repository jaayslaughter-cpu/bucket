"""Research-only collector, explicit historical imports, and as-of exports.

Examples and deployment notes: docs/model_input_archive.md.
Run as ``python -m scripts.collect_model_inputs --help``.
"""

from __future__ import annotations

import argparse
import json
import signal
import threading
from pathlib import Path

from src.ingestion.model_input_archive import (
    InputArchive,
    InputError,
    build_absence_export,
    build_odds_export,
    timestamp,
)
from src.ingestion.model_input_collection import (
    collect_availability,
    collect_odds,
    current_season_start,
    import_odds,
    import_report,
)
from src.utils.volume import resolve_state_root


def _load(path):
    return json.loads(Path(path).read_text()) if path else []


def export_as_of(archive, cutoff, *, mode="system", max_age_hours=6, games=(), players=()):
    availability = archive.as_of("availability", cutoff, mode=mode, max_age_hours=max_age_hours)
    odds = archive.as_of("odds", cutoff, mode=mode, max_age_hours=max_age_hours)
    rows, unresolved = build_absence_export(availability, games, players)
    resolved_odds, unresolved_odds = build_odds_export(odds, games, players)
    if not availability:
        unresolved.append({"reason_code": "MISSING_OR_STALE_REPORT"})
    statuses = {}
    for snapshot in availability:
        for row in snapshot["rows"]:
            statuses[row["status"]] = statuses.get(row["status"], 0) + 1
    return {
        "research_only": True, "cutoff_utc": timestamp(cutoff).isoformat(),
        "selection_mode": mode, "max_age_hours": max_age_hours,
        "availability_snapshots": availability, "odds_snapshots": odds,
        "resolved_availability": rows, "unresolved_availability": unresolved,
        "resolved_odds": resolved_odds, "unresolved_odds": unresolved_odds,
        "coverage": {
            "report_status": availability[0]["status"] if availability else "MISSING_OR_STALE",
            "status_counts": statuses, "resolved_players": len(rows), "unresolved_rows": len(unresolved),
            "paired_odds": sum(len(s["rows"]) for s in odds),
            "resolved_odds": len(resolved_odds), "unresolved_odds": len(unresolved_odds),
            "odds_failure_snapshots": sum(s["status"] not in {"OK", "PARTIAL"} for s in odds),
            "identity_warning": "Provider odds IDs and player names are NOT panel identities.",
            "training_ready": False,
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="default: resolved state root/model_inputs")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("collect", "watch"):
        command = sub.add_parser(name)
        command.add_argument("--season-start", type=int, default=None)
        command.add_argument("--availability-only", action="store_true")
        command.add_argument("--max-events", type=int, default=12)
        command.add_argument("--report-manifest", type=Path,
                             help="JSON array of actual published official PDF URLs; bypass index fetch")
        if name == "watch":
            command.add_argument("--interval-seconds", type=int, default=900)
    imp = sub.add_parser("import-report")
    imp.add_argument("--url", required=True, help="exact published official report URL")
    imp.add_argument("--pdf", type=Path, help="optional local PDF; collection time is still now")
    odds_imp = sub.add_parser("import-odds")
    odds_imp.add_argument("--json", type=Path, required=True, help="documented PropLine event payload")
    exp = sub.add_parser("export")
    exp.add_argument("--cutoff", required=True, help="timezone-aware prediction cutoff")
    exp.add_argument("--mode", choices=["system", "published"], default="system")
    exp.add_argument("--max-age-hours", type=float, default=6)
    exp.add_argument("--games", type=Path, help="explicit dated game/team crosswalk JSON")
    exp.add_argument("--players", type=Path, help="explicit team/player crosswalk JSON")
    exp.add_argument("--output", type=Path, required=True, help="new JSON path; never overwritten")
    args = parser.parse_args(argv)
    archive = InputArchive(args.root or resolve_state_root()[0] / "model_inputs")
    try:
        if args.command == "export":
            output = export_as_of(archive, args.cutoff, mode=args.mode, max_age_hours=args.max_age_hours,
                                  games=_load(args.games), players=_load(args.players))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x") as out:
                json.dump(output, out, indent=2, allow_nan=False)
            print(json.dumps(output["coverage"], sort_keys=True))
            return 0
        if args.command == "import-report":
            result = import_report(archive, args.url, content=args.pdf.read_bytes() if args.pdf else None)
            print(json.dumps(result, sort_keys=True))
            return 0 if result["status"] == "OK" else 1
        if args.command == "import-odds":
            result = import_odds(archive, _load(args.json))
            print(json.dumps(result, sort_keys=True))
            return 0 if result["status"] == "OK" else 1
        if args.command == "watch" and args.interval_seconds < 60:
            raise InputError("minimum collection interval is 60 seconds")
        stop = threading.Event()
        if args.command == "watch":
            signal.signal(signal.SIGTERM, lambda *_: stop.set())
            signal.signal(signal.SIGINT, lambda *_: stop.set())
        while not stop.is_set():
            season = args.season_start if args.season_start is not None else current_season_start()
            availability = collect_availability(
                archive, season_start=season,
                report_urls=_load(args.report_manifest) if args.report_manifest else None,
            )
            odds = ({"status": "DISABLED"} if args.availability_only
                    else collect_odds(archive, max_events=args.max_events))
            print(json.dumps({"availability": availability, "odds": odds}, sort_keys=True), flush=True)
            failed = availability["status"] != "OK" or odds["status"] not in {"OK", "DISABLED"}
            if args.command != "watch":
                return int(failed)
            stop.wait(args.interval_seconds)
        return 0
    except (InputError, OSError, ValueError, KeyError) as exc:
        # Avoid echoing a caller's credential-bearing URL or provider body.
        print(json.dumps({"status": "FAILED", "error_type": type(exc).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
