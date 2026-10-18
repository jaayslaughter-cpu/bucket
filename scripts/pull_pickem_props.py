"""
scripts/pull_pickem_props.py — capture the pick'em boards, or inspect one.

THREE MODES, and the middle one is why this exists:

    --describe <file>    print a recorded board response's REAL paths
    --from-file <file>   run the mapping against that file, write nothing
    (neither)            read the live boards and archive each snapshot

`--describe` is the answer to the only genuinely unknown thing in this
subsystem. These are undocumented internal endpoints; nobody working on this
repository has seen a response, because the build container cannot reach
either host. Rather than write plausible key names into
`config/pickem_sources.yaml` and call them the provider's schema -- inventing
an API payload, which is what produced the 2026-10-18 report of four broken
scrapers -- the paths are configuration, and this prints the ones a real
response actually contains so filling them in is reading.

RESEARCH_ONLY. Captures posted lines and archives them with a timestamp. It
prices nothing, recommends nothing and places nothing.

Usage:
    python -m scripts.pull_pickem_props
    python -m scripts.pull_pickem_props --source underdog
    python -m scripts.pull_pickem_props --describe /tmp/board.json
    python -m scripts.pull_pickem_props --source underdog --from-file /tmp/board.json
    python -m scripts.pull_pickem_props --no-archive

Exit codes:
    0  every requested source was read (an empty board is a read)
    1  at least one source could not be read
    3  every requested source is unconfigured — nothing was attempted
    4  bad arguments
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logger = logging.getLogger("pull_pickem_props")

EXIT_OK = 0
EXIT_SOURCE_FAILED = 1
EXIT_UNCONFIGURED = 3
EXIT_BAD_ARGS = 4


def _describe(path: str) -> int:
    from src.ingestion.pickem_schema import describe_payload

    file = Path(path)
    try:
        payload = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"cannot read {file}: {exc}")
        return EXIT_BAD_ARGS

    paths = describe_payload(payload)
    print(f"\n{file}: {len(paths)} leaf path(s). Copy the ones you want into")
    print("config/pickem_sources.yaml. `rows` is the path to the ARRAY of lines;")
    print("each `fields` path is read RELATIVE to one row of that array.\n")
    for line in paths:
        print(f"    {line}")

    # Where the arrays are matters more than any single leaf: `rows` has to
    # point at one, and a normalised payload has several.
    arrays: list[str] = []

    def find_arrays(node: object, prefix: str = "") -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                here = f"{prefix}.{key}" if prefix else str(key)
                if isinstance(value, list):
                    arrays.append(f"{here}  ({len(value)} item(s))")
                find_arrays(value, here)

    find_arrays(payload)
    if arrays:
        print("\n  arrays in this payload — one of these is `rows`:")
        for entry in arrays:
            print(f"    {entry}")
    print()
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Capture pick'em boards, or inspect a recorded response.",
    )
    parser.add_argument("--source", action="append", default=None,
                        help="repeatable; default every source with a client")
    parser.add_argument("--describe", default=None, metavar="FILE",
                        help="print a recorded response's real paths and exit")
    parser.add_argument("--from-file", default=None, metavar="FILE",
                        help="run one source's mapping against a recorded response")
    parser.add_argument("--config", default=None, help="mapping file to use")
    parser.add_argument("--store-root", default=None, help="snapshot archive root")
    parser.add_argument("--no-archive", action="store_true",
                        help="read the boards but write nothing")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    if args.describe:
        return _describe(args.describe)

    from src.ingestion.pickem import SOURCES, pull_pickem_boards
    from src.ingestion.pickem_schema import (
        STATUS_NOT_CONFIGURED,
        PickemConfigError,
    )

    sources = tuple(args.source) if args.source else SOURCES
    unknown = [s for s in sources if s not in SOURCES]
    if unknown:
        print(f"no pick'em client for {unknown}; have {list(SOURCES)}")
        return EXIT_BAD_ARGS
    if args.from_file and len(sources) != 1:
        print("--from-file reads one recorded response; name exactly one --source")
        return EXIT_BAD_ARGS

    try:
        snapshots = pull_pickem_boards(
            sources,
            config_path=args.config,
            archive=not (args.no_archive or args.from_file),
            store_root=args.store_root,
            from_file=args.from_file,
        )
    except PickemConfigError as exc:
        print(f"\n{exc}\n")
        return EXIT_UNCONFIGURED

    width = max(len(s.source) for s in snapshots)
    print()
    print(f"{'SOURCE'.ljust(width)}  STATUS            LINES  DETAIL")
    print("-" * 100)
    for snap in snapshots:
        http = f"[{snap.http_status}] " if snap.http_status else ""
        print(
            f"{snap.source.ljust(width)}  {snap.status:<16}  "
            f"{len(snap.lines):>5}  {http}{(snap.message or '')[:200]}"
        )
    for snap in snapshots:
        if snap.lines:
            print(f"\n--- {snap.source}: first {min(5, len(snap.lines))} line(s)")
            for line in snap.lines[:5]:
                mult = "" if line.payout_multiplier is None else f" x{line.payout_multiplier}"
                print(
                    f"    {line.player_name:24} {line.market:4} {line.line:>6}{mult}"
                    f"  captured={line.captured_at_utc}"
                )
    print()

    unconfigured = [s for s in snapshots if s.status == STATUS_NOT_CONFIGURED]
    failed = [s for s in snapshots if not s.ok and s.status != STATUS_NOT_CONFIGURED]

    if len(unconfigured) == len(snapshots):
        print(
            "NOTHING WAS ATTEMPTED: every requested source is unconfigured. This "
            "repository ships no board URLs and no field paths on purpose — see "
            "the header of config/pickem_sources.yaml for the three steps, which "
            "start with recording one board response and running --describe."
        )
        return EXIT_UNCONFIGURED
    if failed:
        print(f"COULD NOT READ: {', '.join(s.source for s in failed)}")
        return EXIT_SOURCE_FAILED
    print("Every configured source was read.")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
