"""
scripts/fetch_artifacts.py — pull trained artifacts onto the volume, or push them up.

THE STEP THAT WAS MISSING. `data/**` is gitignored, `.dockerignore` drops
`data/`, and there was no storage client — so a trained artifact reached a
container exactly one way: a person ran `scripts/seed_volume.py`. That is one
manual step in a pipeline that otherwise runs itself, and it is the step that
decides whether the slate scores anything at all.

TWO DIRECTIONS, and only one of them is automatic:

    python -m scripts.fetch_artifacts                  # status
    python -m scripts.fetch_artifacts --pull            # onto this volume
    python -m scripts.fetch_artifacts --push --from DIR # from a training machine

`--pull` is what `scripts/start.sh` runs at boot. `--push` is a human action:
it replaces the artifact that produced every probability now in the database,
so it is not something a scheduler should do while nobody is looking.

AN UNCONFIGURED STORE IS NOT AN ERROR. Exit 0 and a line saying so. A
deployment that seeds by hand is a choice, and failing the boot over it would
break every deployment that was working yesterday.

RESEARCH_ONLY. This moves model files. No odds, no wager.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.artifact_store import (  # noqa: E402
    ENV_STORE_BUCKET,
    ENV_STORE_KEY,
    ENV_STORE_PREFIX,
    ENV_STORE_URL,
    ArtifactStoreError,
    fetch_artifacts,
    is_configured,
    list_remote,
    upload_artifacts,
)
from src.utils.volume import artifact_dir_on, resolve_state_root  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Move trained artifacts between object storage and the "
                    "volume. RESEARCH_ONLY — moves files, prices nothing."
    )
    ap.add_argument("--pull", action="store_true",
                    help="Download complete families onto the state root. What "
                         "scripts/start.sh runs at boot.")
    ap.add_argument("--push", action="store_true",
                    help="Upload complete families from --from. A human action: "
                         "it replaces what produced the probabilities already "
                         "in the database.")
    ap.add_argument("--from", dest="source", default=None, metavar="DIR",
                    help="With --push: where the trained artifacts are.")
    ap.add_argument("--markets", default=None,
                    help="Comma-separated subset, e.g. PTS,REB. Default: all.")
    ap.add_argument("--force", action="store_true",
                    help="With --pull: re-download a family that is already "
                         "current. The local copy produced every probability "
                         "now in the database, so this is not the default.")
    ap.add_argument("--json", action="store_true", help="One JSON object.")
    args = ap.parse_args(argv)

    markets = (
        [m.strip().upper() for m in args.markets.split(",") if m.strip()]
        if args.markets else None
    )

    if not is_configured():
        message = (
            f"No artifact store configured (${ENV_STORE_URL} and "
            f"${ENV_STORE_KEY}). Nothing fetched; seed the volume by hand with "
            f"`python -m scripts.seed_volume --from DIR --apply`. This is not "
            f"an error."
        )
        print(json.dumps({"configured": False, "note": message})
              if args.json else message)
        return 0

    root, how = resolve_state_root()
    dest = artifact_dir_on(root)

    try:
        if args.push:
            if not args.source:
                print("ERROR: --push needs --from DIR", file=sys.stderr)
                return 2
            report = upload_artifacts(Path(args.source), markets=markets)
        elif args.pull:
            report = fetch_artifacts(dest, markets=markets, force=args.force)
        else:
            report = {
                "configured": True,
                "bucket": (
                    f"{Path(args.source).name}" if args.source else None
                ),
                "destination": str(dest),
                "state_root": f"{root} ({how})",
                "remote": [
                    {"name": o.name, "size": o.size, "updated_at": o.updated_at}
                    for o in list_remote()
                ],
            }
    except ArtifactStoreError as exc:
        # The message is already redacted by the store; print it and stop.
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3

    if args.json:
        print(json.dumps(report, indent=2, default=str))
        return 0

    if args.push or args.pull:
        verb = "uploaded" if args.push else "fetched"
        print(f"{verb}: {report.get(verb) or report.get('fetched') or []}")
        for key in ("skipped", "refused", "errors"):
            if report.get(key):
                print(f"{key}: {report[key]}")
    else:
        print(f"state root : {report['state_root']}")
        print(f"destination: {report['destination']}")
        print(f"bucket     : ${ENV_STORE_BUCKET} / ${ENV_STORE_PREFIX}")
        remote = report["remote"]
        print(f"remote     : {len(remote)} object(s)")
        for obj in remote[:24]:
            print(f"  {obj['name']}  {obj['size']}  {obj['updated_at']}")

    # A pull that fetched nothing and errored is worth a nonzero exit, so a
    # deploy script can tell "nothing to do" from "could not do it".
    if args.pull and report.get("errors") and not report.get("fetched"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
