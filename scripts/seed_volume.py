"""
scripts/seed_volume.py — put the trained artifacts where a redeploy keeps them.

THE PROBLEM THIS SOLVES. A fresh container's volume is empty. `score_prob_over`
with no artifact returns an all-null Series with a reason: EVERY ROW ABSTAINS,
the slate job exits 0, the board is empty and nothing looks broken. Seeding the
volume is therefore a step of the deploy, and it was a step with no tool — the
instructions said "train into the volume, or put the artifact plus its
.meta.json sidecar there", and "there" was a path you had to derive from
`config/model_comparison.yaml` by hand.

WHAT IT REFUSES TO DO:

  * copy an artifact with no `.meta.json` sidecar. The sidecar holds the
    `feature_cols` used at training time; scoring is refused without it, so an
    artifact without one is not a model, it is a file. Copying it would turn a
    visible "nothing resolved" into "resolved and unusable".
  * copy the booster alone. ONE MARKET IS SEVERAL FILES. `xgb_adapter.save`
    writes `xgboost_PTS.json` (the classifier), `xgboost_PTS.mean.json` (the
    mean head) and `xgboost_PTS.meta.json`, and `load` warns and sets
    `mean_model = None` when the mean head is absent — so a volume seeded with
    the booster alone returns NULL PROJECTIONS beside live probabilities, which
    is the failure that looks most like a working deployment. Each accepted
    market copies its whole family (`*_PTS.*`), which also carries
    `distribution_PTS.json` and the CatBoost pair when they are there.
  * copy an artifact fit on fewer than `MIN_PLAUSIBLE_TRAIN_ROWS` rows unless
    `--allow-demo` is passed. The demo artifacts in this repository were fit on
    968 synthetic rows; scoring a live slate with one is worse than abstaining.
  * overwrite. `--force` is required, because the artifact on the volume is the
    one that produced every probability now in the database.

IT DOES NOT INVENT A PATH. The destination is
`src.utils.volume.artifact_dir_on(state_root)`, which reads `artifacts_dir`
from the comparison config — the same answer `main.resolve_model_artifact`
reaches. A seeding tool that wrote to a directory of its own choosing
(`/app/data/models/`, say) would report success and leave the resolver finding
nothing.

RESEARCH_ONLY project. This copies files. No odds, no wager, no sizing.

Usage:
    python -m scripts.seed_volume                                   # status
    python -m scripts.seed_volume --from data/external/model_runs/comparison
    python -m scripts.seed_volume --from DIR --apply
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.volume import (  # noqa: E402
    ENV_RAILWAY_VOLUME,
    ENV_STATE_DIR,
    artifact_dir_on,
    resolve_state_root,
)

#: An artifact and the sidecar that makes it scoreable. Both or neither.
SIDECAR_SUFFIX = ".meta.json"


def train_row_count(sidecar: Path) -> int | None:
    """What the sidecar says it was fit on, or None if it will not say."""
    try:
        value = json.loads(sidecar.read_text(encoding="utf-8")).get("train_row_count")
    except Exception:  # noqa: BLE001 — an unreadable sidecar is "it will not say"
        return None
    return value if isinstance(value, int) else None


def pairs_in(directory: Path) -> list[tuple[Path, Path | None]]:
    """
    Every `xgboost_*.json` in `directory`, with its sidecar if it has one.

    The sidecar is returned as None rather than skipped so the caller can SAY
    which artifacts are unusable. Silently ignoring them is how a seeded volume
    ends up one market short with no line in the log about it.
    """
    if not directory.is_dir():
        return []
    out: list[tuple[Path, Path | None]] = []
    for path in sorted(directory.glob("xgboost_*.json")):
        # `xgboost_PTS.json` is a candidate; `xgboost_PTS.mean.json` and
        # `xgboost_PTS.meta.json` are its companions and are copied WITH it by
        # `family_for`. Listing them here as candidates reported three
        # "REFUSED: no sidecar" lines per seeding run for files that were
        # already being copied — alarming, and about nothing.
        if path.name.count(".") != 1:
            continue
        sidecar = path.with_suffix(SIDECAR_SUFFIX)
        out.append((path, sidecar if sidecar.exists() else None))
    return out


def market_of(artifact: Path) -> str:
    """`xgboost_PTS.json` -> `PTS`. The market is the token after the family."""
    return artifact.name.split("_", 1)[1].split(".", 1)[0]


def family_for(directory: Path, market: str) -> list[Path]:
    """
    Every file in `directory` belonging to `market`, across model families.

    ONE MARKET IS SEVERAL FILES and the set is not fixed: XGBoost writes a
    booster, a mean head and a sidecar; CatBoost writes `.cbm` plus `.mean.cbm`;
    the distribution model writes its own JSON. Matching `*_{market}.*` rather
    than naming them keeps a family added later from being silently left behind
    — which would surface as null projections beside live probabilities, the
    failure that looks most like a working deployment.
    """
    return sorted(p for p in directory.glob(f"*_{market}.*") if p.is_file())


def describe_destination() -> dict[str, object]:
    """Where state and artifacts resolve to, and what is already there."""
    root, how = resolve_state_root()
    dest = artifact_dir_on(root)
    present = []
    for artifact, sidecar in pairs_in(dest):
        present.append({
            "artifact": artifact.name,
            "sidecar": sidecar.name if sidecar else None,
            "train_row_count": train_row_count(sidecar) if sidecar else None,
        })
    return {
        "state_root": str(root),
        "resolved_via": how,
        "artifacts_dir": str(dest),
        "artifacts_dir_exists": dest.is_dir(),
        "present": present,
    }


def _print_status(info: dict[str, object]) -> None:
    print(f"state root     : {info['state_root']}  ({info['resolved_via']})")
    print(f"artifacts dir  : {info['artifacts_dir']}"
          + ("" if info["artifacts_dir_exists"] else "   (does not exist yet)"))
    present = info["present"]
    assert isinstance(present, list)
    if not present:
        print("artifacts      : NONE. Every row will abstain, and the slate job "
              "will still exit 0.")
        return
    print("artifacts      :")
    for row in present:
        rows = row["train_row_count"]
        note = (
            "no sidecar — NOT SCOREABLE" if row["sidecar"] is None
            else f"fit on {rows} rows" if rows is not None
            else "sidecar unreadable"
        )
        print(f"  {row['artifact']:<28} {note}")


def seed(
    source: Path,
    *,
    apply: bool,
    force: bool,
    allow_demo: bool,
) -> tuple[int, list[str]]:
    """
    Copy usable artifact/sidecar pairs from `source` onto the state root.

    Returns `(exit_code, lines)`. Nothing is written unless `apply` is true, so
    the default run is a report of what WOULD be copied — a seeding step that
    writes by default is one path typo away from filling the wrong directory.
    """
    from scheduler_worker import MIN_PLAUSIBLE_TRAIN_ROWS

    root, how = resolve_state_root()
    dest = artifact_dir_on(root)
    lines = [f"source      : {source}",
             f"destination : {dest}  (state root {root} via {how})"]

    found = pairs_in(source)
    if not found:
        lines.append(f"ERROR: no xgboost_*.json in {source}.")
        return 2, lines

    planned: list[tuple[str, list[Path]]] = []
    refused = 0
    for artifact, sidecar in found:
        market = market_of(artifact)
        if sidecar is None:
            lines.append(f"  REFUSED {artifact.name}: no {SIDECAR_SUFFIX} sidecar, "
                         f"so scoring with it is refused anyway.")
            refused += 1
            continue
        rows = train_row_count(sidecar)
        if rows is not None and rows < MIN_PLAUSIBLE_TRAIN_ROWS and not allow_demo:
            lines.append(
                f"  REFUSED {artifact.name}: fit on {rows} rows, under "
                f"{MIN_PLAUSIBLE_TRAIN_ROWS} — almost certainly the synthetic "
                f"demo artifact. Pass --allow-demo if that is really intended."
            )
            refused += 1
            continue
        family = family_for(source, market)
        clashes = [f.name for f in family if (dest / f.name).exists()]
        if clashes and not force:
            lines.append(f"  REFUSED {market}: {len(clashes)} file(s) already on "
                         f"the volume ({clashes[:3]}). They produced every "
                         f"probability now in the database; pass --force to "
                         f"replace them.")
            refused += 1
            continue
        planned.append((market, family))

    if not planned:
        lines.append("Nothing to copy.")
        return (1 if refused else 0), lines

    for market, family in planned:
        lines.append(f"  {'copy' if apply else 'would copy'} {market}: "
                     f"{[f.name for f in family]}")
        if apply:
            dest.mkdir(parents=True, exist_ok=True)
            for f in family:
                shutil.copy2(f, dest / f.name)

    if not apply:
        lines.append("Nothing was written. Pass --apply to copy.")
    return (1 if refused else 0), lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Report or seed the durable state root's model artifacts. "
                    "RESEARCH_ONLY — copies files, prices nothing."
    )
    ap.add_argument("--from", dest="source", default=None, metavar="DIR",
                    help="Copy artifact/sidecar pairs from this directory. "
                         "Without it, this only reports.")
    ap.add_argument("--apply", action="store_true",
                    help="With --from: actually copy. Without it, nothing is "
                         "written.")
    ap.add_argument("--force", action="store_true",
                    help="Replace an artifact already on the volume.")
    ap.add_argument("--allow-demo", action="store_true",
                    help="Permit an artifact fit on implausibly few rows. It "
                         "will be scored with; that is the point of refusing.")
    ap.add_argument("--json", action="store_true",
                    help="Status as one JSON object. Ignored with --from.")
    args = ap.parse_args(argv)

    if args.source is None:
        info = describe_destination()
        if args.json:
            print(json.dumps(info, indent=2, default=str))
        else:
            _print_status(info)
            print()
            print(f"Set {ENV_STATE_DIR} or {ENV_RAILWAY_VOLUME} to point this "
                  f"elsewhere. Copy with --from DIR --apply.")
        return 0

    code, lines = seed(
        Path(args.source), apply=args.apply, force=args.force,
        allow_demo=args.allow_demo,
    )
    for line in lines:
        print(line)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
