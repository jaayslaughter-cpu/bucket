"""
scripts/build_shot_panel.py — attach the shot-location layer to the panel.

WHAT THIS IS FOR. src/features/pbp.py's shot-mix columns exist for 2025-26
only, and that module says why that is fatal rather than merely narrow: "A
feature that exists in the validation window and nowhere earlier is not a
feature, it is the shape of a leak, and compare_models_on_panel now refuses
one." This builds the same kind of columns from the NBA's own
``shotchartdetail`` export, which is available per season from 1996 — every
season of this panel.

THE COMPLETENESS CHECK RUNS FIRST AND CAN STOP THE BUILD, exactly as
build_pbp_panel.py's does, and for the reason recorded there: two event logs
supplied to this project named every game, spanned the right dates, carried
no duplicates, and still held 32% and 84% of the events. Rates built from a
partial log are biased by whatever was dropped and nothing downstream can
detect it. The panel's own FGA is the independent measurement, and a season
that fails is SKIPPED rather than silently blended in.

ONE SEASON AT A TIME, because the exports are per season and a bad one should
cost that season rather than the build. The per-season reports are printed and
returned so a caller can see which seasons are in and which are out.

Same-season shot values never reach the output: attach_shot_rolling_features
shifts before rolling, so a row carries what the player's PREVIOUS games
looked like.

RESEARCH ONLY.

Usage:
    python -m scripts.build_shot_panel --shots-dir data/external/shot_detail
    python -m scripts.build_shot_panel --shots-dir DIR --out panel_shots.parquet
"""

from __future__ import annotations

import argparse
import glob
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.shot_zones import (  # noqa: E402
    SHOT_FEATURE_COLS,
    ShotZoneError,
    attach_shot_rolling_features,
    check_shot_completeness,
    summarise_player_games,
)

logger = logging.getLogger("build_shot_panel")

#: The only columns read. A shotchartdetail export carries 24; reading the
#: rest costs memory across thirty seasons and buys nothing. Keep in step with
#: shot_zones.REQUIRED_SHOT_COLS.
SHOT_COLS = (
    "GAME_ID", "PLAYER_ID", "SHOT_DISTANCE", "SHOT_TYPE",
    "SHOT_ZONE_BASIC", "ACTION_TYPE", "SHOT_MADE_FLAG",
)


def load_shot_files(shots_dir: Path, pattern: str = "shotdetail_*.csv") -> dict[str, Path]:
    """Map a season label to its export, from the filename."""
    found = sorted(glob.glob(str(shots_dir / pattern)))
    if not found:
        raise ShotZoneError(
            f"DATA_NOT_AVAILABLE: no shot exports matching {pattern} in "
            f"{shots_dir}"
        )
    out: dict[str, Path] = {}
    for path in found:
        stem = Path(path).stem
        label = stem.rsplit("_", 1)[-1]
        out[label] = Path(path)
    return out


def build(
    panel: pd.DataFrame,
    shots_dir: Path,
    *,
    allow_incomplete: bool = False,
    pattern: str = "shotdetail_*.csv",
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """
    Summarise every usable season's shots, then roll them onto the panel.

    Returns (panel_with_features, per_season_reports). A season whose log
    fails the completeness check contributes NOTHING unless
    ``allow_incomplete`` is set, and the report says so either way.
    """
    files = load_shot_files(shots_dir, pattern)
    logger.info("Found %d shot export(s): %s", len(files), sorted(files))

    reports: list[dict[str, object]] = []
    summaries: list[pd.DataFrame] = []
    for label, path in sorted(files.items()):
        try:
            shots = pd.read_csv(path, usecols=list(SHOT_COLS), low_memory=False)
        except (OSError, ValueError) as exc:
            logger.error("Season %s unreadable (%s) — skipped.", label, exc)
            reports.append({"season": label, "status": "UNREADABLE", "reason": str(exc)})
            continue

        report = check_shot_completeness(shots, panel)
        report["season"] = label
        report["rows"] = int(len(shots))
        reports.append(report)

        if report["status"] != "OK" and not allow_incomplete:
            logger.error(
                "Season %s FAILED the completeness check (%s exact) and is "
                "EXCLUDED. Pass --allow-incomplete to include it anyway, and "
                "read src/features/pbp.py on why that is usually wrong.",
                label, report.get("exact_share"),
            )
            continue
        summaries.append(summarise_player_games(shots))

    if not summaries:
        raise ShotZoneError(
            "DATA_NOT_AVAILABLE: no season passed the completeness check, so "
            "there is nothing to attach. The panel is returned unchanged by "
            "the caller rather than enriched with a partial log."
        )

    summary = pd.concat(summaries, ignore_index=True)
    logger.info(
        "Shot summary across %d usable season(s): %d player-games.",
        len(summaries), len(summary),
    )
    return attach_shot_rolling_features(panel, summary, required=True), reports


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--panel", default="data/external/training_pack/panel.parquet",
                    help="Feature matrix to attach to.")
    ap.add_argument("--shots-dir", required=True,
                    help="Directory of shotdetail_{season}.csv exports.")
    ap.add_argument("--pattern", default="shotdetail_*.csv")
    ap.add_argument("--out", default=None,
                    help="Parquet path (default: <panel dir>/panel_shots.parquet)")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="Include a season that FAILS the FGA check. Rates "
                         "from a partial log look reasonable and are biased.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    logger.setLevel(logging.INFO)

    panel_path = Path(args.panel)
    if not panel_path.exists():
        print(f"ERROR: {panel_path} not found.", file=sys.stderr)
        return 2
    panel = pd.read_parquet(panel_path)
    logger.info("Panel: %d rows x %d cols", len(panel), panel.shape[1])

    try:
        out, reports = build(
            panel, Path(args.shots_dir),
            allow_incomplete=args.allow_incomplete, pattern=args.pattern,
        )
    except ShotZoneError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    target = Path(args.out) if args.out else panel_path.parent / "panel_shots.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(target, index=False)

    print(f"\nWrote {target}  ({len(out):,} rows x {out.shape[1]} columns)")
    print("\n  completeness by season (against the panel's own FGA):")
    for r in reports:
        mark = "ok  " if r.get("status") == "OK" else "SKIP"
        print(f"    {mark} {r.get('season')}  compared={r.get('compared')}  "
              f"exact={r.get('exact_share')}  mean|diff|={r.get('mean_abs_diff')}")
    print("\n  coverage of the shot-zone columns:")
    for col in SHOT_FEATURE_COLS:
        if col in out.columns:
            print(f"    {col:28s} {out[col].notna().mean():6.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
