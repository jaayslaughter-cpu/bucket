"""
scripts/build_pbp_panel.py — attach the play-by-play layer to the panel.

WHY THIS EXISTS AS A SCRIPT. panel_pbp.parquet is the frame the models
actually train on, and until now it was produced by an ad-hoc script that
lived only in a scratch directory. That made the training input
unreproducible: nothing in the repository said how the PBP_* columns got
there, and when the container was recycled the recipe went with it.

WHAT IT REFUSES TO DO. A partial event log is the failure mode this whole
area is shaped around. Twice now a log has named every game, spanned the
right dates, carried no duplicates, and still held only a fraction of each
game's events -- once at 32% and once at 84% coverage, both of which
produce shot-distribution rates that look entirely reasonable. The only
thing that separates "looks fine" from "is fine" is the box score's own
independent count, so this script runs that check FIRST and stops on a
failing season unless explicitly told to continue. Rates built from a
partial log are biased by whatever was dropped, and nothing downstream can
detect it.

Same-game PBP_* values never reach the output. attach_pbp_rolling_features
shifts before rolling, so a row carries what the player's PREVIOUS games
looked like.
"""

from __future__ import annotations

import argparse
import glob
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.pbp import (  # noqa: E402
    PbpFeatureError,
    attach_pbp_rolling_features,
    check_log_completeness,
    prepare_events,
    reconstruct_on_court,
    summarise_player_games,
    validate_on_court_against_minutes,
)

logger = logging.getLogger("build_pbp_panel")


# The only columns src/features/pbp.py reads, plus actionNumber for the
# duplicate check below. The raw export carries about ninety; a full
# multi-season log is millions of rows, so reading the rest costs gigabytes
# of resident memory and buys nothing. Keep this in step with pbp.py.
EVENT_COLS = (
    "gameId", "actionNumber", "actionType", "subType", "period", "clock",
    "orderNumber", "personId", "teamId", "assistPersonId",
    "scoreHome", "scoreAway", "possession", "shotDistance", "shotResult",
)


def load_event_parts(pbp_dir: Path, pattern: str = "*.csv") -> pd.DataFrame:
    """Read every event-log part in a directory into one frame."""
    parts = sorted(glob.glob(str(pbp_dir / pattern)))
    if not parts:
        raise PbpFeatureError(f"DATA_NOT_AVAILABLE: no event-log parts in {pbp_dir}")

    header = pd.read_csv(parts[0], nrows=0).columns
    missing = [c for c in EVENT_COLS if c not in header]
    if missing:
        # Narrowing the read must never silently drop a column pbp.py needs.
        raise PbpFeatureError(
            f"DATA_NOT_AVAILABLE: event log missing {missing}. Found: "
            f"{sorted(header)[:20]}..."
        )
    frames = [pd.read_csv(p, usecols=list(EVENT_COLS), low_memory=False)
              for p in parts]
    events = pd.concat(frames, ignore_index=True)
    del frames
    logger.info("Read %d part(s): %d raw events (%d of %d columns)",
                len(parts), len(events), len(EVENT_COLS), len(header))

    if {"gameId", "actionNumber"}.issubset(events.columns):
        dupes = int(events.duplicated(subset=["gameId", "actionNumber"]).sum())
        if dupes:
            # Overlapping parts would double-count shots and inflate every
            # rate. Drop rather than warn: the log is the input to a count.
            events = events.drop_duplicates(subset=["gameId", "actionNumber"])
            logger.warning("Dropped %d duplicate event(s) on (gameId, actionNumber)", dupes)
        else:
            logger.info("No duplicate events on (gameId, actionNumber)")
    return events


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", default="data/external/training_pack")
    ap.add_argument("--panel", default=None,
                    help="Input panel parquet (default: <pack>/panel.parquet)")
    ap.add_argument("--pbp-dir", default=None,
                    help="Directory of event-log CSV parts (default: <pack>/pbp)")
    ap.add_argument("--out", default=None,
                    help="Output parquet (default: <pack>/panel_pbp.parquet)")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="Build anyway when a season fails the completeness "
                         "check. The resulting rates are biased; only for "
                         "diagnosis, never for a model you intend to trust.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    pack = Path(args.pack)
    panel_path = Path(args.panel) if args.panel else pack / "panel.parquet"
    pbp_dir = Path(args.pbp_dir) if args.pbp_dir else pack / "pbp"
    out_path = Path(args.out) if args.out else pack / "panel_pbp.parquet"

    panel = pd.read_parquet(panel_path)
    logger.info("Panel: %d rows, %d columns from %s",
                len(panel), panel.shape[1], panel_path)

    events = load_event_parts(pbp_dir)

    # FIRST, before anything is computed from the log. See module docstring.
    # check_log_completeness logs a line per season itself; do not repeat it.
    report = check_log_completeness(events, panel)
    if report["failing"]:
        if not args.allow_incomplete:
            logger.error(
                "REFUSING to build: %s failed the completeness check. Supply the "
                "remaining parts, or pass --allow-incomplete to build a frame "
                "whose rates are knowingly biased.",
                ", ".join(report["failing"]),
            )
            return 1
        logger.warning(
            "Building with INCOMPLETE season(s) %s because --allow-incomplete "
            "was passed. Rates for these seasons are biased.",
            ", ".join(report["failing"]),
        )

    # Prepare ONCE. prepare_events is idempotent, so the defensive call
    # inside summarise_player_games returns this same frame rather than
    # copying millions of rows a second time.
    events = prepare_events(events)

    summaries = summarise_player_games(events, panel)

    # The on-court reconstruction rests on an assumption about who started.
    # Check it against the box score's minutes rather than believing it.
    check = validate_on_court_against_minutes(
        reconstruct_on_court(events), panel
    )
    if not check.get("n"):
        logger.warning("on-court reconstruction could not be checked: no overlap "
                       "between the log and the panel on (gameId, personId)")

    before = len(panel)
    out = attach_pbp_rolling_features(panel, summaries)
    if len(out) != before:
        raise PbpFeatureError(
            f"pbp join changed the row count: {before} -> {len(out)}. The join "
            "must not fan out or drop rows."
        )

    pbp_cols = [c for c in out.columns if c.startswith("PBP_")]
    if "SEASON" in out.columns and pbp_cols:
        per = out.groupby("SEASON")[pbp_cols[0]].apply(lambda s: 100 * s.notna().mean())
        for season, pct in per.items():
            logger.info("  %s: %.1f%% of rows have pbp", season, pct)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    logger.info("Wrote %s (%d rows, %d columns)", out_path, len(out), out.shape[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
