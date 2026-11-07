"""
scripts/analyze_losses.py — what the losses are made of, from the ledger.

WHAT IT IS FOR. Nothing in this project said WHERE the model was wrong.
`settlement.metrics` counts W-L-P, ROI and CLV; `settlement.calibration`
scores whether the probabilities were honest. Neither answers the question a
season-long feedback loop exists to answer: is the error one-directional, is
it concentrated in a market, and does it shrink as a player accumulates games
in the current season?

THE EARLY-SEASON QUESTION IS THE REASON THIS EXISTS. A projection built
mostly from last season's baseline may be wrong in a way one built on twenty
current games is not. This bins residuals by how many games the player had
ALREADY PLAYED that season and reports the bias in each bin, so that stops
being a hunch. The arithmetic is in `src/settlement/loss_analysis.py`, which
is pure; this file is the database read and the printing.

IT MEASURES AND RECOMMENDS. IT RETUNES NOTHING. `models/labels.py` already
excluded the halflife shrink family on measurement -- `{M}_HL_SHRINK`
correlates 0.987-0.990 with `{M}_SEASON` across 214,381 rows -- so "add
shrinkage" is a decision that was already taken with numbers, and reversing
it needs numbers, not a script that edits a feature list. Fitting anything to
the residuals of the rows you then report on has no holdout in it at all.

EXIT STATUS IS ABOUT THE ANALYSIS, NOT ABOUT THE MODEL. 0 when the report was
produced, INCLUDING when it honestly reports that there is nothing to analyse
yet -- that is the expected state on the first day a model exists. 2 when the
ledger could not be read. A non-zero exit must mean "this did not run", or a
scheduled copy of it becomes an alarm nobody reads.

Usage:
    python -m scripts.analyze_losses
    python -m scripts.analyze_losses --days 7 --market PTS
    python -m scripts.analyze_losses --json /app/data/loss_analysis.json

RESEARCH_ONLY. Reads a ledger of graded research predictions and reports
arithmetic on them. It prices nothing, sizes nothing and places nothing.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger("analyze_losses")

EXIT_OK = 0
EXIT_LEDGER_UNREADABLE = 2

#: How far back the panel is read when counting prior games. It only has to
#: reach the start of the season the graded rows fall in; 400 matches the
#: live slate's own window.
PANEL_LOOKBACK_DAYS = 400


def prior_game_counts(panel: Any, keys: set[tuple[Any, Any]]) -> dict[tuple[Any, Any], int]:
    """
    ``(player_id, game_date) -> games that player had already played that season``.

    COUNTED FROM THE PANEL, NOT FROM THE LEDGER. The ledger holds only rows
    the model scored; a player's games before the model existed are still
    games he played, and they are what his baseline was built from.

    SEASON-SCOPED, via `features.season.season_start_year`, because "games
    played" that silently spans a summer is not the quantity the early-season
    question is about. STRICTLY BEFORE the graded date, so a count can never
    include the game being predicted.
    """
    import pandas as pd

    from src.features.season import season_start_year

    if panel is None or getattr(panel, "empty", True):
        return {}
    need = {"PLAYER_ID", "GAME_DATE"}
    missing = need - set(panel.columns)
    if missing:
        logger.warning(
            "Panel has no %s, so prior games cannot be counted; the report "
            "will say the early-season question was not measured.",
            ", ".join(sorted(missing)),
        )
        return {}

    frame = panel[["PLAYER_ID", "GAME_DATE"]].copy()
    frame["GAME_DATE"] = pd.to_datetime(frame["GAME_DATE"])
    frame["SEASON"] = season_start_year(frame["GAME_DATE"])

    out: dict[tuple[Any, Any], int] = {}
    by_player: dict[Any, Any] = {
        pid: group.sort_values("GAME_DATE")
        for pid, group in frame.groupby("PLAYER_ID", sort=False)
    }
    for player_id, game_date in keys:
        group = by_player.get(player_id)
        if group is None:
            continue
        stamp = pd.to_datetime(game_date)
        season = season_start_year(pd.Series([stamp])).iloc[0]
        same_season = group[group["SEASON"] == season]
        out[(player_id, game_date)] = int((same_season["GAME_DATE"] < stamp).sum())
    return out


def render(report: dict[str, Any]) -> str:
    """The report as a person reads it. One block per section, numbers aligned."""
    lines: list[str] = []
    add = lines.append
    add("PropIQ loss analysis — RESEARCH_ONLY")
    add("")
    rec = report["record"]
    add(f"  graded rows      {report['n_graded']} of {report['n_rows_in']} read")
    add(f"  with a residual  {report['n_with_residual']}")
    strike = rec["strike_rate"]
    add(
        f"  record           {rec['WIN']}-{rec['LOSS']}-{rec['PUSH']}"
        + (f"  (strike rate {strike:.3f} on {rec['decided']} decided)" if strike is not None else "")
    )
    if report["excluded"]:
        add("  excluded         " + ", ".join(
            f"{reason}: {count}" for reason, count in sorted(report["excluded"].items())
        ))
    add("")

    def block(title: str, body: dict[str, Any]) -> None:
        add(f"  {title}")
        if not body:
            add("      (none)")
            return
        for key, stats in body.items():
            if stats.get("insufficient"):
                add(f"      {key:<10} n={stats['n']:<6} withheld (needs {stats['needs']})")
            elif stats.get("n"):
                add(
                    f"      {key:<10} n={stats['n']:<6} "
                    f"bias {stats['mean_bias']:+.2f}  "
                    f"MAE {stats['mae']:.2f}  RMSE {stats['rmse']:.2f}"
                )
            else:
                add(f"      {key:<10} n=0")

    overall = report["residuals_overall"]
    block("overall", {"actual-proj": overall})
    add("")
    block("by market", report["residuals_by_market"])
    add("")
    block("by side", report["residuals_by_side"])
    add("")
    block("by outcome", report["residuals_by_outcome"])
    add("")

    prior = report["residuals_by_prior_games"]
    if prior.get("status") == "OK":
        block("by prior games played this season", prior["bins"])
        if prior.get("unknown_count"):
            add(f"      ({prior['unknown_count']} row(s) had no countable prior games)")
    else:
        add("  by prior games played this season")
        add(f"      NOT MEASURED — {prior.get('reason')}")
    add("")

    calib = report.get("calibration")
    if calib:
        add("  calibration (settlement.calibration, same report the gate reads)")
        add(f"      {json.dumps(calib, default=str)}")
        add("")

    add("  findings")
    for finding in report["findings"]:
        add(f"      - {finding}")
    add("")
    add("  Nothing above is a recommendation to act, a size, or a claim of")
    add("  profitability. Residuals are arithmetic on graded research rows.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m scripts.analyze_losses",
        description="Residual and calibration analysis over graded prop results.",
    )
    ap.add_argument("--days", type=int, default=30,
                    help="How far back to read graded rows (default 30).")
    ap.add_argument("--market", default=None, help="PTS|REB|AST; default all.")
    ap.add_argument("--json", dest="json_out", default=None,
                    help="Also write the report as JSON to this path.")
    ap.add_argument("--no-prior-games", action="store_true",
                    help="Skip the panel read that counts prior games played.")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    from src.settlement.loss_analysis import analyse_losses
    from src.utils.timezones import format_pacific_iso, now_pacific

    try:
        from src.db.repository import load_graded_prop_residual_rows

        rows = load_graded_prop_residual_rows(
            lookback_days=args.days, market=args.market
        )
    except Exception as exc:  # noqa: BLE001 — the ledger read is the one hard dependency
        logger.error(
            "Could not read the graded ledger: %s. This is a database or "
            "configuration fault, not an empty ledger -- an empty one reports "
            "itself and exits 0.", exc,
        )
        return EXIT_LEDGER_UNREADABLE

    priors: dict[tuple[Any, Any], int] | None = None
    if not args.no_prior_games and rows:
        keys = {(r.get("nba_player_id"), r.get("game_date")) for r in rows}
        try:
            from src.db.repository import load_player_panel

            panel = load_player_panel(lookback_days=PANEL_LOOKBACK_DAYS)
            priors = prior_game_counts(panel, keys)
        except Exception as exc:  # noqa: BLE001 — a bin is worth less than the report
            # DEGRADES, DOES NOT FAIL. The residuals are still worth printing
            # without the early-season split, and `analyse_losses` reports the
            # section as NOT_MEASURED rather than as a negative answer.
            logger.warning(
                "Prior-game counts unavailable (%s); the early-season section "
                "will report that it was not measured.", exc,
            )
            priors = None

    report = analyse_losses(
        rows, prior_games=priors, as_of=format_pacific_iso(now_pacific())
    )
    report["filters"] = {"days": args.days, "market": args.market}

    # THE GATE'S OWN CALIBRATION REPORT, not a second implementation of it.
    try:
        from src.settlement.calibration import calibration_from_graded_rows

        report["calibration"] = calibration_from_graded_rows(rows)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Calibration section unavailable: %s", exc)

    print(render(report))

    if args.json_out:
        target = Path(args.json_out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        logger.info("Report written to %s", target)
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
