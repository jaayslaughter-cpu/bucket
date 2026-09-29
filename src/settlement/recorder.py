"""Turn a slate's projections into PENDING ``prop_results`` rows.

THE HOLE THIS FILLS. ``prop_results`` had a table, a grader
(``settlement/runner.py``, which selects ``outcome_status == 'PENDING'``) and a
metrics layer aggregating W/L/PUSH, stake, profit and both CLV columns — and no
writer anywhere. ``PropResult(`` appeared as a constructor in no module. Every
figure the metrics layer could produce was therefore an aggregate over zero
rows, and a live test would have generated nothing to evaluate, which is the
whole reason for running one.

THESE ROWS ARE PREDICTIONS, NOT WAGERS, and the distinction is enforced rather
than asserted: ``stake_units`` is left NULL by this writer and there is no
parameter to set it. A row here says "the model said OVER 25.5 at this line on
this date"; it says nothing about money. ROI over these rows is undefined until
someone records a stake by hand, and that is correct — PropIQ does not place or
size wagers. What the rows do give is a graded strike rate and, where the line
was priced, CLV: the leakage-safe forward evidence nothing else in the pipeline
retains.

WHY A ROW IS SKIPPED RATHER THAN GUESSED. Every skip below is a case where the
alternative is a confidently wrong ledger entry:

  no line          a prediction with no number to be over or under is not a
                   prop. LINE is null whenever the exact (player, market) join
                   found nothing, which is most rows in a normal slate.
  no probability   PROB_OVER is written only for the market the scoring model
                   was trained for, so the other markets' rows carry None.
                   Recording them would invent a side.
  P(over) == 0.50  no side was predicted. Rounding it to OVER would put a coin
                   flip in the ledger as a call.
  no source        the unique key is (game, player, market, line, side, source)
                   and Postgres treats NULLs in a unique index as distinct, so
                   a source-less row conflicts with nothing and the next run
                   inserts it AGAIN. An unbounded duplicate is worse than a
                   missing row, which is why this one is a hard skip.
  no game id       the grader fetches box scores by game id. A row it cannot
                   join to a game stays PENDING for ever and pollutes the
                   backlog.
  ruled out        ESPN lists the player OUT or DOUBTFUL (``AVAILABILITY`` is
                   WITHHELD, from ``pipeline.scratches``). A prediction on a
                   player who will not dress is not a prediction worth grading:
                   settlement would VOID it, and a VOID row is noise in the
                   backlog rather than evidence. NOTE that only WITHHELD is
                   skipped — UNVERIFIED and UNKNOWN are recorded, because "the
                   feed did not answer" is not "the player is out", and
                   discarding on an unanswered check would silently shrink the
                   evidence base every time ESPN had a bad afternoon.

Every skip is counted and reported by reason. A recorder that silently wrote
fewer rows than it was given would reproduce the defect it exists to fix.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

PENDING = "PENDING"
SIDE_OVER = "OVER"
SIDE_UNDER = "UNDER"

# The columns a prop-line frame must have before it can contribute anything.
# `source` is here and not optional: see the module docstring.
LINE_JOIN_KEYS = ("player_name", "market")
LINE_FIELDS = (
    "source", "over_odds_american", "under_odds_american",
    "is_pickem", "payout_multiplier", "nba_game_id", "nba_player_id",
)


@dataclass
class RecordingReport:
    """What was turned into rows, and why anything else was not."""

    rows: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)

    @property
    def skipped_by_reason(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entry in self.skipped:
            counts[entry["reason"]] = counts.get(entry["reason"], 0) + 1
        return counts

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows_built": len(self.rows),
            "rows_skipped": len(self.skipped),
            "skipped_by_reason": self.skipped_by_reason,
            "examples": self.skipped[:5],
            "note": (
                "These are model predictions recorded for forward grading, not "
                "wagers. No stake is written and none can be set here."
            ),
        }


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    return text or None


def _int_or_none(value: Any) -> int | None:
    number = _finite(value)
    return int(number) if number is not None else None


def _line_lookup(prop_lines: pd.DataFrame | None) -> dict[tuple[str, str], dict[str, Any]]:
    """
    Exact (player_name, market) -> the posted line's own fields.

    EXACT MATCH ONLY, mirroring ``main._attach_prop_lines``. This repository has
    a dedicated fuzzy crosswalk (``ingestion/id_crosswalk.py``); a second, naive
    one here could join one player's price onto another's prediction, which in a
    settlement ledger is not a near miss but a wrong record.
    """
    if prop_lines is None or getattr(prop_lines, "empty", True):
        return {}
    missing = [k for k in LINE_JOIN_KEYS if k not in prop_lines.columns]
    if missing:
        logger.warning(
            "Prop lines frame is missing %s — no source, odds or game id can be "
            "attached, so every row will be skipped for want of a source.",
            missing,
        )
        return {}

    frame = prop_lines.dropna(subset=list(LINE_JOIN_KEYS))
    # keep="last" matches _attach_prop_lines: the most recent capture wins.
    frame = frame.drop_duplicates(subset=list(LINE_JOIN_KEYS), keep="last")
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for _, row in frame.iterrows():
        key = (str(row["player_name"]), str(row["market"]))
        lookup[key] = {f: row.get(f) for f in LINE_FIELDS if f in frame.columns}
    return lookup


def pending_prop_result_rows(
    projections: pd.DataFrame,
    prop_lines: pd.DataFrame | None = None,
    *,
    run_id: str | None = None,
) -> RecordingReport:
    """
    Build PENDING ``prop_results`` rows from an assembled projections frame.

    ``projections`` is ``main.assemble_projections`` output: PLAYER_NAME,
    GAME_ID, GAME_DATE, MARKET, LINE, PROB_OVER, FINAL_PROJECTION. ``prop_lines``
    is the captured board, which supplies the fields a projection does not carry
    — source, the two-way odds, and the pick'em flag.

    Pure: no database, no clock, no network. The report names every skip.
    """
    report = RecordingReport()
    if projections is None or projections.empty:
        return report

    lookup = _line_lookup(prop_lines)

    for _, row in projections.iterrows():
        player = _text(row.get("PLAYER_NAME"))
        market = _text(row.get("MARKET"))
        label = f"{player or '?'} {market or '?'}"

        if not player or not market:
            report.skipped.append({"row": label, "reason": "no player name or market"})
            continue

        game_id = _text(row.get("GAME_ID"))
        game_date = row.get("GAME_DATE")
        line = _finite(row.get("LINE"))
        prob_over = _finite(row.get("PROB_OVER"))

        posted = lookup.get((player, market), {})
        source = _text(posted.get("source"))
        game_id = game_id or _text(posted.get("nba_game_id"))

        if line is None:
            report.skipped.append({"row": label, "reason": "no posted line"})
            continue
        if prob_over is None:
            report.skipped.append({"row": label, "reason": "no model probability"})
            continue
        if not 0.0 <= prob_over <= 1.0:
            report.skipped.append({
                "row": label, "reason": "model probability outside [0, 1]",
            })
            continue
        if prob_over == 0.5:
            report.skipped.append({
                "row": label, "reason": "no side predicted (P(over) is exactly 0.50)",
            })
            continue
        if not source:
            report.skipped.append({
                "row": label,
                "reason": "no line source (a null source defeats the unique key)",
            })
            continue
        if not game_id:
            report.skipped.append({
                "row": label, "reason": "no game id (the grader could never match it)",
            })
            continue
        if pd.isna(game_date):
            report.skipped.append({"row": label, "reason": "no game date"})
            continue
        # Only WITHHELD. See the module docstring on why an unverified check is
        # not a scratch.
        if str(row.get("AVAILABILITY") or "").strip().upper() == "WITHHELD":
            report.skipped.append({
                "row": label,
                "reason": "player is listed OUT or DOUBTFUL (would settle VOID)",
            })
            continue

        side = SIDE_OVER if prob_over > 0.5 else SIDE_UNDER
        odds = _int_or_none(
            posted.get("over_odds_american") if side == SIDE_OVER
            else posted.get("under_odds_american")
        )
        is_pickem = bool(posted.get("is_pickem")) if "is_pickem" in posted else False

        report.rows.append({
            "run_id": run_id,
            "nba_game_id": game_id,
            "nba_player_id": _text(row.get("PLAYER_ID")) or _text(posted.get("nba_player_id")),
            "player_name": player,
            "game_date": game_date,
            "market": market,
            "predicted_line": round(line, 2),
            "predicted_side": side,
            "model_projection": (
                round(proj, 2)
                if (proj := _finite(row.get("FINAL_PROJECTION"))) is not None else None
            ),
            # The COLUMN is prob_over, not "probability of the side taken", and
            # storing the taken side's probability here would silently flip the
            # meaning for every UNDER row.
            "prob_over": round(prob_over, 5),
            "odds": odds,
            "payout_multiplier": _finite(posted.get("payout_multiplier")),
            "source": source,
            "is_pickem": is_pickem,
            "outcome_status": PENDING,
            # Deliberately absent: stake_units. See the module docstring.
            "actual_result": None,
            "did_not_play": False,
        })

    if report.skipped:
        logger.info(
            "Recorded %d prediction(s); skipped %d: %s",
            len(report.rows), len(report.skipped), report.skipped_by_reason,
        )
    return report
