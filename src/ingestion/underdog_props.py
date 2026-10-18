"""
src/ingestion/underdog_props.py — capture Underdog's NBA pick'em board.

APPROVED, AND NARROW. `AGENTS.md` section 5 permits pick'em capture from
Sleeper, Underdog and PrizePicks as timestamped research only; the approval is
recorded in `docs/pickem_props.md` (2026-09-15). This captures a posted line
and, where the board reports one, a payout multiplier. It is not an odds feed:
`PickemPropLine.to_prop_line_row` forces ``is_pickem=True``, so
`market_ev_gate` abstains rather than de-vigging a multiplier.

UNDERDOG LINES ALREADY REACH THIS PIPELINE by another route — as a book key
inside PropLine's payload (`DFS_BOOKS`, `src/ingestion/propline.py`). That
route stays the primary one, because PropLine timestamps and normalises for
us. This client exists for what that route cannot give: the board as Underdog
itself posted it, captured on our own clock, which is what a line-movement
study needs and what `quant.line_diff.pickem_vs_book_line_diff` was written
against.

THE BOARD URL IS NOT IN THIS FILE, and that is deliberate rather than an
omission. Nobody working on this repository has seen a live Underdog
response: the build container cannot reach the host. A plausible endpoint
written from recollection would be an invented API endpoint presented as
fact, which `AGENTS.md` forbids and which is exactly how the 2026-10-18 report
of four broken scrapers came about. The operator supplies the URL and the
field paths in `config/pickem_sources.yaml`, or
``$PROPIQ_UNDERDOG_BOARD_URL``; until then this reports NOT_CONFIGURED and
names what to fill in.

RESEARCH_ONLY. No credential, no wager, no bot-wall bypass.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import requests

from src.ingestion.pickem_board import capture
from src.ingestion.pickem_schema import PickemSnapshot, load_source_config

logger = logging.getLogger(__name__)

SOURCE_NAME = "underdog"


def pull_underdog_board(
    *,
    config_path: str | None = None,
    session: requests.Session | None = None,
    from_file: str | Path | None = None,
) -> PickemSnapshot:
    """
    One timestamped snapshot of Underdog's NBA board.

    Never raises for a board problem: a refusal, a moved path, a restructured
    payload and an empty board are four different snapshot statuses, and a
    caller abstains on all of them without the capture taking down the run.
    """
    source = load_source_config(SOURCE_NAME, config_path)
    snapshot = capture(source, session=session, from_file=from_file)
    logger.info("Underdog board: %s", snapshot.as_dict())
    return snapshot


def underdog_prop_line_rows(**kwargs: Any) -> list[Any]:
    """The snapshot's lines in the pipeline's own row shape, pick'em-flagged."""
    return [line.to_prop_line_row() for line in pull_underdog_board(**kwargs).lines]
