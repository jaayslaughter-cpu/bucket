"""
src/ingestion/sleeper_props.py — capture Sleeper's NBA pick'em board.

SLEEPER PUBLISHES NO READABLE BOARD, and this client is shaped by that rather
than pretending otherwise. The logged-in check recorded in
`docs/pickem_props.md` on 2026-09-15 found `sleeper.com/nba` serving scores
and a season recap, `sleeper.com/picks` serving marketing and an app install,
and Player Picks app-gated with no public web odds path. Nothing since has
been verified from here.

SO THE CAPTURE PATH IS OPERATOR-SUPPLIED JSON. Drop a board response into
`data/external/pickem/sleeper/` and this reads the newest one, maps it under
the same declared field paths every source uses, and timestamps it. That is
the sanctioned route for a board that is not publicly readable; the
alternative -- defeating an app gate -- is banned in
`docs/pickem_props.md` and is not implemented here or anywhere.

If a public board path is ever found, set it as `board_url` (or
``$PROPIQ_SLEEPER_BOARD_URL``) and this client fetches it instead, with no
code change: `pickem_board.capture` already takes either.

WHAT IS DELIBERATELY NOT HERE. Sleeper's public state feed is documented in
`docs/pickem_props.md` as reporting `season_type` and `season_start_date`, and
gating "an empty board is expected" on it would be useful. It is left out
because that observation is a year old and could not be re-verified from this
container, and an unverified endpoint in a capture client is the defect this
module was written in response to. Add it when someone can check it.

Sleeper is also absent from `config/dfs_payouts.yaml` on purpose, for an
unrelated reason worth not confusing with this one: it prices per-leg dynamic
odds rather than a fixed tiered payout matrix, so it does not fit that file's
shape. Captured lines are still real posted numbers and are still research.

RESEARCH_ONLY. No credential, no app-gate bypass, no wager.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import requests

from src.ingestion.pickem_board import capture
from src.ingestion.pickem_schema import PickemSnapshot, load_source_config

logger = logging.getLogger(__name__)

SOURCE_NAME = "sleeper"


def pull_sleeper_board(
    *,
    config_path: str | None = None,
    session: requests.Session | None = None,
    from_file: str | Path | None = None,
) -> PickemSnapshot:
    """One timestamped snapshot of Sleeper's NBA board, from wherever it comes."""
    source = load_source_config(SOURCE_NAME, config_path)
    snapshot = capture(source, session=session, from_file=from_file)
    logger.info("Sleeper board: %s", snapshot.as_dict())
    return snapshot


def sleeper_prop_line_rows(**kwargs: Any) -> list[Any]:
    """The snapshot's lines in the pipeline's own row shape, pick'em-flagged."""
    return [line.to_prop_line_row() for line in pull_sleeper_board(**kwargs).lines]
