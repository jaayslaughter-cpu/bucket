"""Build a slate's recommendation board. One path, two callers.

WHY THIS MODULE EXISTS. The board was assembled inside
``scripts/nba_model_cli.py:decision_board_cmd`` and nowhere else, so the
scheduled worker had nothing to dispatch: ``main.py`` writes projections to
Postgres, and the board CSV that ``notify-discord --source decision-board``
reads was produced only when a person ran a command by hand. A dispatcher wired
to a file nobody writes on a schedule can only ever report that the file is
missing.

Extracted rather than duplicated. A second copy of this sequence would drift
from the CLI's, and the drifting one would be the unattended one.

``src/pipeline/`` was an empty package before this — an ``__init__.py`` and
nothing else. This is what it was presumably for.

RESEARCH_ONLY in the sense that matters here: this builds rows and writes a CSV.
It publishes nothing and places nothing. The calibration gate is applied at the
dispatch surfaces, because that is where a number reaches a person.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_MARKETS: tuple[str, ...] = ("PTS", "REB", "AST")


@dataclass
class SlateBoardResult:
    """The board, where it was written, and the summary a caller echoes."""

    board: list[Any] = field(default_factory=list)
    out: Path | None = None
    written_rows: int = 0
    slate_date: str | None = None
    summary: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.summary,
            "slate_date": self.slate_date,
            "written_rows": self.written_rows,
            "out": str(self.out) if self.out is not None else None,
        }


def build_slate_board(
    panel: pd.DataFrame,
    *,
    out: Path | str,
    markets: Sequence[str] = DEFAULT_MARKETS,
    train_end: str,
    validation_end: str,
    preferred_model: str | None = "distribution",
    min_ev: float = 0.0,
    min_lean: float = 0.0,
    require_valid_book: bool = False,
    recommended_only: bool = False,
    top_n: int | None = None,
    slate_date: str | None = None,
) -> SlateBoardResult:
    """
    Compare models on the panel, expand the slate into rows, write the CSV.

    NO PROP LINES ARE ATTACHED HERE. No archived PropLine pull exists in this
    repository, so no market candidates are joined and nothing is priced — every
    row lands on ``model_lean`` or ``unavailable``. That is the truthful state
    rather than a bug, and it is also why the calibration gate matters: a board
    of unpriced model leans is exactly what must not go out looking priced. Once
    a pull is archived, enrich each row through
    ``decision_board.enrich_row_with_resolved_market``, which applies the
    PropLine-primary / OddsPapi-fallback precedence.
    """
    from src.models.compare import compare_models_on_panel, load_comparison_config
    from src.quant.decision_board import (
        build_decision_board,
        decision_board_summary,
        write_decision_board_csv,
    )
    from src.quant.paper_research import research_slate_from_predictions
    from src.utils.timezones import pacific_calendar_date

    wanted = [str(m).strip().upper() for m in markets if str(m).strip()]
    result = compare_models_on_panel(
        panel,
        markets=wanted,
        train_end=train_end,
        validation_end=validation_end,
        cfg=load_comparison_config(),
    )
    slate = slate_date or str(pacific_calendar_date())
    slate_rows = research_slate_from_predictions(
        result.get("predictions") or [],
        slate_date=slate,
        preferred_model=preferred_model or None,
    )

    board = build_decision_board(
        slate_rows,
        min_ev=min_ev,
        min_lean=min_lean,
        require_valid_book=require_valid_book,
        consider_only=recommended_only,
        top_n=top_n,
    )
    target = Path(out)
    written = write_decision_board_csv(board, target)
    logger.info("Board for %s: %d row(s) -> %s", slate, written, target)

    return SlateBoardResult(
        board=list(board),
        out=target,
        written_rows=written,
        slate_date=slate,
        summary=decision_board_summary(board),
    )
