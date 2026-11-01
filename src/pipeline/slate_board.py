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
    source precedence (PropLine is the only listed source).
    """
    from src.models.compare import compare_models_on_panel, load_comparison_config
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

    return _rank_and_write(
        slate_rows, out=out, slate=slate, min_ev=min_ev, min_lean=min_lean,
        require_valid_book=require_valid_book, recommended_only=recommended_only,
        top_n=top_n,
    )


def _rank_and_write(
    slate_rows: Sequence[Any],
    *,
    out: Path | str,
    slate: str,
    min_ev: float,
    min_lean: float,
    require_valid_book: bool,
    recommended_only: bool,
    top_n: int | None,
) -> SlateBoardResult:
    """
    Rank slate rows, write the CSV, summarise. ONE TAIL, TWO HEADS.

    The two heads differ only in where the rows come from:
    `build_slate_board` compares models over labelled history, and
    `build_board_from_projections` reads what `main.py` already projected
    for tonight. The ranking, the thresholds, the CSV and the summary are
    the same work either way, and a second copy of them would be a second
    definition of what a board IS -- the defect this repository keeps
    removing.
    """
    from src.quant.decision_board import (
        build_decision_board,
        decision_board_summary,
        write_decision_board_csv,
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


#: What `build_board_from_projections` calls the single model behind a
#: projection row. `research_slate_from_predictions` groups detail rows by
#: (event, player, market) and treats each `model_name` as one opinion to
#: arbitrate between; there is exactly one here, so the name's job is to say
#: WHERE the number came from rather than to be chosen between.
PROJECTION_MODEL_NAME = "slate_projection"


def projection_detail_rows(projections: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Persisted projection rows -> the detail-row shape the board reads.

    A RENAME AND NOTHING MORE. No probability is computed, no line is
    attached, no projection is adjusted: every value below is a column
    `main.py` already wrote and `repository.persist_projections` already
    stored. That is the point of this path -- the slate job did the
    modelling at 09:00, and the board's job is to rank and format it.

    ``game_date`` TRAVELS, and the first version of this dropped it. A board
    row must say which game it describes -- see
    `tests/test_board_game_date.py`, whose subject is a board that stamped
    every row with the day it was BUILT while saying nothing about the day
    the game is played.

    ``prediction_std_or_dispersion`` IS LEFT NONE, DELIBERATELY. The
    projections table stores a mean (`final_projection`) and no dispersion,
    so there is no σ to pass. `edge_grades.research_edge_letter_grade`
    documents its own fallback order and drops from the `(mean - line) / σ`
    z to the `|P(over) - 0.5|` proxy, or to `grade_basis="unavailable"`,
    when σ is absent. A fabricated σ -- a league-average, a fraction of the
    mean -- would turn that honest degradation into a confident-looking
    grade resting on a number nobody measured, which is the one thing this
    repository refuses to do with a missing input.
    """
    wanted = (
        "nba_game_id", "nba_player_id", "player_name", "market", "game_date",
        "final_projection", "line", "prob_over", "prob_under", "prob_push",
    )
    missing = [c for c in wanted if c not in projections.columns]
    if missing:
        raise ValueError(
            f"DATA_NOT_AVAILABLE: projection frame lacks {missing}. It should "
            f"be `repository.load_projections`' output; a frame assembled "
            f"elsewhere is not interchangeable with it."
        )

    rows: list[dict[str, Any]] = []
    for record in projections.to_dict("records"):
        market = record.get("market")
        if not market:
            continue
        # THE GAME'S OWN DATE, and dropping it reintroduced the symptom of
        # what `tests/test_board_game_date.py` calls "the most misleading
        # [defect] in the project": a row that cannot say which game it
        # describes, next to a `slate_date` that reads as today.
        # `research_slate_from_predictions` reads `game_date` off the detail
        # row and warns per row when it is absent -- so every live board row
        # carried that warning while the date sat unread in the frame.
        # Raised by review on #32.
        #
        # THE PLAYER IDENTITY FALLS BACK TO THE NAME, because
        # `research_slate_from_predictions` groups on
        # `(event_id, player_id, target_market)` and `nba_player_id` is
        # NULLABLE. Two players in one game and market with no id would
        # group as one board row -- two people collapsed into a single
        # recommendation. `player_name` is NOT NULL on the model, so it is
        # the identity to fall back to rather than None.
        player_id = record.get("nba_player_id") or record.get("player_name")
        rows.append({
            "event_id": record.get("nba_game_id"),
            "game_date": record.get("game_date"),
            "player_id": player_id,
            "player_name": record.get("player_name"),
            "target_market": str(market).upper(),
            "model_name": PROJECTION_MODEL_NAME,
            "prop_line": record.get("line"),
            "prediction_mean": record.get("final_projection"),
            "prediction_std_or_dispersion": None,
            "probability_over_raw": record.get("prob_over"),
            "probability_under_raw": record.get("prob_under"),
            "probability_push_raw": record.get("prob_push"),
        })
    return rows


def build_board_from_projections(
    projections: pd.DataFrame,
    *,
    out: Path | str,
    slate_date: str,
    markets: Sequence[str] = DEFAULT_MARKETS,
    min_ev: float = 0.0,
    min_lean: float = 0.0,
    require_valid_book: bool = False,
    recommended_only: bool = False,
    top_n: int | None = None,
) -> SlateBoardResult:
    """
    A board for TONIGHT, from the projections the slate job already wrote.

    WHY THIS EXISTS RATHER THAN ANOTHER FIX TO `build_slate_board`. That
    function routes through `compare_models_on_panel`, which fits models on
    labelled history and scores a validation window of games with KNOWN
    OUTCOMES -- `compare.py` drops every row whose `over_hit` is null before
    it splits. A game that has not been played has no outcome, so it has no
    label, so it is dropped: the function structurally cannot describe
    tonight. `scripts/nba_model_cli.py:929` says the same thing by
    construction, defaulting to `--train-end 2025-01-15 --validation-end
    2025-02-15 --demo True`.

    Three attempts in two days each got one step closer to that wall:
    #31 supplied the missing feature matrix, #32 attached the forward slate,
    and the forward rows were dropped at the label filter -- a fix that
    could not work, caught in review. The board was asking a model-
    comparison function to forecast.

    THE NUMBERS ALREADY EXIST. `main.py` scores the slate at 09:00 PT and
    persists one row per (player, game, market) with the prop line attached
    and the EV gate applied -- 460 rows on 2026-11-01. Ranking and
    formatting those IS the board, and it needs no model comparison at all.

    `markets` filters; it does not create. A market with no projection row
    contributes nothing rather than an empty placeholder.

    RESEARCH_ONLY, unchanged: this reads rows and writes a CSV. The
    calibration gate still sits at the dispatch surfaces, which is where a
    number reaches a person. With no model artifact every row arrives with
    a null probability and the board says so per row, which is the honest
    state rather than an empty file.
    """
    from src.quant.paper_research import research_slate_from_predictions

    wanted = {str(m).strip().upper() for m in markets if str(m).strip()}
    detail = [
        row for row in projection_detail_rows(projections)
        if not wanted or row["target_market"] in wanted
    ]
    slate_rows = research_slate_from_predictions(
        detail,
        slate_date=str(slate_date),
        # ONE MODEL, NAMED. The default preference is "distribution", which
        # no projection row carries, so leaving it would send
        # `research_slate_from_predictions` down its fallback path to reach
        # the same row by accident instead of on purpose.
        preferred_model=PROJECTION_MODEL_NAME,
    )
    logger.info(
        "Board from projections for %s: %d projection row(s) -> %d detail "
        "row(s) in %s -> %d slate row(s).",
        slate_date, len(projections), len(detail),
        sorted(wanted) or "every market", len(slate_rows),
    )
    return _rank_and_write(
        slate_rows, out=out, slate=str(slate_date), min_ev=min_ev,
        min_lean=min_lean, require_valid_book=require_valid_book,
        recommended_only=recommended_only, top_n=top_n,
    )
