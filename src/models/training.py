"""
src/models/training.py — fit one market's artifacts, in one place.

WHY THIS MODULE EXISTS. `scripts/nba_model_cli.py train-stats` held the only
fit loop in the project: the feature resolution, the numeric/categorical
split, the train/holdout cut and the save. That was fine while a human with a
laptop was the only one who ever trained. It stopped being fine the moment the
deployed worker needed to train too, because the panel the model must learn
from lives in the deployed database and nowhere else — and a second fit loop
in `scheduler_worker` would be a second definition of what training IS.

So the loop moved here and both callers use it. `train-stats` keeps its
flags and its echoes; the worker job keeps its logging and its lock. Neither
owns the decisions below.

WHAT IT DOES NOT DECIDE. It does not choose the panel, the window or where
the artifacts go: the caller passes a built feature matrix and an artifacts
directory. That matters for the deployed case, where the matrix must be the
SAME one the slate scores against — built by the same builder from the same
database — or the artifact's contract names columns the live matrix does not
have, and every row abstains for a reason that looks like a model fault.

THE LABEL IS A RESEARCH STAND-IN, not a book line. `prepare_market_panel`
labels `over_hit` as the realised stat against `{MARKET}_L10`, the player's
trailing ten-game mean, because historical prop lines do not exist for the
seasons there is panel for. A model fitted this way answers "will he beat his
own recent form", which is NOT the question a sportsbook asks, and serving it
against a posted line is a distribution shift that
`labels.mask_probabilities_at_unsupported_lines` exists to bound. Say so
wherever the numbers are reported; do not call it an edge against a book.

RESEARCH_ONLY. This fits a classifier and writes files. It prices nothing,
sizes nothing and places nothing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

#: Columns CatBoost may take natively and XGBoost may not. XGBoost rejects
#: non-numerics outright, so it keeps the numeric list; passing the combined
#: list to both made every run fail XGBoost with a DATA_NOT_AVAILABLE and
#: write only two of the three artifacts -- quietly, since the loop logs and
#: continues. Carried over from `train-stats`, where it was found.
CATEGORICAL_EXTRAS = ("TEAM_ABBREVIATION", "OPPONENT_ABBREVIATION", "SEASON")

MARKETS = ("PTS", "REB", "AST")


class TrainingDataError(RuntimeError):
    """The panel cannot support a fit, and the reason names which part."""


def fit_market_artifacts(
    frame: pd.DataFrame,
    market: str,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    train_end: str | None = None,
    artifacts_dir: Path | str | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Fit every configured component for ``market`` and save what fitted.

    ``frame`` is an already-built feature matrix. It is NOT rebuilt here:
    running the builder over columns that already carry rolling features
    applies a second pass of them.

    THE SPLIT IS BY DATE WHEN ``train_end`` IS GIVEN, and that is the
    configuration to prefer. The fallback is 2/3 by ROW COUNT, which couples
    two unrelated things -- how much history you train on and how much of it
    is held out -- and on a long window holds out years the model never sees.
    Measured numbers and the one-window caveat: `docs/training_window.md`.

    Returns a report rather than printing one, so the CLI can echo it and the
    worker can log it without either guessing what the other did.
    """
    from src.models.compare import (
        build_components,
        load_comparison_config,
        prepare_market_panel,
        resolve_feature_cols,
    )
    from src.models.labels import TARGET_COL, default_feature_cols

    market = market.upper()
    if market not in MARKETS:
        raise TrainingDataError(
            f"{market!r} is not a launch market; they are {', '.join(MARKETS)}"
        )
    if frame is None or getattr(frame, "empty", True):
        raise TrainingDataError(
            "the feature matrix is empty, so there is nothing to fit. The panel "
            "the matrix is built from is what to look at, not this call."
        )

    work = prepare_market_panel(frame, market)
    if start_date:
        work = work[work["GAME_DATE"] >= start_date]
    if end_date:
        work = work[work["GAME_DATE"] <= end_date]
    if work.empty:
        raise TrainingDataError(
            f"no {market} rows between {start_date or 'the start'} and "
            f"{end_date or 'the end'} of the matrix"
        )

    # LABELLED ROWS ARE THE ONES THAT CAN TEACH ANYTHING. A push is left NaN
    # on purpose (grading it as a loss would bias the label set), and a row
    # with no research line cannot be labelled at all -- so a matrix can be
    # large and still carry nothing to fit. Counted here because "0 labelled"
    # and "0 rows" are different faults with different fixes.
    labelled = int(pd.to_numeric(work[TARGET_COL], errors="coerce").notna().sum())
    if not labelled:
        raise TrainingDataError(
            f"{len(work):,} {market} row(s) in the window and NONE carries a "
            f"label: every row lacked either the realised stat or its "
            f"{market}_L10 research line. A player's first ten games cannot "
            f"have an L10, so a panel that is all early-season looks like this."
        )

    cols, dropped = resolve_feature_cols(work, list(default_feature_cols(market)))
    if not cols:
        raise TrainingDataError(
            f"no usable features for {market} in this matrix; "
            f"{len(dropped)} of the default column(s) were absent"
        )
    numeric_cols = list(cols)
    for extra in CATEGORICAL_EXTRAS:
        if extra in work.columns and extra not in cols:
            cols = list(cols) + [extra]

    cfg = config if config is not None else load_comparison_config()
    art = Path(
        artifacts_dir
        if artifacts_dir is not None
        else cfg.get("artifacts_dir", "data/external/model_runs/comparison")
    )
    art.mkdir(parents=True, exist_ok=True)

    if train_end:
        cutoff = pd.Timestamp(train_end)
        train = work[work["GAME_DATE"] <= cutoff]
        val = work[work["GAME_DATE"] > cutoff]
        if train.empty or val.empty:
            raise TrainingDataError(
                f"train_end {train_end} leaves {len(train)} fit row(s) and "
                f"{len(val)} holdout row(s); one side is empty, so there is "
                f"nothing to fit or nothing to calibrate on. The window is "
                f"{work['GAME_DATE'].min().date()} to "
                f"{work['GAME_DATE'].max().date()}."
            )
        split = f"date:{train_end}"
    else:
        mid = len(work) * 2 // 3
        train, val = work.iloc[:mid], work.iloc[mid:]
        if train.empty or val.empty:
            raise TrainingDataError(
                f"{len(work)} row(s) cannot be split 2/3 by row count into a "
                f"fit set and a holdout; pass train_end, or widen the window."
            )
        split = "rows:2/3"

    comps = build_components(market, cols, cfg, xgb_feature_cols=numeric_cols)
    fitted: list[str] = []
    persisted: list[str] = []
    unpersisted: list[str] = []
    failures: dict[str, str] = {}
    for name, model in comps.items():
        try:
            model.fit(train, val)
            # FITTED AND PERSISTED ARE DIFFERENT FACTS, and reporting one as
            # the other is a defect I shipped. `if hasattr(model, "save")`
            # quietly skipped any component without a `save` method and then
            # appended it to `fitted` regardless -- so every run reported
            # "fitted: [distribution, xgboost, catboost, line_aware]" while
            # writing three artifacts. `line_aware` had no `save` at all, and
            # the demo run's own artifacts directory shows it: catboost,
            # distribution and xgboost files, no line_aware file.
            #
            # A COMPONENT THAT CANNOT PERSIST CANNOT BE SERVED, so its weight
            # in the ensemble is unreachable and its absence should be visible
            # in the report rather than inferred from a directory listing.
            if hasattr(model, "save"):
                model.save(art / f"{name}_{market}")
                persisted.append(name)
            else:
                unpersisted.append(name)
                logger.warning(
                    "%s %s fitted but has no save(): nothing was written, so it "
                    "cannot be resolved or served and its share of the ensemble "
                    "is unreachable.", name, market,
                )
            fitted.append(name)
        except Exception as exc:  # noqa: BLE001 — one component, not the market
            # ONE COMPONENT'S FAILURE IS NOT THE MARKET'S, but it is also not
            # nothing: the caller gets it by name in the report, because
            # "fitted 2 of 3" used to be a log line nobody read and the
            # pipeline then resolved an artifact that was never written.
            logger.error("%s %s failed: %s", name, market, exc)
            failures[name] = f"{type(exc).__name__}: {exc}"

    report = {
        "market": market,
        "artifacts_dir": str(art),
        "rows": int(len(work)),
        "labelled_rows": labelled,
        "features": len(numeric_cols),
        "dropped_features": sorted(dropped),
        "split": split,
        "fit_rows": int(len(train)),
        "fit_window": [str(train["GAME_DATE"].min().date()),
                       str(train["GAME_DATE"].max().date())],
        "holdout_rows": int(len(val)),
        "holdout_window": [str(val["GAME_DATE"].min().date()),
                           str(val["GAME_DATE"].max().date())],
        "fitted": fitted,
        "persisted": persisted,
        "fitted_but_not_persisted": unpersisted,
        "failed": failures,
    }
    logger.info("Trained %s: %s", market, report)
    return report
