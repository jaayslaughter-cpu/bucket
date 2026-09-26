"""Walk-forward model comparison and export orchestration."""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, mean_squared_error

from src.models.distribution_adapter import DistributionPropModel
from src.models.ensemble import EnsemblePropModel
from src.models.labels import attach_research_over_labels, default_feature_cols
from src.models.prob_calibration import expected_calibration_error
from src.models.walk_forward import fixed_cutoff_split, sort_by_game_date
from src.utils.timezones import format_pacific_iso, pacific_midnight_utc

logger = logging.getLogger(__name__)


DEFAULT_CATEGORICAL_COLS = ("TEAM_ABBREVIATION", "OPPONENT_ABBREVIATION", "SEASON")

# Columns holding THIS game's realised outcome. They sit in the feature
# matrix because later rows' rolling windows are built from them, but
# using one as a feature hands the model the answer.
#
# The raw stat names are obvious. The efficiency columns are the dangerous
# ones: TS_PCT and SHOT_VOLUME read like engineered features and are not —
# they are same-game box-score quantities, and only their shifted _L5/_L10
# forms are safe to model on.
POSTGAME_ONLY_COLS = frozenset({
    "PTS", "REB", "AST", "PRA", "FG3M", "FG3A", "STL", "BLK", "TOV", "MIN",
    "FGM", "FGA", "FTM", "FTA",
    "TS_PCT", "SHOT_VOLUME", "FT_RATE",
})


class PostgameFeatureError(ValueError):
    """Raised when a feature list contains a same-game outcome column."""


def load_comparison_config(path: Path | str = "config/model_comparison.yaml") -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def resolve_feature_cols(df: pd.DataFrame, cols: list[str]) -> tuple[list[str], list[str]]:
    """
    Keep only features the panel actually has.

    An earlier version created any missing column and filled it with 0.0,
    which let a model train on a wholly invented feature without raising.
    A missing column is now dropped and reported, so the run is narrower
    but honest.
    """
    # Refuse outright rather than dropping: a postgame column in a feature
    # list is a leak, not a narrower run, and silently removing it would
    # hide a mistake the caller needs to see.
    leaking = sorted(set(cols) & POSTGAME_ONLY_COLS)
    if leaking:
        raise PostgameFeatureError(
            f"Feature list contains same-game outcome column(s) {leaking}. "
            "These hold THIS game's result — use their shifted _L5/_L10 "
            "forms instead. (TS_PCT and SHOT_VOLUME look like engineered "
            "features but are raw box-score quantities.)"
        )

    present = [c for c in cols if c in df.columns]
    absent = [c for c in cols if c not in df.columns]
    if absent:
        logger.warning(
            "Dropping %d feature(s) not present in the panel: %s. They are NOT "
            "zero-filled — a fabricated column would silently corrupt training.",
            len(absent), absent,
        )
    return present, absent


def recency_sample_weights(
    train: pd.DataFrame, cfg: dict[str, Any], market: str
) -> "tuple[pd.Series | None, dict[str, Any] | None]":
    """
    Recency weights for this training window, or ``(None, None)`` when off.

    WHY THIS FUNCTION EXISTS. src/models/recency.py computed these weights and
    xgboost_pipeline.py and catboost_pipeline.py both accepted a
    ``sample_weight`` — and nothing in the repository passed one, so a game from
    2018 and a game from last week carried identical influence in every fit.
    This is the missing middle.

    OFF BY DEFAULT, like the blowout layer, because weighting is not free: it
    discards information, and how much is measurable rather than arguable. The
    Kish effective sample size is logged on every run that enables it, so a
    half-life that quietly reduces 80,000 rows to 9,000 says so before it shows
    up as an unstable model.

    NO ``as_of`` IS PASSED. exponential_recency_weights then anchors on the
    newest date in ``train`` — the training window's own end. Handing it the
    validation end would leak the split boundary into the fit, and the function
    refuses a reference date outside the window rather than allowing it.
    """
    recency_cfg = cfg.get("recency") or {}
    if not recency_cfg.get("enabled", False):
        return None, None
    if "GAME_DATE" not in train.columns:
        logger.warning(
            "recency.enabled is true but the training frame has no GAME_DATE — "
            "fitting UNWEIGHTED rather than inventing an ordering."
        )
        return None, None

    from src.models.recency import (
        DEFAULT_HALF_LIFE_DAYS,
        RecencyWeightError,
        exponential_recency_weights,
        recency_weight_report,
    )

    half_life = float(recency_cfg.get("half_life_days", DEFAULT_HALF_LIFE_DAYS))
    try:
        weights = exponential_recency_weights(
            train["GAME_DATE"], half_life_days=half_life
        )
        report = recency_weight_report(train["GAME_DATE"], half_life_days=half_life)
    except RecencyWeightError as exc:
        logger.warning(
            "Market %s: recency weighting refused (%s) — fitting UNWEIGHTED.",
            market, exc,
        )
        return None, None

    report["market"] = market
    logger.info(
        "Market %s: recency weights half_life=%.0fd, %d rows -> effective %s "
        "(%.1f%%), max/min weight %s",
        market, half_life, int(report["n_rows"]),
        report["effective_sample_size"], report["effective_fraction"] * 100.0,
        report["weight_ratio"],
    )
    return weights, report


def _winner_rank_key(scores: dict[str, Any]) -> tuple[float, float, int, float]:
    """
    Sort key for picking a market's winner: Brier, then log loss, then ECE.

    TWO DEFECTS THIS REPLACES, both in `x or 9`:

    1. ``0.0 or 9`` IS 9. A model whose ECE rounded to 0.0000 -- the best
       possible calibration -- was ranked as though its calibration could not be
       measured at all. Measured: `0.0 or 9` and `None or 9` both yield 9, so the
       best and the unmeasurable were indistinguishable.

    2. A GATE FAILURE IS NOT A BAD SCORE. Now that calibration_error comes from
       the gated implementation, it is None whenever fewer than 80% of the
       reliability bins are occupied. Collapsing that to 9 would penalise a model
       for a sparse validation window rather than for being badly calibrated.

    The deliberate rule: models WITH a gated ECE are ordered by it; models
    without one sort after them. "We could not measure this model's calibration"
    is not evidence that it is well calibrated, so it does not win a tie -- but
    it is recorded as a missing measurement rather than as a score of 9.
    """
    brier = scores.get("brier_score")
    log_loss_value = scores.get("log_loss")
    ece = scores.get("calibration_error")
    return (
        float(brier) if brier is not None else float("inf"),
        float(log_loss_value) if log_loss_value is not None else float("inf"),
        0 if ece is not None else 1,
        float(ece) if ece is not None else float("inf"),
    )


def _accepts_sample_weight(fit_callable: Any) -> bool:
    """Does this component's fit take a ``sample_weight`` keyword?"""
    try:
        return "sample_weight" in inspect.signature(fit_callable).parameters
    except (TypeError, ValueError):  # builtins, C extensions
        return False


def prepare_market_panel(panel: pd.DataFrame, market: str) -> pd.DataFrame:
    labeled = attach_research_over_labels(panel, stat=market)  # type: ignore[arg-type]
    return sort_by_game_date(labeled)


def score_binary(y_true: np.ndarray, p: np.ndarray) -> dict[str, float | None]:
    mask = np.isfinite(p) & np.isfinite(y_true)
    if mask.sum() < 5:
        return {"n": float(mask.sum()), "brier": None, "log_loss": None, "mae": None, "rmse": None, "mean_bias": None}
    yt, pp = y_true[mask].astype(int), np.clip(p[mask], 1e-6, 1 - 1e-6)
    return {
        "n": float(len(yt)),
        "brier": float(brier_score_loss(yt, pp)),
        "log_loss": float(log_loss(yt, pp, labels=[0, 1])),
        "mae": None,
        "rmse": None,
        "mean_bias": None,
    }


def score_mean(y_true: np.ndarray, y_hat: np.ndarray) -> dict[str, float | None]:
    mask = np.isfinite(y_true) & np.isfinite(y_hat)
    if mask.sum() < 5:
        return {"mae": None, "rmse": None, "mean_bias": None, "n": float(mask.sum())}
    yt, yh = y_true[mask], y_hat[mask]
    return {
        "n": float(len(yt)),
        "mae": float(mean_absolute_error(yt, yh)),
        "rmse": float(mean_squared_error(yt, yh) ** 0.5),
        "mean_bias": float((yh - yt).mean()),
    }


def build_components(
    market: str,
    feature_cols: list[str],
    cfg: dict[str, Any],
    *,
    include_catboost: bool = True,
    include_xgboost: bool = True,
    xgb_feature_cols: list[str] | None = None,
) -> dict[str, Any]:
    """
    Build model components.

    XGBoost keeps the original numeric ``default_feature_cols`` list (unchanged
    behavior). CatBoost may add categorical columns separately.
    """
    components: dict[str, Any] = {
        "distribution": DistributionPropModel(target_market=market),
    }
    seed = int(cfg.get("random_seed", 42))
    schema = str(cfg.get("feature_schema_version", "fs_v1_shift1_l2"))
    xgb_cols = list(xgb_feature_cols or feature_cols)
    if include_xgboost:
        try:
            from src.models.xgb_adapter import XGBoostAdapter
            from src.models.xgboost_pipeline import split_xgboost_config

            xgb_params, xgb_tuning = split_xgboost_config(cfg.get("xgboost") or {})
            components["xgboost"] = XGBoostAdapter(
                xgb_cols,
                target_market=market,
                feature_schema_version=schema,
                random_state=seed,
                model_params=xgb_params or None,
                tuning=xgb_tuning or None,
            )
        except ImportError as exc:
            logger.warning("XGBoost unavailable: %s", exc)
    if include_catboost:
        try:
            from src.models.catboost_pipeline import CatBoostPropPipeline

            cb_cfg = cfg.get("catboost") or {}
            # Fall back to the known-categorical names when the config does
            # not list them. Without this, a run started outside the repo
            # root (where config/model_comparison.yaml is not found) treated
            # TEAM_ABBREVIATION and friends as numeric, coerced every value
            # to NaN, and dropped the entire training set — surfacing as
            # CatBoost's "Labels variable is empty", which names the wrong
            # thing entirely.
            configured_cats = list(cb_cfg.get("categorical_features") or [])
            if not configured_cats:
                configured_cats = [c for c in DEFAULT_CATEGORICAL_COLS if c in feature_cols]
                if configured_cats:
                    logger.info(
                        "No categorical_features configured — treating %s as categorical "
                        "by name, since coercing them to numeric would empty the panel.",
                        configured_cats,
                    )
            components["catboost"] = CatBoostPropPipeline(
                feature_cols,
                target_market=market,
                feature_schema_version=schema,
                categorical_features=configured_cats,
                hyperparameters={k: v for k, v in cb_cfg.items() if k != "categorical_features"},
                random_seed=seed,
            )
        except ImportError as exc:
            logger.warning("CatBoost unavailable: %s", exc)

    # Line-aware wrapper. It does not replace a model — it trains one of the
    # components above on a frame whose features include the line, so two
    # different lines on the same player-game genuinely produce two different
    # probabilities. A line-blind classifier cannot do that by construction.
    line_cfg = cfg.get("line_aware") or {}
    if line_cfg.get("enabled"):
        base_name = str(line_cfg.get("base", "xgboost"))
        if base_name not in components:
            logger.warning(
                "line_aware base %r is not among the built components %s — "
                "skipping rather than silently wrapping a different model.",
                base_name, sorted(components),
            )
        else:
            try:
                from src.models.line_aware import (
                    DEFAULT_LINE_OFFSETS,
                    LineAwarePropModel,
                )

                offsets = tuple(
                    float(o) for o in (line_cfg.get("offsets") or DEFAULT_LINE_OFFSETS)
                )
                # The factory is handed the augmented column list at fit time,
                # which includes the line features — so the base model is built
                # to see them rather than retrofitted afterwards.
                base_cols = xgb_cols if base_name == "xgboost" else feature_cols

                # The inner build must NOT re-enter this block: it would
                # nest a wrapper inside the wrapper, and fitting it would
                # recurse. Disable the layer for the inner call explicitly.
                base_cfg = {**cfg, "line_aware": {"enabled": False}}

                def _base_factory(cols: list[str], _name: str = base_name):
                    return build_components(
                        market, cols, base_cfg,
                        include_catboost=(_name == "catboost"),
                        include_xgboost=(_name == "xgboost"),
                        xgb_feature_cols=cols,
                    )[_name]

                components["line_aware"] = LineAwarePropModel(
                    _base_factory,
                    stat=market,
                    base_feature_cols=base_cols,
                    offsets=offsets,
                    max_augmented_rows=line_cfg.get("max_augmented_rows"),
                )
            except ImportError as exc:
                logger.warning("line_aware unavailable: %s", exc)
    return components



def apply_calibration(preds, p_over, calibrator) -> np.ndarray:
    """
    Calibrate P(over) and write the calibrated fields onto ``preds``.

    FIT AND APPLY MUST SPEAK THE SAME PROBABILITY. The calibrator is fitted on
    out-of-fold values from ``predict_proba_over``, which is the raw classifier
    P(over) with no push mass removed -- CONDITIONAL on the line not pushing.
    ``probability_over`` has already had push carved out, as
    ``(1 - p_push) * p_raw``. Feeding that in handed the calibrator a different
    quantity than it was fitted on, and scaling the result by ``open_mass``
    then removed the push mass a SECOND time.

    Not a dormant corner: RESEARCH_LINE is a ten-game rolling mean, so about
    11% of its values are whole numbers, where the empirical push rate is 0.9%
    (PTS), 2.0% (REB) and 3.1% (AST).

    So: un-carve to the conditional quantity, calibrate, carve exactly once.
    Returns the calibrated CONDITIONAL probabilities (NaN where unavailable),
    which is what the scoring path compares against ``over_hit`` -- that label
    drops pushes, so it is conditional too.
    """
    push_mass = np.array(
        [float(p.probability_push or 0.0) for p in preds], dtype=float
    )
    open_mass = np.clip(1.0 - push_mass, 0.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        conditional = np.where(open_mass > 0, np.asarray(p_over, dtype=float) / open_mass, np.nan)
    conditional = np.clip(conditional, 0.0, 1.0)

    p_cal = np.full(len(push_mass), np.nan, dtype=float)
    if calibrator is None:
        return p_cal

    usable = np.isfinite(conditional)
    if usable.any():
        p_cal[usable] = calibrator.transform(conditional[usable])
    for pred, value, mass in zip(preds, p_cal, open_mass):
        if np.isfinite(value):
            # The calibrated number is P(over | not a push); scale it into the
            # non-push mass. Subtracting push from the calibrated over instead
            # lets the under go negative whenever isotonic saturates at 1.0 on
            # a whole line.
            conditional_over = float(np.clip(value, 0.0, 1.0))
            pred.probability_over_calibrated = round(conditional_over * mass, 6)
            pred.probability_under_calibrated = round((1.0 - conditional_over) * mass, 6)
    return p_cal


def fit_calibrator_from_earlier_data(
    name: str,
    market: str,
    feature_cols: list[str],
    xgb_cols: list[str],
    cfg: dict[str, Any],
    train: pd.DataFrame,
    *,
    calib_fraction: float = 0.3,
    fitted_model: Any | None = None,
) -> tuple[Any | None, dict[str, Any]]:
    """
    Fit a probability calibrator using only data earlier than the evaluation.

    A calibrator fitted on the predictions it later corrects is worthless —
    it learns the noise it is supposed to smooth. So the training period is
    split chronologically: a fresh copy of the model is fitted on the
    earlier part, asked to predict the later part, and the calibrator is
    fitted on those genuinely out-of-sample probabilities. Validation rows
    are never touched here.

    Returns (calibrator_or_None, info) — info always explains a None.
    """
    from src.models.prob_calibration import choose_calibrator

    min_rows = int((cfg.get("calibration") or {}).get("min_oof_rows", 200))
    if len(train) < min_rows:
        return None, {"reason": f"only {len(train)} training rows, need {min_rows}"}

    # FAST PATH: the fitted model already produced out-of-fold probabilities
    # during its own fit, on the same chronological folds the dispersion
    # used. Refitting the whole component on a separate 70/30 split both
    # duplicates that work and disagrees with it — and it showed the
    # calibrator only the last 30% of the training window.
    oof = getattr(fitted_model, "oof", None) if fitted_model is not None else None
    if oof is not None and getattr(oof, "usable", False):
        y_oof, p_oof = oof.arrays()
        try:
            calibrator, scores = choose_calibrator(y_oof, p_oof)
        except ValueError as exc:
            return None, {"reason": f"shared out-of-fold calibration failed: {exc}"}
        return calibrator, {
            "method": calibrator.method,
            "source": "shared_out_of_fold",
            "n_rows": int(len(y_oof)),
            "scores": scores,
            **oof.as_metadata(),
        }

    ordered = train.sort_values("GAME_DATE") if "GAME_DATE" in train.columns else train
    cut = int(len(ordered) * (1 - calib_fraction))
    earlier, later = ordered.iloc[:cut], ordered.iloc[cut:]
    if earlier.empty or later.empty:
        return None, {"reason": "chronological calibration split produced an empty side"}

    try:
        components = build_components(market, feature_cols, cfg, xgb_feature_cols=xgb_cols)
        if name == "ensemble":
            # Rebuild the blend from components fitted on the earlier window
            # only, so the calibrator never sees its own evaluation rows.
            refit = {}
            for comp_name, comp in components.items():
                try:
                    comp.fit(earlier, later)
                    refit[comp_name] = comp
                except Exception as exc:  # noqa: BLE001
                    logger.debug("calibration re-fit skipped %s: %s", comp_name, exc)
            if len(refit) < 2:
                return None, {"reason": "fewer than two components re-fitted for the ensemble"}
            ens_weights = cfg.get("ensemble_weights") or {}
            fresh = EnsemblePropModel(
                refit,
                weights={k: ens_weights.get(k, 0.0) for k in refit},
                target_market=market,
            )
        else:
            fresh = components.get(name)
            if fresh is None:
                return None, {"reason": f"no component named {name} to re-fit"}
            fresh.fit(earlier, later)
        preds = fresh.predict_rows(later, line_col="RESEARCH_LINE")
    except Exception as exc:  # noqa: BLE001
        return None, {"reason": f"calibration re-fit failed: {exc}"}

    p = np.array([x.probability_over if x.probability_over is not None else np.nan for x in preds])
    y = later["over_hit"].astype(float).to_numpy()
    ok = np.isfinite(p) & np.isfinite(y)
    if int(ok.sum()) < 60:
        return None, {"reason": f"only {int(ok.sum())} usable out-of-sample rows"}

    try:
        calibrator, scores = choose_calibrator(y[ok], p[ok])
    except ValueError as exc:
        return None, {"reason": str(exc)}

    info = {
        "method": calibrator.method,
        # Name the path. Without this a model that fell back here exported
        # calibration_source: null, which reads as "no calibration" rather
        # than "calibrated the slower way" -- and the ensemble falls back
        # whenever a component's out-of-fold frame describes different rows.
        "source": "chronological_refit",
        "n_rows": int(ok.sum()),
        "scores": scores,
        "fit_start_date": str(pd.to_datetime(later["GAME_DATE"]).min().date())
        if "GAME_DATE" in later.columns
        else None,
        "fit_end_date": str(pd.to_datetime(later["GAME_DATE"]).max().date())
        if "GAME_DATE" in later.columns
        else None,
    }
    logger.info(
        "calibrator %s/%s: method=%s rows=%d window=%s..%s",
        market, name, info["method"], info["n_rows"], info["fit_start_date"], info["fit_end_date"],
    )
    return calibrator, info


def compare_models_on_panel(
    panel: pd.DataFrame,
    *,
    markets: list[str],
    train_end: str,
    validation_end: str,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Fit on rows with GAME_DATE <= train_end; score (train_end, validation_end].

    Returns summary rows + detailed prediction rows (no fabricated odds).
    """
    cfg = cfg or load_comparison_config()
    weights = cfg.get("ensemble_weights") or {}
    summary_rows: list[dict[str, Any]] = []
    fit_failures: list[dict[str, Any]] = []
    # Returned rather than only logged: a run whose weighting collapsed the
    # effective sample size is not distinguishable from an unweighted one by
    # its metrics alone, and the caller has to be able to see which it was.
    recency_reports: list[dict[str, Any]] = []
    ensemble_composition: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    calib_rows: list[dict[str, Any]] = []
    calibration_meta: dict[tuple[str, str], dict[str, Any]] = {}
    winners: dict[str, dict[str, Any]] = {}

    for market in markets:
        work = prepare_market_panel(panel, market)
        xgb_cols, dropped = resolve_feature_cols(work, list(default_feature_cols(market)))  # type: ignore[arg-type]
        if not xgb_cols:
            logger.error("Market %s has none of its expected features — skipping.", market)
            continue
        feature_cols = list(xgb_cols)
        # Add categoricals for CatBoost only (XGBoost stays numeric-only)
        for c in ("TEAM_ABBREVIATION", "OPPONENT_ABBREVIATION", "SEASON"):
            if c in work.columns and c not in feature_cols:
                feature_cols = list(feature_cols) + [c]
        # Drop rows with no usable label — never fill a target. The index is
        # reset because fixed_cutoff_split sorts and resets internally, then
        # returns positional labels that must line up with this frame.
        work = work.loc[work["over_hit"].notna()].reset_index(drop=True)
        if work.empty:
            logger.error("Market %s has no labelled rows — skipping.", market)
            continue
        split = fixed_cutoff_split(work, train_end=train_end, validation_end=validation_end)
        train = work.loc[split.train_idx]
        val = work.loc[split.validation_idx]

        # A feature can be present in the panel and still be entirely empty
        # inside the TRAINING window — the market columns only exist for the
        # seasons a line feed covered, which here is the validation season and
        # nothing before it. resolve_feature_cols cannot see this: it runs on
        # the whole panel, where the column looks partially populated.
        #
        # Such a column teaches a model nothing and actively harms one that
        # drops incomplete rows. Worse, a feature observed only in validation
        # is exactly the shape of a leak, so it is refused rather than
        # tolerated.
        empty_in_train = [
            c for c in feature_cols
            if c in train.columns and not train[c].notna().any()
        ]
        if empty_in_train:
            logger.warning(
                "Market %s: dropping %d feature(s) with NO values in the "
                "training window %s: %s. They are populated later in the panel, "
                "so they would be visible only where the model is scored.",
                market, len(empty_in_train), train_end, empty_in_train,
            )
            feature_cols = [c for c in feature_cols if c not in empty_in_train]
            xgb_cols = [c for c in xgb_cols if c not in empty_in_train]
            if not xgb_cols:
                logger.error(
                    "Market %s has no feature with training-window coverage — "
                    "skipping.", market,
                )
                continue

        components = build_components(market, feature_cols, cfg, xgb_feature_cols=xgb_cols)
        weights_series, recency_report = recency_sample_weights(train, cfg, market)
        if recency_report is not None:
            recency_reports.append(recency_report)

        fitted: dict[str, Any] = {}
        for name, model in components.items():
            try:
                # Passed only to components whose fit declares it. Inspected
                # rather than hardcoded: DistributionPropModel estimates a
                # dispersion and takes no weights, and a component added later
                # must not start raising TypeError here.
                if weights_series is not None and _accepts_sample_weight(model.fit):
                    model.fit(train, val, sample_weight=weights_series)
                else:
                    model.fit(train, val)
                fitted[name] = model
            except Exception as exc:  # noqa: BLE001
                # A component that cannot fit is skipped so the run continues,
                # but it is RECORDED. Skipping silently is how catboost ran at
                # 0.50 of the configured ensemble weight while contributing
                # nothing, with no exported artifact saying so.
                logger.warning("fit failed market=%s model=%s: %s", market, name, exc)
                fit_failures.append({
                    "target_market": market,
                    "model_name": name,
                    "configured_ensemble_weight": float(weights.get(name, 0.0)),
                    "error": f"{type(exc).__name__}: {exc}",
                })

        if len(fitted) >= 2:
            component_weights = {k: weights.get(k, 0.0) for k in fitted}
            ens = EnsemblePropModel(
                {k: fitted[k] for k in fitted},
                weights=component_weights,
                target_market=market,
            )
            fitted["ensemble"] = ens

            # What the blend ACTUALLY is, beside what was configured. A
            # component that failed to fit never reaches component_weights at
            # all, so it cannot appear in the ensemble's own dropped-models
            # warning -- its absence is invisible without this record. A
            # component that fitted but carries no configured weight is
            # silently excluded too, which is worth seeing.
            live = {k: v for k, v in component_weights.items() if v > 0}
            total = sum(live.values())
            effective = (
                {k: round(v / total, 6) for k, v in live.items()} if total > 0 else {}
            )
            ensemble_composition.append({
                "target_market": market,
                "configured": dict(weights),
                "effective": effective,
                "failed_to_fit": sorted(
                    set(weights) - set(fitted) - {"ensemble"}
                ),
                "fitted_but_unweighted": sorted(
                    k for k, v in component_weights.items() if v <= 0
                ),
            })
            unweighted = [k for k, v in component_weights.items() if v <= 0]
            if unweighted:
                logger.warning(
                    "Market %s: %s fitted but carry no ensemble weight, so the "
                    "blend excludes them. Add them to ensemble_weights or accept "
                    "that they are model comparisons only.",
                    market, unweighted,
                )
            if effective != {k: round(v / sum(weights.values()), 6)
                             for k, v in weights.items() if v > 0}:
                logger.warning(
                    "Market %s: the ensemble is %s, NOT the configured %s.",
                    market, effective, dict(weights),
                )

        y_true = val["over_hit"].astype(float).to_numpy()
        actual = pd.to_numeric(val[market], errors="coerce").to_numpy() if market in val.columns else np.full(len(val), np.nan)
        market_scores: dict[str, dict[str, Any]] = {}

        for name, model in fitted.items():
            try:
                preds = model.predict_rows(val, line_col="RESEARCH_LINE")
            except Exception as exc:  # noqa: BLE001
                logger.warning("predict failed market=%s model=%s: %s", market, name, exc)
                continue
            p_over = np.array([p.probability_over if p.probability_over is not None else np.nan for p in preds], dtype=float)
            means = np.array([p.prediction_mean if p.prediction_mean is not None else np.nan for p in preds], dtype=float)

            # Calibrate using only data earlier than this evaluation window.
            calibrator, calib_info = fit_calibrator_from_earlier_data(
                name, market, feature_cols, xgb_cols, cfg, train,
                fitted_model=model,
            )
            # Fit and apply must speak the same probability -- see
            # apply_calibration, which un-carves push mass before transforming
            # and carves it back exactly once.
            p_cal = apply_calibration(preds, p_over, calibrator)
            if calibrator is not None:
                calibration_meta[(market, name)] = calib_info
            else:
                logger.info(
                    "calibration skipped market=%s model=%s: %s",
                    market, name, calib_info.get("reason"),
                )
                for pred in preds:
                    pred.warnings.append(f"Uncalibrated: {calib_info.get('reason')}")

            bin_s = score_binary(y_true, p_over)
            mean_s = score_mean(actual, means)
            row = {
                "target_market": market,
                "model_name": name,
                "evaluation_start_date": str(split.validation_start.date()),
                "evaluation_end_date": str(split.validation_end.date()),
                "n_predictions": int(bin_s["n"] or 0),
                "mae": mean_s.get("mae"),
                "rmse": mean_s.get("rmse"),
                "mean_bias": mean_s.get("mean_bias"),
                "brier_score": bin_s.get("brier"),
                "log_loss": bin_s.get("log_loss"),
                "calibration_error": None,
                "calibration_error_ungated": None,
                "calibration_gate_passed": None,
                "calibration_bin_coverage": None,
                # The CALIBRATED counterparts. Without them the comparison
                # fits a calibrator, applies it to the exported predictions,
                # and then scores the raw number — so nothing in the harness
                # could say whether calibration helped or hurt.
                "brier_score_calibrated": None,
                "log_loss_calibrated": None,
                "calibration_error_calibrated": None,
                "calibration_error_calibrated_ungated": None,
                "calibration_gate_passed_calibrated": None,
                "calibration_source": (calib_info or {}).get("source"),
                "calibration_rows": (calib_info or {}).get("n_rows"),
                "interval_coverage": None,
                "notes": "RESEARCH_ONLY; RESEARCH_LINE={stat}_L10; not sportsbook",
            }
            cal_mask = np.isfinite(p_cal) & np.isfinite(y_true)
            if cal_mask.sum() >= 20:
                cal_s = score_binary(y_true[cal_mask], p_cal[cal_mask])
                row["brier_score_calibrated"] = cal_s.get("brier")
                row["log_loss_calibrated"] = cal_s.get("log_loss")
                cal_ece = expected_calibration_error(y_true[cal_mask], p_cal[cal_mask])
                row["calibration_error_calibrated"] = (
                    round(cal_ece["ece"], 4) if cal_ece["ece"] is not None else None
                )
                row["calibration_error_calibrated_ungated"] = (
                    round(cal_ece["ece_ungated"], 4)
                    if cal_ece.get("ece_ungated") is not None else None
                )
                row["calibration_gate_passed_calibrated"] = bool(cal_ece["gate_passed"])

            # ECE through the GATED implementation, not the inline proxy this
            # replaced. The arithmetic is identical -- verified: on a dense
            # sample both give 0.0179 -- so nothing already measured changes
            # value. What changes is the sparse case: the proxy computed
            # 0.0439 from 2 of 10 non-empty bins and reported it as if it
            # meant something, while prob_calibration.expected_calibration_error
            # returns None below 80% bin coverage and keeps the number under
            # ece_ungated. A reliability diagram with two occupied bins is not
            # a calibration measurement, and this harness's numbers feed model
            # selection.
            mask = np.isfinite(p_over) & np.isfinite(y_true)
            if mask.sum() >= 20:
                raw_ece = expected_calibration_error(y_true[mask], p_over[mask])
                table = raw_ece["table"]
                row["calibration_error"] = (
                    round(raw_ece["ece"], 4) if raw_ece["ece"] is not None else None
                )
                row["calibration_error_ungated"] = (
                    round(raw_ece["ece_ungated"], 4)
                    if raw_ece.get("ece_ungated") is not None else None
                )
                row["calibration_gate_passed"] = bool(raw_ece["gate_passed"])
                row["calibration_bin_coverage"] = raw_ece["bin_coverage"]
                if table:
                    for t in table:
                        calib_rows.append(
                            {
                                "target_market": market,
                                "model_name": name,
                                **t,
                                "evaluation_start_date": row["evaluation_start_date"],
                                "evaluation_end_date": row["evaluation_end_date"],
                            }
                        )
            summary_rows.append(row)
            market_scores[name] = row

            for i, pred in enumerate(preds):
                r = val.iloc[i]
                # Calendar game_date + cutoffs presented in Pacific (project display TZ).
                gdate = pd.to_datetime(r.get("GAME_DATE"), errors="coerce")
                game_date_pt = None
                if pd.notna(gdate):
                    # Date-only rows: treat as Pacific calendar date (no feed TZ shown).
                    game_date_pt = str(gdate.date())
                detail_rows.append(
                    {
                        "event_id": pred.event_id,
                        "game_date": game_date_pt,
                        "game_start_pt": None,
                        "player_id": pred.player_id,
                        "player_name": pred.player_name,
                        "player_team": r.get("TEAM_ABBREVIATION"),
                        "opponent": r.get("OPPONENT_ABBREVIATION"),
                        "home_away": "home" if bool(r.get("IS_HOME")) else "away",
                        "target_market": market,
                        "prop_line": pred.prop_line,
                        "line_type": "research_l10",
                        "prediction_mean": pred.prediction_mean,
                        "prediction_std_or_dispersion": pred.prediction_std_or_dispersion,
                        "probability_over_raw": pred.probability_over,
                        "probability_under_raw": pred.probability_under,
                        "probability_push_raw": pred.probability_push,
                        "probability_over_calibrated": pred.probability_over_calibrated,
                        "probability_under_calibrated": pred.probability_under_calibrated,
                        "model_name": pred.model_name,
                        "model_version": pred.model_version,
                        "ensemble_weight": (pred.extras or {}).get("component_weights"),
                        "feature_schema_version": pred.feature_schema_version,
                        # The cutoff is a Pacific CALENDAR DATE, not an instant.
                        # Reading its naive midnight as UTC shifted the display
                        # back a day: 2025-02-01 rendered as 2025-01-31T16:00.
                        "data_cutoff_pt": format_pacific_iso(
                            pacific_midnight_utc(pd.Timestamp(split.train_end).date())
                        ),
                        "prediction_timestamp_pt": format_pacific_iso(pred.prediction_timestamp_utc),
                        "actual_stat_value": float(actual[i]) if np.isfinite(actual[i]) else None,
                        "settlement": None,
                        "warnings": "|".join(pred.warnings) if pred.warnings else None,
                    }
                )

            if name == "catboost" and hasattr(fitted[name], "feature_importance_frame"):
                importance_rows.extend(fitted[name].feature_importance_frame().to_dict(orient="records"))

        # Winner by Brier then log_loss then calibration_error
        ranked = [
            (n, s)
            for n, s in market_scores.items()
            if s.get("brier_score") is not None and n != "ensemble"
        ]
        ranked.sort(key=lambda t: _winner_rank_key(t[1]))
        if ranked:
            winners[market] = {
                "winner": ranked[0][0],
                "brier_score": ranked[0][1]["brier_score"],
                "log_loss": ranked[0][1].get("log_loss"),
                "calibration_error": ranked[0][1].get("calibration_error"),
                "note": "Lowest Brier on validation window; not a profitability claim",
            }

    detail_frame = pd.DataFrame(detail_rows)
    market_rows: list[dict[str, Any]] = []
    bucket_rows: list[dict[str, Any]] = []
    confidence_rows: list[dict[str, Any]] = []
    if not detail_frame.empty:
        from src.models.market_comparison import (
            confidence_is_informative,
            edge_bucket_report,
            model_vs_line_report,
        )

        market_rows = model_vs_line_report(detail_frame).to_dict(orient="records")
        bucket_frame = edge_bucket_report(detail_frame)
        bucket_rows = bucket_frame.to_dict(orient="records")
        if not bucket_frame.empty:
            pairs = bucket_frame[["target_market", "model_name"]].drop_duplicates()
            confidence_rows = [
                confidence_is_informative(
                    bucket_frame,
                    group={"target_market": r.target_market, "model_name": r.model_name},
                )
                for r in pairs.itertuples(index=False)
            ]

    return {
        "summary": summary_rows,
        "predictions": detail_rows,
        "feature_importance": importance_rows,
        "calibration": calib_rows,
        "calibration_meta": {f"{m}|{n}": v for (m, n), v in calibration_meta.items()},
        "model_vs_line": market_rows,
        "edge_buckets": bucket_rows,
        "confidence_verdicts": confidence_rows,
        "winners": winners,
        # Which components failed to fit, and what the ensemble actually is.
        # Exported so a run that quietly lost half its configured weight
        # leaves a record instead of an unremarkable-looking summary row.
        "fit_failures": fit_failures,
        "recency_weighting": recency_reports,
        "ensemble_composition": ensemble_composition,
    }
