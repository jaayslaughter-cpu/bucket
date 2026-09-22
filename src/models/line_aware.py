"""
src/models/line_aware.py — make the classifiers answer AT a line.

THE PROBLEM THIS FIXES (docs/DATA_GAPS.md item 13). ``over_hit`` is
P(stat > RESEARCH_LINE), and the line is not one of the model's inputs.
A fitted classifier therefore returns the identical probability whatever
line it is asked about. Until now the only honest response was to abstain
at any line the labels were not built from, which on real posted lines
means abstaining on nearly every row.

THE FIX. Put the line INTO the features, and train against labels built
at many lines, so the model learns P(stat > L | features, L) rather than
P(stat > one particular L | features).

THE TRAP, stated plainly because it would be invisible in the metrics.
Candidate lines must be generated from PREGAME quantities only. Centre
them on the realised stat — even loosely, even with noise — and the line
feature encodes the outcome. The model would then look superb in
validation and be worthless on a real slate, because at scoring time the
line comes from a sportsbook that cannot see the result either. Every
generator here is anchored to a pregame baseline, and
``assert_lines_are_pregame`` exists to prove it.

AUGMENTATION IS NOT EXTRA DATA. Expanding one player-game into several
(line, label) pairs multiplies rows without adding independent outcomes.
All copies of a game share one realised stat, so they must never be split
across train and validation — that would put the same outcome on both
sides. They share a GAME_DATE, so a chronological split keeps them
together; ``assert_no_augmented_row_straddles`` checks it rather than
trusting it.

RESEARCH_ONLY. Nothing here invents a line: real posted lines are used
when supplied, and generated candidates are labelled as such.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.models.prediction_schema import ModelPrediction

logger = logging.getLogger(__name__)

# Columns this module adds. Feed these to the classifier alongside the
# existing pregame features to make it line-aware.
LINE_FEATURE_COLS = (
    "LINE",
    "LINE_MINUS_BASELINE",
    "LINE_Z",
    "LINE_IS_WHOLE",
)

# Identifies which original player-game an augmented row came from.
SOURCE_ROW_COL = "_line_aug_source_row"
LINE_SOURCE_COL = "line_source"

# Offsets applied to a PREGAME centre, in points, snapped to half-lines.
# Spread wide enough that the model sees the probability curve, not a
# single point on it.
DEFAULT_LINE_OFFSETS = (-6.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 6.0)

MIN_SCALE = 1.0  # floor on the pregame sd, so early-season rows do not explode


def attach_pregame_scale(
    df: pd.DataFrame,
    stat: str,
    *,
    window: int = 10,
    out_col: str | None = None,
) -> pd.DataFrame:
    """
    Shift-1 rolling standard deviation per player-season.

    The line z-score needs a per-player scale, and it must be a PREGAME
    one. Shift and window happen inside a single transform so the window
    cannot reach across a player boundary — the same construction the
    feature builder uses, for the same reason.
    """
    stat = stat.upper()
    target = out_col or f"{stat}_PREGAME_SD"
    work = df.copy()
    if stat not in work.columns:
        raise ValueError(f"DATA_NOT_AVAILABLE: {stat} absent — cannot compute its scale")

    keys = ["PLAYER_ID", "SEASON"] if "SEASON" in work.columns else ["PLAYER_ID"]
    work[stat] = pd.to_numeric(work[stat], errors="coerce")

    def _prior_std(series: pd.Series) -> pd.Series:
        return series.shift(1).rolling(window=window, min_periods=3).std()

    work[target] = work.groupby(keys, sort=False)[stat].transform(_prior_std)
    return work


def _centre_column(df: pd.DataFrame, stat: str) -> str:
    """The pregame estimate a line is measured against."""
    for candidate in (f"{stat}_BASELINE", f"{stat}_L10", f"{stat}_L5", f"{stat}_SEASON"):
        if candidate in df.columns:
            return candidate
    raise ValueError(
        f"DATA_NOT_AVAILABLE: no pregame centre for {stat} "
        f"(looked for {stat}_BASELINE/_L10/_L5/_SEASON). Run build_feature_matrix first."
    )


def attach_line_features(
    df: pd.DataFrame,
    stat: str,
    *,
    line_col: str = "LINE",
) -> pd.DataFrame:
    """
    Derive the features that let a model reason about the line.

    The raw line alone is nearly useless: the model would have to learn a
    separate scale for every player. What generalises is the line's
    position RELATIVE to that player's own pregame form, which is what
    ``LINE_MINUS_BASELINE`` and ``LINE_Z`` carry. A line two points above
    a player's recent average means something similar whether they average
    ten points or thirty.
    """
    stat = stat.upper()
    work = df.copy()
    if line_col not in work.columns:
        raise ValueError(f"DATA_NOT_AVAILABLE: line column {line_col!r} not present")

    centre_col = _centre_column(work, stat)
    scale_col = f"{stat}_PREGAME_SD"
    if scale_col not in work.columns:
        work = attach_pregame_scale(work, stat)

    line = pd.to_numeric(work[line_col], errors="coerce")
    centre = pd.to_numeric(work[centre_col], errors="coerce")
    scale = pd.to_numeric(work[scale_col], errors="coerce")

    # Fall back to sqrt(mean) — the Poisson scale — only where the rolling
    # sd has too little history to exist. This is a scale, not a
    # measurement, so it does not fabricate an observation.
    poisson_scale = np.sqrt(centre.clip(lower=MIN_SCALE))
    scale = scale.fillna(poisson_scale).clip(lower=MIN_SCALE)

    work["LINE"] = line
    work["LINE_MINUS_BASELINE"] = line - centre
    work["LINE_Z"] = (line - centre) / scale
    # A whole-number line can push, which changes what P(over) even means.
    work["LINE_IS_WHOLE"] = (line == line.round()).astype(float)
    work[f"{stat}_LINE_CENTRE"] = centre
    return work


def label_at_line(
    df: pd.DataFrame,
    stat: str,
    *,
    line_col: str = "LINE",
    target_col: str = "over_hit",
) -> pd.DataFrame:
    """
    Label each row at ITS OWN line: 1 over, 0 under, NaN on a push.

    A push is dropped rather than graded, exactly as the single-line
    labeller does — counting an exact tie as a loss would bias the target.
    """
    stat = stat.upper()
    work = df.copy()
    actual = pd.to_numeric(work[stat], errors="coerce")
    line = pd.to_numeric(work[line_col], errors="coerce")

    work[target_col] = pd.Series(pd.NA, index=work.index, dtype="Float64")
    work.loc[actual > line, target_col] = 1.0
    work.loc[actual < line, target_col] = 0.0

    pushes = int((actual == line).sum())
    labelled = int(work[target_col].notna().sum())
    logger.info(
        "Labelled %s at %s: %d rows (%d pushes dropped)",
        stat, line_col, labelled, pushes,
    )
    return work


def _source_row_identity(df: pd.DataFrame) -> pd.Series:
    """
    A STABLE id for the player-game an augmented copy came from.

    Not the positional index. Train and validation are augmented separately
    and each resets its own index, so positional ids collide across the two
    frames: different player-games are handed the same number. That made
    ``assert_no_augmented_row_straddles`` fire on a perfectly clean
    chronological split — and, worse, it would also MISS a real straddle
    whenever the positions happened not to line up. Identity has to come
    from the data.
    """
    keys = [c for c in ("PLAYER_ID", "GAME_ID") if c in df.columns]
    if len(keys) == 2:
        return (
            df["PLAYER_ID"].astype("string").fillna("?")
            + "@"
            + df["GAME_ID"].astype("string").fillna("?")
        )
    if "PLAYER_ID" in df.columns and "GAME_DATE" in df.columns:
        return (
            df["PLAYER_ID"].astype("string").fillna("?")
            + "@"
            + pd.to_datetime(df["GAME_DATE"], errors="coerce").dt.strftime("%Y-%m-%d")
        )
    # No identity columns: fall back to position and say so, because the
    # straddle check is only as good as the id it compares.
    logger.warning(
        "augment_lines: no PLAYER_ID/GAME_ID to identify a source row — falling "
        "back to the positional index, which cannot detect a straddle across "
        "two separately-indexed frames."
    )
    return pd.Series(df.index, index=df.index).astype("string")


def augment_lines(
    df: pd.DataFrame,
    stat: str,
    *,
    offsets: tuple[float, ...] = DEFAULT_LINE_OFFSETS,
    snap_to_half: bool = True,
    keep_real_line_col: str | None = None,
) -> pd.DataFrame:
    """
    Expand each player-game into one row per candidate line.

    Candidates are anchored to the player's PREGAME centre, never to the
    realised stat — see the module docstring. ``keep_real_line_col`` adds
    the genuinely posted line as an extra candidate, marked
    ``line_source='posted'`` so real and generated rows stay separable in
    analysis and in any report.

    The result carries ``_line_aug_source_row`` so callers can verify that
    copies of one game never straddle a train/validation boundary.
    """
    stat = stat.upper()
    work = df.copy().reset_index(drop=True)
    work[SOURCE_ROW_COL] = _source_row_identity(work)

    centre_col = _centre_column(work, stat)
    centre = pd.to_numeric(work[centre_col], errors="coerce")

    frames: list[pd.DataFrame] = []
    for offset in offsets:
        candidate = centre + float(offset)
        if snap_to_half:
            # Books post half-points far more often than whole numbers, and
            # a half-line cannot push. Snapping to .5 keeps the generated
            # board shaped like a real one.
            candidate = np.floor(candidate) + 0.5
        block = work.copy()
        block["LINE"] = candidate
        block[LINE_SOURCE_COL] = "generated"
        block["line_offset"] = float(offset)
        frames.append(block)

    if keep_real_line_col and keep_real_line_col in work.columns:
        posted = work.copy()
        posted["LINE"] = pd.to_numeric(posted[keep_real_line_col], errors="coerce")
        posted[LINE_SOURCE_COL] = "posted"
        posted["line_offset"] = np.nan
        posted = posted.loc[posted["LINE"].notna()]
        if not posted.empty:
            frames.append(posted)
            logger.info("Included %d genuinely posted line(s) as candidates", len(posted))

    out = pd.concat(frames, ignore_index=True)
    out = out.loc[out["LINE"].notna()]
    # A negative line is not a thing a book posts for a counting stat.
    out = out.loc[out["LINE"] > 0]

    out = attach_line_features(out, stat, line_col="LINE")
    out = label_at_line(out, stat, line_col="LINE")

    logger.info(
        "Line augmentation for %s: %d source rows -> %d (line, label) pairs "
        "across %d offsets",
        stat, len(work), len(out), len(offsets),
    )
    return out.reset_index(drop=True)


def assert_lines_are_pregame(
    df: pd.DataFrame,
    stat: str,
    *,
    line_col: str = "LINE",
    max_abs_corr: float = 0.35,
) -> dict[str, float]:
    """
    Fail loudly if the candidate lines encode the outcome.

    A line generated from a pregame baseline correlates with the realised
    stat only as much as the baseline itself does — real, but moderate. A
    line derived from the outcome correlates near 1.0. This is the check
    that separates a legitimate augmentation from the leak that would make
    validation look superb and a live slate look broken.

    Returns the measured correlations. Raises when the line tracks the
    outcome more closely than the pregame centre it was supposedly built
    from, which is the signature of the leak.
    """
    stat = stat.upper()
    actual = pd.to_numeric(df[stat], errors="coerce")
    line = pd.to_numeric(df[line_col], errors="coerce")
    centre_col = _centre_column(df, stat)
    centre = pd.to_numeric(df[centre_col], errors="coerce")

    mask = actual.notna() & line.notna() & centre.notna()
    if mask.sum() < 30:
        raise ValueError(
            f"DATA_NOT_AVAILABLE: only {int(mask.sum())} comparable rows — "
            "too few to test whether the lines are pregame"
        )

    line_corr = float(np.corrcoef(line[mask], actual[mask])[0, 1])
    centre_corr = float(np.corrcoef(centre[mask], actual[mask])[0, 1])

    result = {
        "line_vs_actual_corr": round(line_corr, 4),
        "centre_vs_actual_corr": round(centre_corr, 4),
        "n": int(mask.sum()),
    }

    # The line may not track the outcome more tightly than the pregame
    # estimate it was built from. A small tolerance absorbs sampling noise.
    if abs(line_corr) > abs(centre_corr) + 0.05:
        raise LineLeakageError(
            f"Candidate lines correlate with the realised {stat} at "
            f"{line_corr:.3f}, ABOVE the pregame centre's {centre_corr:.3f}. "
            "The lines encode the outcome — a model trained on these would "
            f"validate beautifully and fail on a real slate. {result}"
        )
    if abs(line_corr) > max_abs_corr and abs(centre_corr) <= max_abs_corr:
        raise LineLeakageError(
            f"Candidate lines correlate with realised {stat} at {line_corr:.3f}, "
            f"above the {max_abs_corr} ceiling. {result}"
        )
    return result


def assert_no_augmented_row_straddles(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    """
    Verify no player-game appears on both sides of a split.

    Augmented copies share one realised outcome. If some land in train and
    others in validation, the model has already seen the answer, and every
    validation metric is optimistic by an amount nobody can estimate after
    the fact.
    """
    if SOURCE_ROW_COL not in train.columns or SOURCE_ROW_COL not in validation.columns:
        return  # not augmented frames — nothing to check
    shared = set(train[SOURCE_ROW_COL]) & set(validation[SOURCE_ROW_COL])
    if shared:
        raise LineLeakageError(
            f"{len(shared)} player-game(s) have augmented copies in BOTH train "
            "and validation. All copies share one realised outcome, so the "
            "model has seen the answer. Split on GAME_DATE, which keeps copies "
            "together."
        )


class LineLeakageError(RuntimeError):
    """Raised when the line feature would carry the outcome into training."""


class LineAwarePropModel:
    """
    Wraps an existing classifier so it answers AT the line it is asked about.

    Architecture is preserved: this does not replace CatBoost or XGBoost,
    it trains one of them on an augmented frame whose features include the
    line. The wrapped model's own defaults, splitting and early stopping
    are untouched.

    ``predict_probability_over`` attaches the line features for the
    requested line and asks the model, so two different lines on the same
    player-game genuinely produce two different probabilities — which is
    the whole point, and is asserted in the tests.
    """

    model_name = "line_aware"

    def __init__(
        self,
        base_factory,
        *,
        stat: str = "PTS",
        base_feature_cols: list[str] | None = None,
        offsets: tuple[float, ...] = DEFAULT_LINE_OFFSETS,
    ) -> None:
        self.stat = stat.upper()
        self.base_feature_cols = list(base_feature_cols or [])
        self.offsets = tuple(offsets)
        self._base_factory = base_factory
        self.model = None
        self.feature_cols: list[str] = []
        self.augmentation_report: dict[str, float] = {}
        # The standardised line range actually seen in training. Outside
        # it the model extrapolates, and a boosted tree extrapolates
        # badly — see predict_probability_over.
        self.trained_z_range: tuple[float, float] | None = None

    def fit(self, panel: pd.DataFrame, validation_data: pd.DataFrame | None = None):
        """
        Augment, verify the lines are pregame, then fit the wrapped model.

        The leakage check runs BEFORE fitting, not after: a model trained
        on outcome-derived lines would post excellent validation numbers,
        and no metric computed afterwards would reveal why.
        """
        augmented = augment_lines(panel, self.stat, offsets=self.offsets)
        self.augmentation_report = assert_lines_are_pregame(augmented, self.stat)

        self.feature_cols = [
            c for c in (*self.base_feature_cols, *LINE_FEATURE_COLS)
            if c in augmented.columns
        ]
        missing_line_cols = [c for c in LINE_FEATURE_COLS if c not in self.feature_cols]
        if missing_line_cols:
            raise ValueError(
                f"DATA_NOT_AVAILABLE: line features {missing_line_cols} absent after "
                "augmentation — the model would not be line-aware at all."
            )

        labelled = augmented.loc[augmented["over_hit"].notna()].reset_index(drop=True)
        # Record the support. Bounds are taken on LINE_Z rather than the raw
        # line because a line of 20.5 is ordinary for a 25-point scorer and
        # far outside anything sane for a 10-point one.
        z = pd.to_numeric(labelled["LINE_Z"], errors="coerce").dropna()
        if not z.empty:
            self.trained_z_range = (
                float(z.quantile(0.01)), float(z.quantile(0.99)),
            )
        self.model = self._base_factory(self.feature_cols)
        # Tell the wrapped model to stop abstaining: the line is now one of
        # its inputs, so the mask that guards a line-blind classifier would
        # discard every answer it produces.
        self.model.line_aware = True
        val = validation_data
        if val is not None and not val.empty:
            val_aug = augment_lines(val, self.stat, offsets=self.offsets)
            assert_lines_are_pregame(val_aug, self.stat)
            val = val_aug.loc[val_aug["over_hit"].notna()].reset_index(drop=True)
            assert_no_augmented_row_straddles(labelled, val)
        self.model.fit(labelled, val)
        logger.info(
            "Line-aware %s fitted on %d (line, label) pairs; %s",
            self.stat, len(labelled), self.augmentation_report,
        )
        return self

    def predict_probability_over(
        self,
        features: pd.DataFrame,
        line: float | pd.Series,
    ) -> pd.Series:
        if self.model is None:
            raise RuntimeError("LineAwarePropModel is not fitted")
        work = self._prepare(features, line)
        probs = self.model.predict_probability_over(work, work["LINE"])

        return self._mask_out_of_support(probs, work)

    def _prepare(self, features: pd.DataFrame, line) -> pd.DataFrame:
        """Attach the line features the wrapped model was fitted with."""
        work = features.copy()
        work["LINE"] = (
            pd.to_numeric(line, errors="coerce") if isinstance(line, pd.Series)
            else float(line)
        )
        return attach_line_features(work, self.stat, line_col="LINE")

    def _mask_out_of_support(self, probs: pd.Series, work: pd.DataFrame) -> pd.Series:
        """
        Blank any probability asked at a line the fit never saw.

        Outside the trained range the model is extrapolating, and a boosted
        tree has no sensible behaviour there — the probability can even tick
        UP with the line, which is impossible for a survival function.
        """
        if self.trained_z_range is None:
            return probs
        low, high = self.trained_z_range
        z = pd.to_numeric(work["LINE_Z"], errors="coerce")
        outside = (z < low) | (z > high) | z.isna()
        n_outside = int(outside.sum())
        if n_outside:
            probs = probs.copy()
            probs.loc[outside] = float("nan")
            logger.warning(
                "%s: %d of %d row(s) asked at a line outside the trained "
                "range (LINE_Z %.2f..%.2f) — abstaining rather than "
                "extrapolating.",
                self.stat, n_outside, len(work), low, high,
            )
        return probs

    # ------------------------------------------------------------------
    # the contract compare.py actually calls
    # ------------------------------------------------------------------

    @property
    def model_version(self) -> str:
        base = getattr(self.model, "model_version", "unknown")
        return f"line_aware/{getattr(self.model, 'model_name', 'base')}@{base}"

    @property
    def target_market(self) -> str:
        return getattr(self.model, "target_market", self.stat)

    @property
    def feature_schema_version(self) -> str:
        return getattr(self.model, "feature_schema_version", "fs_v1_shift1_l2")

    @property
    def dispersion(self):
        """The wrapped model owns the fitted count distribution."""
        return getattr(self.model, "dispersion", None)

    @property
    def oof(self):
        """
        Out-of-fold probabilities from the wrapped model.

        They were produced on the AUGMENTED frame, which is the right sample:
        that is what this model was fitted on, and the calibrator it feeds
        corrects predictions made at a line.
        """
        return getattr(self.model, "oof", None)

    def predict_distribution(self, features: pd.DataFrame) -> pd.DataFrame:
        """Mean and spread come from the wrapped model, unchanged."""
        if self.model is None:
            raise RuntimeError("LineAwarePropModel is not fitted")
        return self.model.predict_distribution(self._prepare(features, np.nan))

    def predict_mean(self, features: pd.DataFrame) -> pd.Series:
        """The projection. A property of the player-game, not of the line."""
        if self.model is None:
            raise RuntimeError("LineAwarePropModel is not fitted")
        work = self._prepare(features, np.nan)
        if hasattr(self.model, "predict_mean"):
            return self.model.predict_mean(work)
        return self.model.predict_distribution(work)["prediction_mean"]

    def predict_rows(
        self,
        features: pd.DataFrame,
        *,
        line_col: str = "RESEARCH_LINE",
    ) -> list[ModelPrediction]:
        """
        Predictions at the requested line, reusing the wrapped model's mean,
        spread and push mass and substituting only the probability.

        Being line-aware changes exactly one thing: WHICH probability answers
        "over this number". The projection, the dispersion and the push mass
        are properties of the player-game, not of the line, so they are taken
        from the wrapped model rather than recomputed — a second copy of that
        arithmetic is a second place for it to drift.

        A row whose line falls outside the trained support gets
        ``probability_over = None`` and a named warning. Clipping an
        abstention into a number is the one outcome worse than not answering.
        """
        if self.model is None:
            raise RuntimeError("LineAwarePropModel is not fitted")
        if line_col not in features.columns:
            raise ValueError(f"DATA_NOT_AVAILABLE: missing {line_col}")

        # The wrapped model was fitted WITH the line features, so it must be
        # given them here too. Passing the raw frame asks it for columns it
        # learned on and cannot find.
        work = self._prepare(features, features[line_col])
        rows = self.model.predict_rows(work, line_col=line_col)
        probs = self._mask_out_of_support(
            self.model.predict_probability_over(work, work["LINE"]), work
        )

        abstained = 0
        for i, prediction in enumerate(rows):
            prediction.model_name = self.model_name
            prediction.model_version = self.model_version

            raw = probs.iloc[i] if i < len(probs) else float("nan")
            if pd.isna(raw):
                abstained += 1
                prediction.probability_over = None
                prediction.probability_under = None
                prediction.warnings = [
                    *prediction.warnings,
                    "line_aware abstained: the requested line is outside the "
                    "range the model was trained on, where a boosted tree "
                    "extrapolates without bound",
                ]
                continue

            p_over = float(np.clip(float(raw), 1e-6, 1.0 - 1e-6))
            # The push mass already computed by the wrapped model from its
            # fitted distribution; the classifier splits only what is left.
            p_push = prediction.probability_push or 0.0
            remaining = max(0.0, 1.0 - float(p_push))
            prediction.probability_over = round(remaining * p_over, 6)
            prediction.probability_under = round(remaining * (1.0 - p_over), 6)

        if abstained:
            logger.info(
                "line_aware %s: %d of %d row(s) abstained on an out-of-support line",
                self.stat, abstained, len(rows),
            )
        return rows

    def probability_curve(
        self,
        features: pd.DataFrame,
        lines: list[float],
    ) -> pd.DataFrame:
        """
        P(over) across a ladder of lines — the shape that proves awareness.

        A line-blind model returns a flat row here. A line-aware one
        returns a curve that falls as the line rises.
        """
        rows = []
        for line in lines:
            probs = self.predict_probability_over(features, float(line))
            usable = probs.notna().sum()
            rows.append({
                "line": float(line),
                "mean_probability_over": float(probs.mean()) if usable else float("nan"),
                "rows_in_support": int(usable),
                "rows_abstained": int(len(probs) - usable),
            })
        return pd.DataFrame(rows)

    def monotonicity_report(
        self,
        features: pd.DataFrame,
        lines: list[float],
    ) -> dict[str, Any]:
        """
        Check P(over) never rises as the line rises.

        P(stat > L) is a survival function in L, so any increase is the
        model contradicting itself. Reported rather than silently patched:
        a violation usually means the ladder reaches past the trained
        support, and that is worth seeing rather than smoothing away.
        """
        curve = self.probability_curve(features, lines)
        usable = curve.loc[curve["mean_probability_over"].notna()]
        values = usable["mean_probability_over"].to_numpy()
        diffs = np.diff(values)
        violations = int((diffs > 1e-9).sum())
        return {
            "lines_evaluated": len(curve),
            "lines_in_support": len(usable),
            "monotonic_non_increasing": bool(violations == 0),
            "violations": violations,
            "largest_increase": float(diffs.max()) if len(diffs) else 0.0,
            "probability_range": (
                float(values.max() - values.min()) if len(values) else 0.0
            ),
        }
