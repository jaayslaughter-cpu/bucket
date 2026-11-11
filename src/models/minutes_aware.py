"""`distribution`'s mean, with the minutes variance it never carried.

RESEARCH_ONLY. Produces a probability and a named abstention. Prices nothing.

WHAT THIS CHANGES, AND WHAT IT DELIBERATELY DOES NOT.
The mean is IDENTICAL to ``DistributionPropModel``'s -- ``{stat}_L2``, which is
``{stat}_BASELINE`` times ``fatigue_multiplier``. Nothing here re-projects a
level. Only the VARIANCE changes, and that is the whole point: the two
components can then be compared on the one thing that differs between them,
instead of on a tangle of mean and variance changes at once.

THE DEFECT IT ADDRESSES, MEASURED RATHER THAN ASSERTED.
``residuals.CountDispersion.variance_for`` is a function of the projected mean
and NOTHING ELSE. Two players projected for 7.4 points get identical
uncertainty, whether one is a 34-minute starter and the other a big whose
minutes swing 12 to 35 by matchup. On the 2026-11-09 slate the board's most
confident rows were overwhelmingly the second kind -- small lines on
low-usage players -- and 288 graded rows came back at a 0.510 strike rate
against a 0.702 mean stated probability, an ECE of 0.1917 against the gate's
0.0500 bar. ``src/models/minutes_model.py`` existed the whole time and nothing
in the repository imported it.

THE DECOMPOSITION. With ``S`` the stat, ``M`` the minutes and ``r`` a per-minute
rate, the law of total variance gives::

    Var(S) = E[Var(S | M)] + Var(E[S | M])
           = phi_cond * r * E[M]  +  r^2 * Var(M)
           = phi_cond * mu        +  mu^2 * (sd_M / E[M])^2

so the whole correction collapses to one scalar per row::

    phi_eff = phi_cond + mu * CV_M^2         where CV_M = sd_M / E[M]

That is handed to the SAME ``over_under_push_from_dispersion`` the other
components use, as a negative binomial whose ``variance_for(mu)`` is
``mu * phi``. One definition of "probability from a mean and a variance"; this
module supplies a variance and borrows the rest.

Note what the form says: the inflation grows with ``mu``. Minutes risk barely
moves a 2-point projection and dominates a 25-point one, which is the
opposite of what a single fitted ``phi`` can express.

WHY ``phi`` MUST BE REFIT AND CANNOT BE BORROWED. This is the trap, and
shipping without it would have been a silent double count.
``DistributionPropModel`` fits dispersion on UNCONDITIONAL residuals -- stat
against its pregame projection -- so minutes variance is ALREADY inside its
phi. Measured on the deployed panel that phi is about 4.5 for points. Adding
``mu * CV_M^2`` on top of 4.5 would count the same uncertainty twice and make
every probability further from honest, not closer. ``fit`` here therefore
measures ``Var(S | M)``: the pregame rate times the minutes ACTUALLY played,
against the realised stat. The two phis are not interchangeable and a loader
that confused them would misprice every row, so the artifact records which
basis it was fitted on and ``load`` refuses the other.

IS USING REALISED MINUTES AT FIT TIME LEAKAGE? No, and the distinction is
worth stating because it looks like it. ``phi_cond`` is a structural parameter
-- how dispersed a stat is around a rate once minutes are known -- estimated
from completed games, exactly as the unconditional phi is. It never becomes a
FEATURE. At scoring time realised minutes do not exist and are not consulted:
``E[M]`` is ``MIN_L2`` and the spread term stands in for what is unknown. The
pregame rate used in the fit is built only from ``.shift(1)`` rollups.

WHAT IT ABSTAINS ON, AND WHY NOT A FALLBACK. A row with no usable ``MIN_SD_L10``
(fewer than three prior games) gets no probability. Substituting a league
average CV, or quietly dropping the minutes term, would make this component
indistinguishable from ``distribution`` on precisely the rows where minutes are
least known -- early-season and new players, which is where the board's worst
rows live.

WHAT IS STILL MISSING. ``CV_M`` is a backward-looking spread over ten games.
It does not know tonight's blowout, an in-game injury, or a coach's rotation
change, so it is a floor on minutes uncertainty rather than a measure of it.
``MinutesModel`` (CatBoost, P10/P50/P90) would give a forward-looking spread;
it has no training entry and no artifact, so it is not used here, and this
module's own abstention is preferable to pretending otherwise.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.models.distribution_adapter import DistributionPropModel
from src.models.residuals import (
    CountDispersion,
    fit_count_dispersion,
    over_under_push_from_dispersion,
)

logger = logging.getLogger(__name__)

#: Written into the artifact and checked on load. The unconditional phi and
#: the conditional one differ by about a factor of two on real panels, so a
#: component that loaded the wrong one would misprice every row while looking
#: perfectly healthy.
DISPERSION_BASIS = "conditional_on_minutes"
UNCONDITIONAL_BASIS = "unconditional"

#: Columns this component needs beyond what `distribution` reads.
MINUTES_LEVEL_COL = "MIN_L2"
MINUTES_SPREAD_COL = "MIN_SD_L10"
MINUTES_BASELINE_COL = "MIN_BASELINE"


class MinutesAwareError(ValueError):
    """A fit or a score that cannot be done honestly."""


def minutes_cv(level: Any, spread: Any) -> float | None:
    """``sd_M / E[M]``, or None where it cannot be formed.

    Returns None rather than 0.0 on a missing spread. Zero is a CLAIM -- that
    the player's minutes never vary -- and it would silently collapse this
    component onto `distribution`.
    """
    try:
        e_m, sd_m = float(level), float(spread)
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(e_m) and np.isfinite(sd_m)):
        return None
    if e_m <= 0.0 or sd_m < 0.0:
        return None
    return sd_m / e_m


def effective_phi(
    dispersion: CountDispersion, mean: Any, cv: float | None
) -> float | None:
    """``phi_eff`` such that ``mu * phi_eff`` is the decomposed variance.

    ``Var_total = Var(S | M) + mu^2 * CV_M^2``, so
    ``phi_eff = Var(S | M)/mu + mu * CV_M^2``.

    THE CONDITIONAL TERM IS READ OFF THE FITTED FAMILY, not off ``phi``. An
    earlier version took ``dispersion.phi`` and substituted 1.0 for any family
    other than negbin -- which silently understated a ZIP or normal fit, whose
    variance at ``mu`` is not ``mu``. ``variance_for`` is the one function that
    knows what each family's variance is, so it is asked.
    """
    if cv is None:
        return None
    try:
        mu = float(mean)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(mu) or mu <= 0.0:
        return None
    conditional_variance = float(dispersion.variance_for(mu))
    if not np.isfinite(conditional_variance) or conditional_variance <= 0.0:
        return None
    return conditional_variance / mu + mu * (cv ** 2)


class MinutesAwarePropModel(DistributionPropModel):
    """`distribution`'s mean; variance decomposed over minutes."""

    model_name = "minutes_aware"
    ARTIFACT_SUFFIX = ".json"

    def fit(
        self,
        train_data: pd.DataFrame,
        validation_data: pd.DataFrame | None = None,
    ) -> "MinutesAwarePropModel":
        """Fit ``Var(S | M)`` -- NOT the unconditional dispersion.

        See the module docstring on why borrowing `distribution`'s phi would
        double count. Refuses rather than falling back: a Poisson default here
        would be combined with a real minutes term and the result would look
        like a fitted model.
        """
        stat = self.target_market
        needed = (stat, "MIN", MINUTES_BASELINE_COL, f"{stat}_BASELINE")
        missing = [c for c in needed if c not in train_data.columns]
        if missing:
            raise MinutesAwareError(
                f"DATA_NOT_AVAILABLE: minutes_aware needs {list(needed)} to fit a "
                f"conditional dispersion; missing {missing}. Refusing to fall back "
                f"to an unconditional phi, which already contains the minutes "
                f"variance this component adds separately"
            )

        work = train_data
        if "GAME_DATE" in work.columns:
            work = work.sort_values("GAME_DATE")

        # The PREGAME rate: prior-game rollups only, both shifted.
        rate = pd.to_numeric(work[f"{stat}_BASELINE"], errors="coerce") / pd.to_numeric(
            work[MINUTES_BASELINE_COL], errors="coerce"
        ).replace(0.0, np.nan)
        # ...times the minutes ACTUALLY played. A structural parameter measured
        # on completed games; never a feature. See the docstring.
        conditional_mean = rate * pd.to_numeric(work["MIN"], errors="coerce")

        self.dispersion = fit_count_dispersion(
            pd.to_numeric(work[stat], errors="coerce").to_numpy(dtype=float),
            conditional_mean.to_numpy(dtype=float),
            market=stat,
        )
        self._fitted = True
        logger.info(
            "minutes_aware %s: conditional dispersion family=%s phi=%.4f on %d row(s). "
            "This phi measures Var(stat | minutes) and is NOT interchangeable with "
            "distribution's unconditional one.",
            stat, self.dispersion.family, self.dispersion.phi,
            self.dispersion.n_train_rows,
        )
        return self

    def _conditional_dispersion(self) -> CountDispersion:
        if self.dispersion is None:
            raise MinutesAwareError(
                "minutes_aware is not fitted; there is no conditional dispersion "
                "to decompose against"
            )
        return self.dispersion

    def predict_probability_over(
        self,
        features: pd.DataFrame,
        line: float | pd.Series,
    ) -> pd.Series:
        means = self.predict_mean(features)
        lines = (
            line if isinstance(line, pd.Series)
            else pd.Series([line] * len(features), index=features.index)
        )
        for col in (MINUTES_LEVEL_COL, MINUTES_SPREAD_COL):
            if col not in features.columns:
                raise MinutesAwareError(
                    f"DATA_NOT_AVAILABLE: minutes_aware needs {col}. Without the "
                    f"minutes spread this component IS `distribution`, and scoring "
                    f"as if it were not would misreport which one answered"
                )
        conditional = self._conditional_dispersion()
        levels = features[MINUTES_LEVEL_COL]
        spreads = features[MINUTES_SPREAD_COL]

        out: list[float | None] = []
        for mu, ln, lvl, sd in zip(means, lines, levels, spreads):
            phi_eff = effective_phi(conditional, mu, minutes_cv(lvl, sd))
            if phi_eff is None:
                out.append(None)
                continue
            # A NEGATIVE BINOMIAL NEEDS VARIANCE STRICTLY ABOVE THE MEAN, and
            # phi_eff is exactly 1 when the fit chose Poisson and the spread is
            # zero. That row is not unpriceable -- its correct distribution is
            # the conditional one, unchanged, with no inflation to add. Forcing
            # a degenerate negbin on it, or abstaining, would both be wrong.
            realised: CountDispersion = (
                CountDispersion(
                    family="negbin",
                    phi=phi_eff,
                    n_train_rows=int(getattr(conditional, "n_train_rows", 0) or 0),
                    selection_scores={},
                )
                if phi_eff > 1.0
                else conditional
            )
            res = over_under_push_from_dispersion(
                float(mu) if pd.notna(mu) else float("nan"),
                float(ln) if pd.notna(ln) else float("nan"),
                realised,
            )
            out.append(res.get("probability_over"))
        return pd.Series(out, index=features.index, dtype="float64")

    def save(self, path: Any) -> None:
        """`distribution`'s payload plus the basis, so the two cannot be swapped."""
        import json
        from pathlib import Path

        super().save(path)
        target = Path(path)
        target = target if target.suffix == ".json" else target.with_suffix(".json")
        payload = json.loads(target.read_text(encoding="utf-8"))
        payload["dispersion_basis"] = DISPERSION_BASIS
        payload["model_name"] = self.model_name
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load(self, path: Any) -> "MinutesAwarePropModel":
        """Load, refusing an artifact fitted on the other basis.

        A `distribution` artifact carries an UNCONDITIONAL phi -- about 4.5 for
        points against roughly half that conditionally. Loaded here it would be
        added to a real minutes term and every probability would be wrong in the
        same direction, with nothing on the face of it to show why.
        """
        import json
        from pathlib import Path

        target = Path(path)
        target = target if target.suffix == ".json" else target.with_suffix(".json")
        if target.exists():
            basis = json.loads(target.read_text(encoding="utf-8")).get(
                "dispersion_basis", UNCONDITIONAL_BASIS
            )
            if basis != DISPERSION_BASIS:
                raise MinutesAwareError(
                    f"{target} was fitted on a {basis!r} dispersion and this "
                    f"component needs {DISPERSION_BASIS!r}. Its phi already "
                    f"contains the minutes variance added here, so loading it "
                    f"would count the same uncertainty twice"
                )
        super().load(target)
        return self
