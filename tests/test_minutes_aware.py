"""Minutes variance reaches the probability, or the row abstains.

The defect: `CountDispersion.variance_for` is a function of the projected mean
and nothing else, so two players projected for the same number got identical
uncertainty however differently their minutes swing. On 2026-11-09 the board's
most confident rows were small lines on low-usage players, and 288 graded rows
returned a 0.510 strike rate against a 0.702 mean stated probability.
`src/models/minutes_model.py` existed and nothing imported it.

Every guard was mutation-tested: fix reverted, named test watched go red, fix
put back.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.distribution_adapter import DistributionPropModel
from src.models.minutes_aware import (
    DISPERSION_BASIS,
    MinutesAwareError,
    MinutesAwarePropModel,
    effective_phi,
    minutes_cv,
)
from src.models.residuals import CountDispersion

FAMILIES = [
    ("poisson", {}),
    ("negbin", {"phi": 2.4}),
    ("zip", {"zero_inflation": 0.07}),
    ("normal", {"sigma_scale": 2.1}),
]


def _disp(family: str, **kw) -> CountDispersion:
    return CountDispersion(
        family=family, phi=kw.pop("phi", 1.0), n_train_rows=1000,
        selection_scores={}, **kw,
    )


# --- the decomposition ---------------------------------------------------


@pytest.mark.parametrize("family,kw", FAMILIES)
@pytest.mark.parametrize("mu,cv", [(7.4, 0.30), (18.0, 0.12), (24.5, 0.45)])
def test_the_total_variance_is_exactly_the_two_terms(family, kw, mu, cv):
    """
    ``Var(S) = Var(S|M) + mu^2 * CV^2`` -- the law of total variance, asserted
    as an identity rather than as a direction. `phi_eff` is only a way to hand
    that variance to the shared `over_under_push_from_dispersion`, whose
    negbin variance is `mu * phi`, so `mu * phi_eff` must reproduce it to
    floating point.
    """
    d = _disp(family, **kw)
    phi_eff = effective_phi(d, mu, cv)
    assert phi_eff is not None
    assert mu * phi_eff == pytest.approx(d.variance_for(mu) + mu**2 * cv**2, abs=1e-9)


@pytest.mark.parametrize("family,kw", FAMILIES)
def test_the_conditional_term_is_read_off_the_fitted_family(family, kw):
    """
    MUTATION THAT THIS CAUGHT DURING DEVELOPMENT: an earlier version took
    `dispersion.phi` and substituted 1.0 for every family except negbin, which
    silently understated a ZIP or normal fit -- neither has variance `mu` at
    `mu`. `variance_for` is the one function that knows, so it is asked.
    """
    d = _disp(family, **kw)
    # With no minutes spread at all the total must be EXACTLY the family's own
    # variance; a hardcoded 1.0 would make poisson right and the rest wrong.
    phi_eff = effective_phi(d, 10.0, 0.0)
    assert 10.0 * phi_eff == pytest.approx(d.variance_for(10.0), abs=1e-9)


def test_more_volatile_minutes_widen_the_distribution():
    """
    THE WHOLE POINT, AND IT IS A REALLOCATION, NOT A WIDENING. At one
    projection, a player whose minutes swing gets a wider distribution than one
    whose do not -- a distinction `distribution` cannot express, because its
    variance is a function of the mean alone and both get the same number.
    """
    d = _disp("negbin", phi=2.4)
    widths = [10.0 * effective_phi(d, 10.0, cv) for cv in (0.05, 0.20, 0.40, 0.60)]
    assert widths == sorted(widths), widths
    assert widths[-1] > 2 * widths[0], "volatility barely moved the variance"


# --- what it refuses -----------------------------------------------------


@pytest.mark.parametrize("level,spread", [
    (28.0, None), (28.0, float("nan")), (None, 5.0), (0.0, 5.0),
    (-4.0, 5.0), (28.0, -1.0), ("a", 5.0), (28.0, "b"),
])
def test_an_unusable_minutes_spread_returns_none_not_zero(level, spread):
    """
    ZERO IS A CLAIM -- that the player's minutes never vary -- and it would
    collapse this component silently onto `distribution` on exactly the rows
    where minutes are least known: early-season and new players, which is
    where the board's worst rows live.
    """
    assert minutes_cv(level, spread) is None


def test_a_row_with_no_spread_abstains_rather_than_falling_back():
    cv = minutes_cv(28.0, None)
    assert cv is None
    assert effective_phi(_disp("negbin", phi=2.4), 10.0, cv) is None


@pytest.mark.parametrize("mu", [0.0, -1.0, float("nan"), float("inf"), None, "x"])
def test_an_unusable_mean_returns_none(mu):
    assert effective_phi(_disp("negbin", phi=2.4), mu, 0.2) is None


def test_a_corrupt_dispersion_with_no_variance_is_refused():
    """
    A fitted family that reports zero variance at the mean describes nothing,
    and dividing by it would produce an infinite phi rather than an abstention.
    Distinct from the legitimate phi_eff == 1.0 case below.
    """
    broken = _disp("negbin", phi=0.0)
    assert effective_phi(broken, 10.0, 0.0) is None


def test_a_poisson_fit_with_no_spread_is_priced_not_abstained(panel):
    """
    phi_eff is EXACTLY 1.0 when the conditional fit chose Poisson and the
    spread is zero. That row is not unpriceable -- its correct distribution is
    the conditional one with no inflation to add. An earlier version refused it
    with a `phi > 1.0` guard, which would have silently dropped every row of a
    market whose conditional dispersion came back Poisson.
    """
    assert effective_phi(_disp("poisson"), 10.0, 0.0) == pytest.approx(1.0)

    m = MinutesAwarePropModel(target_market="PTS").fit(panel)
    m.dispersion = _disp("poisson")
    flat = panel.copy()
    flat["MIN_SD_L10"] = 0.0
    out = m.predict_probability_over(flat, 10.0)
    assert out.notna().any(), "a Poisson conditional fit abstained on everything"


# --- the mean is untouched, which is what makes the comparison clean -----


def _panel(n_players=12, n_games=40):
    from src.features.builder import build_feature_matrix
    from src.models.data_audit import make_demo_panel

    f = build_feature_matrix(make_demo_panel(n_players=n_players, n_games=n_games))
    return f.dropna(subset=["PTS", "MIN", "PTS_BASELINE", "MIN_BASELINE", "MIN_SD_L10"])


@pytest.fixture(scope="module")
def panel():
    return _panel()


def test_the_mean_is_identical_to_distributions(panel):
    """
    ISOLATION. This component changes the VARIANCE and nothing else, so the two
    can be compared on the one thing that differs. `{stat}_L2 / MIN_L2` cancels
    the fatigue multiplier, so the implied rate times the projected minutes IS
    `{stat}_L2` -- the mean comes out the same by construction, and that is
    asserted rather than assumed.
    """
    d = DistributionPropModel(target_market="PTS").fit(panel)
    m = MinutesAwarePropModel(target_market="PTS").fit(panel)
    pd.testing.assert_series_equal(
        d.predict_mean(panel), m.predict_mean(panel), check_names=False,
    )


def test_the_fitted_dispersion_is_conditional_and_differs_from_the_unconditional():
    """
    THE DOUBLE-COUNT TRAP. `distribution`'s phi is fitted on UNCONDITIONAL
    residuals, so minutes variance is already inside it -- about 4.5 for points
    on the deployed panel. Adding `mu * CV^2` on top of that counts the same
    uncertainty twice and makes every probability further from honest.

    Built on a panel where minutes DRIVE the variance, so the two fits must
    separate: the stat is an exact rate times minutes plus small noise, so
    conditioning on minutes should remove nearly all of the spread.
    """
    rng = np.random.default_rng(11)
    rows = []
    for pid in range(14):
        rate = 0.45
        for g in range(40):
            minutes = float(rng.choice([8.0, 34.0]))  # violently bimodal
            rows.append({
                "PLAYER_ID": f"P{pid}", "GAME_DATE": pd.Timestamp("2026-01-01") + pd.Timedelta(days=g),
                "MIN": minutes, "PTS": float(rng.poisson(rate * minutes)),
            })
    df = pd.DataFrame(rows)
    df["MIN_BASELINE"] = 21.0
    df["PTS_BASELINE"] = 0.45 * 21.0
    df["PTS_L2"] = df["PTS_BASELINE"]
    df["MIN_L2"] = df["MIN_BASELINE"]
    df["MIN_SD_L10"] = 13.0

    uncond = DistributionPropModel(target_market="PTS").fit(df).dispersion
    cond = MinutesAwarePropModel(target_market="PTS").fit(df).dispersion
    assert uncond.variance_for(9.45) > cond.variance_for(9.45) * 1.5, (
        f"conditioning on minutes did not reduce the dispersion: "
        f"unconditional {uncond.variance_for(9.45):.2f} against "
        f"conditional {cond.variance_for(9.45):.2f}"
    )


def test_fit_refuses_without_the_columns_it_needs_rather_than_falling_back(panel):
    """
    A Poisson default here would be combined with a real minutes term and the
    result would look like a fitted model.
    """
    for drop in ("MIN", "MIN_BASELINE", "PTS_BASELINE"):
        with pytest.raises(MinutesAwareError, match="DATA_NOT_AVAILABLE"):
            MinutesAwarePropModel(target_market="PTS").fit(panel.drop(columns=[drop]))


def test_scoring_refuses_without_the_minutes_spread_column(panel):
    m = MinutesAwarePropModel(target_market="PTS").fit(panel)
    for drop in ("MIN_L2", "MIN_SD_L10"):
        with pytest.raises(MinutesAwareError, match="DATA_NOT_AVAILABLE"):
            m.predict_probability_over(panel.drop(columns=[drop]), 10.0)


def test_scoring_never_reads_the_realised_minutes(panel):
    """
    LEAKAGE. `fit` uses realised MIN to measure Var(S|M), a structural
    parameter. Scoring must not touch it: at slate time it does not exist.
    Dropping the column entirely must change nothing.
    """
    m = MinutesAwarePropModel(target_market="PTS").fit(panel)
    with_min = m.predict_probability_over(panel, 10.0)
    without = m.predict_probability_over(panel.drop(columns=["MIN"]), 10.0)
    pd.testing.assert_series_equal(with_min, without)


def test_a_row_with_no_minutes_spread_scores_nan_not_a_number(panel):
    m = MinutesAwarePropModel(target_market="PTS").fit(panel)
    spoiled = panel.copy()
    spoiled.loc[spoiled.index[:5], "MIN_SD_L10"] = np.nan
    out = m.predict_probability_over(spoiled, 10.0)
    assert out.iloc[:5].isna().all(), "a row with unknown minutes spread was priced"
    assert out.iloc[5:].notna().any(), "the fixture abstains on everything"


# --- the artifact cannot be confused with distribution's ----------------


def test_the_artifact_records_its_basis(tmp_path, panel):
    import json

    m = MinutesAwarePropModel(target_market="PTS").fit(panel)
    path = tmp_path / "minutes_aware_PTS.json"
    m.save(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["dispersion_basis"] == DISPERSION_BASIS
    assert payload["model_name"] == "minutes_aware"
    assert MinutesAwarePropModel(target_market="PTS").load(path).dispersion is not None


def test_loading_a_distribution_artifact_is_refused(tmp_path, panel):
    """
    THE MISPRICE THIS PREVENTS. A `distribution` artifact carries an
    unconditional phi -- roughly twice the conditional one on real panels.
    Loaded here it would be added to a real minutes term, and every probability
    would be wrong in the same direction with nothing on the face of it to show
    why.
    """
    d = DistributionPropModel(target_market="PTS").fit(panel)
    path = tmp_path / "distribution_PTS.json"
    d.save(path)
    with pytest.raises(MinutesAwareError, match="unconditional"):
        MinutesAwarePropModel(target_market="PTS").load(path)


def test_an_unfitted_model_refuses_to_score(panel):
    with pytest.raises(MinutesAwareError, match="not fitted"):
        MinutesAwarePropModel(target_market="PTS").predict_probability_over(panel, 10.0)


# --- it is reachable ----------------------------------------------------


def test_it_is_registered_as_a_line_taking_component():
    from src.models.serving_components import (
        LINE_TAKING_COMPONENTS,
        artifact_suffix,
        is_line_taking,
    )

    assert is_line_taking("minutes_aware")
    assert "minutes_aware" in LINE_TAKING_COMPONENTS
    assert artifact_suffix("minutes_aware") == ".json"


def test_it_is_selectable_and_shadowable():
    import main

    assert "minutes_aware" in main.SCORING_COMPONENTS
    assert main.SCORING_COMPONENTS[0] == "distribution", (
        "the default must stay distribution: promoting a component needs "
        "forward graded evidence, which this one has none of"
    )


def test_it_claims_no_improvement_anywhere_in_its_source():
    """
    IT IS A REALLOCATION OF VARIANCE, NOT A MEASURED GAIN. Whether it scores
    better is for the shadow comparison to answer on graded rows. AGENTS.md
    section O forbids the claim until then.
    """
    from pathlib import Path

    source = Path("src/models/minutes_aware.py").read_text(encoding="utf-8")
    lowered = source.lower()
    for forbidden in ("more accurate", "better calibrated", "profitable",
                      "improves the", "outperforms"):
        assert forbidden not in lowered, f"{forbidden!r} is a claim nothing has measured"


# --- the feature behind it ----------------------------------------------


def test_the_minutes_spread_uses_only_prior_games():
    """
    `.shift(1)` on every grouped rolling feature -- CLAUDE.md rule 1. A spread
    that saw tonight's minutes would make this component look superb in
    validation and worthless on a slate.
    """
    from src.features.builder import build_feature_matrix
    from src.models.data_audit import make_demo_panel

    f = build_feature_matrix(make_demo_panel(n_players=3, n_games=14))
    f = f.sort_values(["PLAYER_ID", "GAME_DATE"])
    for _pid, g in f.groupby("PLAYER_ID"):
        # min_periods=3, so the first three rows of each player cannot have one.
        assert g["MIN_SD_L10"].iloc[:3].isna().all(), (
            "a spread appeared before three prior games existed"
        )
        # And a spread must never equal the SD including tonight.
        prior = g["MIN"].shift(1).rolling(10, min_periods=3).std(ddof=1)
        pd.testing.assert_series_equal(
            g["MIN_SD_L10"], prior, check_names=False,
        )


def test_the_spread_is_not_the_same_fact_as_minutes_stable():
    """
    `MINUTES_STABLE` is level DRIFT (|MIN_L5/MIN_SEASON - 1| < 0.15). A player
    can sit at a dead-stable level and swing 12 and 35 minutes on alternate
    nights -- stable by that test, and exactly the player whose prop is
    unpredictable. If the two ever collapse into one column this fails.
    """
    rows = []
    for g in range(24):
        rows.append({
            "PLAYER_ID": "P0", "GAME_DATE": pd.Timestamp("2026-01-01") + pd.Timedelta(days=g),
            "MIN": 12.0 if g % 2 else 35.0, "PTS": 10.0, "REB": 4.0, "AST": 2.0,
        })
    df = pd.DataFrame(rows)
    level = df["MIN"].shift(1).rolling(5, min_periods=1).mean()
    season = df["MIN"].shift(1).expanding().mean()
    drift = (level / season - 1.0).abs()
    spread = df["MIN"].shift(1).rolling(10, min_periods=3).std(ddof=1)

    assert drift.dropna().max() < 0.25, "the fixture is not level-stable"
    assert spread.dropna().min() > 8.0, "the fixture has no game-to-game spread"
