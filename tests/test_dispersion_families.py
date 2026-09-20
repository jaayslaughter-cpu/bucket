"""
ZIP and normal dispersion families — fitted, never hardcoded.

The wave packs carried these distributions with fixed constants
(nb_var_scale=1.35, zip_zero_infl=0.12). That 1.35 is the magic number
this project removed once already. These tests pin the replacement: the
same families, with every parameter estimated from data and the family
chosen by held-out likelihood.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.residuals import (
    MAX_ZERO_INFLATION,
    CountDispersion,
    fit_count_dispersion,
    over_under_push_from_dispersion,
)

N = 3000


# --------------------------------------------------------------------------
# The family must follow the data
# --------------------------------------------------------------------------

def test_plain_counts_still_choose_poisson():
    rng = np.random.default_rng(11)
    y = rng.poisson(8.0, N).astype(float)
    fitted = fit_count_dispersion(y, np.full(N, 8.0), market="PTS")
    assert fitted.family == "poisson"
    assert fitted.phi == pytest.approx(1.0)


def test_overdispersed_counts_still_choose_negbin():
    rng = np.random.default_rng(11)
    y = rng.negative_binomial(4, 4 / (4 + 8), N).astype(float)
    fitted = fit_count_dispersion(y, np.full(N, 8.0), market="PTS")
    assert fitted.family == "negbin"
    assert fitted.phi > 1.5


def test_zero_inflated_counts_choose_zip_and_recover_the_rate():
    """A plain Poisson cannot produce 30% zeros at a mean of 8 at any rate."""
    rng = np.random.default_rng(11)
    true_pi = 0.30
    base = rng.poisson(8.0 / (1 - true_pi), N).astype(float)
    y = np.where(rng.random(N) < true_pi, 0.0, base)

    fitted = fit_count_dispersion(y, np.full(N, y.mean()), market="REB")
    assert fitted.family == "zip"
    # Recovered, not assumed — and nowhere near the 0.12 the pack hardcoded.
    assert fitted.zero_inflation == pytest.approx(true_pi, abs=0.05)


def test_tight_symmetric_counts_choose_normal_and_recover_the_scale():
    rng = np.random.default_rng(11)
    mean, true_sd = 30.0, 3.0
    y = np.clip(rng.normal(mean, true_sd, N).round(), 0, None)

    fitted = fit_count_dispersion(y, np.full(N, mean), market="PTS")
    assert fitted.family == "normal"
    # sigma_scale is a multiple of sqrt(mean).
    assert fitted.sigma_scale == pytest.approx(true_sd / np.sqrt(mean), abs=0.08)


def test_no_hardcoded_constant_survives_in_the_source():
    """The pack's constants must not reappear as defaults."""
    source = (Path(__file__).parent.parent / "src/models/residuals.py").read_text()
    assert "1.35" not in source, "the removed variance multiplier is back"
    assert "0.12" not in source, "a hardcoded zero-inflation rate is present"


# --------------------------------------------------------------------------
# Every family must price with its OWN distribution
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("family", "expected"),
    [
        ("poisson", "Poisson"),
        ("negbin", "NegativeBinomial"),
        ("zip", "ZeroInflatedPoisson"),
        ("normal", "Normal"),
    ],
)
def test_each_family_dispatches_to_its_own_push_math(family, expected):
    """An else-branch sweeping every family into Poisson would record the
    family in metadata and ignore it in the arithmetic."""
    dispersion = CountDispersion(
        family=family, phi=1.6, n_train_rows=999, selection_scores={},
        zero_inflation=0.3, sigma_scale=1.2,
    )
    result = over_under_push_from_dispersion(8.0, 8.0, dispersion)

    assert result["status"] == "OK"
    assert result["distribution"] == expected
    total = (
        result["probability_over"]
        + result["probability_under"]
        + result["probability_push"]
    )
    assert total == pytest.approx(1.0, abs=1e-5)
    assert result["probability_push"] > 0, "a whole-number line must carry push mass"


@pytest.mark.parametrize("family", ["poisson", "negbin", "zip", "normal"])
def test_half_point_lines_never_push_in_any_family(family):
    dispersion = CountDispersion(
        family=family, phi=1.6, n_train_rows=999, selection_scores={},
        zero_inflation=0.3, sigma_scale=1.2,
    )
    result = over_under_push_from_dispersion(8.0, 8.5, dispersion)
    assert result["probability_push"] == 0.0


def test_zip_puts_more_mass_on_zero_than_poisson():
    """The whole point of the family."""
    poisson_d = CountDispersion(
        family="poisson", phi=1.0, n_train_rows=999, selection_scores={},
    )
    zip_d = CountDispersion(
        family="zip", phi=1.0, n_train_rows=999, selection_scores={},
        zero_inflation=0.35,
    )
    # P(under 0.5) is P(X == 0).
    p_zero_poisson = over_under_push_from_dispersion(4.0, 0.5, poisson_d)["probability_under"]
    p_zero_zip = over_under_push_from_dispersion(4.0, 0.5, zip_d)["probability_under"]
    assert p_zero_zip > p_zero_poisson + 0.2


def test_variance_grows_with_zero_inflation():
    base = CountDispersion(family="zip", phi=1.0, n_train_rows=9, selection_scores={},
                           zero_inflation=0.0)
    inflated = CountDispersion(family="zip", phi=1.0, n_train_rows=9, selection_scores={},
                               zero_inflation=0.4)
    assert inflated.variance_for(10.0) > base.variance_for(10.0)


# --------------------------------------------------------------------------
# Round trip and backward compatibility
# --------------------------------------------------------------------------

def test_new_parameters_survive_a_round_trip():
    original = CountDispersion(
        family="zip", phi=1.0, n_train_rows=500,
        selection_scores={"zip_mean_nll": -2.1},
        zero_inflation=0.2718, sigma_scale=1.0,
    )
    restored = CountDispersion.from_dict(original.to_dict())
    assert restored == original
    assert restored.zero_inflation == original.zero_inflation


def test_artifacts_written_before_these_families_still_load():
    """An old row must reload as the two-family model it actually was."""
    legacy = {
        "family": "negbin", "phi": 1.87, "n_train_rows": 500,
        "selection_scores": {}, "fallback_reason": None,
    }
    restored = CountDispersion.from_dict(legacy)
    assert restored.family == "negbin"
    assert restored.zero_inflation == 0.0
    assert restored.sigma_scale == 1.0


def test_zero_inflation_is_bounded():
    assert MAX_ZERO_INFLATION < 1.0, (
        "a rate at 1.0 explains the mean entirely with zeros"
    )


# --------------------------------------------------------------------------
# Combo (PRA) variance — routed through the same push math
# --------------------------------------------------------------------------

def test_combo_uses_the_single_push_implementation():
    """A second copy of the over/under/push rule would drift from the first."""
    source = (Path(__file__).parent.parent / "src/models/combo_variance.py").read_text()
    assert "over_under_push_from_dispersion" in source
    assert "line_probs" not in source, "the removed duplicate module is back"


def test_combo_line_probabilities_respect_push_rules():
    from src.models.combo_variance import ComboProjection, pra_line_probabilities

    combo = ComboProjection(mean=25.0, variance=27.5, std=5.244, status="OK")

    whole = pra_line_probabilities(combo, 28.0)
    assert whole["distribution"] == "Normal"
    assert whole["probability_push"] > 0, "a whole PRA line must carry push mass"
    assert sum(
        whole[k] for k in ("probability_over", "probability_under", "probability_push")
    ) == pytest.approx(1.0, abs=1e-5)

    half = pra_line_probabilities(combo, 28.5)
    assert half["probability_push"] == 0.0


def test_combo_abstains_when_the_projection_is_unavailable():
    from src.models.combo_variance import ComboProjection, pra_line_probabilities

    combo = ComboProjection(status="DATA_NOT_AVAILABLE", reason="missing component")
    result = pra_line_probabilities(combo, 28.5)
    assert result["status"] == "DATA_NOT_AVAILABLE"
    assert result["probability_over"] is None


def test_variance_inflation_is_measured_not_assumed():
    """The correlation factor is a quantity, so measure it."""
    import pandas as pd

    from src.models.combo_variance import (
        DEFAULT_COMBO_VAR_FUDGE,
        fit_combo_variance_inflation,
    )

    rng = np.random.default_rng(5)
    # A shared driver makes the components positively correlated, so the
    # variance of the sum must exceed the sum of the variances.
    driver = rng.normal(0, 1, 800)
    frame = pd.DataFrame({
        "PTS": 20 + 5 * driver + rng.normal(0, 2, 800),
        "REB": 6 + 2 * driver + rng.normal(0, 1, 800),
        "AST": 5 + 2 * driver + rng.normal(0, 1, 800),
    })
    measured = fit_combo_variance_inflation(frame)
    assert measured["inflation"] > 1.0
    assert measured["n_rows"] == 800
    assert measured["default_used"] == DEFAULT_COMBO_VAR_FUDGE

    # Genuinely independent components sit near 1.0.
    independent = pd.DataFrame({
        "PTS": rng.normal(20, 5, 800),
        "REB": rng.normal(6, 2, 800),
        "AST": rng.normal(5, 2, 800),
    })
    assert fit_combo_variance_inflation(independent)["inflation"] == pytest.approx(1.0, abs=0.15)


def test_variance_inflation_abstains_on_too_few_rows():
    import pandas as pd

    from src.models.combo_variance import fit_combo_variance_inflation

    tiny = pd.DataFrame({"PTS": [20.0, 21.0], "REB": [5.0, 6.0], "AST": [4.0, 5.0]})
    assert np.isnan(fit_combo_variance_inflation(tiny)["inflation"])
