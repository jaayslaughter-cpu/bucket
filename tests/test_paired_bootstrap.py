"""
tests/test_paired_bootstrap.py — is a metric difference real, or is it noise?

WHAT THIS GUARDS. `compare.py` chose a market's winner on POINT ESTIMATES
alone: Brier 0.2412 beating 0.2418 was promoted as a win with nothing asking
whether the gap would survive a different sample. The properties asserted here
are the ones whose absence is silent — a model credited with an improvement it
did not make looks exactly like a model that improved.

The two that matter most, and both are easy to get wrong:

  * PAIRING. One draw applied to both models, so row difficulty cancels. An
    unpaired version inflates the interval with variance both models agree
    about and hides real improvements.
  * CLUSTERING BY GAME. Eight props from one game share a game script. Treating
    them as eight independent observations understates the interval and calls
    noise significant — the exact error a significance test exists to prevent.

RESEARCH_ONLY.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.models.paired_bootstrap import (
    CLUSTER_GAME,
    CLUSTER_IID,
    METRICS,
    PairedBootstrapError,
    decisive_winner,
    paired_bootstrap,
)

RESAMPLES = 400  # enough to be stable, small enough to keep the suite quick


def _coinflips(n: int, seed: int = 7) -> np.ndarray:
    return (np.random.default_rng(seed).random(n) > 0.5).astype(float)


# --- the contract that stops a model shipping for free -------------------


def test_two_identical_models_are_never_an_improvement():
    """
    THE CORE GUARANTEE. The same predictions twice is a difference of exactly
    zero, so the interval must contain zero and `improved` must be False. If
    this ever passes something through, every comparison downstream is
    meaningless.
    """
    y = _coinflips(400)
    p = np.full(400, 0.5)
    out = paired_bootstrap(y, p, p, "brier", n_resamples=RESAMPLES)
    assert out["difference"] == 0.0
    assert out["ci_low"] <= 0.0 <= out["ci_high"]
    assert out["excludes_zero"] is False
    assert out["improved"] is False


def test_a_hair_of_difference_on_noise_is_not_called_an_improvement():
    """
    THE DEFECT THIS MODULE REPLACES. Two models whose Brier scores differ in
    the fourth decimal on random data: the point estimate has a winner, and
    `compare.py` promoted it. The interval must not.
    """
    rng = np.random.default_rng(11)
    y = (rng.random(600) > 0.5).astype(float)
    baseline = np.clip(rng.normal(0.5, 0.05, 600), 0.01, 0.99)
    candidate = np.clip(baseline + rng.normal(0.0, 0.002, 600), 0.01, 0.99)

    out = paired_bootstrap(y, baseline, candidate, "brier", n_resamples=RESAMPLES)
    # There IS a point-estimate winner...
    assert out["difference"] != 0.0
    # ...and it is not evidence.
    assert out["improved"] is False, (
        "a fourth-decimal gap on noise was called an improvement, which is "
        "exactly the behaviour this module exists to stop"
    )


def test_a_genuinely_better_model_is_detected():
    """
    The test must be able to say yes, or it is only a refusal machine. A model
    that actually knows the labels beats one that guesses, decisively.
    """
    y = _coinflips(500, seed=3)
    guessing = np.full(500, 0.5)
    informed = np.where(y > 0.5, 0.85, 0.15)

    out = paired_bootstrap(y, guessing, informed, "brier", n_resamples=RESAMPLES)
    assert out["difference"] < 0, "lower Brier is better"
    assert out["excludes_zero"] is True
    assert out["improved"] is True


# --- direction is carried with the metric --------------------------------


def test_direction_is_per_metric_so_a_worse_auc_cannot_read_as_a_win():
    """
    Lower is better for Brier, log loss and MAE; HIGHER for AUC. Without the
    direction attached to the metric a caller reads a worse AUC as an
    improvement, which is the one mistake this is supposed to make impossible.
    """
    assert METRICS["brier"][1] is True
    assert METRICS["log_loss"][1] is True
    assert METRICS["mae"][1] is True
    assert METRICS["auc"][1] is False, "higher AUC is better"

    y = _coinflips(500, seed=5)
    informed = np.where(y > 0.5, 0.85, 0.15)
    guessing = np.full(500, 0.5)

    # The INFORMED model is the candidate; its AUC is higher, so on a
    # higher-is-better metric that is an improvement and the sign is positive.
    out = paired_bootstrap(y, guessing, informed, "auc", n_resamples=RESAMPLES)
    assert out["difference"] > 0
    assert out["improved"] is True

    # Reversed, the same magnitude of difference must NOT be an improvement.
    back = paired_bootstrap(y, informed, guessing, "auc", n_resamples=RESAMPLES)
    assert back["difference"] < 0
    assert back["improved"] is False


# --- clustering, which is where this departs from what it was read from ---


def test_clustering_by_game_widens_the_interval_it_should_widen():
    """
    THE REASON THIS IS NOT KEVOCADO'S IMPLEMENTATION. Eight props from one game
    move together. Resampling rows independently treats them as eight
    independent observations and understates the interval; resampling whole
    games does not.

    Built so the correlation is real: every row in a game shares that game's
    outcome, so the clustered interval must be WIDER than the iid one on the
    same data. If they ever match, the clustering is not happening.
    """
    rng = np.random.default_rng(23)
    n_games, per_game = 60, 8
    games = np.repeat(np.arange(n_games), per_game)
    # One outcome per GAME, shared by its rows: maximal within-game correlation.
    game_outcome = (rng.random(n_games) > 0.5).astype(float)
    y = np.repeat(game_outcome, per_game)
    baseline = np.full(y.size, 0.5)
    candidate = np.clip(np.repeat(game_outcome * 0.3 + 0.35, per_game), 0.01, 0.99)

    iid = paired_bootstrap(
        y, baseline, candidate, "brier", n_resamples=RESAMPLES, seed=1
    )
    clustered = paired_bootstrap(
        y, baseline, candidate, "brier", games=games, n_resamples=RESAMPLES, seed=1
    )

    assert iid["cluster"] == CLUSTER_IID
    assert clustered["cluster"] == CLUSTER_GAME
    assert clustered["n_games"] == n_games
    assert iid["n_games"] is None

    iid_width = iid["ci_high"] - iid["ci_low"]
    clustered_width = clustered["ci_high"] - clustered["ci_low"]
    assert clustered_width > iid_width, (
        f"clustered interval ({clustered_width:.5f}) is not wider than iid "
        f"({iid_width:.5f}); within-game correlation is being ignored"
    )


def test_the_result_says_which_resampling_unit_was_used():
    """
    A caller who forgot the game column must not be able to believe they got
    the clustered answer. The weaker result is still returned — a wide-ish
    interval beats none — but it is labelled.
    """
    y = _coinflips(100)
    p = np.full(100, 0.5)
    assert paired_bootstrap(
        y, p, p, "brier", n_resamples=50
    )["cluster"] == CLUSTER_IID


def test_games_must_align_with_the_rows():
    y = _coinflips(50)
    p = np.full(50, 0.5)
    with pytest.raises(PairedBootstrapError, match="games must align"):
        paired_bootstrap(y, p, p, "brier", games=np.arange(7), n_resamples=10)


# --- undefined resamples are dropped, not invented ------------------------


def test_an_undefined_resample_is_dropped_rather_than_scored():
    """
    An AUC draw containing one class only has no value. Scoring it 0.5 would
    drag the interval toward "no difference" for a reason that has nothing to
    do with either model, so those draws are discarded and the count that
    actually counted is reported.
    """
    # Nine zeros and one 1: many draws will contain no positive case at all.
    y = np.array([0.0] * 9 + [1.0])
    baseline = np.full(10, 0.4)
    candidate = np.linspace(0.1, 0.9, 10)

    out = paired_bootstrap(y, baseline, candidate, "auc", n_resamples=300, seed=2)
    assert out["n_resamples_used"] < out["n_resamples"], (
        "no draw was degenerate, so this test is not covering the drop path"
    )
    assert out["n_resamples_used"] > 0


def test_a_fully_degenerate_comparison_raises_rather_than_returning_a_verdict():
    """One class everywhere: AUC is never defined, so there is no interval to
    report. Raising beats returning a confident-looking dictionary."""
    y = np.zeros(20)
    p = np.full(20, 0.5)
    with pytest.raises(PairedBootstrapError, match="degenerate"):
        paired_bootstrap(y, p, p, "auc", n_resamples=50)


# --- the inputs it refuses ------------------------------------------------


@pytest.mark.parametrize("kwargs,match", [
    ({"metric": "accuracy"}, "unknown metric"),
    ({"metric": "brier", "alpha": 0.0}, "alpha must be"),
    ({"metric": "brier", "alpha": 1.0}, "alpha must be"),
    ({"metric": "brier", "n_resamples": 0}, "n_resamples must be positive"),
])
def test_bad_inputs_are_refused_by_name(kwargs, match):
    y = _coinflips(20)
    p = np.full(20, 0.5)
    metric = kwargs.pop("metric")
    with pytest.raises(PairedBootstrapError, match=match):
        paired_bootstrap(y, p, p, metric, **kwargs)


def test_misaligned_predictions_are_refused():
    with pytest.raises(PairedBootstrapError, match="must align"):
        paired_bootstrap(
            _coinflips(20), np.full(20, 0.5), np.full(19, 0.5), "brier",
            n_resamples=10,
        )


def test_an_empty_comparison_is_refused():
    with pytest.raises(PairedBootstrapError, match="nothing to bootstrap"):
        paired_bootstrap([], [], [], "brier", n_resamples=10)


# --- reproducibility ------------------------------------------------------


def test_the_same_seed_gives_the_same_interval():
    """
    Fixed by default so a comparison is reproducible. A per-run seed would let
    somebody re-roll until the interval excluded zero.
    """
    y = _coinflips(300, seed=9)
    baseline = np.full(300, 0.5)
    candidate = np.where(y > 0.5, 0.6, 0.4)
    a = paired_bootstrap(y, baseline, candidate, "brier", n_resamples=RESAMPLES)
    b = paired_bootstrap(y, baseline, candidate, "brier", n_resamples=RESAMPLES)
    assert (a["ci_low"], a["ci_high"]) == (b["ci_low"], b["ci_high"])


# --- decisive_winner, the shape compare.py consumes ----------------------


def test_decisive_winner_reports_a_real_lead_as_decisive():
    y = _coinflips(500, seed=13)
    leader = np.where(y > 0.5, 0.85, 0.15)
    runner_up = np.full(500, 0.5)
    out = decisive_winner(y, leader, runner_up, n_resamples=RESAMPLES)
    assert out["decisive"] is True
    assert "supported" in out["note"]


def test_decisive_winner_calls_a_noise_lead_a_tie_and_says_so():
    """
    The leader is still whatever `compare.py` ranked first — this does not
    re-choose it. What it adds is the sentence that stops the ranking being
    read as a result.
    """
    rng = np.random.default_rng(17)
    y = (rng.random(600) > 0.5).astype(float)
    runner_up = np.clip(rng.normal(0.5, 0.05, 600), 0.01, 0.99)
    leader = np.clip(runner_up + rng.normal(0.0, 0.002, 600), 0.01, 0.99)

    out = decisive_winner(y, leader, runner_up, n_resamples=RESAMPLES)
    assert out["decisive"] is False
    assert "tied on this evidence" in out["note"]
    assert "not a result" in out["note"]


def test_decisive_winner_orients_the_sign_so_the_leader_is_the_candidate():
    """
    A lower-is-better metric must produce a NEGATIVE difference when the leader
    is genuinely better. Getting this backwards would report every real
    improvement as a regression.
    """
    y = _coinflips(400, seed=29)
    leader = np.where(y > 0.5, 0.8, 0.2)
    runner_up = np.full(400, 0.5)
    out = decisive_winner(y, leader, runner_up, n_resamples=RESAMPLES)
    assert out["difference"] < 0
    assert out["candidate"] < out["baseline"]


# --- the compare.py integration ------------------------------------------
#
# `_decisiveness` is what turns the bootstrap into a decision. It reads the
# per-row predictions `compare_models_on_panel` already collects, pairs the top
# two models on the props BOTH scored, and clusters by event_id.


def _detail_row(model, event_id, player_id, line, probability, actual, market="PTS"):
    return {
        "target_market": market,
        "model_name": model,
        "event_id": event_id,
        "player_id": player_id,
        "prop_line": line,
        "probability_over_calibrated": probability,
        "probability_over_raw": probability,
        "actual_stat_value": actual,
    }


def _two_model_rows(n_games, per_game, leader_fn, runner_fn, seed=31):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_games):
        outcome = float(rng.random() > 0.5)
        for p in range(per_game):
            line, actual = 20.0, 20.0 + (5.0 if outcome else -5.0)
            rows.append(_detail_row("alpha", g, p, line, leader_fn(outcome, rng), actual))
            rows.append(_detail_row("beta", g, p, line, runner_fn(outcome, rng), actual))
    return rows


def _ranked(leader="alpha", runner_up="beta"):
    return [
        (leader, {"brier_score": 0.19, "log_loss": 0.55, "calibration_error": 0.02}),
        (runner_up, {"brier_score": 0.24, "log_loss": 0.61, "calibration_error": 0.03}),
    ]


def test_decisiveness_reports_a_real_lead():
    """An informed leader against a coin-flipping runner-up is decisive."""
    from src.models.compare import _decisiveness

    rows = _two_model_rows(
        40, 6,
        leader_fn=lambda o, r: 0.85 if o else 0.15,
        runner_fn=lambda o, r: 0.5,
    )
    out = _decisiveness(_ranked(), rows, "PTS")
    assert out["decisive"] is True
    assert out["runner_up"] == "beta"
    assert out["lead_brier_difference"] < 0
    assert out["lead_n_games"] == 40


def test_decisiveness_calls_a_coin_flip_lead_a_tie():
    """
    THE DEFECT THIS CLOSES. Both models guess; the ranking still has a leader
    because `brier_score` differs in the tuple. The lead must not be decisive.
    """
    from src.models.compare import _decisiveness

    rows = _two_model_rows(
        40, 6,
        leader_fn=lambda o, r: float(np.clip(r.normal(0.5, 0.05), 0.01, 0.99)),
        runner_fn=lambda o, r: float(np.clip(r.normal(0.5, 0.05), 0.01, 0.99)),
    )
    out = _decisiveness(_ranked(), rows, "PTS")
    assert out["decisive"] is False
    assert "tied on this evidence" in out["decisiveness_note"]


def test_decisiveness_clusters_by_event_id():
    """
    Props from one game share a game script. The report must say it resampled
    games, and count them — not rows — or the interval is the understated one.
    """
    from src.models.compare import _decisiveness

    rows = _two_model_rows(
        30, 8,
        leader_fn=lambda o, r: 0.8 if o else 0.2,
        runner_fn=lambda o, r: 0.5,
    )
    out = _decisiveness(_ranked(), rows, "PTS")
    assert out["lead_n_games"] == 30
    assert out["lead_n_rows"] == 30 * 8


def test_an_untestable_lead_is_none_not_false():
    """
    "We could not test this" and "we tested it and it is noise" are different
    states. Collapsing the first into False would let a missing measurement
    read as a measured tie — the defect `_winner_rank_key` already documents
    for ECE.
    """
    from src.models.compare import _decisiveness

    single = _decisiveness([_ranked()[0]], [], "PTS")
    assert single["decisive"] is None
    assert "no lead to test" in single["decisiveness_note"]

    thin = _decisiveness(
        _ranked(),
        _two_model_rows(2, 2,
                        leader_fn=lambda o, r: 0.8,
                        runner_fn=lambda o, r: 0.5),
        "PTS",
    )
    assert thin["decisive"] is None
    assert "below the" in thin["decisiveness_note"]


def test_decisiveness_ignores_other_markets_and_unscored_rows():
    """
    One market at a time, and a row either model left unscored is not evidence
    about either. Mixing markets would compare models on different problems.
    """
    from src.models.compare import _decisiveness

    rows = _two_model_rows(
        40, 6,
        leader_fn=lambda o, r: 0.85 if o else 0.15,
        runner_fn=lambda o, r: 0.5,
    )
    rows += [_detail_row("alpha", 999, 1, 8.5, 0.9, 12.0, market="REB")]
    rows += [_detail_row("alpha", 998, 1, 20.0, None, 25.0)]
    out = _decisiveness(_ranked(), rows, "PTS")
    assert out["lead_n_games"] == 40, "a REB row or an unscored row leaked in"


# --- pairing, which the first version of this file never tested -----------


def test_pairing_narrows_the_interval_against_an_unpaired_draw():
    """
    THE PROPERTY THE MODULE IS NAMED FOR, and the first version of this file
    did not test it: replacing the shared draw with two independent draws left
    every assertion passing. A mutation that survives means the claim is
    undefended.

    Built so pairing demonstrably matters: row difficulty varies hugely and
    BOTH models track it, so the paired difference is near-constant while an
    unpaired difference inherits all of that between-row variance. The paired
    interval must therefore be strictly narrower.
    """
    rng = np.random.default_rng(41)
    n = 600
    y = (rng.random(n) > 0.5).astype(float)
    # Difficulty both models see: confident on easy rows, unsure on hard ones.
    confidence = rng.uniform(0.02, 0.48, n)
    baseline = np.where(y > 0.5, 0.5 + confidence, 0.5 - confidence)
    # The candidate is the baseline nudged the same way on every row.
    candidate = np.clip(
        np.where(y > 0.5, baseline + 0.02, baseline - 0.02), 0.01, 0.99
    )

    paired = paired_bootstrap(
        y, baseline, candidate, "brier", n_resamples=600, seed=5
    )

    # The same computation with the pairing deliberately broken.
    rng2 = np.random.default_rng(5)
    score, _ = METRICS["brier"]
    unpaired_diffs = []
    for _ in range(600):
        a = rng2.integers(0, n, n)
        b = rng2.integers(0, n, n)
        unpaired_diffs.append(
            score(y[a], candidate[a]) - score(y[b], baseline[b])
        )
    lo, hi = np.percentile(unpaired_diffs, [2.5, 97.5])

    paired_width = paired["ci_high"] - paired["ci_low"]
    unpaired_width = float(hi - lo)
    assert paired_width < unpaired_width, (
        f"paired interval ({paired_width:.5f}) is not narrower than unpaired "
        f"({unpaired_width:.5f}); the draw is not actually shared"
    )
    assert paired["improved"] is True

    # DELIBERATELY NOT ASSERTED: that the unpaired interval fails to detect
    # this improvement. It happened to detect it here (hi = -0.0018), and
    # demanding otherwise would pin an incidental property of one constructed
    # dataset rather than the one that matters. Pairing narrows the interval;
    # whether a given effect survives the wider one depends on its size.
