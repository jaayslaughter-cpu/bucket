"""
`settlement.loss_analysis` — the residual report, and what it refuses to say.

WHY THESE TESTS AND NOT OTHERS. The arithmetic is easy; the judgement is not.
What can be got wrong silently here is reporting a bias off four rows,
counting a scratch as a loss, treating "prior games not measured" as "nobody
had played a game", and subtracting the residual the wrong way round so a
bias reads with the wrong sign. Each of those has a test.

The CLI's database read is not exercised here -- there is no database in this
suite, which is also why the arithmetic lives in a pure module.

RESEARCH_ONLY.
"""

from __future__ import annotations

import pytest

from src.settlement.loss_analysis import (
    DEFAULT_PRIOR_GAME_BINS,
    MIN_ROWS_FOR_A_BIAS,
    analyse_losses,
    bin_for_prior_games,
    residual_of,
)


def _row(**kw):
    base = {
        "outcome_status": "WIN",
        "market": "PTS",
        "game_date": "2026-11-06",
        "player_name": "HARNESS PLAYER",
        "nba_player_id": "espn:1",
        "predicted_side": "OVER",
        "predicted_line": 20.5,
        "model_projection": 22.0,
        "prob_over": 0.6,
        "actual_result": 24.0,
        "did_not_play": False,
    }
    base.update(kw)
    return base


# --- the sign, which is the one thing a reader will act on -----------------


def test_a_positive_residual_means_the_player_beat_the_model():
    assert residual_of(_row(model_projection=20.0, actual_result=26.0)) == 6.0
    assert residual_of(_row(model_projection=26.0, actual_result=20.0)) == -6.0


def test_a_missing_side_of_the_subtraction_is_none_not_zero():
    """
    A VOID row has no actual; a row the model never scored has no projection.
    Treating either as 0.0 would drag the mean bias toward a number nobody
    measured.
    """
    assert residual_of(_row(actual_result=None)) is None
    assert residual_of(_row(model_projection=None)) is None
    assert residual_of(_row(actual_result="n/a")) is None
    assert residual_of(_row(model_projection=float("nan"))) is None


# --- the floor on reporting a bias -----------------------------------------


def test_a_bias_is_withheld_below_the_row_floor():
    """
    A bin with four rows has a mean; it does not have a finding. Printing it
    anyway is how a feedback loop starts chasing noise.
    """
    rows = [_row(model_projection=20.0, actual_result=25.0) for _ in range(4)]
    report = analyse_losses(rows)
    overall = report["residuals_overall"]

    assert overall["n"] == 4
    assert overall["mean_bias"] is None, "a 4-row bias was reported as a figure"
    assert overall["insufficient"] is True
    assert overall["needs"] == MIN_ROWS_FOR_A_BIAS
    assert "below the" in report["findings"][0]


def test_a_bias_is_reported_once_there_are_enough_rows():
    rows = [
        _row(model_projection=20.0, actual_result=25.0)
        for _ in range(MIN_ROWS_FOR_A_BIAS)
    ]
    overall = analyse_losses(rows)["residuals_overall"]

    assert overall["n"] == MIN_ROWS_FOR_A_BIAS
    assert overall["mean_bias"] == pytest.approx(5.0)
    assert overall["mae"] == pytest.approx(5.0)
    assert overall["rmse"] == pytest.approx(5.0)
    assert "insufficient" not in overall


# --- what is not a loss ----------------------------------------------------


def test_a_scratch_is_not_counted_as_a_loss():
    """
    `PropResult` records a DNP as VOID for this reason, but a row can carry
    the flag and a graded status if it settled before the scratch was known.
    Counting it would understate the model's real strike rate.
    """
    rows = [_row(outcome_status="LOSS", did_not_play=True) for _ in range(5)]
    rows += [_row(outcome_status="WIN") for _ in range(5)]
    report = analyse_losses(rows)

    assert report["record"]["LOSS"] == 0
    assert report["record"]["WIN"] == 5
    assert report["excluded"]["did not play"] == 5


@pytest.mark.parametrize("status", ["PENDING", "VOID", "", None, "whatever"])
def test_an_ungraded_row_is_excluded_by_name(status):
    report = analyse_losses([_row(outcome_status=status)])
    assert report["n_graded"] == 0
    assert sum(report["excluded"].values()) == 1


def test_the_strike_rate_ignores_pushes():
    rows = [_row(outcome_status="WIN") for _ in range(3)]
    rows += [_row(outcome_status="LOSS")]
    rows += [_row(outcome_status="PUSH") for _ in range(6)]
    rec = analyse_losses(rows)["record"]

    assert rec["decided"] == 4
    assert rec["strike_rate"] == pytest.approx(0.75)


# --- the early-season question, which is why this module exists ------------


@pytest.mark.parametrize(
    "count,label",
    [(0, "0-2"), (2, "0-2"), (3, "3-5"), (9, "6-9"), (10, "10-19"), (82, "20+")],
)
def test_prior_game_bins_cover_the_early_season_finely(count, label):
    assert bin_for_prior_games(count, DEFAULT_PRIOR_GAME_BINS) == label


@pytest.mark.parametrize("bad", [None, -1, "", "many", float("nan")])
def test_an_unknown_prior_game_count_has_no_bin(bad):
    assert bin_for_prior_games(bad, DEFAULT_PRIOR_GAME_BINS) is None


def test_not_measured_is_not_reported_as_zero_games():
    """
    "No counts were supplied" and "every player had played no games" are
    different facts, and the second is a claim. Defaulting the bin to 0-2
    would put every row in the early-season bucket and manufacture exactly
    the finding this module is meant to test for.
    """
    rows = [_row() for _ in range(MIN_ROWS_FOR_A_BIAS)]
    section = analyse_losses(rows, prior_games=None)["residuals_by_prior_games"]

    assert section["status"] == "NOT_MEASURED"
    assert "not the same as" in section["reason"]
    assert any("unanswered" in f for f in analyse_losses(rows)["findings"])


def test_an_early_season_bias_is_found_and_named():
    """
    THE FINDING THIS EXISTS TO PRODUCE. Rows from players with few prior
    games miss badly; rows from players with many are close. The report must
    say so AND must point at re-measuring the shrink family rather than
    re-adding it -- it was excluded at r=0.99 against {M}_SEASON.
    """
    rows, priors = [], {}
    for i in range(MIN_ROWS_FOR_A_BIAS):
        key = (f"espn:early{i}", "2026-11-06")
        rows.append(_row(nba_player_id=key[0], model_projection=20.0, actual_result=26.0))
        priors[key] = 1
    for i in range(MIN_ROWS_FOR_A_BIAS):
        key = (f"espn:late{i}", "2026-11-06")
        rows.append(_row(nba_player_id=key[0], model_projection=20.0, actual_result=20.2))
        priors[key] = 30

    report = analyse_losses(rows, prior_games=priors)
    bins = report["residuals_by_prior_games"]["bins"]

    assert bins["0-2"]["mean_bias"] == pytest.approx(6.0)
    assert bins["20+"]["mean_bias"] == pytest.approx(0.2)
    blob = " ".join(report["findings"])
    assert "Early-season rows carry the larger error" in blob, blob
    assert "RE-MEASURE" in blob, "the finding must not read as a licence to re-add"


def test_no_early_season_pattern_is_reported_as_no_support():
    """The other direction: equal error either side must not read as a finding."""
    rows, priors = [], {}
    for i in range(MIN_ROWS_FOR_A_BIAS):
        for tag, n in (("early", 1), ("late", 30)):
            key = (f"espn:{tag}{i}", "2026-11-06")
            rows.append(
                _row(nba_player_id=key[0], model_projection=20.0, actual_result=21.0)
            )
            priors[key] = n

    blob = " ".join(analyse_losses(rows, prior_games=priors)["findings"])
    assert "does not shrink with games played" in blob
    assert "no support for" in blob


# --- bias versus selection -------------------------------------------------


def test_both_sides_leaning_the_same_way_is_called_a_projection_bias():
    rows = []
    for _ in range(MIN_ROWS_FOR_A_BIAS):
        rows.append(_row(predicted_side="OVER", model_projection=20.0, actual_result=24.0))
        rows.append(_row(predicted_side="UNDER", model_projection=20.0, actual_result=24.0))

    blob = " ".join(analyse_losses(rows)["findings"])
    assert "projection bias rather than a side-selection problem" in blob


def test_sides_leaning_opposite_ways_is_called_selection():
    rows = []
    for _ in range(MIN_ROWS_FOR_A_BIAS):
        rows.append(_row(predicted_side="OVER", model_projection=20.0, actual_result=24.0))
        rows.append(_row(predicted_side="UNDER", model_projection=20.0, actual_result=16.0))

    blob = " ".join(analyse_losses(rows)["findings"])
    assert "selection looks like" in blob


# --- day one -------------------------------------------------------------


def test_an_empty_ledger_says_day_one_rather_than_reporting_zeroes():
    """
    On a freshly trained model there are no graded rows at all. That has to
    read as "nothing yet", not as a model with zero bias.
    """
    report = analyse_losses([])
    assert report["n_graded"] == 0
    assert report["residuals_overall"]["mean_bias"] is None
    assert "nothing to analyse" in report["findings"][0]
    assert "day one" in report["findings"][0]
