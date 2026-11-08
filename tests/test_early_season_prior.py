from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from src.features.early_season_prior import (
    EarlySeasonPriorError,
    GameObservation,
    HistoricalPrior,
    early_season_mean,
)

NOW = datetime(2026, 11, 8, 17, tzinfo=timezone.utc)
PRIOR = HistoricalPrior(
    20.0, 5.0, NOW - timedelta(days=30), "test:prior-trained-before-season",
    "PTS", "2026-27",
)


def observation(i=0, value=30.0):
    return GameObservation(
        f"previous-{i}", "player-1", "PTS", "2026-27", value,
        NOW - timedelta(days=i + 1, hours=1),
        NOW - timedelta(days=i + 1),
    )


def estimate(observations=(), **kwargs):
    params = dict(
        season="2026-27", player_id="player-1", market="PTS",
        prediction_event_id="target", as_of=NOW,
        tipoff_at=NOW + timedelta(hours=3), prior=PRIOR,
    )
    params.update(kwargs)
    return early_season_mean(observations, **params)


@pytest.mark.parametrize("n", [0, 1, 4, 9, 10])
def test_early_season_does_not_need_ten_games(n):
    result = estimate([observation(i) for i in range(n)])
    expected_weight = n / (n + 5.0)
    assert result.observed_games == n
    assert result.current_season_weight == pytest.approx(expected_weight)
    assert result.mean == pytest.approx(expected_weight * 30 + (1 - expected_weight) * 20)


@pytest.mark.parametrize("field,value", [
    ("event_id", "target"), ("season", "2025-26"),
    ("player_id", "different-player"), ("market", "REB"),
    ("available_at", NOW + timedelta(seconds=1)),
    ("completed_at", NOW + timedelta(seconds=1)),
    ("available_at", NOW.replace(tzinfo=None)),
    ("value", float("nan")), ("value", float("inf")), ("value", True),
])
def test_invalid_observation_fails_without_zero_fill(field, value):
    with pytest.raises(EarlySeasonPriorError):
        estimate([replace(observation(), **{field: value})])


def test_duplicate_game_cannot_inflate_sample():
    with pytest.raises(EarlySeasonPriorError, match="duplicate"):
        estimate([observation(), observation()])


@pytest.mark.parametrize("field,value", [
    ("available_at", NOW + timedelta(seconds=1)),
    ("available_at", NOW.replace(tzinfo=None)),
    ("mean", float("nan")), ("equivalent_games", 0.0),
    ("equivalent_games", -1.0), ("equivalent_games", float("inf")),
    ("provenance", ""),
    ("market", "AST"), ("season", "2025-26"),
])
def test_invalid_or_future_prior_is_not_used(field, value):
    with pytest.raises(EarlySeasonPriorError):
        estimate(prior=replace(PRIOR, **{field: value}))


@pytest.mark.parametrize("kwargs", [
    {"as_of": NOW.replace(tzinfo=None)},
    {"tipoff_at": NOW},
    {"tipoff_at": NOW - timedelta(seconds=1)},
    {"prediction_event_id": ""},
])
def test_prediction_identity_and_pregame_cutoff(kwargs):
    with pytest.raises(EarlySeasonPriorError):
        estimate(**kwargs)


def test_negative_fantasy_observations_are_not_clipped():
    result = estimate([observation(value=-2.0)])
    assert result.mean == pytest.approx((5 * 20 - 2) / 6)
