"""Optional, timestamp-checked mean shrinkage for early-season A/B research.

This is an original PropIQ integration candidate, not code from DeVig-AI.
It estimates a mean only: it supplies neither a calibrated probability nor EV.
It is deliberately not registered with builder.py or any serving scorer.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone


class EarlySeasonPriorError(ValueError):
    """Inputs cannot justify an as-of early-season mean."""


@dataclass(frozen=True)
class GameObservation:
    event_id: str
    player_id: str
    market: str
    season: str
    value: float
    completed_at: datetime
    available_at: datetime


@dataclass(frozen=True)
class HistoricalPrior:
    mean: float
    equivalent_games: float
    available_at: datetime
    provenance: str
    market: str
    season: str


@dataclass(frozen=True)
class MeanEstimate:
    mean: float
    observed_games: int
    current_season_weight: float


def _utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise EarlySeasonPriorError("Every timestamp must be timezone-aware")
    return value.astimezone(timezone.utc)


def _identity(value: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise EarlySeasonPriorError("Identity and provenance strings must be nonempty and canonical")
    return value


def _finite(value: float) -> float:
    if isinstance(value, bool):
        raise EarlySeasonPriorError("Boolean values are not numeric observations")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise EarlySeasonPriorError("Non-numeric observation or prior") from exc
    if not math.isfinite(number):
        raise EarlySeasonPriorError("Observation and prior values must be finite")
    return number


def early_season_mean(
    observations: Sequence[GameObservation],
    *,
    season: str,
    player_id: str,
    market: str,
    prediction_event_id: str,
    as_of: datetime,
    tipoff_at: datetime,
    prior: HistoricalPrior,
) -> MeanEstimate:
    """Blend eligible current-season games with an explicitly supplied prior.

    The caller supplies the season key from PropIQ's existing season helper.
    No completed games are silently discarded: duplicates, wrong-season rows,
    target-game observations, and future/unavailable inputs all fail.
    Prior strength must be selected on chronological training folds, not on
    the evaluation period. Unknown historical timestamps must not be invented.
    """
    season = _identity(season)
    player_id, market = _identity(player_id), _identity(market)
    prediction_event_id = _identity(prediction_event_id)
    _identity(prior.provenance)
    if prior.market != market or prior.season != season:
        raise EarlySeasonPriorError("Prior market or target season does not match")
    cutoff, tipoff = _utc(as_of), _utc(tipoff_at)
    if cutoff >= tipoff:
        raise EarlySeasonPriorError("Prediction must precede tipoff")
    if _utc(prior.available_at) > cutoff:
        raise EarlySeasonPriorError("Prior was unavailable at prediction time")
    prior_mean, strength = _finite(prior.mean), _finite(prior.equivalent_games)
    if strength <= 0:
        raise EarlySeasonPriorError("Prior equivalent_games must be positive")

    seen: set[str] = set()
    values: list[float] = []
    for observation in observations:
        event_id = _identity(observation.event_id)
        if event_id == prediction_event_id or event_id in seen:
            raise EarlySeasonPriorError("Target-game observation or duplicate game")
        if observation.season != season:
            raise EarlySeasonPriorError("Observation belongs to a different season")
        if observation.player_id != player_id or observation.market != market:
            raise EarlySeasonPriorError("Observation belongs to a different player or market")
        completed, available = _utc(observation.completed_at), _utc(observation.available_at)
        if completed > available or available > cutoff:
            raise EarlySeasonPriorError("Game was unfinished or unavailable at prediction time")
        seen.add(event_id)
        values.append(_finite(observation.value))

    n = len(values)
    if not n:
        return MeanEstimate(prior_mean, 0, 0.0)
    # A convex combination avoids overflow from multiplying huge prior weights.
    weight = n / (n + strength)
    sample_mean = math.fsum(value / n for value in values)
    estimate = weight * sample_mean + (1.0 - weight) * prior_mean
    if not math.isfinite(estimate):
        raise EarlySeasonPriorError("Mean computation was nonfinite")
    return MeanEstimate(estimate, n, weight)
