"""
How old the fitted artifact is, and whether to keep publishing from it.

WHY THIS EXISTS. Everything else in this project measures whether the artifact
is *usable*: `main.resolve_model_artifact` requires its `.meta.json` sidecar,
`railway_healthcheck` refuses one fit on fewer than `MIN_PLAUSIBLE_TRAIN_ROWS`,
and `publication_gate.calibration_gate` refuses a figure whose calibration
evidence is stale. Nothing measured whether the artifact itself is old. The
sidecar has recorded `saved_at_utc` since the first training run and **no
module ever read it back**.

That is the failure mode an unattended worker is worst at surviving, because it
looks exactly like success. A nightly board stamped with tonight's date goes out
to Discord, every number present, every contract satisfied, scored by a model
fit months earlier. Nothing in the message says so and nothing in the logs is
red. An outright error would be cheaper.

TWO AXES, NOT ONE. The obvious check is file age — now minus `saved_at_utc` —
and on its own it is not enough. A retrain that keeps running but keeps reading
the same stale training frame rewrites `saved_at_utc` every night forever while
the model learns nothing new, so file age alone certifies the exact pipeline it
should catch. So training recency is measured too, from `train_end_date`.

AND TRAINING RECENCY IS NOT MEASURED AGAINST TODAY. It is measured against the
newest game date we actually hold, which the caller passes in (the same figure
`main.market_frames_freshness` computes). Against today, every artifact would
go stale each July and fire an alert every offseason for four months, which is
how a staleness alarm gets muted and then ignored. Against the data we have,
the offseason moves both numbers together and the check stays quiet until a
retrain genuinely falls behind available games.

A MISSING TIMESTAMP IS NOT A PASS. `UNDATED` is reported as its own status and
the gate withholds on it, because "we cannot establish this artifact's age" and
"this artifact is current" are the two readings a silent pass would merge, and
only one of them is safe.

RESEARCH_ONLY. This module reads a sidecar and does arithmetic on dates. It
fetches nothing, scores nothing, sizes nothing and places nothing.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

#: Statuses, mirroring `main.market_frames_freshness` so a reader who knows one
#: already knows the other.
OK = "OK"
STALE = "STALE"
UNDATED = "UNDATED"
MISSING = "MISSING"

#: The sidecar keys this reads. Named constants because they were VERIFIED
#: against the artifacts on disk rather than assumed: `saved_at_utc` and
#: `train_end_date` are present in 8 of the 9 sidecars under
#: `data/external/model_runs`, while `registered_at_utc` — the key
#: `models/artifact_registry.py` builds — appears in none of them, because that
#: module writes a separate `model_registry.jsonl` index that nothing writes to
#: and `resolve_model_artifact` deliberately does not read.
SAVED_AT_KEY = "saved_at_utc"
TRAIN_END_KEY = "train_end_date"

ENV_MAX_ARTIFACT_AGE = "PROPIQ_MAX_ARTIFACT_AGE_DAYS"
ENV_MAX_TRAINING_LAG = "PROPIQ_MAX_TRAINING_LAG_DAYS"

#: Six weeks. Long enough that a deliberate pause between retrains does not trip
#: it, short enough that a silently dead retrain is caught inside one month of
#: the season. This number is OURS TO DECIDE and is pinned by a test, so raising
#: it to make a particular artifact pass fails that test rather than quietly
#: reversing the conclusion. It is not tuned to any artifact now on disk.
DEFAULT_MAX_ARTIFACT_AGE_DAYS = 45

#: Three weeks of games between the end of training and the newest game we hold.
#: Tighter than artifact age on purpose: this one is measured against our own
#: data, so it does not move during the offseason and does not need slack for it.
DEFAULT_MAX_TRAINING_LAG_DAYS = 21

SIDECAR_SUFFIX = ".meta.json"


def _int_env(name: str, default: int, *, low: int, high: int) -> int:
    """
    A bounded integer env var, defaulting rather than raising.

    The fourth small copy of this contract (`main`, `scheduler_worker`,
    `db.session` hold the others) and deliberately not an import: reading an
    artifact's age should not make `src.models` depend on the orchestrator or on
    SQLAlchemy for four lines of parsing.
    """
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("%s=%r is not an integer; using %d", name, raw, default)
        return default
    if not low <= value <= high:
        logger.warning("%s=%d is outside [%d, %d]; using %d", name, value, low, high, default)
        return default
    return value


def max_artifact_age_days() -> int:
    return _int_env(ENV_MAX_ARTIFACT_AGE, DEFAULT_MAX_ARTIFACT_AGE_DAYS, low=1, high=3650)


def max_training_lag_days() -> int:
    return _int_env(ENV_MAX_TRAINING_LAG, DEFAULT_MAX_TRAINING_LAG_DAYS, low=1, high=3650)


def _parse_utc(value: Any) -> datetime | None:
    """ISO-8601 to an aware UTC datetime, or None when it is not one."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    # A trailing Z is valid ISO-8601 and fromisoformat accepts it from 3.11,
    # this project's floor. Normalised anyway so behaviour does not depend on
    # the interpreter's minor version.
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _parse_date(value: Any) -> date | None:
    """A `YYYY-MM-DD` day, or None. Accepts a full timestamp and takes its date."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    parsed = _parse_utc(value)
    if parsed is not None:
        return parsed.date()
    try:
        return date.fromisoformat(value.strip()[:10])
    except ValueError:
        return None


def read_sidecar(artifact_path: Path | str) -> Mapping[str, Any] | None:
    """
    The `.meta.json` beside an artifact, or None when it is absent or unreadable.

    None for both cases on purpose: the caller's next step is the same either
    way (it cannot establish the age), and the distinction is logged here rather
    than pushed into every call site.
    """
    sidecar = Path(artifact_path).with_suffix(SIDECAR_SUFFIX)
    if not sidecar.exists():
        logger.warning("No %s beside %s, so its age cannot be read.", sidecar.name, artifact_path)
        return None
    try:
        loaded = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("%s could not be read (%s), so its age cannot be read.", sidecar, exc)
        return None
    if not isinstance(loaded, Mapping):
        logger.warning("%s is not a JSON object, so its age cannot be read.", sidecar)
        return None
    return loaded


def artifact_freshness(
    sidecar: Mapping[str, Any] | None,
    *,
    now: datetime | None = None,
    newest_game_date: str | date | datetime | None = None,
    market: str | None = None,
) -> dict[str, Any]:
    """
    Age the artifact on both axes and say which axes were actually measured.

    ``newest_game_date`` is the newest game we hold — pass
    ``main.market_frames_freshness(...)["newest_game_date"]``. Omit it and the
    training-recency axis is NOT judged: ``training_lag_days`` comes back None
    and ``checks`` omits it, so a caller cannot read "not measured" as "fine".
    """
    moment = now or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)

    max_age = max_artifact_age_days()
    max_lag = max_training_lag_days()
    out: dict[str, Any] = {
        "status": OK,
        "market": market,
        "saved_at_utc": None,
        "artifact_age_days": None,
        "max_artifact_age_days": max_age,
        "train_end_date": None,
        "training_lag_days": None,
        "max_training_lag_days": max_lag,
        "newest_game_date": None,
        "checks": (),
        "reasons": (),
    }

    if not sidecar:
        out["status"] = MISSING
        out["reasons"] = (
            "No sidecar was supplied, so this artifact's age cannot be "
            "established. An artifact of unknown age is treated as stale.",
        )
        return out

    checks: list[str] = []
    reasons: list[str] = []

    saved_at = _parse_utc(sidecar.get(SAVED_AT_KEY))
    if saved_at is None:
        out["status"] = UNDATED
        reasons.append(
            f"The sidecar has no readable {SAVED_AT_KEY!r} "
            f"(found {sidecar.get(SAVED_AT_KEY)!r}), so the artifact's age "
            "cannot be established. Retrain, or seed an artifact whose sidecar "
            "carries the timestamp training writes."
        )
    else:
        age_days = (moment - saved_at).total_seconds() / 86400.0
        out["saved_at_utc"] = saved_at.isoformat()
        out["artifact_age_days"] = round(age_days, 2)
        checks.append("artifact_age")
        if age_days > max_age:
            out["status"] = STALE
            reasons.append(
                f"The artifact was saved {age_days:.1f} day(s) ago "
                f"({saved_at.date()}), over the {max_age}-day limit "
                f"(${ENV_MAX_ARTIFACT_AGE}). Nothing has retrained since."
            )

    # Training recency, only when the caller supplied what to measure against.
    newest = _parse_date(newest_game_date)
    train_end = _parse_date(sidecar.get(TRAIN_END_KEY))
    out["train_end_date"] = str(train_end) if train_end else None
    out["newest_game_date"] = str(newest) if newest else None

    if newest is not None:
        if train_end is None:
            # Not UNDATED on its own: the artifact's own age may still be
            # readable and is the stronger signal. Said out loud either way.
            reasons.append(
                f"The sidecar has no readable {TRAIN_END_KEY!r} "
                f"(found {sidecar.get(TRAIN_END_KEY)!r}), so training recency "
                "was not measured. Only the artifact's file age was."
            )
        else:
            lag = (newest - train_end).days
            out["training_lag_days"] = lag
            checks.append("training_recency")
            if lag > max_lag:
                out["status"] = STALE
                reasons.append(
                    f"Training stopped at {train_end} and we hold games through "
                    f"{newest} — {lag} day(s) of games the model never saw, over "
                    f"the {max_lag}-day limit (${ENV_MAX_TRAINING_LAG}). A "
                    "retrain that keeps writing a fresh file off a stale frame "
                    "looks current on its timestamp alone."
                )

    out["checks"] = tuple(checks)
    out["reasons"] = tuple(reasons)

    if out["status"] == OK:
        logger.info(
            "Artifact freshness%s: saved %s (%.1f d old, limit %d), trained through %s%s.",
            f" [{market}]" if market else "",
            out["saved_at_utc"],
            out["artifact_age_days"] or 0.0,
            max_age,
            out["train_end_date"],
            f", {out['training_lag_days']} d behind our newest game"
            if out["training_lag_days"] is not None
            else " (recency not measured)",
        )
    else:
        logger.warning(
            "ARTIFACT IS %s%s: %s",
            out["status"],
            f" [{market}]" if market else "",
            " ".join(reasons),
        )
    return out


def artifact_freshness_for(
    artifact_path: Path | str,
    *,
    now: datetime | None = None,
    newest_game_date: str | date | datetime | None = None,
    market: str | None = None,
) -> dict[str, Any]:
    """`artifact_freshness` straight off a path, for callers that hold one."""
    return artifact_freshness(
        read_sidecar(artifact_path),
        now=now,
        newest_game_date=newest_game_date,
        market=market,
    )
