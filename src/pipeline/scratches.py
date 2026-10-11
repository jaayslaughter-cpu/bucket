"""Pre-tip scratch filter, from ESPN's injury report. RESEARCH_ONLY.

THE GAP THIS FILLS is R4 in ``docs/railway_deployment_audit.md``:
``ingestion/inactive_players.py`` and ``features/absences.py`` are wired into the
feature builder, but as a TRAINING-PANEL feature. Nothing dropped a projection
when a player was ruled out AFTER that projection was written. A recommendation
on a player who will not dress is not a weak recommendation; it is a row that
should not exist.

WHY ESPN AND NOT stats.nba.com. The official pregame inactive list comes from
``boxscoresummaryv3``, and ``stats.nba.com`` returns 403 through this
environment's proxy — the existing puller cannot be exercised here at all.
ESPN's public injuries feed is the reachable alternative and needs no credential.

AN UNUSABLE REPORT MUST NOT READ AS A HEALTHY SLATE, which is the rule the whole
module turns on and the one an earlier version of ``projected_available`` got
exactly backwards. When the feed fails, every row is marked UNVERIFIED and
NOTHING is dropped or cleared: the filter reports that it could not check, and a
caller that ignores ``status`` gets rows it can see are unchecked rather than
rows that look cleared. Silence and "everyone is playing" are different claims.

FOUR VALUES, NOT TWO:

  AVAILABLE    no OUT/DOUBTFUL row for this player. Includes players ESPN never
               mentions, because most of a roster is healthy and absent from an
               injury feed.
  WITHHELD     ESPN says OUT or DOUBTFUL. The projection stays in the frame,
               labelled, so the record of what the model said survives — but
               ``settlement.recorder`` does not write it as a prediction and the
               board should not recommend it.
  UNKNOWN      ESPN has a row it could not bucket. Not the same as healthy.
  UNVERIFIED   the feed did not answer. Not the same as healthy either.

MATCHING IS EXACT ON A NORMALISED NAME, NEVER FUZZY — and the normaliser is
now ``ingestion.id_crosswalk.normalise_player_name``, shared so both sides
reduce to the same form. It is deterministic, not a score: that module measured
that no fuzzy cutoff separates ``Jokic``/``Jokić`` (81.8-91.7) from
``Jalen``/``Jaylen`` Williams (96.6), and withholding the wrong player is worse
than withholding nobody. ESPN athlete ids would be better still, but a
projections frame carries NBA ids, so a name is what the two sides share.

DOUBTFUL IS TREATED AS UNAVAILABLE, inherited from
``espn_availability.UNAVAILABLE``. Doubtful players mostly do not play, and the
asymmetry favours dropping a row that might have been fine over recommending one
that will not dress.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

AVAILABILITY_COLUMN = "AVAILABILITY"
DETAIL_COLUMN = "AVAILABILITY_DETAIL"

AVAILABLE = "AVAILABLE"
WITHHELD = "WITHHELD"
UNKNOWN = "UNKNOWN"
UNVERIFIED = "UNVERIFIED"

STATUS_OK = "OK"
STATUS_UNAVAILABLE = "DATA_NOT_AVAILABLE"


@dataclass
class ScratchFilterResult:
    """The annotated frame plus what the check could and could not establish."""

    status: str
    projections: pd.DataFrame
    withheld: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)
    reason: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def verified(self) -> bool:
        """True only when a usable report actually answered."""
        return self.status == STATUS_OK

    def as_dict(self) -> dict[str, Any]:
        counts = (
            self.projections[AVAILABILITY_COLUMN].value_counts().to_dict()
            if AVAILABILITY_COLUMN in getattr(self.projections, "columns", [])
            else {}
        )
        return {
            "status": self.status,
            "verified": self.verified,
            "rows": int(len(self.projections)),
            "by_availability": {str(k): int(v) for k, v in counts.items()},
            "withheld_players": sorted(set(self.withheld))[:20],
            "unknown_players": sorted(set(self.unknown))[:20],
            "reason": self.reason,
            "notes": self.notes,
            "note": (
                "An UNVERIFIED row is not a cleared row. Nothing is dropped when "
                "the feed fails; the rows are marked so a caller can see they "
                "were never checked."
            ),
        }


def _normalise(name: Any) -> str:
    """
    The crosswalk's canonical form, shared so both sides agree.

    THIS USED TO BE LOWERCASE AND WHITESPACE ONLY, and that was a silent safety
    failure rather than a cosmetic gap. ESPN publishes ``Nikola Jokić`` and the
    NBA panel carries ``Nikola Jokic``; those do not compare equal, so a player
    ESPN reported OUT was labelled AVAILABLE and the filter withheld nobody —
    demonstrated before the fix, and pinned by
    ``tests/test_name_crosswalk.py``.

    ``id_crosswalk.normalise_player_name`` strips diacritics, dots and
    apostrophes and spaces out hyphens. It is still EXACT and still not fuzzy,
    which is what the note at the top of this module requires: no score
    threshold can separate ``Jokic``/``Jokić`` from ``Jalen``/``Jaylen``, and
    that module carries the measurement.
    """
    from src.ingestion.id_crosswalk import normalise_player_name

    return normalise_player_name(name)


def apply_scratch_filter(
    projections: pd.DataFrame,
    report: Any | None = None,
    *,
    session: Any | None = None,
    config: Any | None = None,
) -> ScratchFilterResult:
    """
    Label every projection with the player's ESPN availability.

    ``report`` is an ``espn_availability.AvailabilityReport``. Passing one skips
    the fetch, which is how this is tested in an environment where every ESPN
    host is denied; passing None fetches it.

    Never raises on a feed failure. A scheduled slate must not die because an
    injury page moved, and the honest outcome of a failed check is labelled rows
    rather than no rows.
    """
    frame = projections.copy() if projections is not None else pd.DataFrame()
    if frame.empty:
        out = ScratchFilterResult(
            status=STATUS_UNAVAILABLE,
            projections=frame,
            reason="no projections to check",
        )
        return out

    if report is None:
        try:
            from src.ingestion.espn_availability import fetch_injuries

            report = fetch_injuries(config=config, session=session)
        except Exception as exc:  # noqa: BLE001 — a failed check is a state, not a crash
            logger.warning(
                "Scratch filter could not reach ESPN (%s). Rows are marked "
                "%s — NOT cleared.", exc, UNVERIFIED,
            )
            frame[AVAILABILITY_COLUMN] = UNVERIFIED
            frame[DETAIL_COLUMN] = None
            return ScratchFilterResult(
                status=STATUS_UNAVAILABLE,
                projections=frame,
                reason=f"injury feed unreachable: {exc}",
            )

    status = str(getattr(report, "status", "") or "")
    notes = list(getattr(report, "notes", []) or [])

    if status != "OK":
        frame[AVAILABILITY_COLUMN] = UNVERIFIED
        frame[DETAIL_COLUMN] = None
        return ScratchFilterResult(
            status=STATUS_UNAVAILABLE,
            projections=frame,
            reason=(
                f"injury report status is {status!r}, so no player can be shown "
                "available or withheld; every row is unverified"
            ),
            notes=notes,
        )

    from src.ingestion.espn_availability import UNAVAILABLE

    by_name = {
        _normalise(row.player_name): row
        for row in getattr(report, "injuries", []) or []
    }

    labels: list[str] = []
    details: list[str | None] = []
    withheld: list[str] = []
    unknown: list[str] = []

    for name in frame.get("PLAYER_NAME", pd.Series([None] * len(frame))):
        hit = by_name.get(_normalise(name))
        if hit is None:
            labels.append(AVAILABLE)
            details.append(None)
            continue
        if hit.status in UNAVAILABLE:
            labels.append(WITHHELD)
            withheld.append(str(name))
        elif hit.status == "DATA_NOT_AVAILABLE":
            labels.append(UNKNOWN)
            unknown.append(str(name))
        else:
            labels.append(AVAILABLE)
        details.append(hit.detail or hit.status_raw)

    frame[AVAILABILITY_COLUMN] = labels
    frame[DETAIL_COLUMN] = details

    if withheld:
        logger.info(
            "Scratch filter withheld %d projection row(s) for %d player(s) ESPN "
            "lists OUT or DOUBTFUL: %s",
            labels.count(WITHHELD), len(set(withheld)), sorted(set(withheld))[:10],
        )
    return ScratchFilterResult(
        status=STATUS_OK,
        projections=frame,
        withheld=withheld,
        unknown=unknown,
        notes=notes,
    )
