"""Assert the pipeline's own invariants, so a silent failure stops being silent.

RESEARCH_ONLY. Reads nothing and writes nothing: every function here is pure
over inputs the caller supplies, exactly as `settlement.component_comparison`
is. The caller does the reads and the alerting.

WHY THIS EXISTS, AND IT IS NOT A HYPOTHETICAL.
`scheduler_worker.notify_failure` alerts when a job RAISES. Every defect found
in this pipeline over the week of 2026-11-04 to 11-11 was in the other class --
a job that returned ``status: OK`` while doing nothing -- and every one was
found by a person reading Railway logs by hand:

  * ``Loss analysis: graded 0`` on three consecutive nights. Status OK.
  * ``n_consider_with_book_ev: 0`` and ``sources_used: []`` on seven slates
    running, while the EV gate reported 17,210 rows ready. Status OK.
  * The research preview printing "No probability is shown because none was
    computed" on a slate that computed 312 of them. Status OK.
  * 24 graded-ready props left PENDING, which a 14-day age cutoff would have
    dropped without a line in any log.
  * A fitted `line_aware` artifact no resolver could find, so a component
    persisted and verified on every run was unreachable for weeks.
  * `src/models/minutes_model.py`, imported by nothing at all.

None of those raised. A monitor that only watches for exceptions cannot see
any of them, so this watches the OUTPUTS instead and asks whether they are
consistent with each other.

THREE STATUSES, AND THE THIRD IS THE POINT.
``CHECK_OK`` the invariant holds. ``CHECK_ALERT`` it is broken, with a reason
naming what and by how much. ``CHECK_CANNOT_TELL`` there is not enough
evidence to say -- a first night, an empty ledger, a slate too recent to have
graded. A monitor that reports a green OK when it has no evidence is worse
than no monitor: it converts absence of data into a statement that everything
is fine, which is the exact failure mode it exists to catch.

WHAT IT DELIBERATELY DOES NOT DO. It does not decide what to do about a
broken invariant, does not touch a configuration, and does not suppress
anything. It returns findings; `scheduler_worker` routes them to
`notify_failure`, which already scrubs, dedupes per day, and never raises.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Iterable, Mapping, Sequence

logger = logging.getLogger(__name__)

CHECK_OK = "OK"
CHECK_ALERT = "ALERT"
CHECK_CANNOT_TELL = "CANNOT_TELL"

#: A slate younger than this has not had a settlement run against it yet, so
#: an ungraded row is expected rather than stranded. Settlement runs at 03:30
#: PT for the previous Pacific day, so one day is the minimum that can mean
#: anything and two leaves room for a late finish.
DEFAULT_GRACE_DAYS = 2

#: Above this share of an aged slate ungraded, something stopped grading.
#: Voided scratches are legitimately ungraded -- 24 of 312 on 2026-11-09, or
#: 7.7% -- so the bar sits well clear of a normal night's voids while still
#: catching a settlement that did nothing.
DEFAULT_MAX_UNGRADED_SHARE = 0.25

#: A recorded count below this share of the recent median means the slate
#: collapsed: a feed went away, the scratch filter over-withheld, or the line
#: join stopped matching.
DEFAULT_MIN_VOLUME_SHARE = 0.25

#: No record newer than this and the slate job is not recording, whatever its
#: status said.
DEFAULT_MAX_RECORD_AGE_DAYS = 2


@dataclass(frozen=True)
class CheckResult:
    """One invariant, its verdict, and why."""

    name: str
    status: str
    reason: str
    #: Numbers a reader would otherwise have to recompute to trust the verdict.
    detail: Mapping[str, Any] = field(default_factory=dict)

    @property
    def alerting(self) -> bool:
        return self.status == CHECK_ALERT

    def as_dict(self) -> dict[str, Any]:
        return {
            "CHECK": self.name,
            "STATUS": self.status,
            "REASON": self.reason,
            "DETAIL": dict(self.detail),
        }


def _as_date(value: Any) -> date | None:
    """A date, or None. Never today() as a fallback -- see `_counts` below."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text[:10]).date()
    except ValueError:
        return None


def _recorded_count(record: Mapping[str, Any]) -> int:
    counts = record.get("counts") or {}
    by_decision = counts.get("by_decision") or {}
    try:
        return int(by_decision.get("RECORDED") or 0)
    except (TypeError, ValueError):
        return 0


def _recorded_rows(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        r for r in (record.get("rows") or [])
        if isinstance(r, Mapping) and str(r.get("decision") or "") == "RECORDED"
    ]


# --------------------------------------------------------------------------- #
# The checks. Each is pure and each can say it does not know.
# --------------------------------------------------------------------------- #

def records_are_chained(records: Sequence[Mapping[str, Any]]) -> CheckResult:
    """Every record after the first names the one before it, in one chain.

    The chain is the only thing that can show a record was not altered after
    the fact, and a break in it is the difference between an audit trail and a
    directory of files. A fork -- two records claiming the same parent -- means
    two writers, which on one volume means one of them lost.
    """
    name = "records_are_chained"
    if len(records) < 2:
        return CheckResult(
            name, CHECK_CANNOT_TELL,
            f"{len(records)} record(s); a chain needs at least two links to "
            f"have a shape at all",
            {"records": len(records)},
        )

    seen_parents: dict[str, int] = {}
    broken: list[str] = []
    for i, rec in enumerate(records[1:], start=1):
        parent = rec.get("prev_sha256")
        stamp = str(rec.get("as_of_ts") or f"record {i}")
        if not parent:
            broken.append(f"{stamp} names no parent")
            continue
        seen_parents[str(parent)] = seen_parents.get(str(parent), 0) + 1
    forks = [p for p, n in seen_parents.items() if n > 1]
    if forks:
        broken.append(
            f"{len(forks)} parent digest(s) claimed by more than one record, "
            f"so two writers produced records from the same state"
        )
    if broken:
        return CheckResult(
            name, CHECK_ALERT,
            "The decision-record chain is broken: " + "; ".join(broken),
            {"records": len(records), "problems": broken},
        )
    return CheckResult(
        name, CHECK_OK,
        f"{len(records)} record(s) form one unbroken chain",
        {"records": len(records)},
    )


def a_slate_was_recorded_recently(
    records: Sequence[Mapping[str, Any]],
    *,
    as_of: date,
    max_age_days: int = DEFAULT_MAX_RECORD_AGE_DAYS,
) -> CheckResult:
    """The slate job is still recording predictions.

    THE SOURCE OF THE "graded 0 for three nights" CLASS. If nothing is being
    recorded, nothing can ever be graded, and the 04:00 job will report an
    honest zero every night while the real failure sits twelve hours upstream
    in a job that returned OK.
    """
    name = "a_slate_was_recorded_recently"
    dates = [d for d in (_as_date(r.get("slate_date")) for r in records) if d]
    if not dates:
        return CheckResult(
            name, CHECK_CANNOT_TELL,
            "No decision record carries a readable slate date, so there is "
            "nothing to measure an age against",
            {"records": len(records)},
        )
    newest = max(dates)
    age = (as_of - newest).days
    if age > int(max_age_days):
        return CheckResult(
            name, CHECK_ALERT,
            f"The newest decision record is for {newest.isoformat()}, "
            f"{age} day(s) before {as_of.isoformat()}. The slate job has not "
            f"recorded a prediction since, whatever its status reported",
            {"newest_slate": newest.isoformat(), "age_days": age},
        )
    return CheckResult(
        name, CHECK_OK,
        f"The newest record is for {newest.isoformat()}, {age} day(s) old",
        {"newest_slate": newest.isoformat(), "age_days": age},
    )


def aged_slates_are_graded(
    records: Sequence[Mapping[str, Any]],
    graded_game_dates: Iterable[Any],
    *,
    as_of: date,
    grace_days: int = DEFAULT_GRACE_DAYS,
    max_ungraded_share: float = DEFAULT_MAX_UNGRADED_SHARE,
) -> CheckResult:
    """A slate old enough to have settled has most of its rows graded.

    A SHARE AND NOT A COUNT, because slates differ in size by a factor of
    four and a fixed count would mean something different on each.

    WHAT IT CATCHES: a settlement that largely or entirely stopped -- an aged
    slate sitting at 100% ungraded, which is what a broken box-score fetch, a
    bad id join or a job that never ran produces.

    WHAT IT DOES NOT CATCH, STATED RATHER THAN IMPLIED: a SMALL stranded set.
    The 24 props left PENDING on 2026-11-09 were 7.7% of that slate, well
    under this bar. The bar cannot simply be lowered to reach them, because
    VOIDED rows are legitimately ungraded -- `outcome_status` of WIN/LOSS/PUSH
    is what `graded_game_dates` carries -- so a night with many scratches
    would be indistinguishable from a night with stranded rows at any
    threshold low enough to catch 7.7%. Separating the two needs the PENDING
    count per slate, which the caller does not currently read. That specific
    defect is instead fixed at source: `settlement.runner` now VOIDS a prop
    with no box-score line rather than leaving it PENDING for a 14-day cutoff
    to drop.
    """
    name = "aged_slates_are_graded"
    graded_by_date: dict[date, int] = {}
    for value in graded_game_dates:
        d = _as_date(value)
        if d is not None:
            graded_by_date[d] = graded_by_date.get(d, 0) + 1

    aged: list[tuple[date, int, int]] = []
    for rec in records:
        slate = _as_date(rec.get("slate_date"))
        if slate is None or (as_of - slate).days < int(grace_days):
            continue
        recorded = _recorded_count(rec)
        if recorded <= 0:
            continue
        # The record's own rows carry the GAME date, which is what the ledger
        # keys on. A slate date and a game date can differ across midnight.
        game_dates = {
            d for d in (_as_date(r.get("game_date")) for r in _recorded_rows(rec)) if d
        } or {slate}
        graded = sum(graded_by_date.get(d, 0) for d in game_dates)
        aged.append((slate, recorded, graded))

    if not aged:
        return CheckResult(
            name, CHECK_CANNOT_TELL,
            f"No recorded slate is yet {grace_days} day(s) old, so none has "
            f"had a settlement run against it",
            {"grace_days": int(grace_days)},
        )

    bad: list[str] = []
    for slate, recorded, graded in sorted(aged):
        share = 1.0 - (min(graded, recorded) / recorded)
        if share > float(max_ungraded_share):
            bad.append(
                f"{slate.isoformat()}: {recorded - min(graded, recorded)} of "
                f"{recorded} recorded row(s) never graded ({share:.0%})"
            )
    if bad:
        return CheckResult(
            name, CHECK_ALERT,
            "Slates old enough to have settled are still mostly ungraded: "
            + "; ".join(bad)
            + ". A row that can never grade should be VOID, not PENDING",
            {"slates": len(aged), "problems": bad},
        )
    return CheckResult(
        name, CHECK_OK,
        f"{len(aged)} aged slate(s) graded within {max_ungraded_share:.0%}",
        {"slates": len(aged)},
    )


def every_component_answered(
    records: Sequence[Mapping[str, Any]],
) -> CheckResult:
    """The served component and every shadow scored something on the last slate.

    THE DEFECT THIS CATCHES IS THE ONE THAT HID LONGEST. A fitted `line_aware`
    artifact was persisted, checksummed and verified on every run while no
    resolver could find it, so a component the pipeline believed it was
    measuring answered nothing at all. A component that scores zero rows is
    indistinguishable in a status line from one that was never asked.
    """
    name = "every_component_answered"
    if not records:
        return CheckResult(
            name, CHECK_CANNOT_TELL, "No decision records to read", {}
        )
    newest = records[-1]
    rows = _recorded_rows(newest)
    if not rows:
        return CheckResult(
            name, CHECK_CANNOT_TELL,
            "The newest record has no RECORDED rows, so no component was "
            "asked anything",
            {"slate_date": str(newest.get("slate_date") or "")},
        )

    served = sum(1 for r in rows if r.get("prob_over") is not None)
    scored: dict[str, int] = {}
    for r in rows:
        shadow = r.get("shadow") or {}
        if isinstance(shadow, Mapping):
            for component, prob in shadow.items():
                scored[str(component)] = scored.get(str(component), 0) + (
                    1 if prob is not None else 0
                )

    silent = [c for c, n in scored.items() if n == 0]
    detail = {
        "slate_date": str(newest.get("slate_date") or ""),
        "recorded_rows": len(rows),
        "served_scored": served,
        "shadow_scored": scored,
    }
    if served == 0:
        return CheckResult(
            name, CHECK_ALERT,
            f"The SERVING component scored 0 of {len(rows)} recorded row(s) on "
            f"{detail['slate_date']}. Those rows carry a line and no "
            f"probability, which is a resolver or artifact failure, not a quiet slate",
            detail,
        )
    if silent:
        return CheckResult(
            name, CHECK_ALERT,
            f"Shadow component(s) {sorted(silent)} scored 0 of {len(rows)} "
            f"recorded row(s) on {detail['slate_date']}. A component that "
            f"answers nothing cannot be compared and reads as one never asked",
            detail,
        )
    if not scored:
        return CheckResult(
            name, CHECK_CANNOT_TELL,
            f"The serving component scored {served} of {len(rows)} row(s) and "
            f"no shadow was recorded, so there is nothing to compare it with",
            detail,
        )
    return CheckResult(
        name, CHECK_OK,
        f"Served scored {served} of {len(rows)}; shadows "
        + ", ".join(f"{c}={n}" for c, n in sorted(scored.items())),
        detail,
    )


def recorded_volume_has_not_collapsed(
    records: Sequence[Mapping[str, Any]],
    *,
    min_share: float = DEFAULT_MIN_VOLUME_SHARE,
    min_history: int = 3,
) -> CheckResult:
    """The newest slate recorded a normal number of predictions.

    A MEDIAN, NOT A MEAN, and not the previous night alone. Slate sizes vary
    honestly -- 312 on 2026-11-09 against 257 on 11-10, because the second
    night had fewer posted lines -- so a day-on-day comparison would cry wolf
    every other night. What this catches is a collapse: a feed that went away,
    a scratch filter that over-withheld, a line join that stopped matching.
    """
    name = "recorded_volume_has_not_collapsed"
    counts = [_recorded_count(r) for r in records]
    if len(counts) < int(min_history) + 1:
        return CheckResult(
            name, CHECK_CANNOT_TELL,
            f"{len(counts)} record(s); a median needs at least "
            f"{int(min_history)} prior slates to mean anything",
            {"records": len(counts)},
        )
    newest, prior = counts[-1], counts[:-1]
    median = statistics.median(prior)
    if median <= 0:
        return CheckResult(
            name, CHECK_CANNOT_TELL,
            "Every prior slate recorded nothing, so there is no normal volume "
            "to compare against",
            {"newest": newest, "prior_median": median},
        )
    share = newest / median
    detail = {"newest": newest, "prior_median": median, "share": round(share, 4)}
    if share < float(min_share):
        return CheckResult(
            name, CHECK_ALERT,
            f"The newest slate recorded {newest} prediction(s) against a prior "
            f"median of {median:.0f} ({share:.0%}). Something upstream stopped "
            f"supplying rows -- a feed, the line join, or the scratch filter",
            detail,
        )
    return CheckResult(
        name, CHECK_OK,
        f"{newest} recorded against a prior median of {median:.0f} ({share:.0%})",
        detail,
    )


def run_all(
    records: Sequence[Mapping[str, Any]],
    graded_game_dates: Iterable[Any],
    *,
    as_of: date,
) -> list[CheckResult]:
    """Every check, in the order a reader would want them.

    ONE BROKEN CHECK MUST NOT HIDE THE REST, so each is called separately and
    an exception inside one becomes that check's own ALERT rather than
    aborting the sweep. A monitor that dies on its first surprise reports
    nothing about the other four.
    """
    dates = list(graded_game_dates)
    calls = (
        ("a_slate_was_recorded_recently",
         lambda: a_slate_was_recorded_recently(records, as_of=as_of)),
        ("recorded_volume_has_not_collapsed",
         lambda: recorded_volume_has_not_collapsed(records)),
        ("every_component_answered",
         lambda: every_component_answered(records)),
        ("aged_slates_are_graded",
         lambda: aged_slates_are_graded(records, dates, as_of=as_of)),
        ("records_are_chained",
         lambda: records_are_chained(records)),
    )
    out: list[CheckResult] = []
    for name, call in calls:
        try:
            out.append(call())
        except Exception as exc:  # noqa: BLE001 — a check must not kill the sweep
            logger.exception("Self-check %s raised", name)
            out.append(CheckResult(
                name, CHECK_ALERT,
                f"The check itself raised {type(exc).__name__}: {exc}. That is "
                f"a defect in the monitor, and it is reported rather than "
                f"swallowed because a silent monitor is the thing this module "
                f"exists to prevent",
                {"exception": type(exc).__name__},
            ))
    return out


def summarise(results: Sequence[CheckResult]) -> dict[str, Any]:
    """Counts plus the alerting reasons, for a log line and an alert body."""
    by_status: dict[str, int] = {}
    for r in results:
        by_status[r.status] = by_status.get(r.status, 0) + 1
    alerts = [r for r in results if r.alerting]
    return {
        "checks": len(results),
        "by_status": by_status,
        "alerting": [r.name for r in alerts],
        "reasons": [f"{r.name}: {r.reason}" for r in alerts],
        "results": [r.as_dict() for r in results],
    }


__all__ = [
    "CHECK_ALERT", "CHECK_CANNOT_TELL", "CHECK_OK", "CheckResult",
    "DEFAULT_GRACE_DAYS", "DEFAULT_MAX_RECORD_AGE_DAYS",
    "DEFAULT_MAX_UNGRADED_SHARE", "DEFAULT_MIN_VOLUME_SHARE",
    "a_slate_was_recorded_recently", "aged_slates_are_graded",
    "every_component_answered", "records_are_chained",
    "recorded_volume_has_not_collapsed", "run_all", "summarise",
]
