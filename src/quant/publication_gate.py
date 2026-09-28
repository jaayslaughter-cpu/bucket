"""Whether a model-sourced number may be PUBLISHED. RESEARCH_ONLY.

This is the gate between computing a figure and sending it somewhere a person
will read it — Discord, an export, a report. It answers one question: has this
model been shown calibrated on evidence recent and dense enough to stand behind?

WHY PUBLICATION IS THE RIGHT PLACE FOR IT. Computing EV on an uncalibrated model
is fine; it is a diagnostic. Publishing it is not, because a number that reaches
a person gets acted on, and the standing rule in this repository is that nothing
claims profitability until it is verified on leakage-safe forward data. Without a
gate, "the model says 58%" and "the model has been shown to be right 58% of the
time" look identical on the way out.

WHAT THE GATE DOES NOT COVER, and this is the substantive distinction:

  SHARP_BENCHMARK probabilities are the MARKET's, de-vigged from a two-way quote
  on the same contract. They do not rest on this model's calibration at all, so
  the model's ECE is not evidence for or against them and withholding them for a
  missing model backtest would be withholding on an irrelevance. They publish,
  carrying their own caveats (the benchmark must be sharp, the line must match).

  MODEL and MIXED probabilities do rest on it. A mixed entry is gated like a
  model one: every leg has to land, so one model leg puts the model's error on
  the whole card.

  UNSPECIFIED is gated like MODEL. An unrecorded source is not evidence of a
  market-grounded one.

UNKNOWN IS NOT EVIDENCE, which is the rule every refusal below follows. Calibration
evidence that is absent, too sparse to bin, too small to mean anything, or
undated is treated the same as evidence of miscalibration: it does not authorise
publication. In particular an UNDATED report is refused rather than assumed
current — a good ECE from a season ago says nothing about a model serving today's
slate, and the difference between "measured last week" and "measured at some
point" is exactly what a reader cannot recover from the number.

THE THRESHOLD IS A CHOICE, NOT A LAW. ``DEFAULT_MAX_ECE`` of 0.05 means the
model's stated probabilities are off by five percentage points on average across
the reliability bins. That is lenient for a market whose edges are measured in
two or three points, and it is set here as a floor on gross miscalibration
rather than a standard of sharpness. Tighten it per caller; it is an argument.

NOTHING HERE PLACES OR SIZES A WAGER, and passing this gate is not a claim of
profit. It is the absence of one specific disqualification.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Mapping

from src.quant.dfs_payouts import ProbabilitySource
from src.utils.timezones import now_utc, to_utc

logger = logging.getLogger(__name__)

PUBLISH_ALLOWED = "PUBLISH_ALLOWED"
PUBLISH_WITHHELD = "PUBLISH_WITHHELD"

# Mean absolute gap between stated and observed frequency, weighted by bin
# population. See the module docstring on why this is a floor, not a standard.
DEFAULT_MAX_ECE = 0.05

# Below this many graded predictions the ECE is noise. At 50 graded bets a true
# 0.03 ECE and a true 0.08 one are not distinguishable, so a passing number here
# would be luck rather than evidence.
DEFAULT_MIN_SCORED = 100

# Calibration drifts with rotations, role changes and rule enforcement. A report
# older than this is stale evidence about a different regime.
DEFAULT_MAX_AGE_DAYS = 45

# The sources whose numbers rest on THIS model's calibration.
MODEL_DEPENDENT_SOURCES = frozenset({
    ProbabilitySource.MODEL,
    ProbabilitySource.MIXED,
    ProbabilitySource.UNSPECIFIED,
})

PUBLICATION_DISCLAIMER = (
    "RESEARCH_ONLY — passing this gate means the model was not shown grossly "
    "miscalibrated on recent graded results. It is not a profitability claim, "
    "not a recommendation, and not a stake size."
)


@dataclass(frozen=True)
class PublicationVerdict:
    """May this be published, and on what evidence."""

    status: str
    probability_source: ProbabilitySource
    reason: str | None = None
    ece: float | None = None
    max_ece: float = DEFAULT_MAX_ECE
    n_scored: int | None = None
    evidence_as_of: str | None = None
    evidence_age_days: float | None = None
    checks: tuple[str, ...] = field(default_factory=tuple)
    disclaimer: str = PUBLICATION_DISCLAIMER

    @property
    def allowed(self) -> bool:
        return self.status == PUBLISH_ALLOWED

    def as_dict(self) -> dict[str, Any]:
        return {
            "PUBLICATION_STATUS": self.status,
            "PROBABILITY_SOURCE": self.probability_source.value,
            "REASON": self.reason,
            "ECE": self.ece,
            "MAX_ECE": self.max_ece,
            "N_SCORED": self.n_scored,
            "EVIDENCE_AS_OF": self.evidence_as_of,
            "EVIDENCE_AGE_DAYS": (
                round(self.evidence_age_days, 2)
                if self.evidence_age_days is not None else None
            ),
            "CHECKS_PASSED": list(self.checks),
            "DISCLAIMER": self.disclaimer,
        }


def _parse_timestamp(value: Any) -> datetime | None:
    """Parse an ISO-8601 report timestamp to UTC, or None if it is not one."""
    if isinstance(value, datetime):
        return to_utc(value)
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    # A naive string is read as UTC. `report_timestamp_pt` is written by
    # format_pacific_iso and carries its offset, so this only bites on a
    # hand-written value, where the seven or eight hours it costs cannot move a
    # verdict measured in days.
    # A trailing Z is valid ISO-8601 and datetime.fromisoformat accepts it from
    # Python 3.11, which is this project's floor. Normalised anyway so the
    # function does not quietly change behaviour on an older interpreter.
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return to_utc(parsed)


def calibration_gate(
    report: Mapping[str, Any] | None,
    *,
    probability_source: ProbabilitySource = ProbabilitySource.MODEL,
    max_ece: float = DEFAULT_MAX_ECE,
    min_scored: int = DEFAULT_MIN_SCORED,
    max_age_days: float = DEFAULT_MAX_AGE_DAYS,
    now: datetime | None = None,
) -> PublicationVerdict:
    """
    Decide whether a figure from ``probability_source`` may be published.

    ``report`` is a ``paper_calibration.paper_reliability_report``-shaped mapping
    — ``status``, ``n_scored``, ``ece``, ``ece_gate_passed``, and a
    ``report_timestamp_pt``. A raw ``prob_calibration.expected_calibration_error``
    result works too, since the keys it shares are the ones read here; it carries
    no timestamp, so pass ``evidence_as_of`` in the mapping or expect a refusal
    on the staleness check.

    Every refusal names which check failed and what the number was, because
    "withheld" with no figure is indistinguishable from "the pipeline broke".
    """
    source = ProbabilitySource(probability_source)
    checks: list[str] = []

    if source not in MODEL_DEPENDENT_SOURCES:
        return PublicationVerdict(
            status=PUBLISH_ALLOWED,
            probability_source=source,
            reason=(
                "Probabilities came from a de-vigged two-way sharp benchmark, so "
                "they do not rest on this model's calibration and the model's "
                "backtest is not evidence about them. The benchmark being sharp "
                "and the line matching exactly are the assumptions that remain."
            ),
            max_ece=float(max_ece),
            checks=("source_is_market_grounded",),
        )

    if not report:
        return PublicationVerdict(
            status=PUBLISH_WITHHELD,
            probability_source=source,
            reason=(
                "No calibration evidence supplied. A model-sourced probability "
                "with no record of having been right is not publishable — "
                "absent evidence is not evidence of calibration."
            ),
            max_ece=float(max_ece),
        )

    status = str(report.get("status") or "")
    if status and status != "OK":
        return PublicationVerdict(
            status=PUBLISH_WITHHELD,
            probability_source=source,
            reason=(
                f"Calibration report did not produce a figure ({status}): "
                f"{report.get('reason') or 'no reason given'}"
            ),
            max_ece=float(max_ece),
        )
    checks.append("evidence_present")

    n_scored = report.get("n_scored")
    if n_scored is None:
        n_scored = report.get("n_predictions")
    try:
        n_scored = int(n_scored) if n_scored is not None else None
    except (TypeError, ValueError):
        n_scored = None
    if n_scored is None:
        return PublicationVerdict(
            status=PUBLISH_WITHHELD,
            probability_source=source,
            reason=(
                "Calibration report does not say how many graded predictions it "
                "is built on, so the ECE cannot be told from noise."
            ),
            max_ece=float(max_ece), checks=tuple(checks),
        )
    if n_scored < int(min_scored):
        return PublicationVerdict(
            status=PUBLISH_WITHHELD,
            probability_source=source,
            reason=(
                f"Calibration measured on {n_scored} graded prediction(s), below "
                f"the {int(min_scored)} this gate requires. An ECE from a sample "
                "this small is luck, in whichever direction it happens to fall."
            ),
            n_scored=n_scored, max_ece=float(max_ece), checks=tuple(checks),
        )
    checks.append("sample_size")

    # `ece` is None precisely when prob_calibration's own bin-coverage gate
    # failed, and `ece_ungated` is the number it refused to stand behind. Reading
    # the ungated one here would defeat that gate from the outside.
    gate_passed = report.get("ece_gate_passed", report.get("gate_passed"))
    ece = report.get("ece")
    if gate_passed is False or ece is None:
        return PublicationVerdict(
            status=PUBLISH_WITHHELD,
            probability_source=source,
            reason=(
                "The reliability diagram is too sparse for an ECE "
                f"(bin coverage {report.get('bin_coverage')!r}). The ungated "
                f"figure is {report.get('ece_ungated')!r} and is deliberately "
                "not used: a diagram covering a few bins says nothing about the "
                "probabilities that fall outside them."
            ),
            n_scored=n_scored, max_ece=float(max_ece), checks=tuple(checks),
        )
    try:
        ece_value = float(ece)
    except (TypeError, ValueError):
        ece_value = float("nan")
    if not math.isfinite(ece_value):
        return PublicationVerdict(
            status=PUBLISH_WITHHELD,
            probability_source=source,
            reason=f"Calibration error is not a number ({ece!r})",
            n_scored=n_scored, max_ece=float(max_ece), checks=tuple(checks),
        )
    checks.append("ece_present")

    if ece_value > float(max_ece):
        return PublicationVerdict(
            status=PUBLISH_WITHHELD,
            probability_source=source,
            reason=(
                f"Expected calibration error {ece_value:.4f} exceeds the "
                f"{float(max_ece):.4f} allowed. The model's stated probabilities "
                "are off by more than that on average, which is larger than the "
                "edges this pipeline looks for."
            ),
            ece=ece_value, n_scored=n_scored, max_ece=float(max_ece),
            checks=tuple(checks),
        )
    checks.append("ece_within_threshold")

    stamp_raw = (
        report.get("evidence_as_of")
        or report.get("report_timestamp_pt")
        or report.get("report_timestamp")
    )
    stamp = _parse_timestamp(stamp_raw)
    if stamp is None:
        return PublicationVerdict(
            status=PUBLISH_WITHHELD,
            probability_source=source,
            reason=(
                f"Calibration evidence carries no usable timestamp ({stamp_raw!r}), "
                "so its age cannot be established. An undated good ECE could have "
                "been measured on a different season's rotations."
            ),
            ece=ece_value, n_scored=n_scored, max_ece=float(max_ece),
            checks=tuple(checks),
        )

    reference = to_utc(now) if now is not None else now_utc()
    age = reference - stamp
    age_days = age / timedelta(days=1)
    stamp_iso = stamp.isoformat()

    if age_days > float(max_age_days):
        return PublicationVerdict(
            status=PUBLISH_WITHHELD,
            probability_source=source,
            reason=(
                f"Calibration evidence is {age_days:.1f} days old, past the "
                f"{float(max_age_days):.0f}-day limit. Calibration drifts with "
                "rotations and role changes, so this describes a different regime."
            ),
            ece=ece_value, n_scored=n_scored, max_ece=float(max_ece),
            evidence_as_of=stamp_iso, evidence_age_days=age_days,
            checks=tuple(checks),
        )
    # A report dated in the future is a clock or parsing fault, not fresh
    # evidence, and letting it through would make the staleness check trivially
    # bypassable by a wrong timestamp.
    if age_days < 0.0:
        return PublicationVerdict(
            status=PUBLISH_WITHHELD,
            probability_source=source,
            reason=(
                f"Calibration evidence is dated {abs(age_days):.1f} days in the "
                "future, which is a clock or parsing fault rather than fresh "
                "evidence."
            ),
            ece=ece_value, n_scored=n_scored, max_ece=float(max_ece),
            evidence_as_of=stamp_iso, evidence_age_days=age_days,
            checks=tuple(checks),
        )
    checks.append("evidence_is_current")

    return PublicationVerdict(
        status=PUBLISH_ALLOWED,
        probability_source=source,
        reason=(
            f"Calibration error {ece_value:.4f} on {n_scored} graded prediction(s) "
            f"from {age_days:.1f} day(s) ago, within the {float(max_ece):.4f} "
            "allowed. Not a profitability claim."
        ),
        ece=ece_value, n_scored=n_scored, max_ece=float(max_ece),
        evidence_as_of=stamp_iso, evidence_age_days=age_days,
        checks=tuple(checks),
    )
