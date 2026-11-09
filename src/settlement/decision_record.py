"""
A hash-chained record of every slate decision, abstentions included.

THE IDEA IS CITED; THE CODE IS OURS. `jamesz08`-style public pick ledgers and,
directly, the `nba-picks-record` repository supplied 2026-11-08 publish one
JSON per poll covering EVERY game on the slate -- bet, no-bet and not-scored
alike -- chained by `prev_sha256` and stamped by an RFC-3161 authority. That
repository carries no licence, so what is taken here is the reasoning.

WHY A SECOND LEDGER WHEN `prop_results` ALREADY EXISTS. `prop_results` holds
only the rows that CLEARED the recorder. On 2026-11-08 the recorder took 0 of
796 rows, so that table says nothing happened on a day the pipeline scored a
full slate and the board abstained 1,194 times. A strike rate or a calibration
figure computed over the rows a model felt confident about has a denominator
nobody outside the process can see. This records the whole slate.

THE DENOMINATOR IS THE POINT, so the taxonomy is the RECORDER'S, not a second
one. Every entry is RECORDED or SKIPPED and carries the recorder's own reason
string verbatim. Inventing coarse buckets here would be a second classification
of one fact, and the two would drift the first time a skip reason changed.

IT PROVES EXISTENCE, NOT PERFORMANCE. A chained record shows which decision
existed at which time. It is not evidence that a decision was good, it is not a
wager, and nothing here prices, sizes or places anything. Stamping the chain
with an external timestamp authority is a separate step; the chain is useful
without it, because a rewritten record breaks every hash after it.

RESEARCH_ONLY.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

#: Directory under the durable state root. Dated subdirectories keep a season's
#: records listable without loading them.
RECORDS_DIRNAME = "decision_records"

DECISIONS: dict[str, str] = {
    "RECORDED": "written to prop_results for forward grading",
    "SKIPPED": "not recorded; `reason` carries the recorder's own words",
}

#: The fields copied out of a recorded row. A WHITELIST, not a blacklist: the
#: recorder's row dict is free to grow, and a record that mirrored it wholesale
#: would publish the next field added to it without anyone deciding to.
RECORDED_FIELDS: tuple[str, ...] = (
    "player_name", "market", "game_date", "nba_game_id",
    "predicted_line", "predicted_side", "prob_over", "model_projection",
    "odds", "source", "is_pickem",
)

_TIMESTAMP_SAFE = re.compile(r"[^0-9A-Za-z:.+-]")


def canonical_bytes(payload: dict[str, Any]) -> bytes:
    """
    The exact bytes that get hashed.

    Sorted keys and fixed separators, because a hash that depended on dict
    insertion order would break the chain on a Python upgrade rather than on a
    tampered record -- which is the one thing a chain must not do.
    """
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        default=str,
    ).encode("utf-8")


def sha256_of(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def decision_rows(
    recorded: Any,
    *,
    shadow: dict[str, dict[tuple[str, str], float]] | None = None,
) -> list[dict[str, Any]]:
    """
    One entry per projection the recorder saw, from the recorder's own report.

    Reads `RecordingReport.rows` and `.skipped` rather than re-deriving the
    decision from the projections frame. The recorder decides; this reports.
    A second classifier here would disagree with it the first time either
    changed, and the disagreement would be invisible.

    ``shadow`` is ``{component: {(player_name, MARKET): probability}}`` for
    components scored ALONGSIDE the serving one. They are attached to
    RECORDED rows only, under a `shadow` key, and they go nowhere else: not
    to `prop_results`, not to the gate, not to a card. When these rows grade,
    the same outcome scores every component that answered -- which is the
    only way to learn which is better at a posted line, since every published
    measurement ranks them at the trailing average instead.
    """
    out: list[dict[str, Any]] = []
    shadow = shadow or {}
    for row in getattr(recorded, "rows", []) or []:
        entry = {"decision": "RECORDED"}
        entry.update({k: row.get(k) for k in RECORDED_FIELDS})
        key = (str(row.get("player_name") or ""), str(row.get("market") or "").upper())
        # A COMPONENT THAT ABSTAINED IS RECORDED AS ABSTAINING. Omitting it
        # would make "did not answer" indistinguishable from "was not asked",
        # and the second reads as a smaller sample rather than a refusal.
        challengers = {c: probs.get(key) for c, probs in shadow.items()}
        if challengers:
            entry["shadow"] = challengers
        out.append(entry)
    for skip in getattr(recorded, "skipped", []) or []:
        out.append({
            "decision": "SKIPPED",
            "player_name": skip.get("player") or skip.get("row"),
            "market": skip.get("market"),
            "reason": skip.get("reason"),
        })
    return out


def build_record(
    *,
    slate_date: str,
    as_of_ts: str,
    rows: list[dict[str, Any]],
    prev_sha256: str | None,
    model_provenance: dict[str, Any] | None = None,
    gate: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    """
    Return ``(record, sha256_hex)``. Pure: no clock, no filesystem, no network.

    ``as_of_ts`` is passed in rather than read here so the same inputs always
    hash to the same digest -- a function that stamped itself could not be
    tested for that, and the chain is only worth something if it can be.
    """
    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "slate_date": str(slate_date),
        "as_of_ts": str(as_of_ts),
        # BOUND TO WHAT ACTUALLY SERVED. A calibration figure that pools rows
        # across artifacts cannot say which model was being measured; this is
        # the artifact identity for the rows in THIS record.
        "model_provenance": model_provenance or {},
        # What the publication gate said at the time, so a withheld slate is
        # distinguishable from a slate with nothing to say.
        "gate": gate or {},
        "decisions": DECISIONS,
        "prev_sha256": prev_sha256,
        "rows": sorted(
            rows,
            key=lambda r: (
                str(r.get("player_name") or ""), str(r.get("market") or ""),
                str(r.get("decision") or ""),
            ),
        ),
        "counts": _counts(rows),
        "note": (
            "Model predictions recorded for forward grading, not wagers. No "
            "stake is written and none can be set here. A record proves a "
            "decision existed at the stated time; it is not evidence of profit."
        ),
    }
    return body, sha256_of(body)


def _counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Totals the reader would otherwise have to recompute to trust the file."""
    by_decision: dict[str, int] = {}
    by_reason: dict[str, int] = {}
    for row in rows:
        decision = str(row.get("decision") or "")
        by_decision[decision] = by_decision.get(decision, 0) + 1
        if decision == "SKIPPED":
            reason = str(row.get("reason") or "unstated")
            by_reason[reason] = by_reason.get(reason, 0) + 1
    return {"total": len(rows), "by_decision": by_decision, "skipped_by_reason": by_reason}


# --------------------------------------------------------------------------- #
# Filesystem. Separated from everything above, which is pure.
# --------------------------------------------------------------------------- #

def records_dir(root: Path | str) -> Path:
    return Path(root) / RECORDS_DIRNAME


def utc_stamp(as_of_ts: str) -> str:
    """
    ``as_of_ts`` as ``YYYYMMDDTHHMMSSZ``, which sorts lexicographically in the
    order it happened.

    THE FILENAME IS HOW THE CHAIN FINDS ITS HEAD, so it has to sort by time.
    A Pacific local stamp does not: on the November DST change the day holds
    both -07:00 and -08:00 offsets, so `...T01:30:00-07:00` sorts before
    `...T01:15:00-08:00` while happening after it, and the chain would graft
    the next record onto the wrong parent for one hour a year. UTC has no such
    hour. The human-readable local stamp stays inside the record.
    """
    raw = str(as_of_ts)
    try:
        moment = datetime.fromisoformat(raw)
    except ValueError:
        return _TIMESTAMP_SAFE.sub("_", raw)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def record_path(root: Path | str, *, slate_date: str, as_of_ts: str) -> Path:
    """``<root>/decision_records/YYYY/MM/DD/<utc stamp>.json``."""
    parts = str(slate_date).split("-")
    if len(parts) != 3:
        raise ValueError(f"slate_date must be YYYY-MM-DD, got {slate_date!r}")
    return records_dir(root).joinpath(*parts, f"{utc_stamp(as_of_ts)}.json")


def latest_sha256(root: Path | str) -> str | None:
    """
    The digest of the newest record, or None when the chain is empty.

    DERIVED BY SCANNING, not read from a pointer file. A `latest` pointer is a
    second copy of "where the chain ends", and the one time it disagreed with
    the files nobody would know which to believe. Filenames are ISO timestamps
    under YYYY/MM/DD, so the lexicographic maximum is the newest.
    """
    base = records_dir(root)
    if not base.exists():
        return None
    files = sorted(base.rglob("*.json"))
    if not files:
        return None
    newest = files[-1]
    try:
        body = json.loads(newest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning(
            "DECISION CHAIN: the newest record %s could not be read (%s), so "
            "this slate's record will start a NEW chain. The break is visible "
            "in the file rather than papered over.", newest, exc,
        )
        return None
    # Re-hashed from the content rather than trusting a stored digest, so a
    # record edited in place cannot hand us the hash it used to have.
    return sha256_of(body)


def write_record(root: Path | str, body: dict[str, Any]) -> Path:
    """
    Write one record. Atomic: a temp file in the same directory, then replace.

    A half-written record is worse than a missing one -- it breaks the chain at
    a point nobody chose.
    """
    path = record_path(root, slate_date=body["slate_date"], as_of_ts=body["as_of_ts"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_bytes(canonical_bytes(body))
    tmp.replace(path)
    return path


def artifact_provenance(artifacts: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """
    ``{market: {"artifact": basename, "sha256": hex, "bytes": n}}``.

    THE BASENAME, NEVER THE PATH. These records are meant to be publishable,
    and a full path publishes the container's layout for no benefit -- the
    digest is what identifies the generation, not where it happened to sit.

    An artifact that cannot be read is reported as such rather than omitted:
    a market silently missing from provenance reads as "not scored", which is
    a different and much more comforting claim than "scored by something we
    could not identify".
    """
    out: dict[str, dict[str, Any]] = {}
    for market, raw in sorted((artifacts or {}).items()):
        if not raw:
            out[str(market)] = {"artifact": None, "sha256": None, "error": "no artifact recorded"}
            continue
        path = Path(str(raw))
        try:
            blob = path.read_bytes()
        except OSError as exc:
            out[str(market)] = {
                "artifact": path.name, "sha256": None, "error": f"unreadable: {exc.strerror}",
            }
            continue
        out[str(market)] = {
            "artifact": path.name,
            "sha256": hashlib.sha256(blob).hexdigest(),
            "bytes": len(blob),
        }
    return out
