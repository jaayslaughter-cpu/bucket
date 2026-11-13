"""Network orchestration for the append-only research archive.

No new odds source, no database writes, no model or inactive-cache mutation.
Exceptions are classified, never serialized (provider errors may contain keys).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

import requests

from src.ingestion.model_input_archive import (
    InputArchive,
    InputError,
    discover_report_urls,
    normalize_odds,
    parse_report_pdf,
    report_time,
    timestamp,
)
from src.ingestion.propline import (
    API_BASE_URL,
    DEFAULT_MARKET_MAP,
    NBA_SPORT_KEY,
    PropLineAuthError,
    PropLineClient,
)
from src.utils.timezones import UTC
from src.utils.volume import resolve_state_root

DEFAULT_MARKETS = [*DEFAULT_MARKET_MAP, "spreads", "totals"]
MAX_PDF_BYTES = 10 * 1024 * 1024
logger = logging.getLogger(__name__)


def _now():
    return datetime.now(UTC)


def current_season_start(at=None):
    at = at or _now()
    return at.year if at.month >= 7 else at.year - 1


def collect_availability(archive: InputArchive, *, season_start: int,
                         session=None, now=None, report_urls=None) -> dict:
    """Discover the latest actually linked report for the requested season.

    The index is NOT a complete historical archive. Backfill uses explicit
    official report URLs and actual collection time via import_report().
    """
    if not 1946 <= season_start <= 2200:
        raise InputError("invalid season start year")
    session = session or requests.Session()
    index = f"https://official.nba.com/nba-injury-report-{season_start}-{(season_start + 1) % 100:02d}-season/"
    now = now or _now
    try:
        if report_urls is None:
            response = session.get(index, timeout=(10, 45), allow_redirects=False)
            response.raise_for_status()
            if response.status_code != 200:
                raise InputError("unexpected index response")
            links = discover_report_urls(response.text, before=now())
        else:
            # Operator-supplied links, never guessed report filenames. Strictly
            # validate even links in the manifest that are not selected.
            links = sorted({url for url in report_urls if report_time(url) <= now()}, key=report_time)
        links = [url for url in links
                 if (season_start, 7, 1) <= (report_time(url).year, report_time(url).month, report_time(url).day)
                 < (season_start + 1, 7, 1)]
        if not links:
            raise InputError("no published report links")
    except Exception as exc:
        at = now()
        path = archive.write(kind="availability", source="nba_official", source_url=index,
                             observed_at=at, collected_at=at, status="FAILED", rows=[],
                             error_type=type(exc).__name__)
        return {"status": "FAILED", "snapshot": str(path), "reason": "REPORT_DISCOVERY_FAILED"}
    result = import_report(archive, links[-1], session=session, now=now)
    age = (now() - report_time(links[-1])).total_seconds()
    result["report_age_seconds"] = round(age)
    result["discovery"] = "manifest" if report_urls is not None else "official_index"
    if age > 6 * 3600 and result["status"] == "OK":
        result.update(status="PARTIAL", reason="STALE_REPORT")
    return result


def import_odds(archive: InputArchive, payload: dict, *, now=None) -> dict:
    """Import an explicit PropLine event/bookmaker/market payload, not guessed history schemas."""
    collected = (now or _now)()
    rows, rejected = normalize_odds(payload, collected_at=collected, max_quote_age_minutes=None)
    status = "OK" if rows and not rejected else "PARTIAL" if rows else "FAILED"
    observed = max((timestamp(r["observed_at_utc"]) for r in rows), default=collected)
    path = archive.write(kind="odds", source="propline", source_url=API_BASE_URL,
                         provider_event_id=str(payload["id"]), observed_at=observed,
                         collected_at=collected, status=status, rows=rows, rejected=rejected,
                         imported=True)
    return {"status": status, "pairs": len(rows), "snapshot": str(path)}


def import_report(archive: InputArchive, url: str, *, content=None, session=None, now=None) -> dict:
    observed = report_time(url)  # validate BEFORE any request
    now = now or _now
    try:
        if content is None:
            response = (session or requests.Session()).get(url, timeout=(10, 45),
                                                           allow_redirects=False, stream=True)
            try:
                response.raise_for_status()
                if response.status_code != 200:
                    raise InputError("unexpected report response")
                data = bytearray()
                for part in response.iter_content(65536):
                    data.extend(part)
                    if len(data) > MAX_PDF_BYTES:
                        raise InputError("report exceeds size limit")
                content = bytes(data)
            finally:
                response.close()
        if len(content) > MAX_PDF_BYTES:
            raise InputError("report exceeds size limit")
        rows = parse_report_pdf(content, expected_time=observed)
        blob = archive.blob(content)
        collected = now()
        path = archive.write(kind="availability", source="nba_official", source_url=url,
                             observed_at=observed, collected_at=collected, status="OK", rows=rows,
                             report_blob=blob)
        return {"status": "OK", "snapshot": str(path), "rows": len(rows), "report_url": url}
    except Exception as exc:
        collected = now()
        # On an unavailable future report, do not lie about publication time.
        at = min(observed, collected)
        path = archive.write(kind="availability", source="nba_official", source_url=url,
                             observed_at=at, collected_at=collected, status="FAILED", rows=[],
                             error_type=type(exc).__name__)
        return {"status": "FAILED", "snapshot": str(path), "reason": "REPORT_DOWNLOAD_OR_PARSE_FAILED"}


def collect_odds(archive: InputArchive, *, client=None, now=None,
                 max_events=12, horizon_hours=36, markets=None) -> dict:
    """Bounded, quota-aware capture using the existing PropLine client.

    Per-event failures stay separate, so one bad event cannot erase a good
    one or masquerade as a complete successful collection.
    """
    if max_events < 1 or horizon_hours <= 0:
        raise InputError("positive event cap and horizon required")
    now = now or _now
    try:
        client = client or PropLineClient()
        events = client.fetch_events()
    except Exception as exc:
        at = now()
        status = "NOT_CONFIGURED" if isinstance(exc, PropLineAuthError) else "FAILED"
        archive.write(kind="odds", source="propline", source_url=API_BASE_URL,
                      observed_at=at, collected_at=at, status=status, rows=[],
                      error_type=type(exc).__name__)
        return {"status": status, "events": 0, "pairs": 0}
    eligible, problems = [], []
    at = now()
    for event in events:
        try:
            # Discovery can omit sport_key on this NBA-only endpoint.
            if event.get("sport_key", NBA_SPORT_KEY) != NBA_SPORT_KEY:
                raise InputError("non-NBA discovery event")
            start = timestamp(event["commence_time"])
            if at < start <= at + timedelta(hours=horizon_hours):
                if not event.get("id"):
                    raise InputError("missing event id")
                eligible.append(event)
        except (InputError, KeyError, TypeError):
            problems.append({"reason": "INVALID_EVENT"})
    eligible.sort(key=lambda e: (timestamp(e["commence_time"]), str(e["id"])))
    if len(eligible) > max_events:
        problems.append({"reason": "EVENT_CAP", "omitted": len(eligible) - max_events})
    results = []
    for event in eligible[:max_events]:
        event_id = str(event["id"])
        try:
            payload = client.fetch_event_props(event_id, markets or DEFAULT_MARKETS)
            # Only fill absent metadata from the same discovered event, never
            # override contradictory provider data.
            payload = {k: event[k] for k in ("id", "sport_key", "commence_time", "home_team", "away_team")
                       if k in event} | payload
            payload.setdefault("sport_key", NBA_SPORT_KEY)
            if str(payload.get("id")) != event_id:
                raise InputError("event ID mismatch")
            collected = now()
            rows, rejected = normalize_odds(payload, collected_at=collected)
            status = "OK" if rows and not rejected else "PARTIAL" if rows else "FAILED"
            observed = max((timestamp(r["observed_at_utc"]) for r in rows), default=collected)
            path = archive.write(kind="odds", source="propline", source_url=API_BASE_URL,
                                 provider_event_id=event_id, observed_at=observed,
                                 collected_at=collected, status=status, rows=rows, rejected=rejected)
            results.append({"status": status, "pairs": len(rows), "snapshot": str(path)})
        except Exception as exc:
            collected = now()
            path = archive.write(kind="odds", source="propline", source_url=API_BASE_URL,
                                 provider_event_id=event_id, observed_at=collected,
                                 collected_at=collected, status="FAILED", rows=[],
                                 error_type=type(exc).__name__)
            results.append({"status": "FAILED", "pairs": 0, "snapshot": str(path)})
    status = "OK" if results and all(r["status"] == "OK" for r in results) and not problems else "PARTIAL"
    if not results:
        status = "FAILED" if problems else "OK"
    finished = now()
    archive.write(kind="odds", source="propline", source_url=API_BASE_URL,
                  observed_at=finished, collected_at=finished, status=status, rows=[],
                  collection_summary=True, eligible_events=len(eligible), problems=problems,
                  event_results=results)
    return {"status": status, "events": len(results), "pairs": sum(r["pairs"] for r in results),
            "problems": problems}


def run_model_input_collection():
    """Scheduled entrypoint; existing volume, no database/model side effects."""
    archive = InputArchive(resolve_state_root()[0] / "model_inputs")
    manifest = os.environ.get("PROPIQ_MODEL_INPUTS_REPORT_MANIFEST", "").strip()
    try:
        urls = json.loads(Path(manifest).read_text()) if manifest else None
        availability = collect_availability(archive, season_start=current_season_start(), report_urls=urls)
    except (OSError, ValueError, TypeError):
        # Still archive a named failure if the operator's manifest is broken.
        at = _now()
        archive.write(kind="availability", source="nba_official", source_url="https://official.nba.com/",
                      observed_at=at, collected_at=at, status="FAILED", rows=[],
                      error_type="InvalidReportManifest")
        availability = {"status": "FAILED", "reason": "INVALID_REPORT_MANIFEST"}
    odds = collect_odds(archive)
    result = {"availability": availability, "odds": odds}
    logger.info("Research model-input collection: %s", json.dumps(result, sort_keys=True))
    return result


def register_model_input_job(scheduler):
    """Disabled by default: enabling a paid API poll must be deliberate."""
    if os.environ.get("PROPIQ_MODEL_INPUTS_ENABLED", "").strip().lower() not in {"1", "true", "yes", "on"}:
        return
    scheduler.add_job(
        run_model_input_collection, trigger="cron", hour="*", minute="*/15", timezone=UTC,
        id="model_inputs", name="Research-only timestamped availability and PropLine archive",
        max_instances=1, coalesce=True, misfire_grace_time=120, replace_existing=True,
    )
