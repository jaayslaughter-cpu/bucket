"""Injected network and scheduling tests. No credentials or paid requests."""

import json
from datetime import datetime, timedelta

import pytest

from scripts.collect_model_inputs import current_season_start, export_as_of, main
from src.ingestion.model_input_archive import InputArchive, InputError
from src.ingestion.model_input_collection import (
    collect_availability,
    collect_odds,
    import_odds,
    register_model_input_job,
)
from src.utils.timezones import UTC

AT = datetime(2026, 11, 8, 20, 0, tzinfo=UTC)
REPORT = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_2026-11-08_03_00PM.pdf"


def test_current_season_not_last_year():
    assert current_season_start(AT) == 2026
    assert current_season_start(AT.replace(month=1)) == 2025


def test_missing_propline_credential_is_named(tmp_path, monkeypatch):
    monkeypatch.delenv("PROPLINE_API_KEY", raising=False)
    result = collect_odds(InputArchive(tmp_path), now=lambda: AT)
    assert result["status"] == "NOT_CONFIGURED"
    assert result["pairs"] == 0
    assert "NOT_CONFIGURED" in next(tmp_path.glob("snapshots/odds/*.json")).read_text()


def test_report_discovery_failure_is_archived_without_error_body(tmp_path):
    class BrokenSession:
        def get(self, *a, **kw):
            raise RuntimeError("sensitive response body")
    result = collect_availability(InputArchive(tmp_path), season_start=2026,
                                  session=BrokenSession(), now=lambda: AT)
    assert result["status"] == "FAILED"
    content = next(tmp_path.glob("snapshots/availability/*.json")).read_text()
    assert "sensitive" not in content
    assert "RuntimeError" in content


def test_manifest_uses_actual_validated_links_and_latest_eligible_only(tmp_path, monkeypatch):
    seen = []
    def fake_import(archive, url, **kwargs):
        seen.append(url)
        return {"status": "OK"}
    monkeypatch.setattr("src.ingestion.model_input_collection.import_report", fake_import)
    result = collect_availability(InputArchive(tmp_path), season_start=2026,
                                  report_urls=[REPORT, REPORT.replace("03_00PM", "04_00PM")],
                                  now=lambda: AT)
    assert seen == [REPORT]
    assert result["discovery"] == "manifest"


def test_old_manifest_report_is_explicitly_stale(tmp_path, monkeypatch):
    monkeypatch.setattr("src.ingestion.model_input_collection.import_report",
                        lambda *a, **kw: {"status": "OK"})
    result = collect_availability(InputArchive(tmp_path), season_start=2026,
                                  report_urls=[REPORT], now=lambda: AT + timedelta(days=1))
    assert result["status"] == "PARTIAL"
    assert result["reason"] == "STALE_REPORT"


def test_other_season_manifest_cannot_be_todays_data(tmp_path):
    result = collect_availability(InputArchive(tmp_path), season_start=2026,
                                  report_urls=[REPORT.replace("2026-11", "2025-11")],
                                  now=lambda: AT)
    assert result["status"] == "FAILED"


def test_no_cross_sport_spend_and_per_event_failures_remain_named(tmp_path):
    class Client:
        calls = []
        def fetch_events(self):
            return [
                {"id": "bad-sport", "sport_key": "baseball_mlb", "commence_time": (AT + timedelta(hours=2)).isoformat()},
                {"id": "live", "commence_time": (AT - timedelta(minutes=1)).isoformat()},
                {"id": "upcoming", "commence_time": (AT + timedelta(hours=2)).isoformat()},
            ]
        def fetch_event_props(self, event_id, markets):
            self.calls.append(event_id)
            raise RuntimeError("do not archive provider errors with credentials")
    client = Client()
    archive = InputArchive(tmp_path)
    result = collect_odds(archive, client=client, now=lambda: AT)
    assert client.calls == ["upcoming"]
    assert result["status"] == "PARTIAL"
    assert any(s["status"] == "FAILED" for s in archive.as_of("odds", AT))
    assert all("credentials" not in p.read_text() for p in tmp_path.rglob("*.json"))


def test_empty_export_is_unknown_and_not_training_ready(tmp_path):
    result = export_as_of(InputArchive(tmp_path), AT)
    assert result["coverage"]["report_status"] == "MISSING_OR_STALE"
    assert result["coverage"]["training_ready"] is False
    assert result["resolved_availability"] == []
    assert result["unresolved_availability"]


def test_cli_export_is_exclusive_and_handles_naive_time(tmp_path):
    output = tmp_path / "export.json"
    assert main(["--root", str(tmp_path), "export", "--cutoff", AT.isoformat(),
                 "--output", str(output)]) == 0
    saved = output.read_bytes()
    assert main(["--root", str(tmp_path), "export", "--cutoff", AT.isoformat(),
                 "--output", str(output)]) == 1
    assert output.read_bytes() == saved
    assert main(["--root", str(tmp_path), "export", "--cutoff", "2026-11-08T20:00:00",
                 "--output", str(tmp_path / "naive.json")]) == 1


def test_explicit_odds_import_refuses_other_provider_shapes(tmp_path):
    with pytest.raises(InputError):
        import_odds(InputArchive(tmp_path), {"snapshots": []}, now=lambda: AT)


def test_scheduler_is_opt_in_coalesced_and_never_below_entrypoint(monkeypatch):
    class Scheduler:
        jobs = []
        def add_job(self, func, **kwargs):
            self.jobs.append((func, kwargs))
    scheduler = Scheduler()
    monkeypatch.delenv("PROPIQ_MODEL_INPUTS_ENABLED", raising=False)
    register_model_input_job(scheduler)
    assert not scheduler.jobs
    monkeypatch.setenv("PROPIQ_MODEL_INPUTS_ENABLED", "1")
    register_model_input_job(scheduler)
    func, options = scheduler.jobs[0]
    assert callable(func)
    assert options["id"] == "model_inputs"
    assert options["max_instances"] == 1 and options["coalesce"]
    assert options["minute"] == "*/15"


def test_scheduled_job_uses_existing_volume_and_does_not_call_models(tmp_path, monkeypatch):
    from src.ingestion.model_input_collection import run_model_input_collection
    calls = []
    monkeypatch.setenv("PROPIQ_STATE_DIR", str(tmp_path))
    monkeypatch.setattr("src.ingestion.model_input_collection.collect_availability",
                        lambda archive, **kw: calls.append(archive.root) or {"status": "OK"})
    monkeypatch.setattr("src.ingestion.model_input_collection.collect_odds",
                        lambda archive, **kw: {"status": "NOT_CONFIGURED"})
    result = run_model_input_collection()
    assert calls == [tmp_path / "model_inputs"]
    assert result["odds"]["status"] == "NOT_CONFIGURED"


def test_committed_seed_manifest_is_only_real_official_links():
    from pathlib import Path

    from src.ingestion.model_input_archive import report_time
    links = json.loads(Path("config/nba_report_urls.example.json").read_text())
    assert links and all(report_time(url).year == 2026 for url in links)


def test_actual_worker_registers_optional_capture(monkeypatch):
    import scheduler_worker

    class Scheduler:
        def __init__(self):
            self.jobs = []
        def add_job(self, func, **kwargs):
            self.jobs.append(kwargs["id"])

    monkeypatch.setenv("PROPIQ_MODEL_INPUTS_ENABLED", "true")
    scheduler = Scheduler()
    scheduler_worker.build_scheduler(scheduler)
    assert scheduler.jobs.count("model_inputs") == 1
