"""Model artifact registry metadata (Wave 2).

Append-only JSONL index next to saved boosters. No secrets.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REGISTRY_FILENAME = "model_registry.jsonl"


def _sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for pkg in ("numpy", "pandas", "sklearn", "xgboost", "catboost", "scipy", "pydantic"):
        try:
            mod = __import__(pkg if pkg != "sklearn" else "sklearn")
            versions[pkg] = str(getattr(mod, "__version__", "unknown"))
        except ImportError:
            versions[pkg] = "NOT_INSTALLED"
    return versions


def build_registry_record(
    *,
    model_name: str,
    model_version: str,
    target_market: str,
    artifact_path: Path | str,
    feature_schema_version: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path = Path(artifact_path)
    record: dict[str, Any] = {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "model_version": model_version,
        "target_market": target_market,
        "feature_schema_version": feature_schema_version,
        "artifact_path": str(path.resolve()) if path.exists() else str(path),
        "artifact_sha256": _sha256_file(path),
        "package_versions": collect_package_versions(),
        "research_only": True,
    }
    if extra:
        # Never allow secret-like keys
        blocked = {"api_key", "password", "token", "secret", "webhook"}
        safe = {k: v for k, v in extra.items() if not any(b in str(k).lower() for b in blocked)}
        record["extra"] = safe
    return record


def append_registry(
    registry_dir: Path | str,
    record: dict[str, Any],
) -> Path:
    """Append one JSON line to ``model_registry.jsonl`` under ``registry_dir``."""
    root = Path(registry_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / REGISTRY_FILENAME
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")
    logger.info("Registered model artifact in %s", path)
    return path


def register_saved_artifact(
    *,
    registry_dir: Path | str,
    model_name: str,
    model_version: str,
    target_market: str,
    artifact_path: Path | str,
    feature_schema_version: str,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = build_registry_record(
        model_name=model_name,
        model_version=model_version,
        target_market=target_market,
        artifact_path=artifact_path,
        feature_schema_version=feature_schema_version,
        extra=meta,
    )
    append_registry(registry_dir, record)
    return record


def read_registry(registry_dir: Path | str) -> list[dict[str, Any]]:
    path = Path(registry_dir) / REGISTRY_FILENAME
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


RUN_MANIFEST_NAME = "run_manifest.json"


def write_run_manifest(
    run_dir: Path | str,
    *,
    run_id: str | None = None,
    steps: list[dict[str, Any]] | None = None,
    output_files: list[Path | str] | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Forward-only run manifest (Wave 4).

    Records step order + output checksums. Does not mutate prior artifacts —
    callers must write new files under ``run_dir`` rather than rewriting history.
    """
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    rid = run_id or datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    file_rows: list[dict[str, Any]] = []
    for f in output_files or []:
        p = Path(f)
        if not p.is_absolute():
            p = root / p
        file_rows.append(
            {
                "path": str(p.resolve()) if p.exists() else str(p),
                "exists": p.exists(),
                "sha256": _sha256_file(p),
                "bytes": p.stat().st_size if p.exists() else None,
            }
        )
    manifest: dict[str, Any] = {
        "run_id": rid,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "forward_only": True,
        "package_versions": collect_package_versions(),
        "steps": list(steps or []),
        "outputs": file_rows,
        "meta": {},
    }
    if meta:
        blocked = {"api_key", "password", "token", "secret", "webhook"}
        manifest["meta"] = {
            k: v for k, v in meta.items() if not any(b in str(k).lower() for b in blocked)
        }
    path = root / RUN_MANIFEST_NAME
    # Prefer stable name; if present, write a run-scoped sibling and leave original.
    if path.exists():
        path = root / f"run_manifest_{rid}.json"
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    manifest["_manifest_path"] = str(path)
    logger.info("Wrote forward-only run manifest %s", path)
    return manifest


def persist_run_manifest(manifest: dict[str, Any], path: Path | str | None = None) -> Path:
    """Write (or rewrite) a run manifest dict to disk."""
    if path is not None:
        p = Path(path)
    elif manifest.get("_manifest_path"):
        p = Path(str(manifest["_manifest_path"]))
    else:
        raise ValueError("DATA_NOT_AVAILABLE: path or manifest[_manifest_path] required")
    p.parent.mkdir(parents=True, exist_ok=True)
    blob = {k: v for k, v in manifest.items() if not str(k).startswith("_")}
    p.write_text(json.dumps(blob, indent=2, default=str), encoding="utf-8")
    return p


def append_run_step(
    manifest: dict[str, Any],
    *,
    step_name: str,
    input_paths: list[Path | str] | None = None,
    output_paths: list[Path | str] | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """Append one immutable step record to an in-memory manifest dict."""
    step = {
        "step_name": step_name,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": [
            {"path": str(p), "sha256": _sha256_file(Path(p))} for p in (input_paths or [])
        ],
        "outputs": [
            {"path": str(p), "sha256": _sha256_file(Path(p))} for p in (output_paths or [])
        ],
        "notes": notes,
    }
    steps = list(manifest.get("steps") or [])
    steps.append(step)
    manifest["steps"] = steps
    return manifest
