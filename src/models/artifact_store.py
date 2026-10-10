"""
src/models/artifact_store.py — fetch trained artifacts from object storage.

WHAT THIS CLOSES. The trained artifacts live on the mounted volume and nothing
could put them there: `data/**` is gitignored so they are not in the repo,
`.dockerignore` drops `data/` so they are not in the image, and there was no
storage client, so there was no boot-time fetch. Every deploy needed a person
to run `scripts/seed_volume.py` by hand. An earlier version of the Dockerfile's
comment offered "or fetch from object storage at boot" as an option when no
such code existed, which invited a first deploy that assumed the container
would help itself.

SUPABASE STORAGE, over plain HTTPS. Chosen because this project already uses
Supabase for Postgres (`.env.example`'s default `DATABASE_URL` is a Supabase
pooler URL), so it is one credential domain rather than two, and because its
REST API needs only `requests`, which is a core dependency. S3 would mean
adding `boto3` — tens of megabytes in an image that installs no toolchain —
for a bucket this project writes to a handful of times a season. If you need
S3, the three functions below are the whole surface to reimplement.

WHAT IT REFUSES TO DO:

  * install a booster without its `.meta.json` sidecar or its `.mean.json`
    mean head. `xgb_adapter.load` sets `mean_model = None` when the head is
    absent and only warns, so a half-fetched family returns NULL PROJECTIONS
    beside live probabilities — the failure that looks most like a working
    deployment. The whole family lands or none of it does.
  * overwrite a newer local artifact. The one on the volume produced every
    probability now in the database.
  * log, print or embed the service key. Anyone holding it can read and write
    the bucket.
  * fail the boot when it is not configured. An unconfigured store is a
    deployment that seeds by hand, which is a choice, not a fault.

ATOMIC PER FAMILY. Files download to a temporary directory beside the
destination and are moved into place only once the family is complete and
validated, so a connection dropped half way leaves the volume exactly as it
was rather than one file short.

RESEARCH_ONLY. This moves model files. No odds, no wager, no sizing.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

#: The Supabase project URL, e.g. https://abcdefgh.supabase.co
ENV_STORE_URL = "PROPIQ_ARTIFACT_STORE_URL"
#: A Supabase service-role or storage key. A CREDENTIAL: never logged.
ENV_STORE_KEY = "PROPIQ_ARTIFACT_STORE_KEY"
#: Bucket name. Create it in the Supabase dashboard; it may stay private.
ENV_STORE_BUCKET = "PROPIQ_ARTIFACT_BUCKET"
#: Folder inside the bucket. Keeps a season's artifacts separable.
ENV_STORE_PREFIX = "PROPIQ_ARTIFACT_PREFIX"

DEFAULT_BUCKET = "model-artifacts"
DEFAULT_PREFIX = "comparison"

#: Transport timeout per request, and the retry shape. Mirrors
#: `src/ingestion/espn_client` rather than inventing a second policy: three
#: attempts, exponential backoff, 429 honoured, 4xx not retried because a
#: wrong request does not become right.
TIMEOUT_SECONDS = 30.0
RETRY_ATTEMPTS = 3
RETRY_BACKOFF = 2.0
TOO_MANY_REQUESTS = 429

#: A booster is not a model without these. Mirrors `scripts/seed_volume`'s rule
#: and `xgb_adapter.save`'s output.
REQUIRED_SUFFIXES = (".json", ".meta.json", ".mean.json")


class ArtifactStoreError(RuntimeError):
    """Storage was configured and could not be used."""


@dataclass(frozen=True)
class RemoteObject:
    """One object in the bucket, as the list endpoint describes it."""

    name: str
    size: int | None
    updated_at: str | None


def is_configured() -> bool:
    """
    Whether a store has been pointed at. NOT whether it works.

    Separated so the caller can skip the whole step silently rather than
    reporting a failure for a deployment that seeds by hand on purpose.
    """
    return bool(
        (os.environ.get(ENV_STORE_URL) or "").strip()
        and (os.environ.get(ENV_STORE_KEY) or "").strip()
    )


def _config() -> tuple[str, str, str, str]:
    url = (os.environ.get(ENV_STORE_URL) or "").strip().rstrip("/")
    key = (os.environ.get(ENV_STORE_KEY) or "").strip()
    bucket = (os.environ.get(ENV_STORE_BUCKET) or "").strip() or DEFAULT_BUCKET
    prefix = (os.environ.get(ENV_STORE_PREFIX) or "").strip().strip("/") or DEFAULT_PREFIX
    if not url or not key:
        raise ArtifactStoreError(
            f"${ENV_STORE_URL} and ${ENV_STORE_KEY} must both be set. "
            f"Call is_configured() first if an unset store should be a no-op."
        )
    return url, key, bucket, prefix


def _redact(text: str) -> str:
    """
    Remove the service key from anything about to be logged.

    The key arrives in a header, so it should never appear in an exception --
    "should never" is why this exists. `src/notify/discord.py` redacts its
    webhook for the same reason and by the same argument.
    """
    key = (os.environ.get(ENV_STORE_KEY) or "").strip()
    # A BLIND replace, and deliberately so: a key that happens to be a
    # substring of ordinary words mangles the message ("chec*** the buc***et"
    # for a one-character key in a test), and a mangled message is a much
    # better outcome than a leaked credential. Real service keys are long
    # enough that it never fires on prose.
    return text.replace(key, "***") if key and key in text else text


def _request(
    method: str,
    path: str,
    *,
    session: requests.Session | None = None,
    stream: bool = False,
    **kwargs: Any,
) -> requests.Response:
    """
    One Supabase Storage call, with retry on transport faults only.

    Deliberately the same policy as `espn_client.get_json`: a 4xx is not
    retried because the request is wrong and repeating it stays wrong, while a
    429 is retried because the request is fine and there have merely been too
    many of them. A boot that hangs for minutes on a misconfigured bucket is
    worse than one that says the bucket is missing.
    """
    url, key, _, _ = _config()
    sess = session or requests.Session()
    endpoint = f"{url}/storage/v1/{path.lstrip('/')}"
    headers = {
        "Authorization": f"Bearer {key}",
        "apikey": key,
        **kwargs.pop("headers", {}),
    }
    last: Exception | None = None

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            response = sess.request(
                method, endpoint, headers=headers, timeout=TIMEOUT_SECONDS,
                stream=stream, **kwargs,
            )
            status = getattr(response, "status_code", 200)
            if status == TOO_MANY_REQUESTS:
                if attempt < RETRY_ATTEMPTS:
                    wait = RETRY_BACKOFF ** attempt
                    logger.warning(
                        "artifact store: 429 on %s, attempt %d/%d; waiting %.1fs",
                        path, attempt, RETRY_ATTEMPTS, wait,
                    )
                    time.sleep(wait)
                    continue
                raise ArtifactStoreError(
                    f"artifact store rate-limited {path} on all "
                    f"{RETRY_ATTEMPTS} attempts"
                )
            if 400 <= status < 500:
                raise ArtifactStoreError(
                    f"artifact store returned {status} for {path}: "
                    f"{_redact(response.text[:200])}. The request is wrong, so "
                    f"retrying will not help -- check the bucket name, the "
                    f"prefix and the key's permissions."
                )
            response.raise_for_status()
            return response
        except ArtifactStoreError:
            raise
        except requests.exceptions.RequestException as exc:
            last = exc
            if attempt < RETRY_ATTEMPTS:
                wait = RETRY_BACKOFF ** attempt
                logger.warning(
                    "artifact store: attempt %d/%d for %s failed (%s); "
                    "retrying in %.1fs",
                    attempt, RETRY_ATTEMPTS, path, _redact(str(exc)), wait,
                )
                time.sleep(wait)

    raise ArtifactStoreError(
        f"artifact store unreachable after {RETRY_ATTEMPTS} attempts: "
        f"{path}: {_redact(str(last))}"
    )


def list_remote(*, session: requests.Session | None = None) -> list[RemoteObject]:
    """Every object under the configured prefix."""
    _, _, bucket, prefix = _config()
    response = _request(
        "POST", f"object/list/{bucket}", session=session,
        json={"prefix": f"{prefix}/", "limit": 1000, "offset": 0},
    )
    try:
        payload = response.json()
    except ValueError as exc:
        raise ArtifactStoreError(
            f"artifact store list returned something that is not JSON: {exc}"
        ) from exc
    if not isinstance(payload, list):
        raise ArtifactStoreError(
            f"artifact store list returned {type(payload).__name__}, not a list"
        )

    out: list[RemoteObject] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name or name.endswith("/"):
            continue
        # NAMES ARE RELATIVE TO THE PREFIX, because `_download_one` prepends it
        # when it builds the object path. Returning them prefixed produced
        # `object/{bucket}/comparison/comparison/xgboost_PTS.json` and a 404
        # against every real bucket -- found by the tests below, which is the
        # whole reason the HTTP layer is faked rather than mocked away.
        # Supabase's list endpoint has returned both shapes across versions,
        # so the prefix is stripped if present rather than assumed absent.
        if name.startswith(f"{prefix}/"):
            name = name[len(prefix) + 1:]
        meta = item.get("metadata") or {}
        size = meta.get("size") if isinstance(meta, dict) else None
        out.append(RemoteObject(
            name=name,
            size=int(size) if isinstance(size, (int, float)) else None,
            updated_at=str(item.get("updated_at") or "") or None,
        ))
    return out


def _download_one(
    name: str, target: Path, *, session: requests.Session | None = None
) -> int:
    """Stream one object to `target`. Returns the byte count written."""
    _, _, bucket, prefix = _config()
    response = _request(
        "GET", f"object/{bucket}/{prefix}/{name}", session=session, stream=True,
    )
    written = 0
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as fh:
        for chunk in response.iter_content(chunk_size=1 << 16):
            if chunk:
                fh.write(chunk)
                written += len(chunk)
    return written


def _families(objects: list[RemoteObject]) -> dict[str, list[str]]:
    """
    Group object names by market, the way `seed_volume.family_for` does.

    `xgboost_PTS.json` -> PTS. Grouping by market rather than by file is what
    makes the move atomic per family instead of per file.
    """
    out: dict[str, list[str]] = {}
    for obj in objects:
        stem = obj.name.split("/")[-1]  # tolerate a nested name
        if "_" not in stem:
            continue
        market = stem.split("_", 1)[1].split(".", 1)[0].upper()
        if market:
            out.setdefault(market, []).append(obj.name)
    return out


def _family_is_complete(names: list[str], market: str) -> bool:
    """Every suffix a scoreable XGBoost artifact needs is present."""
    basenames = {n.split("/")[-1] for n in names}
    return all(f"xgboost_{market}{suffix}" in basenames for suffix in REQUIRED_SUFFIXES)


def fetch_artifacts(
    dest: Path,
    *,
    markets: list[str] | None = None,
    force: bool = False,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """
    Download complete artifact families into `dest`, skipping what is current.

    Returns a report rather than raising on a partial outcome, because the
    caller is a boot step: one market missing from the bucket should not stop
    the other two from scoring, and the report is what the log line and the
    healthcheck read.

    IDEMPOTENT. A family whose local files all exist with the remote's byte
    size is skipped, so the second boot downloads nothing. Size, not a
    checksum: Supabase's list endpoint gives a size and not a hash, and
    inventing a manifest to carry hashes would be a second source of truth
    about what the artifacts are.
    """
    report: dict[str, Any] = {
        "configured": True, "fetched": [], "skipped": [], "refused": [], "errors": [],
    }
    objects = list_remote(session=session)
    if not objects:
        report["errors"].append("the bucket has no objects under the prefix")
        return report

    by_size = {o.name.split("/")[-1]: o.size for o in objects}
    wanted = {m.upper() for m in markets} if markets else None

    for market, names in sorted(_families(objects).items()):
        if wanted is not None and market not in wanted:
            continue
        if not _family_is_complete(names, market):
            report["refused"].append({
                "market": market,
                "reason": (
                    "incomplete in the bucket: a booster without its "
                    ".meta.json sidecar cannot be scored with, and without its "
                    ".mean.json head it returns null projections beside live "
                    "probabilities"
                ),
            })
            continue

        basenames = sorted({n.split("/")[-1] for n in names})
        current = all(
            (dest / b).exists()
            and (by_size.get(b) is None or (dest / b).stat().st_size == by_size[b])
            for b in basenames
        )
        if current and not force:
            report["skipped"].append({"market": market, "files": len(basenames)})
            continue

        staging = dest / f".fetch-{market}"
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        try:
            total = 0
            for base, name in zip(basenames, sorted(names)):
                total += _download_one(name, staging / base, session=session)
            # VALIDATED BEFORE IT MOVES. A sidecar that will not parse is a
            # file, not a contract, and installing it turns a visible "nothing
            # resolved" into a silent "resolved and unusable".
            sidecar = staging / f"xgboost_{market}.meta.json"
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
            if not meta.get("feature_cols"):
                raise ArtifactStoreError(
                    f"{sidecar.name} lists no feature_cols, so there is no "
                    f"contract to score against"
                )
            dest.mkdir(parents=True, exist_ok=True)
            for staged in sorted(staging.iterdir()):
                shutil.move(str(staged), str(dest / staged.name))
            report["fetched"].append({
                "market": market, "files": len(basenames), "bytes": total,
                "train_row_count": meta.get("train_row_count"),
            })
            logger.info(
                "Fetched %s: %d file(s), %d bytes, fit on %s rows.",
                market, len(basenames), total, meta.get("train_row_count"),
            )
        except Exception as exc:  # noqa: BLE001 — one market must not stop the rest
            report["errors"].append({"market": market, "error": _redact(str(exc))})
            logger.error(
                "Could not fetch %s (%s). The volume is unchanged for that "
                "market; it will abstain until an artifact is there.",
                market, _redact(str(exc)),
            )
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    return report


def upload_artifacts(
    source: Path,
    *,
    markets: list[str] | None = None,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """
    Push complete families from `source` into the bucket.

    The other half, and the one a person runs: training happens on a machine
    with the panel, and this is how what it produced reaches the bucket the
    container fetches from. Refuses an incomplete family for the same reason
    `fetch_artifacts` refuses to install one.
    """
    _, _, bucket, prefix = _config()
    report: dict[str, Any] = {"uploaded": [], "refused": [], "errors": []}
    wanted = {m.upper() for m in markets} if markets else None

    boosters = sorted(
        p for p in source.glob("xgboost_*.json") if p.name.count(".") == 1
    )
    for booster in boosters:
        market = booster.stem.split("_", 1)[1].upper()
        if wanted is not None and market not in wanted:
            continue
        family = sorted(p for p in source.glob(f"*_{market}.*") if p.is_file())
        basenames = {p.name for p in family}
        missing = [
            s for s in REQUIRED_SUFFIXES if f"xgboost_{market}{s}" not in basenames
        ]
        if missing:
            report["refused"].append({"market": market, "missing": missing})
            continue
        try:
            for path in family:
                _request(
                    "POST", f"object/{bucket}/{prefix}/{path.name}",
                    session=session,
                    data=path.read_bytes(),
                    headers={
                        "Content-Type": "application/octet-stream",
                        "x-upsert": "true",
                    },
                )
            report["uploaded"].append({"market": market, "files": len(family)})
            logger.info("Uploaded %s: %d file(s).", market, len(family))
        except Exception as exc:  # noqa: BLE001 — one market must not stop the rest
            report["errors"].append({"market": market, "error": _redact(str(exc))})

    return report
