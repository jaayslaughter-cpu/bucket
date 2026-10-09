# Configuration audit — 2026-10-09

RESEARCH_ONLY. Configuration only: container, process, database, persistence,
secrets, observability. Nothing here places, sizes or automates a wager.

Every line number below is from the tree at this commit. Six pillars, named
explicitly so the mapping can be checked rather than inferred.

---

## 1. Configuration Rating Score

| Pillar | Before today | After | Why |
|---|---|---|---|
| **P1 Container & image** | 6 / 10 | **8 / 10** | Dependencies were entirely uncapped; the entrypoint is now a versioned script. Still never built |
| **P2 Process & lifecycle** | 7 / 10 | **9 / 10** | Signals, replica count and the start sequence were correct or implicit; they are now explicit and tested |
| **P3 Database & connections** | 6 / 10 | **9 / 10** | `pool_recycle` was *never* and there was no connect timeout at all |
| **P4 Persistence & durable state** | 4 / 10 | **8 / 10** | `RAILWAY_VOLUME_MOUNT_PATH` was read nowhere; the artifacts' durability is now resolved, probed and seedable |
| **P5 Secrets & environment** | 9 / 10 | **9 / 10** | Already good: nothing defaulted in a layer, the webhook redacted, `validate_docker` scans for leaks. Unchanged |
| **P6 Observability & scheduling** | 7 / 10 | **8 / 10** | Boot checks were already strong; the probe and `job_defaults` close the two holes |
| **Overall** | **6.5 / 10** | **8.5 / 10** | |

**Not 10/10, and the missing 1.5 is one thing: the image has never been
built.** Every judgement about P1 is reasoned from dependency metadata rather
than from a green build, and two of the in-image checks exist specifically to
prove the `chown` trap that `docs/deploy_railway.md` documents. A score above
8.5 would be a claim this repository cannot support.

---

## 2. Configuration Gaps & Sub-Optimal Settings

### FIXED TODAY

| ID | Where | Was | Now |
|---|---|---|---|
| **F1** | `pyproject.toml:6-28` (`dependencies`), `requirements.txt` throughout | **Every dependency uncapped** (`xgboost>=2.0`, `APScheduler>=3.10`, …). The artifacts live on a volume that survives a redeploy while the image does not, so an uncapped rebuild pairs a fresh major with a booster saved by an older one. `APScheduler>=3.10` is the sharpest case: v4 is a rewrite in which `BlockingScheduler` moves, so the worker would fail at *import* on a redeploy, with the previous image gone | Major caps on both files, kept in step and compared by `tests/test_railway_deploy.py::test_requirements_txt_and_pyproject_do_not_disagree`. `pyarrow` and `ruff` stay uncapped with their reasons in place |
| **F2** | `src/models/xgb_adapter.py:390` (save), `:417` (load) | The sidecar recorded `model_version` and `feature_schema_version` and **no library version at all**, so a version-induced change in probabilities was indistinguishable downstream from a change in the player | `src/models/runtime_versions.py` records the tracked libraries on save and the loader warns on a **major** difference. Never raises: refusing to score a slate over a version string is worse than a warning beside a working booster |
| **F3** | `src/db/session.py:149-166` | `pool_recycle` unset — SQLAlchemy's default is **never** — while this worker leaves a connection idle for the eighteen hours between 09:00 and 03:30 PT. And **no `connect_timeout`, so no bound at all**: a pooler that accepts the TCP connection and never completes the handshake blocks the slate past every tip-off with no error anywhere | `pool_recycle=1800`, `connect_timeout=10`, `pool_timeout=30`, `application_name=propiq-worker`, all bounded `PROPIQ_DB_*` overrides that fall back rather than raising |
| **F4** | `src/utils/volume.py` (new) | `RAILWAY_VOLUME_MOUNT_PATH` **read nowhere in the repository**. Every path was relative to the working directory — correct in the image only because the Dockerfile happens to mount the volume at `/app/data`. Mount it elsewhere and the writes went to the container filesystem, succeeded, and vanished at the next redeploy | `resolve_state_root()` resolves most-explicit-first and returns *how* it chose. The probe WARNs when the root was reached by fallback inside a container |
| **F5** | `scheduler_worker.py:1250-1263` | `BlockingScheduler(timezone=DISPLAY_TZ)` with **no `job_defaults`**, and APScheduler's own default misfire grace is **one second**. Every job registered today passes its own, so nothing was broken — but the next job added without those kwargs would be silently skipped by a two-second delay, and "it did not run" is indistinguishable from "it ran and found nothing" | `job_defaults={max_instances: 1, coalesce: True, misfire_grace_time: 900}` |
| **F6** | `main.py:97-107`, `scripts/railway_healthcheck.py` | A **missing BigDataBall workbook fails the whole slate** (`ingest_market_lines` is unguarded: `FileNotFoundError` → FAILED `pipeline_runs` row → exit 1), the image excludes `data/` and `*.xlsx` on purpose, and **nothing asked about it at boot** | `BIGDATABALL_XLSX` and its default are named once in `main.py` and the probe reports a missing workbook as **FAILURE** before the slate runs |
| **F7** | `scheduler_worker.py:154-175` | The board's market list was parsed inside `run_board` and unreachable from anywhere else, so the probe would have carried a second copy — and a probe checking PTS/REB/AST while the worker is configured for PTS alone reports a failure that is not one | `board_markets()`, with `run_board` AST-asserted to call it |
| **F8** | `tests/test_go_live_readiness.py:140` | The pooling guard read `"pool_size=5" in body` — a substring check that would pass on a **comment** saying so, and that failed on a change which strengthened what it guarded | Reads the kwargs that reach `create_engine` |

### STILL OPEN

| ID | Where | Finding | Severity |
|---|---|---|---|
| **O1** | `Dockerfile:24` | `FROM python:3.11-slim` is a floating tag. A rebuild six months from now gets a different patch release of CPython and of every apt package in the layer. **Not fixed deliberately**: pinning to a digest I cannot resolve or verify here would be a guess written as a fact, and the image has never been built once, so the first build should establish the digest. Pin it at that point | medium |
| **O2** | `Dockerfile:51-53` vs `requirements.txt` | Two dependency declarations for one commit. The caps now agree and a test enforces that, but the *contents* can still diverge — a package added to one and not the other. A single source would need either Nixpacks support for extras or dropping `requirements.txt`, and the latter removes the belt-and-braces for a Nixpacks build | low |
| **O3** | `.dockerignore` | `tests/` and `docs/` are not excluded, so they ship in the image. Harmless and mildly useful (the deploy doc is readable inside the container); listed so it is a decision | low |
| **O4** | `Dockerfile` | No `HEALTHCHECK` instruction. **Deliberate**: Railway does not consume Docker's healthcheck for a worker, and a probe running every 30s would hit the database and the filesystem for nothing. `scripts/railway_healthcheck.py` is run once at boot and on demand instead | none — decision |
| **O5** | `railway.json` `restartPolicyMaxRetries: 3` | After three failed starts the service stays down until someone looks. For a *migration* failure that is right — looping forever would hide the deploy failure. For a transient database outage at boot it means manual intervention. `ALWAYS` would trade one for the other; 3 is the chosen trade | low — decision |
| **O6** | `scheduler_worker.py:1315-1318` | `logging.basicConfig` with a human format, not structured JSON. **Recommended against**, with a drop-in in §3 if wanted anyway: every log line in this worker is deliberately prose that explains a decision ("STATE DIRECTORY … IS NOT WRITABLE … most likely owned by root while this process runs as uid 10001"), and JSON-wrapping them makes the thing an operator actually reads worse in exchange for machine parsing nothing currently consumes | low — decision |
| **O7** | across ~40 `PROPIQ_*` variables | No central `pydantic-settings` model. **Recommended against.** The existing pattern is a bounded reader at each use site that logs and falls back; a settings model centralises validation and changes the failure mode to *refuse to start*, which for an unattended worker is strictly worse — a worker that will not boot cannot tell you why | low — decision |
| **O8** | `railway.json` `numReplicas: 1` | A single point of failure: a container that dies at 08:55 PT misses the slate, since the one-hour grace only helps if it returns inside it. Cannot be raised — two schedulers double-dispatch, and there is no distributed lock. Accepted | medium — accepted |
| **O9** | the whole of P1 | **The image has never been built.** `python -m scripts.validate_docker --preflight` passes 8 checks; the build and the six in-image checks have never run | **high** |

---

## 3. Optimized Drop-In Config Replacements

Applied ones are in the diff. These are the three that are **offered and not
applied**, written out so the decision is reversible by someone who disagrees.

### R1 — pin the base image, after the first successful build (O1)

```dockerfile
# Replaces Dockerfile:24. Resolve the digest from the build that proves the
# image works, not from a guess:
#   docker image inspect python:3.11-slim --format '{{index .RepoDigests 0}}'
FROM python:3.11-slim@sha256:<digest from that build>
```

### R2 — structured JSON logs (O6), if machine parsing ever has a consumer

```python
# Replaces scheduler_worker.py:1315-1318. Opt-in, so the human format stays
# the default and the prose log lines keep working.
def _configure_logging() -> None:
    fmt = (os.environ.get("PROPIQ_LOG_FORMAT") or "text").strip().lower()
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    if fmt != "json":
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
        )
        return

    import json as _json

    class _JsonFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            payload = {
                "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
                "level": record.levelname,
                "logger": record.name,
                "msg": record.getMessage(),
            }
            if record.exc_info:
                payload["exc"] = self.formatException(record.exc_info)
            return _json.dumps(payload)

    handler = logging.StreamHandler()
    handler.setFormatter(_JsonFormatter())
    logging.basicConfig(level=level, handlers=[handler], force=True)
```

**Why it is not applied.** Nothing consumes JSON logs today, and the lines it
would wrap are the ones an operator reads in the platform's log pane. Add it
when a log sink exists, not before.

### R3 — a settings model (O7), if the failure mode is ever wanted

```python
# NOT RECOMMENDED for this worker. Shown so the trade-off is concrete.
from pydantic import Field
from pydantic_settings import BaseSettings

class WorkerSettings(BaseSettings):
    database_url: str
    propline_api_key: str | None = None
    discord_webhook_url: str | None = None
    propiq_slate_hour_pt: int = Field(9, ge=0, le=23)
    propiq_settle_hour_pt: int = Field(3, ge=0, le=23)
    propiq_max_threads: int = Field(2, ge=1, le=64)
    propiq_parlay_ledger: str = "postgres"
```

**Why it is not applied.** `BaseSettings` raises on a bad value at import. This
worker's whole policy is the opposite: `_int_env` logs and falls back,
`check_state_dir` and `check_model_artifact` warn and carry on, and
`build_scheduler` falls back to the default hour on an unusable one — because a
worker that refuses to start cannot report anything, and the settlement job is
still useful with no model. Centralising validation here would convert every
typo into a dead container. The one place where refusing *is* right — a
migration that cannot be applied — already exits nonzero, in
`scripts/start.sh`.

### What was asked for and is deliberately NOT here

| Asked | Verdict |
|---|---|
| `TZ=America/Los_Angeles` | **Not applied.** `Dockerfile:39` keeps `TZ=Etc/UTC`. Storage is UTC throughout and the schedule does not depend on `TZ` — `scheduler_worker` passes `America/Los_Angeles` to APScheduler explicitly. What `TZ` governs is the naive `datetime.now()` calls, which should stay UTC and reproducible; setting it to Pacific would make those shift twice a year while stored timestamps did not, and the resulting off-by-an-hour rows would look like data rather than a setting |
| `SPORTSDATA_API_KEY`, `THE_ODDS_API_KEY` | **Not added.** Neither appears anywhere in this repository, and the Odds API (`ODDS_API_KEY`) is recorded as a banned sportsbook source in `docs/external_feature_harvest.md:90` and `docs/pickem_props.md:61`. Documenting a variable the code does not read is how an operator comes to believe a key is wired up |
| a multi-stage Dockerfile | **Not applied, and it buys nothing here.** A multi-stage build pays off when a build stage installs a toolchain the runtime does not need. This image installs **no toolchain**: `python:3.11-slim` plus `libgomp1`, and every dependency arrives as a wheel. There is no `build-essential`, no `libpq-dev` (hence `psycopg[binary]`), nothing to leave behind. A builder stage would add a layer, a copy and a second place for the dependency list to drift, and would shrink nothing |
| `/app/data/models/` | **Not created.** Artifacts go to `config/model_comparison.yaml`'s `artifacts_dir` (`data/external/model_runs/comparison`), which under a volume at `/app/data` is already on the volume. Writing them to a second location would leave `main.resolve_model_artifact` finding nothing while the files sat there |
| a settlement worker | **Already exists** — `run_settlement` on a 03:30 PT cron, see `docs/automation_audit_2026-10-09.md` §5. Writing a second one would be the duplication this repository keeps finding |
