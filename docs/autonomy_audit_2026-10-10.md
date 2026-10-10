# End-to-end autonomous execution audit — 2026-10-10

RESEARCH_ONLY. "Autonomous" here means *the research pipeline runs unattended*.
Placing a wager, sizing a stake and recording one are not in scope and must
never be automated; this audit separates those **intentional safety
boundaries** from **technical omissions** and says which is which every time.

Every ✅ and ❌ below was checked against the code at commit `ca7bc3a`, not
recalled. Where a criterion as written rests on a false premise, the premise is
corrected rather than answered.

---

## 1. Autonomy verdict

> ### **SEMI-AUTONOMOUS — one-time setup, then hands-free**
>
> The nightly loop needs no human. **Four one-time setup actions** stand
> between a fresh Railway service and that loop, three of which no code can
> take. After them, the system runs unattended and degrades loudly rather than
> silently.
>
> It is **not** "Blocked by Manual Dependencies" — nothing recurring requires a
> person. It is **not** "Fully Autonomous" either, and the gap is honest: a
> volume must be mounted, and **a failure is currently invisible to the
> operator** because no failure path reaches Discord (§5.2).

**What changed today**: the last recurring manual step — putting a trained
model on the volume — is gone. `src/models/artifact_store.py` fetches complete
artifact families from object storage at boot (`scripts/start.sh`, before the
probe). Yesterday that was a person running `scripts/seed_volume.py` every
deploy.

| | Score |
|---|---|
| Nightly loop (what must be hands-free) | **17 / 17 stages** |
| One-time setup | 2 of 4 automatable, **both now automated** |
| Telemetry of failure | **1 / 2** — successes dispatch, failures do not |

---

## 2. The ten criteria, checked

### 1. Ingestion & trigger autonomy

**✅ 1.1 Zero-input scheduling.** Verified by building the real
`BlockingScheduler` (APScheduler 3.11.3) and reading its job table:

```
job_defaults : {'misfire_grace_time': 900, 'coalesce': True, 'max_instances': 1}
jobs         : ['slate', 'settlement']
executor     : ThreadPoolExecutor max_workers = 10
```

`run_slate` calls `main.main([])` — **an empty argv**, so the scheduled path
and a manual one cannot drift. No CLI call, no argument passing.

**Two corrections to the criterion as written:**

* The slate runs at **09:00 PT**, not 10:00 AM (`DEFAULT_SLATE_HOUR = 9`,
  overridable via `PROPIQ_SLATE_HOUR_PT`). 09:00 is before any tip; 10:00 would
  also be fine, but the code says 9.
* Pre-lock is **tip-off minus 35 minutes** by default, bounded 5–180
  (`DEFAULT_PRELOCK_LEAD_MINUTES = 35`) — inside the 30–45 window asked for,
  and it is **one job per game at that game's own tip-off**, not one slate-wide
  check. Armed at boot *and* re-armed inside `run_board` after every slate, so
  a container that starts at 16:00 still covers tonight.

**✅ 1.2 Network & API resilience.** The premise "will a single HTTP 500 crash
the background worker thread" is answered twice over: it cannot, at two
independent levels.

| Client | Daily path | Retry | Timeout | 429 | 4xx |
|---|---|---|---|---|---|
| `espn_client` | ✅ schedule, injuries | 3, exponential | 30s | honours `Retry-After` | not retried |
| `espn_schedule` / `espn_availability` / `espn_game` | ✅ | inherit `espn_client` | — | — | — |
| `propline` (odds) | ✅ | 4 attempts | 20s | honours `Retry-After` | 401/403 not retried |
| `boxscores` (step [3b], `leaguegamelog`) | ✅ daily | ✅ | ✅ | — | — |
| `boxscore_fetcher` (settlement) | ✅ nightly | 3, **linear** | ✅ | ❌ not handled | — |
| `artifact_store` (new) | ✅ boot | 3, exponential | 30s | honours | not retried |
| `starting_positions`, `kaggle_nba`, `basketball_reference` | ❌ hand-run | none | some | — | — |

And **every scheduled entry point catches `Exception`** — `run_slate`,
`run_board`, `run_dispatch`, `run_prelock`, `run_settlement`,
`run_results_card`, `rebuild_calibration_report`, `schedule_prelock_jobs` —
so a raise is logged and the scheduler stays up. APScheduler would also catch
it. One finding: `boxscore_fetcher`'s linear backoff and missing 429 branch
(**P4** below).

### 2. Feature pipeline & model inference autonomy

**✅ 2.1 Dynamic feature assembly.** `build_feature_matrix` contains **zero**
file reads — no `read_csv`, `read_parquet`, `read_excel` or `open()`. It takes
DataFrames. The panel comes from Postgres (`load_player_panel`), refreshed by
the slate itself at step [3b].

The one file dependency, the licensed BigDataBall workbook, was fixed yesterday:
`main.resolve_market_frames` falls back to `team_game_stats` and
`game_market_lines` in Postgres, so the workbook is needed **once, ever**.

**⚠️ One silent capability loss** — see **R2**: the absences layer reads
`PROPIQ_INACTIVE_CACHE_DIR`, which only the hand-run `fetch-inactives` CLI
writes. `load_cached_absences` returns `None` on an empty cache and the builder
omits the layer, so in production **that layer never contributes**. It does not
break scoring: the seeded PTS contract's 38 columns contain no absence feature.

**✅ 2.2 Automatic artifact resolution — but the criterion's premise is wrong
in three ways**, and each one matters:

| As written | Actually |
|---|---|
| `.joblib` / `.pkl` | **`xgboost_{MARKET}.json`** plus `.meta.json` and `.mean.json`. Nothing in this project writes a joblib or a pickle; `.gitignore:95` exists to keep one out of history |
| `/app/data/models/` | **`config/model_comparison.yaml`'s `artifacts_dir`** = `data/external/model_runs/comparison`, which under a volume at `/app/data` is already on the volume. There is no `models/` directory |
| "without throwing missing-file exceptions" | `main.resolve_model_artifact` returns **`(None, reason)`** and never throws; `score_prob_over` returns an all-null Series. It cannot throw a missing-file exception |

A probe written to the criterion as stated would glob `*PTS*.joblib` under
`/app/data/models/` and **report FAILURE on a correctly seeded volume** — which
is precisely what the draft healthcheck did before it was rewritten.

Resolution order, most explicit first: `--model` → `$PROPIQ_MODEL` → newest
`xgboost_{MARKET}.json` **with its `.meta.json` sidecar** → the legacy path.
An artifact without its sidecar is not a candidate, because scoring without
the training-time `feature_cols` is refused anyway.

### 3. Quant engine, EV & sizing autonomy

**✅ 3.1 De-vigging & EV.** `src/quant/contracts.py` (243 lines) and
`src/quant/ev_engine.py` (364 lines) contain **no `try`/`except` at all** — and that is the better
design, not a gap: they guard with **19 explicit checks** (`isfinite`,
`<= 0`, `is None`, `DATA_NOT_AVAILABLE`, abstain paths) so a bad input becomes
a *named abstention* rather than a caught exception. A math domain error cannot
arise from an unguarded `log` of a non-positive number because the guard
precedes it. Multi-leg Pick'em flows through `dfs_payouts` /
`leg_correlation`, whose per-market loop is also guarded.

**❌ 3.2 Advisory sizing is NOT on the automated path.** This is the clearest
technical omission in the audit.

`src/quant/advisory_sizing.recommended_units_for_entry` — fractional Kelly,
hard-capped, tested — is imported in exactly **one** place:
`scripts/nba_model_cli.py:1238`, behind a `--advisory-sizing` flag on a
hand-run command. Searching `src/pipeline/slate_board.py` and
`scheduler_worker.py` for `recommended_units` or `advisory` returns **nothing**.

So the nightly Discord card carries no recommended size. The sizer exists, the
scheduler never calls it. **Patch P1.**

This is a *technical omission*, not a safety boundary — and the distinction
matters: an advisory `recommended_units` is read-only metadata, which an
earlier decision explicitly permitted. **Binding a size to execution remains
prohibited and is not proposed.**

### 4. Database persistence & post-game settlement autonomy

**✅ 4.1 Pre-game snapshotting.** Inside the 09:00 PT slate, hours before any
tip: `main.py:1772` `persist_projections(projections, run_id=run_id)` and
`main.py:1782` `record_pending_prop_results(recorded.rows)`, plus
`insert_prop_snapshots` at `main.py:743` for the posted lines.
`migrations/003_capture_vs_ingest_time.sql` keeps **observation time separate
from ingest time**, so a workbook loaded months later cannot claim its lines
were seen today.

**✅ 4.2 Post-game grading loop.** Already existed; no new script was written,
because a second one would have been duplication.
`run_settlement` on a cron at **03:30 PT** (`max_instances=1`, `coalesce=True`,
six-hour misfire grace — a finished game stays finished, so catching up is
harmless) → `settle_pending_props` grades every `PENDING` `prop_results` row
whose game has finished, bounded by `max_game_age_days=14` so a stuck game id
is not retried forever, cut off on the **Pacific** calendar day → rebuilds the
calibration report → posts the results card. A raise keeps the scheduler up and
records `status: FAILED`.

### 5. Fault tolerance, self-healing & telemetry

**✅ 5.1 Graceful exception recovery in loops.** The per-market scoring loop
(`main.py:1063`) calls `score_prob_over`, which ends in a broad
`except Exception` returning an all-null Series — so one market's corrupt
artifact yields a null column and the loop proceeds to the next. The new
artifact fetch has the same shape per market. `load_cached_absences` catches
per *file* so one bad parquet does not lose the rest.

**❌ 5.2 Webhook & alert dispatch — HALF met, and this is the most important
finding in the audit.**

Successes dispatch. Failures do not.

| Event | Discord? |
|---|---|
| Board / recommendation card | ✅ `run_dispatch` |
| Withdrawal card (late scratch) | ✅ `run_prelock` |
| Daily results card | ✅ `run_results_card` |
| Abstention with its reason | ✅ default on |
| **Boot ERROR** (unseeded volume, csv ledger, unwritable state dir) | ❌ log only |
| **FAILED slate** | ❌ log only |
| **FAILED settlement** | ❌ log only |
| **Probe FAILURE** | ❌ log only |

Verified by grep: `check_state_dir`, `check_model_artifact`,
`check_parlay_ledger`, `run_slate`'s failure path and `run_settlement`'s
failure path contain **zero** references to `discord` or `notify`.

For a system whose notification channel *is* Discord, this means a silent
failure is genuinely silent. **Patch P2 — the highest-value change in this
plan.**

---

## 3. Manual intervention inventory

### Intentional safety boundaries — must stay manual

| | Why |
|---|---|
| **Placing a wager** | PropIQ contacts no operator's order API and never will |
| **Sizing bound to execution** | Advisory `recommended_units` as read-only metadata is permitted; binding it to a bet is not |
| **Recording a stake** | No stake is ever written, so ROI stays undefined until a person logs one. Strike rate and CLV do not need one |
| **Pushing a new artifact to the bucket** | `--push` replaces what produced every probability now in the database. The boot is asserted never to call it |
| **Secret provisioning and rotation** | No credential is defaulted in any image layer |

### Technical omissions — one-time, and two are now closed

| | Automatable? | State |
|---|---|---|
| Mount the volume at `/app/data` | **No** — a platform action | manual, unavoidable |
| `python main.py --init-db` | ~~No~~ | **CLOSED** — `--ensure-tables` at boot |
| Put a model on the volume | ~~No~~ | **CLOSED today** — object-storage fetch at boot |
| Ingest the BigDataBall workbook **once** | **No** — a licensed export; scraping it would breach the licence | manual, unavoidable, once |
| Set `DATABASE_URL`, `PROPLINE_API_KEY`, `DISCORD_WEBHOOK_URL` | **No** — by design | manual, unavoidable |
| Populate the inactive-player cache | **Yes** | open — **R2** |
| Retrain | **Yes, and deliberately not.** An unattended retrain replaces the artifact producing today's probabilities with nobody reading the validation numbers. `docs/training_window.md` also measured that recency does *not* help | open by choice |
| Run `validate_docker` on a Dockerfile change | **Yes** | open — needs a daemon in CI |

---

## 4. Autonomous failure points & race conditions

Checked empirically where possible, and the two most-feared ones are **not**
present.

**✅ Pre-lock jobs do NOT accumulate.** The obvious multi-day leak — one job
armed per game per night, forever. Verified by arming a real one-shot `date`
job, letting it fire, and reading the job table:

```
jobs after add   : ['probe:one-shot', 'settlement', 'slate']
one-shot fired   : True
jobs after firing: ['settlement', 'slate']
```

APScheduler removes a fired `date` job. Memory is flat across nights.

**✅ No thread-pool starvation.** `ThreadPoolExecutor(max_workers=10)` with
`max_instances=1` per job. A 15-game night arms 15 one-shot jobs clustered
around tip-offs; each is an IO-bound injury fetch of a few seconds, and
`PRELOCK_MISFIRE_GRACE_SECONDS = 600` gives any queued job ten minutes. Even
all 15 at the same minute complete far inside the window. No deadlock: no job
waits on another.

**✅ Connections survive an idle day.** Fixed 2026-10-09: `pool_recycle=1800`
(SQLAlchemy's default is *never*, and this worker idles 18 hours between 09:00
and 03:30) and `connect_timeout=10` — **without which there was no bound at
all**, and a pooler that accepted the TCP connection without completing the
handshake would have hung the slate past every tip-off with no error anywhere.

**✅ Bind-parameter ceiling.** Fixed 2026-10-10: all seven bulk upserts share
one `batched()`. `upsert_player_game_logs` runs daily on a whole season from
`leaguegamelog` and would have failed every night past ~3,200 rows.

### Open risks, ranked

| | Risk | Why it stalls |
|---|---|---|
| **F1** | **A failure is invisible** (§5.2) | The worst one, and it is not a crash: the board goes empty or stale and nothing tells anyone. Days pass. → **P2** |
| **F2** | **Market data goes stale silently-ish** | The DB fallback reads whatever was last ingested. `market_frames_freshness` logs `MARKET DATA IS STALE` past 10 days — but only logs (F1). **The workbook on disk today is last season's: newest game 2026-06-13, 118 days before today's slate** |
| **F3** | **`numReplicas` must stay 1** | `max_instances=1` is per *process*. Two replicas = two schedulers = two ingests, two boards, two dispatches of the same card. No distributed lock exists. Pinned in `railway.json` and asserted by a test |
| **F4** | **A single container is a single point of failure** | A container that dies at 08:55 PT misses the slate; the one-hour grace only helps if it returns inside it. Accepted: redundancy needs F3's lock |
| **F5** | **Absences layer permanently absent** | Not a stall — a capability that silently never contributes. → **R2** |
| **F6** | **`boxscore_fetcher` linear backoff, no 429** | A rate-limited stats.nba.com during settlement retries 2s/4s/6s and gives up. Grading catches up next night (6h grace), so it self-heals | 
| **F7** | **Base image tag floats** | A rebuild months later gets a different CPython patch. Dependency majors are now capped; the base is not |

**Explicitly looked for and not found**: unbounded queue growth, a job that
cannot be interrupted (SIGTERM drains the running job), an unclosed session
(`session_scope` commits/rolls back/closes in `finally`), an unclosed cursor
(`execute_script` uses `try/finally`), and a null that reaches the model — the
contract check rejects a short column list by name before scoring.

---

## 5. Hardening patch plan

Ordered by what actually buys autonomy. **P1 and P2 are the whole gap.**

### P1 — call the sizer from the automated path *(closes §3.2)*

`src/pipeline/slate_board.py`, where each recommended row is assembled:

```python
from src.quant.advisory_sizing import recommended_units_for_entry

# ADVISORY ONLY, and read-only metadata by an explicit earlier decision: this
# number is never bound to an execution path, and nothing in this project
# places or sizes a wager. It is attached because a board row that passes the
# publication gate and carries no size makes the operator do the arithmetic
# the project already does better.
row["recommended_units"] = recommended_units_for_entry(...) if gate_passed else None
```

Gate it on the *same* condition the publication gate uses, so a withheld card
never carries a size. ~20 lines plus tests.

### P2 — route failures to Discord *(closes §5.2, F1)*

A `notify_failure()` in `scheduler_worker` that the boot checks and each job's
`except` call, reusing `src/notify/discord.py` (which already redacts the
webhook):

```python
def notify_failure(stage: str, detail: str) -> None:
    """Log AND dispatch. A failure nobody is told about is the one that
    costs a week of empty boards -- the log pane is not a channel anybody
    watches. Never raises: a dispatch that fails must not turn one failure
    into two."""
```

Needs a **de-duplicating guard**: a redeploy loop would otherwise post on every
boot. Suggest at-most-once-per-stage-per-Pacific-day, keyed in Postgres beside
`pipeline_runs`. ~80 lines plus tests. **Highest value in this plan.**

### P3 — a daily heartbeat

`PROPIQ_DISPATCH_ABSTENTIONS` already posts a refusal, so silence means *the
worker did not run*. One line per settlement saying "slate ran, N rows, board
withheld/posted" makes the absence of a message meaningful. Small, and it
depends on P2's dedupe machinery.

### P4 — `boxscore_fetcher`: exponential backoff + 429 *(F6)*

Make it `RETRY_BACKOFF ** attempt` and add the 429 branch
`espn_client`/`propline` already have. Four lines; low urgency because
settlement self-heals, but it is the only client on a daily path missing both.

### R2 — schedule the inactive-player fetch *(F5)*

`save_inactive_players` exists and nothing calls it on a schedule. A step in
the slate before `build_feature_matrix` would make the absences layer live. Do
this **only alongside a retrain**: adding feature columns the current
38-column contract does not name changes nothing, and a contract that *does*
name them must be fit with them present.

### P5 — pin the base image

From the first ordinary-egress build: `FROM python:3.11-slim@sha256:<digest>`.

### Deliberately NOT proposed

| | Why |
|---|---|
| Automated bet placement or sizing bound to execution | Prohibited, permanently |
| Unattended retraining | Replaces the artifact producing today's numbers with nobody checking. P4 of the configuration audit has the staging-path alternative |
| Scraping the BigDataBall workbook | Licensed export |
| `numReplicas > 1` | Needs a distributed lock; buys redundancy for a research job whose worst case is one missing board |
| Structured JSON logging / `pydantic-settings` | `docs/configuration_audit_2026-10-09.md` R2/R3 has both written out, and the reasons they are not applied: the log lines are prose an operator reads, and `BaseSettings` converts every typo into a dead container |
