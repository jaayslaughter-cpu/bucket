# End-to-end automation audit — 2026-10-09

RESEARCH_ONLY. Nothing in this project places a wager, sizes a stake, or
automates one, and nothing below proposes that it should. "Autonomous" here
means *the research pipeline runs without a person present*, and the one step
that is deliberately manual forever is recording a stake.

Scope: the deployed worker (`scheduler_worker.py`), the orchestrator it calls
(`main.py`), and the steps around them that a person still performs.

---

## 1. Automation Coverage Score

Two numbers, because one number hides the thing you want to know.

| | Stages | Automated | Score |
|---|---|---|---|
| **The nightly loop** — what has to happen every day with nobody watching | 17 | 16 | **94%** |
| **The whole system** — including setup, retraining and secret handling | 26 | 18 | **69%** |

**The nightly loop is 94%, and as of 2026-10-10 the missing 6% no longer stops
it dead.** This section said the opposite yesterday and the correction is the
point of re-reading it:

> *Superseded (2026-10-09 reading):* the one unautomated stage in the daily
> loop is the BigDataBall workbook, and `main.ingest_market_lines` does not
> degrade when it is absent — it raises `FileNotFoundError` at step [2], the
> orchestrator records a FAILED `pipeline_runs` row and returns 1. So on a
> fresh container the coverage that matters is not 94%: it is **0% until a
> person uploads one file.**

That was true, and the fix was not to automate the upload. `main.resolve_
market_frames` now falls back to `team_game_stats` and `game_market_lines` in
Postgres — the workbook's own contents, which every run that finds one upserts,
and which survive a redeploy. **The data was never missing; only the file was.**

So the honest reading today: the workbook must be ingested **once, ever**, and
after that a container needs no file and the nightly loop runs unattended at
94%. It still refuses when there is genuinely nothing — neither a workbook nor
a row — because then 11 of the 38 columns a seeded contract names cannot be
built and every row would abstain.

The cost of the fix is a new quiet failure, and it is measured rather than
accepted: the database holds whatever was last ingested, so
`main.market_frames_freshness` reports **MARKET DATA IS STALE** when the newest
team-game row trails the slate by more than `PROPIQ_MAX_MARKET_LAG_DAYS`
(default 10). A workbook run is measured too — a stale file on disk is the same
defect with a different cause.

Scoring rules, so the number can be checked rather than believed:

* A stage counts as automated only if a scheduled job performs it with no
  human action, on a container, with no local machine awake.
* A stage that is *conditional* on durable state (scoring needs a seeded
  artifact) counts as automated — the job runs it — and the condition is
  listed as a manual dependency.
* "Manual by design" stages (recording a stake, provisioning a credential)
  are counted in the denominator of the system score. Excluding them would
  inflate the number by relabelling work as out of scope.

### The nightly loop, stage by stage

| # | Stage | Where | Automated |
|---|---|---|---|
| 1 | Player game logs refreshed | `main.refresh_player_logs`, step [3b] | ✅ since 2026-10-09 |
| 2 | BigDataBall team stats + game markets | step [2] | ✅ **from the database** since 2026-10-10 (**M1**: the workbook itself is ingested once, by hand) |
| 3 | Prop lines (PropLine API) | step [3] | ✅ |
| 4 | Forward slate from ESPN schedule | `attach_forward_slate` | ✅ |
| 5 | Panel load + staleness check | step [4], `panel_freshness` | ✅ |
| 6 | Feature build (incl. fatigue) | step [5] | ✅ |
| 7 | Fatigue verification | step [6] | ✅ |
| 8 | Scoring | step [7] | ✅ (conditional: **D1** seeded artifact) |
| 9 | EV gate verdict | step [8] | ✅ |
| 10 | Persist projections + audit row | step [9] | ✅ |
| 11 | Board build | `run_board` | ✅ |
| 12 | Publication gate | `src/quant/publication_gate.py` | ✅ |
| 13 | Discord dispatch | `run_dispatch` | ✅ |
| 14 | Pre-lock availability re-check | `schedule_prelock_jobs` → `run_prelock`, one job per game at tip − 35 min | ✅ |
| 15 | Grading finished props | `run_settlement` → `settle_pending_props`, 03:30 PT | ✅ |
| 16 | Calibration report rebuild | `rebuild_calibration_report`, inside settlement | ✅ |
| 17 | Results card | `run_results_card`, inside settlement | ✅ |

### The rest of the system

| # | Stage | Automated |
|---|---|---|
| 18 | Schema migrations | ✅ **new 2026-10-09** — `scripts/start.sh` runs `scripts/run_migrations.py` at boot and hard-fails the container if it cannot |
| 19 | Deployment self-probe | ✅ **new** — `scripts/railway_healthcheck.py`, advisory at boot, nonzero exit on demand |
| 20 | Thread-pool capping | ✅ `cap_thread_counts()` from `PROPIQ_MAX_THREADS` |
| 21 | Graceful shutdown | ✅ SIGTERM/SIGINT → scheduler drains the running job |
| 22 | Model training / retraining | ❌ **M2** |
| 23 | Volume mount + artifact seeding | ❌ **M3** (tooling exists: `scripts/seed_volume.py`) |
| 24 | Image build + in-image validation | ❌ **M4** |
| 25 | `main.py --init-db`, once per database | ❌ **M5** |
| 26 | Starting-position ingest | ❌ **M6** |
| — | Recording a stake | ❌ **by design, permanently** |
| — | Secret provisioning and rotation | ❌ **by design** |

---

## 2. Manual Dependency Inventory

| ID | Dependency | Cadence | What happens without it | Can it be automated? |
|---|---|---|---|---|
| **M1** | **BigDataBall team-stats workbook** (`BIGDATABALL_XLSX`) | **once, ever** — and again whenever you want fresher team stats | **Nothing, after the first ingest.** Step [2] reads `team_game_stats` and `game_market_lines` back from Postgres instead. Until 2026-10-10 a missing file was `FileNotFoundError` → FAILED `pipeline_runs` row → exit 1, the whole slate, and the image excludes `data/` and `*.xlsx` deliberately so a fresh container had none. With **neither** a workbook nor a row, step [2] still refuses — and says which | **The fetch: no, and it should not be.** It is a licensed third-party export; scraping it would breach the licence this project is careful about. **The dependency: yes, and now is.** `railway_healthcheck` reports a missing workbook as WARN with the newest row's date when the database can answer, and FAILURE only when it cannot |
| **D1** | **A seeded model artifact** on the volume | once, then per retrain | Every row abstains. The slate runs, writes projections with no probabilities, dispatches an abstention and **exits 0** | Partly: `check_model_artifact()` reports it at boot and the probe now exits nonzero. The *putting it there* cannot be automated — there is no object-storage client in this repository |
| **M2** | Model training | manual, periodic | The artifact ages. Nothing fails; the features drift away from the fit | Technically yes, and deliberately not: an unattended retrain that silently replaced the artifact producing today's probabilities is a worse failure than a stale one. §3 G3 |
| **M3** | Volume mount + seeding | once per environment | §D1 | Mounting: no (platform action). Seeding: `scripts/seed_volume.py --from DIR --apply` |
| **M4** | `python -m scripts.validate_docker` | once per Dockerfile change | **Done 2026-10-10**: built, 15/15 checks pass, container boots and schedules. One caveat in `docs/deploy_railway.md` §2 | Yes — in CI, with a daemon. Worth wiring now that it is known to pass |
| **M5** | `python main.py --init-db` | once per database | The ORM tables do not exist; the SQL migrations alone do not create them | Yes, and it is the natural companion to the boot migration step. §3 G1 |
| **M6** | `scripts/pull_starting_positions.py` | manual | `starting_position` stays NULL. Nothing reads it yet, so nothing breaks | Yes, once a feature consumes it |
| **—** | Recording a stake | per wager | ROI cannot be computed | **Must stay manual.** PropIQ never places a wager and never writes a stake. Automating this would make the project something it has decided not to be |
| **—** | Secrets (`DATABASE_URL`, `PROPLINE_API_KEY`, `DISCORD_WEBHOOK_URL`) | at provision, and at rotation | Nothing runs, or nothing dispatches | **Must stay manual.** No credential is defaulted in any image layer and none should be |

---

## 3. Automation Gaps & Bottlenecks

Ordered by what actually stops a deployment.

### G1 — `--init-db` is not in the boot sequence *(small, real)*

`scripts/start.sh` applies the SQL migrations but does not create the
ORM-defined tables. A brand-new database therefore needs one manual command
that the rest of the boot already has the connection for. The reason it is not
there yet: `create_all` is not transactional alongside the migration run, and
making the boot do two kinds of schema change in two transactions is a worse
failure mode than one documented manual step. **Proposed**, not done: a
`--ensure-tables` step in `run_migrations` that runs `create_all` *before* the
migration transaction, where it is idempotent and additive.

### G2 — ~~a missing workbook is now visible, not solved~~ — SOLVED 2026-10-10

Was: "reports it as FAILURE before the slate … still a FAILURE at 09:00 PT if
nobody acts between the two." The slate no longer fails on a missing workbook
at all; see §1. What survives of this gap is narrower and still real:

**Nobody is told when the probe does find something.** The probe writes to the
deploy log and this project's notification channel is Discord, so a FAILURE at
boot — an unseeded volume, an empty database, pending migrations — is seen only
by whoever opens the platform's log pane. **Proposed, not done**: dispatch the
probe's FAILURE lines to Discord at boot, reusing `src/notify/discord.py`
(which already redacts the webhook). A boot-time dispatch that fires on every
redeploy loop is its own nuisance, and the gating deserves its own design.

**And a stale database is now a thing to watch.** `market_frames_freshness`
logs it and records it in the run's audit row; nothing escalates it. That is
the same shape as `PANEL IS STALE`, which is also only logged, so it is
consistent rather than good.

### G3 — no automated retrain, on purpose

The bottleneck is real: the artifact is a fixed snapshot and the panel moves
daily. The reason not to close it is in `docs/training_window.md` — narrowing
the window to recent seasons was *measured* and it costs accuracy, so "retrain
nightly on recent data" is not the improvement it sounds like. An unattended
retrain also replaces the artifact that produced the probabilities now in the
database, mid-season, with no human looking at the validation numbers.
**Proposed if ever wanted**: a weekly job that trains to a *staging* path and
reports the comparison, leaving the swap manual.

### G4 — the pre-lock re-check cannot re-price

`run_prelock` re-runs the scratch filter and posts a withdrawal card. It does
**not** refit or re-price, and that is correct twice over: refitting 35 minutes
before tip would be absurd, and re-pricing needs a live odds feed this project
cannot reach. Listed so it is not mistaken for an oversight.

### G5 — a single replica is a single point of failure

`numReplicas: 1` is required (two schedulers would double-dispatch), so a
container that dies at 08:55 PT misses the slate: the one-hour misfire grace
only helps if the process comes back inside it. **Proposed**: nothing. A
distributed lock to make two replicas safe is a large change to buy redundancy
for a research job whose worst case is one missing board, and the platform's
`restartPolicyType: ON_FAILURE` covers the common case.

### G6 — ~~the image has never been built~~ — CLEARED 2026-10-10

It builds, 15/15 `validate_docker` checks pass, and the container boots through
`scripts/start.sh` and schedules both jobs in Pacific. The apt list was the
part that turned out to be wrong: the `libgomp1` layer installed a library
nothing loaded (the xgboost wheel vendors its own) and was the only thing in
the build that needed the Debian package index. A build-time import assertion
replaced it.

**What is now the automation gap here**: nothing runs `validate_docker` on a
Dockerfile change. It was unrunnable before, so there was nothing to wire; it
passes now, so a CI job is worth having. **Proposed, not done** — it needs a
daemon in CI and is outside this change.

---

## 4. Autonomous Execution Flow Diagram

Everything below the dashed line runs with no person present.

```
  MANUAL, ONCE PER ENVIRONMENT            MANUAL, PERIODIC
  ────────────────────────────            ────────────────
  mount volume at /app/data               BigDataBall workbook  ──┐  (M1:
  set DATABASE_URL, PROPLINE_API_KEY        ONCE, then optional)  │
                                          train artifacts ───────┤  (M2)
      DISCORD_WEBHOOK_URL                 seed_volume --apply ────┤  (M3)
  main.py --init-db                (G1)   validate_docker ────────┘  (M4)
  ╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
  CONTAINER BOOT          scripts/start.sh
      │
      ├─ run_migrations ──────────── FAIL ⇒ exit nonzero, deploy marked failed
      ├─ railway_healthcheck ─────── FAIL ⇒ logged; the worker starts anyway
      └─ exec scheduler_worker.py
             ├─ cap_thread_counts / check_state_dir
             ├─ check_model_artifact / check_parlay_ledger
             ├─ schedule_prelock_jobs   (armed at boot, not only after 09:00)
             └─ BlockingScheduler, timezone America/Los_Angeles
                    │
   09:00 PT ────────┤ slate   (misfire grace 1h — a late slate is SKIPPED,
                    │          because projecting a tipped game is worse)
                    │   [1] preflight
                    │   [2] market frames ── workbook, else Postgres;
                    │        neither ⇒ refuses and says so       (M1)
                    │        stale ⇒ MARKET DATA IS STALE, runs on
                    │   [3] prop lines + forward slate
                    │  [3b] refresh player_game_logs
                    │   [4] panel + freshness (> 3 days ⇒ PANEL IS STALE)
                    │   [5] features   [6] fatigue verified
                    │   [7] score ───── no artifact ⇒ every row ABSTAINS,
                    │   [8] EV gate              exit 0, looks healthy   (D1)
                    │   [9] persist projections + pipeline_runs
                    │    └─ run_board ─→ publication gate ─→ run_dispatch
                    │                     no calibration evidence ⇒
                    │                     the card carries the reason
                    │
   tip − 35 min ────┤ prelock  (one one-shot job PER GAME, re-armed each slate)
                    │   re-run scratch filter on that game's recommended rows
                    │   OUT / DOUBTFUL ⇒ withdrawal card
                    │   feed unusable ⇒ a card saying the check FAILED
                    │
   03:30 PT ────────┤ settlement  (misfire grace 6h — a finished game stays
                    │              finished, so catching up is harmless)
                    │   settle_pending_props  → grade every PENDING prop
                    │   rebuild_calibration_report → the gate's evidence
                    │   run_results_card → yesterday's settled record
                    └─ ... and the 09:00 slate reads evidence that already
                       includes last night, instead of evidence a day stale.

  NEVER AUTOMATED, BY DESIGN
  ──────────────────────────
  placing a wager · sizing a stake · recording a stake · rotating a credential
```

---

## 5. On the "automated post-game settlement worker"

**It already exists and has since before this audit.** No new script was
written, because writing a second one would have been the duplication this
repository keeps finding.

* `scheduler_worker.build_scheduler` registers `run_settlement` on a cron
  trigger at `PROPIQ_SETTLE_HOUR_PT` / `_MINUTE_PT`, default **03:30 PT**,
  `max_instances=1`, `coalesce=True`, six-hour misfire grace.
* `run_settlement` calls `src.settlement.runner.settle_pending_props`, which
  grades every `PENDING` `prop_results` row whose game has finished, bounded by
  `max_game_age_days=14` so a permanently stuck game id is not retried forever,
  and cut off on the **Pacific** calendar day rather than the host's.
* It then rebuilds the calibration report and posts the previous Pacific day's
  results card. A failed report does not fail settlement: the grading is the
  durable part.
* A raised settlement keeps the scheduler up and records `status: FAILED`.

What this audit added around it: the calibration report's path is now checked
against the durable state root, because settlement writes it at 03:30 and the
slate reads it at 09:00 — on an ephemeral path a redeploy between the two
leaves the gate with no evidence and withholds every card for a reason that is
not the real one.
