# Pre-deployment audit — Railway / containerized cloud

Date: 2026-09-28. RESEARCH_ONLY. Nothing here places or sizes a wager.

**Update, same day.** Three of the findings below have been addressed. The
original text is kept as written, with each correction marked inline, so the
audit still reads as the record of what was found rather than of what was
later done.

| Finding | State |
|---|---|
| Blocker 2 · `prop_results` has no writer (R2) | **fixed** — `settlement/recorder.py` + `repository.record_pending_prop_results`, called from `main.py` |
| Blocker 1 · nothing to deploy (R3) | **fixed** — `Dockerfile`, `.dockerignore`, `scheduler_worker.py` (APScheduler, Pacific-anchored) |
| R8 · unbounded concurrent fits | **fixed** — `max_instances=1` per job and `PROPIQ_MAX_THREADS` capping every thread pool |
| Blocker 3 · nothing survives a restart (R1) | **partly** — the parlay ledger moved to Postgres (`PROPIQ_PARLAY_LEDGER=postgres`, migration 004). **The model artifacts did not.** |

**The verdict is unchanged: DO NOT DEPLOY**, now on one blocker rather than
three. `data/external/model_runs/comparison/` still lives on an ephemeral
filesystem with no volume declared, so the first redeploy leaves
`score_prob_over` with no model and every row abstains. A Railway volume
mounted at `/app/data`, or artifacts fetched from object storage at boot,
clears it — that is a deployment-configuration step this repository cannot
take on its own behalf, and the Dockerfile says so where it would bite.

Audited by tracing the import graph and the filesystem writes, not by
reading status tables. Every claim below names the command or file that
establishes it.

---

## 1. Configuration & Cloud Readiness

**Container config: none exists.** No `Dockerfile`, no
`docker-compose.yml`, no `Procfile`, no `railway.json`/`railway.toml`, no
`deploy/`, no crontab. There is nothing for Railway to build or start.

**Dynamic `PORT`: not applicable, and worth stating explicitly.** This is
not a web service — it is a CLI (`scripts/nba_model_cli.py`) plus a batch
orchestrator (`main.py`). Railway would run it as a *cron/worker* service,
which binds no port. A deployment configured as a web service would be
marked unhealthy for never listening, so the service type matters more here
than `PORT` handling does.

**Dependencies:** `pyproject.toml` + `requirements.txt` present. ML extras
are optional (`catboost` raises a named ImportError pointing at
`pip install 'propiq-analytics[ml]'`), so a minimal image can start and
then fail late on a model step. Pin the ML extra into the image or the
first slate run fails at inference rather than at build.

**Secrets: clean.** No credential literals. The two grep hits for
key-shaped names are env var *names*, not values:
`discord.py:44 ENV_WEBHOOK_URL = "DISCORD_WEBHOOK_URL"` and
`propline.py:47 ENV_API_KEY = "PROPLINE_API_KEY"`. `.env.example` documents
`DATABASE_URL`, `PROPLINE_API_KEY`, `DISCORD_WEBHOOK_URL`,
`PROPIQ_MASTER_GUIDELINE`, `BIGDATABALL_XLSX`, `TZ_DISPLAY`, `LOG_LEVEL`.

**Hardcoded paths: clean.** No `C:\`, no `/Users/`, no absolute local
paths in `src/`, `scripts/` or `main.py`. The only `localhost` occurrence is
`db/session.py:68`, a membership test deciding whether to append
`sslmode=require` — correct behaviour, not a hardcoded host.

**Odds source: PropLine (decided).** PropLine is primary;
`decision_board.py:76` sets `SOURCE_PRECEDENCE = ("propline", "oddspapi")`
with OddsPapi as fallback. The Odds API is not used and must not be added.

---

## 2. Pipeline Wiring & Orphan Code Map

Traced with an AST import graph over `main.py`, `src/`, `scripts/`.

**One trap worth recording:** `features/builder.py:159-162` registers its
feature layers by *string module name*
(`("src.features.absences", "attach_absence_features_layer", "absences")`),
so static import analysis reports `absences`, `sports_ev_features`,
`teammate_cascade` and `scoring_efficiency` as orphans when they are in
fact wired. Any future orphan audit must resolve string references or it
will delete live code.

Resolving string references, 6 of 72 `src` modules are unreferenced by the
production graph, and one of those six — `settlement/cli.py` — is a
`python -m` entrypoint rather than dead code, so **5 are genuinely
orphaned**. It is listed below with the orphans because an import graph
cannot tell the two apart; only reading it can.

### Wired end to end

`preflight` → `ingest_market_lines` / `ingest_prop_lines` →
`build_feature_matrix` (fatigue folded into `{stat}_L2` once, at
`builder.py:517`) → `score_prob_over` → `evaluate_ev_gate` →
`assemble_projections` → `persist_projections` → `projections` table.
Settlement runs separately via `python -m src.settlement.cli settle`.

### Unreferenced by the production graph (5 orphans + 1 entrypoint)

| Module | Lines | State |
|---|---|---|
| `src/features/minutes_weighted.py` | 134 | ~~No production import, no test either.~~ **Registered in the builder and tested** (22 tests). Its columns are deliberately kept OUT of `default_feature_cols`: measured \|r\| 0.976–0.992 against `{STAT}_L5` on the 214,381-row panel puts them inside the band the halflife family was excluded for. Registered as a `feature_ab` layer so `--wire-under-test` can settle it. |
| `src/models/eligibility.py` | 129 | ~~Tested, never called.~~ **Wired** — the cold-start gate withholds a board recommendation, and `ks_feature_drift` reports train/validation covariate drift per market. Both config blocks are now read. |
| `src/models/combo_variance.py` | 282 | Tested, never called. PRA is in `markets_post_launch`, so the combo variance it provides is unreachable. |
| `src/models/feature_spec.py` | — | ~~Tested, never called.~~ **Wired** — `verify_feature_contract` runs in `score_prob_over`, and both model `save` paths write the fingerprint. |
| `src/models/protocol.py` | 48 | A typing `Protocol` no adapter declares conformance to, so nothing enforces the model interface. |
| `src/settlement/cli.py` | 134 | **Not an orphan** — a `python -m` entrypoint with `if __name__ == "__main__"`, invoked by command rather than imported. |

### Config declared but never read

Two tiers, and the difference matters:

**Module orphaned AND config unread** — the behaviour does not happen:
- ~~`eligibility: min_prior_games: 10, min_minutes_l5: 12.0`~~ — **now
  applied.** `compare_models_on_panel` attaches the warnings to each prediction
  row and `decision_board` refuses to RECOMMEND a row carrying them. It still
  RECORDS the prediction: a thin-history row is the evidence where the model is
  weakest, and dropping it from the ledger would bias the calibration toward
  the easy cases.
- ~~`drift: ks_p_threshold: 0.01`~~ — **now reported**, per market and feature,
  returned as `covariate_drift`. Report only: dropping a feature on a KS
  p-value would let the validation window choose the feature set, which is the
  shape of a leak.

**Module wired but config unread** — it runs on hardcoded defaults, so
editing the YAML silently does nothing:
- `arbitration` (`agreement_high`, `edge_high`, …) — used via
  `paper_research.py:15`, but the YAML block has no reader.
- `line_diff: pts_per_prob: 0.03` — `line_diff` is called from
  `paper_research.py:291`; the block has no reader.
- `paper_research`, `pocket_roi` — modules wired through the CLI; blocks
  have no reader.

### Dead-ended write path — **FIXED after this audit**

`prop_results` has a table (`db/models.py:312`), a grader
(`settlement/runner.py:141`, which selects `outcome_status == 'PENDING'`)
and a metrics layer (`settlement/metrics.py`, aggregating W/L/PUSH, stake,
profit and both CLV columns) — and **no writer**. The six `pg_insert`
targets are `TeamGameStat`, `PlayerGameLog`, `GameMarketLine`,
`PropLineSnapshot`, `Projection`, `PipelineRun`. `PropResult(` never
appears as a constructor anywhere.

**Now written.** `settlement/recorder.py:pending_prop_result_rows` turns
assembled projections into PENDING rows and
`db/repository.py:record_pending_prop_results` inserts them, called from
`main.py` right after `persist_projections`. Three things about it matter
for a deployment:

- The rows are **predictions, not wagers**: `stake_units` is never written
  and there is no parameter to set it. Strike rate and CLV are therefore
  available; ROI is not, and will not be until a stake is recorded by hand.
- The upsert is `ON CONFLICT ... DO UPDATE ... WHERE outcome_status =
  'PENDING'`. Without that `WHERE`, re-running a slate after settlement
  would overwrite a graded row's pre-game numbers.
- A row with no line, no probability, no source or no game id is **skipped
  and counted by reason**, not guessed. A null `source` in particular is a
  hard skip: Postgres treats NULLs in a unique index as distinct, so such a
  row would be re-inserted on every run.

---

## 3. Production Risks & Breaking Points

**R1 — Ephemeral filesystem destroys all state (Railway-specific, decisive).**
`.gitignore:28` ignores `data/**`, and seven directories hold real state
with **no volume declared anywhere**:

| Path | What is lost on restart or redeploy |
|---|---|
| `data/external/model_runs/comparison/` | **the trained model artifacts** — `score_prob_over` then finds no model and abstains for every row |
| `data/external/market_store/` | the whole bet lifecycle ledger (`bet_lifecycle.csv`/`.parquet`/`.sqlite`) |
| ~~`data/external/parlay_log/`~~ | **Resolved** — the ledger moved to Postgres (`parlay_tickets`/`parlay_legs`, migration 004). Set `PROPIQ_PARLAY_LEDGER=postgres`; the CSV backend remains the local default |
| `data/external/inactive_players/` | the scratch cache |
| `data/external/training_pack/`, `player_logs/`, `bigdataball/` | ingested panels, re-downloaded each boot |

Postgres holds `projections` and the market snapshots, so those survive.
Everything above does not. Combined with R2, a deployed container would run
slates and keep almost nothing.

**R2 — `prop_results` has no writer, so the feedback loop records nothing.**
Every P/L, ROI and CLV figure `metrics.py` can produce is an aggregate over
zero rows. A live test would generate no evaluable data — which is the
entire point of running one.

> **Resolved.** See "Dead-ended write path" above. One caveat survives: with
> no stake recorded, `metrics.py`'s stake and profit aggregates stay empty by
> design. Strike rate and CLV are the figures a shadow run now produces.

**R3 — No scheduler.** Nothing triggers a slate. Deploying gives a
container that starts, does nothing, and exits.

> **Resolved.** `scheduler_worker.py` runs an APScheduler `BlockingScheduler`
> on America/Los_Angeles: the slate at 09:00 PT (before any tip) and
> settlement at 03:30 PT (after any finish), both overridable by env var.
> Deploy it as a **worker** service — it binds no port, so a web service would
> be marked unhealthy for never listening. It does NOT re-anchor to the day's
> first tip-off: that needs a schedule feed, every data host is denied from
> the environment this was written in, and timing logic never exercised
> against real data would look adaptive while being untested.

**R4 — Late scratches are not handled at tip time.**
`ingestion/inactive_players.py` and `features/absences.py` are wired into
the builder, but as a *training-panel* feature: nothing drops a projection
when a player is ruled out after that projection was written. Also
`stats.nba.com` is denied (403) from this environment's proxy, so the
existing puller cannot be exercised here; ESPN's public injuries feed is
the reachable alternative.

> **Resolved.** `pipeline/scratches.py:apply_scratch_filter` labels every
> projection from ESPN's league-wide injury report, and `main.py` applies it
> right after `assemble_projections`. `settlement/recorder.py` skips a WITHHELD
> row, so a prediction on a player who will not dress is never written as one —
> settlement would VOID it, and a VOID row is backlog noise rather than evidence.
>
> **Four values, not two:** AVAILABLE / WITHHELD / UNKNOWN / **UNVERIFIED**. A
> feed that fails marks every row UNVERIFIED and drops nothing, because "we
> could not ask" is not "everyone is playing". Only WITHHELD is skipped
> downstream, so a bad afternoon at ESPN does not silently shrink the evidence
> base the publication gate is waiting on. Doubtful counts as unavailable.
>
> Matching is exact on a normalised name, never fuzzy: withholding the wrong
> player is worse than withholding nobody.
>
> **Never exercised live** — every ESPN host is denied from this environment, so
> this is fixture-tested only.

**R5 — Train/serve drift is unguarded.** `score_prob_over` loads
`feature_cols` from the model's `.meta.json` sidecar and refuses to score
when columns are missing, which catches *absence*. Column order and dtype
are unchecked because `feature_spec.py` is orphaned.

> **Resolved, and the finding's wording was too broad.** Order and dtype were
> in fact already enforced: `xgboost_pipeline._matrix` selects by `feature_cols`
> in order and refuses a column whose values will not parse as numbers. What
> nothing checked was whether the SIDECAR AND THE ARTIFACT AGREE — a
> `.meta.json` from one fit beside a booster from another. Both `save` methods
> already carry code to delete a stale *mean head* for exactly that reason; the
> classifier had no equivalent guard.
>
> `models/feature_spec.py:verify_feature_contract` now runs in
> `score_prob_over`, and both `save` paths write a fingerprinted `feature_spec`
> block. Two checks: the sidecar against its own fingerprint (catches a
> hand-edited or half-written file — no modelling library can see this), and the
> sidecar against the artifact's own column names.
>
> **What check 2 adds, measured rather than assumed:** on xgboost 3.2.0 a
> permuted or short column list already raises `feature_names mismatch`, so
> `score_prob_over` would have abstained anyway through its broad `except`. What
> changes is the reason an operator reads — xgboost's internal message does not
> say two artifacts came from different fits. It also does not depend on the
> library validating names, which matters if a positionally-indexed model family
> is ever put on the serving path.

**R6 — Rate limits and timeouts: already sound, no action.**
`propline.py` does 4 attempts with exponential backoff, honours
`Retry-After`, parses live quota from response headers, refuses to start
new work below `min_daily_remaining=5`, and does not retry 401/403.
`boxscores.py` uses 3 attempts with `retry_backoff ** attempt`;
`nba_playbyplay.py` mirrors it deliberately.

**R7 — Connection pooling: already sound, no action.**
`db/session.py` sets `pool_pre_ping=True`, `pool_size=5`,
`max_overflow=5`; `session_scope` commits on success, rolls back on
exception, closes in `finally`; no SQLite fallback by design; Supabase
pooler (6543) vs direct (5432) documented. Use the pooler URL for these
short-lived jobs.

**R8 — Concurrency.** Three simultaneous tip-offs means three concurrent
XGBoost/CatBoost fits. Both libraries default to all cores; unbounded on a
shared container this contends badly. (Observed in this session: three
concurrent test suites turned a 126s run into >590s.) Cap thread counts per
process or stagger the runs.

> **Resolved, both ways.** Each scheduled job runs with `max_instances=1` and
> `coalesce=True`, so an overrunning slate is never joined by a second copy and
> a backlog of misfires collapses into one. `PROPIQ_MAX_THREADS` (2 in the
> image) caps OMP, OpenBLAS, MKL, NumExpr and vecLib. Note the libraries
> default to every VISIBLE core, which on a shared container is the host's
> count and not this container's share — so the default is contention, not
> parallelism. Set it to match the plan's CPU allocation.

---

## 4. Final Deployment Verdict

**Status: DO NOT DEPLOY.**

**Explanation.** This is not a marginal call and it is not about code
quality — the model layer, the retry logic, the pooling and the Discord
dispatcher are all in good shape. It is that there is nothing deployable
yet and, if there were, it would not retain its own output.

Three blockers, each independently sufficient:

1. ~~**Nothing to deploy.**~~ **CLEARED.** `Dockerfile` (worker service, no
   port), `.dockerignore` (no `.env`, no `data/`), and `scheduler_worker.py`
   as the start command. **Still never built**, which is a separate thing from
   not existing: `scripts/validate_docker.py` now does the building and
   smoking, its 7 daemon-free preflight checks pass here, and its 6 in-image
   checks — including the two that execute the mounted-volume permission trap
   from `docs/deploy_railway.md` — run wherever a daemon is available.
   `tests/test_validate_docker.py` drives each preflight check to failure, so
   a PASS there means the check can fail.
2. ~~**Nothing would be recorded.**~~ **CLEARED.** `prop_results` now has a
   writer wired into `main.py`, so a shadow run produces gradeable picks and,
   once settled, a strike rate and CLV. P/L still requires a stake the user
   records by hand, which is intended.
3. **Nothing would survive a restart — STILL THE BLOCKER, now narrower.** The
   bet ledger is in Postgres. **The trained model artifacts are not.**
   `data/external/model_runs/comparison/` is still on an ephemeral filesystem
   with no volume (R1), so the first redeploy silently un-trains the models
   and `score_prob_over` abstains on every row. Mount a volume at `/app/data`
   or load the artifacts from object storage at boot.

Minimum to reach WARNING (deployable for shadow testing):
~~write `prop_results`~~ (done); ~~move the parlay ledger into Postgres~~
(done); ~~add a Dockerfile plus a PT-anchored schedule~~ (done); and **either
declare a Railway volume for `data/external/model_runs/` or load model
artifacts from object storage** — the one step left, and the only one that
cannot be taken from inside this repository.

Minimum to reach READY: the above, plus ~~a pre-tip scratch filter (R4)~~
(done), ~~the `FeatureSpec` fingerprint wired at train and serve (R5)~~ (done),
and ~~thread caps on concurrent fits (R8)~~ (done). **All of READY's own items
are closed**; the WARNING blocker above (model artifacts on an ephemeral
filesystem) is the only thing outstanding, and it is a platform step.
