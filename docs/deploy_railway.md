# Deploying the worker (Railway or any container host)

RESEARCH_ONLY. The deployed worker runs research jobs on a clock. It places no
wager, contacts no operator's order API, and sizes no stake.

Read `docs/railway_deployment_audit.md` first for what was wrong and what is
still wrong. **One blocker remains** and it is at the bottom of this page.

---

## 1. Service type

Deploy as a **worker**, not a web service.

`scheduler_worker.py` binds no port. A service configured as a web app is
marked unhealthy for never listening, and the platform will restart it in a
loop — which looks like a crashing app and is not one. This is the most likely
way a first deploy goes wrong.

Three files in the repository root now carry that, so it is version-controlled
rather than typed into a dashboard:

| File | What it fixes |
|---|---|
| `railway.json` | `builder: DOCKERFILE`, `startCommand: bash scripts/start.sh`, `numReplicas: 1`, and **no `healthcheckPath`** — a worker that binds no port would be polled, fail, and restart forever |
| `Procfile` | the buildpack fallback. Declares `worker:` and deliberately **no `web:`** |
| `scripts/start.sh` | the start sequence: migrate → probe → `exec` the worker |

**`numReplicas` must stay 1.** APScheduler's `max_instances=1` is per-process,
so a second replica is a second scheduler with its own clock: two ingests, two
boards, two Discord dispatches of the same card, two settlement passes. There
is no distributed lock in this project.
`tests/test_railway_deploy.py::test_exactly_one_replica_is_configured`.

## 1b. What happens at boot

`scripts/start.sh`, in this order, and the two failure policies differ on
purpose:

1. **`python -m scripts.run_migrations`** — applies pending migrations in one
   transaction. **Hard-fails the container.** A schema behind the code does not
   announce itself; it surfaces at 09:00 PT as a failed insert, hours after the
   deploy looked successful. Exiting nonzero makes the platform show a failed
   deploy, which is the honest signal. Set `PROPIQ_MIGRATE_ON_BOOT=false` where
   a separate release step owns the schema.
2. **`python -m scripts.railway_healthcheck`** — probes the volume, the
   artifacts, the migration ledger, the parlay ledger and the calibration path.
   **Reports and carries on.** It is strict (an unseeded volume is a FAILURE
   there) and the worker's own policy is the opposite and also correct: a
   worker that refuses to start cannot report anything, and the settlement job
   is still useful with no model.
3. **`exec python scheduler_worker.py`** — `exec` so the worker is PID 1 and
   SIGTERM reaches its handler, which shuts the scheduler down *after* the
   running job rather than being SIGKILLed mid-slate.

Run the probe yourself when you want the exit code to mean something:

```
railway run python -m scripts.railway_healthcheck          # 0 OK / 1 FAILURE
railway run python -m scripts.railway_healthcheck --json
python -m scripts.railway_healthcheck --skip-db            # no database needed
```

It calls the real resolvers — `src.utils.volume.resolve_state_root`,
`main.resolve_model_artifact`, `src.db.migrations.status`,
`parlay_log.resolve_ledger_choice`, `scheduler_worker.board_markets` — rather
than globbing paths of its own. A probe that invents its own layout reports
FAILURE on a correct deployment, which is worse than no probe: it gets the
deployment "fixed" in the direction of its own mistake.

## 2. Build

The repository root has a `Dockerfile`; point the service at it.

**It was built on 2026-10-10** — the first time. This page said "it has never
been built … treat the layer ordering and the apt package list as reasoned, not
verified" for months, and the apt package list is precisely what turned out to
be wrong. All 15 checks now pass:

```
python -m scripts.validate_docker                        # preflight, build, smoke
python -m scripts.validate_docker --preflight            # the daemon-free half
python -m scripts.validate_docker --skip-build --tag <t> # smoke an existing image
```

**Phase A, 9 checks, no daemon**: the CMD target exists, every `python -m`
inside it resolves to a real module, a `RUN` layer import-checks the ML extra,
no credential is defaulted in a layer, every installed extra is declared, the
uid is the one this page tells you to chown to, and the ignore file is
evaluated by matching rather than grepped.

**Phase C, 6 checks, inside the built image**: the ML and deploy extras import,
the scheduler builds its job table, the process runs as uid 10001, there is no
`.env` in the image — and **two that exist to test this page**: they mount a
mode-0555 directory at `/app/data` and require `check_state_dir()` to report it
unwritable, then mount a writable one and require it not to false-alarm. Both
pass, so **the trap in §2b below is now demonstrated and not merely
documented.**

The container was then booted end to end: `scripts/start.sh` ran the
migrations, ran the probe, and started the scheduler in `America/Los_Angeles`
with both cron jobs and their misfire graces. Both failure policies were
exercised for real — migrations with no `DATABASE_URL` exit 2 and the worker
never starts; a probe FAILURE is logged and the worker starts anyway.

**One caveat, stated precisely.** The session that built it routes HTTPS
through a CA-re-terminating proxy, so pip inside a build container sees a
self-signed chain for pypi.org. Baking that CA into the image would be wrong —
Railway has no such proxy — so the build used a copy of the `Dockerfile` with
exactly two extra instructions before the pip layer (`COPY` the CA, `ENV
PIP_CERT`). Every other instruction is byte-identical to the committed file.
The first build on a host with ordinary egress exercises the unmodified file.

### The `libgomp1` apt layer is gone, and the comment justifying it was wrong

It read: *"OpenMP, which xgboost and catboost link against. Without it the
image builds and then fails at import."* Reasoned from the libraries' linkage,
and false for these wheels. The first build settled it:

```
$ docker run --rm python:3.11-slim sh -c \
    'pip install xgboost catboost; python -c "import xgboost, catboost"; \
     find / -name "libgomp*"'
xgboost 3.2.0 · catboost 1.2.10
/usr/local/lib/python3.11/site-packages/xgboost.libs/libgomp-e985bcbb.so.1.0.0
```

The **wheel vendors its own libgomp**. The apt layer installed a second copy of
a library nothing loaded, and it was the only thing in the build that needed
the Debian package index — so on a host whose egress policy does not allow
`deb.debian.org`, it failed the build at stage 2 of 8 for a dependency the
image does not have.

**The guarantee moved rather than disappearing.** Vendoring is a property of
the wheel, not of this project: a future release could stop doing it, or a
platform with no wheel could build from source and need system OpenMP. A
`RUN python -c "import xgboost, catboost, sklearn"` layer now asserts at build
time what the apt layer only assumed, and phase A check 7 asserts that layer is
still there — so if a future wheel stops vendoring OpenMP the build goes red
instead of the first unattended inference.

**Point the service at the Dockerfile, not Nixpacks.** Nixpacks reads
`requirements.txt` rather than `pyproject.toml`'s extras. `APScheduler` is now
in both, so a Nixpacks build no longer produces a worker that exits on import —
but there is one build path that is tested by inspection and it is this file.

It installs the `ml`, `db` and `deploy` extras. The ML extras are optional in
`pyproject.toml` so a local checkout stays light — a minimal image would start
cleanly and then fail at the first inference, unattended, after the slate had
already been ingested. Failing at build time is the better trade.

`.dockerignore` keeps `.env` and `data/` out of the build context. A credential
in an image layer survives every later layer that deletes it.

## 2b. The volume, and the one trap in it

Mount a persistent volume at **`/app/data`**. Two things need it:

| Path | Why |
|---|---|
| `data/external/model_runs/comparison/` | the trained artifacts `score_prob_over` loads. Empty on a fresh container → **every row abstains**, and the pipeline does not look broken |
| `/app/data/calibration.json` | the evidence the publication gate reads. Settlement writes it 03:30 PT, the slate reads it 09:00 PT — on the ephemeral layer a redeploy between the two withholds every card for a reason that is not the real one |

**`RAILWAY_VOLUME_MOUNT_PATH` is now read.** Until 2026-10-09 it was read
nowhere in this repository: every path was relative to the working directory,
which is correct in the image *only because* the Dockerfile puts the volume at
`/app/data`. A deploy that mounted it anywhere else wrote to the container
filesystem instead, succeeded, and lost everything at the next redeploy — the
same silent shape as the parlay ledger's CSV default.
`src/utils/volume.resolve_state_root()` resolves, most explicit first:

1. `PROPIQ_STATE_DIR` — an operator override, ahead of everything;
2. `RAILWAY_VOLUME_MOUNT_PATH` — the platform's own answer;
3. `/app/data` when the directory exists — the Dockerfile's mount point;
4. `<repo>/data` — a developer machine with no volume.

It returns *how* it chose, not just the path, and the healthcheck prints it.
"`/app/data` via `RAILWAY_VOLUME_MOUNT_PATH`" and "`/app/data` because the
directory exists" are the same path and different deployments; only one of them
survives a redeploy, and the probe WARNs on the second.

Nothing was relocated: `artifacts_dir` is still
`data/external/model_runs/comparison`, which under a volume at `/app/data` is
already on the volume. **There is no `/app/data/models/` directory** — this
project writes `xgboost_{MARKET}.json` plus a `.meta.json` sidecar into
`artifacts_dir` and nothing else, so a tool looking for `.joblib` or `.pkl`
files under `models/` finds nothing on a correctly seeded volume.

**THE TRAP: a mounted volume shadows the image's `chown`.** The Dockerfile
creates `/app/data` and chowns it to `propiq` (uid 10001), but a volume mounted
there at run time replaces that directory with whatever the platform
provisions — commonly root-owned. The container runs as uid 10001, so the first
write fails with `EACCES`, and because every writer catches broadly you would
see *"calibration report failed"* nightly rather than a permissions problem.

`scheduler_worker.check_state_dir()` probes it at boot and says so. If it fires,
the first log line is an ERROR naming the path and the uid. Two fixes:

1. `chown` the volume to uid 10001 (a one-off `railway run chown -R 10001:10001 /app/data`, or an init step).
2. Run the service as root — drop the `USER propiq` line. Smaller blast radius is worth keeping, so prefer (1).

Nothing is lost while it is unwritable: projections and `prop_results` still go
to Postgres. What stops is the calibration evidence, so every card is withheld.

## 3. Environment

Set these in the platform, never in the image. `.env.example` is the full list;
these are the ones the worker reads.

| Variable | Why |
|---|---|
| `DATABASE_URL` | Required. Use the **pooler** URL (port 6543 on Supabase) — these are short-lived jobs and the direct port exhausts connections. |
| `PROPIQ_PARLAY_LEDGER=postgres` | Already set in the image (Dockerfile line 73). Without it the ledger writes CSVs to an ephemeral disk and a redeploy destroys every ticket's at-bet-time probability and EV. Since 2026-10-09 the worker reports which backend it resolved in the first lines of its log, at ERROR when it is csv while a database is configured — the shape that loses tickets silently, because the CSV writes succeed. |
| `PROPLINE_API_KEY` | The odds source. Without it no line is captured, so every row abstains for want of a market. |
| `PROPIQ_MODEL` | Which fitted artifact to score with. Unset, `main.resolve_model_artifact` tries the comparison `artifacts_dir` for the newest `xgboost_*.json` that has its `.meta.json` sidecar. Set it when the artifact is on the mounted volume. |
| `PROPIQ_FORWARD_SLATE` | Default **on**. Adds rows for games that have not been played, from the ESPN schedule plus each team's recent appearances in the panel. Without it a 09:00 PT run has no rows for tonight — the panel is completed box scores only. |
| `DISCORD_WEBHOOK_URL` | Only if you dispatch. Never logged or printed. |
| `PROPIQ_MAX_THREADS` | Defaults to 2 in the image. Match your plan's CPU allocation — the numeric libraries otherwise see the host's core count, not the container's share. |
| `TZ` | Set to `Etc/UTC` in the image so the container default is a decision rather than the host's. The slate schedule does **not** depend on it: the scheduler passes `America/Los_Angeles` to APScheduler directly. |
| `PROPIQ_SLATE_HOUR_PT` / `_MINUTE_PT` | Defaults 09:00 PT, before any tip. |
| `PROPIQ_SETTLE_HOUR_PT` / `_MINUTE_PT` | Defaults 03:30 PT, after any finish. |
| `PROPIQ_RUN_ON_START` | Off by default. Set it for a single first run; leaving it on means a redeploy loop re-runs the slate each time. |
| `PROPIQ_DISPATCH` | Defaults to on when `DISCORD_WEBHOOK_URL` is set. |
| `PROPIQ_DISPATCH_ABSTENTIONS` | Default on — the house rule is that a refusal is posted too, since silence reads as "nothing good today". |
| `PROPIQ_CALIBRATION_REPORT` | Where the settlement job writes the evidence and the slate job reads it. Default `outputs/calibration.json`. **On a container this must be on the persistent volume**, or the morning run will not see what last night graded. |
| `PROPIQ_BOARD_CSV` | Default `outputs/decision_board.csv`. |
| `PROPIQ_BOARD_TRAIN_END` / `_VALIDATION_END` / `PROPIQ_BOARD_MARKETS` / `PROPIQ_MIN_EV` | Board build parameters, matching the `decision-board` CLI defaults. |
| `PROPIQ_STATE_DIR` | Override where durable state resolves to, ahead of `RAILWAY_VOLUME_MOUNT_PATH`. Unset is right on Railway; set it on a host that mounts the volume somewhere the platform does not announce. |
| `BIGDATABALL_XLSX` | The licensed team-stats export. **Needed once, not per deploy** — see §3b. |
| `PROPIQ_MAX_MARKET_LAG_DAYS` | Default 10. How far the newest `team_game_stats` row may trail the slate before the run logs **MARKET DATA IS STALE**. Looser than the player panel's 3 because the slate refreshes the panel itself every run while the market frames come from a workbook somebody uploads. |
| `PROPIQ_MIGRATE_ON_BOOT` | Default **on**. `scripts/start.sh` applies pending migrations before the worker starts and hard-fails the container if it cannot. Set false where a release step owns the schema. |
| `PROPIQ_DB_POOL_RECYCLE` | Default 1800s. SQLAlchemy's own default is *never*, and this worker leaves a pooled connection idle for the eighteen hours between 09:00 and 03:30. |
| `PROPIQ_DB_CONNECT_TIMEOUT` | Default 10s, passed to libpq. **Without it there is no bound**: a pooler that accepts the TCP connection and never completes the handshake blocks the caller forever, and the slate job has no timeout of its own — it would hang past every tip-off and be noticed as a worker that produced nothing, with no error anywhere. |
| `PROPIQ_DB_POOL_SIZE` / `_MAX_OVERFLOW` / `_POOL_TIMEOUT` | Defaults 5 / 5 / 30. Bounded; an out-of-range or mistyped value falls back rather than raising. |

### Variables this page deliberately does **not** list

Three were asked for and are not here, because documenting a variable the code
does not read is how an operator comes to believe a key is in use:

| Asked for | Why not |
|---|---|
| `SPORTSDATA_API_KEY` | **Appears nowhere in this repository.** No client, no reader, no default. The odds source is `PROPLINE_API_KEY`. |
| `THE_ODDS_API_KEY` | Same, and worse: the Odds API (`ODDS_API_KEY`) is recorded as a **banned sportsbook source** in `docs/external_feature_harvest.md` and `docs/pickem_props.md`. Setting it would not wire anything up; listing it would advertise a source this project has decided against. |
| `TZ=America/Los_Angeles` | The image sets `TZ=Etc/UTC` **on purpose** and that should not change. Storage is UTC throughout, and every slate cutoff is a Pacific *calendar day* that `scheduler_worker` gets by passing `America/Los_Angeles` to APScheduler explicitly — so the schedule does not depend on `TZ` at all. What `TZ` governs is the naive `datetime.now()` calls elsewhere, which should stay UTC and reproducible. Setting it to Pacific would make those calls shift twice a year while the stored timestamps did not, and the resulting off-by-an-hour rows would look like data, not like a setting. |

## 3b. The BigDataBall workbook: once, not per deploy

**This used to fail every scheduled run on a fresh container, and it was the
largest blocker in a deploy — not the model.** `main.ingest_market_lines`
reads the licensed export and was unguarded: a missing path raised
`FileNotFoundError`, which the orchestrator turned into a FAILED
`pipeline_runs` row and exit 1. The whole slate, not a degraded one. And
`.dockerignore` excludes `data/` and `*.xlsx` **deliberately** — a licensed
third-party export does not belong in an image layer — so the container never
had one.

**The data was never missing; only the file was.** Every run that finds a
workbook calls `upsert_team_game_stats` and `upsert_market_lines`, so the
contents are in Postgres, which is the thing that survives a redeploy.
`main.resolve_market_frames` reads them back through
`repository.load_team_game_stats` / `load_game_market_lines`, under the **same
column names** the Elo, defence and market-context layers already accept — so
the fallback frame is interchangeable with the workbook's.

So the step is:

```
# ONCE, where the licensed export is, against the deployment's database:
BIGDATABALL_XLSX=/path/to/2025-2026_NBA_Box_Score_Team-Stats.xlsx python main.py
```

After that a container needs no file. Three outcomes, and the probe reports
which:

| | What step [2] does |
|---|---|
| workbook on disk | reads it, upserts it — the fresher source |
| no workbook, rows in the database | reads them back, logs `NO BIGDATABALL WORKBOOK` naming the fallback |
| no workbook, no rows | **refuses**, naming both the path and the empty tables. 11 of the 38 columns a seeded contract names cannot be built, so every row would abstain — the cause is upstream of the columns, so step [2] says it rather than letting the scorer report missing column names |

**The new thing to watch is staleness, not absence.** The database holds
whatever was last ingested, so a container running for weeks on one upload
computes Elo and `DEF_*` from games that stop before the rows it is scoring —
the layers attach, the columns are present, the contract check passes, and the
numbers are out of date. `main.market_frames_freshness` measures it from the
frame itself and logs **MARKET DATA IS STALE** past
`PROPIQ_MAX_MARKET_LAG_DAYS`. It measures a workbook run too: a stale file on
disk is the same defect with a different cause.

**Only pregame columns are ever read.** `game_market_lines` also stores
`closing_spread`, `closing_total`, the halftime line and three line-movement
columns. `load_game_market_lines` selects `market_context.PREGAME_SOURCE_COLS`
and nothing else, and asserts no closing column came back —
`attach_market_context` happens to project down to the pregame set, so a
`SELECT *` would not leak today, and selecting them anyway would make that
projection the only thing standing between a closing line and a feature matrix.

## 4. Database

**On a container this is automatic**: `scripts/start.sh` runs
`python -m scripts.run_migrations` before the worker starts and hard-fails the
container if a migration cannot be applied. What follows is the manual path,
and what the boot step does.

```
python -m scripts.migrate_db                    # what this database has
python -m scripts.migrate_db --apply --dry-run  # what WOULD be applied
python -m scripts.migrate_db --apply            # apply, in one transaction
python -m scripts.run_migrations                # the same thing, applying by default
python -m scripts.run_migrations --dry-run
```

**Two front doors, one implementation.** `scripts/run_migrations.py` calls
`scripts.migrate_db.main` with `--apply` prepended and contains nothing else —
it does not import `src.db.migrations` and `tests/test_railway_deploy.py`
AST-walks it to keep that true. The reason for two is the default, not the
logic: `migrate_db` must stay read-only where a human is typing, because a
migration tool that writes by default is one typo away from the wrong
`DATABASE_URL`; a container start command is the opposite case, where nobody
is there to pass a flag.

**`--ensure-tables` IS NOT OPTIONAL ON A FIRST DEPLOY**, and that was learned
by doing one against a real Postgres on 2026-10-10. The SQL migrations `ALTER`
tables that `Base.metadata.create_all` is what creates (`projections`,
`player_game_logs`, `prop_line_snapshots`), and `ADD COLUMN IF NOT EXISTS` does
not help when the *table* is absent — 002 fails with `relation "projections"
does not exist`, the runner returns 4, and `scripts/start.sh` hard-fails the
container. **The worker never starts.** `scripts/run_migrations` therefore
always passes it; it is additive and idempotent, so later boots do nothing.

Three more defects surfaced the same way, all of which had passing tests
against SQLite:

* a `%` in a migration **comment** made the file unapplicable —
  `exec_driver_sql` hands the script to psycopg as a format string, so
  `-- a win% that includes pushes` raised `incomplete placeholder: '%'`. The
  runner had never applied a migration to Postgres;
* 002 and 003 wrap themselves in `BEGIN; … COMMIT;`, which is right for `psql`
  and ends the transaction this runner opened — so the "whole run is one
  transaction" guarantee was false. The runner now strips whole-line
  transaction control, leaving `$$` bodies alone;
* every bulk upsert built one statement for every row, over Postgres'
  **65,535 bind-parameter** limit. A season's workbook is 2,644 team-games ×
  25 columns, and `upsert_player_game_logs` runs *daily* on a whole season.
  All seven writers now share one `batched()`.

Verified end to end: 7 migrations applied to an empty database, then 2,644
workbook rows ingested and read back with no workbook on disk.

**This list used to be written out here, and that was the defect.** It named
002 through 005 and went stale the moment 006 was added, so the only way to
learn whether a database had 007 was to query for the column it adds and
infer. `scripts/migrate_db.py` reads the directory, applies what is missing in
version order, and records each file with a checksum in a `schema_migrations`
table — so the database answers the question instead of a document. Status
reporting is the default; `--apply` is required to change anything.

A file edited after it was applied is refused by name rather than re-run: the
checksum catches the one divergence a version number cannot see. There is no
`alembic`; `src/db/migrations.py`'s docstring says why that was the choice.

`python main.py --init-db` creates the ORM-defined tables and exits. The SQL
migrations carry the CHECK constraints and views that `create_all` does not.

## 4b. Seeding the scoring artifact

`data/external/model_runs/` is empty on a fresh container and **every row
abstains** until it is not. `scheduler_worker.check_model_artifact()` reports
that as an ERROR in the first lines of the log, so the state is visible — but
only seeding fixes it.

One command per launch market, run where the panel is:

```
python -m scripts.nba_model_cli train-stats --market PTS \
    --panel <feature-matrix>.parquet \
    --start-date 2018-01-01 --end-date <today> --train-end <today minus ~3 months>
python -m scripts.nba_model_cli train-stats --market REB --panel ...  (same)
python -m scripts.nba_model_cli train-stats --market AST --panel ...  (same)
```

**Pass `--train-end`.** Without it the window is split 2/3 **by row count**,
which ties how much history you train on to how much is withheld: a
2018 → 2026 run fits 142,713 rows and holds out 71,357, so the fit stops in
December 2023. With it the same window fits 200,678 and reaches the present.
Keep a holdout — the components calibrate on it — but a few months is enough.

Artifacts land under `artifacts_dir` from `config/model_comparison.yaml`
(`data/external/model_runs/comparison`), which is where the scoring resolver
looks. `--panel` takes a prebuilt feature matrix and exists because the
loader's own path needs stats.nba.com — denied in CI and in the cloud sessions
this project is developed from, where the artifact previously could not be
trained at all. `scripts/ingest_training_pack.py` builds the matrix.

Each market writes `xgboost_{M}.json` + `.mean.json` + `.meta.json`,
`catboost_{M}.cbm` + `.mean.cbm` + `.meta.json`, and `distribution_{M}.json`.
The `.meta.json` sidecar is the feature contract the scorer checks before using
the booster; a permuted column list is rejected rather than silently
re-ordered.

**THE ARTIFACTS ARE GITIGNORED** (`data/**`), so they travel by volume or
object storage, never by a commit.

### Moving them onto the volume

```
python -m scripts.seed_volume                                       # what is there
python -m scripts.seed_volume --from data/external/model_runs/comparison
python -m scripts.seed_volume --from data/external/model_runs/comparison --apply
```

The destination is `src.utils.volume.artifact_dir_on(state_root)`, which reads
the same `artifacts_dir` the scoring resolver does — a seeding tool that wrote
to a directory of its own choosing would report success and leave the resolver
finding nothing.

**One market is several files, and it copies all of them.** `xgb_adapter.load`
sets `mean_model = None` when `.mean.json` is absent and only warns, so a
volume seeded with the booster alone returns **null projections beside live
probabilities** — the failure that looks most like a working deployment. Each
accepted market copies its whole family (`*_PTS.*`), which carries the mean
head, the sidecar, `distribution_PTS.json` and the CatBoost pair.

It refuses, rather than copying: an artifact with no `.meta.json` sidecar
(scoring is refused without the feature contract, so it is a file and not a
model); an artifact fit on fewer than 5,000 rows, which is the synthetic demo
one, unless `--allow-demo`; and overwriting anything already on the volume
without `--force`, because that artifact produced every probability now in the
database.

**Two things to check after seeding**, both measured on the 2026-10-09 seed:

- The contract must be buildable where it scores. A PTS contract trained with
  the BigDataBall workbook present resolves **38** columns; a live panel built
  without the workbook resolves **27**, and the 11 missing ones (Elo, `MKT_*`,
  `DEF_*`) come from the workbook alone. Train and score with the same inputs,
  or the contract check rejects the artifact on every row.
- Use `--train-end`, per above. And do **not** narrow `--start-date` to buy
  recency: it was measured and it costs accuracy. On one common validation
  window the ensemble's calibrated Brier degrades monotonically as the start
  moves forward — 0.24098 for the full window, 0.24268 from 2024-10 — and two
  seasons of history is the worst configuration for every model.
  `docs/training_window.md`.

## 5. What runs, and when

| Job | Time (PT) | What it does |
|---|---|---|
| `slate` | 09:00 | ingest → features → score → EV gate → persist projections → write PENDING `prop_results` → build the board CSV → dispatch it to Discord, gated |
| `settlement` | 03:30 | grade every PENDING prop whose game has finished → **rebuild the calibration report** → post the previous Pacific day's results card |

The order between them is the dependency: settlement grades last night's games
and recomputes the calibration evidence, so the morning slate reads evidence
that already includes those results rather than evidence a day stale.

Both run with `max_instances=1`: an overrunning job is never joined by a second
copy. A missed slate is **not** run late (one-hour grace) — running late would
project games that have already tipped. A missed settlement is, up to six
hours; a finished game stays finished.

The schedule is fixed, not tip-off-driven. Re-anchoring needs a schedule feed,
and timing logic that has never been exercised against real data would look
adaptive while being untested.

## 5b. Does your own computer need to be on?

**No, if you deploy.** That is what deploying is for. The worker runs in the
platform's container, Postgres is hosted (the `.env.example` default is a
Supabase pooler URL), and the Discord card is posted by the container. Your
machine can be off, asleep, or on a plane at 09:00 PT and the slate still runs.

Four things still need a machine, and none of them is continuous:

| | When | Why |
|---|---|---|
| `python -m scripts.validate_docker` | once, before the first deploy | the build and the six in-image checks need a Docker daemon |
| `python main.py --init-db` | once, per database | creates the ORM-defined tables. The SQL migrations are applied **at boot** by `scripts/start.sh`, so they are no longer a step here — `python -m scripts.migrate_db` remains the way to *inspect* what a database has |
| `nba_model_cli ingest-logs --persist` | **no longer needed per day** | the slate refreshes `player_game_logs` itself (step [3b]). Still useful once to seed history, and to backfill seasons the slate does not touch |
| training model artifacts | once, then whenever you retrain | `data/external/model_runs/` is empty on a fresh container and **every row abstains**. Train into the mounted volume, or train locally and `python -m scripts.seed_volume --from ... --apply` against it. There is still **no boot-time fetch** in this repository — no S3 client, no storage client, nothing that downloads a model |
| recording a stake | whenever you place a bet | PropIQ never places one and never writes a stake. ROI exists only if you log it |

**Yes, if you do not deploy.** Running `python scheduler_worker.py` on your own
machine means the machine must be awake and the process running at both cron
times, 09:00 and 03:30 Pacific. A laptop asleep at 03:30 does not grade last
night's props, which means the 09:00 slate reads day-old calibration evidence.
And the grace periods are deliberately asymmetric:

- **slate: one hour.** A missed slate is **not** run late, because projecting
  games that have already tipped is worse than projecting none.
- **settlement: six hours.** A finished game stays finished, so catching up is
  harmless.

So a local run that wakes at 11:00 silently skips that day's board. That is the
correct behaviour and it is also the reason to deploy rather than self-host.

## 6. What a deployed run produces

- `projections` — the pipeline's deliverable.
- `prop_results` — one PENDING row per prediction with a line, a probability
  and a source, graded by the settlement job. **These are predictions, not
  wagers:** no stake is written and none can be. Strike rate and CLV come out
  of them; ROI does not, and will not until you record a stake yourself.
- `outputs/calibration.json` — the calibration report built from those graded
  rows by `prop-calibration` (and by the settlement job). This is the file the
  publication gate reads, and until it says `status: OK` with enough scored rows
  every dispatched card is withheld.
- `parlay_tickets` / `parlay_legs` — only what you log by hand.

**Dispatch is automatic, and gated.** The slate job builds the board and sends
it. Every board row rests on the model's own probability — a `book_ev` row
*measures* that probability against the market price rather than using the
market's — so the board is gated as MODEL-sourced by
`src/quant/publication_gate.py`. With no calibration evidence the card carries
the gate's reason instead of the rows, which is the correct output while nothing
is settled, not a failure.

Dispatch is on when `DISCORD_WEBHOOK_URL` is set. `PROPIQ_DISPATCH` forces it
either way; `PROPIQ_DISPATCH_ABSTENTIONS=false` stays silent on a withheld or
empty board instead of saying so.

---

## The remaining blocker

One, where there were two on 2026-10-09. It is *visible* rather than silent,
which is a different thing from being fixed.

**1. The volume still has to be mounted and seeded by a person.** Nothing in
this repository can mount a volume or put a model on one. What changed on
2026-10-09 is that the state root is resolved from
`RAILWAY_VOLUME_MOUNT_PATH` and reported, `scripts/seed_volume.py` exists to do
the copying, and `scripts/railway_healthcheck.py` exits **nonzero** on an
unseeded volume — so the "pipeline runs, writes nothing useful, does not look
broken" state now fails a deploy gate instead of producing a week of empty
boards. Mount at `/app/data`, seed, and run the probe before trusting a slate.

**There is still no boot-time fetch.** No S3 client, no storage client, nothing
in this repository downloads a model. An earlier version of this page offered
"fetch from object storage at boot" as option 2; that code does not exist and
listing it invited a first deploy that assumed the container would help itself.

**2. ~~The image has never been built.~~ CLEARED 2026-10-10.** All 15 checks
pass against the built image, the container boots and schedules, and the
`chown` trap above is now *demonstrated* rather than documented. §2 has the
detail and the one caveat (the copy that was built carried two extra
pip-trust instructions).

So the honest state: **one blocker remains, and it is the volume.** Mount it,
ingest the workbook once (§3b), seed the artifacts (§4b), and run the probe.
A deployment before that is a wiring test rather than a shadow run; after it,
the nightly loop runs unattended.
