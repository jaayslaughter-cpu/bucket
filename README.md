# PropIQ Analytics

Pregame NBA player-prop research pipeline: Python services that ingest
licensed box-score data and live prop lines, build leakage-safe features,
score them with XGBoost (plus a CatBoost challenger, a distribution baseline,
and an ensemble), convert projections into calibrated over/under
probabilities, compare them against de-vigged market prices, and dispatch a
decision board to Discord — running as an autonomous Railway worker backed by
PostgreSQL. Single-stat markets (PTS, REB, AST, FG3M, STL, BLK), combination
markets (PRA, PA, PR, RA), and DFS fantasy points are in scope; Underdog-style
streak entries are tracked and graded.

**NBA only. RESEARCH_ONLY** — this repository does not place bets, automate
wagering, or recommend wagers. Nothing here claims a model is profitable, and
nothing should be treated as such until it has been verified on leakage-safe
forward data. `AGENTS.md` §1 lists each hard line and the exact line of code
that enforces it.

## Architecture

```
                    ┌─────────────────────────────────┐
                    │  RAILWAY WORKER                 │
                    │  scheduler_worker.py (no port)  │
                    └─────────────────────────────────┘
    03:30 PT settle job                    09:00 PT slate job
    (post-game grading)                    (daily projections)
            │                                      │
  ┌─────────────────────┐              ┌─────────────────────┐
  │ INGESTION           │              │ INGESTION           │
  │ BigDataBall workbook│              │ PropLine odds API   │
  │  (licensed, manual) │              │ Sleeper / Underdog /│
  │ ESPN box scores     │              │ pick'em props       │
  └─────────────────────┘              └─────────────────────┘
            │                                      │
  ┌─────────────────────┐              ┌─────────────────────┐
  │ PLAYER PANEL        │◄──── merge ──│ FEATURE MATRIX      │
  │ (player game logs)  │              │ build_feature_matrix│
  └─────────────────────┘              │ (shift-1, pregame)  │
            ▲                          └─────────────────────┘
            │                                      │
  ┌─────────────────────┐              ┌─────────────────────┐
  │ TEAM GAME STATS     │              │ COMPOSITE ROLLUPS   │
  │ licensed workbook   │              │ PA/PR/RA/PRA from   │
  │ OR aggregated from  │              │ components          │
  │ player box scores   │              └─────────────────────┘
  └─────────────────────┘                         │
                                       ┌─────────────────────┐
                                       │ MODEL SCORING       │
                                       │ resolve_model_      │
                                       │   artifact()        │
                                       └─────────────────────┘
                                                  │
                                       ┌─────────────────────┐
                                       │ OUTCOME DISTRIBUTION│
                                       │ P(over)/P(under)/   │
                                       │ P(push) at the line │
                                       └─────────────────────┘
                                                  │
  ┌─────────────────────┐              ┌─────────────────────┐
  │ DE-VIG              │─────────────►│ PUBLICATION GATE    │
  │ multiplicative/shin/│              │ calibration evidence│
  │ odds_ratio/log      │              │ (n_scored, ECE)     │
  └─────────────────────┘              └─────────────────────┘
                                                  │
                                       ┌─────────────────────┐
                                       │ DISCORD DISPATCH    │
                                       │ webhook, redacted   │
                                       └─────────────────────┘
                                                  │
  ┌─────────────────────┐              ┌─────────────────────┐
  │ CALIBRATION REPORT  │◄─────────────│ SETTLEMENT          │
  │ (feeds the gate)    │              │ grading, CLV, ledger│
  └─────────────────────┘              └─────────────────────┘
```

The loop is closed: settlement writes the calibration report, and the
publication gate reads it before the next slate dispatches. A system with no
graded games withholds every card, by design — abstaining is a state, not a
failure.

## Layout

| Path | What lives there |
|---|---|
| `main.py` | Orchestrator — one pipeline run; `resolve_model_artifact()` lives here |
| `scheduler_worker.py` | The deployed worker: APScheduler jobs, dispatch, settlement runner |
| `src/ingestion/` | Data loaders: BigDataBall workbook, scraper parquet, PropLine, Sleeper, Underdog, pick'em, ESPN |
| `src/features/` | Feature layers: rolling stats, fatigue, DvP, hot hand, minutes-weighted, defence |
| `src/models/` | XGBoost adapter, CatBoost challenger, distribution model, ensemble, calibration, composite features, feature specs |
| `src/quant/` | De-vig methods, EV engine, advisory (Kelly) sizing, streak engine, publication gate, parlay ledger |
| `src/notify/` | Discord webhook dispatch, payload safety, redaction |
| `src/settlement/` | Post-game grading: box-score fetch, push-aware settlement, CLV, ROI |
| `src/db/` | SQLAlchemy schema (13 tables), engine, repository |
| `src/pipeline/` | Slate board, forward slate, scratches |
| `scripts/` | Diagnostics, ingestion, migration entry points (see below) |
| `config/` | `model_comparison.yaml` (seeds, weights, hyperparameters), `fantasy_scoring.yaml` DFS payouts |
| `migrations/` | SQL migrations, applied and recorded by `python -m scripts.run_migrations` |
| `tests/` | pytest suite (~3,100 tests) |
| `docs/` | Design docs, audits, deployment guide |

## Quickstart

```
git clone https://github.com/jaayslaughter-cpu/bucket.git
cd bucket
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[ml,db,dev]"                       # or: pip install -r requirements.txt
cp .env.example .env                                # then fill in DATABASE_URL
```

Set up PostgreSQL and point `DATABASE_URL` at it, then create the schema:

```
python -m scripts.run_migrations    # applies pending migrations, once per database
python -m scripts.migrate_db        # read-only ledger status
```

Seed team-game data. Either source works; the second needs no licensed file:

```
# (a) the licensed BigDataBall workbook — never committed, once per database
mkdir -p data/external/bigdataball
cp ~/Downloads/2025-2026_NBA_Box_Score_Team-Stats*.xlsx data/external/bigdataball/
python -m scripts.ingest_bigdataball --xlsx data/external/bigdataball/*.xlsx

# (b) from the scraper's parquet — player box scores summed into team rows
python -m scripts.load_scraper_panel --canonical-dir ../nba-scraper-2026-27/data \
    --team-games --dry-run      # reports what it WOULD write, needs no database
python -m scripts.load_scraper_panel --canonical-dir ../nba-scraper-2026-27/data \
    --team-games                # writes, then reads its own rows back
```

Run one slate locally:

```
python main.py --date 2026-10-28   # defaults to today, Pacific
python main.py --no-db             # dry run, nothing persisted
python main.py --init-db           # create tables then exit
```

## Environment variables

Copy `.env.example` to `.env` — it is the annotated, complete reference. The
variables that matter most:

| Variable | Required | Default | Notes |
|---|---|---|---|
| `DATABASE_URL` | yes | — | PostgreSQL. Supabase: use the connection pooler (6543) for the worker, the direct connection (5432) for migrations. `sslmode=require` is appended for remote hosts. |
| `DISCORD_WEBHOOK_URL` | for dispatch | unset (dispatch off) | Dispatch defaults to ON when set, OFF otherwise. Dry-run by default; `--send` actually posts. |
| `BIGDATABALL_XLSX` | once per DB | — | Path to the licensed workbook. Its contents are upserted and survive redeploys, so a container needs no file. |
| `PROPIQ_PARLAY_LEDGER` | on deploys | `postgres` (baked into the image) | CSV is local-only and is destroyed by a redeploy. The worker logs an ERROR if this is CSV while a database is configured. |
| `PROPIQ_CALIBRATION_REPORT` | on deploys | `/app/data/calibration.json` | The publication gate's evidence file. Settlement writes it at 03:30 PT; the slate job reads it at 09:00 PT. |
| `PROPLINE_API_KEY` | for odds | unset (client refuses) | The client refuses to run without it rather than return an empty board that reads like a quiet night. PrizePicks lines reach the pipeline through PropLine's `DFS_BOOKS`; no direct PrizePicks client exists. |
| `PROPIQ_MODEL` | no | newest artifact | Point the worker at a specific artifact without a code change. |
| `PROPIQ_MIGRATE_ON_BOOT` | no | `true` | Set `false` where a separate release step owns the schema. |
| `PROPIQ_SLATE_HOUR_PT` / `PROPIQ_SLATE_MINUTE_PT` | no | `09:00` | Slate job time, 24h Pacific. |
| `PROPIQ_SETTLE_HOUR_PT` / `PROPIQ_SETTLE_MINUTE_PT` | no | `03:30` | Settlement job time, 24h Pacific. |
| `PROPIQ_MAX_MARKET_LAG_DAYS` | no | `10` | How stale the Elo/market frames may be before the slate logs `MARKET DATA IS STALE` (`main.py:683`). |
| `PROPIQ_MAX_PANEL_LAG_DAYS` | no | — | The separate check on the PLAYER panel's newest completed game (`main.py:118`). Not the same variable as the one above. |
| `LOG_LEVEL` | no | `INFO` | INFO on stdout is deliberate — a red line in Railway means something went wrong. |

Display timezone is **not** an environment variable, whatever `.env.example`
suggests: `TZ_DISPLAY` there is read by nothing. `America/Los_Angeles` is a
constant in `src/utils/timezones.py:11`, beside `STORAGE_TZ_NAME = "UTC"`.

Security: never commit `.env`, API keys, or licensed data. The webhook
dispatcher redacts the URL from every error message, and no `DISCORD_BOT_TOKEN`
is needed — webhooks require no bot, gateway, or privileged intents.

## Deploying on Railway

Read `docs/deploy_railway.md` first — it carries the full walkthrough and the
remaining blockers. The essentials:

1. **Deploy as a worker, not a web service.** `scheduler_worker.py` binds no
   port; a web service would be marked unhealthy for never listening and
   restarted forever. `railway.json` sets `builder: DOCKERFILE`,
   `startCommand: bash scripts/start.sh`, `numReplicas: 1`, and deliberately
   no `healthcheckPath`. The Procfile is the buildpack fallback and declares
   `worker:` only.
2. **`numReplicas` must stay 1.** There is no distributed lock; a second
   replica is a second scheduler — two ingests, two dispatches, two settlement
   passes.
3. **Mount the volume at `/app/data`.** Two things must survive a redeploy:
   the trained artifacts (`data/external/model_runs/`) and the calibration
   report (`calibration.json`). There is no boot-time fetch — train into the
   volume, or put the artifact plus its `.meta.json` sidecar there and point
   `$PROPIQ_MODEL` at it. An unseeded volume is an ERROR in the first lines of
   the log, not a silent abstention. The mount arrives root-owned, which
   `scripts/start.sh` handles by chowning the state root and then
   `setpriv`-ing itself down to uid 10001.
4. **Boot sequence** (`scripts/start.sh`, in order):
   - `python -m scripts.run_migrations` — hard-fails the container on failure,
     because a schema behind the code surfaces hours later as a failed insert.
   - `python -m scripts.railway_healthcheck` — reports and carries on. Strict
     by design; an unseeded volume fails here.
   - `exec python scheduler_worker.py` — the worker is PID 1, so SIGTERM
     reaches its handler and shuts down after the running job.

Daily jobs: **slate at 09:00 PT** (ingest, features, score, gate, dispatch) and
**settlement at 03:30 PT** (grade, ledger, calibration report).

## The quantitative engine

- **Markets.** Launch: PTS, REB, AST. Post-launch: FG3M, STL, BLK, PRA.
  Combination markets PA, PR and RA are projected from component rollups (a
  composite mean is a linear sum — no correlation term needed). FPTS is graded
  but not yet projected: it needs a TOV layer whose fatigue haircut has the
  correct sign.
- **De-vigging.** Four methods — `multiplicative` (default), `shin`,
  `odds_ratio`, `logarithmic` — in `src/quant/devig_methods.py`. A missing
  correlation is never defaulted to zero: PTS, REB and AST are all driven by
  minutes and usage, so the covariance term is required, not optional.
- **Fantasy points.** The DFS scoring matrix lives in
  `config/fantasy_scoring.yaml`, dated, per operator:
  `FPTS = 1.0*PTS + 1.2*REB + 1.5*AST + 3.0*STL + 3.0*BLK - 1.0*TOV`. The
  weights are operator-supplied (2026-10-19) and not independently verified
  against a live scoring page — a multiplier is an operator's terms, not a
  fact about basketball, which is why it is config.
- **Sizing.** Fractional Kelly at a 0.25 fraction, capped, in
  `src/quant/advisory_sizing.py`. Zero on a non-positive edge; refuses
  impossible inputs rather than sizing them. Every size it returns carries
  `"AUTO_PLACED": False`.
- **Streaks.** The Underdog-style streak engine (`src/quant/streaks.py`) is
  config-driven: a 2-pick gatekeeper with tier multipliers and sequential
  progression.
- **The publication gate.** Cards are withheld until calibration evidence
  exists — **100** graded predictions with an **ECE at or below 0.05**, from a
  recent report (`src/quant/publication_gate.py:65-70`). With zero graded
  games the system sends nothing, which is the correct behavior for an
  unmeasured model.

## Diagnostics

Every script below is standalone and safe to run locally:

| Command | What it does |
|---|---|
| `python -m scripts.test_autonomous_loop` | Full mock cycle through production code — raw pull, features, quant, gate, dispatch, settlement, ledger commit. No network, no live data. Exit 0 only if every stage passes. |
| `python -m scripts.test_scrapers` | Exercises the ingestion clients against their contracts. |
| `python -m scripts.pipeline_smoke` | Runs the real pipeline stages end-to-end and reports per-stage status. |
| `python -m scripts.verify_wiring` | End-to-end wiring of the seams — `--section N` for one section, `--json` for machine-readable. |
| `python -m scripts.railway_healthcheck` | Deployment gate: probes the volume, artifacts, migration ledger, parlay ledger and calibration path. Exits nonzero when the deployment cannot do its job. |
| `python -m scripts.audit_pipeline` | Deeper audit of panel, folds, and leakage. |
| `python -m scripts.validate_docker` | Validates the Dockerfile's own claims. |
| `python -m pytest tests/ -q` | The full suite (~3,100 tests). |

Expected shape of a healthy `test_autonomous_loop` run: seven `PASS` lines and
`VERDICT: all stages passed`. Anything else names the broken stage.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `HTTP 403` from cdn.nba.com / stats.nba.com in a container | Datacenter IPs are refused. The data plane must run from a residential connection — use `scripts/ingest_from_windows.ps1` from your machine, which ingests through Railway's TCP proxy into the deployed database. |
| `ERROR ... model artifact` at boot | The volume at `/app/data` has no artifact. Train into the volume or place `xgboost_{MARKET}.json` + its `.meta.json` sidecar in `data/external/model_runs/`. |
| First write fails with `EACCES` | The volume mount is commonly root-owned; the container runs as uid 10001. `scheduler_worker.check_state_dir()` names this at boot — see `docs/deploy_railway.md` for the fix. |
| `PROPIQ_PARLAY_LEDGER` ERROR in the first log lines | The ledger is CSV while a database is configured — the CSV writes succeed and are destroyed by the next redeploy. Set it to `postgres`. |
| No Discord messages | Dispatch defaults to OFF without `DISCORD_WEBHOOK_URL`. With the webhook set, sending is still opt-in per invocation (`--send`); a dry run prints the payload. |
| `MARKET DATA IS STALE` in the log | The Elo/market frames are older than `PROPIQ_MAX_MARKET_LAG_DAYS`. Re-ingest the workbook, or re-run `load_scraper_panel --team-games`. |
| Step [2] refuses with empty tables | Neither the workbook nor ingested rows exist. Seed `team_game_stats` once against the deployment's database — the licensed workbook, or `scripts/load_scraper_panel --team-games` from the scraper's parquet. |
| "Written to Postgres" but the next slate reads 0 rows | Fixed, and worth knowing why: `upsert_*` returns rows *handed to* the statement, not rows kept. Both ingest scripts now read their own keys back on a fresh connection and exit 1 if they are not there. |
| Worker restarts in a loop on Railway | The service was deployed as a web app — it binds no port. Deploy as a worker. |
| Duplicate dispatches or double settlement | `numReplicas` is above 1. There is no distributed lock; set it back to 1. |

## Conventions

These are load-bearing; breaking them breaks the research validity.

- **Timestamps.** Storage is UTC. Everything user-facing is
  `America/Los_Angeles` and column names end in `_pt`. Slate dates are Pacific
  calendar days, never UTC midnight.
- **Pregame only.** A feature must have been knowable before tip. Postgame
  box-score fields are targets, never inputs. Every grouped rolling feature
  carries `.shift(1)`.
- **Chronological splits only.** No random K-fold on time-series data.
- **Abstain rather than guess.** When a required field is absent the code
  emits `DATA_NOT_AVAILABLE` and a named reason. It does not fill, invent, or
  substitute another model's output.
- **A write is proved by a read.** Row counts returned by an upsert are rows
  attempted. Persistence is established by reading the rows back on a fresh
  connection.
- **Never commit** `.env`, API keys, or licensed data.

## Status

Working and merged: the DB schema, the BigDataBall loader, the scraper-parquet
loader and team-game aggregation, the ingestion clients, the model interfaces,
chronological splits, over/under/push math, the ensemble, calibration, the
export layer, the settlement engine, the composite markets (PA/PR/RA), the
autonomous-loop harness, and the deployed worker.

**Not yet runnable end-to-end without operator involvement.** Three things are
open, in the order they unblock each other:

1. **Nothing has been ingested.** `team_game_stats` and `game_market_lines`
   are both empty, so the 09:00 PT slate exits 1 at step [2] before any model,
   odds call or gate is reached. Either seeding path above fixes it; both have
   to run from a machine that can reach the database.
2. **No model artifact exists** on the volume, so every row would abstain even
   with data. Training needs the panel from step 1.
3. **The publication gate has no evidence.** `n_scored` is 0 until settlement
   grades real games, and the gate requires 100 with an ECE at or below 0.05.
   Until then every card is withheld — which is the gate working, not failing.

`docs/DATA_GAPS.md` carries the current list and what each gap blocks.
