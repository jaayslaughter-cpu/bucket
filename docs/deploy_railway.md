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

## 2. Build

The repository root has a `Dockerfile`; point the service at it.

**It has never been built.** The environment it was written in has a docker
client but no daemon and a network policy that denies the package index, so
`docker build` has not run against it once. Build it locally before the first
deploy; treat the layer ordering and the apt package list as reasoned, not
verified.

It installs the `ml`, `db` and `deploy` extras. The ML extras are optional in
`pyproject.toml` so a local checkout stays light — a minimal image would start
cleanly and then fail at the first inference, unattended, after the slate had
already been ingested. Failing at build time is the better trade.

`.dockerignore` keeps `.env` and `data/` out of the build context. A credential
in an image layer survives every later layer that deletes it.

## 3. Environment

Set these in the platform, never in the image. `.env.example` is the full list;
these are the ones the worker reads.

| Variable | Why |
|---|---|
| `DATABASE_URL` | Required. Use the **pooler** URL (port 6543 on Supabase) — these are short-lived jobs and the direct port exhausts connections. |
| `PROPIQ_PARLAY_LEDGER=postgres` | Already set in the image. Without it the ledger writes CSVs to an ephemeral disk and a redeploy destroys every ticket's at-bet-time probability and EV. |
| `PROPLINE_API_KEY` | The odds source. Without it no line is captured, so every row abstains for want of a market. |
| `DISCORD_WEBHOOK_URL` | Only if you dispatch. Never logged or printed. |
| `PROPIQ_MAX_THREADS` | Defaults to 2 in the image. Match your plan's CPU allocation — the numeric libraries otherwise see the host's core count, not the container's share. |
| `PROPIQ_SLATE_HOUR_PT` / `_MINUTE_PT` | Defaults 09:00 PT, before any tip. |
| `PROPIQ_SETTLE_HOUR_PT` / `_MINUTE_PT` | Defaults 03:30 PT, after any finish. |
| `PROPIQ_RUN_ON_START` | Off by default. Set it for a single first run; leaving it on means a redeploy loop re-runs the slate each time. |

## 4. Database

Apply the migrations against the target database before the first run:

```
migrations/002_prop_results.sql      # settlement ledger + views
migrations/003_capture_vs_ingest_time.sql
migrations/004_parlay_ledger.sql     # parlay_tickets, parlay_legs
```

`python main.py --init-db` creates the ORM-defined tables and exits. The SQL
migrations carry the CHECK constraints and views that `create_all` does not.

## 5. What runs, and when

| Job | Time (PT) | What it does |
|---|---|---|
| `slate` | 09:00 | ingest → features → score → EV gate → persist projections → write PENDING `prop_results` |
| `settlement` | 03:30 | grade every PENDING prop whose game has finished |

Both run with `max_instances=1`: an overrunning job is never joined by a second
copy. A missed slate is **not** run late (one-hour grace) — running late would
project games that have already tipped. A missed settlement is, up to six
hours; a finished game stays finished.

The schedule is fixed, not tip-off-driven. Re-anchoring needs a schedule feed,
and timing logic that has never been exercised against real data would look
adaptive while being untested.

## 6. What a deployed run produces

- `projections` — the pipeline's deliverable.
- `prop_results` — one PENDING row per prediction with a line, a probability
  and a source, graded by the settlement job. **These are predictions, not
  wagers:** no stake is written and none can be. Strike rate and CLV come out
  of them; ROI does not, and will not until you record a stake yourself.
- `parlay_tickets` / `parlay_legs` — only what you log by hand.

Nothing is dispatched to Discord unless a command is run with `--discord`, and a
model-sourced card is withheld there until `src/quant/publication_gate.py` has
recent, dense, sufficiently large calibration evidence to pass.

---

## The remaining blocker

**Model artifacts do not survive a redeploy.** `score_prob_over` loads them
from `data/external/model_runs/comparison/`, which is on the container's
ephemeral filesystem with no volume declared. On a fresh container it is empty
and **every row abstains** — the pipeline runs, writes nothing useful, and does
not look broken.

Two ways to clear it, both outside this repository:

1. Mount a persistent volume at `/app/data` in the platform, and train once
   into it.
2. Fetch the artifacts from object storage at boot, before the first slate.

Until one is done, treat a deployment as a wiring test rather than a shadow
run.
