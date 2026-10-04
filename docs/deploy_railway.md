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
`docker build` has not run against it once. Treat the layer ordering and the
apt package list as reasoned, not verified.

Build it where a daemon exists, with:

```
python -m scripts.validate_docker          # preflight, build, six smoke checks
python -m scripts.validate_docker --preflight   # the daemon-free half
```

The preflight half runs anywhere and passes today (7 checks: the CMD target
exists, no credential is defaulted in a layer, every installed extra is
declared, the uid is the one this page tells you to chown to, and the ignore
file is evaluated by matching rather than grepped). The build and the six
in-image checks are what remain, and two of them exist to test **this page**:
they mount a mode-0555 directory at `/app/data` and require
`check_state_dir()` to report it unwritable, then mount a writable one and
require it not to false-alarm. Until that runs, the trap below is documented
and not demonstrated.

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
| `PROPIQ_PARLAY_LEDGER=postgres` | Already set in the image. Without it the ledger writes CSVs to an ephemeral disk and a redeploy destroys every ticket's at-bet-time probability and EV. |
| `PROPLINE_API_KEY` | The odds source. Without it no line is captured, so every row abstains for want of a market. |
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

## 4. Database

Apply the migrations against the target database before the first run:

```
migrations/002_prop_results.sql      # settlement ledger + views
migrations/003_capture_vs_ingest_time.sql
migrations/004_parlay_ledger.sql     # parlay_tickets, parlay_legs
migrations/005_projection_under_push.sql  # projections.prob_under, prob_push
```

`python main.py --init-db` creates the ORM-defined tables and exits. The SQL
migrations carry the CHECK constraints and views that `create_all` does not.

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
| `migrations/*.sql` + `python main.py --init-db` | once, per database | applied against the target database by hand |
| training model artifacts | once, then whenever you retrain | `data/external/model_runs/` is empty on a fresh container and **every row abstains**. Train into the mounted volume, or upload the artifacts to object storage and fetch them at boot |
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
