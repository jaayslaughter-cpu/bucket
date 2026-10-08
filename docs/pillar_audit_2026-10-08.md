# Three-pillar audit — 2026-10-08

Read-only audit of channels, mathematics and production stage. Every claim
below was checked against the code in this checkout on this date; where a
thing could not be verified here, it says so and says why. `verify_wiring`'s
own run at the time of writing: **29 passed, 0 failed, 1 warned, 3 skipped**.

Five things the audit brief assumed that this repository does not do, stated up
front so the rest reads correctly:

| brief assumes | actual |
|---|---|
| morning ingestion at **10:00 AM** | slate cron is **09:00 PT** (`DEFAULT_SLATE_HOUR = 9`), settlement 03:30 PT |
| **Pinnacle/Circa** two-way de-vig | no sportsbook feed exists. `SOURCE_PRECEDENCE = ("propline",)` — one pick'em source |
| **NGBoost** location-scale calibration | isotonic (default) or Platt sigmoid, fitted out-of-fold. No NGBoost anywhere |
| minimum **EV ≥ +3.0%** | `min_ev` defaults to **0.0**. The 0.03 constant is `DEFAULT_MIN_LEAN`, a probability lean, not EV |
| an **Odds API** channel | none. Odds reach the system only as pick'em board lines and BigDataBall game markets |

---

## Pillar 1 — Channel audit verdict

### 1a. Data ingestion channels

| channel | module | retry/backoff | reached by the automated slate? | verdict |
|---|---|---|---|---|
| Pick'em prop lines | `ingestion/propline.py` | yes (17 retry/backoff sites) | **yes** — `main.py` step [3] | **CONNECTED** |
| Game market lines | `ingestion/bigdataball.py` | no | **yes** — step [2] | **CONNECTED**, no retry |
| Schedule / tip-offs | `ingestion/espn_schedule.py` | yes (5) | **yes** — forward slate + pre-lock | **CONNECTED** |
| Availability / scratches | `ingestion/espn_availability.py` | **none** | yes, via `pipeline/scratches.py` | **CONNECTED, bare** |
| Official inactive list | `ingestion/inactive_players.py` | none (15 raise/except) | referenced, but **cache is empty** | **DISCONNECTED in effect** |
| Box scores (the panel itself) | `ingestion/boxscores.py` | yes (7) | **NO** — only `nba_model_cli ingest-logs` | **MANUAL** |
| Starting positions | `ingestion/starting_positions.py` | none (16 raise/except) | **NO** — separate pass, never run | **BUILT, NEVER RUN** |
| Play-by-play | `ingestion/nba_playbyplay.py` | yes (3) | **NO** | offline/training only |
| Travel | *not a feed* | — | yes | **DERIVED** — hardcoded `ARENA_COORDS` + haversine |

**The load-bearing finding.** The daily slate does **not refresh box scores.**
`main.py` step [4] is `load_player_panel` — it reads whatever is already in
`player_game_logs`. Writing to that table happens only through
`nba_model_cli ingest-logs` → `upsert_player_game_logs`, which nothing
schedules. So on a deployed container the panel is **frozen at whatever was
last ingested by hand**, and every rolling feature ages silently. Nothing in
the pipeline reports this, because a stale panel is indistinguishable from a
quiet one.

`stats.nba.com` and `site.api.espn.com` are both denied at this environment's
proxy (`CONNECT tunnel failed, response 403`), so no live ingestion channel
could be exercised end-to-end from here.

### 1b. Feature processing channels

All eleven additive layers are registered and all eleven add columns
(`verify_wiring`: "every registered feature layer actually adds columns").
The four the brief names:

| channel | status | detail |
|---|---|---|
| **Fatigue logic** | **ACTIVE, and it bites** | `fatigue_multiplier` folds into `{stat}_L2`. `verify_wiring` proves it modifies the projection rather than riding along: 12 adjusted rows, max \|L2−BASELINE\| = 0.553 |
| **Fatigue load** | **ACTIVE, UNFITTED** | a second, feature-shaped channel; its three parameters are admitted constants (below) |
| **Recency weighting** | **WIRED AND DISABLED** | `recency.enabled: false`, with "DO NOT SET THIS TRUE YET — a known inconsistency is open": `XGBoostAdapter.fit` forwards weights to the final classifier but **not** to its out-of-fold pass, so OOF calibration would be fitted on a different weighting than the model |
| **Usage cascade** | **ABSTAINS** | `teammate_cascade` is explicitly a stub. It needs `BBS_TEAMMATES_OUT` from the inactive-list ingest; that cache is empty, so the columns arrive null and no multiplier is applied |
| **Minute projections** | **NOT IN THE PATH** | `models/minutes_model.py` exists and is referenced **only** from `nba_model_cli.py`. No layer and no inference step calls it. Minutes reach the model as rolling features (`MIN_L5/L10/SEASON`), not as a projection |

Three further layers — `fouls`, `dvp`, `minutes_weighted` — run on every build
and **no market reads a column from any of them**, by design and with the
measurements recorded (`docs/fouls_and_dvp.md`, `docs/minutes_weighted.md`).

### 1c. Execution & alert dispatch channels

| channel | status | detail |
|---|---|---|
| Postgres projections | **CONNECTED** | `persist_projections` + `pipeline_runs` audit row, step [9] |
| Pre-game line snapshots | **CONNECTED** | `prop_line_snapshots` via `pg_insert(PropLineSnapshot)`; `captured_at_utc` is observation time, never `now()` (migration 003) |
| Prop results ledger | **CONNECTED** | `PropResult`, written by `settlement/recorder.py`, called by `main.py` |
| Parlay ledger | **CONNECTED, WRONG DEFAULT FOR A CONTAINER** | `PROPIQ_PARLAY_LEDGER` defaults to **`csv`** — a local file on an ephemeral disk. Must be set to `postgres` on Railway |
| Discord | **CONNECTED** | six embed builders, gated dispatch; a withheld board sends the gate's *reason*, not the rows |
| Advisory sizing metadata | **CONNECTED** | `"AUTO_PLACED": False` is hard-coded and asserted by `verify_wiring` |
| Pre-lock correction | **CONNECTED** | one-shot job per game at tip − 35 min, armed at startup and re-armed by `run_slate` |

**No manual intervention is required for dispatch.** It is required for
ingestion (box scores) and for model artifacts (below).

---

## Pillar 2 — Mathematical blueprint

### 2.1 Probability generation & calibration

**Raw output → p.** Five model families, one trained artifact per market
(`xgboost_{MARKET}.json` + a `.meta.json` feature-contract sidecar):

- `xgboost`, `catboost` — binary classifiers on `over_hit`, `predict_proba`
- `line_aware` — augments each source row across 9 line offsets (capped at
  600,000 augmented rows) so the model sees the line as a feature
- `distribution` — parametric: fits a `CountDispersion` (negbin / ZIP /
  normal / Poisson, selected by mean NLL) and evaluates
  `over_under_push_from_dispersion`
- `ensemble` — blend over the above

**Calibration.** `ProbabilityCalibrator`, **isotonic by default**, Platt
sigmoid available (`LogisticRegression` on a single feature), fitted
**out-of-fold only** — "Never fit on eval fold". Reported as ECE raw vs ECE
calibrated side by side everywhere. **There is no NGBoost and no
location-scale head**; the distributional work is the `distribution` adapter
plus `residuals.CountDispersion`, and the dispersion is used for push mass and
(since 2026-10-05) for the pick'em line transport.

**No sportsbook-shading correction exists.** Nothing models the book's bias,
because there is no book feed to model it against.

**Fatigue decay — the two forms, both unfitted.**

1. *Multiplicative haircut* (`features/fatigue_logic.py`), applied to
   `{stat}_L2`. Not a decay function — a step table:
   `B2B = 0.97`, `3-in-4 = 0.96`, `4-in-5 = 0.94`, `altitude = 0.98`
   (away games at altitude only, neutral sites excluded).
2. *Exponential load* (`features/fatigue_load.py`), a feature rather than a
   haircut:

   `FATIGUE_LOAD_L7 = Σ_g MIN_g · exp(−λ · days_ago_g) · (1 + θ·miles_g/1000 + φ·|tz_shift_g|)`

   with `λ = 0.20`, `θ = 0.05`, `φ = 0.05`. The module's own docstring:
   "**PARAMETERS ARE UNFITTED**, exactly as the constants they are meant to
   replace." λ is the midpoint of a conventionally quoted range; θ and φ are
   deliberately small.

Neither adjusts *projected minutes* — there is no minutes projection in the
path (1b).

### 2.2 De-vigging and pick'em EV

**De-vig: multiplicative only.** `odds_math.multiplicative_devig(a, b)` —
`p_fair = p_raw / Σp_raw`, assuming the vig is spread proportionally. **Shin's
method is not implemented.** No Pinnacle or Circa client exists; the only
two-way prices that reach de-vigging are whatever a `PropMarketSnapshot`
carries with `status=VALID`.

**Multi-leg EV.** `dfs_payouts.evaluate_payout` over the full hit-count
distribution:

- **Independent legs** — `independent_hit_count_distribution`: the exact
  Poisson-binomial recursion (`dp[k] = dp[k]·(1−p) + dp[k−1]·p`), not
  simulation. Verified against brute-force 2ⁿ enumeration to 1e-15 and against
  `scipy.stats.poisson_binom` to 1e-12 where available.
- **Correlated legs** — `parlay.hit_count_distribution`, a Gaussian copula with
  nearest-PSD repair, driven by `leg_correlation` buckets
  (`MIN_PAIRS_PER_BUCKET = 200`, `MIN_GAMES_PER_BUCKET = 50`).
  `independent_hit_count_distribution` documents that the sign of the
  independence error depends on the sign of the correlation **and** on the
  payout curve — a flex can lose EV from positive correlation while a power
  play gains — which is why there is a copula path rather than a correction
  factor.
- `EV = Σ_k P(k hits) · payout_multiple[k] − 1`, with `payout_multiples()`
  exposed as a method so sizing and pricing cannot disagree about how a
  non-paying tier is represented.

**Breakeven for an N-leg entry at multiplier M.** Both forms, and a refusal:

- `per_leg_breakeven_probability() = M^(−1/N)` — the brief's
  `p_breakeven`. Defined **only** for all-or-nothing structures. Worked
  example in the code: 3 legs at 6× → `6^(−1/3) = 55.03%` → synthetic −122.
- `breakeven_joint_probability() = 1/M`.
- `per_leg_synthetic_american()` converts the first to American odds, labelled
  SYNTHETIC because no book offers it.
- For a **flex**, both return `None` on purpose: "with partial payouts the
  breakeven is a surface over the whole count distribution, not a single
  probability, so returning one number would be a category error rather than
  an approximation."

### 2.3 Risk management & advisory sizing

**Two-outcome.** `recommended_units_binary`:
`f* = (b·p − q) / b`, `b = decimal − 1`, then
`units = min(f* · kelly_fraction, max_cap_units)`.
`DEFAULT_KELLY_FRACTION = 0.25`, `DEFAULT_MAX_CAP_UNITS = 3.0`.
One unit = one percent of bankroll. **The cap is applied last and is hard.**

**Tiered.** `recommended_units_multi_outcome` maximises expected log wealth
over the hit-count distribution by `scipy.optimize.minimize_scalar`, using the
same indexing as `evaluate_payout`.

**Safety constraints, all hard-coded:**

- `"AUTO_PLACED": False` in the metadata, always, asserted by `verify_wiring`
  and by `AGENTS.md`'s cited line
- negative or non-finite `kelly_fraction` / `max_cap_units` → refusal, not
  pass-through (a negative fraction previously flowed straight through,
  because `f* < 0` already gives 0 units and `suggested > cap` does not catch it)
- `p` outside the open interval (0, 1) → refusal with a reason, never a size
- `decimal_odds ≤ 1.0` → refusal
- `f* ≤ 0` → zero units

### 2.4 Trigger gates and thresholds

| gate | constant | value | where |
|---|---|---|---|
| Publication: max ECE | `DEFAULT_MAX_ECE` | **0.05** | `publication_gate.py` |
| Publication: min graded | `DEFAULT_MIN_SCORED` | **100** | at 50 graded, a true 0.03 and a true 0.08 ECE are indistinguishable |
| Publication: max evidence age | `DEFAULT_MAX_AGE_DAYS` | **45** | a good ECE from a season ago says nothing about today |
| Board: min probability lean | `DEFAULT_MIN_LEAN` | **0.03** | `decision_board.py` — **a lean, not an EV** |
| Board: min EV | `min_ev` | **0.0** | default; `PROPIQ_MIN_EV` overrides |
| Results card: min sample for a rate | `MIN_SAMPLE_FOR_RATE` | **30** | a strike rate below this is withheld with its reason |
| Dispersion fit | `MIN_ROWS_TO_FIT` | 50 | `residuals.py` |
| OOF usability | `MIN_ROWS_PER_FOLD` / `MIN_USABLE_OOF_ROWS` | 40 / 60 | `oof.py` |
| Market comparison bucket | `MIN_BUCKET_ROWS` | 30 | `market_comparison.py` |
| EV gate | — | requires `status=VALID` **and** verified two-way American odds **and** a finite line | `contracts.market_ev_gate` |

**The expected production verdict today is ABSTAIN on every row**, and that is
correct rather than broken: no model artifact is trained in this checkout, and
the publication gate withholds for want of graded evidence.

### The caveat that qualifies every number above

**O8, self-referential evaluation.** `RESEARCH_LINE` is `{stat}_L10` and
`over_hit` is measured against that same rolling history, so every Brier and
log-loss in this repository — including the DvP arms — **measures form against
form**, not against a posted line. Nothing changes until a posted-line archive
drives line-aware training. Every metric in this audit inherits that.

---

## Pillar 3 — Production stage and blocker list

### Stage: **Refactored Pre-Flight.**

Not Staging-Ready, and the gap is not code quality. The pipeline is wired end
to end, 1,922 tests pass, `verify_wiring` is green, and the container
specification exists — but **three inputs the system needs have never been
supplied**, and two of them cannot be supplied from a developer machine at all.

### Persistence status: **PARTIAL.**

In Postgres: `projections`, `prop_line_snapshots`, `prop_results`,
`parlay_tickets` / `parlay_legs`, `player_game_logs`, `team_game_stats`,
`pipeline_runs`, and now `schema_migrations`.

Still on local disk, and therefore lost on redeploy:

| path | what | consequence |
|---|---|---|
| `data/external/model_runs/` | **trained model artifacts** | a fresh container abstains on **every row** |
| `data/external/parlay_log/` | parlay ledger, **csv by default** | ledger lost unless `PROPIQ_PARLAY_LEDGER=postgres` |
| `data/external/inactive_players/` | injury cache | cascade and absence layers abstain |
| `data/external/starting_positions/` | position cache | DvP abstains |
| `data/external/market_store/`, `bigdataball/`, `training_pack/` | ingest caches | re-fetchable |
| `outputs/` | board CSV | re-derivable |

There is **no S3 client and no object-storage loader** in this repository. The
Dockerfile declares `/app/data` as a volume and no longer claims a boot-time
fetch exists — `scheduler_worker.check_model_artifact()` reports an unseeded
volume as an **ERROR at boot** rather than letting every row abstain quietly.

### Execution & scheduler status: **BUILT.**

`scheduler_worker.py`, APScheduler `BlockingScheduler`, no interactive
execution:

- `slate` — cron 09:00 PT (**not 10:00**; `PROPIQ_SLATE_HOUR` overrides)
- `settlement` — cron 03:30 PT, grades the previous Pacific day
- `prelock` — **one-shot per game at tip − 35 min**, armed at startup and
  re-armed by each `run_slate`, which is the tip-anchored behaviour a fixed
  cron cannot give
- all cron jobs `max_instances=1`, `coalesce=True`, with misfire grace: slate
  one hour (a late slate would project tipped games), settlement six
- every job catches and logs; a failing job does not take the worker down
  (asserted by `verify_wiring`)

### Containerization readiness: **SPECIFIED, UNVERIFIED.**

`Dockerfile` exists: `python:3.11-slim`, non-root `USER propiq`, volume at
`/app/data`, `CMD ["python", "scheduler_worker.py"]`.
`scripts/validate_docker.py` pre-flight passes.

**The image has still never been built.** A `docker` client (29.3.1) is
present in this container but there is no daemon —
`dial unix /var/run/docker.sock: connect: no such file or directory` — so the
build and the six in-image checks cannot run here.

**There is no `railway.json`, `railway.toml` or `Procfile`.** Railway can
build from a Dockerfile without one, but nothing in the repository declares
the volume mount, the restart policy or the healthcheck.

### Blocker list — what remains before `git push railway main`

**P0 — the deploy produces nothing without these.**

1. **Train and seed a model artifact.** No `xgboost_*.json` exists in this
   checkout. Without one every row abstains. Train into the mounted volume, or
   set `PROPIQ_MODEL` to a seeded path.
   `scripts/nba_model_cli.py train-stats --market PTS --start-date … --end-date …`
2. **Schedule the box-score ingest, or accept a frozen panel.** `boxscores.py`
   is not in the automated path. Either add an ingest step ahead of
   `run_slate`, or document that the panel is refreshed by hand — but it
   cannot stay implicit, because a stale panel looks exactly like a quiet one.
3. **Set `PROPIQ_PARLAY_LEDGER=postgres`.** The csv default writes the ledger
   to an ephemeral disk.
4. **Build the image once on a machine with a daemon.**
   `python -m scripts.validate_docker` runs the build and six in-image checks.
5. **Apply the migrations and verify Postgres accepts them.**
   `python -m scripts.migrate_db --apply`. The ledger logic is tested against
   SQLite; whether Postgres accepts the hand-written DDL in 002–008 is the one
   thing only Postgres can answer.

**P1 — the deploy runs but a channel stays dark.**

6. **Pull the inactive lists** (`fetch-inactives`, ~1,230 calls/season, cached)
   — unblocks `teammate_cascade` and the absence layer, both abstaining today.
7. **Pull starting positions** (`scripts/pull_starting_positions.py`) — unblocks
   DvP on the live path. Needs a machine where `stats.nba.com` is reachable;
   run `--limit 5 --dry-run` first, because the five-starters gate settles what
   v3's `position` field means before 1,230 calls are spent.
8. **Add a `railway.json`** declaring the volume mount and restart policy, so
   the platform configuration is in version control rather than in a dashboard.
9. **Decide the model-artifact persistence story.** Mounted volume (seeded by
   hand, which is a recurring manual step) or write an object-storage loader
   (which does not exist). This is the one open item that is a design decision
   rather than a task.

**P2 — correctness and honesty items, none deploy-blocking.**

10. **~~`PRA_L5` / `PRA_L10` / `PRA_SEASON` have two definitions and the second
    silently wins.~~ FIXED 2026-10-08** — and it was worse than written below:
    **five** columns were overwritten, not three (`PRA_BASELINE` and `PRA_L2`
    as well), and `PRA_L2_PACE` was left holding the *discarded* definition, so
    it disagreed with the `PRA_L2` shipped beside it on 8 of 14 rows in a
    fixture with one partial game — including rows where `PACE_MULTIPLIER` was
    exactly 1.0. `builder.attach_pra_from_components` is now the single
    implementation, called from both places; the layer keeps only the four
    suffixes nothing else produces. Every value on the archive panel is
    unchanged. Original finding, for the record:

    **`PRA_L5` / `PRA_L10` / `PRA_SEASON` have two definitions and the second
    silently wins.** `build_feature_matrix` derives `PRA = PTS + REB + AST`
    and rolls it, commenting "a NaN in any component propagates deliberately:
    a partial sum would read as a real total" — then the registered
    `halflife.pra_rollups` layer **overwrites** those columns with
    `PTS_L5 + REB_L5 + AST_L5`, the sum of the component rolling means.
    Measured in this audit: on a clean panel both give 33.000; with one REB
    missing in one prior game the shipped value is **33.000** and the
    propagating definition is **33.500**. The two agree on the archive panel
    (PTS/REB/AST are null on 0 of 214,381 rows) and diverge on the live panel,
    where `player_game_logs.pts/reb/ast` are independently nullable. The
    sum-of-components is arguably the better estimator — it uses all available
    data per stat — but it is a *different* one, and the builder's comment now
    describes behaviour that does not survive the layer stack. **Pick one and
    delete the other.**
11. **Recency weighting stays off** until the `XGBoostAdapter.fit` OOF
    inconsistency is closed and an A/B is run.
12. **O7 — eight config blocks are declared and never read**, so editing the
    YAML changes nothing.
13. **O6 — `Game.tipoff_utc` has no writer.** No longer blocks tip-anchored
    scheduling (`schedule_prelock_jobs` reads `espn_schedule` directly) but the
    column is still unwritten and still wrong to read.
14. **O9 — every entry resolves to `ProbabilitySource.MODEL`.** No sharp
    benchmark feed is in reach, so nothing cross-checks the model's own number.
    Blocked externally.
15. **O8 — self-referential evaluation.** The deepest item on this list and the
    one that qualifies every metric the system reports.

### What `git push railway main` would do today

Boot, log an artifact ERROR, run the 09:00 slate against a panel frozen at the
last manual ingest, score nothing, abstain on every row, and post a withheld
board to Discord with the gate's reason. **Correct behaviour for the current
evidence — and indistinguishable, from the outside, from a working system with
no edges to report.** Items 1 and 2 are what separate those two readings.
