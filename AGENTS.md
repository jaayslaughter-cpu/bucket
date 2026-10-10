# PropIQ — working agreement

**Status: RESEARCH_ONLY.** NBA player props. Every rule below is enforced
somewhere in the code or the test suite, and the enforcement point is named so
you can check rather than trust. If a rule here and the code disagree, the code
is the fact and this file is the bug — fix both.

Adapted from an uploaded pack's `AGENTS.md` and rewritten against this tree;
the original named six docs and a module that do not exist here. See
`docs/go_live_pack_review.md` §4.

---

## 1. The hard lines

| Rule | Where it is enforced |
|---|---|
| No wager is ever placed, and no code path can place one | `src/quant/advisory_sizing.py:114` writes `"AUTO_PLACED": False` into every size it returns |
| Bet sizes are advisory metadata, never bound to an execution script | same; `recommended_units` is read-only output |
| No EV, CLV or stake unless the market is real | `src/quant/contracts.py:187` — `context.status != "VALID"` returns a verdict with a reason and no number |
| Pick'em multipliers are not valid two-way EV | `src/quant/dfs_payouts.py`, `src/quant/dfs_entry.py` — a payout multiple is a payout multiple; a one-sided benchmark is refused outright |
| No card is published on an uncalibrated model | `src/quant/publication_gate.py:150` `calibration_gate()`; absent, sparse, undated or future-dated evidence all withhold |
| No claim words | `tests/test_publication_gate.py:66` fails the build if `"guaranteed"`, `"profitable"`, `"lock"` or `"certain"` appears in the disclaimer |
| Nothing fabricated — lines, odds, props, stats, injuries, endpoints, probabilities, history | explicit `DATA_NOT_AVAILABLE` statuses throughout; see §4 |
| **NBA only. No NCAA, no other sport.** | `main.py:4`; `src/ingestion/propline.py:15` refuses every non-NBA league the provider offers |
| No secret in a repo, an image or an export | `.dockerignore`; `scripts/validate_docker.py` fails the build if any `ENV`/`ARG` defaults a key, token, webhook or DSN |

**Probabilistic language only.** A probability, an interval, or an abstention.
Never "lock", "best bet", "sure thing", "guaranteed", or a profitability claim
about any model. `src/models/edge_grades.py` exists to grade an edge without
implying a stake.

**The user places their own bets, outside PropIQ.** This project logs, grades
and reports. `docs/wave3_paper_research.md`, `docs/decision_board.md`.

---

## 2. Leakage

The whole project is worth nothing if a feature sees the future. Treat this as
the first question about any change, not the last.

- Every grouped rolling feature is `.shift(1)` within player-season. No
  exceptions, including the ones that look harmless.
- `src/features/builder.py:665` `assert_no_lookahead()` runs over whatever the
  layers produced and is not optional.
- Season grouping comes from `src/features/season.py`. A layer that needs a
  season key and has none derives it **privately** and drops it before
  returning — it does not write a public `SEASON`. Four modules used to derive
  their own and one of them was right; `docs/season_key.md` is the write-up.
- Closing lines are refused as features. A number known only at tip is the
  market's final answer, not a prediction. `src/features/market_context.py`.
- Backtests are chronological. No fold may contain a row dated after a row in
  its own training set.

**New features are measured, not argued for.** `scripts/feature_ab.py` runs the
A/B. A column correlating above ~0.97 with one the model already reads is a
second copy of one number, and this repository has excluded whole feature
families on that basis — see `src/models/labels.py` `_EXCLUDED_AS_REDUNDANT`
and `docs/minutes_weighted.md` for a layer that is built, measurable, and
deliberately not shipped. `docs/fouls_and_dvp.md` records the opposite
lesson: both of those layers were first judged impossible here because no
panel on disk carried a foul count or a position, and both columns were in the
source file all along. It also records what happens when the A/B comes back
the other way — DvP improved REB's Brier on every one of four folds for four
of five models — and that a measured win is still not a shipped feature while
the column is null in production.

---

## 3. Layout

```
src/ingestion/   outside data in        (propline, espn_*, bigdataball, kaggle, pbp)
src/features/    leakage-safe columns   (builder + additive layers + season key)
src/models/      fit, score, compare    (xgboost/catboost, feature_spec, labels)
src/quant/       EV, de-vig, gates      (ev_engine, parlay, dfs_*, publication_gate)
src/settlement/  grade and calibrate    (recorder, calibration)
src/pipeline/    slate assembly         (slate_board, scratches)
src/notify/      Discord only           (discord.py)
src/db/          SQLAlchemy models, session, repository
src/utils/       timezones
main.py              one slate run, CLI
scheduler_worker.py  the deployed worker (APScheduler, two cron jobs)
scripts/             CLI, A/B harnesses, audits, docker validation
```

**Dependency direction is downward:** ingestion → features → models → quant →
settlement/notify. One standing exception, and it is narrow:
`src/features/builder.py:121` and `src/features/fatigue_load.py:284` each do a
function-local `from src.models.compare import load_comparison_config` to read
`config/model_comparison.yaml`. That is a config reader, not model code, and
the import is lazy so it cannot create a cycle at import time. Do not widen it:
a features module must not import a fitter, a scorer or a quant gate.

**This is not a Streamlit app** and has no web UI. Notification is Discord.
`scheduler_worker.py` binds no port — deploy it as a **worker** service, which
is the single most likely way a first deploy goes wrong. The `Dockerfile` at
the root builds that worker and `pyproject.toml` declares the extras it
installs; `docs/deploy_railway.md` is the deploy page.

---

## 4. Zero-inference

Every module that cannot answer says so, in a named status, and returns
nothing rather than something plausible:

- `DATA_NOT_AVAILABLE` for a missing input — a column, a feed, a box score.
- `WITHHELD` / `ABSTAIN` for a row the project refuses to publish.
- `UNVERIFIED` for a feed that failed, which is **not** the same as a clean
  feed that said nothing. `src/pipeline/scratches.py` marks every row
  `UNVERIFIED` when the injury feed errors and drops none of them, because an
  unreachable feed is not a healthy roster.

A missing minute count is not zero minutes. A missing rebound is not zero
rebounds. An unknown availability is not "available".

---

## 5. Data and secrets

**Odds:** PropLine (`PROPLINE_API_KEY`) is the source. No other paid sportsbook
API, and no sportsbook HTML scraping. Pick'em capture (Sleeper, Underdog,
PrizePicks) is timestamped research only — `docs/pickem_props.md`.

**Timestamps:** stored in UTC, displayed in `America/Los_Angeles`.
`src/utils/timezones.py:11-12` is the only place those two strings live. Every
slate cutoff is a Pacific **calendar day**.

**Config:** `config/model_comparison.yaml` (model + feature + eligibility +
drift blocks), `config/dfs_payouts.yaml`, and a master guideline at
`config/master_guideline_props.yaml` — not in the repo; `main.py:241`
`load_master_guideline()` returns `None` and the caller logs it. The
`.example` beside it is a placeholder shape, not the real file.

**Secrets:** environment variables only. Never a hardcoded key, credential or
absolute local path. `.env.example` lists every variable the worker reads and
holds no values. Never commit `.env`; rotate anything pasted into a chat.
Nothing — no export, no commit, no Discord payload, no log line — carries a
key or personally identifying data.

---

## 6. Changing this repository

- **Make it fail first.** A new assertion that passes before the fix is not a
  test. Every substantive change in this tree has been mutation-checked:
  revert the fix, watch the named tests go red, put it back.
- Read the actual file before assuming a column, a field, a payload key, a
  target name or a model path. The audits in `scripts/` exist because guesses
  were wrong often enough to be worth automating.
- Preserve existing architecture. Prefer a narrow, named change to a rewrite.
- `outputs/` is for final downloadable deliverables only.
- No external dependency is added for a feature another repository happens to
  have. Read it, measure the idea, implement it natively, say where it came
  from — `docs/external_odds_repos_review.md`,
  `docs/external_feature_harvest.md`, `docs/go_live_pack_review.md`.

### Commands

```bash
python -m pytest -q                              # the whole suite
python -m ruff check src/ tests/ scripts/
python main.py --init-db                         # create tables, exit
python main.py                                   # one slate run
python scheduler_worker.py                       # the deployed worker
python -m scripts.feature_ab --layer <name> --wire-under-test
python -m scripts.validate_docker --preflight    # no daemon needed
```

---

## 7. What is still not true

Kept here on purpose, so nobody reads an ambition as a fact.

- **No model is profitable.** None has been shown profitable, and none may be
  described that way. Forward, leakage-safe, settled results are the only
  evidence that would count, and there are not enough of them yet.
- **The publication gate is withholding every card**, correctly, because
  nothing is settled yet. That is the right output for the current evidence,
  not a failure.
- **Every entry resolves to `ProbabilitySource.MODEL`.** There is no sharp
  two-way NBA player-prop benchmark feed, so nothing cross-checks the model's
  own number. `docs/go_live_readiness.md`.
- **The Docker image is built and smoke-tested** (2026-10-10), where this said
  for months that it never had been. All 15 `scripts/validate_docker.py` checks
  pass against it, including the two that exist to demonstrate the
  volume-permission trap, and the container boots through `scripts/start.sh`
  and schedules both jobs in Pacific. Two things remain true: the base image
  tag still floats, and the build used a copy of the `Dockerfile` with two
  extra pip-trust instructions, because this session's egress re-terminates
  TLS — every other instruction was byte-identical.
  `docs/deploy_railway.md` §2.
- **Model artifacts do not survive a redeploy.** They live under
  `data/external/model_runs/`, on an ephemeral filesystem, and a fresh
  container abstains on every row without looking broken. That one is a
  platform step, not a code change. `docs/railway_deployment_audit.md` §4.
  Since 2026-10-09 it is at least no longer *silent*:
  `resolve_state_root()` in `src/utils/volume.py` reads
  `RAILWAY_VOLUME_MOUNT_PATH` and reports how it chose,
  `scripts/seed_volume.py` does the copying, and
  `scripts/railway_healthcheck.py` exits NONZERO on an unseeded volume. Still
  nobody's code can mount a volume, and **there is no boot-time fetch** — no
  S3 client, no storage client, nothing that downloads a model.
  `docs/deploy_railway.md`.
- **The BigDataBall workbook is needed once, not per deploy — and a stale one
  is now the thing to watch.** Fixed 2026-10-10. It *was* true that a missing
  workbook failed the whole slate: `ingest_market_lines` was unguarded, so
  `FileNotFoundError` at step [2] became a FAILED `pipeline_runs` row and exit
  1, and the image excludes `data/` and `*.xlsx` deliberately, so a container
  nobody uploaded one to failed every scheduled run.
  `main.resolve_market_frames` now falls back to `team_game_stats` and
  `game_market_lines` in Postgres — the workbook's own contents, upserted by
  every run that finds one — because the data was never missing, only the file
  was. With NEITHER it still refuses, naming both. The cost is a quiet failure
  in place of a loud one: the database holds whatever was last ingested, so
  `main.market_frames_freshness` logs MARKET DATA IS STALE past
  `PROPIQ_MAX_MARKET_LAG_DAYS` (10), for a workbook run as well as a fallback.
  Automating the *fetch* remains out of the question — it is a licensed export
  and scraping it would breach the licence. `docs/deploy_railway.md` §3b.
- **Nothing retrains on a schedule.** The artifact is a fixed snapshot and the
  panel moves daily. Deliberate: an unattended retrain replaces the artifact
  that produced the probabilities now in the database, mid-season, with nobody
  looking at the validation numbers. `docs/automation_audit_2026-10-09.md` G3.
- **A trained artifact is not a committed one.** PTS, REB and AST were fitted
  on the archive panel on 2026-10-09 and verified end to end — the resolver
  finds them, `check_model_artifact()` reports INFO rather than the demo
  ERROR, and all three score a production-shaped frame with probabilities in
  [0, 1]. But `data/**` is gitignored, so they live only where they were
  trained: a fresh checkout and a fresh container both have none and abstain
  on every row. Seeding is a volume or object-storage step, not a commit.
  `docs/deploy_railway.md` §4b.
- **More training history beats recency, measured.** Narrowing the window to
  recent seasons was requested, measured and declined: the ensemble's
  calibrated Brier degrades monotonically as the start date moves forward
  (0.24098 full window → 0.24268 from 2024-10, one common validation window).
  What was wrong was the 2/3-by-row split, which threw away a third of
  whatever history it was given; `train-stats --train-end` separates the two
  and the seed now fits 200,678 rows to 2026-01-13 rather than 142,713 to
  2023-12-13. One window, one market, one fold — the direction is established
  and the magnitudes are indicative. `docs/training_window.md`.
- **A forward slate rests on who played recently, not on a roster.**
  `src/pipeline/forward_slate.py` builds tonight's rows from each team's recent
  appearances in the panel, which misses a player returning from a long
  absence. `espn_availability.fetch_roster` would cover that and needs the
  name crosswalk this repository does not have. `docs/integration_audit.md`
  §1.1.
- **No database has had the migration runner pointed at it.**
  `src/db/migrations.py` and `scripts/migrate_db.py` apply `migrations/*.sql`
  in version order and record each file with a checksum in
  `schema_migrations`, and the ledger logic is tested against SQLite — but no
  Postgres is reachable from this checkout, so the one thing still unverified
  is whether Postgres accepts the hand-written DDL in files 002-008. Only
  Postgres can answer that. `python -m scripts.migrate_db` reports before it
  applies anything.
- **The panel refreshes itself, but it has never fetched here.** The slate
  pulls this season's player game logs and upserts them before reading the
  panel (`main.refresh_player_logs`, step [3b]), and `main.panel_freshness`
  measures how far the newest completed game trails the slate whatever the
  refresh reported. Neither has run against a reachable nba.com from this
  checkout, because the proxy denies it — the failure path is exercised, the
  success path is tested only against injected payloads.
- **No starting position has ever been pulled.** `player_game_logs` has the
  column (migration 008), `src/ingestion/starting_positions.py` is the writer
  and `scripts/pull_starting_positions.py` is the pass that runs it, but
  stats.nba.com is denied at this environment's proxy, so the column is NULL
  on every row today and `src/features/dvp.py` still abstains on a live panel.
  Two things follow and neither is a code change: the pull has to run where
  nba.com is reachable, and the one claim the fixtures cannot settle — that
  `boxscoretraditionalv3.position` is a STARTING position and not a listed one
  — is enforced as a refusal on the first real response rather than assumed.
  `docs/fouls_and_dvp.md` section 5.
- **DvP is measured on REB and wired for nothing.** The arm was run on an
  archive panel, which is not the panel production builds from. Wiring it
  needs the pull above to have happened and the arm re-run on a panel that
  carries real positions. `src/models/labels.py`, `docs/fouls_and_dvp.md`
  section 3a.
- **Every board row carries today's date whatever game it describes.**
  `research_slate_from_predictions` drops the detail row's `game_date` and
  stamps its `slate_date` parameter instead. `docs/integration_audit.md` §2.2.

Run `python -m scripts.verify_wiring` for the current state of all of these.
