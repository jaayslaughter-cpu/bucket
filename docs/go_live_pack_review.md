# Review: `PropIQ_GoLive_Readiness_2026-10-01.zip`

Date: 2026-10-02. RESEARCH_ONLY. Comparative analysis of an uploaded archive.
**Nothing was vendored.** Two artifacts were adopted as native work in this
repo's own style and are named below; the rest is recorded with the reason it
was not.

28 files, 187 KB, built by its own `scripts/pack_go_live_readiness.py`.

## 0. What it is, and which lineage it came from

Two of its files are **byte-identical** to this repository's current HEAD:

```
src/features/minutes_weighted.py    8022 bytes   identical
tests/test_minutes_weighted.py     11172 bytes   identical
```

Every other overlapping file is a **different, leaner lineage** of the same
project:

| file | in pack | in this repo |
|---|---|---|
| `src/features/builder.py` | 13,251 | 25,498 |
| `src/models/labels.py` | 4,966 | 21,769 |
| `src/models/feature_spec.py` | 4,603 | 11,755 |
| `scripts/feature_ab.py` | 4,618 | 25,857 |
| `scripts/nba_model_cli.py` | 62,208 | 80,717 |
| `Dockerfile` | 791 | 5,164 |
| `.env.example` | 1,752 | 7,545 |

These are not newer versions of this repo's files. The pack's `builder.py`
imports every layer eagerly and has functions this repo does not
(`build_layer1_baselines`, `_sort_panel`, `_opponent_defensive_rating`); this
repo resolves layers lazily by module path so an absent module simply
contributes no layer. Its `feature_spec.py` is a strict subset of this one.
Copying any of them in would be a regression, so none were.

## 1. Adopted

### `docs/minutes_weighted.md`
The pack's standalone page for the layer, rewritten against this repo's paths
and extended with the three defects wiring it surfaced. The measurement in it
is this repo's own (214,381-row panel), and the pack agrees with it.

### A builder-path alignment test
The pack's `tests/test_builder_minutes_weighted_integration.py` tests the layer
**through `build_feature_matrix`**, which nothing here did — the existing tests
call the layer directly and check the registry for membership. That gap is
real: the builder's layer loop catches `Exception` and logs a warning, so a
layer can be registered, run, and produce nothing with no test failing.

The pack's version was not copied, because it does not measure much: its
assertions (`"PTS_MW_L5" in feat.columns`, `notna().sum() >= 1`) survive the
layer abstaining, and comparing the builder path against the bare layer fails
for a legitimate reason — the builder supplies `MIN_SEASON` while the bare
layer must build a minutes baseline at `min_periods=3`, so the two average
different numbers of weighted games early in a season. Measured, not guessed:
they disagree on the first three rows of each player and agree after.

What went in instead is
`tests/test_minutes_weighted.py::test_the_builder_attaches_each_player_s_values_to_that_player_s_rows`,
where each player scores a different constant, so every non-null value is that
player's own number whatever the window contains. Mutation check: grouping the
layer by `SEASON` alone (values bleeding across players) passes **every**
layer-level test in that file and is caught only by this one.

## 2. What the review turned up in this repo

Checking whether the pack's claim — *"Built in `build_feature_matrix`; not in
`default_feature_cols`"* — actually held end-to-end here found a defect in
three other layers. `src/features/season.py`, `tests/test_season_key.py`,
and `docs/season_key.md`. In short: `halflife`, `hot_hand` and
`sports_ev_features` each derived a missing `SEASON` as `GAME_DATE.dt.year`
(splitting the season at 1 January) **and wrote it into the returned frame**,
so one layer's guess became every later layer's grouping key — and silently
defeated `minutes_weighted`'s abstention, which exists to refuse exactly that
guess. Full write-up in `docs/season_key.md`.

That is the pack's real contribution here, and it is worth saying plainly: the
value was in checking a claim, not in the code.

## 3. Not adopted: `src/ops/` (Wave 6 / 6b)

Eight modules, 1,438 lines: a Celery app, Beat/worker tasks, pure ETA planning,
an ESPN scoreboard slate resolver, a shadow orchestrator, a scratch filter, a
pre-tip snapshot writer, and a paper reconcile. Plus `docker-compose.yml`
(redis + celery-worker + celery-beat + two one-shot services) and 12 tests.

**Four of the eight do not import in this repository**, because they target
modules this lineage does not have:

| pack module | needs | here |
|---|---|---|
| `shadow_orchestrator` | `src.ingestion.espn_context`, `src.quant.discord_notify` | `src/ingestion/espn_availability.py`, `src/notify/discord.py` |
| `reconcile` | `src.quant.paper_ledger`, `src.quant.win_loss_tracker`, `src.quant.discord_notify` | `src/quant/paper_research.py`, `src/notify/discord.py` |
| `pre_tip_snapshot` | `src.db.repository.insert_shadow_snapshot` | no such function |
| `celery_tasks` | `shadow_orchestrator` (transitively) | — |

Both test files import the absent modules too, so neither runs here. Celery and
redis are not dependencies of this project.

Two **are** compatible, and were checked rather than assumed:

- `nba_slate.py` — reads `games.nba_game_id / game_date / tipoff_utc /
  home_team_abbr / away_team_abbr / status`, every one of which exists on
  `src/db/models.py::Game`, and `src.utils.timezones`' four helpers, which all
  exist. It would work as written.
- `eta_dispatch.py` — pure planning, no Celery import, same helpers.

### The design question they raise, which is a real one

This repo schedules with **APScheduler on two fixed Pacific times** (slate
09:00, settlement 03:30) and `docs/deploy_railway.md` says why:

> The schedule is fixed, not tip-off-driven. Re-anchoring needs a schedule
> feed, and timing logic that has never been exercised against real data would
> look adaptive while being untested.

The pack supplies that schedule feed — ESPN's scoreboard — and anchors each
game at `tipoff − 30m`. That is a genuine improvement for late scratches,
which land between a 09:00 slate build and a 19:30 tip. Its own `Dockerfile`
comment is honest that the ESPN fetch is the weak link, and this environment's
network policy 403s every data host, so it cannot be exercised here either.

Adopting it would mean a broker (redis), two more always-on services, and a
`visibility_timeout=86400` that the pack is right to flag in capitals: Redis's
3600s default would redeliver a 10-hour-out reserved task repeatedly. That is
a deliberate architecture decision with a running cost, not a merge. **It is
the user's call, and it is not made here.**

If it is wanted, the smallest honest path is: port `nba_slate.py` +
`eta_dispatch.py` (both compatible today), have the existing APScheduler
`BlockingScheduler` add per-game `DateTrigger` jobs at `tip − 30m` for the
current slate, and skip Celery, redis and compose entirely. One process, no
broker, no visibility timeout to get wrong.

## 4. Not adopted: `Dockerfile`, `.env.example`, `AGENTS.md`

- **`Dockerfile`** (791 bytes) runs as **root**, pins nothing, hardcodes
  `TZ=America/Los_Angeles`, and copies a `config.json` this repo does not have.
  This repo's (5,164 bytes) runs as uid 10001, sets `TZ=Etc/UTC` so the
  container default is a decision rather than the host's, and keeps `.env` and
  `data/` out of the build context. Adopting the pack's would undo
  `docs/railway_deployment_audit.md` steps 1, 3, 4 and 5.
- **`.env.example`** (1,752 bytes) is a subset; this repo's documents all 28
  variables the worker reads.
- **`AGENTS.md`** is a useful project constitution and this repo has none — but
  as written it names six docs and one module that do not exist here
  (`docs/propline.md`, `docs/oddspapi.md`, `docs/betting_decision_layer.md`,
  `docs/wave6_shadow_live.md`, `docs/wave6b_celery_eta.md`,
  `src/ingestion/nba_com_boxscore.py`), plus `config.json`. A constitution that
  cites absent files teaches the wrong map. Worth adapting on request; not
  adopted silently.

Its substantive rules — RESEARCH_ONLY, no auto-wager, no bankroll sizing bound
to model outputs, `.shift(1)` on every grouped rolling feature, UTC storage with
Pacific display, explicit `DATA_NOT_AVAILABLE` over inference, probabilistic
language only, abstain from EV/CLV/stake unless `MarketContext.status ==
"VALID"` with two-way American odds, pick'em multipliers are not valid EV — are
already enforced in code and tests here, and none of them conflicts with
anything in this repository.

## 5. Its own verification block, checked

`README_GOLIVE.md` lists five commands. Against **this** repo:

| command | result here |
|---|---|
| `pip install -e ".[ml,db,ops]"` | no `ops` extra in this `pyproject.toml` (`deploy` is the deploy extra) |
| `pytest tests/test_minutes_weighted.py` | passes — 23 tests |
| `pytest tests/test_wave6_shadow_live.py …` | not present; imports absent modules |
| `feature_ab --layer minutes_weighted --wire-under-test` | present and wired |
| `nba_model_cli celery-health` / `schedule-slate --plan-only` | no such subcommands here |

---

## 6. Pack v2 (`fb10c2f1-`, uploaded 06:48)

A second upload, same filename, 65,888 bytes against the first's 63,634. It is
**v1 plus Docker validation tooling** and nothing else — every other file,
including the two byte-identical to this repo's HEAD, is unchanged.

| | |
|---|---|
| new | `.dockerignore`, `docs/docker_validate.md`, `scripts/validate_docker.ps1`, `scripts/docker_dispatch_smoke.py` |
| changed | `Dockerfile` (now `pip install -e ".[ml,db,ops]"` instead of `requirements.txt` + a bare `celery[redis]`), `README_GOLIVE.md`, `scripts/pack_go_live_readiness.py` |

**None of it is drop-in**, and for one reason: every step validates the
Celery stack from §3, which is not here. `docker compose build celery-worker`
needs a `docker-compose.yml` this repo does not have; `celery-health` and
`schedule-slate --plan-only` are not subcommands of this `nba_model_cli`;
`docker_dispatch_smoke.py` imports `src.ops.celery_tasks`; and
`pip install -e ".[ml,db,ops]"` names an extra this `pyproject.toml` does not
declare (`deploy` is the one that exists). The script is PowerShell, which
also says something useful about where it is meant to run.

### What transferred

The *shape* — build, bring the service up, run a health command **inside the
image**, tear down — is exactly Railway roadmap **step 2**, which has been
open since the Dockerfile was written and which that file admits to in its own
header comment. So `scripts/validate_docker.py` does that for this repo's
image: no compose, no redis, no celery, and in Python rather than PowerShell so
one file covers Windows and Linux.

It also goes somewhere the pack's does not. Two of its six in-image checks
mount a **mode-0555 directory at `/app/data`** and require
`scheduler_worker.check_state_dir()` to report it unwritable, then mount a
writable one and require no false alarm. That is the trap
`docs/deploy_railway.md` §2b describes — a volume mount shadows the image's
`chown`, the container runs as uid 10001, and because every writer catches
broadly the real symptom is a nightly *"calibration report failed"*. The page
has claimed since it was written that `check_state_dir()` catches this. Nothing
had ever executed it.

`tests/test_validate_docker.py` (33 tests) drives every preflight check to
failure against a synthetic tree, because a check that cannot fail reports PASS
either way. Writing them found a real hole: the credential scan missed
`ENV DISCORD_WEBHOOK_URL=https://…` — `WEBHOOK` was required to sit immediately
before the `=` — and missed a secret on an `ENV ... \` continuation line, which
is the shape a leak would actually take in this repo's own Dockerfile.

### What the pack's `.dockerignore` caught that ours missed

Ours is otherwise stricter (it covers `.env.*`, `*.pem`, `*.key`, `secrets/`,
which the pack's does not), but the pack excludes `catboost_info`, `*.zip` and
`*.pdf` and ours did not. `catboost_info/` is a training log CatBoost drops in
the working directory — 33 KB of it is in this checkout right now, gitignored,
and it was going into an image layer. Added, with the matching entries in
`validate_docker.MUST_BE_IGNORED` so the check covers them.

Measured build context after the change: **7.78 MB**, largest entries `tests/`
(4.27) and `src/` (2.65).
