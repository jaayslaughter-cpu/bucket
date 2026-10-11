# Review: 19 external repositories (third round)

Date: 2026-10-05. RESEARCH_ONLY. Comparative analysis only — **nothing was
vendored** and no dependency was added. All 19 were cloned shallow and read;
every claim about data was **executed**, and the one HIGH verdict below rests
on a join actually performed against our panel, not on a README.

**22 URLs were supplied; 19 are unique** (`parlayparlor/nba-prop-prediction-model`,
`kyleskom/NBA-Machine-Learning-Sports-Betting`, `bene-art/props-scorer` and
`adibhar/prop-scout` each appear twice).

## 8 of the 19 were already reviewed, and those verdicts stand

This matters more than any new finding, so it goes first.

| Repository | Where it was reviewed | Verdict then |
|---|---|---|
| `DevanshDaxini/Sports-EV-Bot` | `external_repo_review_2026-09.md`, `external_feature_harvest.md` | "Richest feature library" — already harvested |
| `swar/nba_api` | `external_repo_review_2026-09.md` (2026-09-26 addendum) | MIT, usable as a dependency; endpoints catalogued |
| `mitchelldawkinsjr/NBA-Stat-Spot` | `external_repo_review_2026-09.md` | "Web app. No modelling" |
| `potternate/PropBet` | `external_repo_review_2026-09.md` | 0 Python, TS front end |
| `adibhar/prop-scout` | `external_feature_harvest.md` | B2B flag, same-opponent L5 |
| `kyleskom/NBA-Machine-Learning-Sports-Betting` | `external_feature_harvest.md`, `external_odds_repos_review.md` | reviewed |
| `matthew-hoty/nba-player-prop-analysis-shiny` | `external_repo_review_2026-09.md` | R Shiny dashboard |
| `johntomlinsonn/NBA-Game-Predictor` | `external_repo_review_2026-09.md` (as `GogateVarun`/`loganchoi` variants) | small Keras / notebook |

**And the 2026-09-26 addendum already reached a conclusion this session
ignored.** It says of `leaguedashptdefend`:

> **Supersedes P1.4.** Gives POSITION directly *and* a better defender feature
> than position-bucketed DvP

Position-bucketed DvP was nonetheless built this session
(`src/features/dvp.py`, `docs/fouls_and_dvp.md`). That is a process failure,
not a code defect: the review existed, named the better route, and was not
read before the work started. The layer is leakage-safe and measured, so it
is not wasted — but it is a **proxy for something we were already told how to
get directly**, and the honest status of `dvp.py` is "interim".

## Verdicts: the 11 not previously reviewed

| Target | Py LOC | Licence | Rating |
|---|---|---|---|
| **`shufinskiy/nba_data`** | 108 | MIT | **HIGH — the only one** |
| `optuna/optuna` | 70,227 | MIT | Marginal (a dependency, not a port) |
| `bene-art/props-scorer` | 1,008 | — | Zero |
| `gersimuca/nba-stats-predictor` | 779 | — | Zero |
| `JovaniPink/awesome-nba-data` | 889 | — | Zero (a link list) |
| `parlayparlor/nba-prop-prediction-model` | 335 | — | Zero (one downloader) |
| `joshdscan/NBA-Player-Prop-Prediction` | 136 | — | Zero, one idea noted |
| `whisdev/NBA-prediction-sports-betting` | 41 | — | Zero |
| `prishaa2/nba-scoring-prediction` | 0 | — | Zero (one notebook) |
| `vctrez/ai-agent-nba-props-analyzer` | 0 | — | Zero (4 JS files) |
| `NocturneBear/NBA-Data-2010-2024` | 0 | — | Zero (CSV dump) |

## Pillar 2 is settled across all 19: nothing to take

A scan for every technique the brief names, across all 19 repositories
(`*.py`, `*.ipynb`, `*.js`, `*.ts`, `*.R`):

| Technique | Repos containing it |
|---|---|
| Shin's method | **0 of 19** |
| Copula / covariance between legs | **0 of 19** |
| Negative binomial / `nbinom` | **0 of 19** |
| Zero-truncated distributions | **0 of 19** |
| Proxy rotation | **0 of 19** |

We have all four of the first group (`src/quant/devig_methods.py` with Shin by
bisection, `src/quant/parlay.py` Gaussian copula + `psd_repair.py`,
negbin-vs-Poisson dispersion selection by mean NLL). **Our quant engine is
ahead of every repository in this list on every item the brief asked about.**

The two de-vigging implementations that exist are both **additive
normalisation only**:

- `Sports-EV-Bot/src/core/analyzers/analyzer.py:301` — docstring states
  "Method: Additive normalization ('basic devig')".
- `kyleskom/src/Utils/Kelly_Criterion.py` — full Kelly, binary, no fraction and
  no cap, against our fractional Kelly with a cap and a tiered multi-outcome
  log-wealth solver. Its `american_to_decimal` is also misnamed: it returns
  net odds `b`, not decimal odds (+100 returns 1.0, not 2.0). The Kelly
  formula itself is correct.

## HIGH: `shufinskiy/nba_data`

**2.3 GB, 514 datasets, MIT, and already on local disk** — the shallow clone
pulled the data files, so nothing further needs fetching.

Verified contents, counted rather than read off the README:

| Family | Seasons |
|---|---|
| `nbastats` (stats.nba.com play-by-play) | 58 files, **1996–2024** |
| `shotdetail` | 60 |
| `nbastatsv3` | 60 |
| `pbpstats` | 50 |
| `matchups` | **18** |
| `datanba` (on-court coordinates) | 18 |
| `cdnnba` | 12 |

Two things make this the only HIGH verdict.

**1. It unblocks the pbp layer across the whole panel.** `src/features/pbp.py`
states the limit in its own docstring: "The logs supplied cover 2025-26 only. A
feature that exists in the validation window and nowhere earlier is not a
feature, it is the shape of a leak, and compare_models_on_panel now refuses
one." This supplies 1996–2024. Our panel spans 2018-01-01 → 2026-04-12, so
every season of it except the current one is covered.

**2. `matchups_*` is the defender-level data the 2026-09 review wanted, as
static files.** Measured on `matchups_2024.csv` (56.9 MB, **231,961 rows**):

```
game_id   person_id  position  matchups_person_id  matchup_minutes
22400001  1627759    F         203991              0:34
22400001  1627759    F         1630700             2:57
22400001  1627759    F         1630552             7:51
```

- `matchup_minutes` per (game, offensive player, defensive player) — **actual
  defender assignments**, which is what `dvp.py`'s position bucket is a proxy for.
- `position` — G 46,106 / F 45,752 / C 22,730, NaN 117,373 (starters-only, the
  same shape as the archive's `startingPosition`).
- **It joins.** Against `panel_raw.parquet`: 1,222 of 1,229 matchup games match
  on zero-stripped `GAME_ID`, and **569 of 569 players match on `PLAYER_ID`** —
  same id namespace, no crosswalk needed.

**Why this beats the nba_api route in THIS environment.** The 2026-09-26
addendum named `leagueseasonmatchups`, `boxscorematchupsv3`,
`leaguedashptdefend` and `leaguehustlestatsplayer` as the answers. They are the
better answers — richer columns, per-season freshness — but `stats.nba.com`
returns nothing through this environment's proxy (`curl` with full NBA headers:
HTTP `000`). These files are already local. **The endpoint catalogue is right
for Railway; this is right for here.** Both, not either.

## Marginal: `optuna/optuna`

MIT, 70k LOC, a hyperparameter-optimisation library — a dependency, never a
port. It is the only repository in the list with infrastructure worth reading
(13 backoff sites, 12 pool-configuration sites in its RDB storage layer), but
its pooling is for its own trial storage and does not transfer.

Against our tuning today (`scripts/fit_ensemble_weights.py`,
`scripts/feature_selection.py`, `xgboost_early_stopping`): Optuna would add
TPE/pruning over our current approach. **Not recommended now.** Our open
measurement debt is whether existing features help at all
(`docs/fouls_and_dvp.md`, `docs/minutes_weighted.md` — both unmeasured), and
tuning hyperparameters before knowing which features earn a place optimises
the wrong layer.

## Zero, with reasons

- **`bene-art/props-scorer`** (1,008 LOC) — `scripts/{fetch_data,train_model,evaluate_model}.py`. Standard sklearn pipeline, no prop-specific maths.
- **`gersimuca/nba-stats-predictor`** (779) — Flask + scraper + sklearn. No props, no odds.
- **`JovaniPink/awesome-nba-data`** (889) — a curated link list; the Python is `build_resource_index.py` / `validate_readme.py`, tooling for the README. Useful as a bibliography, zero code value.
- **`parlayparlor/nba-prop-prediction-model`** (335) — one `download_data.py`.
- **`joshdscan/NBA-Player-Prop-Prediction`** (136) — notebooks. **One idea worth recording**: `scripts/get_similar_players.py` builds player comparables from `playerindex` + `boxscoreadvancedv2`. We have no similarity model, and our cold-start path abstains (`prior_games`, `CASCADE_STATUS`). The implementation drops `POSITION`, `HEIGHT` and `WEIGHT` before computing similarity, which is most of what makes two players comparable — so the idea is worth more than this code.
- **`whisdev/NBA-prediction-sports-betting`** (41 Py, 23 JS) — scaffold.
- **`prishaa2/nba-scoring-prediction`**, **`vctrez/ai-agent-nba-props-analyzer`**, **`NocturneBear/NBA-Data-2010-2024`** — 0 Python. The last is a CSV dump superseded by `shufinskiy/nba_data`.

## What the comparison exposed in OUR code

Two findings that are about us, not them.

**1. `line_diff._line_adjust_fair_prob` is a linear fudge whose docstring
claims otherwise.** It reads "Soft log-ish shift: ~3% fair-prob per point" and
the code is `delta = -line_diff * 0.03` — strictly linear, one constant for
every market. Sports-EV-Bot's equivalent
(`analyzer.py::_adjust_for_line_difference`) is **better on two counts**:
per-stat factors (a point of PTS is not a point of AST) and genuine log scaling
with diminishing returns.

Neither is right. **We already own the principled answer and are not using
it**: `src/models/residuals.py::over_under_push_from_dispersion` and the
NGBoost location-scale fits give a distribution per row, and re-pricing at a
different line is that distribution's CDF evaluated at the new line — exact,
not a factor. Porting their heuristic would be replacing one fudge with a
better fudge. The fix is ours to make.

**2. Alembic is declared and unused.** `requirements.txt` pins
`alembic>=1.13.0`; no `alembic.ini`, no `env.py`, and no Python imports it. Our
migrations are hand-numbered `.sql` files (002–007, no 001) with **nothing
recording which have been applied to a given database**. `NBA-Stat-Spot` has a
working `backend/alembic/env.py`, but its pooling is *worse* than ours
(`pool_pre_ping` alone, no `pool_size`, no `pool_recycle`). The gap is not
"adopt Alembic" — our `.sql` files carry `NOT VALID` constraints and explicit
no-backfill reasoning that autogenerate would never produce. The gap is that
**no table records applied migrations**, and the cheap fix is a
`schema_migrations` table, not a framework.

## Pillars 3 and 4: our implementations are ahead

**Ingestion resilience.** `swar/nba_api`'s HTTP layer
(`src/nba_api/stats/library/http.py`) has **a `timeout` parameter and nothing
else** — no retry, no backoff, no 429 handling. Its headers are a subset of
ours: it sets `User-Agent`, `Accept-Language`, `Connection`, `Referer`; we also
set `Origin`, `x-nba-stats-origin` and `x-nba-stats-token`, which are the
headers `stats.nba.com` actually gates on. `Sports-EV-Bot` sleeps
`random.uniform(0.5, 1.5)` and, on a 429, **falls back to a stale cache rather
than retrying** (`prizepicks.py:202`). Our `espn_client.get_json` retries with
exponential backoff and honours `Retry-After`.

**One resilience idea worth taking**: `Sports-EV-Bot/src/sports/nba/injuries.py`
reads **two independent injury feeds** (`INJURY_URL` and `CBS_INJURY_URL`) with
fallback. We use ESPN alone, and `src/pipeline/scratches.py` treats a failed
feed as UNVERIFIED — correct, but a second source would mean fewer UNVERIFIED
slates. Marginal value, cheap.

**Operational utilities.** Nothing. The only `pool_recycle`/PgBouncer hits in
19 repositories are inside Optuna's own trial storage.

## Pillar 1: one genuinely new feature idea, and a leakage warning

`Sports-EV-Bot/src/sports/nba/features.py` is the only real feature library in
the list, and it was already harvested in `external_feature_harvest.md`. What
that harvest did **not** record:

**Its DvP is leaky in two ways, and both are defects we independently found and
fixed this session.** `add_defense_vs_position` (line 181):
1. `df.groupby(['OPPONENT','POSITION'])` then `.rolling(10)` on **player
   rows** — a team faces four or five guards a night, so its "L10" spans about
   two games. Our `dvp.py` aggregates per (defender, game, bucket) first.
2. `league_pos_avg = df.groupby(['POSITION','SEASON_ID'])[stat].transform('median')`
   — a **season-wide median, future-inclusive**, used to build
   `OPP_*_ALLOWED_DIFF`, which **is** in their model's feature list. Ours is an
   as-of expanding mean, shifted.

So its DvP features are leaky *and* used. Independent confirmation of both
findings, and a reason not to port that function.

**Its per-market feature families are leaky too** —
`add_rebound_specific_features` emits `FOUL_TROUBLE_REB_LOSS = (PF >= 4) * -2.5`
from the **current game's** foul count (you cannot know a player will reach four
fouls before tip), plus `ORB_RATE`, `DRB_RATE`, `TEAM_REB_CONCENTRATION` and
`REBOUND_OPPORTUNITY` from same-game box scores. 25 such same-game assignments
across the file. **None of them reach its model**: `train.py::get_features_for_target`
is an explicit allowlist of `_Season`/`_L5`/`_L10`/`_L20`/`_Median`/`_STREAK`/
`_CONSISTENCY` variants, all shift-1. The columns sit on the frame waiting for
someone to add them. Contrast `src/features/fouls.py`, which computes the same
foul-trouble idea from **prior** games.

**What is genuinely new and not leaky:**

| Idea | Where | Why it is new to us |
|---|---|---|
| **Combo-market DvP by linear combination** | `features.py:205-217` | `OPP_PRA_ALLOWED = OPP_PTS + OPP_REB + OPP_AST`, and the same for the `_DIFF`. Allowed means add, so this is valid, and it extends DvP to combo markets for free. We have PRA as a market and no combo DvP |
| **`SB` (steals+blocks) as a market** | `train.py:131` | We do not carry it; it is `STL + BLK` and trivially derivable |
| **First-half markets** (`PTS_1H`, `PRA_1H`, `MIN_1H`) | `train.py:135-138` | We have **no** 1H markets. A real product gap, blocked on 1H box-score data we do not ingest |
| **`{STAT}_LOC_MEAN`** | `train.py:155` | Per-player home/away split mean. We feed `IS_HOME` but no per-player split |

Its `IS_4_IN_6` / `IS_FRESH`, `EXP_POSS`, `_STREAK`, `_CONSISTENCY` and
per-target feature selection are all convergent with what we have
(`is_3_in_4`/`is_4_in_5`, `DEF_PACE_L10`, `{STAT}_STREAK_ABOVE/BELOW`,
`_DEFENSE_BY_MARKET`/`_FORM_BY_MARKET`) — reassuring, not novel.

## Extraction plan

Ordered by value over cost. Nothing here is a code copy; two are data.

**1. Ingest `shufinskiy/nba_data`'s pbp for 1996–2024 (HIGH).** The data is at
`scratchpad/audit2/shufinskiy_nba_data/datasets/nbastats_*.tar.xz`, already
local.
- Extend `scripts/build_pbp_panel.py`, which already takes `--pbp-dir` and
  already runs `check_log_completeness` against the box score's independent
  count. Its `EVENT_COLS` names the 15 columns `src/features/pbp.py` reads;
  diff those against `nbastats_*`'s header **before** building anything.
- The completeness check is the gate, not a formality: that module records two
  logs that "named every game, spanned the right dates, carried no duplicates,
  and still held only a fraction of each game's events — once at 32% and once
  at 84%".
- Then re-run `scripts/feature_ab.py --layer pbp` over the widened span. The
  layer is currently confined to 2025-26.
- Copy the `.tar.xz` files into `data/external/` (gitignored) rather than
  depending on a scratch path.

**2. Replace `dvp.py`'s position bucket with real matchup minutes (HIGH).**
`matchups_*.csv` gives `(game_id, person_id, matchups_person_id,
matchup_minutes, position)`, joins at 99.4% of games and 100% of players, and
covers 18 seasons.
- New module beside `dvp.py`, not inside it: `dvp.py` is measured and
  leakage-safe and should stay until the replacement is measured too.
- The leakage rule is the same and is the whole job: a matchup prior must be
  built from the defender's **prior** games, shift-1 within season, exactly as
  `build_opponent_allowed` does.
- Keep `dvp.py` as the fallback for rows the matchup data does not cover, and
  say so in both docstrings.
- Measure with `scripts/feature_ab.py --layer <new> --wire-under-test` against
  `--layer dvp`, which is the comparison that decides which survives.

**3. Combo-market DvP (LOW cost, small win). DONE, with two deviations.**
`src/features/dvp.py` now emits `DVP_PRA_ALLOWED_L10` and
`DVP_PRA_INDEX_L10`. Write-up: `docs/fouls_and_dvp.md` section 2's
"Combination markets".

- **`DVP_PR_*`, `DVP_PA_*` and `DVP_RA_*` were NOT added.** PR, PA and RA are
  not markets in this project (`labels.LAUNCH_MARKETS` is PTS/REB/AST,
  `POST_LAUNCH_MARKETS` is FG3M/STL/BLK/PRA), so nothing could read them and
  `feature_ab._dvp_for_market` has no market to route them to. This item asked
  for six columns computed on every build for nobody — the state AGENTS.md
  records four feature layers sitting in. `dvp.DVP_COMBOS` is the one place to
  turn one on, guarded by a test that every combo is a market.
- **"Linear combinations of the existing per-stat columns" is not what was
  built**, and the item's own premise is why. "Allowed means add" holds only
  while the means share a denominator, and both of this layer's aggregations
  skip nulls per column. The sum therefore happens PER PLAYER-GAME, masked to
  rows where every part is known, and the one column then travels the same
  path a base stat travels. On today's panel the two agree to 7.1e-15 (no
  component is ever null in the archive), so it is a guard rather than a fix —
  but `player_game_logs` has pts, reb and ast independently nullable, so the
  live panel can produce exactly the partial row that breaks it.

The `_INDEX` caveat this item named was right and it matters more than it
looks: the summed trio has mean 3.009 against the re-derived column's 1.001
and correlates with it at only 0.917, so it is not even a monotone restatement.
Also measured, and worth stating against this item's "small win": combining
averages the position split away. `PRA`'s between-bucket ratio is 1.11, the
flattest of the four, because rebounds (C-heavy, 2.17x) and assists (G-heavy,
1.96x) cancel.

**The arm has now been run and the answer is no.** `line_aware` Brier is worse
on 4 of 4 folds at 2.8–4x the fold spread, `catboost` calibrated likewise at
0/4, `ensemble` and `xgboost` are nil, and raw ECE improves — the same
Brier-worse / ECE-better pattern `labels._EXCLUDED_AS_REDUNDANT` already
documents for two excluded families, read the same way, because Brier is the
metric these calls are made on. Not wired; the columns stay as
`minutes_weighted`'s do. Full table: `docs/fouls_and_dvp.md` section 3b.

So this item's "small win" was optimistic, and so was my own restatement of it:
the geometry predicted a weak effect and I wrote exactly that, where the
measured answer is **adverse**. A column carrying little signal is not merely
weak in a gradient-boosted model — it is one more split candidate competing
with features that do carry signal.

**4. Fix our own line-diff with dispersion, not their factors (MEDIUM).
DONE.** `src/quant/line_diff.py` now inverts the fitted distribution at the
book's line to recover the mean the book's price implies, then re-evaluates at
the pick'em line. `LineDiffResult.method` is `"dispersion"` or `"heuristic"`,
with `method_reason` naming why the fallback ran. The "soft log-ish shift"
docstring is corrected: the fallback is strictly linear and there is no
logarithm in the module. Tests: `tests/test_line_diff_dispersion.py`.

What the fix buys, measured: the same one-point move is worth **+0.162** of
fair probability off a 4.5 line and **+0.069** off a 24.5 line, where the flat
`0.03` said the same thing for both. The heuristic understated both, badly at
low lines.

**Three things this item did not know, found while doing it.**

- **It was fixing a dormant path.** `adjusted_fair_prob_over` is read by
  nothing outside the module's own tests, and `enrich_row_with_pickem` — the
  only caller of `pickem_vs_book_line_diff` — has no caller of its own. So the
  flat `0.03` was not mispricing anything in production. Worth doing before
  something reaches for it; not worth describing as a pricing fix.
- **A separate incoherence, in the function being replaced.** `side="under"`
  flipped the shift's sign, so one pair of lines and one price returned
  `adjusted_fair_prob_over` of 0.5408 for the over and 0.4808 for the under.
  P(over) at a line is a property of the line. It is now side-independent, and
  `side` is gone from the private helper's signature — a parameter that cannot
  change the answer invites the belief that it can. Nothing in production
  changes: the one call site hardcodes `"over"`.
- **The comparison had to be made conditional on no push.** A two-way book
  price de-vigs to a two-outcome probability — a push is voided, not priced —
  while `over_under_push_from_dispersion` correctly reports three. Comparing
  them directly would understate the book's view at every whole-number line by
  exactly the push mass. Half-point lines coincide, which is why this is easy
  to miss.

**And one bug of mine, caught by its own test.** The inversion's mean ceiling
capped the bracket's *growth* but not its *start*, so a line above half the
ceiling began the search past it, the growth loop never ran, and the guard
could not fire — an uninvertible price came back as a confident mean. The test
that asserted the refusal got an answer instead.

**5. A `schema_migrations` table (LOW). DONE, both halves.**
`src/db/migrations.py` + `scripts/migrate_db.py` apply the files in version
order and record each with a sha256 in a `schema_migrations` table;
`alembic` is dropped from `requirements.txt` and `pyproject.toml`. Tests:
`tests/test_db_migrations.py`.

The item offered "either drop `alembic` or wire it". **Dropping was right and
wiring was not**, for a reason the item did not have: these migrations are not
mechanical DDL. 006 and 008 are each ~60 lines of reasoning about why a column
is nullable and why a default would be a fabrication, and that reasoning is
the valuable part of the file. Converting them to `op.add_column` revisions
would either lose it or duplicate it somewhere it can drift. Alembic earns its
keep when migrations are generated from model diffs; here they are written by
hand on purpose, and what was missing was never the authoring tool. It was the
ledger.

**What the item under-specified, and what the implementation added.**

- **The checksum, not just the version.** A version number cannot catch a file
  edited AFTER it was applied: every row looks present and correct while the
  repository and the database have diverged. Drift is a refusal, not a
  warning — re-running an applied migration is not the fix, and guessing which
  half of an edited file is already in place is worse.
- **One transaction for the whole run.** A migration that applied and then
  failed to record would be re-run against a schema it had already changed.
  Each file and its ledger row commit together or not at all.
- **No `009_schema_migrations.sql`.** A ledger that itself has to be applied by
  hand before anything can be recorded reproduces the problem it solves. The
  runner creates it from the ORM model with `checkfirst=True`, which also lets
  the whole thing be tested against SQLite while production stays Postgres.
- **Status is the default.** A migration tool whose no-argument behaviour
  changes the schema is one typo away from the wrong `DATABASE_URL`.
- **`docs/deploy_railway.md` was the real victim.** It carried the
  authoritative list of migrations to run, named 002–005, and had been stale
  since 006. It now points at the runner, because a database answering the
  question cannot go stale.

**Not recommended:** Optuna (optimises the wrong layer while feature value is
unmeasured), Sports-EV-Bot's DvP or per-market families (leaky), its
line-adjustment factors (a better fudge is still a fudge), kyleskom's Kelly
(ours is strictly more general), and a second injury feed (real but low value
against the open blockers).

## Security

A credential scan across all 19 repositories — `api_key|apikey|secret|token|
password` assigned a 16+ character literal, excluding placeholders and
`os.environ`/`getenv`/`process.env` reads — found **one real hit**:

| File | Finding |
|---|---|
| `mitchelldawkinsjr/NBA-Stat-Spot/docker-compose.dev.yml:8` | `POSTGRES_PASSWORD=nba_props_password` |
| `mitchelldawkinsjr/NBA-Stat-Spot/docker-compose.prod.yml:94` | `POSTGRES_PASSWORD=nba_props_password` — the same literal, in the **prod** compose file |

Low consequence in itself — it is a compose-local Postgres for a container that
is not ours, not a third-party API key — but it is a hardcoded credential in a
file named `prod`, and it is the anti-pattern our own `.env.example` and
`Dockerfile` are explicitly built to avoid ("NO SECRETS ARE BAKED IN. There is
no COPY of .env, no ARG for a key, and no default credential"). Nothing to take
from this repository's deployment setup.

No API keys or tokens were found in any of the 19. The `jspdf.src.js` hits in
the same repository's vendored Highcharts bundle are empty-string library
defaults, not credentials.

**An earlier draft of this section said the scan "returned nothing".** That was
wrong: the first pass covered `*.py`, `*.ipynb`, `*.js`, `*.ts` and `*.R` and
did not include `*.yml`, which is where the hit is. Recorded because a security
section that overstates a clean result is worse than no security section.
