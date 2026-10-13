# Candidate repository evaluation — 28 public NBA betting/modelling repos

**Date:** 2026-10-11 · **Scope:** NBA only · **RESEARCH_ONLY.** Nothing here is
a wagering recommendation, and no claim here is a claim of profit.

A dual-lens evaluation of 28 pasted GitHub repositories against PropIQ —
**Pillar 1** (modelling / probability quality) and **Pillar 2** (autonomy and
infrastructure) — with four deliverables: Strategic Verdict, Model Extraction
Checklist, Autonomy & Infrastructure Extraction Checklist, and Integration Risk
Analysis.

## 0. Method, and the thing that reframed the exercise

The 28 URLs deduplicate to 26 distinct repositories. **25 cloned**;
`Risky-Scout/nba-player-props-model` failed (private or deleted) and is
unevaluated — read every count below as covering 25. Each was cloned shallow
and read, not summarised from its README.

**Then the decisive check: 18 of the 25 have already been reviewed in this
repository.** `docs/external_repo_review_2026-10.md` covers 14 of them with
explicit ratings, `docs/external_repo_review_2026-10-05.md` and
`docs/external_repo_review_2026-09.md` add four more, and
`docs/external_feature_harvest.md` records a feature harvest across several.
I found this only after profiling all 25 — the cost of not checking our own
`docs/` first.

Better still, **those reviews were acted on**, which is why so much of this
pass comes back redundant:

| 2026-10-05 finding | State on 2026-10-11 |
|---|---|
| §2.1 **HIGH** — `EdgarParra565`'s PSD repair + shrinkage: "we have no PSD repair", port it our way, report how far the matrix moved, abstain above a threshold (`docs/external_repo_review_2026-10.md:127`) | **Done.** `src/quant/psd_repair.py:105` with a refusal path, wired into `src/quant/parlay.py:317` |
| §3 — `mattdspecht`'s `player_roll10_pf`: "we have **no PF feature at all**" | **Done.** `src/features/fouls.py:110` — `PF_L5`, `PF_L10`, `PF_SEASON`, `PF_PER_MIN_L10`, `PF_TROUBLE_RATE_L10` |
| §3 — explicit minutes×efficiency interaction, "we do not carry it explicitly" | **Partly.** `src/features/matchup.py:101` carries `PTS_PER_MIN` matchup features; the named `usg_x_opp_drtg` form is still absent |
| §3 — LightGBM quantile band for the stat (`docs/external_repo_review_2026-10.md:189`) | **Still open.** Carried forward below as **M1** |

So the genuinely new surface is **7 repositories**: `Kevocado/NBA_Predictor`,
`julianthant/sports_betting`, `PerryGraham/Betting-Model-NBA`,
`SmittyDukes/Minutes-Extension-ML-API`, `hugohamynen/nba-predictor-dashboard`,
`jensenbenjamin/NBA_Betting_Model` (0 LOC) and `api-evangelist/thrivefantasy`
(0 LOC). The two substantial ones are `julianthant` (~88k LOC, 106 test files,
but **zero** `.shift(1)` calls) and `Kevocado` (~31k LOC, 86 test files, 21
`.shift(1)` calls) — and `Kevocado` is where the single best finding on this
page comes from.

**Citation convention.** A backticked citation with no repository prefix, such
as `src/quant/devig_methods.py:136`, points into **this** tree, and
`tests/test_deploy_audit_citations.py` checks it still resolves. A citation into
a candidate carries the repository and an `@`, as in
`EdgarParra565/player-performance-forecaster @ nba_model/data/hourly_update.py:74-128`;
those lines are from the clone read on 2026-10-11 and nothing here can keep them
fresh. The same test checks the `@` form is only ever used for a path that is
**not** in our tree, so it cannot be used to dodge the line check on one of ours.

**One correction worth recording.** An early grep of mine reported several
PropIQ capabilities as ABSENT. That grep had a malformed alternation (`\|` is a
literal pipe under `grep -E`). Redone correctly, every Pillar 1 technique the
brief names is already present. A tooling bug nearly produced a "we should port
this" list of things we already have.

## 1. Deliverable — Strategic Verdict

**Do not port code from any of the 25. Take three ideas, cite them, write them
ourselves.** Four independent findings, any one of which would be close to
decisive.

### 1.1 Licensing makes most of the set unusable as source

**20 of 25 carry no licence file.** No licence means all rights reserved:
copying their code here is a copyright problem, not a style preference. Six
have one and all six are MIT — `Andresperez397/nba-shot-making`,
`DanielTomaro13/Basketball-Modelling`, `Khavel/proplab-mcp`,
`mattdspecht/NBA-Contextual-Modeling`,
`parlayparlor/nba-prop-prediction-model`, `roni-altshuler/nba_predictor` — and
one of those (`Khavel/proplab-mcp`) contains **0 lines of Python**, so five are
usable in practice.

This inverts the exercise. Every repository that is interesting on Pillar 1 or
Pillar 2 — `julianthant` (the only full Shin/power/equal-margin de-vig suite
with tests), `EdgarParra565`, `Kevocado` — is **unlicensed**. The licensed ones
are smaller and mostly weaker. Neither prior review did a licence audit; this
is the main new contribution of this pass.

### 1.2 Pillar 1: PropIQ already implements every technique named

Verified by reading our files, not assuming them:

| Technique | Where it already lives |
|---|---|
| Shin (1993) de-vig, + multiplicative, odds-ratio, power | `src/quant/devig_methods.py:136` |
| Count overdispersion (negbin, zero-inflated, family by likelihood) | `src/models/residuals.py:79` |
| Copula / correlated same-game legs | `src/quant/parlay.py`, `src/quant/leg_correlation.py` |
| Nearest-PSD correlation repair | `src/quant/psd_repair.py:105` |
| Isotonic + Platt calibration | `src/models/prob_calibration.py:34` |
| Poisson-binomial hit-count distribution | `src/quant/dfs_payouts.py:513` |
| Chronological / purged cross-validation | `src/models/oof.py:125` |
| Quantile models (P10/P50/P90) — **for minutes** | `src/models/minutes_model.py:57` |
| Travel distance (haversine) | `src/features/schedule.py:69` |
| Altitude | `src/features/fatigue_logic.py:22` |
| Push-conditional probability | `src/quant/line_diff.py:103` |
| Calibration gate before publication | `src/quant/publication_gate.py:150` |
| Fouls, usage, defender matchup, blowout | `src/features/fouls.py`, `absences.py`, `dvp.py`, `shot_zones.py` |

On "shin": grepping for it matched 12 of 25, almost all substring noise
(`washington`, `shooting`). There are exactly **two** real implementations —
`julianthant/sports_betting @ backend/src/nba/utils/odds_math.py:169`
(unlicensed) and
`roni-altshuler/nba_predictor @ backend/services/prediction/market.py:160`
(MIT). Both are two-outcome only. Ours is n-way, and — the difference that
matters — ours **raises** when the solver cannot bracket a root
(`src/quant/devig_methods.py:99`), where `roni-altshuler`'s `_shin_z` returns
`0.0` and then re-normalises, silently degrading to proportional de-vig with no
signal to the caller. Porting that would be a downgrade.

### 1.3 Pillar 2: no candidate has an in-process scheduler at all

**Zero of 25 use APScheduler, Celery, or any in-process scheduler.** Everything
that schedules does so externally: macOS launchd property lists
(`BangLetsGetIt` 13, `EdgarParra565` 4, `DanielTomaro13` 1) or GitHub Actions
cron (7 repositories, 1–2 workflows each).

`EdgarParra565` has the most elaborate deployment story, and reading it settles
the question: its autonomy is a **launchd-driven Chrome scraper on a residential
IP**, whose own README states it "**will not work in GitHub Actions or a
container**" because sportsbooks fingerprint the TLS stack, and which has been
**off since 2026-09-28** over geo-blocks. Its Fly.io app is a read-only front
end fed by hourly SQLite snapshots published from the owner's Mac. That is not
a more autonomous architecture than ours; it is a less autonomous one with a
human and a laptop inside the loop. The container-native equivalent of its best
part — the publish/poll artifact hand-off — already exists as
`src/models/artifact_store.py` and `scripts/fetch_artifacts.py`.

Correcting an earlier note of mine: I had recorded `EdgarParra565` as doing
**proxy rotation**. It does not. Those matches are `X-Forwarded-For`
trusted-hop parsing for rate limiting. There is no proxy rotation in the set,
and we would not want it.

### 1.4 The one place a candidate is genuinely ahead of us

`Kevocado/NBA_Predictor` refuses to publish a snapshot built on a stale model:
it compares the artifact's `trained_at` against now **at publish time** and
raises, rather than warning
(`Kevocado/NBA_Predictor @ src/nba_predictor/public_snapshot.py:1-20`). Its
stated reason is the failure that prompted it — a snapshot stamped "generated
today" while serving models trained three weeks earlier, where "nothing on the
page said so."

We had no equivalent. That is **A2** below, it was the most valuable thing in
all 25 repositories, and it is now built.

## 2. Deliverable — Model Extraction Checklist

One open item and one record-only. Both are **ideas to implement ourselves**,
and both come from `mattdspecht/NBA-Contextual-Modeling` — MIT, ~3.1k LOC, and
the only licensed candidate that is leakage-disciplined (24 `.shift(1)` calls
against 15 `rolling(` calls).

### M1 — Quantile band for the stat itself *(candidate; open since 2026-10-05)*

- **Not new.** `docs/external_repo_review_2026-10.md:189` already identified
  this and framed it better than I would have: an empirical q10/q90 band is
  distribution-free, and it is a candidate answer to the case our `prob_push`
  column deliberately leaves NULL. This entry carries it forward with two
  additions.
- **Addition 1 — the licence is clean.** `mattdspecht` is MIT, so unlike most
  of this set the idea can be taken from a source we are allowed to read and
  cite.
- **Addition 2 — what must NOT come with it.** Their conversion of the interval
  into a probability is strictly worse than ours: they rebuild a *symmetric*
  normal with `scale = half_width * 0.78` and then set
  `prob_under = 1.0 - prob_over`
  (`mattdspecht/NBA-Contextual-Modeling @ nba/modeling/inference.py:393-396`).
  That discards the asymmetry the quantile models exist to capture **and**
  ignores pushes, which `src/quant/line_diff.py:103` exists to handle. If we
  adopt the quantile heads, they must feed the fitted count distribution, never
  replace it.
- **The gap in our tree.** `src/models/residuals.py:79` predicts variance as
  `mean * phi` with **one phi per market**, so spread scales with the
  projection and with nothing else: two players with the same projected points
  get the same implied spread even if one has far more volatile minutes. Note
  we already run quantile heads for *minutes*
  (`src/models/minutes_model.py:57`), so this is the same technique applied one
  layer out, not a new dependency.
- **Gate.** It must beat the global phi on leakage-safe walk-forward
  calibration (`src/models/walk_forward.py`), measured, with a mutation test
  proving the comparison is real. Until then it stays a candidate.

### M2 — `prev_arena` lag as a travel primitive *(record only, no action)*

`mattdspecht/NBA-Contextual-Modeling @ nba/etl/preprocess.py:112-114` lags
arena latitude/longitude and date to get travel between consecutive games —
the same construction as our `TRAVEL_MILES` (`src/features/schedule.py:69`),
and the 2026-10-05 review already found their altitude and rest handling
*worse* than ours. **No action.** Recorded so the next reader does not
re-derive it.

### Explicitly not extracted

| Candidate | Why not |
|---|---|
| `julianthant` Shin/power/equal-margin suite + tests | Unlicensed; `src/quant/devig_methods.py` already covers all four n-way with a failing-solver contract. Also **zero** `.shift(1)` calls across ~88k LOC |
| `roni-altshuler` closed-form Shin | MIT, but two-outcome only and silently degrades to proportional on solver failure |
| `EdgarParra565` negative binomial | Selects the count family by a moment heuristic at call time; ours selects by likelihood and persists `dispersion_selection` into the sidecar |
| `EdgarParra565` `simulate_sgp` | Hardcodes `corr: float = 0.6` and feeds a hand-built covariance to `multivariate_normal` with no PSD repair — the very gap its own `correlation_calibration.py` closed for us |
| `Kevocado` retrain pipeline metrics | Records log-loss/Brier/AUC but does not gate promotion on them; `src/quant/publication_gate.py:150` already gates on ECE with a minimum sample |
| `arjun2garg/NBA-Correlation` | Nothing beyond `src/quant/leg_correlation.py` fitted on our own data |
| Everything in `docs/external_feature_harvest.md`'s "Not taken" table | Already decided, with reasons |

## 3. Deliverable — Autonomy & Infrastructure Extraction Checklist

Two items. The set is weaker than PropIQ on this pillar almost everywhere, and
most of what looks like autonomy is a laptop — but these two are real.

### A2 — Refuse to publish on a stale model artifact — **IMPLEMENTED 2026-10-11**

- **Source.** `Kevocado/NBA_Predictor @ src/nba_predictor/public_snapshot.py:1-20`:
  compare the model's `trained_at` against now at publish time and **raise**
  when it exceeds a named `MAX_MODEL_AGE_DAYS`. Their notes are explicit that a
  refusal raises rather than warns, because the alternative is a staleness
  banner the reader has to notice — "and the whole point is that they would
  not."
- **The gap in our tree, verified.** `scripts/railway_healthcheck.py:152`
  checked that artifacts *exist*, parse, and were not fit on demo-sized data —
  never how old they are. There was no `model_age` or `MAX_MODEL_AGE` anywhere.
  We bound **data** freshness (`main.market_frames_freshness`, default 10 days)
  but not **model** freshness.
- **The key is `saved_at_utc`, which took inspecting the artifacts to establish.**
  I first wrote that `src/models/artifact_registry.py:52` put
  `registered_at_utc` in every sidecar. It does not: that module builds a
  separate append-only `model_registry.jsonl`, **nothing writes to it**, and
  `main.py:182` says in as many words that it is
  deliberately not read because "resolving through a dead module is how
  `oddspapi` stayed in the source precedence for months". What training actually
  persists is `saved_at_utc`, present in 8 of the 9 sidecars under
  `data/external/model_runs` — and `tests/test_artifact_freshness.py` now
  asserts that coupling, so a training change that drops the key fails a test
  instead of silently withholding every board.
- **Why it matters here specifically.** We are a nightly Discord worker with no
  human in the loop. If a retrain silently stops, the board keeps going out,
  stamped with tonight's date, on an artifact that could be months old, and
  nothing anywhere says so. This is the autonomous-operation failure mode that
  looks most like success.
- **Built with TWO axes, where the source has one.** File age alone certifies
  the pipeline it should catch: a retrain that keeps running against the same
  stale frame rewrites `saved_at_utc` nightly forever while the model learns
  nothing. So training recency is measured too, from `train_end_date` — and
  **against the newest game we hold, not against today**
  (`src/db/repository.py:469` is the one reader in that module with no upper
  window, because a window there would hide exactly the lag it looks for).
  Against today, every artifact would go stale each July and the alarm would
  fire for four months, which is how a staleness alarm gets muted.
- **Where it lives.** Measurement in `src/models/artifact_freshness.py:177`
  (`OK` / `STALE` / `UNDATED` / `MISSING`, reporting *which* axes ran so
  "not measured" cannot be read as "fine"); the refusal in
  `src/quant/publication_gate.py:378` as a second verdict beside
  `calibration_gate`, combined by `first_refusal` so the reason that fired is
  the reason shown; wired into the nightly dispatch at
  `scheduler_worker.py:1424` and into the deploy probe as a FAILURE at
  `scripts/railway_healthcheck.py:320`.
- **Licence note.** `Kevocado` is unlicensed, so this is the idea and the
  reasoning, cited — not their code. Every line is ours.

### A1 — Cross-process advisory lock on the write path *(real gap, small fix)*

- **Source.**
  `EdgarParra565/player-performance-forecaster @ nba_model/data/hourly_update.py:74-128`
  takes a **non-blocking** `fcntl.flock` and exits `75` (`EX_TEMPFAIL`) when it
  is already held, so an overlapping run is *skipped* rather than queued; its
  publisher separately exits `3` when the database is mid-write rather than
  shipping a torn snapshot.
- **The gap in our tree.** `scheduler_worker.py:2000-2003` sets
  `max_instances: 1` and `coalesce: True`, which is correct and sufficient
  **within one process** and does nothing about a second one: an operator
  running `railway run python main.py` while the slate job is firing can
  interleave writes to the artifact directory and the parlay ledger. We run
  `numReplicas: 1`, so this is operator concurrency, not scaling — which is the
  kind that happens once and is hard to diagnose afterwards.
- **Shape of the fix.** A non-blocking lock around the artifact/ledger write
  path, taken by both the scheduled job and the manual entry point, which
  **declines** rather than waits and says so in the log.
- **Caveat on their exit codes.** `EX_TEMPFAIL`-and-relaunch works only because
  launchd retries. Our container must not treat a declined lock as a boot
  failure; it is a successful no-op.

### Explicitly not extracted

| Candidate | Why not |
|---|---|
| launchd LaunchAgents (`BangLetsGetIt` 13, `EdgarParra565` 4) | macOS-only; we deploy a Linux container |
| `EdgarParra565` scraping Chrome on a residential IP | Its own README says it will not run in a container, and it is disabled over geo-blocks. Also a sportsbook-scraping path we ruled out in favour of the licensed PropLine feed |
| GitHub Actions cron (7 repos) | Weaker than an in-process scheduler for a slate that must react to tip-off times; we already have misfire grace and coalescing |
| `EdgarParra565` Mac→object-store→Fly snapshot hand-off | Sound pattern, already ours: `src/models/artifact_store.py`, `scripts/fetch_artifacts.py` |
| Streamlit / Shiny front ends (`matthew-hoty`, `hugohamynen`, `EdgarParra565`) | Out of scope: this is not a Streamlit app; notifications go to Discord |
| `Khavel/proplab-mcp` | MIT, but 0 lines of Python |

### Worth one sentence, though it is not extractable

`Kevocado/NBA_Predictor @ src/nba_predictor/models/candidate_race.py:5` records
a **negative** result in its own module docstring — a ridge candidate at
log-loss 0.7059 against a naive 0.6868, "worse than doing nothing clever" — and
keeps the module as the record of three things it now refuses to do. That is
the same discipline as our own "make it fail first" rule, arrived at
independently. Not code to take; worth knowing someone else landed on it.

## 4. Deliverable — Integration Risk Analysis

Ordered by damage if ignored.

1. **Copyright (highest).** 20 of 25 are unlicensed, and *all three* ideas
   above come from unlicensed or idea-only sources (`Kevocado` unlicensed,
   `EdgarParra565` unlicensed, `mattdspecht` MIT). Any copied function,
   docstring or test from the unlicensed ones is an infringement that lives in
   our git history permanently. **Mitigation:** all three extractions are ideas
   with cited sources, to be written from scratch; this document names which
   repository each came from so the provenance is not lost.

2. **Re-doing work we have already done.** The real cost this pass found is not
   a bad port — it is that 18 of 25 were already reviewed and I profiled all 25
   before checking `docs/`. **Mitigation:** the table in §0 states what was
   already decided and what has since been implemented, so the next pass starts
   from there. Any future candidate review should grep `docs/external_*` for
   the repository name first.

3. **NCAA contamination — violates a standing instruction.** Several candidates
   are NBA+NCAA: `spahnrl/prj_BookieX` (3,729 files matching),
   `BangLetsGetIt/NBA-NCAA-Betting-Models` (89), `api-evangelist/wager-api`
   (12), with smaller counts in `julianthant`, `roni-altshuler` and
   `DevanshDaxini`. The standing rule is **no NCAA**. **Mitigation:** none of
   those is a source for anything above; no extraction touches a file in them.

4. **Banned odds sources re-entering through a port.** `BangLetsGetIt` and
   CourtSide are built on the Odds API, which
   `docs/external_feature_harvest.md:90` bans outright. Porting a feature from
   a repository built on a banned feed tends to drag the feed's payload shape,
   and then its key, in with it. **Mitigation:** our odds come from PropLine via
   `PROPLINE_API_KEY` and nothing else, and `scripts/railway_healthcheck.py`
   fails when that variable is absent.

5. **Secret leakage from candidate fixtures.** Not hypothetical: a fixture I
   wrote earlier in this project used Supabase's real `sbp_` token prefix and
   GitHub push protection correctly blocked the push. Candidate repositories are
   full of example keys and committed `.env` samples. **Mitigation:** nothing
   from these clones is copied, and the clones live in the session scratchpad,
   outside the repository. (The 2026-10-05 review's own credential scan across
   17 of them found no hardcoded keys, which is reassuring but not a licence to
   copy.)

6. **Importing a weaker contract on top of a stronger one.** The subtlest risk,
   and the one that nearly happened twice: `roni-altshuler`'s Shin returns a
   silent fallback where ours raises, and `mattdspecht`'s probability step sets
   `prob_under = 1 - prob_over`. Both are *regressions* dressed as features. A
   port preserving the candidate's error handling would quietly undo deliberate
   design in `src/quant/devig_methods.py:99` and `src/quant/line_diff.py:103`.
   **Mitigation:** M1's gate is a measured walk-forward comparison, and its text
   names exactly which part must not come with it.

7. **A2's own failure mode.** A staleness refusal that is too tight turns a
   working nightly board off for a reason the operator cannot see, which is a
   worse outage than the one it prevents. **Mitigation:** the constant is named
   and pinned by a test, the healthcheck reports it as a FAILURE *before* the
   publication path refuses anything, and the existing Discord failure alert
   (`notify_failure`) already carries the reason out to a human.

8. **Leakage.** `julianthant` has **zero** `.shift(1)` calls across ~88k LOC
   with 106 test files; `DanielTomaro13` (4,024 LOC) and `roni-altshuler`
   (16,495 LOC) also have none. `roni-altshuler` is game-level so some lagging
   may be in SQL; `DanielTomaro13` has no tests at all. **Mitigation:** the
   project rule — every grouped rolling feature shifts, enforced in `tests/` —
   applies to anything we write regardless of where the idea came from. The one
   repository we take a model idea from, `mattdspecht`, is the one that shifts.

9. **Scope creep.** The set holds four repositories of 49k–88k LOC. Reading them
   is cheap; adopting their architecture is not. **Mitigation:** the three
   extractions are one feature flag, one constant, and one lock.

## 5. What this evaluation did not do

- `Risky-Scout/nba-player-props-model` could not be cloned and is unevaluated.
- No candidate's model was *run*, and no claim here compares measured
  performance — only method and contract. M1 is explicitly gated on a
  measurement we have not taken.
- **A2 is implemented** (see its entry above), with eight mutations each
  failing a named test. **A1 and M1 are not** — they are scoped findings.
- A2 refuses to publish; it does not refuse to *score*. A stale artifact still
  produces a board CSV, which is deliberate: the file is the record an operator
  inspects to see what was withheld and why.
- No candidate code was copied into this repository. The clones are in the
  session scratchpad and are not part of the tree.
