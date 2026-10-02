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
- `src/features/builder.py:570` `assert_no_lookahead()` runs over whatever the
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
deliberately not shipped.

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
`src/features/builder.py:74` and `src/features/fatigue_load.py:284` each do a
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
`config/master_guideline_props.yaml` — not in the repo; `main.py:97`
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
- **The Docker image has never been built.** `scripts/validate_docker.py`
  preflight passes; the build and the six in-image checks need a machine with
  a daemon.
- **Model artifacts do not survive a redeploy.** They live under
  `data/external/model_runs/`, on an ephemeral filesystem, and a fresh
  container abstains on every row without looking broken. This is the one
  remaining deploy blocker and it is a platform step, not a code change.
  `docs/railway_deployment_audit.md` §4.
