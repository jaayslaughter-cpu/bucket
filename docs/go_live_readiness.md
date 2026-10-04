# Go-live readiness — audited 2026-09-28, reconciled 2026-10-04

RESEARCH_ONLY. Nothing here places or sizes a wager.

**This page was stale and was sending readers at the wrong work.** It said
scheduling was *ABSENT* and `prop_results` had *no writer*. Both were closed by
`8ed808c`, **dated the same day as this audit** — so the page was overtaken
within hours and never revisited. Anyone following its priority list would have
rebuilt a worker, a Dockerfile, a prop-results writer and a parlay ledger that
all already existed, and missed every blocker that is actually open.

Reconciled against `6d9a130`. Every verdict below was re-checked against the
tree rather than carried forward, and two of the original findings turned out
to have been **wrong when written** — those are corrected, not quietly
dropped.

**For the current state, run the checker rather than trusting this page:**

```bash
python -m scripts.verify_wiring     # counts move as items close; run it
```

`tests/test_go_live_readiness.py` pins this page in both directions: what it
calls fixed must stay fixed, and what it lists as open must stay open — when
one is closed, that test fails on purpose and says to update this page. That
failure is the mechanism that stops a week-stale audit happening twice.

Companion pages, each narrower and newer than this one:
`docs/integration_audit.md` (the seams, with a runnable verifier),
`docs/railway_deployment_audit.md` (deploy), `docs/deploy_railway.md` (how),
`docs/decision_board.md` (the board), `AGENTS.md` §7 (what is still not true).

---

## What is still open

Promoted to the top, because this is the part that drives work. Each line is a
FAIL or WARN from `scripts/verify_wiring`, so it is checkable rather than
asserted.

| # | Open item | Where it is established |
|---|---|---|
| **O1** | **Every board row carries today's date whatever game it describes.** `research_slate_from_predictions` drops the detail row's `game_date` and stamps its `slate_date` parameter. | §2.2 |
| **O2** | **The board's train/validation window is pinned to two 2025 dates** that never advance, and `run_board` re-fits every model on every slate job. | §2.3 |
| **O3** | **`src/ingestion/id_crosswalk.py` does not exist** and three modules name it as the fix for name-format mismatch; the exact-name join silently records zero gradeable rows when formats differ. | §1.2, §3 |
| **O4** | **Model artifacts do not survive a redeploy** — ephemeral filesystem, no volume declared. A platform step, not a code change. | `docs/railway_deployment_audit.md` §4 |
| **O5** | **A slate-level EV verdict is persisted as a per-row claim** (`market_status`). | §2.4 |
| **O6** | **`Game.tipoff_utc` has no writer.** The column exists and nothing populates it, so any tip-anchored scheduling has no times to anchor to. | `grep tipoff_utc src/` |
| **O7** | **Eight config blocks are declared and never read**, so editing the YAML changes nothing. | `docs/railway_deployment_audit.md` §2 |
| **O8** | **Self-referential evaluation.** `RESEARCH_LINE` is `{stat}_L10` and `over_hit` is measured against that same rolling history, so Brier and log-loss measure form against form until a posted-line archive drives line-aware training. | `src/models/labels.py`, `docs/DATA_GAPS.md` |
| **O9** | **Every entry resolves to `ProbabilitySource.MODEL`.** There is no sharp two-way NBA player-prop benchmark feed in reach, so nothing cross-checks the model's own number and the publication gate treats every board as model-sourced. Blocked externally, not by this repository. | `src/quant/dfs_entry.py`, `src/quant/publication_gate.py` |

**The two that gated everything else are closed** (see below), so a deployed
worker can now reach rows and a model. **O1 is what remains of the misleading
class**: a board row carrying today's date whatever game it describes could
mislead a reader rather than merely disappoint one, and it is the next thing to
fix.

### Closed since this page was reconciled

| Was | Closed |
|---|---|
| **No rows existed for a slate that had not been played.** `load_player_panel` reads completed box scores and `_filter_to_slate` keeps only the slate date, so a 09:00 PT run found an empty intersection and exited 0 with `success_no_data` — every day, without looking broken | 2026-10-04. `src/pipeline/forward_slate.py` adds one row per (player, scheduled game) from the ESPN schedule's **pre-tip** games plus each team's recent appearances in the panel, carrying **no box-score stat** so the rolling features read each player's own prior real games and the forward row has nothing of its own to leak. The lineup comes from the panel rather than `fetch_roster` on purpose: a roster fetch would need the ESPN-name → NBA-name crosswalk that still does not exist (O3). A denied schedule leaves the panel untouched and says so. `PROPIQ_FORWARD_SLATE` turns it off. |
| **A scheduled run could not find a model even when one was trained.** Scoring defaulted to `models/xgb_prop_over.json`, a directory that does not exist; `train-stats` writes to the comparison `artifacts_dir`; `run_slate` passes no `--model` | 2026-10-04. `main.resolve_model_artifact` tries `--model`, then `PROPIQ_MODEL`, then the newest `xgboost_*.json` in `artifacts_dir` **that has its `.meta.json` sidecar**, then the legacy path — and returns a reason naming every path it tried. The worker needs no argv plumbing: it calls `main.main([])` in-process, so the env var reaches the resolver directly. `artifact_registry` is deliberately **not** read; nothing writes to it, and resolving through a dead module is how `oddspapi` survived in the source precedence for months. |
| **The schedule and roster sources were unwired.** `espn_schedule.load_slate` was imported only by the CLI and `fetch_roster` by nothing | 2026-10-04 for the schedule, via the forward slate above. `fetch_roster` is **still uncalled**, and now deliberately: see the crosswalk note. |
| **`Projection` stored only `prob_over`**, so a whole line's push mass was unrecoverable and `1 - prob_over` was the wrong under | 2026-10-04. `prob_under` and `prob_push` columns, `migrations/005_projection_under_push.sql`, populated through `paper_research.resolve_two_way_model_probs`: a half line gets an exact under and a zero push; a **whole or unknown** line gets NULL for both plus the refusal reason in `notes`. A binary classifier has no push mass to split out, and that is recorded rather than guessed. |
| **No daily W/L reconciliation embed** — four builders, none reporting a settled day | 2026-10-04. `build_win_loss_embed` is the fifth, sent from the settlement job by `scheduler_worker.run_results_card`. It withholds a strike rate under 30 decided props with the reason, reports ROI only when a stake was actually recorded, and carries the metrics layer's CLV caveat with the CLV figure. Not behind the calibration gate, deliberately: that gate stops an uncalibrated model *probability* reaching a person, and this card carries none. |

---

## Original verdicts, then and now

The 2026-09-28 column is what this page said. The 2026-10-04 column is what the
tree does, re-checked item by item.

| # | Item | 2026-09-28 | 2026-10-04 | What changed |
|---|------|---|---|---|
| 1.1 | Late scratches / roster re-pull | PARTIAL + BLOCKED | **PASS** | `src/pipeline/scratches.py` applies a pre-tip filter from `espn_availability`, wired in `main.py`. A failed feed marks every row `UNVERIFIED` and drops nothing. |
| 1.2 | Rate limits, retries, daily caps | PASS | **PASS** | Unchanged; re-verified — `max_attempts=4`, `backoff_seconds=2.0`, `Retry-After` honoured, `min_daily_remaining=5`. |
| 1.3 | Feature alignment train ≡ serve | PARTIAL | **PASS** | `verify_feature_contract` is called in `score_prob_over`, and `catboost_pipeline` / `xgb_adapter` write the contract into the sidecar. See the correction below. |
| 2.1 | Fatigue applied, not bypassed | PASS | **PASS** | Unchanged; `build_feature_matrix` still calls `attach_fatigue_column` itself and folds the multiplier into every `{stat}_L2` exactly once. |
| 2.2 | Confidence / EV thresholds | PRESENT, inert | **PRESENT, inert** | Still correct, and still untested against real traffic. |
| 3.1 | Pre-game snapshotting | PARTIAL | **PARTIAL** | `migrations/003` separated observation time from ingest time, which was the sharp edge. No frozen pre-tip bundle is written. |
| 3.2 | Parlay & prop tracking in the DB | **FAIL — top blocker** | **PASS** | `pg_insert(PropResult)` exists in `repository.py`, `settlement/recorder.py` is its writer, and `main.py` calls it. `ParlayTicketRow` / `ParlayLegRow` + `migrations/004_parlay_ledger.sql` put parlays in Postgres with `PROPIQ_PARLAY_LEDGER=postgres`. |
| 3.3 | Daily result reconciliation | PARTIAL (blocked by 3.2) | **PARTIAL** | Unblocked: `settle_pending_props` grades, `metrics.get_performance_summary` aggregates, and the settlement job rebuilds the calibration report. Still no W/L embed (O9), and ROI needs a stake only a person records. |
| 4.1 | Automated scheduling | **ABSENT** | **PASS** | `Dockerfile`, `.dockerignore`, `scheduler_worker.py` (APScheduler, two PT-anchored cron jobs, `max_instances=1`), `APScheduler>=3.10` in both `requirements.txt` and `pyproject.toml`. Never built in CI — `scripts/validate_docker.py` is the build-and-smoke path. |
| 4.2 | Discord dispatcher formatting | PASS, one gap | **PASS** | The gap — no daily W/L summary — is closed. `build_win_loss_embed` is the fifth builder and `scheduler_worker.run_results_card` sends it from the settlement job. |
| 4.3 | Connection pooling | PASS | **PASS** | Unchanged: `pool_pre_ping=True`, `pool_size=5`, `max_overflow=5`, commit / rollback / `finally: close()`, `sslmode=require` for remote hosts, and `expire_on_commit=False` so ORM rows survive the session. |

---

## Two findings that were wrong when written

Kept, because a correction that is quietly deleted teaches nothing.

**1.3 said "column presence is checked; column order and dtype are not."**
That was false at the time. `xgboost_pipeline._matrix` does
`df[self.feature_cols]` — selecting in the frozen order — then coerces to
numeric and **raises** `DATA_NOT_AVAILABLE` on any value that was present and
failed to parse. Order and dtype were both enforced. What `FeatureSpec` added
later is the *reason*: xgboost already refuses a permuted or short column list
with its own `feature_names mismatch`, and the fingerprint check names the
market and the mismatched artifact instead.

**The OddsPapi fallback was called "existing tested architecture."** It had no
client module, no key reader and no ingestion path in this repository. What was
tested was the precedence *walker*, not a feed. Deleted 2026-10-04 on request;
`SOURCE_PRECEDENCE` is `("propline",)`, and `verify_wiring` now checks that
every name in it has an ingestion client. See `docs/decision_board.md`.

---

## The commits that closed the original P0s and P1s

Traced with `git log --diff-filter=A` and `git log -S` rather than read off the
subject lines — a first draft of this table guessed from subjects and got three
of five wrong. `tests/test_go_live_readiness.py` now parses this table and
checks each attribution against git.

| Original gap | Closed by |
|---|---|
| P0 write `prop_results` | `8ed808c` added `src/settlement/recorder.py`; `e3ad94d` added the availability skip; `3c77aee` built the calibration report from the graded rows |
| P0 parlay tickets and legs into Postgres | `8ed808c` added `migrations/004_parlay_ledger.sql` |
| P1 pre-tip scratch filter | `e3ad94d` added `src/pipeline/scratches.py`, wired in `main.py` |
| P1 wire `FeatureSpec` | `a27ff25` |
| P2 scheduling | `8ed808c` added `scheduler_worker.py` and `Dockerfile`; `3c77aee` added the board build and gated dispatch; `d9147da` hardened the image (non-root uid, `APScheduler` into `requirements.txt`, the full `.env.example`); `ed43c26` added `scripts/validate_docker.py` |
| P1 daily W/L embed | **closed 2026-10-04** — `build_win_loss_embed`, sent by `scheduler_worker.run_results_card` |
| P2 `prob_under` / `prob_push` on `Projection` | **closed 2026-10-04** — both columns, `migrations/005_projection_under_push.sql`, resolved through `paper_research.resolve_two_way_model_probs` |

`8ed808c` is dated **2026-09-28 — the same day as this audit.** The two "top
blocker" P0s were closed hours after being written up, which is why a page
frozen at that date reads so badly: it is not months stale, it was overtaken
immediately and never revisited.

---

## The three conflicts, settled

1. **Odds source: PropLine, and now the only one.** The Odds API is not used
   and must not be added. OddsPapi is deleted — see the correction above.

2. **Celery / Redis / Railway: resolved as recommended.** The recommendation
   was "Docker plus PT-anchored cron first, Celery only if per-game staggering
   proves necessary." That is what was built: APScheduler in one process, no
   broker. The `visibility_timeout: 86400` warning remains correct for anyone
   who adds Celery later — Redis's 1-hour default redelivers a long-ETA task —
   and `docs/go_live_pack_review.md` §3 reviews the uploaded Celery
   implementation and says what the cheaper path would be. Per-game staggering
   is still not built, and O8 is why: nothing populates a tip-off time.

3. **"Only flag high-probability plays": unchanged and still correct.**
   `build_decision_board` takes `min_ev`, `min_lean`, `require_valid_book` and
   `consider_only`, and nothing ranks by EV without VALID two-way odds and a
   no-vig fair probability. With every row abstaining, the threshold gates
   nothing — the right behaviour, and not evidence the threshold works.
