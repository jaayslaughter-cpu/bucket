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

**O-numbers are stable and are never reused.** A closed item keeps its number
and moves to the table below, so the open list starts at O4 rather than being
renumbered — renumbering silently repoints every cross-reference at a
different item. That has already happened once: `tests/test_forward_slate.py`
opens "O1 — rows for a slate that has not been played" and
`tests/test_under_and_push.py` opens "O8 — the under and the push", from an
earlier numbering, so a bare O-number in a test docstring older than
2026-10-05 may not mean what it means here. Cite the item, not just the
number.

| # | Open item | Where it is established |
|---|---|---|
| **O4** | **Model artifacts do not survive a redeploy** — ephemeral filesystem; the volume is declared in the Dockerfile but must be mounted and SEEDED by hand. There is no boot-time fetch from object storage and the Dockerfile no longer claims there is. `scheduler_worker.check_model_artifact()` now reports an unseeded volume as an ERROR at boot instead of letting every row abstain silently. A platform step, not a code change. | `docs/railway_deployment_audit.md` §4 |
| **O6** | **`Game.tipoff_utc` has no writer** — the DB column exists and nothing populates it. NO LONGER BLOCKS TIP-ANCHORED SCHEDULING: `scheduler_worker.schedule_prelock_jobs` reads tip-offs straight from `espn_schedule.load_slate`, which parses them, rather than from the table. The column is still unwritten and still wrong to read. | `grep tipoff_utc src/` |
| **O7** | **Eight config blocks are declared and never read**, so editing the YAML changes nothing. | `docs/railway_deployment_audit.md` §2 |
| **O8** | **Self-referential evaluation.** `RESEARCH_LINE` is `{stat}_L10` and `over_hit` is measured against that same rolling history, so Brier and log-loss measure form against form until a posted-line archive drives line-aware training. | `src/models/labels.py`, `docs/DATA_GAPS.md` |
| **O9** | **Every entry resolves to `ProbabilitySource.MODEL`.** There is no sharp two-way NBA player-prop benchmark feed in reach, so nothing cross-checks the model's own number and the publication gate treats every board as model-sourced. Blocked externally, not by this repository. | `src/quant/dfs_entry.py`, `src/quant/publication_gate.py` |

**The misleading class is now empty.** O1 and O2 were the two defects that
could mislead a reader rather than merely disappoint one, and both are closed
below. What remains (O4-O9) either narrows the output or is blocked outside
this repository; none of it publishes a claim that is not true.

**O3 was listed open here after it had been closed**, which is its own small
version of the same problem — a readiness page that outlives its facts teaches
the wrong map. `src/ingestion/id_crosswalk.py` exists, three modules route
through it, and `verify_wiring`'s name-mismatch guard passes. It is in the
closed table below where it belongs.

### Closed since this page was reconciled

| Was | Closed |
|---|---|
| **O1 — every board row carried today's date whatever game it described.** `research_slate_from_predictions` stamped its `slate_date` parameter and never read the detail row's own `game_date` (written at `compare.py:839`); `ResearchSlateRow` had no game-date field at all. A board built today from a window ending 2025-02-15 produced February rows, each stamped today, with nothing saying otherwise — a reader could not tell a projection from a backtest row | 2026-10-05. `game_date` is carried on `ResearchSlateRow` AND `BettingDecisionCandidate`, so it reaches the CSV dispatch reads and the Discord embed a person sees. It deliberately does **not** fall back to `slate_date` — inheriting the stamp is the bug, and a null says "unknown" where a copy would say "today". An off-slate row warns naming both dates; the embed banners the card and marks each row. `tests/test_board_game_date.py` (14 tests); `verify_wiring`'s board-date guard went FAIL → PASS |
| **O2 — the board's train/validation window was pinned to two 2025 dates.** `run_board` fell back to `train_end="2025-01-15"` / `validation_end="2025-02-15"`, and `compare_models_on_panel` scores `(train_end, validation_end]`, so the window receded one day further into the past on every run and nothing reported it | 2026-10-05. `scheduler_worker.board_window()` anchors on the Pacific calendar day: fit `<= yesterday`, score `(yesterday, today]`, which is exactly the rows `forward_slate` wrote for today's scheduled games. An explicit override still wins, because a backtest board is a legitimate thing to build — what it no longer does is happen by accident: a `validation_end` in the past is logged as a backtest with its lag in days, and the window travels on `run_board`'s result. `verify_wiring`'s window guard went WARN → PASS |
| **O5 — a slate-level EV verdict was persisted as a per-row claim.** `evaluate_ev_gate` asked the gate about each captured prop line and collapsed the answers into one status; `assemble_projections` wrote that onto every row and `persist_projections` stores it per row as `market_status`. One priced prop out of three hundred labelled all three hundred READY_FOR_EVALUATION — a claim about a different row, in the only form the column is ever read | 2026-10-05. `evaluate_ev_gate` returns `by_key`, one verdict per `(player_name, market)`, joined onto the rows on the same key `_attach_prop_lines` uses for LINE. **A second defect surfaced while fixing it:** the loop built its `MarketContext` without `line`, and `market_ev_gate` refuses a context with no finite line, so every row abstained for a reason the caller had created and a perfect two-way price would have abstained too; `market`, `player_name`, `is_pickem` and `payout_multiplier` were dropped on the same floor, so pick'em rows were never routed either. `migrations/007` adds `market_status_reason`, because DATA_NOT_AVAILABLE has two causes — no line reached this row, or a line was posted and refused — that lead a reader to opposite conclusions |
| **One market per run.** `score_prob_over` takes a single `model_path`, so only the market its sidecar named could carry a probability; AST, REB and FG3M came out null and `settlement.recorder` skips a row with no probability, making three of four markets ungradeable however many models were trained. `resolve_model_artifact` already took a `market` and globbed `xgboost_{MARKET}.json`; nothing called it that way | 2026-10-05. `score_prob_over_by_market` resolves and scores one artifact per market; `assemble_projections` takes a market → Series mapping and still accepts a bare Series. An explicit `--model`/`$PROPIQ_MODEL` is scored ONCE, with its market from its own sidecar — looping an explicit path would score one booster four times and hand three results to markets it was not fit for. A market with no artifact stays null rather than borrowing one, and a sidecar claiming a different market than it was resolved for is skipped |
| **The worker had no tip-anchored job**, so nothing re-checked a slate between the 09:00 PT board and tip-off. This module's own docstring said a cron expression cannot express a slate — "tip-offs move by hours" — and it then shipped two fixed clocks anyway, because the schedule feed was out of reach when it was written | 2026-10-05. `schedule_prelock_jobs` arms ONE one-shot job per pre-tip game at that game's own tip-off minus 35 minutes (bounded 5-180, `PROPIQ_PRELOCK_LEAD_MINUTES`), armed at boot and again after each board build, idempotent on a `(slate date, event id)` job id. `run_prelock` re-runs the scratch filter on that game's RECOMMENDED rows and posts a withdrawal card naming any player now OUT or DOUBTFUL — the caller `src/pipeline/scratches.py` was written for and never had. It deliberately refits nothing and re-prices nothing, and a test pins what it must not reach for. A game with no tip-off, a slot already past, and a denied schedule are each skipped with a stated reason rather than guessed at |
| **O3 — `src/ingestion/id_crosswalk.py` did not exist** and three modules named it as the fix for name-format mismatch | 2026-10-04, and this page went on listing it as open until 2026-10-05. The module is deterministic rather than fuzzy (NFKD, combining marks stripped, then EQUAL): no cutoff separates `Jokic`/`Jokić` at 81.8-91.7 from `Jalen`/`Jaylen Williams` at 96.6. The port found a live safety bug in `scratches._normalise`, which was lowercase-only, so a player ESPN reported OUT was labelled AVAILABLE |
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
| 4.2 | Discord dispatcher formatting | PASS, one gap | **PASS** | The gap — no daily W/L summary — is closed. `build_win_loss_embed` is the fifth builder and `scheduler_worker.run_results_card` sends it from the settlement job. `build_prelock_correction_embed` is the sixth, sent by `run_prelock`: the only surface here that RETRACTS a published recommendation, which is why it is not an abstention embed with a composed string. |
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
