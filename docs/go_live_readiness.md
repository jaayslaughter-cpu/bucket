# Go-live readiness audit — 2026-09-28

RESEARCH_ONLY. Nothing here places or sizes a wager.

Audited against the shadow-live checklist by inspecting the tree, not by
reading the packs' own status tables. Every FAIL below is backed by a
command whose output is quoted in the finding.

## Verdict

| # | Item | Verdict |
|---|------|---------|
| 1.1 | Late scratches / roster re-pull | **PARTIAL + BLOCKED** |
| 1.2 | API rate limits, retries, daily caps | **PASS** |
| 1.3 | Feature alignment train ≡ serve | **PARTIAL** |
| 2.1 | Adjustment overrides (fatigue) applied not bypassed | **PASS** |
| 2.2 | Confidence / EV thresholds | **PRESENT, inert pre-season** |
| 3.1 | Pre-game snapshotting | **PARTIAL** |
| 3.2 | Parlay & prop tracking in the DB | **FAIL — top blocker** |
| 3.3 | Daily result reconciliation | **PARTIAL (blocked by 3.2)** |
| 4.1 | Automated scheduling (Docker / cron) | **ABSENT** |
| 4.2 | Discord dispatcher formatting | **PASS, one gap** |
| 4.3 | Connection pooling | **PASS** |

## 3.2 — the blocker: `prop_results` has no writer

Two proven holes, both of which make the feedback loop a no-op.

**`prop_results` is read and graded but never written.** The repo has six
`pg_insert` targets:

    src/db/repository.py:50    pg_insert(TeamGameStat)
    src/db/repository.py:126   pg_insert(PlayerGameLog)
    src/db/repository.py:190   pg_insert(GameMarketLine)
    src/db/repository.py:216   pg_insert(PropLineSnapshot)
    src/db/repository.py:332   pg_insert(Projection)
    src/db/repository.py:360   pg_insert(PipelineRun)

`PropResult` is not among them, and `PropResult(` never appears as a
constructor anywhere in `src/`, `scripts/` or `main.py`. Meanwhile
`settlement/runner.py:141` selects rows `WHERE outcome_status ==
'PENDING'` to grade, and `settlement/metrics.py` aggregates wins, losses,
pushes, stake units, profit units and both CLV columns off that same
table. So the grader updates rows nothing creates, and every P&L and CLV
number the metrics layer can report is an aggregate over zero rows.

**Parlays never reach Postgres at all.** `src/db/` contains no parlay
table. `parlay_log.py:521,525` writes `parlay_tickets.csv` and
`parlay_legs.csv` under `data/external/parlay_log/` — container-local
files, not the queryable store retraining needs.

Until a pick is persisted at the moment it is taken, nothing in section 3
can be verified, and the reconciliation job in 3.3 has nothing to
reconcile.

## What already holds up

**1.2 rate limits** are better than the checklist asks.
`ingestion/propline.py` sets `max_attempts=4` with exponential
`backoff_seconds=2.0`, honours `Retry-After`, parses live quota from
response headers into `PropLineQuota`, refuses to begin new work below
`min_daily_remaining=5` so a long run cannot exhaust the quota mid-slate,
and deliberately does not retry 401/403 because a credential problem does
not improve on the second attempt. `boxscores.py` uses
`retry_attempts=3` with `retry_backoff ** attempt`; `nba_playbyplay.py`
mirrors it deliberately so the two behave identically.

**2.1 fatigue is applied exactly once.** `build_feature_matrix` calls
`attach_fatigue_column` itself (`builder.py:419`) and folds
`fatigue_multiplier` into every `{stat}_L2` (`builder.py:517`). `main.py`
verifies that rather than re-applying it; applying it twice
double-counts the multiplier, which an earlier revision did.

**4.3 pooling** needs nothing. `db/session.py` sets `pool_pre_ping=True`
(survives the Supabase pooler dropping idle connections), `pool_size=5`,
`max_overflow=5`; `session_scope` commits on success, rolls back on
exception and closes in `finally`; there is no SQLite fallback by design;
`sslmode=require` is appended for non-local hosts; and the pooler port
(6543) vs direct port (5432) tradeoff is documented in the module.

**4.2 Discord** treats the webhook as the credential it is: regex
validated, redacted on every log path, `assert_payload_safe` refuses
payloads carrying credentials or PII, claim language is blocked, and
Discord's real limits (25 fields per embed, 10 embeds per message,
character caps) are enforced before sending rather than discovered as a
400. Three builders exist: decision board, parlay, abstention.

## Gaps, in priority order

**P0 — write `prop_results`.** A `persist_prop_results` upsert beside the
existing five, called where a pick is actually taken, keyed so a re-run
overwrites its own rows. This unblocks 3.2 and 3.3 together.

**P0 — parlay tickets and legs into Postgres.** Two tables mirroring
`ParlayTicketRecord` / `ParlayLegRecord`, keeping `assert_export_safe` on
the way in. The CSV path stays as the local mirror.

**P1 — pre-tip scratch filter.** `ingestion/inactive_players.py` and
`features/absences.py` are wired into the builder (layer order is
`sports_ev -> absences -> teammate_cascade`, which is load-bearing), but
they are a *training-panel* feature, not a T-30 filter: nothing drops a
projection when a player is ruled out after that projection was written.
Also `stats.nba.com` is denied 403 at this environment's proxy, so the
existing puller cannot be exercised from here — an ESPN-injuries source
is the reachable alternative.

**P1 — wire `FeatureSpec`.** `score_prob_over` does load `feature_cols`
from the model's `.meta.json` sidecar, refuses to score when columns are
missing, and abstains per-row at unsupported lines. But
`models/feature_spec.py`, which computes the SHA fingerprint that would
catch train/serve drift, has **zero callers outside itself**. Column
presence is checked; column order and dtype are not.

**P1 — daily W/L reconciliation embed.** No formatter exists for the
checklist's daily win/loss summary.

**P2 — scheduling.** Absent entirely: no Dockerfile, no compose, no
`deploy/`, no Procfile, no railway config, no crontab, and no
celery / redis / apscheduler in `requirements.txt` or `pyproject.toml`.

**P2 — `prob_under` / `prob_push` on `Projection`.**
`residuals.over_under_push_from_dispersion` computes all three legs and
has six production callers, but `Projection` stores only `prob_over`, so
a whole-number line's push mass is unrecoverable after the fact and
`1 - prob_over` is the wrong under.

## Three conflicts to settle before building

1. **The Odds API.** The proposed 06:00 PT slate initializer "fetches
   baseline odds from The Odds API" and pulls the schedule from
   BallDontLie or SportsData.io. Every doc in this project states *No The
   Odds API*, with PropLine as primary. Not wiring a banned source on the
   strength of a schedule sketch — confirm which source is intended.

2. **Celery / Redis / Railway.** The repo has no scheduler at all, so this
   is three new infra dependencies rather than a change to an existing
   one. The `visibility_timeout: 86400` point is correct and a real trap:
   Redis's 1-hour default redelivers a long-ETA task and the pipeline runs
   many times for one game. But per-game T-30 staggering is achievable
   with cron plus a dynamic dispatcher and no broker. Recommend Docker +
   PT-anchored cron first, and Celery only if per-game staggering proves
   necessary in practice.

3. **"Only flag high-probability plays."** `decision_board` already takes
   `min_ev`, `require_valid_book`, `min_leg_prob` and `consider_only`, and
   by the project's own rule nothing ranks by EV without VALID two-way
   odds and a no-vig fair probability. With boards EMPTY pre-season every
   row abstains, so the threshold currently gates nothing. That is the
   correct behaviour, not a misconfiguration — but it means the threshold
   is untested against real traffic and should not be read as verified.
