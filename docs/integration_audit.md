# End-to-end integration audit

Date: 2026-10-02. RESEARCH_ONLY. Scope: the **seams** — whether stage A hands
stage B what stage B accepts — not model quality.

Reproduce every finding below:

```bash
python -m scripts.verify_wiring            # 20 PASS · 5 FAIL · 5 WARN · 2 SKIP
python -m scripts.verify_wiring --json     # machine-readable
python -m pytest tests/test_verify_wiring.py -q
```

The verifier runs with **no database, no network, no trained artifact and no
credential**, on a synthetic panel shaped like `load_player_panel`'s output. It
therefore cannot tell you the model is any good. It tells you the pipe is
connected. `scripts/audit_pipeline.py` (folds, learning, metric reproduction)
and `scripts/audit_leakage.py` (splits) already cover the rest and need the real
parquet.

**Four of my own first-draft checks failed for the wrong reason** — a wrong
`PickemLeg` signature, the wrong sizing entry point, an EV-gate fixture missing
its posted line, and a recorder fixture missing the board join. Those were
harness bugs, not findings, and `tests/test_verify_wiring.py` pins the fixtures
so they cannot drift back.

---

## 1. Broken connections & disconnected modules

### 1.1 CLOSED (2026-10-04) — nothing produced rows for a slate that had not been played

`load_player_panel` selects `PlayerGameLog`, which is **completed box scores**.
`main._filter_to_slate` then keeps only rows whose `GAME_DATE` equals the slate
date. At 09:00 PT, before any tip, that intersection is empty, so the deployed
worker takes the `success_no_data` branch and exits 0 — every single day.
`main.py` says as much in its own log line:

> the player game log holds completed games, so a future slate will not appear
> here until those games are played and ingested

`scripts/nba_model_cli.py:predict_slate` has the same shape and reports
`NO_ROWS`. **No path in the repository builds a forward slate.**

Both halves of the fix are already in the tree, tested, and imported by nothing
in the production path:

| module | gives | imported by |
|---|---|---|
| `src/ingestion/espn_schedule.py` `load_slate()` | games, tip-offs, `pregame_only` | `scripts/nba_model_cli.py` only |
| `src/ingestion/espn_availability.py` `fetch_roster()` | players per ESPN team id | **nothing** |

**Fixed.** `src/pipeline/forward_slate.py` adds one row per (player, scheduled
game) from the schedule's **pre-tip** games plus each team's recent appearances
in the panel, with **every box-score column NaN** so the rolling features read
each player's own prior real games and the forward row has nothing of its own
to leak — measured by value in `tests/test_forward_slate.py`, which also runs
`assert_no_lookahead` over the result.

The lineup comes from the panel rather than `fetch_roster`, on purpose: a
roster fetch is keyed by ESPN athlete name and would need the crosswalk in §1.2
that still does not exist, so a name mismatch would silently drop a player.
"Who appeared for this team in its last few games" is a narrower claim than a
roster — it misses a player returning from a long absence, which is stated in
the module rather than hidden — but it is a true one, and the ESPN injury feed
already removes the ruled-out through `src/pipeline/scratches.py`.

Wired into `main.py` behind `PROPIQ_FORWARD_SLATE` (default on). A denied or
unreachable schedule leaves the panel untouched with a named reason, so the run
degrades to exactly what it did before rather than failing.

### 1.2a CLOSED (2026-10-04) — OddsPapi was in the source precedence with no client

`decision_board.SOURCE_PRECEDENCE` read `("propline", "oddspapi")`. OddsPapi
had **no client module, no key reader, no ingestion path**: a string in a tuple
and prose in a dozen docstrings. `BetLifecycleRecord.source` also *defaulted*
to `"oddspapi"` and no caller anywhere set it, so every paper ticket in the
ledger carried the provenance of a feed that does not exist.

Deleted. The precedence is `("propline",)`, the ledger's default source is
`None` (unknown, not a vendor name), and the walker is unchanged and still
generic — `resolve_market` and now `enrich_row_with_resolved_market` both take
a `precedence`, and the fallback path is tested against a second source the
test invents. `verify_wiring` section 3 gained a check that **every name in the
precedence has an ingestion client**, which is the generalised form of this
defect and of 1.2 below.

Measured while deleting it: the precedence **ranks, it does not allow-list** —
an unlisted source sorts last and is still priced. Removing the name removed a
false promise of a fallback; it did not add a filter, and none is needed while
nothing can fetch that source.

### 1.2 `src/ingestion/id_crosswalk.py` does not exist, and three modules name it as the fix

Every exact-name join in the project defers to a module that was never written:

- `main.py:590` (`assemble_projections` docstring) and `main.py:681` (the
  all-unmatched warning)
- `src/settlement/recorder.py:133`
- `src/pipeline/scratches.py:35`
- `src/settlement/runner.py:81,178` name it in prose without a path

### 1.3 Orphans

Only two modules in `src/` have no importer, and the scan is in the verifier so
it cannot rot:

| module | lines | note |
|---|---|---|
| `src/models/protocol.py` | 48 | a typing `Protocol` nothing declares conformance to; no test |
| `src/settlement/cli.py` | 134 | standalone entry point, run as `python -m`; not an error |

Dead public functions: **16 of 321**. Most are harmless second doors
(`attach_eligibility_warnings` beside the `eligibility_warnings_for_row` that
`compare.py` actually calls). Two matter:

- **`espn_availability.fetch_roster`** — see 1.1.
- **`models/artifact_registry.py`** `register_saved_artifact` / `read_registry` —
  an append-only index of where artifacts were written, which nothing writes and
  nothing reads. It is the missing half of 2.1.

---

## 2. Schema & interface mismatches

### 2.1 CLOSED (2026-10-04) — scoring read one path, training wrote another, and the worker passed neither

| | path |
|---|---|
| `main.MODEL_ARTIFACT_DEFAULT` (what `score_prob_over` loads) | `models/xgb_prop_over.json` — **the `models/` directory does not exist** |
| `config/model_comparison.yaml:artifacts_dir` (where `train-stats` writes) | `data/external/model_runs/comparison/xgboost_{MARKET}.json` |

`main.py` bridged them only via `--model`, and `scheduler_worker.run_slate`
calls `main.main(argv or [])` — **no arguments** — so a scheduled run scored
nothing *even after a model was trained into the right place*.

**Fixed.** `main.resolve_model_artifact` tries `--model`, then `PROPIQ_MODEL`,
then the newest `xgboost_*.json` in `artifacts_dir` **that has its
`.meta.json` sidecar** (one without it cannot be scored with anyway), then the
legacy path — and returns a reason naming every path it tried rather than a
missing-file message for a path nobody chose. `score_prob_over` no longer
defaults to a path nothing writes.

The worker needed **no argv plumbing**: it calls `main.main([])` in-process, so
`PROPIQ_MODEL` set on the service reaches the resolver through the shared
environment. `artifact_registry` (1.3) is deliberately **not** read — nothing
writes to it, and resolving through a dead module is how `oddspapi` survived in
the source precedence for months.

### 2.2 `research_slate_from_predictions` drops the game date and stamps today's

`compare.py:839` emits `"game_date": game_date_pt` on every detail row.
`paper_research.research_slate_from_predictions` **never reads it**: `slate_date`
is a parameter stamped onto every output row (line 387), and `ResearchSlateRow`
has no game-date field at all.

Combined with 2.3 this is the worst finding in the audit. `run_board`'s defaults
score the window `(2025-01-15, 2025-02-15]`, so **every board row is a
February-2025 prediction carrying today's date**, and nothing downstream can
tell. The Discord card takes its title from `frame["slate_date"].iloc[0]`.

### 2.3 The board's training window is pinned to two 2025 dates

`scheduler_worker.run_board` falls back to `train_end="2025-01-15"` and
`validation_end="2025-02-15"` when `PROPIQ_BOARD_TRAIN_END` / `_VALIDATION_END`
are unset. They never advance. The window recedes further into the past every
day and nothing reports it. It also means `run_board` **re-fits every model on
every slate job**.

### 2.4 A slate-level EV verdict is written as a per-row claim

`main.evaluate_ev_gate` loops every prop, calls `market_ev_gate` per row, then
returns **counts plus one status**. `assemble_projections` writes that single
status into `MARKET_STATUS` on every row, and `repository.py:331` persists it
per row as `market_status`. One ready prop out of three hundred labels all three
hundred `READY_FOR_EVALUATION`. The per-row verdicts are computed and discarded.

### 2.5 One artifact, one market

`score_prob_over` takes a single `model_path` and writes `PROB_OVER` only for
that artifact's `target_market` — which is **correct** and deliberate
(`assemble_projections` refuses to copy a points probability onto rebounds). The
consequence is structural: `REB`, `AST` and `FG3M` can never carry a probability
in one run, and the recorder skips a row with no probability, so a three-market
board is a one-market ledger.

---

## 3. Silent failure risks

Ranked by how much output is lost with no error raised.

| # | Path | What is lost | Signal today |
|---|---|---|---|
| 1 | forward slate (1.1) | **everything** | `success_no_data`, exit 0 |
| 2 | model path (2.1) | every probability | one WARNING line |
| 3 | board dates (2.2 + 2.3) | correctness, not volume — historical rows read as tonight's | none |
| 4 | prop-line name join | the whole settlement ledger | total failure warns; **partial does not** |
| 5 | `MARKET_STATUS` (2.4) | per-row truth | none |

On #4: `recorder._line_lookup` joins on exact `(player_name, market)` and
`source` comes **only** from the board. The verifier runs the real case — the
panel's `Nikola Jokic` against a board's `Nikola Jokić` — and the row is skipped
for *"no line source (a null source defeats the unique key)"*. A systematic
name-format difference therefore records **zero** gradeable rows while the run
reports success. `main._attach_prop_lines` warns only when *every* row is
unmatched; a half-matched board is silent. Market keys are fine — PropLine
normalises through `DEFAULT_MARKET_MAP` and logs unmapped keys.

### What is sound, checked rather than assumed

- **Database lifecycle.** `session_scope` has commit / rollback / `finally:
  close()`, `pool_pre_ping=True`, `pool_size=5`, and `expire_on_commit=False` —
  so the ORM rows `load_player_panel` reads after the session closes are not
  detached-expired. I went looking for that bug and it is not there.
- **Train/serve column contract.** `verify_feature_contract` enforces **order**,
  not just membership: the verifier swaps two columns in a real sidecar and the
  check rejects it.
- **Exception safety.** All four `run_*` jobs catch and log; the worker survives
  a failed slate. The broad handlers in `compare.py`, `main.py` and `discord.py`
  each return a **named reason** rather than swallowing. A scan for silent
  handlers returns ~95 hits, and almost all are narrow coercion handlers
  returning `None` — which is this project's zero-inference idiom, not a defect.
- **The publication gate.** No report → withheld. The real producer's output on
  400 synthetic calibrated rows → allowed (ECE 0.0476, n=400). Both sides agree
  on the report schema.
- **Dtypes.** IDs stay `str` (matching `String(32)`) and `GAME_DATE` stays
  `datetime64` through the builder.
- **Feature layers.** All nine registered layers add non-null columns on a
  synthetic panel; fatigue genuinely moves `{stat}_L2` off `{stat}_BASELINE`; a
  `SEASON`-less panel comes back `SEASON`-less (`docs/season_key.md`).
- **Advisory sizing.** `RECOMMENDED_UNITS=2.96` at quarter Kelly on f*=0.1185,
  `AUTO_PLACED=False`.
- **Pick'em contracts.** A two-leg entry against a `{2: 3.0}` structure returns
  `PAYOUT_EV_READY` — no hard block, no `EV undefined` exception.

---

## 4. Remediation, in dependency order

1. **Wire a forward slate** (1.1). `espn_schedule.load_slate().pregame_only` for
   the games, `espn_availability.fetch_roster` for the players, joined onto the
   panel's history so the rolling features still read real prior games. Until
   this lands, nothing else in this list changes the output.
2. **Make the model reachable** (2.1). Smallest version: a `PROPIQ_MODEL`
   variable read by `run_slate` and passed as `--model`. Better: have
   `score_prob_over` resolve the newest artifact per market through
   `artifact_registry`, which exists for this.
3. **Carry the game date onto the board row** (2.2). Add a `game_date` field to
   `ResearchSlateRow`, populate it from the detail row, and have
   `build_slate_board` refuse a row whose game date is not the slate date. That
   turns 2.3 from a silent staleness into a loud refusal.
4. **Roll the board window** (2.3), or derive it from the panel's max date.
5. **Keep the EV verdict per row** (2.4) — return the per-prop verdicts
   `evaluate_ev_gate` already computes, and key them on `(player, market)`.
6. **Write `id_crosswalk.py`** (1.2), or delete the five references and state the
   exact-match rule as final. A named fix that does not exist is worse than an
   acknowledged limitation.
7. Score every configured market (2.5) once 2 is done.

Items 1 and 2 are deployment blockers in the plain sense: without them a
deployed worker runs on schedule, logs cleanly, and produces nothing. Item 3 is
the one that could mislead a reader rather than merely disappoint one, which is
why it ranks above the rest.
