# Why zero plays reach Discord — early-season audit, 2026-11-01

**Verdict: the cause is not a filter, a sample-size guardrail, or model
conservatism. There is no model probability to filter.** None of the three
thresholds named in the brief is ever evaluated — but for **two different
reasons**, and collapsing them into one would misstate the chain:

| Threshold | Why it never runs |
|---|---|
| `min_ev` | gated on the probability: `p_side is None` on 460 of 460 rows, so control never reaches the comparison |
| `min_lean` | same — and its live value is **0.0**, not the 0.03 default, so there is nothing to relax even once it does run |
| `min_prior_games` | **nothing to do with the probability.** It is skipped because the live projections adapter emits no `eligibility_warnings` column at all, so the cold-start gate is not wired into this path |

The first two are starved. The third is disconnected. Relaxing any of them
changes nothing, and the third is not even a filter that is firing.

This document corrects three premises in the brief, because the remedies they
imply would not have worked and two of them would have weakened guardrails for
no gain.

---

## 1. The measured funnel

From today's slate job, run `run_20261101T090000-0800_c87fc7`
(2026-11-01 09:00 PST, Railway deployment `5cb02cf8`):

| Stage | Count | Source |
|---|---|---|
| Panel rows loaded | 2,106 | `Slate filter: 115 of 2106 panel rows fall on 2026-11-01` |
| — of those, today's slate | 115 | same line |
| Projection rows assembled | 460 | `Assembled 460 projection rows across 4 markets` |
| — with a matched prop line | 226 | same line |
| — **with a probability** | **0** | same line (`0 with a probability from no market`) |
| — cleared the EV gate (`contracts.py`) | **224** | same line (`224 READY_FOR_EVALUATION per row`) |
| Withheld by the scratch filter | 40 (10 players OUT/DOUBTFUL) | `src.pipeline.scratches` |
| **RECOMMENDED rows** | **0** | — |

Two numbers decide the whole question.

**`0 with a probability`.** `main.score_prob_over_by_market` resolves one
artifact per market and, per its own contract at `main.py:1127`, *"A market with
no artifact is SKIPPED, not filled. Its rows keep a null probability."* No
artifact resolves for any market — the boot probe has said so on every deploy
since 2026-10-31:

> `NO SCORING ARTIFACT RESOLVED … Every row will ABSTAIN: the slate job will
> ingest, build features and write projections with NO probability, exit 0 and
> look healthy.`

**`224 READY_FOR_EVALUATION`.** `contracts.market_ev_gate` passed **224 of the
226 rows that had a price — 99.1%**. The de-vig ran. The gate the brief suspects
of wiping out edge is the one part of the chain demonstrably not blocking
anything.

### Where the rows actually die

`src/quant/decision_board.py:442-503`. The classifier tries three paths in
order, and **every one of them requires `p_side is not None`**:

```python
book_ok = row.book_status == "VALID" and odds is not None and ev is not None   # :442

if book_ok and p_side is not None:        # :444  → book_ev,  tests min_ev
elif require_valid_book:                  # :459
elif p_side is not None:                  # :466  → model_lean, tests min_lean
elif row.pickem_line is not None:         # :493
else:
    why = f"DATA_NOT_AVAILABLE: cannot resolve P({side})"   # :503  ← all 460 rows
```

`min_ev` is read at line 448, `min_lean` at line 471. With `p_side` null,
control reaches line 503 and **neither comparison executes**. This is the
bottleneck, and it is a missing input, not a strict test.

---

## 2. Three premise corrections

### (a) `contracts.py` holds no EV or Kelly threshold at all

The brief asks whether `EV% > 0.03` or `fractional Kelly > 0.02` is too strict
for early-season markets. Neither exists in that file. `market_ev_gate` is a
**permission** gate — it answers *may this row be priced as a two-way market*
and returns `"ev": None` by construction, documented in its own docstring: *"the
gate grants permission, it does not price anything."*

The real thresholds, and their actual values:

| Threshold | Tested at | Default | Overridable? | Effect at the default |
|---|---|---|---|---|
| `min_ev` | `decision_board.py:448` | **0.0** (`slate_board.py:248`) | **yes** — `PROPIQ_MIN_EV`, parsed at `scheduler_worker.py:1482` | any EV strictly above zero recommends |
| `min_lean` | `decision_board.py:471` | **0.0** (`slate_board.py:249`) | no — `run_board` never passes it | any lean above zero recommends — **P(side) > 0.500**, over or under |

Two corrections to the first version of this table, both from review on #32.

**`min_lean` on the live path is 0.0, not 0.03.** `DEFAULT_MIN_LEAN = 0.03` at
`decision_board.py:124` is that module's own default, reached only by a caller
that does not override it. `build_board_from_projections` declares
`min_lean: float = 0.0` (`slate_board.py:249`) and `run_board` passes only
`min_ev` (`scheduler_worker.py:1543-1548`), so the scheduled board takes the
0.0. Nothing in the environment can raise it; closing it needs a code change.

**`min_ev`'s 0.0 is a DEFAULT, not a measured live value**, and the first
version of this table called it live. It is read from the environment:

```python
min_ev = float(os.environ.get(ENV_MIN_EV) or 0.0)   # scheduler_worker.py:1482
```

So `PROPIQ_MIN_EV` can raise it, and **I have not read the deployed
environment to find out whether it is set** — Railway's variable listing is
all-or-nothing and would print the database password and the PropLine API key
into a transcript. Worth confirming before anyone acts on the §4 numbers.

It does not change this document's verdict, and here is why it cannot: an
override can only move the floor **upward**, and `min_ev` is never evaluated at
all (§1). A stricter floor cannot explain zero plays when the comparison does
not run. What it would change is §4 Step 3's starting point — if
`PROPIQ_MIN_EV` is already 0.02, that row is satisfied and only `min_lean`
needs closing.

So the live `min_lean` is zero and the live `min_ev` is zero-or-higher. The
moment an artifact lands, this board recommends on an edge of +0.0001 if the
override is unset, and on a model lean of half a basis point **with no price
behind it at all** regardless, because that path's floor is zero and not
configurable. That is the opposite problem to the brief's, it is worse, and §4
Step 3 addresses it in the other direction.

### (b) The shrinkage has no prior-season prior to over-weight

The brief asks whether prior-season baselines are weighted so heavily that
predicted means collapse onto market lines. They are not weighted at all,
because they are not there.

`src/features/halflife.py:67,81`:

```python
w = n_prior / (n_prior + shrink_k)                    # k = 8, n = games THIS season
out[f"{stat}_HL_SHRINK"] = w * hl + (1.0 - w) * season
```

The shrink target is `{stat}_SEASON` — the **current** season's expanding mean
(`builder.py:542`, grouped `["PLAYER_ID", "SEASON"]` at `:524`), not a prior-season
baseline. `n_prior` is also within-season (`halflife._prior_game_counts` groups
on `player_season_keys`). So at n = 3 games, w = 3/11 = 0.27 and the estimator
puts 73% of its weight on **the simple mean of the same three games it is
shrinking**. It borrows no strength from anywhere. Early-season output is
therefore *noisier* than mid-season, not tighter — the reverse of the premise.

Note the one place history *would* flow in if it existed: `{stat}_L5` and
`{stat}_L10` group on `PLAYER_ID` alone (`builder.py:538-540`), so they roll
**across** the season boundary. A player's first game of 2026-27 would carry an
L5 built from last season's final five — if the panel held them. It does not;
see §3.

### (c) `builder.py` drops nobody, and the cold-start gate is not in the live path

On ingestion gaps: the rolling means use `min_periods=1` (`builder.py:299`), and
the baseline degrades rather than dropping the row (`builder.py:711`):

```python
df[f"{stat}_BASELINE"] = blended.fillna(df[f"{stat}_L5"]).fillna(df[f"{stat}_L10"])
# "Early-season rows have no season mean yet; fall back to what exists
#  rather than dropping the row or inventing a value."
```

`builder.py` filtered out zero players. The 460 rows are the proof.

On the sample-size guardrail: `src/models/eligibility.py:33` does set
`min_prior_games: int = 10`, and with a panel mean of ~3.8 games per player
(441 of 2,106 rows are a player's first, per the foul layer's log) that gate
*would* abstain essentially the entire pool. **But it is not reached.**
`eligibility_warnings` is attached only by `compare_models_on_panel`
(`compare.py:935`), which is the **backtest** path. The live board introduced in
#32 reads persisted projections through
`slate_board.projection_detail_rows` (`slate_board.py:165`), which emits no
`eligibility_warnings` column, and `decision_board.py:523` therefore iterates an
empty list on every live row.

So the cold-start guardrail is not choking off action. It is not firing at all —
which is a gap I introduced in #32, in the *dangerous* direction: once a model
lands, the live board will recommend thin-history rows with nothing withholding
them. See §4, patch B.

---

## 3. The real chain, and why "relax the floors" is not the fix

The blocker is the missing artifact. The reason there is no artifact is the
reason the brief is really about:

**The panel holds 2,106 rows — about 6.5% of one NBA season, roughly 11 days.**
A regular season is ~1,230 games × ~26 players ≈ 32,000 player-game rows. There
is no prior-season history in the panel at all.

That single fact explains everything downstream:

1. ~2,100 rows cannot support a leakage-safe fixed-cutoff train/validation split
   per market with a trustworthy ECE.
2. `{stat}_L5` / `{stat}_L10` have no prior season to roll back into, so every
   early-season window is thin even though the code is willing to use history.
3. `eligibility`'s 10-game floor is unsatisfiable, because the panel does not
   contain ten games for almost anyone.
4. With 0 probabilities, `settlement/recorder` skips every row, `prop_results`
   stays empty, calibration never builds, and the publication gate (100 graded
   predictions, ECE ≤ 0.05) cannot open.

**The history already has a loader, deployed and switched off.** The ESPN
history backfill is registered on every boot and names its own state:

> `scheduled {'id': 'espn_history_backfill', 'name': 'Extend the panel backwards
> from ESPN (off unless configured)', 'trigger': "cron[hour='4', minute='30']"}`

It is dormant because `PROPIQ_ESPN_HISTORY_SEASONS` is unset. Setting it is the
unblock, and it **weakens no guardrail**: it supplies the sample the guardrails
are asking for instead of lowering what they ask.

Lowering `min_prior_games` from 10 to 2 would do the opposite — it would keep the
sample thin and remove the protection, and it would do so most aggressively for
genuine rookies, which is the one population where that floor is doing real work
and where no backfill can help.

---

## 4. Calibration plan

Ordered by dependency. Steps 1–2 are prerequisites; nothing after them can be
measured until they are done.

### Step 1 — supply the history (no code change)

Set `PROPIQ_ESPN_HISTORY_SEASONS` to the prior seasons to pull. The backfill
chunks at 60 days/night from 04:30 PT with a 60-minute lead on the slate job, is
event-level resumable, and re-examines nights it already holds. Two prior
seasons takes the panel from ~2,100 rows to ~60–65k.

**Then re-check `0 market-line rows`.** Today's run logged *"PARTIAL MARKET DATA
from the database: 176 team-game row(s), 0 market-line row(s)"* and no
BigDataBall workbook. That matters for step 2 specifically: a model trained
while the market-context layer is present will carry those columns in its
contract, and scoring **abstains** if a later run cannot supply them. Train on
the feature set production will actually have, or the artifact reproduces this
same zero-play outcome through a different door.

### Step 2 — train one artifact per market, then measure

One file per market by design. Until the sidecar resolves, every number below is
unmeasurable.

### Step 3 — the thresholds, with the direction reversed

Once probabilities exist, the early-season risk is **too many plays, not too
few**, because `min_ev = 0.0` recommends on an edge of one hundredth of a basis
point from a model with no measured calibration. Suggested bootstrap values, to
be revisited against observed ECE rather than assumed:

| Parameter | Now | Early-season bootstrap | Why |
|---|---|---|---|
| `min_ev` | 0.0 **by default**; `PROPIQ_MIN_EV` may already raise it — check before changing | **0.02** | an uncalibrated model's +0.5% edge is indistinguishable from noise; 2% survives a plausible calibration error. Settable from the environment, no code change |
| `min_lean` | **0.0** on the live path, with no override | **0.05**, or disable the `model_lean` path entirely — needs a code change, `run_board` does not pass it | a lean recommendation has *no price behind it*, and at 0.0 any P(side) > 0.500 qualifies — over OR under; with calibration unmeasured that is an opinion, not an edge. This is the most permissive gate in the pipeline and the one to close first |
| `min_prior_games` | 10 | **keep 10** | satisfy it with step 1, do not lower it |
| `shrink_k` | 8 | **8**, and point the target at a prior-season mean | see below |
| Kelly multiplier | n/a — advisory only | leave advisory | `advisory_sizing` stamps `AUTO_PLACED: False`; nothing sizes or places |

### Step 4 — the shrinkage change that would actually help

The useful early-season change is not a different `k`. It is giving the
estimator something to shrink *toward*: a prior-season per-player mean, so
`HL_SHRINK` at n = 2 leans on last year's 70 games instead of this year's two.
That is a real modelling change, it needs step 1 to have any data, and it must
be validated leakage-safe before it is believed. I have not written it, because
on today's panel it has nothing to read and no way to be tested.

### What cannot be answered yet

The brief asks whether probabilities are clustering in 48–52%. **There are zero
probabilities on record**, so there is no distribution to inspect. Any number I
gave for that spread would be invented. It becomes measurable one day after
step 2.

---

## 5. Patches

**Patch A — the "bootstrap mode" as specified: not written, because it is a
no-op.** Relaxing `min_ev`, `min_lean` or `min_prior_games` cannot produce a
play tonight, and for the two distinct reasons set out at the top: the first two
are never *reached*, because they are gated on a probability that is null on
every row — and both are already **0.0** on the live path, so there is no slack
in them to take up. The third is not relaxable at all here, because it is not
*connected* to this path (§2c). Shipping it would look like the fix
and change nothing — the same failure as the withdrawn forward-slate patch on
#32. The only way such a patch could emit plays is by fabricating a probability
where the model produced none, which is both forbidden and would put
recommendations on the board with nothing behind them.

**Patch B — restore cold-start withholding on the live board.** Real, small,
and mine to fix: #32's live path dropped `eligibility_warnings`, so once an
artifact lands the board will recommend rows the model had almost nothing to
learn from. The abstention is specific to *recommending* — the prediction is
still recorded and graded, because thin-history rows are exactly where
calibration needs evidence.

**Patch C — one definition of "prior games".** `builder.py:522` already computes
`CAREER_GAMES_PRIOR = by_player.cumcount()`, and `eligibility.prior_game_counts`
recomputes the same quantity independently at `eligibility.py:23`, sorting by
`(player, date)` first. On a panel not already in date order the two disagree,
and the gate then reads a different number from the one the features were built
against.

Patches B and C are not written into this branch: #32 is under review and
adding them would widen it.
