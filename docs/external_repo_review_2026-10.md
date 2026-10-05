# Review: 17 external repositories + the local juice-reel pack

Date: 2026-10-05. RESEARCH_ONLY. Comparative analysis only — **nothing was
vendored**, and no dependency was added. Every repository was cloned shallow
and read; where a claim is made about behaviour it was **executed**, because
two of the headline findings below are docstrings that contradict their own
code.

Scope rule observed: `BangLetsGetIt/NBA-NCAA-Betting-Models` is majority NCAA,
which this project excludes. It was read to answer the question; nothing from
its NCAA tree is portable here regardless of quality.

---

## Verdicts at a glance

| Target | Python LOC | Rating |
|---|---|---|
| **`PropIQ_JuiceReel_Local` (the uploaded pack)** | 16,016 **new** to us | **HIGH — by a wide margin** |
| `EdgarParra565/player-performance-forecaster` | 58,876 | **Marginal, with one High item** |
| `mattdspecht/NBA-Contextual-Modeling` | 3,076 | **Marginal** |
| `arjun2garg/NBA-Correlation` | 6,822 | **Marginal** |
| `roni-altshuler/nba_predictor` | 15,691 | Zero / redundant |
| `joshdspotify96-jpg/plus-ev-model` | 11,479 | **Zero — and mathematically broken** |
| `BangLetsGetIt/NBA-NCAA-Betting-Models` | 72,067 | Zero (out of scope + subpar) |
| `spahnrl/prj_BookieX` | 49,209 | Zero for our purposes |
| `DanielTomaro13/Basketball-Modelling` | 4,024 | Zero |
| `aadi-shah26/nba-prop-bet` | 3,691 | Zero |
| `Andresperez397/nba-shot-making` | 1,250 | Zero |
| `kpundhir/Prop-Model` (listed twice) | 168 | Zero |
| `api-evangelist/wager-api` | 0 | Zero (OpenAPI specs, no code) |
| `Khavel/proplab-mcp` | 0 (TypeScript) | Zero |
| `danielabboudi/NBA`, `atr777/nba-predictions`, `paulkellar2023/nbai-bets`, `Risky-Scout/nba-player-props-model` | 0 | Zero (static sites / empty) |

**Security:** a credential scan across all 17 (`api_key|secret|token|password`
assigned a 16+ character literal, excluding placeholders) returned **nothing**.
No hardcoded keys found anywhere.

---

## 1. The uploaded pack — HIGH VALUE, and the one that matters

66 modules / **16,016 lines** that our tree does not have. This is not an
external repository to mine; it is the other half of this project. Four items
are worth porting before anything else on this page.

### 1.1 `src/ingestion/id_crosswalk.py` (96 lines) — port first

**It is the module three of our own files name and we do not have.**
`main.py:590`, `main.py:681`, `src/settlement/recorder.py:133` and
`src/pipeline/scratches.py:35` all tell the reader to route name variance
"through `ingestion/id_crosswalk.py`" — a path that does not exist. It is open
item **O3** on `docs/go_live_readiness.md`, and it is the reason
`docs/integration_audit.md` §3 ranks a silent name mismatch as the fourth-worst
silent failure in the pipeline: one diacritic (`Nikola Jokic` vs `Nikola Jokić`)
records **zero** gradeable rows while the run reports success.

What it is: `rapidfuzz.process.extractOne` with `token_sort_ratio` and
`score_cutoff=85`, returning a Pydantic `CrosswalkMatch` whose miss path is
`status="DATA_NOT_AVAILABLE"` with a reason. That is already our abstention
idiom, and **`rapidfuzz>=3.6.0` is already in our `requirements.txt`** — zero
new dependency.

**One hardening required before porting.** `PlayerNameRecord` carries a `team`
field and `match_one` never reads it. A names-only fuzzy match will pair
`Jalen`/`Jaylen` and same-surname relatives, and in a settlement ledger a near
miss is a wrong record, not a rounding error. Require a team match, or use team
as a tiebreak and abstain on an ambiguous pair.

### 1.2 `src/features/fatigue_fit.py` (87 lines) — closes our own DATA_GAPS #16

Fits **empirical** B2B / 3-in-4 / altitude multipliers as mean residual ratios
against baseline, and abstains when there are too few rows. Our multipliers
(0.97 / 0.96 / 0.94 and the altitude tax) are **unfitted heuristics folded into
every `{stat}_L2`** — our own `docs/DATA_GAPS.md` lists that as a gap. This
replaces a guess with a measurement, in 87 lines, with the refusal path already
written.

### 1.3 `src/features/matchup_overlay.py` (790 lines) — defensive matchup interaction

Opponent defense-vs-position: L10 rolling mean of stats allowed by the opponent
to a position bucket, with `shift(1)` hygiene stated in the docstring and a
named provenance. Our `src/features/defense.py` carries opponent *defensive
rating* and allowed rates; **position-bucketed** DvP is a different and finer
feature. Verify against `defense.py` before porting — I did not confirm every
column is new — but the position bucketing is not something our tree does.

### 1.4 `docs/validation_*.md` (20 reports) + `src/models/validation.py` (390 lines)

**The most useful thing in the pack, and it is not code.** Out-of-sample
validation per market per season, with reliability tables. Read
`validation_2024-25_PTS.md` and the honest conclusion is unavoidable:

```
P(Over) — XGBoost:  N=12,765  accuracy=0.5419  log_loss=0.6915  brier=0.249
  bin 0.7-0.8 :  153 rows, mean pred 0.7344, mean outcome 0.5425
  bin 0.8-0.9 :   16 rows, mean pred 0.8263, mean outcome 0.6250
```

A constant 0.5 predictor scores Brier 0.25 and log-loss 0.6931. This model is
**0.001 better than a coin flip**, and it is *confidently wrong exactly where a
board would surface a row*: predictions near 0.73 realise at 0.54.

That is evidence about **our** approach, not only theirs — it is the same
feature family against the same self-referential `RESEARCH_LINE` target that
our `docs/go_live_readiness.md` **O8** flags. It is the strongest external
confirmation we have that the publication gate should keep withholding, and it
should be read before anyone promotes a feature on intuition.

### 1.5 The rest, briefly

- `blowout_features.py` + `_v2.py` (689) — our `blowout` layer is off by
  default and measured redundant; compare before adopting.
- `win_loss_tracker.py` (646) — we built the results *embed* this week (O9);
  this is a fuller tracker behind it.
- `line_probs.py` (243) — line-aware probabilities, our **O8** target.
- `six_factor_rapm.py` (197), `bi_asof.py` / `bi_interactions.py` (406),
  `transformers.py` (407), `extended_box.py`, `player_game_market.py`.
- ~20 ingestion clients (`nba_api_dvp`, `nba_stats_advanced`, `pbp_boxscores`
  at 869 lines, `underdog_props`, `prizepicks_props`, `sleeper_props`,
  `nba_com_boxscore`, `nba_official_archive`).
- `src/ops/*` — the Celery/ETA stack already reviewed in
  `docs/go_live_pack_review.md` §3; that assessment stands.

---

## 2. `player-performance-forecaster` — Marginal, one High item

### 2.1 HIGH — `nba_model/model/correlation_calibration.py` (118 lines)

**This closes a real failure in our parlay path.** It supplies what we lack:

- `_nearest_psd` — eigenvalue clipping (`eigh`, clip to `eps`, reconstruct);
- `_to_psd_correlation` — PSD projection, rescale to unit diagonal, absolute cap;
- **shrinkage toward the identity** (`(1-w)·corr + w·I`, default w=0.15);
- return the identity outright when the sample is under `min_games` — abstain
  to independence rather than trust a noisy empirical correlation.

We have the sample gates (`min_pairs`/`min_games`, and ours is better — it
distinguishes pairs from games), off-diagonal range validation, and
`estimate_tetrachoric_correlation`, which is the right estimator for binary
outcomes. We have **no PSD repair and no shrinkage**, and
`src/quant/parlay.py:297` raises `ParlayError` on a Cholesky failure with the
message *"Check the pairwise values against each other rather than adjusting
one in isolation"* — an instruction for a human to do by hand what
`_nearest_psd` does in ten lines.

**Executed, not asserted.** An entirely ordinary pairwise matrix — A correlates
+0.75 with both B and C, while B and C correlate −0.40 — is not a joint
distribution:

```
eigenvalues            [-0.2794  1.4  1.8794]     cholesky: FAILS
after _to_psd_correlation: [0.0  1.293  1.707]    cholesky: OK
max |change| per entry: 0.1554
```

Today that ticket cannot be priced at all.

**Port it our way, not theirs.** A silent repair is against house style, and
0.1554 is a large move. Port `_nearest_psd` + shrinkage, then **return how far
the matrix moved** alongside the result, and abstain above a threshold instead
of repairing quietly. Raise `eps` above `1e-8`: their clip leaves the minimum
eigenvalue at exactly 0, and `cholesky` wants positive *definite*.

### 2.2 ZERO — `nba_model/model/minutes_projection.py` (25 lines)

The brief asks whether anything here has "a dedicated, robust minutes model
that accounts for blowouts, foul trouble, or rotation shifts." This repository's
is **one `if` statement**: a flat 12% haircut when `|spread| >= 10`. Our
`src/models/minutes_model.py` (350 lines, q0.1/q0.5/q0.9 quantile models) is
already more. Ignore.

### 2.3 Code smells

`nba_model/web/subscriptions.py` and a Streamlit UI — a product surface we have
no use for ("this is not a Streamlit app"). `fillna(0.0)` over percentage
columns is the soft-zero anti-pattern we removed from `compare`.

---

## 3. `NBA-Contextual-Modeling` — Marginal

### Worth referencing

- **`player_roll10_pf`** — rolling personal fouls as foul-trouble/minutes risk.
  We have **no PF feature at all**. Cheap, plausibly not redundant with minutes.
  The single cleanest idea in the three mid-sized repos.
- **LightGBM native quantile regression** (`objective="quantile",
  alpha=0.10/0.90`) for a prediction interval. We derive over/under/push from a
  *parametric* dispersion. An empirical q10/q90 band is distribution-free — and
  it is a candidate answer to the case our new `prob_push` column deliberately
  leaves NULL: a whole line where a binary classifier has no push mass to split
  out (**O8 is closed, but by refusal, not by measurement**).
- Explicit interactions: `expected_pts_prior = pts_per_min × ema5_minutes`,
  `usg_x_opp_drtg`, `pts_x_team_scoring`, justified as "trees sometimes miss
  interactions at shallow depth." The minutes×efficiency decomposition is the
  standard prop prior and we do not carry it explicitly.

### Redundant

Its travel/altitude is **worse than ours**. Altitude is a two-arena boolean
(`{"ball arena", "vivint arena"}` — and "Vivint" is Utah's former name, so the
table is already stale). We have `ARENA_COORDS` **plus `ARENA_HISTORY`**, a
date-aware relocation table that exists precisely to avoid that staleness, plus
`haversine_miles`.

Their `days_rest > 10 → zero travel, cap rest at 10` is a **cruder fix for a
problem we solved properly**: we partition rest by `SEASON`, so the offseason
gap never appears, and a real 12-day in-season layoff keeps its information
instead of being flattened to 10.

### Code smells

`opp_roll10_pts_scored` is computed by a broken merge and then overwritten by a
`# Simpler: recompute` block — dead code left in place. `is_playoffs` is derived
from `game_month >= 4`, which mislabels April regular-season games.

---

## 4. `NBA-Correlation` — Marginal

- **`compute_rolling_covariances`** — rolling 20-game within-player correlation
  between PTS/AST and PTS/REB **as a feature**. We do not have this, and for
  PRA-style same-game legs it is the right shape. But `.fillna(0.0)` makes an
  unknown correlation read as "no correlation" — the invented value this project
  refuses. Port the idea, keep the NaN.
- `scripts/experiments/analyze_stat_phi.py` — **redundant and weaker than
  ours.** It reports mean/std φ and the share of pairs over |φ| > 0.10/0.15,
  treating pairs within a game as independent observations. Our
  `scripts/leg_correlation_dependence_check.py` measures
  `P(both over) − P(a)·P(b)` **with game-clustered uncertainty**, which is the
  statistic that survives the clustering their method ignores; and φ understates
  the latent correlation that `estimate_tetrachoric_correlation` targets.
- Worth stealing cheaply: their `validate()` asserts data-quality invariants
  (`usage_share ∈ [0,1]`, `pace ∈ [85,130]`). Note `days_rest ∈ [0,7]` would
  fail on a real All-Star break.

---

## 5. `nba_predictor` — Zero, and a lesson about reading code

Its `devig_shin` docstring says: *"For two outcomes Shin has a closed form, so
there is no root-finding and no convergence to babysit."* **The function
immediately below it bisects for 80 iterations.** The docstring is false.

Executed against ours across the full price range:

```
    odds        theirs               ours            max|diff|
-110/-110  (0.50000,0.50000)  (0.50000,0.50000)     1.9e-13
-450/340   (0.79545,0.20455)  (0.79545,0.20455)     5.6e-13
-8000/2500 (0.97460,0.02540)  (0.97460,0.02540)     1.4e-13
```

Same formula, same answer, and **ours is better**: our bracket is
`[0, 1-1e-12]` against their `[0, 0.9]`, which can fail to bracket on a large
overround and then silently return `z=0` (proportional de-vig) without saying
so — exactly the silent-fallback our `devig_two_way` refuses to do.

One idea worth taking: `MIN_BOOKSUM = 0.90` / `MAX_BOOKSUM = 1.30` with the
message *"these two legs are probably not the same game."* A booksum sanity
range is a cheap guard against a mispaired two-way quote. Check it against the
equivalent message already at `src/quant/devig_methods.py:232` first.

---

## 6. `plus-ev-model` — Zero, and actively wrong

`src/core/devig.py` looks like the most on-topic file in the batch. It is
broken twice over, and both bugs were confirmed by running it:

**`american_to_decimal` has a sign error.** For negative odds it returns
`-100/abs(odds) + 1`, so −110 becomes **0.0909** instead of 1.909. Downstream,
`decimal_to_probability` guards `< 0` but 0.0909 is positive, so it slips
through and returns an "implied probability" of **11.0011**.

**`balanced_devig` is degenerate.** It assigns the *same* average odds to both
sides (`avg_odds1 = avg_odds2 = (odds1+odds2)/2`), so after normalisation it
returns `(0.5, 0.5)` for every input:

```
 -110/-110  raw=(11.0011,11.0011) -> (0.5, 0.5)
 -450/+340  raw=( 1.2857, 0.2273) -> (0.5, 0.5)
 -200/+170  raw=( 2.0000, 0.3704) -> (0.5, 0.5)
```

A de-vig that discards the market and returns a coin flip, built on a
probability above 1. **Do not reference this file.** The only salvageable
concept is `calculate_true_probability`'s **multi-book weighted average** across
sharp books — we price from a single source by precedence — and that is an idea,
not code to take.

---

## 7. The rest

**`NBA-NCAA-Betting-Models`** — 72k lines, and the keyword matrix lied about it:
60 files matching "tracking" are **bet-pick tracking JSON**, not player tracking
data. Its rate limiting is bare `time.sleep(4.5)` / `sleep(90)` with no
`Retry-After` and no exponential backoff — far below our PropLine client
(`max_attempts=4`, `backoff_seconds=2.0`, `Retry-After` honoured, quota floor
at `min_daily_remaining=5`). Also majority NCAA (out of scope) and contains an
`auto_bet_helper`, a concept our architecture forbids outright. **Zero.**

**`prj_BookieX`** — 49k Python lines but **5.3 GB of committed data** (deleted
locally before it exhausted this container's disk). Nothing matched any audit
criterion: no de-vig, no correlation, no minutes model, no calibration, no
scheduling. Zero for our purposes.

**`Basketball-Modelling`, `nba-prop-bet`, `nba-shot-making`, `Prop-Model`** —
1k–4k lines of standard box-score rollups and a sklearn fit. `nba-prop-bet`'s
`odds.py` is ordinary proportional de-vig. `Prop-Model` is a single 168-line
script. Nothing we lack. Zero.

**`wager-api`** — not code: 69 YAML OpenAPI specs describing sportsbook APIs. It
has no implementation to extract, and the 62 "rate limit" hits are *spec prose*
about providers' limits. If a provider were ever approved it would be a useful
reference for request shapes; it is not a module.

**`proplab-mcp`** — a TypeScript MCP server, no Python. Out of band for this
pipeline.

**`danielabboudi/NBA`, `atr777/nba-predictions`, `paulkellar2023/nbai-bets`,
`Risky-Scout/nba-player-props-model`** — static marketing sites (`index.html`,
PNGs, one `.mp4`) or empty. **`Risky-Scout/nba-player-props-model` cloned with
zero files.** No code at all.

---

## 8. Integration recommendation — extract vs ignore

**Extract, in this order:**

1. **`id_crosswalk.py`** from the pack → `src/ingestion/id_crosswalk.py`.
   Add the team-match hardening, a test that a diacritic matches and a
   `Jalen`/`Jaylen` pair abstains, then wire it into the three call sites that
   already name it (`main._attach_prop_lines`, `settlement/recorder._line_lookup`,
   `pipeline/scratches`). Closes **O3** and the fourth-ranked silent failure in
   `docs/integration_audit.md`. **No new dependency.**
2. **`_nearest_psd` + identity shrinkage** from `correlation_calibration.py` →
   `src/quant/leg_correlation.py`. Report the repair magnitude and abstain above
   a threshold rather than repairing silently; raise `eps` above `1e-8`. Turns a
   `ParlayError` into a priced ticket with a recorded adjustment.
3. **`fatigue_fit.py`** from the pack → replaces unfitted multipliers with
   fitted ones. Keep the A/B discipline: measure Brier before promoting.
4. **`player_roll10_pf`** (rolling personal fouls) → a two-line feature in the
   builder, then `scripts/feature_ab.py --layer` to decide. Cheap to test,
   cheap to reject.
5. **`matchup_overlay.py`** (position-bucketed DvP) → verify against
   `src/features/defense.py` first; port only the columns that are genuinely new.

**Read, do not port:** the pack's `docs/validation_*.md`. Twenty measured
reports showing accuracy 0.5419 and Brier 0.249 against a self-referential
line, with the top reliability bins inverted. It is the best argument on this
page for leaving the publication gate shut.

**Ignore outright:** `plus-ev-model` (broken arithmetic), `nba_predictor`
(redundant, with a false docstring), `NBA-NCAA-Betting-Models` (out of scope,
subpar ingestion, auto-bet), `prj_BookieX`, and the ten repositories with no
usable code.

**Zero external dependencies were added, and nothing was vendored.** Every
item above is a native reimplementation with its source named here.
