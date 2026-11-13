# External quant & feature audit — 18 repositories

**Date:** 2026-11-13 · **Scope:** NBA only · **Status:** RESEARCH_ONLY

Nothing in this document prices, sizes or places a wager. Where an external
repository reports its own backtest figures, those figures are attributed to it
and are **not** reproduced as findings of this project.

Citations here name a **file and function**, never a line number. The existing
citation anchors in `docs/` are pinned to line numbers and have had to be
re-pinned six times in the last three days; this document does not add to that.

---

## Summary, stated before the detail

The honest headline is that **12 of the 18 repositories contributed nothing**,
and that most of what this audit was commissioned to look for — de-vigging,
Kelly sizing, parlay pricing, copulas, composite props, DFS scoring, cold-start
shrinkage — **already exists here, in a stronger form than any of the 18 has.**
Specifically:

| Pillar asked about | What the externals have | What `src/quant/` already has |
|---|---|---|
| De-vigging | Multiplicative only (`p_i / Σp`) | 4 methods — multiplicative, **Shin**, odds-ratio, logarithmic — plus `method_spread` to measure disagreement between them |
| Kelly sizing | Fractional Kelly, one hardcoded fraction | `advisory_sizing` — fractional Kelly, hard cap, negative-fraction guard, and an explicit no-execution boundary |
| Parlay / outcome distributions | `math.prod(probs)`, independence assumed | `parlay.hit_count_distribution` / `copula_joint_probability` — exact **Poisson-binomial** when no correlation is supplied, Gaussian copula when one is; **22** payout structures in `config/dfs_payouts.yaml` (Underdog power 2–8 / flex 3–8, PrizePicks power 2–6 / flex 3–6), i.e. Power **and Flex partial-win tiers**, covering the 2–5 leg range in full |
| Correlation / dependency | One hardcoded ρ per role category | `leg_correlation` — **empirical tetrachoric** ρ per (relationship × market-pair) across same-player / same-team / opposing-team / different-game, dual-gated on 200 pairs **and** 50 distinct games, with the over/under sign correction `ρ·s_a·s_b` |
| PSD repair | Shrink off-diagonals until Cholesky succeeds | `psd_repair.nearest_correlation` (Higham) |
| Composite props | — | `composite_markets` — PA/PR/RA/PRA/FPTS with `composite_mean` and `composite_variance` under a correlation matrix, reading the DFS matrix `1.0·PTS + 1.2·REB + 1.5·AST + 3.0·STL + 3.0·BLK − 1.0·TOV` from `config/fantasy_scoring.yaml`. **Provenance caveat:** that file records those weights as *user-supplied on 2026-10-19 and not verified against a live operator page from this environment* — they are the operator's terms as recorded, not a checked fact, and this audit did not verify them either |
| Early-season cold start | Linear ramp over 5 games | `early_season_prior` — Bayesian shrinkage with `equivalent_games` prior strength and an `available_at` timestamp gate that makes the blend leakage-safe |

So this audit's value is **not** a list of things to copy. It is four specific
things worth taking, one of which matters a great deal, and four defects in the
external code that must not be imported.

---

## 1. Quant & Feature Value Matrix

Grade = value **to this project**, given what it already has. A repository can
be competent and still grade Low here.

| # | Repository | Quant | Feature | What it actually contributes |
|---|---|---|---|---|
| 1 | `omedakbari/nba-edge-model` | **High** | Medium | `market_blend.blend` — market-anchored logit blend. `market_blend.shift_line` — line-scaled σ. The single most valuable find of the 18. |
| 2 | `kevon306/lockit-labs` | **Medium-High** | Low | `review.devig_pair` — no-vig consensus **averaged across every book posting both sides**. `find_closing_snapshot` — a true-close window flag for CLV quality. |
| 3 | `mroditall2x/NBA-Model-` | Low | **Medium-High** | `rot_blend.streak` / `RET_F` — return-from-absence minutes ramp. The one genuine *feature* gap it exposed. |
| 4 | `Karandeep-Singhhh/pick-and-roll` | **Medium** | Low | `combo.js rhoBetween` — starter/bench **role sub-split** within a team (`team_SS`/`team_SB`/`team_BB`, starter ≥ 28 min). A refinement our buckets lack. Its copula and PSD repair are both weaker than ours. |
| 5 | `kyleskom/NBA-Machine-Learning-Sports-Betting` | **Low** | Low | Nothing to port. Its `american_to_decimal` is **wrong** (see §4). Most-starred of the 18, which is why the defect is worth naming. |
| 6 | `serocky/nba-props-alerts` | Low | Low | Multiplicative de-vig and a fantasy-points weighting we already hold. No new method. |
| 7 | `jonxsong/CourtVision` | Low | **Medium** | Not quant — but it bundles per-player `usagePct`, `pace`, `offensiveRating`, `defensiveRating`, `trueShootingPct`, `PIE` from `nba_stats`. Directly relevant to our **empty team pace/efficiency columns** (the absent BigDataBall workbook). Evidence those fields are reachable without that workbook. |
| 8 | `swar/nba_api` | Low | Low | The canonical NBA stats client. A data-access library, no quant. (Its 17 "kelly" matches are player names in a static roster file.) |
| 9 | `hugohamynen/nba-predictor-dashboard` | Low | Low | Dashboard. No de-vig, Kelly, parlay, correlation, shrinkage or DFS code. |
| 10 | `jlemay90/nba-prop-intel` | Low | Low | As above — zero hits across all six probes. |
| 11 | `stevopotpie/courtline` | Low | Low | As above. |
| 12 | `jriordan55/nba-news-scanner` | Low | Low | News scraping. Out of scope. |
| 13 | `VishalT25/propscope` | Low | Low | Zero hits across all six probes. |
| 14 | `bttmly/nba` | Low | Low | JS data client. Zero quant. |
| 15 | `Nico1223XX/nba-beta-data` | Low | Low | One file. Zero quant. |
| 16 | `xtheredshirtx/fanduel-player-data` | Low | Low | Scraped payloads. Zero quant. |
| 17 | `alvinycheung/nba-picks-record` | Low | Low | Record-keeping. Zero quant. |
| 18 | `adam-suver/basketball-stats-aid` | Low | Low | Zero quant. |

**NCAA:** none of the retained material is NCAA, and none was examined for it.

---

## 2. Actionable Mathematical Upgrades

### Upgrade 1 — Market-anchored logit blend · **HIGH** · shipped

**Source:** `omedakbari/nba-edge-model`, `src/models/market_blend.py`.

**The formula.** With `m = logit(p_market)` and `d = logit(p_model) − m`:

```
z = a + b·m + c·d
p = expit(z)
```

**Why it matters here, from measured figures.** `prop_result_calibration_report`
on 533 graded rows:

```
mean stated probability   0.7077
observed win rate         0.4765   (254-279, 95% CI [43.4%, 51.9%])
ECE                       0.2311   at bin coverage 1.0 (5/5 bins)
Brier, served             0.3011
Brier, constant 0.5       0.2500
```

A served Brier **worse than a constant 0.5** means the stated numbers currently
cost more than saying nothing. Isotonic/sigmoid recalibration
(`src/models/ledger_calibration.py`) is the other half of the answer and cannot
be the whole of it: **a monotone map cannot change the ordering of rows**, so if
the ranking carries little signal, recalibration only compresses every row
toward the base rate. This blend attacks the other axis — *which quantity gets
calibrated* — by anchoring on the market and giving our disagreement a fitted
coefficient.

**Why the logit form and not an average.** It nests both endpoints exactly:

- `a=0, b=1, c=0` → **exactly the market**; the model is ignored.
- `a=0, b=1, c=1` → **exactly the model**; the market is ignored.
- `0 < c < 1` → the market, nudged by our disagreement.

So `c` is a **measurement**, not a dial: it answers "how much is our
disagreement with the closing price worth?" If the fit returns `c ≈ 0`, the
honest reading is that our model adds nothing to the market — and that is a
finding to report, not a failure to hide. No other blend shape can return that
answer legibly.

**Their coefficients are not copied.** The audited repo reports `c ≈ 0.15–0.23`
against its own model and its own book sample. Those describe their model.
There are **no default coefficients** in the shipped module; `fit_market_blend`
fits on our rows or refuses.

**The data prerequisite, which bounds this today.** A no-vig probability needs
*both* sides of a price. `prop_results` stores a single `odds` column (the taken
side), so the graded ledger alone cannot produce one.
`projections.odds_over_american` / `.odds_under_american` arrived in
**migration 010**, so the pair is recoverable for rows written from 010 onward
and for **no row written before it**. The fitter is gated and reports
`INSUFFICIENT_DATA` until post-010 history accumulates. Recovering `p_market`
is a `projections`↔`prop_results` join, which is the caller's to make — the
module stays pure, for the reason `settlement.calibration.taken_side_pairs`
gives: one reader of the ledger contract.

### Upgrade 2 — Line-scaled σ for the line-shift fallback · **MEDIUM-HIGH** · specified, not shipped

**Source:** `omedakbari/nba-edge-model`, `market_blend.shift_line`.

```python
sig = max(s0 + s1 * book_line, 0.3)
mu  = book_line + sig * norm.ppf(clip(book_p_over, 0.02, 0.98))
p   = clip(1 - norm.cdf((target_line - mu) / sig), 0.01, 0.99)
```

**Our dispersion path already dominates this** and should stay primary:
`line_diff._line_adjust_fair_prob` inverts a *fitted count dispersion* at the
book line to recover the implied mean, evaluates at the pick'em line, and
accounts for **push mass** explicitly. A Normal is the wrong family for a
discrete count and carries no push mass at all.

**Their contribution is to our *fallback*.** When no dispersion is on hand, we
currently apply `pts_per_prob = 0.03` — a flat 3 % of probability per point of
line, for every stat at every line level. That is clearly wrong in a way worth
fixing: **a point of line is worth far more at AST 4.5 than at PTS 27.5.**
`σ = s0 + s1·line` has the right first-order shape, because count variance grows
with the mean.

**Not shipped, deliberately,** because `s0`/`s1` must be fitted on *our* panel
per market, and shipping the functional form with their coefficients — or with
guessed ones — would publish a number this project has never measured. The fit
is a scripted measurement; the patch is one function in `line_diff.py` plus a
named `method` string so it can never be mistaken for the dispersion path.

### Upgrade 3 — Multi-book no-vig consensus · **MEDIUM** · specified, not shipped

**Source:** `kevon306/lockit-labs`, `review.devig_pair`.

For each bookmaker posting **both** sides, de-vig that book's pair, then average
the resulting probabilities across books.

`decision_board` currently selects **one** book by a precedence ranking
(`_source_rank`). Averaging across books reduces single-book noise and
single-book bias. Two subtleties that make it correct rather than merely
plausible, both of which their implementation gets right:

1. Average the **no-vig probabilities**, not the raw prices.
2. Include **only** books posting both sides — otherwise a one-sided quote's
   vig leaks into the consensus unremoved.

Not shipped because it changes which price the board reports, which is a
serving change and therefore an A/B measurement, not a patch.

### Upgrade 4 — Role dimension on same-team correlation buckets · **LOW-MEDIUM** · reported, not shipped

**Source:** `Karandeep-Singhhh/pick-and-roll`, `combo.js rhoBetween` —
`team_SS` / `team_SB` / `team_BB`, with starter defined as `starter_min ?? 28`.

Our `leg_correlation` pools **all** same-team pairs into one bucket per market
pair; theirs distinguishes starter/starter, starter/bench and bench/bench. The
refinement is real: two starters' usage competes more directly than a starter's
and a bench player's.

**Reported rather than patched, with the arithmetic for why.** Adding role as a
bucket dimension splits the same-team sample three ways against gates of **200
pairs *and* 50 distinct games**. On this season's graded history that very
likely fails the gates, and a bucket fitted below them is noise — which
`leg_correlation`'s own comment records learning the expensive way: *25 leg rows
from one game produce 2,700 pairs, and six buckets cleared a 200-pair gate on
that alone with ρ between −0.059 and +0.053.* The right sequence is to measure
the split's sample sizes first and widen the dimension only if the gates clear.
Shipping it blind would convert a gated refusal into an ungated number.

### Upgrade 5 — Return-from-absence ramp · **MEDIUM-HIGH** (feature side) · shipped

**Source:** `mroditall2x/NBA-Model-`, `rot_blend.streak` and `RET_F`.

**The gap, and why it is a prop-pricing gap specifically.** `absences.py`
answers *who is out tonight*; `teammate_cascade.py` answers *how many of his
teammates are out*. **Neither answers "is *he* just back, and from how long
out?"** — and that is a question every rolling feature gets wrong by
construction. Each `{STAT}_L5` / `{STAT}_L10` averages his last *appearances*,
which are the games **before** the absence. On his first game back those windows
describe a healthy player at full minutes — exactly the night he is least likely
to play them. The feature set therefore **overstates him on the one night the
error is predictable.**

**How the absence is observed with no injury feed.** `teammate_cascade` records
the load-bearing property of this panel: it holds only players who *appeared*
(median 10 rows per team-game, no row with `MIN == 0`). An absence leaves **no
row**, so a gap in a player's appearances measured against his team's schedule
*is* the absence. Nothing is inferred from a status string that may be stale at
tip-off.

**What is deliberately not ported.** Their hardcoded return factor
(`base * RET_F`) and 20-minute regular threshold are magnitudes fitted to
someone else's rotation data. The shipped module emits the **count** and the
**flag** and lets the model learn the ramp — which is both honest and strictly
more general, since a 1-game and a 15-game absence do not deserve the same
multiplier and one constant cannot express that.

---

## 3. Ready-to-Commit Code Patches

Both land as **pure, tested math outside the serving path**. Per AGENTS.md, a
new component reaches the model matrix only by winning a holdout in
`scripts/feature_ab.py`; neither is wired in.

### `src/quant/market_blend.py` (new) + `tests/test_market_blend.py` — 49 tests

`logit` / `expit` (clipped, overflow-safe), `MarketBlendCoefficients`,
`blend_probability`, `fit_market_blend` (logistic regression, dual-gated).

Gates, matching `leg_correlation`'s posture rather than inventing a new one:

- `MIN_ROWS_FOR_FIT = 400` — three free parameters on fewer fits noise.
- `MIN_SLATES_FOR_FIT = 30` — **the independent unit is the slate.** Rows from
  one night share pace, officiating and injury news, so 2,000 rows drawn from
  three nights are not 2,000 observations.

The fitter is verified by **recovering coefficients from data generated by the
blend itself** — the only way to test a fitter without asserting its own output
back at it. A separate test confirms it returns `c ≈ 0` when the model's
disagreement is pure noise, because that is the finding the module exists to be
able to report.

**Mutation-tested (make it fail first).** Five guards, each reverted, each
turning its named test red:

| Mutation | Test that went red |
|---|---|
| slate gate removed | `test_enough_rows_drawn_from_too_few_slates_is_still_refused` |
| sign of `c` flipped | `test_the_blend_shrinks_an_overconfident_model_toward_the_market` |
| silent fallback to the market | `test_an_unusable_coefficient_set_refuses_instead_of_falling_back` |
| probability clip removed | `test_logit_of_a_certainty_is_finite_rather_than_infinite` |
| no-disagreement check removed | `test_a_model_that_never_disagrees_cannot_identify_c` |

One contract **cannot** be guarded in code and is therefore asserted as a
footgun: both arguments must describe the **same side**. Pass the market's
P(over) with the model's P(under) and `d` holds a side flip rather than a
disagreement — undetectable from two floats.
`test_mixing_the_sides_produces_a_wrong_number_silently` pins the size of that
error so it cannot quietly shrink below notice.

### `src/features/return_ramp.py` (new) + `tests/test_return_ramp.py` — 19 tests

Emits `RET_GAMES_MISSED`, `RET_IS_RETURNING`,
`RET_FIRST_APPEARANCE_FOR_TEAM`, `RET_BASELINE_MIN_L5`, plus `RET_STATUS`.
Registered in `src/features/builder.py` as layer 13, alongside `box_rates`,
`fouls`, `dvp` and `minutes_weighted` — i.e. **not** in the feature contract.

Design decisions that are load-bearing:

- **A trade is not a return.** The gap is computed within (player, team); a
  first appearance for a new team reports NaN, not a count invented from two
  unrelated calendars.
- **A first appearance is unknown, not zero.** Reporting 0 missed would claim
  he played a previous game he has no relationship to.
- **A game no row covers is nobody's absence.** The schedule is built from the
  panel, so a team game absent from it cannot be charged to the roster — the
  evidence for it is simply not here.

**A real bug was caught during development and is recorded in the test that
now covers it.** The first spelling of the baseline was
`groupby(...)["min"].shift(1).rolling(W).mean()`, which groups **only the
shift** — `.rolling` then runs down the whole Series and its window **crosses
player boundaries**. On the smoke sample it gave the second player a baseline
of 30.0 on his *first* appearance, borrowed from the previous player's rows,
where the honest answer is NaN. The fix puts both steps inside
`.transform(...)`. This is exactly CLAUDE.md rule 1's failure mode, and it was
visible only because the smoke run printed per-player values.

**Mutation-tested.** Five guards, five named tests red:

| Mutation | Test that went red |
|---|---|
| rolling moved back outside the groupby | `test_the_baseline_window_never_crosses_from_one_player_to_another` |
| `.shift(1)` dropped | `test_the_baseline_never_contains_tonights_minutes` |
| first appearance counted as 0 missed | `test_a_first_appearance_is_unknown_rather_than_zero` |
| gap computed across teams | `test_a_first_appearance_for_a_new_team_is_unknown_not_a_long_absence` |
| internal re-sort removed | `test_the_emitted_columns_stay_aligned_to_the_rows_they_describe` |

The last one **initially passed under mutation** — it asserted that
`PLAYER_ID`, `GAME_ID` and `MIN` returned in input order, but those are carried
on the copied frame and are never reordered, so it could not fail. A test that
cannot fail is not a test. It was rewritten to pin each *emitted* value to the
(player, game) it describes, computed once ordered and once shuffled; it now
fails when the re-sort is removed.

---

## 4. Defects found — do NOT port

1. **`kyleskom/.../Kelly_Criterion.py` — `american_to_decimal` is wrong.**
   It returns `american/100` for positive odds (+150 → **1.5**) and
   `100/abs(american)` for negative (−110 → **0.909**). Both **omit the
   returned stake**; correct values are 2.5 and 1.909. Every Kelly fraction
   downstream is therefore miscalibrated, and a decimal "odds" below 1.0 is not
   a price at all. This is the most-starred repository of the 18, which is
   precisely why the defect is recorded rather than assumed absent.

2. **`pick-and-roll/combo.js` — `cholesky` falls back to the identity matrix.**
   After shrinking off-diagonals by ×0.85 down to 0.05 without achieving a
   factorisation, it returns the identity — **silently pricing a correlated
   parlay as independent.** That is the exact failure `leg_correlation` refuses
   to commit: an unresolved pair is *named* so the caller can abstain rather
   than treat "unknown" as "independent". Our `psd_repair.nearest_correlation`
   is the principled alternative.

3. **`pick-and-roll/combo.js` — `rhoBetween` has no side handling.** It computes
   ρ between *players*, not between *legs with sides*. Copying ρ unchanged onto
   a mixed over/under pair gives the copula the wrong **sign** of dependence.
   Ours applies `ρ · s_a · s_b`, verified by simulation: at latent ρ = 0.6 with
   0.55 marginals, φ(over,over) = +0.4083 and φ(over,under) = −0.4083.

4. **`omedakbari/.../ev.py` — `entry_ev` multiplies leg probabilities.**
   `math.prod(probs)` assumes independence; its own docstring admits
   *"same-game legs are correlated and this will misstate them."* For a
   **Flex** entry it is worse than a bias: correlation moves probability mass
   out of the middle of the hit-count distribution, so a partial-win tier and a
   power play move in **opposite** directions. `parlay_builder` already handles
   this exactly.

---

## 5. What to do next, in order

1. **Join `projections`↔`prop_results` and fit the blend.** The one number that
   matters is `c`. If it is near zero, our disagreement with the market is worth
   nothing and that redirects the whole modelling effort; if it is meaningfully
   positive, the blend is the largest single calibration win available. Gated at
   400 rows / 30 slates, so this becomes answerable as post-010 history
   accumulates — not today.
2. **Read `ledger_calibration`'s first AUC** (04:00 PT job). AUC decides whether
   recalibration can help at all; `c` decides whether anchoring can. They are
   complementary measurements and should be read together.
3. **Fit `s0`/`s1` per market** and replace the flat `pts_per_prob` fallback.
4. **Measure the same-team role split's sample sizes** before adding the
   dimension.
5. **Run `scripts/feature_ab.py` on `return_ramp`** once a slate's worth of
   rows carries it.
