# External repo review — 35 NBA betting/prop repositories vs PropIQ

RESEARCH_ONLY. Comparative analysis for design ideas. **Zero external
dependency**: nothing here is vendored, imported or copied. 26 of the 35
repositories carry no licence at all (see the licence table at the end), so
concept-level reimplementation is the only clean path as well as the required
one.

All 35 were cloned and read. Every claim about an **external** repository cites
the file and line it came from. Claims about **PropIQ** cite a file and line
where one exists; where the point is that we have no equivalent, that is stated
as an absence rather than given a reference. Where a repository's README claims
something its code does not do, that is stated.

---

## 0. The headline: five of the best ideas here are modules we already wrote

The most valuable finding is not a new feature. It is that the external repos
independently converged on five things PropIQ **already has code for** and
left unconnected or unreached. These are the cheapest wins available because the hard part
is done.

| Our module | State today | What the external repo shows |
|---|---|---|
| `src/features/teammate_cascade.py` | Wired at `builder.py:150`, **abstains on every row** because `BBS_OUT_FLAG` does not exist in the panel (`teammate_cascade.py:35`) | `tredaman5/src/common/injury_context.py` names the exact source: nba_api `boxscoresummaryv2` → `InactivePlayers` result set, one call per game, cached to one parquet per season |
| `src/models/minutes_model.py` | Trains and saves an artifact from `nba_model_cli.py:411`; its projection is **never a feature** for any prop model (no `MIN_PROJ`/`PROJ_MIN` anywhere) | `tredaman5/src/minutes/` trains minutes first, then feeds the projection into the points/rebounds/assists models |
| `src/models/prob_calibration.py` → `expected_calibration_error` (has an 80% bin-occupancy gate) and `select_model_dual` | The gated version is called **only** from `src/quant/paper_calibration.py:80`, and with a looser `min_bin_coverage=0.5`. `compare.py:616-623` computes its own **ungated** "simple ECE proxy", and `compare.py:691` **ranks models on it**. `select_model_dual` is an orphan | `conorwalsh99/src/model_selection.py:100-104` is where this guard comes from — the same 0.8 threshold. Our implementation is already correct; our main comparison path bypasses it |
| `src/models/recency.py` | **Orphan** — imported by nothing in `src/` or `scripts/`. (Note: recency *features* are NOT missing — `src/features/halflife.py:33` already runs shifted `ewm(halflife=...)`, producing 29 `_HL` / `_HL_SHRINK` columns. The orphan is training-time recency **sample weights**, which `xgboost_pipeline.py:252-269` supports but nothing supplies) | `DevanshDaxini/src/sports/nba/features.py:127,134` runs EWMA at **two spans** (10 and 30) alongside flat L5/L10. We run a single half-life — a tuning question, not a missing capability |
| `src/features/minutes_weighted.py` | **Orphan**; already contains the rule `MIN < 0.70 × prior_MIN_mean → weight 0.5` (`minutes_weighted.py:9`) | `DevanshDaxini/.../features.py:108-113` (`DNP_MIN_THRESHOLD = 5`, line 110) masks those rows to NaN *before* rolling, so a blowout bench ride cannot drag a player's L10 down |

Confirmed panel state (`data/external/training_pack/panel.parquet`, 214,381
rows × 196 columns): no `POSITION`, no rolling medians, no inactive/injury
column, no schedule-density column, no head-to-head column.
`CASCADE_USAGE_MULT` exists and is `NaN` on every row by design. Exponential
recency **is** present, as the `_HL` family (29 columns), so it is not a gap
despite no column being named `EWM`.

---

## 1. Feature discovery — what they have that we do not

### 1.1 Defence vs Position, normalised to the positional median

`DevanshDaxini/src/sports/nba/features.py:181-217`. Two steps:

```
OPP_{stat}_ALLOWED      = groupby([OPPONENT, POSITION])[stat].shift(1).rolling(10).mean()
OPP_{stat}_ALLOWED_DIFF = OPP_{stat}_ALLOWED - league_median(POSITION, SEASON)
```

The second step is the insight, and their comment states it well: *"Allowed 25
pts means nothing if league avg is 26."* Our `DEF_*_ALLOWED_PER100_L10` family
is opponent-level only — it cannot distinguish a team that funnels points to
guards from one that concedes them to bigs.

**Blocker:** we have no `POSITION`. **Unblocker, from another repo in this
set:** `rissicay/src/props.py:132`, `_classify_position(fg3m, reb)` derives a
coarse position class from a player's own shooting/rebounding profile — inputs
we already carry (`FG3M_L10`, `REB_L10`). No new ingestion source needed.

**Defect to avoid:** their `league_pos_avg` uses
`groupby([POSITION, SEASON_ID])[stat].transform('median')` — a whole-season
median including games after the row. Small in magnitude, but it is future
information and disqualified under our own rule. Use an expanding/as-of median.

### 1.2 Usage vacuum — sum vacated usage share, don't count bodies

Two repos solve the same problem at different quality levels, and the best
version is a hybrid of them:

- `tredaman5/src/common/injury_context.py:92-145` — `TEAMMATES_OUT_ROTATION`:
  count the players from the team's **most recent prior game** who played
  ≥15 minutes and appear on **tonight's official inactive list**. The docstring
  reasons explicitly about why this is not leakage, and it is correct: prior
  box score plus tonight's pregame announcement.
- `DevanshDaxini/.../features.py:261-307` — `MISSING_USAGE`: instead of a
  count, **sum the lagged usage rate** of the absent rotation players. A team
  missing 30% of its usage is a different situation from one missing two
  low-usage bench players, and a count cannot tell them apart.

**Recommended synthesis:** tredaman5's data source (official inactive list) +
DevanshDaxini's usage-weighted aggregation. That is exactly the
`BBS_OUT_FLAG` + `CASCADE_USAGE_MULT` pair our stub already declares.

**Two defects to avoid in their version:**
1. `player_latest_usage = df.sort_values('GAME_DATE').groupby('PLAYER_ID').tail(1)`
   takes each player's **last row in the whole dataset**, not their last row
   before the game being predicted — despite the comment claiming "BEFORE that
   game". A January row gets April's usage.
2. "Expected but did not play" is inferred from a missing box-score row, which
   conflates injury, rest, trade and G-League assignment. A traded player looks
   permanently absent from his old team, inflating `MISSING_USAGE` for the rest
   of the season. The official inactive list does not have this problem.

### 1.3 Smaller feature gaps, all cheap

| Feature | Source | Why it is not redundant with what we have |
|---|---|---|
| Rolling **medians** (`_L5_Median`, `_L10_Median`) | `DevanshDaxini/.../features.py:137-138` | A median is robust to the one 40-point outburst that moves a mean; for a skewed count stat it is a genuinely different signal |
| **Schedule density** — games in trailing 7 days | `DevanshDaxini/.../features.py:311`; `Kalshi/props/fatigue_model.py:282` | We have `days_rest` and `IS_B2B_*`, which describe the last gap only. Three games in five nights with one day between each is invisible to both |
| **Head-to-head** player-vs-opponent history | `DevanshDaxini/.../features.py:419` | Absent from our panel entirely |
| Player's **home/away split means**, selected by tonight's venue | `DevanshDaxini/.../features.py:393` | We carry `IS_HOME` as a flag; we do not carry the player's own home-vs-away scoring level |
| Market-specific rebound features: opponent shot volume, opponent 3PT rate | `DevanshDaxini/.../features.py:528` | More opponent attempts = more available rebounds; 3PT rate changes rebound distance. Our REB features are player- and defence-side, not opponent-volume-side |

---

## 2. Techniques worth taking

### 2.1 Anchor player projections to the market's implied team total

`DanielTomaro13/src/sim.py:97-101` — `_anchor_to_winprob` shifts the two team
score means so the implied margin reproduces a target win probability, **while
holding the total fixed**.

The prop analogue: our panel already carries `MKT_IMPLIED_TEAM_TOTAL`
(`market_context.py:132`, `(total − spread) / 2`). Today it is used only as a
model *feature*. It can also be used as a **constraint**: if the sum of our
projected points across a team's rotation is 118 and the market's implied team
total is 111, one of the two is wrong, and the discrepancy is a measurable
signal rather than a number to ignore. This is the most original idea in the
set and the input is already ingested.

It must be evaluated, not assumed: the correct experiment is whether the
reconciliation residual improves Brier/ECE as a feature, before anything is
rescaled by it.

### 2.2 An ECE that cannot be gamed by bin sparsity — we have the guard and bypass it

`conorwalsh99/src/model_selection.py:100-104` — the only peer-reviewed
repository in the set (*Machine Learning with Applications*, 2024,
doi:10.1016/j.mlwa.2024.100539). When selecting on calibration they require
**at least 80% of 20 bins to be non-empty**, and return the worst possible
score otherwise. The reason is that a model which always predicts 0.52
concentrates in one bin and can post an excellent ECE while being useless.

**We already implemented this**, to the same threshold:
`src/models/prob_calibration.py:139`, `min_bin_coverage: float = 0.8`, returning
`ece: None` and `gate_passed: False` when coverage falls short — and
`select_model_dual` implements the paper's accuracy-vs-calibration dual
selection alongside it.

**But the main comparison harness does not use either.**
`src/models/compare.py:616-623` computes an inline, **ungated** figure it calls
a "Simple ECE proxy" straight off `reliability_table`, and assigns it to
`calibration_error`. `compare.py:691` then sorts models by
`(brier_score, log_loss, calibration_error)` — so an ungated ECE participates
in model ranking, which is precisely what the guard exists to prevent.
`select_model_dual` is called from nowhere.

This makes it a wiring fix, not a new feature, and it retroactively affects
every `calibration_error` figure our comparison runs have reported.

### 2.3 Kelly bounded by the probability interval

`ethangu16/src/utils/betting_advanced.py:123-149` — compute Kelly at the point
estimate *and* at both ends of the probability CI, then take the **minimum**.

We already produce `joint_probability_stderr` (`src/quant/parlay.py:117`, set
at `:639`) and abstain on relative noise (`MAX_RELATIVE_STANDARD_ERROR`,
`parlay.py:72`). Propagating that interval into the displayed fractional-Kelly
figure is a natural extension and stays MANUAL_ONLY: it makes the stake
suggestion shrink when the model is unsure, rather than only when the edge is
thin.

### 2.4 Two cheap reporting upgrades

- **Brier Skill Score**, `1 − Brier/0.25` (`ethangu16/src/evaluation/calibration_analysis.py:65`).
  0.25 is the Brier of always predicting 0.5, so the score reads directly as
  "how much better than a coin flip". More legible than a raw Brier in a report.
- **Calibration circuit-breaker** (`Kalshi/models/calibrator.py:37,270,346`):
  `halt_threshold = 0.25`, and `should_halt(market)` when the rolling Brier
  exceeds it. **We have no equivalent** — grep for `halt`/`circuit` across
  `src/` returns nothing, and `parlay_log.py:731` (`leg_calibration_frame`)
  produces the `(model_prob, hit)` pairs but nothing consumes them as a gate.
  Framed our way: a market whose live Brier has decayed past the coin-flip line
  should stop being published until it is re-examined.
- **Seed-averaged model selection** (`conorwalsh99/src/model_selection.py:117,137`):
  fit over several random seeds and average before comparing models. Our A/B
  harness compares deltas against fold SD, which is related but does not remove
  single-seed selection noise.

---

## 3. Where PropIQ is already ahead — do not "upgrade" backwards

| Area | PropIQ | Best of the 35 |
|---|---|---|
| **Projection → probability** | `residuals.py`: four families (poisson, negbin, zip, normal) chosen by **out-of-sample NLL**, with a continuity correction so the normal's likelihood is comparable to a discrete one (`residuals.py:189-195`) | `tredaman5`: one global sigma per market, set to test-set RMSE (`predict_slate.py:83`). `rissicay`: a hardcoded Poisson-or-Normal rule (`props.py:181`) plus `sigma = projection × cv` (`props.py:191-196`) — the best of the others, still well behind |
| **Fair probability** | No-vig de-vigging before any EV claim | `conorwalsh99/src/simulate.py:160`, `tredaman5/src/betting/edge_calculator.py:50`, `ethangu16` all use `1/odds` — the **vigged** implied probability — and call the difference an edge |
| **Parlay pricing** | Gaussian copula, tetrachoric buckets gated on pairs **and** distinct games, matrix validated for symmetry/unit-diagonal/PSD, same-game tickets refused without a quoted combined price | `snandyala13/backtest_real_lines.py:503,687`: `profit = hits*2.5 − n  # ~+250 odds for 2-leg` — a hardcoded round-number payout, for both SGP and cross-game |
| **Calibration** | Shared out-of-fold predictions, isotonic-vs-Platt selection, ECE reported raw and calibrated, and a gated ECE implementation that matches the peer-reviewed guard (though `compare.py` bypasses it — §2.2) | `Kalshi/models/calibrator.py`: Platt only, fitted on a trailing 30-day window of settled predictions, no out-of-fold discipline |
| **Leakage discipline** | shift-1 everywhere, as-of cutoffs required not defaulted, fold boundaries grouped by calendar day | See §4.1 — the largest repo's headline backtest violates all three |

---

## 4. Cautions — claims in these repos that their code does not support

### 4.1 `snandyala13/AIBall`: the README's ROI ladder is a selection artifact

Its README reports 34,697 predictions over 1,727 games with hit rate rising
monotonically from 52.5% to 81.5% as an "ML threshold" tightens, and
"+13.4% ROI". Three defects in `backtest_real_lines.py` make those numbers
uninterpretable:

1. **`pipeline.build_game_context(home_team, away_team)`** (line 244) takes
   **no date argument** — `data_pipeline.py:1006` confirms the signature is
   `(home_team, away_team, player_names)`. Its own docstring says it includes
   "Live DraftKings props" and "Injury report". Every historical prediction is
   built from data current at run time.
2. **`get_injuries_for_team(team_id)`** (line 90) calls a live
   `/player_injuries` endpoint with no date filter, and those injuries are
   applied to a game from October 2024.
3. **`player_names = list(actual_stats.keys())`** (line 253) where
   `actual_stats` is filtered to `mins >= 10` from the **box score of the game
   being predicted** (line 215). The model is only asked about players who
   turned out to play. Every DNP and every short-minutes night — precisely the
   hardest props — is removed by construction.

The monotonic ladder is what selection on post-hoc information looks like. This
is the failure mode our own "do not use future information when backtesting"
rule exists to prevent, and it is worth keeping as a reference example.

### 4.2 `Cuisine1234-hash/Kalshi-...`: the most impressive-looking repo is largely inert

34,887 lines and by far the best module names in the set
(`bayesian_updater.py`, `teammate_cascade.py`, `fatigue_model.py`,
`pace_efficiency.py`). The ideas are the right ideas. The code is not ready to
learn from:

**`props/fatigue_model.py`** — the docstring says "All adjustments empirically
calibrated from database" and lists specific effects (B2B −2.1 pts, age 30+
−3.2, west-to-east −1.2). Those are **hardcoded literals** (lines 102-143) with
no fitting code anywhere, and every input that would modulate them is a stub:
`_get_player_age` → `None`, `_get_player_usage` → `None`,
`_get_travel_direction` → always `LOCAL`, `_estimate_timezone_changes` → `0`,
`_is_road_trip_game` → `False` (lines 493-527). So the age, usage and travel
multipliers never fire. And `_compute_days_rest` (line 529) passes `player_id`
into the `home_team_id`/`away_team_id` columns, so the query never matches and
rest is always 0 — i.e. every player is treated as permanently on a back-to-back.

**`props/teammate_cascade.py`** — right concept, six problems:
- `delta_std = float(np.std(stats_without_vals - stats_with_vals))` (line 471)
  subtracts two **unpaired arrays of different lengths**; the `ValueError` is
  swallowed by the bare `except Exception` at line 504, so cascade effects silently
  vanish whenever the two samples differ in size, which is nearly always.
- `MIN_SAMPLE_SIZE = 2` (line 31) — a delta fitted on two games.
- The "teammate" queries (lines 406-411, 421) join on `game_id` and `half` only,
  never on **team**, so two players on *opposing* teams count as teammates.
- The `games_with` query has **no date filter**, so it can include games after
  `as_of_date`; only `games_without` filters, and with `<=`.
- `usage_shift = pts_boost * 0.005` (line 312) — an invented coefficient,
  labelled "Heuristic".
- `avg_confidence = np.mean([boosts[s][2] ...])` (line 315) averages index
  `[2]`, which `compute_absence_boost` returns as the **CI upper bound**, and
  calls the result a confidence.

The `_apply_shrinkage` idea (shrink a small-sample delta toward the population
mean) is sound and worth keeping. Its `_compute_confidence` is not:
`cv = std / (abs(std) + 1.0)` is labelled "coefficient of variation" but is a
units-dependent saturating function of the SD alone, so it penalises points
(SD≈8 → 0.11) far more than rebounds (SD≈2 → 0.33) for no statistical reason.

### 4.3 Smaller mislabels

- `ethangu16/src/utils/betting_advanced.py:55` —
  `posterior = prior_weight*prior + (1-prior_weight)*likelihood` is a **linear
  blend**, documented as "using Bayes' theorem".
- `anshs527/src/paper_trading.py:435` — `calculate_confidence_correlation`
  computes bucketed win rates by confidence tier, not a correlation. The
  underlying report (realised hit rate per confidence tier) is worth having;
  we already store `confidence_tier` in the parlay log.
- `DevanshDaxini/.../features.py:127` — the `_L20` column is `ewm(span=10)`,
  and the comment's "~20-game half-life" and "2× more weight" are both wrong
  (span=10 gives a ≈3.5-game half-life and ≈54× the weight on the latest game
  versus 20 back). The technique is right; the annotation is not.
- `digsallday/...` covers NFL, NBA, **NCAAF and NCAAB**. Out of scope under our
  NBA-only rule. The one transferable piece is
  `Bootstrap/parameter_estimation.py` — bootstrap CIs on fitted parameters.

---

## 5. Prioritised recommendations

Ordered by (expected accuracy gain) ÷ (work), highest first. Every item is a
native reimplementation; nothing is vendored.

**P0 — unblock what we already built**

1. **Ingest the official inactive list → populate `BBS_OUT_FLAG`.** Source:
   nba_api `boxscoresummaryv2` `InactivePlayers` result set, per game, cached
   to one parquet per season (pattern: `tredaman5/src/common/injury_context.py`).
   This alone switches on `teammate_cascade.py`, which is already wired into
   `builder.py` and abstaining on every row today. Our existing
   `CASCADE_TEAMMATE_OUTS` becomes real, and `CASCADE_USAGE_MULT` becomes
   fittable. ~1,200 cached calls per season, paid once.
2. **Feed `MinutesModel`'s projection into the prop models as a feature.** The
   model exists and trains; nothing consumes its output. Gate it behind the A/B
   harness against the `MIN_L5`/`MIN_L10` baseline, exactly as
   `tredaman5/src/minutes/train.py:144-147` benchmarks itself against
   `MIN_ROLL_5`.
3. **Route `compare.py` through `prob_calibration.expected_calibration_error`**
   instead of its inline ungated "simple ECE proxy" (`compare.py:616-623`), and
   decide deliberately whether `calibration_error` should keep participating in
   the model ranking at `compare.py:691` when the gate fails. The gated
   implementation already exists at `prob_calibration.py:139` with the paper's
   0.8 threshold; nothing new has to be written. Smallest diff in this list, and
   it hardens every calibration number our comparison runs have reported.
   Consider wiring the orphaned `select_model_dual` at the same time.

**P1 — new features, each through the existing A/B harness**

4. **Derive `POSITION` from the box-score profile**
   (`rissicay/src/props.py:132` pattern), then add **defence-vs-position
   normalised to an as-of positional median**. Highest expected signal of any
   new feature here, because opponent-level defence cannot express funnelling.
5. **`MISSING_USAGE`** — sum of the as-of lagged usage share of absent rotation
   players (depends on P0.1). Strictly more informative than a count.
6. **Add a second EWMA span** alongside the existing `_HL` family (we run one
   half-life; they run spans 10 and 30). Separately, decide whether the orphaned
   `recency.py` should supply training-time sample weights — `xgboost_pipeline.py:252-269`
   already accepts them and nothing passes any.
7. **Wire `minutes_weighted.py`**, or add near-DNP masking before rolling. Pick
   one: our down-weight rule and their NaN-mask solve the same problem.
8. **Rolling medians**, **schedule density** (games in trailing 7 days), and
   the **player's home/away split means**. Cheap, independent, and each is a
   one-layer addition.

**P2 — techniques**

9. **Market-anchoring residual**: compare the sum of projected player points
   for a team against `MKT_IMPLIED_TEAM_TOTAL` and expose the discrepancy as a
   feature. Novel, and the input is already in the panel.
10. **Kelly bounded by the probability interval** — display
    `min(kelly(p), kelly(p_lo), kelly(p_hi))`. Stays MANUAL_ONLY.
11. **Calibration circuit-breaker** — stop publishing a market whose rolling
    Brier passes 0.25, and report **Brier Skill Score** alongside raw Brier.
12. **Seed-averaged model selection** in the comparison harness.

**Explicitly not recommended**

- Any staking automation. Every repo here that sizes bets does it automatically;
  ours stays MANUAL_ONLY.
- `Kalshi/props/fatigue_model.py`'s coefficients — they are unfitted literals,
  and the docstring calling them "empirically calibrated" is the exact failure
  mode we audit for. If we want a fatigue layer, fit it on our own panel.
- `Kalshi/props/twitter_alpha.py` (social-media signal) and its live/halftime
  modules — we are pregame, Discord-notified, and not an execution system.

---

## 6. Coverage: all 35 repositories

| Repository | Py LOC | Licence | Verdict |
|---|---|---|---|
| Cuisine1234-hash/Kalshi-Trading-bot-code-for-nba | 34,887 | none | Best module *ideas*; fatigue + cascade largely inert (§4.2). Take: shrinkage, circuit-breaker |
| DevanshDaxini/Sports-EV-Bot | 10,467 | none | **Richest feature library.** Take: DvP-normalised, MISSING_USAGE, EWMA, medians, DNP-mask, density |
| tredaman5/NBA-Player-Prop-Model | 4,799 | LICENSE | **Cleanest architecture.** Take: official inactive list, minutes→prop wiring, baseline benchmarking |
| conorwalsh99/ml-for-sports-betting | 5,038 | MIT | Peer-reviewed. Take: ECE occupancy guard, seed-averaged selection |
| DanielTomaro13/Basketball-Modelling | 4,024 | MIT | Take: market-anchoring (`_anchor_to_winprob`) |
| ethangu16/nba-betting-ev-model | 2,926 | none | Take: Kelly-with-uncertainty, Brier Skill Score. Caution: mislabelled "Bayes" |
| rissicay/nba-betting-model | 7,950 | none | Take: position-from-profile. Prop math behind ours; Streamlit, not our surface |
| snandyala13/AIBall | 12,055 | none | **Cautionary** (§4.1). Backtest not leakage-safe; ROI ladder is a selection artifact |
| anshs527/nba-betting-model | 6,383 | none | Parlay bookkeeping + paper trading. Take: hit rate by confidence tier |
| digsallday/Beating-the-House | 4,698 | none | Contains NCAAF/NCAAB — out of scope. Take: bootstrap CIs on parameters |
| n1ops/nba-betting-predictor | 2,347 | none | Consistency/trend features, Discord output. Nothing we lack |
| abhisaradev/betting-the-regression | 5,885 | MIT | Experience/tier features, WNBA. Out of NBA scope |
| geloganu/NBA_Over_Under_Models | 891 | MIT | Team over/under notebooks. Not player props |
| BettingApp-hcai/betting_edge | 4,377 | none | Football/soccer-focused despite the name. Not applicable |
| Joe-Ferrara/predicting-nba-games | 2,472 | none | Game scores/spread. Not props |
| KamranSHussain/NBA-Propositions-Forecasting-App | 3,625 | MIT | Mostly bundled JSON data; thin modelling |
| Exidekat/math178project | 911 | none | Coursework notebooks; Kelly demo |
| clandgrebe/predicting-nba-games-ml | notebooks | none | Coursework; game outcomes |
| DavidKatzman/Basketball_Betting_Model | notebooks | none | SPSS `.sav` + 2 notebooks; no pipeline |
| GogateVarun/NBA-Game-Predictor | 126 | none | Small Keras game predictor |
| loganchoi/NBA-Game-Predictor | notebook | none | One notebook + slide deck |
| mmyoung77/betModel | notebooks | none | Two exploratory notebooks |
| chogan72/BasketballBettingModel | 492 | none | Two scripts + xlsx |
| kpundhir/Prop-Model | 168 | none | Stub |
| J-Nguyeners/NBA-Model | 0 | none | Spreadsheet only |
| jordangalexander/nba-player-props | 3,407 | none | Mostly CSV; light scraping |
| adamrajkotwala/machine-learning-parlay-generator | 486 | none | Parlay generator; correlation handling is rolling averages only, well behind our copula |
| Cuisine…/pregame_bot (same repo) | — | — | Duplicated subtree of the Kalshi props modules |
| mitchelldawkinsjr/NBA-Stat-Spot | 31,273 (JS/TS) | none | Web app. No modelling |
| WFord26/BetTrack | 6,742 (TS) | MIT | Bet tracker. No modelling |
| TopTrenDev/sports-betting-platform | 1,858 | none | TS platform scaffold |
| potternate/PropBet | 0 Py | MIT | TS front end |
| ejjlittle/ev-betting-model | 633 | MIT | Mostly front end |
| nihalafs11/StatsPicksNBA | 1,374 | none | Small app |
| adam-suver/basketball-stats-aid | Java | none | Java/SQL coursework |
| matthew-hoty/nba-player-prop-analysis-shiny | R | none | R Shiny dashboard |

**Licence summary:** 26 of 35 carry no licence (all rights reserved by
default), 8 are MIT, 1 has an unidentified `LICENSE` file. Concept-level
reimplementation only — which is the standing constraint regardless.
