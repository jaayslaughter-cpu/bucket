# Data gaps and missing modules

What is absent from this repository today, what each absence blocks, and
what has to arrive before a model comparison means anything.

Nothing in this file is a guess about your data. Every "MISSING" was
verified by reading the code and opening the workbook.

## 1. Modules written to fill the gaps — read the provenance note

Five modules were imported by the packs but did not exist. They have now
been **written fresh for this repository**. They are not recovered
originals.

| Module | Status |
|---|---|
| `src/features/builder.py` | New. Pregame-only rolling features, shift-1 discipline, `assert_no_lookahead` |
| `src/features/fatigue_logic.py` | New. Schedule density + altitude. **Multipliers are unfitted heuristics** |
| `src/features/schedule.py` | New. Team rest, travel miles, rest advantage, arena relocation history |
| `src/features/team_strength.py` | New. Elo with margin-of-victory damping and offseason regression |
| `src/models/labels.py` | New. `RESEARCH_LINE` / `over_hit`, pushes dropped not graded |
| `src/models/xgboost_pipeline.py` | New. **Not your prior baseline** — see below |
| `src/quant/contracts.py` | New. `MarketContext`, `market_ev_gate`, two-way de-vig |
| `src/ingestion/boxscores.py` | New. Player game logs from the NBA stats API |
| `src/ingestion/basketball_reference.py` | New. SR season tables. Prior-season only — refuses same-season joins |
| `src/features/market_context.py` | New. Pregame opening spread/total + implied team totals. Refuses closing lines |
| `src/models/line_aware.py` | **Now wired** into build_components as the `line_aware` component |

**The XGBoost baseline is new code.** No earlier PropIQ baseline was
available, so `xgboost_pipeline.py` was written from scratch. A comparison
against it tells you which of two models written at the same time scored
better on the same split. It says nothing about beating an incumbent
production model, because there is no incumbent here. The file carries
this warning in its own docstring.

**The fatigue multipliers are guesses.** 0.97 for a back-to-back, 0.96 for
a 3-in-4, 0.94 for a 4-in-5, 0.98 for away-at-altitude. They are
conservative and documented, but nobody fitted them. Fitting them against
real player logs is open work.

## 2. Missing data

The `player_game_logs` table is the input to every model. It is empty, and
nothing in this repository populates it.

The BigDataBall workbook (verified: sheet `NBA-2025-26-TEAM`, 2,644 rows,
1,322 games, 10/21/2025 onward, 57 columns) is **team-level only**. Its
`PTS`, `A`, `TOT`, `ST`, `BL`, and `MIN` columns are team totals, not
player lines.

| Field | Status | Note |
|---|---|---|
| game date | AVAILABLE | workbook `DATE`; `games.game_date` |
| pregame / tipoff timestamp | **MISSING** | `Game.tipoff_utc` exists but nothing populates it; the workbook has no tip time |
| player id | NEEDS VALIDATION | schema enforces `uq_player_game`, but the table is empty; workbook has only starter *names* |
| event / game id | AVAILABLE | workbook `GAME-ID` is the NBA.com id (`0022500001`) |
| player team / opponent | AVAILABLE | via the verified NBA.com team map |
| home / away | NEEDS VALIDATION | workbook `VENUE (R/H/N)`. `IS_HOME` is deliberately forced `True` for the 12 neutral-site rows to suppress a fatigue altitude tax, so it does not literally mean "home" |
| minutes | **MISSING** | at player level |
| points / rebounds / assists | **MISSING** | at player level |
| made threes | **MISSING** | at player level |
| steals / blocks | **MISSING** | at player level |
| game total / spread | NEEDS VALIDATION | opening + closing present for 1,322 games, but **no capture timestamps** |
| injury / starter status | NEEDS VALIDATION | workbook `STARTING LINEUPS` gives 5 names per team-game, but it is published *postgame* — see leakage note below |
| sportsbook prop line / odds / timestamp | **MISSING** | `prop_line_snapshots` is well-designed and empty. No two-way odds source is wired |
| historical settlement | NEEDS VALIDATION | `src/settlement/` implements it correctly, but it needs player box scores first |

## 3. Defects found by inspection — status

**Fixed**

1. ~~All three models return the same prediction mean.~~ CatBoost and
   XGBoost now each fit a **regression head** alongside the classifier.
   MAE, RMSE and mean bias differ per model and can discriminate. Verified:
   before the fix all four models reported byte-identical MAE.

2. ~~Silent feature fabrication.~~ `_soft_fill` is gone. Missing features
   are dropped and logged; missing targets drop the row. Nothing is
   zero-filled.

3. ~~Calibration is never called.~~ Wired in. Each model/market gets a
   calibrator fitted on a **strictly earlier** window, chosen between
   isotonic and sigmoid by held-out Brier.

4. ~~Hardcoded dispersion.~~ The 1.35 variance multiplier is gone.
   Dispersion is fitted from **out-of-fold** residuals and the family
   (Poisson vs Negative Binomial) is selected by held-out log-likelihood.
   In-sample residuals were tried first and were visibly too tight — every
   boosted model collapsed to Poisson φ=1.0. Out-of-fold gives φ≈1.3–2.1.

5. ~~Cross-player rolling leakage.~~ Found by a test written for this
   repo: applying `.rolling()` to a groupby-shifted Series rolls across
   player boundaries, so a player's debut inherited the previous player's
   last game. Both shift and window now happen inside one per-group call.

6. ~~Duplicate push math.~~ `line_probs.py` is removed; its rule lived in
   two places and would have drifted.

7. ~~No opponent-strength feature.~~ `team_strength.py` adds pre-game Elo.
   Validated against the real 2025-26 workbook: Brier 0.212 and log loss
   0.613 against 0.25 / 0.693 baselines, 67.2% accuracy on 2,116
   team-games. Only `elo_pre` is exposed; `elo_post` never reaches the
   feature matrix.

8. ~~Rest computed across season boundaries.~~ `fatigue_logic` grouped by
   player alone, so a player's first game of a season read as roughly 150
   days of rest. Now partitioned by season.

9. ~~No test of the projection against the line.~~ `market_comparison.py`
   adds `MAE(line) - MAE(projection)` and hit rate bucketed by how strongly
   the model leaned. Both run against `RESEARCH_LINE` today and become a
   real market test unchanged once prop lines land; `is_market_line` keeps
   the two from being confused. The confidence verdict requires the spread
   to clear two standard errors, because with an uninformative model the
   top bucket outscores the bottom about half the time by chance.

**Fixed after code review** (see PR #1 comments for the full list)

10. ~~Early stopping read the scoring set.~~ `compare.py` handed `val` to
    CatBoost as its `eval_set` and then reported metrics on that same
    `val`, so the iteration count was chosen from the labels being scored.
    Early stopping now carves its holdout from the tail of `train`. On the
    demo panel this moved CatBoost PTS Brier 0.2380 → 0.2451 while leaving
    the distribution and XGBoost models untouched, since neither used an
    `eval_set` — that unchanged control is what shows the fix is surgical.

11. ~~Ensemble published invented certainty.~~ Probabilities accumulated
    from `0.0`, so a row no component could score came out as "certainly
    under". Now null with a warning.

12. ~~Calibrated under could go negative~~, ~~`MinutesModel` zero-filled
    absent features~~, ~~the EV gate accepted pick'em rows carrying odds~~,
    ~~`ModelMetadata` dropped `feature_schema_version`~~, ~~the distribution
    model lost its dispersion on reload~~, ~~one model's P(Over) was written
    into every market~~, ~~the ensemble's two probability APIs disagreed on
    whole-number lines~~.

12b. Second review pass, all fixed and regression-tested (each test was
    verified to fail against the unfixed code): ~~a date-only cutoff
    rendered a day early in `data_cutoff_pt`~~, ~~Elo rated a team against
    itself when two rows shared an abbreviation~~, ~~blank market rows were
    marked VALID because NaN is not None~~, ~~audit row counts could exceed
    the row total~~, ~~`is_valid_probability` accepted out-of-range
    components that happened to sum to 1~~, ~~a zero stake was silently
    re-staked at one unit~~, ~~the XGBoost adapter saved only its
    classifier, so a reloaded model returned null projections and
    different probabilities~~, ~~an unparseable minutes string was graded
    as a DNP and voided~~, ~~PRA was advertised as a market but had no
    column~~, ~~`captured_at_utc` defaulted to now(), fabricating an
    observation time that CLV and line movement are measured against~~,
    ~~the settlement README documented a CLI that did not exist~~.

12c. Wave merge, third pass — all four previously-open merge items closed:
    ~~config/model_comparison.yaml was stale~~ (merged and WIRED, since
    config nothing reads states a value the system is not using);
    ~~scoring_efficiency.py was referenced and absent~~ (written with the
    real True Shooting formula, and FGA/FTA/OREB/DREB added to both
    ingesters so it has genuine inputs); ~~PACE_MULTIPLIER was a
    fabricated 1.0~~ (now measured from box scores via the standard
    possessions estimate, null where it cannot be measured);
    ~~prob_calibration lacked ECE~~ (ported, with the sparse-bin coverage
    gate that stops a two-bin reliability diagram winning selection).

    Deliberately NOT merged, with reasons:

    - `_soft_fill` (compare.py). This is the silent feature fabrication
      removed as item 2 above. Taking the wave version back would reverse
      a documented fix.
    - `zip.zero_infl` / `combo.var_fudge` (config). Both are now fitted;
      setting them would reintroduce the hardcoded dispersion of item 4.
    - `propiq_analyst.py`. A Streamlit dashboard, which is out of scope by
      instruction, and it imports four modules (`propiq_logic_v2`,
      `matchup_overlay`, `ml_learner`, `vault_store`) that exist in no
      pack and in no version of this repository. Merging it would add a
      file that cannot run and re-add a dependency deliberately dropped.

**Still open**

13. **The classifiers can now be made line-aware.** `over_hit` is
    P(stat > RESEARCH_LINE) and the line was not a model input, so a
    fitted classifier returned the identical probability at every line.
    Measured on the demo panel: a line-blind XGBoost returns 0.3879 at
    8.5, 12.5, 16.5 and 20.5 alike — a range of exactly 0.0000.

    `src/models/line_aware.py` puts the line into the features and trains
    against labels built at many lines, so the model learns
    P(stat > L | features, L). The same measurement on the same panel
    gives 0.768 → 0.052 across that ladder, monotonically non-increasing.

    Two things make this safe rather than merely impressive:

    - **Candidate lines are generated from pregame quantities only.**
      Anchor them on the realised stat and the line feature carries the
      outcome; validation would look superb and a live slate would fail,
      because at scoring time the book cannot see the result either.
      `assert_lines_are_pregame` runs before every fit and rejects both
      an exact leak and a noisy one.
    - **Augmented copies of one game share one outcome**, so they must
      never straddle a train/validation split. They share a GAME_DATE, so
      a chronological split keeps them together;
      `assert_no_augmented_row_straddles` checks rather than assumes.

    The model abstains outside the standardised line range it was trained
    on. A boosted tree extrapolates badly there — the raw curve ticked
    back UP at the top of the ladder, which is impossible for a survival
    function. That was found by the monotonicity test, not reasoned about
    in advance.

    **Still to do:** train against genuinely posted lines rather than
    generated candidates. `augment_lines(keep_real_line_col=...)` already
    accepts them and marks them `line_source='posted'`; it needs a
    PropLine archive, which starts accumulating from the first
    `ingest-props` run. Generated candidates are a bridge, not the
    destination — they teach the shape of the probability curve, not the
    market's own view of where the line belongs.

14. **Orchestrator paths carry known defects** that cannot be verified
    until player data lands: `main.py` scores the whole lookback panel
    rather than the requested slate, and `predict-slate` does not actually
    score. The database defects previously listed here are fixed and
    regression-tested — credential percent-encoding, the projections
    upsert keying on a per-execution UUID, the ROI aggregates counting
    unsettled rows, and box-score name collisions.

7. **The evaluation target is self-referential.** `over_hit` is defined
   against `RESEARCH_LINE = {stat}_L10` while the projection derives from
   the same rolling history. Brier and log loss measure "is recent form
   above medium-term form", not prop skill. Only real prop lines fix this.

8. **The minutes model is trained but not consumed.** `MinutesModel` now
   has `save()`/`load()` (mean head plus every quantile head, refusing a
   partial artifact rather than silently narrowing the interval), and
   `train-minutes` writes the artifact instead of discarding it.

   It is still not consumed by `compare.py` or the orchestrator, and that
   is deliberate rather than pending: feeding projected minutes into the
   stat projections changes every number the pipeline produces. That is a
   modelling decision, not a wiring fix, and it wants real player data to
   evaluate against before it is made.

9. **Calibrator selection leak (minor).** `choose_calibrator()` selects a
   method on a clean 70/30 chronological split, then refits the deployed
   calibrator on all rows including the 30% used to select it.

10. **PRA is implemented; the other combos are not.** PRA is derived as
    the exact sum of `PTS + REB + AST` before the rolling step, so it gets
    the same shift-1 features as any other stat and its dispersion is
    fitted on *realised* PRA residuals. Modelling the realised total
    directly sidesteps the correlation problem rather than solving it —
    no joint simulation is needed, because the components are never
    combined as independent marginals. PR, PA and RA are not derived; they
    would follow the same pattern. A NaN in any component propagates, so a
    partial sum is never presented as a total.

11. **No no-vig market comparison, CLV, or ROI** in the comparison path.
    Blocked on two-way odds with capture timestamps.

## 4. Leakage traps to avoid when the data arrives

- **`STARTING LINEUPS`** sits in a postgame workbook. Using confirmed
  starters as a pregame feature leaks. Join starters only from a source
  timestamped before tip.
- **`CLOSING SPREAD` / `CLOSING TOTAL`** are known only at tip. Using them
  as features for a projection made hours earlier is look-ahead. Opening
  values are the safe choice. **Now enforced**:
  `src/features/market_context.py` raises `ClosingLineLeakageError` on any
  closing column offered as a feature, and closing values are reachable only
  through `closing_line_value()`, which is settlement.
- **Basketball-Reference season tables** (per game, per 100 possessions,
  play-by-play, adjusted shooting) are SEASON AGGREGATES. Joined onto their own season they leak
  the future into every game: a season TS% is computed from the game being
  predicted and from every game after it. The `Awards` column is the same
  trap at its most extreme — award shares are voted after the season ends.
  `src/ingestion/basketball_reference.py` refuses that join outright
  (`SeasonAggregateLeakageError`); only PRIOR-season aggregates are
  admissible, and within-season form must come from the shift-1 rollers in
  `builder.py`.

## 5. What unblocks the most, fastest

1. **Player game logs.** Run `nba-model ingest-logs` (or
   `load_player_game_logs()`) on a machine with network access to
   `stats.nba.com`. One call per season. This single item unblocks every
   model, and nothing else can substitute for it — the workbook is
   team-level.
2. **Real prop lines with capture timestamps.** This is what turns
   `RESEARCH_LINE` from a labelled proxy into an actual benchmark, and is
   a hard prerequisite for any claim about edge, CLV, or ROI.
3. **Fitted fatigue multipliers**, replacing the heuristics in
   `fatigue_logic.py`, once item 1 gives something to fit against.

Until item 1 lands, every number this repository produces comes from the
synthetic demo panel and means nothing about real NBA performance.
