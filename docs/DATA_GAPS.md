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

**Still open**

13. **The classifiers ignore the line they are scored at.** `CatBoost`,
    `XGBoostAdapter` and `XGBoostPropPipeline` are binary classifiers
    trained against one line definition, so asking for a probability at a
    different line returns the same number. The distribution path handles
    arbitrary lines correctly because it derives them from a fitted count
    distribution. Making the classifiers line-aware means retraining with
    the line as a feature — a modelling change, not a patch. Until then,
    treat classifier probabilities as valid only at the line they were
    labelled against.

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

8. **The minutes model is orphaned.** `MinutesModel` is built but never
   used by `compare.py`, and is the only model with no `save()`/`load()`.

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
  values are the safe choice.

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
