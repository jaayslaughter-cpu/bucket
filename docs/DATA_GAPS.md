# Data gaps and missing modules

What is absent from this repository today, what each absence blocks, and
what has to arrive before a model comparison means anything.

Nothing in this file is a guess about your data. Every "MISSING" was
verified by reading the code and opening the workbook.

## 1. Missing modules

These are imported by code in this repository but do not exist here. Until
they land, `compare-models` and the `main.py` orchestrator cannot run.

| Module | Imported by | What its absence blocks |
|---|---|---|
| `src/features/builder.py` | `scripts/nba_model_cli.py`, `main.py`, tests | **Everything.** Provides `build_feature_matrix`, `assert_no_lookahead`, `attach_fatigue_column`, and the `{stat}_L2` / `{stat}_BASELINE` / `fatigue_multiplier` columns every model reads |
| `src/models/labels.py` | `src/models/compare.py`, CLI | `attach_research_over_labels` and `default_feature_cols` — without them there is no `over_hit` target and no feature list |
| `src/models/xgboost_pipeline.py` | `src/models/xgb_adapter.py`, `main.py` | The XGBoost baseline itself. **Do not let anyone regenerate this from scratch** — it is the thing the challenger is being compared against, and a reinvented copy would not be a fair baseline |
| `src/quant/contracts.py` | `main.py` | `MarketContext` and `market_ev_gate`, the abstention gate that stops EV being computed without two-way odds |
| `src/ingestion/boxscores.py` | `scripts/nba_model_cli.py` | `load_player_game_logs` — the real (non-demo) data path |

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

## 3. Known defects in the merged code

Found by inspection, not yet fixed. Listed so they are not mistaken for
working behaviour.

1. **All three models return the same prediction mean.** `XGBoostAdapter`,
   `CatBoostPropPipeline`, and `DistributionPropModel` each return
   `{market}_L2` from `predict_mean()`. MAE, RMSE, and mean bias are
   therefore identical across models and cannot pick a winner. Neither
   boosted model actually predicts a mean — both are pure classifiers.

2. **The evaluation target is self-referential.** `over_hit` is defined
   against `RESEARCH_LINE = {stat}_L10`, and the projection is
   `{stat}_L2` — both derived from the same rolling history. Brier and log
   loss currently measure "is his 2-game average above his 10-game
   average", not prop skill. The code labels this honestly; it still means
   no current metric is a real benchmark.

3. **Silent feature fabrication.** `compare.py::_soft_fill()` creates any
   missing feature column and fills it with `0.0`. A model can train on a
   wholly invented column with no error raised.

4. **Calibration is never called.** `src/models/prob_calibration.py` is
   complete but nothing invokes it, so `probability_over_calibrated` is
   always null in the exports.

5. **The minutes model is orphaned.** `MinutesModel` is built but never
   used by `compare.py`, and it is the only model with no `save()`/`load()`.

6. **`MIN_SEASON` may leak.** If it is a full-season average it leaks the
   future into every early-season row. Unverifiable until
   `src/features/builder.py` arrives.

7. **CatBoost train/serve skew.** `fit()` drops rows with NaN features;
   `predict_probability_over()` fills them with `0.0`.

8. **Calibrator selection leak (minor).** `choose_calibrator()` selects a
   method on a clean 70/30 chronological split, then refits the deployed
   calibrator on all rows including the 30% used to select it.

## 4. Leakage traps to avoid when the data arrives

- **`STARTING LINEUPS`** sits in a postgame workbook. Using confirmed
  starters as a pregame feature leaks. Join starters only from a source
  timestamped before tip.
- **`CLOSING SPREAD` / `CLOSING TOTAL`** are known only at tip. Using them
  as features for a projection made hours earlier is look-ahead. Opening
  values are the safe choice.

## 5. What unblocks the most, fastest

1. **Player game logs** for the 2025-26 season. This single item unblocks
   every model. The workbook cannot provide it.
2. **`src/features/builder.py` and `src/models/labels.py`**, so the
   feature and target contracts are the real ones rather than reinvented.
3. **`src/models/xgboost_pipeline.py`**, so the baseline is genuinely your
   baseline.
4. **Real prop lines with capture timestamps**, which is what turns
   `RESEARCH_LINE` from a labeled proxy into an actual benchmark.

Items 1–3 are prerequisites for any honest comparison. Item 4 is a
prerequisite for any claim about market edge, CLV, or ROI.
