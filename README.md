# PropIQ Analytics

Pre-game NBA player-prop research. **NBA only. RESEARCH_ONLY** — this
repository does not place bets, automate wagering, or recommend wagers.

## What this is

A research pipeline that projects player stat lines from pregame-only
features, turns those projections into probabilities at a given line, and
then **compares competing models against each other** on chronological,
out-of-sample data.

The model comparison is the point. Nothing here claims a model is
profitable, and nothing should be treated as such until it has been
verified on leakage-safe forward data.

```
pregame-only features
    -> minutes model
    -> stat projection + uncertainty
    -> full outcome distribution
    -> P(over) / P(under) / P(push) at the posted line
    -> probability calibration
    -> compare against a de-vigged market price (when one exists)
    -> track Brier, log loss, calibration, CLV
```

## Layout

| Path | What lives there |
|---|---|
| `main.py` | Orchestrator — one pipeline run, writes to Postgres |
| `src/db/` | SQLAlchemy schema (10 tables), engine, queries |
| `src/ingestion/` | Data loaders. Currently: the BigDataBall workbook |
| `src/models/` | XGBoost adapter, CatBoost challenger, distribution model, ensemble, calibration, walk-forward splits, exports |
| `src/settlement/` | Post-game grading: box-score fetch, push-aware settlement, CLV, ROI |
| `src/utils/` | Timezone helpers |
| `scripts/nba_model_cli.py` | CLI entry point |
| `config/` | `model_comparison.yaml` — seeds, weights, hyperparameters |
| `migrations/` | SQL migrations |
| `outputs/` | Downloadable CSV/Parquet deliverables only |
| `tests/` | pytest suite |

## Setup

```bash
pip install -e ".[ml,db,dev]"      # or: pip install -r requirements.txt
cp .env.example .env               # then fill in DATABASE_URL
```

The BigDataBall workbook is **licensed data and is never committed**. Put
your copy at the path `BIGDATABALL_XLSX` points to:

```bash
mkdir -p data/external/bigdataball
cp ~/Downloads/2025-2026_NBA_Box_Score_Team-Stats*.xlsx data/external/bigdataball/
```

## Conventions

These are load-bearing; breaking them breaks the research validity.

- **Timestamps.** Storage is UTC. Everything user-facing is
  `America/Los_Angeles` and column names end in `_pt`. Slate dates are
  Pacific calendar days, never UTC midnight.
- **Pregame only.** A feature must have been knowable before tip. Postgame
  box-score fields are targets, never inputs.
- **Chronological splits only.** No random K-fold on time-series data.
- **Abstain rather than guess.** When a required field is absent the code
  emits `DATA_NOT_AVAILABLE` and a named reason. It does not fill, invent,
  or substitute another model's output.
- **Never commit** `.env`, API keys, or licensed data.

## Status

Working and merged: the DB schema, the BigDataBall loader, the model
interfaces, chronological splits, over/under/push math, the ensemble,
calibration, the export layer, the settlement engine, and the CLI.

**Not yet runnable end-to-end.** The pipeline needs player-level game logs
and several modules that are not in this repository yet. See
`docs/DATA_GAPS.md` for the current list and what each one blocks.
