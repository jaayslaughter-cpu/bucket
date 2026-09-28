# Wave 3 — Manual paper-research layer

**Status: RESEARCH_ONLY · MANUAL_ONLY**

You place bets yourself. PropIQ surfaces edges, logs what you entered, and
grades outcomes so the model can be audited and improved.

## Guarantees

| Rule | Behavior |
|------|----------|
| Placement | **Never** calls a book / Kalshi / DFS order API |
| Bankroll | **Never** auto-sizes stake (no Kelly from model) |
| Odds | EV only when `MarketContext.status=VALID` — **PropLine primary, OddsPapi fallback** (see [decision_board.md](decision_board.md)) |
| Pick’em | Sleeper / Underdog / PrizePicks — lines only, not VALID two-way |
| Purpose | Paper / shadow research for model improvement |

## Workflow

1. Build a research slate (`research-slate` CLI or compare-models exports).
2. Decide and bet **outside** PropIQ.
3. Log the bet you took (`log-manual-bet`) with line, side, American odds, stake.
4. After tipoff, grade with actuals (`grade-pending` / box scores).
5. Read `paper-report` for ROI-by-stat, pending count, and probability calibration.

## Module

- `src/quant/paper_research.py` — slate rows, manual log, improvement report
- `src/quant/historical_store.py` — grade-before-append lifecycle (Wave 4)
- CLI: `python -m scripts.nba_model_cli research-slate | log-manual-bet | paper-report`

## Disclaimer

Manual paper research is not live P&L, not a lock, and not automated wagering.
