# Settlement — Win/Loss/Push tracker (NBA only)

## Run

```bash
psql "$DATABASE_URL" -f migrations/002_prop_results.sql   # once
python -m src.settlement.cli settle                        # grade PENDING props
python -m src.settlement.cli summary --days 30             # W-L-P / strike rate / ROI / CLV
```

## Three things that are easy to get wrong, and how these handle them

**1. PUSH vs LOSS.** A push is an exact tie and is only possible on a
whole-number line — 25.5 cannot tie. The engine refuses to emit a PUSH
against a fractional line, and a DB CHECK rejects it independently
(verified: `PUSH on 25.5 -> rejected`). A push returns the stake: profit
is exactly 0, never negative.

**2. DNP is VOID, not LOSS.** A scratched player produces a voided prop.
Grading it as a loss would understate the model's real strike rate.
Voids are excluded from W-L and ROI, and reported separately.

**3. ROI ≠ CLV.** These are different measurements and are never summed:
- **ROI** needs a stake and a payout price. Pick'em rows carry a payout
  multiplier, not two-way American odds, so ROI *abstains* for them —
  they still count in W-L-P, and `unpriced_graded_n` shows how many were
  excluded.
- **CLV** measures whether the line/price moved toward you after you took
  it. It's a market-quality signal, not profit. Line movement and price
  movement are tracked as separate fields (`clv_line_points` /
  `clv_prob_points`) because conflating them is the most common CLV error.

## Denominators

- `graded_n`  = WIN + LOSS + PUSH
- `decided_n` = WIN + LOSS  ← strike rate uses this
- `priced_n`  = settled rows with real odds ← ROI uses this

Both are always returned. A strike rate under 30 decided props emits a
low-sample warning.

## Data source

`https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{GAME_ID}.json`
— real, unauthenticated, no mock mode. Requires browser-like headers
(bot-detected otherwise). Refuses to return a box score unless
`gameStatus == 3` (Final), so props are never graded on a partial line.

Missing stat components raise rather than defaulting to 0 — a missing
rebound count is not zero rebounds, and treating it as such would
mis-grade every Over as a LOSS.

## NCAA

Not implemented. Would need a different provider, id scheme, and name
crosswalk. Add `ncaa_boxscore_fetcher.py` implementing the same
`extract_player_stats()` contract rather than widening the NBA module.
