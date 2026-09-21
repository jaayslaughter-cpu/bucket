# Betting decision layer

**Status: RESEARCH_ONLY · MANUAL_ONLY**

When it is time to bet, PropIQ ranks Over and Under so **you** choose. You
place the wager outside PropIQ. No auto-placement. No Kelly.

## Guarantees

| Rule | Behavior |
|------|----------|
| Placement | **Never** calls a book / DFS order API |
| Bankroll | **Never** auto-sizes stake |
| EV | Only from `MarketContext.status=VALID` two-way American odds |
| Source | **PropLine primary, OddsPapi fallback** — and the source is on every row |
| Pick'em | Line / line-diff research only — never VALID EV |
| Whole lines | Refuses silent `1 − P(over)` for the under when a push is possible |
| Language | Refuses to emit "lock", "best bet", "guaranteed" and the like |

The first three are enforced in code, not just documented:
`test_layer_cannot_reach_a_book_or_size_a_stake` greps the module for
network and staking symbols, and `_assert_no_claims` raises on the banned
vocabulary before a row can be written.

## What you get

| Field | Meaning |
|---|---|
| `side` | `over` or `under` — **both sides always expanded** |
| `decision_status` | `CONSIDER` or `ABSTAIN` |
| `decision_basis` | `book_ev` · `model_lean` · `pickem_line_only` · `unavailable` |
| `book_ev` | Side EV, only when a source cleared the gate |
| `preferred_side` | Higher-EV side, **only when both sides are priced** |
| `why` | Short human-readable reason, including every refusal |
| `rank` | CONSIDER first, then band, then score |
| `book_source` | `propline` or `oddspapi` — which feed priced this row |
| `book_fallback_used` | True when PropLine was present but unusable |
| `book_sources_skipped` | What was passed over, and the gate's reason for each |
| `line_can_push` | True on a whole line (and on an unknown one) |
| `model_prob` | P(this side) — never P(over) for an under row |
| `rank_score` | EV for priced rows, lean for unpriced ones (banded, never mixed) |

## Source precedence

`resolve_market` walks `SOURCE_PRECEDENCE = ("propline", "oddspapi")` and
takes the first source whose market clears the EV gate — **by precedence,
not by arrival order**. `enrich_row_with_resolved_market(row, candidates)`
is the entry point that applies it and stamps the source onto the row;
`enrich_row_with_book` prices whatever snapshot it is handed. The common fallback is a PrizePicks or Underdog
row: a pick'em board publishes a payout multiplier rather than a two-way
price, so it cannot be de-vigged, PropLine is skipped for pricing, and
OddsPapi prices the row instead. That is recorded, never silent:
`fallback_used=True` and `sources_skipped` carries the gate's own reason.

The pick'em line itself survives on the row for line-diff research even
when it cannot price anything.

## The four bases, and why they are banded

Only one of the four is a price.

- **`book_ev`** — two-way American odds cleared the gate; the EV is real.
- **`model_lean`** — no priced market. The model leans, and a lean is not an
  edge, because nothing here says what it costs.
- **`pickem_line_only`** — a pick'em board posted a line. EV stays undefined.
- **`unavailable`** — nothing to say.

Rows are sorted by **status, then band, then score**. A model lean can never
outrank a priced edge, however large the lean. Putting the two on one
numeric scale would imply they are comparable, and they are not: one has a
price behind it and the other does not.

## The push rule

On a whole-number line N the bet has three outcomes: over (> N), under
(< N), and push (= N). So:

```
1 − P(over)  =  P(under) + P(push)
```

Using the complement as P(under) books **every push as an under win** and
inflates the under's EV by exactly the push mass. This layer refuses it
whenever a push is possible, including when the line is unknown — an unseen
line cannot be shown to be a half-line, and the safe default is the one that
refuses. Half-lines cannot push, so there the complement is exact and is
used.

A refused under does **not** cost the over its EV. The de-vig needs both
*prices* (which the gate has already guaranteed), but each side's EV needs
only its own probability. So a whole line with no explicit P(under) still
produces a real over EV, the under abstains with a named reason, and
`preferred_side` is `None` because the pair is not comparable.

To price both sides of a whole line, supply `model_p_under` (and
`model_p_push`). `research_slate_from_predictions` now carries both through
from the model's own over/under/push output.

## Workflow

```bash
# 1. Build the board
PYTHONPATH=. python scripts/nba_model_cli.py decision-board --demo

# 2. Scan CONSIDER rows
open outputs/demo/decision_board.csv

# 3. Bet outside PropIQ (book / pick'em app) — by hand, at your own size

# 4. Log what you actually took
PYTHONPATH=. python scripts/nba_model_cli.py log-manual-bet \
    --game-id G1 --prop-stat PTS --line 24.5 --side under \
    --odds -110 --model-prob 0.58 --model-prob-side 0.42 --unit-stake 1

# 5. Grade and audit
PYTHONPATH=. python scripts/nba_model_cli.py paper-report
```

### `--model-prob` is P(the side you took)

`--model-prob` is **P(the side you took)** — P(over) for an over, P(under)
for an under. One quantity, stored as logged and read back unchanged by
both `paper-report` and `paper-calibration`.

This matters because the alternative convention (always store P(over), and
derive the under as `1 − P(over)`) is wrong on exactly the lines where it is
used: a whole line's complement includes the push mass. Storing P(side)
removes the conversion, and with it the bug.

`candidate_to_manual_bet_fields` emits the right value for the row you
picked, so you can copy it straight across.

## Flags

| Flag | Effect |
|---|---|
| `--min-ev 0.02` | CONSIDER only when VALID book EV clears 2% |
| `--require-valid-book` | Demote every non-`book_ev` row to ABSTAIN |
| `--min-lean 0.05` | For unpriced rows: how far past P=0.50 the lean must be |
| `--consider-only` | Drop ABSTAIN rows |
| `--top-n 25` | Keep the top ranked candidates |

## Modules

- `src/quant/decision_board.py` — source precedence, expand / rank / CSV, and
  the handoff fields for the manual log (`build_decision_board`,
  `BettingDecisionCandidate`, `candidate_to_manual_bet_fields`)
- `src/quant/paper_research.py` — `resolve_two_way_model_probs` /
  `model_prob_for_side` hold the push rule
- `src/quant/paper_research.py` — dual-side slate + manual log
- CLI: `decision-board` · `research-slate` · `log-manual-bet` · `paper-report`

## Current state

No archived PropLine pull exists in this repository yet, so
`decision-board` attaches no market candidates and **every row lands on
`model_lean` or `unavailable`**. That is the truthful state, not a bug:
there is nothing priced to rank. Once a PropLine pull is archived, pass it
as `markets` to `decision_board_from_slate` and the `book_ev` band fills in.

## Audit provenance

An external audit (2026-09-20) reviewed this tree and found 17 defects,
including two that are pinned here by regression test in
`tests/test_audit_fixes.py`:

- `PACE_MULTIPLIER` divided each team's rolling pace by a **season-wide**
  league mean, so every early-season row was measured against games that had
  not happened yet. The baseline is now an as-of expanding daily mean, and
  the test proves it by deleting later games and checking that no earlier
  value moves.
- `line_diff` and `enrich_row_with_book` both advertised
  `MarketContext | PropMarketSnapshot` but read `market.total`, which
  `MarketContext` does not have — every `MarketContext` caller raised
  `AttributeError`.

The audit independently found the same push-mass defect described above,
which is the second reason it is worth stating twice.

## Parlays

`src/quant/parlay.py` prices a ticket someone is considering. It does not
select tickets, and it cannot produce one from this repository today —
every leg needs a real price, and no odds exist yet.

The one thing it will not do is multiply leg probabilities together and
call that a parlay. That product is correct only for independent legs, and
same-game legs never are: a player's points and his team's total move
together. So `evaluate_parlay` **refuses** a ticket whose legs share a
`game_id` unless it is given a correlation matrix. Joint probability comes
from a Gaussian copula, which reduces exactly to the product at R = I, so
independence is a point in the same model rather than a separate path. R is
the **tetrachoric** correlation — the latent normal one, not the observed
correlation of the 0/1 outcomes.

It also refuses a leg on a whole line: a push voids that leg and re-prices
the whole ticket at the remaining legs' odds, which a win/lose model does
not represent.

`calibration_amplification` is the number worth reading before any of this
is used. At a 5% per-leg optimism:

| legs | believed | true | overstated by |
|---|---|---|---|
| 2 | 0.3364 | 0.3036 | 10.8% |
| 3 | 0.1951 | 0.1673 | 16.6% |
| 5 | 0.0656 | 0.0508 | 29.2% |

A parlay is where an uncalibrated model's error compounds fastest, which
makes it the worst available way to express an unproven edge rather than
the best.

## Disclaimer

Decision board output is a research ranking for your judgment — not a lock,
not live P&L, and not automated wagering. Nothing here has been shown to be
profitable, and a positive EV is a statement about the model's probability
being right, which is exactly what has not been established yet.
