# Timestamped pick'em prop-line sources

Status: **RESEARCH_ONLY** — timestamped prop-line **research capture only**.
Does **not** place wagers or size bankroll.

**User approval (2026-09-15):** Sleeper, Underdog, and PrizePicks are approved as
**timestamped prop-line sources for research capture only**.

## Season timing

NBA pick’em player props are **not** expected on the boards yet as of
**2026-09-15**. Sleeper’s own public state feed reports
`season_type=off` and **`season_start_date=2026-10-20`**
(`GET https://api.sleeper.app/v1/state/nba`). Lines show up when the season
starts in **October**. Until then, successful pulls that return no rows are
**`EMPTY`** (expected) — never invent lines.

Web notes (logged-in check 2026-09-15):
- [sleeper.com/nba](https://sleeper.com/nba) = Scores / season recap
- [sleeper.com/picks](https://sleeper.com/picks) = Picks **marketing** + app install (no live board UI even when logged in)
- Live Player Picks appear to be **app-gated**; no public web odds board path found yet

| Source | Client | Live board status (pre-October) |
|--------|--------|----------------------------------|
| **Underdog** | `src/ingestion/underdog_props.py` | Endpoint works; board often **EMPTY** until October season open |
| **PrizePicks** | `src/ingestion/prizepicks_props.py` | May also **403 Cloudflare** from non-browser hosts → `DATA_NOT_AVAILABLE` |
| **Sleeper** | `src/ingestion/sleeper_props.py` | Public fantasy API has **no** pick'em board. Web: [sleeper.com/nba](https://sleeper.com/nba) = **Scores/recap** (not props); [sleeper.com/picks](https://sleeper.com/picks) = Picks marketing / app gate. Use local JSON under `data/external/pickem/sleeper/` or set `pickem.sleeper.board_url` when a stable path is known |

## Capture contract

- Schema: `src/ingestion/pickem_schema.py` → `PickemPropLine` / `PickemSnapshot`
  - Every line carries `captured_at_utc` (UTC)
  - Board pulls are keyed by that timestamp
- Batch: `src/ingestion/pickem.py` → `pull_pickem_boards()`
- Store: `src/ingestion/pickem_store.py` → `PickemResearchStore`
  - Layout: `data/external/pickem/snapshots/{source}/{YYYY}/{MM}/{source}_{YYYYMMDDTHHMMSSZ}.json`
  - Flat index: `lines_index.csv` (append-only when lines exist)
- CLI: `python scripts/pull_pickem_props.py`

## Important: not OddsPapi EV

Pick'em rows carry **lines + optional multipliers**, not verified two-way American
odds. Keep using OddsPapi for `MarketContext` EV/CLV. Do not mark pick'em rows
as `MarketContext.status=VALID` for stake math.

### Wave 4: pick'em vs book line-diff

Helper: `src/quant/line_diff.py` → `pickem_vs_book_line_diff(...)`.

- Runs **only** when the book side is OddsPapi `status=VALID` with two-way American odds.
- Returns pick'em−book line delta + a soft fair-prob adjustment for research display.
- Never treats pick'em multipliers as VALID odds; never sizes stake from this helper.

Player metadata / injuries remain on `src/ingestion/sleeper.py` (separate from props).

## Still banned

- `ODDS_API_KEY` / SportsGameOdds / other paid sportsbook APIs
- Direct HTML scrapes of sportsbook sites
- Automated wager placement on any destination
