# Timestamped pick'em prop-line sources

> ## STATUS: NOT BUILT. This is a DESIGN, not a description.
>
> **Verified against the tree on 2026-10-18, and against the full git history:
> none of the capture modules below has ever been committed.** `git log --all
> --diff-filter=A` finds no `pickem`, `underdog`, `prizepicks` or `sleeper`
> module at any commit. There is nothing to run, nothing to fix, and nothing
> that can be failing.
>
> This document previously read in the present tense — a status table saying
> Underdog's "Endpoint works" and PrizePicks "May also 403 Cloudflare" — and
> on 2026-10-18 that produced a bug report about four broken scrapers, three
> of which do not exist. Same defect as the one commit `6d9a130` fixed by
> deleting OddsPapi: *a source precedence with no client*. The design is kept
> because the approval below is real and the plan is still the right one; the
> tense is not.
>
> **What DOES exist** is the consumer side: `src/quant/line_diff.py`
> (`pickem_vs_book_line_diff`), called by `src/quant/paper_research.py`. It
> has no pick'em rows to read and will not until the capture layer below is
> written.
>
> `tests/test_pickem_docs_match_the_tree.py` now fails if this document names
> a module that is not in the tree and not marked planned.

Status: timestamped prop-line **capture**. This document is about ingesting
lines, not about acting on them. Nothing in the capture path places a wager;
recommendations and sizing live in `quant.decision_board`, `quant.dfs_entry` and
`quant.advisory_sizing`.

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

| Source | Client (PLANNED — none of these exist) | What was expected of it |
|--------|----------------------------------------|-------------------------|
| **Underdog** | `src/ingestion/underdog_props.py` (planned) | Board often **EMPTY** until the October season open |
| **PrizePicks** | `src/ingestion/prizepicks_props.py` (planned) | Expected to be bot-walled from non-browser hosts. **We do not bypass bot protection** — see "Still banned" |
| **Sleeper** | `src/ingestion/sleeper_props.py` (planned) | Public fantasy API has **no** pick'em board. Live Player Picks appear app-gated; no public web odds board path was found. A capture path here would be operator-supplied JSON under `data/external/pickem/sleeper/`, not a scrape |

Every path in the capture contract below is likewise **planned**:

- `src/ingestion/pickem_schema.py` (planned)
- `src/ingestion/pickem.py` (planned)
- `src/ingestion/pickem_store.py` (planned)
- `scripts/pull_pickem_props.py` (planned)
- `src/ingestion/sleeper.py` (planned)

## Capture contract (PLANNED — none of these exist)

- Schema: `src/ingestion/pickem_schema.py` (planned) → `PickemPropLine` / `PickemSnapshot`
  - Every line carries `captured_at_utc` (UTC)
  - Board pulls are keyed by that timestamp
- Batch: `src/ingestion/pickem.py` → `pull_pickem_boards()` (planned)
- Store: `src/ingestion/pickem_store.py` → `PickemResearchStore` (planned)
  - Layout (planned): `data/external/pickem/snapshots/{source}/{YYYY}/{MM}/{source}_{YYYYMMDDTHHMMSSZ}.json`
  - Flat index: `lines_index.csv` (append-only when lines exist)
- CLI: `scripts/pull_pickem_props.py` (planned)

## Important: a pick'em multiplier is not EV

Pick'em rows carry **lines + optional multipliers**, not verified two-way
American odds. EV and CLV need a two-way price, which only a sportsbook feed
posts — PropLine here. Do not mark pick'em rows as
`MarketContext.status=VALID` for stake math.

### Wave 4: pick'em vs book line-diff

Helper: `src/quant/line_diff.py` → `pickem_vs_book_line_diff(...)`.

- Runs **only** when the book side is `status=VALID` with two-way American odds. Source-neutral: it asks the EV gate, not the vendor's name.
- Returns pick'em−book line delta + a soft fair-prob adjustment for research display.
- Never treats pick'em multipliers as VALID odds; never sizes stake from this helper.

Player metadata / injuries were to live on `src/ingestion/sleeper.py` (planned;
separate from props). Injury data today comes from ESPN —
`src/ingestion/espn_availability.py`.

## Still banned

- `ODDS_API_KEY` / SportsGameOdds / other paid sportsbook APIs
- Direct HTML scrapes of sportsbook sites
- Automated wager placement on any destination
- **Circumventing bot protection.** TLS-impersonation libraries (`curl_cffi`,
  `tls-client`), headless-browser fingerprint spoofing, residential-proxy
  rotation or captcha solving, used to get past a Cloudflare / Akamai /
  DataDome wall. A 403 from an edge guard is the operator declining to serve
  an automated client, and the approval recorded above is to CAPTURE a public
  board, not to defeat an access control. Where a board is not publicly
  readable the sanctioned paths are the provider's own documented API, or
  operator-supplied JSON dropped under `data/external/pickem/` (planned).
