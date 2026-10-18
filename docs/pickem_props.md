# Timestamped pick'em prop-line sources

> ## STATUS: BUILT, SHIPPED UNCONFIGURED.
>
> The Underdog and Sleeper capture clients were written on 2026-10-18. What
> this document described for a year, and did not have, now exists:
> `src/ingestion/pickem_schema.py`, `src/ingestion/pickem_board.py`,
> `src/ingestion/pickem_store.py`, `src/ingestion/pickem.py`,
> `src/ingestion/underdog_props.py`, `src/ingestion/sleeper_props.py`,
> `scripts/pull_pickem_props.py`, `config/pickem_sources.yaml`.
>
> **They ship with no board URLs and no field paths, on purpose.** Nobody
> working on this repository has seen a live response from either board: the
> build container's network policy denies both hosts, so no payload shape here
> is anybody's recollection. Each source reports `NOT_CONFIGURED` — never an
> empty board — and names what to fill in. The three steps are in the header of
> `config/pickem_sources.yaml` and start with recording one response and
> running `python -m scripts.pull_pickem_props --describe <file>`, which prints
> that response's real paths.
>
> **History, kept because the lesson was expensive.** Until 2026-10-18 this
> document described the whole subsystem in the present tense — a status table
> saying Underdog's "Endpoint works" and PrizePicks "May also 403 Cloudflare" —
> while `git log --all --diff-filter=A` showed none of it had ever been
> committed. That produced a bug report about four broken scrapers, three of
> which did not exist. Same defect as the one commit `6d9a130` fixed by
> deleting OddsPapi: *a source precedence with no client*.
> `tests/test_pickem_docs_match_the_tree.py` now fails if this document names a
> path that is neither in the tree nor marked planned.
>
> Still not written, deliberately:
>
> - `src/ingestion/prizepicks_props.py` (planned) — bot-walled, and getting
>   past one is banned below.
> - `src/ingestion/sleeper.py` (planned) — player metadata. Injuries come from
>   ESPN via `src/ingestion/espn_availability.py`, so nothing waits on it.

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

| Source | Client | State |
|--------|--------|-------|
| **Underdog** | `src/ingestion/underdog_props.py` | Built. Unconfigured: set `board_url` (or `$PROPIQ_UNDERDOG_BOARD_URL`) and the field paths. Underdog lines already reach the pipeline via PropLine's `DFS_BOOKS`; this captures the board on our own clock, which that route cannot give |
| **Sleeper** | `src/ingestion/sleeper_props.py` | Built. Reads **operator-supplied JSON** from `data/external/pickem/sleeper/`, because the 2026-09-15 logged-in check found no readable public board: sleeper.com/nba is scores and recap, sleeper.com/picks is marketing and an app install, Player Picks app-gated. If a public path is ever found, set `board_url` and the same client fetches it with no code change |
| **PrizePicks** | `src/ingestion/prizepicks_props.py` (planned) | Not written. Its board was expected behind a bot wall; see "Still banned" |

Shared by both: `src/ingestion/pickem_board.py` reads the bytes (wire or file),
`src/ingestion/pickem_schema.py` holds the contract and the path mapper,
`src/ingestion/pickem_store.py` archives each snapshot, and
`src/ingestion/pickem.py` runs the batch.

**Not wired into the slate.** `pickem.pickem_prop_line_rows()` is the seam, and
`main.ingest_prop_lines` does not call it yet: with every source unconfigured it
would add a step that can only report nothing.

## Capture contract

- Schema: `src/ingestion/pickem_schema.py` → `PickemPropLine` / `PickemSnapshot` → `PickemPropLine` / `PickemSnapshot`
  - Every line carries `captured_at_utc` (UTC)
  - Board pulls are keyed by that timestamp
- Batch: `src/ingestion/pickem.py` → `pull_pickem_boards()`
- Store: `src/ingestion/pickem_store.py` → `PickemResearchStore`
  - Layout: `data/external/pickem/snapshots/{source}/{YYYY}/{MM}/{source}_{YYYYMMDDTHHMMSSZ}.json`
  - Flat index: `lines_index.csv` (append-only when lines exist)
- CLI: `scripts/pull_pickem_props.py`

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
  operator-supplied JSON dropped under `data/external/pickem/`, which is how
  `src/ingestion/sleeper_props.py` reads Sleeper.
