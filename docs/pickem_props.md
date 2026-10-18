# Timestamped pick'em prop-line sources

> ## STATUS: BUILT AND MAPPED. SHIPPED SWITCHED OFF, BECAUSE IT COSTS MONEY.
>
> **Provider decision (2026-10-18):** Underdog and Sleeper boards are captured
> through the Apify Actors `zen-studio/underdog-player-props` and
> `zen-studio/sleeper-player-props`, not by scraping either site ourselves.
> The field mapping in `config/pickem_sources.yaml` is written from those
> Actors' own documented output records.
>
> **This bills per line: $0.15 per 1,000, and filtered-out lines are not
> billed.** Both sources therefore ship `enabled: false`, and
> `PickemSourceConfig.configured` reads that flag, so nothing can run and
> spend until somebody flips it deliberately. The Actor listing publishes
> per-league counts for other leagues but not NBA (MLB ~2,300/day ≈ $0.35,
> NFL ~3,900 ≈ $0.59, all 21 leagues ~11,500 ≈ $1.73), so treat a full NBA
> board as roughly $0.29–$0.59 a pull and **measure it on the first run**. The
> free tier is 5 lifetime runs at 50 results each, which is enough to verify
> the mapping for nothing.
>
> **Why a paid provider rather than a direct client.** A direct client would
> mean reading Underdog's internal endpoint ourselves, and the thing that
> makes that work from a datacentre is the bot-wall handling this document
> bans below. The Actor operates its own residential-proxy infrastructure —
> that is a service we buy, not evasion we build, the same relationship this
> project already has with PropLine. `AGENTS.md` §5 permits it under the
> pick'em carve-out ("Pick'em capture ... is timestamped research only"); it
> is not another sportsbook odds API and it is not us scraping.
>
> **What is NOT verified.** No live run has happened from this repository —
> this build container's network policy denies `api.apify.com`. The mapping
> matches the published output records and is exercised against the Underdog
> listing's full 46-field example in `tests/test_pickem_capture.py`. Sleeper's
> listing publishes no full example, so only the fields it states are shared
> with the Underdog Actor are mapped and the rest are left empty rather than
> assumed. Check both for free before trusting either:
>
>     python -m scripts.pull_pickem_props --describe /tmp/board.json
>     python -m scripts.pull_pickem_props --source underdog --from-file /tmp/board.json
>
> **An Underdog row is a two-way price, not a pick'em row.** The Actor returns
> `higher_american_odds` and `lower_american_odds` at
> `payout_multiplier: "1.0"`, and `ingestion/dfs_pricing` — the one rule this
> and `ingestion/propline` both read — does **not** flag it. The first version
> of the capture adapter did, dropping both prices and setting two flags that
> make `market_ev_gate` abstain. See that module's docstring. PrizePicks is
> still pick'em however numeric its prices look, and a boost on **either**
> side makes a row incomparable.
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
> - `src/ingestion/prizepicks_props.py` (planned) — `zen-studio/prizepicks-player-props`
>   would fit the same transport, but PropLine already delivers PrizePicks
>   lines through `DFS_BOOKS` at no per-line cost. A second reader of one board.
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
| **Underdog** | `src/ingestion/underdog_props.py` | Built and mapped from the Actor's documented 46-field record. `enabled: false` — flipping it starts billing. NBA-only input, with `statTypes` narrowed to the markets this project models, which is both a cost control and a correctness one |
| **Sleeper** | `src/ingestion/sleeper_props.py` | Built. Same transport; only the fields its listing states are shared with the Underdog Actor are mapped, the rest left empty rather than assumed. Operator-supplied JSON in `data/external/pickem/sleeper/` remains the free path and is what `--from-file` reads |
| **PrizePicks** | `src/ingestion/prizepicks_props.py` (planned) | Not written. Its lines arrive via PropLine's `DFS_BOOKS` already |

Shared by all of them: `src/ingestion/pickem_board.py` reads the bytes (Apify
run, public URL, or recorded file), `src/ingestion/pickem_schema.py` holds the
contract and the path mapper, `src/ingestion/dfs_pricing.py` decides whether a
row is a two-way price, `src/ingestion/pickem_store.py` archives each snapshot,
and `src/ingestion/pickem.py` runs the batch.

**Not wired into the slate.** `pickem.pickem_prop_line_rows()` is the seam, and
`main.ingest_prop_lines` does not call it: a step that bills per line does not
belong in an unattended slate run until somebody has watched it cost what they
expect.

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

### Which rows are actually pick'em

`src/ingestion/dfs_pricing.py` is the single reader. PrizePicks is a pick'em
board however numeric its prices look; **Underdog at payout multiplier 1.0 is
a genuine two-way price and is not flagged**; a multiplier away from 1.0 on
either side is a boost or discount and makes the row incomparable; PrizePicks
`goblin` and `demon` are different products from its `standard` line. The
paragraph above applies to rows that rule calls pick'em — not to every row
that happens to come from a DFS operator.

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
