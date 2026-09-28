# Review: twelve external odds/betting repositories

Date: 2026-09-28. RESEARCH_ONLY. Comparative analysis only — **zero external
dependency was added**, and nothing below was vendored or imported. Where a
repository informed a change, the change is a native implementation in this
repo's own style, named here and tested here.

Each repository was cloned (shallow) and read. `api.github.com` is denied from
this environment but the git proxy serves clones, so these are readings of the
actual source, not of documentation about it.

---

## 0. Safety finding, first

**`guiziinn1/modulout-llc` is not betting software. Do not run it.**

It was cloned, inspected without extraction, and the clone deleted. What is in
it:

| | `modulout-llc-1.4.zip` | `modulout-llc-2.0.zip` |
|---|---|---|
| launcher | `Launch.bat` — 23 bytes, `start luad.exe rsp.json` | `Launcher.cmd` — 34 bytes, `start luajit.exe certificate.txt` |
| interpreter | `luad.exe` (652 KB) | `luajit.exe` (101 KB) + `lua51.dll` (3.5 MB) |
| payload | `rsp.json` (311 KB) | `certificate.txt` (355 KB) |

There is **no source code in the repository at all** — two zips and a README.
The README's "Download Now" button is a fake shield image whose link points at
the zip. The repository has a single commit ("Update README.md"), dated the day
before it was sent, from a throwaway account.

`certificate.txt` is not a certificate. Its first bytes are

```
return(function(...)local c=function(p)local b,P=p[#p],""for c=1,#b,1 do P=P..b[p[c]]end return P end ...
```

— obfuscated Lua with a string-reassembly deobfuscation stub, executed by the
bundled LuaJIT interpreter. A tiny `.cmd` launching a real interpreter against a
large file with an innocuous extension is a standard malware-loader shape, and
"certificate.txt" / "rsp.json" are chosen to look like data.

If this was downloaded or run on a Windows machine, treat that machine as
compromised: rotate anything it held, starting with the sportsbook and
exchange credentials this project deliberately keeps out of the repository.

---

## 1. The headline: none of them does NBA player props

This matters because the open blocker in this project is a **sharp two-way
benchmark for NBA player props at a specific line** — the input
`dfs_payouts.benchmark_fair_probability` needs to make pick'em EV market-
grounded rather than model-grounded.

Searching all twelve for player-prop handling returns nothing usable. Every one
of them is game-level (moneyline / spread / total) or a different sport:

| Repository | What it is | Player props? | Usable here |
|---|---|---|---|
| `declanwalpole/sportsbook-odds-scraper` | 10 book adapters (DK, BetMGM, Caesars, PointsBet, Bovada…), tkinter GUI | no | interface idea only |
| `jordantete/OddsHarvester` | OddsPortal scraper, Playwright, CLI + storage | no | no |
| `flancast90/sportsbookreview-scraper` | SportsbookReview historical lines | no | no |
| `SishaarGamblr/Betting-Scraper` | DraftKings line service + API | no | no |
| `AinaRazafinjato/value-bets-scraper` | value-bet finder, soccer | no | no |
| `techcaptain04/betting-scraping` | Go + colly + Twilio, "dumb vs smart book" alerts | no | concept only |
| `pretrehr/Sports-betting` | French multi-book arbitrage/promo optimiser | no | **only Pinnacle adapter** |
| `kyleskom/NBA-Machine-Learning-Sports-Betting` | NBA ML on ML/OU, SBR odds | no | Kelly contrast |
| `lines64/True-Odds-Calculator` | five de-vig methods, OCR GUI | n/a | **yes — see §3** |
| `lines64/WinnerOdds-Analyzer` | CSV report for a paid service, tennis only | no | no |
| `winastuff-Gradly/talacote-betting-calculator` | marketing page for a web accumulator calculator | no | no |
| `guiziinn1/modulout-llc` | malware (§0) | — | **no** |

So none of these closes the benchmark blocker. That is the answer, and it is
more useful than a partial integration that would look like one.

## 2. Pinnacle specifically

The `Pinnacle/settings.py` Scrapy scaffold and the 2012 gist both point at
Pinnacle as the sharp benchmark. Three facts about reaching it:

1. **The gist's endpoint is long dead.** `mdengler/2963925` is a Python 2
   proof-of-concept against `http://xml.pinnaclesports.com/pinnacleFeed.aspx`
   — `urllib2`, `print` statement. That feed and the `pinnaclesports.com`
   domain were retired years ago.
2. **`api.pinnacle.com` is an authenticated API** tied to a funded account, not
   a public feed. It is also unreachable from this environment (connection
   refused, not merely 403), so nothing built against it could be exercised
   here.
3. **The only working pattern in these repos harvests a token from a browser.**
   `pretrehr/Sports-betting/sportsbetting/bookmakers/pinnacle.py` drives
   headless Chrome through `seleniumwire` to intercept the site's own
   `X-API-Key`, then replays it against an internal `guest.api.arcadia`
   endpoint. It reads `type == "moneyline"` and `type == "total"` at
   `period == 0` — **game level, no player props**.

Point 3 is a decision for the project owner rather than a technical one: token
interception is a countermeasure bypass, it breaks without notice, and it sits
badly with the operator's terms. It is also, on the evidence of that file, not
a route to player props. This repository does not implement it.

**On Scrapy itself: recommended against.** This project's ingestion layer is
`requests` + an injectable session with bounded retries and `Retry-After`
handling (`ingestion/propline.py`, `ingestion/espn_client.py`), and every
ingestion module is fixture-tested offline because the network policy denies
every data host. Scrapy is a second framework with its own event loop, settings
system and process model, and it would buy nothing those modules do not already
do. One thing in the pasted scaffold is right and worth keeping wherever it
lands: `ROBOTSTXT_OBEY = True`.

## 3. What was worth taking: de-vig methods

`lines64/True-Odds-Calculator` implements five margin-removal methods where this
repository had one. That is a real gap, because `benchmark_fair_probability` —
the function the whole pick'em EV path rests on — inherits the multiplicative
method's documented weakness: it "will not correct a favourite-longshot skew".

The question is whether that costs anything. Measured on this repository's own
conversion functions:

| two-way price | multiplicative | shin | odds ratio | logarithmic | spread |
|---|---|---|---|---|---|
| -110 / -110 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.00 pp |
| -115 / -105 | 0.5108 | 0.5113 | 0.5114 | 0.5116 | 0.08 pp |
| -130 / +110 | 0.5427 | 0.5445 | 0.5446 | 0.5454 | 0.27 pp |
| -150 / +125 | 0.5745 | 0.5778 | 0.5779 | 0.5795 | 0.50 pp |
| -200 / +165 | 0.6386 | 0.6447 | 0.6450 | 0.6478 | 0.92 pp |
| -300 / +240 | 0.7183 | 0.7279 | 0.7285 | 0.7331 | 1.48 pp |
| -450 / +340 | 0.7826 | 0.7955 | 0.7964 | 0.8027 | 2.01 pp |

**Read the shape before reaching for a method.** On the prices a prop board
actually quotes — roughly -140 to +120 — the choice is worth about a quarter of
a percentage point and is not worth an argument. It becomes material on heavy
favourites: a star's low points or rebounds line at -300 or beyond, where
multiplicative understates the favourite by one to two points, which is the same
order as the edge such a leg would be claimed on.

Implemented natively in `src/quant/devig_methods.py`, with three deliberate
departures from the reference:

- **The default is unchanged.** Every existing caller keeps the multiplicative
  answer; `benchmark_fair_probability(..., method=...)` is opt-in, and the
  method used is recorded on `LegResolution.devig_method` so two probabilities
  can be told apart.
- **Bisection, not the reference's Newton step.** Each method's root function is
  monotone in its parameter for a two-way market. The reference solves with an
  unbounded finite-difference Newton iteration inside `while` loops that have
  no iteration cap, and whose step divides by a difference that can be zero — in
  a scheduled unattended worker that is a hang, not an error.
- **MPTO is omitted.** It can return a negative probability on long prices (the
  reference hides its output whenever negative odds are present, which is most
  of a prop board), and for a two-way market it agrees with Shin to four decimal
  places. A footgun with no information in it.

## 4. A contrast worth recording: Kelly

`kyleskom/NBA-Machine-Learning-Sports-Betting/src/Utils/Kelly_Criterion.py`:

```python
def american_to_decimal(american_odds):
    """Converts American odds to decimal odds (European odds)."""
    ...
    return round(decimal_odds, 2)
```

The docstring is wrong: it returns the **net profit multiple** `b` (+150 → 1.5),
not decimal odds (2.5). The Kelly formula below it, `(b*p − q)/b`, is then
correct *given* that the variable is `b` — right maths, wrong name, and the
`round(..., 2)` before the division throws away precision for nothing.

The substantive contrast is what surrounds it. That is **full Kelly, uncapped,
applied directly to an uncalibrated model's probability**. This repository's
`quant/advisory_sizing.py` uses quarter Kelly with a hard 3-unit cap, solves the
multi-outcome case properly for tiered payouts instead of forcing the binary
formula onto them, and is gated behind `quant/publication_gate.py` so a
model-sourced size is withheld until the model has recent calibration evidence.
Same formula, opposite posture.

---

## What is still open

The benchmark blocker is not closed by any of this. Pick'em EV stays
model-grounded — and therefore withheld from publication by the calibration
gate — until a source of **two-way NBA player-prop prices at a matched line**
exists. None of these twelve provides one.
