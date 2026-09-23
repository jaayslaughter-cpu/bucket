# External feature harvest (RESEARCH_ONLY)

**RESEARCH_ONLY · harvest-only · do not merge foreign code.** Ideas below are for later implementation on PropIQ data with `.shift(1)` on grouped rollings. Markets: PropLine primary (never invent lines). Pick’em capture is research display only — not VALID EV.

**Sports-EV-Bot (DevanshDaxini) remains the primary feature cookbook.** Other repos add complementary form/matchup/UI/lifecycle patterns only.

**None of these repos ship multi-season sportsbook player-prop archives.** Training actuals = PropIQ boxes/PBP; market lines = PropLine / BigDataBall (team) / approved pick’em capture. Do not treat foreign CSVs or graded ledgers as PropLine-VALID market panels.

Sources covered: Sports-EV-Bot · VinceDiR · m4nr44j · heetshah · PrizePicks/PropBet · prop-scout (adibhar) · BangLetsGetIt NBA-NCAA-Betting-Models · CourtSide (eval notes).

---

## By theme (source → idea → PropIQ layer)

### Rolling form, usage, schedule density

| Idea | Source | Note | Likely layer |
|------|--------|------|--------------|
| `.shift(1)` rollings; L5/L10 medians | Sports-EV-Bot (DevanshDaxini) | **Primary cookbook** — leakage-safe form stats | `src/features` |
| Streak / consistency | Sports-EV-Bot | Hit streaks, variance of recent outcomes | `src/features` |
| EXP_POSS, usage vacuum | Sports-EV-Bot | Pace/usage context when teammates out | `src/features` |
| Schedule density: B2B, 4-in-6, GAMES_7D, DAYS_REST | Sports-EV-Bot | Fatigue / rest density | `src/features` |
| Chronological split + recency weights | Sports-EV-Bot | Train/val method, not a column | `src/models` |
| Walk-forward edge thresholds | Sports-EV-Bot | Method for deciding when edge is actionable | decision board / `src/models` |
| B2B flag | prop-scout (adibhar) | Same-day / consecutive-game fatigue bit | `src/features` |
| `shift().rolling` L5 means | prop-scout (adibhar) | Confirms Sports-EV-Bot leakage pattern; keep shift-before-roll | `src/features` |
| Same-opponent L5 (date-filtered) | prop-scout (adibhar) | Prior games vs same opp only, strictly before game-T | `src/features` |
| HOME indicator | prop-scout (adibhar) | Venue context | `src/features` |
| ORTG / DRTG context (as-of only) | prop-scout (adibhar) | Team offense/defense ratings known before tip — never post-game | `src/features` |

### Volume, efficiency, screening, grading

| Idea | Source | Note | Likely layer |
|------|--------|------|--------------|
| days_rest | VinceDiR Prop_Betting_Regression | Rest days before game | `src/features` |
| Rolling volume (FGA/FTA/2PA/3PA) | VinceDiR | Shifted shot-attempt rates | `src/features` |
| TS% / GmSc / BPM | VinceDiR | Efficiency / production proxies | `src/features` |
| LassoCV feature screening | VinceDiR | Model selection method | `src/models` |
| Fuzzy name match → rapidfuzz | VinceDiR | Align external names (PropIQ already prefers rapidfuzz) | ingestion / matching |
| Hit-rate vs line grading | VinceDiR | Post-hoc research metric vs posted line | decision board / research reports |

### Comps, injuries, aggregates, consistency

| Idea | Source | Note | Likely layer |
|------|--------|------|--------------|
| Opponent-similar games | m4nr44j nba-betting | Comp set by opponent profile | `src/features` |
| Minutes-scaled comps | m4nr44j | Scale counting stats to expected MIN | `src/features` |
| Injury missing PTS/REB/AST by position | m4nr44j | Vacuum from out teammates by pos | `src/features` |
| Teammate / opp G-F-C aggregates | m4nr44j | Frontcourt/backcourt context | `src/features` |
| Consistency CV | m4nr44j | Coefficient of variation on recent form | `src/features` |

### Matchup context + probabilistic over

| Idea | Source | Note | Likely layer |
|------|--------|------|--------------|
| Rest / home / opp-allowed | heetshah nba-datamgmt | Context + defense-allowed rates | `src/features` |
| Normal CDF P(over) with calibrated σ | heetshah | Calibrate σ from residuals — **not** hardcoded | `src/models` (prob); board displays only |

### Hit-rate display / prop cards (UI research)

| Idea | Source | Note | Likely layer |
|------|--------|------|--------------|
| L5/L10/L15 hit-rate vs posted line | PrizePicks cheat-sheet / PropBet | Display only; needs real posted line | decision board UI |
| Prop card actual-vs-line | PrizePicks cheat-sheet / PropBet | Research card layout | decision board UI |
| UI prop cards (form + line + context) | prop-scout (adibhar) | Card layout for research board; lines must come from PropLine / approved capture | decision board UI |

### Market routing + pick lifecycle (patterns only)

| Idea | Source | Note | Likely layer |
|------|--------|------|--------------|
| Book priority / fallback pattern | BangLetsGetIt NBA-NCAA-Betting-Models | Prefer primary book, fall back when quote missing — map to PropLine book preference, not Odds API | market context / PropLine client |
| Pick track → grade → ledger lifecycle | BangLetsGetIt / CourtSide eval | MANUAL_ONLY paper: log pick, grade vs result, append ledger | Wave 3 paper research / decision board |
| Shared props engine concept | BangLetsGetIt | One prop pipeline reused across sports — concept only; PropIQ stays NBA-first modular | architecture note |

### Reference-only / weak labels (do not train as markets)

| Item | Source | PropIQ stance |
|------|--------|---------------|
| Limited Brunson PTS+line CSV | prop-scout (adibhar) | **Reference-only** demo panel — not PropLine-VALID; do not use as training market lines or EV |
| Graded JSON pick ledgers | BangLetsGetIt / CourtSide | **Weak labels** for paper-research grading UX — not multi-season training panels; not VALID odds archives |

---

## Not features / skip

Keep this list strong. Do not invent, require, or port:

| Skip | Why |
|------|-----|
| Odds API / `ODDS_API_KEY` | Banned sportsbook source (BangLetsGetIt / CourtSide / others use it — skip entirely) |
| Synthetic odds (e.g. PrizePicks +100, boosted UD) | Not VALID two-way American odds / EV |
| Invented / AI-score EV | Fabricated edge; quant abstains without VALID PropLine two-way odds |
| Kelly / auto-bet / stake sizing from model | RESEARCH_ONLY — no bankroll or automated wager placement |
| BettingPros / BR / RealGM / sportsbook HTML scrapers | Non-approved HTML scrapers |
| Projection-as-feature | Leakage / circular; projections are outputs, not inputs |
| Same-game W/L (or final score) as features | Outcome leakage into game-T rows |
| Pick’em as VALID EV | Capture/research only; quant abstains without VALID PropLine two-way odds |
| Foreign graded ledgers as training panels | Weak labels / UX only — not multi-season sportsbook prop archives |
| Brunson (or any single-player) demo line CSV as VALID market | Reference-only; PropLine / approved capture for lines |

---

## Layer map (quick)

| Layer | What belongs here from harvest |
|-------|--------------------------------|
| `src/features` | Shifted rollings/medians (Sports-EV-Bot primary + prop-scout L5), B2B/HOME, same-opp L5 date-filtered, as-of ORTG/DRTG, rest/schedule density, volume/efficiency, usage vacuum, comps, injury-by-position, G-F-C aggregates, consistency CV, rest/home/opp-allowed |
| `src/models` | Recency weights, chronological/walk-forward splits, LassoCV screening, calibrated-σ Normal CDF P(over), edge-threshold methods |
| Market context | Book priority/fallback **pattern** only — PropLine primary path; never Odds API |
| Decision board UI | Hit-rate vs line, actual-vs-line prop cards, pick track→grade→ledger (MANUAL_ONLY), walk-forward edge flags — research/paper only |

Implement later on PropIQ contracts only; no foreign code merges.

---

## PropIQ status of each idea (added by audit, 2026-09-23)

Harvest notes only — no foreign code is imported and nothing here is a
dependency. Where an idea already exists in PropIQ, the module is named.

| Idea | Status in PropIQ |
|---|---|
| `.shift(1)` rollings, L5/L10 | **done** — `src/features/builder.py`, the core contract |
| Streak / consistency | **done** — `src/features/sports_ev_features.py` (built, unused by any feature list) |
| Usage proxy / usage vacuum | **partial** — `USAGE_PROXY_L10` built, unused; teammate cascade is a stub |
| Schedule density: B2B, rest, travel | **done** — `src/features/schedule.py`, and among the top features |
| Chronological split + recency weights | **done** / **orphaned** — split yes; `src/models/recency.py` is not wired |
| B2B flag, HOME | **done** — `IS_B2B_FIRST` ranks top-10 for all three markets |
| Same-opponent L5 (date-filtered) | **absent** — genuinely missing |
| ORTG / DRTG as-of | **done** — `src/features/defense.py`, per 100 possessions |
| days_rest | **done** — top-6 feature for PTS, REB and AST |
| Rolling volume (FGA/FTA/3PA) | **done** — pbp shot mix, `src/features/pbp.py` |
| TS% | **done** — `src/features/scoring_efficiency.py` |
| LassoCV screening | **superseded** — permutation importance, `scripts/feature_selection.py` |
| Hit-rate vs line grading | **done** — decision board |
| Opponent-similar games / comps | **absent** |
| Minutes-scaled comps | **absent** — `src/features/minutes_weighted.py` exists, unwired |
| Injury vacuum by position | **absent** — no injury source ingested |
| Consistency CV | **absent** |
| Normal CDF P(over) with fitted σ | **done** — `src/models/residuals.py` fits the family by held-out log-likelihood |
| Book priority / fallback | **done** — PropLine primary, oddspapi fallback |
| Pick → grade → ledger | **done** — `src/quant/historical_store.py`, MANUAL_ONLY |

The "skip" list is already honoured: no Odds API, no synthetic odds, no Kelly
or stake sizing, no projection-as-feature, and `POSTGAME_ONLY_COLS` refuses
same-game outcomes in any feature list — verified by the audit's check 7.

**The largest genuine gap** is not on this list: 87 of the 147 numeric columns
the builder produces are read by no model. Several harvest ideas marked
"done" above are done *and unused*.
