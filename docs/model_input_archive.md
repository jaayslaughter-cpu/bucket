# PropIQ Timestamped Model-Input Archive

This is a research-only collection and point-in-time preparation layer, not a
change to serving. It archives official availability reports and comparable
two-sided PropLine quotes, preserves source and collection timestamps, and
exports unresolved identities and collection failures instead of filling zeros.

## What is included

- **Availability:** The requested season's official report page supplies actual
  PDF links. A geometry-based parser retains game date, matchup, team, player,
  status, reason and the original name. Eastern report times convert to UTC,
  including DST. The PDF's internal report time must match its URL.
- **Odds:** The existing NBA-only PropLine client captures player props, spreads
  and totals. Two sides must agree on bookmaker, market, player, line and market
  snapshot time. Player provider IDs remain explicitly in the provider
  namespace. Conflicting player IDs, duplicate sides, malformed prices,
  suspended markets, withdrawn selections, stale live markets and post-tip
  updates are refused.
- **Storage:** Canonical JSON snapshots and source PDFs are content-addressed.
  Atomic hard links prevent overwrites, identical replays deduplicate, snapshot
  checksums are verified on reads, and credential-like keys are refused.
  The archive contains normalized quotes and selected side metadata, not
  authenticated request objects or unfiltered response bodies.
- **Selection:** The latest complete official report replaces the older report
  globally. An old OUT row cannot survive simply because the player or team is
  absent from the newer snapshot. Per-event quote snapshots replace their
  predecessors, including explicit failures.
- **Scheduling:** An opt-in job runs every 15 minutes in the existing scheduler
  on its existing volume. It coalesces missed runs and permits one instance.
  Nothing is defined below the scheduler entrypoint.

The current-season report page describes continually updated reports and links
the PDFs used for collection. It is not presented here as a complete
season-long historical archive ([NBA official report page](https://official.nba.com/nba-injury-report-2026-27-season/)).

## Install and run

Run from the repository root. Python 3.11 or newer is required by the project;
the existing `pdfplumber` dependency is now also declared in the deploy extra.

```bash
pip install -e '.[ml,db,deploy]'

# Current season is derived at runtime; --season-start 2026 means 2026-27.
python -m scripts.collect_model_inputs collect --season-start 2026

# Free source only. Does not request odds or require a PropLine credential.
python -m scripts.collect_model_inputs collect --availability-only

# Optional standalone loop, not needed alongside the scheduler job.
python -m scripts.collect_model_inputs watch --interval-seconds 900
```

Provide `PROPLINE_API_KEY` through the existing environment/secret manager.
Never put a credential in a URL, manifest, file committed to Git, or command
argument. Without a usable key, collection archives `NOT_CONFIGURED`, returns
nonzero for a one-shot run, and does not invent quotes. This status includes a
client auth error and is not proof that a production service lacks a key.

Default archive root:

```text
<resolve_state_root()>/model_inputs/
  snapshots/availability/<sha256>.json
  snapshots/odds/<sha256>.json
  reports/<sha256>.pdf
```

`resolve_state_root()` honors `PROPIQ_STATE_DIR`, the Railway volume mount, and
the existing fallback rules. For a one-off alternative, put `--root PATH`
before the subcommand.

## Blocked index fallback

Direct Python requests to the official index returned HTTP 403 in this
environment, while direct downloads of verified PDF links succeeded. Railway
reachability has not been tested. The default index path records failure, not
a successful empty report; it does not bypass access controls.

Supply a JSON array of actual published official PDF links when the index
cannot be fetched. `config/nba_report_urls.example.json` contains links
observed on the official page during this build, not a generated date grid
([published report links](https://official.nba.com/nba-injury-report-2026-27-season/)).

```bash
python -m scripts.collect_model_inputs collect \
  --season-start 2026 \
  --report-manifest config/nba_report_urls.example.json \
  --availability-only
```

The manifest is a seed example, not a self-updating feed. An operator or a
permitted upstream collector must refresh it with newly published links.
The collector validates every entry, filters to the requested season, ignores
future links and chooses the latest eligible report. A report older than six
hours returns `PARTIAL` with `STALE_REPORT`; its true publication time is
unchanged, and a six-hour export omits it. No missing report means “healthy.”

## Historical imports

Backfill only source material you actually have or can access legitimately.
No endpoint guesses, fabricated source dates, licensed-source scraping, or
additional paid odds providers are introduced.

```bash
# An exact published official URL, downloaded now:
python -m scripts.collect_model_inputs import-report \
  --url https://ak-static.cms.nba.com/referee/injury/Injury-Report_2026-11-08_03_00PM.pdf

# Or the corresponding already-downloaded PDF:
python -m scripts.collect_model_inputs import-report \
  --url https://ak-static.cms.nba.com/referee/injury/Injury-Report_2026-11-08_03_00PM.pdf \
  --pdf /path/to/the-same-report.pdf

# A saved PropLine event/bookmaker/market/outcome payload:
python -m scripts.collect_model_inputs import-odds --json /path/to/propline-event.json
```

The example PDF was downloaded and parsed in this environment
([verified November 8 report](https://ak-static.cms.nba.com/referee/injury/Injury-Report_2026-11-08_03_00PM.pdf)).
The importer verifies its internal timestamp against its URL. It does not
authenticate the provenance of an operator-supplied local file.

The odds importer accepts only the existing documented event-shaped schema,
including `sport_key`, `id`, `commence_time`, `bookmakers`, `markets`,
`last_update` and two-sided `outcomes`. It does not guess the shape of an
arbitrary `/odds/history` response. Existing provider-history adapters can
produce an explicit event-shaped payload; availability of any history still
depends on the provider, plan and actual returned data.

Imports retain actual collection time now, even for an older publication.
Historical imports waive only the live quote-age limit, not source timestamps,
two-sided pairing, finite prices, withdrawal checks or pregame checks.
Neither live capture nor import calls a first observed spread “opening.”

## Point-in-time export

```bash
python -m scripts.collect_model_inputs export \
  --cutoff 2026-11-13T06:45:00Z \
  --mode system \
  --max-age-hours 6 \
  --games /path/to/game-crosswalk.json \
  --players /path/to/player-crosswalk.json \
  --output /path/to/new-asof-export.json
```

Output files are created exclusively; existing files are not overwritten.
Omit the crosswalk arguments to inspect raw coverage and unresolved rows.

- **System mode:** Both publication and actual collection must be at or before
  the prediction cutoff. An imported old report cannot pretend the system
  possessed it in the past.
- **Published mode:** Uses source publication time for a counterfactual
  historical replay. The result says `selection_mode: published` and retains
  collection times. This is hindsight-assisted availability, not evidence
  of what the running system knew or that an old URL's contents never changed.
- **Freshness:** The default maximum age is six hours for exported reports
  and odds. Live quote normalization separately requires a market update
  within 30 minutes of collection. Neither threshold is a model finding.
- **Pregame odds:** Both the quote update and prediction cutoff must precede
  the event start. Outcome last-change times can differ because the shared
  market snapshot, not coincident last changes, establishes coexistence.
- **Failures:** A failed latest report stays failed instead of falling back
  to older OUT/AVAILABLE rows. Failure and not-submitted records remain
  separate from resolved player rows. No zero absence count is synthesized.

Use each prediction's real timestamp. A single today's export is not a
historical training panel and cannot safely be joined to every past game.

## Explicit identity crosswalks

The report carries names and teams, not the panel's player/game IDs. Supply
reviewed mappings derived from your real roster, schedule and panel. Examples
below use placeholders, not real identities or event mappings.

Game mapping JSON, one entry per game/team:

```json
[
  {
    "game_date": "2026-11-13",
    "matchup": "AAA@BBB",
    "team": "Exact Official Team Name",
    "game_id": "YOUR_EXISTING_PANEL_GAME_ID",
    "team_abbreviation": "AAA",
    "provider_event_id": "ACTUAL_PROPLINE_EVENT_ID_IF_KNOWN"
  }
]
```

Player mapping JSON:

```json
[
  {
    "team": "Exact Official Team Name",
    "player_name": "Exact Player Name",
    "player_id": "YOUR_EXISTING_PANEL_PLAYER_ID"
  }
]
```

Names use the existing deterministic Unicode/punctuation normalization;
suffixes remain significant. No fuzzy matching is introduced. Odds mappings
require a unique provider-event-to-panel-game mapping and a unique player
within that game's mapped teams. Ambiguous matches stay unresolved. Both
participating teams should be represented in the game map.

Copy IDs exactly from a single panel namespace. ESPN identifiers retain the
`espn:` prefix; a numeric provider ID is not automatically an NBA player ID.
The code never silently strips a provider namespace into a panel identity.
Crosswalk correctness remains an operator responsibility.

Exports contain:

- **Resolved availability:** `GAME_ID`, `PLAYER_ID`, team, date, status, reason
  and report observation time.
- **Resolved odds:** Canonical game/player IDs alongside the original provider
  identifiers, paired prices, line and both timestamps. Game-level rows have
  no player ID.
- **Unresolved data:** Original facts plus missing/ambiguous-identity reasons,
  including team-not-submitted records.
- **Coverage:** Status counts, resolution counts, paired quote counts and
  failure counts. `training_ready` is deliberately always false.

This layer does not write `PROPIQ_INACTIVE_CACHE_DIR`, populate DB market tables,
derive vacated usage, treat coach-decision DNPs as injuries, generate workbook
data, or attach these features to live scoring. In particular, a partially
mapped team is not a complete absence list.

## Railway enablement

Merge only after reviewing the patch and testing the container in staging.
No Railway service or production setting was changed by this build.

1. Build the normal image, which now includes `pdfplumber` in its deploy extra.
2. Confirm the existing worker can write to its mounted persistent state root.
3. Supply the existing `PROPLINE_API_KEY` securely if collecting odds.
4. Verify source reachability from that container with a one-shot run.
5. If the index is blocked, maintain a permitted manifest on the volume and set
   `PROPIQ_MODEL_INPUTS_REPORT_MANIFEST=/app/data/nba_report_urls.json`.
6. Set `PROPIQ_MODEL_INPUTS_ENABLED=1` only when ready for repeated API usage.
7. Redeploy the existing worker and confirm the `model_inputs` job and snapshot
   statuses. Disable the flag to stop future scheduled polls.

Do not also run the standalone watch loop against the same volume. Multiple
replicas need an external single-writer arrangement; `max_instances=1` only
protects one scheduler process. Atomic storage prevents overwrites, not duplicate
paid requests. The collector caps each pass at 12 upcoming events within 36
hours and uses the client's existing quota checks, but account limits and
costs still need operator review.

Archive reads currently scan snapshot files. Before a season-scale archive,
measure export latency and disk usage, add an indexed manifest if warranted,
and adopt an explicit retention/backup policy. No automatic deletion is
introduced here.

## Model admission

This collection work does not establish any accuracy improvement. Before
model admission, backfill enough real source data, resolve identities against
the actual panel, review per-team and per-game missingness, and build
prediction-cutoff-specific candidate inputs. Do not fill unknown coverage with
zero absences, neutral spreads, invented prices or assumed healthy players.

Then evaluate the candidate through `scripts/feature_ab.py` on chronological,
season-aware folds and identical rows against the existing baseline. Any new
grouped rolling feature must use `.shift(1)` inside player-season groups.
Serving remains unchanged until a measured, reviewed result supports wiring.
