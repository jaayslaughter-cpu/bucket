# Operator-supplied Sleeper board responses

`src/ingestion/sleeper_props.py` reads the **newest `*.json`** in this
directory. Sleeper publishes no readable public pick'em board — the logged-in
check recorded in `docs/pickem_props.md` on 2026-09-15 found scores and recap
at sleeper.com/nba, marketing and an app install at sleeper.com/picks, and
Player Picks app-gated — so this is the sanctioned capture path. Defeating an
app gate is not, and is banned in that document.

Drop a board response in, then check the mapping against it before trusting
it:

    python -m scripts.pull_pickem_props --describe <file>
    python -m scripts.pull_pickem_props --source sleeper --from-file <file>

`*.json` here is gitignored: these are third-party postings, and the archive
that keeps them with a timestamp is `data/external/pickem/snapshots/`.
