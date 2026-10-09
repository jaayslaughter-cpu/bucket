#!/usr/bin/env bash
#
# scripts/start.sh — the container's start command: migrate, probe, then run.
#
# WHY A SCRIPT AND NOT THREE PLATFORM SETTINGS. The order matters and getting it
# wrong is silent. Migrations must be applied before the worker reads the
# schema, and the probe must run after them or it reports pending migrations it
# just watched get applied. A start command typed into a dashboard is also not
# in version control, which is where the one sequence that has to be right
# belongs.
#
# THE TWO FAILURE POLICIES ARE DIFFERENT, ON PURPOSE:
#
#   migrations  -> HARD FAIL. A schema behind the code does not announce
#                  itself; it surfaces at 09:00 PT as a failed insert, hours
#                  after the deploy looked successful. Exiting nonzero here
#                  makes the platform show a failed deploy, which is the
#                  honest signal. The run is one transaction, so a failure
#                  leaves the database on the last fully applied migration.
#
#   healthcheck -> REPORT AND CARRY ON. It is strict by design (an unseeded
#                  volume is a FAILURE there), and `scheduler_worker` has the
#                  opposite and also-correct policy: a worker that refuses to
#                  start cannot report anything, and the settlement job is
#                  still useful with no model. So its findings are logged and
#                  the worker starts anyway. Run it yourself through
#                  `railway run python -m scripts.railway_healthcheck` when you
#                  want the exit code to mean something.
#
# RESEARCH_ONLY. Nothing here places, sizes or automates a wager.

set -euo pipefail

cd "$(dirname "$0")/.."

echo "PropIQ start: migrate -> probe -> worker. RESEARCH_ONLY."

# Default ON. This script is a container entrypoint, so the DATABASE_URL in
# scope is the platform's injected one and nobody is there to pass --apply.
# `scripts/migrate_db` keeps the read-only default for the case where a human
# is typing. Set this false where migrations are applied by a separate release
# step and the worker must not touch the schema.
case "${PROPIQ_MIGRATE_ON_BOOT:-true}" in
  1|true|TRUE|True|yes|YES)
    echo "--- migrations"
    python -m scripts.run_migrations
    ;;
  *)
    echo "--- migrations skipped (PROPIQ_MIGRATE_ON_BOOT=${PROPIQ_MIGRATE_ON_BOOT})"
    ;;
esac

echo "--- deployment healthcheck (advisory; the worker starts either way)"
if ! python -m scripts.railway_healthcheck; then
  echo "HEALTHCHECK REPORTED FAILURE — the lines above say which check and why."
  echo "The worker is starting anyway: it still settles finished games, and a"
  echo "worker that will not boot cannot tell you any of this. An unseeded"
  echo "volume means every row ABSTAINS while the slate job still exits 0."
fi

echo "--- worker"
# exec so the worker is PID 1 and receives SIGTERM directly: `scheduler_worker`
# installs a handler that shuts the scheduler down AFTER the running job, and a
# shell in between would swallow the signal and let the platform SIGKILL it
# mid-slate.
exec python scheduler_worker.py
