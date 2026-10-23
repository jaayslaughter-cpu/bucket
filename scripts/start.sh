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

# --- THE VOLUME'S OWNER, FIXED AS ROOT, AND ROOT THEN GIVEN UP ---------------
#
# WHY THIS BLOCK EXISTS. A Railway volume is mounted by the platform as root
# (docs.railway.com/volumes#permissions), and a mount REPLACES the directory
# the Dockerfile created and chowned -- so /app/data arrives root-owned however
# carefully the image prepared it. This container's `USER propiq` is uid 10001,
# so the first write fails with EACCES, and because every writer here catches
# broadly the symptom is "calibration report failed" nightly rather than a
# permissions problem. MEASURED on this deployment, 2026-10-23 18:05 UTC:
#
#   [FAIL] state_root_writable: /app/data EXISTS AND THE WRITE WAS REFUSED
#   for uid 10001 ... Permission denied: '/app/data/.propiq_write_probe.1'
#
# Railway's own documented remedy is RAILWAY_RUN_UID=0, which starts the
# container as root. That ALONE would leave the worker running as root for the
# rest of the deployment's life, which is exactly the blast radius the
# Dockerfile's `USER propiq` exists to avoid. So root is used for one thing --
# taking ownership of the mount -- and given up before anything else runs.
#
# THE chown PERSISTS, because it is written to the volume and not to the
# container layer: a later deploy finds the state root already owned by 10001
# and the chown is a no-op. The ownership outlives RAILWAY_RUN_UID itself, so
# removing that variable later costs the drop, not the fix.
#
# IT RUNS BEFORE THE ARTIFACT FETCH AND THE PROBE, both of which write to the
# state root. Fixing ownership after them would report a fault that had just
# been repaired, and the fetch would have already failed for want of it.
#
# WHERE the state root is comes from `resolve_state_root` -- the one resolver
# the slate, the probe and the artifact store already share. Hardcoding
# /app/data here would be a second reader of that fact, and it would silently
# skip the chown on any deployment whose volume is mounted elsewhere.
#
# `setpriv` RATHER THAN `su` OR `runuser` because it EXECS instead of forking:
# those two leave a parent as PID 1 that would swallow the SIGTERM
# `scheduler_worker` installs a handler for, and the worker would be SIGKILLed
# mid-slate -- the exact failure the `exec` at the bottom of this file exists
# to prevent. `--clear-groups`, NOT `--init-groups`: the latter requires a
# passwd entry for the uid and fails outright without one ("uid 10001 not
# found", measured), while propiq has no supplementary groups to initialise.
#
# THE DROP IS PROVEN BEFORE IT IS RELIED ON. The same invocation is tried on
# `true` first, so a setpriv that is absent or refused is REPORTED and the boot
# continues, rather than `set -e` killing a container that would otherwise have
# run perfectly well as root.
#
# A SENTINEL GUARDS AGAINST A BOOT LOOP: if the drop somehow left this process
# as root, the re-executed script must not try again forever.
PROPIQ_RUN_UID=10001
PROPIQ_RUN_GID=10001

if [ "$(id -u)" -eq 0 ] \
   && [ -z "${PROPIQ_PRIVS_DROPPED:-}" ] \
   && [ "${PROPIQ_DROP_PRIVS:-true}" != "false" ]; then
  echo "--- running as root: taking the state root, then dropping privileges"

  state_root="$(python -c 'from src.utils.volume import resolve_state_root; print(resolve_state_root()[0])' 2>/dev/null || true)"

  if [ -n "${state_root}" ] && [ -d "${state_root}" ]; then
    if chown -R "${PROPIQ_RUN_UID}:${PROPIQ_RUN_GID}" "${state_root}"; then
      echo "    chowned ${state_root} to ${PROPIQ_RUN_UID}:${PROPIQ_RUN_GID}"
    else
      echo "    chown of ${state_root} FAILED -- the probe below will say so."
    fi
  else
    echo "    no state root directory to chown (${state_root:-unresolved})."
  fi

  if setpriv --reuid="${PROPIQ_RUN_UID}" --regid="${PROPIQ_RUN_GID}" \
             --clear-groups true 2>/dev/null; then
    echo "    dropping to uid ${PROPIQ_RUN_UID} and re-running this script"
    export PROPIQ_PRIVS_DROPPED=1
    exec setpriv --reuid="${PROPIQ_RUN_UID}" --regid="${PROPIQ_RUN_GID}" \
                 --clear-groups "$0" "$@"
  fi

  echo "    WARNING: setpriv is unavailable or refused the drop, so THE WORKER"
  echo "    IS CONTINUING AS ROOT. The state root's ownership is fixed either"
  echo "    way; this is a blast-radius regression, not a functional one."
fi

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

# BEFORE THE PROBE, so the probe reports what the volume has after the fetch
# rather than what it had before one. Advisory like the probe: an unconfigured
# store exits 0 saying so, and a store that cannot be reached leaves the volume
# exactly as it was -- which is a deployment that seeds by hand, not a broken
# one. The worker's own boot check then says whether anything is there.
echo "--- artifact store (advisory; no-op when unconfigured)"
if ! python -m scripts.fetch_artifacts --pull; then
  echo "ARTIFACT FETCH REPORTED A PROBLEM — the lines above say which market."
  echo "The volume is unchanged for those; they will ABSTAIN until an artifact"
  echo "is there. Boot continues: settlement still runs, and a worker that will"
  echo "not start cannot tell you any of this."
fi

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
