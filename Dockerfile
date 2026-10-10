# PropIQ Analytics — the deployed worker. RESEARCH_ONLY.
#
# Builds the scheduler, not a web service. It binds NO PORT: a Railway service
# configured as a web app would be marked unhealthy for never listening, so
# deploy this as a WORKER service. That is a configuration choice outside this
# file and the most likely way a first deploy goes wrong.
#
# NO SECRETS ARE BAKED IN. There is no COPY of .env, no ARG for a key, and no
# default credential. DATABASE_URL, PROPLINE_API_KEY and DISCORD_WEBHOOK_URL are
# supplied by the platform at run time; see .env.example for the full list. A
# key in an image layer survives every later layer that deletes it.
#
# BUILT AND SMOKE-TESTED 2026-10-10, after months of being reasoned about
# rather than run. `python -m scripts.validate_docker --skip-build` passes all
# 15 checks against the built image, including the two that exist to
# demonstrate the volume-permission trap documented at the bottom of this file:
# a mode-0555 mount at /app/data makes check_state_dir() report it unwritable,
# and a writable one is not false-alarmed. The container was then booted and
# ran `scripts/start.sh` end to end -- migrate, probe, scheduler up in
# America/Los_Angeles with both cron jobs and their misfire graces.
#
# ONE CAVEAT, STATED PRECISELY. The session that built it routes HTTPS through
# a CA-re-terminating proxy, so pip inside a build container sees a self-signed
# chain for pypi.org. Baking that CA into this file would be wrong -- Railway
# has no such proxy -- so the build used a copy of this file with exactly two
# extra instructions before the pip layer (`COPY` the CA, `ENV PIP_CERT`).
# Every other instruction, this file's nine stages included, is byte-identical
# to what was built. The first build on a host with ordinary egress will
# exercise the unmodified file; nothing here is now unverified by inspection
# alone, but that one difference is real.
#
# THE apt LAYER IS GONE and the comment that justified it was wrong; see below.
#
# THE ML EXTRA IS INSTALLED ON PURPOSE. catboost and xgboost are optional in
# pyproject so a local checkout stays light, and a minimal image would start
# cleanly and then fail at the first inference — late, unattended, and after the
# slate had already been ingested. Failing at build time is the better trade.

FROM python:3.11-slim

# - PYTHONDONTWRITEBYTECODE: the filesystem is ephemeral; .pyc files are waste.
# - PYTHONUNBUFFERED: without it, logs sit in a buffer and a crash loses the
#   lines that would explain it.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# SET EXPLICITLY so the container's default is a decision rather than the
# host's. Every slate cutoff in this project is a Pacific CALENDAR DAY and the
# scheduler passes America/Los_Angeles to APScheduler directly, so none of the
# timing logic depends on this value — it governs the naive datetime.now()
# calls elsewhere, which should be UTC and reproducible rather than inherited.
ENV TZ=Etc/UTC

# NO apt LAYER. This file had one, installing libgomp1, with the comment
# "OpenMP, which xgboost and catboost link against. Without it the image builds
# and then fails at import". That was reasoned from the libraries' linkage and
# it is FALSE for the wheels this project installs, which was established the
# first time the image was actually built (2026-10-10):
#
#   $ docker run --rm python:3.11-slim sh -c 'pip install xgboost catboost; \
#       python -c "import xgboost, catboost; print(xgboost.__version__)"'
#   xgboost 3.2.0
#   catboost 1.2.10
#   /usr/local/lib/python3.11/site-packages/xgboost.libs/libgomp-e985bcbb.so.1.0.0
#
# The WHEEL vendors its own libgomp. The apt layer installed a second copy of a
# library nothing loaded, and it was the only thing in this build that needed
# the Debian package index -- so on a host whose egress policy does not allow
# deb.debian.org the build failed at step 2 of 8 for a dependency the image
# does not have.
#
# THE GUARANTEE MOVED RATHER THAN DISAPPEARING. Vendoring is a property of the
# wheel, not of this project: a future xgboost or catboost could stop doing it,
# or a platform with no wheel could build from source and need system OpenMP.
# The import check below asserts at BUILD time what the apt layer only assumed,
# which is the thing the old comment actually wanted.

WORKDIR /app

# Dependency metadata first so a code change does not reinstall the world.
COPY pyproject.toml README.md ./
COPY src/__init__.py src/__init__.py
RUN pip install --no-cache-dir -e ".[ml,db,deploy]"

# FAIL HERE, NOT AT THE FIRST INFERENCE. Importing xgboost and catboost is what
# resolves OpenMP -- from the wheel's own vendored copy, or from the system if a
# future wheel stops vendoring one. An image that starts cleanly and then dies
# unattended at 09:00 PT, after the slate has already been ingested, is the
# failure this whole block exists to prevent; `python -c "import ..."` costs one
# layer and converts it into a red build.
#
# It also catches a build that resolved the ML extra to nothing, which is the
# other way this image can look fine and score nothing.
RUN python -c "import xgboost, catboost, sklearn; \
    print('ml extra OK:', xgboost.__version__, catboost.__version__)"

COPY . .

# Non-root. Nothing here needs to write outside /app, and a container process
# that can rewrite its own image is a larger blast radius than this job needs.
RUN useradd --create-home --uid 10001 propiq \
    && mkdir -p /app/data /app/outputs \
    && chown -R propiq:propiq /app
USER propiq

# Cap the numeric libraries' thread pools. XGBoost, CatBoost, OpenMP and BLAS
# each default to every VISIBLE core, which on a shared container is the host's
# count and not this container's share — so the default is contention, not
# parallelism. Override at deploy time to match the plan's CPU allocation.
ENV PROPIQ_MAX_THREADS=2

# The ledger must not live on the container filesystem: `data/**` is gitignored
# and a redeploy destroys it, taking every ticket's at-bet-time probability and
# EV with it. Postgres is the only durable option in the image.
ENV PROPIQ_PARLAY_LEDGER=postgres

# STATE THAT MUST OUTLIVE A REDEPLOY LIVES UNDER /app/data, so mount the
# Railway volume there. Two things depend on it:
#
#   data/external/model_runs/  the trained artifacts score_prob_over loads. On
#                              a fresh container this is empty and EVERY ROW
#                              ABSTAINS — the pipeline runs, writes nothing
#                              useful, and does not look broken. Not shipped in
#                              this image: TRAIN INTO THE VOLUME, or put the
#                              artifact plus its .meta.json sidecar there and
#                              point $PROPIQ_MODEL at it.
#
#                              THERE IS NO BOOT-TIME FETCH. An earlier version
#                              of this comment said "or fetch from object
#                              storage at boot", and no such code exists in
#                              this repository — no S3 client, no Supabase
#                              Storage client, nothing that downloads a model.
#                              Saying otherwise invited a first deploy that
#                              assumed the container would help itself.
#                              scheduler_worker.check_model_artifact() probes
#                              for the artifact at boot and names what it
#                              found, so an unseeded volume is an ERROR in the
#                              first lines of the log rather than a silent
#                              abstention every night.
#   calibration.json           the evidence the publication gate reads. The
#                              settlement job writes it at 03:30 PT and the
#                              slate job reads it at 09:00 PT; on the ephemeral
#                              layer a redeploy between those two leaves the
#                              gate with no evidence, so every card is withheld
#                              for a reason that is not the real one.
ENV PROPIQ_CALIBRATION_REPORT=/app/data/calibration.json

# THE SCHEMA IS APPLIED AT BOOT by scripts/start.sh, which hard-fails the
# container when a migration cannot be applied. Set PROPIQ_MIGRATE_ON_BOOT=false
# where a separate release step owns the schema and this worker must not touch
# it. `scripts/migrate_db` keeps its read-only default for the case where a
# human is typing; `scripts/run_migrations` is the applying front door.

# THE VOLUME MOUNT SHADOWS THE chown ABOVE. /app/data is created and chowned to
# propiq at build time, but a volume mounted there at RUN time replaces it with
# whatever the platform provisions — commonly root-owned. This container runs as
# uid 10001, so the first write can fail with EACCES, and the worker's broad
# except would report it as a failed calibration report rather than a
# permissions problem. scheduler_worker.check_state_dir() probes it at boot and
# names it; see docs/deploy_railway.md for the fix if it fires.

# THE START SEQUENCE IS A SCRIPT, NOT THIS LINE. scripts/start.sh applies the
# pending migrations, runs the deployment healthcheck, and then execs the
# worker. The order matters and getting it wrong is silent: a schema behind the
# code surfaces at 09:00 PT as a failed insert, hours after the deploy looked
# successful. `exec` in that script keeps the worker as PID 1 so SIGTERM
# reaches its handler, which shuts the scheduler down AFTER the running job
# rather than being SIGKILLed mid-slate.
#
# railway.json repeats this as `startCommand` because the platform's own
# setting wins over the image's CMD when one is set, and a start command typed
# into a dashboard is not in version control.
CMD ["bash", "scripts/start.sh"]
