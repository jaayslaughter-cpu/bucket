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
# NOT BUILT OR RUN ANYWHERE YET. The environment this was written in has the
# docker client but no daemon, and its network policy denies the package index,
# so no `docker build` of this file has ever executed. Every instruction here is
# reasoned from the repository's own dependency metadata, not from a green
# build. Build it once locally before trusting a deploy.
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

# libgomp1 is OpenMP, which xgboost and catboost link against. Without it the
# image builds and then fails at import, which is the same late failure the ML
# extra is installed to avoid.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependency metadata first so a code change does not reinstall the world.
COPY pyproject.toml README.md ./
COPY src/__init__.py src/__init__.py
RUN pip install --no-cache-dir -e ".[ml,db,deploy]"

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

# MODEL ARTIFACTS ARE NOT IN THIS IMAGE and are not written to a volume by
# anything here. data/external/model_runs/ is where score_prob_over looks; on a
# fresh container it is empty and every row abstains. Mount a Railway volume at
# /app/data, or fetch the artifacts from object storage at boot, BEFORE relying
# on a deployed slate run. This comment is the warning, not a fix.

CMD ["python", "scheduler_worker.py"]
