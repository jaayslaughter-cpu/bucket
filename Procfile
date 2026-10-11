# PropIQ Analytics — RESEARCH_ONLY. A WORKER, not a web service: nothing here
# binds a port, so a `web:` process would be marked unhealthy for never
# listening and restarted in a loop. That is why this file declares no `web`.
#
# This is the BUILDPACK fallback. The supported build path is the Dockerfile
# (railway.json points at it), and `scripts/start.sh` is the same sequence
# either way: migrate, probe, then run.
worker: bash scripts/start.sh
