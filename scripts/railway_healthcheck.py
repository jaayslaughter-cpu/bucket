"""
scripts/railway_healthcheck.py — is this deployment able to do its job?

RESEARCH_ONLY. This probes configuration and storage. It fetches no odds,
scores no slate, places no wager and prints no credential.

WHAT THIS IS FOR. `scheduler_worker.main` already checks three things at boot
(`check_state_dir`, `check_model_artifact`, `check_parlay_ledger`) and its
policy is to WARN AND CARRY ON: a worker that refuses to start cannot report
anything, and the settlement job is still useful with no model. That policy is
right for the worker and useless for a deploy gate — you want one command that
EXITS NONZERO when the deployment cannot do its job, runnable from
`railway run` before anybody waits for 09:00 PT to find out.

So this asks the same questions and answers them with an exit code:

    0  OK      — every check passed; warnings may still be printed.
    1  FAILURE — at least one check means the pipeline cannot do its job.
    2  the probe itself could not run (an import failed, say).

WHY IT CALLS THE REAL RESOLVERS. Every path and every default here comes from
the module that owns it: `src.utils.volume.resolve_state_root`,
`main.resolve_model_artifact`, `src.quant.parlay_log.resolve_ledger_choice`,
`src.db.migrations.status`, `scheduler_worker.board_markets`. A healthcheck
that globs for `*.joblib` under a path of its own invention tests the
healthcheck's assumptions, not the deployment: it reports FAILURE on a
correctly seeded volume, or — worse — SUCCESS on a broken one. This project
writes `xgboost_{MARKET}.json` plus a `.meta.json` sidecar into
`config/model_comparison.yaml`'s `artifacts_dir`, and that is only knowable by
asking the resolver.

NOT A RAILWAY HTTP HEALTHCHECK. The service binds no port; there is nothing
for the platform to poll. Do not set `healthcheckPath` — see `railway.json`.

Usage:
    python -m scripts.railway_healthcheck            # human-readable, exit code
    python -m scripts.railway_healthcheck --json     # one JSON object
    python -m scripts.railway_healthcheck --skip-db  # when no database is reachable
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OK = "OK"
WARN = "WARN"
FAIL = "FAIL"

#: Only these cost an exit code. WARN exists for the things that are normal on
#: a first boot — no calibration report has been written yet — and for the ones
#: that are a choice rather than a fault.
FAILING_STATUSES = (FAIL,)


@dataclass
class Check:
    """One question, its answer, and why the answer matters."""

    name: str
    status: str
    detail: str
    data: dict[str, Any] = field(default_factory=dict)


def _check_state_root() -> list[Check]:
    """
    Where durable state resolves to, HOW, and whether it can be written.

    The `how` is the point. "/app/data via RAILWAY_VOLUME_MOUNT_PATH" and
    "/app/data because the directory exists" are the same path and different
    deployments: the second is the image's own mount point with no volume
    attached, which writes to the container filesystem, succeeds, and loses
    everything at the next redeploy.
    """
    from src.utils.volume import (
        ENV_RAILWAY_VOLUME,
        ENV_STATE_DIR,
        artifact_dir_on,
        resolve_state_root,
    )

    root, how = resolve_state_root()
    out = [
        Check(
            "state_root",
            OK,
            f"{root} ({how})",
            {"path": str(root), "resolved_via": how,
             "artifacts_dir": str(artifact_dir_on(root))},
        )
    ]

    # On a container, a state root nobody named is the ephemeral-disk shape.
    explicit = bool(
        (os.environ.get(ENV_STATE_DIR) or "").strip()
        or (os.environ.get(ENV_RAILWAY_VOLUME) or "").strip()
    )
    if not explicit and Path("/app").is_dir():
        out.append(Check(
            "volume_attached", WARN,
            f"Neither {ENV_RAILWAY_VOLUME} nor {ENV_STATE_DIR} is set, so state "
            f"resolved to {root} by fallback. Inside a container that is the "
            "EPHEMERAL layer unless a volume is mounted there: the writes "
            "succeed and a redeploy destroys the artifacts and the calibration "
            "evidence. Mount a volume at /app/data.",
            {"resolved_via": how},
        ))

    # WHETHER IT EXISTED BEFORE THIS PROBE is a finding of its own. A volume
    # that failed to mount leaves the path absent, and creating it here would
    # make "not mounted" read as "mounted and empty" — the probe would then
    # confirm its own directory as writable and say nothing. It still creates
    # it, because refusing to probe is worse than probing a path it made, but
    # it reports which happened.
    existed = root.is_dir()
    probe = root / ".propiq_healthcheck_probe"
    try:
        root.mkdir(parents=True, exist_ok=True)
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        if existed:
            out.append(Check("state_root_writable", OK, f"{root} is writable"))
        else:
            out.append(Check(
                "state_root_writable", WARN,
                f"{root} DID NOT EXIST and this probe created it. It is "
                f"writable, but a path that was absent is not a mounted "
                f"volume: check the mount, and check {ENV_STATE_DIR} and "
                f"{ENV_RAILWAY_VOLUME} for a typo.",
                {"path": str(root), "created_by_probe": True},
            ))
    except OSError as exc:
        uid = os.getuid() if hasattr(os, "getuid") else "?"
        out.append(Check(
            "state_root_writable", FAIL,
            f"{root} IS NOT WRITABLE ({exc}). A mounted volume shadows the "
            f"image's chown and is commonly root-owned while this process runs "
            f"as uid {uid}. chown it to that uid; see docs/deploy_railway.md.",
            {"path": str(root), "uid": str(uid)},
        ))
    return out


def _check_model_artifacts(markets: list[str]) -> list[Check]:
    """
    One check per market, through `main.resolve_model_artifact`.

    An absent artifact is a FAILURE here and a warning in the worker, and the
    difference is deliberate: the worker still has a settlement job to run,
    while a deploy with no artifact scores nothing. `score_prob_over` returns
    an all-null Series and EVERY ROW ABSTAINS — the slate runs, exits 0 and
    does not look broken, which is this worker's most expensive failure mode.
    """
    from main import ENV_MODEL, resolve_model_artifact
    from scheduler_worker import MIN_PLAUSIBLE_TRAIN_ROWS
    from src.utils.volume import resolve_state_root

    state_root = resolve_state_root()[0].resolve()
    out: list[Check] = []
    for market in markets:
        name = f"model_artifact[{market}]"
        try:
            path, how = resolve_model_artifact(market=market)
        except Exception as exc:  # noqa: BLE001 — the resolver's error is the answer
            out.append(Check(name, FAIL, f"the resolver raised: {exc}"))
            continue

        if path is None:
            out.append(Check(name, FAIL, f"nothing resolved: {how}"))
            continue
        if not Path(path).exists():
            hint = (
                f" ${ENV_MODEL} points at a path this filesystem does not have."
                if how == ENV_MODEL else ""
            )
            out.append(Check(
                name, FAIL,
                f"resolved to {path} (via {how}) and it does not exist.{hint} "
                "Train into the volume, or seed the artifact and its "
                ".meta.json sidecar there.",
                {"path": str(path), "resolved_via": how},
            ))
            continue

        sidecar = Path(path).with_suffix(".meta.json")
        if not sidecar.exists():
            out.append(Check(
                name, FAIL,
                f"{path} has no {sidecar.name}. Scoring is refused without the "
                "feature_cols used at training time, so the artifact is unusable.",
                {"path": str(path), "resolved_via": how},
            ))
            continue

        rows: Any = None
        try:
            rows = json.loads(sidecar.read_text(encoding="utf-8")).get("train_row_count")
        except Exception as exc:  # noqa: BLE001 — an unreadable sidecar is the answer
            out.append(Check(
                name, WARN,
                f"{path} is present but {sidecar.name} could not be read ({exc}).",
                {"path": str(path), "resolved_via": how},
            ))
            continue

        data = {"path": str(path), "resolved_via": how, "train_row_count": rows}
        if isinstance(rows, int) and rows < MIN_PLAUSIBLE_TRAIN_ROWS:
            out.append(Check(
                name, FAIL,
                f"{path} was fit on {rows} rows, under the {MIN_PLAUSIBLE_TRAIN_ROWS} "
                "this project treats as the floor for a real one — this is almost "
                "certainly the synthetic demo artifact. Scoring a live slate with "
                "it is worse than abstaining.",
                data,
            ))
        else:
            out.append(Check(
                name, OK,
                f"{path} (via {how}), fit on {rows} rows", data,
            ))

        # THE ARTIFACT CAN BE PRESENT AND STILL NOT SURVIVE A REDEPLOY.
        # `resolve_model_artifact` reads `artifacts_dir` from the comparison
        # config, which ships as the RELATIVE `data/external/model_runs/
        # comparison` — correct in the image only because WORKDIR is /app and
        # the Dockerfile says to mount the volume at /app/data. Mount it
        # anywhere else and the resolver still reads /app/data: the artifact
        # loads, the slate scores, and the next deploy starts from nothing.
        # So the question is not "is it there" but "is it there DURABLY".
        try:
            durable = Path(path).resolve().is_relative_to(state_root)
        except OSError:
            durable = True  # unreadable is a different finding, reported above
        if not durable:
            out.append(Check(
                f"model_artifact_durable[{market}]", WARN,
                f"{Path(path).resolve()} is not under the durable state root "
                f"{state_root}, so a redeploy destroys it and every row then "
                f"abstains. Either mount the volume at /app/data (what the "
                f"Dockerfile documents) or point ${ENV_MODEL} onto the volume.",
                {"path": str(Path(path).resolve()), "state_root": str(state_root)},
            ))
    return out


def _check_bigdataball_workbook() -> list[Check]:
    """
    Is the licensed workbook where step [2] will look for it?

    THE BIGGEST SILENT BLOCKER IN A FRESH DEPLOY, and nothing reported it
    before. `main.ingest_market_lines` is NOT guarded: a missing path raises
    `FileNotFoundError`, the orchestrator's outer handler records a FAILED
    `pipeline_runs` row and returns 1, and the slate produces nothing at all.
    The image excludes it on purpose — `.dockerignore` drops `data/` and
    `*.xlsx`, because a licensed third-party export does not belong in an image
    layer — so a container that nobody uploaded it to fails EVERY scheduled run
    at 09:00 PT, having looked healthy at boot.

    Failing loudly there is the right behaviour, not a bug: the workbook
    supplies the Elo, `MKT_*` and `DEF_*` columns, 11 of the 38 a trained
    contract names, and a run that quietly built 27 would be rejected by the
    contract check on every row instead — same empty board, worse diagnosis.
    What was missing is the question being asked BEFORE the slate.
    """
    from main import DEFAULT_BIGDATABALL_XLSX, ENV_BIGDATABALL

    configured = (os.environ.get(ENV_BIGDATABALL) or "").strip()
    path = Path(configured or DEFAULT_BIGDATABALL_XLSX)
    if path.exists():
        return [Check(
            "bigdataball_workbook", OK, str(path),
            {"path": str(path), "configured": bool(configured)},
        )]
    return [Check(
        "bigdataball_workbook", FAIL,
        f"{path} does not exist, and step [2] does not degrade: it raises "
        f"FileNotFoundError and the WHOLE SLATE fails. The image excludes "
        f"`data/` and `*.xlsx` on purpose, so upload the licensed export onto "
        f"the volume and point ${ENV_BIGDATABALL} at it. Without it the Elo, "
        f"MKT_* and DEF_* columns cannot be built at all.",
        {"path": str(path)},
    )]


def _check_database(skip: bool) -> list[Check]:
    """
    Connect, then ask the migration ledger what this database is missing.

    A pending migration is a FAILURE, not a warning: the code expects columns
    the database does not have, and the first thing to find out is an insert at
    09:00 PT. Drift — a migration file edited after it was applied — is the
    failure a version number cannot see, so it is reported separately.
    """
    if skip:
        return [Check("database", WARN, "--skip-db: not checked")]

    try:
        from src.db.session import get_engine
    except Exception as exc:  # noqa: BLE001 — a missing driver is the answer
        return [Check("database", FAIL, f"cannot import the database layer: {exc}")]

    try:
        engine = get_engine()
    except Exception as exc:  # noqa: BLE001 — an unset DATABASE_URL is the answer
        return [Check(
            "database", FAIL,
            f"no usable connection: {exc}. Set DATABASE_URL — there is "
            "deliberately no SQLite fallback, because a run that wrote nowhere "
            "the rest of the stack can read would look successful.",
        )]

    out: list[Check] = []
    try:
        from src.db.migrations import status

        with engine.begin() as conn:
            report = status(conn)
    except Exception as exc:  # noqa: BLE001 — the driver's own error is the answer
        return [Check("database", FAIL, f"connected or queried and failed: {exc}")]

    out.append(Check("database", OK, "reachable"))

    pending = [f.filename for f in report["pending"]]
    if pending:
        out.append(Check(
            "migrations", FAIL,
            f"{len(pending)} pending: {pending}. Run "
            "`python -m scripts.run_migrations --apply`.",
            {"pending": pending},
        ))
    else:
        out.append(Check(
            "migrations", OK,
            f"{len(report['applied'])} applied, none pending",
            {"applied": sorted(report["applied"])},
        ))

    if report["drift"]:
        out.append(Check(
            "migration_drift", FAIL,
            "these were applied and the file has changed since: "
            f"{[d.filename for d in report['drift']]}. Every version looks "
            "present while the repository and the database have diverged.",
            {"drift": [d.filename for d in report["drift"]]},
        ))
    if report["orphaned_ledger_rows"]:
        out.append(Check(
            "migration_orphans", WARN,
            f"{report['orphaned_ledger_rows']} ledger row(s) have no file in this "
            "checkout — this database carries a migration the code cannot show you.",
        ))
    return out


def _check_ledger(db_configured: bool) -> list[Check]:
    """
    Which parlay-ledger backend this environment resolved to.

    csv while a database is configured is the shape that loses tickets
    silently: the CSV writes succeed, `data/**` is gitignored and ephemeral,
    and the ledger holds each ticket's at-bet-time probability and EV — the one
    thing that cannot be recomputed after the game.
    """
    try:
        from src.quant.parlay_log import ENV_LEDGER_BACKEND, resolve_ledger_choice

        choice = resolve_ledger_choice()
    except Exception as exc:  # noqa: BLE001 — a typo'd backend raises, by design
        return [Check("parlay_ledger", FAIL, str(exc))]

    if choice == "postgres":
        return [Check("parlay_ledger", OK, "postgres", {"backend": choice})]
    if db_configured:
        return [Check(
            "parlay_ledger", FAIL,
            f"csv while a database is configured. Set "
            f"{ENV_LEDGER_BACKEND}=postgres: the CSVs land on the ephemeral "
            "layer, the writes succeed, and a redeploy takes every ticket's "
            "at-bet-time probability and EV with it.",
            {"backend": choice},
        )]
    return [Check(
        "parlay_ledger", WARN,
        "csv, and no database is configured either", {"backend": choice},
    )]


def _check_calibration_report() -> list[Check]:
    """
    Where the publication gate's evidence lives, and whether it is there yet.

    Absent is a WARNING, not a failure: nothing has written it until the first
    settlement job runs at 03:30 PT. A path OFF the state root is the real
    finding, because that is the configuration where settlement writes it at
    03:30 and a redeploy before 09:00 leaves the gate with no evidence — so
    every card is withheld for a reason that is not the real one.
    """
    from scheduler_worker import DEFAULT_CALIBRATION_REPORT, ENV_CALIBRATION_REPORT
    from src.utils.volume import resolve_state_root

    configured = (os.environ.get(ENV_CALIBRATION_REPORT) or "").strip()
    path = Path(configured or DEFAULT_CALIBRATION_REPORT)
    root = resolve_state_root()[0].resolve()
    out: list[Check] = []

    try:
        durable = path.resolve().is_relative_to(root)
    except OSError:
        durable = False
    if not durable:
        out.append(Check(
            "calibration_report_location", WARN,
            f"{path} is not under the durable state root {root}. The settlement "
            f"job writes it at 03:30 PT and the slate reads it at 09:00 PT; on "
            f"an ephemeral path a redeploy between the two withholds every card. "
            f"Set {ENV_CALIBRATION_REPORT} under {root}.",
            {"path": str(path), "state_root": str(root)},
        ))

    out.append(Check(
        "calibration_report", OK if path.exists() else WARN,
        f"{path}" + ("" if path.exists() else
                     " does not exist yet — normal until the first settlement "
                     "run; until then the publication gate has no evidence and "
                     "withholds every card."),
        {"path": str(path), "exists": path.exists()},
    ))
    return out


def _check_dispatch() -> list[Check]:
    """
    Whether a Discord webhook is configured — NEVER what it is.

    The URL is a credential: anyone holding it can post to that channel. Only
    its presence is reported, and `src/notify/discord.py` redacts it from every
    error and log line for the same reason.
    """
    configured = bool((os.environ.get("DISCORD_WEBHOOK_URL") or "").strip())
    if configured:
        return [Check("discord_webhook", OK, "configured (value not shown)")]
    return [Check(
        "discord_webhook", WARN,
        "DISCORD_WEBHOOK_URL is unset, so nothing is dispatched. The pipeline "
        "still runs and still writes to Postgres.",
    )]


def run_checks(*, skip_db: bool = False) -> list[Check]:
    """Every check, in the order a first deploy breaks them."""
    db_configured = bool(
        (os.environ.get("DATABASE_URL") or "").strip()
        or (os.environ.get("PGHOST") or "").strip()
    )

    try:
        from scheduler_worker import board_markets

        markets = list(board_markets())
    except Exception as exc:  # noqa: BLE001 — fall back rather than skip the check
        markets = ["PTS", "REB", "AST"]
        print(f"note: could not read the configured markets ({exc}); "
              f"checking {markets}", file=sys.stderr)

    checks: list[Check] = []
    checks += _check_state_root()
    checks += _check_model_artifacts(markets)
    checks += _check_bigdataball_workbook()
    checks += _check_database(skip_db)
    checks += _check_ledger(db_configured)
    checks += _check_calibration_report()
    checks += _check_dispatch()
    return checks


def verdict(checks: list[Check]) -> str:
    """FAILURE if anything failed, OK otherwise. Warnings never fail a deploy."""
    return "FAILURE" if any(c.status in FAILING_STATUSES for c in checks) else "OK"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Probe a PropIQ deployment. Exits nonzero when it cannot "
                    "do its job. RESEARCH_ONLY — no odds, no wager."
    )
    ap.add_argument("--json", action="store_true",
                    help="One JSON object, for a deploy script to read.")
    ap.add_argument("--skip-db", action="store_true",
                    help="Do not touch the database. For a build-time probe, "
                         "where DATABASE_URL is not injected.")
    args = ap.parse_args(argv)

    try:
        checks = run_checks(skip_db=args.skip_db)
    except Exception as exc:  # noqa: BLE001 — the probe failing is its own answer
        payload = {"verdict": "ERROR", "error": str(exc)}
        print(json.dumps(payload) if args.json
              else f"ERROR: the healthcheck could not run: {exc}",
              file=sys.stderr)
        return 2

    result = verdict(checks)
    if args.json:
        print(json.dumps({
            "verdict": result,
            "checks": [
                {"name": c.name, "status": c.status, "detail": c.detail, **c.data}
                for c in checks
            ],
        }, indent=2, default=str))
    else:
        print("PropIQ deployment healthcheck — RESEARCH_ONLY")
        print()
        for c in checks:
            print(f"  [{c.status:<4}] {c.name}: {c.detail}")
        print()
        failed = [c.name for c in checks if c.status in FAILING_STATUSES]
        warned = [c.name for c in checks if c.status == WARN]
        print(f"VERDICT: {result}"
              + (f" — failed: {failed}" if failed else "")
              + (f" (warnings: {warned})" if warned else ""))

    return 1 if result == "FAILURE" else 0


if __name__ == "__main__":
    raise SystemExit(main())
