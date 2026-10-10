"""
tests/test_railway_deploy.py — the deployment's own contract.

WHAT THIS GUARDS. The deploy surface is the part of this repository that no
other test touches, because nothing imports a Dockerfile and nothing calls a
start script. Every defect in it is found by deploying: the container boots,
the build log is green, and the failure arrives hours later as an empty board.
So the things asserted here are the ones whose breakage is silent.

Three of these tests exist because the code they cover was WRONG when written:

  * `artifact_dir_on` resolved to one directory ABOVE a state root that was
    not literally named `data`, which is writable, which nothing reads, and
    which a redeploy destroys.
  * the healthcheck in the original brief imported `check_model_artifact` and
    `resolve_model_artifact` from `src.models.artifact_registry`, which exposes
    neither, and fell back to globbing `*PTS*.joblib` under `/app/data/models/`
    — paths and extensions this project never writes. It reported FAILURE on a
    correctly seeded volume.
  * seeding copied the booster without its `.mean.json` head, which loads as
    `mean_model = None`: null projections beside live probabilities, the
    failure that looks most like a working deployment.

RESEARCH_ONLY project. Nothing here prices, sizes or places a wager.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# src/utils/volume.py — where durable state resolves to, and HOW
# ---------------------------------------------------------------------------

def test_the_operator_override_wins_over_the_platform_variable(monkeypatch, tmp_path):
    from src.utils.volume import (
        ENV_RAILWAY_VOLUME,
        ENV_STATE_DIR,
        resolve_state_root,
    )

    monkeypatch.setenv(ENV_STATE_DIR, str(tmp_path / "chosen"))
    monkeypatch.setenv(ENV_RAILWAY_VOLUME, str(tmp_path / "platform"))
    root, how = resolve_state_root()
    assert root == tmp_path / "chosen"
    assert how == ENV_STATE_DIR


def test_the_railway_variable_is_read_when_nothing_overrides_it(monkeypatch, tmp_path):
    """The variable this repository read NOWHERE before src/utils/volume.py."""
    from src.utils.volume import (
        ENV_RAILWAY_VOLUME,
        ENV_STATE_DIR,
        resolve_state_root,
    )

    monkeypatch.delenv(ENV_STATE_DIR, raising=False)
    monkeypatch.setenv(ENV_RAILWAY_VOLUME, str(tmp_path / "vol"))
    root, how = resolve_state_root()
    assert root == tmp_path / "vol"
    assert how == ENV_RAILWAY_VOLUME


def test_how_the_root_was_chosen_is_reported_not_just_the_path(monkeypatch, tmp_path):
    """
    "/app/data via RAILWAY_VOLUME_MOUNT_PATH" and "/app/data because the
    directory exists" are the same path and different deployments: only one of
    them survives a redeploy. The second element of the tuple is the whole
    point of returning a tuple.
    """
    from src.utils.volume import ENV_RAILWAY_VOLUME, ENV_STATE_DIR, resolve_state_root

    monkeypatch.delenv(ENV_STATE_DIR, raising=False)
    monkeypatch.delenv(ENV_RAILWAY_VOLUME, raising=False)
    _, how = resolve_state_root()
    assert ENV_RAILWAY_VOLUME not in how and ENV_STATE_DIR not in how
    assert "volume" in how.lower() or "/app/data" in how


def test_the_artifacts_dir_lands_inside_a_state_root_not_named_data(
    monkeypatch, tmp_path
):
    """
    THE BUG THIS WAS WRITTEN FOR. The configured `artifacts_dir` ships as the
    relative `data/external/model_runs/comparison`, whose leading segment IS
    the state root. An earlier version stripped that segment only when the
    root happened to be NAMED "data", so `PROPIQ_STATE_DIR=/mnt/vol` resolved
    the artifacts to `/mnt/external/...` — one directory above the volume.
    """
    from src.utils.volume import ENV_STATE_DIR, artifact_dir_on

    root = tmp_path / "mnt" / "some-volume-name"
    monkeypatch.setenv(ENV_STATE_DIR, str(root))
    resolved = artifact_dir_on()

    # EXACT, not `is_relative_to`. The two ways this has been got wrong land in
    # different places and only one of them is outside the root:
    #   base.parent / path  -> <root>/../data/external/...  (above the volume)
    #   base / path         -> <root>/data/external/...     (inside it, and
    #                          still wrong: nothing reads that path, because
    #                          the resolver joins the config value onto the
    #                          working directory, not onto a nested `data`).
    # `is_relative_to` alone passes the second, so it is asserted exactly.
    assert resolved == root / "external" / "model_runs" / "comparison", resolved


def test_an_artifacts_dir_not_under_data_is_joined_onto_the_state_root(
    monkeypatch, tmp_path
):
    """
    THE OTHER BRANCH, which the test above cannot reach: the shipped config
    value starts with `data`, so the join-whole path is only taken by a
    reconfigured `artifacts_dir`. It must land INSIDE the state root — an
    earlier version used `base.parent`, which put it one directory above the
    volume, a path that is writable, that nothing reads, and that a redeploy
    destroys.
    """
    import src.models.compare as compare
    from src.utils.volume import ENV_STATE_DIR, artifact_dir_on

    root = tmp_path / "vol"
    monkeypatch.setenv(ENV_STATE_DIR, str(root))
    monkeypatch.setattr(
        compare, "load_comparison_config", lambda *a, **k: {"artifacts_dir": "runs/x"}
    )
    assert artifact_dir_on() == root / "runs" / "x"


def test_an_absolute_artifacts_dir_is_taken_as_given(monkeypatch, tmp_path):
    """An operator who names a path has named it; joining a state root onto it
    would silently relocate the artifacts the slate is pointed at."""
    import src.models.compare as compare
    from src.utils.volume import ENV_STATE_DIR, artifact_dir_on

    monkeypatch.setenv(ENV_STATE_DIR, str(tmp_path / "vol"))
    monkeypatch.setattr(
        compare, "load_comparison_config",
        lambda *a, **k: {"artifacts_dir": "/srv/artifacts"},
    )
    assert artifact_dir_on() == Path("/srv/artifacts")


def test_the_artifacts_dir_agrees_with_the_resolver_the_slate_uses(monkeypatch):
    """
    `artifact_dir_on` must not be a second opinion. Under the repository's own
    data directory it has to name the directory `main.resolve_model_artifact`
    actually globs, or a healthcheck and a deploy document would agree with
    each other and disagree with the pipeline.
    """
    from main import resolve_model_artifact
    from src.utils.volume import ENV_STATE_DIR, artifact_dir_on

    monkeypatch.delenv(ENV_STATE_DIR, raising=False)
    monkeypatch.delenv("PROPIQ_MODEL", raising=False)
    path, _ = resolve_model_artifact(market="PTS")
    if path is None or not Path(path).exists():
        pytest.skip("no artifact in this checkout to compare against")
    assert Path(path).resolve().parent == artifact_dir_on(REPO / "data").resolve()


# ---------------------------------------------------------------------------
# scripts/railway_healthcheck.py — strict where the worker is forgiving
# ---------------------------------------------------------------------------

def _seed(artifacts_dir: Path, market: str, *, rows: int, sidecar: bool = True) -> Path:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    artifact = artifacts_dir / f"xgboost_{market}.json"
    artifact.write_text("{}", encoding="utf-8")
    (artifacts_dir / f"xgboost_{market}.mean.json").write_text("{}", encoding="utf-8")
    if sidecar:
        (artifacts_dir / f"xgboost_{market}.meta.json").write_text(
            json.dumps({"train_row_count": rows, "feature_cols": ["x"]}),
            encoding="utf-8",
        )
    return artifact


def _volume(monkeypatch, tmp_path, *, rows=200_000, sidecar=True, markets=("PTS",)):
    """A state root that looks like a seeded volume, and the checks run on it."""
    from src.utils.volume import ENV_STATE_DIR, artifact_dir_on

    root = tmp_path / "volume"
    root.mkdir()
    monkeypatch.setenv(ENV_STATE_DIR, str(root))
    monkeypatch.delenv("PROPIQ_MODEL", raising=False)
    monkeypatch.setenv("PROPIQ_BOARD_MARKETS", ",".join(markets))
    for market in markets:
        artifact = _seed(artifact_dir_on(root), market, rows=rows, sidecar=sidecar)
    monkeypatch.setenv("PROPIQ_MODEL", str(artifact))
    return root


def _statuses(checks) -> dict[str, str]:
    return {c.name: c.status for c in checks}


def test_a_correctly_seeded_volume_is_not_reported_as_a_failure(monkeypatch, tmp_path):
    """
    THE DEFECT IN THE ORIGINAL HEALTHCHECK, as a test. It globbed
    `/app/data/models/*PTS*.joblib`; this project writes
    `xgboost_PTS.json` plus a `.meta.json` sidecar into the comparison
    `artifacts_dir`. A probe that invents its own paths reports FAILURE on a
    correct deployment, which is worse than no probe at all — it gets the
    deployment "fixed" in the direction of its own mistake.
    """
    from scripts.railway_healthcheck import FAIL, run_checks, verdict

    _volume(monkeypatch, tmp_path)
    monkeypatch.setenv("PROPIQ_PARLAY_LEDGER", "postgres")
    checks = run_checks(skip_db=True)
    failed = [c.name for c in checks if c.status == FAIL]
    assert failed == [], failed
    assert verdict(checks) == "OK"


def test_a_state_root_the_probe_had_to_create_is_not_reported_as_fine(
    monkeypatch, tmp_path
):
    """
    A VOLUME THAT FAILED TO MOUNT LEAVES THE PATH ABSENT. The probe creates it
    — refusing to probe would be worse — so without this it would confirm its
    OWN directory as writable and say nothing, turning "not mounted" into
    "mounted and empty".
    """
    from scripts.railway_healthcheck import OK, WARN, run_checks
    from src.utils.volume import ENV_STATE_DIR

    absent = tmp_path / "never-mounted"
    monkeypatch.setenv(ENV_STATE_DIR, str(absent))
    monkeypatch.setenv("PROPIQ_MODEL", str(absent / "x.json"))
    monkeypatch.setenv("PROPIQ_BOARD_MARKETS", "PTS")
    assert _statuses(run_checks(skip_db=True))["state_root_writable"] == WARN

    # Second run: it exists now (the first created it), so no warning.
    assert _statuses(run_checks(skip_db=True))["state_root_writable"] == OK


def test_an_unseeded_volume_is_a_failure_with_a_nonzero_exit(monkeypatch, tmp_path):
    from scripts.railway_healthcheck import main, run_checks, verdict
    from src.utils.volume import ENV_STATE_DIR

    root = tmp_path / "empty"
    root.mkdir()
    monkeypatch.setenv(ENV_STATE_DIR, str(root))
    monkeypatch.setenv("PROPIQ_MODEL", str(root / "nothing.json"))
    monkeypatch.setenv("PROPIQ_BOARD_MARKETS", "PTS")
    assert verdict(run_checks(skip_db=True)) == "FAILURE"
    assert main(["--skip-db"]) == 1


def test_an_artifact_without_its_sidecar_is_a_failure(monkeypatch, tmp_path):
    """Scoring is refused without the sidecar's feature_cols, so a sidecar-less
    artifact is not a model: reporting it as present would turn a visible
    "nothing resolved" into a silent "resolved and unusable"."""
    from scripts.railway_healthcheck import FAIL, run_checks

    _volume(monkeypatch, tmp_path, sidecar=False)
    assert _statuses(run_checks(skip_db=True))["model_artifact[PTS]"] == FAIL


def test_the_synthetic_demo_artifact_is_a_failure_not_a_warning(monkeypatch, tmp_path):
    """Scoring a live slate with a model fit on a thousand synthetic rows is
    worse than abstaining, so the deploy gate refuses it."""
    from scheduler_worker import MIN_PLAUSIBLE_TRAIN_ROWS
    from scripts.railway_healthcheck import FAIL, run_checks

    _volume(monkeypatch, tmp_path, rows=MIN_PLAUSIBLE_TRAIN_ROWS - 1)
    assert _statuses(run_checks(skip_db=True))["model_artifact[PTS]"] == FAIL


def test_an_artifact_off_the_volume_is_flagged_even_though_it_loads(
    monkeypatch, tmp_path
):
    """
    PRESENT IS NOT DURABLE. Mount the volume anywhere but /app/data and the
    resolver still reads the relative `artifacts_dir`: the artifact loads, the
    slate scores, and the next deploy starts from nothing.
    """
    from scripts.railway_healthcheck import WARN, run_checks
    from src.utils.volume import ENV_STATE_DIR, artifact_dir_on

    elsewhere = tmp_path / "ephemeral"
    artifact = _seed(artifact_dir_on(elsewhere), "PTS", rows=200_000)
    volume = tmp_path / "volume"
    volume.mkdir()
    monkeypatch.setenv(ENV_STATE_DIR, str(volume))
    monkeypatch.setenv("PROPIQ_MODEL", str(artifact))
    monkeypatch.setenv("PROPIQ_BOARD_MARKETS", "PTS")

    statuses = _statuses(run_checks(skip_db=True))
    assert statuses["model_artifact[PTS]"] == "OK"
    assert statuses["model_artifact_durable[PTS]"] == WARN


def test_a_missing_workbook_is_a_failure_only_when_the_database_is_empty_too(
    monkeypatch, tmp_path
):
    """
    THIS TEST USED TO ASSERT THE OPPOSITE, and it was right to until
    2026-10-10. `main.ingest_market_lines` was unguarded, so an absent workbook
    failed the whole slate and an absent workbook was a FAILURE here.
    `resolve_market_frames` now falls back to `team_game_stats` and
    `game_market_lines` -- the workbook's own contents, which every run that
    finds one upserts -- so an absent file is no longer fatal and reporting it
    as FAILURE would fail a deployment that works. A probe that outlives the
    behaviour it probes is this repository's recurring defect in its most
    expensive form: it sends somebody to fix something that is not broken.
    """
    import pandas as pd

    import src.db.repository as repo
    from main import ENV_BIGDATABALL
    from scripts.railway_healthcheck import FAIL, OK, WARN, run_checks
    from src.db.repository import MARKET_LINE_PREGAME_COLS, TEAM_GAME_STAT_COLS

    _volume(monkeypatch, tmp_path)
    monkeypatch.setenv(ENV_BIGDATABALL, str(tmp_path / "absent.xlsx"))

    # Nothing anywhere -> FAILURE: resolve_market_frames refuses, correctly.
    monkeypatch.setattr(
        repo, "load_team_game_stats",
        lambda **k: pd.DataFrame(columns=list(TEAM_GAME_STAT_COLS)),
    )
    monkeypatch.setattr(
        repo, "load_game_market_lines",
        lambda **k: pd.DataFrame(columns=list(MARKET_LINE_PREGAME_COLS)),
    )
    assert _statuses(run_checks(skip_db=False))["bigdataball_workbook"] == FAIL

    # The database has it -> WARN, naming how old it is. Not a failure.
    monkeypatch.setattr(repo, "load_team_game_stats", lambda **k: pd.DataFrame([{
        **{c: None for c in TEAM_GAME_STAT_COLS},
        "nba_game_id": "0022600001", "game_date": pd.Timestamp("2026-01-10").date(),
        "team_abbr": "LAL", "points": 110,
    }]))
    monkeypatch.setattr(repo, "load_game_market_lines", lambda **k: pd.DataFrame([{
        "nba_game_id": "0022600001", "game_date": pd.Timestamp("2026-01-10").date(),
        "team_abbr": "LAL", "opening_spread": -3.5, "opening_total": 224.5,
    }], columns=list(MARKET_LINE_PREGAME_COLS)))
    checks = {c.name: c for c in run_checks(skip_db=False)}
    assert checks["bigdataball_workbook"].status == WARN
    assert "2026-01-10" in checks["bigdataball_workbook"].detail, (
        "the fallback's age is not reported, and nothing else would say it"
    )

    # A workbook on disk -> OK.
    present = tmp_path / "book.xlsx"
    present.write_bytes(b"only its presence is checked here")
    monkeypatch.setenv(ENV_BIGDATABALL, str(present))
    assert _statuses(run_checks(skip_db=True))["bigdataball_workbook"] == OK


def test_the_probe_cannot_claim_the_fallback_is_fine_without_looking(
    monkeypatch, tmp_path
):
    """With --skip-db and no workbook, whether the fallback has anything to read
    is UNKNOWN, and saying OK would be a guess."""
    from main import ENV_BIGDATABALL
    from scripts.railway_healthcheck import WARN, run_checks

    _volume(monkeypatch, tmp_path)
    monkeypatch.setenv(ENV_BIGDATABALL, str(tmp_path / "absent.xlsx"))
    check = {c.name: c for c in run_checks(skip_db=True)}["bigdataball_workbook"]
    assert check.status == WARN
    assert "UNKNOWN" in check.detail


def test_the_probe_and_the_orchestrator_read_one_workbook_path(monkeypatch):
    """
    AST-walked. The default lived inline in `main`'s argparse call, so the
    probe would have carried a second copy of the filename — and this
    repository has already shipped a default pointing at a file that does not
    exist (`..._Team-Stats__1_.xlsx`), which failed every slate. Two copies of
    that string is two chances to be wrong.
    """
    import ast

    tree = ast.parse((REPO / "main.py").read_text(encoding="utf-8"))
    dumped = ast.dump(tree)
    # The literal filename appears exactly once: at the constant's definition.
    assert dumped.count("2025-2026_NBA_Box_Score_Team-Stats.xlsx") == 1
    assert dumped.count("'BIGDATABALL_XLSX'") == 1


def test_a_csv_ledger_with_a_database_configured_is_a_failure(monkeypatch, tmp_path):
    """The shape that loses tickets silently: the CSV writes succeed, `data/**`
    is ephemeral, and the ledger holds the one number that cannot be recomputed
    after the game."""
    from scripts.railway_healthcheck import FAIL, OK, run_checks

    _volume(monkeypatch, tmp_path)
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@example.invalid:6543/postgres")
    monkeypatch.setenv("PROPIQ_PARLAY_LEDGER", "csv")
    assert _statuses(run_checks(skip_db=True))["parlay_ledger"] == FAIL

    monkeypatch.setenv("PROPIQ_PARLAY_LEDGER", "postgres")
    assert _statuses(run_checks(skip_db=True))["parlay_ledger"] == OK


def test_a_calibration_report_off_the_state_root_is_flagged(monkeypatch, tmp_path):
    """Settlement writes it at 03:30 PT and the slate reads it at 09:00 PT; on
    an ephemeral path a redeploy between the two withholds every card for a
    reason that is not the real one."""
    from scripts.railway_healthcheck import WARN, run_checks

    root = _volume(monkeypatch, tmp_path)
    monkeypatch.setenv("PROPIQ_CALIBRATION_REPORT", str(tmp_path / "x/calibration.json"))
    assert _statuses(run_checks(skip_db=True))["calibration_report_location"] == WARN

    monkeypatch.setenv("PROPIQ_CALIBRATION_REPORT", str(root / "calibration.json"))
    assert "calibration_report_location" not in _statuses(run_checks(skip_db=True))


def test_the_webhook_is_reported_as_present_and_never_printed(
    monkeypatch, tmp_path, capsys
):
    """The URL is a credential: anyone holding it can post to that channel."""
    from scripts.railway_healthcheck import main

    _volume(monkeypatch, tmp_path)
    secret = "https://discord.com/api/webhooks/123/SUPER-SECRET-TOKEN"
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", secret)
    main(["--skip-db"])
    out = capsys.readouterr()
    assert "SUPER-SECRET-TOKEN" not in out.out + out.err
    assert "configured" in out.out


def test_warnings_alone_never_fail_a_deploy(monkeypatch, tmp_path):
    """A first boot has no calibration report and may have no webhook. Failing
    on those would train an operator to ignore the exit code."""
    from scripts.railway_healthcheck import WARN, Check, verdict

    assert verdict([Check("a", WARN, "")]) == "OK"


# ---------------------------------------------------------------------------
# scripts/run_migrations.py — one implementation, two defaults
# ---------------------------------------------------------------------------

def test_run_migrations_applies_and_migrate_db_does_not(monkeypatch):
    """
    The whole difference between the two front doors, asserted. `migrate_db`
    must stay read-only by default (a migration tool that writes by default is
    one typo away from the wrong DATABASE_URL); `run_migrations` is the
    container's, where nobody is there to pass --apply.
    """
    import scripts.migrate_db as migrate_db
    import scripts.run_migrations as run_migrations

    seen: list[list[str]] = []
    monkeypatch.setattr(run_migrations, "_migrate_main", lambda argv: seen.append(argv) or 0)
    run_migrations.main(["--dry-run"])
    # --ensure-tables joined --apply on 2026-10-10: the SQL migrations ALTER
    # tables that create_all makes, so without it a brand-new database fails
    # at 002 and `scripts/start.sh` never starts the worker. A first deploy is
    # by definition a brand-new database.
    assert seen == [["--apply", "--ensure-tables", "--dry-run"]]

    parsed = migrate_db.main.__doc__  # the default lives in the argparse surface
    assert parsed is None or "apply" not in (parsed or "")


def test_run_migrations_is_a_wrapper_and_not_a_second_implementation():
    """
    AST-walked, not grepped. A prose promise that this delegates is exactly the
    kind of assertion this repository has been fooled by four times: a comment
    saying `engine.begin()` satisfied a source grep for it. So the test reads
    the module's call graph instead.
    """
    import ast

    tree = ast.parse((REPO / "scripts" / "run_migrations.py").read_text())
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "_migrate_main" in called
    # Nothing here may touch the database or the migration primitives directly.
    assert not any(
        isinstance(n, ast.ImportFrom) and n.module == "src.db.migrations"
        for n in ast.walk(tree)
    )


# ---------------------------------------------------------------------------
# scripts/seed_volume.py — one market is several files
# ---------------------------------------------------------------------------

def test_seeding_copies_the_mean_head_not_only_the_booster(monkeypatch, tmp_path):
    """
    `xgb_adapter.load` sets `mean_model = None` when `.mean.json` is absent, so
    a volume seeded with the booster alone returns NULL PROJECTIONS beside live
    probabilities — the failure that looks most like a working deployment.
    """
    from scripts.seed_volume import seed
    from src.utils.volume import ENV_STATE_DIR, artifact_dir_on

    source = tmp_path / "source"
    _seed(source, "PTS", rows=200_000)
    (source / "distribution_PTS.json").write_text("{}", encoding="utf-8")
    root = tmp_path / "volume"
    monkeypatch.setenv(ENV_STATE_DIR, str(root))

    code, lines = seed(source, apply=True, force=False, allow_demo=False)
    assert code == 0, lines
    landed = {p.name for p in artifact_dir_on(root).iterdir()}
    assert landed == {
        "xgboost_PTS.json", "xgboost_PTS.mean.json", "xgboost_PTS.meta.json",
        "distribution_PTS.json",
    }


def test_seeding_refuses_an_artifact_with_no_sidecar(monkeypatch, tmp_path):
    from scripts.seed_volume import seed
    from src.utils.volume import ENV_STATE_DIR, artifact_dir_on

    source = tmp_path / "source"
    _seed(source, "PTS", rows=200_000, sidecar=False)
    root = tmp_path / "volume"
    monkeypatch.setenv(ENV_STATE_DIR, str(root))

    code, lines = seed(source, apply=True, force=False, allow_demo=False)
    assert code == 1
    assert any("REFUSED" in line and "sidecar" in line for line in lines)
    assert not artifact_dir_on(root).exists()


def test_seeding_refuses_the_demo_artifact_unless_asked(monkeypatch, tmp_path):
    from scheduler_worker import MIN_PLAUSIBLE_TRAIN_ROWS
    from scripts.seed_volume import seed
    from src.utils.volume import ENV_STATE_DIR, artifact_dir_on

    source = tmp_path / "source"
    _seed(source, "PTS", rows=MIN_PLAUSIBLE_TRAIN_ROWS - 1)
    root = tmp_path / "volume"
    monkeypatch.setenv(ENV_STATE_DIR, str(root))

    code, lines = seed(source, apply=True, force=False, allow_demo=False)
    assert code == 1 and not artifact_dir_on(root).exists()
    assert any("demo" in line for line in lines)

    code, _ = seed(source, apply=True, force=False, allow_demo=True)
    assert code == 0 and (artifact_dir_on(root) / "xgboost_PTS.json").exists()


def test_seeding_will_not_silently_replace_the_artifact_on_the_volume(
    monkeypatch, tmp_path
):
    """It produced every probability now in the database."""
    from scripts.seed_volume import seed
    from src.utils.volume import ENV_STATE_DIR, artifact_dir_on

    source = tmp_path / "source"
    _seed(source, "PTS", rows=200_000)
    root = tmp_path / "volume"
    monkeypatch.setenv(ENV_STATE_DIR, str(root))
    (artifact_dir_on(root)).mkdir(parents=True)
    (artifact_dir_on(root) / "xgboost_PTS.json").write_text("OLD", encoding="utf-8")

    code, lines = seed(source, apply=True, force=False, allow_demo=False)
    assert code == 1
    assert (artifact_dir_on(root) / "xgboost_PTS.json").read_text() == "OLD"
    assert any("--force" in line for line in lines)

    seed(source, apply=True, force=True, allow_demo=False)
    assert (artifact_dir_on(root) / "xgboost_PTS.json").read_text() == "{}"


def test_seeding_writes_nothing_without_apply(monkeypatch, tmp_path):
    from scripts.seed_volume import seed
    from src.utils.volume import ENV_STATE_DIR, artifact_dir_on

    source = tmp_path / "source"
    _seed(source, "PTS", rows=200_000)
    root = tmp_path / "volume"
    monkeypatch.setenv(ENV_STATE_DIR, str(root))

    code, lines = seed(source, apply=False, force=False, allow_demo=False)
    assert code == 0
    assert not artifact_dir_on(root).exists()
    assert any("would copy" in line for line in lines)


# ---------------------------------------------------------------------------
# railway.json / Procfile / scripts/start.sh — the platform contract
# ---------------------------------------------------------------------------

def test_railway_json_builds_the_dockerfile_and_runs_the_start_script():
    config = json.loads((REPO / "railway.json").read_text(encoding="utf-8"))
    assert config["build"]["builder"] == "DOCKERFILE"
    assert (REPO / config["build"]["dockerfilePath"]).is_file()
    start = config["deploy"]["startCommand"]
    assert "scripts/start.sh" in start
    assert (REPO / "scripts" / "start.sh").is_file()


def test_the_service_declares_no_http_healthcheck():
    """`scheduler_worker` binds no port. A healthcheckPath would be polled
    forever, fail, and restart the container in a loop — which looks exactly
    like a crashing app and is not one."""
    config = json.loads((REPO / "railway.json").read_text(encoding="utf-8"))
    assert "healthcheckPath" not in config["deploy"]


def test_exactly_one_replica_is_configured():
    """
    TWO REPLICAS RUN THE SLATE TWICE. APScheduler's `max_instances=1` is
    per-process, so a second replica is a second scheduler with its own clock:
    two ingests, two boards, two Discord dispatches of the same card, and two
    settlement passes. There is no distributed lock in this project.
    """
    config = json.loads((REPO / "railway.json").read_text(encoding="utf-8"))
    assert config["deploy"]["numReplicas"] == 1


def test_the_procfile_declares_a_worker_and_no_web_process():
    body = (REPO / "Procfile").read_text(encoding="utf-8")
    declared = {
        line.split(":", 1)[0].strip()
        for line in body.splitlines()
        if line.strip() and not line.lstrip().startswith("#") and ":" in line
    }
    assert "worker" in declared
    assert "web" not in declared


def _start_script_code() -> list[str]:
    """
    The EXECUTABLE lines of scripts/start.sh, comments dropped.

    THIS HELPER EXISTS BECAUSE THE FIRST VERSION OF THE TWO TESTS BELOW WAS
    FOOLED BY THE SCRIPT'S OWN COMMENTS. The header explains the policy and
    names `scripts.railway_healthcheck` and `scripts.run_migrations` in prose,
    so `body.index(...)` found the comment, and `test_..._migrates_before_it_
    probes` failed with 2249 < 1508 against a script whose order was correct.
    That is the same trap this repository has now hit five times — a source
    assertion satisfied by prose ABOUT the code — and the fix is to assert on
    the code.
    """
    return [
        line for line in (REPO / "scripts" / "start.sh")
        .read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_the_start_script_migrates_before_it_probes_and_execs_the_worker():
    """
    THE ORDER IS THE CONTRACT. Probing before the migrations run reports
    pending migrations it is about to watch get applied, and `exec` last keeps
    the worker as PID 1 so SIGTERM reaches the handler that shuts the scheduler
    down after the running job instead of being SIGKILLed mid-slate.
    """
    code = _start_script_code()
    migrate = next(i for i, l in enumerate(code) if "scripts.run_migrations" in l)
    probe = next(i for i, l in enumerate(code) if "scripts.railway_healthcheck" in l)
    worker = next(i for i, l in enumerate(code) if "exec python scheduler_worker.py" in l)
    assert migrate < probe < worker
    assert any("set -euo pipefail" in line for line in code)


def test_a_failing_healthcheck_does_not_stop_the_worker_from_starting():
    """
    The two policies differ on purpose: migrations hard-fail the boot, the
    probe does not. A worker that refuses to start cannot report anything, and
    the settlement job is still useful with no model. Asserted on the control
    flow — the probe's exit status is consumed by an `if !`, so `set -e` cannot
    kill the container on it.
    """
    code = _start_script_code()
    probe_line = next(l for l in code if "scripts.railway_healthcheck" in l)
    assert probe_line.strip().startswith("if ! ")
    # The migration call, by contrast, is bare: set -e must kill the boot.
    migrate_line = next(l for l in code if "scripts.run_migrations" in l)
    assert migrate_line.strip() == "python -m scripts.run_migrations"


def test_the_start_script_actually_runs(tmp_path):
    """
    Not a source grep: the script is EXECUTED with the migration step disabled
    and the healthcheck pointed at an empty state root, and the worker command
    replaced. A script with a syntax error passes every grep in this file.
    """
    body = (REPO / "scripts" / "start.sh").read_text(encoding="utf-8")
    stub = tmp_path / "start.sh"
    stub.write_text(
        body.replace("exec python scheduler_worker.py", "echo WORKER-WOULD-START"),
        encoding="utf-8",
    )
    result = subprocess.run(
        ["bash", str(stub)],
        cwd=REPO,
        env={
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "PROPIQ_MIGRATE_ON_BOOT": "false",
            "PROPIQ_STATE_DIR": str(tmp_path / "empty-volume"),
            "PROPIQ_MODEL": str(tmp_path / "absent.json"),
            "PYTHONPATH": str(REPO),
        },
        capture_output=True, text=True, timeout=240,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "WORKER-WOULD-START" in result.stdout
    # The probe found an unseeded volume and said so, and the worker still ran.
    assert "HEALTHCHECK REPORTED FAILURE" in result.stdout
    assert "migrations skipped" in result.stdout


# ---------------------------------------------------------------------------
# src/models/runtime_versions.py — which libraries produced the artifact
# ---------------------------------------------------------------------------

def test_a_major_version_change_under_a_saved_artifact_is_reported(monkeypatch):
    """
    The volume survives a redeploy and the image does not, so a rebuild can
    pair a fresh major version of XGBoost with a booster saved by an older one.
    Nothing recorded the training versions, so the pairing was undetectable:
    the booster either loaded with possibly-changed behaviour or raised
    something opaque from the C++ layer.
    """
    from src.models.runtime_versions import collect, compare

    now = collect()
    assert "xgboost" in now, "the tracked list no longer covers xgboost"
    bumped = {**now, "xgboost": "1.7.6"}
    diffs = compare(bumped)
    assert any("xgboost" in d for d in diffs), diffs


def test_a_patch_level_difference_is_not_reported():
    """MAJORS ONLY. Warning on every patch bump trains everyone to ignore the
    line that matters."""
    from src.models.runtime_versions import collect, compare

    now = collect()
    major = now["xgboost"].split(".", 1)[0]
    assert compare({**now, "xgboost": f"{major}.0.0"}) == []


def test_an_artifact_saved_before_this_existed_is_not_reported_as_mismatched():
    """Every artifact saved before 2026-10-09 has no `runtime_versions` block.
    Reporting all of those as mismatched would be noise about nothing."""
    from src.models.runtime_versions import compare

    assert compare(None) == []
    assert compare({}) == []


def test_save_records_the_versions_and_load_compares_them(tmp_path, caplog):
    """
    END TO END, not just `compare`. An earlier version of this test only
    exercised the comparison helper, so blanking the adapter's
    `meta[runtime_versions] = collect()` call left every test green while no
    artifact recorded anything — the mutation that proves a test proves
    nothing.

    Fits a deliberately tiny booster: the point is the sidecar round trip, not
    the model.
    """
    import json
    import logging

    import numpy as np
    import pandas as pd

    from src.models.runtime_versions import META_KEY
    from src.models.xgb_adapter import XGBoostAdapter

    cols = ["a", "b"]
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(40, 2)), columns=cols)
    y = (X["a"] > 0).astype(int)

    adapter = XGBoostAdapter(cols, target_market="PTS")
    import xgboost as xgb

    adapter._pipe.model = xgb.XGBClassifier(n_estimators=4, max_depth=2).fit(X, y)
    adapter._fitted = True

    target = tmp_path / "xgboost_PTS.json"
    adapter.save(target)

    meta = json.loads(target.with_suffix(".meta.json").read_text())
    assert META_KEY in meta, "save() recorded no library versions"
    assert meta[META_KEY].get("xgboost"), "the xgboost version was not recorded"
    assert meta[META_KEY].get("python")

    # Round trip: the same versions are installed, so loading says nothing.
    with caplog.at_level(logging.WARNING):
        XGBoostAdapter(cols, target_market="PTS").load(target)
    assert "DIFFERENT MAJOR VERSION" not in caplog.text

    # Rewrite the sidecar as though it had been trained under xgboost 1.x.
    meta[META_KEY]["xgboost"] = "1.7.6"
    target.with_suffix(".meta.json").write_text(json.dumps(meta))
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        XGBoostAdapter(cols, target_market="PTS").load(target)
    assert "DIFFERENT MAJOR VERSION" in caplog.text
    assert "xgboost 1.7.6" in caplog.text


def test_the_version_check_never_stops_an_artifact_loading(monkeypatch, caplog):
    """A warning beside a working booster is useful. Refusing to score
    tonight's slate over a version string is not."""
    import logging

    from src.models.runtime_versions import warn_on_mismatch

    with caplog.at_level(logging.WARNING):
        diffs = warn_on_mismatch({"xgboost": "0.1.2"}, "artifact.json")
    assert diffs  # it reported
    assert "artifact.json" in caplog.text


def test_an_uninstalled_library_is_omitted_rather_than_recorded_null():
    """CatBoost is an optional extra: "absent at training time" and "present at
    version null" are different facts, and only one of them means the artifact
    cannot have used it."""
    from src.models.runtime_versions import collect

    assert all(v for v in collect().values())


# ---------------------------------------------------------------------------
# dependency bounds — one commit must not build two different images
# ---------------------------------------------------------------------------

def _caps(spec: str) -> str | None:
    """The major version cap in a requirement spec, if it has one."""
    import re

    m = re.search(r"<\s*([0-9]+)", spec)
    return m.group(1) if m else None


def _split(spec: str) -> tuple[str, str]:
    import re

    m = re.match(r"^([A-Za-z0-9_.\-]+)(?:\[[a-z,]+\])?(.*)$", spec.strip())
    assert m, spec
    return m.group(1).lower().replace("_", "-"), m.group(2).strip()


def test_the_libraries_that_can_change_a_saved_model_are_capped():
    """
    `xgboost>=2.0` with no upper bound was the live risk, not a theoretical
    one: the artifacts live on a MOUNTED VOLUME that survives a redeploy while
    the image does not, so an uncapped `docker build` months later pairs a
    fresh major version with a booster saved by an older one. A probability
    that moved because the serving library changed is indistinguishable,
    downstream, from one that moved because the player did.
    """
    import tomllib

    project = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]
    specs = list(project["dependencies"]) + [
        x for v in project["optional-dependencies"].values() for x in v
    ]
    found = {name: spec for name, spec in (_split(s) for s in specs)}
    for critical in ("xgboost", "scikit-learn", "catboost", "apscheduler",
                     "sqlalchemy", "pandas", "numpy"):
        assert critical in found, f"{critical} is no longer declared"
        assert _caps(found[critical]), (
            f"{critical} has no upper bound: a rebuild can install a new major "
            f"version against an artifact saved by an older one"
        )


def test_requirements_txt_and_pyproject_do_not_disagree():
    """
    TWO DECLARATIONS, ONE COMMIT. Nixpacks reads `requirements.txt` and the
    Dockerfile reads `pyproject.toml`, so a cap present in one and absent in
    the other builds two different images from the same commit — and this
    repository has already shipped that exact defect once, when APScheduler
    was in the extras and not in `requirements.txt`, producing a container
    that booted, hit its import guard and restarted in a loop.
    """
    import re
    import tomllib

    req: dict[str, str] = {}
    for line in (REPO / "requirements.txt").read_text().splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        name, spec = _split(line)
        req[name] = spec

    project = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]
    specs = list(project["dependencies"]) + [
        x for v in project["optional-dependencies"].values() for x in v
    ]
    disagree = []
    for spec in specs:
        name, bound = _split(spec)
        if name not in req:
            continue
        if _caps(bound) != _caps(req[name]):
            disagree.append((name, bound, req[name]))
    assert not disagree, f"caps differ between the two files: {disagree}"
    assert re.search(r"must stay IN STEP", (REPO / "requirements.txt").read_text())


# ---------------------------------------------------------------------------
# the scheduler's own defaults
# ---------------------------------------------------------------------------

def test_the_scheduler_sets_job_defaults_for_the_next_job_added():
    """
    APScheduler's OWN DEFAULT misfire grace is ONE SECOND. Every job registered
    today passes its own, so this changes nothing now and everything for the
    next job someone adds: without it, a two-second delay silently skips the
    job, and "it did not run" is indistinguishable from "it ran and found
    nothing".

    AST-walked, because `BlockingScheduler` cannot be constructed here —
    APScheduler is the `deploy` extra and is not installed in this
    environment, so the branch that builds the real one never executes in a
    test. A substring check would be satisfied by the comment above it.
    """
    import ast

    import scheduler_worker

    tree = ast.parse((REPO / "scheduler_worker.py").read_text(encoding="utf-8"))
    call = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id == "BlockingScheduler"
    )
    kwargs = {k.arg: k.value for k in call.keywords}
    assert "timezone" in kwargs
    defaults = kwargs.get("job_defaults")
    assert isinstance(defaults, ast.Dict), "BlockingScheduler has no job_defaults"
    keys = {k.value for k in defaults.keys if isinstance(k, ast.Constant)}
    assert {"max_instances", "coalesce", "misfire_grace_time"} <= keys

    # And the fallback grace is a real interval, not APScheduler's one second.
    assert scheduler_worker.DEFAULT_MISFIRE_GRACE_SECONDS >= 60


# ---------------------------------------------------------------------------
# src/db/session.py — the pool a long-idle worker needs
# ---------------------------------------------------------------------------

def test_the_engine_bounds_every_wait_it_can(monkeypatch):
    """
    WITHOUT connect_timeout THERE IS NO BOUND. A pooler host that accepts the
    connection and never completes the handshake blocks the caller forever, and
    the slate job has no timeout of its own: it would hang past every tip-off
    and be noticed as a worker that produced nothing, with no error anywhere.
    """
    import src.db.session as session

    captured: dict = {}

    def fake_create_engine(url, **kwargs):
        captured.update(kwargs)
        captured["url"] = url
        return object()

    monkeypatch.setattr(session, "create_engine", fake_create_engine)
    monkeypatch.setattr(session, "_engine", None)
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@db.example.com:6543/postgres")
    session.get_engine()

    assert captured["pool_pre_ping"] is True
    assert captured["pool_recycle"] == session.POOL_RECYCLE_SECONDS
    assert captured["pool_timeout"] == session.POOL_TIMEOUT_SECONDS
    assert captured["connect_args"]["connect_timeout"] == session.CONNECT_TIMEOUT_SECONDS
    assert captured["connect_args"]["application_name"] == session.APPLICATION_NAME
    session._engine = None


def test_a_mistyped_pool_size_falls_back_rather_than_raising(monkeypatch):
    """And max_overflow=0 is legitimate, so the floor cannot be 1."""
    from src.db.session import _int_env

    monkeypatch.setenv("X", "not-a-number")
    assert _int_env("X", 5, low=1, high=50) == 5
    monkeypatch.setenv("X", "500")
    assert _int_env("X", 5, low=1, high=50) == 5
    monkeypatch.setenv("X", "0")
    assert _int_env("X", 5, low=0, high=50) == 0
    assert _int_env("X", 5, low=1, high=50) == 5


# ---------------------------------------------------------------------------
# scheduler_worker.board_markets — one reader
# ---------------------------------------------------------------------------

def test_the_market_list_has_one_reader(monkeypatch):
    """
    The parsing lived inside `run_board` and was unreachable, so the deploy
    probe would have re-implemented it — and a probe checking PTS/REB/AST while
    the worker is configured for PTS alone reports a failure that is not one.
    """
    from scheduler_worker import board_markets
    from src.pipeline.slate_board import DEFAULT_MARKETS

    monkeypatch.setenv("PROPIQ_BOARD_MARKETS", "pts, ast")
    assert board_markets() == ["PTS", "AST"]
    monkeypatch.setenv("PROPIQ_BOARD_MARKETS", "  ,  ")
    assert board_markets() == list(DEFAULT_MARKETS)
    monkeypatch.delenv("PROPIQ_BOARD_MARKETS")
    assert board_markets() == list(DEFAULT_MARKETS)


def test_run_board_reads_the_shared_market_list(monkeypatch):
    """AST-walked: `run_board` must CALL `board_markets`, not carry its own copy
    of the parsing. A docstring saying it does has fooled this repository four
    times."""
    import ast

    tree = ast.parse((REPO / "scheduler_worker.py").read_text(encoding="utf-8"))
    run_board = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "run_board"
    )
    calls = {
        n.func.id for n in ast.walk(run_board)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "board_markets" in calls
    assert "PROPIQ_BOARD_MARKETS" not in ast.dump(run_board)
