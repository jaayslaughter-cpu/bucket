"""
Training and seeding the scoring artifact.

Two things blocked it, and neither was the model.

ONE: `train-stats` had no `--panel`. `evaluate` and `scripts/feature_ab.py`
both accepted a prebuilt feature matrix; the one command that produces the
artifact the whole pipeline needs did not. So on any machine where
stats.nba.com is denied — this environment, and any CI — the artifact could
not be trained at all, and `scheduler_worker.check_model_artifact()` reported
an ERROR at boot with no way to clear it.

TWO: `main.py`'s `--bigdataball` default named a file that does not exist —
`..._Team-Stats__1_.xlsx`, where the export is `..._Team-Stats.xlsx`.
`load_bigdataball_workbook` raises on a missing path, so the slate failed
outright with defaults. Loudly, which is the one mercy — but it also meant the
team and market frames never reached `build_feature_matrix`, so the `DEF_*`,
`MKT_*` and Elo columns a trained artifact's contract names could not be built
at all, and an artifact trained WITH them would have failed its own feature
contract on every live row.

The artifacts themselves are gitignored (`data/**`), so they cannot be
committed and these tests do not assert their presence — a test that required
a trained artifact would fail on every fresh checkout. What is pinned is the
two things that made training possible, and the contract the artifact has to
satisfy.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


# --- the CLI can train without the network ------------------------------

def test_train_stats_accepts_a_prebuilt_panel():
    """
    The option that makes the artifact reachable offline. Checked on the
    signature rather than by running a fit, which would take minutes.
    """
    import inspect

    from scripts.nba_model_cli import train_stats

    params = inspect.signature(train_stats).parameters
    assert "panel" in params, (
        "train-stats cannot take a prebuilt feature matrix, so it cannot run "
        "where stats.nba.com is denied"
    )


def test_the_panel_option_is_actually_passed_to_the_loader():
    """
    An option parsed and dropped is worse than an absent one: the command
    would accept --panel, ignore it, and try the network anyway.
    """
    import inspect

    from scripts import nba_model_cli

    source = inspect.getsource(nba_model_cli.train_stats)
    assert "_load_real_or_demo(demo, seasons, season_type, panel)" in source, (
        "--panel is parsed but never reaches _load_real_or_demo"
    )


def test_the_loader_reads_the_panel_it_is_given(tmp_path):
    """Behavioural: a frame on disk comes back, and is not rebuilt."""
    from scripts.nba_model_cli import _load_real_or_demo

    frame = pd.DataFrame({
        "PLAYER_ID": ["1"], "GAME_ID": ["g"], "GAME_DATE": [pd.Timestamp("2025-01-01")],
        "PTS": [20.0], "PTS_L10": [18.0],
    })
    path = tmp_path / "panel.parquet"
    frame.to_parquet(path, index=False)

    out, is_demo = _load_real_or_demo(False, None, None, str(path))
    assert is_demo is False
    assert len(out) == 1 and "PTS_L10" in out.columns


def test_a_missing_panel_is_refused_rather_than_silently_falling_back(tmp_path):
    """
    Falling back to the network loader on a typo'd path would hit a denied
    endpoint and report THAT as the failure, hiding the real cause.
    """
    from scripts.nba_model_cli import _load_real_or_demo

    with pytest.raises(SystemExit):
        _load_real_or_demo(False, None, None, str(tmp_path / "nope.parquet"))


# --- the workbook default ------------------------------------------------

def test_the_bigdataball_default_names_a_file_the_repo_actually_uses():
    """
    THE DEFAULT POINTED AT A FILE THAT DOES NOT EXIST. `__1_.xlsx` was a
    download suffix that never matched the export on disk, and
    `load_bigdataball_workbook` raises on a missing path, so `python main.py`
    with no arguments could not get past step [2].

    BY AST, NOT BY GREP. The first version of this test asserted the string
    "__1_.xlsx" was absent from main.py — and the comment recording that the
    old default was wrong CONTAINS it, so the test failed on the explanation
    of its own fix. The mirror image of a comment masking a defect, and the
    same lesson: read the code, not the prose around it.
    """
    import ast

    tree = ast.parse((ROOT / "main.py").read_text(encoding="utf-8"))
    default: str | None = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "add_argument"):
            continue
        if not (node.args and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "--bigdataball"):
            continue
        for kw in node.keywords:
            if kw.arg != "default":
                continue
            # default=os.environ.get("BIGDATABALL_XLSX", "<path>")
            assert isinstance(kw.value, ast.Call), "the default is no longer overridable"
            literals = [a.value for a in kw.value.args if isinstance(a, ast.Constant)]
            assert literals and literals[0] == "BIGDATABALL_XLSX"
            default = literals[1] if len(literals) > 1 else None

    assert default, "could not read the --bigdataball default from main.py"
    assert "__1_" not in default, (
        f"the default workbook path is back to a name that does not exist: {default!r}"
    )
    assert default.endswith("2025-2026_NBA_Box_Score_Team-Stats.xlsx"), default


def test_the_workbook_default_is_overridable_by_environment():
    """A licensed export lives wherever the operator put it."""
    source = (ROOT / "main.py").read_text(encoding="utf-8")
    assert 'os.environ.get(\n                            "BIGDATABALL_XLSX"' in source


def test_a_missing_workbook_raises_rather_than_building_a_panel_without_it():
    """
    The failure has to be loud. A workbook that silently did not load would
    build a panel with no DEF_*, MKT_* or Elo columns — and an artifact whose
    contract names them then abstains on every row, which is the
    looks-healthy-produces-nothing shape.
    """
    import main

    with pytest.raises(FileNotFoundError, match="workbook not found"):
        main.ingest_market_lines(Path("data/external/bigdataball/does-not-exist.xlsx"),
                                 persist=False)


# --- the contract the artifact must satisfy -----------------------------

def test_a_trained_artifact_is_only_usable_where_its_features_can_be_built():
    """
    MEASURED WHEN THE ARTIFACTS WERE SEEDED, and the reason the workbook
    default mattered. The PTS contract resolves 38 columns against a panel
    built WITH the workbook and 27 against one built without it: 11 of them —
    Elo, MKT_* and DEF_* — come from the workbook alone. Train with them and
    score without, and the contract check rejects the artifact on every row.

    Asserted as a relationship rather than as two magic numbers, so it still
    means something when the feature list changes.
    """
    import scripts.verify_wiring as vw
    from src.features.builder import build_feature_matrix
    from src.models.compare import resolve_feature_cols
    from src.models.labels import default_feature_cols

    without_workbook = build_feature_matrix(vw.synthetic_panel())
    wanted = list(default_feature_cols("PTS"))
    resolved, dropped = resolve_feature_cols(without_workbook, wanted)

    workbook_only = [
        c for c in dropped
        if c.startswith(("MKT_", "DEF_")) or "ELO" in c.upper()
    ]
    assert workbook_only, (
        "a panel built without the workbook now resolves the MKT_/DEF_/Elo "
        "columns, so either the builder gained another source or this test is "
        "measuring the wrong thing"
    )
    assert len(resolved) < len(wanted)


@pytest.mark.skipif(
    not (ROOT / "data/external/model_runs/comparison/xgboost_PTS.meta.json").exists(),
    reason="no artifact trained in this checkout; data/** is gitignored",
)
def test_a_seeded_artifact_is_not_the_demo_one():
    """
    Runs only where an artifact has been trained. check_model_artifact logs an
    ERROR -- not a warning -- when train_row_count is small enough to be the
    demo artifact, because scoring a live slate with a model fit on a thousand
    synthetic rows is worse than abstaining.
    """
    import json

    meta = json.loads(
        (ROOT / "data/external/model_runs/comparison/xgboost_PTS.meta.json").read_text()
    )
    assert meta["target_market"] == "PTS"
    assert meta["train_row_count"] > 50_000, meta["train_row_count"]
    assert meta["feature_cols"], "the sidecar carries no feature contract"
    assert meta["train_start_date"] < meta["train_end_date"]
