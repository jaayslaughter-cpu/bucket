"""O2 — finding the artifact that was actually trained.

`score_prob_over` defaulted to `models/xgb_prop_over.json`, a directory that
does not exist. `train-stats` writes `xgboost_{MARKET}.json` plus its
`.meta.json` under `config/model_comparison.yaml`'s `artifacts_dir`. The two
never coincided, and `scheduler_worker.run_slate` calls `main.main([])` with no
`--model` — so a scheduled run scored nothing EVEN AFTER a model was trained
into the right place, and said only "no fitted model at models/...".
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import main
from main import ENV_MODEL, MODEL_ARTIFACT_DEFAULT, resolve_model_artifact


@pytest.fixture(autouse=True)
def _no_inherited_env(monkeypatch):
    monkeypatch.delenv(ENV_MODEL, raising=False)


def artifact(directory: Path, market: str = "PTS", *, sidecar: bool = True) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    model = directory / f"xgboost_{market}.json"
    model.write_text("{}", encoding="utf-8")
    if sidecar:
        (directory / f"xgboost_{market}.meta.json").write_text(
            json.dumps({"feature_cols": ["PTS_L5"], "target_market": market}),
            encoding="utf-8",
        )
    return model


def point_at(monkeypatch, directory: Path) -> None:
    """
    Make the comparison config report ``directory`` as its artifacts_dir.

    `resolve_model_artifact` imports `load_comparison_config` INSIDE the
    function, so patching the attribute on the module object is what reaches
    it — patching a name in `main` would not.
    """
    import src.models.compare as compare

    monkeypatch.setattr(
        compare, "load_comparison_config",
        lambda *a, **k: {"artifacts_dir": str(directory)},
    )


# --- the resolution order --------------------------------------------------

def test_an_explicit_model_flag_wins_over_everything(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_MODEL, str(tmp_path / "from_env.json"))
    path, how = resolve_model_artifact(tmp_path / "explicit.json")
    assert path == tmp_path / "explicit.json"
    assert how == "--model"


def test_the_environment_variable_is_read_when_no_flag_is_given(monkeypatch, tmp_path):
    """
    This is what makes a SCHEDULED run able to find a model: the worker calls
    main.main([]) with no arguments, so an env var is the only channel.
    """
    target = tmp_path / "on_the_volume.json"
    monkeypatch.setenv(ENV_MODEL, str(target))
    path, how = resolve_model_artifact()
    assert path == target
    assert how == ENV_MODEL


def test_the_comparison_artifacts_dir_is_searched(monkeypatch, tmp_path):
    """Where train-stats actually writes — the half that was never read."""
    point_at(monkeypatch, tmp_path)
    expected = artifact(tmp_path)
    path, how = resolve_model_artifact()
    assert path == expected
    assert str(tmp_path) in how


def test_a_market_narrows_the_search(monkeypatch, tmp_path):
    point_at(monkeypatch, tmp_path)
    artifact(tmp_path, "PTS")
    reb = artifact(tmp_path, "REB")
    path, _how = resolve_model_artifact(market="reb")
    assert path == reb


def test_the_newest_artifact_wins_when_several_match(monkeypatch, tmp_path):
    import os
    import time

    point_at(monkeypatch, tmp_path)
    old = artifact(tmp_path, "PTS")
    new = artifact(tmp_path, "REB")
    os.utime(old, (time.time() - 10_000, time.time() - 10_000))
    path, _how = resolve_model_artifact()
    assert path == new


def test_an_artifact_without_its_sidecar_is_not_a_candidate(monkeypatch, tmp_path, caplog):
    """
    Scoring without the sidecar's feature_cols is refused downstream anyway, so
    offering one here would resolve a path that cannot be scored with.
    """
    point_at(monkeypatch, tmp_path)
    artifact(tmp_path, "PTS", sidecar=False)
    with caplog.at_level("WARNING"):
        path, how = resolve_model_artifact()
    assert path is None or path == MODEL_ARTIFACT_DEFAULT
    assert "no .meta.json sidecar" in caplog.text


def test_nothing_found_returns_a_reason_naming_every_path_tried(monkeypatch, tmp_path):
    point_at(monkeypatch, tmp_path / "empty")
    path, reason = resolve_model_artifact()
    assert path is None
    assert ENV_MODEL in reason
    assert "empty" in reason
    assert str(MODEL_ARTIFACT_DEFAULT) in reason
    assert "train-stats" in reason, "the reason should say how to produce one"


def test_a_broken_comparison_config_does_not_take_the_resolver_down(monkeypatch, tmp_path):
    import src.models.compare as compare

    monkeypatch.setattr(
        compare, "load_comparison_config",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("bad yaml")),
    )
    path, reason = resolve_model_artifact()
    assert path is None
    assert "data/external/model_runs/comparison" in reason


# --- scoring handles the absence ------------------------------------------

def test_score_prob_over_abstains_on_a_resolved_none():
    out = main.score_prob_over(
        pd.DataFrame({"PTS_L5": [20.0]}), pd.DataFrame({"line": [24.5]}), None
    )
    assert out.isna().all()


def test_score_prob_over_no_longer_defaults_to_a_path_nobody_writes():
    import inspect

    default = inspect.signature(main.score_prob_over).parameters["model_path"].default
    assert default is None, (
        "defaulting to models/xgb_prop_over.json is the bug: it is a path "
        "nothing in this repository writes to"
    )


# --- the worker needs no argv plumbing ------------------------------------

def test_the_scheduled_worker_reaches_the_resolver_through_the_environment():
    """
    `run_slate` calls main.main(argv or []) in-process, so PROPIQ_MODEL set on
    the service reaches the resolver with no argument passing at all. That is
    why the worker needs no --model and this test asserts the shape rather
    than a string in the file.
    """
    import scheduler_worker as w

    body = Path(w.__file__).read_text(encoding="utf-8")
    assert "main.main(argv or [])" in body
    assert main.resolve_model_artifact.__module__ == "main"
    # os.environ is the shared channel; both read the same process environment.
    assert "import os" in body


def test_the_model_path_is_recorded_in_the_run_summary():
    body = Path(main.__file__).read_text(encoding="utf-8")
    assert 'stage_summary["model"]' in body, (
        "which artifact scored a run is provenance; a run that abstained for "
        "want of a model should say so in pipeline_runs"
    )
    assert '"resolved_by"' in body


def test_the_resolver_does_not_read_a_registry_nothing_writes():
    """
    `models/artifact_registry.py` would be the right index for this and nothing
    writes to it. Resolving through a dead module is how `oddspapi` stayed in
    the source precedence for months.
    """
    body = Path(main.__file__).read_text(encoding="utf-8")
    block = body[body.index("def resolve_model_artifact"):]
    block = block[: block.index("\ndef ")]
    assert "read_registry" not in block
    assert "artifact_registry" in block, "the decision should be stated, not silent"
