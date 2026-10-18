"""
Seeding must verify the resolver can see what it just wrote.

FOUND BY MEASUREMENT, 2026-10-16, while seeding the artifacts for the Railway
deploy. `seed_volume` copies to ``utils.volume.artifact_dir_on(state_root)``;
`main.resolve_model_artifact` reads ``artifacts_dir`` from
``config/model_comparison.yaml``, which ships RELATIVE
(``data/external/model_runs/comparison``) and is therefore resolved against the
WORKING DIRECTORY. Those two agree only when the state root sits at
``<cwd>/data`` -- i.e. the documented ``/app/data`` mount with ``WORKDIR=/app``:

    state root /app/data                -> /app/data/external/model_runs/comparison   (resolver looks here)
    state root /var/lib/postgresql/data -> /var/lib/postgresql/data/external/...       (resolver never looks)

The deploy's volume is at ``/var/lib/postgresql/data``. Seeding it printed
three reassuring "copy" lines and left the resolver finding nothing -- exactly
the failure this script's own docstring says it exists to prevent:

    "A seeding tool that wrote to a directory of its own choosing
     (/app/data/models/, say) would report success and leave the resolver
     finding nothing."

It prevented that for its DESTINATION and never checked the OUTCOME. So the
guard asks the resolver, after copying, whether the file it will actually score
with is the one just written.

NOT MERELY "SOMETHING RESOLVED". With a repo copy of the artifacts present, the
resolver answers happily -- with a file on the ephemeral container layer that a
redeploy destroys. The guard requires the resolved path to live UNDER THE STATE
ROOT, which is the only thing seeding was for.

RESEARCH_ONLY. This copies files and then checks where they landed.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.seed_volume import seed
from src.utils.volume import ENV_STATE_DIR, artifact_dir_on


def _usable_family(directory: Path, market: str, rows: int = 200_678) -> None:
    """An xgboost artifact family big enough that the demo guard accepts it."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"xgboost_{market}.json").write_text("{}", encoding="utf-8")
    (directory / f"xgboost_{market}.mean.json").write_text("{}", encoding="utf-8")
    (directory / f"xgboost_{market}.meta.json").write_text(
        json.dumps({"n_train_rows": rows, "feature_cols": ["A"],
                    "saved_at_utc": "2026-10-09T02:29:15+00:00",
                    "train_end_date": "2026-01-13"}),
        encoding="utf-8",
    )


def test_a_seed_the_resolver_cannot_see_is_an_error(tmp_path, monkeypatch, capsys):
    """
    THE PRODUCTION SHAPE: a state root that is not ``<cwd>/data``. The copy
    succeeds and the resolver looks somewhere else entirely.
    """
    source = tmp_path / "trained"
    _usable_family(source, "PTS")
    volume = tmp_path / "var" / "lib" / "postgresql" / "data"
    monkeypatch.setenv(ENV_STATE_DIR, str(volume))
    monkeypatch.delenv("PROPIQ_MODEL", raising=False)

    code, lines = seed(source, apply=True, force=False, allow_demo=False)
    text = "\n".join(lines)

    # The files really were copied -- this is not a copy failure.
    assert (artifact_dir_on(volume) / "xgboost_PTS.json").exists()

    assert code != 0, f"a seed the resolver cannot see exited 0:\n{text}"
    assert "WILL NOT BE USED" in text or "cannot see" in text.lower(), text
    assert str(volume) in text, "the message does not name the state root seeded"


def test_the_guard_names_the_mount_that_would_work(tmp_path, monkeypatch):
    """An operator reading the failure needs the remedy, not just the fault."""
    source = tmp_path / "trained"
    _usable_family(source, "PTS")
    monkeypatch.setenv(ENV_STATE_DIR, str(tmp_path / "elsewhere"))
    monkeypatch.delenv("PROPIQ_MODEL", raising=False)

    _code, lines = seed(source, apply=True, force=False, allow_demo=False)
    text = "\n".join(lines)
    assert "/app/data" in text, (
        "the failure does not say where the volume has to be mounted for the "
        f"resolver and the seeder to agree:\n{text}"
    )


def test_a_seed_the_resolver_does_see_is_fine(tmp_path, monkeypatch):
    """
    The success branch. ``$PROPIQ_MODEL`` pointing at the seeded file is the
    other configuration in which the resolver and the state root agree, and it
    is the one a test can arrange without changing the working directory.
    """
    source = tmp_path / "trained"
    _usable_family(source, "PTS")
    volume = tmp_path / "volume"
    monkeypatch.setenv(ENV_STATE_DIR, str(volume))
    monkeypatch.setenv(
        "PROPIQ_MODEL", str(artifact_dir_on(volume) / "xgboost_PTS.json")
    )

    code, lines = seed(source, apply=True, force=False, allow_demo=False)
    text = "\n".join(lines)
    assert code == 0, text
    assert "WILL NOT BE USED" not in text
    assert "verified" in text.lower(), (
        f"a successful seed says nothing about having been verified:\n{text}"
    )


def test_an_override_pointing_off_the_volume_is_caught(tmp_path, monkeypatch):
    """
    The other way a seed goes to waste: ``$PROPIQ_MODEL`` wins over
    ``artifacts_dir``, so a stale override means the freshly seeded family is
    copied and then ignored. The guard checks what WILL be scored, so it sees
    this too.
    """
    source = tmp_path / "trained"
    _usable_family(source, "PTS")
    stray = tmp_path / "stray" / "xgboost_PTS.json"
    stray.parent.mkdir(parents=True)
    stray.write_text("{}", encoding="utf-8")
    (stray.parent / "xgboost_PTS.meta.json").write_text(
        json.dumps({"n_train_rows": 200_678, "feature_cols": ["A"]}),
        encoding="utf-8",
    )
    monkeypatch.setenv(ENV_STATE_DIR, str(tmp_path / "volume"))
    monkeypatch.setenv("PROPIQ_MODEL", str(stray))

    code, lines = seed(source, apply=True, force=False, allow_demo=False)
    text = "\n".join(lines)
    assert code != 0, text
    assert str(stray) in text, (
        f"the message does not name the override that wins instead:\n{text}"
    )


def test_a_dry_run_does_not_claim_to_have_verified_anything(tmp_path, monkeypatch):
    """Nothing was written, so there is nothing to verify and no failure to
    report: the guard must not fire on a report-only run."""
    source = tmp_path / "trained"
    _usable_family(source, "PTS")
    monkeypatch.setenv(ENV_STATE_DIR, str(tmp_path / "var" / "data"))
    monkeypatch.delenv("PROPIQ_MODEL", raising=False)

    code, lines = seed(source, apply=False, force=False, allow_demo=False)
    text = "\n".join(lines)
    assert code == 0, text
    assert "WILL NOT BE USED" not in text
    assert "Nothing was written" in text
