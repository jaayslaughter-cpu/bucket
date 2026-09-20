"""
tests/test_projection_roundtrip.py

WHY THIS EXISTS: v2 shipped three silent-null bugs on the seam between
`main.assemble_projections` (which writes the DataFrame) and
`db.repository.persist_projections` (which reads it):

    assembler wrote      persister read          result
    ---------------      --------------          ------
    "BASELINE"           "BASELINE_PROJECTION"   always NULL
    (never written)      "FATIGUE_NOTES"         always NULL

The v2 contract tests guarded main.py against PropIQ's interfaces but
never tested this patch's OWN internal seam — same class of bug as v1,
one layer in. These tests close that gap by parsing both sides and
asserting every key the persister reads is actually produced.

Pure AST/static analysis — no DB connection, no PropIQ tree required, so
it runs everywhere including this patch directory.
"""

import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


REPO_ROOT = Path(__file__).parent.parent


def _keys_read_by_persist_projections() -> set[str]:
    """Every `r.get("KEY")` inside persist_projections."""
    source = (REPO_ROOT / "src" / "db" / "repository.py").read_text()
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "persist_projections":
            keys = set()
            for sub in ast.walk(node):
                if (
                    isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Attribute)
                    and sub.func.attr == "get"
                    and isinstance(sub.func.value, ast.Name)
                    and sub.func.value.id == "r"
                    and sub.args
                    and isinstance(sub.args[0], ast.Constant)
                ):
                    keys.add(sub.args[0].value)
            return keys
    raise AssertionError("persist_projections not found in repository.py")


def _keys_written_by_assemble_projections() -> set[str]:
    """
    Every column key assemble_projections produces — both dict literals
    inside the DataFrame construction AND subscript assignments like
    `out["LINE"] = ...` afterwards. Missing the second form would make
    this test produce false failures.
    """
    source = (REPO_ROOT / "main.py").read_text()
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "assemble_projections":
            keys = set()
            for sub in ast.walk(node):
                # form 1: {"KEY": value} inside pd.DataFrame({...})
                if isinstance(sub, ast.Dict):
                    for k in sub.keys:
                        if isinstance(k, ast.Constant) and isinstance(k.value, str):
                            keys.add(k.value)
                # form 2: out["KEY"] = value
                if isinstance(sub, ast.Assign):
                    for target in sub.targets:
                        if (
                            isinstance(target, ast.Subscript)
                            and isinstance(target.slice, ast.Constant)
                            and isinstance(target.slice.value, str)
                        ):
                            keys.add(target.slice.value)
            return keys
    raise AssertionError("assemble_projections not found in main.py")


def test_every_persisted_key_is_actually_produced():
    """
    The bug that shipped: persister read BASELINE_PROJECTION and
    FATIGUE_NOTES, assembler produced neither -> silent NULLs in Postgres
    with no error anywhere.
    """
    read = _keys_read_by_persist_projections()
    written = _keys_written_by_assemble_projections()

    missing = read - written
    assert not missing, (
        f"persist_projections reads key(s) {sorted(missing)} that "
        f"assemble_projections never writes. These would persist as silent "
        f"NULLs. Assembler produces: {sorted(written)}"
    )


def test_baseline_key_specifically():
    """Regression guard for the exact bug found in the v2 audit."""
    read = _keys_read_by_persist_projections()
    assert "BASELINE_PROJECTION" not in read, (
        "persist_projections is reading BASELINE_PROJECTION again — the "
        "assembler writes BASELINE."
    )
    assert "BASELINE" in read, "persist_projections should read the BASELINE key"


def test_fatigue_notes_is_produced():
    """Second silent-null: FATIGUE_NOTES was read but never written."""
    written = _keys_written_by_assemble_projections()
    assert "FATIGUE_NOTES" in written, (
        "assemble_projections must emit FATIGUE_NOTES — persist_projections "
        "reads it into the fatigue_notes column."
    )


def test_load_player_panel_actually_uses_its_parameters():
    """
    Third bug: slate_date and lookback_days were accepted and silently
    ignored, so every call loaded the whole table — no date bound, no
    leakage guard at the query level.
    """
    source = (REPO_ROOT / "src" / "db" / "repository.py").read_text()
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "load_player_panel":
            body_names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
            assert "slate_date" in body_names, "load_player_panel ignores slate_date"
            assert "lookback_days" in body_names, "load_player_panel ignores lookback_days"

            # And it must actually constrain the query
            has_where = any(
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Attribute)
                and sub.func.attr == "where"
                for sub in ast.walk(node)
            )
            assert has_where, (
                "load_player_panel builds no .where() clause — the date "
                "parameters are not reaching the query."
            )
            return
    raise AssertionError("load_player_panel not found")


def test_fatigue_notes_helper_uses_real_flag_columns():
    """
    _fatigue_notes must read the columns fatigue_logic actually emits
    (is_back_to_back / is_3_in_4 / is_4_in_5), not invented ones.
    """
    source = (REPO_ROOT / "main.py").read_text()
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_fatigue_notes":
            body = ast.get_source_segment(source, node) or ""
            for real_col in ("is_back_to_back", "is_3_in_4", "is_4_in_5"):
                assert real_col in body, f"_fatigue_notes does not reference {real_col}"
            return
    raise AssertionError("_fatigue_notes not found in main.py")
