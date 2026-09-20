"""
tests/test_orchestrator_contracts.py

These tests exist because v1 of main.py was written against ASSUMED
interfaces and shipped four wrong integrations. They assert that the
orchestrator's assumptions match the REAL PropIQ source.

They are skipped automatically when the full PropIQ tree isn't present
(this package is a patch, not a standalone repo) — but once merged, they
fail loudly if a refactor renames a column or changes a signature out
from under main.py.
"""

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

propiq_available = True
try:
    from src.features import builder, fatigue_logic
    from src.models.xgboost_pipeline import XGBoostPropPipeline
    from src.quant.contracts import MarketContext, market_ev_gate
except ImportError:
    propiq_available = False

pytestmark = pytest.mark.skipif(
    not propiq_available,
    reason="Full PropIQ tree not present — merge this patch into the main repo first.",
)


def test_fatigue_column_name_is_lowercase():
    """
    main.py's FATIGUE_COL must match what attach_fatigue_column actually
    emits. v1 assumed 'FATIGUE_MULTIPLIER' (uppercase) and would have
    raised its own RuntimeError on every run.
    """
    from main import FATIGUE_COL

    source = inspect.getsource(fatigue_logic.attach_fatigue_column)
    assert f'df["{FATIGUE_COL}"]' in source, (
        f"main.FATIGUE_COL={FATIGUE_COL!r} not assigned in attach_fatigue_column"
    )
    assert FATIGUE_COL.islower(), "Real column is lowercase fatigue_multiplier"


def test_builder_already_applies_fatigue():
    """
    build_feature_matrix calls attach_fatigue_column internally. The
    orchestrator must VERIFY, not re-apply — re-applying would
    double-count the multiplier into {stat}_L2.
    """
    source = inspect.getsource(builder)
    assert "attach_fatigue_column" in source, (
        "builder no longer calls attach_fatigue_column — main.py's verify-only "
        "approach is no longer valid and must be revisited."
    )


def test_no_baseline_projection_column_is_assumed():
    """
    v1 invented BASELINE_PROJECTION; real columns are {stat}_BASELINE /
    {stat}_L2. Checks executable code only — the module docstring
    legitimately mentions the old name when documenting the fix.
    """
    import ast

    import main

    tree = ast.parse(inspect.getsource(main))
    # Collect every string constant that is NOT a docstring
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc:
                docstrings.add(doc)

    code_strings = [
        n.value for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value not in docstrings
    ]
    identifiers = [n.id for n in ast.walk(tree) if isinstance(n, ast.Name)]
    attributes = [n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)]

    offenders = [s for s in code_strings + identifiers + attributes if "BASELINE_PROJECTION" in s]
    assert not offenders, (
        f"main.py uses BASELINE_PROJECTION in executable code {offenders} — that "
        f"column does not exist in the real feature builder."
    )

    source = inspect.getsource(main)
    assert "_L2" in source, "main.py should consume the real {stat}_L2 columns"


def test_xgboost_pipeline_requires_feature_cols():
    """
    XGBoostPropPipeline.__init__ takes feature_cols positionally. v1's bare
    XGBoostPropPipeline() always raised and silently skipped scoring.
    """
    sig = inspect.signature(XGBoostPropPipeline.__init__)
    params = list(sig.parameters)
    assert "feature_cols" in params
    assert sig.parameters["feature_cols"].default is inspect.Parameter.empty, (
        "feature_cols is required — main.py must pass it explicitly"
    )


def test_ev_gate_requires_two_way_american_odds():
    """
    BigDataBall game lines do NOT satisfy the gate. Confirm the gate
    abstains when two-way prop odds are absent, so the pipeline's
    abstention is genuine rather than incidental.
    """
    ctx_no_odds = MarketContext(game_id="0022500001", status="VALID")
    verdict = market_ev_gate(ctx_no_odds)
    assert verdict["status"] == "DATA_NOT_AVAILABLE"
    assert verdict["ev"] is None

    # A posted line is required too: EV is a claim about a probability AT a
    # number, so a ready verdict without one would be meaningless.
    ctx_full = MarketContext(
        game_id="0022500001",
        status="VALID",
        line=25.5,
        over_odds_american=-110,
        under_odds_american=-110,
    )
    assert market_ev_gate(ctx_full)["status"] == "READY_FOR_EVALUATION"


def test_main_calls_the_ev_gate():
    """v1 listed an EV stage it never actually invoked.

    Asserts on a real Call node, not on the substring. main.py's module
    docstring mentions `market_ev_gate` twice while documenting this very
    correction, so a substring check passes even with the executable call
    deleted — the exact regression this test exists to catch would go
    unnoticed. The sibling docstring-filtering test below makes the same
    distinction.
    """
    import ast

    import main

    tree = ast.parse(inspect.getsource(main))
    called = {
        node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", None)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    assert "market_ev_gate" in called, (
        "main.py must actually CALL market_ev_gate, not merely mention it"
    )
