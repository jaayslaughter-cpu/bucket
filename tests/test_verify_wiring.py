"""The wiring verifier's own fixtures, pinned to the real APIs.

Writing `scripts/verify_wiring.py` produced four FAILs that were bugs in the
CHECKS, not in the code under audit: PickemLeg takes a MarketContext rather than
a market string, the sizing entry point is recommended_units_binary, the EV gate
needs a posted line as well as a two-way pair, and the recorder reads `source`
only from the captured board. A check that fails for the wrong reason is worse
than no check — it spends the reader's attention and then is ignored.

So the fixtures are pinned here. If a signature moves, this fails rather than
the verifier quietly reporting a break that is not there.
"""

from __future__ import annotations

import inspect

import pandas as pd
import pytest

from scripts import verify_wiring as vw


def test_the_synthetic_panel_carries_what_the_repository_returns():
    """
    The panel fixture has to match ``load_player_panel``'s output, or section 1
    proves nothing about production.
    """
    import re

    repo = (vw.ROOT / "src" / "db" / "repository.py").read_text(encoding="utf-8")
    block = repo[repo.index("def load_player_panel"):]
    block = block[: block.index("\ndef ")] if "\ndef " in block[10:] else block
    produced = set(re.findall(r'"([A-Z_0-9]+)":\s*r\.', block))
    panel = vw.synthetic_panel()
    missing = sorted(produced - set(panel.columns))
    assert not missing, (
        f"load_player_panel returns {missing}, which the fixture does not have"
    )


def test_the_panel_fixture_keeps_ids_as_strings():
    """db/models.py declares them String(32); an int fixture would test nothing."""
    panel = vw.synthetic_panel()
    assert panel["PLAYER_ID"].map(type).eq(str).all()
    assert panel["GAME_ID"].map(type).eq(str).all()
    assert pd.api.types.is_datetime64_any_dtype(panel["GAME_DATE"])


# --- the four signatures a wrong fixture got wrong once --------------------

def test_pickem_leg_takes_a_market_context_not_a_string():
    from src.quant.dfs_entry import PickemLeg

    fields = set(inspect.signature(PickemLeg).parameters)
    assert "market" in fields
    assert "player_name" not in fields, (
        "the player's name lives on the MarketContext, not on the leg"
    )
    assert "line" not in fields, "the line lives on the MarketContext"


def test_route_pickem_entry_takes_the_payout_structure_first():
    from src.quant.dfs_entry import route_pickem_entry

    params = list(inspect.signature(route_pickem_entry).parameters)
    assert params[0] == "structure"
    assert params[1] == "legs"


def test_the_sizing_entry_point_is_named_what_the_verifier_calls():
    from src.quant import advisory_sizing

    assert hasattr(advisory_sizing, "recommended_units_binary")
    assert not hasattr(advisory_sizing, "advisory_size"), (
        "if a wrapper is added, update verify_wiring's check with it"
    )


def test_the_advisory_size_dict_uses_the_keys_the_verifier_reads():
    from src.quant.advisory_sizing import recommended_units_binary

    d = recommended_units_binary(0.58, 1.91).as_dict()
    for key in ("RECOMMENDED_UNITS", "KELLY_FRACTION_APPLIED",
                "FULL_KELLY_FRACTION", "AUTO_PLACED"):
        assert key in d, f"verify_wiring reads {key}"
    assert d["AUTO_PLACED"] is False


def test_the_ev_gate_needs_a_line_as_well_as_a_pair():
    """
    The check that read the gate's correct refusal as a break. Pinned so the
    fixture and the gate's actual requirements cannot drift apart again.
    """
    from datetime import datetime, timezone

    from src.quant.contracts import MarketContext, market_ev_gate

    base = dict(
        game_id="g1", market="PTS", player_name="P", source="propline",
        captured_at_utc=datetime.now(timezone.utc), status="VALID",
        over_odds_american=-110, under_odds_american=-110,
    )
    assert market_ev_gate(MarketContext(**base, line=24.5))["status"] == "READY_FOR_EVALUATION"
    assert market_ev_gate(MarketContext(**base, line=None))["status"] != "READY_FOR_EVALUATION"


def test_the_recorder_takes_source_only_from_the_captured_board():
    from src.settlement.recorder import LINE_FIELDS, LINE_JOIN_KEYS

    assert "source" in LINE_FIELDS
    assert LINE_JOIN_KEYS == ("player_name", "market"), (
        "the verifier's name-mismatch check depends on this being the join"
    )


def test_the_calibration_producer_emits_what_the_gate_reads():
    """
    A hand-written fixture used `generated_at`; the gate reads
    `evidence_as_of` / `report_timestamp_pt` / `report_timestamp`. The verifier
    now builds its fixture with the real producer, which this pins.
    """
    import numpy as np

    from src.quant.dfs_payouts import ProbabilitySource
    from src.quant.publication_gate import calibration_gate
    from src.settlement.calibration import calibration_from_graded_rows

    rng = np.random.default_rng(7)
    rows = []
    for _ in range(400):
        p = float(rng.uniform(0.05, 0.95))
        rows.append({
            "outcome_status": "WIN" if rng.random() < p else "LOSS",
            "prob_over": p, "predicted_side": "OVER", "predicted_line": 24.5,
        })
    report = calibration_from_graded_rows(rows)
    assert report["status"] == "OK", report.get("reason")
    verdict = calibration_gate(report, probability_source=ProbabilitySource.MODEL)
    assert verdict.allowed, (
        f"the producer's own output does not satisfy the gate: {verdict.reason}"
    )


# --- the verifier runs, and its verdict is not vacuous --------------------

def test_the_verifier_runs_end_to_end_and_reports_both_outcomes(capsys):
    """
    It must not crash, and it must produce PASSes as well as FAILs: a run that
    is all FAIL usually means the harness is broken, and one that is all PASS
    on this tree would mean the checks stopped looking.
    """
    code = vw.main(["--section", "1", "--section", "3", "--section", "6"])
    out = capsys.readouterr().out
    assert "passed," in out
    assert "PASS" in out
    assert isinstance(code, int) and code >= 0


def test_the_verifier_reports_a_check_that_raises_as_a_failure():
    """
    `guard` exists so a broken check is a FAIL with its traceback type, never a
    crash that hides every check after it.
    """
    report = vw.Report()
    report.section("probe")

    def explode() -> None:
        raise RuntimeError("boom")

    vw.guard(report, "deliberately broken")(explode)
    assert report.count("FAIL") == 1
    assert "RuntimeError: boom" in report.results[0].detail


@pytest.mark.parametrize("section", sorted(vw.SECTIONS))
def test_every_section_is_runnable_on_its_own(section):
    code = vw.main(["--section", section])
    assert isinstance(code, int)
