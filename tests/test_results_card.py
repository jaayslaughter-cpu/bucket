"""O9 — the daily results card.

`src/notify/discord.py` had four embed builders and none reported a settled
day. The gap had been open since the 2026-09-28 readiness audit, which named it
"P1 — daily W/L reconciliation embed."

A results card is the easiest place in a research system to start implying a
profit claim, so the tests below spend most of their effort on what it REFUSES:
a strike rate under the minimum sample, an ROI figure with no stake behind it,
and a CLV number without its caveat.
"""

from __future__ import annotations

import json

import pytest

from src.notify.discord import DiscordDispatchError, build_win_loss_embed
from src.settlement.metrics import MIN_SAMPLE_FOR_RATE, PerformanceSummary


def _summary(**record) -> PerformanceSummary:
    s = PerformanceSummary()
    for key, value in record.items():
        setattr(s.record, key, value)
    return s


def _text(embed) -> str:
    return json.dumps(embed).lower()


def _field(embed, name: str) -> str:
    for f in embed["fields"]:
        if f["name"] == name:
            return f["value"]
    raise AssertionError(f"no {name!r} field in {[f['name'] for f in embed['fields']]}")


# --- it exists and reports the record --------------------------------------

def test_the_builder_exists_and_is_the_fifth():
    """O9's whole content: four builders, none of them a results card."""
    import re
    from pathlib import Path

    body = (Path(__file__).parent.parent / "src" / "notify" / "discord.py").read_text()
    builders = set(re.findall(r"^def (build_\w+_embed)", body, re.M))
    assert "build_win_loss_embed" in builders
    assert len(builders) == 5, sorted(builders)


def test_the_record_is_reported_as_w_l_p():
    embed = build_win_loss_embed(
        _summary(graded_n=12, wins=7, losses=4, pushes=1, decided_n=11)
    )
    assert "7-4-1" in _field(embed, "Record")
    assert "12 graded" in _field(embed, "Record")


def test_pending_props_are_named_rather_than_folded_into_the_record():
    embed = build_win_loss_embed(_summary(graded_n=5, wins=3, losses=2, pending=9))
    assert "9 still pending" in _field(embed, "Record")


def test_voids_are_shown_when_present_and_hidden_when_not():
    assert "void" in _field(
        build_win_loss_embed(_summary(graded_n=4, wins=2, losses=1, voids=1)), "Record"
    )
    assert "void" not in _field(
        build_win_loss_embed(_summary(graded_n=3, wins=2, losses=1)), "Record"
    )


def test_a_day_with_nothing_graded_says_so_rather_than_going_silent():
    embed = build_win_loss_embed(_summary(graded_n=0), slate_date="2026-10-04")
    assert "nothing was graded" in _text(embed)
    assert "not a day with no value" in _text(embed)
    assert embed["title"] == "Results — 2026-10-04"


# --- what it refuses -------------------------------------------------------

def test_a_strike_rate_under_the_minimum_sample_is_withheld_with_its_reason():
    """
    `MIN_SAMPLE_FOR_RATE` is 30 and exists because a rate over a handful of
    props is noise. Printing 66.7% over three props is the single most
    misleading number this card could carry.
    """
    embed = build_win_loss_embed(
        _summary(graded_n=3, wins=2, losses=1, decided_n=3, strike_rate_pct=66.7),
        min_sample_for_rate=MIN_SAMPLE_FOR_RATE,
    )
    rate = _field(embed, "Strike rate")
    assert "withheld" in rate
    assert "66.7" not in rate, "the withheld figure leaked into the card anyway"
    assert "3 decided" in rate and "30" in rate


def test_a_strike_rate_over_the_minimum_sample_is_shown():
    """Withholding has to be about the sample, not a refusal to ever report."""
    embed = build_win_loss_embed(
        _summary(graded_n=40, wins=22, losses=18, decided_n=40, strike_rate_pct=55.0),
        min_sample_for_rate=MIN_SAMPLE_FOR_RATE,
    )
    rate = _field(embed, "Strike rate")
    assert "55.0%" in rate and "40 decided" in rate
    assert "withheld" not in rate


def test_roi_is_not_computable_without_a_recorded_stake():
    """
    Nothing in this pipeline writes a stake — the recorder deliberately never
    writes `stake_units`. "ROI 0.00%" would read as a flat day rather than as
    no data.
    """
    embed = build_win_loss_embed(_summary(graded_n=20, wins=11, losses=9, decided_n=20))
    roi = _field(embed, "ROI")
    assert "not computable" in roi
    assert "no stake is recorded" in roi
    assert "0.00%" not in roi


def test_roi_is_reported_when_a_stake_was_actually_recorded():
    s = _summary(graded_n=40, wins=22, losses=18, decided_n=40)
    s.roi.staked_units = 40.0
    s.roi.profit_units = 2.0
    s.roi.roi_pct = 5.0
    roi = _field(build_win_loss_embed(s), "ROI")
    assert "+5.00%" in roi and "40.00 unit(s) staked" in roi


def test_the_metrics_layer_s_own_roi_note_travels_with_the_number():
    s = _summary(graded_n=10, wins=6, losses=4, decided_n=10)
    s.roi.note = "No priced props settled — ROI not computable."
    assert "no priced props settled" in _field(build_win_loss_embed(s), "ROI").lower()


def test_clv_never_appears_without_its_caveat():
    s = _summary(graded_n=40, wins=22, losses=18, decided_n=40)
    s.clv.n_with_line_clv = 40
    s.clv.avg_clv_line_points = 0.25
    s.clv.avg_clv_prob_points = 0.013
    clv = _field(build_win_loss_embed(s), "CLV")
    assert "+0.250" in clv
    assert "not profit" in clv
    assert "not evidence of future returns" in clv


def test_no_closing_lines_is_stated_rather_than_shown_as_zero_clv():
    clv = _field(build_win_loss_embed(_summary(graded_n=5, wins=3, losses=2)), "CLV")
    assert "no closing lines captured" in clv


def test_the_card_carries_no_claim_language():
    s = _summary(graded_n=40, wins=35, losses=5, decided_n=40, strike_rate_pct=87.5)
    s.roi.staked_units, s.roi.profit_units, s.roi.roi_pct = 40.0, 20.0, 50.0
    blob = _text(build_win_loss_embed(s))
    for word in ("lock", "guaranteed", "best bet", "sure thing", "free money"):
        assert word not in blob
    # The shared RESEARCH_FOOTER, whose substantive claim holds for a results
    # card too. Its wording ("Recommendation, not a promise of an outcome")
    # was written for the board and reads slightly off here; one footer across
    # every card is the deliberate trade, since a second copy would drift.
    assert "does not place the wager" in blob, "the research footer is missing"
    assert "graded predictions, not wagers" in blob


def test_a_warning_injected_into_the_summary_cannot_smuggle_a_claim_through():
    """
    `warnings` comes from the metrics layer and is rendered verbatim, so the
    builder has to run it through the same guard as everything else.
    """
    s = _summary(graded_n=5, wins=5, losses=0, decided_n=5)
    s.warnings = ["this one is a lock"]
    with pytest.raises(DiscordDispatchError):
        build_win_loss_embed(s)


def test_the_metrics_clv_note_survives_the_dispatcher_s_own_guard():
    """
    It did not. The note read "does not guarantee future profitability", and
    the guard bans "guarantee" by substring — deliberately bluntly, so it
    fires on a negated claim too. The note was reworded; the guard was not
    weakened. This pins both halves.
    """
    from src.notify.discord import FORBIDDEN_CLAIM_WORDS
    from src.settlement.metrics import ClvBlock

    note = ClvBlock().note
    assert note
    for word in FORBIDDEN_CLAIM_WORDS:
        assert word not in note.lower(), (
            f"the metrics CLV note contains {word!r}, so any card carrying it "
            "will be refused by the dispatcher"
        )


# --- it is wired into the settlement job -----------------------------------

def test_the_settlement_job_sends_the_card():
    import scheduler_worker as w

    assert hasattr(w, "run_results_card")
    body = (
        __import__("pathlib").Path(__file__).parent.parent / "scheduler_worker.py"
    ).read_text(encoding="utf-8")
    settle = body[body.index("def run_settlement"):]
    settle = settle[: settle.index("\ndef ")]
    assert "run_results_card()" in settle, (
        "the builder exists but nothing calls it on a schedule — which is the "
        "shape O9 was in to begin with"
    )


def test_the_card_is_off_without_a_webhook():
    import scheduler_worker as w

    out = w.run_results_card()
    assert out["status"] == "SKIPPED"


def test_the_card_reports_yesterday_not_today():
    """
    Settlement runs at 03:30 PT and grades games that finished the previous
    Pacific day. A card dated today would be empty every morning.
    """
    body = (
        __import__("pathlib").Path(__file__).parent.parent / "scheduler_worker.py"
    ).read_text(encoding="utf-8")
    card = body[body.index("def run_results_card"):]
    assert "timedelta(days=1)" in card


def test_a_failure_in_the_card_does_not_take_the_worker_down():
    import ast
    from pathlib import Path

    tree = ast.parse((Path(__file__).parent.parent / "scheduler_worker.py").read_text())
    fn = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "run_results_card"
    )
    assert any(isinstance(n, ast.Try) for n in ast.walk(fn))
