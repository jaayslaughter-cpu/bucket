"""
tests/test_distribution_sweep.py — the sweep's own contract.

WHAT IS WORTH TESTING. Not the families or the tails — those are
`residuals`'s and are tested where they live. What is this script's own:

  * it refuses to run without lines somebody actually posted, which is the
    exact bug it was written in response to;
  * it scores the threshold, not the whole distribution, and reports the two
    rankings separately so a disagreement is visible;
  * pushes are dropped rather than scored as an over or an under;
  * a thin market is reported unranked instead of ranked on noise;
  * and it changes nothing.

RESEARCH_ONLY.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts import distribution_sweep as sweep


def _rows(n_games=60, per_game=6, market="PTS", seed=5, line_offset=0.5):
    """
    A panel with REAL posted lines offset from the projection, so no family is
    handed the degenerate p=0.5 case that `line = expected_value` produces.
    """
    rng = np.random.default_rng(seed)
    out = []
    for g in range(n_games):
        for p in range(per_game):
            mean = float(rng.uniform(8.0, 28.0))
            # Overdispersed counts: variance above the mean, so negbin has
            # something real to find.
            actual = float(rng.negative_binomial(4, 4 / (4 + mean)))
            out.append({
                "market": market,
                "event_id": g,
                "player_id": p,
                "mean": mean,
                "line": mean + line_offset,
                "actual": actual,
            })
    return pd.DataFrame(out)


# --- the refusal that is the whole point ---------------------------------


def test_it_refuses_to_run_without_a_line_column():
    """
    THE BUG THIS SCRIPT EXISTS BECAUSE OF. A sweep that defaults the line to
    the projection measures its own harness: the symmetric families evaluate
    to ~0.5 at the mean and lose by construction. There is no default here —
    absent lines is a refusal.
    """
    frame = _rows(10, 2).drop(columns=["line"])
    with pytest.raises(sweep.SweepError, match="missing required column"):
        sweep._check_columns(frame)


def test_the_required_columns_say_what_each_must_contain():
    """The refusal has to be actionable, so each name carries its meaning —
    and the line's entry says explicitly that a projection will not do."""
    assert "line" in sweep.REQUIRED_COLUMNS
    assert "never the projection" in sweep.REQUIRED_COLUMNS["line"]
    for column in ("market", "actual", "mean"):
        assert sweep.REQUIRED_COLUMNS[column]


def test_a_missing_file_is_named_rather_than_tracebacked(tmp_path):
    with pytest.raises(sweep.SweepError, match="no such file"):
        sweep._load(tmp_path / "absent.csv")


# --- the two rankings, reported separately -------------------------------


def test_it_reports_the_nll_pick_and_the_threshold_pick_separately():
    """
    THE QUESTION THE SCRIPT ASKS. `fit_count_dispersion` picks by mean NLL
    over the whole support; the board only ever asks about the posted line.
    Both rankings must appear, with a flag when they differ — collapsing them
    to one answer would hide the finding.
    """
    result = sweep.score_market(_rows(), "PTS")
    assert result["status"] == "RANKED"
    assert result["live_selector_picks"] in sweep.FAMILIES
    assert result["threshold_score_picks"] in sweep.FAMILIES
    assert isinstance(result["selectors_disagree"], bool)
    assert result["selectors_disagree"] == (
        result["live_selector_picks"] != result["threshold_score_picks"]
    )


def test_every_family_is_scored_on_both_objectives():
    """
    Each family carries its NLL *and* its Brier, so a reader can see the two
    objectives rank them differently rather than taking the verdict on trust.
    """
    result = sweep.score_market(_rows(), "PTS")
    for family in sweep.FAMILIES:
        entry = result["families"][family]
        if entry["status"] != "OK":
            continue
        assert entry["nll"] > 0
        assert 0.0 <= entry["brier"] <= 1.0
        assert entry["log_loss"] > 0
        assert entry["n_scored"] > 0


def test_the_probabilities_come_from_the_shared_tail_not_a_second_one():
    """
    NEITHER FAMILY NOR TAIL IS REIMPLEMENTED. Every probability must come from
    `residuals.over_under_push_from_dispersion` — the function the live path
    calls. A second implementation here would score a distribution this
    project does not actually use.
    """
    from pathlib import Path

    source = Path("scripts/distribution_sweep.py").read_text(encoding="utf-8")
    assert "over_under_push_from_dispersion" in source
    assert "fit_count_dispersion" in source
    for reimplemented in ("nbinom.", "poisson.", "norm.cdf", "scipy"):
        assert reimplemented not in source, (
            f"{reimplemented} suggests a second tail implementation here"
        )


# --- pushes, and thin markets --------------------------------------------


def test_a_push_is_not_scored_as_an_over_or_an_under():
    """
    A stat landing exactly on a whole-number line is a push. Scoring it either
    way charges a family for an outcome it correctly called neither.
    """
    frame = _rows(60, 6)
    # Force a block of exact pushes on a whole-number line.
    frame.loc[frame.index[:40], "line"] = 20.0
    frame.loc[frame.index[:40], "actual"] = 20.0

    result = sweep.score_market(frame, "PTS")
    assert result["n_pushes"] == 40
    assert result["n_decided_rows"] == len(frame) - 40


def test_a_thin_market_is_unranked_rather_than_ranked_on_noise():
    """
    Below the row floor a ranking is a statement about the sample. Reporting
    NOT_RANKED with the count beats reporting a winner nobody should use.
    """
    result = sweep.score_market(_rows(4, 2), "PTS")
    assert result["status"] == "NOT_RANKED"
    assert "below the" in result["reason"]


# --- the disagreement is tested, not just announced ----------------------


def _zero_inflated_rows(n_games=100, per_game=8, seed=1, zero_rate=0.25):
    """
    A panel the two objectives genuinely disagree about.

    MEASURED, NOT CONTRIVED. On zero-inflated counts the NLL selector picks
    `zip` — correctly, the data *is* zero-inflated — while the threshold score
    picks `negbin`. The mechanism is the finding this sweep exists to surface:
    ZIP earns its NLL on the zero spike, and a posted line sits nowhere near
    the zero, so at the threshold NB's tail shape prices better. Reproduced on
    seeds 1, 2 and 3.

    This is the regime real low-usage props live in: STL, BLK and FG3M for a
    bench player are genuinely zero-inflated.
    """
    rng = np.random.default_rng(seed)
    out = []
    for g in range(n_games):
        for p in range(per_game):
            mean = float(rng.uniform(2.0, 30.0))
            actual = 0.0 if rng.random() < zero_rate else float(rng.poisson(mean))
            out.append({
                "market": "PTS", "event_id": g, "player_id": p,
                "mean": mean, "line": mean + 0.5, "actual": actual,
            })
    return pd.DataFrame(out)


def test_the_two_objectives_really_can_disagree():
    """
    THE FINDING, as a test. If this ever stops disagreeing the sweep has
    nothing to report and the premise behind it needs re-examining.
    """
    result = sweep.score_market(_zero_inflated_rows(), "PTS")
    assert result["status"] == "RANKED"
    assert result["selectors_disagree"] is True
    assert result["live_selector_picks"] == "zip"
    assert result["threshold_score_picks"] == "negbin"
    # ZIP wins the objective it is scored on and loses the one that pays.
    families = result["families"]
    assert families["zip"]["nll"] < families["negbin"]["nll"]
    assert families["negbin"]["brier"] < families["zip"]["brier"]


def test_a_disagreement_is_put_through_the_paired_bootstrap(monkeypatch):
    """
    A Brier gap between two families is subject to the same sampling noise as
    a gap between two models, so a disagreement must be qualified rather than
    reported as a result.

    The first version of this test ran on data whose selectors happened to
    agree and called `pytest.skip`, which is a test that does not run. It uses
    the zero-inflated panel now, which disagrees deterministically.
    """
    captured: dict = {}

    def fake_decisive(truth, leader, runner_up, **kwargs):
        captured["games_passed"] = kwargs.get("games") is not None
        captured["metric"] = kwargs.get("metric")
        return {
            "decisive": False,
            "note": "tied on this evidence",
            "ci_low": -0.01,
            "ci_high": 0.01,
            "cluster": "game",
        }

    import src.models.paired_bootstrap as pb

    monkeypatch.setattr(pb, "decisive_winner", fake_decisive)

    result = sweep.score_market(_zero_inflated_rows(), "PTS")
    assert result["selectors_disagree"] is True
    assert captured["games_passed"] is True, "the bootstrap was not clustered"
    assert captured["metric"] == "brier"
    assert result["disagreement_decisive"] is False
    assert "tied on this evidence" in result["disagreement_note"]


def test_a_real_disagreement_gets_a_real_interval():
    """
    Unmocked: the disagreement must arrive with an interval and a clustering
    label, so a reader can see whether the gap is worth acting on.
    """
    result = sweep.score_market(_zero_inflated_rows(), "PTS")
    assert result["disagreement_cluster"] == "game"
    low, high = result["disagreement_ci"]
    assert low <= high
    assert isinstance(result["disagreement_decisive"], bool)


def test_the_disagreement_branch_clusters_by_game():
    """
    Asserted on the code, because whether a given random sample produces a
    disagreement is not something a test should depend on. `event_id` is the
    clustering unit and it must be read from the frame.
    """
    from pathlib import Path

    source = Path("scripts/distribution_sweep.py").read_text(encoding="utf-8")
    assert "GAME_COLUMN" in source
    assert sweep.GAME_COLUMN == "event_id"
    assert "games=games" in source
    # And the weaker iid interval must be labelled when the column is absent.
    assert "narrower than it should be" in source


# --- it changes nothing ---------------------------------------------------


def test_the_report_says_the_live_selector_is_untouched(tmp_path):
    """
    The point of a research script is that somebody reads it and decides. A
    report that looked like a change would invite a switch nobody approved —
    and switching the family needs a new .meta.json contract on the volume.
    """
    source = tmp_path / "rows.csv"
    _rows().to_csv(source, index=False)
    report = sweep.sweep(source)
    assert report["research_only"] is True
    assert "untouched" in report["changes_nothing"]
    assert ".meta.json" in report["changes_nothing"]


def test_the_cli_writes_the_report_and_exits_zero(tmp_path, capsys):
    source = tmp_path / "rows.csv"
    _rows().to_csv(source, index=False)
    out = tmp_path / "sweep.json"

    assert sweep.main(["--from-file", str(source), "--out", str(out)]) == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["research_only"] is True
    assert written["markets"]
    printed = capsys.readouterr().out
    assert "RESEARCH_ONLY" in printed
    assert "what runs live" in printed


def test_the_cli_refuses_a_file_without_lines(tmp_path, capsys):
    source = tmp_path / "rows.csv"
    _rows().drop(columns=["line"]).to_csv(source, index=False)
    assert sweep.main(["--from-file", str(source), "--out", str(tmp_path / "x.json")]) == 2
    assert "missing required column" in capsys.readouterr().err
