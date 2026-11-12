"""
tests/test_market_coverage.py — what the slate asks for vs what can be fit.

TWO LISTS OF MARKETS, IN TWO MODULES, AND NOTHING TIED THEM TOGETHER.

    main.DEFAULT_STATS       ("PTS", "REB", "AST", "FG3M")   what the slate scores
    training.MARKETS         ("PTS", "REB", "AST")           what training can fit

The difference is FG3M, so every slate since 2026-11-07 logged

    No artifact for ['FG3M'], so those rows carry no probability and the
    recorder will skip them.

and moved on. The behaviour is right -- an absent artifact must abstain, not
guess -- but the DISAGREEMENT was undeclared. It read as a nightly surprise,
and the decision that produced it (FG3M is unmodelled, not merely untrained)
lived nowhere.

So the gap is pinned here, in both directions:

  * nothing may be added to `DEFAULT_STATS` without either a model or an
    entry in `MARKETS_WITHOUT_A_MODEL`;
  * nothing may be listed as unmodelled while a model for it exists.

AND THE BOOT LINE THAT OVERSTATED ITS SCOPE. `check_model_artifact` resolves
ONE artifact for a boot report. Its log line used to say "every other market
abstains this run", which was true before `score_prob_over_by_market` existed
and false afterwards. On 2026-11-12 it was read as current and reported as a
finding: that one market scored per night, chosen by file mtime. The live logs
refuted it within minutes. A test now holds the line to what it can support.

RESEARCH_ONLY. Consistency between two declarations. No model, no wager.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _market_sets() -> tuple[set[str], set[str], set[str]]:
    import main
    from src.models.training import MARKETS, MARKETS_WITHOUT_A_MODEL

    return (
        {m.upper() for m in main.DEFAULT_STATS},
        {m.upper() for m in MARKETS},
        {m.upper() for m in MARKETS_WITHOUT_A_MODEL},
    )


def test_every_scored_market_is_either_fittable_or_declared_unmodelled() -> None:
    """
    THE DISAGREEMENT IS THE WHOLE POINT, so it is stated exactly.

    A market the slate loops over must either have a model this project can
    fit, or be named as one it cannot. A fifth market added to
    `DEFAULT_STATS` fails here until somebody says which it is.
    """
    scored, fittable, unmodelled = _market_sets()
    unexplained = scored - fittable - unmodelled
    assert not unexplained, (
        f"{sorted(unexplained)} appear in main.DEFAULT_STATS with no model and "
        f"no entry in training.MARKETS_WITHOUT_A_MODEL. The slate will ask for "
        f"them every night and skip every row, silently. Either fit them or "
        f"name them."
    )


def test_nothing_is_declared_unmodelled_while_a_model_exists() -> None:
    """
    The other direction, which is how a stale declaration rots.

    The day FG3M becomes fittable, leaving it in `MARKETS_WITHOUT_A_MODEL`
    would document a limitation that no longer exists -- and the next person
    reading it would not train the thing that already trains.
    """
    _, fittable, unmodelled = _market_sets()
    contradiction = fittable & unmodelled
    assert not contradiction, (
        f"{sorted(contradiction)} is listed as having no model AND is in "
        f"training.MARKETS, which cannot both be true"
    )


def test_the_unmodelled_set_is_exactly_what_the_slate_cannot_score() -> None:
    """
    No spare entries. A market named as unmodelled that the slate never asks
    for is a note about nothing, and it will outlive whatever prompted it.
    """
    scored, _, unmodelled = _market_sets()
    stray = unmodelled - scored
    assert not stray, (
        f"{sorted(stray)} is named unmodelled but is not in DEFAULT_STATS, so "
        f"nothing ever asks for it"
    )


def test_fg3m_is_the_gap_and_is_named_as_such() -> None:
    """
    The regression, pinned to today's actual state rather than to a shape.

    If this fails because FG3M was trained, delete it along with the entry --
    that is the good outcome. If it fails because the difference changed to
    some other market, the change wants deliberate attention.
    """
    scored, fittable, unmodelled = _market_sets()
    assert scored - fittable == {"FG3M"}, (
        f"the scored-but-unfittable set is {sorted(scored - fittable)}, not "
        f"{{'FG3M'}}; the market coverage changed"
    )
    assert unmodelled == {"FG3M"}


def test_the_boot_check_does_not_claim_to_know_what_the_slate_will_score(
) -> None:
    """
    MAKE THE WRONG CLAIM UNSAYABLE.

    `check_model_artifact` resolves one artifact for a boot report. It cannot
    know which markets the slate will score, because that is decided per
    market at run time by `score_prob_over_by_market`. The old wording --
    "every other market abstains this run" -- asserted otherwise and was read
    as evidence.

    Asserted on the function's STRING LITERALS, not its raw source, and that
    distinction earned itself immediately: the first version scanned the
    source text and failed on the comment that records the old wording in
    order to explain why it was removed. The claim lives in what gets
    LOGGED. A comment quoting the mistake is the opposite of making it.
    """
    import scheduler_worker

    source = inspect.getsource(scheduler_worker.check_model_artifact)
    tree = ast.parse(source)  # module-level fn: already at column 0
    literals = [
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    logged = " ".join(literals)

    for forbidden in (
        "every other market abstains",
        "is the only one that can carry",
    ):
        assert forbidden not in logged, (
            f"check_model_artifact still LOGS {forbidden!r}. It resolves one "
            f"artifact for a boot report and cannot know what the slate will "
            f"score."
        )
    assert "does NOT say which" in logged, (
        "the boot line no longer states the limit of what it knows"
    )


def test_the_slate_scores_per_market_rather_than_once() -> None:
    """
    The fact the corrected log line defers to, asserted on the real call.

    `main` must reach `score_prob_over_by_market` -- the per-market loop --
    and not a single `score_prob_over` for the whole slate. This is what made
    the old boot line false, so it is worth holding in place.
    """
    source = (REPO / "main.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "score_prob_over_by_market" in calls, (
        "nothing calls the per-market scorer, so the slate is back to one "
        "artifact for every market"
    )


def test_a_market_with_no_artifact_is_skipped_not_filled() -> None:
    """
    The abstention FG3M relies on, driven rather than assumed.

    With no artifact on disk the scorer must return no entry for that market,
    so `settlement.recorder` skips the row. Filling it -- with 0.5, or with
    another market's probability -- would publish a number nobody computed.
    """
    import pandas as pd

    import main

    features = pd.DataFrame({
        "PLAYER_ID": ["p1"], "PLAYER_NAME": ["A B"],
        "GAME_ID": ["g1"], "GAME_DATE": [pd.Timestamp("2026-11-11")],
    })
    lines = pd.DataFrame({
        "PLAYER_NAME": ["A B"], "market": ["FG3M"], "line": [2.5],
    })

    out = main.score_prob_over_by_market(
        features, lines, markets=("FG3M",),
    )
    assert "FG3M" not in out, (
        f"a market with no artifact produced a probability anyway: {out}"
    )
