"""
`build_research_preview_embed` -- the card that makes the slate's props
visible while the publication gate is still shut.

WHAT THIS FILE IS ABOUT. The preview exists because with no scoring artifact
every board row is ABSTAIN, so the recommendation card correctly names nobody
and an operator has nowhere to look but the CSV. Its whole value is that the
rows on it are real props. A row that is not a prop -- no posted line, so
nothing for the projection to disagree with -- is worse than absent: it
occupies one of the card's limited rows and says nothing.

The dispatch wiring (the flag, the gate's verdict riding along, one row per
(player, market), the leaning side) is covered in
`tests/test_scheduler_worker.py`. This file calls the builder directly,
because the builder is what decides.
"""

from __future__ import annotations

from types import SimpleNamespace

from src.notify.discord import build_research_preview_embed


def _row(player, market, side, line, proj, status="ABSTAIN"):
    return SimpleNamespace(
        player_name=player,
        target_market=market,
        side=side,
        line=line,
        prediction_mean=proj,
        decision_status=status,
    )


def _withheld(reason="fewer than 100 graded predictions"):
    return SimpleNamespace(allowed=False, reason=reason)


def test_a_prop_with_no_posted_line_is_not_previewable(monkeypatch):
    """
    NaN IS NOT None, which is the whole defect.

    On 2026-11-04 the slate assembled 1620 projection rows and only 754
    carried a matched prop line; the settlement recorder named the other 866
    "no posted line". Read back through pandas, those rows carry `line` as
    NaN rather than None, and:

      * `float(nan) - float(proj)` is `nan`, which is neither `> 0` nor
        `< 0`, so the leaning-side filter rejects neither side;
      * `abs(nan) > abs(x)` is false against everything, so such a row never
        loses a comparison it is already holding;
      * it renders as "`+nan` PLAYER PTS OVER nan".

    Measured at that slate's proportion of line-less rows, 11 of the card's
    15 rows came out `nan` -- and the four real ones were not even the
    highest-lean props, because NaN sort keys corrupt the ordering around
    them.
    """
    rows = [
        # NO LINE, listed first so insertion order cannot hide the bug.
        _row("HARNESS NOLINE", "PTS", "over", float("nan"), 19.5),
        _row("HARNESS NOLINE", "PTS", "under", float("nan"), 19.5),
        # A real prop, with a SMALLER lean than a nan row sorts ahead of.
        _row("HARNESS JOKIC", "PTS", "under", 27.5, 29.1),
        _row("HARNESS JOKIC", "PTS", "over", 27.5, 29.1),
    ]

    embed = build_research_preview_embed(
        rows, slate_date="2026-11-04", publication=_withheld()
    )
    body = embed["description"]

    assert "nan" not in body.lower(), (
        "a projection with no posted line was rendered as `nan` on the card"
    )
    assert "HARNESS NOLINE" not in body, (
        "a player with no posted line was listed as a prop to look at"
    )
    # AND THE REAL PROP SURVIVED, so this cannot pass by showing nothing.
    assert "HARNESS JOKIC" in body and "27.5" in body, (
        "the line-less rows were dropped and so was the real prop beside them"
    )
    # THE COUNT IS OF WHAT IS PREVIEWABLE, not of what arrived. A headline
    # claiming 2 props above a card showing 1 is the same lie in miniature.
    assert "1 prop(s) with a posted line" in body, (
        f"the headline counted rows it refused to show: {body[:200]!r}"
    )


def test_a_line_less_slate_says_so_rather_than_printing_nothing():
    """
    EVERY row line-less is the honest-absence case, not an empty card.

    This is reachable: a slate whose prop feed returns nothing at all still
    produces projections, and all of them would land here.
    """
    rows = [
        _row("HARNESS NOLINE", "PTS", "over", float("nan"), 19.5),
        _row("HARNESS NOLINE", "AST", "over", None, 4.2),
    ]

    body = build_research_preview_embed(
        rows, slate_date="2026-11-04", publication=_withheld()
    )["description"]

    assert "nothing to preview" in body
    assert "nan" not in body.lower()
    # THE GATE'S REASON IS STILL ON IT. An empty preview must not read as
    # "the gate opened and there was nothing good".
    assert "withheld from publication" in body


def test_the_ordering_is_by_absolute_lean_once_the_nan_rows_are_gone():
    """
    The card's one ordering claim, asserted on a set NaN would have scrambled.

    `+1.6` beating `-10.0` was the observable symptom of NaN keys in the
    sort, so a fixture that mixes a large negative lean with a small positive
    one pins the thing that was actually broken.
    """
    rows = [
        _row("SMALL LEAN", "PTS", "over", 17.0, 20.0),      # +3.0
        _row("NO LINE", "PTS", "over", float("nan"), 19.5),  # dropped
        _row("BIG LEAN", "PTS", "under", 23.0, 13.0),       # -10.0
    ]

    body = build_research_preview_embed(
        rows, slate_date="2026-11-04", publication=_withheld()
    )["description"]

    shown = [ln for ln in body.splitlines() if ln.startswith("`")]
    assert len(shown) == 2, shown
    assert "BIG LEAN" in shown[0], f"-10.0 must outrank +3.0: {shown}"
    assert "SMALL LEAN" in shown[1], shown


def test_max_rows_caps_the_card_and_keeps_the_largest_leans():
    """
    The cap is a display budget, not a filter on which props count: the
    headline still names how many there are.
    """
    rows = []
    for i in range(40):
        # lean grows with i, so the last ones must be the ones shown
        rows.append(_row(f"PLAYER {i:02d}", "PTS", "over", 10.0, 10.0 + i * 0.5))

    body = build_research_preview_embed(
        rows, slate_date="2026-11-04", publication=_withheld(), max_rows=5
    )["description"]

    shown = [ln for ln in body.splitlines() if ln.startswith("`")]
    assert len(shown) == 5
    assert "40 prop(s) with a posted line" in body, (
        "the cap must not change the count: PLAYER 00's lean is exactly 0.0, "
        "which is still a prop with a posted line, just the least interesting"
    )
    assert "PLAYER 39" in shown[0]
    assert "PLAYER 00" not in body, "the smallest lean was shown over larger ones"
