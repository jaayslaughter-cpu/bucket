"""
The board embed must show a size, and must count what it evaluated.

TWO GAPS FOUND IN THE 2026-10-29 PRODUCTION SWEEP, and they are the only two
things in that brief the repository did not already do.

  * `BettingDecisionCandidate` carries EV, odds, model probability, side and
    line, and NO size. The DFS entry embed has shown a recommended stake since
    it was written (`build_dfs_entry_embed`); the board -- the card an operator
    actually reads every morning -- never has. A reader got "EV 4.2%" and no
    answer to "how much".
  * A slate where nothing qualified said "Nothing recommended on this slate."
    with no denominator, so it read identically whether the pipeline evaluated
    412 lines and liked none of them or evaluated zero. Those are a quiet night
    and a broken pipeline, and they must not look the same in a channel.

RESEARCH_ONLY. A displayed size is advisory metadata; `advisory_sizing` stamps
`AUTO_PLACED: False` on every one it returns and nothing here places anything.
"""

from __future__ import annotations

from src.notify.discord import build_decision_board_embed


def _row(**over):
    """One board row, as the dispatcher builds it from the board CSV."""
    base = {
        "player_name": "HARNESS PLAYER",
        "target_market": "PTS",
        "side": "over",
        "line": 24.5,
        "model_prob": 0.58,
        "american_odds": -110,
        "book_ev": 0.042,
        "book_source": "harnessbook",
        "decision_status": "RECOMMENDED",
        "decision_basis": "book_ev",
        "game_date": "2026-10-29",
    }
    base.update(over)
    return type("Row", (), base)()


def _text(embed) -> str:
    parts = [embed.get("description", "")]
    for field in embed.get("fields", []):
        parts.append(f"{field.get('name', '')} {field.get('value', '')}")
    return "\n".join(parts)


def test_a_priced_row_shows_the_size_its_caller_computed():
    """
    THE NUMBER THE READER NEEDS NEXT. An EV percentage answers "is there an
    edge"; it does not answer "how much", and the operator reading this card
    has to answer the second question somewhere.

    DISPLAYED, NOT COMPUTED. `src/notify/` may not import the sizer --
    `test_no_module_under_src_imports_this_one` forbids it so sizing cannot
    reach a dispatch path -- so the row arrives carrying the number, exactly
    as `build_dfs_entry_embed` takes its `advisory_size` as a parameter.
    """
    embed = build_decision_board_embed(
        [_row(advisory_units=2.95)], slate_date="2026-10-29",
    )
    text = _text(embed)
    assert "2.95u" in text, f"the size the caller supplied was not shown: {text[:400]}"
    assert "Kelly" in text


def test_a_row_with_no_size_shows_none_rather_than_zero():
    """
    A row the caller could not size must not read "0u", which is a
    recommendation to stake nothing on a real edge rather than the absence of
    one. The caller sends None; the embed prints nothing.
    """
    embed = build_decision_board_embed(
        [_row(advisory_units=None)], slate_date="2026-10-29",
    )
    text = _text(embed)
    assert "**0u**" not in text and "**0.0u**" not in text
    assert "advisory" not in text, "a row with no size still rendered the blurb"


def test_the_notifier_does_not_reach_for_the_sizer_itself():
    """
    THE GUARD THAT CAUGHT THIS. The first version of the board embed imported
    `recommended_units_binary` directly, which is the exact case
    `test_no_module_under_src_imports_this_one` exists for: "a size is held by
    a caller that already computed it, never fetched by a module that sends or
    places anything." Asserted here too, at the one module most tempted to.
    """
    source = (
        __import__("pathlib").Path("src/notify/discord.py")
        .read_text(encoding="utf-8")
    )
    assert "import recommended_units_binary" not in source
    assert "advisory_sizing import" not in source


def test_an_empty_slate_says_how_many_lines_it_evaluated():
    """
    A QUIET NIGHT AND A BROKEN PIPELINE MUST NOT READ THE SAME. "Nothing
    recommended" with no denominator is consistent with both. The count of
    rows the board carried is the one number that separates them.
    """
    evaluated = [_row(decision_status="ABSTAIN") for _ in range(412)]
    embed = build_decision_board_embed(evaluated, slate_date="2026-10-29")
    text = _text(embed)
    assert "412" in text, (
        f"an empty slate did not say how many lines it evaluated: {text[:300]}"
    )
    assert "0 qualifying" in text or "0 of 412" in text


def test_a_board_with_no_rows_at_all_still_says_so():
    """Zero evaluated is its own state and is not an exception."""
    embed = build_decision_board_embed([], slate_date="2026-10-29")
    assert _text(embed).strip()


# --- the half that computes: scheduler_worker, which is outside src/ -------


def _board_row(**over):
    base = {"model_prob": 0.58, "american_odds": -110}
    base.update(over)
    return type("Row", (), base)()


def test_the_caller_sizes_a_real_edge():
    from scheduler_worker import _attach_advisory_units

    rows = [_board_row()]
    _attach_advisory_units(rows)
    assert rows[0].advisory_units is not None
    assert rows[0].advisory_units > 0


def test_the_caller_sends_none_on_a_negative_edge_not_zero():
    """
    `recommended_units_binary` returns 0.0 both for "no edge here" and for
    inputs it refuses. Passing that 0.0 through would print "0u" on the card.
    The None/zero decision lives here, in the caller, because the notifier is
    not allowed to know how to size.
    """
    from scheduler_worker import _attach_advisory_units

    rows = [_board_row(model_prob=0.40)]
    _attach_advisory_units(rows)
    assert rows[0].advisory_units is None, (
        f"a 40% probability at -110 is a negative edge and was sized "
        f"{rows[0].advisory_units}"
    )


def test_an_unsizable_row_does_not_fail_the_card():
    """One row that cannot be sized must not cost the whole dispatch: a card
    without a size beats no card."""
    from scheduler_worker import _attach_advisory_units

    rows = [_board_row(american_odds=None), _board_row(model_prob="n/a"),
            _board_row()]
    _attach_advisory_units(rows)
    assert rows[0].advisory_units is None
    assert rows[1].advisory_units is None
    assert rows[2].advisory_units > 0
