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


def test_a_board_with_no_rows_at_all_says_zero_evaluated():
    """
    Zero evaluated is its own state and is not an exception -- and it is a
    DIFFERENT state from "412 evaluated, none qualified", so the marker is
    pinned rather than the test merely asserting the embed is non-empty,
    which it was and which could not fail. Flagged in review on 2026-10-30.
    """
    embed = build_decision_board_embed([], slate_date="2026-10-29")
    text = _text(embed)
    assert "0 lines evaluated" in text
    assert not embed.get("fields"), "a board with no rows rendered row fields"


def test_a_board_with_recommendations_also_carries_the_denominator():
    """
    "3 recommended" out of 400 evaluated and out of 6 evaluated are very
    different nights. The count was shown only when nothing qualified.
    """
    rows = [_row(advisory_units=2.95) for _ in range(3)]
    rows += [_row(decision_status="ABSTAIN") for _ in range(409)]
    text = _text(build_decision_board_embed(rows, slate_date="2026-10-29"))
    assert "412" in text, f"no denominator on a board with plays: {text[:300]}"


# --- the half that computes: scheduler_worker, which is outside src/ -------


def _board_row(**over):
    """A board row with the fields `advisory_size_for_candidate` reads.

    Richer than the embed fixture on purpose: the real sizer refuses a row
    the board did not recommend, a row with no VALID book, and a pushable
    line with no two sides to condition on. A fixture carrying only
    model_prob and odds would be refused by all three and prove nothing.
    """
    base = {
        "decision_status": "RECOMMENDED",
        "book_status": "VALID",
        "american_odds": -110,
        "model_prob": 0.58,
        "model_p_over": 0.58,
        "model_p_under": 0.42,
        "model_p_push": 0.0,
        "line_can_push": False,
    }
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
    Every refusal and every non-positive edge comes back as 0.0 units. Passing
    that through would print "0u" on the card, which reads as a
    recommendation to stake nothing on a real edge. The None/zero decision
    lives here because the notifier prints whatever it is given.
    """
    from scheduler_worker import _attach_advisory_units

    rows = [_board_row(model_prob=0.40, model_p_over=0.40, model_p_under=0.60)]
    _attach_advisory_units(rows)
    assert rows[0].advisory_units is None


def test_a_pushable_line_is_sized_on_the_no_push_probability():
    """
    THE REASON THIS GOES THROUGH `advisory_size_for_candidate` AT ALL. Binary
    Kelly's `q` is P(lose), and on a whole-number line a push RETURNS the
    stake. Sizing on the raw `model_prob` counts returned money as lost and
    understates the size. The helper conditions on no-push --
    `model_prob / (model_p_over + model_p_under)` -- so the same row sizes
    HIGHER than the naive reduction would give it.

    Flagged in review on 2026-10-30, after the first version of
    `_attach_advisory_units` hand-rolled `recommended_units_binary` on the raw
    probability and silently understated every whole-number line.
    """
    from scheduler_worker import _attach_advisory_units

    pushable = [_board_row(
        line_can_push=True, model_p_push=0.07,
        model_prob=0.55, model_p_over=0.55, model_p_under=0.38,
    )]
    naive = [_board_row(model_prob=0.55, model_p_over=0.55, model_p_under=0.45)]
    _attach_advisory_units(pushable)
    _attach_advisory_units(naive)

    assert pushable[0].advisory_units is not None
    assert naive[0].advisory_units is not None
    assert pushable[0].advisory_units > naive[0].advisory_units, (
        f"the push was not conditioned out: pushable sized "
        f"{pushable[0].advisory_units} against a naive {naive[0].advisory_units}, "
        f"but 0.55/(0.55+0.38) is a bigger edge than 0.55 flat"
    )


def test_a_pushable_line_with_no_two_sides_is_refused_not_guessed():
    """The helper refuses rather than sizing a three-outcome probability
    against a two-outcome price. That refusal must reach the card as no size,
    not as a zero."""
    from scheduler_worker import _attach_advisory_units

    rows = [_board_row(
        line_can_push=True, model_p_push=0.07,
        model_p_over=None, model_p_under=None,
    )]
    _attach_advisory_units(rows)
    assert rows[0].advisory_units is None


def test_a_row_the_board_did_not_recommend_is_not_sized():
    """Sizing a row the gate withheld would contradict the gate."""
    from scheduler_worker import _attach_advisory_units

    rows = [_board_row(decision_status="ABSTAIN")]
    _attach_advisory_units(rows)
    assert rows[0].advisory_units is None


def test_a_row_with_no_valid_book_is_not_sized():
    """Kelly needs a price. A model probability alone sizes nothing."""
    from scheduler_worker import _attach_advisory_units

    rows = [_board_row(book_status="DATA_NOT_AVAILABLE", american_odds=None)]
    _attach_advisory_units(rows)
    assert rows[0].advisory_units is None


def test_an_unsizable_row_does_not_fail_the_card():
    """One row that cannot be sized must not cost the whole dispatch: a card
    without a size beats no card."""
    from scheduler_worker import _attach_advisory_units

    rows = [_board_row(model_prob="n/a"), _board_row(american_odds=None),
            _board_row()]
    _attach_advisory_units(rows)
    assert rows[0].advisory_units is None
    assert rows[1].advisory_units is None
    assert rows[2].advisory_units > 0


def test_the_worker_does_not_size_with_the_binary_sizer_directly():
    """
    ONE SIZER, AND IT IS THE PUSH-AWARE ONE. `recommended_units_binary` on a
    raw board `model_prob` understates every whole-number line; the hand-
    rolled call that did exactly that is the thing this review round removed.

    ASSERTED ON THE IMPORTS, NOT THE SOURCE TEXT. The first version of this
    test read the file and searched for the name -- and failed on the two
    COMMENTS that explain why the binary sizer is not used here. A source
    assertion satisfied, or broken, by prose about the code is the trap this
    repository documents hitting five times; see
    `test_the_advice_does_not_restate_the_loaders_required_columns`.
    """
    import ast
    import pathlib

    tree = ast.parse(pathlib.Path("scheduler_worker.py").read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)

    assert "recommended_units_binary" not in imported, (
        "the worker imports the binary sizer, which understates every "
        "whole-number line"
    )
    assert "advisory_size_for_candidate" in imported


# --- the wiring, which neither half above could prove ----------------------


def test_run_dispatch_puts_the_computed_size_on_the_card(monkeypatch, tmp_path):
    """
    THE HOLE BOTH HALVES LEFT. The helper was tested, the embed was tested,
    and deleting `_attach_advisory_units(rows)` from `run_dispatch` passed the
    whole suite -- the two ends were proved and the wire between them was not.
    Flagged in review on 2026-10-30.

    Feeds a priced board CSV through the real `run_dispatch` with the webhook
    transport captured, and asserts the units reach the rendered embed.
    """
    import json

    import pandas as pd

    import scheduler_worker as worker

    board = tmp_path / "board.csv"
    pd.DataFrame([{
        "slate_date": "2026-10-29", "game_date": "2026-10-29",
        "player_name": "HARNESS PLAYER", "target_market": "PTS",
        "side": "over", "line": 24.5,
        "model_prob": 0.58, "model_p_over": 0.58, "model_p_under": 0.42,
        "model_p_push": 0.0, "line_can_push": False,
        "american_odds": -110, "book_status": "VALID", "book_ev": 0.042,
        "book_source": "harnessbook",
        "decision_status": "RECOMMENDED", "decision_basis": "book_ev",
    }]).to_csv(board, index=False)

    monkeypatch.setenv(worker.ENV_BOARD_CSV, str(board))
    monkeypatch.setenv(worker.ENV_DISPATCH, "1")
    monkeypatch.setenv(worker.ENV_WEBHOOK_URL,
                       "https://discord.com/api/webhooks/1/harness-token")

    # An allowed publication verdict, so the gate does not replace the rows
    # with its refusal -- the gate itself is exercised in
    # tests/test_publication_gate.py and is not what this test is about.
    import src.notify.discord as discord_mod
    import src.quant.publication_gate as gate_mod

    def _allowed(*a, **k):
        return type("V", (), {
            "allowed": True, "reason": None, "as_dict": lambda self: {},
        })()

    monkeypatch.setattr(gate_mod, "calibration_gate", _allowed)
    monkeypatch.setattr(worker, "artifact_freshness_verdict", _allowed)
    monkeypatch.setattr(gate_mod, "first_refusal", lambda *v: v[0])

    captured: list[dict] = []

    def fake_send(embeds, **kwargs):
        captured.append({"embeds": embeds})
        return type("R", (), {"status": "OK", "reason": None})()

    # Patched on the SOURCE module: `run_dispatch` imports these
    # function-locally, so the name it binds is resolved at call time from
    # `src.notify.discord`, not from `scheduler_worker`.
    monkeypatch.setattr(discord_mod, "send_embeds", fake_send)

    worker.run_dispatch()

    assert captured, "run_dispatch sent nothing"
    rendered = json.dumps(captured[0]["embeds"])
    assert "2.95u" in rendered or "u** advisory" in rendered, (
        f"the size never reached the card -- the wire from "
        f"_attach_advisory_units to the embed is cut: {rendered[:500]}"
    )
