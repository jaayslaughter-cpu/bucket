"""A board row must say which game it describes — B2 from the pre-flight audit.

THE DEFECT, and it was the most misleading one in the project.
``research_slate_from_predictions`` stamped every row with its ``slate_date``
parameter and never read the detail row's own ``game_date``, which
``compare_models_on_panel`` writes at compare.py:839. ``ResearchSlateRow`` had
no game-date field at all, so:

  * a board built today from a validation window ending 2025-02-15 produced
    rows about February 2025 games;
  * every one carried today's date;
  * the Discord embed titled itself with that date;
  * and nothing anywhere said otherwise.

A reader could not distinguish a projection for tonight from a backtest row.
That is worse than an empty board, because an empty board disappoints and this
one misleads.

The two dates are different quantities and both now travel to the CSV and the
embed:

    slate_date  WHEN THE BOARD WAS BUILT. One value per run.
    game_date   THE DATE OF THE GAME THIS ROW DESCRIBES. Per row, from the
                detail row. None when unknown — deliberately NOT a copy of
                slate_date, because inheriting the stamp is the bug.
"""

from __future__ import annotations

from src.quant.decision_board import board_to_dataframe, build_decision_board
from src.quant.paper_research import ResearchSlateRow, research_slate_from_predictions

SLATE = "2026-10-05"


def detail(game_date: str | None = "2025-02-10", **over: object) -> dict:
    row = {
        "event_id": "0022500001",
        "player_id": "2000001",
        "player_name": "A. Player",
        "player_team": "LAL",
        "opponent": "BOS",
        "target_market": "PTS",
        "prop_line": 24.5,
        "prediction_mean": 25.1,
        "prediction_std_or_dispersion": 5.0,
        "probability_over_raw": 0.56,
        "probability_under_raw": 0.44,
        "probability_push_raw": None,
        "model_name": "distribution",
    }
    if game_date is not None:
        row["game_date"] = game_date
    row.update(over)
    return row


def board_row(**kw) -> ResearchSlateRow:
    rows = research_slate_from_predictions(
        [detail(**kw)], slate_date=SLATE, preferred_model="distribution"
    )
    assert rows, "the fixture must produce exactly one board row"
    return rows[0]


# --- the field exists and carries the game's own date -------------------

def test_the_row_carries_the_games_own_date_not_the_board_stamp():
    row = board_row(game_date="2025-02-10")
    assert row.game_date == "2025-02-10"
    assert row.slate_date == SLATE
    assert row.game_date != row.slate_date


def test_a_row_for_today_carries_today():
    row = board_row(game_date=SLATE)
    assert row.game_date == SLATE
    assert row.slate_date == SLATE


def test_a_missing_game_date_is_null_and_does_not_inherit_the_stamp():
    """
    Inheriting the stamp IS the bug. A null says "unknown"; a copy would say
    "today" about a game nobody checked.
    """
    row = board_row(game_date=None)
    assert row.game_date is None
    assert row.slate_date == SLATE


# --- an off-slate row says so ------------------------------------------

def test_an_off_slate_row_warns_and_names_both_dates():
    row = board_row(game_date="2025-02-10")
    blob = " ".join(row.warnings)
    assert "2025-02-10" in blob
    assert SLATE in blob
    assert "backtest" in blob.lower()


def test_a_row_on_the_slate_date_does_not_warn_about_its_date():
    row = board_row(game_date=SLATE)
    assert not any("backtest" in w.lower() for w in row.warnings)


def test_a_dateless_row_warns_that_it_cannot_say_which_game():
    row = board_row(game_date=None)
    assert any("no game date" in w.lower() for w in row.warnings)


# --- it reaches the CSV dispatch reads ---------------------------------

def test_the_date_survives_into_the_decision_board_and_its_csv():
    """
    The row is not what dispatch reads. build_decision_board builds
    BettingDecisionCandidate and write_decision_board_csv dumps THAT, so a
    field that stops at ResearchSlateRow never reaches a reader.
    """
    row = board_row(game_date="2025-02-10")
    board = build_decision_board([row], min_ev=-1.0, require_valid_book=False)
    assert board, "the fixture must produce at least one candidate"
    assert all(c.game_date == "2025-02-10" for c in board)

    frame = board_to_dataframe(board)
    assert "game_date" in frame.columns
    assert set(frame["game_date"]) == {"2025-02-10"}
    # Both dates, because either alone is ambiguous.
    assert "slate_date" in frame.columns


# --- and into the embed a person actually sees -------------------------

def _embed(candidates, slate_date=SLATE) -> dict:
    from src.notify.discord import build_decision_board_embed

    return build_decision_board_embed(candidates, slate_date=slate_date)


def _embed_blob(candidates, slate_date=SLATE) -> str:
    return repr(_embed(candidates, slate_date=slate_date))


def _row_values(candidates, slate_date=SLATE) -> str:
    """
    Only the per-row field values, with the banner EXCLUDED.

    The banner lists every off-slate date, so asserting on the whole embed
    passes whether or not the individual rows are marked — the first version of
    these tests did exactly that, and deleting the per-row note left it green.
    A reader scanning fields rather than the preamble needs the per-row mark.
    """
    embed = _embed(candidates, slate_date=slate_date)
    return " ".join(str(f.get("value", "")) for f in embed.get("fields") or [])


def test_the_embed_says_loudly_when_the_rows_are_not_tonights_slate():
    row = board_row(game_date="2025-02-10")
    board = build_decision_board([row], min_ev=-1.0, require_valid_book=False)
    blob = _embed_blob(board)
    assert "2025-02-10" in blob
    assert "NOT TONIGHT'S SLATE" in blob


def test_the_embed_does_not_cry_wolf_on_a_board_that_is_tonights():
    row = board_row(game_date=SLATE)
    board = build_decision_board([row], min_ev=-1.0, require_valid_book=False)
    blob = _embed_blob(board)
    assert "NOT TONIGHT'S SLATE" not in blob
    assert "no game date" not in blob


def test_the_embed_flags_a_row_with_no_game_date():
    row = board_row(game_date=None)
    board = build_decision_board([row], min_ev=-1.0, require_valid_book=False)
    blob = _embed_blob(board)
    assert "no game date" in blob.lower()


def test_each_off_slate_ROW_is_marked_not_only_the_banner():
    """
    The banner is the preamble; these are the fields. A reader who scans the
    rows must see the mark on the row, so this asserts on the field values with
    the banner excluded.
    """
    row = board_row(game_date="2025-02-10")
    board = build_decision_board([row], min_ev=-1.0, require_valid_book=False)
    values = _row_values(board)
    assert "2025-02-10" in values
    assert f"not {SLATE}" in values


def test_a_row_on_the_slate_date_carries_no_per_row_date_noise():
    """Repeating today's date on every row would train a reader to ignore it,
    which is how the mark on the one row that matters gets missed."""
    row = board_row(game_date=SLATE)
    board = build_decision_board([row], min_ev=-1.0, require_valid_book=False)
    values = _row_values(board)
    assert "game date" not in values.lower()


def test_a_dateless_ROW_is_marked_not_only_the_banner():
    row = board_row(game_date=None)
    board = build_decision_board([row], min_ev=-1.0, require_valid_book=False)
    assert "no game date" in _row_values(board).lower()


# --- the guard that catches a regression -------------------------------

def test_verify_wiring_would_fail_if_the_field_were_dropped_again():
    """
    scripts/verify_wiring.py's "a board row keeps the date of the game it
    describes" looks for any of game_date / event_date / game_start_pt being
    truthy on the row. This asserts the contract that check depends on, so the
    two cannot drift apart silently.
    """
    row = board_row(game_date="2025-02-10")
    carried = {
        f for f in ("game_date", "event_date", "game_start_pt")
        if getattr(row, f, None)
    }
    assert carried, "verify_wiring's board-date guard would report a FAIL"


# --- the live chain: CSV round-trip into the embed ----------------------

def test_the_date_survives_the_csv_round_trip_dispatch_actually_does(tmp_path):
    """
    THE LINK THE OTHER TESTS SKIP. They go row -> candidate -> embed in
    memory. The deployed worker does not: run_board writes a CSV, run_dispatch
    reads it back with pd.read_csv and rebuilds duck-typed row objects from
    `frame.to_dict("records")`, and only then builds the embed. A column that
    is dropped, renamed or NaN-ed by that round trip would leave every
    in-memory test green and the live card still misleading.

    This reproduces that exact reconstruction.
    """
    import pandas as pd

    from src.notify.discord import build_decision_board_embed
    from src.quant.decision_board import write_decision_board_csv

    row = board_row(game_date="2025-02-10")
    board = build_decision_board([row], min_ev=-1.0, require_valid_book=False)

    # THROUGH A REAL FILE, because that is where dtype coercion happens.
    target = tmp_path / "decision_board.csv"
    write_decision_board_csv(board, target)
    frame = pd.read_csv(target)
    assert "game_date" in frame.columns

    # run_dispatch's own reconstruction, verbatim in shape.
    rows = [
        type("Row", (), {k: (None if pd.isna(v) else v) for k, v in r.items()})()
        for r in frame.to_dict("records")
    ]
    # One candidate per SIDE, so a single research row yields over and under.
    assert {getattr(r, "game_date", None) for r in rows} == {"2025-02-10"}

    slate = str(frame["slate_date"].iloc[0])
    blob = repr(build_decision_board_embed(rows, slate_date=slate))
    assert "NOT TONIGHT'S SLATE" in blob
    assert "2025-02-10" in blob


def test_a_csv_with_an_empty_game_date_cell_reads_as_unknown_not_as_today(tmp_path):
    """
    A null round-trips through CSV as an empty cell and comes back as NaN,
    which run_dispatch maps to None. It must not become the string "nan" --
    that would print as a date and read as a real one.
    """
    import pandas as pd

    from src.notify.discord import build_decision_board_embed
    from src.quant.decision_board import write_decision_board_csv

    row = board_row(game_date=None)
    board = build_decision_board([row], min_ev=-1.0, require_valid_book=False)
    target = tmp_path / "decision_board.csv"
    write_decision_board_csv(board, target)
    frame = pd.read_csv(target)
    assert frame["game_date"].isna().all()

    rows = [
        type("Row", (), {k: (None if pd.isna(v) else v) for k, v in r.items()})()
        for r in frame.to_dict("records")
    ]
    assert all(getattr(r, "game_date", "sentinel") is None for r in rows)

    blob = repr(build_decision_board_embed(rows, slate_date=SLATE))
    assert "no game date" in blob.lower()
    assert "nan" not in blob.lower().replace("financ", "")
