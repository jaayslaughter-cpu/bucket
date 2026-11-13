"""
tests/test_return_ramp.py -- the night the rolling windows are wrong.

WHY THIS LAYER EXISTS. Every `{STAT}_L5` and `{STAT}_L10` on this panel
averages a player's last APPEARANCES. On his first game back from an absence
those windows describe him healthy and at full minutes, which is the one night
he is least likely to play them -- so the feature set overstates him exactly
where the error is predictable. `absences.py` answers "who is out tonight" and
`teammate_cascade.py` answers "how many of his teammates are out"; neither
answers "is HE just back, and from how long out?".

HOW THE ABSENCE IS SEEN AT ALL. The panel holds only players who appeared, so
an absence leaves no row, and a gap in a player's appearances measured against
his team's schedule IS the absence. These tests build panels with rows
deliberately removed, because that is what the real data looks like.

RESEARCH_ONLY. Schedule arithmetic. No wager.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.return_ramp import (
    BASELINE_WINDOW,
    EMITTED,
    RETURNING_THRESHOLD,
    attach_return_ramp_features,
)


def panel(
    appearances: dict[str, list[int]],
    *,
    n_games: int = 6,
    team: str = "LAL",
    minutes: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    A panel of APPEARANCES ONLY, which is what this project's panel is.

    `appearances` maps a player to the team-game indices he played. A game he
    did not play simply has no row, exactly as in production.
    """
    minutes = minutes or {}
    rows = []
    for game in range(n_games):
        for player, played in appearances.items():
            if game in played:
                rows.append({
                    "PLAYER_ID": player,
                    "TEAM_ABBREVIATION": team,
                    "GAME_ID": f"002260000{game}",
                    "GAME_DATE": f"2026-11-{game + 1:02d}",
                    "MIN": float(minutes.get(player, 30.0)),
                })
    return pd.DataFrame(rows)


def test_a_player_who_misses_two_games_reports_two_and_is_flagged() -> None:
    """The arithmetic, on the case the layer is named for."""
    out = attach_return_ramp_features(panel({"A": [0, 1, 4, 5], "B": list(range(6))}))
    back = out[(out["PLAYER_ID"] == "A") & (out["GAME_ID"] == "0022600004")]
    assert float(back["RET_GAMES_MISSED"].iloc[0]) == 2.0
    assert float(back["RET_IS_RETURNING"].iloc[0]) == 1.0


def test_a_player_who_plays_every_game_reports_zero_and_is_never_flagged() -> None:
    out = attach_return_ramp_features(panel({"A": list(range(6)), "B": list(range(6))}))
    a = out[out["PLAYER_ID"] == "A"]
    # The first appearance has no prior game and is NaN by design; the rest 0.
    assert a["RET_GAMES_MISSED"].iloc[0] != a["RET_GAMES_MISSED"].iloc[0]  # NaN
    assert (a["RET_GAMES_MISSED"].iloc[1:] == 0.0).all()
    assert (a["RET_IS_RETURNING"].iloc[1:] == 0.0).all()


def test_a_long_absence_reports_its_true_length() -> None:
    """A 1-game absence and a 4-game absence must be distinguishable, which is
    the whole reason this emits a count and not just a flag."""
    out = attach_return_ramp_features(panel({"A": [0, 5], "B": list(range(6))}))
    back = out[(out["PLAYER_ID"] == "A") & (out["GAME_ID"] == "0022600005")]
    assert float(back["RET_GAMES_MISSED"].iloc[0]) == 4.0


def test_one_missed_game_is_not_a_return() -> None:
    """RETURNING_THRESHOLD is 2: a single missed game is routine rest."""
    assert RETURNING_THRESHOLD == 2
    out = attach_return_ramp_features(panel({"A": [0, 2, 3], "B": list(range(6))}))
    back = out[(out["PLAYER_ID"] == "A") & (out["GAME_ID"] == "0022600002")]
    assert float(back["RET_GAMES_MISSED"].iloc[0]) == 1.0
    assert float(back["RET_IS_RETURNING"].iloc[0]) == 0.0


def test_a_first_appearance_is_unknown_rather_than_zero() -> None:
    """
    Reporting 0 missed games on a player's first row would say "he played the
    previous game", which is a claim about a game he has no relationship to.
    """
    out = attach_return_ramp_features(panel({"A": list(range(6))}))
    first = out.iloc[0]
    assert pd.isna(first["RET_GAMES_MISSED"])
    assert pd.isna(first["RET_IS_RETURNING"])
    assert float(first["RET_FIRST_APPEARANCE_FOR_TEAM"]) == 1.0


def test_the_first_appearance_flag_separates_unknown_from_abstained() -> None:
    """
    A model handed only the count cannot tell "no prior game" from "this layer
    produced nothing". The flag is what makes those two NaNs different.
    """
    out = attach_return_ramp_features(panel({"A": list(range(3))}))
    assert out["RET_FIRST_APPEARANCE_FOR_TEAM"].tolist() == [1.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# LEAKAGE -- CLAUDE.md rule 1
# ---------------------------------------------------------------------------


def test_the_baseline_never_contains_tonights_minutes() -> None:
    """
    THE MOST DIRECT LEAK AVAILABLE ON THIS PANEL.

    Change only the LAST row's MIN and the baseline on that row must not move:
    it is the mean of his PREVIOUS appearances. If it moves, the window
    contains the target's own game.
    """
    low = panel({"A": list(range(4))}, minutes={"A": 20.0})
    high = low.copy()
    high.loc[high.index[-1], "MIN"] = 99.0

    out_low = attach_return_ramp_features(low)
    out_high = attach_return_ramp_features(high)
    assert (
        float(out_low["RET_BASELINE_MIN_L5"].iloc[-1])
        == float(out_high["RET_BASELINE_MIN_L5"].iloc[-1])
        == 20.0
    )


def test_the_baseline_window_never_crosses_from_one_player_to_another() -> None:
    """
    THE BUG THE FIRST VERSION OF THIS MODULE SHIPPED WITH, caught on the smoke
    sample before it reached a test.

    `groupby(...)["min"].shift(1).rolling(W).mean()` groups only the SHIFT.
    `.rolling` then runs down the whole Series and its window crosses player
    boundaries, so the second player's FIRST appearance took a baseline of
    30.0 borrowed from the first player's rows, where the honest answer is
    NaN. Both players here have constant and different minutes, so any
    crossing shows up as a value that is neither 30 nor 10 -- or, on a first
    appearance, as a number where there should be none.
    """
    frame = panel(
        {"A": list(range(5)), "B": list(range(5))},
        minutes={"A": 30.0, "B": 10.0},
    )
    out = attach_return_ramp_features(frame)

    for player, expected in (("A", 30.0), ("B", 10.0)):
        rows = out[out["PLAYER_ID"] == player]
        assert pd.isna(rows["RET_BASELINE_MIN_L5"].iloc[0]), (
            f"{player}'s first appearance has no prior game, so it can have "
            "no baseline; a number here was borrowed from another player"
        )
        later = rows["RET_BASELINE_MIN_L5"].iloc[1:]
        assert (later == expected).all(), later.tolist()


def test_the_baseline_respects_its_window_length() -> None:
    """A mean over the previous five appearances, not over all of them."""
    assert BASELINE_WINDOW == 5
    minutes = [10.0, 10.0, 10.0, 10.0, 10.0, 40.0, 40.0]
    frame = panel({"A": list(range(7))}, n_games=7)
    frame["MIN"] = minutes
    out = attach_return_ramp_features(frame)
    # The last row's window is his previous five appearances -- rows 1..5,
    # i.e. 10,10,10,10,40 -> 16.0. NOT rows 2..6: the shift means the window
    # closes on the appearance BEFORE this one, so tonight's 40 is excluded.
    # (A first pass asserted 22.0 by counting the window from the wrong end.)
    assert float(out["RET_BASELINE_MIN_L5"].iloc[-1]) == pytest.approx(16.0)
    # And the row before it sees only tens, which pins the window's far edge.
    assert float(out["RET_BASELINE_MIN_L5"].iloc[-2]) == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# a trade is not a return
# ---------------------------------------------------------------------------


def test_a_first_appearance_for_a_new_team_is_unknown_not_a_long_absence() -> None:
    """
    His appearances for his old team say nothing about games missed for the
    new one. Counting across the two schedules would invent an absence from
    two unrelated calendars.
    """
    old = panel({"A": [0, 1, 2]}, team="BOS")
    new = panel({"A": [3, 4]}, team="LAL")
    out = attach_return_ramp_features(pd.concat([old, new], ignore_index=True))
    lal = out[out["TEAM_ABBREVIATION"] == "LAL"]
    assert pd.isna(lal["RET_GAMES_MISSED"].iloc[0])
    assert float(lal["RET_FIRST_APPEARANCE_FOR_TEAM"].iloc[0]) == 1.0
    assert float(lal["RET_GAMES_MISSED"].iloc[1]) == 0.0


# ---------------------------------------------------------------------------
# abstention, and the shape of the output
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dropped", ["PLAYER_ID", "TEAM_ABBREVIATION", "GAME_ID",
                                     "GAME_DATE", "MIN"])
def test_a_missing_input_abstains_whole_and_names_the_column(dropped: str) -> None:
    """
    A partial set a caller cannot tell from a complete one is worse than none:
    the same posture `box_rates` takes when migration 011's columns are absent.
    """
    frame = panel({"A": list(range(4))}).drop(columns=[dropped])
    out = attach_return_ramp_features(frame)
    assert out["RET_STATUS"].iloc[0].startswith("DATA_NOT_AVAILABLE")
    assert dropped in out["RET_STATUS"].iloc[0]
    for column in EMITTED:
        assert out[column].isna().all()


def test_an_unparseable_date_abstains_rather_than_ordering_by_accident() -> None:
    """The team's schedule order is derived from the date; without it the
    team-game numbering would be whatever order the rows happened to arrive in."""
    frame = panel({"A": list(range(4))})
    frame["GAME_DATE"] = "not-a-date"
    out = attach_return_ramp_features(frame)
    assert "GAME_DATE" in out["RET_STATUS"].iloc[0]
    assert out["RET_GAMES_MISSED"].isna().all()


def test_the_emitted_columns_stay_aligned_to_the_rows_they_describe() -> None:
    """
    THE TEST THAT FIRST ASSERTED NOTHING, which is worth recording.

    Its original form checked that PLAYER_ID, GAME_ID and MIN came back in
    the input's order. Those columns are carried on the copied frame and are
    never reordered, so the assertion held even with the internal re-sort
    deleted -- it could not fail, and a test that cannot fail is not a test.

    The real hazard is the EMITTED columns being written back in the layer's
    internal sort order while the rest of the frame stays in the caller's, so
    that every value lands on the wrong row. This version pins each emitted
    value to the (player, game) it belongs to, computed once on an ordered
    frame and once on a shuffled one. Deleting the re-sort now fails it.
    """
    ordered = panel({"A": [0, 1, 4], "B": list(range(6))})
    shuffled = ordered.sample(frac=1.0, random_state=11).reset_index(drop=True)

    out_ordered = attach_return_ramp_features(ordered)
    out_shuffled = attach_return_ramp_features(shuffled)

    def by_row(frame: pd.DataFrame) -> dict[tuple[str, str], float]:
        return {
            (str(r["PLAYER_ID"]), str(r["GAME_ID"])): (
                -1.0 if pd.isna(r["RET_GAMES_MISSED"]) else float(r["RET_GAMES_MISSED"])
            )
            for _, r in frame.iterrows()
        }

    assert by_row(out_shuffled) == by_row(out_ordered), (
        "an emitted value landed on a different row once the input was "
        "shuffled, so the internal sort order leaked into the output"
    )
    # And the carried columns are still in the caller's order, as before.
    assert out_shuffled["PLAYER_ID"].tolist() == shuffled["PLAYER_ID"].tolist()


def test_a_game_no_player_row_covers_is_not_counted_as_missed_by_anyone() -> None:
    """
    The schedule is built from the panel, so a team game with no rows at all
    does not exist here. Counting it as missed would charge every player on
    the roster with an absence for which this panel holds no evidence.
    """
    # Nobody has a row for team-game 2: it is absent from the panel entirely.
    out = attach_return_ramp_features(panel({"A": [0, 1, 3], "B": [0, 1, 3]}))
    for player in ("A", "B"):
        rows = out[out["PLAYER_ID"] == player]
        assert (rows["RET_GAMES_MISSED"].iloc[1:] == 0.0).all(), (
            "a game the panel does not contain was counted as an absence"
        )


def test_the_layer_emits_exactly_what_it_advertises() -> None:
    out = attach_return_ramp_features(panel({"A": list(range(4))}))
    for column in EMITTED:
        assert column in out.columns
    assert out["RET_STATUS"].iloc[0] == "OK"


# ---------------------------------------------------------------------------
# REAL-WORLD DATA SHAPES -- added 2026-11-13, after probing the layer with
# malformed input rather than the clean panels above.
# ---------------------------------------------------------------------------


def test_duplicate_rows_for_one_game_never_produce_a_negative_count() -> None:
    """
    THE DEFECT NINETEEN PASSING TESTS DID NOT SEE.

    Every test above supplies one row per appearance, which is what a healthy
    panel holds. A panel that carries a DUPLICATE (player, team, game) row --
    a data defect, but a reachable one -- made the first version of this layer
    difference a team-game number against ITSELF: 0 - 1 = -1, emitted as
    RET_GAMES_MISSED = -1.0.

    A negative count of games missed is meaningless, it did not crash, and it
    would have travelled into the A/B feature matrix looking like a number.
    That is the same shape as the outage earlier in this project's week: a
    fault invisible to tests built on clean input and visible only against a
    real-world shape.
    """
    rows = [
        {"PLAYER_ID": 1, "TEAM_ABBREVIATION": "LAL", "GAME_ID": "g0",
         "GAME_DATE": "2026-11-01", "MIN": 30.0},
        # The duplicate.
        {"PLAYER_ID": 1, "TEAM_ABBREVIATION": "LAL", "GAME_ID": "g0",
         "GAME_DATE": "2026-11-01", "MIN": 30.0},
        {"PLAYER_ID": 1, "TEAM_ABBREVIATION": "LAL", "GAME_ID": "g1",
         "GAME_DATE": "2026-11-02", "MIN": 30.0},
    ]
    out = attach_return_ramp_features(pd.DataFrame(rows))

    counts = out["RET_GAMES_MISSED"].dropna()
    assert (counts >= 0).all(), (
        f"a negative games-missed reached the output: {counts.tolist()}"
    )
    # Both duplicate rows describe the same game, so both must carry the same
    # value -- one right answer and one nonsense answer is not a fix.
    duplicated = out[out["GAME_ID"] == "g0"]["RET_GAMES_MISSED"]
    assert duplicated.isna().all(), duplicated.tolist()
    # And the following game still reads as played-the-last-one.
    assert float(out[out["GAME_ID"] == "g1"]["RET_GAMES_MISSED"].iloc[0]) == 0.0


def test_a_duplicated_game_counts_once_in_the_baseline_window() -> None:
    """
    A duplicate must not be allowed to weight one night twice. With a 40-minute
    game duplicated and four 10-minute games after it, a window that counted
    the duplicate twice would pull the mean above the honest figure.
    """
    rows = [
        {"PLAYER_ID": 1, "TEAM_ABBREVIATION": "LAL", "GAME_ID": "g0",
         "GAME_DATE": "2026-11-01", "MIN": 40.0},
        {"PLAYER_ID": 1, "TEAM_ABBREVIATION": "LAL", "GAME_ID": "g0",
         "GAME_DATE": "2026-11-01", "MIN": 40.0},
    ] + [
        {"PLAYER_ID": 1, "TEAM_ABBREVIATION": "LAL", "GAME_ID": f"g{g}",
         "GAME_DATE": f"2026-11-{g + 1:02d}", "MIN": 10.0}
        for g in range(1, 5)
    ]
    out = attach_return_ramp_features(pd.DataFrame(rows))
    # Last row's window: g0(40), g1(10), g2(10), g3(10) -> 17.5.
    # Counting g0 twice would give (40+40+10+10+10)/5 = 22.0.
    assert float(out["RET_BASELINE_MIN_L5"].iloc[-1]) == pytest.approx(17.5)


def test_a_missing_team_or_game_id_degrades_rather_than_crashing() -> None:
    """A null join key must not take the slate down; it costs its own row."""
    rows = [
        {"PLAYER_ID": 1, "TEAM_ABBREVIATION": None, "GAME_ID": "g0",
         "GAME_DATE": "2026-11-01", "MIN": 30.0},
        {"PLAYER_ID": 1, "TEAM_ABBREVIATION": "LAL", "GAME_ID": None,
         "GAME_DATE": "2026-11-02", "MIN": 30.0},
        {"PLAYER_ID": 1, "TEAM_ABBREVIATION": "LAL", "GAME_ID": "g2",
         "GAME_DATE": "2026-11-03", "MIN": 30.0},
    ]
    out = attach_return_ramp_features(pd.DataFrame(rows))
    assert len(out) == 3
    assert out["RET_STATUS"].iloc[0] == "OK"


def test_an_empty_panel_passes_through_without_raising() -> None:
    """A slate with no rows is a real state; it must not be an exception."""
    empty = pd.DataFrame(columns=["PLAYER_ID", "TEAM_ABBREVIATION", "GAME_ID",
                                  "GAME_DATE", "MIN"])
    out = attach_return_ramp_features(empty)
    assert len(out) == 0
    for column in EMITTED:
        assert column in out.columns


def test_the_row_count_is_never_changed_by_this_layer() -> None:
    """
    A feature layer that merges must not duplicate or drop rows. The gap is
    computed on a deduplicated frame and merged back `many_to_one`, so a
    regression that broke that validation would multiply panel rows -- and a
    silently longer panel is a very expensive defect downstream.
    """
    frame = panel({"A": [0, 1, 4], "B": list(range(6))})
    assert len(attach_return_ramp_features(frame)) == len(frame)
    # And with duplicates present.
    doubled = pd.concat([frame, frame.head(2)], ignore_index=True)
    assert len(attach_return_ramp_features(doubled)) == len(doubled)
