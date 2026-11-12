"""
tests/test_box_rates.py — the reader migration 011 exists for.

MIGRATION 009 REFUSED TO ADD THESE COLUMNS, and said what would change its
mind:

    A column nothing reads is a column that rots [...] If a feature later
    needs FG%, that is its own migration with its own reason.

So this file carries two jobs, not one. It tests the rates, and it holds the
migration honest: `test_the_migration_has_a_reader` fails if this layer stops
reading any of the five columns 011 added, which is the condition under which
they would become the rotting columns 009 warned about.

THE FULL CHAIN THAT WAS OPEN AT THE FAR END, for whoever reads this next:

    espn_game        splits "11-18" into FGM/FGA
    espn_panel       STAT_MAP maps all five through
    scraper_panel    COLUMN_MAP DROPPED them -- "no column in
                     player_game_logs, and inventing one here would be a
                     schema change disguised as an import"
    player_game_logs no columns, per migration 009
    migration 009    "that is its own migration with its own reason"

Nothing in that chain was wrong. Each link pointed correctly at the next and
the last one pointed at a future decision. Migration 011 made it.

RESEARCH_ONLY. Box-score ratios on shifted windows. No wager.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.box_rates import (
    RATE_WINDOWS,
    REQUIRED_COLS,
    THREE_POINT_BONUS,
    attach_box_rate_features,
    effective_field_goal_percentage,
    free_throw_percentage,
    offensive_rebound_share,
    three_point_attempt_rate,
)

POSTGAME = ("EFG_PCT", "FT_PCT", "FG3A_RATE", "OREB_SHARE")


def _panel(players: tuple[str, ...] = ("p1", "p2"), games: int = 6) -> pd.DataFrame:
    """A panel shaped like `load_player_panel`'s output, post-011."""
    rows = []
    for pid in players:
        for g in range(games):
            rows.append({
                "PLAYER_ID": pid,
                "SEASON": "2026-27",
                "GAME_DATE": pd.Timestamp("2026-11-01") + pd.Timedelta(days=g),
                "FGM": 6 + g, "FGA": 16,
                "FG3M": 2, "FG3A": 6,
                "FTM": 3, "FTA": 4,
                "OREB": 2, "DREB": 6,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLAUDE.md rule one. Every grouped rolling feature, no exceptions.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("window", RATE_WINDOWS)
def test_no_window_can_see_the_game_it_describes(window: int) -> None:
    """
    THE ONE THAT MATTERS MOST, and asserted against an independent
    recomputation rather than against a property of the output.

    A first attempt at this check used `groupby(...).first()`, which skips
    nulls -- so it reported "first game is not NaN" and proved nothing in
    either direction. The window at row i must equal the mean of rows
    strictly before i, computed outside the layer.
    """
    df = _panel()
    out = attach_box_rate_features(df)
    same_game = effective_field_goal_percentage(df["FGM"], df["FG3M"], df["FGA"])

    for pid in df["PLAYER_ID"].unique():
        mask = out["PLAYER_ID"] == pid
        got = out.loc[mask, f"EFG_PCT_L{window}"].tolist()
        prior = same_game[mask].reset_index(drop=True)

        assert np.isnan(got[0]), (
            f"{pid}'s first game has a prior-window value, so the window "
            f"includes the game it describes"
        )
        for i in range(1, len(got)):
            lo = max(0, i - window)
            expected = prior.iloc[lo:i].mean()
            assert got[i] == pytest.approx(expected), (
                f"{pid} game {i}: window {window} gave {got[i]}, the mean of "
                f"games [{lo}:{i}] is {expected}"
            )


def test_a_window_never_crosses_a_player_boundary() -> None:
    """
    Two players in one frame, and the second must not inherit the first.

    The shift and the roll happen inside one groupby transform for exactly
    this reason; done separately, player two's first game would carry player
    one's last.
    """
    df = _panel(players=("p1", "p2"))
    out = attach_box_rate_features(df)

    for pid in ("p1", "p2"):
        first = out.loc[out["PLAYER_ID"] == pid].iloc[0]
        for rate in POSTGAME:
            for window in RATE_WINDOWS:
                assert np.isnan(first[f"{rate}_L{window}"]), (
                    f"{pid}'s first game carries {rate}_L{window}, which can "
                    f"only have come from another player"
                )


def test_the_same_game_rates_are_dropped() -> None:
    """
    They are scaffolding for the shifted means and must not reach a model.

    A model handed this game's eFG% has been told how the game went -- the
    same reason `scoring_efficiency` drops TS_PCT.
    """
    out = attach_box_rate_features(_panel())
    for rate in POSTGAME:
        assert rate not in out.columns, (
            f"{rate} is a POSTGAME quantity and survived into the output"
        )
        assert f"{rate}_L{RATE_WINDOWS[0]}" in out.columns


# ---------------------------------------------------------------------------
# The arithmetic, including what each rate refuses to answer.
# ---------------------------------------------------------------------------


def test_efg_weights_a_three_at_one_and_a_half_twos() -> None:
    """eFG% = (FGM + 0.5*FG3M) / FGA. The 0.5 is arithmetic, not a knob."""
    # 10 made of 20, four of them threes -> (10 + 2) / 20 = 0.60
    got = effective_field_goal_percentage(
        pd.Series([10.0]), pd.Series([4.0]), pd.Series([20.0]),
    )
    assert got.iloc[0] == pytest.approx(0.60)
    assert THREE_POINT_BONUS == 0.5
    # And it exceeds FG% whenever any three was made, which is the point.
    plain_fg = 10.0 / 20.0
    assert got.iloc[0] > plain_fg


@pytest.mark.parametrize(
    "fn,args",
    [
        (effective_field_goal_percentage, (5.0, 1.0, 0.0)),
        (free_throw_percentage, (0.0, 0.0)),
        (three_point_attempt_rate, (0.0, 0.0)),
        (offensive_rebound_share, (0.0, 0.0)),
    ],
)
def test_an_empty_denominator_is_undefined_not_zero(fn, args) -> None:
    """
    NO ATTEMPTS IS NOT BAD SHOOTING, and 0.0 would say it was.

    A zero would drag every rolling mean down with a value meaning "did not
    shoot" -- the reasoning `scoring_efficiency` gives and migration 011
    repeats for its NULL columns.
    """
    series = [pd.Series([float(a)]) for a in args]
    assert np.isnan(fn(*series).iloc[0])


def test_oreb_share_is_a_split_and_is_not_called_a_percentage() -> None:
    """
    THE NAME IS THE ASSERTION.

    True OREB% is OREB / (OREB + opponent DREB) -- a share of AVAILABLE
    offensive rebounds, needing the opponent's defensive rebounds. This is
    the player's own offensive/defensive split, which is a role signal and a
    different quantity. Two offensive and six defensive is 0.25 of the
    player's own rebounds, and says nothing about how many were there to get.
    """
    got = offensive_rebound_share(pd.Series([2.0]), pd.Series([6.0]))
    assert got.iloc[0] == pytest.approx(0.25)

    out = attach_box_rate_features(_panel())
    for column in out.columns:
        assert "OREB_PCT" not in column, (
            f"{column} names itself a percentage; this layer cannot compute "
            f"OREB% without the opponent's defensive rebounds"
        )


# ---------------------------------------------------------------------------
# Abstention, and the migration's own justification.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dropped", REQUIRED_COLS)
def test_the_layer_abstains_whole_when_any_column_is_absent(dropped: str) -> None:
    """
    A panel written before migration 011 gets no rates, not partial ones.

    Deriving the two rates that happen to be possible would put a column in
    the matrix whose presence depended on which migration had run -- the
    train/serve gap migration 009 was written to close.
    """
    df = _panel().drop(columns=[dropped])
    out = attach_box_rate_features(df)

    assert list(out.columns) == list(df.columns), (
        f"with {dropped} absent the layer emitted "
        f"{set(out.columns) - set(df.columns)}"
    )


def test_a_frame_with_no_player_id_is_returned_untouched() -> None:
    df = _panel().drop(columns=["PLAYER_ID"])
    out = attach_box_rate_features(df)
    assert list(out.columns) == list(df.columns)


def test_the_migration_has_a_reader() -> None:
    """
    MIGRATION 009'S CONDITION, ENFORCED.

    009 refused these columns because "a column nothing reads is a column
    that rots", and allowed them only with "its own reason" -- a reader. If
    this layer stops reading any of the five, 011's columns become exactly
    what 009 refused to create, and this fails.
    """
    for column in ("FGM", "FG3A", "FTM", "OREB", "DREB"):
        assert column in REQUIRED_COLS, (
            f"migration 011 added {column.lower()} and nothing reads it now"
        )


def test_the_panel_carries_what_the_layer_requires() -> None:
    """
    THE GAP THAT MADE THIS WORK NECESSARY, pinned shut.

    The columns were stored but not surfaced: `load_player_panel` is what the
    feature builder sees, and a column absent there is absent from the matrix
    however well the table holds it. Asserted against the emitting code, not
    a live database, since this suite runs with no DATABASE_URL.
    """
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1]
              / "src" / "db" / "repository.py").read_text(encoding="utf-8")
    for column in REQUIRED_COLS:
        assert f'"{column}": r.' in source, (
            f"load_player_panel does not emit {column}, so the feature "
            f"builder cannot read it"
        )


def test_a_garbled_denominator_yields_nan_not_infinity() -> None:
    """
    MUTATION THAT SURVIVED THE FIRST ATTEMPT, and the reason is instructive.

    Replacing `fga.where(fga > 0)` with an unmasked divide changed nothing
    for the zero case, because pandas already returns NaN for 0/0 -- so the
    obvious mutation proved the mask untested when it was merely
    under-specified. What the mask actually guards is a NEGATIVE denominator,
    which no real box score has and migration 011's CHECK constraints forbid,
    but which a garbled value or a mis-parsed cell could produce:

        2 / (2 + -2)  ->  masked NaN,  unmasked +inf
        5 / -1        ->  masked NaN,  unmasked -5.0

    Either would be worse than useless. An inf poisons every rolling mean it
    enters, permanently and silently; a negative eFG% reads as a real figure.
    NaN is the one answer that propagates as "unknown".
    """
    # Denominator sums to zero from a negative component.
    assert np.isnan(
        offensive_rebound_share(pd.Series([2.0]), pd.Series([-2.0])).iloc[0]
    )
    # Outright negative denominators.
    assert np.isnan(
        effective_field_goal_percentage(
            pd.Series([5.0]), pd.Series([1.0]), pd.Series([-1.0]),
        ).iloc[0]
    )
    assert np.isnan(
        free_throw_percentage(pd.Series([3.0]), pd.Series([-4.0])).iloc[0]
    )
    assert np.isnan(
        three_point_attempt_rate(pd.Series([2.0]), pd.Series([-6.0])).iloc[0]
    )


def test_no_emitted_feature_is_infinite_on_a_garbled_row() -> None:
    """
    The same guarantee through the layer, which is where it matters.

    One poisoned row must not make a player's whole window infinite.
    """
    df = _panel()
    df.loc[2, "FGA"] = -1
    df.loc[2, "DREB"] = -2
    out = attach_box_rate_features(df)

    emitted = [c for c in out.columns if c.endswith(
        tuple(f"_L{w}" for w in RATE_WINDOWS)
    )] + ["EFG_PCT_TREND"]
    for column in emitted:
        values = pd.to_numeric(out[column], errors="coerce")
        assert not np.isinf(values.dropna()).any(), (
            f"{column} carries an infinity, which will poison every window "
            f"it enters"
        )


def test_a_rising_shooter_has_a_positive_trend() -> None:
    """
    MUTATION THAT SURVIVED: flipping EFG_PCT_TREND to L10 - L5.

    The subtraction order IS the feature's meaning -- the docstring promises
    "positive means converting above their own baseline" -- and nothing
    asserted it, so a model could have been handed the inverse of the signal
    it was named for. `scoring_efficiency.TS_PCT_TREND` has the same shape
    and the same exposure.

    A player improving game over game must read positive once both windows
    are populated: the short window tracks the recent, better games and the
    long one still carries the earlier, worse ones.
    """
    rows = []
    for g in range(12):
        rows.append({
            "PLAYER_ID": "riser", "SEASON": "2026-27",
            "GAME_DATE": pd.Timestamp("2026-11-01") + pd.Timedelta(days=g),
            # Made shots climb on a fixed attempt count, so eFG% climbs.
            "FGM": 4 + g, "FGA": 20,
            "FG3M": 1, "FG3A": 5, "FTM": 3, "FTA": 4,
            "OREB": 2, "DREB": 6,
        })
    out = attach_box_rate_features(pd.DataFrame(rows))

    last = out.iloc[-1]
    assert last["EFG_PCT_L5"] > last["EFG_PCT_L10"], (
        "the fixture does not actually rise, so this proves nothing"
    )
    assert last["EFG_PCT_TREND"] > 0, (
        f"a steadily improving shooter has a negative trend "
        f"({last['EFG_PCT_TREND']}), so the subtraction is inverted"
    )
    assert last["EFG_PCT_TREND"] == pytest.approx(
        last["EFG_PCT_L5"] - last["EFG_PCT_L10"]
    )
