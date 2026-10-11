"""
One definition of what an NBA game id looks like, at every join that uses one.

WHAT THIS GUARDS, found by audit on 2026-10-11. The feature joins keyed on
``GAME_ID`` stripped whitespace and compared the result literally. NBA game ids
are canonically ten characters, zero-padded (``0022500001``), and
``src/ingestion/boxscores.py`` has always normalised them that way — but the
cached training-pack panels carry them eight-wide with the leading zeros gone
(``22500001``). Joining the two gave **0 matches out of 1322**, so every
``MKT_*``, ``DEF_*`` and Elo column arrived 100% NaN.

WHY THAT WAS WORSE THAN AN ERROR. Those columns carry 21.7%-27.5% of each
booster's gain, and XGBoost does not fail on NaN — it sends the row down the
default branch. Measured on the real PTS booster, blanking them shifts the
predicted probability by a mean of 0.14 while the slate exits 0. The layers did
warn and did leave the columns null rather than filling a neutral value, which
is why this was recoverable at all; nothing made the join itself robust.

ONE READER IS THE FIX, which is the lesson this repository keeps relearning
(``board_markets``, the PRA rollups, the parlay ledger -- each was two readers
of one fact that disagreed). ``utils.ids.normalise_game_id`` is that reader, and
the AST test below stops a fourth hand-rolled copy appearing.

RESEARCH_ONLY. Pure key normalisation; nothing here fetches, prices or sizes.
"""

from __future__ import annotations

import ast
import inspect

import pandas as pd
import pytest

from src.utils.ids import normalise_game_id

# --- the normaliser itself ------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("22500001", "0022500001"),      # the cached-panel form
        ("0022500001", "0022500001"),    # already canonical, unchanged
        (" 0022500001 ", "0022500001"),  # whitespace, as the old code handled
        (22500001, "0022500001"),        # an int round-trip through parquet
        ("42400305", "0042400305"),      # playoffs
    ],
)
def test_it_canonicalises_to_ten_wide(raw, expected):
    assert normalise_game_id(pd.Series([raw]))[0] == expected


def test_it_leaves_a_longer_id_alone_rather_than_truncating():
    """
    Truncating would invent a different game. An over-long id is not something
    this function knows how to fix, so it passes through and the join simply
    does not match -- which is visible, where a silent truncation is not.
    """
    assert normalise_game_id(pd.Series(["00225000019"]))[0] == "00225000019"


def test_missing_stays_missing():
    out = normalise_game_id(pd.Series([None, "", "22500001"]))
    assert out[0] is None or pd.isna(out[0])
    assert out[1] is None or pd.isna(out[1])
    assert out[2] == "0022500001"


# --- the joins that depend on it ------------------------------------------


def _panel(game_ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "GAME_ID": game_ids,
        "TEAM_ABBREVIATION": ["LAL"] * len(game_ids),
        "GAME_DATE": pd.to_datetime(["2026-01-20"] * len(game_ids)),
        "PLAYER_ID": [str(i) for i in range(len(game_ids))],
    })


def _market_lines(game_ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "nba_game_id": game_ids,
        "game_date": pd.to_datetime(["2026-01-20"] * len(game_ids)),
        "team_abbr": ["LAL"] * len(game_ids),
        "opening_spread": [-4.5] * len(game_ids),
        "opening_total": [228.5] * len(game_ids),
        "status": ["VALID"] * len(game_ids),
    })


def test_market_features_join_across_an_id_width_mismatch():
    """
    THE AUDIT FINDING, as a test. Panel eight-wide, workbook ten-wide — the
    exact shape that produced 0 of 1322 matches.
    """
    from src.features.market_context import attach_market_context

    out = attach_market_context(_panel(["22500001"]), _market_lines(["0022500001"]))
    assert out["MKT_OPENING_TOTAL"].notna().all(), (
        "the market join still fails across an id-width mismatch, so MKT_* "
        "would be 100% NaN on any panel whose ids lost their padding"
    )
    assert float(out["MKT_OPENING_TOTAL"].iloc[0]) == pytest.approx(228.5)


def test_market_features_join_still_works_when_both_sides_are_canonical():
    """The fix must not break the case that already worked."""
    from src.features.market_context import attach_market_context

    out = attach_market_context(_panel(["0022500001"]), _market_lines(["0022500001"]))
    assert out["MKT_OPENING_TOTAL"].notna().all()


def test_a_genuinely_absent_game_stays_unmatched():
    """
    Normalising must not manufacture a match. Two different games are still two
    different games after padding.
    """
    from src.features.market_context import attach_market_context

    out = attach_market_context(_panel(["22500001"]), _market_lines(["0022599999"]))
    assert out["MKT_OPENING_TOTAL"].isna().all()


def test_elo_joins_across_an_id_width_mismatch():
    """`team_strength` failed identically: 305,614 of 305,614 rows unmatched."""
    from src.features.team_strength import attach_elo_features

    elo = pd.DataFrame({
        "game_id": ["0022500001"],
        "team": ["LAL"],
        "elo_pre": [1550.0],
        "opp_elo_pre": [1480.0],
        "elo_diff": [70.0],
        "elo_win_probability": [0.6],
    })
    out = attach_elo_features(_panel(["22500001"]), elo)
    assert out["TEAM_ELO_PRE"].notna().all(), (
        "Elo still does not join across an id-width mismatch"
    )


# --- and only one definition of it ----------------------------------------


def test_no_layer_hand_rolls_its_own_game_id_padding():
    """
    AST-walked, so a docstring mentioning zfill cannot satisfy it.

    A fourth copy of this rule is how the first three came to disagree. Any
    module that needs it imports the one reader.
    """
    import pathlib

    offenders = []
    for path in pathlib.Path("src").rglob("*.py"):
        if path.name == "ids.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "zfill"
            ):
                offenders.append(f"{path}:{node.lineno}")
    assert not offenders, (
        "these modules pad a game id themselves instead of calling "
        f"utils.ids.normalise_game_id: {offenders}"
    )


def test_the_normaliser_is_the_one_the_joins_actually_call():
    """
    Both joins must route through it — an import that nothing calls would let
    the AST test above pass while the joins stayed literal.
    """
    from src.features import market_context, team_strength

    for mod in (market_context, team_strength):
        src = inspect.getsource(mod)
        assert "normalise_game_id" in src, f"{mod.__name__} does not use the normaliser"
        tree = ast.parse(src)
        called = {
            n.func.id
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        }
        assert "normalise_game_id" in called, (
            f"{mod.__name__} imports the normaliser but never calls it"
        )


# --- fix 3: the builder must say what went wrong --------------------------


def test_attach_team_pace_on_an_already_built_frame_says_so():
    """
    FOUND BY AUDIT, 2026-10-11. Handing `build_feature_matrix` a panel that had
    already been through it raised a bare ``KeyError: 'PACE_MULTIPLIER'`` from
    inside a merge — the column collided, pandas suffixed both copies to
    ``_x``/``_y``, and the next line read the unsuffixed name that no longer
    existed. The message named a column, not the mistake, which sent me looking
    for a missing feature when the real fault was re-entry.
    """
    from src.features.builder import FeatureFrameError, attach_team_pace

    built = pd.DataFrame({
        "TEAM_ABBREVIATION": ["LAL", "LAL"],
        "GAME_ID": ["0022500001", "0022500002"],
        "GAME_DATE": pd.to_datetime(["2026-01-20", "2026-01-22"]),
        "SEASON": ["2025-26"] * 2,
        "FGA": [88, 90], "OREB": [10, 11], "TOV": [12, 13], "FTA": [20, 22],
        # the markers of a frame that has already been built
        "PACE_ROLL": [99.0, 100.0],
        "PACE_MULTIPLIER": [1.01, 1.02],
    })
    with pytest.raises(FeatureFrameError, match="already"):
        attach_team_pace(built)


def test_attach_team_pace_still_works_on_a_raw_frame():
    """The guard must not refuse the case it exists to protect."""
    from src.features.builder import attach_team_pace

    raw = pd.DataFrame({
        "TEAM_ABBREVIATION": ["LAL"] * 6,
        "GAME_ID": [f"002250000{i}" for i in range(1, 7)],
        "GAME_DATE": pd.to_datetime([f"2026-01-{10+i}" for i in range(6)]),
        "SEASON": ["2025-26"] * 6,
        "FGA": [88, 90, 85, 92, 87, 91], "OREB": [10, 11, 9, 12, 10, 11],
        "TOV": [12, 13, 11, 14, 12, 13], "FTA": [20, 22, 18, 24, 20, 21],
    })
    out = attach_team_pace(raw)
    assert "PACE_MULTIPLIER" in out.columns
    assert len(out) == len(raw)
