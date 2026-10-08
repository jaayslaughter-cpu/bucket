"""
PRA had two definitions and the second silently won.

`build_feature_matrix` rolled the PRA column to produce PRA_L5/L10/SEASON and
then PRA_BASELINE/L2/L2_PACE from them. The registered `halflife.pra_rollups`
layer then OVERWROTE five of those with the sums of the component columns —
inside a block whose own comment reads "Each of these ONLY adds columns; none
rewrites the core L2/L5/L10/BASELINE set above".

WHY IT WAS INVISIBLE. The two agreed on every row of the archive panel, where
PTS, REB and AST are null on 0 of 214,381 rows. They differ only when a
component is missing, which is reachable only on the LIVE panel —
`player_game_logs.pts`, `.reb` and `.ast` are independently nullable.

WHAT IT ACTUALLY BROKE. PRA_L2_PACE. The builder computed it from the PRA_L2
it had just built; the layer replaced PRA_L2 and not PRA_L2_PACE; the two then
disagreed on 8 of 14 rows in a fixture with one partial game, including rows
where PACE_MULTIPLIER was exactly 1.0. Two columns that are the same number up
to a multiplier, holding different definitions, side by side.

The surviving definition is the component sum. The tests below pin it, pin the
one case where it is NOT used (the raw label), and pin the invariant the
builder's comment claims — for every layer, not just the one that broke it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.features.builder as builder
from src.features.builder import (
    PRA_COMPONENTS,
    attach_pra_from_components,
    build_feature_matrix,
)
from src.features.halflife import PRA_LAYER_SUFFIXES

#: Every suffix anything in the pipeline produces for PRA.
ALL_SUFFIXES = (
    "L5", "L10", "SEASON", "BASELINE", "L2", "L2_PACE",
    "HL", "HL_SHRINK", "L2_HL",
)


def panel(seed: int = 1, nulls: tuple[tuple[int, str], ...] = ()) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(3):
        for i in range(14):
            rows.append({
                "PLAYER_ID": f"p{p}", "GAME_ID": f"g{p}_{i}",
                "GAME_DATE": pd.Timestamp("2025-01-01") + pd.Timedelta(days=2 * i),
                "SEASON": "2024-25",
                "TEAM_ABBREVIATION": ["LAL", "BOS", "DEN"][p],
                "OPPONENT_ABBREVIATION": ["BOS", "LAL", "MIA"][p],
                "IS_HOME": bool(i % 2), "IS_NEUTRAL_SITE": False,
                "MIN": float(rng.integers(20, 38)),
                "PTS": float(rng.integers(5, 35)),
                "REB": float(rng.integers(1, 14)),
                "AST": float(rng.integers(0, 12)),
                "FG3M": float(rng.integers(0, 6)),
                "STL": float(rng.integers(0, 4)),
                "BLK": float(rng.integers(0, 3)),
                "FGA": float(rng.integers(8, 24)),
                "FTA": float(rng.integers(0, 9)),
                "OREB": float(rng.integers(0, 5)),
                "TOV": float(rng.integers(0, 5)),
            })
    out = pd.DataFrame(rows)
    for row, col in nulls:
        out.loc[row, col] = np.nan
    return out


PARTIAL = ((3, "REB"), (20, "REB"), (31, "AST"), (5, "PTS"))


def _component_sum(frame: pd.DataFrame, suffix: str) -> pd.Series | None:
    cols = [f"{stat}_{suffix}" for stat in PRA_COMPONENTS]
    if not all(c in frame.columns for c in cols):
        return None
    total = pd.to_numeric(frame[cols[0]], errors="coerce")
    for col in cols[1:]:
        total = total + pd.to_numeric(frame[col], errors="coerce")
    return total


# --- one definition, everywhere ------------------------------------------

@pytest.mark.parametrize("nulls", [(), PARTIAL], ids=["clean", "partial components"])
@pytest.mark.parametrize("suffix", ALL_SUFFIXES)
def test_every_pra_rollup_is_the_sum_of_its_components(suffix, nulls):
    """
    The whole family, on a clean panel AND on one with missing components.
    Parametrised over suffixes rather than asserted in a loop so a failure
    names which one drifted.
    """
    frame = build_feature_matrix(panel(nulls=nulls))
    column = f"PRA_{suffix}"
    assert column in frame.columns, f"{column} is no longer produced at all"
    expected = _component_sum(frame, suffix)
    assert expected is not None
    got = frame[column]
    assert (
        (got.isna() & expected.isna()) | np.isclose(got, expected, equal_nan=True)
    ).all(), f"{column} is not the sum of {[f'{s}_{suffix}' for s in PRA_COMPONENTS]}"


def test_pra_l2_pace_agrees_with_pra_l2_times_the_pace_multiplier():
    """
    THE DEFECT THE DOUBLE DEFINITION ACTUALLY CAUSED. PRA_L2_PACE was built
    from the definition that was then overwritten, so it disagreed with the
    PRA_L2 shipped beside it on 8 of 14 rows — including rows where the
    multiplier was exactly 1.0, which is as plain as a contradiction gets.
    """
    frame = build_feature_matrix(panel(nulls=PARTIAL))
    assert "PACE_MULTIPLIER" in frame.columns, "fixture no longer produces pace"
    lhs = frame["PRA_L2_PACE"]
    rhs = frame["PRA_L2"] * frame["PACE_MULTIPLIER"]
    mismatched = int(
        (~((lhs.isna() & rhs.isna()) | np.isclose(lhs, rhs, equal_nan=True))).sum()
    )
    assert mismatched == 0, (
        f"{mismatched} rows where PRA_L2_PACE != PRA_L2 * PACE_MULTIPLIER — the "
        "two carry different definitions of PRA again"
    )


def test_the_rolled_pra_column_is_not_what_the_features_use():
    """
    Pins which of the two definitions survived, by showing the OTHER one gives
    a different answer on the same data. Without this the suite would pass if
    both definitions were quietly swapped back.
    """
    source = panel(nulls=PARTIAL)
    frame = build_feature_matrix(source.copy())

    # The discarded definition: roll the PRA column itself.
    work = source.copy()
    work["GAME_DATE"] = pd.to_datetime(work["GAME_DATE"])
    work = work.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)
    work["PRA_direct"] = work["PTS"] + work["REB"] + work["AST"]
    rolled = (
        work.groupby("PLAYER_ID", sort=False)["PRA_direct"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    )
    shipped = frame.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)["PRA_L5"]
    differing = int(
        (~((shipped.isna() & rolled.isna()) | np.isclose(shipped, rolled, equal_nan=True))).sum()
    )
    assert differing > 0, (
        "the two definitions agree on this fixture, so it is not exercising "
        "the divergence and this test proves nothing"
    )


# --- the exception: the LABEL propagates ---------------------------------

def test_the_raw_pra_label_still_propagates_a_missing_component():
    """
    The one place the component sum is WRONG. A feature is an estimate of an
    expectation and may use each component's own denominator; a LABEL is the
    thing that happened, and a partial sum would read as a real total and
    train the model on a wrong target.
    """
    source = panel(nulls=PARTIAL)
    frame = build_feature_matrix(source.copy())
    assert frame["PRA"].isna().sum() == len(PARTIAL), (
        "the PRA label no longer propagates a missing component, so a row "
        "missing one stat now carries a total that looks real"
    )


# --- the invariant the builder claims about itself -----------------------

def test_no_additive_layer_rewrites_a_column_the_builder_already_owns():
    """
    THE INVARIANT THAT WAS FALSE. build_feature_matrix's own comment says of
    the additive layers: "Each of these ONLY adds columns; none rewrites the
    core L2/L5/L10/BASELINE set above." halflife.pra_rollups rewrote five.

    Checked for EVERY registered layer rather than the one that broke it,
    because the next violation will not be in the same module.

    WHAT IT CANNOT SEE, stated rather than left to be discovered: it compares
    VALUES, so a layer that rewrites a column with an identical number is
    invisible to it. That is deliberate and it is also why
    `test_the_layer_owns_only_the_suffixes_nothing_else_produces` exists —
    putting the core suffixes back into the layer's list while both callers
    use the one shared helper is harmless to every number and still a
    redundancy worth refusing, and only the by-name test catches it.
    """
    source = panel(nulls=PARTIAL)
    original = builder._ADDITIVE_FEATURE_LAYERS
    offenders: dict[str, list[str]] = {}
    try:
        for name, _ in original:
            builder._ADDITIVE_FEATURE_LAYERS = [
                (n, f) for n, f in original if n != name
            ]
            without = build_feature_matrix(source.copy())
            builder._ADDITIVE_FEATURE_LAYERS = original
            with_it = build_feature_matrix(source.copy())

            changed = []
            for col in without.columns:
                if col not in with_it.columns:
                    continue
                a, b = without[col], with_it[col]
                if a.dtype.kind not in "fiub" or b.dtype.kind not in "fiub":
                    continue
                an, bn = pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")
                if not ((an.isna() & bn.isna()) | np.isclose(an, bn, equal_nan=True)).all():
                    changed.append(col)
            if changed:
                offenders[name] = sorted(changed)
    finally:
        builder._ADDITIVE_FEATURE_LAYERS = original

    assert not offenders, (
        "these layers rewrite columns that exist without them, which the "
        f"builder's own comment says none of them do: {offenders}"
    )


def test_the_layer_owns_only_the_suffixes_nothing_else_produces():
    """
    The narrowing, pinned by name. If a suffix is added back to the layer's
    list that the builder also produces, the double definition returns.
    """
    core = {"L5", "L10", "SEASON", "BASELINE", "L2", "L2_PACE"}
    assert not (set(PRA_LAYER_SUFFIXES) & core), (
        f"the layer claims a suffix the builder owns: "
        f"{sorted(set(PRA_LAYER_SUFFIXES) & core)}"
    )
    assert set(PRA_LAYER_SUFFIXES) == {"L15", "HL", "HL_SHRINK", "L2_HL"}


def test_the_layer_no_longer_writes_the_label_a_second_time():
    """The builder owns PRA. Two writers of one column is how this started."""
    import inspect

    from src.features import halflife

    source = inspect.getsource(halflife.attach_pra_component_rollups)
    assert 'out["PRA"] =' not in source


# --- the shared helper ---------------------------------------------------

def test_the_helper_skips_a_suffix_whose_components_are_not_all_present():
    """A part-sum named for a total is the thing being avoided."""
    frame = pd.DataFrame({"PTS_L5": [10.0], "REB_L5": [5.0]})  # no AST_L5
    out = attach_pra_from_components(frame.copy(), ("L5",))
    assert "PRA_L5" not in out.columns


def test_the_layer_delegates_to_the_shared_helper(monkeypatch):
    """
    Two implementations of one formula is what this whole file is about, so a
    second one is not allowed to grow back quietly.

    BEHAVIOURAL, because the first version of this test was a source grep for
    "attach_pra_from_components" — and the function's own DOCSTRING names the
    helper, so replacing the call with a hand-rolled loop left the substring
    present and the test green. The same trap as a comment that mentions the
    call it is supposed to be proving. Here the helper is replaced and the
    layer has to actually reach it.
    """
    from src.features import halflife

    calls: list[tuple[str, ...]] = []

    def _spy(frame, suffixes):
        calls.append(tuple(suffixes))
        return frame

    monkeypatch.setattr(builder, "attach_pra_from_components", _spy)
    halflife.attach_pra_component_rollups(
        pd.DataFrame({"PTS": [1.0], "REB": [1.0], "AST": [1.0]})
    )
    assert calls == [tuple(PRA_LAYER_SUFFIXES)], (
        "the layer did not call the shared definition — it has its own "
        f"summing loop again (calls seen: {calls})"
    )
