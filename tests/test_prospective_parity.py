from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.prospective_parity import FeatureParityError, assert_prospective_parity


def frame() -> pd.DataFrame:
    return pd.DataFrame({
        "GAME_ID": ["A", "B"],
        "PLAYER_ID": ["1", "2"],
        "PTS_L5": [20.0, 22.0],
        "MIN_L5": [30.0, np.nan],
    })


def compare(batch, live, **kwargs):
    return assert_prospective_parity(
        batch, live, keys=["GAME_ID", "PLAYER_ID"],
        features=["PTS_L5", "MIN_L5"], **kwargs,
    )


def test_row_order_is_not_semantic():
    result = compare(frame(), frame().iloc[::-1])
    assert result.rows == 2
    assert result.max_absolute_error == {"PTS_L5": 0.0, "MIN_L5": 0.0}
    assert result.compared_values == {"PTS_L5": 2, "MIN_L5": 1}


@pytest.mark.parametrize("side", ["batch", "live"])
def test_duplicate_identity_is_not_hidden_by_intersection(side):
    a, b = frame(), frame()
    duplicate = pd.concat([frame(), frame().iloc[[0]]], ignore_index=True)
    if side == "batch":
        a = duplicate
    else:
        b = duplicate
    with pytest.raises(FeatureParityError, match="duplicate identity"):
        compare(a, b)


@pytest.mark.parametrize("mutation", [
    "missing_row", "extra_row", "wrong_event", "null_key", "missing_column",
    "value", "missingness", "infinity", "nonnumeric", "all_missing",
    "empty", "duplicate_column",
])
def test_parity_rejects_bad_evidence(mutation):
    a, b = frame(), frame()
    if mutation == "missing_row":
        b = b.iloc[[0]]
    elif mutation == "extra_row":
        extra = b.iloc[[0]].copy()
        extra["GAME_ID"] = "C"
        b = pd.concat([b, extra], ignore_index=True)
    elif mutation == "wrong_event":
        b.loc[0, "GAME_ID"] = "C"
    elif mutation == "null_key":
        b.loc[0, "PLAYER_ID"] = None
    elif mutation == "missing_column":
        b = b.drop(columns="PTS_L5")
    elif mutation == "value":
        b.loc[0, "PTS_L5"] += 0.1
    elif mutation == "missingness":
        b.loc[1, "MIN_L5"] = 0.0
    elif mutation == "infinity":
        a.loc[0, "PTS_L5"] = b.loc[0, "PTS_L5"] = np.inf
    elif mutation == "nonnumeric":
        b["PTS_L5"] = ["bad", "22"]
    elif mutation == "all_missing":
        a["MIN_L5"] = b["MIN_L5"] = np.nan
    elif mutation == "empty":
        a, b = a.iloc[:0], b.iloc[:0]
    elif mutation == "duplicate_column":
        b = pd.concat([b, b[["PTS_L5"]]], axis=1)
    with pytest.raises(FeatureParityError):
        compare(a, b)


@pytest.mark.parametrize("tolerance", [-1.0, np.inf, np.nan, True])
def test_invalid_tolerance(tolerance):
    with pytest.raises(FeatureParityError):
        compare(frame(), frame(), absolute_tolerance=tolerance)


def test_tolerance_is_explicit_and_absolute():
    b = frame()
    b.loc[0, "PTS_L5"] += 1e-9
    compare(frame(), b, absolute_tolerance=1e-8)
    with pytest.raises(FeatureParityError):
        compare(frame(), b, absolute_tolerance=1e-12)


def test_empty_keys_are_not_a_parity_test():
    with pytest.raises(FeatureParityError):
        assert_prospective_parity(frame(), frame(), keys=[], features=["PTS_L5"])
