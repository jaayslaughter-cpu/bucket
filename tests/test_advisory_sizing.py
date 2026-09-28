"""Advisory unit sizing. READ-ONLY reference numbers, never execution.

The arithmetic is checkable by hand for the binary case and pinned against it
for the general one.
"""

from __future__ import annotations

import math

import pytest

from src.quant.advisory_sizing import (
    DEFAULT_KELLY_FRACTION,
    DEFAULT_MAX_CAP_UNITS,
    recommended_units_binary,
    recommended_units_for_entry,
    recommended_units_multi_outcome,
)
from src.quant.dfs_payouts import (
    DfsPayoutStructure,
    ProbabilitySource,
    evaluate_pickem_entry,
    independent_hit_count_distribution,
)

FIXTURE = "TEST FIXTURE — not a real platform's payout table"


def american_to_decimal(american: int) -> float:
    return 1.0 + (american / 100.0 if american > 0 else 100.0 / abs(american))


# --- the binary closed form --------------------------------------------

@pytest.mark.parametrize("p,american,expected_f", [
    (0.58, -110, 0.1180),
    (0.48, 120, 0.0467),
])
def test_binary_kelly_matches_the_worked_examples(p, american, expected_f):
    out = recommended_units_binary(p, american_to_decimal(american))
    assert out.full_kelly_fraction == pytest.approx(expected_f, abs=1e-4)
    assert out.recommended_units == pytest.approx(
        round(expected_f * DEFAULT_KELLY_FRACTION * 100, 2), abs=0.02
    )


def test_a_negative_edge_sizes_to_zero_not_to_a_negative_number():
    """0.51 at -115 is below breakeven; 'bet the other side' is not an action."""
    out = recommended_units_binary(0.51, american_to_decimal(-115))
    assert out.full_kelly_fraction < 0
    assert out.recommended_units == 0.0
    assert "no edge" in out.reason


def test_the_cap_is_hard_and_reports_the_uncapped_suggestion():
    out = recommended_units_binary(0.90, 3.0, max_cap_units=2.5)
    assert out.recommended_units == 2.5
    assert out.capped is True
    assert "uncapped suggestion was" in out.reason


def test_exactly_breakeven_sizes_to_zero():
    """p = 1/decimal is the breakeven, where f* is exactly 0."""
    out = recommended_units_binary(1 / 3, 3.0)
    assert out.full_kelly_fraction == pytest.approx(0.0, abs=1e-12)
    assert out.recommended_units == 0.0


@pytest.mark.parametrize("bad_p", [0.0, 1.0, 1.05, -0.1, float("nan")])
def test_an_impossible_probability_is_refused_not_sized(bad_p):
    out = recommended_units_binary(bad_p, 3.0)
    assert out.recommended_units == 0.0
    assert "not a probability" in out.reason


@pytest.mark.parametrize("bad_d", [1.0, 0.5, 0.0, -2.0])
def test_odds_that_return_no_profit_are_refused(bad_d):
    out = recommended_units_binary(0.6, bad_d)
    assert out.recommended_units == 0.0
    assert "exceed 1.0" in out.reason


# --- the general solver, and why it is needed --------------------------

def test_the_general_solver_reduces_to_the_closed_form():
    """Measured floor is ~1.3e-9; 1e-8 is asserted, not a comfort tolerance."""
    p, multiple = 0.40, 3.0
    binary = recommended_units_binary(p, multiple)
    general = recommended_units_multi_outcome([1 - p, p], [0.0, multiple])
    assert general.full_kelly_fraction == pytest.approx(
        binary.full_kelly_fraction, abs=1e-8
    )
    assert general.recommended_units == pytest.approx(binary.recommended_units)


def test_a_flex_sized_by_its_top_tier_alone_is_badly_understated():
    """The substantive reason the binary formula is not enough.

    A 6-pick flex paying 25x/2.6x/0.25x at p=0.60: sizing from P(all hit) and
    the 25x multiple alone discards the two partial-hit tiers, which are part of
    the return. Measured: 0.17 units against 1.51, a factor of about nine.
    """
    n, p = 6, 0.60
    tiers = {6: 25.0, 5: 2.6, 4: 0.25}
    probabilities = independent_hit_count_distribution([p] * n)
    multiples = [tiers.get(k, 0.0) for k in range(n + 1)]

    top_tier_only = recommended_units_binary(probabilities[-1], tiers[6])
    whole_distribution = recommended_units_multi_outcome(probabilities, multiples)

    assert whole_distribution.recommended_units > top_tier_only.recommended_units * 5
    assert whole_distribution.method == "multi_outcome"


def test_a_non_positive_ev_entry_sizes_to_zero():
    probabilities = independent_hit_count_distribution([0.5, 0.5, 0.5])
    out = recommended_units_multi_outcome(probabilities, [0.0, 0.0, 0.0, 2.0])
    assert out.recommended_units == 0.0
    assert "EV is" in out.reason


def test_misaligned_probabilities_and_payouts_abstain():
    out = recommended_units_multi_outcome([0.5, 0.5], [0.0, 1.0, 5.0])
    assert out.recommended_units == 0.0
    assert "aligned by hit count" in out.reason


def test_a_distribution_that_does_not_sum_to_one_abstains():
    out = recommended_units_multi_outcome([0.2, 0.2], [0.0, 5.0])
    assert out.recommended_units == 0.0
    assert "not 1" in out.reason


def test_a_zero_paying_tier_does_not_produce_nan():
    """log(1 - f + f*0) goes to -inf at f = 1, so the domain must be kept open."""
    probabilities = independent_hit_count_distribution([0.7, 0.7])
    out = recommended_units_multi_outcome(probabilities, [0.0, 0.0, 3.0])
    assert math.isfinite(out.recommended_units)
    assert math.isfinite(out.full_kelly_fraction)


# --- routing from a priced entry ---------------------------------------

def test_an_all_or_nothing_entry_routes_to_the_binary_form():
    structure = DfsPayoutStructure(3, {3: 6.0}, source=FIXTURE)
    entry = evaluate_pickem_entry(
        structure, [0.6, 0.6, 0.6], source=ProbabilitySource.SHARP_BENCHMARK
    )
    multiples = [structure.payout_for(k) for k in range(4)]
    out = recommended_units_for_entry(entry, multiples)
    assert out.method == "binary"


def test_a_tiered_entry_routes_to_the_solver():
    structure = DfsPayoutStructure(3, {3: 3.25, 2: 1.09}, source=FIXTURE)
    entry = evaluate_pickem_entry(
        structure, [0.6, 0.6, 0.6], source=ProbabilitySource.SHARP_BENCHMARK
    )
    multiples = [structure.payout_for(k) for k in range(4)]
    out = recommended_units_for_entry(entry, multiples)
    assert out.method == "multi_outcome"


def test_an_unpriced_entry_gets_no_size():
    structure = DfsPayoutStructure(3, {3: 6.0}, source=FIXTURE)
    abstained = evaluate_pickem_entry(structure, [0.6, 0.6])   # wrong leg count
    out = recommended_units_for_entry(abstained, [0, 0, 0, 6.0])
    assert out.recommended_units == 0.0
    assert "cannot be sized" in out.reason


# --- the guardrails ----------------------------------------------------

def test_no_bankroll_is_an_input_anywhere():
    """No bankroll in means no currency amount out, structurally."""
    import inspect

    import src.quant.advisory_sizing as mod

    for name, fn in inspect.getmembers(mod, inspect.isfunction):
        if fn.__module__ != mod.__name__:
            continue
        params = inspect.signature(fn).parameters
        offenders = [
            p for p in params
            if any(w in p.lower() for w in ("bankroll", "balance", "dollar", "usd", "stake"))
        ]
        assert not offenders, f"{name} takes {offenders}, so it could return money"


def test_the_output_is_labelled_advisory_and_carries_no_currency():
    out = recommended_units_binary(0.58, 3.0).as_dict()
    assert out["ADVISORY_ONLY"] is True
    assert not [
        k for k in out
        if any(w in k.upper() for w in ("DOLLAR", "USD", "AMOUNT", "BANKROLL"))
    ], f"a currency field reached the caller: {sorted(out)}"


def test_no_module_under_src_imports_this_one():
    """
    The guardrail: advisory sizing must not reach an execution or dispatch path.

    Checked across ALL of src/ and on real IMPORTS, not on a substring in a
    subset of directories. The earlier version collected every file mentioning
    the module and then kept only those whose path contained "/notify/" or
    "dispatch" — so a future src/execution/orders.py importing it would have
    been collected and then filtered straight back out, which is the one case
    the guardrail exists for. Matching the substring also counted a docstring
    that merely names the module.

    Nothing under src/ may import it at all. A size is held by a caller that
    already computed it (the CLI), never fetched by a module that sends or
    places anything.
    """
    import ast
    import pathlib

    offenders = []
    for path in pathlib.Path("src").rglob("*.py"):
        if "advisory_sizing" in path.name:
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""] + [a.name for a in node.names]
            else:
                continue
            if any("advisory_sizing" in n for n in names):
                offenders.append(str(path))
                break

    assert not offenders, (
        f"a module under src/ imports advisory sizing: {offenders}. A size may "
        "be shown by a caller that already holds it, never fetched by the "
        "sender, an execution path, or anything that places an order."
    )


def test_quarter_kelly_is_the_default_and_the_cap_is_three_units():
    assert DEFAULT_KELLY_FRACTION == 0.25
    assert DEFAULT_MAX_CAP_UNITS == 3.0


# --- the sizing controls themselves -------------------------------------

@pytest.mark.parametrize("fraction", [-0.25, float("nan"), float("inf")])
def test_an_unusable_kelly_fraction_returns_no_size_rather_than_a_negative_one(
    fraction
):
    """
    f* is POSITIVE on a real edge, so f* * -0.25 * 100 is a negative number of
    units and the `suggested > cap` test does not catch it. The module's stated
    invariant — never a negative size — was written about f* and did not hold
    for the multiplier applied to it.
    """
    out = recommended_units_binary(0.60, 2.0, kelly_fraction=fraction)
    assert out.recommended_units == 0.0
    assert "kelly_fraction" in out.reason


@pytest.mark.parametrize("cap", [-1.0, float("nan")])
def test_an_unusable_cap_returns_no_size(cap):
    out = recommended_units_binary(0.60, 2.0, max_cap_units=cap)
    assert out.recommended_units == 0.0
    assert "max_cap_units" in out.reason


def test_a_zero_kelly_fraction_is_allowed_and_sizes_nothing():
    """Zero is a legitimate setting — "compute it but do not size" — not an error."""
    out = recommended_units_binary(0.60, 2.0, kelly_fraction=0.0)
    assert out.recommended_units == 0.0
    assert out.full_kelly_fraction > 0.0, "the edge itself is still reported"


def test_the_cap_is_hard_even_when_it_is_not_a_whole_number_of_cents():
    """
    Capping and then rounding lets the answer back over the limit: min(...) of a
    2.555 cap is 2.555, which rounds to 2.56. The cap is applied last.
    """
    out = recommended_units_binary(0.95, 10.0, kelly_fraction=1.0, max_cap_units=2.555)
    assert out.capped is True
    assert out.recommended_units <= 2.555


# --- misaligned payout vectors ------------------------------------------

def test_a_truncated_payout_vector_abstains_instead_of_sizing_the_wrong_tier():
    """
    The binary fast path used to skip the length check the multi-outcome path
    made: probabilities[-1] is P(all hit) while a truncated multiples[-1] is a
    middle tier's return, so the entry would be sized against a payout it does
    not have.
    """
    class Evaluation:
        status = "PAYOUT_EV_READY"
        count_probabilities = [0.05, 0.20, 0.35, 0.40]   # 3-pick card

    out = recommended_units_for_entry(Evaluation(), [0.0, 0.0, 6.0])  # one short
    assert out.recommended_units == 0.0
    assert "aligned by hit count" in out.reason


def test_an_entry_with_no_count_distribution_abstains_rather_than_raising():
    class Evaluation:
        status = "PAYOUT_EV_READY"
        count_probabilities = []

    out = recommended_units_for_entry(Evaluation(), [0.0, 0.0, 3.0])
    assert out.recommended_units == 0.0
    assert "aligned by hit count" in out.reason


def test_an_aligned_all_or_nothing_entry_still_takes_the_binary_path():
    """The check must not break the routing it guards."""
    class Evaluation:
        status = "PAYOUT_EV_READY"
        count_probabilities = [0.05, 0.20, 0.35, 0.40]

    out = recommended_units_for_entry(Evaluation(), [0.0, 0.0, 0.0, 6.0])
    assert out.method == "binary"
    assert out.recommended_units > 0.0
