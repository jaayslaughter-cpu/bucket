"""Routing a pick'em row from the EV gate to the payout engine.

The gate returns ``route = PICKEM_ENTRY_ROUTE`` on a pick'em row. Until
``src/quant/dfs_entry.py`` existed, nothing read that field, so the advertised
path was unreachable and every pick'em row still ended at an abstention. These
tests pin the reading of it, and pin the refusals that stop a mismatched
benchmark from being priced as if it matched.

The odds and lines below are TEST FIXTURES. No real operator board or book
quote is asserted anywhere.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.quant.contracts import MarketContext, PropMarketSnapshot
from src.quant.dfs_entry import (
    LEG_ABSTAIN,
    LEG_READY,
    PickemLeg,
    combine_probability_sources,
    resolve_leg_probability,
    route_pickem_entry,
)
from src.quant.dfs_payouts import (
    PAYOUT_EV_ABSTAIN,
    PAYOUT_EV_READY,
    DfsPayoutStructure,
    ProbabilitySource,
    benchmark_fair_probability,
)

FIXTURE = "TEST FIXTURE — not a real platform's payout table"


def power(n: int, multiple: float) -> DfsPayoutStructure:
    return DfsPayoutStructure(n, {n: multiple}, source=FIXTURE, label=f"{n}-pick power")


def pickem_row(line: float = 25.5, *, game_id: str = "g1") -> MarketContext:
    """An operator's row: a payout multiplier, no two-way price."""
    return MarketContext(
        game_id=game_id,
        status="VALID",
        market="PTS",
        line=line,
        payout_multiplier=3.0,
        is_pickem=True,
        source="test-operator",
    )


def benchmark_leg(
    leg_id: str,
    *,
    line: float = 25.5,
    benchmark_line: float | None = 25.5,
    over: int = -130,
    under: int = 110,
    side: str = "over",
) -> PickemLeg:
    return PickemLeg(
        leg_id=leg_id,
        market=pickem_row(line, game_id=leg_id),
        side=side,
        benchmark_over_american=over,
        benchmark_under_american=under,
        benchmark_line=benchmark_line,
        benchmark_source="test-benchmark",
    )


# --- the route is actually read -----------------------------------------

def test_a_pickem_row_resolves_through_the_route_rather_than_dead_ending():
    """The whole point: a routed row gets a probability, not an abstention."""
    out = resolve_leg_probability(benchmark_leg("l1"))
    assert out.status == LEG_READY
    assert out.probability == pytest.approx(benchmark_fair_probability(-130, 110))
    assert out.probability_source is ProbabilitySource.SHARP_BENCHMARK


def test_a_two_way_book_row_is_refused_as_a_pickem_leg():
    """
    A row that CLEARS the gate is a sportsbook market, not a pick'em leg.

    Pricing it against an operator's payout table would measure a book's quote
    against a different product's matrix.
    """
    two_way = MarketContext(
        game_id="g1", status="VALID", market="PTS", line=25.5,
        over_odds_american=-110, under_odds_american=-110,
    )
    out = resolve_leg_probability(
        PickemLeg("l1", two_way, model_probability=0.6)
    )
    assert out.status == LEG_ABSTAIN
    assert "two-way market" in out.reason
    assert out.probability is None


def test_an_invalid_row_passes_the_gates_own_reason_through():
    stale = MarketContext(game_id="g1", status="DATA_NOT_AVAILABLE", is_pickem=True)
    out = resolve_leg_probability(PickemLeg("l1", stale, model_probability=0.6))
    assert out.status == LEG_ABSTAIN
    assert "not VALID" in out.reason


def test_a_snapshot_is_narrowed_to_a_context_like_the_gate_expects():
    snapshot = PropMarketSnapshot(
        game_id="g1", status="VALID", market="PTS", line=25.5,
        payout_multiplier=3.0, is_pickem=True, bookmaker="test-operator",
    )
    out = resolve_leg_probability(
        PickemLeg("l1", snapshot, model_probability=0.58)
    )
    assert out.status == LEG_READY
    assert out.line == 25.5


# --- line matching is exact and hard -------------------------------------

def test_a_benchmark_on_a_different_line_abstains_rather_than_falling_back():
    """
    Half a point is a different contract, and the model number is not a
    substitute for it. Falling back would hide a join fault behind a plausible
    probability.
    """
    leg = benchmark_leg("l1", line=25.5, benchmark_line=24.5)
    leg = PickemLeg(
        leg_id=leg.leg_id, market=leg.market, side=leg.side,
        benchmark_over_american=leg.benchmark_over_american,
        benchmark_under_american=leg.benchmark_under_american,
        benchmark_line=leg.benchmark_line,
        benchmark_source=leg.benchmark_source,
        model_probability=0.62,          # available, and must NOT be used
    )
    out = resolve_leg_probability(leg)
    assert out.status == LEG_ABSTAIN
    assert "does not match" in out.reason
    assert out.probability is None, "fell back to the model on a line mismatch"


def test_a_matching_line_is_accepted_through_float_round_tripping():
    """25.5 parsed from JSON and from YAML must still be the same contract."""
    out = resolve_leg_probability(
        benchmark_leg("l1", line=25.5, benchmark_line=25.5 + 1e-12)
    )
    assert out.status == LEG_READY


def test_a_benchmark_without_a_line_cannot_be_shown_to_match():
    out = resolve_leg_probability(benchmark_leg("l1", benchmark_line=None))
    assert out.status == LEG_ABSTAIN
    assert "no line" in out.reason


def test_an_operator_row_without_a_line_abstains():
    row = MarketContext(
        game_id="g1", status="VALID", market="PTS", line=None,
        payout_multiplier=3.0, is_pickem=True,
    )
    out = resolve_leg_probability(PickemLeg("l1", row, model_probability=0.6))
    assert out.status == LEG_ABSTAIN
    assert "no finite line" in out.reason


# --- a one-sided benchmark is not a benchmark ---------------------------

@pytest.mark.parametrize("over,under", [(-130, None), (None, 110)])
def test_a_one_sided_benchmark_is_refused_rather_than_used_with_its_hold(over, under):
    leg = PickemLeg(
        leg_id="l1", market=pickem_row(), side="over",
        benchmark_over_american=over, benchmark_under_american=under,
        benchmark_line=25.5, model_probability=0.6,
    )
    out = resolve_leg_probability(leg)
    assert out.status == LEG_ABSTAIN
    assert "one side" in out.reason
    assert out.probability is None


# --- the model fallback, and what it is labelled ------------------------

def test_the_model_is_used_only_when_no_benchmark_was_supplied_at_all():
    out = resolve_leg_probability(
        PickemLeg("l1", pickem_row(), model_probability=0.58)
    )
    assert out.status == LEG_READY
    assert out.probability == pytest.approx(0.58)
    assert out.probability_source is ProbabilitySource.MODEL


def test_a_leg_with_no_probability_source_at_all_abstains():
    out = resolve_leg_probability(PickemLeg("l1", pickem_row()))
    assert out.status == LEG_ABSTAIN
    assert "no probability source" in out.reason


@pytest.mark.parametrize("bad", [0.0, 1.0, 1.4, -0.2, float("nan")])
def test_a_model_probability_outside_the_open_unit_interval_abstains(bad):
    out = resolve_leg_probability(
        PickemLeg("l1", pickem_row(), model_probability=bad)
    )
    assert out.status == LEG_ABSTAIN
    assert "(0, 1)" in out.reason


def test_an_unknown_side_abstains():
    out = resolve_leg_probability(
        PickemLeg("l1", pickem_row(), side="middle", model_probability=0.6)
    )
    assert out.status == LEG_ABSTAIN
    assert "over" in out.reason


def test_the_under_side_takes_the_other_half_of_the_devig():
    leg = benchmark_leg("l1", side="under")
    out = resolve_leg_probability(leg)
    assert out.probability == pytest.approx(
        benchmark_fair_probability(-130, 110, side="under")
    )
    assert out.probability == pytest.approx(
        1.0 - benchmark_fair_probability(-130, 110), abs=1e-9
    )


# --- source combination --------------------------------------------------

def test_a_uniform_source_survives_and_a_mixed_one_is_named():
    B, M = ProbabilitySource.SHARP_BENCHMARK, ProbabilitySource.MODEL
    assert combine_probability_sources([B, B, B]) is B
    assert combine_probability_sources([M, M]) is M
    assert combine_probability_sources([B, B, M]) is ProbabilitySource.MIXED
    assert combine_probability_sources([]) is ProbabilitySource.UNSPECIFIED


# --- the entry ----------------------------------------------------------

def test_a_fully_benchmarked_entry_is_priced_and_market_grounded():
    structure = power(3, 6.0)
    legs = [benchmark_leg(f"l{i}") for i in range(3)]
    out = route_pickem_entry(structure, legs)

    assert out.status == PAYOUT_EV_READY
    assert out.payout is not None
    p = benchmark_fair_probability(-130, 110)
    assert out.payout.expected_value == pytest.approx(p**3 * 6.0 - 1.0, abs=1e-9)
    assert out.probability_source is ProbabilitySource.SHARP_BENCHMARK
    assert "Market-grounded" in out.payout.disclaimer


def test_a_mixed_entry_is_labelled_mixed_not_market_grounded():
    structure = power(2, 3.0)
    legs = [
        benchmark_leg("l0"),
        PickemLeg("l1", pickem_row(game_id="l1"), model_probability=0.6),
    ]
    out = route_pickem_entry(structure, legs)
    assert out.status == PAYOUT_EV_READY
    assert out.probability_source is ProbabilitySource.MIXED
    assert "weakest leg" in out.payout.disclaimer


def test_one_unresolved_leg_abstains_the_whole_entry():
    """
    Dropping it and pricing the rest would price a SHORTER slip, which against
    the same payout table reads as a better one.
    """
    structure = power(3, 6.0)
    legs = [
        benchmark_leg("l0"),
        benchmark_leg("l1"),
        benchmark_leg("l2", benchmark_line=24.5),  # mismatched
    ]
    out = route_pickem_entry(structure, legs)
    assert out.status == PAYOUT_EV_ABSTAIN
    assert out.payout is None
    assert "l2" in out.reason and "does not match" in out.reason
    assert sum(leg.status == LEG_READY for leg in out.legs) == 2


def test_a_leg_count_mismatch_abstains_before_any_leg_is_priced():
    out = route_pickem_entry(power(3, 6.0), [benchmark_leg("l0"), benchmark_leg("l1")])
    assert out.status == PAYOUT_EV_ABSTAIN
    assert "different products" in out.reason
    assert out.payout is None


def test_no_legs_abstains():
    out = route_pickem_entry(power(2, 3.0), [])
    assert out.status == PAYOUT_EV_ABSTAIN
    assert "no legs" in out.reason


def test_a_correlation_matrix_is_passed_through_to_the_copula():
    """Same legs, correlated: the answer must differ from the independent one."""
    structure = power(3, 6.0)
    legs = [benchmark_leg(f"l{i}") for i in range(3)]
    independent = route_pickem_entry(structure, legs)

    rho = np.array([[1.0, 0.35, 0.35], [0.35, 1.0, 0.35], [0.35, 0.35, 1.0]])
    correlated = route_pickem_entry(structure, legs, correlation=rho, n_sims=200_000)

    assert correlated.status == PAYOUT_EV_READY
    # positive correlation raises P(all hit) on same-side legs
    assert (
        correlated.payout.probability_all_hit
        > independent.payout.probability_all_hit
    )


def test_the_entry_is_reachable_from_the_cli(tmp_path):
    """
    The wiring, end to end: a JSON slip in, a priced entry out.

    The route the gate returns was unreachable while no command read it, which
    is the state this test exists to keep from coming back. The payout table
    here is a TEST FIXTURE written into tmp_path, not the shipped catalogue.
    """
    import json

    pytest.importorskip("typer")
    from typer.testing import CliRunner

    from scripts.nba_model_cli import app

    catalog = tmp_path / "payouts.yaml"
    catalog.write_text(
        "as_of: '2026-01-01'\n"
        "source: 'TEST FIXTURE — invented multiples'\n"
        "platforms:\n  testop:\n    power:\n      2: {2: 3.0}\n"
    )
    slip = tmp_path / "slip.json"
    slip.write_text(json.dumps({
        "structure": "testop:power:2",
        "legs": [
            {"leg_id": "a", "market": "PTS", "line": 25.5, "side": "over",
             "payout_multiplier": 3.0,
             "benchmark_over_american": -130, "benchmark_under_american": 110,
             "benchmark_line": 25.5, "benchmark_source": "fixture"},
            {"leg_id": "b", "market": "REB", "line": 7.5, "side": "under",
             "payout_multiplier": 3.0,
             "benchmark_over_american": -125, "benchmark_under_american": 105,
             "benchmark_line": 7.5, "benchmark_source": "fixture"},
        ],
    }))

    result = CliRunner().invoke(app, [
        "dfs-entry", "--entry", str(slip), "--catalog", str(catalog),
    ])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["STATUS"] == PAYOUT_EV_READY
    assert payload["ENTRY"]["PROBABILITY_SOURCE"] == "SHARP_BENCHMARK"
    assert payload["RESEARCH_STATUS"] == "RESEARCH_ONLY"
    assert payload["ADVISORY_SIZE"]["ADVISORY_ONLY"] is True
    assert "as_of" in payload["STRUCTURE_SOURCE"]


def test_the_cli_exits_non_zero_when_the_entry_abstains(tmp_path):
    """An abstention must not read as a priced entry to a shell script."""
    import json

    pytest.importorskip("typer")
    from typer.testing import CliRunner

    from scripts.nba_model_cli import app

    catalog = tmp_path / "payouts.yaml"
    catalog.write_text(
        "as_of: '2026-01-01'\n"
        "source: 'TEST FIXTURE — invented multiples'\n"
        "platforms:\n  testop:\n    power:\n      2: {2: 3.0}\n"
    )
    slip = tmp_path / "slip.json"
    slip.write_text(json.dumps({
        "structure": "testop:power:2",
        "legs": [
            {"leg_id": "a", "line": 25.5, "payout_multiplier": 3.0,
             "benchmark_over_american": -130, "benchmark_under_american": 110,
             "benchmark_line": 24.5},          # mismatched line
            {"leg_id": "b", "line": 7.5, "payout_multiplier": 3.0,
             "model_probability": 0.6},
        ],
    }))

    result = CliRunner().invoke(app, [
        "dfs-entry", "--entry", str(slip), "--catalog", str(catalog),
    ])
    assert result.exit_code == 3, result.output
    payload = json.loads(result.output)
    assert payload["STATUS"] == PAYOUT_EV_ABSTAIN
    assert "does not match" in payload["REASON"]
    assert "ADVISORY_SIZE" not in payload, "an abstaining entry must not be sized"


def _fixture_catalog(tmp_path):
    catalog = tmp_path / "payouts.yaml"
    catalog.write_text(
        "as_of: '2026-01-01'\n"
        "source: 'TEST FIXTURE — invented multiples'\n"
        "platforms:\n  testop:\n    power:\n      2: {2: 3.0}\n"
    )
    return catalog


def test_the_cli_withholds_a_model_sourced_card_from_discord_without_evidence(tmp_path):
    """
    The calibration gate sits in front of publication, not in front of pricing.

    The JSON still shows its numbers — it is a diagnostic. The embed does not,
    because in a channel an ungated model EV reads exactly like a verified one.
    """
    import json

    pytest.importorskip("typer")
    from typer.testing import CliRunner

    from scripts.nba_model_cli import app

    slip = tmp_path / "slip.json"
    slip.write_text(json.dumps({
        "structure": "testop:power:2",
        "legs": [
            {"leg_id": "a", "line": 25.5, "payout_multiplier": 3.0,
             "model_probability": 0.62},
            {"leg_id": "b", "line": 7.5, "payout_multiplier": 3.0,
             "model_probability": 0.60},
        ],
    }))

    result = CliRunner().invoke(app, [
        "dfs-entry", "--entry", str(slip),
        "--catalog", str(_fixture_catalog(tmp_path)), "--discord",
    ])
    assert result.exit_code == 0, result.output
    assert "withheld from publication" in result.output
    # the diagnostic JSON is still complete
    payload = json.loads(result.output[result.output.rindex("{\n  \"STATUS\""):])
    assert payload["ENTRY"]["PAYOUT_EV"] is not None
    assert payload["PUBLICATION"]["PUBLICATION_STATUS"] == "PUBLISH_WITHHELD"
    assert payload["DISCORD"]["status"] == "DRY_RUN"


def test_the_cli_publishes_a_benchmark_sourced_card_without_a_model_backtest(tmp_path):
    import json

    pytest.importorskip("typer")
    from typer.testing import CliRunner

    from scripts.nba_model_cli import app

    slip = tmp_path / "slip.json"
    slip.write_text(json.dumps({
        "structure": "testop:power:2",
        "legs": [
            {"leg_id": "a", "line": 25.5, "payout_multiplier": 3.0,
             "benchmark_over_american": -130, "benchmark_under_american": 110,
             "benchmark_line": 25.5},
            {"leg_id": "b", "line": 7.5, "payout_multiplier": 3.0,
             "benchmark_over_american": -125, "benchmark_under_american": 105,
             "benchmark_line": 7.5},
        ],
    }))

    result = CliRunner().invoke(app, [
        "dfs-entry", "--entry", str(slip),
        "--catalog", str(_fixture_catalog(tmp_path)), "--discord",
    ])
    assert result.exit_code == 0, result.output
    assert "withheld from publication" not in result.output
    assert "SHARP_BENCHMARK" in result.output
    payload = json.loads(result.output[result.output.rindex("{\n  \"STATUS\""):])
    assert payload["PUBLICATION"]["PUBLICATION_STATUS"] == "PUBLISH_ALLOWED"


def test_the_cli_names_the_known_keys_when_the_structure_is_unknown(tmp_path):
    import json

    pytest.importorskip("typer")
    from typer.testing import CliRunner

    from scripts.nba_model_cli import app

    catalog = tmp_path / "payouts.yaml"
    catalog.write_text(
        "as_of: '2026-01-01'\n"
        "source: 'TEST FIXTURE — invented multiples'\n"
        "platforms:\n  testop:\n    power:\n      2: {2: 3.0}\n"
    )
    slip = tmp_path / "slip.json"
    slip.write_text(json.dumps({
        "structure": "nosuch:power:2",
        "legs": [{"leg_id": "a", "line": 25.5, "model_probability": 0.6}],
    }))

    result = CliRunner().invoke(app, [
        "dfs-entry", "--entry", str(slip), "--catalog", str(catalog),
    ])
    assert result.exit_code == 2
    assert "testop:power:2" in result.output


def test_the_entry_serialises_every_leg_and_its_reason():
    structure = power(2, 3.0)
    legs = [benchmark_leg("l0"), benchmark_leg("l1", benchmark_line=24.5)]
    payload = route_pickem_entry(structure, legs).as_dict()

    assert payload["STATUS"] == PAYOUT_EV_ABSTAIN
    assert [leg["LEG_ID"] for leg in payload["LEGS"]] == ["l0", "l1"]
    assert payload["LEGS"][1]["REASON"]
    assert payload["LEGS"][0]["PROBABILITY_SOURCE"] == "SHARP_BENCHMARK"
    assert "ENTRY" not in payload, "an abstaining entry must not carry pricing"
