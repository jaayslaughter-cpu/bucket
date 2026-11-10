"""The slip builder: one leg per game, or no slip.

Every guard in `src.quant.parlay_builder` was mutation-tested — the fix
reverted, the named test watched go red, the fix put back. The mutations that
survived a first attempt are named in the tests they forced.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.quant.decision_board import BettingDecisionCandidate
from src.quant.dfs_payouts import (
    ProbabilitySource,
    independent_hit_count_distribution,
    load_payout_catalog,
)
from src.quant.parlay_builder import (
    SLIP_ABSTAIN,
    SLIP_LEG_COUNTS,
    SLIP_READY,
    LegRejection,
    ParlayBuilderError,
    SlipLeg,
    admit_leg,
    build_slip,
    build_slips,
    eligible_legs,
    leg_key,
)


def row(**kw) -> BettingDecisionCandidate:
    """A board row that is admissible unless a test spoils one field."""
    base = dict(
        slate_date="2026-11-10",
        event_id="0022600321",
        player_id="P1",
        player_name="Player One",
        target_market="PTS",
        side="over",
        line=24.5,
        model_prob=0.57,
        decision_status="RECOMMENDED",
    )
    base.update(kw)
    return BettingDecisionCandidate(**base)


def board(n: int, **kw) -> list[BettingDecisionCandidate]:
    """``n`` admissible rows, each in its own game and on its own player."""
    return [
        row(event_id=f"G{i}", player_id=f"P{i}", player_name=f"Player {i}", **kw)
        for i in range(n)
    ]


@pytest.fixture(scope="module")
def catalog():
    return load_payout_catalog()


# --- the same-game prohibition --------------------------------------------


def test_a_second_leg_from_the_same_game_is_refused_and_the_game_is_named():
    """
    THE CONSTRAINT THE WHOLE MODULE TURNS ON. Two legs in one game share a
    pace, a game script and a minutes budget, so the exact Poisson-binomial
    the slip is priced with is not entitled to its independence assumption.

    The refusal must also be NAMED. A silently dropped row reads as "no such
    row existed", which is how a filter chain reports success while emitting
    nothing.
    """
    rows = [
        row(event_id="SAME", player_id="P1"),
        row(event_id="SAME", player_id="P2", player_name="Player Two"),
        row(event_id="OTHER", player_id="P3", player_name="Player Three"),
    ]
    legs, rejections = eligible_legs(rows, 2)

    assert [leg.event_id for leg in legs] == ["SAME", "OTHER"]
    assert len({leg.event_id for leg in legs}) == 2, "two legs shared a game"
    same_game = [r for r in rejections if "SAME" in r.reason and "already on this slip" in r.reason]
    assert same_game, f"the same-game refusal was not reported: {rejections}"


def test_a_same_game_board_yields_no_slip_rather_than_a_shorter_one(catalog):
    """
    A SHORTER SLIP AGAINST AN UNCHANGED PAYOUT TABLE READS AS A BETTER ONE.
    Three rows, all in one game, asked for a 3-pick: exactly one leg is
    admissible, and handing back that one leg to be priced against the 3-pick
    table would claim a 6.0x product on a single pick.

    MUTATION THAT SURVIVED THE FIRST ATTEMPT: returning the partial list from
    `eligible_legs` instead of `[]`. The first version of this test only
    asserted `len(legs) != 3`, which a 1-leg return also satisfies. It now
    asserts the empty list AND that the priced slip abstains.
    """
    rows = [row(event_id="SAME", player_id=f"P{i}", player_name=f"P {i}") for i in range(3)]
    legs, rejections = eligible_legs(rows, 3)
    assert legs == [], f"a shortfall returned {len(legs)} leg(s) to be priced"

    evaluation = build_slip(rows, catalog["underdog:power:3"], structure_key="underdog:power:3")
    assert evaluation.status == SLIP_ABSTAIN
    assert evaluation.payout is None, "a 1-leg slip was priced against a 3-pick table"
    assert "1 eligible leg(s) on this board against the 3" in (evaluation.reason or ""), (
        f"the shortfall did not say how many were eligible: {evaluation.reason!r}"
    )
    assert evaluation.rejections, "the refused rows vanished"


def test_the_shortfall_reason_counts_eligible_legs_rather_than_saying_fewer():
    """
    "FEWER THAN 5" IS NOT ACTIONABLE AND "4 OF 5" IS. The `_select` split
    exists only so this reason can carry the count; a caller that cannot tell
    0 eligible from 4 cannot tell a dead board from a near miss.
    """
    from src.quant.dfs_payouts import structure_from_mapping

    five = structure_from_mapping({
        "n_picks": 5, "payouts": {5: 20.0}, "source": "test", "label": "t:p:5",
    })
    evaluation = build_slip(board(4), five, structure_key="t:p:5")
    assert "4 eligible leg(s) on this board against the 5" in (evaluation.reason or ""), (
        evaluation.reason
    )


# --- the other admission rules --------------------------------------------


def test_a_second_prop_on_one_player_is_refused():
    """Two props on one player are strongly dependent, and operators forbid it."""
    rows = [
        row(event_id="G1", player_id="P1", target_market="PTS"),
        row(event_id="G2", player_id="P1", target_market="REB"),
        row(event_id="G3", player_id="P2", player_name="Player Two"),
    ]
    legs, rejections = eligible_legs(rows, 2)

    assert [leg.player_id for leg in legs] == ["P1", "P2"]
    assert any("already" in r.reason and "on this slip" in r.reason for r in rejections)


def test_both_sides_of_one_prop_cannot_sit_on_one_slip():
    """
    THE DEGENERATE CASE. Over and under on the same prop loses exactly one leg
    by construction, and pricing it as two independent legs at 0.57 each reports
    P(both hit) = 0.32 for an event whose probability is zero. Caught by the
    same-player rule, asserted here separately because the same-player rule
    could be narrowed to same-market without this test noticing.
    """
    rows = [
        row(event_id="G1", player_id="P1", side="over"),
        row(event_id="G1", player_id="P1", side="under", model_prob=0.43),
    ]
    legs, _ = eligible_legs(rows, 2)
    assert legs == [], "a slip carried both sides of one prop"


def test_a_whole_number_line_is_refused_because_a_void_resizes_the_slip():
    """
    THE SUBTLE ONE. A whole line can tie; the operator voids a tied leg and
    re-sizes the slip, so a 5-pick with one void pays the 4-pick table. The
    fixed HITS -> MULTIPLE matrix cannot represent that, and conditioning on
    no-push — which `advisory_sizing` correctly does for a two-way PRICE —
    would answer a different question: the EV given nothing tied, not the EV
    of the entry.
    """
    outcome = admit_leg(row(line=25.0))
    assert isinstance(outcome, LegRejection)
    assert "whole number" in outcome.reason and "void" in outcome.reason


def test_a_half_line_is_admitted_even_though_the_rows_push_flag_defaults_to_true():
    """
    WHY THE PREDICATE IS RE-DERIVED FROM THE FLOAT RATHER THAN READ OFF THE ROW.
    `BettingDecisionCandidate.line_can_push` defaults to True and is only set
    by `expand_row_to_candidates`; a row that reached this module from a CSV or
    a JSON round-trip may carry the stale default, or no such key at all.
    Trusting the flag would refuse every half line on such a board — and a
    missing flag must never read as "cannot push" either.
    """
    candidate = row(line=24.5)
    assert candidate.line_can_push is True, "the fixture no longer exercises the trap"
    assert isinstance(admit_leg(candidate), SlipLeg)
    assert isinstance(admit_leg({"line": 24.5, "event_id": "G", "player_id": "P",
                                 "target_market": "PTS", "side": "over",
                                 "model_prob": 0.57,
                                 "decision_status": "RECOMMENDED"}), SlipLeg)


def test_a_pickem_line_that_disagrees_with_the_board_line_abstains():
    """
    The model's probability was computed at the board's line. If the operator
    posted another, that probability prices a different leg — the same rule
    `dfs_entry` applies to a benchmark, and for the same reason: half a point
    on a scoring prop is routinely several points of probability.
    """
    outcome = admit_leg(row(line=24.5, pickem_line=25.5))
    assert isinstance(outcome, LegRejection)
    assert "24.5" in outcome.reason and "25.5" in outcome.reason

    agreeing = admit_leg(row(line=24.5, pickem_line=24.5))
    assert isinstance(agreeing, SlipLeg), "a matching pick'em line was refused"


def test_a_row_the_board_withheld_never_reaches_a_slip():
    """Slipping an ABSTAIN would contradict the gate that withheld it."""
    outcome = admit_leg(row(decision_status="ABSTAIN"))
    assert isinstance(outcome, LegRejection)
    assert "RECOMMENDED" in outcome.reason


@pytest.mark.parametrize("prob", [None, 0.0, 1.0, float("nan"), "0.57"])
def test_a_row_without_a_usable_probability_is_refused(prob):
    outcome = admit_leg(row(model_prob=prob) if prob != "0.57" else {
        "model_prob": "not a number", "decision_status": "RECOMMENDED",
    })
    assert isinstance(outcome, LegRejection), f"{prob!r} was priced"


def test_a_row_with_no_event_id_is_refused_rather_than_assumed_distinct():
    """
    THE SAME-GAME PROHIBITION IS WHAT ENTITLES THIS MODULE TO THE INDEPENDENT
    POISSON-BINOMIAL, so a row whose game is unknown cannot be let through on
    the assumption that it is a different one. Treating a blank event_id as
    distinct would let an entire same-game stack onto one slip.
    """
    outcome = admit_leg(row(event_id=""))
    assert isinstance(outcome, LegRejection)
    assert "event_id" in outcome.reason

    blanks = [row(event_id="", player_id=f"P{i}", player_name=f"P {i}") for i in range(3)]
    legs, _ = eligible_legs(blanks, 2)
    assert legs == [], "rows with no game formed a slip"


def test_a_row_with_no_player_id_is_refused():
    outcome = admit_leg(row(player_id=""))
    assert isinstance(outcome, LegRejection)
    assert "player_id" in outcome.reason


def test_a_one_leg_slip_is_refused_outright():
    """A single belongs on the board, not in the slip builder."""
    for bad in (1, 0, -3, True, 2.0, "2"):
        with pytest.raises(ParlayBuilderError):
            eligible_legs(board(3), bad)  # type: ignore[arg-type]


def test_the_leg_key_distinguishes_the_two_sides_of_one_prop():
    """
    A row identified by player alone cannot tell the over from the under, and
    `used_leg_ids` keyed that way would treat them as the same leg.
    """
    over = leg_key(row(side="over"))
    under = leg_key(row(side="under"))
    assert over != under, "the two sides of one prop share a leg key"
    assert leg_key(row(line=24.5)) != leg_key(row(line=25.5))


# --- the pricing ----------------------------------------------------------


def test_the_slip_is_priced_against_the_structure_it_names(catalog):
    """
    The entry EV must equal the Poisson-binomial expectation of the named
    table — computed here independently of the module under test, from the
    exact recursion and the catalogue's own multiples.
    """
    structure = catalog["underdog:flex:4"]
    rows = board(4)
    evaluation = build_slip(rows, structure, structure_key="underdog:flex:4")

    assert evaluation.status == SLIP_READY
    assert len(evaluation.legs) == 4
    counts = independent_hit_count_distribution([0.57] * 4)
    expected = sum(
        counts[k] * structure.payout_for(k) for k in range(5)
    ) - 1.0
    assert evaluation.payout is not None
    assert evaluation.payout.expected_value == pytest.approx(expected, abs=1e-12)


def test_the_slip_is_priced_with_no_correlation_matrix_and_says_why(catalog):
    """
    `correlation=None` is only correct BECAUSE same-game legs are forbidden,
    and a consumer must never have to assume that. Every evaluation records
    the basis, including that residual cross-game correlation is not modelled
    and is not zero.

    MUTATION THAT SURVIVED THE FIRST ATTEMPT: deleting the
    `independence_basis` default. The dataclass field still existed as None and
    the first version of this test only checked the attribute was present.
    """
    evaluation = build_slip(board(3), catalog["underdog:power:3"], structure_key="k")
    basis = evaluation.independence_basis
    assert "one leg per game" in basis
    assert "is not zero" in basis, "the residual correlation was claimed away"
    assert evaluation.payout is not None
    assert evaluation.payout.expected_value_stderr in (None, 0.0), (
        "a simulation standard error appeared, so the copula path ran"
    )


def test_every_slip_is_recorded_as_model_sourced(catalog):
    """
    There is no sharp two-way NBA prop benchmark feed here, so a slip's EV is
    arithmetic on the model's own claim. Recording SHARP_BENCHMARK or leaving
    it UNSPECIFIED would let a reader take it for market-grounded.
    """
    evaluation = build_slip(board(2), catalog["underdog:power:2"], structure_key="k")
    assert evaluation.payout is not None
    assert evaluation.payout.probability_source is ProbabilitySource.MODEL
    assert "calibration" in evaluation.payout.disclaimer.lower()


def test_every_priced_slip_carries_the_payout_tables_date(catalog):
    """
    Provenance that lives only in a comment at the top of a YAML file does not
    travel with the number once one slip is evaluated on its own. A stale table
    is the single largest error term here: the 2026-09-28 -> 2026-11-10
    revision moved entry EV by up to 20 points and flipped two signs.
    """
    evaluation = build_slip(board(2), catalog["underdog:power:2"], structure_key="k")
    assert evaluation.payout is not None
    assert "2026-11-10" in (evaluation.payout.structure_source or "")
    assert "not verified" in (evaluation.payout.structure_source or "")


# --- building the set of slips --------------------------------------------


def test_the_builder_covers_two_through_five_legs(catalog):
    """The user's sequencing: 2-to-5 legs per slip."""
    assert SLIP_LEG_COUNTS == (2, 3, 4, 5)
    slips = build_slips(board(9), catalog, platform="underdog", variant="power")
    assert [s.n_legs_requested for s in slips] == [2, 3, 4, 5]
    assert all(s.status == SLIP_READY for s in slips), [s.reason for s in slips]
    assert [len(s.legs) for s in slips] == [2, 3, 4, 5]
    for slip in slips:
        assert len({leg.event_id for leg in slip.legs}) == len(slip.legs)


def test_a_leg_count_the_catalogue_does_not_offer_abstains_with_its_key_named(catalog):
    """
    A missing tier that vanishes from the output reads as "the operator does
    not offer that slip size". prizepicks:flex has no 2-pick, because a flex
    needs room to miss one.
    """
    slips = build_slips(board(9), catalog, platform="prizepicks", variant="flex")
    two = next(s for s in slips if s.n_legs_requested == 2)
    assert two.status == SLIP_ABSTAIN
    assert "prizepicks:flex:2" in (two.reason or "")
    assert two.payout is None


def test_by_default_each_slip_starts_from_the_top_of_the_board(catalog):
    """
    The slips are ALTERNATIVES a reader picks between, not a portfolio. Giving
    the 5-leg slip the board's 5th-to-9th best legs would make it look worse
    for a reason that has nothing to do with the payout table.
    """
    slips = build_slips(board(9), catalog, platform="underdog", variant="power")
    first = [leg.leg_id for leg in slips[0].legs]
    assert [leg.leg_id for leg in slips[-1].legs][:2] == first


def test_distinct_legs_never_reuses_a_leg_across_slips(catalog):
    """
    The other reading: several slips considered together. Sharing a leg makes
    their outcomes dependent while each is priced as if it stood alone.
    """
    slips = build_slips(
        board(14), catalog, platform="underdog", variant="power", distinct_legs=True,
    )
    assert all(s.status == SLIP_READY for s in slips), [s.reason for s in slips]
    seen: list[str] = []
    for slip in slips:
        seen.extend(leg.leg_id for leg in slip.legs)
    assert len(seen) == len(set(seen)) == 14, "a leg appeared on two slips"


def test_a_reused_leg_is_refused_with_a_reason_rather_than_skipped():
    legs, rejections = eligible_legs(board(3), 2, used_leg_ids={leg_key(board(3)[0])})
    assert len(legs) == 2
    assert any("already used on another slip" in r.reason for r in rejections)


# --- the hard lines -------------------------------------------------------


def test_the_builder_places_no_wager_and_promises_no_outcome():
    """
    AGENTS.md section 1. A slip is a research object; nothing here contacts an
    operator, and no output may read as a promise.
    """
    source = Path("src/quant/parlay_builder.py").read_text(encoding="utf-8")
    for forbidden in (
        "place_bet", "submit_entry", "requests.post", "httpx.post",
        "AUTO_PLACED", "guaranteed", "lock of the",
    ):
        assert forbidden not in source, f"{forbidden!r} appeared in a research module"


def test_the_builder_hardcodes_no_payout_multiple():
    """
    Every multiple must come from `config/dfs_payouts.yaml`, which carries the
    date it was transcribed. A number inlined here would be sportsbook data
    invented in code, and it would never go stale in a way anyone could see.
    """
    source = Path("src/quant/parlay_builder.py").read_text(encoding="utf-8")
    body = "\n".join(
        line for line in source.splitlines()
        if not line.lstrip().startswith("#")
    )
    for multiple in ("3.0x", "6.0x", "10.0x", "37.5", "150.0"):
        assert multiple not in body, f"{multiple} is hardcoded; read the catalogue"


# --- today's catalogue ----------------------------------------------------


def test_todays_catalogue_offers_every_slip_size_the_builder_asks_for(catalog):
    """
    2-to-5 legs on the all-or-nothing entry for both operators. A flex needs
    room to miss, so it starts at 3 — asserted separately so a missing 3-pick
    flex cannot hide behind the power tiers.
    """
    for platform in ("underdog", "prizepicks"):
        for n in SLIP_LEG_COUNTS:
            assert f"{platform}:power:{n}" in catalog, f"{platform} power {n} missing"
        for n in (3, 4, 5):
            if platform == "prizepicks" and n == 5:
                continue
            assert f"{platform}:flex:{n}" in catalog


def test_the_catalogue_is_dated_and_unverified_and_says_so(catalog):
    """
    An operator term with no date cannot be audited, and one this environment
    has never read from a live page must not read as verified. Both travel on
    every structure, not just in the file's header comment.

    MUTATION THAT SURVIVED THE FIRST ATTEMPT: reverting `as_of` to the stale
    2026-09-28 while leaving the `source` prose alone. `load_payout_catalog`
    renders `f"{source} (as_of {as_of})"`, and the prose carries its own copy of
    the date — so "2026-11-10 appears in structure.source" was satisfied by the
    prose and said nothing about `as_of`. TWO READERS OF ONE FACT, the defect
    this repository keeps removing. The assertion now names the loader's own
    rendering of `as_of`, and the test below forbids the two dates from
    drifting apart at all.
    """
    for key, structure in catalog.items():
        assert "(as_of 2026-11-10)" in structure.source, (
            f"{key} does not carry the catalogue's as_of date: {structure.source!r}"
        )
        assert "not verified" in structure.source, f"{key} reads as verified"


def test_the_catalogues_two_dates_cannot_drift_apart():
    """
    `as_of` and the date written into the `source` prose are the same fact
    recorded twice, and a file where only one was updated would report the new
    table under the old date or the reverse. Neither is auditable.
    """
    import re

    import yaml

    payload = yaml.safe_load(Path("config/dfs_payouts.yaml").read_text(encoding="utf-8"))
    as_of = str(payload["as_of"])
    in_prose = re.findall(r"\d{4}-\d{2}-\d{2}", str(payload["source"]))
    assert in_prose, "the source prose records no date"
    assert set(in_prose) == {as_of}, (
        f"as_of is {as_of} but the source prose says {sorted(set(in_prose))}"
    )


def test_the_streak_curve_and_the_standard_entry_have_diverged():
    """
    `config/streaks.yaml` noted that its 2-6 tiers coincided with
    `underdog.power`. On today's tables 7 and 8 do NOT: the standard entry pays
    75x and 150x where the streak pays 65x and 120x. That divergence is exactly
    why the two were never allowed to share one table, and if it ever stops
    being asserted the note goes stale silently.
    """
    import yaml

    streaks = yaml.safe_load(Path("config/streaks.yaml").read_text(encoding="utf-8"))
    curve = streaks["platforms"]["underdog"]["streaks"]
    payouts = yaml.safe_load(Path("config/dfs_payouts.yaml").read_text(encoding="utf-8"))
    power = payouts["platforms"]["underdog"]["power"]

    for n in (2, 3, 4, 5, 6):
        assert float(curve[n]) == float(power[n][n]), f"tier {n} drifted apart"
    for n in (7, 8):
        assert float(curve[n]) != float(power[n][n]), (
            f"tier {n} now coincides; the note in streaks.yaml is stale"
        )
