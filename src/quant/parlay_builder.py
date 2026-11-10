"""Build 2-to-5 leg pick'em slips from the board, one leg per game.

RESEARCH_ONLY. Nothing here places a wager, sizes a stake, or contacts an
operator. It selects which board rows could sit on one slip and prices the
slip against a payout matrix whose provenance it carries.

WHY THIS MODULE EXISTS, AND WHAT IT IS NOT.
``dfs_payouts`` prices a slip once its leg probabilities are in hand.
``dfs_entry`` routes ONE slip's legs through the EV gate. Neither of them ever
decided WHICH rows go on a slip, so no slip was ever assembled: the board
emitted single rows, the payout engine waited for a slip, and the two never
met. That is the same "N readers of one fact, no writer" shape this repository
keeps producing -- a route nobody reads, a config block with no reader, a table
with a grader and no writer -- and this module is the missing writer.

THE SAME-GAME PROHIBITION IS THE POINT, NOT A DETAIL.
At most one leg per game, and at most one leg per player. Two reasons, and the
first is arithmetic rather than taste:

  1. ``evaluate_pickem_entry`` with ``correlation=None`` computes the EXACT
     Poisson-binomial, which assumes the legs are independent. Two overs on
     players in the same game are not: they share a pace, a blowout, a
     referee crew and a minutes budget. ``parlay.evaluate_parlay`` already
     refuses a same-game pair that arrives without a correlation matrix, for
     exactly this reason, and this module does not go around that refusal --
     it removes the case. Forbidding same-game legs is what MAKES the
     independent path the right one here, so the constraint and the pricing
     choice are one decision, not two.
  2. A correlation matrix for NBA teammate props has never been estimated in
     this repository from settled data. ``parlay.estimate_tetrachoric_correlation``
     exists and nothing has fed it a graded sample. Passing a guessed matrix
     would be inventing the one number the correlated path turns on.

INDEPENDENCE ACROSS GAMES IS AN APPROXIMATION AND IS NOT CLAIMED TO BE EXACT.
Legs in different games still share league-wide pace, a national-TV script and
whatever the model got wrong about the whole slate at once. That residual is
much smaller than a shared game script, but it is not zero, and its sign is
unknown without a matrix. Every evaluation this module returns records
``independence_basis`` so a reader is never left to assume the number was exact.

SELECTION IS GREEDY ON THE BOARD'S OWN RANK, DELIBERATELY NOT AN OPTIMISER.
The legs are taken in the order the board already ranked them, skipping any that
would break a constraint. Searching subsets for the highest-EV slip would be
selecting the rows where the model is most overconfident -- the payout matrix
is fixed, so maximising entry EV over subsets maximises the model's own
probabilities, and a miscalibrated model's largest numbers are its most wrong
ones. The board's ranking is the only ordering with a documented basis, so it
is the one used.

WHAT A LEG MUST HAVE, AND WHY EACH REFUSAL IS NAMED RATHER THAN SILENT.
A dropped leg that leaves no trace reads as "no such row existed", which is how
a filter chain ends up emitting nothing while every stage reports success --
the defect the user hit when no slip reached Discord. Every rejection here
carries the row's key and a reason.

  not RECOMMENDED      the board withheld it; putting it on a slip would
                       contradict the gate that withheld it.
  no usable probability  nothing to price.
  no line              a probability with no line prices nothing.
  a line that can push  THE SUBTLE ONE. A whole-number line can tie, and an
                       operator VOIDS a tied leg and re-sizes the slip -- a
                       5-pick with one void pays the 4-pick table. That is a
                       different product, and the fixed HITS -> MULTIPLE matrix
                       cannot represent it. Conditioning on no-push, as
                       ``advisory_sizing`` correctly does for a two-way PRICE,
                       would answer a different question here: it gives the EV
                       of the slip GIVEN nothing tied, not the EV of the entry.
                       Refused.
  pick'em line mismatch  the model's probability was computed at the board's
                       line. If the operator posted a different one, that
                       probability prices a different leg. Half a point on a
                       scoring prop is routinely several points of
                       probability, so this abstains rather than falling back,
                       the same rule ``dfs_entry`` applies to a benchmark.
  same game            above.
  same player          above, and operators forbid it anyway.

THE PAYOUT TABLE IS THE OTHER HALF OF THE ANSWER AND IT IS NOT VERIFIED.
Every structure comes from ``load_payout_catalog``, which folds the catalogue's
``as_of`` into each structure's ``source``. No run in this environment has read
a live operator page, so a slip's EV is only as good as the day the table was
transcribed. ``config/dfs_payouts.yaml`` records what the 2026-09-28 to
2026-11-10 revision moved: up to 20 points of entry EV, and the SIGN on two
tiers.

AND THE LEG PROBABILITIES ARE THE MODEL'S OWN.
``ProbabilitySource.MODEL`` on every slip, because there is no sharp two-way NBA
prop benchmark feed here. A positive entry EV is therefore arithmetic on the
model's claim, not evidence of an edge, and it is wrong by about whatever the
model's calibration is wrong by. Read it beside the graded ECE, never instead.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from src.quant.dfs_entry import LINE_MATCH_TOLERANCE
from src.quant.dfs_payouts import (
    PAYOUT_EV_ABSTAIN,
    DfsPayoutStructure,
    PayoutEvaluation,
    ProbabilitySource,
    evaluate_pickem_entry,
)

logger = logging.getLogger(__name__)

#: The slip sizes this project builds. The user's sequencing is 2-to-5 legs per
#: slip; Underdog STREAKS are a different product, built one leg at a time by
#: `src.quant.streaks`, and nothing here touches them.
SLIP_LEG_COUNTS: tuple[int, ...] = (2, 3, 4, 5)

SLIP_READY = "SLIP_READY"
SLIP_ABSTAIN = "DATA_NOT_AVAILABLE"

#: Recorded on every evaluation so a consumer never has to assume what the
#: Poisson-binomial was entitled to assume.
INDEPENDENCE_BASIS = (
    "At most one leg per game and per player, so no two legs share a game "
    "script and the exact Poisson-binomial is used rather than a copula. "
    "Residual cross-game correlation (league pace, slate-wide model error) is "
    "not modelled and is not zero."
)


class ParlayBuilderError(ValueError):
    """A request that does not describe a buildable slip at all."""


@dataclass(frozen=True)
class SlipLeg:
    """One admitted leg, flattened off the board row that produced it."""

    leg_id: str
    event_id: str
    player_id: str
    player_name: str | None
    market: str
    side: str
    line: float
    probability: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "LEG_ID": self.leg_id,
            "EVENT_ID": self.event_id,
            "PLAYER_ID": self.player_id,
            "PLAYER": self.player_name or self.player_id,
            "MARKET": self.market,
            "SIDE": self.side,
            "LINE": self.line,
            "PROBABILITY": round(self.probability, 6),
        }


@dataclass(frozen=True)
class LegRejection:
    """A row that was not admitted, and the reason, so nothing vanishes."""

    leg_id: str
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {"LEG_ID": self.leg_id, "REASON": self.reason}


@dataclass
class SlipEvaluation:
    """One priced slip, or one named refusal to price one."""

    status: str
    n_legs_requested: int
    structure_key: str
    legs: list[SlipLeg] = field(default_factory=list)
    rejections: list[LegRejection] = field(default_factory=list)
    payout: PayoutEvaluation | None = None
    reason: str | None = None
    independence_basis: str = INDEPENDENCE_BASIS

    @property
    def expected_value(self) -> float | None:
        return None if self.payout is None else self.payout.expected_value

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "STATUS": self.status,
            "N_LEGS_REQUESTED": self.n_legs_requested,
            "STRUCTURE": self.structure_key,
            "REASON": self.reason,
            "INDEPENDENCE_BASIS": self.independence_basis,
            "LEGS": [leg.as_dict() for leg in self.legs],
            "REJECTED": [r.as_dict() for r in self.rejections],
        }
        if self.payout is not None:
            out["ENTRY"] = self.payout.as_dict()
        return out


def _field(row: Any, name: str) -> Any:
    """Read one field off a board candidate or a plain mapping.

    The same duck-typing ``advisory_sizing.advisory_size`` uses, for the same
    reason: the board hands out ``BettingDecisionCandidate`` objects and the
    CSV/JSON round-trip hands out dicts, and a builder that only accepted one
    of them would be unusable from whichever side it did not pick.
    """
    if isinstance(row, Mapping):
        return row.get(name)
    return getattr(row, name, None)


def leg_key(row: Any) -> str:
    """A stable, readable identity for a board row.

    Player AND market AND side AND line: a row identified by player alone
    cannot distinguish the over from the under on the same prop, and a slip
    carrying both sides of one prop loses exactly one leg by construction while
    being priced as two independent legs.
    """
    return ":".join(
        str(_field(row, name) or "")
        for name in ("event_id", "player_id", "target_market", "side", "line")
    )


def _is_whole_line(line: float) -> bool:
    """A whole line can tie; a half line cannot.

    Deliberately the same predicate as ``decision_board.line_can_push``, which
    reads ``features.season.is_whole_number_line``. It is re-derived here from
    the float rather than read off the row, because a row that reached this
    module as a dict may not carry ``line_can_push`` at all, and a missing flag
    must not read as "cannot push".
    """
    return abs(line - round(line)) < 1e-9


def admit_leg(row: Any) -> SlipLeg | LegRejection:
    """Admit one board row as a slip leg, or refuse it with a reason.

    Order matters only for which reason is reported first; every check is
    independent. See the module docstring for why each one is here.
    """
    key = leg_key(row)

    status = _field(row, "decision_status")
    if status != "RECOMMENDED":
        return LegRejection(
            key,
            f"decision_status is {status!r}, not RECOMMENDED: putting it on a "
            f"slip would contradict the gate that withheld it",
        )

    raw_prob = _field(row, "model_prob")
    try:
        probability = float(raw_prob)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return LegRejection(key, f"model_prob is {raw_prob!r}, so there is nothing to price")
    if not (math.isfinite(probability) and 0.0 < probability < 1.0):
        return LegRejection(
            key, f"model_prob {probability!r} is not a probability in (0, 1)"
        )

    raw_line = _field(row, "line")
    try:
        line = float(raw_line)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return LegRejection(
            key,
            f"line is {raw_line!r}; a probability with no line prices nothing, "
            f"because the number is only a probability OF something",
        )
    if not math.isfinite(line):
        return LegRejection(key, f"line {line!r} is not finite")

    if _is_whole_line(line):
        return LegRejection(
            key,
            f"line {line} is a whole number and can tie. The operator voids a "
            f"tied leg and re-sizes the slip, so an N-pick with one void pays "
            f"the (N-1)-pick table -- a different product the fixed payout "
            f"matrix cannot represent",
        )

    pickem_line = _field(row, "pickem_line")
    if pickem_line is not None:
        try:
            posted = float(pickem_line)
        except (TypeError, ValueError):
            return LegRejection(
                key, f"pickem_line is {pickem_line!r}, which is not a line"
            )
        if abs(posted - line) > LINE_MATCH_TOLERANCE:
            return LegRejection(
                key,
                f"the model's probability was computed at {line} but the "
                f"operator posted {posted}; that probability prices a "
                f"different leg, and half a point is routinely several points "
                f"of probability",
            )

    event_id = str(_field(row, "event_id") or "").strip()
    if not event_id:
        return LegRejection(
            key,
            "no event_id, so whether this leg shares a game with another "
            "cannot be established. The same-game prohibition is what entitles "
            "this module to the independent Poisson-binomial, so an unknown "
            "game is refused rather than assumed distinct",
        )

    player_id = str(_field(row, "player_id") or "").strip()
    if not player_id:
        return LegRejection(
            key, "no player_id, so a duplicate player on one slip cannot be ruled out"
        )

    return SlipLeg(
        leg_id=key,
        event_id=event_id,
        player_id=player_id,
        player_name=_field(row, "player_name"),
        market=str(_field(row, "target_market") or ""),
        side=str(_field(row, "side") or ""),
        line=line,
        probability=probability,
    )


def eligible_legs(
    candidates: Iterable[Any],
    n_legs: int,
    *,
    used_leg_ids: Sequence[str] | set[str] | None = None,
) -> tuple[list[SlipLeg], list[LegRejection]]:
    """Take the first ``n_legs`` admissible rows, one per game and per player.

    Returns ``([], rejections)`` when ``n_legs`` cannot be met, NOT a shorter
    list. A 3-leg slip handed back where 5 were asked for would be priced
    against whichever table the caller had in hand, and a shorter slip against
    an unchanged payout table reads as a better one -- the same trap
    ``dfs_entry.route_pickem_entry`` refuses by abstaining on a leg count
    mismatch.

    ``used_leg_ids`` lets a caller build several slips off one board without
    repeating a leg across them, which would make the slips' outcomes
    dependent while each was priced as if it stood alone.
    """
    if not isinstance(n_legs, int) or isinstance(n_legs, bool) or n_legs < 2:
        raise ParlayBuilderError(
            f"n_legs must be an integer of at least 2, got {n_legs!r}: a "
            f"one-leg 'slip' is a single and belongs on the board, not here"
        )

    legs, rejections, _ = _select(candidates, n_legs, used_leg_ids)
    return legs, rejections


def _select(
    candidates: Iterable[Any],
    n_legs: int,
    used_leg_ids: Sequence[str] | set[str] | None,
) -> tuple[list[SlipLeg], list[LegRejection], int]:
    """``eligible_legs`` plus the count that WERE admissible.

    Split out so ``build_slip`` can say "3 of the 5 asked for" instead of
    "fewer than 5". ``eligible_legs`` returns an empty list on a shortfall
    deliberately (see its docstring), and a caller that cannot tell 0 from 4
    cannot write a reason a reader can act on.
    """
    spent = {str(x) for x in (used_leg_ids or ())}
    legs: list[SlipLeg] = []
    rejections: list[LegRejection] = []
    games: set[str] = set()
    players: set[str] = set()

    for row in candidates:
        if len(legs) == n_legs:
            break
        outcome = admit_leg(row)
        if isinstance(outcome, LegRejection):
            rejections.append(outcome)
            continue
        if outcome.leg_id in spent:
            rejections.append(LegRejection(
                outcome.leg_id,
                "already used on another slip from this board; reusing it "
                "would make the two slips' outcomes dependent while each is "
                "priced as if it stood alone",
            ))
            continue
        if outcome.event_id in games:
            rejections.append(LegRejection(
                outcome.leg_id,
                f"game {outcome.event_id} is already on this slip. Same-game "
                f"legs share a pace, a game script and a minutes budget, so "
                f"they are not independent and the exact Poisson-binomial "
                f"would misprice the slip in a direction that depends on the "
                f"sign of a correlation this repository has never estimated",
            ))
            continue
        if outcome.player_id in players:
            rejections.append(LegRejection(
                outcome.leg_id,
                f"player {outcome.player_name or outcome.player_id} is already "
                f"on this slip; two props on one player are strongly dependent "
                f"and operators forbid the pairing",
            ))
            continue
        legs.append(outcome)
        games.add(outcome.event_id)
        players.add(outcome.player_id)

    if len(legs) < n_legs:
        return [], rejections, len(legs)
    return legs, rejections, len(legs)


def build_slip(
    candidates: Iterable[Any],
    structure: DfsPayoutStructure,
    *,
    structure_key: str = "",
    used_leg_ids: Sequence[str] | set[str] | None = None,
) -> SlipEvaluation:
    """Select legs for one structure and price the slip against it.

    ``correlation`` is deliberately never passed: the legs are one-per-game by
    construction, which is the whole justification for the exact
    Poisson-binomial. See the module docstring -- the constraint and the
    pricing choice are one decision.
    """
    label = structure_key or structure.label or f"{structure.n_picks}-pick"
    n_legs = int(structure.n_picks)
    legs, rejections, n_admissible = _select(candidates, n_legs, used_leg_ids)
    out = SlipEvaluation(
        status=SLIP_ABSTAIN,
        n_legs_requested=n_legs,
        structure_key=label,
        legs=legs,
        rejections=rejections,
    )
    if not legs:
        out.reason = (
            f"{n_admissible} eligible leg(s) on this board against the "
            f"{n_legs} this structure needs, so no {n_legs}-pick slip was "
            f"built. {len(rejections)} row(s) were refused and every reason is "
            f"on this evaluation rather than discarded, because a filter chain "
            f"that drops rows silently reports success while emitting nothing"
        )
        return out

    payout = evaluate_pickem_entry(
        structure,
        [leg.probability for leg in legs],
        source=ProbabilitySource.MODEL,
        correlation=None,
    )
    out.payout = payout
    out.reason = payout.reason
    out.status = SLIP_READY if payout.status != PAYOUT_EV_ABSTAIN else SLIP_ABSTAIN
    return out


def build_slips(
    candidates: Sequence[Any],
    catalog: Mapping[str, DfsPayoutStructure],
    *,
    platform: str,
    variant: str,
    leg_counts: Sequence[int] = SLIP_LEG_COUNTS,
    distinct_legs: bool = False,
) -> list[SlipEvaluation]:
    """One slip per requested leg count, for one platform and variant.

    ``distinct_legs=False`` (the default) builds each slip from the top of the
    board, so the 2-leg slip's legs also appear on the 3-leg slip. That is the
    right default for a research board: the slips are ALTERNATIVES a reader
    picks between, not a portfolio to buy together, and giving the 5-leg slip
    the board's 5th-to-9th best legs instead of its best 5 would make it look
    worse for a reason that has nothing to do with the payout table.

    ``distinct_legs=True`` is for the other reading -- several slips considered
    together -- where sharing a leg makes their outcomes dependent while each
    is priced as if it stood alone.

    A leg count with no structure in the catalogue abstains with that named,
    rather than being skipped: a missing tier that vanishes reads as "the
    operator does not offer that slip size".
    """
    out: list[SlipEvaluation] = []
    spent: set[str] = set()
    for n_legs in leg_counts:
        key = f"{platform}:{variant}:{n_legs}"
        structure = catalog.get(key)
        if structure is None:
            out.append(SlipEvaluation(
                status=SLIP_ABSTAIN,
                n_legs_requested=int(n_legs),
                structure_key=key,
                reason=(
                    f"no payout structure {key} in the catalogue, so a "
                    f"{n_legs}-pick slip on this platform cannot be priced. "
                    f"Nothing here invents a multiplier"
                ),
            ))
            continue
        evaluation = build_slip(
            candidates,
            structure,
            structure_key=key,
            used_leg_ids=spent if distinct_legs else None,
        )
        if distinct_legs:
            spent.update(leg.leg_id for leg in evaluation.legs)
        out.append(evaluation)
    return out
