"""
src/quant/leg_correlation.py — fit parlay leg correlations from realised games.

This is step three of the chain that ends in a priced ticket: a real panel,
then real posted prices, then the correlations that say how the legs move
together. ``parlay.evaluate_parlay`` refuses a same-game ticket without
them, and this is where they come from.

YOU CANNOT FIT CORRELATIONS PER PLAYER PAIR. Two specific players share a
few dozen games at most, and a correlation fitted on a few dozen binary
outcomes is noise — noise that moves the parlay probability in whichever
direction happens to flatter the ticket. So pairs are pooled into BUCKETS
by the relationship between the legs and the two markets involved:

    same_player   PTS x REB   one player's own two markets
    same_team     PTS x PTS   two teammates
    opposing_team PTS x REB   a player and an opponent
    different_game            no shared game; assumed 0 unless fitted

A bucket pools every realised pair that matches it, across all players and
all games, which is what makes the sample large enough to mean anything.
The price of that is a prior rather than a bespoke number: two teammates
get the league's same-team PTS x PTS correlation, not their own.

AS-OF DISCIPLINE. A correlation fitted on the season that includes the game
being predicted leaks, in exactly the way a season-wide mean leaks. ``as_of``
keeps only games strictly before the slate, and it is required rather than
defaulted, because a silent "use everything" is the failure this guards.

LINES. The outcome of a leg is defined against a line. Where real posted
lines are not archived, a research proxy column is used and the fitted
correlations inherit whatever bias that proxy carries — which is recorded
in the report rather than hidden.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.models.labels import RESEARCH_LINE_COL, RESEARCH_LINE_SOURCE
from src.quant.parlay import ParlayError, ParlayLeg, estimate_tetrachoric_correlation

logger = logging.getLogger(__name__)

RESEARCH_STATUS = "RESEARCH_ONLY"

Relationship = str  # "same_player" | "same_team" | "opposing_team" | "different_game"

SAME_PLAYER = "same_player"
SAME_TEAM = "same_team"
OPPOSING_TEAM = "opposing_team"
DIFFERENT_GAME = "different_game"

# Below this a bucket is reported but never used: a correlation fitted on a
# small sample is noise, and noise here flatters the ticket.
MIN_PAIRS_PER_BUCKET = 200

# And below THIS, independently. Pairs within one game are not independent
# observations -- they are drawn from a single night's blowout, pace and
# officiating, and they reuse the same handful of outcomes. 25 leg rows from
# ONE game produce 2,700 pairs, and six buckets cleared the 200-pair gate on
# that alone with rho between -0.059 and +0.053, which is noise wearing an
# n of 600. The independent unit is the GAME, so buckets are gated on both.
MIN_GAMES_PER_BUCKET = 50


class LegCorrelationError(RuntimeError):
    """Raised when correlations cannot be fitted from what was supplied."""


@dataclass(frozen=True)
class CorrelationBucket:
    """One fitted relationship/market-pair prior."""

    relationship: Relationship
    market_a: str
    market_b: str
    rho: float | None
    n_pairs: int
    usable: bool
    reason: str | None = None
    # Distinct GAMES contributing. Pairs within one game reuse the same
    # outcomes, so n_pairs alone overstates the evidence by a factor of the
    # per-game pair count -- see MIN_GAMES_PER_BUCKET.
    n_games: int = 0

    @property
    def key(self) -> tuple[str, str, str]:
        a, b = sorted((self.market_a, self.market_b))
        return (self.relationship, a, b)


@dataclass
class CorrelationPriors:
    """Fitted buckets plus the provenance needed to judge them."""

    buckets: dict[tuple[str, str, str], CorrelationBucket] = field(default_factory=dict)
    as_of: str | None = None
    n_games: int = 0
    n_player_games: int = 0
    line_source: str | None = None
    min_pairs: int = MIN_PAIRS_PER_BUCKET
    min_games: int = MIN_GAMES_PER_BUCKET
    research_status: str = RESEARCH_STATUS

    def get(self, relationship: str, market_a: str, market_b: str) -> CorrelationBucket | None:
        a, b = sorted((market_a, market_b))
        return self.buckets.get((relationship, a, b))

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "relationship": b.relationship,
                "market_a": b.market_a,
                "market_b": b.market_b,
                "rho": b.rho,
                "n_pairs": b.n_pairs,
                "n_games": b.n_games,
                "usable": b.usable,
                "reason": b.reason,
            }
            for b in self.buckets.values()
        ])

    def summary(self) -> dict[str, Any]:
        usable = [b for b in self.buckets.values() if b.usable]
        return {
            "research_status": self.research_status,
            "as_of": self.as_of,
            "n_games": self.n_games,
            "n_player_games": self.n_player_games,
            "line_source": self.line_source,
            "min_pairs": self.min_pairs,
            "min_games": self.min_games,
            "n_buckets": len(self.buckets),
            "n_usable": len(usable),
            "usable": sorted(
                ({"key": "/".join(b.key), "rho": round(b.rho, 4),
                  "n_pairs": b.n_pairs, "n_games": b.n_games}
                 for b in usable),
                key=lambda d: -abs(d["rho"]),
            ),
        }


# ---------------------------------------------------------------------------
# realised outcomes
# ---------------------------------------------------------------------------


def _default_line_col(
    market: str, columns: Sequence[str], markets: Sequence[str],
) -> str | None:
    """Which column grades ``market``'s legs when the caller named none.

    This defaulted to the single shared column ``RESEARCH_LINE``, which the
    panel does not carry -- labels.attach_research_line derives it per market
    from ``{stat}_L10`` and attaches it to ONE market's training frame. So the
    default fit skipped every market and raised "No usable markets", measured
    on the real 214,381-row panel: the layer evaluate_parlay depends on could
    not run at its own defaults. And a single shared column cannot be the line
    for three markets anyway -- it would grade REB legs against PTS's line.

    So the per-market ``{stat}_L10`` comes first. RESEARCH_LINE is accepted
    only when exactly one market is being fitted, where it is unambiguous.
    """
    available = set(columns)
    per_market = f"{market}_{RESEARCH_LINE_SOURCE}"
    if per_market in available:
        return per_market
    if len(markets) == 1 and RESEARCH_LINE_COL in available:
        return RESEARCH_LINE_COL
    return None


def realised_leg_outcomes(
    panel: pd.DataFrame,
    *,
    markets: Sequence[str],
    line_col_for: Mapping[str, str] | None = None,
    as_of: str | pd.Timestamp,
    side: str = "over",
) -> pd.DataFrame:
    """
    Turn a player-game panel into one realised binary leg outcome per row.

    ``as_of`` is required: only games strictly BEFORE it are used, because a
    correlation fitted on the game being predicted leaks into it.

    A row is dropped when its line is unknown, or when the stat lands exactly
    on a whole line — that is a push, which is neither a win nor a loss and
    must not be coerced into either.
    """
    required = {"GAME_ID", "GAME_DATE"}
    missing = required - set(panel.columns)
    if missing:
        raise LegCorrelationError(f"Panel is missing {sorted(missing)}")
    if as_of is None:
        raise LegCorrelationError(
            "as_of is required. Fitting on games at or after the slate leaks the "
            "outcome being predicted into the correlation used to predict it."
        )

    player_key = "PLAYER_ID" if "PLAYER_ID" in panel.columns else "PLAYER_NAME"
    if player_key not in panel.columns:
        raise LegCorrelationError("Panel has neither PLAYER_ID nor PLAYER_NAME")

    work = panel.copy()
    work["GAME_DATE"] = pd.to_datetime(work["GAME_DATE"], errors="coerce")
    cutoff = pd.Timestamp(as_of)
    work = work[work["GAME_DATE"].notna() & (work["GAME_DATE"] < cutoff)]
    if work.empty:
        raise LegCorrelationError(
            f"No games strictly before {cutoff.date()}; nothing to fit on"
        )

    lines = dict(line_col_for or {})
    frames: list[pd.DataFrame] = []
    for market in markets:
        if market not in work.columns:
            logger.warning("leg_correlation: market %s absent from the panel", market)
            continue
        line_col = lines.get(market) or _default_line_col(market, work.columns, markets)
        if line_col is None or line_col not in work.columns:
            logger.warning(
                "leg_correlation: no line column for %s (looked for %r); market "
                "skipped", market, line_col or f"{market}_{RESEARCH_LINE_SOURCE}",
            )
            continue
        block = work[[
            "GAME_ID", "GAME_DATE", player_key, market, line_col,
        ]].copy()
        for col in ("TEAM_ABBREVIATION", "OPPONENT_ABBREVIATION"):
            block[col] = work[col] if col in work.columns else None

        stat = pd.to_numeric(block[market], errors="coerce")
        line = pd.to_numeric(block[line_col], errors="coerce")
        block = block.assign(market=market, stat=stat, line=line, line_col=line_col)
        block = block[block["stat"].notna() & block["line"].notna()]

        # A push is neither a win nor a loss. Coercing it to either biases
        # every correlation fitted from these outcomes.
        block = block[block["stat"] != block["line"]]
        over = (block["stat"] > block["line"]).astype(float)
        block["outcome"] = over if side == "over" else 1.0 - over
        frames.append(block[[
            "GAME_ID", "GAME_DATE", player_key, "TEAM_ABBREVIATION",
            "OPPONENT_ABBREVIATION", "market", "line_col", "outcome",
        ]].rename(columns={player_key: "player_key"}))

    if not frames:
        raise LegCorrelationError(
            f"No usable markets among {list(markets)}. Each needs its stat column "
            "and a line column in the panel."
        )
    return pd.concat(frames, ignore_index=True)


def classify_relationship(row_a: Mapping[str, Any], row_b: Mapping[str, Any]) -> Relationship:
    """How two legs in the same game relate. Drives which bucket they pool into."""
    if row_a.get("GAME_ID") != row_b.get("GAME_ID"):
        return DIFFERENT_GAME

    # Same player only when BOTH identities are known and equal. Two unknown
    # players are not the same player, and treating them as one would apply a
    # same-player correlation (a player's own PTS and REB move together far
    # more than two people's do) to two different people.
    key_a, key_b = row_a.get("player_key"), row_b.get("player_key")
    known = (
        key_a is not None and key_b is not None
        and not pd.isna(key_a) and not pd.isna(key_b)
    )
    if known and key_a == key_b:
        return SAME_PLAYER
    team_a, team_b = row_a.get("TEAM_ABBREVIATION"), row_b.get("TEAM_ABBREVIATION")
    if team_a is not None and team_b is not None and not pd.isna(team_a) and not pd.isna(team_b):
        return SAME_TEAM if team_a == team_b else OPPOSING_TEAM
    # Same game, unknown teams: pooling it as same-team would invent a
    # relationship, so it goes to the weaker, more conservative bucket.
    return OPPOSING_TEAM


# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------


def fit_leg_correlations(
    panel: pd.DataFrame,
    *,
    as_of: str | pd.Timestamp,
    markets: Sequence[str] = ("PTS", "REB", "AST"),
    line_col_for: Mapping[str, str] | None = None,
    min_pairs: int = MIN_PAIRS_PER_BUCKET,
    min_games: int = MIN_GAMES_PER_BUCKET,
    max_rows_per_game: int = 200,
) -> CorrelationPriors:
    """
    Fit tetrachoric correlations per (relationship, market, market) bucket.

    Every pair of legs that co-occurred in a game before ``as_of`` is pooled
    into its bucket, and each bucket is fitted only when it clears BOTH
    ``min_pairs`` and ``min_games``. Buckets that do not clear them are
    RETURNED, marked unusable with a reason, rather than dropped — "we could
    not fit this" is a different statement from "these legs are independent",
    and the caller needs to be able to tell them apart.

    ``min_games`` is not redundant with ``min_pairs``. Pairs inside one game
    share that night's blowout, pace and officiating and reuse the same
    outcomes, so they are not independent observations: 25 leg rows from a
    SINGLE game produce 2,700 pairs, and six buckets cleared a 200-pair gate
    on that alone. The game is the independent unit.

    ``max_rows_per_game`` caps the LEG ROWS taken from one game, not the pairs
    they generate — 200 rows still yield up to 19,900 pairs.
    """
    outcomes = realised_leg_outcomes(
        panel, markets=markets, line_col_for=line_col_for, as_of=as_of,
    )

    pairs: dict[tuple[str, str, str], list[tuple[float, float]]] = {}
    games: dict[tuple[str, str, str], set[str]] = {}
    for game_id, block in outcomes.groupby("GAME_ID", sort=False):
        rows = block.to_dict("records")
        if len(rows) > max_rows_per_game:
            rows = rows[:max_rows_per_game]
        for a, b in combinations(rows, 2):
            relationship = classify_relationship(a, b)
            if relationship == DIFFERENT_GAME:
                continue
            market_a, market_b = a["market"], b["market"]
            outcome_a, outcome_b = a["outcome"], b["outcome"]
            if market_b < market_a:
                market_a, market_b = market_b, market_a
                outcome_a, outcome_b = outcome_b, outcome_a
            bucket_key = (relationship, market_a, market_b)
            pairs.setdefault(bucket_key, []).append(
                (float(outcome_a), float(outcome_b))
            )
            games.setdefault(bucket_key, set()).add(str(game_id))

    buckets: dict[tuple[str, str, str], CorrelationBucket] = {}
    for key, observed in pairs.items():
        relationship, market_a, market_b = key
        n = len(observed)
        n_games = len(games.get(key, ()))
        if n < int(min_pairs) or n_games < int(min_games):
            short = (
                f"only {n} realised pairs, need {min_pairs}"
                if n < int(min_pairs)
                else f"only {n_games} distinct game{'' if n_games == 1 else 's'}, "
                     f"need {min_games} (the {n} pairs reuse those games' "
                     "outcomes and are not independent observations)"
            )
            buckets[key] = CorrelationBucket(
                relationship=relationship, market_a=market_a, market_b=market_b,
                rho=None, n_pairs=n, n_games=n_games, usable=False,
                reason=f"{short}. Not fitted — this is 'unknown', not 'independent'.",
            )
            continue
        arr = np.asarray(observed, dtype=float)
        try:
            rho = estimate_tetrachoric_correlation(
                arr[:, 0], arr[:, 1], min_observations=int(min_pairs),
            )
        except ParlayError as exc:
            buckets[key] = CorrelationBucket(
                relationship=relationship, market_a=market_a, market_b=market_b,
                rho=None, n_pairs=n, n_games=n_games, usable=False, reason=str(exc),
            )
            continue
        buckets[key] = CorrelationBucket(
            relationship=relationship, market_a=market_a, market_b=market_b,
            rho=float(rho), n_pairs=n, n_games=n_games, usable=True,
        )

    priors = CorrelationPriors(
        buckets=buckets,
        as_of=str(pd.Timestamp(as_of).date()),
        n_games=int(outcomes["GAME_ID"].nunique()),
        n_player_games=int(len(outcomes)),
        # The columns ACTUALLY used, not the requested default. This label is
        # the record of what bias the fitted correlations inherit, and one that
        # names a column the fit did not read is worse than none.
        line_source=", ".join(
            f"{m}={c}" for m, c in sorted(
                outcomes.drop_duplicates("market")
                .set_index("market")["line_col"].astype(str).items()
            )
        ),
        min_pairs=int(min_pairs),
        min_games=int(min_games),
    )
    logger.info("leg_correlation: %s", priors.summary())
    return priors


# ---------------------------------------------------------------------------
# applying priors to a prospective ticket
# ---------------------------------------------------------------------------



def _side_sign(leg: Any) -> int | None:
    """+1 for an over leg, -1 for an under leg, None when it cannot be told.

    WHY THIS EXISTS. The fitted buckets are OVER/OVER correlations --
    realised_leg_outcomes defaults to side="over" and stores
    ``1.0 - over`` only when asked for the other side -- but a ticket's legs
    each pick a side, and model_prob is P(THIS side wins). Copying rho
    unchanged onto a mixed over/under pair gives the copula the wrong sign of
    dependence, which moves the joint probability the wrong way and with it
    the breakeven and EV.

    Verified by simulation rather than asserted: with a latent over/over rho of
    0.6 and 0.55 marginals, phi(over_a, over_b) = +0.4083 and
    phi(over_a, under_b) = -0.4083, summing to 0.000000, and a latent rho of
    -0.6 reproduces the under pairing. So the correction is rho * s_a * s_b.

    Returns None for an unrecognised or absent side. The CALLER decides what
    that means: correlation_for_legs reads it as "over", matching the side
    realised_leg_outcomes fits by default, and logs that it did so. Refusing
    outright would turn every side-agnostic over/over ticket into an
    abstention, which fixes nothing; the real paths (paper_research) do
    populate the field, so a declared under leg is signed correctly.
    """
    raw = getattr(leg, "side", None)
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if text in ("over", "o", "more"):
        return 1
    if text in ("under", "u", "less"):
        return -1
    return None


def correlation_for_legs(
    legs: Sequence[ParlayLeg],
    priors: CorrelationPriors,
    *,
    leg_teams: Mapping[str, str] | None = None,
    leg_players: Mapping[str, str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Build the matrix ``evaluate_parlay`` needs, from fitted priors.

    Returns the matrix and a list of unresolved pairs. A same-game pair with
    no usable bucket is left at 0 AND named in that list, so the caller can
    refuse rather than quietly treat "unknown" as "independent".
    """
    size = len(legs)
    matrix = np.eye(size, dtype=float)
    unresolved: list[str] = []
    teams = dict(leg_teams or {})
    players = dict(leg_players or {})

    for i, j in combinations(range(size), 2):
        a, b = legs[i], legs[j]
        if not a.game_id or not b.game_id or a.game_id != b.game_id:
            continue
        row_a = {
            "GAME_ID": a.game_id, "player_key": players.get(a.leg_id, a.player_name),
            "TEAM_ABBREVIATION": teams.get(a.leg_id),
        }
        row_b = {
            "GAME_ID": b.game_id, "player_key": players.get(b.leg_id, b.player_name),
            "TEAM_ABBREVIATION": teams.get(b.leg_id),
        }
        relationship = classify_relationship(row_a, row_b)
        bucket = priors.get(relationship, str(a.market), str(b.market))
        if bucket is None or not bucket.usable or bucket.rho is None:
            unresolved.append(
                f"{a.leg_id}x{b.leg_id} ({relationship} {a.market}/{b.market}): "
                + (bucket.reason if bucket and bucket.reason else "no fitted bucket")
            )
            continue
        # rho * s_a * s_b -- see _side_sign. Same sign for over/over and for
        # under/under, flipped for a mixed pair.
        #
        # An ABSENT side is read as "over", which is the side
        # realised_leg_outcomes fits by default, so a caller that never
        # populated the field gets exactly the behaviour it had before. It is
        # logged rather than silent: a leg that MEANT under and omitted the
        # field would be signed wrongly, and the only cure for that is to say
        # so. Refusing instead would convert every side-agnostic over/over
        # ticket into an abstention, which fixes nothing and breaks callers.
        sign_a, sign_b = _side_sign(a), _side_sign(b)
        for leg, sign in ((a, sign_a), (b, sign_b)):
            if sign is None:
                logger.info(
                    "leg %s declares no side; reading it as OVER to match the "
                    "side realised_leg_outcomes fits by default. Populate "
                    "ParlayLeg.side to price an under leg correctly.",
                    leg.leg_id,
                )
        matrix[i, j] = matrix[j, i] = (
            float(bucket.rho) * (sign_a or 1) * (sign_b or 1)
        )

    return matrix, unresolved
