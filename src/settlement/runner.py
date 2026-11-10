"""
src/settlement/runner.py — settle PENDING props against real box scores.

Flow:
    1. Query prop_results WHERE outcome_status = 'PENDING'
    2. Group by nba_game_id (one CDN fetch per game, not per prop)
    3. Fetch the FINAL box score; skip the whole game if not final
    4. Grade each prop via evaluator.settle_prop
    5. Write outcome, actual_result, profit, CLV, raw payload back

Failure policy: a game that can't be fetched leaves its props PENDING
and logs why. Nothing is ever graded from partial or absent data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import requests
from sqlalchemy import select

from src.db.models import PropResult
from src.db.session import session_scope
from src.settlement.boxscore_fetcher import (
    BoxScoreError,
    GameNotFinalError,
    fetch_player_stats_for_game,
)
from src.settlement.closing_lines import capture_for_dates, closing_for_prop
from src.settlement.evaluator import (
    Outcome,
    SettlementError,
    compute_clv,
    settle_prop,
)
from src.settlement.names import normalise_player_name
from src.utils.timezones import pacific_calendar_date

logger = logging.getLogger(__name__)


@dataclass
class SettlementReport:
    games_attempted: int = 0
    games_settled: int = 0
    games_not_final: int = 0
    games_failed: int = 0
    props_graded: int = 0
    wins: int = 0
    losses: int = 0
    pushes: int = 0
    voids: int = 0
    props_unmatched: int = 0
    errors: list[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []

    def as_dict(self) -> dict[str, Any]:
        return {
            "games_attempted": self.games_attempted,
            "games_settled": self.games_settled,
            "games_not_final": self.games_not_final,
            "games_failed": self.games_failed,
            "props_graded": self.props_graded,
            "record": f"{self.wins}-{self.losses}-{self.pushes}",
            "voids": self.voids,
            "props_unmatched": self.props_unmatched,
            "errors": self.errors[:10],
        }


#: Why `_match_player` declined. THE CALLER USED TO GUESS THIS AND GUESSED
#: WRONG: it logged "Likely a name-format mismatch; route through
#: id_crosswalk" for every failure, unconditionally, including for players
#: `espn_boxscore` had ALREADY identified in the line above as scratches who
#: did not travel. Two readers of one fact, and the less-informed one reached
#: the log -- acting on it would have added crosswalk aliases for names that
#: are spelled perfectly correctly. The matcher knows which case it is, so it
#: says.
MATCH_AMBIGUOUS = "ambiguous_name"
MATCH_ABSENT = "absent_from_box_score"


def _match_player(
    stats_by_name: dict[str, dict], player_name: str
) -> tuple[dict | None, str | None]:
    """
    Exact match, then a single normalised retry (casefold + strip
    punctuation). Deliberately NOT fuzzy: mis-matching one player onto
    another's prop would produce a confidently wrong settlement. Genuine
    name variance should be routed through PropIQ's id_crosswalk.

    Returns ``(stats, None)`` on a match, or ``(None, reason)`` where reason is
    ``MATCH_AMBIGUOUS`` or ``MATCH_ABSENT`` -- see the note above those.
    """
    if player_name in stats_by_name:
        candidate = stats_by_name[player_name]
        if candidate.get("ambiguous_name"):
            logger.warning(
                "Refusing to settle %r: two players in this game share that name. "
                "Grading either would be a coin flip on whose line it was.",
                player_name,
            )
            return None, MATCH_AMBIGUOUS
        return candidate, None

    target = normalise_player_name(player_name)
    norm = normalise_player_name
    for name, stats in stats_by_name.items():
        if norm(name) == target:
            if stats.get("ambiguous_name"):
                logger.warning(
                    "Refusing to settle %r: normalises to a name shared by two "
                    "players in this game.",
                    player_name,
                )
                return None, MATCH_AMBIGUOUS
            return stats, None
    return None, MATCH_ABSENT


#: What `extract_player_stats` has always implied without saying: a row with
#: no explicit source came from the NBA CDN, because until the ESPN fallback
#: existed that was the only producer.
CDN_RESULT_SOURCE = "nba_cdn_liveData"
ESPN_RESULT_SOURCE = "espn_site_api_summary"


def _result_source(player_stats: dict) -> str:
    """The source that actually produced this stat line."""
    return ESPN_RESULT_SOURCE if player_stats.get("source") == "espn" else CDN_RESULT_SOURCE


# `stake_or_default` USED TO LIVE HERE AND IT WAS THE DEFECT. It turned the
# recorder's deliberate NULL stake into one unit at grading time, so every
# graded prediction booked a notional 1-unit bet and the results card reported
# "ROI -3.83% on 288.00 unit(s) staked" for props that were never recommended
# -- the board was WITHHELD that night -- never staked and never shown to
# anyone as a pick. The NULL is now passed through and `settle_prop` abstains
# from ROI on it, which is what `recorder.py`'s docstring and
# `notify.discord.build_win_loss_embed`'s both already claimed happened.
#
# Do not reintroduce a default here. A caller that genuinely stakes a row
# passes `stake_units` explicitly; acquiring a stake by omission is how a
# research ledger becomes a betting record nobody decided to keep.


def settle_pending_props(
    max_game_age_days: int = 14,
    session_http: requests.Session | None = None,
    dry_run: bool = False,
) -> SettlementReport:
    """
    Settle every PENDING prop whose game has finished.

    `max_game_age_days` bounds the lookback so a permanently-stuck row
    (e.g. a game id that never posts) doesn't get retried forever.
    """
    report = SettlementReport()
    # Pacific calendar day, not the host's local date: an NBA slate that
    # runs past midnight UTC is still the same Pacific game day, and a
    # UTC-dated cutoff would drop the most recent night's props.
    cutoff = pacific_calendar_date() - timedelta(days=max_game_age_days)
    http = session_http or requests.Session()
    # ONE SLATE FETCH PER RUN, not per game. Without this the ESPN fallback
    # re-fetched the whole slate for every game: a twelve-game CDN outage made
    # about 156 requests instead of thirteen, against a host whose 429 would
    # then strand the grading the fallback exists to rescue.
    espn_cache: dict[str, Any] = {}

    with session_scope() as db:
        pending = db.execute(
            select(PropResult)
            .where(PropResult.outcome_status == "PENDING")
            .where(PropResult.game_date >= cutoff)
            .order_by(PropResult.game_date)
        ).scalars().all()

        if not pending:
            logger.info("No PENDING props within the last %d days.", max_game_age_days)
            return report

        by_game: dict[str, list[PropResult]] = {}
        for row in pending:
            by_game.setdefault(row.nba_game_id, []).append(row)

        # CLOSING LINES, ONCE PER RUN. CLV is the only validator that needs no
        # history: it compares the line taken against THAT SAME GAME's close,
        # so it reads on the first game of a season while win/loss needs
        # thousands of settled props. The machinery was complete and inert --
        # `compute_clv`, the columns and the results card all existed and
        # nothing populated `closing_line`.
        #
        # A FAILURE HERE MUST NOT COST THE GRADING RUN. A closing line is a
        # market-quality measure, not a settlement input: a prop with no quote
        # still grades to WIN/LOSS/PUSH and simply carries no CLV.
        closing_lookup: dict = {}
        try:
            closing_lookup = capture_for_dates(
                {row.game_date for row in pending if row.game_date},
                session=http,
            )
        except Exception as exc:  # noqa: BLE001 — see above
            logger.warning(
                "Closing-line capture failed (%s); every prop grades without "
                "CLV. Grading itself is unaffected.", exc,
            )

        logger.info("Settling %d pending props across %d games", len(pending), len(by_game))

        for game_id, props in by_game.items():
            report.games_attempted += 1
            try:
                # THE FALLBACK'S TWO INPUTS, both already in hand. ESPN's
                # payload carries no NBA game id, so `espn_boxscore` identifies
                # the event from the players being graded; without these it
                # declines rather than guess which game it is reading. The CDN
                # path ignores both.
                stats_by_name, raw_payload = fetch_player_stats_for_game(
                    game_id,
                    session=http,
                    game_date=props[0].game_date,
                    expected_players={p.player_name for p in props if p.player_name},
                    espn_cache=espn_cache,
                )
            except GameNotFinalError as exc:
                report.games_not_final += 1
                logger.info("Game %s not final — leaving %d props PENDING. %s",
                            game_id, len(props), exc)
                continue
            except BoxScoreError as exc:
                report.games_failed += 1
                report.errors.append(f"{game_id}: {exc}")
                logger.warning("Game %s fetch failed — props stay PENDING. %s", game_id, exc)
                continue

            for prop in props:
                player_stats, miss = _match_player(stats_by_name, prop.player_name)
                if player_stats is None and miss == MATCH_AMBIGUOUS:
                    # STAYS PENDING, and this is the one case where that is
                    # right: the player IS in the box score, twice, and a
                    # crosswalk entry really would resolve it. Voiding would
                    # discard a row that is recoverable.
                    report.props_unmatched += 1
                    logger.warning(
                        "Two players in game %s share the name %r — left "
                        "PENDING. Resolvable: add the id_crosswalk entry that "
                        "says which one this prop belongs to.",
                        game_id, prop.player_name,
                    )
                    continue
                if player_stats is None:
                    # VOID, NOT PENDING. The game is final and its box score
                    # was fetched whole, so a player with no line in it
                    # produced no stat and this prop can never grade to
                    # WIN/LOSS/PUSH -- PENDING was a state it would sit in
                    # until the 14-day age cutoff dropped it silently.
                    #
                    # AND THE CAUSE IS NOT ASSERTED, because from here it
                    # cannot be told. A scratch who did not dress and a
                    # spelling this matcher cannot resolve are the SAME
                    # observation: absent from a complete box score. The old
                    # message picked one ("Likely a name-format mismatch") and
                    # pointed at the wrong fix for the common case.
                    # `espn_boxscore` logs the players it could not find on the
                    # slate, which is the evidence that separates them.
                    report.props_unmatched += 1
                    logger.warning(
                        "No box-score line for %r in final game %s — VOIDED, "
                        "excluded from W-L and ROI. Either a scratch who did "
                        "not dress (see the box-score fetcher's own list for "
                        "this slate) or a spelling this matcher cannot resolve; "
                        "from here those are the same observation. If the "
                        "player did play, the fix is an id_crosswalk entry.",
                        prop.player_name, game_id,
                    )
                    void = settle_prop(
                        market=prop.market,
                        predicted_line=prop.predicted_line,
                        predicted_side=prop.predicted_side,
                        player_stats=None,
                    )
                    if not dry_run:
                        prop.outcome_status = str(void.outcome)
                        prop.actual_result = None
                        prop.stake_units = None
                        prop.profit_units = None
                        prop.settlement_note = (
                            "No box-score line for this player in a final game, "
                            "so the prop can never grade to WIN/LOSS/PUSH. "
                            "VOIDED and excluded from W-L and ROI. The cause is "
                            "a scratch who did not dress OR a name this matcher "
                            "could not resolve; from the settlement runner those "
                            "are the same observation and neither is asserted "
                            "here."
                        )
                        prop.settled_at_utc = datetime.now(timezone.utc)
                        # DELIBERATELY NOT SET: `did_not_play`, `minutes_played`,
                        # `result_source` and `raw_boxscore_json`. Writing
                        # did_not_play=True would assert the scratch reading over
                        # the name-mismatch one, which is the exact
                        # over-confidence this change removes; and there is no
                        # box-score row to record provenance for or to store.
                    report.props_graded += 1
                    report.voids += 1
                    continue

                try:
                    result = settle_prop(
                        market=prop.market,
                        predicted_line=prop.predicted_line,
                        predicted_side=prop.predicted_side,
                        player_stats=player_stats,
                        odds=prop.odds,
                        did_not_play=player_stats.get("did_not_play", False),
                        minutes_played=player_stats.get("minutes_played"),
                        # PASSED THROUGH, NEVER DEFAULTED. See the note where
                        # `stake_or_default` used to be.
                        stake_units=prop.stake_units,
                        # THE OPERATOR, FOR FPTS ONLY. A fantasy score is
                        # scored on the book's own multipliers, so the row's
                        # source IS the scoring matrix; every other market
                        # ignores this. An operator with no configured matrix
                        # raises and the prop stays PENDING with a named
                        # reason, rather than being graded on whichever
                        # operator's terms happened to be the default.
                        platform=prop.source,
                    )
                except (SettlementError, ValueError, TypeError, AttributeError) as exc:
                    report.errors.append(f"{game_id}/{prop.player_name}/{prop.market}: {exc}")
                    logger.warning("Could not grade %s %s: %s", prop.player_name, prop.market, exc)
                    continue

                # FILL THE CLOSE BEFORE MEASURING AGAINST IT. These two
                # columns existed and nothing ever wrote them, so `compute_clv`
                # returned {None, None} on every row ever graded.
                #
                # The LINE is the main close; the ODDS are the price of the SAME
                # POINT that was taken. `compute_clv` keeps the line move and
                # the price move separate because "reporting one as the other is
                # the most common CLV error", and handing it the main line's
                # price would let a line move leak into clv_prob_points and read
                # as a price edge.
                #
                # RESOLVED INTO LOCALS, NOT ONTO THE ROW. `dry_run` must write
                # nothing, and these two assignments sat ABOVE the only
                # `if not dry_run` guard: `session_scope()` commits on a clean
                # exit and does not care that the caller asked for a rehearsal,
                # so a dry run persisted every close it captured.
                closing_line = prop.closing_line
                closing_odds = prop.closing_odds
                # EITHER ONE MISSING IS A REASON TO LOOK. The two numbers come
                # from different rungs -- the main line, and the price of the
                # point actually taken -- so a row can carry a close and still
                # have no same-point price. Gating on the line alone left those
                # rows permanently without one, and the price measure is the
                # half that abstains without it.
                if closing_line is None or closing_odds is None:
                    main_line, same_point_odds = closing_for_prop(
                        closing_lookup,
                        source=prop.source,
                        market=prop.market,
                        player_name=prop.player_name,
                        taken_line=prop.predicted_line,
                        taken_side=prop.predicted_side,
                    )
                    # Only ever ADDS a close: whichever field already holds a
                    # number keeps it, so a re-run cannot overwrite what was
                    # true at the time with a later capture.
                    if closing_line is None:
                        closing_line = main_line
                    if closing_odds is None:
                        closing_odds = same_point_odds

                # CLV is a market-quality signal, not part of grading. A bad
                # closing price must not abort the surrounding transaction and
                # leave every other prop in this batch stuck PENDING.
                try:
                    clv = compute_clv(
                        predicted_line=prop.predicted_line,
                        predicted_side=prop.predicted_side,
                        closing_line=closing_line,
                        bet_odds=prop.odds,
                        closing_odds=closing_odds,
                    )
                except SettlementError as exc:
                    report.errors.append(
                        f"{game_id}/{prop.player_name}/{prop.market}: CLV unavailable ({exc})"
                    )
                    logger.warning(
                        "CLV skipped for %s %s (%s) — the prop is still graded",
                        prop.player_name, prop.market, exc,
                    )
                    clv = {"clv_line_points": None, "clv_prob_points": None}

                if not dry_run:
                    prop.closing_line = closing_line
                    prop.closing_odds = closing_odds
                    prop.outcome_status = str(result.outcome)
                    prop.actual_result = result.actual_result
                    prop.stake_units = result.stake_units
                    prop.profit_units = result.profit_units
                    prop.minutes_played = player_stats.get("minutes_played")
                    prop.did_not_play = player_stats.get("did_not_play", False)
                    prop.clv_line_points = clv["clv_line_points"]
                    prop.clv_prob_points = clv["clv_prob_points"]
                    prop.settlement_note = result.note
                    # PROVENANCE IS NOT DECORATION. This was hardcoded, so
                    # every ESPN-graded row claimed the NBA CDN as its source.
                    # The settlement audit's whole value is being able to ask
                    # "where did this number come from" and get a true answer;
                    # a row that lies about its source cannot be re-checked
                    # against the feed it names. The fetched row says which
                    # source produced it, and the CDN value stands otherwise.
                    prop.result_source = _result_source(player_stats)
                    prop.settled_at_utc = datetime.now(timezone.utc)
                    prop.raw_boxscore_json = {
                        "gameId": game_id,
                        "player": player_stats,
                    }

                report.props_graded += 1
                if result.outcome is Outcome.WIN:
                    report.wins += 1
                elif result.outcome is Outcome.LOSS:
                    report.losses += 1
                elif result.outcome is Outcome.PUSH:
                    report.pushes += 1
                elif result.outcome is Outcome.VOID:
                    report.voids += 1

            report.games_settled += 1

    logger.info("Settlement complete: %s", report.as_dict())
    return report
