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
from decimal import Decimal
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


def _match_player(stats_by_name: dict[str, dict], player_name: str) -> dict | None:
    """
    Exact match, then a single normalised retry (casefold + strip
    punctuation). Deliberately NOT fuzzy: mis-matching one player onto
    another's prop would produce a confidently wrong settlement. Genuine
    name variance should be routed through PropIQ's id_crosswalk.
    """
    if player_name in stats_by_name:
        candidate = stats_by_name[player_name]
        if candidate.get("ambiguous_name"):
            logger.warning(
                "Refusing to settle %r: two players in this game share that name. "
                "Grading either would be a coin flip on whose line it was.",
                player_name,
            )
            return None
        return candidate

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
                return None
            return stats
    return None


#: What `extract_player_stats` has always implied without saying: a row with
#: no explicit source came from the NBA CDN, because until the ESPN fallback
#: existed that was the only producer.
CDN_RESULT_SOURCE = "nba_cdn_liveData"
ESPN_RESULT_SOURCE = "espn_site_api_summary"


def _result_source(player_stats: dict) -> str:
    """The source that actually produced this stat line."""
    return ESPN_RESULT_SOURCE if player_stats.get("source") == "espn" else CDN_RESULT_SOURCE


def stake_or_default(stake: Decimal | None) -> Decimal:
    """Default a NULL stake to one unit — and only a NULL one.

    Written as an explicit None check because ``stake or Decimal(1)`` treats
    Decimal("0") as falsey: a row deliberately staked at zero would be
    re-staked at one unit and booked into profit_units as a real bet.
    """
    return Decimal(1) if stake is None else stake


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
                player_stats = _match_player(stats_by_name, prop.player_name)
                if player_stats is None:
                    report.props_unmatched += 1
                    logger.warning(
                        "No box-score match for %r in game %s — left PENDING. "
                        "Likely a name-format mismatch; route through id_crosswalk.",
                        prop.player_name, game_id,
                    )
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
                        stake_units=stake_or_default(prop.stake_units),
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
