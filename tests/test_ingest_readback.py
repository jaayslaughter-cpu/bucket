"""
The read-back that is the only evidence an ingest wrote anything.

WHY THIS FILE EXISTS. Every `upsert_*` in `src/db/repository.py` returns
``len(rows)`` -- the rows handed to the statement, not the rows the database
kept -- and for five days (2026-10-23 to 2026-10-28) that number was printed
as "Written to Postgres. 2,644 rows" while `team_game_stats` held nothing.
`ingested_row_counts` and `ingested_keys_present` are what replaced the claim,
so what they actually count is worth asserting rather than assuming.

AGAINST A REAL DATABASE, in-memory SQLite, following
`tests/test_artifact_freshness.py`: the assertions are about which rows a
SELECT returns and how the result is intersected, neither of which is
dialect-specific, and a test that needs Postgres to run is a test that does
not run.

RESEARCH_ONLY.
"""

from __future__ import annotations

import contextlib
from datetime import date

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import src.db.repository as repository
from src.db.models import Base, GameMarketLine, PlayerGameLog, TeamGameStat


def _bind(monkeypatch, team_keys=(), market_keys=()):
    """`repository.session_scope` bound to a database holding these keys."""
    from sqlalchemy.dialects.postgresql import JSONB
    from sqlalchemy.ext.compiler import compiles

    # SQLite's DDL compiler cannot render JSONB. Registered HERE rather than
    # at module scope, following `tests/test_artifact_freshness.py`: at module
    # scope, importing this file teaches SQLAlchemy to compile a Postgres type
    # for SQLite in every other test in the session, and a test that changes
    # global behaviour by being imported is how a suite starts passing for the
    # wrong reason. Flagged in review, 2026-10-29.
    @compiles(JSONB, "sqlite")
    def _jsonb_as_json(type_, compiler, **kw):  # noqa: ARG001
        return "JSON"

    engine = create_engine("sqlite://")
    Base.metadata.create_all(
        bind=engine,
        tables=[TeamGameStat.__table__, GameMarketLine.__table__,
                PlayerGameLog.__table__],
    )
    factory = sessionmaker(bind=engine)
    day = date(2026, 10, 29)
    with factory() as seed:
        for game, team in team_keys:
            seed.add(TeamGameStat(
                nba_game_id=game, game_date=day, team_abbr=team,
                opponent_abbr="XXX", is_home=True, points=100.0,
                source="bigdataball",
            ))
        for game, team in market_keys:
            seed.add(GameMarketLine(
                nba_game_id=game, game_date=day, team_abbr=team,
                source="bigdataball", status="VALID",
            ))
        seed.commit()

    @contextlib.contextmanager
    def _scope():
        with factory() as session:
            yield session

    monkeypatch.setattr(repository, "session_scope", _scope)


def test_the_row_counts_are_the_tables_own(monkeypatch):
    _bind(monkeypatch,
          team_keys=[("0022600001", "LAL"), ("0022600001", "GSW")],
          market_keys=[("0022600001", "LAL")])
    counts = repository.ingested_row_counts()
    assert counts == {
        "team_game_stats": 2, "game_market_lines": 1, "player_game_logs": 0,
    }


def test_a_key_the_table_does_not_hold_is_not_counted(monkeypatch):
    """
    THE DIFFERENCE BETWEEN THE TWO FUNCTIONS, AND THE WHOLE POINT OF THE
    SECOND. The table has two rows, so `ingested_row_counts` says
    "team_game_stats: 2" and an absolute check passes -- while NEITHER of the
    keys this run parsed is in there. After the first successful ingest that
    is the ordinary shape of the failure.
    """
    _bind(monkeypatch, team_keys=[("0022600001", "LAL"), ("0022600001", "GSW")])
    assert repository.ingested_row_counts()["team_game_stats"] == 2

    present = repository.ingested_keys_present(
        team_games=[("0022600099", "BOS"), ("0022600099", "NYK")],
    )
    assert present == {"team_game_stats": 0}


def test_a_partial_landing_is_counted_partially(monkeypatch):
    _bind(monkeypatch, team_keys=[("0022600001", "LAL")])
    present = repository.ingested_keys_present(
        team_games=[("0022600001", "LAL"), ("0022600001", "GSW")],
    )
    assert present == {"team_game_stats": 1}


def test_a_repeated_key_is_counted_once(monkeypatch):
    """A frame may carry a key twice -- `game_market_lines` is unique on
    (game, team, SOURCE) -- and 3 of 2 present would be a verification that
    cannot be read."""
    _bind(monkeypatch, market_keys=[("0022600001", "LAL")])
    present = repository.ingested_keys_present(
        market_lines=[("0022600001", "LAL"), ("0022600001", "LAL")],
    )
    assert present == {"game_market_lines": 1}


def test_a_table_with_no_keys_asked_for_is_not_reported(monkeypatch):
    """
    NOT ZERO -- ABSENT. A caller that parsed no market lines must not be told
    "game_market_lines: 0 present", which reads as a failure of a write that
    was never attempted.
    """
    _bind(monkeypatch, team_keys=[("0022600001", "LAL")])
    present = repository.ingested_keys_present(
        team_games=[("0022600001", "LAL")],
    )
    assert present == {"team_game_stats": 1}
    assert "game_market_lines" not in present
    assert repository.ingested_keys_present() == {}


def test_a_key_asked_for_as_a_number_still_matches(monkeypatch):
    """
    THE SIDE THAT CAN ACTUALLY ARRIVE WRONG. A parsed frame hands over
    whatever pandas inferred, and an all-numeric `nba_game_id` column infers
    as int64 -- so `1` would be asked for where `"1"` is stored, and a
    healthy write would be reported as a lost one. That is the false alarm
    this verification would be retired for.

    ONLY THIS SIDE IS ASSERTED. The stored side is coerced too, but SQLite
    gives a VARCHAR column TEXT affinity and converts an inserted int on the
    way in, so there is no way to make the DATABASE hand back a non-string
    here and a test of it would pass whether the coercion were there or not.
    Rule 2: an assertion that cannot fail is not a test, so it is not
    claimed.
    """
    _bind(monkeypatch, team_keys=[("1", "LAL")])
    present = repository.ingested_keys_present(team_games=[(1, "LAL")])
    assert present == {"team_game_stats": 1}


@pytest.mark.parametrize("table", ["team_game_stats", "game_market_lines"])
def test_both_ingest_filled_tables_can_be_verified(monkeypatch, table):
    """The market half used to be unverifiable: nothing read it back, and
    nothing required it. It is the table the EV path abstains without."""
    _bind(monkeypatch,
          team_keys=[("0022600001", "LAL")],
          market_keys=[("0022600001", "LAL")])
    present = repository.ingested_keys_present(
        team_games=[("0022600001", "LAL")],
        market_lines=[("0022600001", "LAL")],
    )
    assert present[table] == 1


# --- the second id namespace -------------------------------------------


def _bind_players(monkeypatch, player_ids):
    """`repository.session_scope` bound to a panel holding these player ids."""
    from datetime import date as _date

    from sqlalchemy.dialects.postgresql import JSONB
    from sqlalchemy.ext.compiler import compiles

    @compiles(JSONB, "sqlite")
    def _jsonb_as_json(type_, compiler, **kw):  # noqa: ARG001
        return "JSON"

    engine = create_engine("sqlite://")
    Base.metadata.create_all(bind=engine, tables=[PlayerGameLog.__table__])
    factory = sessionmaker(bind=engine)
    with factory() as seed:
        for i, pid in enumerate(player_ids):
            seed.add(PlayerGameLog(
                nba_game_id=f"g{i}", nba_player_id=str(pid),
                player_name=f"P{i}", game_date=_date(2026, 10, 29),
                season="2026-27", source="test",
            ))
        seed.commit()

    @contextlib.contextmanager
    def _scope():
        with factory() as session:
            yield session

    monkeypatch.setattr(repository, "session_scope", _scope)


def test_a_single_namespace_panel_reports_one_namespace(monkeypatch):
    _bind_players(monkeypatch, ["espn:1", "espn:2", "espn:3"])
    assert repository.player_id_namespaces() == {"espn": 3}


def test_a_mixed_panel_is_visible_rather_than_silent(monkeypatch):
    """
    THE FAILURE THIS EXISTS FOR. `builder.py:518` groups rolling features by
    PLAYER_ID, so the same player under two namespaces is two players with
    half a history each -- a confident PTS_L10 over the wrong games, with no
    error anywhere. A mixed panel has to be VISIBLE before it is written to,
    because afterwards nothing downstream can tell.
    """
    _bind_players(monkeypatch, ["espn:1", "espn:2", "203507", "1629029"])
    counts = repository.player_id_namespaces()
    assert counts == {"espn": 2, "nba": 2}
    assert len(counts) > 1, "mixing must be detectable from this alone"


def test_an_empty_panel_has_no_namespaces(monkeypatch):
    _bind_players(monkeypatch, [])
    assert repository.player_id_namespaces() == {}


def test_an_espn_team_game_is_visible_to_the_newest_game_query(monkeypatch):
    """
    WITHOUT THE NAMESPACE BRANCH THESE ROWS ARE NOT MISCLASSIFIED, THEY ARE
    INVISIBLE. An `espn:` id fails both the length test and the season-digit
    test, so the newest-game query returned None over a table full of them --
    the slate would report "nothing ingested" against data it had just
    written, which is the exact five-day failure in reverse.
    """
    from datetime import date as _date

    _bind(monkeypatch, team_keys=[("espn:401909881", "LAL")])
    with repository.session_scope() as s:
        row = s.query(TeamGameStat).first()
        row.game_date = _date(2026, 10, 28)
        s.commit()
    assert repository.newest_team_game_date() is not None, (
        "an espn-namespaced team game is invisible to the regular-season filter"
    )


def test_an_nba_id_with_a_playoff_digit_is_still_excluded(monkeypatch):
    """The new branch widens the filter for ONE namespace, not for everything:
    a playoff NBA id must still be filtered out."""
    _bind(monkeypatch, team_keys=[("0042600123", "LAL")])
    assert repository.newest_team_game_date() is None
