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


# --- the history backfill's resume point ----------------------------------


def _bind_panel(monkeypatch, rows):
    """
    `session_scope` bound to a database holding these panel rows.

    Each row is ``(date, player, source)`` or ``(date, player, source,
    game_id)``. THE GAME ID IS EXPLICIT where it matters: the three-tuple
    form derives a unique one per row, which made the "same night, two
    players" case in `test_the_backfill_resume_point_is_read_from_the
    _database` two separate GAMES -- so nothing asserted that two players
    from ONE game collapse to one id. Raised by review on #31.
    """
    from sqlalchemy.dialects.postgresql import JSONB
    from sqlalchemy.ext.compiler import compiles

    @compiles(JSONB, "sqlite")
    def _jsonb_as_json(type_, compiler, **kw):  # noqa: ARG001
        return "JSON"

    engine = create_engine("sqlite://")
    Base.metadata.create_all(bind=engine, tables=[PlayerGameLog.__table__])
    factory = sessionmaker(bind=engine)
    with factory() as seed:
        for index, row in enumerate(rows):
            day, player, source = row[0], row[1], row[2]
            game_id = row[3] if len(row) > 3 else f"espn:{40000 + index}"
            seed.add(PlayerGameLog(
                nba_game_id=game_id, nba_player_id=player,
                player_name=f"HARNESS {player}", game_date=day,
                season="2024-25", team_abbr="LAL", opponent_abbr="XXX",
                is_home=True, pts=20, source=source,
            ))
        seed.commit()

    @contextlib.contextmanager
    def _scope():
        with factory() as session:
            yield session

    monkeypatch.setattr(repository, "session_scope", _scope)


def test_the_backfill_resume_point_is_read_from_the_database(monkeypatch):
    """
    The dates ESPN has already supplied, as ISO strings.

    THE RESUME POINT HAS TO COME FROM HERE. A state file beside the database
    is a second source of truth about what is ingested, and a high-water mark
    cannot express a hole in the middle left by a night ESPN was down for.
    """
    _bind_panel(monkeypatch, [
        # One night, TWO GAMES, with two players sharing the first. Both
        # distinctions matter: the two games must stay two ids, and the two
        # players in one game must collapse to one. The first version of
        # this test gave every row its own game id, so it only ever
        # exercised the second half of that.
        (date(2024, 10, 22), "espn:1", "espn", "espn:40000"),
        (date(2024, 10, 22), "espn:2", "espn", "espn:40000"),
        (date(2024, 10, 22), "espn:3", "espn", "espn:40001"),
        (date(2024, 10, 24), "espn:4", "espn", "espn:40002"),
    ])
    assert repository.panel_game_ids_by_date("espn") == {
        "2024-10-22": {"espn:40000", "espn:40001"},
        "2024-10-24": {"espn:40002"},
    }, "the resume point is keyed by GAME, not by row and not by date"


def test_the_resume_point_reads_the_panel_not_the_team_table(monkeypatch):
    """
    THE HOLE THE FIRST VERSION HAD. It read `team_game_stats`, and
    `aggregate_team_games` DECLINES a (date, team) a higher-precedence
    source already owns — BigDataBall outranks `espn_agg`. So a night the
    licensed workbook holds gets no `espn_agg` team row however many times
    ESPN is read for it, and the backfill would re-fetch that night for
    ever while never reporting itself complete.

    `upsert_player_game_logs` has no precedence guard: it writes every row
    it is handed. The panel is also the thing the backfill exists to fill,
    so it is the honest place to ask what is already in. Raised by review
    on #31.
    """
    from sqlalchemy.dialects.postgresql import JSONB
    from sqlalchemy.ext.compiler import compiles

    @compiles(JSONB, "sqlite")
    def _jsonb_as_json(type_, compiler, **kw):  # noqa: ARG001
        return "JSON"

    engine = create_engine("sqlite://")
    Base.metadata.create_all(
        bind=engine, tables=[PlayerGameLog.__table__, TeamGameStat.__table__],
    )
    factory = sessionmaker(bind=engine)
    day = date(2024, 10, 22)
    with factory() as seed:
        # ESPN supplied the panel for this night...
        seed.add(PlayerGameLog(
            nba_game_id="espn:40001", nba_player_id="espn:1",
            player_name="HARNESS ONE", game_date=day, season="2024-25",
            team_abbr="LAL", opponent_abbr="GSW", is_home=True, pts=20,
            source="espn",
        ))
        # ...while the team row for it belongs to the licensed workbook,
        # which outranks espn_agg, so no espn_agg team row exists.
        seed.add(TeamGameStat(
            nba_game_id="0022400001", game_date=day, team_abbr="LAL",
            opponent_abbr="GSW", is_home=True, points=100.0,
            source="bigdataball",
        ))
        seed.commit()

    @contextlib.contextmanager
    def _scope():
        with factory() as session:
            yield session

    monkeypatch.setattr(repository, "session_scope", _scope)

    assert repository.panel_game_ids_by_date("espn") == {
        "2024-10-22": {"espn:40001"},
    }, (
        "the night is in the panel but the team table has no espn_agg row "
        "for it; keyed on the team table the backfill would re-fetch it for "
        "ever"
    )


def test_another_sources_night_is_not_a_night_espn_has_read(monkeypatch):
    """
    KEYED ON SOURCE, and this is the assertion that pins it. A night the
    licensed workbook owns is NOT a night ESPN has supplied: counting it
    would leave the backfill a hole it believes it has filled.
    """
    _bind_panel(monkeypatch, [
        (date(2024, 10, 22), "espn:1", "espn"),
        (date(2024, 10, 23), "2544", "bigdataball"),
        (date(2024, 10, 24), "201939", "scraper"),
    ])
    assert set(repository.panel_game_ids_by_date("espn")) == {"2024-10-22"}
    assert set(repository.panel_game_ids_by_date("bigdataball")) == {"2024-10-23"}


def test_a_source_with_no_rows_reports_no_dates_rather_than_raising(monkeypatch):
    """The first run's state. An exception here would fail the backfill
    before it read a single night."""
    _bind_panel(monkeypatch, [(date(2024, 10, 22), "2544", "bigdataball")])
    assert repository.panel_game_ids_by_date("espn") == {}


def test_a_partially_ingested_night_is_visible_as_the_games_it_holds(
    monkeypatch,
):
    """
    THE DEFECT A DATE-LEVEL MARKER HID, raised by review on #31.

    `fetch_player_rows` records a failed event and keeps the others, so a
    night that lost one game to a 404 is written and INCOMPLETE. Returning
    dates alone made that night indistinguishable from a complete one, so
    the backfill skipped it for ever and the missing game's players carried
    a rolling window short one game, silently.

    Keyed by game id, the gap is visible: the caller subtracts what it holds
    from what ESPN lists for that date and asks for the difference.
    """
    _bind_panel(monkeypatch, [
        (date(2024, 10, 22), "espn:1", "espn", "espn:40000"),
        (date(2024, 10, 22), "espn:2", "espn", "espn:40001"),
    ])
    held = repository.panel_game_ids_by_date("espn")
    assert held == {"2024-10-22": {"espn:40000", "espn:40001"}}
    # ESPN listed three games that night; the third never landed.
    listed = {"espn:40000", "espn:40001", "espn:40002"}
    assert listed - held["2024-10-22"] == {"espn:40002"}, (
        "the missing game must be derivable from what the panel holds"
    )


# --- the projections read side, against a real database -------------------


def _bind_projections(monkeypatch, runs):
    """
    `session_scope` bound to a database holding these projection runs.

    ``runs`` is ``[(run_id, created_at, game_date, [(player, market, line,
    prob_over), ...]), ...]``. Against real SQLite rather than a stub,
    because what is asserted below is which rows a query RETURNS and in
    which order — neither of which a monkeypatched loader can be wrong
    about, and both of which the query can.
    """
    from src.db.models import Projection

    engine = create_engine("sqlite://")
    Base.metadata.create_all(bind=engine, tables=[Projection.__table__])
    factory = sessionmaker(bind=engine)
    with factory() as seed:
        for run_id, created, game_date, rows in runs:
            for player, market, line, prob in rows:
                seed.add(Projection(
                    run_id=run_id, created_at=created, game_date=game_date,
                    nba_player_id=player, player_name=f"HARNESS {player}",
                    nba_game_id="espn:401909903", market=market,
                    final_projection=20.0, baseline_projection=19.0,
                    line=line, prob_over=prob, market_status="DATA_NOT_AVAILABLE",
                ))
        seed.commit()

    @contextlib.contextmanager
    def _scope():
        with factory() as session:
            yield session

    monkeypatch.setattr(repository, "session_scope", _scope)


def test_a_re_run_cannot_double_the_board_because_the_schema_forbids_it(
    monkeypatch,
):
    """
    THE DEFECT MY FIRST VERSION OF THIS LOADER GUARDED AGAINST DID NOT EXIST,
    and the guard introduced one that did.

    `projections` is UNIQUE on (nba_game_id, player_name, market) -- run_id
    is NOT in the key -- and `persist_projections` upserts on that identity
    while refreshing run_id. So one (game, player, market) has exactly one
    row, ever, carrying whichever run last wrote it. A re-run OVERWRITES;
    it cannot double.

    This test asserts that from the database side: inserting the same key
    twice is a conflict, not two rows.
    """
    from datetime import datetime, timezone

    import pytest
    from sqlalchemy.exc import IntegrityError

    day = date(2026, 11, 1)
    with pytest.raises(IntegrityError):
        _bind_projections(monkeypatch, [
            ("run_morning", datetime(2026, 11, 1, 9, 0, tzinfo=timezone.utc), day,
             [("espn:1", "PTS", 21.5, None)]),
            ("run_rerun", datetime(2026, 11, 1, 11, 0, tzinfo=timezone.utc), day,
             [("espn:1", "PTS", 22.5, 0.56)]),
        ])


def test_a_partial_re_run_keeps_every_row_rather_than_the_newest_runs_only(
    monkeypatch,
):
    """
    THE DEFECT THE RUN FILTER WOULD HAVE CAUSED. A re-run that covers only
    SOME rows -- fewer markets after a config change, fewer players after
    some games tipped -- leaves the untouched rows carrying the PREVIOUS
    run_id, because the upsert only refreshes the rows it writes.

    Selecting `run_id == newest` would have dropped those rows, and the
    board would have shown fewer than the slate projected with nothing
    saying so. Filtering by DATE returns all of them, which is correct:
    each is still the newest value for its own key.
    """
    from datetime import datetime, timezone

    day = date(2026, 11, 1)
    _bind_projections(monkeypatch, [
        # PTS was re-scored at 11:00; REB and AST still carry the 09:00 run.
        ("run_rerun", datetime(2026, 11, 1, 11, 0, tzinfo=timezone.utc), day,
         [("espn:1", "PTS", 22.5, 0.56)]),
        ("run_morning", datetime(2026, 11, 1, 9, 0, tzinfo=timezone.utc), day,
         [("espn:2", "REB", 8.5, None), ("espn:3", "AST", 4.5, None)]),
    ])

    out = repository.load_projections("2026-11-01")
    assert len(out) == 3, (
        "a row the re-run did not touch was dropped; the board would be "
        f"narrower than the slate with nothing saying so. Got:\n{out}"
    )
    assert set(out["market"]) == {"PTS", "REB", "AST"}
    assert set(out["run_id"]) == {"run_rerun", "run_morning"}, (
        "provenance is preserved, so a mixed-run slate is visible"
    )


def test_projections_are_filtered_to_the_slate_date(monkeypatch):
    """A board for tonight must not rank yesterday's rows. The filter is an
    ISO string parsed to a date -- a string compared against a Date column
    matches nothing in some backends and everything in others."""
    from datetime import datetime, timezone

    _bind_projections(monkeypatch, [
        ("run_yesterday", datetime(2026, 10, 31, 9, 0, tzinfo=timezone.utc),
         date(2026, 10, 31), [("espn:1", "PTS", 20.5, None)]),
        ("run_today", datetime(2026, 11, 1, 9, 0, tzinfo=timezone.utc),
         date(2026, 11, 1), [("espn:2", "REB", 8.5, None)]),
    ])

    today = repository.load_projections("2026-11-01")
    assert today["run_id"].tolist() == ["run_today"]
    yesterday = repository.load_projections("2026-10-31")
    assert yesterday["run_id"].tolist() == ["run_yesterday"]


def test_no_run_returns_an_empty_frame_WITH_the_columns(monkeypatch):
    """
    So a caller's `[col]` access fails on a missing COLUMN rather than on a
    missing frame — the contract `load_game_market_lines` already keeps.
    `slate_board.projection_detail_rows` reads nine of these by name and
    would raise a bare KeyError on a shapeless empty frame.
    """
    _bind_projections(monkeypatch, [])

    out = repository.load_projections("2026-11-01")
    assert out.empty
    assert list(out.columns) == list(repository.PROJECTION_BOARD_COLS)
    for col in ("nba_game_id", "market", "line", "prob_over", "final_projection"):
        assert col in out.columns
