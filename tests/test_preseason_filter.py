"""
The live slate must not treat a preseason fixture as an ordinary game.

FOUND IN PRODUCTION, 2026-10-12. The deployed worker read ESPN's scoreboard for
2026-10-12 and armed the full pipeline on five PRESEASON games:

    espn_schedule: 5 game(s) for 2026-10-12 PT (5 pre-tip); unmapped teams: ['LON']
    scheduled {'id': 'prelock:2026-10-12:401914130',
               'name': 'Pre-lock availability re-check — LON @ POR', ...}

Two faults in one line.

1. PRESEASON. Nothing between `espn_schedule.parse_scoreboard` and the slate
   reads ESPN's season type, so a preseason fixture is indistinguishable from a
   regular-season one. The panel the model is fit on is the opposite:
   `scripts/ingest_training_pack.py` filters to ``IS_REGULAR_SEASON``
   unless asked not to, and `src/ingestion/kaggle_nba.py` warns in as many
   words that "preseason
   minutes and rotations do not describe the same competition". Starters sit,
   bench players play thirty minutes, and the features describing both are
   built from games where neither happens.

2. AN OPPONENT THAT IS NOT AN NBA TEAM. ``LON`` has no entry in
   ``ESPN_TO_NBA_TEAM`` and is correctly NOT guessed into a nearest match --
   but the game was armed anyway. Every DEF_*, DvP and MKT_* column for the
   Portland players in it has no counterparty to be computed against.

WHY THIS MATTERED ONLY TODAY. Every row abstains for want of a scoring
artifact, so the five games produced nothing. The moment the volume is seeded
the same slate produces confident-looking numbers instead, and that is the
failure this repository treats as the worst kind: exit 0, a full board, nothing
in the logs.

THE SHAPE OF THE FIX. One reader for "may this fixture be projected" --
``SlateGame.exclusion_reason``, with ``SlateResult.projectable`` as the list
both consumers take -- rather than a preseason test in the pipeline and another
in the pre-lock arming, which is the two-readers defect this repository has
fixed five times.

AN EXCLUDE-LIST, NOT AN INCLUDE-LIST. Only what is positively identified as
preseason is dropped. An include-list of {regular, postseason} would silently
blank the board on any fixture ESPN types differently -- the play-in and the
in-season tournament are the cases nobody here can verify from this
environment, and a dark board on a real game night is worse than a noisy one.

RESEARCH_ONLY. A filter on which fixtures may be projected.
"""

from __future__ import annotations

from datetime import date

import pytest

from src.ingestion.espn_schedule import (
    SEASON_TYPE_PRESEASON,
    SEASON_TYPE_REGULAR,
    SlateGame,
    parse_scoreboard,
)


def _event(
    event_id: str = "401765432",
    iso: str = "2025-03-15T02:30Z",
    home: str = "GSW",
    away: str = "BOS",
    state: str = "pre",
    season: dict | None = None,
) -> dict:
    """The documented shape, with ESPN's per-event ``season`` block optional.

    Same provenance caveat as `tests/test_espn_schedule.py`: derived from the
    documented response schema, not captured live. This environment denies
    outbound CONNECT to site.api.espn.com, which is why the parser reads every
    plausible location for the season type instead of trusting one.
    """
    event = {
        "id": event_id,
        "date": iso,
        "shortName": f"{away} @ {home}",
        "status": {"type": {"state": state, "shortDetail": "7:30 PM PT"}},
        "competitions": [{
            "id": event_id,
            "competitors": [
                {"homeAway": "home", "team": {"id": "9", "abbreviation": home}},
                {"homeAway": "away", "team": {"id": "2", "abbreviation": away}},
            ],
        }],
    }
    if season is not None:
        event["season"] = season
    return event


def _payload(*events: dict, season: dict | None = None) -> dict:
    out: dict = {"leagues": [{"abbreviation": "NBA"}], "events": list(events)}
    if season is not None:
        out["season"] = season
    return out


# --- the production fault --------------------------------------------------


def test_a_preseason_game_is_not_projectable():
    """THE FAULT ITSELF: 2026-10-12's slate, as a test."""
    result = parse_scoreboard(
        _payload(_event(season={"year": 2027, "type": SEASON_TYPE_PRESEASON}))
    )
    assert len(result.games) == 1, "the game is still reported, never dropped"
    game = result.games[0]
    assert game.is_preseason is True
    assert game.exclusion_reason is not None
    assert "preseason" in game.exclusion_reason.lower()
    assert result.projectable == [], (
        "a preseason fixture reached the projectable slate -- the model is fit "
        "on regular-season rows only (scripts/ingest_training_pack.py)"
    )


def test_a_regular_season_game_is_projectable():
    """The other half: the filter must not blank an ordinary night."""
    result = parse_scoreboard(
        _payload(_event(season={"year": 2026, "type": SEASON_TYPE_REGULAR}))
    )
    assert result.games[0].is_preseason is False
    assert result.games[0].exclusion_reason is None
    assert len(result.projectable) == 1


def test_an_unrecognised_opponent_is_not_projectable():
    """``LON @ POR``: no crosswalk entry, so no counterparty for any DEF_*
    or DvP column. Reported, not guessed, and not projected either."""
    result = parse_scoreboard(
        _payload(_event(home="POR", away="LON",
                        season={"year": 2027, "type": SEASON_TYPE_REGULAR}))
    )
    game = result.games[0]
    assert game.away_team == "LON", "the raw code stays visible"
    assert "LON" in result.unmapped_teams
    assert game.exclusion_reason is not None
    assert "LON" in game.exclusion_reason
    assert result.projectable == []


# --- what "unknown" must mean ---------------------------------------------


def test_an_absent_season_type_stays_projectable_and_says_so():
    """
    THE OUTAGE THIS AVOIDS. The season type cannot be verified against the live
    endpoint from here, so a payload without it must behave as it did before
    this filter existed -- loudly. Excluding unknowns would mean one renamed
    ESPN key turns every slate into an empty board that looks like a quiet
    night.
    """
    result = parse_scoreboard(_payload(_event()))
    game = result.games[0]
    assert game.season_type is None
    assert game.is_preseason is None, "unknown is not False"
    assert game.exclusion_reason is None, (
        "an unreported season type blanked the slate -- a renamed key would "
        "take the whole board down silently"
    )
    assert len(result.projectable) == 1
    assert any("season type" in n.lower() for n in result.notes), (
        "nothing in the result says the season type was never read"
    )


def test_the_payload_level_season_is_used_when_the_event_has_none():
    """ESPN's scoreboard carries a season block for the request as a whole;
    for a single-date query that is the season type of its events."""
    result = parse_scoreboard(
        _payload(_event(), season={"year": 2027, "type": SEASON_TYPE_PRESEASON})
    )
    game = result.games[0]
    assert game.is_preseason is True
    assert game.season_type_source == "payload.season", (
        f"the value was read from {game.season_type_source!r}, so a reader "
        "cannot tell whether it described this event or the request"
    )
    assert result.projectable == []


def test_a_named_preseason_is_caught_when_the_number_is_not_what_we_think():
    """
    The numbering is the one thing no test here can verify, so the name is read
    too: either signal saying preseason is enough.
    """
    result = parse_scoreboard(
        _payload(_event(season={"year": 2027, "type": 7, "slug": "preseason"}))
    )
    assert result.games[0].is_preseason is True
    assert result.projectable == []


def test_the_postseason_is_projectable_and_the_mismatch_is_recorded():
    """
    Deliberate, and narrower than the preseason case. The training pack filters
    to IS_REGULAR_SEASON too (`scripts/ingest_training_pack.py`), so a playoff
    is also off-distribution -- but excluding it would blank every April board,
    which is a bigger decision than this fix and not the one that was asked
    for. It is projected, and the note says why that is a known compromise.
    """
    result = parse_scoreboard(
        _payload(_event(season={"year": 2026, "type": 3, "slug": "post-season"}))
    )
    assert result.games[0].exclusion_reason is None
    assert len(result.projectable) == 1
    assert any("postseason" in n.lower() for n in result.notes)


# --- a game in progress is still not projectable --------------------------


def test_projectable_is_still_pre_tip_only():
    """``projectable`` must keep what ``pregame_only`` guaranteed: a game under
    way or final has a box score, which is the answer, not a projection."""
    result = parse_scoreboard(_payload(
        _event(event_id="1", state="pre", season={"type": SEASON_TYPE_REGULAR}),
        _event(event_id="2", state="post", season={"type": SEASON_TYPE_REGULAR}),
        _event(event_id="3", state="in", season={"type": SEASON_TYPE_REGULAR}),
    ))
    assert [g.espn_event_id for g in result.projectable] == ["1"]


# --- the consumers ---------------------------------------------------------


def _slate(*games: SlateGame):
    """A real ``SlateResult``, because the point is what the consumers do with
    the actual object ``load_slate`` hands them."""
    from src.ingestion.espn_schedule import SlateResult

    return SlateResult(status="OK", slate_date_pt=date(2026, 10, 12),
                       games=list(games))


def _game(event_id: str, *, season_type: int, tipoff, home="BOS", away="LAL"):
    return SlateGame(
        espn_event_id=event_id, tipoff_utc=tipoff,
        slate_date_pt=date(2026, 10, 12), home_team=home, away_team=away,
        state="pre", status_detail=None, venue=None, venue_city=None,
        is_neutral_site=None, season_type=season_type,
        season_type_source="event.season",
    )


def test_the_pre_lock_arming_skips_a_preseason_game(monkeypatch):
    """
    THE PRODUCTION LINE ITSELF, as a test:

        scheduled {'id': 'prelock:2026-10-12:401914130',
                   'name': 'Pre-lock availability re-check — LON @ POR', ...}

    Behavioural, not a grep: the earlier version of this test searched
    ``schedule_prelock_jobs``'s source for the word "projectable", which the
    explanatory comment satisfied on its own — reverting the code to
    ``pregame_only`` left it green. A weak test is worse than none, because it
    reports coverage it does not have.
    """
    from datetime import datetime, timedelta, timezone

    import scheduler_worker as worker
    from tests.test_scheduler_worker import FakeScheduler

    monkeypatch.delenv("PROPIQ_PRELOCK", raising=False)
    monkeypatch.delenv("PROPIQ_PRELOCK_LEAD_MINUTES", raising=False)
    now = datetime(2026, 10, 12, 18, 0, tzinfo=timezone.utc)
    sched = FakeScheduler()
    out = worker.schedule_prelock_jobs(sched, now=now, slate=_slate(
        _game("401914130", season_type=SEASON_TYPE_PRESEASON,
              tipoff=now + timedelta(hours=2), home="POR", away="GSW"),
        _game("401898402", season_type=SEASON_TYPE_REGULAR,
              tipoff=now + timedelta(hours=4), home="ATL", away="OKC"),
    ))

    assert out["armed"] == 1, (
        "a tip-off job was armed for a preseason fixture — the 2026-10-12 "
        "production behaviour"
    )
    assert len(sched.jobs) == 1
    assert "401898402" in str(sched.jobs[0])


def test_the_forward_slate_builds_no_rows_for_a_preseason_game(monkeypatch):
    """The other consumer, also behavioural: no panel rows, so nothing to
    score, so nothing to publish."""
    from datetime import datetime, timedelta, timezone

    import src.ingestion.espn_schedule as sched
    from src.pipeline.forward_slate import attach_forward_slate
    from tests.test_forward_slate import forward_only, panel

    now = datetime(2026, 10, 12, 18, 0, tzinfo=timezone.utc)
    result = _slate(
        _game("401700001", season_type=SEASON_TYPE_PRESEASON,
              tipoff=now + timedelta(hours=2), home="LAL", away="BOS"),
        _game("401700002", season_type=SEASON_TYPE_REGULAR,
              tipoff=now + timedelta(hours=4), home="BOS", away="LAL"),
    )
    monkeypatch.setattr(sched, "load_slate", lambda *a, **k: result)
    out = attach_forward_slate(panel(), slate_date="2026-10-12")

    assert set(forward_only(out.panel)["GAME_ID"]) == {"401700002"}, (
        "preseason rows were built, and a seeded artifact would have scored them"
    )
    assert any("not projected" in n and "preseason" in n for n in out.notes), (
        "the result does not say which game was dropped or why"
    )
    assert any("1 projectable game(s) of 2 pre-tip" in n for n in out.notes), (
        "the counts do not distinguish projectable from pre-tip, so the note "
        "describes the wrong number"
    )


def test_both_consumers_read_the_projectable_attribute(monkeypatch):
    """
    ONE READER, asserted on the AST rather than on a substring: the word
    appears in both files' comments, so ``"projectable" in source`` is
    satisfied by prose. This requires an actual ``getattr(x, "projectable",
    ...)`` call in each consumer.
    """
    import ast
    import inspect

    import scheduler_worker as worker
    from src.pipeline import forward_slate

    def reads_projectable(source: str) -> bool:
        tree = ast.parse(inspect.cleandoc(source))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value == "projectable"
            ):
                return True
        return False

    assert reads_projectable(inspect.getsource(worker.schedule_prelock_jobs)), (
        "the pre-lock arming never reads .projectable, so it will arm a "
        "tip-off job for a preseason fixture"
    )
    assert reads_projectable(inspect.getsource(forward_slate.attach_forward_slate)), (
        "the forward slate never reads .projectable, so preseason rows will be "
        "built and scored"
    )


def test_an_all_preseason_slate_is_loud_rather_than_empty():
    """
    A board that goes dark must never read as a quiet night. With games on the
    schedule and none of them projectable, the reason belongs in the result,
    not only in a log line nobody greps.
    """
    result = parse_scoreboard(_payload(
        _event(event_id="1", season={"type": SEASON_TYPE_PRESEASON}),
        _event(event_id="2", season={"type": SEASON_TYPE_PRESEASON}),
    ))
    assert result.status == "OK", "the fetch worked; it is the slate that is empty"
    assert result.projectable == []
    text = " ".join(result.notes).lower()
    assert "preseason" in text
    assert "2" in text, "the count of excluded games is not stated"
    assert "no projectable games" in text, (
        "nothing in the result distinguishes an all-excluded slate from an "
        "ordinary empty one -- the board goes dark and reads as a quiet night"
    )


# --- the field itself ------------------------------------------------------


def test_slate_game_exposes_the_season_type_it_read():
    """``as_dict`` is what the CSV and the CLI show; a filter that drops games
    invisibly is not auditable."""
    result = parse_scoreboard(
        _payload(_event(season={"year": 2027, "type": SEASON_TYPE_PRESEASON}))
    )
    row = result.games[0].as_dict()
    assert row["SEASON_TYPE"] == SEASON_TYPE_PRESEASON
    assert row["EXCLUSION_REASON"]


@pytest.mark.parametrize("raw", ["2", 2, " 2 "])
def test_the_season_type_is_read_whether_it_arrives_as_text_or_a_number(raw):
    """ESPN returns ids as strings in some blocks and integers in others, and
    this is the key the whole filter turns on."""
    result = parse_scoreboard(_payload(_event(season={"type": raw})))
    assert result.games[0].season_type == SEASON_TYPE_REGULAR
    assert result.games[0].is_preseason is False


def test_a_hand_built_slate_game_still_works_without_the_new_fields():
    """The dataclass is constructed positionally in places; the new fields must
    be optional or every caller breaks."""
    game = SlateGame(
        espn_event_id="1", tipoff_utc=None, slate_date_pt=date(2026, 10, 12),
        home_team="POR", away_team="GSW", state="pre", status_detail=None,
        venue=None, venue_city=None, is_neutral_site=None,
    )
    assert game.season_type is None
    assert game.is_preseason is None
    assert game.exclusion_reason is None


def test_a_slate_object_without_the_filter_falls_back_loudly(caplog):
    """
    FAILS OPEN, LOUDLY. Both consumers read ``projectable`` through getattr, so
    a slate-like object that predates the property (a test double, a caller
    outside this repository) would otherwise produce an empty slate — a
    pipeline that silently stops writing rows, which is the worse failure of
    the two. It falls back to pre-tip games and says so.
    """
    import logging

    import src.ingestion.espn_schedule as sched
    from src.pipeline.forward_slate import attach_forward_slate
    from tests.test_forward_slate import Game, forward_only, panel

    class OldSlate:
        status = "OK"
        notes: list[str] = []
        unmapped_teams: list[str] = []
        games = [Game("401700001", "LAL", "BOS")]
        pregame_only = [Game("401700001", "LAL", "BOS")]

    original = sched.load_slate
    sched.load_slate = lambda *a, **k: OldSlate()
    try:
        with caplog.at_level(logging.WARNING):
            out = attach_forward_slate(panel(), slate_date="2025-11-15")
    finally:
        sched.load_slate = original

    assert set(forward_only(out.panel)["GAME_ID"]) == {"401700001"}, (
        "the fallback dropped the rows instead of building them"
    )
    assert any("no .projectable" in r.getMessage() for r in caplog.records), (
        "the fallback is silent, so a slate that skipped the preseason filter "
        "looks identical to one that applied it"
    )


def test_the_events_own_season_type_beats_the_requests(monkeypatch):
    """
    cubic, PR #9 (P1): the flat ``event["seasonType"]`` was read AFTER
    ``payload.season``, so a scoreboard whose request-level block said "regular
    season" overrode an event that said "preseason" — and that game stayed
    projectable. The leak the whole filter exists to close, in the one case
    where the two disagree.
    """
    event = _event()
    event["seasonType"] = SEASON_TYPE_PRESEASON
    result = parse_scoreboard(
        _payload(event, season={"year": 2027, "type": SEASON_TYPE_REGULAR})
    )
    game = result.games[0]
    assert game.season_type_source == "event.seasonType", (
        f"read from {game.season_type_source!r}: the request's season block "
        "overrode the event's own"
    )
    assert game.is_preseason is True
    assert result.projectable == []


def test_the_request_level_block_is_still_the_last_resort():
    """The other half: with nothing on the event, the request's block is better
    than nothing — it just must not outrank the event."""
    result = parse_scoreboard(
        _payload(_event(), season={"year": 2027, "type": SEASON_TYPE_PRESEASON})
    )
    assert result.games[0].season_type_source == "payload.season"
    assert result.projectable == []


def test_an_all_excluded_slate_is_not_reported_as_an_off_day(monkeypatch):
    """
    cubic, PR #9: `build_forward_rows` saw the empty projectable list and
    reported status EMPTY with "no scheduled games — a valid off-day", and
    `slate.notes` — which carries the NO PROJECTABLE GAMES marker — was never
    carried over. The run summary then read as a quiet night, which is the
    confusion the exclusion machinery exists to prevent, one layer out.
    """
    from datetime import datetime, timedelta, timezone

    import src.ingestion.espn_schedule as sched
    from src.pipeline.forward_slate import attach_forward_slate
    from tests.test_forward_slate import panel

    now = datetime(2026, 10, 12, 18, 0, tzinfo=timezone.utc)
    result = _slate(
        _game("401700001", season_type=SEASON_TYPE_PRESEASON,
              tipoff=now + timedelta(hours=2), home="LAL", away="BOS"),
        _game("401700002", season_type=SEASON_TYPE_PRESEASON,
              tipoff=now + timedelta(hours=4), home="BOS", away="LAL"),
    )
    result.notes.append("2 preseason game(s) excluded from the projectable slate")
    monkeypatch.setattr(sched, "load_slate", lambda *a, **k: result)
    out = attach_forward_slate(panel(), slate_date="2026-10-12")

    assert out.status == "ALL_EXCLUDED", (
        f"status {out.status!r}: an all-excluded slate is indistinguishable "
        "from an off-day"
    )
    text = " ".join(out.notes)
    assert "NO PROJECTABLE GAMES" in text
    assert "not an off-day" in text
    assert "preseason game(s) excluded" in text, (
        "the schedule's own notes were dropped, so the reason never reaches "
        "the run summary"
    )


def test_every_exclusion_reason_reaches_the_notes(monkeypatch):
    """cubic, PR #9: the reasons were sliced to five, so on a heavily excluded
    slate the later games vanished from the summary."""
    from datetime import datetime, timedelta, timezone

    import src.ingestion.espn_schedule as sched
    from src.pipeline.forward_slate import attach_forward_slate
    from tests.test_forward_slate import panel

    now = datetime(2026, 10, 12, 18, 0, tzinfo=timezone.utc)
    games = [
        _game(f"4017000{i:02d}", season_type=SEASON_TYPE_PRESEASON,
              tipoff=now + timedelta(hours=2 + i), home="LAL", away="BOS")
        for i in range(7)
    ]
    games.append(_game("401700099", season_type=SEASON_TYPE_REGULAR,
                       tipoff=now + timedelta(hours=12), home="BOS", away="LAL"))
    monkeypatch.setattr(sched, "load_slate", lambda *a, **k: _slate(*games))
    out = attach_forward_slate(panel(), slate_date="2026-10-12")

    dropped = [n for n in out.notes if "not projected" in n]
    assert len(dropped) == 7, (
        f"{len(dropped)} of 7 exclusions reached the notes — the rest were "
        "silently truncated"
    )


def test_the_armed_log_does_not_call_the_projectable_count_pre_tip(
    monkeypatch, caplog
):
    """
    cubic, PR #9 (P3): "armed %d of %d pre-tip game(s)" kept its wording after
    `games` became the projectable list, so a slate with two pre-tip games of
    which one was preseason reported "armed 1 of 1 pre-tip game(s)". The
    text-outliving-its-facts defect this PR fixes explicitly elsewhere.
    """
    import logging
    from datetime import datetime, timedelta, timezone

    import scheduler_worker as worker
    from tests.test_scheduler_worker import FakeScheduler

    monkeypatch.delenv("PROPIQ_PRELOCK", raising=False)
    monkeypatch.delenv("PROPIQ_PRELOCK_LEAD_MINUTES", raising=False)
    now = datetime(2026, 10, 12, 18, 0, tzinfo=timezone.utc)
    slate = _slate(
        _game("401914130", season_type=SEASON_TYPE_PRESEASON,
              tipoff=now + timedelta(hours=2), home="POR", away="GSW"),
        _game("401898402", season_type=SEASON_TYPE_REGULAR,
              tipoff=now + timedelta(hours=4), home="ATL", away="OKC"),
    )
    with caplog.at_level(logging.INFO):
        out = worker.schedule_prelock_jobs(FakeScheduler(), now=now, slate=slate)

    text = " ".join(r.getMessage() for r in caplog.records)
    assert "armed 1 of 1 pre-tip" not in text, (
        "the log calls the projectable count 'pre-tip', so one of two pre-tip "
        "games vanishes from the denominator"
    )
    assert "armed 1 of 1 projectable game(s) (2 pre-tip, 1 excluded)" in text
    assert out["pretip"] == 2 and out["projectable"] == 1
    assert out["excluded"] and "preseason" in out["excluded"][0]


# --- cubic's follow-ups on PR #10 -----------------------------------------


def test_the_off_day_note_is_dropped_not_argued_with(monkeypatch):
    """
    cubic, PR #10: the all-excluded block only APPENDED, so
    `build_forward_rows`' "no scheduled games — a valid off-day" stayed two
    lines above "this is not an off-day" in the same summary — the exact
    contradiction the block exists to remove.
    """
    from datetime import datetime, timedelta, timezone

    import src.ingestion.espn_schedule as sched
    from src.pipeline.forward_slate import attach_forward_slate
    from tests.test_forward_slate import panel

    now = datetime(2026, 10, 12, 18, 0, tzinfo=timezone.utc)
    result = _slate(_game("401700001", season_type=SEASON_TYPE_PRESEASON,
                          tipoff=now + timedelta(hours=2), home="LAL", away="BOS"))
    monkeypatch.setattr(sched, "load_slate", lambda *a, **k: result)
    out = attach_forward_slate(panel(), slate_date="2026-10-12")

    text = " ".join(out.notes)
    assert "no scheduled games" not in text, (
        "the off-day note survives beside the not-an-off-day note"
    )
    assert "NO PROJECTABLE GAMES" in text


def test_the_no_projectable_marker_is_not_emitted_twice(monkeypatch):
    """cubic, PR #10: `espn_schedule` already appends the marker under the same
    condition, and this copied slate.notes wholesale."""
    from datetime import datetime, timedelta, timezone

    import src.ingestion.espn_schedule as sched
    from src.pipeline.forward_slate import attach_forward_slate
    from tests.test_forward_slate import panel

    now = datetime(2026, 10, 12, 18, 0, tzinfo=timezone.utc)
    result = _slate(_game("401700001", season_type=SEASON_TYPE_PRESEASON,
                          tipoff=now + timedelta(hours=2), home="LAL", away="BOS"))
    # What parse_scoreboard really puts there.
    result.notes.append(
        "NO PROJECTABLE GAMES: all 1 pre-tip game(s) were excluded."
    )
    monkeypatch.setattr(sched, "load_slate", lambda *a, **k: result)
    out = attach_forward_slate(panel(), slate_date="2026-10-12")

    markers = [n for n in out.notes if "NO PROJECTABLE GAMES" in n]
    assert len(markers) == 1, f"the marker appears {len(markers)} times: {markers}"


def test_an_empty_panel_outranks_the_all_excluded_status(monkeypatch):
    """cubic, PR #10: an empty panel means nothing could be projected whatever
    the filter did, so DATA_NOT_AVAILABLE must not be downgraded."""
    from datetime import datetime, timedelta, timezone

    import pandas as pd

    import src.ingestion.espn_schedule as sched
    from src.pipeline.forward_slate import attach_forward_slate

    now = datetime(2026, 10, 12, 18, 0, tzinfo=timezone.utc)
    result = _slate(_game("401700001", season_type=SEASON_TYPE_PRESEASON,
                          tipoff=now + timedelta(hours=2), home="LAL", away="BOS"))
    monkeypatch.setattr(sched, "load_slate", lambda *a, **k: result)
    out = attach_forward_slate(pd.DataFrame(), slate_date="2026-10-12")

    assert out.status == "DATA_NOT_AVAILABLE", (
        f"status {out.status!r}: the empty panel is the root cause and it was "
        "overwritten by the exclusion status"
    )


def test_the_fallback_path_publishes_no_projectable_counts(monkeypatch, caplog):
    """
    cubic, PR #10: with a slate object that predates `.projectable`, ``games``
    is the UNFILTERED pre-tip list, so the line called excluded fixtures
    projectable and the absent `.excluded` zero exclusions — two false numbers
    presented as measurements. "Not measured" must not read as zero.
    """
    import logging
    from datetime import datetime, timedelta, timezone

    import scheduler_worker as worker
    from tests.test_scheduler_worker import FakeScheduler

    monkeypatch.delenv("PROPIQ_PRELOCK", raising=False)
    monkeypatch.delenv("PROPIQ_PRELOCK_LEAD_MINUTES", raising=False)
    now = datetime(2026, 10, 12, 18, 0, tzinfo=timezone.utc)

    class OldSlate:
        pregame_only = [
            _game("401", season_type=SEASON_TYPE_REGULAR,
                  tipoff=now + timedelta(hours=2)),
        ]

    with caplog.at_level(logging.INFO):
        out = worker.schedule_prelock_jobs(FakeScheduler(), now=now, slate=OldSlate())

    text = " ".join(r.getMessage() for r in caplog.records)
    assert "UNFILTERED" in text and "not known" in text
    assert "projectable game(s)" not in text, (
        "the fallback path still calls the unfiltered list projectable"
    )
    assert out["filtered"] is False
    # ALL THREE, cubic PR #10: leaving `pretip` unasserted would let a
    # regression publish one numeric count and still pass.
    assert out["projectable"] is None, out["projectable"]
    assert out["pretip"] is None, out["pretip"]
    assert out["excluded"] is None, (
        "absent counts are published as numbers, so 0 exclusions reads as a "
        "measurement rather than as unmeasured"
    )
