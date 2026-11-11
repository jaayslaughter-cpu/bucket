"""Lineup stints from the ESPN plays already stored.

THE MODULE ARRIVED WITHOUT TESTS. It was delivered on 2026-11-11 with a
report stating four bugs were "pinned by tests"; no test file came with it.
These are those tests, plus one for a fifth bug the delivered code still had:
ESPN's minute-less clock form made the builder silently skip substitutions and
cost a whole team's stints on the first real game it was run against.

Every guard is mutation-tested: fix reverted, named test watched go red, fix
put back.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.features.stints import (
    PLAYERS_ON_FLOOR,
    REGULATION_PERIOD_SECONDS,
    StintError,
    build_stints,
    parse_clock_seconds,
    period_length,
    player_seconds_on_court,
    validate_stints_against_minutes,
)

GAME = "espn:401909945"
TEAM = "4"
FIVE = ["p1", "p2", "p3", "p4", "p5"]


def _play(seq, *, period=1, clock="12:00", type_id="0", participants="",
          team=TEAM, game=GAME):
    return {
        "NBA_GAME_ID": game, "SEQUENCE_NUMBER": seq, "PERIOD": period,
        "CLOCK_DISPLAY": clock, "ESPN_TEAM_ID": team,
        "PLAY_TYPE_ID": type_id, "PLAY_TYPE_TEXT": "",
        "PARTICIPANT_ATHLETE_IDS": participants,
    }


def _starters(players=None, *, team=TEAM, game=GAME):
    return pd.DataFrame([
        {"NBA_GAME_ID": game, "ESPN_TEAM_ID": team, "ESPN_ATHLETE_ID": p}
        for p in (players or FIVE)
    ])


# --- the clock, which is where the delivered code was wrong ---------------


@pytest.mark.parametrize("value,expected", [
    ("12:00", 720.0), ("7:01", 421.0), ("0:00", 0.0), (" 1:53 ", 113.0),
    # THE FORM THAT WAS MISSED. ESPN drops the minute under one minute.
    ("48.6", 48.6), ("13.4", 13.4), ("0.0", 0.0),
])
def test_both_espn_clock_forms_parse(value, expected):
    assert parse_clock_seconds(value) == pytest.approx(expected)


@pytest.mark.parametrize("value", ["", "12:", ".", "x", None, "1:2:3", 12])
def test_an_unparseable_clock_is_nan_not_zero(value):
    """Zero would read as "the period ended here" and close every stint."""
    assert parse_clock_seconds(value) != parse_clock_seconds(value)


def test_an_unparseable_clock_refuses_the_team_rather_than_skipping_the_sub():
    """
    THE FIFTH BUG, AND IT COST HALF A GAME. The delivered code `continue`d on
    an unparseable clock. `_floor_state_invariant_report` never reads the
    clock, so it APPLIED that substitution while the builder skipped it; the
    next real substitution then found its `leaving` player already absent,
    removed nobody, appended the entering one, and produced a SIX-player
    lineup. The five-player backstop discarded that whole team -- so one of
    the two teams in the 2026-11-09 Bulls game built no stints while the other
    validated cleanly at 0.35 minutes mean error.

    Without a clock the stint's seconds cannot be computed, so a refusal with
    a reason is the only honest outcome.
    """
    plays = pd.DataFrame([
        _play(1, clock="12:00"),
        _play(2, clock="not a clock", type_id="584", participants="p6;p1"),
        _play(3, clock="5:00", type_id="584", participants="p7;p2"),
    ])
    out = build_stints(plays, _starters())
    assert out.empty, "a substitution with no clock was priced into a stint"


# --- the floor-state invariant -------------------------------------------


def test_a_contradictory_substitution_stream_refuses_the_team():
    """
    The leaving player must be on the floor and the entering one must not. A
    stream that contradicts itself cannot have its floor state recovered, and
    a stint built on it would have four or six players.
    """
    plays = pd.DataFrame([
        # p9 was never on the floor, so it cannot leave.
        _play(1, clock="10:00", type_id="584", participants="p6;p9"),
    ])
    assert build_stints(plays, _starters()).empty


def test_a_team_without_five_starters_is_refused():
    plays = pd.DataFrame([_play(1, clock="10:00", type_id="584",
                                participants="p6;p1")])
    assert build_stints(plays, _starters(["p1", "p2", "p3"])).empty


# --- only substitutions move the floor -----------------------------------


def test_a_two_participant_play_that_is_not_a_substitution_is_ignored():
    """
    THE BUG THE REPORT NAMES THIRD. The feed lists two participants on far
    more than substitutions -- a made three names the shooter and the
    assister, a steal names both players -- and reading those pairs as
    substitutions produced 103 impossible subs in one game. The type id (584)
    is the discriminator; the participants are only its operands.
    """
    plays = pd.DataFrame([
        # An assisted basket: two participants, NOT type 584.
        _play(1, clock="11:00", type_id="101", participants="p1;p2"),
        _play(2, clock="6:00", type_id="584", participants="p6;p1"),
    ])
    out = build_stints(plays, _starters())
    assert not out.empty, "a legitimate substitution was lost"
    lineups = {frozenset(l.split(";")) for l in out["LINEUP"]}
    assert frozenset(FIVE) in lineups, (
        "the assist pair was applied as a substitution and the starting five "
        "never appears"
    )


# --- every period opens and closes with its own five ---------------------


def test_each_period_opens_at_its_own_length():
    """
    THE BUG THE REPORT NAMES SECOND. A period opens at the tip, not at its
    first substitution, so a player who never left showed only his bench
    time. The first stint of every period is the starters, for the full
    period if nobody is substituted.
    """
    plays = pd.DataFrame([_play(1, period=2, clock="12:00", type_id="0")])
    out = build_stints(plays, _starters())
    assert len(out) == 1
    assert out["PERIOD"].iloc[0] == 2
    assert out["STINT_SECONDS"].iloc[0] == pytest.approx(REGULATION_PERIOD_SECONDS)


def test_a_periods_seconds_sum_to_the_period_length():
    """Two stints in one period must account for the whole period."""
    plays = pd.DataFrame([
        _play(1, clock="12:00"),
        _play(2, clock="8:00", type_id="584", participants="p6;p1"),
    ])
    out = build_stints(plays, _starters())
    assert out["STINT_SECONDS"].sum() == pytest.approx(REGULATION_PERIOD_SECONDS)


def test_each_period_closes_with_the_five_that_was_on_the_floor():
    """
    THE BUG THE REPORT NAMES FOURTH. Every period's closing stint was
    credited to the GAME's final five, which gave bench players minutes they
    played in no period -- a 3.4-minute mean error.
    """
    plays = pd.DataFrame([
        _play(1, period=1, clock="12:00"),
        _play(2, period=1, clock="6:00", type_id="584", participants="p6;p1"),
        _play(3, period=2, clock="12:00"),
        _play(4, period=2, clock="6:00", type_id="584", participants="p7;p2"),
    ])
    out = build_stints(plays, _starters())
    p1_close = out.loc[out["PERIOD"] == 1, "LINEUP"].iloc[-1].split(";")
    assert "p7" not in p1_close, (
        "period 1 closed with a player who only entered in period 2"
    )
    for lineup in out["LINEUP"]:
        assert len(lineup.split(";")) == PLAYERS_ON_FLOOR, lineup


def test_overtime_is_five_minutes_not_twelve():
    assert period_length(4) == REGULATION_PERIOD_SECONDS
    assert period_length(5) == 300
    assert period_length(0) != period_length(0)  # NaN


# --- failure isolation ---------------------------------------------------


def test_one_games_refusal_does_not_cost_the_others():
    """The same failure-isolation rule the panel fetch follows."""
    good = [_play(1, clock="12:00"),
            _play(2, clock="8:00", type_id="584", participants="p6;p1")]
    bad = [_play(1, game="espn:BAD", clock="10:00", type_id="584",
                 participants="p6;p9")]
    plays = pd.DataFrame(good + bad)
    starters = pd.concat([_starters(), _starters(game="espn:BAD")])
    out = build_stints(plays, starters)
    assert set(out["NBA_GAME_ID"]) == {GAME}, (
        "the contradictory game took the clean one with it"
    )


def test_missing_columns_refuse_rather_than_producing_empty_stints():
    with pytest.raises(StintError, match="DATA_NOT_AVAILABLE"):
        build_stints(pd.DataFrame([{"NBA_GAME_ID": GAME}]), _starters())
    with pytest.raises(StintError, match="DATA_NOT_AVAILABLE"):
        build_stints(pd.DataFrame(), _starters())


# --- the per-player view and its validation ------------------------------


def test_seconds_on_court_sums_every_stint_a_player_appears_in():
    plays = pd.DataFrame([
        _play(1, clock="12:00"),
        _play(2, clock="6:00", type_id="584", participants="p6;p1"),
    ])
    oc = player_seconds_on_court(build_stints(plays, _starters()))
    # The keys carry the panel's `espn:` prefix -- see
    # `test_on_court_ids_join_to_the_panels_player_id` for why this is the
    # contract and not an accident. THIS TEST ASSERTED THE BARE FORM UNTIL
    # 2026-11-11, which is how it agreed with a join that matched nothing.
    by_player = dict(zip(oc["ESPN_ATHLETE_ID"], oc["SECONDS_ON_COURT"]))
    # p2..p5 never left: the whole period.
    assert by_player["espn:p2"] == pytest.approx(REGULATION_PERIOD_SECONDS)
    # p1 played the first six minutes, p6 the last six.
    assert by_player["espn:p1"] == pytest.approx(360.0)
    assert by_player["espn:p6"] == pytest.approx(360.0)


def test_the_validation_reports_n_zero_rather_than_dividing_by_nothing():
    assert validate_stints_against_minutes(pd.DataFrame(), pd.DataFrame()) == {"n": 0.0}


def test_the_validation_measures_against_the_box_scores_own_minutes():
    plays = pd.DataFrame([
        _play(1, clock="12:00"),
        _play(2, clock="6:00", type_id="584", participants="p6;p1"),
    ])
    oc = player_seconds_on_court(build_stints(plays, _starters()))
    # `espn:`-prefixed, because that is what `load_player_panel` emits.
    panel = pd.DataFrame([
        {"GAME_ID": GAME, "PLAYER_ID": f"espn:{p}", "MIN": 12.0}
        for p in ("p2", "p3", "p4", "p5")
    ] + [{"GAME_ID": GAME, "PLAYER_ID": "espn:p1", "MIN": 6.0},
         {"GAME_ID": GAME, "PLAYER_ID": "espn:p6", "MIN": 6.0}])
    out = validate_stints_against_minutes(oc, panel)
    assert out["n"] == 6.0
    assert out["mean_abs_error_minutes"] == pytest.approx(0.0, abs=1e-9)
    assert out["within_2_minutes"] == 1.0


def test_a_substitution_bringing_on_a_player_already_on_the_floor_is_refused():
    """
    THE CASE THE DELIVERED BACKSTOP MISSED. An ENTERING player who is already
    out there makes the filter remove the leaver and the append add a
    duplicate: `p1;p1;p3;p4;p5` -- five entries, FOUR players. The backstop
    counted parts, so it passed, and the row credited p1 twice and p2 not at
    all.

    Both the floor-state invariant and the (now distinct-counting) backstop
    catch this. They overlap on purpose, which is why neither can be killed
    by a mutation on its own; see the note in the module.
    """
    plays = pd.DataFrame([
        _play(1, clock="12:00"),
        _play(2, clock="6:00", type_id="584", participants="p1;p2"),
    ])
    assert build_stints(plays, _starters()).empty


def test_no_shipped_lineup_ever_repeats_a_player():
    """The property, over a stream with many substitutions."""
    plays = [_play(1, clock="12:00")]
    for i, (enter, leave) in enumerate(
        [("p6", "p1"), ("p7", "p2"), ("p1", "p6"), ("p8", "p3")], start=2
    ):
        plays.append(_play(i, clock=f"{12 - i}:00", type_id="584",
                           participants=f"{enter};{leave}"))
    out = build_stints(pd.DataFrame(plays), _starters())
    assert not out.empty
    for lineup in out["LINEUP"]:
        players = lineup.split(";")
        assert len(players) == len(set(players)) == PLAYERS_ON_FLOOR, lineup


# --- the lineup predicate, tested directly -------------------------------
#
# The floor-state invariant rejects every input that would reach the backstop,
# so no end-to-end test can show the predicate itself is wrong -- and the
# delivered one WAS wrong. Tested here as a function for that reason.


@pytest.mark.parametrize("lineup", [
    "p1;p2;p3;p4;p5",
    "p5;p4;p3;p2;p1",   # order is not part of validity
])
def test_five_distinct_players_is_a_lineup(lineup):
    from src.features.stints import lineup_is_valid

    assert lineup_is_valid(lineup)


@pytest.mark.parametrize("lineup,why", [
    ("p1;p1;p3;p4;p5", "five entries, four players -- what the delivered "
                       "part-count check passed"),
    ("p1;p2;p3;p4", "four"),
    ("p1;p2;p3;p4;p5;p6", "six"),
    ("p1;p2;p3;p4;", "an empty id padding it to five"),
    (";;;;", "all empty"),
    ("", "nothing"),
])
def test_anything_else_is_not_a_lineup(lineup, why):
    from src.features.stints import lineup_is_valid

    assert not lineup_is_valid(lineup), why


# ---------------------------------------------------------------------------
# The join to the panel. Added 2026-11-11, after it joined nothing.
# ---------------------------------------------------------------------------


def test_on_court_ids_join_to_the_panels_player_id() -> None:
    """
    THE VALIDATION'S JOIN KEY, WHICH MATCHED ZERO ROWS AS SHIPPED.

    `LINEUP` carries ESPN's bare athlete ids. The panel's `PLAYER_ID` is the
    same id with this project's `espn:` prefix (`load_player_panel` builds it
    from `player_id_nba`, which `espn_panel` writes through
    `espn_player_id`). The first draft emitted the bare form, so
    `validate_stints_against_minutes` -- an INNER join -- matched nothing on
    every real night, returned `n: 0`, and the night's job read that as a
    correlation of zero.

    Measured on 2026-11-09's eight games: 185 player-games, none of which
    joined. With the prefix: 185 of 185, correlation 0.9996.
    """
    from src.features.stints import player_seconds_on_court

    stints = pd.DataFrame([
        {"NBA_GAME_ID": "espn:401909952", "ESPN_TEAM_ID": "13",
         "LINEUP": "2490149;3146557;3907498;4066648;4433134",
         "PERIOD": 1, "STINT_SECONDS": 600.0},
    ])
    out = player_seconds_on_court(stints)

    assert len(out) == 5
    for athlete in out["ESPN_ATHLETE_ID"]:
        assert athlete.startswith("espn:"), (
            f"{athlete!r} is the bare ESPN id; the panel's PLAYER_ID is "
            f"espn:-prefixed, so this cannot join and the validation that "
            f"rests on it silently measures nothing"
        )


def test_the_validation_measures_a_panel_shaped_frame() -> None:
    """
    End to end on the real column names, which is what the bug needed.

    The test above asserts the prefix; this asserts the join it exists for,
    against a frame named the way `load_player_panel` names one -- `GAME_ID`,
    `PLAYER_ID`, `MIN`. A prefix fix that still failed to join would pass the
    first test and fail this one.
    """
    from src.features.stints import (
        player_seconds_on_court,
        validate_stints_against_minutes,
    )

    stints = pd.DataFrame([
        {"NBA_GAME_ID": "espn:401909952", "ESPN_TEAM_ID": "13",
         "LINEUP": "2490149;3146557;3907498;4066648;4433134",
         "PERIOD": 1, "STINT_SECONDS": 600.0},
    ])
    on_court = player_seconds_on_court(stints)
    panel = pd.DataFrame([
        {"GAME_ID": "espn:401909952", "PLAYER_ID": f"espn:{pid}", "MIN": 10.0}
        for pid in ("2490149", "3146557", "3907498", "4066648", "4433134")
    ])

    result = validate_stints_against_minutes(on_court, panel)

    assert result["n"] == 5.0, (
        f"the validation joined {result.get('n')} of 5 player-games; an "
        f"inner join that matches nothing reports no error at all"
    )
    assert result["mean_abs_error_minutes"] == pytest.approx(0.0)
    assert result["within_2_minutes"] == 1.0


# ---------------------------------------------------------------------------
# The night's job. A check that did not run is not a check that failed.
# ---------------------------------------------------------------------------


def _stint_job_env(monkeypatch, tmp_path, panel):
    """Drive the real 05:30 job body with its reads stubbed."""
    import scheduler_worker as worker
    import src.db.repository as repo

    plays = pd.DataFrame([
        _play(1, clock="12:00"),
        _play(2, clock="6:00", type_id="584", participants="p6;p1"),
    ])
    monkeypatch.setattr(repo, "load_espn_plays", lambda *_a, **_k: plays)
    monkeypatch.setattr(repo, "load_espn_starters", lambda *_a, **_k: _starters())
    monkeypatch.setattr(repo, "load_player_panel", lambda *_a, **_k: panel)
    monkeypatch.setattr(
        "src.utils.volume.resolve_state_root", lambda: (tmp_path, "test"),
    )
    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(
        worker, "notify_failure",
        lambda stage, detail, **kw: sent.append((stage, detail)),
    )
    monkeypatch.setenv(worker.ENV_STINTS, "true")
    return worker, sent


def test_an_unjoined_validation_is_reported_as_unmeasured_not_as_failed(
    monkeypatch, tmp_path,
) -> None:
    """
    THE DEFECT THIS REPLACED. The job read `validation.get("corr", 0) < 0.9`,
    so a validation that joined nothing scored zero and sent a nightly notice
    saying the reconstruction had fallen below 0.9 correlation -- a verdict on
    a measurement that was never taken. A false alarm every night is how a
    real one gets ignored, so the two states get two different messages.

    The panel here is non-empty and joins nothing, which is exactly the shape
    the id-prefix bug produced in production.
    """
    panel = pd.DataFrame([
        {"GAME_ID": GAME, "PLAYER_ID": "nba:999", "MIN": 12.0},
    ])
    worker, sent = _stint_job_env(monkeypatch, tmp_path, panel)

    out = worker.run_lineup_stints()

    assert out["status"] == "OK", out
    assert out["validation_status"] == "CANNOT_TELL", out
    assert sent, "the job measured nothing and said nothing"
    stage, detail = sent[0]
    assert stage == "lineup-stints"
    assert "NOT validated" in detail, detail
    assert "0.9" not in detail, (
        f"the notice still claims a correlation verdict it does not hold: "
        f"{detail!r}"
    )


def test_a_validation_that_agrees_with_the_box_score_sends_nothing(
    monkeypatch, tmp_path,
) -> None:
    """The pass case, so CANNOT_TELL is not simply always returned."""
    panel = pd.DataFrame([
        {"GAME_ID": GAME, "PLAYER_ID": f"espn:{p}", "MIN": 12.0}
        for p in ("p2", "p3", "p4", "p5")
    ] + [{"GAME_ID": GAME, "PLAYER_ID": "espn:p1", "MIN": 6.0},
         {"GAME_ID": GAME, "PLAYER_ID": "espn:p6", "MIN": 6.0}])
    worker, sent = _stint_job_env(monkeypatch, tmp_path, panel)

    out = worker.run_lineup_stints()

    assert out["validation_status"] == "OK", out
    assert out["validation"]["n"] == 6.0, out["validation"]
    assert not sent, f"a clean validation still alerted: {sent}"


def test_a_validation_that_disagrees_with_the_box_score_alerts(
    monkeypatch, tmp_path,
) -> None:
    """
    THE OTHER VERDICT, which the CANNOT_TELL test cannot reach.

    MUTATION THAT SURVIVED THE FIRST ATTEMPT: making the `corr < 0.9` branch
    unreachable. Every other test here either joins cleanly or joins nothing,
    so a reconstruction that joins and is WRONG had no case -- which is the
    branch that matters most, because it is the one that catches a broken
    substitution stream.

    SECOND MUTATION THAT SURVIVED: moving the threshold from 0.9 to 0.0. The
    first version of this test used an ANTI-correlated panel, whose
    correlation is negative and so below either bar -- it proved the branch
    runs but not that 0.9 is the number. The disagreement here is mild:
    correlation about 0.11, which a 0.9 bar catches and a 0.0 bar waves
    through. A reconstruction can be badly wrong while still pointing the
    right way, and that is the case worth catching.
    """
    panel = pd.DataFrame([
        {"GAME_ID": GAME, "PLAYER_ID": "espn:p2", "MIN": 12.0},
        {"GAME_ID": GAME, "PLAYER_ID": "espn:p3", "MIN": 12.0},
        {"GAME_ID": GAME, "PLAYER_ID": "espn:p4", "MIN": 2.0},
        {"GAME_ID": GAME, "PLAYER_ID": "espn:p5", "MIN": 2.0},
        {"GAME_ID": GAME, "PLAYER_ID": "espn:p1", "MIN": 6.0},
        {"GAME_ID": GAME, "PLAYER_ID": "espn:p6", "MIN": 6.0},
    ])
    worker, sent = _stint_job_env(monkeypatch, tmp_path, panel)

    out = worker.run_lineup_stints()

    assert out["validation_status"] == "ALERT", out
    assert out["validation"]["n"] == 6.0, out["validation"]
    assert 0.0 < out["validation"]["corr"] < 0.9, (
        f"the disagreement must sit strictly between the two candidate "
        f"thresholds or it cannot tell them apart: {out['validation']}"
    )
    assert sent, "the reconstruction disagreed with the box score in silence"
    stage, detail = sent[0]
    assert stage == "lineup-stints"
    assert "0.9" in detail, detail
    assert "NOT validated" not in detail, (
        f"a measured disagreement was reported as an unmeasured one: {detail!r}"
    )
