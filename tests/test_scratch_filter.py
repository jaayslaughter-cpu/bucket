"""The pre-tip scratch filter (audit finding R4).

One rule carries this module, and an earlier version of the code underneath it
got the rule exactly backwards: AN UNUSABLE INJURY FEED MUST NOT READ AS A
HEALTHY SLATE. Most of these tests are about the feed failing, because that is
the case where a wrong answer looks like a right one.

The injury rows below are TEST FIXTURES. No live ESPN response is asserted
anywhere — every ESPN host is denied from this environment.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.ingestion.espn_availability import AvailabilityReport, InjuryRow
from src.pipeline.scratches import (
    AVAILABILITY_COLUMN,
    AVAILABLE,
    DETAIL_COLUMN,
    STATUS_OK,
    STATUS_UNAVAILABLE,
    UNKNOWN,
    UNVERIFIED,
    WITHHELD,
    apply_scratch_filter,
)


def projections(*names: str) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "PLAYER_NAME": n, "GAME_ID": "0022500123", "MARKET": "PTS",
            "LINE": 25.5, "PROB_OVER": 0.6,
        }
        for n in (names or ("DEMO_A",))
    ])


def injury(name: str, status: str, *, detail: str | None = "left ankle") -> InjuryRow:
    return InjuryRow(
        player_name=name, espn_athlete_id=None, espn_team_id=None,
        team_abbreviation="LAL", status=status, status_raw=status.title(),
        detail=detail, reported_date="2026-09-29",
    )


def report(*rows: InjuryRow, status: str = "OK") -> AvailabilityReport:
    return AvailabilityReport(status=status, injuries=list(rows))


# --- an unusable feed labels, it never clears ---------------------------

def test_a_failed_report_marks_every_row_unverified_and_drops_nothing():
    rows = projections("DEMO_A", "DEMO_B", "DEMO_C")
    out = apply_scratch_filter(rows, report(status="DATA_NOT_AVAILABLE"))

    assert out.status == STATUS_UNAVAILABLE
    assert out.verified is False
    assert len(out.projections) == 3, "a failed check must not drop rows"
    assert set(out.projections[AVAILABILITY_COLUMN]) == {UNVERIFIED}
    assert out.withheld == []
    assert "every row is unverified" in out.reason


def test_an_unreachable_feed_marks_every_row_unverified(monkeypatch):
    """A scheduled slate must not die because an injury page moved."""
    import src.ingestion.espn_availability as espn

    def boom(**kwargs):
        raise RuntimeError("403 from the proxy")

    monkeypatch.setattr(espn, "fetch_injuries", boom)

    out = apply_scratch_filter(projections("DEMO_A"))
    assert out.status == STATUS_UNAVAILABLE
    assert list(out.projections[AVAILABILITY_COLUMN]) == [UNVERIFIED]
    assert "unreachable" in out.reason
    assert "403" in out.reason


def test_unverified_is_not_available():
    """
    The distinction the whole module exists for, asserted directly so no future
    refactor can collapse the two into one truthy value.
    """
    out = apply_scratch_filter(projections("DEMO_A"), report(status="FAILED"))
    assert UNVERIFIED != AVAILABLE
    assert out.projections[AVAILABILITY_COLUMN].iloc[0] != AVAILABLE


def test_a_failed_check_reports_itself_as_unverified_in_the_summary():
    out = apply_scratch_filter(projections("DEMO_A"), report(status="FAILED"))
    payload = out.as_dict()
    assert payload["verified"] is False
    assert payload["by_availability"] == {UNVERIFIED: 1}
    assert "not a cleared row" in payload["note"]


# --- a usable feed -------------------------------------------------------

def test_a_player_listed_out_is_withheld():
    out = apply_scratch_filter(
        projections("DEMO_A", "DEMO_B"), report(injury("DEMO_A", "OUT")),
    )
    assert out.status == STATUS_OK
    assert out.verified is True
    labels = dict(zip(out.projections["PLAYER_NAME"], out.projections[AVAILABILITY_COLUMN]))
    assert labels == {"DEMO_A": WITHHELD, "DEMO_B": AVAILABLE}
    assert out.withheld == ["DEMO_A"]


def test_doubtful_is_withheld_too():
    """
    Doubtful players mostly do not play, and the asymmetry favours withholding a
    row that might have been fine over recommending one that will not dress.
    """
    out = apply_scratch_filter(
        projections("DEMO_A"), report(injury("DEMO_A", "DOUBTFUL")),
    )
    assert out.projections[AVAILABILITY_COLUMN].iloc[0] == WITHHELD


@pytest.mark.parametrize("status", ["AVAILABLE", "PROBABLE", "QUESTIONABLE"])
def test_a_listed_but_playing_player_is_available(status):
    out = apply_scratch_filter(projections("DEMO_A"), report(injury("DEMO_A", status)))
    assert out.projections[AVAILABILITY_COLUMN].iloc[0] == AVAILABLE
    assert out.withheld == []


def test_a_row_espn_could_not_bucket_is_unknown_not_available():
    out = apply_scratch_filter(
        projections("DEMO_A"), report(injury("DEMO_A", "DATA_NOT_AVAILABLE")),
    )
    assert out.projections[AVAILABILITY_COLUMN].iloc[0] == UNKNOWN
    assert out.unknown == ["DEMO_A"]
    assert out.withheld == []


def test_a_player_absent_from_the_feed_is_available():
    """Most of a roster is healthy and simply not on an injury report."""
    out = apply_scratch_filter(
        projections("DEMO_A"), report(injury("SOMEONE_ELSE", "OUT")),
    )
    assert out.projections[AVAILABILITY_COLUMN].iloc[0] == AVAILABLE


def test_the_detail_travels_so_a_reader_knows_why():
    out = apply_scratch_filter(
        projections("DEMO_A"), report(injury("DEMO_A", "OUT", detail="left ankle")),
    )
    assert out.projections[DETAIL_COLUMN].iloc[0] == "left ankle"


# --- matching ------------------------------------------------------------

def test_matching_ignores_case_and_extra_whitespace():
    out = apply_scratch_filter(
        projections("  demo_a  "), report(injury("DEMO_A", "OUT")),
    )
    assert out.projections[AVAILABILITY_COLUMN].iloc[0] == WITHHELD


def test_matching_is_not_fuzzy():
    """
    Withholding the wrong player is worse than withholding nobody, and this
    repository has a dedicated crosswalk for name resolution.
    """
    out = apply_scratch_filter(
        projections("DEMO_AB"), report(injury("DEMO_A", "OUT")),
    )
    assert out.projections[AVAILABILITY_COLUMN].iloc[0] == AVAILABLE


def test_every_row_for_one_player_is_labelled():
    """A player has a row per market; a scratch withholds all of them."""
    frame = pd.concat([projections("DEMO_A")] * 3, ignore_index=True)
    frame["MARKET"] = ["PTS", "REB", "AST"]
    out = apply_scratch_filter(frame, report(injury("DEMO_A", "OUT")))
    assert list(out.projections[AVAILABILITY_COLUMN]) == [WITHHELD] * 3
    assert out.as_dict()["by_availability"] == {WITHHELD: 3}


# --- edges ---------------------------------------------------------------

def test_an_empty_frame_abstains_without_touching_the_network(monkeypatch):
    import src.ingestion.espn_availability as espn

    monkeypatch.setattr(
        espn, "fetch_injuries",
        lambda **kw: pytest.fail("an empty frame must not trigger a fetch"),
    )
    out = apply_scratch_filter(pd.DataFrame())
    assert out.status == STATUS_UNAVAILABLE
    assert out.projections.empty


def test_a_frame_with_no_player_name_column_labels_everything_available():
    """
    Nothing can be matched, so nothing can be shown withheld. This is the one
    case where AVAILABLE is reached without a check, and it is reached because
    the frame carries no player to check rather than because a feed said so.
    """
    frame = pd.DataFrame([{"GAME_ID": "1", "MARKET": "PTS"}])
    out = apply_scratch_filter(frame, report(injury("DEMO_A", "OUT")))
    assert list(out.projections[AVAILABILITY_COLUMN]) == [AVAILABLE]


def test_the_input_frame_is_not_mutated():
    frame = projections("DEMO_A")
    apply_scratch_filter(frame, report(injury("DEMO_A", "OUT")))
    assert AVAILABILITY_COLUMN not in frame.columns


def test_the_report_notes_are_carried_through():
    rep = AvailabilityReport(status="OK", injuries=[], notes=["two rows collided"])
    out = apply_scratch_filter(projections("DEMO_A"), rep)
    assert out.notes == ["two rows collided"]


# --- the CLI surface, which is how these modules became reachable --------

def _runner():
    pytest.importorskip("typer")
    from typer.testing import CliRunner

    return CliRunner()


def test_the_injuries_command_reports_a_failed_feed_as_a_failure(monkeypatch):
    """
    "Nobody is hurt" and "we could not ask" must not look the same to a shell
    script, so an unreachable feed exits non-zero rather than printing 0 rows.
    """
    import src.ingestion.espn_availability as espn
    from scripts.nba_model_cli import app

    def boom(**kw):
        raise RuntimeError("403 from the proxy")

    monkeypatch.setattr(espn, "fetch_injuries", boom)
    result = _runner().invoke(app, ["espn-injuries"])
    assert result.exit_code == 2
    assert "unreachable" in result.output
    assert "403" in result.output


def test_the_injuries_command_exits_non_zero_on_a_non_ok_report(monkeypatch):
    import src.ingestion.espn_availability as espn
    from scripts.nba_model_cli import app

    monkeypatch.setattr(
        espn, "fetch_injuries",
        lambda **kw: AvailabilityReport(status="DATA_NOT_AVAILABLE", notes=["empty"]),
    )
    result = _runner().invoke(app, ["espn-injuries"])
    assert result.exit_code == 3
    assert "DATA_NOT_AVAILABLE" in result.output


def test_the_injuries_command_lists_who_is_out(monkeypatch, tmp_path):
    import json

    import src.ingestion.espn_availability as espn
    from scripts.nba_model_cli import app

    monkeypatch.setattr(
        espn, "fetch_injuries",
        lambda **kw: report(injury("DEMO_A", "OUT"), injury("DEMO_B", "QUESTIONABLE")),
    )
    out = tmp_path / "injuries.csv"
    result = _runner().invoke(app, ["espn-injuries", "--out", str(out)])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["rows"] == 2
    assert payload["out_or_doubtful"] == 1
    assert payload["names"] == ["DEMO_A"]
    assert out.exists()


def test_the_slate_command_surfaces_unmapped_teams_rather_than_hiding_them(
    monkeypatch,
):
    """An unmappable code means a game that cannot be joined to the panel."""
    import json
    from datetime import date, datetime, timezone

    import src.ingestion.espn_schedule as sched
    from scripts.nba_model_cli import app

    game = sched.SlateGame(
        espn_event_id="401", tipoff_utc=datetime(2026, 10, 21, 2, tzinfo=timezone.utc),
        slate_date_pt=date(2026, 10, 20), home_team="LAL", away_team="GSW",
        state="pre", status_detail="7:00 PM PT", venue="Crypto.com Arena",
        venue_city="Los Angeles", is_neutral_site=False,
    )
    monkeypatch.setattr(
        sched, "load_slate",
        lambda *a, **k: sched.SlateResult(
            status="OK", slate_date_pt=date(2026, 10, 20), games=[game],
            unmapped_teams=["ZZZ"], notes=["one code had no mapping"],
        ),
    )
    result = _runner().invoke(app, ["espn-slate", "--date", "2026-10-20"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["games"] == 1
    assert payload["pregame"] == 1
    assert payload["unmapped_teams"] == ["ZZZ"]
    assert payload["matchups"] == ["GSW @ LAL"]


def test_the_slate_command_exits_non_zero_when_the_scoreboard_abstains(monkeypatch):
    import src.ingestion.espn_schedule as sched
    from scripts.nba_model_cli import app

    monkeypatch.setattr(
        sched, "load_slate",
        lambda *a, **k: sched.SlateResult(
            status="DATA_NOT_AVAILABLE", notes=["payload was not an object"],
        ),
    )
    result = _runner().invoke(app, ["espn-slate"])
    assert result.exit_code == 3
    assert "not an object" in result.output


def test_the_boxscore_command_separates_played_from_inactive(monkeypatch, tmp_path):
    """
    A post-game DNP is not the pre-tip injury report. The command labels which
    it is showing, because conflating them is how a settled scratch gets read as
    a pre-tip signal it never was.
    """
    import json

    import src.ingestion.espn_game as game_mod
    from scripts.nba_model_cli import app

    def row(name, dnp):
        return game_mod.BoxScoreRow(
            espn_event_id="401", espn_athlete_id=None, player_name=name,
            espn_team_id="13", did_not_play=dnp,
            stats={} if dnp else {"PTS": 24.0},
        )

    monkeypatch.setattr(
        game_mod, "fetch_summary",
        lambda *a, **k: game_mod.GameSummary(
            status="OK", espn_event_id="401",
            box_score=[row("PLAYED_A", False), row("SCRATCH_B", True)],
        ),
    )
    out = tmp_path / "box.csv"
    result = _runner().invoke(
        app, ["espn-boxscore", "--event-id", "401", "--out", str(out)]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["box_score_rows"] == 2
    assert payload["played"] == 1
    assert payload["inactive_names"] == ["SCRATCH_B"]
    assert out.exists()
