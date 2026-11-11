"""
The pre-lock availability re-check finding its own game's rows.

THE DEFECT, OBSERVED LIVE. On 2026-11-09 the board held 312 RECOMMENDED rows
and all six pre-lock jobs logged "no recommended row on the board for this
game". The job scoped by `player_team`/`opponent`, and the board carries
NEITHER: `projection_detail_rows` builds its rows without them because
`persist_projections` has no team column at all. So the filter matched
nothing, every night, since the job was written -- a player ruled out after
the 09:00 slate was never withdrawn before tip.

The existing tests passed throughout because their fixture populates
`player_team`. A fixture that cannot reproduce the production row shape
cannot catch a production defect, so the board below is the live shape.

RESEARCH_ONLY. Synthetic rows; nothing here is a real recommendation.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

import scheduler_worker as worker

EVENT = "0022600321"
OTHER_EVENT = "0022600322"


def _live_board(tmp_path, rows=None):
    """
    The shape the board ACTUALLY has: an event id, and no team on any row.
    """
    rows = rows if rows is not None else [
        (EVENT, "A. Player", "RECOMMENDED"),
        (OTHER_EVENT, "B. Player", "RECOMMENDED"),
    ]
    frame = pd.DataFrame([
        {
            "slate_date": "2026-11-09", "game_date": "2026-11-09",
            "event_id": event, "player_id": f"20000{i}", "player_name": name,
            # NULL, exactly as production writes them.
            "player_team": None, "opponent": None,
            "target_market": "PTS", "side": "over", "line": 24.5,
            "decision_status": status, "model_prob": 0.61,
        }
        for i, (event, name, status) in enumerate(rows)
    ])
    path = tmp_path / "decision_board.csv"
    frame.to_csv(path, index=False)
    return path


class _Report:
    def __init__(self, injuries=(), status="OK"):
        self.status = status
        self.injuries = list(injuries)
        self.notes: list[str] = []


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.delenv("PROPIQ_PRELOCK", raising=False)
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    monkeypatch.setattr(
        "src.ingestion.espn_availability.fetch_injuries", lambda **k: _Report()
    )


def test_a_board_with_no_team_columns_is_still_scoped_to_this_game(
    monkeypatch, tmp_path,
):
    """
    THE LIVE DEFECT. Every production row has a null team, so the only scope
    the job had matched nothing and the check silently examined zero rows.
    """
    monkeypatch.setenv("PROPIQ_BOARD_CSV", str(_live_board(tmp_path)))
    out = worker.run_prelock(event_id=EVENT, home="DAL", away="IND")

    assert out["status"] != "SKIPPED", (
        f"the job found no rows for its own game: {out.get('reason')}"
    )
    assert out["checked"] == 1, (
        "the check looked at the wrong number of rows for this game"
    )
    assert out["scoped_by"] == "event_id"


def test_another_games_rows_are_not_checked_under_this_games_name(
    monkeypatch, tmp_path,
):
    """
    The counterpart failure. A job that cannot scope must not fall back to
    the whole board: it would withdraw other games' players at this game's
    tip and post the result naming this game.
    """
    monkeypatch.setenv("PROPIQ_BOARD_CSV", str(_live_board(tmp_path)))
    out = worker.run_prelock(event_id=EVENT, home="DAL", away="IND")
    assert out["checked"] == 1, "rows from the other game were swept in"


def test_a_game_with_no_recommendation_says_so_and_names_the_scope(
    monkeypatch, tmp_path,
):
    board = _live_board(tmp_path, rows=[(EVENT, "A. Player", "ABSTAIN")])
    monkeypatch.setenv("PROPIQ_BOARD_CSV", str(board))
    out = worker.run_prelock(event_id=EVENT, home="DAL", away="IND")

    assert out["status"] == "SKIPPED"
    assert out["reason"] == "no recommended rows"
    # The scope is reported, so "none for this game" can be told apart from
    # "could not tell which rows are this game's".
    assert out["scoped_by"] == "event_id"


def test_an_unscopeable_board_refuses_rather_than_checking_everything(
    monkeypatch, tmp_path,
):
    """
    No event match and no usable team column. The old code left `mine = frame`
    here, so the job would have scratch-checked EVERY game. An unscopeable
    check is a check that did not happen, and it has to say so.
    """
    monkeypatch.setenv("PROPIQ_BOARD_CSV", str(_live_board(tmp_path)))
    out = worker.run_prelock(event_id="an-id-on-no-row", home=None, away=None)

    assert out["status"] == "DATA_NOT_AVAILABLE"
    assert "scoped" in out["reason"]
    assert "checked" not in out, "an unscopeable job reported rows as checked"


def test_a_board_written_before_event_id_still_scopes_by_team(
    monkeypatch, tmp_path,
):
    """
    The team filter stays as a fallback, and the result says when it was what
    scoped the check -- a board without `event_id` is a board from an older
    writer, not a reason to check nothing.
    """
    frame = pd.DataFrame([{
        "slate_date": "2026-11-09", "game_date": "2026-11-09",
        "player_id": "2000001", "player_name": "A. Player",
        "player_team": "DAL", "opponent": "IND",
        "target_market": "PTS", "side": "over", "line": 24.5,
        "decision_status": "RECOMMENDED", "model_prob": 0.61,
    }])
    path = tmp_path / "decision_board.csv"
    frame.to_csv(path, index=False)
    monkeypatch.setenv("PROPIQ_BOARD_CSV", str(path))

    out = worker.run_prelock(event_id=EVENT, home="DAL", away="IND")
    assert out["status"] != "DATA_NOT_AVAILABLE"
    assert out["scoped_by"] == "player_team/opponent"
    assert out["checked"] == 1


def test_a_player_ruled_out_is_still_withdrawn_on_a_live_shaped_board(
    monkeypatch, tmp_path,
):
    """
    Scoping is only worth fixing if the withdrawal it feeds still works.
    """
    class _Injury:
        player_name, status, status_raw = "A. Player", "OUT", "Out (knee)"
        detail = "Out (knee)"

    monkeypatch.setattr(
        "src.ingestion.espn_availability.fetch_injuries",
        lambda **k: _Report([_Injury()]),
    )
    monkeypatch.setenv("PROPIQ_BOARD_CSV", str(_live_board(tmp_path)))
    out = worker.run_prelock(event_id=EVENT, home="DAL", away="IND")

    assert out["checked"] == 1
    assert out["withheld"] == 1, "the ruled-out player was not withdrawn"


# --- the send's own failure reason -------------------------------------


def test_a_failed_prelock_send_carries_the_webhook_reason(monkeypatch, tmp_path):
    """
    THE SEND SITE THE 2026-11-11 AUDIT CHANGED AND DID NOT TEST.

    `send_embeds` returns a result whose `reason` carries the HTTP status and
    the redacted response body, and this job returned only `sent.status`. A
    pre-lock card failing read as "FAILED" at the one moment a withdrawal
    matters most -- a player ruled out after the 09:00 slate, minutes before
    tip. `notify_failure` is deliberately not used on a dispatch path (see its
    docstring: alerting about a broken webhook through that webhook is a loop
    with no exit), so the return value is where the detail has to live.

    The audit's patch covered three send sites and tested one; two mutations
    survived its own test suite. This is one of the two.
    """
    monkeypatch.setenv("PROPIQ_BOARD_CSV", str(_live_board(tmp_path)))
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/1/x")
    monkeypatch.setenv("PROPIQ_PRELOCK", "true")
    monkeypatch.setattr(
        "src.ingestion.espn_availability.fetch_injuries",
        # ATTRIBUTES, NOT DICT KEYS: `scratches.apply_scratch_filter` reads
        # `row.player_name`, so a dict here fails before the send is reached.
        lambda **k: _Report(injuries=[
            SimpleNamespace(player_name="A. Player", status="OUT", team="DAL",
                            detail="ruled out", status_raw="OUT"),
        ]),
    )

    def fake_send(embeds, config=None, **kw):
        return type("R", (), {
            "status": "FAILED",
            "reason": "HTTP 429: rate limited",
            "http_status": 429,
            "embeds_sent": 0,
        })()

    import src.notify.discord as discord_mod

    monkeypatch.setattr(discord_mod, "send_embeds", fake_send)

    out = worker.run_prelock(event_id=EVENT, home="DAL", away="IND")

    assert out["status"] == "FAILED"
    assert out.get("reason") == "HTTP 429: rate limited", (
        f"the webhook's reason did not survive the return: {sorted(out)}"
    )
    assert out.get("http_status") == 429
    # AND THE SCOPE STILL TRAVELS, as every other return on this job carries it.
    assert out.get("scoped_by") == "event_id"
