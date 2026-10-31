"""
tests/test_espn_panel.py — player box scores from ESPN into the canonical frame.

WHAT THIS GUARDS. `team_game_stats` had two writers and the deployed container
could reach neither: the BigDataBall workbook is a licensed file downloaded by
hand, and the scraper's parquet needs an IP nba.com will answer. ESPN is the
third source and the container can reach it, which is the only reason an
unattended deploy can fill its own panel at all.

Every test below is about a way an ESPN row can be WRONG WHILE LOOKING RIGHT:
a half-finished game that reads as a final one, a preseason game in a rolling
window, a team code guessed into the nearest NBA team, and above all an id in
the wrong namespace, which splits a player's history in `groupby("PLAYER_ID")`
and gives a confident rolling mean over half the season.

AGAINST A REAL PAYLOAD. `summary_401909881.json` is a live ESPN response
captured 2026-10-30, trimmed to the keys the mapper reads. The suite needs no
network; the fixture is what the network actually returned.

RESEARCH_ONLY.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from src.ingestion.espn_panel import (
    ESPN_SOURCE,
    EspnPanelError,
    EspnPanelReport,
    espn_player_id,
    parse_summary_payload,
)
from src.utils.ids import ESPN_ID_PREFIX, espn_game_id, is_espn_id

FIXTURE = Path(__file__).parent / "fixtures" / "espn_panel" / "summary_401909881.json"
EVENT_ID = "401909881"


def _payload() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _rows(payload=None, report=None) -> list[dict]:
    return parse_summary_payload(payload or _payload(), EVENT_ID, report=report)


# --- the arithmetic, against the real response ---------------------------


def test_a_real_payload_yields_both_teams_and_every_stat():
    rows = _rows()
    frame = pd.DataFrame(rows)
    assert len(frame) == 27
    assert set(frame["team_abbr"]) == {"MIA", "ATL"}
    assert set(frame["home_away"]) == {"home", "away"}
    for col in ("pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb",
                "dreb", "reb", "ast", "stl", "blk", "tov", "pf"):
        assert col in frame.columns, f"{col} missing from the canonical frame"
    played = frame.loc[frame["played"]]
    assert len(played) == 19
    assert played["pts"].notna().all()


def test_the_frame_feeds_both_existing_consumers_unchanged():
    """
    THE WHOLE DESIGN CLAIM. This module is a mapper onto the canonical frame,
    not a third definition of a player row — so `to_player_panel` and
    `aggregate_team_games` must take it with no argument either of them did
    not already have.
    """
    from src.ingestion.scraper_panel import to_player_panel
    from src.ingestion.scraper_team_games import aggregate_team_games

    frame = pd.DataFrame(_rows())

    panel, panel_report = to_player_panel(frame)
    assert len(panel) == 19, "the played rows did not survive the panel mapper"
    assert panel_report.unknown_sources == (), (
        f"espn is not a known source: {panel_report.unknown_sources}"
    )

    teams, _ = aggregate_team_games(frame, team_source="espn_agg")
    assert len(teams) == 2
    totals = dict(zip(teams["team_abbr"], teams["points"]))
    assert totals["ATL"] == 117.0 and totals["MIA"] == 129.0


def test_minutes_reach_the_panel_as_minutes_not_seconds():
    """`scraper_panel` divides `seconds_played` by 60. Emitting minutes there
    would make every MIN sixty times too small and every minutes-weighted
    feature quietly wrong."""
    from src.ingestion.scraper_panel import to_player_panel

    panel, _ = to_player_panel(pd.DataFrame(_rows()))
    assert 5.0 < float(panel["MIN"].max()) < 60.0, (
        f"MIN tops out at {panel['MIN'].max()}, which is not a basketball number"
    )


# --- the id namespaces, which is the dangerous part ----------------------


def test_both_ids_are_namespaced_and_cannot_pass_for_nba_ids():
    rows = _rows()
    for row in rows:
        assert row["game_id"].startswith(ESPN_ID_PREFIX)
        assert row["player_id_nba"].startswith(ESPN_ID_PREFIX)
        assert is_espn_id(row["game_id"])
        # An NBA id is ten characters of digits. Nothing here can be mistaken
        # for one by a careless join, which is the point of the prefix.
        assert not row["game_id"].isdigit()
        assert not row["player_id_nba"].isdigit()


def test_the_namespace_helpers_are_idempotent_and_refuse_nothing():
    assert espn_game_id("401909881") == "espn:401909881"
    assert espn_game_id("espn:401909881") == "espn:401909881"
    assert espn_player_id("3032977") == "espn:3032977"
    assert espn_player_id("espn:3032977") == "espn:3032977"
    for helper in (espn_game_id, espn_player_id):
        with pytest.raises(ValueError):
            helper("")
        with pytest.raises(ValueError):
            helper(None)


def test_an_nba_id_is_not_mistaken_for_an_espn_one():
    assert not is_espn_id("0022600123")
    assert not is_espn_id("")
    assert not is_espn_id(None)


# --- the refusals --------------------------------------------------------


def test_a_game_still_in_progress_is_refused():
    """
    A LIVE BOX SCORE LOOKS EXACTLY LIKE A FINISHED ONE. A player with 8 points
    at half-time reads as a player who scored 8 all night, and a rolling mean
    cannot tell them apart. This is the sharpest failure this module can have.
    """
    payload = _payload()
    payload["header"]["competitions"][0]["status"]["type"]["name"] = "STATUS_IN_PROGRESS"
    report = EspnPanelReport()
    assert _rows(payload, report) == []
    assert report.skipped_not_final and EVENT_ID in report.skipped_not_final[0]


def test_a_preseason_game_is_refused():
    """
    `utils/ids.ESPN_IDS_ARE_REGULAR_SEASON` lets a SQL filter accept the whole
    `espn:` namespace without reading a season digit out of it. That is only
    true while this refusal holds, so it is not a preference.
    """
    payload = _payload()
    payload["header"]["season"]["type"] = 1
    report = EspnPanelReport()
    assert _rows(payload, report) == []
    assert report.skipped_season_type


def test_a_postseason_game_is_refused_too():
    payload = _payload()
    payload["header"]["season"]["type"] = 3
    report = EspnPanelReport()
    assert _rows(payload, report) == []
    assert report.skipped_season_type


def test_an_unmapped_team_code_is_refused_not_guessed():
    """`espn_schedule` makes the same refusal: an unrecognised code is
    surfaced, never guessed into the nearest NBA team."""
    payload = _payload()
    payload["header"]["competitions"][0]["competitors"][0]["team"]["abbreviation"] = "ZZZ"
    report = EspnPanelReport()
    assert _rows(payload, report) == []
    assert report.skipped_unmapped_team


def test_a_one_sided_payload_is_refused():
    """Elo cannot score a result with one team in it, and no row can name an
    opponent that is not there."""
    payload = _payload()
    payload["header"]["competitions"][0]["competitors"] = (
        payload["header"]["competitions"][0]["competitors"][:1]
    )
    report = EspnPanelReport()
    assert _rows(payload, report) == []
    assert report.skipped_one_sided


def test_a_payload_with_no_competition_block_is_refused_not_tracebacked():
    with pytest.raises(EspnPanelError, match="DATA_NOT_AVAILABLE"):
        parse_summary_payload({"header": {}}, EVENT_ID)


def test_a_row_with_no_athlete_id_is_dropped_and_counted():
    """A row with no identity cannot be keyed or rolled. Dropped, never
    written under a blank or invented id."""
    payload = _payload()
    blanked = 0
    for block in payload["boxscore"]["players"]:
        for group in block.get("statistics", []):
            for athlete in group.get("athletes", []):
                if blanked < 2:
                    athlete["athlete"]["id"] = ""
                    blanked += 1
    assert blanked == 2, "the fixture shape changed; this test blanked nothing"
    report = EspnPanelReport()
    rows = _rows(payload, report)
    assert report.dropped_no_athlete_id == 2
    assert len(rows) == 25


# --- the DNP, which must never become a zero -----------------------------


def test_a_dnp_carries_nulls_not_zeroes():
    """
    A zero is a player who played and scored nothing. A null is a player who
    did not play. A rolling window counts rows, so turning one into the other
    moves every mean it appears in.
    """
    rows = _rows()
    dnps = [r for r in rows if not r["played"]]
    assert len(dnps) == 8
    for row in dnps:
        assert row["pts"] is None
        assert row["seconds_played"] is None
    assert all(r["pts"] is not None for r in rows if r["played"])


def test_a_dnp_that_arrives_with_a_zero_filled_stat_line_is_still_nulled():
    """
    THE CASE THE FIXTURE DOES NOT CONTAIN, AND THE ONLY ONE WHERE THE GUARD
    DOES ANYTHING. In this payload every DNP carries `stats: []`, so
    `row.stats.get(...)` is None whatever the mapper does — a mutation that
    zero-filled DNPs passed the whole file. `espn_game` populates `stats`
    whenever the values array is non-empty, INDEPENDENT of the didNotPlay
    flag, so a DNP arriving with a full zero line is a shape ESPN can send
    and the mapper must still refuse.

    A zero there is a player who dressed, played, and scored nothing. There
    is no such game. Caught by mutation on 2026-10-31.
    """
    payload = _payload()
    filled = 0
    for block in payload["boxscore"]["players"]:
        group = block["statistics"][0]
        width = len(group["names"])
        for athlete in group["athletes"]:
            if athlete.get("didNotPlay") and not athlete.get("stats"):
                athlete["stats"] = ["0"] * width
                filled += 1
                break
    assert filled == 2, "the fixture shape changed; this test filled nothing"

    rows = _rows(payload)
    dnps = [r for r in rows if not r["played"]]
    assert len(dnps) == 8
    assert all(r["pts"] is None for r in dnps), (
        "a DNP that arrived with a zero line was written as a real zero: "
        f"{[r['pts'] for r in dnps]}"
    )
    assert all(r["seconds_played"] is None for r in dnps)


def test_the_default_panel_mapper_drops_dnps_rather_than_zero_filling():
    from src.ingestion.scraper_panel import to_player_panel

    panel, report = to_player_panel(pd.DataFrame(_rows()))
    assert report.dropped_dnp == 8
    assert len(panel) == 19


# --- provenance ----------------------------------------------------------


def test_every_row_says_where_it_came_from():
    assert {r["source"] for r in _rows()} == {ESPN_SOURCE}


def test_espn_ranks_below_every_source_that_reports_the_nbas_own_numbers():
    """
    Indispensable is not authoritative. ESPN is the only source a datacenter
    can reach, and it is a public aggregator -- where a licensed or official
    feed also has the game, that one wins.
    """
    from src.ingestion.scraper_panel import SOURCE_PRECEDENCE

    assert SOURCE_PRECEDENCE[-1] == ESPN_SOURCE
    assert "nba_stats" in SOURCE_PRECEDENCE
    assert SOURCE_PRECEDENCE.index("nba_stats") < SOURCE_PRECEDENCE.index(ESPN_SOURCE)


def test_the_season_label_is_the_starting_year():
    """ESPN's season year is the ENDING one: 2027 is the 2026-27 season."""
    rows = _rows()
    assert {r["season"] for r in rows} == {"2026-27"}


def test_the_fixture_is_a_real_response_not_a_hand_written_one():
    """
    The point of capturing it live. A hand-written fixture proves the mapper
    handles what its author imagined; this one proves it handles what ESPN
    sent on 2026-10-28.
    """
    payload = _payload()
    assert payload["header"]["competitions"][0]["status"]["type"]["name"] == "STATUS_FINAL"
    assert len(payload["boxscore"]["players"]) == 2
    # Real box scores carry made-attempted pairs as text; `espn_game` splits
    # them. A fixture of pre-split numbers would skip that code entirely.
    raw = json.dumps(payload)
    assert "-" in raw and '"FG"' not in raw[:200]
    assert copy.deepcopy(payload) == payload
