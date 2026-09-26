"""
Tests for src/ingestion/inactive_players.py.

This is the input src/features/teammate_cascade.py has been abstaining for want
of, so the load-bearing properties are the two that decide whether the layer
comes on with TRUE information:

  * the join actually matches — the panel stores GAME_ID unpadded and the NBA
    returns it zero-padded to ten, so a naive merge matches nothing at all;
  * "not fetched" stays distinguishable from "nobody was out", because only one
    of those is evidence.

Everything runs offline: stats.nba.com is denied at this environment's proxy, so
the fetch is injected rather than performed.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.ingestion.inactive_players import (
    INACTIVE_COLUMNS,
    InactiveListError,
    attach_teammate_out_counts,
    fetch_inactive_players,
    fetch_many_inactive_players,
    load_cached_inactive_players,
    parse_inactive_players,
    save_inactive_players,
    team_id_to_abbreviation,
)

TEAM_MAP = {"1610612747": "LAL", "1610612744": "GSW", "1610612738": "BOS"}


def _v3_payload(game_id="0021700548", rows=None):
    """The v3 data-set shape nba_api's get_data_sets produces."""
    if rows is None:
        rows = [
            [game_id, 1610612747, 201566, "Russell", "Westbrook", "0"],
            [game_id, 1610612744, 201939, "Stephen", "Curry", "30"],
        ]
    return {
        "GameSummary": {"headers": ["GAME_ID"], "data": [[game_id]]},
        "InactivePlayers": {
            "headers": [
                "gameId", "teamId", "personId", "firstName", "familyName", "jerseyNum",
            ],
            "data": rows,
        },
    }


def _v2_payload(rows=None):
    if rows is None:
        rows = [[201566, "Russell", "Westbrook", "0", 1610612747, "Los Angeles", "Lakers", "LAL"]]
    return {
        "InactivePlayers": {
            "headers": [
                "PLAYER_ID", "FIRST_NAME", "LAST_NAME", "JERSEY_NUM",
                "TEAM_ID", "TEAM_CITY", "TEAM_NAME", "TEAM_ABBREVIATION",
            ],
            "data": rows,
        },
    }


# --- parsing both endpoint versions --------------------------------------


def test_the_v3_shape_parses_to_the_normalised_schema():
    frame = parse_inactive_players(_v3_payload(), "0021700548")
    assert list(frame.columns)[: len(INACTIVE_COLUMNS)] == list(INACTIVE_COLUMNS)
    assert len(frame) == 2
    assert set(frame["PLAYER_NAME"]) == {"Russell Westbrook", "Stephen Curry"}
    assert frame["GAME_ID"].tolist() == ["0021700548"] * 2
    # Ids are keys. As integers they lose the padding a join needs.
    assert frame["PLAYER_ID"].dtype == "string"
    assert frame["TEAM_ID"].dtype == "string"


def test_the_v2_shape_parses_too_and_keeps_its_abbreviation():
    """v2 is the fallback, not the default — the library warns its data is
    missing for games on or after 2025-04-10 — but it must still parse, and it
    carries TEAM_ABBREVIATION directly so no team map is needed."""
    frame = parse_inactive_players(_v2_payload(), 21700548)
    assert frame["PLAYER_NAME"].tolist() == ["Russell Westbrook"]
    assert frame["TEAM_ABBREVIATION"].tolist() == ["LAL"]
    # The id we asked for wins: v2 rows carry no gameId of their own.
    assert frame["GAME_ID"].tolist() == ["0021700548"]


def test_an_empty_inactive_list_keeps_its_game_as_a_coverage_sentinel():
    """A game where everyone dressed is a fact, and it must not raise. It must
    also not return ZERO rows: the game would then vanish from the concatenated
    frame, and a verified "nobody out" would be indistinguishable from a game
    nobody pulled. One sentinel row carries the coverage, with no player on it.
    """
    frame = parse_inactive_players(_v3_payload(rows=[]), "0021700548")
    assert len(frame) == 1
    assert list(frame.columns) == list(INACTIVE_COLUMNS)
    assert frame["GAME_ID"].tolist() == ["0021700548"]
    # No player, so nothing can count it as an absence.
    assert frame["PLAYER_ID"].isna().all()


def test_a_missing_data_set_raises_rather_than_reading_as_nobody_out():
    """An absent InactivePlayers field means the response did not contain it.
    Treating that as an empty list would make a broken fetch indistinguishable
    from a healthy roster."""
    with pytest.raises(InactiveListError, match="no InactivePlayers data set"):
        parse_inactive_players({"GameSummary": {"headers": ["X"], "data": []}}, "1")


def test_an_unrecognised_column_shape_is_refused():
    payload = {"InactivePlayers": {"headers": ["who", "what"], "data": [["a", "b"]]}}
    with pytest.raises(InactiveListError, match="neither the v2 nor the v3 shape"):
        parse_inactive_players(payload, "1")


# --- fetching -------------------------------------------------------------


def test_v2_is_tried_when_v3_has_nothing_to_give():
    """v3 first by preference; v2 only if v3 fails. Pins the order, since
    defaulting to v2 silently loses every game after 2025-04-10."""
    seen: list[str] = []

    def _fetch(game_id, endpoint):
        seen.append(endpoint)
        if endpoint == "boxscoresummaryv3":
            raise RuntimeError("v3 exploded")
        return _v2_payload()

    frame = fetch_inactive_players("0021700548", fetch=_fetch)
    assert seen == ["boxscoresummaryv3", "boxscoresummaryv2"]
    assert len(frame) == 1


def test_every_endpoint_failing_names_both_in_the_error():
    def _fetch(game_id, endpoint):
        raise RuntimeError(f"{endpoint} down")

    with pytest.raises(InactiveListError, match="boxscoresummaryv3.*boxscoresummaryv2"):
        fetch_inactive_players("1", fetch=_fetch)


def test_failures_are_returned_not_swallowed():
    """A caller that asked for three games and got two must be able to tell, or
    a partial pull becomes a season where some games had nobody injured."""
    def _fetch(game_id, endpoint):
        if game_id == "0021700002":
            raise RuntimeError("nope")
        return _v3_payload(game_id=game_id)

    frame, failures = fetch_many_inactive_players(
        ["0021700001", "0021700002", "0021700003"], fetch=_fetch, pause_seconds=0,
    )
    assert frame["GAME_ID"].nunique() == 2
    assert len(failures) == 1
    assert failures[0]["game_id"] == "0021700002"


def test_every_game_failing_raises():
    def _fetch(game_id, endpoint):
        raise RuntimeError("all down")

    with pytest.raises(InactiveListError, match="no inactive list could be fetched"):
        fetch_many_inactive_players(["1", "2"], fetch=_fetch, pause_seconds=0)


def test_overlapping_pulls_do_not_double_count():
    def _fetch(game_id, endpoint):
        return _v3_payload(game_id=game_id)

    frame, _ = fetch_many_inactive_players(
        ["0021700001", "0021700001"], fetch=_fetch, pause_seconds=0,
    )
    assert len(frame) == 2  # two inactive players, one game, not four rows


def test_the_cache_round_trips_ids_as_text(tmp_path):
    """Written as integers, TEAM_ID and a padded GAME_ID come back having lost
    their form — the same defect the parlay log's CSV reader had."""
    frame = parse_inactive_players(_v3_payload(), "0021700548")
    save_inactive_players(frame, "2017-18", root=tmp_path)
    back = load_cached_inactive_players("2017-18", root=tmp_path)
    assert back is not None
    assert back["GAME_ID"].tolist() == ["0021700548"] * 2
    assert back["GAME_ID"].dtype == "string"
    assert back["PLAYER_ID"].dtype == "string"
    assert load_cached_inactive_players("1999-00", root=tmp_path) is None


# --- the join, which is where this earns its place ------------------------


def _panel():
    """Two teams, one game, appearances only — as the real panel is."""
    return pd.DataFrame({
        # UNPADDED, exactly as data/external/training_pack/panel.parquet stores it
        "GAME_ID": ["21700548"] * 4 + ["21700999"] * 2,
        "TEAM_ABBREVIATION": ["LAL", "LAL", "GSW", "GSW", "BOS", "BOS"],
        "PLAYER_ID": ["1", "2", "3", "4", "5", "6"],
        "PTS": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    })


def test_the_join_survives_the_game_id_padding_mismatch():
    """The panel stores GAME_ID as '21700548' and the NBA returns '0021700548'.
    Merged as-is, NOTHING matches and the count is null on every row — the layer
    would keep abstaining while holding the data.
    """
    inactives = parse_inactive_players(_v3_payload(game_id="0021700548"), "0021700548")
    out = attach_teammate_out_counts(_panel(), inactives, team_map=TEAM_MAP)

    lal = out[(out["GAME_ID"] == "21700548") & (out["TEAM_ABBREVIATION"] == "LAL")]
    gsw = out[(out["GAME_ID"] == "21700548") & (out["TEAM_ABBREVIATION"] == "GSW")]
    assert lal["BBS_TEAMMATES_OUT"].tolist() == [1, 1]
    assert gsw["BBS_TEAMMATES_OUT"].tolist() == [1, 1]
    assert (lal["BBS_INACTIVE_SOURCE"] == "official_inactive_list").all()


def test_a_game_absent_from_the_pull_is_unknown_not_zero():
    """"We did not fetch this game" and "nobody was out" are different
    statements, and only one is evidence."""
    inactives = parse_inactive_players(_v3_payload(game_id="0021700548"), "0021700548")
    out = attach_teammate_out_counts(_panel(), inactives, team_map=TEAM_MAP)

    other = out[out["GAME_ID"] == "21700999"]
    assert other["BBS_TEAMMATES_OUT"].isna().all()
    assert (other["BBS_INACTIVE_SOURCE"] == "DATA_NOT_AVAILABLE").all()


def test_a_fetched_game_with_nobody_out_is_a_real_zero():
    """Distinguished from the case above: this game WAS pulled, and the honest
    answer is 0, not unknown."""
    pulled = parse_inactive_players(_v3_payload(game_id="0021700548"), "0021700548")
    # BOS played in 21700999, which the pull covers, but no BOS player was out.
    also_pulled = parse_inactive_players(
        _v3_payload(game_id="0021700999",
                    rows=[["0021700999", 1610612744, 201939, "Stephen", "Curry", "30"]]),
        "0021700999",
    )
    inactives = pd.concat([pulled, also_pulled], ignore_index=True)
    out = attach_teammate_out_counts(_panel(), inactives, team_map=TEAM_MAP)

    bos = out[out["TEAM_ABBREVIATION"] == "BOS"]
    assert bos["BBS_TEAMMATES_OUT"].tolist() == [0, 0]
    assert (bos["BBS_INACTIVE_SOURCE"] == "official_inactive_list").all()


def test_the_count_is_per_team_not_per_game():
    """Three Lakers out and one Warrior out is not "four out" for everyone."""
    rows = [
        ["0021700548", 1610612747, 1001, "A", "One", "1"],
        ["0021700548", 1610612747, 1002, "B", "Two", "2"],
        ["0021700548", 1610612747, 1003, "C", "Three", "3"],
        ["0021700548", 1610612744, 2001, "D", "Four", "4"],
    ]
    inactives = parse_inactive_players(_v3_payload(rows=rows), "0021700548")
    out = attach_teammate_out_counts(_panel(), inactives, team_map=TEAM_MAP)
    assert out[out["TEAM_ABBREVIATION"] == "LAL"]["BBS_TEAMMATES_OUT"].tolist() == [3, 3]
    assert out[out["TEAM_ABBREVIATION"] == "GSW"]["BBS_TEAMMATES_OUT"].tolist() == [1, 1]


def test_a_panel_without_the_join_keys_is_refused():
    with pytest.raises(InactiveListError, match="missing"):
        attach_teammate_out_counts(pd.DataFrame({"PLAYER_ID": ["1"]}), pd.DataFrame())


def test_the_team_map_is_injectable_so_it_needs_no_nba_api():
    mapped = team_id_to_abbreviation(
        [{"id": 1610612747, "abbreviation": "LAL"}, {"id": None, "abbreviation": "X"}]
    )
    assert mapped == {"1610612747": "LAL"}


# --- and the point of all of it: the cascade layer comes on ---------------


def test_the_cascade_layer_stops_abstaining_once_the_counts_are_attached():
    """
    The whole purpose. teammate_cascade abstains on all 214,381 panel rows for
    want of an absence input, and a per-row BBS_OUT_FLAG cannot supply it: the
    panel holds only players who APPEARED, so such a flag is 0 everywhere and
    the layer's team_outs - flag arithmetic yields zero teammates out.
    """
    from src.features.teammate_cascade import attach_teammate_cascade_stub

    inactives = parse_inactive_players(_v3_payload(game_id="0021700548"), "0021700548")
    panel = attach_teammate_out_counts(_panel(), inactives, team_map=TEAM_MAP)
    out = attach_teammate_cascade_stub(panel)

    played = out[out["GAME_ID"] == "21700548"]
    assert (played["CASCADE_TEAMMATE_OUTS"] == 1).all()
    assert (played["CASCADE_STATUS"] == "NEEDS_VERIFIED_PAIRWISE").all()

    # The unpulled game still abstains, and says why.
    unpulled = out[out["GAME_ID"] == "21700999"]
    assert (unpulled["CASCADE_STATUS"] == "DATA_NOT_AVAILABLE").all()
    assert unpulled["CASCADE_NOTES"].str.contains("unknown, not absent").all()

    # And no usage multiplier is invented anywhere.
    assert out["CASCADE_USAGE_MULT"].isna().all()


def test_a_per_row_flag_panel_of_appearances_only_still_abstains():
    """Pins the reason the counted path exists. With BBS_OUT_FLAG alone on a
    panel of appearances, every flag is 0 and the layer sees no absences —
    which is what it did before this change."""
    from src.features.teammate_cascade import attach_teammate_cascade_stub

    panel = _panel()
    panel["BBS_OUT_FLAG"] = 0  # true of every row: they all played
    out = attach_teammate_cascade_stub(panel)
    assert (out["CASCADE_TEAMMATE_OUTS"] == 0).all()
    assert (out["CASCADE_STATUS"] == "NO_TEAMMATE_OUTS").all()


# --- the two cases my first pass only appeared to cover --------------------


def test_a_game_whose_whole_inactive_list_is_empty_is_zero_not_unknown():
    """
    The case test_a_fetched_game_with_nobody_out_is_a_real_zero did NOT cover.
    That one gave the game an inactive row for the OTHER team, so the game was
    present in the frame and the merge produced 0 for the team without outs. It
    passed while this was broken.

    Here the game's inactive list is empty ENTIRELY. Before the sentinel row it
    contributed nothing to the frame, so a game fetched successfully with nobody
    out came back as DATA_NOT_AVAILABLE — destroying the one distinction this
    module is built around.
    """
    def _fetch(game_id, endpoint):
        if game_id == "0021700777":
            return _v3_payload(rows=[])          # nobody out, fetched fine
        return _v3_payload(game_id=game_id)

    frame, failures = fetch_many_inactive_players(
        ["0021700548", "0021700777"], fetch=_fetch, pause_seconds=0,
    )
    assert not failures
    assert set(frame["GAME_ID"]) == {"0021700548", "0021700777"}

    panel = pd.DataFrame({
        "GAME_ID": ["21700548", "21700777", "21799999"],
        "TEAM_ABBREVIATION": ["LAL", "GSW", "BOS"],
        "PLAYER_ID": ["1", "2", "3"],
    })
    out = attach_teammate_out_counts(panel, frame, team_map=TEAM_MAP)
    by_game = out.set_index("GAME_ID")

    # Fetched, nobody out -> a real 0.
    assert by_game.loc["21700777", "BBS_TEAMMATES_OUT"] == 0
    assert by_game.loc["21700777", "BBS_INACTIVE_SOURCE"] == "official_inactive_list"
    # Fetched, somebody out -> counted.
    assert by_game.loc["21700548", "BBS_TEAMMATES_OUT"] == 1
    # Never fetched -> still unknown. The sentinel must not make everything 0.
    assert pd.isna(by_game.loc["21799999", "BBS_TEAMMATES_OUT"])
    assert by_game.loc["21799999", "BBS_INACTIVE_SOURCE"] == "DATA_NOT_AVAILABLE"


def test_v3_rows_keep_their_team_when_mixed_with_v2_fallback_rows():
    """
    fetch_many_inactive_players mixes endpoint versions across games — v3 for
    most, v2 where v3 failed. v3 rows carry only teamId; v2 rows carry the
    abbreviation. A frame holding both is neither "column absent" nor "all
    null", so an all-or-nothing mapping test skipped the mapping entirely: every
    v3 row lost its team, was dropped from the count, and its game reported
    DATA_NOT_AVAILABLE — a game we fetched, where a player WAS out, reading as
    unknown.
    """
    v3 = parse_inactive_players(
        _v3_payload(game_id="0021700548",
                    rows=[["0021700548", 1610612747, 201566, "Russell", "Westbrook", "0"]]),
        "0021700548",
    )
    v2 = parse_inactive_players(
        _v2_payload(rows=[[201939, "Stephen", "Curry", "30", 1610612744,
                           "San Francisco", "Warriors", "GSW"]]),
        "0021700777",
    )
    mixed = pd.concat([v3, v2], ignore_index=True)
    # The precondition that defeats an all-or-nothing check.
    assert mixed["TEAM_ABBREVIATION"].isna().any()
    assert mixed["TEAM_ABBREVIATION"].notna().any()

    panel = pd.DataFrame({
        "GAME_ID": ["21700548", "21700777"],
        "TEAM_ABBREVIATION": ["LAL", "GSW"],
        "PLAYER_ID": ["1", "2"],
    })
    out = attach_teammate_out_counts(panel, mixed, team_map=TEAM_MAP)
    by_game = out.set_index("GAME_ID")
    assert by_game.loc["21700548", "BBS_TEAMMATES_OUT"] == 1   # the v3 game
    assert by_game.loc["21700777", "BBS_TEAMMATES_OUT"] == 1   # the v2 game
    assert (out["BBS_INACTIVE_SOURCE"] == "official_inactive_list").all()
