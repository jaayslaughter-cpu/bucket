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

import sys

import pandas as pd
import pytest

from src.ingestion.inactive_players import (
    INACTIVE_COLUMNS,
    InactiveListError,
    attach_absence_features,
    fetch_inactive_players,
    fetch_many_inactive_players,
    load_cached_inactive_players,
    parse_inactive_players,
    resolve_team_abbreviations,
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
    out = attach_absence_features(_panel(), inactives, team_map=TEAM_MAP)

    lal = out[(out["GAME_ID"] == "21700548") & (out["TEAM_ABBREVIATION"] == "LAL")]
    gsw = out[(out["GAME_ID"] == "21700548") & (out["TEAM_ABBREVIATION"] == "GSW")]
    assert lal["BBS_TEAMMATES_OUT"].tolist() == [1, 1]
    assert gsw["BBS_TEAMMATES_OUT"].tolist() == [1, 1]
    assert (lal["BBS_INACTIVE_SOURCE"] == "official_inactive_list").all()


def test_a_game_absent_from_the_pull_is_unknown_not_zero():
    """"We did not fetch this game" and "nobody was out" are different
    statements, and only one is evidence."""
    inactives = parse_inactive_players(_v3_payload(game_id="0021700548"), "0021700548")
    out = attach_absence_features(_panel(), inactives, team_map=TEAM_MAP)

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
    out = attach_absence_features(_panel(), inactives, team_map=TEAM_MAP)

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
    out = attach_absence_features(_panel(), inactives, team_map=TEAM_MAP)
    assert out[out["TEAM_ABBREVIATION"] == "LAL"]["BBS_TEAMMATES_OUT"].tolist() == [3, 3]
    assert out[out["TEAM_ABBREVIATION"] == "GSW"]["BBS_TEAMMATES_OUT"].tolist() == [1, 1]


def test_a_panel_without_the_join_keys_is_refused():
    with pytest.raises(InactiveListError, match="missing"):
        attach_absence_features(pd.DataFrame({"PLAYER_ID": ["1"]}), pd.DataFrame())


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
    panel = attach_absence_features(_panel(), inactives, team_map=TEAM_MAP)
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
    out = attach_absence_features(panel, frame, team_map=TEAM_MAP)
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
    out = attach_absence_features(panel, mixed, team_map=TEAM_MAP)
    by_game = out.set_index("GAME_ID")
    assert by_game.loc["21700548", "BBS_TEAMMATES_OUT"] == 1   # the v3 game
    assert by_game.loc["21700777", "BBS_TEAMMATES_OUT"] == 1   # the v2 game
    assert (out["BBS_INACTIVE_SOURCE"] == "official_inactive_list").all()


def test_an_unrecognised_schema_is_refused_whether_or_not_it_has_rows():
    """
    The shape check sat INSIDE the non-empty branch, so a zero-row table with
    unrecognised columns became a coverage sentinel — recorded as a verified
    "nobody out", turning that game's counts into 0 — while the SAME broken
    schema carrying one row was refused. A schema break decided by row count
    converts a changed response into false evidence precisely when there is
    nothing to cross-check it against.
    """
    garbage = ["who", "what", "eh"]
    for rows in ([], [["a", "b", "c"]]):
        payload = {"InactivePlayers": {"headers": garbage, "data": rows}}
        with pytest.raises(InactiveListError, match="neither the v2 nor the v3 shape"):
            parse_inactive_players(payload, "0021700548")

    # And a RECOGNISED schema still sentinels when empty and parses when not.
    assert len(parse_inactive_players(_v3_payload(rows=[]), "0021700548")) == 1
    assert len(parse_inactive_players(_v3_payload(), "0021700548")) == 2


# --- vacated usage: the sum, not the count -------------------------------


def _usage_panel():
    """Star 900 and bench 901 build a usage history, then miss game 3.

    Player 902 has never appeared, so he has no prior usage to contribute.
    500/501 are the two who DID play game 3, i.e. the rows a prop model scores.
    """
    return pd.DataFrame({
        "GAME_ID": ["21700001", "21700001", "21700002", "21700002",
                    "21700003", "21700003"],
        "GAME_DATE": pd.to_datetime([
            "2018-01-01", "2018-01-01", "2018-01-03", "2018-01-03",
            "2018-01-05", "2018-01-05",
        ]),
        "TEAM_ABBREVIATION": ["LAL"] * 6,
        "PLAYER_ID": ["900", "901", "900", "901", "500", "501"],
        "USAGE_PROXY_L10": [0.30, 0.05, 0.28, 0.06, 0.20, 0.21],
    })


def _three_out_of_game_three():
    return parse_inactive_players(_v3_payload(game_id="0021700003", rows=[
        ["0021700003", 1610612747, 900, "Star", "Player", "1"],
        ["0021700003", 1610612747, 901, "Bench", "Guy", "2"],
        ["0021700003", 1610612747, 902, "Never", "Played", "3"],
    ]), "0021700003")


def test_vacated_usage_sums_each_absent_players_prior_level():
    """
    A count cannot tell a team missing 30% of its usage from one missing two
    end-of-bench players. The sum can, and it must take each absent player's
    level from his last game BEFORE the one he missed: 0.28 + 0.06 from game 2,
    not 0.30 + 0.05 from game 1.
    """
    out = attach_absence_features(
        _usage_panel(), _three_out_of_game_three(), team_map={"1610612747": "LAL"},
    )
    row = out[out["GAME_ID"] == "21700003"].iloc[0]

    assert row["BBS_TEAMMATES_OUT"] == 3
    assert row["BBS_VACATED_USAGE"] == pytest.approx(0.34)      # 0.28 + 0.06
    # 902 never played, so there is no prior level to add. Counted, not filled:
    # an invented default would put fabricated usage into the one feature whose
    # purpose is measuring what is missing.
    assert row["BBS_VACATED_USAGE_UNKNOWN"] == 1


def test_vacated_usage_never_reaches_forward_for_a_later_appearance():
    """The leakage property, stated as a difference. A player's usage AFTER the
    game he missed must not enter that game's sum — the as-of join is backward
    and excludes exact matches, so adding a huge later game changes nothing."""
    panel = _usage_panel()
    inactives = _three_out_of_game_three()
    before = attach_absence_features(panel, inactives, team_map={"1610612747": "LAL"})
    baseline = before[before["GAME_ID"] == "21700003"]["BBS_VACATED_USAGE"].iloc[0]

    # Player 900 returns two days later with a wildly higher usage.
    later = pd.concat([panel, pd.DataFrame({
        "GAME_ID": ["21700004"],
        "GAME_DATE": pd.to_datetime(["2018-01-07"]),
        "TEAM_ABBREVIATION": ["LAL"],
        "PLAYER_ID": ["900"],
        "USAGE_PROXY_L10": [0.99],
    })], ignore_index=True)
    after = attach_absence_features(later, inactives, team_map={"1610612747": "LAL"})
    assert after[after["GAME_ID"] == "21700003"]["BBS_VACATED_USAGE"].iloc[0] == (
        pytest.approx(baseline)
    )


def test_vacated_usage_is_unknown_when_the_usage_column_is_absent():
    """No usage column means the sum cannot be formed. It reports unknown rather
    than a zero that would read as "nobody important was out"."""
    panel = _usage_panel().drop(columns=["USAGE_PROXY_L10"])
    out = attach_absence_features(
        panel, _three_out_of_game_three(), team_map={"1610612747": "LAL"},
    )
    row = out[out["GAME_ID"] == "21700003"].iloc[0]
    assert row["BBS_TEAMMATES_OUT"] == 3            # the count still works
    assert row["BBS_VACATED_USAGE"] == 0.0          # nothing could be summed
    assert row["BBS_VACATED_USAGE_UNKNOWN"] == 3    # and all three say so


def test_a_game_never_pulled_has_unknown_vacated_usage_too():
    """The count and the sum must agree about coverage: an unfetched game is NA
    for both, never 0.0 for one of them."""
    out = attach_absence_features(
        _usage_panel(), _three_out_of_game_three(), team_map={"1610612747": "LAL"},
    )
    other = out[out["GAME_ID"] == "21700001"]
    assert other["BBS_TEAMMATES_OUT"].isna().all()
    assert other["BBS_VACATED_USAGE"].isna().all()
    assert (other["BBS_INACTIVE_SOURCE"] == "DATA_NOT_AVAILABLE").all()


# --- the layer that puts it in the pipeline -------------------------------


def test_the_absence_layer_is_registered_before_the_cascade_that_reads_it():
    """Order is load-bearing: teammate_cascade consumes BBS_TEAMMATES_OUT, so a
    layer producing it must run first. Registered the other way round, the
    cascade would abstain on every build no matter what was fetched."""
    from src.features.builder import _additive_feature_layers

    labels = [label for label, _ in _additive_feature_layers()]
    assert "absences" in labels, labels
    assert "teammate_cascade" in labels, labels
    assert labels.index("absences") < labels.index("teammate_cascade")


def test_the_layer_abstains_with_named_columns_when_no_cache_exists(tmp_path):
    """Until the pull has run there is nothing to join. The columns still appear
    — a MISSING column and a NULL column send the cascade down different paths,
    and only one of them means "no absence input exists at all"."""
    from src.features.absences import ABSENCE_COLUMNS, attach_absence_features_layer

    panel = _usage_panel()
    out = attach_absence_features_layer(panel, cache_root=tmp_path)
    for column in ABSENCE_COLUMNS:
        assert column in out.columns
        assert out[column].isna().all()
    assert (out["BBS_INACTIVE_SOURCE"] == "DATA_NOT_AVAILABLE").all()


def test_the_layer_uses_a_cache_once_one_exists(tmp_path):
    """And once the pull has run, the same layer produces real numbers from the
    parquet without any further wiring.

    The cached frame is the one fetch-inactives writes, abbreviations already
    resolved. That matters: v3 returns team ids only, and the layer has no team
    map to pass, so a cache of raw ids sends it to nba_api — absent from the
    optional 'stats' extra, hence absent in CI, where this abstained and
    returned nulls while passing locally on a machine that had the extra.
    """
    from src.features.absences import attach_absence_features_layer

    cached = resolve_team_abbreviations(
        _three_out_of_game_three(), {"1610612747": "LAL"}
    )
    save_inactive_players(cached, "2017-18", root=tmp_path)
    out = attach_absence_features_layer(_usage_panel(), cache_root=tmp_path)
    row = out[out["GAME_ID"] == "21700003"].iloc[0]
    assert row["BBS_TEAMMATES_OUT"] == 3
    assert row["BBS_VACATED_USAGE"] == pytest.approx(0.34)
    assert row["BBS_INACTIVE_SOURCE"] == "official_inactive_list"


def test_a_cache_written_by_the_pull_needs_no_nba_api(tmp_path, monkeypatch):
    """The pull machine has the optional 'stats' extra; the machine building
    features need not. Resolving abbreviations at write time is what makes the
    cache self-contained — without it this abstained and returned nulls on any
    box without nba_api, which is how CI read it.
    """
    from src.features.absences import attach_absence_features_layer

    cached = resolve_team_abbreviations(
        _three_out_of_game_three(), {"1610612747": "LAL"}
    )
    save_inactive_players(cached, "2017-18", root=tmp_path)

    # A None entry makes `from nba_api.stats.static import teams` raise
    # ImportError, exactly as a missing extra does.
    for name in ("nba_api", "nba_api.stats", "nba_api.stats.static"):
        monkeypatch.setitem(sys.modules, name, None)

    out = attach_absence_features_layer(_usage_panel(), cache_root=tmp_path)
    row = out[out["GAME_ID"] == "21700003"].iloc[0]
    assert row["BBS_TEAMMATES_OUT"] == 3
    assert row["BBS_VACATED_USAGE"] == pytest.approx(0.34)
    assert row["BBS_INACTIVE_SOURCE"] == "official_inactive_list"


def test_an_unresolved_cache_abstains_by_name_rather_than_reading_as_healthy():
    """The other half of the same contract: a cache that kept only team ids, on a
    machine with no way to map them, must say DATA_NOT_AVAILABLE — never 0 outs.
    """
    from src.features.absences import ABSENCE_COLUMNS

    saved = {k: sys.modules.get(k) for k in
             ("nba_api", "nba_api.stats", "nba_api.stats.static")}
    try:
        for name in saved:
            sys.modules[name] = None
        with pytest.raises(InactiveListError, match="nba_api is not installed"):
            attach_absence_features(_usage_panel(), _three_out_of_game_three())
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    assert ABSENCE_COLUMNS  # the layer above turns that refusal into null columns


def test_the_usage_proxy_layer_runs_before_the_absence_layer_that_sums_it():
    """BBS_VACATED_USAGE is a sum of USAGE_PROXY_L10, which src.features.
    sports_ev_features creates. Registered after the absence layer, the column
    did not exist when absences ran: every absent player's prior usage read as
    unknown and the sum was 0.0 on every covered row — a feature that was inert
    while reporting success.
    """
    from src.features.builder import _additive_feature_layers
    from src.ingestion.inactive_players import DEFAULT_USAGE_COLUMN

    labels = [label for label, _ in _additive_feature_layers()]
    assert DEFAULT_USAGE_COLUMN == "USAGE_PROXY_L10"
    assert "sports_ev" in labels, labels
    assert labels.index("sports_ev") < labels.index("absences"), labels
    assert labels.index("absences") < labels.index("teammate_cascade"), labels


def test_the_usage_column_really_comes_from_the_layer_registered_first():
    """Pins the dependency the ordering test asserts, so a rename of the column
    or a move of its producer cannot leave that test passing vacuously."""
    from src.features.sports_ev_features import attach_sports_ev_features
    from src.ingestion.inactive_players import DEFAULT_USAGE_COLUMN

    panel = pd.DataFrame({
        "GAME_ID": ["21700001", "21700002"],
        "GAME_DATE": pd.to_datetime(["2018-01-01", "2018-01-03"]),
        "SEASON": ["2017-18", "2017-18"],
        "TEAM_ABBREVIATION": ["LAL", "LAL"],
        "PLAYER_ID": ["900", "900"],
        "FGA": [20.0, 18.0], "FTA": [4.0, 2.0], "TOV": [3.0, 1.0],
    })
    assert DEFAULT_USAGE_COLUMN not in panel.columns
    assert DEFAULT_USAGE_COLUMN in attach_sports_ev_features(panel).columns


def test_an_absence_is_counted_even_when_the_panel_has_no_date_for_that_game():
    """A date is needed to look up PRIOR usage. It is not needed to COUNT an
    absence, and dropping undated rows lost them from both: a game the pull
    covered came back with zero teammates out, which is a fabricated healthy
    roster rather than an abstention.
    """
    panel = _usage_panel()
    panel.loc[panel["GAME_ID"] == "21700003", "GAME_DATE"] = pd.NaT

    out = attach_absence_features(
        panel, _three_out_of_game_three(), team_map={"1610612747": "LAL"},
    )
    row = out[out["GAME_ID"] == "21700003"].iloc[0]
    assert row["BBS_TEAMMATES_OUT"] == 3
    assert row["BBS_VACATED_USAGE_UNKNOWN"] == 3   # no date, so no prior level
    assert row["BBS_VACATED_USAGE"] == 0.0
    assert row["BBS_INACTIVE_SOURCE"] == "official_inactive_list"


def test_one_game_written_twice_with_different_padding_stays_one_row(tmp_path):
    """Two cache files, one holding '0021700003' and one whose ids were read back
    as integers, are the same game. Deduped before padding they both survived,
    then collapsed to a single join key and duplicated every panel row for that
    game.
    """
    from src.features.absences import load_cached_absences

    padded = resolve_team_abbreviations(
        _three_out_of_game_three(), {"1610612747": "LAL"}
    )
    unpadded = padded.copy()
    unpadded["GAME_ID"] = unpadded["GAME_ID"].str.lstrip("0")
    save_inactive_players(padded, "2017-18", root=tmp_path)
    save_inactive_players(unpadded, "2017-18-int", root=tmp_path)

    combined = load_cached_absences(tmp_path)
    assert len(combined) == 3, combined[["GAME_ID", "PLAYER_ID"]]
    assert set(combined["GAME_ID"]) == {"0021700003"}

    panel = _usage_panel()
    out = attach_absence_features(panel, combined, team_map={"1610612747": "LAL"})
    assert len(out) == len(panel)
    row = out[out["GAME_ID"] == "21700003"].iloc[0]
    assert row["BBS_TEAMMATES_OUT"] == 3


def test_the_cache_directory_can_be_relocated_by_environment(tmp_path, monkeypatch):
    """Where the cache lives is a setting, not a code edit — the pull and the
    feature build can run on different machines with different layouts."""
    from src.features.absences import ENV_CACHE_DIR, load_cached_absences

    assert load_cached_absences() is None          # the isolated default is empty
    save_inactive_players(
        resolve_team_abbreviations(_three_out_of_game_three(), {"1610612747": "LAL"}),
        "2017-18", root=tmp_path,
    )
    monkeypatch.setenv(ENV_CACHE_DIR, str(tmp_path))
    found = load_cached_absences()
    assert found is not None and len(found) == 3


def test_the_cli_writes_a_cache_with_abbreviations_already_resolved(
    tmp_path, monkeypatch
):
    """The claim the test above depends on, checked on the writer rather than
    assumed. fetch-inactives runs where nba_api is installed; the feature build
    need not. If the command wrote raw team ids, the layer would abstain on any
    machine without the optional extra and the cache would be useless there.
    """
    pytest.importorskip("typer")
    from typer.testing import CliRunner

    import src.ingestion.inactive_players as module
    from scripts.nba_model_cli import app

    monkeypatch.chdir(tmp_path)
    panel_path = tmp_path / "panel.parquet"
    pd.DataFrame({
        "GAME_ID": ["0021700003"],
        "GAME_DATE": pd.to_datetime(["2018-01-05"]),
        "SEASON": ["2017-18"],
    }).to_parquet(panel_path, index=False)

    # v3 shape: team ids, no abbreviation — what the real endpoint returns.
    fetched = _three_out_of_game_three()
    assert "TEAM_ABBREVIATION" not in fetched.columns
    monkeypatch.setattr(
        module, "fetch_many_inactive_players",
        lambda ids, **kwargs: (fetched.copy(), []),
    )
    # A fixed map stands in for nba_api's static table, so this needs no extra.
    monkeypatch.setattr(
        module, "team_id_to_abbreviation", lambda teams=None: {"1610612747": "LAL"}
    )

    out_path = tmp_path / "inactive.parquet"
    result = CliRunner().invoke(
        app, ["fetch-inactives", "--panel", str(panel_path), "--out", str(out_path)]
    )
    assert result.exit_code == 0, result.stdout
    written = pd.read_parquet(out_path)
    assert "TEAM_ABBREVIATION" in written.columns
    assert set(written["TEAM_ABBREVIATION"]) == {"LAL"}
