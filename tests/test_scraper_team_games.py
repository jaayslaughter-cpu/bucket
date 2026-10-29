"""
tests/test_scraper_team_games.py — team-game rows summed from player rows.

WHAT THIS GUARDS. `team_game_stats` had one writer, the licensed BigDataBall
workbook, which is why the only manual step in this project exists and why the
slates of 2026-10-26, 10-27 and 10-28 each died at step [2]. A team's box score
is the sum of its players', so the frame can be built from data the scraper is
already bringing in.

The arithmetic is the easy part. Every test below is about a way the sum can be
WRONG WHILE LOOKING RIGHT, because that is the whole risk here: a team row is
fifteen plausible integers, and nothing downstream can tell a real one from a
total summed over four ingested players or over a "Team Totals" line.

RESEARCH_ONLY.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.ingestion.scraper_team_games import (
    AUTHORITATIVE_SOURCE,
    MIN_PLAYERS_PER_TEAM_GAME,
    STAT_SUMS,
    TEAM_SOURCE,
    TeamAggregateError,
    aggregate_team_games,
)


def _player(team: str, opponent: str, home: str, idx: int, **over) -> dict:
    row = {
        "game_id": "0022600123",
        "game_date": "2026-10-23",
        "season": "2026-27",
        "season_type": "Regular Season",
        "team_abbr": team,
        "opponent_abbr": opponent,
        "home_away": home,
        "player_name": f"{team} Player {idx}",
        "player_id_nba": f"{team}{idx:03d}",
        "played": True,
        "pts": 10.0, "fgm": 4.0, "fga": 9.0, "fg3m": 1.0, "fg3a": 3.0,
        "ftm": 1.0, "fta": 2.0, "oreb": 1.0, "dreb": 3.0, "reb": 4.0,
        "ast": 2.0, "stl": 1.0, "blk": 0.0, "tov": 1.0, "pf": 2.0,
    }
    row.update(over)
    return row


def _game(home_players: int = 10, away_players: int = 10, **over) -> pd.DataFrame:
    rows = [
        _player("LAL", "GSW", "home", i, **over) for i in range(home_players)
    ] + [
        _player("GSW", "LAL", "away", i, **over) for i in range(away_players)
    ]
    return pd.DataFrame(rows)


# --- the arithmetic ------------------------------------------------------


def test_a_team_row_is_the_sum_of_its_players():
    frame = _game()
    out, report = aggregate_team_games(frame)

    assert len(out) == 2
    assert report.team_games_out == 2
    lal = out.loc[out["team_abbr"] == "LAL"].iloc[0]

    # Every one of the fifteen, against the sum of the ten player rows.
    for player_col, team_col in STAT_SUMS.items():
        expected = frame.loc[frame["team_abbr"] == "LAL", player_col].sum()
        assert lal[team_col] == expected, f"{team_col} != sum of {player_col}"

    assert lal["points"] == 100.0  # 10 players x 10 points
    assert lal["is_home"]
    assert lal["opponent_abbr"] == "GSW"
    assert lal["source"] == TEAM_SOURCE


def test_the_two_sides_get_opposite_home_flags():
    out, _ = aggregate_team_games(_game())
    by_team = out.set_index("team_abbr")["is_home"]
    assert bool(by_team["LAL"]) is True
    assert bool(by_team["GSW"]) is False


def test_a_dnp_contributes_nothing_without_nulling_the_total():
    """
    The scraper marks a DNP `played=False` with NaN stats. For a TEAM total
    that is exactly right -- the player added nothing -- but a sum that
    propagated the NaN would null the whole column and the row would be
    useless rather than wrong, which is harder to notice.
    """
    rows = [_player("LAL", "GSW", "home", i) for i in range(9)]
    dnp = _player("LAL", "GSW", "home", 99, played=False)
    for col in STAT_SUMS:
        dnp[col] = float("nan")
    rows.append(dnp)
    rows += [_player("GSW", "LAL", "away", i) for i in range(10)]

    out, _ = aggregate_team_games(pd.DataFrame(rows))
    lal = out.loc[out["team_abbr"] == "LAL"].iloc[0]
    assert lal["points"] == 90.0, "the DNP row changed the total"
    assert pd.notna(lal["points"]), "a NaN propagated and nulled the column"


# --- the ways a sum can be wrong while looking right --------------------


def test_a_team_totals_row_is_refused_rather_than_doubling_everything():
    """
    THE SHARPEST FAILURE THIS MODULE CAN HAVE. A "Team Totals" line summed
    alongside the players doubles every figure, and the result is fifteen
    plausible integers that nothing downstream can question. The scraper's
    Basketball-Reference parser drops these, but inheriting a guarantee is not
    holding one.
    """
    rows = [_player("LAL", "GSW", "home", i) for i in range(10)]
    totals = _player("LAL", "GSW", "home", 0, player_name="Team Totals")
    for col in STAT_SUMS:
        totals[col] = float(sum(r[col] for r in rows))
    rows.append(totals)
    rows += [_player("GSW", "LAL", "away", i) for i in range(10)]

    out, report = aggregate_team_games(pd.DataFrame(rows))
    lal = out.loc[out["team_abbr"] == "LAL"].iloc[0]

    assert lal["points"] == 100.0, (
        f"the Team Totals row was summed with the players: points={lal['points']} "
        f"instead of 100"
    )
    assert report.dropped_non_player_rows == 1


@pytest.mark.parametrize("name", ["Team Totals", "team totals", "  Reserves  "])
def test_non_player_row_names_are_matched_case_and_space_insensitively(name):
    rows = [_player("LAL", "GSW", "home", i) for i in range(10)]
    rows.append(_player("LAL", "GSW", "home", 0, player_name=name, pts=999.0))
    rows += [_player("GSW", "LAL", "away", i) for i in range(10)]
    out, _ = aggregate_team_games(pd.DataFrame(rows))
    assert out.loc[out["team_abbr"] == "LAL", "points"].iloc[0] == 100.0


@pytest.mark.parametrize("bad", [None, "", "   ", "None", "nan"])
def test_a_row_with_no_player_identity_is_not_summed(bad):
    """An unresolved crosswalk leaves `player_id_nba` null rather than
    guessing -- and that is also the shape an aggregate row has."""
    rows = [_player("LAL", "GSW", "home", i) for i in range(10)]
    rows.append(_player("LAL", "GSW", "home", 0, player_id_nba=bad, pts=500.0))
    rows += [_player("GSW", "LAL", "away", i) for i in range(10)]

    out, report = aggregate_team_games(pd.DataFrame(rows))
    assert out.loc[out["team_abbr"] == "LAL", "points"].iloc[0] == 100.0
    assert report.dropped_no_identity == 1


def test_a_short_roster_is_skipped_and_named_without_losing_the_good_games():
    """
    A total summed over four ingested players is arithmetically fine and
    factually wrong, and nothing downstream would catch it. So it is refused
    and NAMED -- an import that silently halves a team's points is worse than
    one that writes nothing.

    TWO GAMES, deliberately. My first version used a single game with one
    short roster and asserted the function returned without it; it raised
    instead, and correctly -- dropping that team left the game one-sided, and
    a game with one team in it is refused by the next rule. That conflated the
    two guards. With a second, complete game present, this tests the roster
    rule alone AND the thing that matters operationally: one bad team-game
    must not take the night's good ones down with it.
    """
    short = _game(home_players=4, away_players=10)
    good = _game()
    good["game_id"] = "0022600200"
    good["game_date"] = "2026-10-24"

    out, report = aggregate_team_games(pd.concat([short, good], ignore_index=True))

    assert set(out["nba_game_id"]) == {"0022600200"}, (
        "the complete game was lost along with the incomplete one"
    )
    assert len(out) == 2
    assert report.skipped_short_roster
    assert "LAL" in report.skipped_short_roster[0]
    assert "4 players" in report.skipped_short_roster[0]
    assert "0022600123" in report.skipped_one_sided_games


def test_a_full_roster_at_the_threshold_is_accepted():
    """The guard must not be so tight it refuses a real short-handed team."""
    out, report = aggregate_team_games(
        _game(home_players=MIN_PLAYERS_PER_TEAM_GAME,
              away_players=MIN_PLAYERS_PER_TEAM_GAME)
    )
    assert len(out) == 2
    assert not report.skipped_short_roster


def test_a_one_sided_game_is_skipped():
    """Elo cannot score a result with one team in it, and `defense` cannot
    find an opponent that is not there."""
    rows = [_player("LAL", "GSW", "home", i) for i in range(10)]
    frame = pd.DataFrame(rows)
    with pytest.raises(TeamAggregateError, match="no complete team-game"):
        aggregate_team_games(frame)


def test_preseason_rows_are_not_aggregated():
    rows = [
        _player("LAL", "GSW", "home", i, season_type="Pre Season")
        for i in range(10)
    ] + [
        _player("GSW", "LAL", "away", i, season_type="Pre Season")
        for i in range(10)
    ]
    with pytest.raises(TeamAggregateError):
        aggregate_team_games(pd.DataFrame(rows))


# --- precedence: a licensed row is left alone --------------------------


def test_a_row_the_licensed_source_owns_is_declined():
    """
    THE REASON THIS CANNOT BE LEFT TO THE UPSERT. `upsert_team_game_stats`'s
    conflict clause refreshes `points, pace, off_eff, def_eff, poss,
    opponent_abbr, is_home, is_neutral_site` -- not the fifteen box-score
    columns and not `source`. Writing over a BigDataBall row would replace its
    points, leave its own fg/fga/reb, and leave it labelled `bigdataball`: a
    row stitched from two sources claiming to be from one.
    """
    owned = {("0022600123", "LAL"): AUTHORITATIVE_SOURCE}
    out, report = aggregate_team_games(_game(), owned_by=owned)

    assert set(out["team_abbr"]) == {"GSW"}, "a licensed row was overwritten"
    assert report.declined_to_authoritative == ["0022600123/LAL"]


def test_a_row_this_writer_owns_is_refreshed():
    """Its own previous output is not sacred: re-running must update it."""
    owned = {("0022600123", "LAL"): TEAM_SOURCE}
    out, report = aggregate_team_games(_game(), owned_by=owned)
    assert set(out["team_abbr"]) == {"LAL", "GSW"}
    assert not report.declined_to_authoritative


def test_every_row_owned_by_the_licensed_source_is_not_an_error():
    owned = {
        ("0022600123", "LAL"): AUTHORITATIVE_SOURCE,
        ("0022600123", "GSW"): AUTHORITATIVE_SOURCE,
    }
    out, report = aggregate_team_games(_game(), owned_by=owned)
    assert out.empty
    assert report.team_games_out == 0
    assert len(report.declined_to_authoritative) == 2


def test_the_provenance_is_never_the_licensed_label():
    out, _ = aggregate_team_games(_game())
    assert set(out["source"]) == {TEAM_SOURCE}
    assert TEAM_SOURCE != AUTHORITATIVE_SOURCE


# --- what is deliberately absent --------------------------------------


def test_possessions_and_efficiency_are_left_null():
    """
    `builder.attach_team_pace` owns the one possessions formula
    (`FGA - OREB + TOV + 0.44 * FTA`) and `defense.py` says "already in
    attach_team_pace; it is not repeated here". Nothing reads the stored
    columns, so computing them here would be a second definition of the
    formula -- and `repository.TEAM_GAME_STAT_COLS` would then hand two
    different answers to the same question.
    """
    out, _ = aggregate_team_games(_game())
    for absent in ("poss", "pace", "off_eff", "def_eff"):
        assert absent not in out.columns, (
            f"{absent} was computed here; the builder owns it"
        )


def test_neutral_site_is_false_and_that_is_a_known_loss():
    """
    RECORDED RATHER THAN HIDDEN. BigDataBall sets this from a venue column
    the scraper has no equivalent for. False is wrong for about six games a
    season, and the consequence is specific: `fatigue_logic` applies its
    altitude tax on `IS_HOME == False`, and this flag is what stops a
    neutral-site game in Denver or Utah taxing BOTH teams.

    If a venue field ever arrives, this test is where the change starts.
    """
    out, _ = aggregate_team_games(_game())
    assert set(out["is_neutral_site"]) == {False}


def test_the_output_satisfies_what_elo_and_defense_require():
    """
    The two consumers, read from THEIR constants rather than restated here.
    """
    from src.features.defense import REQUIRED_TEAM_COLS

    out, _ = aggregate_team_games(_game())
    for col in REQUIRED_TEAM_COLS:
        assert col in out.columns, f"defense.py requires {col}"

    # team_strength accepts the DB names and normalises them itself.
    for col in ("nba_game_id", "game_date", "team_abbr", "points"):
        assert col in out.columns

    from src.features.team_strength import _normalize_team_games

    normalised = _normalize_team_games(out)
    assert {"game_id", "game_date", "team", "points"} <= set(normalised.columns)


# --- refusing rather than half-working --------------------------------


def test_an_empty_frame_is_refused():
    with pytest.raises(TeamAggregateError, match="DATA_NOT_AVAILABLE"):
        aggregate_team_games(pd.DataFrame())


def test_a_frame_missing_a_stat_column_is_refused_naming_it():
    frame = _game().drop(columns=["tov"])
    with pytest.raises(TeamAggregateError) as exc:
        aggregate_team_games(frame)
    assert "tov" in str(exc.value)


def test_the_game_id_goes_through_the_project_s_own_normaliser():
    frame = _game()
    frame["game_id"] = 22600123          # a parquet round-trip can do this
    out, _ = aggregate_team_games(frame)
    assert set(out["nba_game_id"]) == {"0022600123"}


def test_the_report_accounts_for_the_rows_it_was_given():
    rows = [_player("LAL", "GSW", "home", i) for i in range(10)]
    rows.append(_player("LAL", "GSW", "home", 0, player_name="Team Totals"))
    rows.append(_player("LAL", "GSW", "home", 0, player_id_nba=None))
    rows += [_player("GSW", "LAL", "away", i) for i in range(10)]

    _, report = aggregate_team_games(pd.DataFrame(rows))
    assert report.rows_in == 22
    assert report.dropped_non_player_rows == 1
    assert report.dropped_no_identity == 1
    assert report.team_games_out == 2
    assert any("non-player" in line for line in report.lines())
