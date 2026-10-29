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
        "source": "nba_live",
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
    conflict clause now refreshes every written column, `source` included, so
    the write no longer stitches a row out of two sources -- it replaces one
    wholesale. For this writer that is worse, not better: an aggregated row
    carries no `poss` and no spread, so taking over a licensed BigDataBall
    team-game would downgrade it and relabel it `scraper_agg`, and nothing in
    the statement can tell that apart from a correction. Only the caller knows
    which source outranks which.
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
    (`FGA - OREB + TOV + 0.44 * FTA`), so computing it here would be a second
    definition of it and `repository.TEAM_GAME_STAT_COLS` would then hand two
    different answers to the same question.

    SOMETHING DOES READ THE STORED COLUMN, and this docstring used to say
    nothing did: `src/features/defense.py` prefers `team_games.poss` where a
    row has one and estimates from FGA/OREB/TOV/FTA where it does not.
    Leaving the column NULL is therefore not an omission the readers have to
    work around -- it is how they are told to use the one formula. Writing an
    aggregated `poss` here would make `defense.py` prefer OUR number over
    `attach_team_pace`'s for exactly these rows, which is the second
    definition arriving by the back door. Corrected in review on 2026-10-29,
    together with the per-row coalesce that makes the NULL safe
    (`tests/test_defense.py::test_one_writer_s_missing_possessions_are_estimated_not_nulled`).
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


# --- one row per player, across sources ---------------------------------


def test_the_same_player_from_three_sources_is_summed_once():
    """
    THE DEFECT THIS MODULE WOULD HAVE SHIPPED WITH, and the reason 27 tests
    passed over it: every fixture here was single-source.

    The scraper writes one row per (source, game, player) — `nba_live`,
    `nba_stats` and `basketball_reference` each contribute a copy — so summing
    the frame as it arrives multiplies every team total by the number of
    enabled sources. The operator's own `.env` has USE_NBA_STATS=1 and
    USE_BASKETBALL_REFERENCE=1, which triples it.

    Neither of the other guards would notice: a short roster looks for FEWER
    rows and duplicates give MORE, and the two-teams rule is satisfied either
    way. Found in review, 2026-10-29.
    """
    rows = []
    for source in ("nba_live", "nba_stats", "basketball_reference"):
        for team, opponent, side in (("LAL", "GSW", "home"), ("GSW", "LAL", "away")):
            rows += [
                _player(team, opponent, side, i, source=source) for i in range(10)
            ]
    frame = pd.DataFrame(rows)
    assert len(frame) == 60, "three sources x two teams x ten players"

    out, report = aggregate_team_games(frame)

    lal = out.loc[out["team_abbr"] == "LAL"].iloc[0]
    assert lal["points"] == 100.0, (
        f"points={lal['points']} — the per-source copies were summed, so the "
        f"total is multiplied by the number of sources"
    )
    assert report.deduplicated_source_copies == 40
    assert report.team_games_out == 2


def test_the_surviving_copy_is_the_precedence_winner():
    """Whole-row precedence, the same order `to_player_panel` declares: a
    blended row would be a box score no source ever reported."""
    rows = []
    for team, opponent, side in (("LAL", "GSW", "home"), ("GSW", "LAL", "away")):
        for i in range(10):
            rows.append(_player(team, opponent, side, i,
                                source="basketball_reference", pts=99.0))
            rows.append(_player(team, opponent, side, i,
                                source="nba_stats", pts=10.0))
    out, _ = aggregate_team_games(pd.DataFrame(rows))
    lal = out.loc[out["team_abbr"] == "LAL"].iloc[0]
    assert lal["points"] == 100.0, (
        f"points={lal['points']}; nba_stats outranks basketball_reference, so "
        f"10 per player should have won over 99"
    )


def test_the_precedence_is_read_from_the_player_panel_not_restated():
    import inspect

    from src.ingestion.scraper_panel import SOURCE_PRECEDENCE as panel_order
    from src.ingestion.scraper_team_games import aggregate_team_games as agg

    assert "precedence" in inspect.signature(agg).parameters
    assert inspect.signature(agg).parameters["precedence"].default == panel_order


def test_a_frame_with_no_source_column_is_refused():
    """Without it the copies cannot be collapsed, and a frame that cannot be
    deduplicated must not be summed."""
    frame = _game().drop(columns=["source"])
    with pytest.raises(TeamAggregateError) as exc:
        aggregate_team_games(frame)
    assert "source" in str(exc.value)


# --- an unknown side, and a played player with a null stat --------------


def test_a_team_game_with_no_home_away_is_refused_not_called_away():
    """
    `astype(str).eq("home")` turns a missing value into "nan" and then into
    False, so the row would be written as an away game. `fatigue_logic` reads
    IS_HOME for the altitude tax and `team_strength` for home advantage — a
    silent False is a wrong feature, not a missing one.
    """
    rows = [_player("LAL", "GSW", None, i) for i in range(10)]
    rows += [_player("GSW", "LAL", "away", i) for i in range(10)]
    with pytest.raises(TeamAggregateError, match="no complete team-game"):
        aggregate_team_games(pd.DataFrame(rows))


def test_a_played_player_with_a_null_stat_is_reported_not_silently_zero():
    """
    `sum(min_count=1)` treats one player's null as zero. For a DNP that is
    right — he added nothing. For a PLAYED row it is a data gap, and the team
    total comes out silently LOW. The row is still written, because a total
    short by one line beats no row at all, but it is never silent.
    """
    rows = [_player("LAL", "GSW", "home", i) for i in range(9)]
    gap = _player("LAL", "GSW", "home", 9)
    gap["pts"] = float("nan")          # played, but points unknown
    rows.append(gap)
    rows += [_player("GSW", "LAL", "away", i) for i in range(10)]

    out, report = aggregate_team_games(pd.DataFrame(rows))
    lal = out.loc[out["team_abbr"] == "LAL"].iloc[0]
    assert lal["points"] == 90.0, "the gap did not behave as zero"
    assert report.stat_gaps, "a low total was not reported"
    assert "LAL" in report.stat_gaps[0]
    assert any("TOTALS LOW" in line for line in report.lines())


def test_a_dnp_s_nulls_are_not_counted_as_a_gap():
    """The distinction that makes the warning worth having: a DNP's nulls are
    correct, so flagging them would make the signal noise."""
    rows = [_player("LAL", "GSW", "home", i) for i in range(9)]
    dnp = _player("LAL", "GSW", "home", 9, played=False)
    for col in STAT_SUMS:
        dnp[col] = float("nan")
    rows.append(dnp)
    rows += [_player("GSW", "LAL", "away", i) for i in range(10)]

    _, report = aggregate_team_games(pd.DataFrame(rows))
    assert not report.stat_gaps, f"a DNP was reported as a gap: {report.stat_gaps}"


def test_a_dnp_that_arrived_as_the_text_false_is_not_counted_as_played():
    """
    `"False"` IS TRUTHY. `.astype(bool)` on the string makes a DNP read as a
    player who played, and a DNP's nulls would then be reported as a stat gap
    on every team-game the scraper ever loads -- a warning line that is always
    there is a warning line nobody reads. `scraper_panel._as_bool` already
    owns this read, so it is borrowed rather than rewritten.
    """
    rows = []
    for team, opponent, side in (("LAL", "GSW", "home"), ("GSW", "LAL", "away")):
        for i in range(10):
            rows.append(_player(team, opponent, side, i))
    # One DNP per team, arriving as TEXT, with the whole line null.
    for team, opponent, side in (("LAL", "GSW", "home"), ("GSW", "LAL", "away")):
        rows.append(_player(team, opponent, side, 99, played="False",
                            **{col: None for col in STAT_SUMS}))

    out, report = aggregate_team_games(pd.DataFrame(rows))
    assert report.stat_gaps == [], (
        f"a text DNP was counted as a played player with missing stats: "
        f"{report.stat_gaps}"
    )
    # And the totals are still the ten who played.
    assert out.loc[out["team_abbr"] == "LAL", "points"].iloc[0] == 100.0
