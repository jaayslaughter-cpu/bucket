"""
tests/test_scraper_panel.py — the scraper's parquet, mapped onto the panel.

WHY THE FIXTURE IS NOT HAND-WRITTEN. `tests/fixtures/scraper_canonical/` was
produced by running nba-scraper-2026-27's OWN `map_live_boxscore` over
nba-scraper-2026-27's OWN fixture payload, and captured. So these tests run
against the producer's real output shape rather than my reading of its README,
and the scraper does not have to be installed for CI to exercise it. If the
scraper's canonical schema changes, the committed fixture goes stale and
`test_the_fixture_still_carries_every_column_the_mapper_reads` is what says so.

The four rows in it are the four cases that matter, which is why that payload
was chosen:

  * LeBron James   — played, 1954 seconds, 19 points
  * Bronny James   — DNP: `played=False`, stats NaN ("unavailable", not zero)
  * Stephen Curry  — played, away side
  * Nikola Jokić   — `played=True` with ZERO seconds and 0.0 points: a real
                     zero-minute appearance, which must NOT be confused with
                     the DNP above. Also a non-ASCII name, which is the bug
                     class that made four of the scraper's own tests fail on
                     Windows.

RESEARCH_ONLY.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from src.ingestion.scraper_panel import (
    COLUMN_MAP,
    DEFAULT_SEASON_TYPES,
    REQUIRED_COLUMNS,
    SOURCE_PRECEDENCE,
    ScraperPanelError,
    read_canonical_boxscores,
    resolve_dataset_dir,
    to_player_panel,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "scraper_canonical"


@pytest.fixture
def canonical() -> pd.DataFrame:
    frame, _ = read_canonical_boxscores(FIXTURE_DIR)
    return frame


# --- the contract with the producer --------------------------------------


def test_the_fixture_still_carries_every_column_the_mapper_reads(canonical):
    """
    THE CONTRACT TEST. The fixture came from the scraper's own mapper; if its
    canonical schema ever drops or renames a column this mapper reads, this is
    the failure, rather than a column of nulls written to Postgres.
    """
    missing_required = [c for c in REQUIRED_COLUMNS if c not in canonical.columns]
    assert not missing_required, missing_required
    missing_mapped = [c for c in COLUMN_MAP if c not in canonical.columns]
    assert not missing_mapped, missing_mapped


def test_the_panel_satisfies_the_upsert_s_OWN_stated_requirement(canonical):
    """
    Read out of `repository.py` rather than restated here, so adding a
    required column there fails this instead of failing in production. The
    upsert raises `DATA_NOT_AVAILABLE` on a missing one.
    """
    body = (ROOT / "src" / "db" / "repository.py").read_text(encoding="utf-8")
    match = re.search(r"required = \{([^}]*)\}", body)
    assert match, "could not find the upsert's `required` set in repository.py"
    required = set(re.findall(r'"(\w+)"', match.group(1)))
    assert required, "parsed an empty requirement set, so this proves nothing"

    panel, _ = to_player_panel(canonical)
    assert required <= set(panel.columns), required - set(panel.columns)


# --- a DNP is not a zero -------------------------------------------------


def test_a_dnp_is_dropped_but_a_real_zero_minute_appearance_is_kept(canonical):
    """
    THE DISTINCTION THE WHOLE MODULE TURNS ON. Both rows have zero seconds.
    One player was unavailable; the other dressed, checked in and did nothing.
    Collapsing them would either invent a zero-point game or discard a real one.
    """
    panel, report = to_player_panel(canonical)
    names = set(panel["PLAYER_NAME"])

    assert "Bronny James" not in names, "a DNP reached the panel by default"
    assert report.dropped_dnp == 1

    assert "Nikola Jokić" in names, (
        "a real zero-minute appearance was dropped as if it were a DNP"
    )
    jokic = panel.loc[panel["PLAYER_NAME"] == "Nikola Jokić"].iloc[0]
    assert jokic["MIN"] == 0.0
    assert jokic["PTS"] == 0.0
    assert not pd.isna(jokic["PTS"]), "a played zero was turned into unknown"


def test_the_string_False_is_not_a_player_who_played():
    """
    `Series.astype(bool)` on the STRING "False" is True, because every
    non-empty string is truthy. A source or a round-trip that handed over
    `played` as text would then turn a DNP into a player who played and
    scored nothing — a fabricated zero-point game, which is strictly worse
    than dropping the row.
    """
    frame = _synthetic([
        {"player_id_nba": "1630228", "player_name": "Bronny James",
         "played": "False", "pts": None},
        {"player_id_nba": "2544", "played": "True"},
    ])
    panel, report = to_player_panel(frame)
    assert list(panel["PLAYER_NAME"]) == ["LeBron James"]
    assert report.dropped_dnp == 1


def test_including_dnps_writes_nulls_and_never_zeros(canonical):
    """`--include-dnp` is allowed to add the row. It is not allowed to claim
    the player scored nothing: migrations 006 and 009 exist because a
    defaulted zero reads as a clean game."""
    panel, report = to_player_panel(canonical, include_dnp=True)
    assert report.dropped_dnp == 0
    bronny = panel.loc[panel["PLAYER_NAME"] == "Bronny James"].iloc[0]
    for col in ("PTS", "REB", "AST", "FG3M", "FGA", "FTA", "STL", "BLK", "TOV", "PF"):
        assert pd.isna(bronny[col]), f"{col} was filled in for a player who did not play"


# --- units ---------------------------------------------------------------


def test_minutes_are_minutes(canonical):
    """`seconds_played` is seconds. `player_game_logs.minutes` is minutes, and
    `features/minutes_weighted` compares it against prior minute means — so
    passing seconds through would be a 60x feature, not a unit nit."""
    panel, _ = to_player_panel(canonical)
    lebron = panel.loc[panel["PLAYER_NAME"] == "LeBron James"].iloc[0]
    assert lebron["MIN"] == pytest.approx((32 * 60 + 34) / 60.0)
    assert 32 < lebron["MIN"] < 33


def test_the_game_id_goes_through_the_project_s_own_normaliser():
    """`utils.ids.normalise_game_id` is the single reader of the fact that
    these are ten-wide strings; a parquet round-trip can hand back an int."""
    frame = _synthetic([
        {"game_id": 22600123, "player_id_nba": "2544", "source": "nba_live"},
    ])
    panel, _ = to_player_panel(frame)
    assert panel["GAME_ID"].iloc[0] == "0022600123"


def test_home_away_becomes_a_boolean(canonical):
    panel, _ = to_player_panel(canonical)
    by_name = panel.set_index("PLAYER_NAME")["IS_HOME"]
    assert bool(by_name["LeBron James"]) is True
    assert bool(by_name["Stephen Curry"]) is False


# --- the duplicate key ---------------------------------------------------


def _synthetic(overrides: list[dict]) -> pd.DataFrame:
    """A minimal frame in the scraper's canonical shape."""
    rows = []
    for over in overrides:
        row = {
            "game_id": "0022600123", "player_id_nba": "2544",
            "player_name": "LeBron James", "game_date": "2026-10-23",
            "season": "2026-27", "season_type": "Regular Season",
            "home_away": "home", "played": True, "seconds_played": 1800.0,
            "source": "nba_live", "team_abbr": "LAL", "opponent_abbr": "GSW",
            "pts": 20.0, "reb": 7.0, "ast": 8.0, "fg3m": 2.0, "fga": 15.0,
            "fta": 4.0, "stl": 1.0, "blk": 1.0, "tov": 3.0, "pf": 2.0,
        }
        row.update(over)
        rows.append(row)
    return pd.DataFrame(rows)


def test_three_sources_for_one_player_game_collapse_to_one_row():
    frame = _synthetic([
        {"source": "basketball_reference", "pts": 21.0},
        {"source": "nba_live", "pts": 20.0},
        {"source": "nba_stats", "pts": 19.0},
    ])
    panel, report = to_player_panel(frame)
    assert len(panel) == 1
    assert report.duplicate_rows_resolved == 2
    assert panel["SOURCE"].iloc[0] == "nba_stats", (
        f"precedence is {SOURCE_PRECEDENCE}, so nba_stats should win"
    )


def test_the_winning_row_is_taken_WHOLE_and_never_blended():
    """
    A column-wise merge across sources would produce a box-score line that no
    source ever reported — worse than either source's own version of it, and
    impossible to reconcile against anything.

    THE FIRST VERSION OF THIS TEST COULD NOT FAIL. It gave every source a
    different value for every stat and asserted the winner's values came
    through. But a blend via `groupby().first()` takes the first NON-NULL
    value per column, and when the winning row has all of them populated that
    is the winning row's value anyway — so the mutation changed nothing the
    test looked at. Verified: swapping `drop_duplicates` for
    `groupby(...).first()` left all 30 tests green.

    A blend is only observable where the WINNER HAS A HOLE that a loser could
    fill. That is also the realistic case: Basketball-Reference carries
    `game_score`, the live feed does not, and the sources disagree about which
    stats they report at all. So the winning row below leaves `pf` and `tov`
    unreported, and the assertion is that they stay unreported.
    """
    frame = _synthetic([
        # The winner, with two holes in it.
        {"source": "nba_stats", "pts": 19.0, "reb": 7.0, "ast": 8.0,
         "tov": None, "pf": None},
        # Losers that COULD fill those holes, and differ everywhere else.
        {"source": "nba_live", "pts": 20.0, "reb": 8.0, "ast": 9.0,
         "tov": 4.0, "pf": 2.0},
        {"source": "basketball_reference", "pts": 21.0, "reb": 9.0,
         "ast": 10.0, "tov": 5.0, "pf": 3.0},
    ])
    panel, _ = to_player_panel(frame)
    assert len(panel) == 1
    row = panel.iloc[0]

    assert row["SOURCE"] == "nba_stats"
    assert (row["PTS"], row["REB"], row["AST"]) == (19.0, 7.0, 8.0)
    # THE ASSERTION THE MUTATION TRIPS ON: a hole in the winning row is a
    # hole in the panel, not an invitation to borrow from another source.
    assert pd.isna(row["TOV"]), (
        "TOV was filled from a lower-precedence source — this row is now a "
        "blend that no source ever reported"
    )
    assert pd.isna(row["PF"]), (
        "PF was filled from a lower-precedence source — same defect"
    )


def test_a_source_nobody_declared_ranks_LAST_not_first():
    """
    `.map(dict.get)` returning None for an unknown source would sort it ahead
    of everything. A source this project has never evaluated must not silently
    outrank the league's own feed — and it must be reported.
    """
    frame = _synthetic([
        {"source": "some_new_scraper", "pts": 99.0},
        {"source": "nba_stats", "pts": 19.0},
    ])
    panel, report = to_player_panel(frame)
    assert panel["SOURCE"].iloc[0] == "nba_stats"
    assert panel["PTS"].iloc[0] == 19.0
    assert "some_new_scraper" in report.unknown_sources


def test_two_different_players_in_one_game_are_not_a_duplicate():
    frame = _synthetic([
        {"player_id_nba": "2544", "player_name": "LeBron James"},
        {"player_id_nba": "201939", "player_name": "Stephen Curry"},
    ])
    panel, report = to_player_panel(frame)
    assert len(panel) == 2
    assert report.duplicate_rows_resolved == 0


def test_the_same_player_in_two_different_games_is_not_a_duplicate():
    frame = _synthetic([
        {"game_id": "0022600123"},
        {"game_id": "0022600200", "game_date": "2026-10-25"},
    ])
    panel, _ = to_player_panel(frame)
    assert len(panel) == 2


# --- what must never enter the panel -------------------------------------


def test_preseason_is_excluded_by_default():
    """`boxscores.BoxScoreLoadConfig.season_type` is "Regular Season", so the
    existing panel has no exhibition games. Admitting them here would put
    October exhibition minutes inside every _L5 window."""
    frame = _synthetic([
        {"season_type": "Pre Season", "game_id": "0012600001"},
        {"season_type": "Regular Season", "game_id": "0022600123"},
    ])
    panel, report = to_player_panel(frame)
    assert list(panel["GAME_ID"]) == ["0022600123"]
    assert report.dropped_wrong_season_type == 1
    assert DEFAULT_SEASON_TYPES == ("Regular Season",)


def test_a_season_type_can_be_asked_for_explicitly():
    frame = _synthetic([
        {"season_type": "Playoffs", "game_id": "0042600101"},
        {"season_type": "Regular Season", "game_id": "0022600123"},
    ])
    panel, _ = to_player_panel(
        frame, season_types=("Regular Season", "Playoffs")
    )
    assert set(panel["GAME_ID"]) == {"0042600101", "0022600123"}


@pytest.mark.parametrize("bad", [None, "", "  ", "None", "nan"])
def test_a_row_whose_player_identity_did_not_resolve_is_dropped(bad):
    """
    `player_id_nba` is nullable in the source BECAUSE the Basketball-Reference
    crosswalk is allowed to fail rather than guess at a name. Such a row
    cannot be keyed, so it is dropped and counted — never defaulted to
    something that would collide with a real player.
    """
    frame = _synthetic([
        {"player_id_nba": bad, "player_name": "Unresolved Guy"},
        {"player_id_nba": "2544"},
    ])
    panel, report = to_player_panel(frame)
    assert list(panel["PLAYER_NAME"]) == ["LeBron James"]
    assert report.dropped_no_player_id == 1


def test_is_starter_never_becomes_starting_position(canonical):
    """
    `is_starter` is a bool. `starting_position` is G/F/C with a CHECK from
    migration 008 and a meaning `features/dvp.py` reads as POS_BUCKET. There
    is no position in this payload, so the column must be ABSENT rather than
    invented — `update_starting_positions` is what fills it.
    """
    panel, _ = to_player_panel(canonical)
    assert "STARTING_POSITION" not in panel.columns
    assert "IS_STARTER" not in panel.columns


def test_provenance_is_recorded_per_row(canonical):
    """The comment in `upsert_player_game_logs` exists because 214,381 Kaggle
    rows once claimed to come from stats.nba.com. SOURCE must be the row's own
    source, not a constant."""
    frame = _synthetic([
        {"player_id_nba": "2544", "source": "nba_stats"},
        {"player_id_nba": "201939", "source": "basketball_reference"},
    ])
    panel, _ = to_player_panel(frame)
    assert set(panel["SOURCE"]) == {"nba_stats", "basketball_reference"}


def test_a_non_ascii_player_name_survives(canonical):
    """The bug class that broke the scraper's own suite on Windows: text I/O
    without an explicit encoding. If a parquet round-trip or this mapper ever
    mangles it, the crosswalk stops matching and the row silently disappears."""
    panel, _ = to_player_panel(canonical)
    assert "Nikola Jokić" in set(panel["PLAYER_NAME"])


# --- refusing rather than reporting success ------------------------------


def test_an_empty_frame_is_refused():
    with pytest.raises(ScraperPanelError, match="DATA_NOT_AVAILABLE"):
        to_player_panel(pd.DataFrame())


def test_a_frame_that_filters_down_to_nothing_is_refused_with_the_counts():
    """The failure mode this guards is the one the repo keeps removing: an
    import that discarded everything and reported success."""
    frame = _synthetic([{"season_type": "Pre Season"}])
    with pytest.raises(ScraperPanelError) as exc:
        to_player_panel(frame)
    assert "DATA_NOT_AVAILABLE" in str(exc.value)
    assert "wrong season type" in str(exc.value)


def test_a_parquet_missing_a_column_is_refused_naming_it(tmp_path):
    d = tmp_path / "canonical" / "player_boxscore" / "game_date=2026-10-23"
    d.mkdir(parents=True)
    pd.DataFrame({"game_id": ["0022600123"]}).to_parquet(
        d / "nba_live__0022600123.parquet", index=False
    )
    with pytest.raises(ScraperPanelError) as exc:
        read_canonical_boxscores(tmp_path)
    assert "DATA_NOT_AVAILABLE" in str(exc.value)
    assert "player_id_nba" in str(exc.value)


def test_a_directory_with_no_parquet_says_to_run_the_scraper(tmp_path):
    (tmp_path / "canonical" / "player_boxscore").mkdir(parents=True)
    with pytest.raises(ScraperPanelError, match="run_daily.py"):
        read_canonical_boxscores(tmp_path)


def test_a_missing_directory_is_refused(tmp_path):
    with pytest.raises(ScraperPanelError, match="does not exist"):
        read_canonical_boxscores(tmp_path / "nope")


# --- the path puzzle ----------------------------------------------------


def test_all_three_spellings_of_the_directory_resolve_to_the_dataset():
    """An operator may reasonably point at the data root, the canonical
    directory, or the dataset directory. Guessing wrong and globbing a parent
    would read play_by_play parquet into a box-score mapper."""
    dataset = FIXTURE_DIR / "player_boxscore"
    assert resolve_dataset_dir(FIXTURE_DIR) == dataset
    assert resolve_dataset_dir(dataset) == dataset


def test_a_directory_that_is_not_the_dataset_is_refused_with_what_it_looked_for(tmp_path):
    (tmp_path / "canonical" / "play_by_play").mkdir(parents=True)
    with pytest.raises(ScraperPanelError, match="player_boxscore"):
        resolve_dataset_dir(tmp_path)


# --- the report is the only trace, so it has to add up -------------------


def test_the_report_accounts_for_every_row_it_was_given():
    frame = _synthetic([
        {"season_type": "Pre Season"},                                  # dropped
        {"player_id_nba": None, "player_name": "Unresolved"},           # dropped
        {"player_id_nba": "1630228", "played": False, "pts": None},     # dropped
        {"player_id_nba": "2544", "source": "nba_stats"},               # kept
        {"player_id_nba": "2544", "source": "nba_live"},                # duplicate
    ])
    panel, report = to_player_panel(frame)
    assert report.rows_in == 5
    accounted = (
        report.rows_out
        + report.dropped_wrong_season_type
        + report.dropped_no_player_id
        + report.dropped_no_game_id
        + report.dropped_dnp
        + report.duplicate_rows_resolved
    )
    assert accounted == report.rows_in, "\n".join(report.lines())
    assert report.rows_out == len(panel) == 1
