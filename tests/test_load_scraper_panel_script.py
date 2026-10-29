"""
tests/test_load_scraper_panel_script.py — the operator-facing parquet loader.

WHAT IS WORTH TESTING HERE. Not the mapping (`tests/test_scraper_panel.py`)
and not the aggregation (`tests/test_scraper_team_games.py`); those are tested
where they live. What is this script's own: that a dry run can be used to
validate what it would write BEFORE it writes it, that it needs no database to
do so, and that it does not report a write it has not read back.

THE LAST ONE IS THE ONE THAT COST SOMETHING. `upsert_*` returns `len(rows)` --
rows handed to the statement, not rows kept -- and for five days that number
was printed as "Written to Postgres. 2,644 rows" while the table held nothing.
This script writes a SECOND table through the same kind of call, so it gets the
same read-back.

RESEARCH_ONLY.
"""

from __future__ import annotations

import pandas as pd
import pytest

from scripts import load_scraper_panel as script


def _player(team: str, opponent: str, home: str, idx: int, **over) -> dict:
    """A canonical scraper row, satisfying BOTH mappers' required columns.

    `scraper_panel` needs season/seconds_played/played; `scraper_team_games`
    needs team_abbr/opponent_abbr and the fifteen stat columns. The loader
    reads one parquet and hands the same frame to both, so a fixture that
    only satisfies one of them would not exercise the path at all.
    """
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
        "seconds_played": 1800,
        "source": "nba_live",
        "pts": 10.0, "fgm": 4.0, "fga": 9.0, "fg3m": 1.0, "fg3a": 3.0,
        "ftm": 1.0, "fta": 2.0, "oreb": 1.0, "dreb": 3.0, "reb": 4.0,
        "ast": 2.0, "stl": 1.0, "blk": 0.0, "tov": 1.0, "pf": 2.0,
    }
    row.update(over)
    return row


def _frame(players: int = 10) -> pd.DataFrame:
    return pd.DataFrame(
        [_player("LAL", "GSW", "home", i) for i in range(players)]
        + [_player("GSW", "LAL", "away", i) for i in range(players)]
    )


@pytest.fixture
def parquet(monkeypatch):
    """Stand in for the parquet read; the frame is the fixture above."""
    from src.ingestion import scraper_panel

    monkeypatch.setattr(
        scraper_panel, "read_canonical_boxscores",
        lambda _dir: (_frame(), 1),
    )


def _forbid_database(monkeypatch):
    """Every repository call this script can make, wired to explode."""
    import src.db.repository as repository

    def boom(*a, **kw):
        raise AssertionError("a dry run reached for the database")

    for name in ("ingested_row_counts", "team_game_sources",
                 "upsert_player_game_logs", "upsert_team_game_stats"):
        monkeypatch.setattr(repository, name, boom)


def _stub_database(monkeypatch, counts, before=None, keys_present=None):
    import src.db.repository as repository

    seen = {"counts_calls": 0, "team_rows": None, "player_rows": None}
    series = [before if before is not None else counts, counts]

    def _counts():
        i = min(seen["counts_calls"], len(series) - 1)
        seen["counts_calls"] += 1
        return series[i]

    monkeypatch.setattr(repository, "ingested_row_counts", _counts)
    monkeypatch.setattr(repository, "team_game_sources", lambda: {})
    monkeypatch.setattr(
        repository, "upsert_player_game_logs",
        lambda df: seen.__setitem__("player_rows", len(df)) or len(df),
    )
    monkeypatch.setattr(
        repository, "upsert_team_game_stats",
        lambda df: seen.__setitem__("team_rows", len(df)) or len(df),
    )
    # By default every key this run parsed is present. The key read-back is
    # tested against a real (SQLite) database in
    # `tests/test_ingest_readback.py`; here it stands in so the script's own
    # branches can be reached without one.
    monkeypatch.setattr(
        repository, "ingested_keys_present",
        lambda **kw: {
            "team_game_stats": (
                len(set(kw.get("team_games", ())))
                if keys_present is None else keys_present
            )
        },
    )
    return seen


# --- the dry run must be able to validate the new writer -----------------


def test_a_dry_run_aggregates_team_games_and_needs_no_database(
    parquet, monkeypatch, capsys,
):
    """
    THE VALIDATION STEP CAME AFTER THE STEP IT VALIDATES. The aggregation used
    to sit below the dry-run return, so the only way to see whether it refused
    a short roster, a one-sided game or an unknown home/away was to write to
    the database first. Flagged in review on 2026-10-29.
    """
    _forbid_database(monkeypatch)
    assert script.main(["--canonical-dir", "x", "--team-games", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "team-games out            2" in out
    assert "nothing was written" in out.lower()


def test_a_dry_run_says_which_check_it_could_not_make(
    parquet, monkeypatch, capsys,
):
    """
    `owned_by` is the one part of the aggregation that needs a connection, so
    a dry run cannot tell the operator which keys a licensed ingest already
    owns. Said out loud: a count that would shrink on the real run must not
    read as the number that will land.
    """
    _forbid_database(monkeypatch)
    assert script.main(["--canonical-dir", "x", "--team-games", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "ownership of existing team_game_stats rows" in out


def test_a_dry_run_without_team_games_does_not_aggregate(
    parquet, monkeypatch, capsys,
):
    _forbid_database(monkeypatch)
    assert script.main(["--canonical-dir", "x", "--dry-run"]) == 0
    assert "team-games out" not in capsys.readouterr().out


# --- the write is proved by a read, not by the upsert's own count --------


def test_the_team_game_write_is_read_back(parquet, monkeypatch, capsys):
    seen = _stub_database(monkeypatch, {
        "player_game_logs": 20, "team_game_stats": 2, "game_market_lines": 0,
    }, before={
        "player_game_logs": 0, "team_game_stats": 0, "game_market_lines": 0,
    })
    assert script.main(["--canonical-dir", "x", "--team-games"]) == 0
    out = capsys.readouterr().out
    assert seen["team_rows"] == 2
    assert "read" in out and "fresh connection" in out
    assert "team_game_stats      2 row(s)  (+2 this run)" in out
    assert "2 of this run's 2 team-game key(s) present" in out
    # THE WORD MATTERS. `upsert_*` returns rows handed to the statement, so
    # "Upserted 2,644 rows" is the sentence an operator reported back as
    # confirmation while the table held nothing. Only the read-back claims
    # persistence; the write claims an attempt.
    assert "Upserted" not in out
    assert "Attempted 20 player game-log row(s)" in out
    assert "Attempted 2 team-game stat row(s)" in out


def test_an_empty_table_after_writing_team_games_is_a_failure(
    parquet, monkeypatch, capsys,
):
    """
    THE FIVE-DAY BUG, IN THE SECOND WRITER. `upsert_team_game_stats` returns
    the rows it was handed, so it says "2" whether or not the transaction
    survived; only a reader on a fresh connection -- which is what the 09:00
    PT slate is -- can tell.
    """
    _stub_database(monkeypatch, {
        "player_game_logs": 20, "team_game_stats": 0, "game_market_lines": 0,
    }, keys_present=0)
    assert script.main(["--canonical-dir", "x", "--team-games"]) == 1
    captured = capsys.readouterr()
    assert "are NOT in the table" in captured.err
    assert "step [2]" not in captured.out, "success was claimed over no rows"


def test_a_table_with_other_runs_rows_but_not_this_one_s_is_a_failure(
    parquet, monkeypatch, capsys,
):
    """
    A ROW COUNT IS NOT PROOF OF THIS WRITE. 2,644 rows from an earlier
    ingest and not one of the two keys this run aggregated: the ordinary
    shape of the failure after the first successful load, and what a
    non-empty check passes on.
    """
    _stub_database(monkeypatch, {
        "player_game_logs": 20, "team_game_stats": 2644,
        "game_market_lines": 0,
    }, keys_present=0)
    assert script.main(["--canonical-dir", "x", "--team-games"]) == 1
    captured = capsys.readouterr()
    assert "0 of this run's 2 team-game key(s) present" in captured.out
    assert "are NOT in the table" in captured.err


def test_an_empty_player_panel_after_a_persisting_run_is_a_failure(
    parquet, monkeypatch, capsys,
):
    _stub_database(monkeypatch, {
        "player_game_logs": 0, "team_game_stats": 2, "game_market_lines": 0,
    })
    assert script.main(["--canonical-dir", "x"]) == 1
    assert "player_game_logs is EMPTY" in capsys.readouterr().err


def test_a_read_back_that_cannot_connect_does_not_claim_success(
    parquet, monkeypatch, capsys,
):
    """UNVERIFIED is not VERIFIED-EMPTY and is certainly not success."""
    import src.db.repository as repository

    _stub_database(monkeypatch, {
        "player_game_logs": 20, "team_game_stats": 2, "game_market_lines": 0,
    })
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:          # the baseline, before the write
            return {"player_game_logs": 0, "team_game_stats": 0}
        raise OSError("connection refused")

    monkeypatch.setattr(repository, "ingested_row_counts", flaky)
    assert script.main(["--canonical-dir", "x", "--team-games"]) == 1
    assert "COULD NOT VERIFY" in capsys.readouterr().err


def test_a_baseline_that_cannot_be_read_does_not_fail_the_run(
    parquet, monkeypatch, capsys,
):
    """
    The delta is a courtesy; the read-back is the evidence. A baseline that
    cannot be taken must not abort a write that then verifies fine -- and must
    not print a delta it cannot compute.
    """
    import src.db.repository as repository

    _stub_database(monkeypatch, {
        "player_game_logs": 20, "team_game_stats": 2, "game_market_lines": 0,
    })
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("connection refused")
        return {"player_game_logs": 20, "team_game_stats": 2,
                "game_market_lines": 0}

    monkeypatch.setattr(repository, "ingested_row_counts", flaky)
    assert script.main(["--canonical-dir", "x", "--team-games"]) == 0
    out = capsys.readouterr().out
    assert "this run)" not in out, "a delta was printed with no baseline"
    assert "team_game_stats      2 row(s)" in out
