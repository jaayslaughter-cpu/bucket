"""
The operator-facing BigDataBall ingest wrapper.

WHAT IS WORTH TESTING. Not the parsing or the upserts — those are
`ingest_market_lines`'s and are tested where they live. What is this script's
own: that it calls that shared path rather than loading the workbook a second
way, that a wrong path is named rather than tracebacked, that --dry-run writes
nothing, and that "wrote nothing" is not reported as success.

RESEARCH_ONLY.
"""

from __future__ import annotations

import pandas as pd

from scripts import ingest_bigdataball as script


def _frames(team_rows: int = 3, market_rows: int = 2) -> tuple:
    team = pd.DataFrame({"nba_game_id": ["0022500001"] * team_rows})
    market = pd.DataFrame({
        "nba_game_id": [f"002250000{i}" for i in range(market_rows)],
        "status": ["VALID"] * market_rows,
    })
    return team, market


def _keyed_frames(games=("0022500001", "0022500002")) -> tuple:
    """Frames that carry the key pair, so the per-key read-back applies.

    `_frames` deliberately does not: it covers the fallback, where a frame
    cannot be verified by key and only the row count is available.
    """
    rows = [
        {"nba_game_id": g, "team_abbr": t, "status": "VALID"}
        for g in games for t in ("LAL", "GSW")
    ]
    return pd.DataFrame(rows), pd.DataFrame(rows)


def _stub_verification(monkeypatch, counts=None):
    """
    Stand in for the post-write read-back.

    `scripts/ingest_bigdataball` now reads the tables back on a fresh
    connection before it will say "Written to Postgres", so every test of a
    PERSISTING run needs that call stubbed or it tries to reach a real
    database. The verification itself is tested separately below.
    """
    import src.db.repository as repository

    monkeypatch.setattr(
        repository, "ingested_row_counts",
        lambda: counts if counts is not None else {
            "team_game_stats": 3, "game_market_lines": 3, "player_game_logs": 0,
        },
    )
    # By default every parsed key is present: the frames most tests use carry
    # no `team_abbr`, so the key check does not apply to them at all, and the
    # ones that do are asserting something else.
    _tables = {"team_games": "team_game_stats",
               "market_lines": "game_market_lines"}
    monkeypatch.setattr(
        repository, "ingested_keys_present",
        lambda **kw: {_tables[arg]: len(set(keys)) for arg, keys in kw.items()},
    )


def test_it_calls_the_slate_s_own_ingest_and_does_not_reload_the_workbook(
    monkeypatch, tmp_path, capsys,
):
    """
    ONE LOADER. `main.ingest_market_lines` is what step [2] calls when the
    workbook exists, so this calls that — the parsing, the canonical column
    names and both upserts are shared. A second loader here is the
    two-definitions defect with the licensed export as its subject.
    """
    import main as orchestrator

    wb = tmp_path / "wb.xlsx"
    wb.write_bytes(b"not really xlsx; the loader is stubbed")
    seen: dict = {}

    def fake(path, persist=True):
        seen["path"], seen["persist"] = path, persist
        return _frames()

    monkeypatch.setattr(orchestrator, "ingest_market_lines", fake)
    _stub_verification(monkeypatch)
    assert script.main(["--xlsx", str(wb)]) == 0
    assert seen["path"] == wb
    assert seen["persist"] is True
    out = capsys.readouterr().out
    assert "team_game_stats rows : 3" in out
    assert "Written to Postgres" in out


def test_a_dry_run_persists_nothing(monkeypatch, tmp_path, capsys):
    import main as orchestrator

    wb = tmp_path / "wb.xlsx"
    wb.write_bytes(b"stub")
    seen: dict = {}

    def fake(path, persist=True):
        seen["persist"] = persist
        return _frames()

    monkeypatch.setattr(orchestrator, "ingest_market_lines", fake)
    assert script.main(["--xlsx", str(wb), "--dry-run"]) == 0
    assert seen["persist"] is False, "--dry-run must not persist"
    assert "nothing was written" in capsys.readouterr().out.lower()


def test_a_missing_workbook_is_named_rather_than_tracebacked(tmp_path, capsys):
    """
    The likeliest failure and the cheapest to report. Checked before anything
    is imported or connected, so it still works where the dependencies are
    not installed.
    """
    missing = tmp_path / "nope.xlsx"
    assert script.main(["--xlsx", str(missing)]) == 2
    err = capsys.readouterr().err
    assert str(missing) in err
    assert "never downloads or scrapes" in err


def test_writing_nothing_is_not_reported_as_success(
    monkeypatch, tmp_path, capsys,
):
    """
    THE OUTCOME THAT LOOKS LIKE SUCCESS AND IS NOT. Two empty frames write no
    rows, so step [2] would still refuse afterwards — and an operator who saw
    exit 0 would believe the blocker was cleared and go looking somewhere else.
    """
    import main as orchestrator

    wb = tmp_path / "wb.xlsx"
    wb.write_bytes(b"stub")
    monkeypatch.setattr(
        orchestrator, "ingest_market_lines",
        lambda path, persist=True: (pd.DataFrame(), pd.DataFrame()),
    )
    assert script.main(["--xlsx", str(wb)]) == 1
    assert "WROTE NOTHING" in capsys.readouterr().out


def test_a_loader_failure_suggests_the_two_things_that_cause_it(
    monkeypatch, tmp_path, capsys,
):
    """A worksheet name — one of the two ways a different export variant
    fails. Reported, not raised at an operator."""
    import main as orchestrator

    wb = tmp_path / "wb.xlsx"
    wb.write_bytes(b"stub")

    def explode(path, persist=True):
        raise ValueError("Worksheet named 'NBA-2025-26-TEAM' not found")

    monkeypatch.setattr(orchestrator, "ingest_market_lines", explode)
    assert script.main(["--xlsx", str(wb)]) == 1
    err = capsys.readouterr().err
    assert "--sheet" in err


# --- the advice must match the fault --------------------------------------
#
# MEASURED ON A REAL RUN, 2026-10-24. The first version printed the workbook
# advice for every failure, so a wrong database password was answered with
# "If this names a worksheet, pass --sheet". The operator goes looking at
# their spreadsheet while the fault is in a connection string.


def test_a_database_failure_is_not_blamed_on_the_workbook(
    monkeypatch, tmp_path, capsys,
):
    """
    THE EXACT PRODUCTION FAILURE. An unsubstituted password placeholder
    reaches Postgres and is rejected like any wrong password; the remedy is
    $DATABASE_URL, and nothing about a worksheet or a column.
    """
    import main as orchestrator

    wb = tmp_path / "wb.xlsx"
    wb.write_bytes(b"stub")

    class OperationalError(Exception):
        """Stands in for sqlalchemy.exc.OperationalError by module."""
    OperationalError.__module__ = "sqlalchemy.exc"

    def explode(path, persist=True):
        # TEST-NET-3 (RFC 5737), not the real endpoint from the log. A test
        # fixture is a bad place to keep a production address. cubic, PR #23.
        raise OperationalError(
            'connection failed: connection to server at "203.0.113.10", port '
            '27751 failed: FATAL:  password authentication failed for user '
            '"postgres"'
        )

    monkeypatch.setattr(orchestrator, "ingest_market_lines", explode)
    assert script.main(["--xlsx", str(wb)]) == 1
    err = capsys.readouterr().err
    assert "DATABASE_URL" in err
    assert "placeholder" in err
    assert "--sheet" not in err, "a connection error is not a worksheet problem"
    assert "WORKSHEET" not in err, "nor a worksheet problem"


def _exc(name: str, module: str) -> type[Exception]:
    """An exception class with a given name and module, as the classifier sees
    it — so the branches can be covered without importing SQLAlchemy."""
    cls = type(name, (Exception,), {})
    cls.__module__ = module
    return cls


def test_the_classifier_matches_each_fault_to_its_own_remedy():
    """
    Asserted on `advice_for` directly, so each branch is covered without one
    subprocess-shaped test per branch. Classified on the exception's MODULE
    and NAME, never its message: psycopg's wording is version- and
    locale-dependent, and this repository has been bitten before by a check
    that read prose.
    """
    op = _exc("OperationalError", "sqlalchemy.exc")
    iface = _exc("InterfaceError", "psycopg")
    cfg = _exc("DatabaseConfigError", "src.db.session")

    assert script.advice_for(op("nope")) == script._DB_UNREACHABLE_ADVICE
    assert script.advice_for(iface("nope")) == script._DB_UNREACHABLE_ADVICE
    # UNSET is not WRONG: pointing at a password that does not exist yet
    # sends the operator to check nothing.
    assert script.advice_for(cfg("none configured")) == script._DB_UNSET_ADVICE
    assert script.advice_for(
        ValueError("Worksheet named 'X' not found")
    ) == script._SHEET_ADVICE
    assert script.advice_for(KeyError("GAME-ID")) == script._COLUMN_ADVICE
    # Nothing recognisable names the possibilities rather than guessing one.
    assert script.advice_for(RuntimeError("boom")) == script._GENERIC_ADVICE


def test_a_rejected_row_is_not_blamed_on_the_connection():
    """
    A DataError is a value the column's type refused and an IntegrityError is
    a constraint the rows violated. Both come from the WORKBOOK'S CONTENTS,
    reached only after the connection succeeded — so answering them with
    "check $DATABASE_URL" is the same misdirection in the other direction.
    cubic, PR #23.
    """
    for name in ("DataError", "IntegrityError"):
        cls = _exc(name, "sqlalchemy.exc")
        advice = script.advice_for(cls("value too long for type character(3)"))
        assert advice == script._ROW_ADVICE
        assert "DATABASE_URL" not in " ".join(advice)


def test_an_unknown_database_error_is_not_guessed():
    """
    The branches are an ALLOWLIST. A ProgrammingError — a schema the
    migrations have not applied — is confidently neither a connection fault
    nor a bad row, so it must fall through to the advice that names the
    possibilities instead of picking the wrong one.
    """
    cls = _exc("ProgrammingError", "sqlalchemy.exc")
    assert script.advice_for(cls('relation "team_game_stats" does not exist')) \
        == script._GENERIC_ADVICE


def test_the_database_advice_names_the_two_ways_the_url_is_wrong():
    """
    Both were hit for real tonight: an unsubstituted `...` placeholder, and
    the private `postgres.railway.internal` host, which does not resolve from
    outside the platform.
    """
    lines = " ".join(script._DB_UNREACHABLE_ADVICE)
    assert "placeholder" in lines
    assert "postgres.railway.internal" in lines


def test_the_advice_does_not_restate_the_loaders_required_columns():
    """
    `load_bigdataball_workbook` already raises "missing required columns
    (GAME-ID / DATE / TEAM / VENUE)" and that message is printed verbatim one
    line above this advice. The first version of the advice listed three of
    the four — it omitted VENUE — which is what a second reader of someone
    else's fact eventually does. So the advice must point AT that error, not
    carry its own copy of the list.

    ASSERTED ON THE ADVICE STRINGS, not the file's source. The first version
    of this test read the whole module and tripped over the COMMENT that
    explains why the columns are not restated — a source assertion satisfied
    by prose about the code, which is the trap this repository has hit five
    times. What the operator reads is the tuples.
    """
    advice = " ".join(
        script._COLUMN_ADVICE + script._GENERIC_ADVICE + script._ROW_ADVICE
    )
    for column in ("GAME-ID", "VENUE"):
        assert column not in advice, (
            f"{column} is restated in the advice; the loader's own error "
            f"already names every required column"
        )
    assert "error above" in " ".join(script._COLUMN_ADVICE)


def test_the_sheet_override_lands_where_the_default_lives(
    monkeypatch, tmp_path,
):
    """
    `sheet_name` is a default on `load_bigdataball_workbook`, not a parameter
    step [2] threads through, so an override has to be applied where the
    default lives rather than passed down.
    """
    import main as orchestrator
    from src.ingestion import bigdataball

    original = bigdataball.DEFAULT_SHEET
    try:
        wb = tmp_path / "wb.xlsx"
        wb.write_bytes(b"stub")
        monkeypatch.setattr(
            orchestrator, "ingest_market_lines",
            lambda path, persist=True: _frames(),
        )
        _stub_verification(monkeypatch)
        assert script.main(["--xlsx", str(wb), "--sheet", "MY-TAB"]) == 0
        assert bigdataball.DEFAULT_SHEET == "MY-TAB"
    finally:
        bigdataball.DEFAULT_SHEET = original


def test_it_never_reaches_an_odds_api():
    """
    Step [2] only. A full `main.py` run reaches step [3] once the ingest
    succeeds and spends a slate's worth of PropLine quota; this must not.
    """
    from pathlib import Path

    source = Path("scripts/ingest_bigdataball.py").read_text(encoding="utf-8")
    for forbidden in ("PropLineClient", "fetch_player_props", "requests.get",
                      "normalize_event_props", "run_slate"):
        assert forbidden not in source, f"{forbidden} would spend quota"


# --- the read-back, which is the only proof of a write -------------------


def test_an_empty_table_after_a_persisting_run_is_a_failure(
    monkeypatch, tmp_path, capsys,
):
    """
    THE FIVE-DAY BUG, PINNED. This script printed "Written to Postgres."
    unconditionally, directly under the PARSE counts. On 2026-10-25 an
    operator reported "Written to Postgres. 2,644 rows / 1,322 games" and the
    slates of 10-26, 10-27 and 10-28 each logged
    `team_game_stats query window -> 0 rows`. Both statements were true: 2,644
    rows were parsed, and none were in the table afterwards.

    Nothing in the path had asked the database what it held — and the upsert
    return values would not have helped, because they are `len(rows)`, the
    rows handed to the statement rather than the rows kept.
    """
    import main as orchestrator

    wb = tmp_path / "wb.xlsx"
    wb.write_bytes(b"stub")
    monkeypatch.setattr(
        orchestrator, "ingest_market_lines",
        lambda path, persist=True: _frames(),
    )
    _stub_verification(monkeypatch, {
        "team_game_stats": 0, "game_market_lines": 0, "player_game_logs": 0,
    })

    assert script.main(["--xlsx", str(wb)]) == 1

    captured = capsys.readouterr()
    assert "team_game_stats is EMPTY" in captured.err
    assert "Written to Postgres and read back" not in captured.out, (
        "success was claimed over an empty table"
    )


def test_the_verified_counts_are_printed_not_the_parsed_ones(
    monkeypatch, tmp_path, capsys,
):
    """
    The two numbers must be distinguishable. The parse said 3 rows; the table
    holds 2,644. Printing only one of them is how "2,644" came to be read as
    a confirmation.
    """
    import main as orchestrator

    wb = tmp_path / "wb.xlsx"
    wb.write_bytes(b"stub")
    monkeypatch.setattr(
        orchestrator, "ingest_market_lines",
        lambda path, persist=True: _frames(),
    )
    _stub_verification(monkeypatch, {
        "team_game_stats": 2644, "game_market_lines": 2644,
        "player_game_logs": 17,
    })

    assert script.main(["--xlsx", str(wb)]) == 0
    out = capsys.readouterr().out
    assert "team_game_stats rows : 3" in out          # parsed
    assert "2,644 row(s)" in out                       # verified
    assert "read back" in out
    assert "player_game_logs" in out


def test_a_verification_that_cannot_connect_does_not_claim_success(
    monkeypatch, tmp_path, capsys,
):
    """A read-back that fails leaves the write UNVERIFIED, which is not the
    same as verified-empty and is certainly not success."""
    import main as orchestrator
    import src.db.repository as repository

    wb = tmp_path / "wb.xlsx"
    wb.write_bytes(b"stub")
    monkeypatch.setattr(
        orchestrator, "ingest_market_lines",
        lambda path, persist=True: _frames(),
    )

    def boom():
        raise OSError("connection refused")

    monkeypatch.setattr(repository, "ingested_row_counts", boom)

    assert script.main(["--xlsx", str(wb)]) == 1
    captured = capsys.readouterr()
    assert "COULD NOT VERIFY" in captured.err
    assert "Written to Postgres" not in captured.out


def test_a_dry_run_never_reaches_the_verification(monkeypatch, tmp_path):
    """A dry run writes nothing, so there is nothing to read back — and it
    must not need a database to say so."""
    import main as orchestrator
    import src.db.repository as repository

    wb = tmp_path / "wb.xlsx"
    wb.write_bytes(b"stub")
    monkeypatch.setattr(
        orchestrator, "ingest_market_lines",
        lambda path, persist=True: _frames(),
    )

    called = {"n": 0}

    def counted():
        called["n"] += 1
        return {"team_game_stats": 0}

    monkeypatch.setattr(repository, "ingested_row_counts", counted)

    assert script.main(["--xlsx", str(wb), "--dry-run"]) == 0
    assert called["n"] == 0, "a dry run reached for the database"


def test_a_run_whose_own_keys_are_not_in_the_table_fails(
    monkeypatch, tmp_path, capsys,
):
    """
    A ROW COUNT IS NOT PROOF OF THIS RUN'S WRITE. The table holds 2,644 rows
    from some earlier ingest and not one of the four keys this run parsed —
    which is the normal shape of the failure after the first successful
    ingest, and exactly what the absolute check passed on. Flagged in review
    on 2026-10-29.
    """
    import main as orchestrator
    import src.db.repository as repository

    wb = tmp_path / "wb.xlsx"
    wb.write_bytes(b"stub")
    monkeypatch.setattr(
        orchestrator, "ingest_market_lines",
        lambda path, persist=True: _keyed_frames(),
    )
    _stub_verification(monkeypatch, {
        "team_game_stats": 2644, "game_market_lines": 2644,
        "player_game_logs": 0,
    })
    monkeypatch.setattr(
        repository, "ingested_keys_present",
        lambda **kw: {"team_game_stats": 0, "game_market_lines": 0},
    )

    assert script.main(["--xlsx", str(wb)]) == 1
    captured = capsys.readouterr()
    assert "0 of this run's 4 team-game key(s) present" in captured.out
    assert "are NOT in the table" in captured.err
    assert "Written to Postgres and read back" not in captured.out


def test_market_lines_that_do_not_land_are_a_failure_too(
    monkeypatch, tmp_path, capsys,
):
    """
    THE HALF THAT USED TO BE UNCHECKED. Only `team_game_stats` was required,
    so a workbook whose spread columns all failed to write still exited 0 —
    and `game_market_lines` is the table the EV path abstains without.
    """
    import main as orchestrator
    import src.db.repository as repository

    wb = tmp_path / "wb.xlsx"
    wb.write_bytes(b"stub")
    monkeypatch.setattr(
        orchestrator, "ingest_market_lines",
        lambda path, persist=True: _keyed_frames(),
    )
    _stub_verification(monkeypatch, {
        "team_game_stats": 4, "game_market_lines": 0, "player_game_logs": 0,
    })
    monkeypatch.setattr(
        repository, "ingested_keys_present",
        lambda **kw: {"team_game_stats": 4, "game_market_lines": 0},
    )

    assert script.main(["--xlsx", str(wb)]) == 1
    assert "game_market_lines" in capsys.readouterr().err


def test_the_delta_separates_a_re_run_from_a_write_that_vanished(
    monkeypatch, tmp_path, capsys,
):
    """
    Both leave the table the same size. A re-ingest of the same workbook is
    all updates and legitimately adds nothing, so the delta alone cannot
    decide — which is why the keys are asked for as well, and why the delta
    is reported rather than acted on.
    """
    import main as orchestrator

    wb = tmp_path / "wb.xlsx"
    wb.write_bytes(b"stub")
    monkeypatch.setattr(
        orchestrator, "ingest_market_lines",
        lambda path, persist=True: _keyed_frames(),
    )
    _stub_verification(monkeypatch, {
        "team_game_stats": 4, "game_market_lines": 4, "player_game_logs": 0,
    })

    assert script.main(["--xlsx", str(wb)]) == 0
    out = capsys.readouterr().out
    # Baseline and post-write counts come from the same stub, so a re-run's
    # delta is what is printed: zero, beside four keys present.
    assert "(+0 this run)" in out
    assert "4 of this run's 4 team-game key(s) present" in out
    assert "Written to Postgres and read back" in out


def test_the_key_helper_reads_the_frame_rather_than_assuming_it():
    """
    No columns, no keys — and then the row count is the only check available.
    Asserted on the helper because the alternative is a KeyError inside a
    verification step that exists to avoid false confidence.

    ONE DEFINITION, in `src/db/repository.py`: both this script and
    `scripts/load_scraper_panel.py` verify their writes by this key, and a
    second reading of it would be the defect this repository removes.
    """
    from src.db.repository import team_game_keys

    assert team_game_keys(pd.DataFrame()) == []
    assert team_game_keys(pd.DataFrame({"nba_game_id": ["1"]})) == []
    assert team_game_keys(None) == []
    frame = pd.DataFrame({
        "nba_game_id": ["1", "1", "2"], "team_abbr": ["LAL", "LAL", "GSW"],
    })
    assert team_game_keys(frame) == [("1", "LAL"), ("2", "GSW")]
