"""
tests/test_market_frames_fallback.py — step [2] must not die on a missing file.

THE BLOCKER. `main.ingest_market_lines` reads the licensed BigDataBall
workbook and a missing path raised `FileNotFoundError`, which the
orchestrator's outer handler turned into a FAILED `pipeline_runs` row and
exit 1 — the WHOLE slate, not a degraded one. The image excludes `data/` and
`*.xlsx` deliberately, because a licensed third-party export does not belong
in an image layer, so a deployed container had no workbook and every
scheduled run died at 09:00 PT having looked healthy at boot.

THE DATA WAS NEVER MISSING; ONLY THE FILE WAS. Every run that finds a workbook
upserts its contents into `team_game_stats` and `game_market_lines`, and
Postgres survives a redeploy. `main.resolve_market_frames` reads them back.

What is asserted here, in the order it matters:

  1. the column names the loaders select are the ones the consuming layers
     already require — the whole fallback rests on that and nothing else
     checks it;
  2. no closing-line column is ever selected;
  3. the three branches: workbook, database, and the refusal when there is
     neither;
  4. staleness is measured, because the fix trades a loud failure for a quiet
     one and that quiet one has to be visible.

RESEARCH_ONLY project. No odds, no wager.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import main

REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# 1 + 2. the column contracts the fallback rests on
# ---------------------------------------------------------------------------

def test_the_team_loader_selects_what_the_consuming_layers_require():
    """
    The fallback works for exactly one reason: `team_strength.
    _normalize_team_games` and `defense.REQUIRED_TEAM_COLS` already read the
    DATABASE column names, so a frame built straight from the ORM is
    interchangeable with the workbook's. If either layer's requirements move,
    the fallback silently produces a narrower matrix — the layers omit what
    they cannot build — so the requirement is asserted rather than assumed.
    """
    from src.db.repository import TEAM_GAME_STAT_COLS
    from src.features.defense import REQUIRED_TEAM_COLS

    missing = set(REQUIRED_TEAM_COLS) - set(TEAM_GAME_STAT_COLS)
    assert not missing, f"the team loader does not select {sorted(missing)}"

    # Elo's own requirements, via the aliases it accepts.
    for needed in ("nba_game_id", "game_date", "team_abbr", "points"):
        assert needed in TEAM_GAME_STAT_COLS

    # And the per-100 rates defence computes need the box-score counts.
    for needed in ("fga", "oreb", "tov", "fta", "poss"):
        assert needed in TEAM_GAME_STAT_COLS, (
            f"{needed} is not selected, so build_team_defense falls back to a "
            f"different possession estimate than a workbook run uses"
        )


def test_the_market_loader_selects_exactly_the_pregame_columns():
    from src.db.repository import MARKET_LINE_PREGAME_COLS
    from src.features.market_context import PREGAME_SOURCE_COLS

    assert MARKET_LINE_PREGAME_COLS == PREGAME_SOURCE_COLS


def test_no_closing_line_column_is_ever_selected():
    """
    `game_market_lines` stores closing_spread, closing_total, the halftime line
    and three line-movement columns — every one of them information from after
    the game was priced. `attach_market_context` happens to project down to the
    pregame columns, so a `SELECT *` would not leak TODAY; selecting them
    anyway would make that projection the only thing between a closing line and
    a feature matrix.
    """
    from src.db.repository import MARKET_LINE_PREGAME_COLS, TEAM_GAME_STAT_COLS
    from src.features.market_context import CLOSING_ONLY_COLS

    for cols in (MARKET_LINE_PREGAME_COLS, TEAM_GAME_STAT_COLS):
        assert not (set(cols) & CLOSING_ONLY_COLS), sorted(set(cols) & CLOSING_ONLY_COLS)


def test_the_loaders_bound_the_window_on_both_sides(monkeypatch):
    """
    The UPPER bound is a leakage guard: a backfill for an old slate must not
    load games played after it. Asserted on the SQL the loader builds, because
    a loader that accepted `slate_date` and ignored it is a defect this
    repository has already shipped once — `load_player_panel` did exactly that.
    """
    import src.db.repository as repo

    captured: list[str] = []

    class _FakeSession:
        def execute(self, stmt):
            captured.append(str(stmt.compile(compile_kwargs={"literal_binds": True})))

            class _R:
                @staticmethod
                def scalars():
                    class _S:
                        @staticmethod
                        def all():
                            return []
                    return _S()
            return _R()

    from contextlib import contextmanager

    @contextmanager
    def _scope():
        yield _FakeSession()

    monkeypatch.setattr(repo, "session_scope", _scope)
    repo.load_team_game_stats(slate_date="2026-01-15", lookback_days=30)
    repo.load_game_market_lines(slate_date="2026-01-15", lookback_days=30)

    assert len(captured) == 2
    for sql in captured:
        assert "2026-01-15" in sql, f"the upper bound is not in the query: {sql}"
        assert "2025-12-16" in sql, f"the lower bound is not in the query: {sql}"
    # And the market query takes VALID rows only.
    assert "VALID" in captured[1]


# ---------------------------------------------------------------------------
# 3. the three branches of resolve_market_frames
# ---------------------------------------------------------------------------

def _team_frame(dates: list[str]) -> pd.DataFrame:
    from src.db.repository import TEAM_GAME_STAT_COLS

    rows = []
    for i, d in enumerate(dates):
        row = {c: None for c in TEAM_GAME_STAT_COLS}
        row.update({
            "nba_game_id": f"00226000{i:02d}", "game_date": pd.Timestamp(d).date(),
            "team_abbr": "LAL", "opponent_abbr": "BOS", "points": 110,
            "is_home": True, "is_neutral_site": False,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def _market_frame(n: int = 1) -> pd.DataFrame:
    from src.db.repository import MARKET_LINE_PREGAME_COLS

    return pd.DataFrame([{
        "nba_game_id": f"00226000{i:02d}", "game_date": pd.Timestamp("2026-01-10").date(),
        "team_abbr": "LAL", "opening_spread": -3.5, "opening_total": 224.5,
    } for i in range(n)], columns=list(MARKET_LINE_PREGAME_COLS))


def _stub_loaders(monkeypatch, team: pd.DataFrame, market: pd.DataFrame):
    import src.db.repository as repo

    monkeypatch.setattr(repo, "load_team_game_stats", lambda **k: team)
    monkeypatch.setattr(repo, "load_game_market_lines", lambda **k: market)


def test_a_present_workbook_is_still_read_and_still_persists(monkeypatch, tmp_path):
    """The fallback must not change the path that works."""
    book = tmp_path / "book.xlsx"
    book.write_bytes(b"stand-in; ingest_market_lines is stubbed")
    seen: list[bool] = []

    def _fake_ingest(path, persist=True):
        seen.append(persist)
        return _team_frame(["2026-01-10"]), _market_frame()

    monkeypatch.setattr(main, "ingest_market_lines", _fake_ingest)
    out = main.resolve_market_frames(book, persist=True, slate_date="2026-01-12")

    assert out["source"] == "workbook"
    assert seen == [True], "the workbook path no longer persists what it reads"
    assert not out["team_games"].empty and not out["market_lines"].empty


def test_a_missing_workbook_reads_the_database_instead_of_failing(
    monkeypatch, tmp_path, caplog
):
    """
    THE FIX, as a test. Before this, the next line of the slate was a FAILED
    pipeline_runs row.
    """
    import logging

    _stub_loaders(monkeypatch, _team_frame(["2026-01-10", "2026-01-11"]), _market_frame(2))
    with caplog.at_level(logging.WARNING):
        out = main.resolve_market_frames(
            tmp_path / "absent.xlsx", persist=True, slate_date="2026-01-12",
        )

    assert out["source"] == "database"
    assert len(out["team_games"]) == 2
    assert len(out["market_lines"]) == 2
    assert "NO BIGDATABALL WORKBOOK" in caplog.text, (
        "the fallback happened silently; which source produced the frames is "
        "the difference between ingesting and reading back"
    )


def test_no_workbook_and_an_empty_database_still_refuses(monkeypatch, tmp_path):
    """
    NOT A SILENT DEGRADATION. With neither, the Elo, MKT_* and DEF_* layers are
    omitted — 11 of the 38 columns a seeded contract names — and every row
    abstains. That abstention does name the missing columns, but the cause is
    upstream of them, so step [2] says so instead.
    """
    from src.db.repository import MARKET_LINE_PREGAME_COLS, TEAM_GAME_STAT_COLS

    _stub_loaders(
        monkeypatch,
        pd.DataFrame(columns=list(TEAM_GAME_STAT_COLS)),
        pd.DataFrame(columns=list(MARKET_LINE_PREGAME_COLS)),
    )
    with pytest.raises(main.MarketFramesUnavailable) as exc:
        main.resolve_market_frames(tmp_path / "absent.xlsx", slate_date="2026-01-12")

    message = str(exc.value)
    assert "team_game_stats" in message and "game_market_lines" in message
    assert "BIGDATABALL_XLSX" in message, "the refusal does not say how to fix it"


def test_one_empty_frame_degrades_with_a_warning_rather_than_refusing(
    monkeypatch, tmp_path, caplog
):
    """
    An operator who trained without the workbook has a 27-column contract that
    a partial frame satisfies, and the builder already omits a layer whose
    input is absent. So this is a warning, not a refusal — but it is not
    silence either.
    """
    import logging

    from src.db.repository import MARKET_LINE_PREGAME_COLS

    _stub_loaders(
        monkeypatch,
        _team_frame(["2026-01-10"]),
        pd.DataFrame(columns=list(MARKET_LINE_PREGAME_COLS)),
    )
    with caplog.at_level(logging.WARNING):
        out = main.resolve_market_frames(tmp_path / "absent.xlsx", slate_date="2026-01-12")

    assert out["source"] == "database"
    assert "PARTIAL MARKET DATA" in caplog.text


def test_the_orchestrator_calls_the_resolver_not_the_raw_ingest():
    """
    AST-walked. The whole blocker was that step [2] called
    `ingest_market_lines` directly, and a docstring saying otherwise has fooled
    this repository five times.
    """
    import ast

    tree = ast.parse((REPO / "main.py").read_text(encoding="utf-8"))
    entry = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    called = {
        n.func.id for n in ast.walk(entry)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "resolve_market_frames" in called
    assert "ingest_market_lines" not in called, (
        "main() calls ingest_market_lines directly again, so a missing "
        "workbook fails the whole slate once more"
    )


# ---------------------------------------------------------------------------
# 4. the quiet failure the fix introduces
# ---------------------------------------------------------------------------

def test_stale_market_data_is_reported(monkeypatch, caplog):
    """
    THE COST OF THE FIX, MEASURED. Reading the frames back from Postgres
    removed a hard failure and put a quiet one in its place: the database holds
    whatever was last ingested, so a container running for weeks on one upload
    computes Elo and opponent-defence features from games that stopped before
    the rows it is scoring. The layers attach, the columns are present, the
    contract check passes, and the numbers are simply out of date.
    """
    import logging

    monkeypatch.delenv(main.ENV_MAX_MARKET_LAG, raising=False)
    stale = _team_frame(["2025-11-01"])
    with caplog.at_level(logging.WARNING):
        out = main.market_frames_freshness(stale, "2026-01-12", source="database")

    assert out["status"] == "STALE"
    assert out["lag_days"] > out["max_lag_days"]
    assert out["newest_game_date"] == "2025-11-01"
    assert "MARKET DATA IS STALE" in caplog.text


def test_fresh_market_data_is_not_reported_as_stale(monkeypatch):
    monkeypatch.delenv(main.ENV_MAX_MARKET_LAG, raising=False)
    out = main.market_frames_freshness(
        _team_frame(["2026-01-11"]), "2026-01-12", source="workbook",
    )
    assert out["status"] == "OK"
    assert out["lag_days"] == 1
    assert out["source"] == "workbook"


def test_a_workbook_run_is_measured_too_not_only_the_fallback():
    """A stale workbook on disk is the same defect with a different cause, and
    only reading the frame itself catches both."""
    import ast

    tree = ast.parse((REPO / "main.py").read_text(encoding="utf-8"))
    entry = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    calls = [
        n for n in ast.walk(entry)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id == "market_frames_freshness"
    ]
    assert len(calls) == 1, "freshness is measured once"
    # Passed the frame, not a source-dependent branch.
    assert any(isinstance(a, ast.Name) and a.id == "team_games_df" for a in calls[0].args)

    # UNCONDITIONALLY. `len(calls) == 1` does not say that: wrapping the call
    # in `if frames["source"] == "database":` leaves exactly one call and one
    # `team_games_df` argument, and that mutation survived this test until the
    # check below was added. So no enclosing `if` may contain it.
    for node in ast.walk(entry):
        if not isinstance(node, ast.If):
            continue
        nested = [
            n for n in ast.walk(node)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id == "market_frames_freshness"
        ]
        assert not nested, (
            "market_frames_freshness is inside a conditional, so a workbook "
            "run can skip it -- and a stale workbook on disk is the same "
            "defect as a database nobody has re-ingested into"
        )


def test_the_limit_falls_back_rather_than_raising_on_a_typo(monkeypatch):
    monkeypatch.setenv(main.ENV_MAX_MARKET_LAG, "not-a-number")
    out = main.market_frames_freshness(
        _team_frame(["2026-01-11"]), "2026-01-12", source="database",
    )
    assert out["max_lag_days"] == main.DEFAULT_MAX_MARKET_LAG_DAYS


def test_an_empty_or_undated_frame_is_named_not_called_fresh():
    """An empty frame has no newest game, and reporting lag 0 would read as
    perfectly fresh."""
    assert main.market_frames_freshness(
        pd.DataFrame(), "2026-01-12", source="database")["status"] == "EMPTY"
    undated = _team_frame(["2026-01-11"])
    undated["game_date"] = None
    assert main.market_frames_freshness(
        undated, "2026-01-12", source="database")["status"] == "UNDATED"


# ---------------------------------------------------------------------------
# 5. the only assertion that proves the whole thing works
# ---------------------------------------------------------------------------

def test_a_database_shaped_frame_builds_the_elo_market_and_defence_columns():
    """
    THE ONE TEST THAT MATTERS, and everything above it is branch logic.

    The fallback rests on a single claim: a frame built straight from the ORM
    is interchangeable with the workbook's, because `team_strength.
    _normalize_team_games`, `defense.REQUIRED_TEAM_COLS` and
    `market_context.PREGAME_SOURCE_COLS` all already read the DATABASE column
    names. The column-contract tests above check that claim against the
    constants; this one checks it against the builder, which is the only thing
    that can be wrong in a way the constants agree about.

    It matters because the failure is silent: `build_feature_matrix` OMITS a
    layer whose input it cannot read rather than raising, so a fallback frame
    the layers quietly rejected would produce a 27-column matrix, and the only
    symptom would be every row abstaining with a list of missing column names
    at 09:00 PT.
    """
    from src.db.repository import MARKET_LINE_PREGAME_COLS, TEAM_GAME_STAT_COLS
    from src.features.builder import build_feature_matrix

    # Two teams over six alternating games: enough for Elo to move off its
    # initial rating and for the defence layer to have an opponent to rate.
    games, team_rows, market_rows = [], [], []
    for i in range(6):
        gid = f"002260000{i}"
        date = pd.Timestamp("2026-01-02") + pd.Timedelta(days=2 * i)
        games.append((gid, date))
        for team, opp, home, pts in (("LAL", "BOS", True, 112 + i), ("BOS", "LAL", False, 104 + i)):
            row = {c: None for c in TEAM_GAME_STAT_COLS}
            row.update({
                "nba_game_id": gid, "game_date": date.date(), "team_abbr": team,
                "opponent_abbr": opp, "is_home": home, "is_neutral_site": False,
                "points": pts, "fg": 40, "fga": 88, "fg3": 12, "fg3a": 33,
                "ft": 18, "fta": 22, "oreb": 10, "dreb": 33, "reb": 43,
                "ast": 25, "stl": 7, "blk": 5, "tov": 13, "pf": 19,
                "poss": 99.5, "pace": 99.5, "off_eff": 112.0, "def_eff": 104.0,
            })
            team_rows.append(row)
            market_rows.append({
                "nba_game_id": gid, "game_date": date.date(), "team_abbr": team,
                "opening_spread": -3.5 if home else 3.5, "opening_total": 224.5,
            })

    # PROJECTED ONTO THE LOADER'S OWN COLUMN LISTS, not onto the dicts built
    # above. The first version of this test built the frames from literal dicts
    # and then used the constants only to pad missing keys, so dropping
    # "points" from TEAM_GAME_STAT_COLS left this test green -- it was
    # exercising the dict, not the loader's selection. Projecting means the
    # frames carry exactly what `load_team_game_stats` and
    # `load_game_market_lines` would return and nothing else.
    team_games = pd.DataFrame(team_rows)[list(TEAM_GAME_STAT_COLS)]
    market_lines = pd.DataFrame(market_rows)[list(MARKET_LINE_PREGAME_COLS)]

    panel = pd.DataFrame([{
        "PLAYER_ID": "1", "PLAYER_NAME": "A Player", "GAME_ID": gid,
        "GAME_DATE": date, "SEASON": "2025-26", "TEAM_ABBREVIATION": "LAL",
        "OPPONENT_ABBREVIATION": "BOS", "IS_HOME": True, "IS_NEUTRAL_SITE": False,
        "MIN": 34.0, "PTS": 24.0 + i, "REB": 7.0, "AST": 5.0,
    } for i, (gid, date) in enumerate(games)])

    matrix = build_feature_matrix(
        panel, team_games=team_games, market_lines=market_lines,
    )

    # One column from each of the three layers the workbook alone used to feed.
    elo = [c for c in matrix.columns if "ELO" in c.upper()]
    market = [c for c in matrix.columns if c.startswith("MKT_")]
    defence = [c for c in matrix.columns if c.startswith("DEF_")]
    assert elo, "the Elo layer was omitted: the DB frame did not satisfy it"
    assert market, "the market-context layer was omitted"
    assert defence, "the opponent-defence layer was omitted"

    # And no closing-line information rode along.
    from src.features.market_context import assert_no_closing_lines

    assert_no_closing_lines(matrix.columns)
