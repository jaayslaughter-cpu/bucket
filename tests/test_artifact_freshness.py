"""
A2 — the artifact's own age, and the refusal that rests on it.

WHAT THIS GUARDS. Every other check in this project asks whether the fitted
artifact is USABLE. None asked whether it was CURRENT, and the sidecar has
recorded `saved_at_utc` since the first training run with no reader. An
unattended worker that keeps publishing from a dead retrain looks exactly like
one that is working, which is why the refusal is a refusal and not a warning.

All sidecars below are TEST FIXTURES. No real training result is asserted, and
nothing here fetches odds, scores a slate or sizes anything.
"""

from __future__ import annotations

import ast
import inspect
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.models import artifact_freshness as af
from src.quant.dfs_payouts import ProbabilitySource
from src.quant.publication_gate import (
    PUBLISH_ALLOWED,
    PUBLISH_WITHHELD,
    artifact_freshness_gate,
    first_refusal,
)

NOW = datetime(2026, 10, 11, 12, 0, tzinfo=timezone.utc)


def sidecar(
    *,
    saved_days_ago: float | None = 2.0,
    train_end: str | None = "2026-01-13",
    **extra,
) -> dict:
    out: dict = {"train_row_count": 200_678}
    if saved_days_ago is not None:
        stamp = NOW.timestamp() - saved_days_ago * 86400.0
        out[af.SAVED_AT_KEY] = datetime.fromtimestamp(stamp, timezone.utc).isoformat()
    if train_end is not None:
        out[af.TRAIN_END_KEY] = train_end
    out.update(extra)
    return out


# --- the two axes -------------------------------------------------------


def test_a_recent_artifact_trained_close_to_our_newest_game_is_ok():
    fresh = af.artifact_freshness(
        sidecar(saved_days_ago=2.0, train_end="2026-01-13"),
        now=NOW,
        newest_game_date="2026-01-20",
        market="PTS",
    )
    assert fresh["status"] == af.OK
    assert fresh["artifact_age_days"] == pytest.approx(2.0, abs=0.01)
    assert fresh["training_lag_days"] == 7
    assert set(fresh["checks"]) == {"artifact_age", "training_recency"}
    assert fresh["reasons"] == ()
    assert artifact_freshness_gate(fresh).status == PUBLISH_ALLOWED


def test_an_artifact_older_than_the_limit_is_stale_and_withheld():
    fresh = af.artifact_freshness(
        sidecar(saved_days_ago=af.DEFAULT_MAX_ARTIFACT_AGE_DAYS + 1), now=NOW
    )
    assert fresh["status"] == af.STALE
    verdict = artifact_freshness_gate(fresh)
    assert verdict.status == PUBLISH_WITHHELD
    # The refusal must carry the number AND the knob, or the operator has to go
    # looking for both.
    assert "46.0 day(s) ago" in verdict.reason
    assert af.ENV_MAX_ARTIFACT_AGE in verdict.reason
    assert verdict.artifact_age_days == pytest.approx(46.0, abs=0.01)


def test_a_fresh_file_off_a_stale_training_frame_is_still_stale():
    """
    THE CASE FILE AGE ALONE CANNOT SEE, and the reason there are two axes.

    A retrain that runs nightly but keeps reading the same frame rewrites
    `saved_at_utc` forever while the model learns nothing. On file age this
    artifact is two days old; it has never seen three months of games.
    """
    fresh = af.artifact_freshness(
        sidecar(saved_days_ago=2.0, train_end="2026-01-13"),
        now=NOW,
        newest_game_date="2026-04-13",
    )
    assert fresh["status"] == af.STALE
    assert fresh["artifact_age_days"] == pytest.approx(2.0, abs=0.01)
    assert fresh["training_lag_days"] == 90
    verdict = artifact_freshness_gate(fresh)
    assert verdict.status == PUBLISH_WITHHELD
    assert "90 day(s) of games the model never saw" in verdict.reason


def test_the_offseason_does_not_fire_a_false_alarm():
    """
    Training recency is measured against THE DATA WE HOLD, not against today.

    Measured against today, every artifact goes stale each July and the alarm
    fires for four months — which is how a staleness alarm gets muted. Here the
    season ended in June and today is October: both numbers are old together,
    the lag is zero, and the check is quiet.
    """
    fresh = af.artifact_freshness(
        sidecar(saved_days_ago=2.0, train_end="2026-06-18"),
        now=NOW,
        newest_game_date="2026-06-18",
    )
    assert fresh["status"] == af.OK
    assert fresh["training_lag_days"] == 0
    assert artifact_freshness_gate(fresh).allowed


# --- an unknown age is not a young one ----------------------------------


def test_a_sidecar_with_no_saved_at_is_undated_not_ok():
    fresh = af.artifact_freshness(sidecar(saved_days_ago=None), now=NOW)
    assert fresh["status"] == af.UNDATED
    assert fresh["artifact_age_days"] is None
    verdict = artifact_freshness_gate(fresh)
    assert verdict.status == PUBLISH_WITHHELD
    assert af.SAVED_AT_KEY in verdict.reason


def test_an_unparseable_saved_at_is_undated():
    fresh = af.artifact_freshness({af.SAVED_AT_KEY: "last Tuesday"}, now=NOW)
    assert fresh["status"] == af.UNDATED
    assert not artifact_freshness_gate(fresh).allowed


def test_no_sidecar_at_all_is_missing_and_withheld():
    fresh = af.artifact_freshness(None, now=NOW)
    assert fresh["status"] == af.MISSING
    assert fresh["checks"] == ()
    verdict = artifact_freshness_gate(fresh)
    assert verdict.status == PUBLISH_WITHHELD
    assert "unknown age" in verdict.reason.lower() or "cannot be established" in verdict.reason


def test_the_gate_withholds_on_no_freshness_at_all():
    assert artifact_freshness_gate(None).status == PUBLISH_WITHHELD
    assert artifact_freshness_gate({}).status == PUBLISH_WITHHELD


def test_an_ok_status_with_nothing_measured_is_not_a_pass():
    """
    The hole a status-only reader would leave.

    `status` defaults to OK and only a failing axis moves it, so a freshness
    dict on which NO axis ran reports OK. Reading that alone would turn
    "nothing was checked" into "everything passed" — the precise shape of
    defect this repository keeps finding.
    """
    hand_made = {"status": af.OK, "checks": (), "artifact_age_days": None}
    verdict = artifact_freshness_gate(hand_made)
    assert verdict.status == PUBLISH_WITHHELD
    assert "no freshness check actually ran" in verdict.reason


def test_recency_omitted_is_reported_as_not_measured_not_as_fine():
    fresh = af.artifact_freshness(sidecar(), now=NOW)  # no newest_game_date
    assert fresh["training_lag_days"] is None
    assert "training_recency" not in fresh["checks"]
    assert fresh["checks"] == ("artifact_age",)
    # Still publishable: one axis DID run and passed. The point is that the
    # dict says which.
    assert artifact_freshness_gate(fresh).allowed


def test_a_missing_train_end_says_so_without_claiming_recency_was_checked():
    fresh = af.artifact_freshness(
        sidecar(train_end=None), now=NOW, newest_game_date="2026-04-13"
    )
    assert "training_recency" not in fresh["checks"]
    assert fresh["training_lag_days"] is None
    assert any(af.TRAIN_END_KEY in r for r in fresh["reasons"])
    # The artifact's own age still carried the verdict.
    assert fresh["status"] == af.OK


# --- the market-grounded carve-out --------------------------------------


def test_a_market_grounded_probability_is_not_gated_on_our_model_age():
    """A de-vigged sharp price does not rest on our artifact, so its age is not
    a fact about it. Gating it would withhold a figure over something it never
    used."""
    stale = af.artifact_freshness(sidecar(saved_days_ago=999), now=NOW)
    assert stale["status"] == af.STALE
    verdict = artifact_freshness_gate(
        stale, probability_source=ProbabilitySource.SHARP_BENCHMARK
    )
    assert verdict.status == PUBLISH_ALLOWED
    assert verdict.checks == ("source_is_market_grounded",)


@pytest.mark.parametrize(
    "source",
    [ProbabilitySource.MODEL, ProbabilitySource.MIXED, ProbabilitySource.UNSPECIFIED],
)
def test_every_model_dependent_source_is_gated(source):
    stale = af.artifact_freshness(sidecar(saved_days_ago=999), now=NOW)
    assert not artifact_freshness_gate(stale, probability_source=source).allowed


# --- the limits are ours to decide, and pinned --------------------------


def test_the_default_limits_are_pinned():
    """
    Loosening a limit must fail a test rather than quietly reverse a
    conclusion. If a retrain cadence genuinely changes, change the constant AND
    this line, deliberately.
    """
    assert af.DEFAULT_MAX_ARTIFACT_AGE_DAYS == 45
    assert af.DEFAULT_MAX_TRAINING_LAG_DAYS == 21


def test_the_limits_are_env_overridable_within_bounds(monkeypatch):
    monkeypatch.setenv(af.ENV_MAX_ARTIFACT_AGE, "7")
    fresh = af.artifact_freshness(sidecar(saved_days_ago=10), now=NOW)
    assert fresh["status"] == af.STALE
    assert fresh["max_artifact_age_days"] == 7


def test_a_nonsense_limit_falls_back_rather_than_raising(monkeypatch):
    monkeypatch.setenv(af.ENV_MAX_ARTIFACT_AGE, "not-a-number")
    fresh = af.artifact_freshness(sidecar(saved_days_ago=2), now=NOW)
    assert fresh["max_artifact_age_days"] == af.DEFAULT_MAX_ARTIFACT_AGE_DAYS
    monkeypatch.setenv(af.ENV_MAX_ARTIFACT_AGE, "0")  # below the floor
    fresh = af.artifact_freshness(sidecar(saved_days_ago=2), now=NOW)
    assert fresh["max_artifact_age_days"] == af.DEFAULT_MAX_ARTIFACT_AGE_DAYS


# --- reading the sidecar off disk ---------------------------------------


def test_read_sidecar_returns_none_for_absent_and_corrupt(tmp_path):
    artifact = tmp_path / "xgboost_PTS.json"
    artifact.write_text("{}", encoding="utf-8")
    assert af.read_sidecar(artifact) is None  # no sidecar

    artifact.with_suffix(af.SIDECAR_SUFFIX).write_text("{not json", encoding="utf-8")
    assert af.read_sidecar(artifact) is None  # unreadable

    artifact.with_suffix(af.SIDECAR_SUFFIX).write_text("[1, 2]", encoding="utf-8")
    assert af.read_sidecar(artifact) is None  # not an object


def test_artifact_freshness_for_reads_a_real_sidecar(tmp_path):
    artifact = tmp_path / "xgboost_PTS.json"
    artifact.write_text("{}", encoding="utf-8")
    artifact.with_suffix(af.SIDECAR_SUFFIX).write_text(
        json.dumps(sidecar(saved_days_ago=100)), encoding="utf-8"
    )
    fresh = af.artifact_freshness_for(artifact, now=NOW, market="PTS")
    assert fresh["status"] == af.STALE
    assert fresh["market"] == "PTS"


# --- the contract this whole feature rests on ---------------------------


def test_the_sidecars_on_disk_actually_carry_the_key_this_reads():
    """
    THE KEY WAS VERIFIED, NOT ASSUMED — and this keeps it verified.

    `artifact_registry.build_registry_record` writes `registered_at_utc`, which
    looks like the right key and is NOT in any sidecar: it goes into a separate
    `model_registry.jsonl` that nothing writes and `resolve_model_artifact`
    deliberately does not read. The key training actually persists is
    `saved_at_utc`. If a change to the training path drops it, every artifact
    becomes UNDATED and every board is withheld — so the coupling is asserted
    rather than left to be discovered in production.
    """
    found = sorted(Path("data/external/model_runs").rglob(f"*{af.SIDECAR_SUFFIX}"))
    if not found:
        pytest.skip("no artifacts in this checkout to assert the contract against")

    # The demo minutes sidecar predates the key and is not a scoring artifact.
    scoring = [p for p in found if "demo" not in p.parts and "minutes" not in p.name]
    if not scoring:
        pytest.skip("only demo artifacts present")

    missing = []
    for path in scoring:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not af._parse_utc(payload.get(af.SAVED_AT_KEY)):
            missing.append(path.name)
    assert not missing, (
        f"these sidecars carry no readable {af.SAVED_AT_KEY!r}: {missing}. "
        "Training must persist it, or every artifact reads as UNDATED and "
        "every board is withheld."
    )


def test_the_newest_game_date_query_applies_no_upper_window():
    """
    The windows elsewhere in `db.repository` are LEAKAGE GUARDS. This reader is
    not a feature and must not acquire one: an upper bound would make "how far
    does our data run" answer "up to the slate", which is the one answer that
    cannot detect a lagging retrain. Checked on the AST so a docstring saying
    so cannot satisfy it.
    """
    from src.db import repository

    tree = ast.parse(inspect.getsource(repository.newest_team_game_date))
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "_window" not in called, (
        "newest_team_game_date called _window. The windows in this module stop "
        "a FEATURE from seeing past its slate; this is not a feature, and an "
        "upper bound here hides exactly the staleness it exists to find."
    )


# --- the combiner -------------------------------------------------------


def test_first_refusal_keeps_the_reason_that_fired():
    allowed = artifact_freshness_gate(
        af.artifact_freshness(sidecar(), now=NOW)
    )
    withheld = artifact_freshness_gate(None)
    assert first_refusal(allowed, withheld) is withheld
    assert first_refusal(withheld, allowed) is withheld
    assert first_refusal(allowed, allowed) is allowed


def test_first_refusal_refuses_to_invent_permission():
    with pytest.raises(ValueError, match="DATA_NOT_AVAILABLE"):
        first_refusal()
    with pytest.raises(ValueError, match="DATA_NOT_AVAILABLE"):
        first_refusal(None, None)


# --- it actually bites, in the paths that publish -----------------------


def _stale_artifact(tmp_path, *, days: float = 400.0) -> Path:
    artifact = tmp_path / "xgboost_PTS.json"
    artifact.write_text("{}", encoding="utf-8")
    stamp = datetime.now(timezone.utc).timestamp() - days * 86400.0
    artifact.with_suffix(af.SIDECAR_SUFFIX).write_text(
        json.dumps({
            "train_row_count": 200_678,
            af.SAVED_AT_KEY: datetime.fromtimestamp(stamp, timezone.utc).isoformat(),
            af.TRAIN_END_KEY: "2024-01-13",
        }),
        encoding="utf-8",
    )
    return artifact


def _good_calibration_report(tmp_path) -> Path:
    """A report the CALIBRATION gate allows, so a withheld board can only be
    the artifact gate firing."""
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps({
            "status": "OK",
            "n_scored": 400,
            "ece": 0.03,
            "ece_ungated": 0.03,
            "ece_gate_passed": True,
            "bin_coverage": 0.9,
            "report_timestamp_pt": datetime.now(timezone.utc).isoformat(),
        }),
        encoding="utf-8",
    )
    return path


def test_the_worker_withholds_a_board_scored_by_a_stale_artifact(
    monkeypatch, tmp_path
):
    """
    THE WHOLE POINT, as behaviour rather than as a unit.

    A board with a clean calibration report and a 400-day-old artifact. Before
    this gate existed the numbers went out, every contract satisfied, stamped
    with tonight's date. The assertion that matters is the last one: the player
    name must not reach the embed.
    """
    import pandas as pd

    import scheduler_worker as worker

    sent = {}

    def fake_send(embeds, config=None, **kw):
        sent["embeds"] = list(embeds)
        return type("R", (), {"status": "OK"})()

    board = tmp_path / "board.csv"
    pd.DataFrame([{
        "slate_date": "2026-10-11", "decision_status": "RECOMMENDED",
        "decision_basis": "model_lean", "player_name": "DEMO_A",
        "target_market": "PTS", "side": "over", "line": 24.5,
        "model_prob": 0.71, "book_ev": None, "american_odds": None,
        "book_source": None,
    }]).to_csv(board, index=False)

    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/1/x")
    monkeypatch.setenv(worker.ENV_DISPATCH, "true")
    monkeypatch.setenv(worker.ENV_BOARD_CSV, str(board))
    monkeypatch.setenv(worker.ENV_BOARD_MARKETS, "PTS")
    monkeypatch.setenv(
        worker.ENV_CALIBRATION_REPORT, str(_good_calibration_report(tmp_path))
    )
    monkeypatch.setenv("PROPIQ_MODEL", str(_stale_artifact(tmp_path)))

    import src.notify.discord as discord_mod

    monkeypatch.setattr(discord_mod, "send_embeds", fake_send)

    out = worker.run_dispatch()
    assert out["status"] == "OK"
    assert out["publication"]["PUBLICATION_STATUS"] == "PUBLISH_WITHHELD"
    # It must be the ARTIFACT's refusal, not the calibration gate's.
    assert "artifact" in out["publication"]["REASON"].lower()
    assert out["publication"]["ARTIFACT_AGE_DAYS"] > 45
    assert "DEMO_A" not in str(sent["embeds"][0])


def test_the_worker_publishes_when_the_artifact_is_current(monkeypatch, tmp_path):
    """
    The other half, or the test above would pass with a gate that always
    withholds.
    """
    import pandas as pd

    import scheduler_worker as worker

    sent = {}

    def fake_send(embeds, config=None, **kw):
        sent["embeds"] = list(embeds)
        return type("R", (), {"status": "OK"})()

    board = tmp_path / "board.csv"
    pd.DataFrame([{
        "slate_date": "2026-10-11", "decision_status": "RECOMMENDED",
        "decision_basis": "model_lean", "player_name": "DEMO_A",
        "target_market": "PTS", "side": "over", "line": 24.5,
        "model_prob": 0.71, "book_ev": None, "american_odds": None,
        "book_source": None,
    }]).to_csv(board, index=False)

    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/1/x")
    monkeypatch.setenv(worker.ENV_DISPATCH, "true")
    monkeypatch.setenv(worker.ENV_BOARD_CSV, str(board))
    monkeypatch.setenv(worker.ENV_BOARD_MARKETS, "PTS")
    monkeypatch.setenv(
        worker.ENV_CALIBRATION_REPORT, str(_good_calibration_report(tmp_path))
    )
    monkeypatch.setenv("PROPIQ_MODEL", str(_stale_artifact(tmp_path, days=2.0)))

    import src.notify.discord as discord_mod

    monkeypatch.setattr(discord_mod, "send_embeds", fake_send)

    out = worker.run_dispatch()
    assert out["publication"]["PUBLICATION_STATUS"] == "PUBLISH_ALLOWED"
    assert "DEMO_A" in str(sent["embeds"][0])


def test_a_market_with_no_artifact_at_all_withholds_the_board(monkeypatch, tmp_path):
    """An unresolvable artifact is an unknown age, which this gate refuses to
    read as a young one — nothing scored those rows from a model."""
    import scheduler_worker as worker

    monkeypatch.setenv(worker.ENV_BOARD_MARKETS, "PTS")
    monkeypatch.setenv("PROPIQ_MODEL", str(tmp_path / "does-not-exist.json"))
    verdict = worker.artifact_freshness_verdict()
    assert verdict.status == PUBLISH_WITHHELD
    assert "MISSING" in verdict.reason


def test_the_healthcheck_fails_on_a_stale_artifact(monkeypatch, tmp_path):
    """
    Said at DEPLOY time, before a channel sees anything — the dispatch gate is
    the last line, not the only one.
    """
    from scripts import railway_healthcheck as hc

    monkeypatch.setenv("PROPIQ_MODEL", str(_stale_artifact(tmp_path)))
    checks = hc._check_artifact_freshness(["PTS"], True)
    assert [c.status for c in checks] == [hc.FAIL]
    assert "over the 45-day limit" in checks[0].detail
    assert hc.verdict(checks) == "FAILURE"

    monkeypatch.setenv("PROPIQ_MODEL", str(_stale_artifact(tmp_path, days=1.0)))
    checks = hc._check_artifact_freshness(["PTS"], True)
    assert [c.status for c in checks] == [hc.OK]
    assert hc.verdict(checks) == "OK"


def test_the_healthcheck_runs_the_freshness_check_for_every_board_market(
    monkeypatch, tmp_path
):
    """
    A check that silently covers fewer markets than the worker boards is the
    defect `board_markets` was extracted to prevent. Asserted on the assembled
    list, not on the function in isolation.
    """
    from scripts import railway_healthcheck as hc

    monkeypatch.setenv("PROPIQ_BOARD_MARKETS", "PTS,REB")
    monkeypatch.setenv("PROPIQ_MODEL", str(_stale_artifact(tmp_path, days=1.0)))
    names = {c.name for c in hc.run_checks(skip_db=True)}
    assert {"artifact_freshness[PTS]", "artifact_freshness[REB]"} <= names
    assert "artifact_freshness[AST]" not in names


# --- the newest-game-date query, against a real database ----------------


def _sqlite_session_scope(rows: list[str]):
    """
    `session_scope` bound to an in-memory SQLite holding `rows` team-games.

    SQLite, following `tests/test_postgres_only_defects.py`: the assertion here
    is about which row `func.max` returns, which is not dialect-specific, and a
    test that needs Postgres to run is a test that does not run.
    """
    import contextlib
    from datetime import date as _date

    from sqlalchemy import create_engine
    from sqlalchemy.dialects.postgresql import JSONB
    from sqlalchemy.ext.compiler import compiles
    from sqlalchemy.orm import sessionmaker

    from src.db.models import Base, TeamGameStat

    # `team_game_stats.raw_json` is JSONB, which SQLite's DDL compiler cannot
    # render. Teaching it to emit JSON affects the SQLITE dialect only, and
    # this project has no SQLite path in production ("There is no SQLite
    # fallback by design" — src/db/session.py), so nothing real can take it.
    # Registered here rather than at module scope so importing this file does
    # not alter SQLAlchemy for every other test.
    @compiles(JSONB, "sqlite")
    def _jsonb_as_json(type_, compiler, **kw):  # noqa: ARG001
        return "JSON"

    engine = create_engine("sqlite://")
    # Only the one table: creating the whole schema would make an unrelated
    # model's column type a reason this test fails.
    Base.metadata.create_all(bind=engine, tables=[TeamGameStat.__table__])
    factory = sessionmaker(bind=engine)
    with factory() as seed:
        for i, row in enumerate(rows):
            # A bare date keeps the regular-season default every existing
            # caller relied on; a (date, game_id) pair is how the season-type
            # filter is exercised.
            day, game_id = row if isinstance(row, tuple) else (row, f"00226000{i:02d}")
            seed.add(TeamGameStat(
                nba_game_id=game_id,
                game_date=_date.fromisoformat(day),
                team_abbr="LAL",
                opponent_abbr="BOS",
                is_home=True,
            ))
        seed.commit()

    @contextlib.contextmanager
    def scope():
        session = factory()
        try:
            yield session
            session.commit()
        finally:
            session.close()

    return scope


def test_the_newest_game_date_is_the_maximum_over_the_whole_table(monkeypatch):
    """
    Inserted out of order on purpose: an ingest that backfills is when this
    number matters most, and it is when "the maximum" and "whichever row came
    back first" diverge.

    WHAT THIS DOES AND DOES NOT COVER, measured rather than assumed. Mutating
    the query to `func.min` fails this test, and so does a `LIMIT 1` with no
    `ORDER BY`. Taking the LAST row of an unordered `select(game_date)` does
    NOT fail it: `game_date` is indexed, so SQLite answers from a covering
    index scan in ascending order and the last row happens to be the maximum.
    That coincidence is a property of this harness, not of the contract, so it
    is written down here instead of being left to look like coverage.
    """
    from src.db import repository

    monkeypatch.setattr(
        repository,
        "session_scope",
        _sqlite_session_scope(["2026-01-20", "2026-04-13", "2026-02-02"]),
    )
    assert repository.newest_team_game_date() == "2026-04-13"


def test_an_empty_table_has_no_newest_game_date(monkeypatch):
    """A first deploy with no ingest is not a broken one — and None means the
    recency axis is not measured, never that it passed."""
    from src.db import repository

    monkeypatch.setattr(repository, "session_scope", _sqlite_session_scope([]))
    assert repository.newest_team_game_date() is None

    fresh = af.artifact_freshness(
        sidecar(train_end="2024-01-13"),
        now=NOW,
        newest_game_date=repository.newest_team_game_date(),
    )
    assert fresh["training_lag_days"] is None
    assert "training_recency" not in fresh["checks"]


# --- the training-lag comparison must be LIKE AGAINST LIKE -----------------
#
# FOUND 2026-10-18, two days before opening night. The gate read:
#
#   [FAIL] artifact_freshness[PTS]: trained through 2026-01-13, we hold games
#          through 2026-06-13 — 151 day(s), over the 21-day limit
#
# and no retrain could clear it. `scripts/ingest_training_pack` decision 2
# DROPS non-regular-season rows, so `train_end_date` can never pass the end of
# the regular season — while the comparison date came from a table whose newest
# row was a June FINALS game. Measured on the 2025-26 workbook: the regular
# season ends 2026-04-12 and the newest game of any kind is 2026-06-13, so the
# smallest lag any honest retrain could reach was 62 days against a limit of
# 21. The gate was not failing; it was unsatisfiable.
#
# A gate nobody can satisfy gets its limit raised or its result ignored, and
# both of those are worse than the gate not existing. Like against like: the
# same retrain lands at 0.


def test_the_newest_game_date_excludes_playoff_games(monkeypatch):
    """
    THE FIX. The playoff row is newer and must not be the one the comparison
    uses, because training never sees its kind.
    """
    from src.db import repository

    monkeypatch.setattr(
        repository,
        "session_scope",
        _sqlite_session_scope([
            ("2026-04-12", "0022501230"),   # last regular-season game
            ("2026-04-16", "0052500101"),   # play-in
            ("2026-06-13", "0042500406"),   # Finals
        ]),
    )
    assert repository.newest_team_game_date() == "2026-04-12", (
        "the comparison date is a non-regular-season game, so no artifact "
        "trained on regular-season rows can ever be current against it"
    )


@pytest.mark.parametrize(
    "game_id,kind",
    [
        ("0012500001", "preseason"),
        ("0032500001", "all-star"),
        ("0042500406", "playoffs"),
        ("0052500101", "play-in"),
        ("0062500001", "NBA Cup final"),
    ],
)
def test_no_non_regular_season_kind_sets_the_comparison_date(monkeypatch, game_id, kind):
    """Each kind separately: one of them slipping through reinstates the fault
    on its own, and they arrive at different times of year."""
    from src.db import repository

    monkeypatch.setattr(
        repository,
        "session_scope",
        _sqlite_session_scope([
            ("2026-04-12", "0022501230"),
            ("2026-06-13", game_id),
        ]),
    )
    assert repository.newest_team_game_date() == "2026-04-12", (
        f"a {kind} game set the training-lag comparison date"
    )


def test_an_unreadable_game_id_is_excluded_rather_than_assumed_regular(monkeypatch):
    """
    Defaulting an id the parser cannot read to "regular" would quietly admit
    exactly the rows the filter exists to drop.
    """
    from src.db import repository

    monkeypatch.setattr(
        repository,
        "session_scope",
        _sqlite_session_scope([
            ("2026-04-12", "0022501230"),
            ("2026-06-13", "not-an-id"),
        ]),
    )
    assert repository.newest_team_game_date() == "2026-04-12"


def test_a_playoff_only_table_is_not_measured_rather_than_measured_wrongly(monkeypatch):
    """
    A backfill holding only playoff rows has no regular-season date to compare
    against. None means "this axis is not measured" — never a pass, and never
    the June date.
    """
    from src.db import repository

    monkeypatch.setattr(
        repository,
        "session_scope",
        _sqlite_session_scope([("2026-06-13", "0042500406")]),
    )
    assert repository.newest_team_game_date() is None


def test_a_playoff_only_table_says_why_it_cannot_measure(monkeypatch, caplog):
    """"No rows" and "no REGULAR-SEASON rows" are different conditions, and the
    second reads as the first unless it is said."""
    import logging

    from src.db import repository

    monkeypatch.setattr(
        repository,
        "session_scope",
        _sqlite_session_scope([("2026-06-13", "0042500406")]),
    )
    with caplog.at_level(logging.WARNING):
        repository.newest_team_game_date()
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "NONE identifiable as regular season" in text


def test_the_season_type_rule_is_not_restated_as_sql():
    """
    One reader. The filter runs in the database -- this function's own
    docstring argues against loading a frame to take one maximum -- so the
    digit and its position have to come from `utils.ids`, not be written out
    again as a SQL literal here.
    """
    import ast
    import inspect

    from src.db import repository

    src = inspect.getsource(repository.newest_team_game_date)
    tree = ast.parse(src.lstrip())
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert "REGULAR_SEASON_ID_DIGIT" in names, (
        "the regular-season digit is hardcoded here instead of read from "
        "utils.ids, so the two definitions can drift"
    )
    assert "SEASON_TYPE_SQL_POSITION" in names
    # the literal itself must not appear
    assert '"2"' not in src and "'2'" not in src


def test_the_gate_is_satisfiable_by_a_retrain_on_the_data_we_hold():
    """
    THE CONSEQUENCE, with the real workbook rather than fixtures: a retrain
    through the last regular-season game must PASS. Before the fix the same
    retrain read 62 days and failed, and there was nothing a retrain could do
    about it.
    """
    import pandas as pd

    from src.ingestion.bigdataball import load_bigdataball_workbook
    from src.utils.ids import is_regular_season_game_id

    book = "data/external/bigdataball/2025-2026_NBA_Box_Score_Team-Stats.xlsx"
    if not Path(book).is_file():  # pragma: no cover - licensed export
        pytest.skip("licensed workbook not present")

    stats, _ = load_bigdataball_workbook(book)
    dates = pd.to_datetime(stats["game_date"])
    regular = stats["nba_game_id"].map(is_regular_season_game_id)
    newest_regular = dates[regular].max().date()
    newest_any = dates.max().date()

    assert newest_any > newest_regular, (
        "this workbook holds no post-season games, so it cannot demonstrate "
        "the defect — the assertions below would pass trivially"
    )

    fresh = af.artifact_freshness(
        sidecar(train_end=str(newest_regular)),
        now=NOW,
        newest_game_date=str(newest_regular),
    )
    assert fresh["training_lag_days"] == 0
    # `checks` names the axes that WERE measured; `status` is the verdict.
    assert "training_recency" in fresh["checks"]
    assert fresh["status"] == af.OK

    # and the comparison it replaced, for the record
    stale = af.artifact_freshness(
        sidecar(train_end=str(newest_regular)),
        now=NOW,
        newest_game_date=str(newest_any),
    )
    assert stale["training_lag_days"] == (newest_any - newest_regular).days
    assert stale["status"] != af.OK, (
        "the comparison that made the gate unsatisfiable now reads as healthy"
    )


def test_a_truncated_game_id_cannot_pass_the_filter(monkeypatch):
    """
    The length check earns its place: `substr('002', 3, 1)` is '2', so a
    truncated or malformed id would match the digit test alone and set the
    comparison date to whatever date it carried.
    """
    from src.db import repository

    monkeypatch.setattr(
        repository,
        "session_scope",
        _sqlite_session_scope([
            ("2026-04-12", "0022501230"),
            ("2026-06-13", "002"),
        ]),
    )
    assert repository.newest_team_game_date() == "2026-04-12"
