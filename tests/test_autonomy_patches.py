"""
tests/test_autonomy_patches.py — the two findings that were the autonomy gap.

`docs/autonomy_audit_2026-10-10.md` named two technical omissions and called
them the whole gap between semi- and fully-autonomous operation:

  P1 (§3.2) ADVISORY SIZING WAS NOT ON THE AUTOMATED PATH. Fractional Kelly
     existed, capped and tested, and `recommended_units_for_entry` was imported
     in exactly one place: a hand-run CLI command behind a flag. The nightly
     board carried no recommended size.

  P2 (§5.2) NO FAILURE PATH REACHED DISCORD. Successes dispatched; a boot
     ERROR, a FAILED slate, a FAILED settlement and a probe FAILURE were
     log-only. In Railway's log view every line is already red -- Python
     logging writes to stderr and Railway classifies stderr as error severity
     -- so a real failure is indistinguishable from an INFO line by eye.

WHERE P1 LIVES IS NOT AN ACCIDENT. Two existing guards fired when it was first
written inside `src/quant/decision_board.py`, and both were right:
`test_no_module_under_src_imports_this_one` ("a size is held by a caller that
already computed it, never fetched by a module that sends or places anything")
and `test_layer_cannot_reach_a_book_or_size_a_stake` (no "kelly" identifier in
that module). The sizing therefore lives in `scheduler_worker`, which is not
under `src/` and sends nothing, and the ranking layer keeps no concept of
stake. Tests below pin that placement so it cannot drift back.

RESEARCH_ONLY. Every number here is advisory metadata. Nothing in this project
places or sizes a wager, and no test below asserts otherwise.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

import scheduler_worker as worker

REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# P1 — the sizer, and the refusals that matter more than the number
# ---------------------------------------------------------------------------

class _Candidate:
    """A RECOMMENDED board row with a valid price and a half-point line."""

    decision_status = "RECOMMENDED"
    book_status = "VALID"
    american_odds = -110
    model_prob = 0.56
    model_p_over = 0.56
    model_p_under = 0.44
    model_p_push = 0.0
    line_can_push = False
    player_name = "A Player"
    target_market = "PTS"
    side = "over"


def _cand(**overrides):
    c = _Candidate()
    for k, v in overrides.items():
        setattr(c, k, v)
    return c


def test_a_recommended_row_with_a_valid_price_is_sized():
    from src.quant.advisory_sizing import advisory_size_for_candidate

    sized = advisory_size_for_candidate(_cand())
    assert sized.method == "binary"
    assert sized.recommended_units > 0
    assert sized.as_dict()["AUTO_PLACED"] is False


def test_a_withheld_row_is_never_sized():
    """Sizing a row the board declined would contradict the gate that
    withheld it."""
    from src.quant.advisory_sizing import advisory_size_for_candidate

    sized = advisory_size_for_candidate(_cand(decision_status="ABSTAIN"))
    assert sized.method == "none"
    assert sized.recommended_units == 0.0
    assert "RECOMMENDED" in (sized.reason or "")


def test_no_valid_price_means_no_size():
    """
    `f* = (bp - q)/b` needs a price. The house rule is that nothing is ranked
    or priced without valid odds, so a model probability alone sizes nothing.
    """
    from src.quant.advisory_sizing import advisory_size_for_candidate

    for kwargs in (
        {"book_status": "DATA_NOT_AVAILABLE"},
        {"american_odds": None},
        {"american_odds": 0},
    ):
        sized = advisory_size_for_candidate(_cand(**kwargs))
        assert sized.method == "none", kwargs
        assert sized.recommended_units == 0.0


@pytest.mark.parametrize("prob", [None, 0.0, 1.0, 1.4, -0.2, float("nan")])
def test_an_unusable_probability_is_refused(prob):
    from src.quant.advisory_sizing import advisory_size_for_candidate

    sized = advisory_size_for_candidate(_cand(model_prob=prob))
    assert sized.method == "none"


def test_a_pushable_line_is_sized_conditional_on_no_push():
    """
    THE PART THAT IS EASY TO GET WRONG. Binary Kelly is two-outcome: `q` is
    P(lose). On a whole-number line a push is a third outcome and the stake is
    RETURNED, so `q = 1 - p` counts returned money as lost. A two-way American
    price de-vigs to a TWO-outcome fair probability because a push is voided
    rather than priced -- the same reduction `line_diff._over_given_no_push`
    applies -- so the model's probability must be taken conditional on no push.
    """
    from src.quant.advisory_sizing import (
        advisory_size_for_candidate,
        recommended_units_binary,
    )
    from src.quant.odds_math import american_to_decimal

    pushable = _cand(
        line_can_push=True, model_p_push=0.08,
        model_prob=0.52, model_p_over=0.52, model_p_under=0.40,
    )
    sized = advisory_size_for_candidate(pushable)
    assert sized.method == "binary"

    expected = recommended_units_binary(
        0.52 / (0.52 + 0.40), american_to_decimal(-110),
    )
    assert sized.recommended_units == pytest.approx(expected.recommended_units)

    # And it is NOT the raw-probability answer, or this test would pass on the
    # bug it exists for.
    raw = recommended_units_binary(0.52, american_to_decimal(-110))
    assert sized.recommended_units != pytest.approx(raw.recommended_units)
    assert "no push" in (sized.reason or "")


def test_push_mass_with_no_two_sides_is_refused():
    """Nothing to condition on, so sizing would compare a three-outcome
    probability against a two-outcome price."""
    from src.quant.advisory_sizing import advisory_size_for_candidate

    sized = advisory_size_for_candidate(_cand(
        line_can_push=True, model_p_push=0.08,
        model_p_over=None, model_p_under=None,
    ))
    assert sized.method == "none"
    assert "condition on" in (sized.reason or "")


def test_the_size_is_capped():
    from src.quant.advisory_sizing import DEFAULT_MAX_CAP_UNITS, advisory_size_for_candidate

    sized = advisory_size_for_candidate(_cand(model_prob=0.95, model_p_under=0.05))
    assert sized.recommended_units <= DEFAULT_MAX_CAP_UNITS
    assert sized.capped is True


# --- where it lives, which the existing guards decided ----------------------

def test_the_ranking_layer_still_has_no_concept_of_stake():
    """
    Pins the placement the two existing guards require. P1 was first written
    inside `decision_board` and both fired; this keeps it from drifting back.
    """
    src = (REPO / "src" / "quant" / "decision_board.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert "advisory_sizing" not in (node.module or "")
        elif isinstance(node, ast.Import):
            assert not any("advisory_sizing" in a.name for a in node.names)

    from src.quant.decision_board import BettingDecisionCandidate

    for field in BettingDecisionCandidate.model_fields:
        assert "kelly" not in field.lower()
        assert "stake" not in field.lower()


def test_the_worker_attaches_sizing_and_it_is_the_worker_that_does():
    """
    `scheduler_worker` is not under `src/` and sends nothing, so it is the
    caller the guards permit to hold a size.
    """
    tree = ast.parse((REPO / "scheduler_worker.py").read_text(encoding="utf-8"))
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "attach_board_sizing"
    )
    imported = {
        (n.module or "") for n in ast.walk(fn) if isinstance(n, ast.ImportFrom)
    }
    assert any("advisory_sizing" in m for m in imported)

    run_board = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "run_board"
    )
    called = {
        n.func.id for n in ast.walk(run_board)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "attach_board_sizing" in called


def test_sizing_writes_the_columns_into_the_board_csv(tmp_path):
    """The CSV is what `run_dispatch` reads, so this is how a size reaches
    Discord."""
    import pandas as pd

    csv = tmp_path / "board.csv"
    pd.DataFrame([
        {"player_name": "A Player", "target_market": "PTS", "side": "over"},
        {"player_name": "B Player", "target_market": "REB", "side": "under"},
    ]).to_csv(csv, index=False)

    board = [_cand(), _cand(decision_status="ABSTAIN")]
    out = worker.attach_board_sizing(board, str(csv))

    assert out["status"] == "OK"
    assert out["sized"] == 1 and out["declined"] == 1
    frame = pd.read_csv(csv)
    assert frame["RECOMMENDED_UNITS"].notna().sum() == 1
    # A declined row gets NaN, never 0.0: zero would read as "sized, and the
    # answer is nothing".
    assert pd.isna(frame["RECOMMENDED_UNITS"].iloc[1])
    # A DECLINED row must say why; a clean uncapped size legitimately has no
    # note. The first version asserted every row had one and failed on the
    # sized row -- the test was wrong, not the sizer: a note explains a refusal
    # or a cap, and "sized cleanly" is neither.
    assert pd.notna(frame["SIZING_NOTE"].iloc[1]), "the refusal gave no reason"
    assert "RECOMMENDED" in str(frame["SIZING_NOTE"].iloc[1])
    assert (frame["AUTO_PLACED"] == False).all()  # noqa: E712 — a column compare


def test_a_row_count_mismatch_refuses_rather_than_mispairing(tmp_path):
    """
    The positional join is the one assumption here, and pairing one row's size
    with another's price is worse than no size at all.
    """
    import pandas as pd

    csv = tmp_path / "board.csv"
    pd.DataFrame([{"player_name": "only one row"}]).to_csv(csv, index=False)
    out = worker.attach_board_sizing([_cand(), _cand()], str(csv))

    assert out["status"] == "FAILED"
    assert "mismatch" in out["reason"]
    assert "RECOMMENDED_UNITS" not in pd.read_csv(csv).columns


def test_sizing_never_fails_the_board(tmp_path):
    """The board is the durable output and is already written when this runs."""
    assert worker.attach_board_sizing(None, "/nonexistent/board.csv")["status"] == "SKIPPED"
    assert worker.attach_board_sizing([], "/nonexistent/board.csv")["status"] == "SKIPPED"
    # A real board against an unreadable CSV reports FAILED, never raises.
    assert worker.attach_board_sizing([_cand()], "/nonexistent/x.csv")["status"] == "FAILED"


# ---------------------------------------------------------------------------
# P2 — failures reach Discord, and credentials do not
# ---------------------------------------------------------------------------

@pytest.fixture
def alerting(monkeypatch, tmp_path):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL",
                       "https://discord.com/api/webhooks/123/token-value-here")
    monkeypatch.setenv("PROPIQ_STATE_DIR", str(tmp_path))
    monkeypatch.delenv("PROPIQ_ALERT_FAILURES", raising=False)
    monkeypatch.delenv("PROPIQ_ALERT_DEDUPE", raising=False)
    sent: list[dict] = []

    def _fake_send(embeds, **kwargs):
        sent.append({"embeds": list(embeds), "kwargs": kwargs})

        class _R:
            status = "SENT"
        return _R()

    import src.notify.discord as discord
    monkeypatch.setattr(discord, "send_embeds", _fake_send)
    return sent


def test_a_failure_is_dispatched(alerting):
    out = worker.notify_failure("slate", "the slate raised ValueError: nope")
    assert out["status"] == "SENT"
    assert len(alerting) == 1
    embed = alerting[0]["embeds"][0]
    assert "slate" in embed["title"]
    assert "nope" in embed["description"]
    assert "RESEARCH_ONLY" in embed["footer"]["text"]


def test_alerting_is_off_without_a_webhook(monkeypatch, tmp_path):
    """A worker with no webhook should not spend a boot trying to post."""
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    monkeypatch.setenv("PROPIQ_STATE_DIR", str(tmp_path))
    assert worker.notify_failure("slate", "x")["status"] == "SKIPPED"


def test_the_same_stage_does_not_alert_twice_in_one_pacific_day(alerting):
    """A redeploy loop would otherwise post the same unseeded-volume alert on
    every boot."""
    assert worker.notify_failure("boot:model-artifact", "no artifact")["status"] == "SENT"
    assert worker.notify_failure("boot:model-artifact", "no artifact")["status"] == "DEDUPED"
    assert len(alerting) == 1
    # A DIFFERENT stage still gets through.
    assert worker.notify_failure("slate", "unrelated")["status"] == "SENT"
    assert len(alerting) == 2


def test_dedupe_fails_OPEN_when_its_store_is_unusable(alerting, monkeypatch):
    """
    The store lives on the mounted volume, and "the volume is not writable" is
    one of the failures this reports. A dedupe that suppressed on its own
    failure would silence exactly the alert it could not record. A duplicate is
    a nuisance; a missing alert is the problem P2 exists for.
    """
    monkeypatch.setattr(
        worker, "_alert_state_path",
        lambda: (_ for _ in ()).throw(OSError("volume is read-only")),
    )
    assert worker.notify_failure("boot:state-dir", "unwritable")["status"] == "SENT"
    assert worker.notify_failure("boot:state-dir", "unwritable")["status"] == "SENT"
    assert len(alerting) == 2


def test_a_dispatch_failure_never_raises(alerting, monkeypatch):
    """One failure must not become two."""
    import src.notify.discord as discord

    monkeypatch.setattr(
        discord, "send_embeds",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("webhook is gone")),
    )
    out = worker.notify_failure("slate", "the slate raised")
    assert out["status"] == "FAILED"
    assert "webhook is gone" in out["error"]


def test_the_dispatch_path_never_alerts_about_itself():
    """
    Alerting about a broken webhook through that webhook is a loop with no
    exit, so `run_dispatch` must not call `notify_failure` at all.
    """
    tree = ast.parse((REPO / "scheduler_worker.py").read_text(encoding="utf-8"))
    dispatch = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "run_dispatch"
    )
    called = {
        n.func.id for n in ast.walk(dispatch)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "notify_failure" not in called


# --- the credential scrubber, which is why this is safe to send -------------

@pytest.mark.parametrize("raw,must_not_contain", [
    ("connect failed: postgresql://postgres:s3cr3t@db.supabase.co:6543/postgres",
     "s3cr3t"),
    ("PROPLINE_API_KEY=pk_live_abcdef123456 was refused", "pk_live_abcdef123456"),
    ("PROPIQ_ARTIFACT_STORE_KEY=longsecretvalue rejected", "longsecretvalue"),
    ("posting to https://discord.com/api/webhooks/123/AbCdEf-ghIJKL failed",
     "AbCdEf-ghIJKL"),
])
def test_credentials_are_scrubbed_before_an_alert_leaves(raw, must_not_contain):
    """
    THE ALERT BODY IS AN EXCEPTION STRING. psycopg and SQLAlchemy both put the
    connection URL in their messages, and the URL carries the password -- so
    routing failures to an outbound channel without this would publish
    DATABASE_URL to a chat the first time the database refused a connection.
    That is worse than the silence P2 set out to fix.
    """
    cleaned = worker.scrub_alert(raw)
    assert must_not_contain not in cleaned
    assert "***" in cleaned


def test_the_scrubber_leaves_ordinary_text_alone():
    """Or every alert would be unreadable and nobody would read them."""
    plain = "the slate raised ValueError: no rows for 2026-10-11"
    assert worker.scrub_alert(plain) == plain


def test_a_dispatched_alert_body_carries_no_credential(alerting):
    worker.notify_failure(
        "settlement",
        "OperationalError: could not connect to "
        "postgresql://postgres:hunter2@db.supabase.co:6543/postgres",
    )
    body = json.dumps(alerting[0]["embeds"][0])
    assert "hunter2" not in body
    assert "***" in body


# --- the boot checks, read from what they return ---------------------------

def test_boot_problems_are_read_from_the_checks_return_values(alerting):
    """
    The three checks stay pure -- they already return structured dicts and
    their policy is warn-and-carry-on. Threading a dispatch callback through
    them would couple three probes to an outbound channel for no gain.
    """
    out = worker.notify_boot_problems(
        {"path": "/app/data", "writable": False, "error": "EACCES"},
        {"resolved": False, "reason": "no artifact found"},
        {"status": "UNUSABLE", "ledger": "postgres"},
        {"ok": True, "message": "Database reachable."},
    )
    assert len(out) == 3
    stages = {a["stage"] for a in out}
    assert stages == {"boot:state-dir", "boot:model-artifact", "boot:parlay-ledger"}
    assert len(alerting) == 3


def test_a_healthy_boot_alerts_nothing(alerting):
    out = worker.notify_boot_problems(
        {"path": "/app/data", "writable": True},
        {"resolved": True, "exists": True, "path": "/app/data/x.json"},
        {"status": "OK", "ledger": "postgres"},
        {"ok": True, "message": "Database reachable."},
    )
    assert out == []
    assert alerting == []


def test_the_demo_artifact_is_reported_using_the_key_that_is_actually_set(alerting):
    """
    This branch read `demo_suspected` when it was first written -- a key
    nothing writes -- so it was dead code that would have stayed silent on the
    demo artifact. `check_model_artifact` sets `implausible_train_rows`.
    """
    out = worker.notify_boot_problems(
        {"writable": True},
        {"resolved": True, "exists": True, "implausible_train_rows": True,
         "train_row_count": 968},
        {"status": "OK"},
        {"ok": True, "message": "Database reachable."},
    )
    assert [a["stage"] for a in out] == ["boot:model-artifact-demo"]
    assert "968" in json.dumps(alerting[0]["embeds"][0])


def test_a_local_csv_ledger_is_not_alerted(alerting):
    """LOCAL is a deliberate local choice, not a failure."""
    assert worker.notify_boot_problems(
        {"writable": True}, {"resolved": True, "exists": True},
        {"status": "LOCAL", "ledger": "csv"},
        {"ok": True, "message": "Database reachable."},
    ) == []
    assert alerting == []


def test_an_unreachable_database_is_alerted_at_boot(alerting):
    """
    FOUND IN PRODUCTION, 2026-10-11. The Railway container booted at 03:22 PT
    reporting nothing wrong. `DATABASE_URL` was unset, so settlement failed at
    03:30 and the slate refused at 09:00 -- the first signal came more than
    five hours after a boot that had every fact it needed to say so.

    The three existing checks ask about the volume, the artifact and the
    ledger. None asks whether the database answers, and on this deployment the
    ledger check only caught it indirectly: it reports "the first parlay write
    will fail", which names a consequence rather than the cause.
    """
    out = worker.notify_boot_problems(
        {"writable": True},
        {"resolved": True, "exists": True},
        {"status": "OK", "ledger": "postgres"},
        {"ok": False, "message": "No database connection configured."},
    )
    assert [a["stage"] for a in out] == ["boot:database"]
    assert "No database connection configured" in json.dumps(alerting[0]["embeds"][0])


def test_a_reachable_database_alerts_nothing(alerting):
    assert worker.notify_boot_problems(
        {"writable": True}, {"resolved": True, "exists": True},
        {"status": "OK"}, {"ok": True, "message": "Database reachable."},
    ) == []
    assert alerting == []


def test_the_boot_database_check_never_raises(monkeypatch):
    """
    A probe that raises at boot takes the worker down before the scheduler
    starts, which is strictly worse than the silence it replaces.
    """
    import src.db.session as session

    def explode():
        raise RuntimeError("driver exploded on import")

    monkeypatch.setattr(session, "healthcheck", explode)
    out = worker.check_database()
    assert out["ok"] is False
    assert "driver exploded" in out["message"]


def test_the_boot_database_alert_carries_no_credential(alerting):
    """
    SQLAlchemy and psycopg both put the connection URL -- password included --
    into their error messages, and this check hands that message straight to an
    outbound channel. `scrub_alert` is what stands between the two.
    """
    leak = (
        "Database connection failed: could not connect to "
        "postgresql://propiq:sup3rs3cret@db.example.com:5432/propiq"
    )
    worker.notify_boot_problems(
        {"writable": True}, {"resolved": True, "exists": True},
        {"status": "OK"}, {"ok": False, "message": leak},
    )
    body = json.dumps(alerting[0]["embeds"][0])
    assert "sup3rs3cret" not in body
    assert "propiq:***@db.example.com" in body


def test_main_routes_the_boot_checks_into_the_alerter():
    """
    AST-walked: every boot check must RUN in `main` and its return value must
    reach `notify_boot_problems`, not be discarded.

    THIS TEST USED TO COUNT ARGUMENTS AND NOTHING ELSE, and a mutation that
    replaced the database result with the literal `{"ok": True}` passed it --
    the arity was right and the check had been unwired. Counting arguments
    cannot tell a routed result from a hardcoded pass, so each argument is now
    required to be a NAME, and each check is required to be called.
    """
    tree = ast.parse((REPO / "scheduler_worker.py").read_text(encoding="utf-8"))
    main_fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    calls = [
        n for n in ast.walk(main_fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id == "notify_boot_problems"
    ]
    assert len(calls) == 1
    args = calls[0].args
    assert len(args) == 4, "all four check results must be passed"

    literals = [ast.dump(a) for a in args if not isinstance(a, ast.Name)]
    assert not literals, (
        "a boot check result was passed as a literal rather than as the "
        f"variable the check returned, so the check is not wired: {literals}"
    )

    called = {
        n.func.id for n in ast.walk(main_fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    for check in ("check_state_dir", "check_model_artifact",
                  "check_parlay_ledger", "check_database"):
        assert check in called, f"main does not call {check}"

    # And the names passed must be the ones those checks were assigned to, or
    # four correctly-shaped variables could still be the wrong four.
    assigned = {
        target.id: node.value.func.id
        for node in ast.walk(main_fn)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    routed = [assigned.get(a.id) for a in args]
    assert routed == [
        "check_state_dir", "check_model_artifact",
        "check_parlay_ledger", "check_database",
    ], f"the arguments do not come from the four checks, in order: {routed}"


# ---------------------------------------------------------------------------
# the odds source, and why its absence is a FAILURE rather than a warning
# ---------------------------------------------------------------------------

def test_a_missing_odds_key_does_not_crash_the_slate_which_is_the_problem(
    monkeypatch,
):
    """
    VERIFIED, AND THE FIRST GUESS WAS WRONG. The expectation was that a missing
    `PROPLINE_API_KEY` raised and failed the slate loudly. It does not:
    `pull_nba_prop_lines` catches `PropLineAuthError` and returns a
    DATA_NOT_AVAILABLE snapshot, step [3] logs one WARNING and skips it, and
    the run continues with no posted lines -- so every row abstains for want of
    a market, the board is empty, and the run exits 0.

    The quieter behaviour is the worse one, and because it is a warning rather
    than an error `notify_failure` does not alert on it either. This test pins
    the behaviour so the probe check below is known to be guarding something
    real.
    """
    import main

    monkeypatch.delenv("PROPLINE_API_KEY", raising=False)
    frame = main.ingest_prop_lines(None, persist=False)
    assert len(frame) == 0, "step [3] no longer degrades; re-check the probe's reason"


def test_the_probe_fails_when_the_odds_key_is_absent(monkeypatch, tmp_path):
    from scripts.railway_healthcheck import FAIL, OK, run_checks
    from src.utils.volume import ENV_STATE_DIR

    root = tmp_path / "volume"
    root.mkdir()
    monkeypatch.setenv(ENV_STATE_DIR, str(root))
    monkeypatch.setenv("PROPIQ_BOARD_MARKETS", "PTS")
    monkeypatch.setenv("PROPIQ_MODEL", str(root / "absent.json"))

    monkeypatch.delenv("PROPLINE_API_KEY", raising=False)
    statuses = {c.name: c.status for c in run_checks(skip_db=True)}
    assert statuses["odds_source"] == FAIL

    monkeypatch.setenv("PROPLINE_API_KEY", "a-value")
    statuses = {c.name: c.status for c in run_checks(skip_db=True)}
    assert statuses["odds_source"] == OK


def test_the_probe_never_prints_the_odds_key(monkeypatch, tmp_path, capsys):
    from scripts.railway_healthcheck import main as probe_main
    from src.utils.volume import ENV_STATE_DIR

    monkeypatch.setenv(ENV_STATE_DIR, str(tmp_path))
    monkeypatch.setenv("PROPIQ_BOARD_MARKETS", "PTS")
    monkeypatch.setenv("PROPLINE_API_KEY", "UNMISTAKABLE-ODDS-KEY-VALUE")
    probe_main(["--skip-db"])
    out = capsys.readouterr()
    assert "UNMISTAKABLE-ODDS-KEY-VALUE" not in out.out + out.err


def test_the_probe_makes_no_live_api_call(monkeypatch, tmp_path):
    """
    A probe that called the API on every boot would spend quota on a redeploy
    loop. Asserted by making any request through the client's session fail
    loudly: the check must not touch it.
    """
    import requests

    from scripts.railway_healthcheck import run_checks
    from src.utils.volume import ENV_STATE_DIR

    monkeypatch.setenv(ENV_STATE_DIR, str(tmp_path))
    monkeypatch.setenv("PROPIQ_BOARD_MARKETS", "PTS")
    monkeypatch.setenv("PROPLINE_API_KEY", "a-value")

    def _boom(*a, **k):
        raise AssertionError("the probe made a network call")

    monkeypatch.setattr(requests.Session, "request", _boom)
    monkeypatch.setattr(requests.Session, "get", _boom)
    run_checks(skip_db=True)


# ---------------------------------------------------------------------------
# P4 — the one daily-path client that had neither exponential backoff nor 429
# ---------------------------------------------------------------------------

def test_the_boxscore_fetcher_backs_off_exponentially(monkeypatch):
    """
    2s, 4s, 8s -- not 2s, 4s, 6s.

    RETRY_ATTEMPTS IS RAISED TO 4 HERE, and that is the point of the test
    rather than an incidental detail. At the shipped setting of 3 the loop
    sleeps twice, and `2 * attempt` and `2 ** attempt` are IDENTICAL for
    attempts 1 and 2 -- so the first version of this test passed against the
    linear backoff it was written to catch. The two only diverge at the third
    sleep. Stated plainly: the exponential change is a no-op at the shipped
    setting and matters only if the attempt budget ever rises, which is
    exactly why it is pinned here instead of assumed.
    """
    import requests

    from src.settlement import boxscore_fetcher as bf

    waits: list[float] = []
    monkeypatch.setattr(bf.time, "sleep", lambda s: waits.append(s))
    monkeypatch.setattr(bf, "RETRY_ATTEMPTS", 4)

    class _Session:
        def get(self, *a, **k):
            raise requests.exceptions.ConnectionError("down")

    with pytest.raises(bf.BoxScoreError):
        bf.fetch_boxscore("0022600001", session=_Session())

    assert waits == [2.0, 4.0, 8.0], "linear backoff would give [2, 4, 6]"


def test_a_429_is_retried_and_honours_retry_after(monkeypatch):
    from src.settlement import boxscore_fetcher as bf

    waits: list[float] = []
    monkeypatch.setattr(bf.time, "sleep", lambda s: waits.append(s))
    calls: list[int] = []

    class _Resp:
        def __init__(self, status, headers=None, payload=None):
            self.status_code = status
            self.headers = headers or {}
            self._payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self._payload

    class _Session:
        def get(self, *a, **k):
            calls.append(1)
            if len(calls) == 1:
                return _Resp(429, {"Retry-After": "7"})
            return _Resp(200, payload={"game": "final"})

    assert bf.fetch_boxscore("0022600001", session=_Session()) == {"game": "final"}
    assert waits == [7.0], "Retry-After was not honoured"
    assert len(calls) == 2


def test_a_429_without_retry_after_falls_back_to_the_backoff(monkeypatch):
    from src.settlement import boxscore_fetcher as bf

    waits: list[float] = []
    monkeypatch.setattr(bf.time, "sleep", lambda s: waits.append(s))

    class _Resp:
        status_code = 429
        headers: dict = {}

        def raise_for_status(self):
            pass

    class _Session:
        def get(self, *a, **k):
            return _Resp()

    with pytest.raises(bf.BoxScoreError, match="rate-limited"):
        bf.fetch_boxscore("0022600001", session=_Session())
    assert waits == [bf.RETRY_BACKOFF ** 1, bf.RETRY_BACKOFF ** 2]
