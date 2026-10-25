"""
scripts/test_autonomous_loop.py — one full mock cycle, through PRODUCTION code.

WHAT THIS IS. The loop a scheduled Railway worker actually runs, end to end,
with every EXTERNAL boundary mocked and every INTERNAL function real:

    raw odds pull -> feature build -> de-vig / Kelly -> publication gate
    -> Discord dispatch -> post-game settlement -> ledger commit

No network call is made. No environment variable is read for data. The
PropLine HTTP session, the Discord POST, and the database file are fakes the
script owns; everything between them is the code Railway runs.

WHAT THIS PROVES, and what it deliberately does not:

    PROVED   the production functions compose: parse a real-shaped odds
             payload, build leakage-safe features, de-vig without dividing
             by zero, size with quarter-Kelly, respect the calibration gate
             in BOTH directions, build and "send" a safe Discord payload,
             grade settled props, and commit ledger rows that a FRESH
             connection can read back.
    NOT PROVED  anything about live Railway readiness — credentials,
             connectivity, the mounted volume, Postgres itself. Those are
             `railway_healthcheck` and `verify_wiring`'s jobs, and this
             script refuses to overlap them. Passing this script means the
             loop's CODE is wired; it does not mean tonight's slate will
             score.

DESIGN RULES, from the repository's own conventions:

    * A missing production module is a FAILURE, not a skip. The point is
      to notice a broken wire, so ImportError is stage zero's subject.
    * The DB is a throwaway SQLite file built DIRECTLY with SQLAlchemy,
      never through `src/db/session.py` — the same division
      `tests/test_db_migrations.py` documents: session.py refuses a SQLite
      fallback for production on purpose, and this script honors that by
      not touching it. The commit is verified through a second, fresh
      connection because a commit only one connection can see is a commit
      nothing else in the stack can read.
    * The calibration gate is exercised, not bypassed: the synthetic
      evidence that opens it is labelled TEST-ONLY below, and the ledger
      rows it produces carry the label too.
    * Zero-embed dispatch returns DATA_NOT_AVAILABLE rather than raising.
      Silence in a channel is a bug; a named reason is a state.

USAGE
    python scripts/test_autonomous_loop.py
Exit 0 only if every stage passes.
"""

from __future__ import annotations

import json
import sys
import tempfile
import traceback
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

# The script may be run from the repo root or anywhere else; the production
# imports below need the repo root on sys.path.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


class StageResult:
    def __init__(self, name: str) -> None:
        self.name = name
        self.ok = False
        self.detail = ""

    def pass_(self, detail: str) -> None:
        self.ok = True
        self.detail = detail

    def fail(self, detail: str) -> None:
        self.ok = False
        self.detail = detail


# ---------------------------------------------------------------------------
# stage 1 — raw pull, through the production PropLine client (mocked HTTP)
# ---------------------------------------------------------------------------


class _FakeResponse:
    """The response attributes PropLineClient actually reads."""

    def __init__(self, payload: object, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.headers: dict[str, str] = {}
        self.text = json.dumps(payload)

    def json(self) -> object:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeSession:
    """A requests.Session whose GETs return fixture payloads by URL shape."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        now = datetime.now(timezone.utc)

        # PropLine's own staleness rule: an outcome is live when its last
        # sighting is not older than the market's last update. Same instant
        # for both, so the fixture outcome is live.
        stamp = now.isoformat()
        self.events_payload = [
            {
                "id": "evt-harness-1",
                "commence_time": (now + timedelta(hours=6)).isoformat(),
            }
        ]
        self.props_payload = {
            "id": "evt-harness-1",
            "commence_time": (now + timedelta(hours=6)).isoformat(),
            "bookmakers": [
                {
                    "key": "harnessbook",
                    "title": "Harness Book",
                    "markets": [
                        {
                            "key": "player_points",
                            "last_update": stamp,
                            "outcomes": [
                                {
                                    "description": "Harness Player",
                                    "name": "Over",
                                    "point": 24.5,
                                    "price": -110,
                                    "last_change_at": stamp,
                                },
                                {
                                    "description": "Harness Player",
                                    "name": "Under",
                                    "point": 24.5,
                                    "price": -110,
                                    "last_change_at": stamp,
                                },
                            ],
                        }
                    ],
                }
            ],
        }

    def get(self, url: str, **_kwargs: object) -> _FakeResponse:
        self.calls.append(url)
        if "/odds" in url:
            return _FakeResponse(self.props_payload)
        return _FakeResponse(self.events_payload)


def stage_raw_pull(slot: StageResult) -> None:
    from src.ingestion.propline import pull_nba_prop_lines

    session = _FakeSession()
    result = pull_nba_prop_lines(
        api_key="harness-test-key",  # TEST-ONLY; the fake session ignores it
        session=session,
        max_events=1,
    )

    # One snapshot per source; each carries paired OVER/UNDER rows.
    snapshots = [s for s in result if s.status == "VALID"]
    if not snapshots:
        slot.fail(
            f"ingest produced no VALID snapshot; statuses: "
            f"{[(s.source, s.status) for s in result]}"
        )
        return
    lines = [line for snap in snapshots for line in snap.lines]
    if not lines:
        slot.fail("VALID snapshot carried zero lines")
        return
    row = lines[0]
    if row.player_name != "Harness Player":
        slot.fail(f"player parsed as {row.player_name!r}")
        return
    if row.over_odds_american != -110 or row.under_odds_american != -110:
        slot.fail(
            f"odds parsed as {row.over_odds_american}/"
            f"{row.under_odds_american}, expected -110/-110"
        )
        return

    # Failure isolation: a payload with NO bookmaker is a snapshot that
    # says so, not an exception that takes the run down.
    empty_session = _FakeSession()
    empty_session.props_payload = {"id": "evt-harness-1", "bookmakers": []}
    empty_result = pull_nba_prop_lines(
        api_key="harness-test-key", session=empty_session, max_events=1
    )
    if not empty_result:
        slot.fail("empty-payload run returned no snapshot at all")
        return

    slot.pass_(
        f"{len(lines)} VALID prop line(s) parsed from mocked HTTP; "
        f"empty payload isolated (status {empty_result[0].status}, run survived)"
    )


# ---------------------------------------------------------------------------
# stage 2 — feature build, through the production builder
# ---------------------------------------------------------------------------


def _synthetic_panel():
    """A small panel with the columns build_feature_matrix reads."""
    import pandas as pd

    rows = []
    start = datetime(2026, 10, 1, tzinfo=timezone.utc)
    for player in ("A", "B", "C"):
        pts = 18.0 if player == "A" else 12.0
        for i in range(14):
            rows.append(
                {
                    "PLAYER_ID": f"2039{player}",
                    "PLAYER_NAME": f"HARNESS {player}",
                    "GAME_ID": f"002260000{i:02d}",
                    "GAME_DATE": start + timedelta(days=i),
                    "SEASON": "2026-27",
                    "PTS": pts + i * 0.5,
                    "REB": 5.0 + (i % 3),
                    "AST": 4.0,
                    "FG3M": 1.5,
                    "STL": 1.0,
                    "BLK": 0.5,
                    "TOV": 2.0,
                    "MIN": 32.0,
                }
            )
    return pd.DataFrame(rows)


def stage_features(slot: StageResult) -> None:
    from src.features.builder import build_feature_matrix

    panel = _synthetic_panel()
    features = build_feature_matrix(panel)

    if "PTS_L10" not in features.columns:
        slot.fail(f"PTS_L10 missing; columns like {list(features.columns)[:8]}")
        return

    # THE LEAKAGE CHECK, REWRITTEN. The original compared the REPORTED
    # `LAST_INCLUDED_GAME_DATE` column against `GAME_DATE`. That column is
    # computed by a different code path from the rollups, so removing the
    # `.shift(1)` from the player rolling mean left it untouched: the matrix
    # leaked the current game into PTS_L5 and this stage still printed
    # "no look-ahead". Verified — the mutation passed all seven stages.
    # `builder.assert_no_lookahead` compares the same reported column and also
    # passed, so calling it instead would not have helped either.
    #
    # So the window is checked ARITHMETICALLY, on known values. A single
    # player, five strictly increasing scores: the last row's PTS_L5 must be
    # the mean of the FOUR PRIOR games (25.0) and not the mean including
    # today's (30.0). That is rule 1 of CLAUDE.md, and it is the one property
    # that makes every number downstream meaningful.
    import pandas as pd

    probe = pd.DataFrame({
        "PLAYER_ID": ["9999"] * 5,
        "GAME_DATE": pd.date_range("2026-10-01", periods=5),
        "PTS": [10.0, 20.0, 30.0, 40.0, 50.0],
    })
    probed = build_feature_matrix(probe).sort_values("GAME_DATE")
    last_l5 = probed["PTS_L5"].iloc[-1]
    if pd.isna(last_l5):
        slot.fail("PTS_L5 is null on a player's fifth game, so the window never filled")
        return
    prior_mean = 25.0        # (10 + 20 + 30 + 40) / 4
    including_today = 30.0   # (10 + 20 + 30 + 40 + 50) / 5
    if abs(float(last_l5) - including_today) < 1e-9:
        slot.fail(
            f"LEAKAGE: PTS_L5 is {float(last_l5)}, which is the mean INCLUDING "
            f"the current game. The .shift(1) is not being applied — every "
            f"feature can see the game it is predicting."
        )
        return
    if abs(float(last_l5) - prior_mean) > 1e-9:
        slot.fail(
            f"PTS_L5 is {float(last_l5)}, expected {prior_mean} from the four "
            f"prior games. The window is neither prior-only nor "
            f"including-today, so it is something else again."
        )
        return

    # A player's first game can have no history at all.
    first = probed["PTS_L5"].iloc[0]
    if not pd.isna(first):
        slot.fail(
            f"a player's first game carries PTS_L5={first}, so it is seeing "
            f"its own game or another player's history"
        )
        return

    slot.pass_(
        f"{len(features)} row(s), PTS_L10 built; window verified "
        f"arithmetically as prior-games-only (PTS_L5={float(last_l5)} on a "
        f"10/20/30/40/50 panel, not 30.0), first game null "
        f"({features['PTS_L10'].notna().sum()} scored rows)"
    )


# ---------------------------------------------------------------------------
# stage 3 — de-vig, Kelly, CLV: the production math, no division by zero
# ---------------------------------------------------------------------------


def stage_quant(slot: StageResult) -> None:
    from src.quant.advisory_sizing import recommended_units_binary
    from src.quant.devig_methods import devig_two_way, method_spread
    from src.quant.ev_engine import compute_clv

    # A standard two-sided market: devig must sum to ~1 and the methods
    # must agree on a price this balanced.
    multiplicative = devig_two_way(-110, -110)
    shin = devig_two_way(-110, -110, method="shin")
    total = multiplicative.fair_prob_a + multiplicative.fair_prob_b
    if not 0.98 <= total <= 1.02:
        slot.fail(f"de-vigged probabilities sum to {total:.4f}, not ~1")
        return
    diagnostics = method_spread(-110, -110)
    spread = diagnostics.get("spread")
    if spread is None or spread > 0.01:
        slot.fail(f"de-vig methods disagree by {spread!r} on -110/-110")
        return
    if abs(multiplicative.fair_prob_a - shin.fair_prob_a) > 0.01:
        slot.fail("multiplicative and Shin disagree beyond rounding on -110/-110")
        return

    # Quarter-Kelly by default; zero on a negative edge, never negative.
    #
    # THE HOLE THIS BLOCK FIRST HAD. The "on edge" case used the de-vigged
    # probability of a -110/-110 market, which is 0.500 — exactly no edge, so
    # the size came back 0.0. Together with the -EV and impossible-input cases,
    # which must also be 0.0, every assertion here was satisfied by 0.0: a
    # sizing function stubbed to `return 0` would have passed the whole stage.
    # A POSITIVE size on a real edge is therefore asserted explicitly.
    decimal_odds = 100.0 / 110.0 + 1.0  # -110 American
    no_edge = recommended_units_binary(multiplicative.fair_prob_a, decimal_odds)
    if no_edge.recommended_units != 0.0:
        slot.fail(
            f"a fair 50/50 priced at -110 is a NEGATIVE edge after vig, so it "
            f"must size 0, not {no_edge.recommended_units}"
        )
        return

    edged = recommended_units_binary(0.55, decimal_odds)
    if not edged.recommended_units > 0:
        slot.fail(
            f"a genuine 55% edge at -110 sized {edged.recommended_units}; "
            f"the sizer never returns anything but zero, so nothing here "
            f"exercises it"
        )
        return

    # And the guardrail: a bigger edge must not scale without bound.
    bigger = recommended_units_binary(0.70, decimal_odds)
    if bigger.recommended_units > 10.0:
        slot.fail(
            f"a 70% edge sized {bigger.recommended_units} units — the "
            f"fractional-Kelly cap is not holding"
        )
        return

    flat = recommended_units_binary(0.45, decimal_odds)
    if flat.recommended_units != 0.0:
        slot.fail(
            f"negative-EV input returned size {flat.recommended_units}, "
            "expected 0"
        )
        return

    # Pathological inputs must be refused, not crash: p outside (0,1),
    # odds <= 1, and a pickem-shaped market with no vig to remove — which
    # must still return fair probabilities, never negative ones.
    refused = recommended_units_binary(1.05, decimal_odds)
    if refused.recommended_units != 0.0:
        slot.fail("impossible probability was sized rather than refused")
        return
    try:
        pickem = devig_two_way(-100, -100)
        if pickem.fair_prob_a < 0 or pickem.fair_prob_b < 0:
            slot.fail("pickem de-vig produced a negative probability")
            return
        if abs(pickem.fair_prob_a - 0.5) > 1e-9:
            slot.fail(
                f"pickem de-vig is not 50/50: {pickem.fair_prob_a}"
            )
            return
    except ZeroDivisionError:
        slot.fail("de-vig divided by zero on a symmetric -100/-100 market")
        return

    clv = compute_clv(-110, -105)
    if clv.reason and "required" in clv.reason:
        slot.fail(f"CLV refused valid inputs: {clv.reason}")
        return

    slot.pass_(
        f"devig ok (sum={total:.4f}, method spread={spread:.4f}), "
        f"quarter-Kelly ok ({edged.recommended_units:.2f}u on a 55% edge, "
        f"0 on no-edge and on -EV, capped at {bigger.recommended_units:.2f}u), "
        "no ZeroDivisionError on pathological inputs, CLV computed"
    )


# ---------------------------------------------------------------------------
# stage 4 — the publication gate, both directions, never bypassed
# ---------------------------------------------------------------------------


def stage_gate(slot: StageResult) -> None:
    from src.quant.dfs_payouts import ProbabilitySource
    from src.quant.publication_gate import calibration_gate

    source = ProbabilitySource.MODEL

    # No evidence at all: withheld.
    v1 = calibration_gate(None, probability_source=source)
    if v1.allowed:
        slot.fail("gate ALLOWED with no calibration report")
        return
    if not v1.reason:
        slot.fail("withheld verdict carries no reason")
        return

    # Zero scored games: withheld. This is the state a fresh deploy is in,
    # and the gate holding here is what "hands-free" MEANS: the system
    # abstains rather than sending numbers nothing graded.
    v2 = calibration_gate({"n_scored": 0, "ece": None}, probability_source=source)
    if v2.allowed:
        slot.fail("gate ALLOWED with n_scored=0")
        return

    # TEST-ONLY synthetic evidence in the shape a scored season produces:
    # enough graded predictions, a passing ECE, a fresh timestamp. Nothing
    # in this script claims the numbers are real.
    now = datetime.now(timezone.utc)
    v3 = calibration_gate(
        {
            "status": "OK",  # TEST-ONLY
            "n_scored": 150,  # TEST-ONLY; above the gate's 100 minimum
            "ece": 0.03,  # TEST-ONLY; below the gate's 0.05 maximum
            "ece_gate_passed": True,  # TEST-ONLY
            "report_timestamp_pt": now.isoformat(),  # TEST-ONLY
        },
        probability_source=source,
    )
    if not v3.allowed:
        slot.fail(
            f"gate refused adequate synthetic evidence ({v3.reason}) — "
            "it can never open, which makes it a block, not a gate"
        )
        return

    slot.pass_(
        "withholds on no-evidence and n_scored=0; opens on labelled "
        f"TEST-ONLY evidence — first refusal reason: {v1.reason[:70]}..."
    )


# ---------------------------------------------------------------------------
# stage 5 — Discord dispatch, production payload rules (mocked POST)
# ---------------------------------------------------------------------------


def stage_dispatch(slot: StageResult) -> None:
    from src.notify.discord import (
        DiscordConfig,
        assert_payload_safe,
        build_abstention_embed,
        redact_webhook,
        send_embeds,
    )

    captured: list[dict[str, object]] = []

    def fake_post(url: str, **kwargs: object) -> _FakeResponse:
        captured.append({"url": url, **dict(kwargs)})
        return _FakeResponse({}, status_code=204)

    # A withheld night must still dispatch — the abstention itself is the
    # message, so a channel is never silently empty.
    embed = build_abstention_embed(
        "calibration evidence below threshold (TEST-ONLY harness run)"
    )
    result = send_embeds(
        [embed],
        config=DiscordConfig(dry_run=False, max_retries=2),
        webhook_url="https://discord.com/api/webhooks/123456789/harness-token",
        transport=fake_post,
    )
    if result.status != "OK":
        slot.fail(f"dispatch returned {result.status}: {result.reason}")
        return
    if not captured:
        slot.fail("transport was never called")
        return
    if "discord.com/api/webhooks" not in str(captured[0]["url"]):
        slot.fail(f"transport called with {captured[0]['url']!r}")
        return

    # The payload that left the machine must be inspectable and safe.
    payload = captured[0].get("json")
    if not isinstance(payload, dict) or not payload.get("embeds"):
        slot.fail(f"payload malformed: {str(payload)[:120]}")
        return
    assert_payload_safe(payload)  # raises on a credential in the payload
    if "harness-token" in json.dumps(payload):
        slot.fail("webhook token leaked into the payload")
        return

    # Redaction is the last line: an error that echoes the URL must come
    # back masked.
    redacted = redact_webhook(
        "boom at https://discord.com/api/webhooks/123456789/harness-token"
    )
    if "harness-token" in redacted:
        slot.fail("redact_webhook left the token in the text")
        return

    # Zero-embed dispatch is a named state, not an exception.
    empty = send_embeds(
        [],
        config=DiscordConfig(dry_run=False, max_retries=2),
        webhook_url="https://discord.com/api/webhooks/123456789/harness-token",
        transport=fake_post,
    )
    if empty.status != "DATA_NOT_AVAILABLE":
        slot.fail(f"zero-embed dispatch returned {empty.status}, expected DATA_NOT_AVAILABLE")
        return

    slot.pass_(
        "payload dispatched (204 mocked), safe, token redacted; "
        "zero-embed path returns DATA_NOT_AVAILABLE without raising"
    )


# ---------------------------------------------------------------------------
# stage 6 — settlement grading + ledger commit (isolated DB, fresh read)
# ---------------------------------------------------------------------------


def stage_settlement(slot: StageResult) -> None:
    from src.settlement.evaluator import settle_prop

    # Grade the four outcomes a ledger actually sees, through the
    # production grader. Box-score stats use the box-score field names
    # the evaluator reads ("points", not the panel's "PTS").
    win = settle_prop(
        market="PTS", predicted_line=Decimal("24.5"), predicted_side="OVER",
        player_stats={"points": 28}, odds=-110,
    )
    loss = settle_prop(
        market="PTS", predicted_line=Decimal("24.5"), predicted_side="OVER",
        player_stats={"points": 21}, odds=-110,
    )
    push = settle_prop(
        market="PTS", predicted_line=Decimal("25.0"), predicted_side="OVER",
        player_stats={"points": 25}, odds=-110,
    )
    void = settle_prop(
        market="PTS", predicted_line=Decimal("24.5"), predicted_side="OVER",
        player_stats=None, did_not_play=True,
    )
    if str(win.outcome) != "WIN":
        slot.fail(f"28 over 24.5 graded {win.outcome}")
        return
    if str(loss.outcome) != "LOSS":
        slot.fail(f"21 over 24.5 graded {loss.outcome}")
        return
    if str(push.outcome) != "PUSH":
        slot.fail(f"25 on 25.0 graded {push.outcome}")
        return
    if str(void.outcome) != "VOID":
        slot.fail(f"DNP graded {void.outcome}")
        return

    # Ledger commit on an isolated throwaway database, following the
    # repository's own test convention (tests/test_db_migrations.py): the
    # engine is built DIRECTLY, never through src/db/session.py, which
    # refuses a SQLite fallback for production on purpose.
    from sqlalchemy import create_engine, select
    from sqlalchemy.orm import Session

    from src.db.models import Base, PropResult

    # A TEMPORARY directory, not the repo root. The original wrote
    # `.harness_ledger.db` beside the source and unlinked it at the end, so a
    # stage that failed in between left a stray database in a working tree
    # whose .gitignore does not mention it.
    tmpdir = tempfile.TemporaryDirectory(prefix="propiq-harness-")
    ledger_path = Path(tmpdir.name) / "harness_ledger.db"
    engine = create_engine(f"sqlite:///{ledger_path}")
    Base.metadata.create_all(engine, tables=[PropResult.__table__])

    def ledger_row(row_id: int, player: str, line: str, graded, note: str) -> PropResult:
        outcome = str(graded.outcome)
        # Explicit id: SQLite will not autoincrement the Postgres-side
        # BigInteger primary key, and PK generation is Postgres behavior,
        # not the thing this stage tests. The commit and the fresh read
        # are.
        #
        # One player per row: the table's UNIQUE(game, player, market,
        # line, side, source) is real production behavior — one recorded
        # prop per player per game — and four grades of the SAME prop is
        # a shape the ledger refuses on purpose.
        #
        # The PUSH row carries the whole-number line it was graded on:
        # the table's own CHECK (push => whole line) is the repo's rule
        # that half-point lines cannot tie.
        return PropResult(
            id=row_id,
            projection_id=f"harness-{player.lower()}",
            run_id="harness-run-1",
            nba_game_id="0022600001",
            nba_player_id=f"2039{player}",
            player_name=f"HARNESS {player}",
            game_date=datetime(2026, 10, 24, tzinfo=timezone.utc).date(),
            market="PTS",
            predicted_line=Decimal(line),
            predicted_side="OVER",
            model_projection=Decimal("26.0"),
            prob_over=Decimal("0.61"),
            odds=-110,
            payout_multiplier=Decimal("1.909"),
            source="harness",
            is_pickem=False,
            actual_result=graded.actual_result,
            minutes_played=Decimal("32.0") if outcome != "VOID" else None,
            did_not_play=outcome == "VOID",
            outcome_status=outcome,
            stake_units=graded.stake_units,
            profit_units=graded.profit_units,
            settlement_note=note,
            settled_at_utc=datetime.now(timezone.utc),
        )

    note = "TEST-ONLY synthetic settlement — harness run"
    with Session(engine) as session:
        session.add_all(
            [ledger_row(1, "A", "24.5", win, note),
             ledger_row(2, "B", "24.5", loss, note),
             ledger_row(3, "C", "25.0", push, note),
             ledger_row(4, "D", "24.5", void, note)]
        )
        session.commit()

    # A FRESH connection reading the rows back is the whole point of a
    # commit: one only the writing connection can see is a commit nothing
    # else in the stack can read.
    with engine.connect() as fresh:
        rows = list(fresh.execute(select(PropResult)))
    grades = {row.outcome_status for row in rows}
    if grades != {"WIN", "LOSS", "PUSH", "VOID"}:
        slot.fail(f"fresh connection read back {sorted(grades)}")
        return

    # The ledger rows must carry their TEST-ONLY label — synthetic
    # evidence must never be mistakable for a graded season.
    if not all("TEST-ONLY" in (row.settlement_note or "") for row in rows):
        slot.fail("a ledger row lost its TEST-ONLY label")
        return

    # And the graded rows must carry the grader's own profit figures, not
    # numbers this script invented.
    profits = sorted(
        f"{row.profit_units}" for row in rows if row.profit_units is not None
    )
    if len(profits) < 3:
        slot.fail(f"expected W/L/P profit figures, got {profits}")
        return

    engine.dispose()
    tmpdir.cleanup()

    slot.pass_(
        "WIN/LOSS/PUSH/VOID graded; 4 rows committed and read back through "
        "a fresh connection, each labelled TEST-ONLY, grader's own profits"
    )


# ---------------------------------------------------------------------------
# stage 0 — preflight: every production module the loop needs must import
# ---------------------------------------------------------------------------


PREFLIGHT_MODULES = [
    "src.ingestion.propline",
    "src.features.builder",
    "src.quant.devig_methods",
    "src.quant.advisory_sizing",
    "src.quant.ev_engine",
    "src.quant.publication_gate",
    "src.notify.discord",
    "src.settlement.evaluator",
    "src.db.models",
    "main",
    "scheduler_worker",
]


def stage_preflight(slot: StageResult) -> None:
    missing = []
    for module in PREFLIGHT_MODULES:
        try:
            __import__(module)
        except Exception as exc:  # a broken wire is the subject here
            missing.append(f"{module}: {type(exc).__name__}: {exc}")
    if missing:
        slot.fail("; ".join(missing))
        return
    slot.pass_(f"all {len(PREFLIGHT_MODULES)} production modules import cleanly")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    print("PropIQ autonomous-loop harness — TEST-ONLY, no network, no live data")
    results: list[StageResult] = []

    stages = [
        ("PREFLIGHT — production modules import", stage_preflight),
        ("RAW PULL — production PropLine client, mocked HTTP", stage_raw_pull),
        ("FEATURES — production builder, synthetic panel", stage_features),
        ("QUANT — de-vig / quarter-Kelly / CLV", stage_quant),
        ("PUBLICATION GATE — real gate, both directions", stage_gate),
        ("DISPATCH — production Discord payload, mocked POST", stage_dispatch),
        ("SETTLEMENT + LEDGER — grading, commit, fresh read-back", stage_settlement),
    ]

    for name, fn in stages:
        slot = StageResult(name)
        try:
            fn(slot)
        except Exception:  # a stage crashing IS its result
            slot.fail("EXCEPTION: " + traceback.format_exc(limit=3))
        results.append(slot)
        mark = "PASS" if slot.ok else "FAIL"
        first_line = slot.detail.splitlines()[0] if slot.detail else ""
        print(f"[{mark}] {first_line[:110]}")

    print("\n" + "=" * 70)
    failed = [r for r in results if not r.ok]
    for r in results:
        print(f"  {'PASS' if r.ok else 'FAIL'}  {r.name}")
        if not r.ok:
            print(f"       {r.detail}")

    if failed:
        print(f"\nVERDICT: {len(failed)} stage(s) FAILED — the loop is NOT hands-free.")
        print("This harness proves code wiring only; fix the failures above, then")
        print("run railway_healthcheck and verify_wiring for deploy readiness.")
        return 1

    print("\nVERDICT: all stages passed — the production loop composes end-to-end")
    print("under mocked boundaries. This is NOT live-readiness proof; run")
    print("`python main.py railway_healthcheck` and `python main.py verify_wiring`")
    print("against the deployed environment for that.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
