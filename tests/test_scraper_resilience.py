"""
What an ingestion client must survive: a rate limiter, and a provider that
changes the shape of its own payload.

REPORTED 2026-10-18 as "our Propline, Underdog, Sleeper and PrizePicks
scrapers are all failing". Three of those four have no client in this tree
and never have -- `tests/test_pickem_docs_match_the_tree.py` is the guard
for how that belief arose. This file is about the one that does exist, and
about the two defects the report led to, both measured rather than assumed:

1. THREE READERS OF ONE RULE. `Retry-After` was parsed in three places --
   `settlement/boxscore_fetcher`, `ingestion/espn_client`, `ingestion/
   propline` -- and they disagreed. ESPN returned nan for ``nan`` (because
   `min(nan, 60.0)` is nan) and `time.sleep(nan)` raises ValueError.
   PropLine returned inf for ``inf`` (OverflowError), 86400.0 for ``86400``
   (a slate job asleep for a day, with no timeout above it), and 0.0 for
   both ``nan`` and ``-1`` -- retrying a rate limiter instantly.

2. A PROVIDER'S RESTRUCTURED PAYLOAD ENDED THE SLATE. `normalize_event_props`
   was called outside the per-event guard, and `main.ingest_prop_lines`
   guarded only its import -- while its docstring promised that an absent
   source "is not a pipeline failure". An AttributeError from a shape change
   escaped both and took out features, scoring and notification over an
   OPTIONAL source.

RESEARCH_ONLY. No odds are fetched here; every payload is a literal.
"""

from __future__ import annotations

import ast
import inspect
import math

import pytest

import main as pipeline
import src.ingestion.propline as propline
from src.ingestion.espn_client import _retry_after_seconds as espn_retry_after
from src.settlement.boxscore_fetcher import retry_after_seconds as boxscore_retry_after
from src.utils.http_retry import MAX_RETRY_AFTER_SECONDS, retry_after_seconds


class _Resp:
    def __init__(self, value):
        self.headers = {"Retry-After": value} if value is not None else {}


def _propline_retry_after(value, fallback=1.0):
    return propline.PropLineClient._retry_after_seconds(_Resp(value), fallback)


ALL_THREE = (
    ("propline", _propline_retry_after),
    ("espn", lambda v, fallback=1.0: espn_retry_after(_Resp(v), fallback)),
    ("boxscore", lambda v, fallback=1.0: boxscore_retry_after(_Resp(v).headers, fallback)),
)


# --- 1. one rule ------------------------------------------------------------


@pytest.mark.parametrize("client,parse", ALL_THREE)
@pytest.mark.parametrize("hostile", ["nan", "inf", "-inf", "-1", "0"])
def test_no_client_returns_a_duration_that_sleep_refuses(client, parse, hostile):
    """
    Every one of these raised from `time.sleep` before the fix, in at least
    one of the three clients -- turning a rate limit, which is a wait, into a
    crash.
    """
    seconds = parse(hostile)
    assert math.isfinite(seconds), f"{client} returned {seconds!r} for Retry-After: {hostile}"
    assert seconds > 0, f"{client} returned {seconds!r} -- an instant retry against a limiter"


@pytest.mark.parametrize("client,parse", ALL_THREE)
def test_no_client_will_sleep_a_slate_job_for_an_hour(client, parse):
    assert parse("86400") == MAX_RETRY_AFTER_SECONDS, (
        f"{client} would honour a 24-hour Retry-After inside a job whose "
        f"window closes at tip-off"
    )


@pytest.mark.parametrize("client,parse", ALL_THREE)
def test_a_real_retry_after_is_still_honoured(client, parse):
    """The cap and the hardening must not swallow the header's actual purpose."""
    assert parse("5") == 5.0


def test_the_three_clients_agree_on_every_value():
    """
    The point of consolidating. Any value, same answer -- otherwise the
    weakest reader decides what happens in production.
    """
    for value in ["nan", "inf", "-1", "0", "5", "86400", "Wed, 21 Oct 2026 07:28:00 GMT", None]:
        answers = {name: parse(value) for name, parse in ALL_THREE}
        assert len(set(answers.values())) == 1, f"Retry-After {value!r}: {answers}"


def test_the_http_date_form_falls_back_rather_than_being_guessed():
    assert retry_after_seconds({"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}, 3.0) == 3.0


def test_there_is_exactly_one_implementation_of_the_rule():
    """
    AST-walked across the clients: each must DELEGATE. A fourth client that
    writes `float(response.headers["Retry-After"])` of its own reintroduces
    the drift this consolidation removed.
    """
    import pathlib

    offenders = []
    for path in [
        "src/ingestion/propline.py",
        "src/ingestion/espn_client.py",
        "src/settlement/boxscore_fetcher.py",
        "src/ingestion/nba_playbyplay.py",
    ]:
        source = pathlib.Path(path).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports_the_rule = any(
            isinstance(n, ast.ImportFrom) and n.module == "src.utils.http_retry"
            for n in ast.walk(tree)
        )
        if not imports_the_rule:
            offenders.append(path)
    assert offenders == [], f"these parse Retry-After without the shared rule: {offenders}"


# --- 2. a restructured payload ----------------------------------------------


RESTRUCTURED = {
    "bookmakers keyed by book name": {
        "id": "e1",
        "commence_time": "2026-10-18T02:00:00Z",
        "bookmakers": {"underdog": {"markets": []}},
    },
    "markets keyed by market name": {
        "id": "e1",
        "commence_time": "2026-10-18T02:00:00Z",
        "bookmakers": [{"key": "underdog", "markets": {"player_points": {}}}],
    },
    "outcomes keyed by side": {
        "id": "e1",
        "commence_time": "2026-10-18T02:00:00Z",
        "bookmakers": [
            {"key": "underdog", "markets": [{"key": "player_points", "outcomes": {"over": {}}}]}
        ],
    },
    "outcomes replaced by a string": {
        "id": "e1",
        "commence_time": "2026-10-18T02:00:00Z",
        "bookmakers": [
            {"key": "underdog", "markets": [{"key": "player_points", "outcomes": "n/a"}]}
        ],
    },
}


def _client_returning(payload):
    class _FakeClient:
        quota = type("_Q", (), {"as_dict": staticmethod(dict)})()

        def fetch_events(self):
            return [{"id": "e1"}, {"id": "e2"}]

        def fetch_event_props(self, *a, **k):
            return payload

    return lambda **kwargs: _FakeClient()


@pytest.mark.parametrize("shape", sorted(RESTRUCTURED))
def test_a_restructured_payload_abstains_instead_of_ending_the_run(shape, monkeypatch):
    monkeypatch.setattr(propline, "PropLineClient", _client_returning(RESTRUCTURED[shape]))
    snapshots = propline.pull_nba_prop_lines()
    assert len(snapshots) == 1
    assert snapshots[0].status == "DATA_NOT_AVAILABLE", (
        "a shape this parser cannot read was reported as a VALID empty board, "
        "which downstream cannot tell from a quiet night"
    )


@pytest.mark.parametrize("shape", sorted(RESTRUCTURED))
def test_the_message_says_the_shape_changed_rather_than_that_there_were_no_lines(
    shape, monkeypatch
):
    """
    "No lines" and "we could not read the response" send an operator to two
    completely different places. The snapshot has to say which it is.
    """
    monkeypatch.setattr(propline, "PropLineClient", _client_returning(RESTRUCTURED[shape]))
    message = propline.pull_nba_prop_lines()[0].message or ""
    assert "shape not understood" in message
    assert "restructured" in message


def test_one_unreadable_event_does_not_discard_the_readable_ones(monkeypatch):
    """
    Per-event isolation, not all-or-nothing: a provider that restructures one
    game's response must cost that game, not the slate.
    """
    good = {
        "id": "e2",
        "commence_time": "2026-10-18T02:00:00Z",
        "bookmakers": [
            {
                "key": "fanduel",
                "markets": [
                    {
                        "key": "player_points",
                        "last_update": "2026-10-18T01:00:00Z",
                        "outcomes": [
                            {"name": "Over", "description": "A Player", "point": 24.5,
                             "price": -110, "last_update": "2026-10-18T01:00:00Z"},
                            {"name": "Under", "description": "A Player", "point": 24.5,
                             "price": -110, "last_update": "2026-10-18T01:00:00Z"},
                        ],
                    }
                ],
            }
        ],
    }
    bad = RESTRUCTURED["bookmakers keyed by book name"]
    payloads = iter([bad, good])

    class _FakeClient:
        quota = type("_Q", (), {"as_dict": staticmethod(dict)})()

        def fetch_events(self):
            return [{"id": "e1"}, {"id": "e2"}]

        def fetch_event_props(self, *a, **k):
            return next(payloads)

    monkeypatch.setattr(propline, "PropLineClient", lambda **k: _FakeClient())
    snapshots = propline.pull_nba_prop_lines()
    valid = [s for s in snapshots if s.status == "VALID"]
    assert valid, "the readable event's lines were thrown away with the unreadable one's"
    assert sum(len(s.lines) for s in valid) >= 1


def test_the_slate_survives_a_prop_source_that_raises(monkeypatch):
    """
    `ingest_prop_lines` promised this in its docstring and did not do it: only
    the IMPORT was guarded. The pull itself was not, so step [3] ended the run.
    """
    def _boom(*a, **k):
        raise RuntimeError("provider restructured everything")

    monkeypatch.setattr(propline, "pull_nba_prop_lines", _boom)
    df = pipeline.ingest_prop_lines({}, persist=False)
    assert df.empty, "a failed prop pull invented rows"


def test_the_failure_is_loud_even_though_it_is_survived(monkeypatch, caplog):
    """Isolation must not become silence: the run continues, and the log says
    why there are no prop lines in it."""
    import logging

    def _boom(*a, **k):
        raise RuntimeError("provider restructured everything")

    monkeypatch.setattr(propline, "pull_nba_prop_lines", _boom)
    with caplog.at_level(logging.ERROR):
        pipeline.ingest_prop_lines({}, persist=False)
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "PROP-LINE INGEST FAILED" in text
    assert "slate continues" in text


def test_the_isolation_is_around_the_pull_not_only_the_import():
    """
    AST: the call to `pull_nba_prop_lines` must sit inside a Try. The original
    defect was a try that covered the import statement and nothing else, which
    reads as handled and is not.
    """
    tree = ast.parse(inspect.getsource(pipeline.ingest_prop_lines).lstrip())
    guarded = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Name)
                and inner.func.id == "pull_nba_prop_lines"
            ):
                guarded = True
    assert guarded, "the prop pull is not inside a try — step [3] can still end the run"


# --- what is NOT here -------------------------------------------------------


def test_a_renamed_key_still_yields_no_rows_rather_than_raising():
    """
    The parser reads with `.get` and `or []` throughout, so a RENAMED container
    is already handled and must stay that way. Only RESTRUCTURING -- a list
    becoming an object -- was the crash.
    """
    rows = propline.normalize_event_props(
        {"id": "e1", "commence_time": "2026-10-18T02:00:00Z", "books": []}
    )
    assert rows == []


# --- the diagnostic itself --------------------------------------------------
#
# A diagnostic that lies is worse than none: the 2026-10-18 report came from a
# status table that described clients nobody had written. So the script is
# tested for exactly that -- that it reports from the tree and the wire, and
# that "no client" and "broken client" stay different words.


import scripts.test_scrapers as diag  # noqa: E402


def test_a_source_with_no_client_is_absent_not_failed(capsys):
    """
    PrizePicks, which stays unwritten on purpose. Underdog and Sleeper WERE
    absent when this file was written and are not any more -- the clients
    were built on 2026-10-18 -- which is why this now names the one platform
    the claim is still true of.
    """
    code = diag.main(["--source", "prizepicks"])
    out = capsys.readouterr().out
    assert "ABSENT" in out
    assert code == diag.EXIT_OK, "a platform nobody wrote a client for was reported as a failure"


def test_absent_says_how_those_lines_do_reach_the_pipeline(capsys):
    """
    PrizePicks lines are not missing -- they arrive as a book key inside
    PropLine's payload. An operator told only "ABSENT" would go and build a
    scraper this project does not need, into a bot wall it must not bypass.
    """
    diag.main(["--source", "prizepicks"])
    out = capsys.readouterr().out
    assert "DFS_BOOKS" in out


@pytest.mark.parametrize("source", ["underdog", "sleeper"])
def test_a_built_but_unconfigured_client_is_neither_absent_nor_failed(source, capsys):
    """
    Three distinct states, and conflating any two of them misleads: ABSENT is
    "nobody wrote it", UNCONFIGURED is "it is written and ships with no board
    paths, because nobody here has seen a response", FAILED is "something
    that should work did not". Only the last is a non-zero exit.
    """
    code = diag.main(["--source", source])
    out = capsys.readouterr().out
    assert "UNCONFIGURED" in out
    assert "ABSENT" not in out
    assert code == diag.EXIT_OK
    assert "config/pickem_sources.yaml" in out, (
        "it says the source is unconfigured but not which file to edit"
    )


def test_a_missing_key_is_a_failure_with_a_nonzero_exit(monkeypatch, capsys):
    monkeypatch.delenv("PROPLINE_API_KEY", raising=False)
    code = diag.main(["--source", "propline", "--offline"])
    assert code == diag.EXIT_SOURCE_FAILED
    assert "PROPLINE_API_KEY" in capsys.readouterr().out


def test_the_key_itself_is_never_printed(monkeypatch, capsys):
    monkeypatch.setenv("PROPLINE_API_KEY", "sk-live-do-not-print-me")
    diag.main(["--source", "propline", "--offline"])
    assert "do-not-print-me" not in capsys.readouterr().out


def test_an_empty_board_is_an_answer_not_a_failure(monkeypatch, capsys):
    monkeypatch.setenv("PROPLINE_API_KEY", "x")
    monkeypatch.setattr(
        propline,
        "pull_nba_prop_lines",
        lambda *a, **k: [
            propline.PropLineSnapshotResult(
                source="propline", status="DATA_NOT_AVAILABLE",
                message="No upcoming NBA events — expected off-season.",
            )
        ],
    )
    code = diag.main(["--source", "propline"])
    out = capsys.readouterr().out
    assert "EMPTY" in out
    assert code == diag.EXIT_OK
    assert "off-season" in out


def test_parsed_lines_are_printed_so_the_shape_can_be_eyeballed(monkeypatch, capsys):
    """The ask was "print status codes and sample parsed prop lines" — the
    sample is what tells you a parser is reading the right fields."""
    monkeypatch.setenv("PROPLINE_API_KEY", "x")
    row = propline.PropLineRow(
        source="underdog", player_name="A Player", market="PTS", line=24.5,
        over_odds_american=-110, under_odds_american=-110,
    )
    monkeypatch.setattr(
        propline,
        "pull_nba_prop_lines",
        lambda *a, **k: [
            propline.PropLineSnapshotResult(source="underdog", status="VALID", lines=[row])
        ],
    )
    code = diag.main(["--source", "propline", "--sample", "3"])
    out = capsys.readouterr().out
    assert code == diag.EXIT_OK
    assert "A Player" in out and "24.5" in out and "PTS" in out


def test_the_source_table_cannot_claim_a_client_that_is_not_there(capsys):
    """
    The table says where each client lives; the filesystem decides whether it
    does. If someone adds an entry pointing at a module they have not written,
    the script must say FAILED rather than print a confident ABSENT or crash.
    """
    import pathlib

    for name, spec in diag.SOURCES.items():
        module = spec["module"]
        if module is None:
            continue
        assert (pathlib.Path(".") / module).is_file(), (
            f"the diagnostic claims {name} has a client at {module}, which is "
            f"not in the tree"
        )
