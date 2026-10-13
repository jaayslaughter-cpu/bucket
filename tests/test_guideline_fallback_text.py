"""
The missing-guideline warning must describe the fallback the code has.

FOUND IN A RAILWAY LOG, 2026-10-12 (deploy 4ded47f4, the 09:00 PT slate):

    WARNING propiq.main | MASTER GUIDELINE NOT FOUND at
    config/master_guideline_props.yaml — prop-source selection falls back to
    config.json `pickem.approved_sources`.

It does not. Grepped across the repository, the string ``config.json`` appears
in exactly one place: that warning. Nothing reads such a file, there is no
``pickem.approved_sources`` reader anywhere, and ``config.json`` is neither in
the tree nor gitignored — it has never existed. What `ingest_prop_lines`
actually does with no guideline is leave ``books = None`` and pass that to
``pull_nba_prop_lines``, which applies NO bookmaker filter at all.

So the warning sends an operator to create a file that will be ignored, and
hides the real behaviour: every source PropLine returns is accepted, including
ones they may not have approved. That is the opposite of reassuring.

Documentation outliving its facts, which this repository keeps two test files
to catch — in a log line, where neither of them looks.

RESEARCH_ONLY. The text of a warning.
"""

from __future__ import annotations

import logging
from pathlib import Path

import main


def _preflight_warnings(monkeypatch, caplog) -> str:
    """Run `preflight` with no guideline and no database, and return its log.

    Through the real function rather than a helper extracted for the test: the
    warning's job is to appear in a boot log, and a function that exists only
    so a test can call it would not prove that.
    """
    monkeypatch.setenv("PROPIQ_MASTER_GUIDELINE", "/nonexistent/guideline.yaml")
    with caplog.at_level(logging.WARNING):
        report = main.preflight(require_db=False)
    assert report["master_guideline"] == {"applied": False}
    return " ".join(r.getMessage() for r in caplog.records)


def test_the_warning_does_not_name_a_file_nothing_reads(monkeypatch, caplog):
    text = _preflight_warnings(monkeypatch, caplog)

    assert "MASTER GUIDELINE NOT FOUND" in text
    assert "config.json" not in text, (
        "the warning still points at config.json, which no code in this "
        "repository reads and which has never existed in the tree"
    )
    assert "pickem.approved_sources" not in text, (
        "the warning still names a key with no reader"
    )


def test_the_warning_states_what_actually_happens(monkeypatch, caplog):
    """The real consequence, in the operator's words: no book filter, so every
    source the feed returns is accepted."""
    text = _preflight_warnings(monkeypatch, caplog).lower()
    assert "no bookmaker filter" in text or "no book filter" in text, (
        "the warning does not say that every source is accepted, which is what "
        "books=None does in ingest_prop_lines"
    )
    assert "approved_prop_sources" in text, (
        "the warning does not name the key that WOULD restrict the sources"
    )


def test_the_warning_names_the_path_that_is_actually_read(monkeypatch, caplog):
    """
    cubic, PR #9: the warning named the DEFAULT path while the loader honoured
    $PROPIQ_MASTER_GUIDELINE, so an operator with the override set was told to
    create a file that would never be read.
    """
    monkeypatch.setenv("PROPIQ_MASTER_GUIDELINE", "/custom/elsewhere.yaml")
    with caplog.at_level(logging.WARNING):
        main.preflight(require_db=False)

    text = " ".join(r.getMessage() for r in caplog.records)
    assert "/custom/elsewhere.yaml" in text, (
        "the warning names the default path, not the override that is actually "
        "resolved, so the file the operator creates will be ignored"
    )


def test_the_missing_guideline_path_really_leaves_the_filter_unset(monkeypatch):
    """
    cubic, PR #9: the AST check only proved ``books = None`` appeared somewhere
    in the function. This drives the real call and asserts what
    `pull_nba_prop_lines` receives, so a later assignment that re-enables
    filtering cannot pass.
    """
    import sys
    import types

    seen: dict[str, object] = {}

    def fake_pull(bookmakers=None, **kwargs):
        seen["bookmakers"] = bookmakers
        return []

    module = types.ModuleType("src.ingestion.propline")
    module.pull_nba_prop_lines = fake_pull
    monkeypatch.setitem(sys.modules, "src.ingestion.propline", module)

    main.ingest_prop_lines(None)
    assert "bookmakers" in seen, "pull_nba_prop_lines was never called"
    assert seen["bookmakers"] is None, (
        f"the missing-guideline path passed bookmakers={seen['bookmakers']!r}, "
        "so the warning's claim that no filter is applied is false"
    )

    # And the other direction: a guideline DOES restrict them.
    seen.clear()
    main.ingest_prop_lines({"approved_prop_sources": ["underdog"]})
    assert seen["bookmakers"] == ["underdog"]


def test_the_example_file_does_not_contradict_the_warning():
    """
    cubic, PR #9: the warning points an operator at the .example, which still
    said the pipeline "falls back to config.json's pickem.approved_sources",
    called its own shape "a GUESS", and told the reader to change
    `ingest_prop_lines` to read whatever keys they use -- code that already
    reads `approved_prop_sources`. AGENTS.md: when code and doc disagree, fix
    both.
    """
    text = (
        Path(__file__).resolve().parents[1]
        / "config/master_guideline_props.yaml.example"
    ).read_text(encoding="utf-8")
    assert "config.json" not in text, "the template still promises a config.json fallback"
    assert "GUESS" not in text
    assert "approved_prop_sources" in text
    assert "every source the feed returns is accepted" in text.lower(), (
        "the template does not state the consequence of its own absence"
    )


def test_the_key_the_warning_names_is_the_key_the_code_reads():
    """
    WHY THE OLD TEXT SURVIVED: nothing tied it to the code. This does. The
    warning tells an operator which key restricts the sources; if
    `ingest_prop_lines` ever reads a different one, this fails and the message
    has to follow.
    """
    import ast
    import inspect
    import textwrap

    source = inspect.getsource(main.ingest_prop_lines)
    keys = {
        node.value
        for node in ast.walk(ast.parse(textwrap.dedent(source)))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert "approved_prop_sources" in keys, (
        "ingest_prop_lines no longer reads `approved_prop_sources`, so the "
        "guideline warning names a key that does nothing"
    )

    # And the absent-guideline path must still leave the book filter unset,
    # which is what makes "every source is accepted" true.
    tree = ast.parse(textwrap.dedent(source))
    assigns_none = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(getattr(t, "id", None) == "books" for t in n.targets)
        and isinstance(n.value, ast.Constant) and n.value.value is None
    ]
    assert assigns_none, (
        "`books` is no longer defaulted to None, so the warning's claim that "
        "no bookmaker filter is applied may be false"
    )


def test_an_empty_override_does_not_crash_preflight(monkeypatch, caplog):
    """
    cubic, PR #10: an empty ``$PROPIQ_MASTER_GUIDELINE`` became ``Path('.')``,
    which always exists, so `load_master_guideline` went on to ``open('.')``
    and preflight died with IsADirectoryError instead of logging the warning.
    Every other env reader in main.py guards with ``(… or "").strip()``.
    """
    for value in ("", "   "):
        monkeypatch.setenv("PROPIQ_MASTER_GUIDELINE", value)
        assert main.master_guideline_path() == main.MASTER_GUIDELINE_DEFAULT_PATH
        with caplog.at_level(logging.WARNING):
            report = main.preflight(require_db=False)
        assert report["master_guideline"] == {"applied": False}


def test_a_padded_override_is_stripped_not_taken_literally(monkeypatch):
    monkeypatch.setenv("PROPIQ_MASTER_GUIDELINE", "  /custom/x.yaml  ")
    assert str(main.master_guideline_path()) == "/custom/x.yaml"
