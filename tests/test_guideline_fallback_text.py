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
