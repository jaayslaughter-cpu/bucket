"""
Serving P(over) at the POSTED line, from the fitted count distribution.

WHY THIS FILE EXISTS. Three consecutive live slates scored 0 of ~199 rows per
market with a healthy, freshly trained artifact on disk. The cause was not a
missing model, a bad threshold or a broken webhook: the serving path resolved
the XGBoost CLASSIFIER, which can only answer at the line its labels were
built from, and `mask_probabilities_at_unsupported_lines` correctly refused to
publish that number beside a sportsbook's line.

These tests pin the four things that made that failure possible and silent:
the resolver demanding a sidecar the distribution artifact was never designed
to have, the posted line not reaching the model at all, a missing line
becoming a number, and the component default itself.

RESEARCH_ONLY.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import main as propiq_main
from src.models.distribution_adapter import DistributionPropModel


def _panel(n: int = 400, market: str = "PTS") -> pd.DataFrame:
    """Rows with the one column the distribution model reads, plus a label."""
    rng = np.random.default_rng(11)
    actual = rng.poisson(18.0, n).astype(float)
    return pd.DataFrame({
        "PLAYER_ID": [f"espn:{i % 40}" for i in range(n)],
        "PLAYER_NAME": [f"Player {i % 40:02d}" for i in range(n)],
        "GAME_DATE": pd.date_range("2026-01-01", periods=n, freq="h"),
        f"{market}_L2": rng.normal(18.0, 2.0, n).clip(1.0),
        market: actual,
        "RESEARCH_LINE": rng.normal(18.0, 2.0, n).clip(1.0),
        "over_hit": (actual > 18.0).astype(float),
    })


@pytest.fixture
def fitted_artifact(tmp_path):
    """A real fitted DistributionPropModel on disk, not a stub."""
    frame = _panel()
    model = DistributionPropModel(target_market="PTS")
    model.fit(frame, frame)
    target = tmp_path / "distribution_PTS.json"
    model.save(target)
    return target


# --- the resolver's sidecar rule ------------------------------------------


def test_the_distribution_artifact_resolves_without_a_meta_sidecar(
    monkeypatch, tmp_path, fitted_artifact
):
    """
    THE BUG THAT HID A USABLE MODEL. The resolver kept only candidates with a
    `.meta.json` beside them — correct for a booster, whose feature_cols live
    there — and `distribution_{MARKET}.json` carries target_market and the
    fitted dispersion INSIDE the file. Requiring a sidecar it never had
    rejected the only component that can answer at a posted line.
    """
    assert not fitted_artifact.with_suffix(".meta.json").exists()
    monkeypatch.setattr(
        "src.models.compare.load_comparison_config",
        lambda *a, **k: {"artifacts_dir": str(tmp_path)},
    )
    path, how = propiq_main.resolve_model_artifact(
        market="PTS", component="distribution"
    )
    assert path == fitted_artifact, how


def test_the_booster_still_needs_its_sidecar(monkeypatch, tmp_path):
    """The rule was right for xgboost and must stay for it: without
    feature_cols the column order is a guess."""
    (tmp_path / "xgboost_PTS.json").write_text("{}")
    monkeypatch.setattr(
        "src.models.compare.load_comparison_config",
        lambda *a, **k: {"artifacts_dir": str(tmp_path)},
    )
    monkeypatch.delenv("PROPIQ_MODEL", raising=False)
    monkeypatch.setattr(propiq_main, "MODEL_ARTIFACT_DEFAULT", tmp_path / "absent.json")
    path, _how = propiq_main.resolve_model_artifact(market="PTS", component="xgboost")
    assert path is None


def test_the_legacy_xgboost_default_is_never_served_as_a_distribution(
    monkeypatch, tmp_path
):
    legacy = tmp_path / "xgb_prop_over.json"
    legacy.write_text("{}")
    monkeypatch.setattr(propiq_main, "MODEL_ARTIFACT_DEFAULT", legacy)
    monkeypatch.setattr(
        "src.models.compare.load_comparison_config",
        lambda *a, **k: {"artifacts_dir": str(tmp_path / "empty")},
    )
    path, _how = propiq_main.resolve_model_artifact(
        market="PTS", component="distribution"
    )
    assert path is None, "a booster was handed to a caller asking for a distribution"


# --- scoring at the posted line -------------------------------------------


def _prop_lines(names, line=20.5, market="PTS"):
    return pd.DataFrame(
        [{"player_name": n, "market": market, "line": line} for n in names]
    )


def test_probabilities_are_produced_at_the_posted_line(fitted_artifact):
    """
    THE WHOLE POINT. Rows with a posted line must come back with a probability
    in (0, 1) — the thing three live slates could not do.
    """
    features = _panel(80).drop(columns=["RESEARCH_LINE"])
    lines = _prop_lines(sorted(set(features["PLAYER_NAME"])))

    scored = propiq_main.score_prob_over_distribution(
        features, lines, fitted_artifact
    )
    values = pd.to_numeric(scored, errors="coerce")

    assert values.notna().sum() == len(features), (
        "every row had a posted line, so every row should carry a probability"
    )
    assert ((values > 0.0) & (values < 1.0)).all()
    assert scored.attrs["target_market"] == "PTS"


def test_the_probability_moves_with_the_line_which_the_classifier_could_not_do():
    """
    THE PROPERTY THAT MAKES THIS COMPONENT THE RIGHT ONE. A classifier returns
    the same number at every line; this must not. P(over) at a low line has to
    exceed P(over) at a high one.
    """
    frame = _panel()
    model = DistributionPropModel(target_market="PTS")
    model.fit(frame, frame)
    features = _panel(50)

    low = pd.to_numeric(model.predict_probability_over(features, 8.5))
    high = pd.to_numeric(model.predict_probability_over(features, 30.5))

    assert (low > high).all(), "the probability did not fall as the line rose"


def test_a_row_with_no_posted_line_abstains_rather_than_scoring_zero(fitted_artifact):
    """A line we do not have is not a line of zero: there is nothing to be over."""
    features = _panel(40).drop(columns=["RESEARCH_LINE"])
    named = sorted(set(features["PLAYER_NAME"]))[:5]
    scored = propiq_main.score_prob_over_distribution(
        features, _prop_lines(named), fitted_artifact
    )
    values = pd.to_numeric(scored, errors="coerce")

    covered = features["PLAYER_NAME"].isin(named)
    assert values[covered].notna().all()
    assert values[~covered].isna().all(), (
        "a row with no posted line was given a probability anyway"
    )


def test_the_posted_line_reaches_the_model_through_the_shared_crosswalk():
    """
    A SECOND NAME MATCHER WOULD SPLIT THE PIPELINE. The projection join
    resolves book spellings through `id_crosswalk.resolve_board_names`; if this
    path did not, a player whose line reaches the board would carry no
    probability. "Alexandre Sarr" is a live alias.
    """
    features = pd.DataFrame({
        "PLAYER_NAME": ["Alex Sarr", "Nobody Here"],
        "PTS_L2": [14.0, 9.0],
    })
    lines = _prop_lines(["Alexandre Sarr"], line=15.5)

    got = propiq_main.posted_line_for_market(features, lines, "PTS")
    assert got.iloc[0] == pytest.approx(15.5), (
        "the book's 'Alexandre Sarr' did not resolve onto the panel's 'Alex Sarr'"
    )
    assert pd.isna(got.iloc[1])


def test_a_market_with_no_lines_yields_no_posted_line():
    features = pd.DataFrame({"PLAYER_NAME": ["Alex Sarr"], "PTS_L2": [14.0]})
    got = propiq_main.posted_line_for_market(
        features, _prop_lines(["Alex Sarr"], market="REB"), "PTS"
    )
    assert pd.isna(got.iloc[0])
    assert propiq_main.posted_line_for_market(
        features, pd.DataFrame(), "PTS"
    ).isna().all()


# --- the component default -------------------------------------------------


def test_distribution_is_the_default_component(monkeypatch):
    monkeypatch.delenv(propiq_main.ENV_SCORING_COMPONENT, raising=False)
    assert propiq_main.scoring_component() == "distribution"


@pytest.mark.parametrize("value,expected", [
    ("xgboost", "xgboost"), ("DISTRIBUTION", "distribution"), (" xgboost ", "xgboost"),
])
def test_the_component_can_be_overridden(monkeypatch, value, expected):
    monkeypatch.setenv(propiq_main.ENV_SCORING_COMPONENT, value)
    assert propiq_main.scoring_component() == expected


def test_an_unknown_component_falls_back_loudly(monkeypatch, caplog):
    import logging

    monkeypatch.setenv(propiq_main.ENV_SCORING_COMPONENT, "catboost_maybe")
    with caplog.at_level(logging.ERROR, logger="propiq.main"):
        assert propiq_main.scoring_component() == "distribution"
    assert "catboost_maybe" in caplog.text


# --- the alert that renamed what it was diagnosing ------------------------


def test_the_alert_body_survives_discord_markdown():
    """
    THE DEFECT AS IT REACHED THE CHANNEL. The boot alert said
    "PROPIQ_MODEL unset, no xgboost_*.json" and Discord showed
    "PROPIQMODEL unset, no xgboost*.json" — the two underscores are a markdown
    PAIR, so everything between them was italicised and both were eaten. The
    operator was told to check a variable that does not exist under that name.
    """
    from src.notify.discord import escape_markdown

    raw = "PROPIQ_MODEL unset, no xgboost_*.json with a sidecar"
    out = escape_markdown(raw)

    assert "\\_" in out, "underscores are not escaped, so Discord will eat them"
    assert "\\*" in out
    # Every identifier survives once Discord unescapes it.
    assert out.replace("\\", "") == raw


@pytest.mark.parametrize("char", ["_", "*", "`", "~", "|", ">"])
def test_every_markdown_character_is_escaped(char):
    from src.notify.discord import escape_markdown

    assert escape_markdown(f"a{char}b") == f"a\\{char}b"


def test_a_backslash_is_escaped_first_and_not_twice():
    """
    Order matters: escaping the backslash after the others would double every
    escape they added and print literal backslashes to the operator.
    """
    from src.notify.discord import escape_markdown

    assert escape_markdown("a\\b") == "a\\\\b"
    assert escape_markdown("C:\\path_name") == "C:\\\\path\\_name"


def test_the_alert_escapes_before_truncating(monkeypatch):
    """
    Truncation after escaping, or a cut can land between a backslash and the
    character it escapes and leave a dangling escape on the card.
    """
    import inspect

    import scheduler_worker as worker

    source = inspect.getsource(worker.notify_failure)
    assert "_escaped_alert_body(safe)[:1500]" in source, (
        "the alert truncates before escaping, so a 1500-character cut can "
        "split an escape sequence"
    )


def test_the_redaction_marker_stays_readable_through_escaping():
    r"""
    TWO TRUTHS AT ONCE. Identifiers must survive Discord's markdown, and the
    credential scrubber's marker must still look like a redaction. Escaping
    everything satisfied the first and broke the second: a scrubbed URL came
    out as `propiq:\*\*\*@host`. The credential was redacted either way --
    way -- this is about the marker being legible.
    """
    import scheduler_worker as worker

    body = worker._escaped_alert_body(
        "could not connect to postgresql://propiq:***@db.example.com/propiq"
    )
    assert "propiq:***@db.example.com" in body, body
    assert "\\*" not in body
    # And the identifiers are still escaped in the same string.
    mixed = worker._escaped_alert_body("PROPIQ_MODEL unset; user:***@host")
    assert "PROPIQ\\_MODEL" in mixed
    assert ":***@host" in mixed
