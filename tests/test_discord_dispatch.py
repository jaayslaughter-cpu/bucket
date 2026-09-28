"""
Tests for src/notify/discord.py.

Three properties carry this module: the webhook URL never escapes, the
payload never promises an outcome, and nothing is sent unless someone asked
for it to be.
"""

from __future__ import annotations

import pytest

from src.notify.discord import (
    ENV_WEBHOOK_URL,
    MAX_EMBEDS_PER_MESSAGE,
    MAX_FIELDS_PER_EMBED,
    DiscordConfig,
    DiscordDispatchError,
    assert_payload_safe,
    build_abstention_embed,
    build_decision_board_embed,
    build_parlay_embed,
    load_webhook_url,
    redact_webhook,
    send_embeds,
)
from src.quant.parlay import ParlayLeg, evaluate_parlay
from src.quant.parlay_log import ticket_from_evaluation

VALID_URL = "https://discord.com/api/webhooks/123456789012345678/AbC-dEf_123"

LEGS = [
    ParlayLeg("L1", 0.60, -110, game_id="G1", line=24.5,
              player_name="DEMO_A", market="PTS", side="over"),
    ParlayLeg("L2", 0.55, -115, game_id="G2", line=7.5,
              player_name="DEMO_B", market="AST", side="over"),
]


def _ticket():
    return ticket_from_evaluation(evaluate_parlay(LEGS), LEGS, slate_date="2026-09-21")


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


class _Recorder:
    """A transport that records calls instead of making them."""

    def __init__(self, *responses):
        self.responses = list(responses) or [_FakeResponse(204)]
        self.calls: list[dict] = []

    def __call__(self, url, json=None, timeout=None):
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        return self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]


# --- the credential ------------------------------------------------------


def test_the_webhook_url_is_never_in_an_error_message(monkeypatch):
    monkeypatch.delenv(ENV_WEBHOOK_URL, raising=False)
    with pytest.raises(DiscordDispatchError) as unset:
        load_webhook_url()
    assert ENV_WEBHOOK_URL in str(unset.value)

    with pytest.raises(DiscordDispatchError) as wrong:
        load_webhook_url("https://evil.example/hook")
    assert "evil.example" not in str(wrong.value)   # the value is not echoed


def test_redaction_masks_the_token_but_keeps_the_id():
    redacted = redact_webhook(f"POST to {VALID_URL} failed")
    assert "AbC-dEf_123" not in redacted
    assert "123456789012345678" in redacted
    assert "REDACTED" in redacted


def test_a_valid_url_loads_from_the_environment(monkeypatch):
    monkeypatch.setenv(ENV_WEBHOOK_URL, VALID_URL)
    assert load_webhook_url() == VALID_URL


def test_a_payload_carrying_a_credential_is_refused():
    with pytest.raises(DiscordDispatchError, match="Refusing"):
        assert_payload_safe({"description": f"hook is {VALID_URL}"})
    with pytest.raises(DiscordDispatchError, match="Refusing"):
        assert_payload_safe({"description": "postgresql://user:pw@host/db"})
    with pytest.raises(DiscordDispatchError, match="Refusing"):
        assert_payload_safe({"description": "PROPLINE_API_KEY=abc"})
    assert_payload_safe({"description": "EV +3.21% at -110"})    # clean passes


def test_send_refuses_rather_than_posting_a_secret():
    result = send_embeds(
        [{"title": "x", "description": f"hook {VALID_URL}"}],
        config=DiscordConfig(dry_run=False),
        transport=_Recorder(),
    )
    assert result.status == "REFUSED"
    assert result.embeds_sent == 0


# --- what may be said ----------------------------------------------------


def test_claim_words_are_refused_in_any_embed():
    for phrase in ("this is a lock", "best bet of the night", "guaranteed winner"):
        with pytest.raises(DiscordDispatchError):
            build_abstention_embed(phrase)


def test_the_research_footer_travels_with_every_embed():
    ticket, legs = _ticket()
    for embed in (
        build_parlay_embed(ticket, legs),
        build_decision_board_embed([], slate_date="2026-09-21"),
        build_abstention_embed("No two-way price reached the gate"),
    ):
        assert "RESEARCH_ONLY" in embed["footer"]["text"]
        assert "never places or sizes" in embed["footer"]["text"]


def test_a_stake_is_labelled_as_the_users_own():
    ticket, legs = _ticket()
    embed = build_parlay_embed(ticket, legs)
    stake = next(f for f in embed["fields"] if f["name"] == "Stake")
    assert "**your** figure" in stake["value"]
    assert "does not size bets" in stake["value"]


def test_an_unpriced_board_says_so_rather_than_looking_empty():
    class Row:
        decision_status = "CONSIDER"
        decision_basis = "model_lean"
        player_name = "DEMO_A"
        target_market = "PTS"
        side = "under"
        line = 24.5
        model_prob = 0.71
        book_ev = None
        american_odds = None
        book_source = None

    embed = build_decision_board_embed([Row(), Row()], slate_date="2026-09-21")
    assert "none priced" in embed["description"]
    assert all("model_lean" in f["value"] for f in embed["fields"])
    assert all("No priced market" in f["value"] for f in embed["fields"])


# --- sending -------------------------------------------------------------


def test_dry_run_is_the_default_and_sends_nothing():
    ticket, legs = _ticket()
    recorder = _Recorder()
    result = send_embeds([build_parlay_embed(ticket, legs)], transport=recorder)

    assert result.status == "DRY_RUN"
    assert result.dry_run is True
    assert result.embeds_sent == 0
    assert recorder.calls == []                  # nothing left the machine
    assert result.payload_preview["embeds"]


def test_a_successful_post_reports_204():
    ticket, legs = _ticket()
    recorder = _Recorder(_FakeResponse(204))
    result = send_embeds(
        [build_parlay_embed(ticket, legs)],
        config=DiscordConfig(dry_run=False),
        webhook_url=VALID_URL,
        transport=recorder,
    )
    assert result.status == "OK"
    assert result.http_status == 204
    assert result.embeds_sent == 1
    assert len(recorder.calls) == 1
    assert recorder.calls[0]["json"]["embeds"]


def test_rate_limit_is_retried_using_discords_own_retry_after():
    """Guessing a backoff against a rate limiter is how a webhook gets disabled."""
    recorder = _Recorder(
        _FakeResponse(429, {"retry_after": 0.01}), _FakeResponse(204),
    )
    result = send_embeds(
        [build_abstention_embed("nothing today")],
        config=DiscordConfig(dry_run=False, backoff_seconds=0.01),
        webhook_url=VALID_URL,
        transport=recorder,
    )
    assert result.status == "OK"
    assert len(recorder.calls) == 2


def test_a_client_error_is_not_retried():
    recorder = _Recorder(_FakeResponse(400, text="Invalid Form Body"))
    result = send_embeds(
        [build_abstention_embed("nothing today")],
        config=DiscordConfig(dry_run=False, max_retries=3, backoff_seconds=0.01),
        webhook_url=VALID_URL,
        transport=recorder,
    )
    assert result.status == "FAILED"
    assert result.http_status == 400
    assert len(recorder.calls) == 1          # a 400 will not improve on a retry


def test_a_transport_error_is_redacted_before_it_is_reported():
    def explode(url, json=None, timeout=None):
        raise ConnectionError(f"failed to reach {url}")

    result = send_embeds(
        [build_abstention_embed("nothing today")],
        config=DiscordConfig(dry_run=False, max_retries=1, backoff_seconds=0.01),
        webhook_url=VALID_URL,
        transport=explode,
    )
    assert result.status == "FAILED"
    assert "AbC-dEf_123" not in (result.reason or "")
    assert "REDACTED" in (result.reason or "")


def test_discord_limits_are_enforced_before_sending():
    assert send_embeds(
        [{"title": "x"}] * (MAX_EMBEDS_PER_MESSAGE + 1),
        config=DiscordConfig(dry_run=False), transport=_Recorder(),
    ).status == "DATA_NOT_AVAILABLE"

    assert send_embeds([], transport=_Recorder()).status == "DATA_NOT_AVAILABLE"

    crowded = build_decision_board_embed(
        [type("R", (), {
            "decision_status": "CONSIDER", "decision_basis": "book_ev",
            "player_name": f"P{i}", "target_market": "PTS", "side": "over",
            "line": 24.5, "model_prob": 0.6, "book_ev": 0.03,
            "american_odds": -110, "book_source": "propline",
        })() for i in range(60)],
        max_rows=60,
    )
    assert len(crowded["fields"]) <= MAX_FIELDS_PER_EMBED
    assert crowded["fields"][-1]["name"] == "…"
