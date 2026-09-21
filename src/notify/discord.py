"""
src/notify/discord.py — Discord webhook dispatch for research output.

Status: RESEARCH_ONLY · MANUAL_ONLY. This delivers a message to a person.
It is not a betting layer: nothing here places a wager, confirms one, or
tells you how much to risk. Discord is the last step of a research pipeline,
not the first step of an automated one.

THE WEBHOOK URL IS A CREDENTIAL. A Discord webhook URL ends in a token, and
anyone holding it can post to that channel as you. So it is read from the
environment, never committed, never written into a payload, and never echoed
in an error message or a log line — ``redact_webhook`` is applied on every
path that could surface it. ``artifact_registry`` already refuses to persist
a key named "webhook"; this module is the other half of that rule.

WHAT THE EMBED MAY SAY. The same vocabulary guard the decision board uses
applies here: no "lock", no "best bet", no "guaranteed". A notification is
the most quotable artifact the system produces — it is the thing that gets
screenshotted — so it carries the research disclaimer and the abstention
reasons rather than only the rows that cleared.

NO STAKE IS EVER SUGGESTED. The embed shows a stake only when you logged one
yourself, labelled as yours. PropIQ does not size bets, and an embed that
implied otherwise would make it look as though it does.

DISCORD'S OWN LIMITS are enforced before sending, because exceeding them
returns a 400 that reads like a bug in your data: 25 fields per embed, 256
characters of title, 1024 of field value, 4096 of description, 6000 total
per embed, 10 embeds per message.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

logger = logging.getLogger(__name__)

ENV_WEBHOOK_URL = "DISCORD_WEBHOOK_URL"
PLACEMENT_MODE = "MANUAL_ONLY"
RESEARCH_STATUS = "RESEARCH_ONLY"

RESEARCH_FOOTER = (
    "RESEARCH_ONLY · MANUAL_ONLY — research ranking, not a bet instruction. "
    "PropIQ never places or sizes wagers."
)

# Discord's documented limits. Exceeding one returns a 400 that reads like a
# data bug, so the payload is trimmed to fit before it is ever sent.
MAX_EMBEDS_PER_MESSAGE = 10
MAX_FIELDS_PER_EMBED = 25
MAX_TITLE = 256
MAX_DESCRIPTION = 4096
MAX_FIELD_NAME = 256
MAX_FIELD_VALUE = 1024
MAX_FOOTER = 2048
MAX_EMBED_TOTAL = 6000

COLOR_CONSIDER = 0x2E8B57   # sea green
COLOR_ABSTAIN = 0x708090    # slate grey
COLOR_REFUSED = 0xB22222    # firebrick

_WEBHOOK_PATTERN = re.compile(
    r"^https://(?:\w+\.)?discord(?:app)?\.com/api/webhooks/\d+/[\w-]+$"
)

# Mirrors the decision board's guard: a notification never promises an outcome.
FORBIDDEN_CLAIM_WORDS: frozenset[str] = frozenset({
    "lock", "locks", "guaranteed", "guarantee", "best bet", "bestbet",
    "sure thing", "can't lose", "cant lose", "free money", "max bet",
})

_SECRET_PATTERNS = (
    re.compile(r"discord(?:app)?\.com/api/webhooks/", re.I),
    re.compile(r"api[_-]?key|secret|password|bearer|token", re.I),
    re.compile(r"postgres(ql)?://|mysql://|mongodb(\+srv)?://", re.I),
)


class DiscordDispatchError(RuntimeError):
    """Raised when a notification cannot be built or sent."""


@dataclass(frozen=True)
class DiscordConfig:
    """Dispatch settings. The URL is never stored on the object."""

    timeout_seconds: float = 10.0
    max_retries: int = 3
    backoff_seconds: float = 2.0
    # Default ON. The first outbound channel in a research system should not
    # be able to fire by accident; sending is an explicit choice.
    dry_run: bool = True
    username: str | None = "PropIQ Research"


@dataclass
class DispatchResult:
    """What happened, with nothing sensitive in it."""

    status: str = "DATA_NOT_AVAILABLE"
    reason: str | None = None
    http_status: int | None = None
    dry_run: bool = True
    embeds_sent: int = 0
    payload_preview: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "http_status": self.http_status,
            "dry_run": self.dry_run,
            "embeds_sent": self.embeds_sent,
            "placement_mode": PLACEMENT_MODE,
            "research_status": RESEARCH_STATUS,
            "payload_preview": self.payload_preview,
        }


# ---------------------------------------------------------------------------
# the credential
# ---------------------------------------------------------------------------


def redact_webhook(text: Any) -> str:
    """
    Replace any webhook URL with its id and a masked token.

    Applied to every error and log line. A failed request that echoes its URL
    has published the credential into wherever logs go.
    """
    return re.sub(
        r"(https://(?:\w+\.)?discord(?:app)?\.com/api/webhooks/)(\d+)/[\w-]+",
        r"\1\2/***REDACTED***",
        str(text),
    )


def load_webhook_url(explicit: str | None = None) -> str:
    """
    Read the webhook URL from the environment. Never hardcoded, never logged.

    Raises with the variable NAME and never its value, so a misconfiguration
    message can be pasted anywhere safely.
    """
    url = (explicit or os.environ.get(ENV_WEBHOOK_URL) or "").strip()
    if not url:
        raise DiscordDispatchError(
            f"{ENV_WEBHOOK_URL} is not set. Copy .env.example to .env and paste "
            "your channel's webhook URL there. It is a credential: never commit "
            "it, and rotate it if it has ever been pasted into a chat or a "
            "screenshot."
        )
    if not _WEBHOOK_PATTERN.match(url):
        raise DiscordDispatchError(
            f"{ENV_WEBHOOK_URL} does not look like a Discord webhook URL "
            "(expected https://discord.com/api/webhooks/<id>/<token>). The value "
            "is not shown here on purpose."
        )
    return url


def assert_payload_safe(payload: Any, *, path: str = "payload") -> None:
    """Refuse to send a payload carrying a credential or a connection string."""
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            assert_payload_safe(value, path=f"{path}.{key}")
        return
    if isinstance(payload, (list, tuple)):
        for i, item in enumerate(payload):
            assert_payload_safe(item, path=f"{path}[{i}]")
        return
    if isinstance(payload, str):
        for pattern in _SECRET_PATTERNS:
            if pattern.search(payload):
                raise DiscordDispatchError(
                    f"{path}: payload matches {pattern.pattern!r}. Refusing to "
                    "post a credential or connection string into a channel."
                )


def _assert_no_claims(text: str) -> str:
    lowered = str(text).lower()
    hit = next((w for w in FORBIDDEN_CLAIM_WORDS if w in lowered), None)
    if hit:
        raise DiscordDispatchError(
            f"Refusing to post {text!r}: {hit!r} states an outcome this system "
            "cannot support."
        )
    return text


# ---------------------------------------------------------------------------
# embed building
# ---------------------------------------------------------------------------


def _clip(text: Any, limit: int) -> str:
    value = _assert_no_claims(str(text))
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _pct(value: Any) -> str:
    return "—" if value is None else f"{float(value) * 100:+.2f}%"


def _prob(value: Any) -> str:
    return "—" if value is None else f"{float(value):.1%}"


def _odds(value: Any) -> str:
    return "—" if value is None else f"{int(value):+d}"


def _fit_embed(embed: dict[str, Any]) -> dict[str, Any]:
    """Trim an embed to Discord's limits rather than letting it 400."""
    if "title" in embed:
        embed["title"] = embed["title"][:MAX_TITLE]
    if "description" in embed:
        embed["description"] = embed["description"][:MAX_DESCRIPTION]
    fields = embed.get("fields") or []
    trimmed: list[dict[str, Any]] = []
    total = len(embed.get("title", "")) + len(embed.get("description", ""))
    # Reserve a slot for the "… n more" marker when trimming, or the marker
    # itself pushes the embed to 26 fields — the 400 this function prevents.
    budget = (
        MAX_FIELDS_PER_EMBED if len(fields) <= MAX_FIELDS_PER_EMBED
        else MAX_FIELDS_PER_EMBED - 1
    )
    for entry in fields[:budget]:
        name = str(entry.get("name", ""))[:MAX_FIELD_NAME]
        value = str(entry.get("value", ""))[:MAX_FIELD_VALUE]
        if total + len(name) + len(value) > MAX_EMBED_TOTAL - MAX_FOOTER:
            break
        total += len(name) + len(value)
        trimmed.append({"name": name, "value": value, "inline": bool(entry.get("inline"))})
    if len(trimmed) < len(fields):
        trimmed.append({
            "name": "…",
            "value": f"{len(fields) - len(trimmed)} more rows omitted to fit Discord's limits",
            "inline": False,
        })
    embed["fields"] = trimmed
    return embed


def build_decision_board_embed(
    candidates: Sequence[Any],
    *,
    slate_date: str | None = None,
    max_rows: int = 10,
) -> dict[str, Any]:
    """
    One embed for a decision board.

    Shows the basis of every row, not just its number, so a `model_lean` is
    never mistaken for a priced edge in a channel where the column headers
    are gone.
    """
    considered = [c for c in candidates if getattr(c, "decision_status", "") == "CONSIDER"]
    priced = [c for c in considered if getattr(c, "decision_basis", "") == "book_ev"]

    lines: list[str] = []
    if not considered:
        lines.append("No row met the threshold. Nothing to look at.")
    elif not priced:
        lines.append(
            f"{len(considered)} rows lean, **none priced** — no two-way odds "
            "reached the gate, so no EV was computed for any of them."
        )
    else:
        lines.append(
            f"{len(priced)} priced · {len(considered) - len(priced)} unpriced leans "
            f"· {len(candidates) - len(considered)} abstained"
        )

    fields: list[dict[str, Any]] = []
    for c in considered[:max_rows]:
        basis = getattr(c, "decision_basis", "unavailable")
        name = (
            f"{getattr(c, 'player_name', None) or getattr(c, 'player_id', '?')} — "
            f"{getattr(c, 'target_market', '?')} {getattr(c, 'side', '?').upper()} "
            f"{getattr(c, 'line', '—')}"
        )
        if basis == "book_ev":
            value = (
                f"EV **{_pct(getattr(c, 'book_ev', None))}** at "
                f"{_odds(getattr(c, 'american_odds', None))} · "
                f"model {_prob(getattr(c, 'model_prob', None))} · "
                f"{getattr(c, 'book_source', None) or 'book'}"
            )
        else:
            value = (
                f"`{basis}` — model {_prob(getattr(c, 'model_prob', None))}. "
                "No priced market, so no EV."
            )
        fields.append({"name": name, "value": value, "inline": False})

    embed = {
        "title": f"Decision board — {slate_date or 'slate'}",
        "description": "\n".join(lines),
        "color": COLOR_CONSIDER if priced else COLOR_ABSTAIN,
        "fields": fields,
        "footer": {"text": RESEARCH_FOOTER[:MAX_FOOTER]},
    }
    return _fit_embed(embed)


def build_parlay_embed(ticket: Any, legs: Sequence[Any]) -> dict[str, Any]:
    """
    One embed for a logged parlay ticket.

    A stake appears only when the user recorded one, labelled as theirs. The
    model does not size bets and the embed must not read as though it does.
    """
    price = getattr(ticket, "ticket_american_price", None)
    joint = getattr(ticket, "joint_probability", None)
    stderr = getattr(ticket, "joint_probability_stderr", None)
    breakeven = getattr(ticket, "breakeven_probability", None)
    ev = getattr(ticket, "ev_at_bet_time", None)
    independent = getattr(ticket, "independent_probability", None)

    description = [
        f"**{len(legs)} legs** at **{_odds(price)}**",
        f"Model {_prob(joint)}"
        + (f" ± {float(stderr):.2%}" if stderr is not None else "")
        + f" · breakeven {_prob(breakeven)} · EV {_pct(ev)}",
    ]
    if joint is not None and independent is not None:
        delta = float(joint) - float(independent)
        description.append(
            f"Correlation moved the joint probability {delta:+.2%} against the "
            "naive product of the legs."
        )

    fields = []
    for leg in legs:
        fields.append({
            "name": _clip(
                f"{getattr(leg, 'player_name', None) or getattr(leg, 'leg_id', '?')} — "
                f"{getattr(leg, 'market', '?')} "
                f"{str(getattr(leg, 'side', '') or '').upper()} "
                f"{getattr(leg, 'line', '—')}",
                MAX_FIELD_NAME,
            ),
            "value": _clip(
                f"{_odds(getattr(leg, 'taken_odds_american', None))} · "
                f"model {_prob(getattr(leg, 'model_prob', None))} · "
                f"edge {_pct(getattr(leg, 'edge_vs_devig', None))} · "
                f"{getattr(leg, 'book_source', None) or 'book'}",
                MAX_FIELD_VALUE,
            ),
            "inline": False,
        })

    stake = getattr(ticket, "unit_stake", None)
    if stake is not None:
        fields.append({
            "name": "Stake",
            "value": (
                f"{float(stake):g}u — **your** figure, recorded as logged. "
                "PropIQ does not size bets."
            ),
            "inline": False,
        })

    embed = {
        "title": _clip(
            f"Parlay ticket {getattr(ticket, 'ticket_id', '')[:8]} — "
            f"{getattr(ticket, 'slate_date', None) or 'slate'}",
            MAX_TITLE,
        ),
        "description": "\n".join(description),
        "color": COLOR_CONSIDER,
        "fields": fields,
        "footer": {"text": RESEARCH_FOOTER[:MAX_FOOTER]},
    }
    return _fit_embed(embed)


def build_abstention_embed(reason: str, *, title: str = "No ticket") -> dict[str, Any]:
    """
    Post the refusal too.

    Silence reads as "nothing good today". A named reason distinguishes that
    from "the pipeline could not run", which is the difference between
    trusting the system and guessing at it.
    """
    return _fit_embed({
        "title": _clip(title, MAX_TITLE),
        "description": _clip(reason, MAX_DESCRIPTION),
        "color": COLOR_REFUSED,
        "fields": [],
        "footer": {"text": RESEARCH_FOOTER[:MAX_FOOTER]},
    })


# ---------------------------------------------------------------------------
# sending
# ---------------------------------------------------------------------------


def send_embeds(
    embeds: Sequence[Mapping[str, Any]],
    *,
    config: DiscordConfig | None = None,
    webhook_url: str | None = None,
    content: str | None = None,
    transport: Callable[..., Any] | None = None,
) -> DispatchResult:
    """
    POST embeds to the configured webhook, or preview them when dry_run.

    ``dry_run`` defaults to True: the payload comes back for inspection and
    nothing leaves the machine. ``transport`` is injectable so the dispatch
    path can be tested without a network.

    A 429 is honoured using Discord's own ``retry_after``; guessing a backoff
    against a rate limiter is how a webhook gets disabled.
    """
    config = config or DiscordConfig()
    if not embeds:
        return DispatchResult(
            status="DATA_NOT_AVAILABLE", reason="No embeds to send",
            dry_run=config.dry_run,
        )
    if len(embeds) > MAX_EMBEDS_PER_MESSAGE:
        return DispatchResult(
            status="DATA_NOT_AVAILABLE",
            reason=(
                f"{len(embeds)} embeds exceeds Discord's limit of "
                f"{MAX_EMBEDS_PER_MESSAGE} per message; split the batch"
            ),
            dry_run=config.dry_run,
        )

    payload: dict[str, Any] = {"embeds": [dict(e) for e in embeds]}
    if config.username:
        payload["username"] = config.username
    if content:
        payload["content"] = _clip(content, 2000)

    try:
        assert_payload_safe(payload)
    except DiscordDispatchError as exc:
        return DispatchResult(
            status="REFUSED", reason=str(exc), dry_run=config.dry_run,
        )

    if config.dry_run:
        return DispatchResult(
            status="DRY_RUN",
            reason="dry_run=True — nothing was sent. Pass dry_run=False to post.",
            dry_run=True,
            embeds_sent=0,
            payload_preview=payload,
        )

    try:
        url = load_webhook_url(webhook_url)
    except DiscordDispatchError as exc:
        return DispatchResult(status="DATA_NOT_AVAILABLE", reason=str(exc), dry_run=False)

    post = transport
    if post is None:
        import requests

        post = requests.post

    last_reason: str | None = None
    for attempt in range(1, int(config.max_retries) + 1):
        try:
            response = post(url, json=payload, timeout=config.timeout_seconds)
        except Exception as exc:  # noqa: BLE001 — redacted, then retried
            last_reason = redact_webhook(f"{type(exc).__name__}: {exc}")
            logger.warning(
                "discord attempt %d/%d failed: %s", attempt, config.max_retries, last_reason,
            )
            if attempt < config.max_retries:
                time.sleep(config.backoff_seconds ** attempt)
            continue

        status = int(getattr(response, "status_code", 0))
        if status in (200, 204):
            logger.info("discord: delivered %d embed(s)", len(embeds))
            return DispatchResult(
                status="OK", http_status=status, dry_run=False,
                embeds_sent=len(embeds),
            )
        if status == 429:
            # Discord tells you how long to wait. Guessing gets you disabled.
            wait = config.backoff_seconds
            try:
                wait = float(response.json().get("retry_after", wait))
            except Exception:  # noqa: BLE001 — fall back to the configured backoff
                pass
            last_reason = f"rate limited (429), retry_after={wait}s"
            logger.warning("discord: %s", last_reason)
            if attempt < config.max_retries:
                time.sleep(wait)
            continue

        body = redact_webhook(getattr(response, "text", ""))[:300]
        last_reason = f"HTTP {status}: {body}"
        # 4xx other than 429 will not improve on a retry.
        if 400 <= status < 500:
            return DispatchResult(
                status="FAILED", reason=last_reason, http_status=status, dry_run=False,
            )
        if attempt < config.max_retries:
            time.sleep(config.backoff_seconds ** attempt)

    return DispatchResult(
        status="FAILED",
        reason=redact_webhook(last_reason or "exhausted retries"),
        dry_run=False,
    )


def preview_json(result: DispatchResult) -> str:
    """Pretty-print a dry-run payload for eyeballing before you send."""
    return json.dumps(result.payload_preview, indent=2, ensure_ascii=False)
