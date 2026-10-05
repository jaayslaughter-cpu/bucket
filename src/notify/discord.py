"""
src/notify/discord.py — Discord webhook dispatch for recommendations.

This delivers a message to a person. It does not place a wager or confirm one:
Discord is the last step of this pipeline, not the first step of an automated
one, and there is no order API behind it.

THE WEBHOOK URL IS A CREDENTIAL. A Discord webhook URL ends in a token, and
anyone holding it can post to that channel as you. So it is read from the
environment, never committed, never written into a payload, and never echoed
in an error message or a log line — ``redact_webhook`` is applied on every
path that could surface it. ``artifact_registry`` already refuses to persist
a key named "webhook"; this module is the other half of that rule.

WHAT THE EMBED MAY SAY. It may recommend a side and a size. It may NOT promise
an outcome: the decision board's vocabulary guard applies here unchanged — no
"lock", no "best bet", no "guaranteed". Recommending and promising are different
acts and only the first is supportable. A notification is the most quotable
artifact this system produces — it is the thing that gets screenshotted — so it
carries the basis of each recommendation and the abstention reasons, not only
the rows that cleared.

TWO KINDS OF STAKE, AND THE EMBED KEEPS THEM APART. A RECOMMENDED size comes
from ``quant.advisory_sizing`` (fractional Kelly, capped) and is labelled as the
recommendation. A LOGGED stake is what the reader actually staked, labelled as
theirs. Collapsing the two would make a suggestion look like a record of a bet
that happened.

THE CALIBRATION GATE IS NOT BYPASSED HERE. ``build_dfs_entry_embed`` takes a
``publication`` verdict and posts the gate's reason INSTEAD of the numbers when
it withholds. A model-sourced recommendation with no graded results behind it is
exactly the thing that reads, in a channel, as though it had some.

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
    "Recommendation, not a promise of an outcome — each row names the basis it "
    "rests on. PropIQ does not place the wager."
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
    publication: Any = None,
) -> dict[str, Any]:
    """
    One embed for a board of recommendations.

    Shows the BASIS of every row, not just its number. In a channel the column
    headers are gone, and a recommendation resting on the model alone must not
    read like one resting on a de-vigged price.

    THE PUBLICATION GATE APPLIES HERE TOO, and until this parameter existed it
    did not. Three docstrings in this repository claimed the gate stopped an
    uncalibrated board row from being dispatched; only the DFS path actually
    called it, so the board's rows went out ungated. That is the same
    "documented guard that is not wired" defect this codebase keeps producing.

    EVERY BOARD ROW IS MODEL-SOURCED, including a ``book_ev`` one — and that is
    the subtlety worth stating. A book_ev row's EV compares THE MODEL's
    probability against a de-vigged market price: the price is the benchmark it
    is measured against, not the probability being used. So there is no
    equivalent here of ``dfs_entry``'s SHARP_BENCHMARK case, where the market's
    own probability is the input. The whole board is gated as MODEL.

    ``publication`` withheld -> the gate's reason replaces the rows. None ->
    published unguarded, which is only right where the caller already gated.
    """
    if publication is not None and not getattr(publication, "allowed", False):
        return build_abstention_embed(
            "Board withheld from publication: "
            f"{getattr(publication, 'reason', None) or 'no reason given'}",
            title=f"Recommendations — {slate_date or 'slate'}",
        )

    considered = [
        c for c in candidates if getattr(c, "decision_status", "") == "RECOMMENDED"
    ]
    priced = [c for c in considered if getattr(c, "decision_basis", "") == "book_ev"]

    lines: list[str] = []
    if not considered:
        lines.append("Nothing recommended on this slate.")
    elif not priced:
        lines.append(
            f"**{len(considered)} recommended, none priced** — no two-way odds "
            "reached the gate, so every one of these rests on the model alone "
            "and none has a market price behind it."
        )
    else:
        lines.append(
            f"**{len(priced)} recommended on a price** · "
            f"{len(considered) - len(priced)} on the model alone "
            f"· {len(candidates) - len(considered)} abstained"
        )

    # THE DATES THE ROWS ACTUALLY DESCRIBE. A board is built from a scored
    # VALIDATION WINDOW, so its rows can be about any day that window covers --
    # and the title carries slate_date, which reads as "tonight". Until this
    # check existed a card titled with today's date could be entirely February
    # backtest rows with nothing saying so. This is the loudest place that can
    # be said, so it is said first, before any row.
    off_slate = sorted({
        str(seen) for seen in (
            getattr(c, "game_date", None) for c in considered
        ) if seen and str(seen) != str(slate_date)
    })
    undated = sum(1 for c in considered if not getattr(c, "game_date", None))
    if off_slate:
        lines.insert(0, (
            f"⚠️ **NOT TONIGHT'S SLATE.** {len(off_slate)} other game date(s) "
            f"appear below ({', '.join(off_slate[:4])}"
            f"{', …' if len(off_slate) > 4 else ''}). These are BACKTEST rows "
            f"scored from a past validation window, not projections for "
            f"{slate_date or 'the slate date'}."
        ))
    if undated:
        lines.insert(0, (
            f"⚠️ **{undated} row(s) carry no game date**, so which game they "
            f"describe cannot be shown. The heading date is when this board "
            f"was built, not when the games are played."
        ))

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
                "No priced market, so no EV and no market check on this one."
            )
        # Per row, and only when it is not the slate date: repeating today's
        # date on every row would be noise, while an off-slate date on one row
        # is the thing a reader has to see.
        row_date = getattr(c, "game_date", None)
        if row_date and str(row_date) != str(slate_date):
            value += f"\n⚠️ game date **{row_date}**, not {slate_date}."
        elif not row_date:
            value += "\n⚠️ no game date on this row."
        fields.append({"name": name, "value": value, "inline": False})

    embed = {
        "title": f"Recommendations — {slate_date or 'slate'}",
        "description": "\n".join(lines),
        "color": COLOR_CONSIDER if priced else COLOR_ABSTAIN,
        "fields": fields,
        "footer": {"text": RESEARCH_FOOTER[:MAX_FOOTER]},
    }
    return _fit_embed(embed)


def build_parlay_embed(ticket: Any, legs: Sequence[Any]) -> dict[str, Any]:
    """
    One embed for a LOGGED parlay ticket — a record, not a recommendation.

    The stake shown here is the one the reader actually staked, labelled as
    theirs. A recommended size is a different number from a different place
    (``quant.advisory_sizing``), and this embed must not let the two be read as
    one: a suggestion rendered like a record implies a bet that happened.
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
            "name": "Stake logged",
            "value": (
                f"{float(stake):g}u — **your** figure, as recorded in the ledger. "
                "This is what was staked, not a recommended size."
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


def build_dfs_entry_embed(
    evaluation: Any,
    *,
    publication: Any = None,
    slate_date: str | None = None,
    advisory_size: Any = None,
) -> dict[str, Any]:
    """
    One embed for a DFS pick'em entry priced by ``quant.dfs_entry``.

    THE PUBLICATION GATE DECIDES WHAT THIS SHOWS, which is the point of taking
    it as an argument rather than leaving it to the caller's discipline. When
    ``publication`` is withheld, the numbers are replaced by the gate's reason:
    a model-sourced EV whose model has not been shown calibrated is exactly the
    figure that reads, in a channel, as though it had been. Pass the verdict
    from ``quant.publication_gate.calibration_gate``; passing None publishes the
    numbers unguarded and is only right where the caller has already gated.

    Where a probability came from is stated on the face of the embed. A
    market-grounded entry and a model-grounded one are different claims and look
    identical once the column headers are gone.
    """
    entry = getattr(evaluation, "payout", None)
    legs = list(getattr(evaluation, "legs", []) or [])
    status = str(getattr(evaluation, "status", "") or "")
    source = getattr(evaluation, "probability_source", None)
    source_label = getattr(source, "value", None) or "UNSPECIFIED"
    structure = getattr(evaluation, "structure_label", None) or "entry"
    title = f"Recommended DFS entry — {structure}" + (
        f" · {slate_date}" if slate_date else ""
    )

    if entry is None or status != "PAYOUT_EV_READY":
        return build_abstention_embed(
            f"Not priced: {getattr(evaluation, 'reason', None) or 'no reason given'}",
            title=title,
        )

    if publication is not None and not getattr(publication, "allowed", False):
        return build_abstention_embed(
            "Priced but withheld from publication: "
            f"{getattr(publication, 'reason', None) or 'no reason given'}",
            title=title,
        )

    ev = getattr(entry, "expected_value", None)
    p_all = getattr(entry, "probability_all_hit", None)
    breakeven = getattr(entry, "breakeven_joint_probability", None)

    description = [
        f"**{len(legs)} legs** · model {_prob(p_all)}"
        + (f" · breakeven {_prob(breakeven)}" if breakeven is not None else "")
        + f" · EV {_pct(ev)}",
        f"Probabilities: **{source_label}**",
    ]
    disclaimer = getattr(entry, "disclaimer", None)
    if disclaimer:
        description.append(str(disclaimer))

    fields: list[dict[str, Any]] = []
    for leg in legs:
        fields.append({
            "name": _clip(
                f"{getattr(leg, 'leg_id', '?')} "
                f"{str(getattr(leg, 'side', '') or '').upper()} "
                f"{getattr(leg, 'line', '—')}",
                MAX_FIELD_NAME,
            ),
            "value": _clip(
                f"model {_prob(getattr(leg, 'probability', None))} · "
                f"{getattr(getattr(leg, 'probability_source', None), 'value', '?')}"
                + (f" · {benchmark}" if (benchmark := getattr(
                    leg, "benchmark_source", None)) else ""),
                MAX_FIELD_VALUE,
            ),
            "inline": False,
        })

    if advisory_size is not None:
        units = getattr(advisory_size, "recommended_units", None)
        fields.append({
            "name": "Recommended stake",
            "value": _clip(
                f"**{float(units):g}u** — percent of bankroll, fractional Kelly "
                f"({getattr(advisory_size, 'kelly_fraction_applied', '?')} of full) "
                "and capped. Kelly is optimal only if the probabilities are right, "
                "so on a model-sourced entry this size inherits the model's "
                "calibration error. PropIQ does not place it.",
                MAX_FIELD_VALUE,
            ) if units is not None else "—",
            "inline": False,
        })

    return _fit_embed({
        "title": _clip(title, MAX_TITLE),
        "description": "\n".join(description),
        "color": COLOR_CONSIDER,
        "fields": fields,
        "footer": {"text": RESEARCH_FOOTER[:MAX_FOOTER]},
    })


def build_win_loss_embed(
    summary: Any,
    *,
    slate_date: str | None = None,
    min_sample_for_rate: int = 30,
) -> dict[str, Any]:
    """
    The day's settled record. History, not a forecast.

    This is the one card in this module that reports what ALREADY HAPPENED, and
    that is exactly why it needs more care than the others rather than less. A
    results card is the easiest place in a research system to start implying a
    profit claim, so three things are refused here on purpose:

    1. **A strike rate below ``min_sample_for_rate`` is not shown as a rate.**
       ``settlement/metrics.MIN_SAMPLE_FOR_RATE`` is 30 and exists because a
       rate over a handful of props is noise. The count is still reported; the
       percentage is withheld with the reason.
    2. **ROI is reported only when a stake was recorded.** Nothing in this
       pipeline writes a stake — ``settlement/recorder.py`` deliberately never
       writes ``stake_units`` — so ROI is normally undefined, and the metrics
       layer's own note says which case applies. Printing "ROI 0.00%" over zero
       staked units would read as a flat month rather than as no data.
    3. **CLV is never presented as profit.** The metrics layer ships that
       sentence itself and it is passed through rather than paraphrased.

    ``summary`` is a ``settlement.metrics.PerformanceSummary`` — read by
    attribute so this module does not import the settlement layer, matching how
    the board embed reads its rows.
    """
    record = getattr(summary, "record", None)
    roi = getattr(summary, "roi", None)
    clv = getattr(summary, "clv", None)
    warnings = list(getattr(summary, "warnings", None) or [])

    graded = int(getattr(record, "graded_n", 0) or 0)
    decided = int(getattr(record, "decided_n", 0) or 0)
    pending = int(getattr(record, "pending", 0) or 0)

    fields: list[dict[str, Any]] = []

    if record is not None:
        wins = int(getattr(record, "wins", 0) or 0)
        losses = int(getattr(record, "losses", 0) or 0)
        pushes = int(getattr(record, "pushes", 0) or 0)
        voids = int(getattr(record, "voids", 0) or 0)
        line = f"**{wins}-{losses}-{pushes}**"
        if voids:
            line += f"  ({voids} void)"
        fields.append({
            "name": "Record",
            "value": _clip(f"{line}\n{graded} graded, {pending} still pending",
                           MAX_FIELD_VALUE),
            "inline": True,
        })

        rate = getattr(record, "strike_rate_pct", None)
        if rate is None:
            rate_text = "—"
        elif decided < int(min_sample_for_rate):
            rate_text = (
                f"withheld\n{decided} decided prop(s) is under the "
                f"{int(min_sample_for_rate)} this project treats as the "
                "minimum for a rate rather than noise"
            )
        else:
            rate_text = f"**{float(rate):.1f}%** over {decided} decided"
        fields.append({
            "name": "Strike rate",
            "value": _clip(rate_text, MAX_FIELD_VALUE),
            "inline": True,
        })

    if roi is not None:
        staked = getattr(roi, "staked_units", None)
        roi_pct = getattr(roi, "roi_pct", None)
        note = str(getattr(roi, "note", "") or "")
        if staked and roi_pct is not None:
            roi_text = (
                f"{float(roi_pct):+.2f}% on {float(staked):.2f} unit(s) staked"
            )
        else:
            roi_text = (
                "not computable — no stake is recorded by this pipeline, and "
                "none can be"
            )
        if note:
            roi_text += f"\n{note}"
        fields.append({
            "name": "ROI",
            "value": _clip(roi_text, MAX_FIELD_VALUE),
            "inline": False,
        })

    if clv is not None:
        n_line = int(getattr(clv, "n_with_line_clv", 0) or 0)
        avg_line = getattr(clv, "avg_clv_line_points", None)
        avg_prob = getattr(clv, "avg_clv_prob_points", None)
        if n_line:
            clv_text = (
                f"line {avg_line:+.3f} pts, probability {_pct(avg_prob)} "
                f"over {n_line} prop(s)"
                if avg_line is not None
                else f"{n_line} prop(s) with a closing line"
            )
        else:
            clv_text = "no closing lines captured, so no CLV"
        clv_note = str(getattr(clv, "note", "") or "")
        if clv_note:
            clv_text += f"\n{clv_note}"
        fields.append({
            "name": "CLV",
            "value": _clip(clv_text, MAX_FIELD_VALUE),
            "inline": False,
        })

    for warning in warnings[:3]:
        fields.append({
            "name": "Note",
            "value": _clip(warning, MAX_FIELD_VALUE),
            "inline": False,
        })

    if graded == 0:
        description = (
            "Nothing was graded for this period. That is the pipeline reporting "
            "its own state — no prop reached a final box score — and not a day "
            "with no value in it."
        )
    else:
        description = (
            "Settled results for props this pipeline recorded as predictions. "
            "These are graded predictions, not wagers: no stake was placed by "
            "this system and none is recorded."
        )

    title = "Results" if slate_date is None else f"Results — {slate_date}"
    return _fit_embed({
        "title": _clip(title, MAX_TITLE),
        "description": _clip(description, MAX_DESCRIPTION),
        "color": COLOR_ABSTAIN if graded == 0 else COLOR_CONSIDER,
        "fields": fields,
        "footer": {"text": RESEARCH_FOOTER[:MAX_FOOTER]},
    })


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
