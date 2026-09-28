"""Outbound notification channels.

Notifications only. Nothing in this package places, sizes or confirms a
wager — it delivers research output to a person, who decides.
"""

from src.notify.discord import (
    DiscordConfig,
    DiscordDispatchError,
    DispatchResult,
    build_decision_board_embed,
    build_parlay_embed,
    load_webhook_url,
    redact_webhook,
    send_embeds,
)

__all__ = [
    "DiscordConfig",
    "DiscordDispatchError",
    "DispatchResult",
    "build_decision_board_embed",
    "build_parlay_embed",
    "load_webhook_url",
    "redact_webhook",
    "send_embeds",
]
