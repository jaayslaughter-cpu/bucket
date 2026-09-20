"""Multi-model confidence arbitration (Wave 4).

BookieX-inspired: agreement across models + edge magnitude → HIGH / MODERATE /
LOW / DISAGREE / ABSTAIN. RESEARCH_ONLY — not a stake or lock signal.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Literal

import numpy as np

ConfidenceTier = Literal["HIGH", "MODERATE", "LOW", "DISAGREE", "ABSTAIN"]

DISCLAIMER = (
    "RESEARCH_ONLY confidence tier from model agreement — "
    "not a bet recommendation, stake size, or guaranteed outcome."
)


def _side_from_p(p: float) -> str:
    if p >= 0.5:
        return "over"
    return "under"


def arbitrate_probabilities(
    model_probs: dict[str, float | None],
    *,
    exclude: set[str] | None = None,
    agreement_high: float = 0.75,
    agreement_mod: float = 0.60,
    edge_high: float = 0.08,
    edge_mod: float = 0.04,
) -> dict[str, Any]:
    """
    Arbitrate a dict of model_name → P(over).

    Ensemble is excluded by default so component agreement is meaningful.
    """
    exclude = exclude if exclude is not None else {"ensemble"}
    usable: dict[str, float] = {}
    for name, p in model_probs.items():
        if name in exclude or p is None:
            continue
        try:
            pf = float(p)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(pf):
            continue
        usable[name] = float(np.clip(pf, 1e-6, 1.0 - 1e-6))

    if len(usable) < 2:
        return {
            "confidence_tier": "ABSTAIN",
            "side_lean": None,
            "agreement_rate": None,
            "mean_p_over": float(next(iter(usable.values()))) if usable else None,
            "edge_vs_half": None,
            "n_models": len(usable),
            "votes_over": 0,
            "votes_under": 0,
            "component_probs": usable,
            "disclaimer": DISCLAIMER,
            "reason": "Need >=2 component models with finite P(over)",
        }

    sides = {n: _side_from_p(p) for n, p in usable.items()}
    votes_over = sum(1 for s in sides.values() if s == "over")
    votes_under = len(sides) - votes_over
    if votes_over == votes_under:
        lean = "split"
        agreement = 0.5
    elif votes_over > votes_under:
        lean = "over"
        agreement = votes_over / len(sides)
    else:
        lean = "under"
        agreement = votes_under / len(sides)

    mean_p = float(np.mean(list(usable.values())))
    edge = abs(mean_p - 0.5)

    if lean == "split" or agreement < 0.55:
        tier: ConfidenceTier = "DISAGREE"
    elif agreement >= agreement_high and edge >= edge_high:
        tier = "HIGH"
    elif agreement >= agreement_mod or edge >= edge_mod:
        tier = "MODERATE"
    else:
        tier = "LOW"

    return {
        "confidence_tier": tier,
        "side_lean": lean if lean != "split" else None,
        "agreement_rate": round(agreement, 4),
        "mean_p_over": round(mean_p, 4),
        "edge_vs_half": round(edge, 4),
        "n_models": len(usable),
        "votes_over": votes_over,
        "votes_under": votes_under,
        "component_probs": {k: round(v, 4) for k, v in usable.items()},
        "disclaimer": DISCLAIMER,
        "reason": None,
    }


def attach_arbitration_to_predictions(
    detail_rows: list[dict[str, Any]],
    *,
    exclude: set[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Group rows by (event_id, player_id, target_market) and stamp arbitration
    fields onto every row in the group.
    """
    groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for i, row in enumerate(detail_rows):
        key = (row.get("event_id"), row.get("player_id"), row.get("target_market"))
        groups[key].append(i)

    out = [dict(r) for r in detail_rows]
    for idxs in groups.values():
        probs: dict[str, float | None] = {}
        for i in idxs:
            name = out[i].get("model_name")
            if not name:
                continue
            probs[str(name)] = out[i].get("probability_over_raw")
        arb = arbitrate_probabilities(probs, exclude=exclude)
        for i in idxs:
            out[i]["confidence_tier"] = arb["confidence_tier"]
            out[i]["arbitration_side_lean"] = arb["side_lean"]
            out[i]["arbitration_agreement"] = arb["agreement_rate"]
            out[i]["arbitration_n_models"] = arb["n_models"]
    return out


def summarize_arbitration(detail_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Market-level counts of confidence tiers (unique player-game keys)."""
    seen: set[tuple[Any, ...]] = set()
    counts: dict[str, int] = defaultdict(int)
    for row in detail_rows:
        key = (row.get("event_id"), row.get("player_id"), row.get("target_market"))
        if key in seen:
            continue
        seen.add(key)
        tier = row.get("confidence_tier") or "ABSTAIN"
        counts[str(tier)] += 1
    return {
        "unique_keys": len(seen),
        "tier_counts": dict(counts),
        "disclaimer": DISCLAIMER,
    }
