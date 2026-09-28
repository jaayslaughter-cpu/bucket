"""FeatureSpec — freeze train ≡ slate feature column order (Wave 5b).

Inspired by live-edge FeatureSpec: one ordered contract so compare training
and slate inference cannot silently drift. RESEARCH_ONLY — no wager placement.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Sequence

import pandas as pd
from pydantic import BaseModel, Field

from src.models.labels import default_feature_cols

SCHEMA_VERSION = "fs_v2_wave5b"


class FeatureSpec(BaseModel):
    """Ordered feature contract for one market / model family."""

    name: str = "prop_over"
    version: str = SCHEMA_VERSION
    market: str
    features: list[str] = Field(default_factory=list)
    categorical: list[str] = Field(default_factory=list)
    notes: str = "train≡slate column freeze; leakage-safe cols only"

    def fingerprint(self) -> str:
        payload = {
            "version": self.version,
            "market": self.market,
            "features": list(self.features),
            "categorical": list(self.categorical),
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def missing_from(self, columns: Sequence[str] | pd.Index) -> list[str]:
        have = set(columns)
        return [c for c in self.features if c not in have]

    def assert_compatible(self, columns: Sequence[str] | pd.Index) -> None:
        missing = self.missing_from(columns)
        if missing:
            raise ValueError(
                f"DATA_NOT_AVAILABLE: FeatureSpec missing columns {missing} "
                f"(market={self.market} fingerprint={self.fingerprint()})"
            )

    def select(
        self,
        df: pd.DataFrame,
        *,
        fill_value: float | None = None,
    ) -> pd.DataFrame:
        """Return frame with exactly ``features`` columns in frozen order.

        Default leaves NaN in place (zero-inference). Pass an explicit
        ``fill_value`` only when a booster requires finite inputs and the
        imputation policy is logged upstream.
        """
        self.assert_compatible(df.columns)
        out = df.loc[:, self.features].copy()
        for c in self.features:
            if c in self.categorical:
                continue
            series = pd.to_numeric(out[c], errors="coerce")
            if fill_value is not None:
                series = series.fillna(fill_value)
            out[c] = series
        return out

    def to_meta(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "market": self.market,
            "features": list(self.features),
            "categorical": list(self.categorical),
            "fingerprint": self.fingerprint(),
            "n_features": len(self.features),
            "notes": self.notes,
        }


def build_prop_feature_spec(
    market: str,
    available_columns: Sequence[str] | pd.Index | None = None,
    *,
    categorical: Sequence[str] | None = None,
    include_wave5a: bool = True,
    schema_version: str = SCHEMA_VERSION,
) -> FeatureSpec:
    """
    Build a frozen feature list for ``market``.

    Intersects ``default_feature_cols`` (+ Wave 5a/5b additives) with available
    columns when provided so train and slate share the same order.
    """
    m = market.upper()
    base = list(default_feature_cols(m if m in {"PTS", "REB", "AST", "FG3M", "STL", "BLK", "PRA"} else "PTS"))  # type: ignore[arg-type]

    extras: list[str] = [
        "days_rest",
        "is_back_to_back",
    ]
    if include_wave5a:
        extras.extend(
            [
                "USAGE_PROXY_L10",
                f"{m}_STREAK_ABOVE",
                f"{m}_STREAK_BELOW",
                f"{m}_HOT_Z",
                "MINUTES_STABLE",
            ]
        )

    ordered: list[str] = []
    for c in base + extras:
        if c not in ordered:
            ordered.append(c)

    cats = [c for c in (categorical or []) if c]
    for c in cats:
        if c not in ordered:
            ordered.append(c)

    if available_columns is not None:
        have = set(available_columns)
        ordered = [c for c in ordered if c in have]
        cats = [c for c in cats if c in have]

    return FeatureSpec(
        name=f"prop_over_{m.lower()}",
        version=schema_version,
        market=m,
        features=ordered,
        categorical=cats,
    )


def specs_match(a: FeatureSpec, b: FeatureSpec) -> bool:
    return a.fingerprint() == b.fingerprint()
