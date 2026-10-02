"""FeatureSpec — freeze train ≡ slate feature column order (Wave 5b).

One ordered contract so training and slate inference cannot silently drift.
RESEARCH_ONLY — no wager placement.

WHAT WAS ACTUALLY UNGUARDED, which is narrower than audit finding R5 claimed.
R5 said "column order and dtype are unchecked". Order and dtype are in fact
already enforced on the serving path: ``xgboost_pipeline._matrix`` selects by
``feature_cols`` in order and refuses a column whose values will not parse as
numbers. What nothing checked was whether THE SIDECAR AND THE ARTIFACT AGREE.

That is the drift that matters here, and this repository has the exact hazard
that produces it: both ``save`` methods carry explicit code to delete a stale
mean-head file, because an artifact from one fit sitting beside metadata from
another would otherwise be reloaded as current. The classifier sidecar had no
such guard. A ``.meta.json`` from run A next to a booster from run B scores
happily, and the probabilities are a different model's, read through the wrong
column list.

TWO CHECKS, and ``verify_feature_contract`` performs both:

  1. THE SIDECAR AGAINST ITSELF. ``feature_spec.fingerprint`` is written at
     train time over the ordered feature list. Recomputing it at serve time from
     ``feature_cols`` and comparing catches a sidecar that was hand-edited,
     truncated, or half-written — the file contradicting its own hash. No
     modelling library can do this: it is a property of the file, not the model.
  2. THE SIDECAR AGAINST THE ARTIFACT. XGBoost stores its own column names
     inside the model and preserves them through ``save_model``/``load_model``
     (verified on xgboost 3.2.0: ``get_booster().feature_names`` round-trips).
     Comparing that list, in order, to the sidecar's detects a mismatched pair.

HOW MUCH CHECK 2 ADDS ON THE XGBOOST PATH, STATED HONESTLY. Measured on xgboost
3.2.0: predicting with a named DataFrame whose columns are merely PERMUTED
raises ``feature_names mismatch``, and so does a missing column. XGBoost already
refuses both. So on this path check 2 is NOT the difference between garbage and
an abstention — ``score_prob_over`` would have abstained anyway, through its
broad ``except``.

What it changes is WHICH REASON the operator sees. Without it the log reads
``P(Over) skipped: feature_names mismatch: [...] [...]`` — an xgboost internal
message that does not say a sidecar and an artifact came from different fits, and
does not suggest where to look. With it the reason names the mismatch, the market
and the fingerprint. An abstention nobody can act on is only half of an
abstention.

The check also does not depend on the library validating names, which matters if
a model family that indexes features positionally is ever put on the serving
path — there, the same mismatch would be silent.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

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


# ---------------------------------------------------------------------------
# train/serve verification
# ---------------------------------------------------------------------------

META_KEY = "feature_spec"


def spec_from_artifact_meta(meta: Mapping[str, Any]) -> FeatureSpec:
    """
    Rebuild the contract from a model's ``.meta.json``.

    Built from ``feature_cols``, not from the stored ``feature_spec.features``,
    and that is deliberate: ``feature_cols`` is the list the pipeline actually
    selects with, so it is the list whose hash is worth checking. Reading the
    stored copy instead would verify the sidecar against itself in the one way
    that cannot fail.
    """
    cols = [str(c) for c in (meta.get("feature_cols") or [])]
    block = meta.get(META_KEY) or {}
    return FeatureSpec(
        name=str(block.get("name") or "prop_over"),
        version=str(block.get("version") or SCHEMA_VERSION),
        market=str(meta.get("target_market") or block.get("market") or "UNKNOWN"),
        features=cols,
        categorical=[str(c) for c in (
            meta.get("categorical_features") or block.get("categorical") or []
        )],
    )


def first_order_difference(
    expected: Sequence[str], actual: Sequence[str]
) -> str | None:
    """
    Describe how two ordered column lists differ, or None when they match.

    Names the FIRST position that differs rather than only reporting set
    differences: two lists holding the same columns in a different order are the
    silent-garbage case, and a set comparison calls them equal.
    """
    expected = [str(c) for c in expected]
    actual = [str(c) for c in actual]
    if expected == actual:
        return None

    missing = [c for c in expected if c not in set(actual)]
    extra = [c for c in actual if c not in set(expected)]
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"absent from the artifact: {missing[:8]}")
        if extra:
            parts.append(f"present only in the artifact: {extra[:8]}")
        return "; ".join(parts)

    for i, (want, got) in enumerate(zip(expected, actual)):
        if want != got:
            return (
                f"same columns in a different ORDER — position {i} is {got!r} in "
                f"the artifact and {want!r} in the sidecar"
            )
    return f"different lengths: sidecar {len(expected)}, artifact {len(actual)}"


def verify_feature_contract(
    meta: Mapping[str, Any],
    booster_feature_names: Sequence[str] | None = None,
) -> tuple[FeatureSpec, str | None]:
    """
    Check a model artifact against its sidecar. Returns ``(spec, problem)``.

    ``problem`` is None when everything agrees, and otherwise a sentence naming
    what disagreed — for a caller that abstains with a reason rather than
    raising. Scoring through a contract that does not match the artifact does not
    fail loudly; it returns confident probabilities from the wrong model.

    ``booster_feature_names`` absent means the artifact could not say what it was
    trained on. That is reported as a problem only when the sidecar claims a
    fingerprint, because an OLDER sidecar predates this check entirely and
    refusing every previously trained model would be a migration, not a guard.
    """
    spec = spec_from_artifact_meta(meta)
    block = meta.get(META_KEY) or {}
    recorded = block.get("fingerprint")

    if not spec.features:
        return spec, "the sidecar lists no feature_cols, so there is no contract to check"

    if recorded:
        if recorded != spec.fingerprint():
            stored = [str(c) for c in (block.get("features") or [])]
            detail = first_order_difference(spec.features, stored) or "unknown difference"
            return spec, (
                f"the sidecar contradicts its own fingerprint ({recorded} recorded, "
                f"{spec.fingerprint()} recomputed from feature_cols): {detail}. "
                "A half-written or hand-edited sidecar is not a contract."
            )
    if booster_feature_names is not None:
        difference = first_order_difference(spec.features, booster_feature_names)
        if difference:
            return spec, (
                "the model artifact and its sidecar describe different feature "
                f"sets — {difference}. This is a sidecar from one fit beside an "
                "artifact from another. XGBoost would also refuse it, with a "
                "message that does not say so; this one names it."
            )
    return spec, None
