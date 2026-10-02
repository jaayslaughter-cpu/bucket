"""End-to-end wiring verification — the seams, not the models.

RESEARCH_ONLY. Reads nothing private, writes nothing outside a temp dir,
contacts no network, places no wager.

    python -m scripts.verify_wiring              # every section
    python -m scripts.verify_wiring --section 3  # one section
    python -m scripts.verify_wiring --json       # machine-readable

WHAT THIS IS FOR, AND WHAT IT IS NOT.

``scripts/audit_pipeline.py`` already covers the panel, the folds, whether the
model is learning and whether exported metrics reproduce — but it needs the real
214k-row parquet and the outputs directory. ``scripts/audit_leakage.py`` covers
the splits. Neither covers the SEAMS: whether stage A hands stage B what stage B
expects, and whether a stage that silently produces nothing is distinguishable
from one that correctly abstains.

This script runs on a synthetic panel with no database, no network, no trained
artifact and no credential, so it can run in CI and on a laptop. It therefore
cannot tell you the model is any good. It tells you the pipe is connected.

Each check is PASS, FAIL, WARN or SKIP:

  PASS  the seam holds, demonstrated on data this script constructed
  FAIL  a break: one side hands over something the other does not accept
  WARN  wired, but with a property that will silently reduce output in
        production — the class of defect that looks like a quiet night
  SKIP  needs something absent here (an artifact, a database), named

Exit code is the number of FAILs. WARNs do not fail the run, because several of
them are known design gaps with their own entries in docs/.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

PANEL_STATS = ("PTS", "REB", "AST", "FG3M", "STL", "BLK", "TOV", "MIN")


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

@dataclass
class Result:
    section: str
    name: str
    status: str
    detail: str = ""


@dataclass
class Report:
    results: list[Result] = field(default_factory=list)
    _section: str = ""

    def section(self, title: str) -> None:
        self._section = title
        print(f"\n{title}")

    def _add(self, status: str, name: str, detail: str) -> None:
        self.results.append(Result(self._section, name, status, detail))
        head = f"  {status:<5} {name}"
        print(head if not detail else f"{head}\n          {detail}")

    def ok(self, name: str, detail: str = "") -> None:
        self._add("PASS", name, detail)

    def bad(self, name: str, detail: str) -> None:
        self._add("FAIL", name, detail)

    def warn(self, name: str, detail: str) -> None:
        self._add("WARN", name, detail)

    def skip(self, name: str, detail: str) -> None:
        self._add("SKIP", name, detail)

    def count(self, status: str) -> int:
        return sum(1 for r in self.results if r.status == status)


def guard(report: Report, name: str) -> Callable[[Callable[[], None]], None]:
    """Run a check; an exception in the check is itself a FAIL, never a crash."""
    def run(fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 — a broken check must still report
            report.bad(name, f"the check itself raised: {type(exc).__name__}: {exc}")
    return run


# --------------------------------------------------------------------------
# synthetic panel — shaped like what repository.load_player_panel returns
# --------------------------------------------------------------------------

def synthetic_panel(n_players: int = 4, n_games: int = 24) -> pd.DataFrame:
    """
    A panel with the columns ``load_player_panel`` builds, and its dtypes.

    IDs are strings because ``db/models.py`` declares them ``String(32)`` and
    the repository passes them through unchanged; a check that used ints would
    pass on data production never sees.
    """
    rng = np.random.default_rng(11)
    rows = []
    for p in range(n_players):
        for g in range(n_games):
            rows.append({
                "PLAYER_ID": f"20000{p:02d}",
                "PLAYER_NAME": f"Player {p}",
                "GAME_ID": f"00225{g:05d}",
                "GAME_DATE": pd.Timestamp("2025-10-21") + pd.Timedelta(days=2 * g),
                "SEASON": "2025-26",
                "TEAM_ABBREVIATION": ["LAL", "BOS", "DEN", "MIA"][p % 4],
                "OPPONENT_ABBREVIATION": ["BOS", "LAL", "MIA", "DEN"][p % 4],
                "IS_HOME": bool(g % 2),
                "IS_NEUTRAL_SITE": False,
                "MIN": float(rng.integers(24, 38)),
                "PTS": float(rng.integers(8, 34)),
                "REB": float(rng.integers(1, 13)),
                "AST": float(rng.integers(0, 11)),
                "FG3M": float(rng.integers(0, 6)),
                "STL": float(rng.integers(0, 4)),
                "BLK": float(rng.integers(0, 3)),
                "TOV": float(rng.integers(0, 5)),
                "FGA": float(rng.integers(8, 24)),
                "FTA": float(rng.integers(0, 9)),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 1. ingestion -> features
# --------------------------------------------------------------------------

def section_1(report: Report) -> pd.DataFrame | None:
    report.section("1. ingestion -> feature pipeline")
    from src.features.builder import (
        _ADDITIVE_FEATURE_LAYERS,
        ROLLING_STATS,
        build_feature_matrix,
    )

    panel = synthetic_panel()
    features: pd.DataFrame | None = None

    g = guard(report, "the panel's own columns satisfy build_feature_matrix")

    def _build() -> None:
        nonlocal features
        features = build_feature_matrix(panel)
        made = [c for c in features.columns if c.endswith("_L2")]
        if not made:
            report.bad(
                "the panel's own columns satisfy build_feature_matrix",
                "no {stat}_L2 columns were produced from a panel carrying every "
                "column load_player_panel returns",
            )
            return
        report.ok(
            "the panel's own columns satisfy build_feature_matrix",
            f"{len(features.columns)} columns, {len(made)} layer-2 projections",
        )
    g(_build)
    if features is None:
        return None

    # every ROLLING_STATS member present in the panel must get its rollups
    g = guard(report, "every panel stat reaches its rolling features")

    def _rollups() -> None:
        missing = [
            s for s in ROLLING_STATS
            if s in panel.columns and f"{s}_L5" not in features.columns
        ]
        if missing:
            report.bad("every panel stat reaches its rolling features",
                       f"present in the panel but no _L5 column: {missing}")
        else:
            report.ok("every panel stat reaches its rolling features",
                      f"{len([s for s in ROLLING_STATS if s in panel.columns])} stats")
    g(_rollups)

    # Each additive layer must ADD columns. A layer that is registered and
    # silently abstains is the defect this catches: build_feature_matrix's loop
    # catches Exception and logs, so a dead layer costs nothing visible.
    g = guard(report, "every registered feature layer actually adds columns")

    def _layers() -> None:
        inert = []
        for label, attach in _ADDITIVE_FEATURE_LAYERS:
            before = set(panel.columns)
            try:
                out = attach(panel.copy())
            except Exception as exc:  # noqa: BLE001
                inert.append(f"{label} (raised {type(exc).__name__})")
                continue
            added = set(out.columns) - before
            if not added:
                inert.append(f"{label} (added nothing)")
                continue
            if all(out[c].isna().all() for c in added):
                inert.append(f"{label} (added {len(added)} all-null columns)")
        names = [label for label, _ in _ADDITIVE_FEATURE_LAYERS]
        if inert:
            report.warn(
                "every registered feature layer actually adds columns",
                f"{len(names)} registered; inert on this panel: {inert}. A layer "
                "that abstains here may still work on the real panel — but the "
                "builder's loop swallows the reason either way.",
            )
        else:
            report.ok("every registered feature layer actually adds columns",
                      f"{len(names)}: {', '.join(names)}")
    g(_layers)

    # fatigue must MODIFY the baseline, not sit beside it
    g = guard(report, "fatigue modifies the projection rather than riding along")

    def _fatigue() -> None:
        from main import FATIGUE_COL
        if FATIGUE_COL not in features.columns:
            report.bad("fatigue modifies the projection rather than riding along",
                       f"no {FATIGUE_COL} column")
            return
        mult = pd.to_numeric(features[FATIGUE_COL], errors="coerce")
        moved = mult.notna() & (mult != 1.0)
        if not moved.any():
            report.warn(
                "fatigue modifies the projection rather than riding along",
                "the multiplier is 1.0 on every synthetic row, so this panel "
                "cannot demonstrate the effect (no back-to-backs in it)",
            )
            return
        diff = (
            pd.to_numeric(features.loc[moved, "PTS_L2"], errors="coerce")
            - pd.to_numeric(features.loc[moved, "PTS_BASELINE"], errors="coerce")
        ).abs()
        if float(diff.max() or 0) == 0.0:
            report.bad("fatigue modifies the projection rather than riding along",
                       "PTS_L2 equals PTS_BASELINE on rows whose multiplier is not 1.0")
        else:
            report.ok("fatigue modifies the projection rather than riding along",
                      f"{int(moved.sum())} adjusted rows, max |L2-BASELINE| = {diff.max():.3f}")
    g(_fatigue)

    # dtypes across the transform
    g = guard(report, "ids stay strings and dates stay datetimes through the builder")

    def _dtypes() -> None:
        problems = []
        for col in ("PLAYER_ID", "GAME_ID"):
            if col in features.columns and not all(
                isinstance(v, str) for v in features[col].dropna().head(50)
            ):
                problems.append(f"{col} is no longer str (db declares String(32))")
        if "GAME_DATE" in features.columns and not pd.api.types.is_datetime64_any_dtype(
            features["GAME_DATE"]
        ):
            problems.append("GAME_DATE is not datetime64")
        if problems:
            report.bad("ids stay strings and dates stay datetimes through the builder",
                       "; ".join(problems))
        else:
            report.ok("ids stay strings and dates stay datetimes through the builder")
    g(_dtypes)

    # the builder must not fabricate a season label (docs/season_key.md)
    g = guard(report, "no layer publishes a season it invented")

    def _season() -> None:
        from src.features.season import SEASON_KEY_COL
        bare = build_feature_matrix(panel.drop(columns=["SEASON"]))
        leaked = [c for c in ("SEASON", SEASON_KEY_COL) if c in bare.columns]
        if leaked:
            report.bad("no layer publishes a season it invented",
                       f"a SEASON-less panel came back carrying {leaked}")
        else:
            report.ok("no layer publishes a season it invented",
                      "a SEASON-less panel stays SEASON-less; minutes_weighted abstains")
    g(_season)

    return features


# --------------------------------------------------------------------------
# 2. features -> model inference
# --------------------------------------------------------------------------

def section_2(report: Report, features: pd.DataFrame | None) -> None:
    report.section("2. feature pipeline -> model inference")

    from src.models.compare import load_comparison_config

    cfg = load_comparison_config()
    art_dir = ROOT / str(cfg.get("artifacts_dir", "data/external/model_runs/comparison"))

    # The default path main.py scores with, against where training writes.
    g = guard(report, "the path scoring reads and the path training writes agree")

    def _paths() -> None:
        from main import MODEL_ARTIFACT_DEFAULT
        scoring = ROOT / MODEL_ARTIFACT_DEFAULT
        trained = sorted(art_dir.glob("xgboost_*.json")) if art_dir.exists() else []
        if scoring.exists():
            report.ok("the path scoring reads and the path training writes agree",
                      f"{MODEL_ARTIFACT_DEFAULT} exists")
            return
        report.bad(
            "the path scoring reads and the path training writes agree",
            f"score_prob_over defaults to {MODEL_ARTIFACT_DEFAULT} (absent); "
            f"train-stats writes to {art_dir.relative_to(ROOT) if art_dir.exists() else art_dir}"
            f" ({len(trained)} xgboost_*.json there). main.py needs --model to "
            "bridge them, and scheduler_worker.run_slate calls main.main([]) with "
            "no arguments — so a scheduled run scores nothing even when a model "
            "has been trained.",
        )
    g(_paths)

    # the train/serve contract check, exercised against a real sidecar if present
    g = guard(report, "the feature contract check accepts a matching artifact")

    def _contract() -> None:
        from src.models.feature_spec import verify_feature_contract
        sidecars = sorted(art_dir.rglob("xgboost_*.meta.json")) if art_dir.exists() else []
        if not sidecars:
            report.skip("the feature contract check accepts a matching artifact",
                        f"no xgboost_*.meta.json under {art_dir}")
            return
        meta = json.loads(sidecars[0].read_text(encoding="utf-8"))
        cols = meta.get("feature_cols")
        if not cols:
            report.bad("the feature contract check accepts a matching artifact",
                       f"{sidecars[0].name} has no feature_cols")
            return
        _spec, problem = verify_feature_contract(meta, list(cols))
        if problem:
            report.bad("the feature contract check accepts a matching artifact",
                       f"{sidecars[0].name}: {problem}")
        else:
            report.ok("the feature contract check accepts a matching artifact",
                      f"{sidecars[0].relative_to(ROOT)}, {len(cols)} columns")
    g(_contract)

    g = guard(report, "the feature contract check rejects a permuted column list")

    def _contract_neg() -> None:
        from src.models.feature_spec import verify_feature_contract
        sidecars = sorted(art_dir.rglob("xgboost_*.meta.json")) if art_dir.exists() else []
        if not sidecars:
            report.skip("the feature contract check rejects a permuted column list",
                        "no sidecar to permute")
            return
        meta = json.loads(sidecars[0].read_text(encoding="utf-8"))
        cols = list(meta.get("feature_cols") or [])
        if len(cols) < 2:
            report.skip("the feature contract check rejects a permuted column list",
                        "fewer than two columns")
            return
        swapped = [cols[1], cols[0], *cols[2:]]
        _spec, problem = verify_feature_contract(meta, swapped)
        if problem:
            report.ok("the feature contract check rejects a permuted column list",
                      "order is enforced, not just membership")
        else:
            report.bad("the feature contract check rejects a permuted column list",
                       "a swapped pair was accepted — column ORDER is unchecked, so "
                       "an artifact from another fit would score silently")
    g(_contract_neg)

    # whatever the model emits must be a probability
    g = guard(report, "scored probabilities are in [0, 1] or null")

    def _probs() -> None:
        if features is None:
            report.skip("scored probabilities are in [0, 1] or null", "no feature frame")
            return
        from main import MODEL_ARTIFACT_DEFAULT, score_prob_over
        scored = score_prob_over(features, pd.DataFrame(), ROOT / MODEL_ARTIFACT_DEFAULT)
        vals = pd.to_numeric(pd.Series(scored), errors="coerce").dropna()
        if vals.empty:
            report.skip("scored probabilities are in [0, 1] or null",
                        "no model artifact, so every row abstained — which is the "
                        "documented behaviour, not a range violation")
            return
        bad = vals[(vals < 0) | (vals > 1)]
        if len(bad):
            report.bad("scored probabilities are in [0, 1] or null",
                       f"{len(bad)} value(s) outside [0,1], e.g. {bad.iloc[0]}")
        else:
            report.ok("scored probabilities are in [0, 1] or null", f"{len(vals)} values")
    g(_probs)

    # one artifact scores one market
    g = guard(report, "a one-market model does not claim the other markets")

    def _one_market() -> None:
        from main import DEFAULT_STATS, assemble_projections
        if features is None:
            report.skip("a one-market model does not claim the other markets", "no features")
            return
        probs = pd.Series(0.55, index=features.index)
        probs.attrs["target_market"] = "PTS"
        proj = assemble_projections(
            features, probs, {"status": "DATA_NOT_AVAILABLE"}, stats=DEFAULT_STATS
        )
        if proj.empty:
            report.skip("a one-market model does not claim the other markets",
                        "no projections assembled")
            return
        claimed = sorted(proj.loc[proj["PROB_OVER"].notna(), "MARKET"].unique())
        if claimed != ["PTS"]:
            report.bad("a one-market model does not claim the other markets",
                       f"PROB_OVER written for {claimed}, but the artifact's "
                       "target_market is PTS")
        else:
            others = sorted(set(proj["MARKET"]) - {"PTS"})
            report.warn(
                "a one-market model does not claim the other markets",
                f"correct: PROB_OVER only for PTS. But score_prob_over takes ONE "
                f"model_path, so {others} can never carry a probability in a single "
                "run, and the recorder skips a row with no probability.",
            )
    g(_one_market)


# --------------------------------------------------------------------------
# 3. quant
# --------------------------------------------------------------------------

def section_3(report: Report) -> None:
    report.section("3. quant engine & pick'em contracts")

    g = guard(report, "the EV gate refuses a market with no two-way odds")

    def _gate_refuses() -> None:
        from src.quant.contracts import MarketContext, market_ev_gate
        v = market_ev_gate(MarketContext(
            game_id="g1", source="pickem", captured_at_utc=None,
            over_odds_american=None, under_odds_american=None, status="VALID",
        ))
        if v["status"] == "READY_FOR_EVALUATION":
            report.bad("the EV gate refuses a market with no two-way odds",
                       "a one-sided market cleared the gate")
        else:
            report.ok("the EV gate refuses a market with no two-way odds",
                      f"{v['status']}: {v.get('reason')}")
    g(_gate_refuses)

    g = guard(report, "the EV gate passes a real two-way price")

    def _gate_passes() -> None:
        from datetime import datetime, timezone

        from src.quant.contracts import MarketContext, market_ev_gate
        # A posted LINE is required as well as the pair — an earlier version of
        # this check omitted it and read the gate's correct refusal as a break.
        v = market_ev_gate(MarketContext(
            game_id="g1", market="PTS", player_name="Player 0", line=24.5,
            source="propline", captured_at_utc=datetime.now(timezone.utc),
            over_odds_american=-110, under_odds_american=-110, status="VALID",
        ))
        if v["status"] != "READY_FOR_EVALUATION":
            report.bad("the EV gate passes a real two-way price",
                       f"a -110/-110 VALID market was refused: {v.get('reason')}. "
                       "The gate would then abstain on every priced row too.")
        else:
            report.ok("the EV gate passes a real two-way price", v["status"])
    g(_gate_passes)

    # the per-row verdict vs the per-slate scalar
    g = guard(report, "MARKET_STATUS is a per-row claim, not a slate aggregate")

    def _granularity() -> None:
        src = (ROOT / "main.py").read_text(encoding="utf-8")
        writes_scalar = 'ev_verdict["status"]' in src and '"MARKET_STATUS"' in src
        per_row = re.search(r'"MARKET_STATUS":\s*\w+\[', src)
        if writes_scalar and per_row:
            report.warn(
                "MARKET_STATUS is a per-row claim, not a slate aggregate",
                "evaluate_ev_gate loops every prop and returns COUNTS plus one "
                "status; assemble_projections writes that single status onto every "
                "row, and repository persists it per row as `market_status`. One "
                "ready prop out of three hundred therefore labels all three "
                "hundred READY_FOR_EVALUATION.",
            )
        else:
            report.ok("MARKET_STATUS is a per-row claim, not a slate aggregate")
    g(_granularity)

    g = guard(report, "pick'em entry EV evaluates without a hard block")

    def _pickem() -> None:
        from src.quant.contracts import MarketContext
        from src.quant.dfs_entry import PickemLeg, route_pickem_entry
        from src.quant.dfs_payouts import DfsPayoutStructure

        # `market` is the OPERATOR's row carrying the multiplier, not a string.
        legs = [
            PickemLeg(
                leg_id=f"L{i}",
                market=MarketContext(
                    game_id="0022500001", market="PTS", player_name=f"Player {i}",
                    line=20.5, payout_multiplier=3.0, is_pickem=True,
                    source="demo_operator", status="VALID",
                ),
                side="over",
                model_probability=0.58,
            )
            for i in range(2)
        ]
        structure = DfsPayoutStructure(n_picks=2, payouts={2: 3.0},
                                       source="verify_wiring synthetic")
        ev = route_pickem_entry(structure, legs)
        status = getattr(ev, "status", None) or getattr(ev, "route", None)
        report.ok("pick'em entry EV evaluates without a hard block",
                  f"2-leg entry returned {type(ev).__name__} status={status!r} "
                  "(an abstention is a valid outcome; an exception is not)")
    g(_pickem)

    g = guard(report, "advisory sizing never marks a stake as placed")

    def _sizing() -> None:
        from src.quant.advisory_sizing import recommended_units_binary
        size = recommended_units_binary(0.58, 1.91)
        d = size.as_dict() if hasattr(size, "as_dict") else dict(size)
        if d.get("AUTO_PLACED") is not False:
            report.bad("advisory sizing never marks a stake as placed",
                       f"AUTO_PLACED is {d.get('AUTO_PLACED')!r}, expected False")
        else:
            report.ok(
                "advisory sizing never marks a stake as placed",
                f"RECOMMENDED_UNITS={d.get('RECOMMENDED_UNITS')} at "
                f"{d.get('KELLY_FRACTION_APPLIED')} Kelly "
                f"(full f*={d.get('FULL_KELLY_FRACTION'):.4f}), AUTO_PLACED=False",
            )
    g(_sizing)


# --------------------------------------------------------------------------
# 4. settlement, database, storage
# --------------------------------------------------------------------------

def section_4(report: Report, features: pd.DataFrame | None) -> None:
    report.section("4. settlement, database & storage")

    g = guard(report, "a transactional scope commits, rolls back and closes")

    def _scope() -> None:
        src = (ROOT / "src" / "db" / "session.py").read_text(encoding="utf-8")
        need = {
            "commit": "session.commit()" in src,
            "rollback": "session.rollback()" in src,
            "close in finally": re.search(r"finally:\s*\n\s*session\.close\(\)", src) is not None,
            "pool_pre_ping": "pool_pre_ping=True" in src,
            "expire_on_commit=False": "expire_on_commit=False" in src,
        }
        missing = [k for k, v in need.items() if not v]
        if missing:
            report.bad("a transactional scope commits, rolls back and closes",
                       f"session.py is missing: {missing}")
        else:
            report.ok("a transactional scope commits, rolls back and closes",
                      "commit/rollback/finally-close, pool_pre_ping, "
                      "expire_on_commit=False (ORM rows stay readable after close)")
    g(_scope)

    g = guard(report, "a missing DATABASE_URL is reported, not raised")

    def _healthcheck() -> None:
        import os

        from src.db.session import healthcheck
        saved = {k: os.environ.pop(k, None) for k in
                 ("DATABASE_URL", "PGHOST", "PGUSER", "PGPASSWORD", "PGDATABASE")}
        try:
            import src.db.session as sess
            sess._engine = None
            ok, message = healthcheck()
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v
            import src.db.session as sess
            sess._engine = None
        if ok:
            report.warn("a missing DATABASE_URL is reported, not raised",
                        "healthcheck returned ok with no connection settings")
        elif not message:
            report.bad("a missing DATABASE_URL is reported, not raised",
                       "returned False with no reason, so a log reader cannot act")
        else:
            report.ok("a missing DATABASE_URL is reported, not raised",
                      message.split(".")[0][:90])
    g(_healthcheck)

    g = guard(report, "the recorder writes one gradeable row per usable prediction")

    def _recorder() -> None:
        from src.settlement.recorder import pending_prop_result_rows
        proj = pd.DataFrame({
            "PLAYER_ID": ["2000001", "2000002", "2000003"],
            "PLAYER_NAME": ["A", "B", "C"],
            "GAME_ID": ["0022500001"] * 3,
            "GAME_DATE": [pd.Timestamp("2025-11-01")] * 3,
            "MARKET": ["PTS"] * 3,
            "LINE": [24.5, 24.5, None],          # third has no line
            "PROB_OVER": [0.61, None, 0.55],     # second has no probability
            "AVAILABILITY": ["AVAILABLE"] * 3,
            "MARKET_STATUS": ["DATA_NOT_AVAILABLE"] * 3,
        })
        # `source` comes ONLY from the captured board, joined on exact
        # (player_name, market). Passing None skips every row for want of a
        # source, which an earlier version of this check mistook for a defect.
        lines = pd.DataFrame({
            "player_name": ["A", "B", "C"],
            "market": ["PTS"] * 3,
            "line": [24.5, 24.5, 24.5],
            "source": ["propline"] * 3,
            "nba_game_id": ["0022500001"] * 3,
        })
        out = pending_prop_result_rows(proj, lines, run_id="verify")
        rows = getattr(out, "rows", None)
        skipped = getattr(out, "skipped", None)
        if rows is None:
            report.bad("the recorder writes one gradeable row per usable prediction",
                       f"RecordingReport has no .rows ({type(out).__name__})")
            return
        if len(rows) != 1:
            report.bad("the recorder writes one gradeable row per usable prediction",
                       f"expected 1 of 3 rows gradeable (one lacks a line, one lacks "
                       f"a probability), got {len(rows)}; skips={skipped}")
        else:
            report.ok("the recorder writes one gradeable row per usable prediction",
                      f"1 of 3 kept; the rest skipped with reasons: "
                      f"{[s['reason'] for s in skipped]}")
    g(_recorder)

    g = guard(report, "a name mismatch in the board is reported, not silent")

    def _name_join() -> None:
        from src.settlement.recorder import pending_prop_result_rows
        proj = pd.DataFrame({
            "PLAYER_NAME": ["Nikola Jokic"], "GAME_ID": ["0022500001"],
            "GAME_DATE": [pd.Timestamp("2025-11-01")], "MARKET": ["PTS"],
            "LINE": [24.5], "PROB_OVER": [0.61], "AVAILABILITY": ["AVAILABLE"],
        })
        lines = pd.DataFrame({           # same player, a different name format
            "player_name": ["Nikola Jokić"], "market": ["PTS"], "line": [24.5],
            "source": ["propline"], "nba_game_id": ["0022500001"],
        })
        out = pending_prop_result_rows(proj, lines, run_id="verify")
        crosswalk = ROOT / "src" / "ingestion" / "id_crosswalk.py"
        if len(out.rows) == 0 and not crosswalk.exists():
            report.bad(
                "a name mismatch in the board is reported, not silent",
                "one diacritic skipped the row for 'no line source', and the "
                "crosswalk five modules name as the fix "
                "(src/ingestion/id_crosswalk.py) does not exist. A systematic "
                "name-format difference between the board and the NBA panel "
                "therefore records ZERO gradeable rows while the run reports "
                "success.",
            )
        elif len(out.rows) == 0:
            report.warn("a name mismatch in the board is reported, not silent",
                        f"row skipped: {[s['reason'] for s in out.skipped]}; "
                        "a crosswalk exists but this path does not use it")
        else:
            report.ok("a name mismatch in the board is reported, not silent",
                      "the join tolerated a name-format difference")
    g(_name_join)

    g = guard(report, "the recorder reads an availability column somebody writes")

    def _availability_writer() -> None:
        rec = (ROOT / "src" / "settlement" / "recorder.py").read_text(encoding="utf-8")
        if "AVAILABILITY" not in rec:
            report.skip("the recorder reads an availability column somebody writes",
                        "recorder does not mention AVAILABILITY")
            return
        # The writer assigns through a module constant, not the literal string,
        # so a literal-only search reports a break that is not there.
        writers = []
        for p in list((ROOT / "src").rglob("*.py")) + [ROOT / "main.py"]:
            if p.name == "recorder.py" or "__pycache__" in str(p):
                continue
            body = p.read_text(encoding="utf-8")
            consts = re.findall(r"^(\w+)\s*=\s*[\"']AVAILABILITY[\"']", body, re.M)
            names = [r"\[.AVAILABILITY.\]", r'"AVAILABILITY":'] + [
                rf"\[{c}\]" for c in consts
            ]
            if any(re.search(pat, body) for pat in names):
                writers.append(p.relative_to(ROOT).as_posix())
        if not writers:
            report.bad("the recorder reads an availability column somebody writes",
                       "recorder filters on AVAILABILITY and no module writes it — "
                       "every row would read as unknown")
        else:
            report.ok("the recorder reads an availability column somebody writes",
                      f"written by {writers}")
    g(_availability_writer)

    g = guard(report, "the box-score grader refuses a game that is not final")

    def _boxscore() -> None:
        src = (ROOT / "src" / "settlement" / "boxscore_fetcher.py").read_text(encoding="utf-8")
        if "gameStatus" not in src:
            report.bad("the box-score grader refuses a game that is not final",
                       "no gameStatus guard — a partial line could grade an Over as a LOSS")
        else:
            report.ok("the box-score grader refuses a game that is not final",
                      "gameStatus is checked before stats are read")
    g(_boxscore)

    g = guard(report, "the calibration report the gate reads can be produced")

    def _calibration() -> None:
        from src.quant.dfs_payouts import ProbabilitySource
        from src.quant.publication_gate import calibration_gate
        withheld = calibration_gate(None, probability_source=ProbabilitySource.MODEL)
        if withheld.allowed:
            report.bad("the calibration report the gate reads can be produced",
                       "the gate ALLOWED publication with no report at all")
            return
        # Built by the REAL producer from synthetic graded rows, not hand-written.
        # A hand-written fixture used the wrong timestamp key and the gate
        # correctly withheld on it, which read as a break in the gate.
        from src.settlement.calibration import calibration_from_graded_rows
        rng = np.random.default_rng(7)
        rows = []
        for _ in range(400):
            p_over = float(rng.uniform(0.05, 0.95))
            hit = bool(rng.random() < p_over)      # well-calibrated by construction
            rows.append({
                "outcome_status": "WIN" if hit else "LOSS",
                "prob_over": p_over,
                "predicted_side": "OVER",
                "predicted_line": 24.5,
            })
        produced = calibration_from_graded_rows(rows)
        allowed = calibration_gate(produced, probability_source=ProbabilitySource.MODEL)
        if produced.get("status") != "OK":
            report.warn(
                "the calibration report the gate reads can be produced",
                f"no report -> withheld, correctly. But the producer returned "
                f"status={produced.get('status')!r} on 400 well-calibrated "
                f"synthetic rows: {produced.get('reason')}",
            )
        elif not allowed.allowed:
            report.bad(
                "the calibration report the gate reads can be produced",
                f"the producer emitted status=OK (ece={produced.get('ece')}, "
                f"n_scored={produced.get('n_scored')}) and the gate still "
                f"withheld: {allowed.reason}. The two sides disagree about the "
                "report's own schema, so the gate can never open.",
            )
        else:
            report.ok(
                "the calibration report the gate reads can be produced",
                f"no report -> withheld; the producer's own output on 400 "
                f"calibrated rows -> allowed (ece={produced.get('ece'):.4f}, "
                f"n_scored={produced.get('n_scored')})",
            )
    g(_calibration)


# --------------------------------------------------------------------------
# 5. background execution & dispatch
# --------------------------------------------------------------------------

def section_5(report: Report) -> None:
    report.section("5. background execution & dispatcher")

    g = guard(report, "the scheduler registers the jobs it documents")

    def _jobs() -> None:
        try:
            import apscheduler  # noqa: F401
        except ImportError:
            report.skip("the scheduler registers the jobs it documents",
                        "APScheduler not installed here: pip install -e '.[deploy]'")
            return
        import scheduler_worker as w
        jobs = w.describe(w.build_scheduler())
        ids = {j["id"] for j in jobs}
        if ids != {"slate", "settlement"}:
            report.bad("the scheduler registers the jobs it documents",
                       f"expected slate+settlement, got {sorted(ids)}")
            return
        bad = [j["id"] for j in jobs if j["max_instances"] != 1 or not j["coalesce"]]
        if bad:
            report.bad("the scheduler registers the jobs it documents",
                       f"{bad} allow a second concurrent copy or do not coalesce")
        else:
            report.ok("the scheduler registers the jobs it documents",
                      "; ".join(f"{j['id']} grace={j['misfire_grace_time']}s" for j in jobs))
    g(_jobs)

    g = guard(report, "a failing job does not take the worker down")

    def _safety() -> None:
        src = (ROOT / "scheduler_worker.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        wrapped, unwrapped = [], []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith("run_"):
                has = any(isinstance(n, ast.Try) for n in ast.walk(node))
                (wrapped if has else unwrapped).append(node.name)
        if unwrapped:
            report.bad("a failing job does not take the worker down",
                       f"no try/except in {unwrapped}")
        else:
            report.ok("a failing job does not take the worker down",
                      f"{', '.join(wrapped)} each catch and log")
    g(_safety)

    g = guard(report, "the scheduled slate passes the arguments it needs")

    def _argv() -> None:
        src = (ROOT / "scheduler_worker.py").read_text(encoding="utf-8")
        if "main.main(argv or [])" in src and "--model" not in src:
            report.warn(
                "the scheduled slate passes the arguments it needs",
                "run_slate calls main.main(argv or []) with no --model, and no "
                "environment variable overrides the model path, so a scheduled "
                "run always reads main.MODEL_ARTIFACT_DEFAULT. See section 2.",
            )
        else:
            report.ok("the scheduled slate passes the arguments it needs")
    g(_argv)

    g = guard(report, "a withheld board sends the gate's reason, not the rows")

    def _embed() -> None:
        from src.notify.discord import build_decision_board_embed
        from src.quant.dfs_payouts import ProbabilitySource
        from src.quant.publication_gate import calibration_gate
        verdict = calibration_gate(None, probability_source=ProbabilitySource.MODEL)
        row = type("Row", (), {
            "player_name": "Player 0", "target_market": "PTS", "side": "over",
            "research_line": 24.5, "status": "RECOMMENDED", "book_ev_over": 0.08,
            "slate_date": "2026-10-02",
        })()
        embed = build_decision_board_embed([row], slate_date="2026-10-02",
                                           publication=verdict)
        blob = json.dumps(embed).lower()
        leaked = [w for w in ("guaranteed", "lock", "sure thing", "profitable")
                  if w in blob]
        if leaked:
            report.bad("a withheld board sends the gate's reason, not the rows",
                       f"the embed contains banned claim words: {leaked}")
        elif "24.5" in blob:
            report.bad("a withheld board sends the gate's reason, not the rows",
                       "a row's line reached a WITHHELD embed")
        else:
            report.ok("a withheld board sends the gate's reason, not the rows",
                      "no row content, no claim words")
    g(_embed)

    # the board's dates
    g = guard(report, "a board row keeps the date of the game it describes")

    def _board_dates() -> None:
        from src.quant.paper_research import research_slate_from_predictions
        detail = [{
            "event_id": "0022500001", "player_id": "2000001", "player_name": "A",
            "target_market": "PTS", "game_date": "2025-02-10",
            "prop_line": 24.5, "prediction_mean": 25.1,
            "prediction_std_or_dispersion": 5.0,
            "probability_over_raw": 0.56, "probability_under_raw": 0.44,
            "probability_push_raw": None, "model_name": "distribution",
        }]
        rows = research_slate_from_predictions(detail, slate_date="2026-10-02",
                                              preferred_model="distribution")
        if not rows:
            report.skip("a board row keeps the date of the game it describes",
                        "no board row produced from the synthetic detail row")
            return
        r = rows[0]
        carried = {
            f for f in ("game_date", "event_date", "game_start_pt")
            if getattr(r, f, None)
        }
        if r.slate_date == "2026-10-02" and not carried:
            report.bad(
                "a board row keeps the date of the game it describes",
                "the detail row's game_date=2025-02-10 is dropped and the row is "
                "stamped slate_date=2026-10-02. ResearchSlateRow has no game-date "
                "field, so a prediction from the validation window is "
                "indistinguishable from one for tonight. With "
                "scheduler_worker.run_board's default train_end=2025-01-15 / "
                "validation_end=2025-02-15, EVERY board row is historical and "
                "every one claims today.",
            )
        else:
            report.ok("a board row keeps the date of the game it describes",
                      f"carries {sorted(carried)}")
    g(_board_dates)

    g = guard(report, "the board's training window is not pinned to a past date")

    def _window() -> None:
        src = (ROOT / "scheduler_worker.py").read_text(encoding="utf-8")
        pinned = re.findall(r'or\s+"(20\d\d-\d\d-\d\d)"', src)
        if pinned:
            report.warn(
                "the board's training window is not pinned to a past date",
                f"run_board falls back to fixed dates {pinned} when "
                "PROPIQ_BOARD_TRAIN_END / _VALIDATION_END are unset. They never "
                "advance, so the window recedes further into the past every day "
                "and nothing reports it.",
            )
        else:
            report.ok("the board's training window is not pinned to a past date")
    g(_window)


# --------------------------------------------------------------------------
# 6. dangling references and orphans
# --------------------------------------------------------------------------

def section_6(report: Report) -> None:
    report.section("6. dangling references & orphans")

    g = guard(report, "every module path cited in code exists")

    def _dangling() -> None:
        cited: dict[str, list[str]] = {}
        files = [p for p in list((ROOT / "src").rglob("*.py"))
                 + list((ROOT / "scripts").rglob("*.py"))
                 + [ROOT / "main.py", ROOT / "scheduler_worker.py"]
                 if "__pycache__" not in str(p)]
        for p in files:
            if p.name == "verify_wiring.py":
                continue  # it names the path in order to report it
            for m in re.finditer(r"\b((?:src/)?(?:ingestion|features|models|quant|"
                                 r"settlement|pipeline|notify|db)/\w+\.py)\b",
                                 p.read_text(encoding="utf-8")):
                path = m.group(1)
                full = ROOT / (path if path.startswith("src/") else f"src/{path}")
                if not full.exists():
                    cited.setdefault(path, []).append(p.relative_to(ROOT).as_posix())
        if cited:
            lines = [f"{path} cited by {sorted(set(who))}" for path, who in cited.items()]
            report.bad("every module path cited in code exists", "; ".join(lines))
        else:
            report.ok("every module path cited in code exists")
    g(_dangling)

    g = guard(report, "no production module is unreachable")

    def _orphans() -> None:
        imported: set[str] = set()
        for p in ROOT.rglob("*.py"):
            if "__pycache__" in str(p) or ".venv" in str(p):
                continue
            try:
                tree = ast.parse(p.read_text(encoding="utf-8"))
            except SyntaxError:
                continue
            for n in ast.walk(tree):
                if isinstance(n, ast.ImportFrom) and n.module:
                    imported.add(n.module)
                elif isinstance(n, ast.Import):
                    imported.update(a.name for a in n.names)
        orphans = []
        for p in sorted((ROOT / "src").rglob("*.py")):
            if p.name == "__init__.py" or "__pycache__" in str(p):
                continue
            dotted = p.relative_to(ROOT).with_suffix("").as_posix().replace("/", ".")
            if dotted not in imported:
                orphans.append(f"{p.relative_to(ROOT).as_posix()} "
                               f"({len(p.read_text().splitlines())} lines)")
        if orphans:
            report.warn("no production module is unreachable",
                        f"nothing imports: {orphans}")
        else:
            report.ok("no production module is unreachable")
    g(_orphans)

    g = guard(report, "a schedule and a roster source exist for a forward slate")

    def _forward() -> None:
        main_src = (ROOT / "main.py").read_text(encoding="utf-8")
        worker_src = (ROOT / "scheduler_worker.py").read_text(encoding="utf-8")
        has_schedule = (ROOT / "src/ingestion/espn_schedule.py").exists()
        has_roster = "def fetch_roster" in (
            ROOT / "src/ingestion/espn_availability.py"
        ).read_text(encoding="utf-8")
        wired = "espn_schedule" in main_src or "espn_schedule" in worker_src
        if has_schedule and has_roster and not wired:
            report.bad(
                "a schedule and a roster source exist for a forward slate",
                "load_player_panel reads PlayerGameLog — COMPLETED games only — and "
                "_filter_to_slate keeps rows whose GAME_DATE equals the slate, so a "
                "09:00 PT run finds zero rows for games that have not been played "
                "and returns success_no_data. espn_schedule.load_slate (games + "
                "tipoffs) and espn_availability.fetch_roster (players per team) both "
                "exist and are tested; neither is imported by main.py or "
                "scheduler_worker.py.",
            )
        elif wired:
            report.ok("a schedule and a roster source exist for a forward slate")
        else:
            report.skip("a schedule and a roster source exist for a forward slate",
                        f"schedule={has_schedule} roster={has_roster}")
    g(_forward)


# --------------------------------------------------------------------------

SECTIONS: dict[str, str] = {
    "1": "ingestion -> features",
    "2": "features -> inference",
    "3": "quant",
    "4": "settlement / db",
    "5": "scheduler / dispatch",
    "6": "dangling refs / orphans",
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--section", action="append", choices=sorted(SECTIONS),
                    help="run only these sections (repeatable)")
    ap.add_argument("--json", action="store_true", help="emit results as JSON")
    args = ap.parse_args(argv)
    wanted = set(args.section or SECTIONS)

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    report = Report()
    features = None
    if "1" in wanted:
        features = section_1(report)
    elif "2" in wanted:
        features = section_1(Report())  # quiet build, section 2 needs the frame
    if "2" in wanted:
        section_2(report, features)
    if "3" in wanted:
        section_3(report)
    if "4" in wanted:
        section_4(report, features)
    if "5" in wanted:
        section_5(report)
    if "6" in wanted:
        section_6(report)

    print(
        f"\n{report.count('PASS')} passed, {report.count('FAIL')} failed, "
        f"{report.count('WARN')} warned, {report.count('SKIP')} skipped"
    )
    if args.json:
        print(json.dumps([r.__dict__ for r in report.results], indent=2))
    return report.count("FAIL")


if __name__ == "__main__":
    sys.exit(main())
