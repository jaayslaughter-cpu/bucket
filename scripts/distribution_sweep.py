"""
scripts/distribution_sweep.py — is the family chosen on the right objective?

THE QUESTION. `fit_count_dispersion` picks a market's distribution family with

    family = min(ranked, key=lambda pair: scores[pair[0]])[1]

where every score is a MEAN NEGATIVE LOG LIKELIHOOD of the stat value. That
asks "which family describes this player's whole count distribution best?".

It is not the question the board answers. The board answers "will this stat
clear THIS posted line?", and a line is one threshold, sitting in a narrow band
near the middle of the distribution. NLL is summed over the entire support --
the zero, the long right tail, every outcome no money is on. A family can
describe the whole distribution better and still price the one threshold worse,
and nothing in this repository has ever checked whether it does.

So this re-ranks the four families by a THRESHOLD-LOCAL score -- Brier and log
loss on the realised over/under at real posted lines -- and reports whether
that ranking disagrees with the NLL ranking the live path uses. Disagreement is
evidence the selector optimises the wrong thing. Agreement is evidence it does
not, which is equally worth knowing and cannot be assumed.

WHY REAL POSTED LINES, NEVER `line = expected_value`. This was read off a
candidate repository's own warning (EdgarParra565/player-performance-forecaster,
surveyed 2026-10-24), which is honest about the bug in its own sweep:

    "the sweep's default `line = expected_value` makes symmetric distributions
     (normal / student_t / uniform) degenerate -- prob_over evaluates to ~0.5
     at the mean and the 0.55-edge bet filter rejects every wager... Asymmetric
     distributions dominate the avg_roi ranking by default; this is a property
     of the backtest harness, not a claim that they fit player props better."

Setting the line to the projection guarantees the asymmetric families win. Any
sweep that does it is measuring its own harness. This one refuses to run
without a column of lines somebody actually posted.

IT REIMPLEMENTS NEITHER FAMILY NOR TAIL. The parameters come from
`residuals.fit_count_dispersion` and every probability from
`residuals.over_under_push_from_dispersion` -- the same two functions the live
path calls. A second implementation here would be measuring a distribution this
project does not score with, which is the defect this tree keeps removing.

AND IT ASKS WHETHER THE GAP IS REAL. A Brier difference between two families is
subject to the same sampling noise as a difference between two models, so the
comparison goes through `models.paired_bootstrap.decisive_winner`, clustered by
game.

IT CHANGES NOTHING. Output is a JSON report and a table under `outputs/`. The
live selector is untouched: switching it is a decision for a human holding this
report, and it would need a new `.meta.json` contract on the mounted volume.

    PYTHONPATH=. python -m scripts.distribution_sweep --from-file rows.csv
    PYTHONPATH=. python -m scripts.distribution_sweep --from-file rows.csv --json

RESEARCH_ONLY. This scores distributions. It prices nothing and sizes nothing.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger("propiq.distribution_sweep")

#: The four families `residuals` fits. Named here only to iterate them; the
#: fitted parameters for each come from `fit_count_dispersion`.
FAMILIES: tuple[str, ...] = ("poisson", "negbin", "zip", "normal")

#: The column names this sweep needs, and what each must contain.
REQUIRED_COLUMNS: dict[str, str] = {
    "market": "PTS / REB / AST / ... — one sweep per market",
    "line": "the line a book or board ACTUALLY posted, never the projection",
    "actual": "the realised stat value",
    "mean": "the model's projected mean for that player-game",
}

#: Used to cluster the bootstrap. Optional, and its absence is reported rather
#: than silently downgrading the interval.
GAME_COLUMN = "event_id"

#: Below this many scored rows a market's ranking is sampling noise and is
#: reported as unranked rather than ranked badly.
MIN_ROWS_PER_MARKET = 100

DEFAULT_OUT = Path("outputs/distribution_sweep.json")


class SweepError(RuntimeError):
    """Raised when the inputs cannot support the comparison."""


def _load(path: Path):
    import pandas as pd

    if not path.exists():
        raise SweepError(f"no such file: {path}")
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _check_columns(frame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        detail = "; ".join(f"{c}: {REQUIRED_COLUMNS[c]}" for c in missing)
        raise SweepError(f"missing required column(s) {missing}. {detail}")


def _dispersion_for(family: str, fitted: Any, n_rows: int):
    """
    A CountDispersion pinned to one family, carrying the parameters the
    selector fitted for it.

    The parameters are NOT refitted per family here: `fit_count_dispersion`
    already fits phi, the zero-inflation and the sigma scale and reports all
    of them in `selection_scores`, so reading them keeps one fitter.
    """
    from src.models.residuals import MIN_PHI, CountDispersion

    scores = fitted.selection_scores
    return CountDispersion(
        family=family,  # type: ignore[arg-type]
        phi=float(scores.get("fitted_phi", MIN_PHI)) if family == "negbin" else MIN_PHI,
        n_train_rows=n_rows,
        selection_scores=dict(scores),
        zero_inflation=(
            float(scores.get("fitted_zero_inflation", 0.0)) if family == "zip" else 0.0
        ),
        sigma_scale=(
            float(scores.get("fitted_sigma_scale", 1.0)) if family == "normal" else 1.0
        ),
    )


def score_market(frame, market: str) -> dict[str, Any]:
    """
    Rank the four families for one market by NLL and by threshold score.

    Pushes are dropped from the threshold score: a stat landing exactly on a
    whole-number line is neither an over nor an under, so scoring it as either
    would charge a family for an outcome it correctly called a push.
    """
    import numpy as np

    from src.models.residuals import (
        fit_count_dispersion,
        over_under_push_from_dispersion,
    )

    rows = frame[frame["market"] == market]
    y = np.asarray(rows["actual"], dtype=float)
    mu = np.asarray(rows["mean"], dtype=float)
    lines = np.asarray(rows["line"], dtype=float)

    usable = np.isfinite(y) & np.isfinite(mu) & np.isfinite(lines)
    # A push is not an over/under outcome.
    decided = usable & (y != lines)
    if int(decided.sum()) < MIN_ROWS_PER_MARKET:
        return {
            "market": market,
            "status": "NOT_RANKED",
            "reason": (
                f"{int(decided.sum())} decided row(s), below the "
                f"{MIN_ROWS_PER_MARKET} this sweep needs to rank anything"
            ),
            "n_rows": int(usable.sum()),
            "n_pushes": int((usable & (y == lines)).sum()),
        }

    fitted = fit_count_dispersion(y[usable], mu[usable], market=market)
    nll = {
        family: float(fitted.selection_scores[f"{family}_mean_nll"])
        for family in FAMILIES
    }

    y_d, mu_d, line_d = y[decided], mu[decided], lines[decided]
    over = (y_d > line_d).astype(float)

    per_family: dict[str, Any] = {}
    probabilities: dict[str, Any] = {}
    for family in FAMILIES:
        dispersion = _dispersion_for(family, fitted, int(usable.sum()))
        p_over = np.array([
            over_under_push_from_dispersion(float(m), float(ln), dispersion)
            .get("probability_over")
            for m, ln in zip(mu_d, line_d, strict=True)
        ], dtype=object)
        valid = np.array([v is not None for v in p_over])
        if valid.sum() < MIN_ROWS_PER_MARKET:
            per_family[family] = {
                "status": "NOT_SCORED",
                "reason": f"only {int(valid.sum())} row(s) priced",
                "nll": nll[family],
            }
            continue
        p = np.clip(np.asarray(p_over[valid], dtype=float), 1e-12, 1 - 1e-12)
        truth = over[valid]
        per_family[family] = {
            "status": "OK",
            "nll": nll[family],
            "brier": float(np.mean((p - truth) ** 2)),
            "log_loss": float(-np.mean(truth * np.log(p) + (1 - truth) * np.log(1 - p))),
            "mean_p_over": float(p.mean()),
            "base_rate_over": float(truth.mean()),
            "n_scored": int(valid.sum()),
        }
        probabilities[family] = (truth, p, valid)

    scored = {f: s for f, s in per_family.items() if s["status"] == "OK"}
    if not scored:
        return {
            "market": market,
            "status": "NOT_RANKED",
            "reason": "no family priced enough rows to score",
            "families": per_family,
        }

    by_nll = min(nll, key=lambda f: nll[f])
    by_brier = min(scored, key=lambda f: scored[f]["brier"])

    out: dict[str, Any] = {
        "market": market,
        "status": "RANKED",
        "n_decided_rows": int(decided.sum()),
        "n_pushes": int((usable & (y == lines)).sum()),
        "live_selector_picks": by_nll,
        "threshold_score_picks": by_brier,
        "selectors_disagree": by_nll != by_brier,
        "families": per_family,
        "fitted_family": fitted.family,
        "fallback_reason": fitted.fallback_reason,
    }

    # IS THE DISAGREEMENT REAL? A Brier gap between families is subject to the
    # same sampling noise as a gap between models, so it goes through the same
    # test — clustered by game, since props from one game move together.
    if out["selectors_disagree"] and by_nll in probabilities:
        from src.models.paired_bootstrap import PairedBootstrapError, decisive_winner

        truth, p_leader, valid = probabilities[by_brier]
        _, p_runner, _ = probabilities[by_nll]
        games = None
        if GAME_COLUMN in rows.columns:
            games = np.asarray(rows[GAME_COLUMN])[decided][valid]
        try:
            verdict = decisive_winner(
                truth, p_leader, p_runner, metric="brier", games=games
            )
            out["disagreement_decisive"] = verdict["decisive"]
            out["disagreement_note"] = verdict["note"]
            out["disagreement_ci"] = [verdict["ci_low"], verdict["ci_high"]]
            out["disagreement_cluster"] = verdict["cluster"]
        except PairedBootstrapError as exc:
            out["disagreement_decisive"] = None
            out["disagreement_note"] = f"could not be tested: {exc}"
        if games is None:
            out["disagreement_note"] += (
                f" (no {GAME_COLUMN} column, so the interval is the iid one "
                f"and narrower than it should be)"
            )
    return out


def sweep(path: Path) -> dict[str, Any]:
    frame = _load(path)
    _check_columns(frame)
    markets = sorted(str(m) for m in frame["market"].dropna().unique())
    if not markets:
        raise SweepError("no markets in the input")
    results = [score_market(frame, market) for market in markets]
    disagreeing = [
        r["market"] for r in results if r.get("selectors_disagree")
    ]
    return {
        "research_only": True,
        "source": str(path),
        "n_rows": int(len(frame)),
        "markets": results,
        "markets_where_selectors_disagree": disagreeing,
        "verdict": (
            "The NLL selector and the threshold score choose different families "
            f"for {len(disagreeing)} market(s): {disagreeing}. The live selector "
            "optimises the whole distribution; the board only ever asks about "
            "the posted line."
            if disagreeing else
            "The NLL selector and the threshold score agree on every ranked "
            "market, so selecting on NLL costs nothing measurable here."
        ),
        "changes_nothing": (
            "A report. The live family selector in residuals.fit_count_dispersion "
            "is untouched; switching it needs a human decision and a new "
            ".meta.json contract on the mounted volume."
        ),
    }


def _print_human(report: dict[str, Any]) -> None:
    print()
    print("Distribution sweep — RESEARCH_ONLY")
    print("=" * 78)
    for market in report["markets"]:
        if market["status"] != "RANKED":
            print(f"  {market['market']:<6} NOT RANKED — {market['reason']}")
            continue
        print(f"  {market['market']:<6} {market['n_decided_rows']:,} decided rows, "
              f"{market['n_pushes']:,} pushes")
        for family in FAMILIES:
            s = market["families"][family]
            if s["status"] != "OK":
                print(f"      {family:<8} not scored — {s['reason']}")
                continue
            print(f"      {family:<8} nll={s['nll']:.4f}  brier={s['brier']:.5f}  "
                  f"logloss={s['log_loss']:.4f}  n={s['n_scored']:,}")
        print(f"      NLL picks       : {market['live_selector_picks']}  (what runs live)")
        print(f"      Threshold picks : {market['threshold_score_picks']}")
        if market["selectors_disagree"]:
            print(f"      DISAGREE — {market.get('disagreement_note', '')}")
        print()
    print("=" * 78)
    print(f"VERDICT: {report['verdict']}")
    print(report["changes_nothing"])
    print()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Re-rank the count families by threshold score against REAL "
                    "posted lines, and report whether that disagrees with the "
                    "NLL ranking the live path uses. RESEARCH_ONLY."
    )
    ap.add_argument("--from-file", required=True, metavar="PATH", dest="source",
                    help="CSV or Parquet with columns: "
                         + ", ".join(REQUIRED_COLUMNS))
    ap.add_argument("--out", default=str(DEFAULT_OUT), metavar="PATH",
                    help=f"Where to write the JSON report (default {DEFAULT_OUT}).")
    ap.add_argument("--json", action="store_true", dest="as_json",
                    help="Print the report instead of the table.")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)-8s %(name)s | %(message)s",
    )
    try:
        report = sweep(Path(args.source).expanduser())
    except SweepError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.as_json:
        print(json.dumps(report, indent=2))
    else:
        _print_human(report)
        print(f"Written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
