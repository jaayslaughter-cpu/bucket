"""
scripts/audit_leakage.py — prove the panel and its splits cannot see the future.

Six checks. Each one is a way a model can post excellent validation numbers
that mean nothing, and each fails loudly rather than warning.

  1 LOOKAHEAD FLAG   Every row's features must be built from games strictly
                     before its own. This is build_feature_matrix's own
                     assertion, re-run here so the audit is self-contained.

  2 DELETION TEST    The strongest one. Rebuild the features from a panel with
                     every game from date T onward deleted, and compare the
                     surviving rows to the full build. A feature that peeks at
                     later games will move. Nothing else catches an as-of mean
                     that was quietly computed season-wide.

  3 SPLIT ORDERING   No validation row may be dated on or before train_end.

  4 ROW OVERLAP      No (player, game) may appear in both sides of a split.

  5 GAME STRADDLE    No single game may have rows on both sides of a cutoff.
                     Teammates share a game state, so a game split down the
                     middle leaks it.

  6 TARGET GIVEAWAY  No feature THE MODEL ACTUALLY USES may separate the
                     same-game label almost perfectly. Scored over
                     default_feature_cols, not over every column in the panel:
                     the panel necessarily carries this game's PTS, FGM and
                     MIN, because that is where the labels come from, and
                     scoring those measures nothing but the arithmetic.

  7 POSTGAME COLUMN  No market's feature list may name a column that is only
                     known after tip-off. This is the check that makes 6
                     meaningful -- together they say the raw stats are in the
                     panel and no model can reach them.

RESEARCH ONLY.

Usage:
    python -m scripts.audit_leakage --panel data/external/training_pack/panel.parquet
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("audit_leakage")

# Columns that are identifiers or labels rather than features.
NON_FEATURE = {
    "PLAYER_ID", "PLAYER_NAME", "GAME_ID", "GAME_DATE", "SEASON",
    "TEAM_ABBREVIATION", "OPPONENT_ABBREVIATION", "LAST_INCLUDED_GAME_DATE",
    "FEATURE_SCHEMA_VERSION", "SOURCE", "GAME_TYPE", "DNP_COMMENT",
    "IS_REGULAR_SEASON", "RESEARCH_LINE", "over_hit", "line_type",
}


class LeakageFound(AssertionError):
    """Raised when a check fails. Never downgraded to a warning."""


def _auc(y: np.ndarray, x: np.ndarray) -> float:
    """Rank AUC, NaN-safe, direction-agnostic (returns max(auc, 1-auc))."""
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 50:
        return float("nan")
    y, x = y[ok], x[ok]
    pos, neg = y == 1, y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    ranks = pd.Series(x).rank().to_numpy()
    auc = (ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum())
    return float(max(auc, 1.0 - auc))


def check_lookahead_flag(panel: pd.DataFrame) -> str:
    from src.features.builder import assert_no_lookahead

    assert_no_lookahead(panel)
    return f"every one of {len(panel):,} rows is built from strictly earlier games"


def check_deletion(panel: pd.DataFrame, raw_builder, cut_quantile: float = 0.7) -> str:
    """Rebuild with later games deleted; nothing earlier may move."""
    cut = panel["GAME_DATE"].quantile(cut_quantile)
    full, truncated = raw_builder(None), raw_builder(cut)

    key = ["PLAYER_ID", "GAME_ID"]
    numeric = [
        c for c in full.select_dtypes(include=[np.number]).columns
        if c not in NON_FEATURE
    ]
    joined = full.merge(truncated[key + numeric], on=key, suffixes=("_f", "_t"))
    if joined.empty:
        raise LeakageFound("deletion test compared zero rows — the rebuild failed")

    moved: list[tuple[str, int, float]] = []
    for col in numeric:
        a, b = joined[f"{col}_f"], joined[f"{col}_t"]
        both = a.notna() & b.notna()
        if not both.any():
            continue
        delta = (a[both] - b[both]).abs()
        n = int((delta > 1e-9).sum())
        if n:
            moved.append((col, n, float(delta.max())))
    if moved:
        worst = sorted(moved, key=lambda t: -t[1])[:6]
        lines = "\n".join(f"      {c}: {n} rows moved, max |delta| {d:.6g}"
                          for c, n, d in worst)
        raise LeakageFound(
            f"{len(moved)} feature(s) changed when games from "
            f"{cut.date()} onward were deleted, so they were reading them:\n{lines}"
        )
    return (f"{len(joined):,} rows x {len(numeric)} features unchanged after "
            f"deleting every game from {cut.date()}")


def check_split(panel: pd.DataFrame, train_end: str, validation_end: str) -> str:
    from src.models.walk_forward import fixed_cutoff_split

    work = panel.sort_values("GAME_DATE").reset_index(drop=True)
    split = fixed_cutoff_split(work, train_end=train_end, validation_end=validation_end)
    train, val = work.loc[split.train_idx], work.loc[split.validation_idx]
    if train.empty or val.empty:
        raise LeakageFound(f"split produced an empty side ({len(train)}/{len(val)})")

    cutoff = pd.Timestamp(train_end)
    late = train["GAME_DATE"] > cutoff
    if late.any():
        raise LeakageFound(f"{int(late.sum())} training rows are after train_end")
    early = val["GAME_DATE"] <= cutoff
    if early.any():
        raise LeakageFound(f"{int(early.sum())} validation rows are on or before train_end")
    return (f"train {train['GAME_DATE'].min().date()}..{train['GAME_DATE'].max().date()} "
            f"({len(train):,}) then validation "
            f"{val['GAME_DATE'].min().date()}..{val['GAME_DATE'].max().date()} "
            f"({len(val):,})")


def check_row_overlap(panel: pd.DataFrame, train_end: str, validation_end: str) -> str:
    from src.models.walk_forward import fixed_cutoff_split

    work = panel.sort_values("GAME_DATE").reset_index(drop=True)
    split = fixed_cutoff_split(work, train_end=train_end, validation_end=validation_end)
    ident = work["PLAYER_ID"].astype(str) + "@" + work["GAME_ID"].astype(str)
    shared = set(ident.loc[split.train_idx]) & set(ident.loc[split.validation_idx])
    if shared:
        raise LeakageFound(
            f"{len(shared)} (player, game) pairs appear on BOTH sides, e.g. "
            f"{sorted(shared)[:3]}"
        )
    dupes = int(ident.duplicated().sum())
    if dupes:
        raise LeakageFound(f"{dupes} duplicate (player, game) rows in the panel")
    return f"no (player, game) on both sides; no duplicates in {len(ident):,} rows"


def check_game_straddle(panel: pd.DataFrame, train_end: str) -> str:
    """Teammates share a game state, so a game split across the cutoff leaks it."""
    cutoff = pd.Timestamp(train_end)
    side = np.where(panel["GAME_DATE"] <= cutoff, "train", "val")
    per_game = pd.DataFrame({"GAME_ID": panel["GAME_ID"].astype(str), "side": side})
    counts = per_game.groupby("GAME_ID")["side"].nunique()
    straddling = counts[counts > 1]
    if len(straddling):
        raise LeakageFound(
            f"{len(straddling)} game(s) have rows on both sides of {train_end}, "
            f"e.g. {list(straddling.index[:3])}"
        )
    return f"none of {counts.size:,} games straddles {train_end}"


def check_postgame_columns(panel: pd.DataFrame, markets: list[str]) -> str:
    """No feature list may name a column known only after tip-off."""
    from src.models.compare import POSTGAME_ONLY_COLS
    from src.models.labels import default_feature_cols

    offenders: dict[str, list[str]] = {}
    for market in markets:
        named = sorted(set(default_feature_cols(market)) & POSTGAME_ONLY_COLS)
        if named:
            offenders[market] = named
    if offenders:
        raise LeakageFound(
            "feature list(s) name same-game outcome columns: "
            + "; ".join(f"{m}: {c}" for m, c in offenders.items())
        )
    in_panel = sorted(set(panel.columns) & POSTGAME_ONLY_COLS)
    return (f"{len(in_panel)} postgame column(s) sit in the panel ({', '.join(in_panel[:6])}"
            f"{'...' if len(in_panel) > 6 else ''}) and no market's feature list names one")


def check_target_giveaway(
    panel: pd.DataFrame, market: str, threshold: float = 0.95
) -> str:
    from src.models.labels import attach_research_over_labels, default_feature_cols

    work = attach_research_over_labels(panel, stat=market)
    work = work[work["over_hit"].notna()]
    y = work["over_hit"].astype(float).to_numpy()
    numeric = set(work.select_dtypes(include=[np.number]).columns)
    feats = [
        c for c in default_feature_cols(market)
        if c in numeric and c not in NON_FEATURE and c != market
    ]
    if not feats:
        raise LeakageFound(f"{market}: none of its features are present in the panel")
    scored = [(c, _auc(y, work[c].to_numpy(dtype=float))) for c in feats]
    scored = [(c, a) for c, a in scored if np.isfinite(a)]
    hot = [(c, a) for c, a in scored if a >= threshold]
    if hot:
        lines = "\n".join(f"      {c}: AUC {a:.4f}" for c, a in sorted(hot, key=lambda t: -t[1]))
        raise LeakageFound(
            f"{len(hot)} feature(s) separate the {market} label almost perfectly, "
            f"which usually means the outcome under another name:\n{lines}"
        )
    top = sorted(scored, key=lambda t: -t[1])[:5]
    return (f"{market}: strongest single feature is {top[0][0]} at AUC {top[0][1]:.4f} "
            f"(threshold {threshold}); next " +
            ", ".join(f"{c} {a:.3f}" for c, a in top[1:4]))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--panel", default="data/external/training_pack/panel.parquet")
    ap.add_argument("--pack", default="data/external/training_pack")
    ap.add_argument("--train-end", default="2025-10-01")
    ap.add_argument("--validation-end", default="2026-04-12")
    ap.add_argument("--markets", default="PTS,REB,AST")
    ap.add_argument("--deletion-seasons", default="2024-25,2025-26",
                    help="Seasons to rebuild for the deletion test. The whole "
                         "panel would be rebuilt twice, which is slow.")
    ap.add_argument("--skip-deletion", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    panel = pd.read_parquet(args.panel)
    panel["GAME_DATE"] = pd.to_datetime(panel["GAME_DATE"])
    markets = [m.strip().upper() for m in args.markets.split(",") if m.strip()]

    print(f"Panel: {len(panel):,} rows x {panel.shape[1]} columns, "
          f"{panel['GAME_DATE'].min().date()} -> {panel['GAME_DATE'].max().date()}\n")

    checks: list[tuple[str, callable]] = [
        ("1 lookahead flag", lambda: check_lookahead_flag(panel)),
        ("3 split ordering", lambda: check_split(panel, args.train_end, args.validation_end)),
        ("4 row overlap", lambda: check_row_overlap(panel, args.train_end, args.validation_end)),
        ("5 game straddle", lambda: check_game_straddle(panel, args.train_end)),
    ]
    for market in markets:
        checks.append((f"6 target giveaway [{market}]",
                       lambda m=market: check_target_giveaway(panel, m)))
    checks.append(("7 postgame column", lambda: check_postgame_columns(panel, markets)))

    if not args.skip_deletion:
        seasons = [s.strip() for s in args.deletion_seasons.split(",") if s.strip()]
        builder = _make_rebuilder(Path(args.pack), seasons)
        checks.insert(1, ("2 deletion test", lambda: check_deletion(
            panel[panel["SEASON"].isin(seasons)], builder)))

    failures = 0
    for name, fn in checks:
        try:
            detail = fn()
        except LeakageFound as exc:
            failures += 1
            print(f"  FAIL  {name}\n        {exc}")
        except Exception as exc:  # noqa: BLE001 — an audit that crashes is a failed audit
            failures += 1
            print(f"  ERROR {name}\n        {type(exc).__name__}: {exc}")
        else:
            print(f"  pass  {name}  —  {detail}")

    print()
    if failures:
        print(f"{failures} check(s) FAILED. Do not train on this panel until they pass.")
        return 1
    print("All checks passed. The panel's features and splits do not read the future.")
    return 0


def _make_rebuilder(pack: Path, seasons: list[str]):
    """Return build(cut) -> feature matrix, optionally truncated at ``cut``."""
    from scripts.ingest_training_pack import build_team_games
    from src.features.builder import build_feature_matrix
    from src.ingestion.kaggle_nba import (
        describe_schema,
        load_team_crosswalk,
        normalize_player_box_scores,
    )

    raw = pd.read_csv(pack / "player_boxes" / "PlayerStatistics_2018_to_2026.csv",
                      low_memory=False)
    crosswalk = load_team_crosswalk(pack / "reference" / "TeamHistories.csv")
    base = normalize_player_box_scores(raw, describe_schema(raw), team_crosswalk=crosswalk)
    base = base[base["IS_REGULAR_SEASON"] & base["SEASON"].isin(seasons)]
    minutes = pd.to_numeric(base["MIN"], errors="coerce")
    base = base[minutes.notna() & (minutes > 0)].copy()
    base["GAME_ID"] = base["GAME_ID"].astype(str)

    bdb_team = pack / "market" / "bigdataball_team_game_stats.csv"
    bdb = pd.read_csv(bdb_team) if bdb_team.exists() else None
    lines_path = pack / "market" / "bigdataball_game_market_lines.csv"

    def build(cut):
        work = base if cut is None else base[base["GAME_DATE"] < cut]
        work = work.copy()
        team_games = build_team_games(work, bdb)
        lines = None
        if lines_path.exists():
            lines = pd.read_csv(lines_path)
            lines["nba_game_id"] = lines["nba_game_id"].astype(str)
            if cut is not None:
                lines = lines[pd.to_datetime(lines["game_date"]) < cut]
        return build_feature_matrix(work, team_games=team_games, market_lines=lines)

    return build


if __name__ == "__main__":
    raise SystemExit(main())
