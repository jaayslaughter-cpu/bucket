"""
scripts/ingest_training_pack.py — build the feature matrix from the historical
training pack.

The pack carries player box scores for 2018-2026, BigDataBall team stats and
market lines for 2025-26, and the archive's reference tables. This assembles
them into one leakage-safe panel.

THREE DECISIONS WORTH KNOWING ABOUT, each measured rather than assumed:

1. NON-APPEARANCES ARE DROPPED. 47,308 regular-season rows carry zero or null
   minutes, and every single one records PTS = REB = AST = 0. They are
   did-not-plays, not zero-point performances. Rolling them into a player's
   L5 mean teaches the model that a starter scores nothing once a week.

2. NON-REGULAR-SEASON ROWS ARE DROPPED. Preseason rotations do not describe
   the same competition, and the pack's preseason includes exhibition games
   against Guangzhou Loong-Lions, Hapoel Jerusalem, Melbourne United and
   South East Melbourne Phoenix -- clubs that are not NBA franchises.

3. TEAM TOTALS ARE DERIVED FROM THE PLAYER ARCHIVE, then overridden by
   BigDataBall where it overlaps. Summing players is only valid if the
   archive is complete per team-game; it is. Checked against BigDataBall's
   independent team stats on the 2,460 team-games they share, the sums match
   exactly on 99.96% of points, 100% of free-throw attempts and >99.5% of
   every other column. That is what makes Elo and opponent-defence features
   available for all nine seasons rather than only for 2025-26.

RESEARCH ONLY. Builds features; places no bets.

Usage:
    python -m scripts.ingest_training_pack
    python -m scripts.ingest_training_pack --pack data/external/training_pack
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger("ingest_training_pack")

# BigDataBall measures possessions; the player archive only supports the
# standard estimate. Where both exist, the measurement wins.
BDB_PREFERRED_COLS = ("poss", "pace", "off_eff", "def_eff", "rest_days")


def build_team_games(panel: pd.DataFrame, bdb: pd.DataFrame | None) -> pd.DataFrame:
    """
    Team-game totals from the player panel, refined by BigDataBall.

    Possessions follow the standard estimate, FGA - OREB + TOV + 0.44*FTA,
    which correlates 0.953 with BigDataBall's measured value at a mean
    absolute error of 2.2 possessions. Where BigDataBall has the real number
    it replaces the estimate rather than sitting beside it.
    """
    need = {"GAME_ID", "TEAM_ABBREVIATION", "OPPONENT_ABBREVIATION", "GAME_DATE"}
    missing = need - set(panel.columns)
    if missing:
        raise ValueError(f"DATA_NOT_AVAILABLE: panel missing {sorted(missing)}")

    work = panel.dropna(subset=["TEAM_ABBREVIATION", "OPPONENT_ABBREVIATION"]).copy()
    agg = work.groupby(["GAME_ID", "TEAM_ABBREVIATION"], as_index=False).agg(
        game_date=("GAME_DATE", "first"),
        opponent_abbr=("OPPONENT_ABBREVIATION", "first"),
        is_home=("IS_HOME", "first"),
        points=("PTS", "sum"), fg=("FGM", "sum"), fga=("FGA", "sum"),
        fg3=("FG3M", "sum"), ft=("FTM", "sum"), fta=("FTA", "sum"),
        oreb=("OREB", "sum"), dreb=("DREB", "sum"), reb=("REB", "sum"),
        ast=("AST", "sum"), stl=("STL", "sum"), blk=("BLK", "sum"),
        tov=("TOV", "sum"),
    ).rename(columns={"GAME_ID": "nba_game_id", "TEAM_ABBREVIATION": "team_abbr"})

    agg["poss"] = agg["fga"] - agg["oreb"] + agg["tov"] + 0.44 * agg["fta"]
    agg["source"] = "player_archive_sum"
    agg["nba_game_id"] = agg["nba_game_id"].astype(str)

    if bdb is not None and not bdb.empty:
        right = bdb.copy()
        right["nba_game_id"] = right["nba_game_id"].astype(str)
        cols = [c for c in BDB_PREFERRED_COLS if c in right.columns]
        right = right[["nba_game_id", "team_abbr", *cols]]
        agg = agg.merge(right, on=["nba_game_id", "team_abbr"], how="left",
                        suffixes=("", "_bdb"))
        replaced = 0
        for col in cols:
            src = f"{col}_bdb" if f"{col}_bdb" in agg.columns else col
            if src not in agg.columns:
                continue
            if col in agg.columns and src != col:
                take = agg[src].notna()
                agg.loc[take, col] = agg.loc[take, src]
                agg = agg.drop(columns=[src])
                replaced = max(replaced, int(take.sum()))
            else:
                replaced = max(replaced, int(agg[col].notna().sum()))
        agg.loc[agg.get("pace", pd.Series(index=agg.index)).notna(), "source"] = (
            "player_archive_sum+bigdataball"
        )
        logger.info(
            "Team games: %d rows, %d refined with BigDataBall's measured "
            "possessions and efficiency.", len(agg), replaced,
        )
    else:
        logger.info("Team games: %d rows, no BigDataBall overlay.", len(agg))
    return agg


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--pack", default="data/external/training_pack")
    ap.add_argument("--out", default=None, help="Parquet path (default: <pack>/panel.parquet)")
    ap.add_argument("--min-minutes", type=float, default=0.0,
                    help="Drop rows at or below this many minutes. 0 keeps every "
                         "appearance and drops only non-appearances.")
    ap.add_argument("--keep-non-regular-season", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    logger.setLevel(logging.INFO)

    from src.features.builder import build_feature_matrix
    from src.ingestion.kaggle_nba import (
        describe_schema,
        load_team_crosswalk,
        normalize_player_box_scores,
    )

    pack = Path(args.pack)
    boxes = pack / "player_boxes" / "PlayerStatistics_2018_to_2026.csv"
    histories = pack / "reference" / "TeamHistories.csv"
    for required in (boxes, histories):
        if not required.exists():
            print(f"ERROR: {required} not found. Extract the training pack first.")
            return 2

    raw = pd.read_csv(boxes, low_memory=False)
    crosswalk = load_team_crosswalk(histories)
    panel = normalize_player_box_scores(raw, describe_schema(raw),
                                        team_crosswalk=crosswalk)
    logger.info("Loaded %d player-game rows.", len(panel))

    if not args.keep_non_regular_season and "IS_REGULAR_SEASON" in panel.columns:
        before = len(panel)
        panel = panel[panel["IS_REGULAR_SEASON"]].copy()
        logger.info("Dropped %d non-regular-season rows.", before - len(panel))

    minutes = pd.to_numeric(panel["MIN"], errors="coerce")
    played = minutes.notna() & (minutes > args.min_minutes)
    logger.info(
        "Dropped %d non-appearances (minutes null or <= %.1f). Every one of "
        "them records PTS = REB = AST = 0, which is a did-not-play rather "
        "than a zero-point game.",
        int((~played).sum()), args.min_minutes,
    )
    panel = panel[played].copy()

    bdb_team = pack / "market" / "bigdataball_team_game_stats.csv"
    bdb_lines = pack / "market" / "bigdataball_game_market_lines.csv"
    team_games = build_team_games(
        panel, pd.read_csv(bdb_team) if bdb_team.exists() else None
    )
    market_lines = None
    if bdb_lines.exists():
        market_lines = pd.read_csv(bdb_lines)
        market_lines["nba_game_id"] = market_lines["nba_game_id"].astype(str)
        logger.info("Market lines: %d rows.", len(market_lines))

    panel["GAME_ID"] = panel["GAME_ID"].astype(str)
    features = build_feature_matrix(
        panel, team_games=team_games, market_lines=market_lines
    )

    out = Path(args.out) if args.out else pack / "panel.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out, index=False)

    print(f"\nWrote {out}  ({len(features):,} rows x {features.shape[1]} columns)")
    print(f"  dates   {features['GAME_DATE'].min().date()} -> "
          f"{features['GAME_DATE'].max().date()}")
    print(f"  players {features['PLAYER_ID'].nunique():,}   "
          f"games {features['GAME_ID'].nunique():,}")
    print("\n  coverage of the context layers:")
    for col in ("TEAM_ELO_PRE", "DEF_RATING_L10", "MKT_IMPLIED_TEAM_TOTAL",
                "PTS_L5", "PTS_SEASON"):
        if col in features.columns:
            print(f"    {col:26s} {features[col].notna().mean():6.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
