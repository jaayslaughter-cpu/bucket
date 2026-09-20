"""Team Elo ratings from final scores.

WHY THIS EXISTS: nothing else in the feature set carries opponent quality.
`OPPONENT_ABBREVIATION` is a bare label, so a model can only learn "games
against DEN tend to look like this" — it cannot tell a 60-win Denver from
a lottery Denver two seasons later. Opponent strength drives pace, blowout
risk, and therefore minutes, which drives every counting-stat prop.

Two details make this more than a naive win/loss rating:

**Margin-of-victory damping.** A rating that simply rewards big wins lets
strong teams inflate without bound, because blowouts correlate with the
rating gap that produced them. The multiplier below shrinks the update as
the winner's pre-game edge grows, so beating a much weaker team by 30
moves the needle far less than the same margin against a peer.

**Offseason regression.** Rosters turn over, so carrying a rating intact
across a summer overstates what is known. Each new season pulls every team
part-way back to the league mean.

LEAKAGE: only ``elo_pre`` may ever be used as a feature. It is the rating
as it stood before the game was played. ``elo_post`` incorporates the
result and is a postgame field — using it to predict that same game is
circular. ``attach_elo_features`` exposes only pre-game columns for this
reason.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)

BASE_RATING = 1500.0
LEAGUE_MEAN = 1505.0


@dataclass(frozen=True)
class EloConfig:
    """Tunable Elo parameters. Defaults follow the widely published NBA values.

    These are conventional starting points, not values fitted against this
    repository's data. Treat them as a baseline to tune, not as estimates.
    """

    k_factor: float = 20.0
    home_advantage: float = 100.0
    offseason_regression: float = 0.75  # weight kept on last season's rating
    league_mean: float = LEAGUE_MEAN
    base_rating: float = BASE_RATING


def expected_score(rating: float, opponent_rating: float, home_advantage: float = 0.0) -> float:
    """Win probability implied by a rating gap, before the game is played."""
    gap = (rating + home_advantage) - opponent_rating
    return 1.0 / (1.0 + 10.0 ** (-gap / 400.0))


def margin_multiplier(margin: float, winner_rating_edge: float) -> float:
    """Damp the rating update by margin of victory and by the winner's edge.

    The ``0.006 * winner_rating_edge`` term in the denominator is the part
    that matters: without it, a favourite winning big gains as much as an
    underdog doing the same, and ratings drift upward on games that were
    never in doubt.
    """
    return ((abs(float(margin)) + 3.0) ** 0.8) / (7.5 + 0.006 * float(winner_rating_edge))


def _normalize_team_games(team_games: pd.DataFrame) -> pd.DataFrame:
    """Accept either the DB column names or the panel's, return one schema."""
    aliases = {
        "nba_game_id": "game_id",
        "GAME_ID": "game_id",
        "game_date": "game_date",
        "GAME_DATE": "game_date",
        "team_abbr": "team",
        "TEAM_ABBREVIATION": "team",
        "opponent_abbr": "opponent",
        "OPPONENT_ABBREVIATION": "opponent",
        "is_home": "is_home",
        "IS_HOME": "is_home",
        "points": "points",
        "PTS": "points",
        "season": "season",
        "SEASON": "season",
        "is_neutral_site": "is_neutral_site",
        "IS_NEUTRAL_SITE": "is_neutral_site",
    }
    work = team_games.rename(columns={k: v for k, v in aliases.items() if k in team_games.columns})

    required = {"game_id", "game_date", "team", "points"}
    missing = required - set(work.columns)
    if missing:
        raise ValueError(f"DATA_NOT_AVAILABLE: team games missing {sorted(missing)}")

    work["game_date"] = pd.to_datetime(work["game_date"])
    work["points"] = pd.to_numeric(work["points"], errors="coerce")
    if "is_home" not in work.columns:
        work["is_home"] = False
    if "is_neutral_site" not in work.columns:
        work["is_neutral_site"] = False
    if "season" not in work.columns:
        # Seasons run October to June, so a January game belongs to the
        # season that started the previous calendar year.
        work["season"] = work["game_date"].apply(
            lambda d: f"{d.year}-{str(d.year + 1)[2:]}" if d.month >= 10
            else f"{d.year - 1}-{str(d.year)[2:]}"
        )
    return work


def compute_team_elo(
    team_games: pd.DataFrame,
    config: EloConfig | None = None,
) -> pd.DataFrame:
    """
    Walk games in date order and produce one row per team-game.

    Returns game_id, team, opponent, game_date, season, elo_pre, opp_elo_pre,
    elo_diff, elo_win_probability, elo_post. Only the ``_pre`` columns and
    those derived from them are safe to use as features.

    Games missing either side's score are skipped without updating ratings —
    grading a rating off a half-known result would corrupt every later game.
    """
    config = config or EloConfig()
    work = _normalize_team_games(team_games)

    ratings: dict[str, float] = {}
    current_season: str | None = None
    rows: list[dict[str, object]] = []
    skipped = 0
    initialised: set[str] = set()

    for (_date, game_id), pair in work.sort_values(["game_date", "game_id"]).groupby(
        ["game_date", "game_id"], sort=True
    ):
        if len(pair) != 2:
            skipped += 1
            continue

        season = str(pair["season"].iloc[0])
        if current_season is not None and season != current_season:
            for team in ratings:
                ratings[team] = (
                    config.offseason_regression * ratings[team]
                    + (1.0 - config.offseason_regression) * config.league_mean
                )
            logger.info("Season %s: regressed %d ratings toward the mean", season, len(ratings))
        current_season = season

        row_a, row_b = pair.iloc[0], pair.iloc[1]
        if pd.isna(row_a["points"]) or pd.isna(row_b["points"]):
            skipped += 1
            continue

        for row in (row_a, row_b):
            team = str(row["team"])
            if team not in ratings:
                ratings[team] = config.base_rating
                initialised.add(team)

        team_a, team_b = str(row_a["team"]), str(row_b["team"])
        pre_a, pre_b = ratings[team_a], ratings[team_b]

        neutral = bool(row_a.get("is_neutral_site", False)) or bool(row_b.get("is_neutral_site", False))
        if neutral:
            hca_a = hca_b = 0.0
        else:
            hca_a = config.home_advantage if bool(row_a.get("is_home", False)) else 0.0
            hca_b = config.home_advantage if bool(row_b.get("is_home", False)) else 0.0

        expected_a = expected_score(pre_a + hca_a, pre_b + hca_b)
        score_a = float(row_a["points"])
        score_b = float(row_b["points"])
        actual_a = 1.0 if score_a > score_b else 0.0 if score_a < score_b else 0.5

        winner_edge = abs((pre_a + hca_a) - (pre_b + hca_b))
        if actual_a == 0.5:
            multiplier = 1.0
        else:
            winner_pre = (pre_a + hca_a) if actual_a == 1.0 else (pre_b + hca_b)
            loser_pre = (pre_b + hca_b) if actual_a == 1.0 else (pre_a + hca_a)
            winner_edge = winner_pre - loser_pre
            multiplier = margin_multiplier(score_a - score_b, winner_edge)

        shift = config.k_factor * (actual_a - expected_a) * multiplier
        post_a, post_b = pre_a + shift, pre_b - shift
        ratings[team_a], ratings[team_b] = post_a, post_b

        # Both sides' home advantage must enter the probability. Applying
        # only the team's own leaves every away row overstated, which shows
        # up as a uniform gap between predicted and observed win rates.
        for team, opponent, pre, opp_pre, post, hca, opp_hca in (
            (team_a, team_b, pre_a, pre_b, post_a, hca_a, hca_b),
            (team_b, team_a, pre_b, pre_a, post_b, hca_b, hca_a),
        ):
            rows.append(
                {
                    "game_id": str(game_id),
                    "game_date": pd.Timestamp(_date),
                    "season": season,
                    "team": team,
                    "opponent": opponent,
                    "elo_pre": round(pre, 4),
                    "opp_elo_pre": round(opp_pre, 4),
                    "elo_diff": round(pre - opp_pre, 4),
                    "elo_win_probability": round(expected_score(pre + hca, opp_pre + opp_hca), 6),
                    "elo_post": round(post, 4),
                }
            )

    if skipped:
        logger.warning(
            "Elo skipped %d game(s) without exactly two scored rows — ratings unchanged for them",
            skipped,
        )
    logger.info(
        "Elo computed for %d team-games across %d teams (%d seeded at %.0f)",
        len(rows), len(ratings), len(initialised), config.base_rating,
    )
    return pd.DataFrame(rows)


def attach_elo_features(panel: pd.DataFrame, elo_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Join pre-game Elo onto a player panel by (GAME_ID, TEAM_ABBREVIATION).

    Deliberately exposes only pre-game columns. Unmatched rows keep NaN and
    are counted in the log rather than being filled — a fabricated rating
    would read as a real measurement.
    """
    if elo_frame.empty:
        logger.warning("Elo frame is empty — no team-strength features attached")
        return panel.copy()

    required = {"GAME_ID", "TEAM_ABBREVIATION"}
    if not required.issubset(panel.columns):
        raise ValueError(f"DATA_NOT_AVAILABLE: panel missing {sorted(required - set(panel.columns))}")

    keep = ["game_id", "team", "elo_pre", "opp_elo_pre", "elo_diff", "elo_win_probability"]
    lookup = elo_frame[keep].rename(
        columns={
            "game_id": "GAME_ID",
            "team": "TEAM_ABBREVIATION",
            "elo_pre": "TEAM_ELO_PRE",
            "opp_elo_pre": "OPP_ELO_PRE",
            "elo_diff": "ELO_DIFF",
            "elo_win_probability": "ELO_WIN_PROB",
        }
    )

    out = panel.copy()
    out["GAME_ID"] = out["GAME_ID"].astype(str)
    lookup["GAME_ID"] = lookup["GAME_ID"].astype(str)
    out = out.merge(lookup, on=["GAME_ID", "TEAM_ABBREVIATION"], how="left")

    unmatched = int(out["TEAM_ELO_PRE"].isna().sum())
    if unmatched:
        logger.warning(
            "%d of %d panel rows had no Elo match — left null, not filled. "
            "Usually a team-abbreviation or game-id format mismatch.",
            unmatched, len(out),
        )
    return out
