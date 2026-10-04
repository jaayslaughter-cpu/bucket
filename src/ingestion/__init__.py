"""Ingestion layer.

Only modules that exist in this repository are exported, so a missing module
can never be mistaken for a silently empty data source. Re-export a loader
here as it lands.

Deliberately not listed as "coming": a loader named in a docstring and absent
from the tree reads as a plan, and this project has been bitten twice by that
— OddsPapi sat in the decision board's source precedence with no client at
all, and `id_crosswalk.py` is still named as the fix for name-format mismatch
by three modules while not existing.
"""

from src.ingestion.basketball_reference import (
    SR_ATTRIBUTION,
    BasketballReferenceError,
    SeasonAggregateLeakageError,
    attach_prior_season_features,
    describe_sr_csv,
    describe_sr_table,
    prior_season_features,
    read_sr_season_csv,
    season_totals,
    team_splits,
)
from src.ingestion.bigdataball import load_bigdataball_workbook, load_team_map
from src.ingestion.boxscores import (
    BoxScoreFetchError,
    BoxScoreLoadConfig,
    load_player_game_logs,
)
from src.ingestion.propline_history import (
    bets_to_grade_payload,
    clv_from_closing,
    describe_resolution_payload,
    normalize_closing_odds,
    normalize_clv_grade,
    plan_history_window,
)

__all__ = [
    "BasketballReferenceError",
    "BoxScoreFetchError",
    "BoxScoreLoadConfig",
    "SR_ATTRIBUTION",
    "SeasonAggregateLeakageError",
    "attach_prior_season_features",
    "bets_to_grade_payload",
    "clv_from_closing",
    "describe_resolution_payload",
    "describe_sr_csv",
    "describe_sr_table",
    "load_bigdataball_workbook",
    "load_player_game_logs",
    "load_team_map",
    "normalize_closing_odds",
    "normalize_clv_grade",
    "plan_history_window",
    "prior_season_features",
    "read_sr_season_csv",
    "season_totals",
    "team_splits",
]
