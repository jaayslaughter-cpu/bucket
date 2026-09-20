"""Ingestion layer: box scores, PBP aggregation, Sleeper, ID crosswalk."""

from src.ingestion.bbs_client import BbsClient, BbsClientConfig, attach_bbs_injury_features
from src.ingestion.boxscores import BoxScoreLoadConfig, load_player_game_logs
from src.ingestion.oddspapi_client import (
    HISTORICAL_COVERAGE_START,
    OddsPapiClient,
    OddsPapiConfig,
    OddsPapiError,
    default_snapshot_dir,
)
from src.ingestion.nba_official_archive import (
    OfficialArchiveLoadConfig,
    load_official_player_game_logs,
    resolve_official_archive_zip,
)
from src.ingestion.databallr_pdf import (
    DataballrPdfConfig,
    list_databallr_pdf_coverage,
    load_databallr_pdf_snapshots,
)
from src.ingestion.databallr_views_paste import (
    DataballrViewPasteConfig,
    load_all_databallr_view_pastes,
    load_databallr_view_paste,
    parse_databallr_view_paste,
)
from src.ingestion.databallr_zts_paste import (
    DataballrZtsPasteConfig,
    load_databallr_zts_paste,
    parse_databallr_zts_paste,
)
from src.ingestion.id_crosswalk import PlayerIdCrosswalk, PlayerNameRecord
from src.ingestion.pbp_boxscores import (
    ModernPbpConfig,
    PbpLoadConfig,
    list_available_pbp_seasons,
    list_modern_pbp_years,
    load_modern_player_games,
    load_pbp_boxscores,
)
from src.ingestion.nba_cdn_pbp import NbaCdnPbpClient, NbaCdnPbpConfig
from src.ingestion.espn_pbp import EspnPbpClient, EspnPbpConfig
from src.ingestion.nba_stats_pbp import NbaStatsPbpClient, NbaStatsPbpConfig
from src.ingestion.pbp_schema import UnifiedPbpEvent, UnifiedPbpGame
from src.ingestion.pbp_transformer import (
    transform_espn_summary,
    transform_nba_cdn_playbyplay,
    transform_nba_stats_playbyplay,
    unified_events_to_frame,
)
from src.ingestion.six_factor import load_partial_ots_from_shotquality, load_six_factor_csv
from src.ingestion.shotquality import ShotQualityLoadConfig, load_shotquality_off
from src.ingestion.sleeper import SleeperClient
from src.ingestion.pickem_schema import PickemPropLine, PickemSnapshot
from src.ingestion.pickem import pull_pickem_boards
from src.ingestion.pickem_store import PickemResearchStore, PickemStoreConfig
from src.ingestion.underdog_props import UnderdogPropsClient, UnderdogPropsConfig
from src.ingestion.prizepicks_props import PrizePicksPropsClient, PrizePicksPropsConfig
from src.ingestion.sleeper_props import SleeperPropsClient, SleeperPropsConfig
from src.ingestion.basketball_index import (
    BasketballIndexLoadConfig,
    BasketballIndexSnapshot,
    load_basketball_index_snapshots,
    load_bi_export_file,
)
from src.ingestion.bigdataball import load_bigdataball_workbook, load_team_map

__all__ = [
    "BbsClient",
    "BbsClientConfig",
    "attach_bbs_injury_features",
    "BoxScoreLoadConfig",
    "load_player_game_logs",
    "HISTORICAL_COVERAGE_START",
    "OddsPapiClient",
    "OddsPapiConfig",
    "OddsPapiError",
    "default_snapshot_dir",
    "OfficialArchiveLoadConfig",
    "load_official_player_game_logs",
    "resolve_official_archive_zip",
    "DataballrPdfConfig",
    "list_databallr_pdf_coverage",
    "load_databallr_pdf_snapshots",
    "DataballrZtsPasteConfig",
    "load_databallr_zts_paste",
    "parse_databallr_zts_paste",
    "DataballrViewPasteConfig",
    "load_all_databallr_view_pastes",
    "load_databallr_view_paste",
    "parse_databallr_view_paste",
    "ModernPbpConfig",
    "PbpLoadConfig",
    "list_available_pbp_seasons",
    "list_modern_pbp_years",
    "load_modern_player_games",
    "load_pbp_boxscores",
    "NbaCdnPbpClient",
    "NbaCdnPbpConfig",
    "EspnPbpClient",
    "EspnPbpConfig",
    "NbaStatsPbpClient",
    "NbaStatsPbpConfig",
    "UnifiedPbpEvent",
    "UnifiedPbpGame",
    "transform_espn_summary",
    "transform_nba_cdn_playbyplay",
    "transform_nba_stats_playbyplay",
    "unified_events_to_frame",
    "PlayerIdCrosswalk",
    "PlayerNameRecord",
    "SleeperClient",
    "PickemPropLine",
    "PickemSnapshot",
    "pull_pickem_boards",
    "PickemResearchStore",
    "PickemStoreConfig",
    "UnderdogPropsClient",
    "UnderdogPropsConfig",
    "PrizePicksPropsClient",
    "PrizePicksPropsConfig",
    "SleeperPropsClient",
    "SleeperPropsConfig",
    "ShotQualityLoadConfig",
    "load_shotquality_off",
    "load_six_factor_csv",
    "load_partial_ots_from_shotquality",
    "BasketballIndexLoadConfig",
    "BasketballIndexSnapshot",
    "load_basketball_index_snapshots",
    "load_bi_export_file",
    "load_bigdataball_workbook",
    "load_team_map",
]
