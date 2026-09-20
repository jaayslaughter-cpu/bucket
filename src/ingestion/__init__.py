"""Ingestion layer.

Only modules that exist in this repository are exported. The wider PropIQ
tree carries additional loaders (box scores, play-by-play, pick'em boards,
OddsPapi, the player id crosswalk); re-export them here as they land, so a
missing module can never be mistaken for a silently empty data source.
"""

from src.ingestion.bigdataball import load_bigdataball_workbook, load_team_map

__all__ = [
    "load_bigdataball_workbook",
    "load_team_map",
]
