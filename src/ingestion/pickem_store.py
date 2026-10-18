"""
src/ingestion/pickem_store.py — keep every board snapshot, keyed by when we read it.

WHY A STORE AND NOT A TABLE. These are research captures of a one-way price.
They do not belong in `prop_line_snapshots` next to two-way sportsbook quotes
where something could later read them as de-viggable, and the whole value of
a pick'em capture is the SERIES -- the same board read at 10:00, 13:00 and
18:00 -- which is a file per read, not a row that gets upserted over.

LAYOUT, as `docs/pickem_props.md` specifies it:

    data/external/pickem/snapshots/{source}/{YYYY}/{MM}/{source}_{stamp}.json
    data/external/pickem/snapshots/lines_index.csv

The index is APPEND-ONLY and is written only when a snapshot carries lines. A
NOT_CONFIGURED or refused read still writes its snapshot file -- a dark board
is evidence, and the gap in the series is the finding -- but it adds no index
row, because the index is "lines we hold" and an empty read holds none.

NOTHING HERE IS A SECRET. Board responses are public postings. The writer
still refuses a payload carrying anything that looks like a credential, so a
future board that starts echoing a token cannot quietly put it on disk.

RESEARCH_ONLY. Writes JSON and a CSV.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.ingestion.pickem_schema import PickemSnapshot

logger = logging.getLogger(__name__)

DEFAULT_ROOT = "data/external/pickem/snapshots"
ENV_ROOT = "PROPIQ_PICKEM_STORE"
INDEX_NAME = "lines_index.csv"

INDEX_COLUMNS = (
    "ingested_at_utc",
    "source",
    "player_name",
    "market",
    "line",
    "payout_multiplier",
    "captured_at_utc",
    "team_abbr",
    "opponent_abbr",
    "game_date",
    "snapshot_file",
)

#: Keys whose presence in a board payload stops it being written. Public
#: boards send none of these; a board that starts sending one is a board we
#: should not be archiving verbatim.
SECRET_KEY_HINTS = (
    "authorization", "api_key", "apikey", "access_token", "refresh_token",
    "password", "secret", "session_token", "bearer", "cookie",
)


def _stamp(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


def find_secret_like_keys(payload: Any, *, limit: int = 5) -> list[str]:
    """Any key in the payload that looks like a credential, by path."""
    found: list[str] = []

    def walk(node: Any, path: str) -> None:
        if len(found) >= limit:
            return
        if isinstance(node, dict):
            for key, value in node.items():
                here = f"{path}.{key}" if path else str(key)
                if str(key).strip().lower().replace("-", "_") in SECRET_KEY_HINTS:
                    found.append(here)
                walk(value, here)
        elif isinstance(node, (list, tuple)):
            for index, value in enumerate(node[:20]):
                walk(value, f"{path}.{index}" if path else str(index))

    walk(payload, "")
    return found


@dataclass
class PickemResearchStore:
    """The snapshot archive on disk."""

    root: Path

    @classmethod
    def from_env(cls, root: str | Path | None = None) -> "PickemResearchStore":
        import os

        return cls(Path(root or os.environ.get(ENV_ROOT) or DEFAULT_ROOT))

    @property
    def index_path(self) -> Path:
        return self.root / INDEX_NAME

    def snapshot_path(self, snapshot: PickemSnapshot) -> Path:
        moment = snapshot.ingested_at_utc or datetime.now(UTC)
        return (
            self.root
            / snapshot.source
            / f"{moment.astimezone(UTC):%Y}"
            / f"{moment.astimezone(UTC):%m}"
            / f"{snapshot.source}_{_stamp(moment)}.json"
        )

    def write(
        self, snapshot: PickemSnapshot, *, payload: Any = None
    ) -> tuple[Path, int]:
        """
        Archive one snapshot. Returns ``(path, index_rows_appended)``.

        The raw payload is stored alongside the parsed lines so a mapping can
        be re-run over it later: when a board restructures, the only way to
        tell a genuinely empty night from a stale mapping is to re-parse what
        was actually received.
        """
        secrets = find_secret_like_keys(payload) if payload is not None else []
        if secrets:
            # Refuse rather than redact: a board that echoes a credential is
            # a situation for a person to look at, and a half-scrubbed
            # archive is worse than none.
            raise ValueError(
                f"refusing to archive {snapshot.source}: the payload carries "
                f"key(s) that look like a credential ({secrets}). Nothing was "
                f"written. A public board sends none of these."
            )

        path = self.snapshot_path(snapshot)
        path.parent.mkdir(parents=True, exist_ok=True)
        body: dict[str, Any] = {
            "snapshot": snapshot.as_dict(),
            "lines": [
                {
                    "player_name": line.player_name,
                    "market": line.market,
                    "line": line.line,
                    "payout_multiplier": line.payout_multiplier,
                    "captured_at_utc": (
                        line.captured_at_utc.isoformat() if line.captured_at_utc else None
                    ),
                    "team_abbr": line.team_abbr,
                    "opponent_abbr": line.opponent_abbr,
                    "game_date": line.game_date.isoformat() if line.game_date else None,
                    "board_id": line.board_id,
                    "player_id": line.player_id,
                }
                for line in snapshot.lines
            ],
            "raw_payload": payload,
        }
        path.write_text(json.dumps(body, indent=2, default=str), encoding="utf-8")

        appended = self._append_index(snapshot, path)
        logger.info(
            "pick'em snapshot %s (%s, %d line(s), %d index row(s))",
            path, snapshot.status, len(snapshot.lines), appended,
        )
        return path, appended

    def _append_index(self, snapshot: PickemSnapshot, path: Path) -> int:
        if not snapshot.lines:
            return 0
        self.root.mkdir(parents=True, exist_ok=True)
        new_file = not self.index_path.exists()
        with self.index_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if new_file:
                writer.writerow(INDEX_COLUMNS)
            for line in snapshot.lines:
                writer.writerow([
                    snapshot.ingested_at_utc.isoformat() if snapshot.ingested_at_utc else "",
                    snapshot.source,
                    line.player_name,
                    line.market,
                    line.line,
                    "" if line.payout_multiplier is None else line.payout_multiplier,
                    line.captured_at_utc.isoformat() if line.captured_at_utc else "",
                    line.team_abbr or "",
                    line.opponent_abbr or "",
                    line.game_date.isoformat() if line.game_date else "",
                    str(path),
                ])
        return len(snapshot.lines)
