"""
src/ingestion/pickem.py — read every configured pick'em board, once.

THE BATCH, AND NOTHING ELSE. Each source's client owns its own quirks; this
owns only the loop, the isolation and the archive. One board failing must not
cost the others: that is the same defect that took the slate down on
2026-10-18, when a restructured PropLine payload escaped step [3] and ended
the run, and it is not being rebuilt here one layer over.

A SOURCE THAT IS NOT CONFIGURED IS NOT A FAILURE. This repository ships no
board URLs and no field paths, because nobody here has seen a live response;
`NOT_CONFIGURED` is the honest state and `pull_pickem_boards` reports it
without raising, so a scheduled capture on an unconfigured deployment logs
what to fill in instead of paging somebody.

RESEARCH_ONLY. Captures posted lines and archives them with a timestamp.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.ingestion.pickem_schema import (
    STATUS_NOT_CONFIGURED,
    STATUS_UNAVAILABLE,
    PickemSnapshot,
)

logger = logging.getLogger(__name__)

#: Every source with a client. PrizePicks is deliberately absent: its board
#: was expected behind a bot wall, `docs/pickem_props.md` bans getting past
#: one, and its lines already arrive through PropLine's `DFS_BOOKS`.
SOURCES = ("underdog", "sleeper")


def _puller(source: str) -> Any:
    if source == "underdog":
        from src.ingestion.underdog_props import pull_underdog_board

        return pull_underdog_board
    if source == "sleeper":
        from src.ingestion.sleeper_props import pull_sleeper_board

        return pull_sleeper_board
    raise ValueError(f"no pick'em client for {source!r}; have {SOURCES}")


def pull_pickem_boards(
    sources: tuple[str, ...] | list[str] | None = None,
    *,
    config_path: str | None = None,
    archive: bool = True,
    store_root: str | Path | None = None,
    from_file: str | Path | None = None,
) -> list[PickemSnapshot]:
    """
    One snapshot per source, isolated from each other and archived.

    `from_file` applies to a single named source and is how a mapping is
    checked against a recorded response before anyone trusts it on the wire.
    """
    wanted = tuple(sources or SOURCES)
    if from_file is not None and len(wanted) != 1:
        raise ValueError(
            "--from-file reads one recorded response, so name exactly one "
            f"source; got {wanted}"
        )

    store = None
    if archive:
        from src.ingestion.pickem_store import PickemResearchStore

        store = PickemResearchStore.from_env(store_root)

    snapshots: list[PickemSnapshot] = []
    for source in wanted:
        try:
            snapshot = _puller(source)(
                config_path=config_path, from_file=from_file
            )
        except Exception as exc:  # noqa: BLE001 — one board, isolated
            logger.exception("pick'em capture %s raised: %s", source, exc)
            snapshots.append(PickemSnapshot(
                source=source,
                status=STATUS_UNAVAILABLE,
                message=f"the capture itself raised: {type(exc).__name__}: {exc}",
            ))
            continue

        if store is not None and snapshot.status != STATUS_NOT_CONFIGURED:
            # An unconfigured source is archived nowhere: there is no board
            # response to keep, and a directory of empty snapshots would make
            # the series look read-but-dark rather than never-set-up.
            try:
                store.write(snapshot, payload=None)
            except Exception as exc:  # noqa: BLE001 — the read still counts
                logger.error(
                    "pick'em snapshot for %s was captured but not archived: %s",
                    source, exc,
                )
        snapshots.append(snapshot)

    logger.info(
        "pick'em capture: %s",
        {s.source: f"{s.status}/{len(s.lines)}" for s in snapshots},
    )
    return snapshots


def pickem_prop_line_rows(**kwargs: Any) -> list[Any]:
    """
    Every captured line in the pipeline's own row shape, pick'em-flagged.

    This is the seam a future `main.ingest_prop_lines` would use. It is NOT
    wired into the slate yet, on purpose: with every source NOT_CONFIGURED it
    would add a step that can only report nothing, and the slate already has
    one source it cannot reach.
    """
    rows: list[Any] = []
    for snapshot in pull_pickem_boards(**kwargs):
        rows.extend(line.to_prop_line_row() for line in snapshot.lines)
    return rows
