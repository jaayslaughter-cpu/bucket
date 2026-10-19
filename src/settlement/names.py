"""
src/settlement/names.py — one way to compare two player names.

THIS FILE IS A RE-EXPORT, DELIBERATELY. `ingestion.id_crosswalk` already owns
the rule: decompose Unicode and drop combining marks (ć -> c, ņ -> n),
lowercase, delete dots and apostrophes so ``P.J.`` and ``PJ`` agree and
``De'Aaron`` and ``DeAaron`` agree, turn hyphens into spaces, collapse
whitespace.

THREE READERS, AND I ADDED THE THIRD. `runner._match_player` folded case and
kept alphanumerics; `espn_boxscore.select_event_by_players` first stripped and
lowercased; and this module was created to end that split -- by writing a
FOURTH rule that still kept accented letters. So the ESPN fallback could not
match ``Nikola Jokić`` in a box score to a pending ``Nikola Jokic`` prop,
which is precisely the case `id_crosswalk` exists for, and
`verify_wiring` already demonstrates it resolving. Found by cubic on PR #16,
in the commit whose message claimed to have consolidated to one reader.

Settlement imports from here rather than reaching into `ingestion` directly,
so the dependency is stated once and a future change has one place to look --
but there is no second implementation to drift.
"""

from __future__ import annotations

from src.ingestion.id_crosswalk import normalise_player_name

__all__ = ["normalise_player_name"]
