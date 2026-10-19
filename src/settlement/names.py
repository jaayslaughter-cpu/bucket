"""
src/settlement/names.py — one way to compare two player names.

ONE RULE, ONE READER. `runner._match_player` folds case and strips every
non-alphanumeric character before comparing, so "P.J. Washington" and
"PJ Washington" are the same player. `espn_boxscore.select_event_by_players`
compares names too, and its first version used a weaker rule -- strip and
lower, punctuation intact. Two normalisers meant the SELECTOR could fail to
recognise a player the MATCHER would have matched, so a punctuation-only
difference in one name could make the whole event unidentifiable and leave a
night's props PENDING for a reason nothing logged.

Same defect shape as `Retry-After`, the DFS pricing rule and the
calibration-report path: copies agree until one is changed.
"""

from __future__ import annotations


def normalise_player_name(name: str) -> str:
    """Casefold and drop everything that is not alphanumeric."""
    return "".join(ch for ch in str(name).casefold() if ch.isalnum())
