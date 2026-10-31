"""
One definition of what an NBA game id looks like.

WHY THIS MODULE EXISTS. NBA game ids are ten characters, zero-padded: a regular
season game is ``0022500001``, a playoff game ``0042400305``. Three places in
this repository needed that fact and only one of them had it.
``ingestion/boxscores.py`` normalised on the way in; ``features/market_context``
and ``features/team_strength`` stripped whitespace and compared literally. The
cached training-pack panels carry ids eight-wide with the leading zeros gone, so
joining them against a workbook matched **0 of 1322 games** and every ``MKT_*``,
``DEF_*`` and Elo column came out 100% NaN.

WHY THAT WAS WORSE THAN A CRASH. Those columns carry 21.7%-27.5% of each
booster's gain, and XGBoost does not fail on a NaN feature -- it takes the
default branch. Measured on the real PTS booster, blanking them moves the
predicted probability by a mean of 0.14 while the run exits 0 and the logs stay
green. The two layers did warn and did leave the columns null rather than
filling a neutral value, which is the only reason it was recoverable.

THIS IS CANONICALISATION, NOT GUESSING, which is the line this project draws
everywhere else. Padding ``22500001`` to ``0022500001`` does not invent
information: the ten-wide form is what the NBA's own ids are, and
``boxscores.py`` has written them that way since the module was added. An id
that is already ten or more characters is returned untouched -- truncating a
longer one would name a DIFFERENT GAME, so it passes through and the join
simply fails to match, which is visible where a silent truncation is not.

RESEARCH_ONLY. String normalisation. Nothing here reads a market or sizes
anything.
"""

from __future__ import annotations

import pandas as pd

#: The canonical width of an NBA game id, as `ingestion/boxscores.py` has
#: always written it and as the NBA's own payloads report it.
GAME_ID_WIDTH = 10


def normalise_game_id(values: pd.Series) -> pd.Series:
    """
    Canonical ten-wide NBA game ids, as strings.

    Accepts whatever a parquet round-trip or a JSON payload produced -- ints,
    floats that were ints, padded strings, unpadded strings, surrounding
    whitespace -- and returns the ten-character form. Missing stays missing:
    a null id is not a game, and ``"0000000000"`` would be a lie about one.
    """
    text = values.astype("object").map(_one)
    return pd.Series(text, index=values.index, name=values.name)


def _one(value: object) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        if pd.isna(value):  # pandas NA / NaT
            return None
    except (TypeError, ValueError):
        pass
    # A float that is really an int: 22500001.0 -> "22500001", never
    # "22500001.0", which would pad to the wrong width and match nothing.
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    text = str(value).strip()
    if not text:
        return None
    return text.zfill(GAME_ID_WIDTH)


def normalise_one_game_id(value: object) -> str | None:
    """
    The same rule for a single id rather than a column.

    `settlement/boxscore_fetcher` holds one game id at a time, and a second
    hand-rolled `zfill` there is exactly the duplication this module exists to
    remove, so it gets a scalar door into the same function.
    """
    return _one(value)


# ---------------------------------------------------------------------------
# season type, from the game id itself
# ---------------------------------------------------------------------------
#
# An NBA game id encodes what KIND of game it is in its third character, and
# nothing else in this project's data does. `team_game_stats` has no
# season_type column and the BigDataBall workbook publishes none, so the id is
# the only signal available -- and it is a reliable one.
#
# MEASURED against the 2025-26 workbook (2,644 rows) rather than taken from
# documentation:
#
#     002  regular season   2025-10-21 .. 2026-04-12   2,460 rows
#     005  play-in          2026-04-14 .. 2026-04-17       12
#     004  playoffs         2026-04-18 .. 2026-06-13      170
#     006  NBA Cup final    2025-12-16                      2
#
# The date ranges do not overlap and they fall in the real calendar order, which
# is what makes the mapping trustworthy rather than a guess about a prefix.

SEASON_TYPE_PRESEASON = "preseason"
SEASON_TYPE_REGULAR = "regular"
SEASON_TYPE_ALL_STAR = "all_star"
SEASON_TYPE_PLAYOFFS = "playoffs"
SEASON_TYPE_PLAY_IN = "play_in"
SEASON_TYPE_CUP_FINAL = "cup_final"
SEASON_TYPE_UNKNOWN = "unknown"

#: Third character of a normalised 10-character game id -> what kind of game.
_SEASON_TYPE_BY_DIGIT = {
    "1": SEASON_TYPE_PRESEASON,
    "2": SEASON_TYPE_REGULAR,
    "3": SEASON_TYPE_ALL_STAR,
    "4": SEASON_TYPE_PLAYOFFS,
    "5": SEASON_TYPE_PLAY_IN,
    "6": SEASON_TYPE_CUP_FINAL,
}


#: Where the season-type digit sits, 1-BASED, for a SQL ``substr``. A caller
#: that filters in the database must not restate the mapping above in SQL --
#: it reads this and `REGULAR_SEASON_ID_DIGIT` instead, so there is still one
#: definition of which character means what.
SEASON_TYPE_SQL_POSITION = 3
REGULAR_SEASON_ID_DIGIT = "2"


#: THE SECOND ID NAMESPACE, and the only one. An ESPN event id is "401909881"
#: -- nine characters, with "1" where an NBA id carries its season-type digit.
#: It is not an NBA game id and `espn_schedule`'s docstring has said since it
#: was written that the two "must never be joined directly".
#:
#: WHY A PREFIX RATHER THAN A SYNTHESISED NBA ID. The container can reach
#: ESPN and cannot reach nba.com (403 from cdn, read timeout from stats, on an
#: honest User-Agent -- measured 2026-10-30), so ESPN is the only box-score
#: source an unattended deploy has. Minting an NBA-SHAPED id for those rows
#: would be the one unrecoverable mistake available here: the moment a
#: licensed or scraped ingest covers the same night, the invented id either
#: collides with a real game or sits beside it as a duplicate, and nothing
#: downstream could tell which row is the real one. A prefixed id cannot be
#: mistaken for an NBA id by anything, including a careless join.
ESPN_ID_PREFIX = "espn:"

#: An ``espn:`` row is REGULAR SEASON BY CONSTRUCTION, which is why a filter
#: can accept the whole namespace without reading a digit out of it:
#: `ingestion/espn_panel` refuses any game whose ESPN season type is not 2,
#: so a preseason or playoff game never reaches the table wearing this prefix.
#: The guarantee lives here, next to the prefix, because the SQL that relies
#: on it is in `db/repository` and the code that enforces it is in
#: `ingestion/` -- neither is a good home for a fact the other depends on.
ESPN_IDS_ARE_REGULAR_SEASON = True


def is_espn_id(value: object) -> bool:
    """Whether a game id is in the ESPN namespace rather than the NBA's."""
    return str(value or "").startswith(ESPN_ID_PREFIX)


def espn_game_id(event_id: object) -> str:
    """``401909881`` -> ``espn:401909881``, idempotently."""
    raw = str(event_id or "").strip()
    if not raw:
        raise ValueError("DATA_NOT_AVAILABLE: an ESPN event id is required")
    return raw if raw.startswith(ESPN_ID_PREFIX) else f"{ESPN_ID_PREFIX}{raw}"


def season_type_from_game_id(value: object) -> str:
    """
    What kind of game this id belongs to, or ``SEASON_TYPE_UNKNOWN``.

    Returns UNKNOWN rather than guessing REGULAR for an id it cannot read. A
    caller filtering to regular-season games must drop what it cannot
    identify: defaulting an unreadable id to "regular" would quietly admit
    playoff rows to a comparison that exists to exclude them.
    """
    normalised = normalise_one_game_id(value)
    if normalised is None or len(normalised) < 3:
        return SEASON_TYPE_UNKNOWN
    return _SEASON_TYPE_BY_DIGIT.get(normalised[2], SEASON_TYPE_UNKNOWN)


def is_regular_season_game_id(value: object) -> bool:
    """Whether this id is a regular-season game. Unreadable ids are not."""
    return season_type_from_game_id(value) == SEASON_TYPE_REGULAR
