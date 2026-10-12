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
