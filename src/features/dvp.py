"""
src/features/dvp.py — opponent defence split by the position it is defending.

WHAT THIS ADDS OVER ``src/features/defense.py``. That layer answers "how good
is tonight's opponent at defending?" with nine per-100-possession team rates.
It cannot answer "how good is tonight's opponent at defending SOMEONE LIKE
THIS PLAYER?", and those are different questions: a team can be average
overall while conceding to guards and smothering centres, and every player on
the slate receives the identical ``DEF_RATING_L10`` from it. A
defence-versus-position term is the first column in this project that varies
by WHO the player is as well as by whom he is playing.

THREE BUCKETS, NOT FIVE. The archive's own ``startingPosition`` takes exactly
three values — G, F, C — because the NBA records a starting five as two
guards, two forwards and a centre. Inventing PG/SG/SF/PF distinctions it does
not carry would be a guess dressed as data, and ``normalise_bucket`` collapses
the finer spellings other sources use onto these three rather than pretending
to recover them.

(One naming collision, worth stating because it bit the first draft: "PF" is
power forward in a position string and PERSONAL FOULS as a panel column —
see src/features/fouls.py. ``normalise_bucket`` reads position strings only,
and maps "PF" to F.)

HOW A PLAYER GETS A BUCKET, AND WHY IT IS AS-OF. ``startingPosition`` is
populated for starters only: 88,986 of the 214,381 panel rows (41.5%). A
player's bucket here is therefore the MODAL POSITION OF HIS OWN PRIOR
STARTS, expanding and shifted — never his current game's designation, which
would tell the model he is in tonight's starting five and so leak tonight's
minutes. Measured on the real panel: 87.2% of rows receive a bucket this
way, and on the 88,007 starter rows that have both, the as-of bucket equals
that night's actual designation 89.6% of the time. The remaining 27,349 rows
belong to players with no prior start, and they get NO bucket and NO DvP
columns. A fallback to a roster flag was measured and rejected: the archive's
``Players.csv`` guard/forward/centre flags disagree with the player's own
modal start for 15% of pure-flag players and for most hybrids (FLAG=F yet
modal C for 41 players, FLAG=G yet modal F for 56), and 251 of the panel's
1,659 players carry no flag at all. A 15%-wrong bucket silently mixes two
defensive populations.

THE AGGREGATION IS A MEAN PER PLAYER-GAME, AND THAT IS THE WHOLE DEFENCE
AGAINST THE BUG ``defense.py`` EXISTS FOR. That module refuses to sum the
player panel, because a sum over the players PRESENT IN THE PANEL measures
roster coverage as much as it measures defence — drop one player per
team-game and "points allowed" moves 33%. This layer has no choice but to
read the player panel, because the player panel is the only place a position
lives. So it takes a MEAN over the bucket's player-games rather than a sum:
coverage changes the sample size, not the level. Thin samples are then
handled at the WINDOW, by ``ROLL_MIN_PERIODS`` over the defender's games, and
not per game -- see below for why that distinction is not a detail.

MEASURED AND CHANGED, because the first version of this layer shipped a
column that was null for every centre in the league. Two faults, one cause:

  * A per-game minimum sample size of two player-games per bucket looked like
    prudence and was fatal. ``startingPosition`` names exactly one centre per
    team-game, so the C bucket's per-game sample is ALWAYS one, and every
    centre row came back NaN -- a third of the slate silently absent from a
    feature whose whole purpose is to distinguish positions. The minimum is
    gone; the ten-game window is where a thin sample is refused.
  * Aggregating over observed starters only measured what a defence allowed
    to opposing STARTERS, while the feature joins onto every bucketed player,
    bench included. Those have to be one population. The aggregation now
    counts a completed game under its observed designation where there is one
    and under the player's as-of bucket otherwise, which is leakage-safe for
    the same reason the as-of bucket is and raises the population from five
    players a side to everyone on the floor with a bucket.

With both fixed, 87.2% of rows receive a bucket and 81.7% also match their
opponent's prior form against it. The allowed means then separate exactly as
a position split should, over the 175,209 matched rows:

              PTS    REB    AST    BLK     rows
  C         10.92   7.31   1.87   1.02   33,089
  F         11.44   4.63   2.10   0.48   69,684
  G         12.89   3.36   3.65   0.30   72,436

A centre's matchup concedes 2.2x the rebounds and 3.4x the blocks a guard's
does, per player-game, and the INDEX columns centre all three buckets on
1.00 (C 1.012 / F 0.997 / G 0.999 on points). That is the scale argument
below stated as a measurement rather than a prediction: the raw column holds
three distributions and the index holds one.

WHY THE ROLLING IS OVER THE DEFENDER'S GAMES, NOT OVER PLAYER ROWS. The
implementation this is modelled on grouped player rows by (season, opponent,
position) and called ``.rolling(10)`` on them. A team faces four or five
guards a night, so that "L10" spanned about two games — the window's name was
wrong by a factor of four or five, and a team's number moved on the identity
of the fourth guard it faced rather than on anything about the team. Here the
per-game bucket mean is computed FIRST, one value per (defender, game,
bucket), and the ten-game window is then ten of the defender's games.

LEAKAGE, in the three places it could enter:

  1. The player's bucket is expanding-shifted over his own prior starts.
  2. The defender's allowed means are shift-1 rolling within the defending
     team's season, so tonight is never in tonight's number.
  3. The league baseline behind the INDEX columns is an AS-OF expanding daily
     mean, shifted — not a season-wide mean. A season-wide mean folds games
     that have not been played into an October index; ``defense.py`` records
     that exact bug being found and fixed, and the source this layer is
     adapted from reintroduced it as a ``groupby(["SEASON", ...]).median()``.

The one thing the aggregation may legitimately read is each PAST game's own
``startingPosition``, because by the time a row's features are built those
games are complete and their lineups are public. That observed bucket is used
INTERNALLY ONLY and is never written to the panel: a column saying "this
player started tonight" is minutes information about the current game, and
``tests/test_dvp.py`` asserts it does not escape. ``POS_BUCKET``, the one
position column that IS written, is the as-of estimate and carries no claim
about tonight's lineup.

WHY BOTH ``ALLOWED`` AND ``INDEX`` ARE EMITTED, unlike
``DEF_RATING_INDEX_L10`` which is excluded as redundant. There the index and
the raw rate describe ONE population — the team — and correlate at r = 0.999,
so a model receives one number twice. Here the raw column does not mean the
same thing on every row: centres, forwards and guards concede on different
scales, so ``DVP_PTS_ALLOWED_L10`` silently mixes three distributions in one
column while the index — allowed divided by what the league allows THAT
BUCKET — is comparable across all three. The index is the modelling column
and the raw value is kept for reporting, which is the reverse of the
defence layer's split for a stated reason.

NOT WIRED INTO ANY MODEL YET. Absent from ``labels.default_feature_cols`` on
purpose; ``scripts/feature_ab.py --layer dvp --wire-under-test`` is where a
defence-versus-position term earns its place or does not.

AND IT IS TRAINING-ONLY TODAY, which is stated here rather than discovered
later. ``STARTING_POSITION`` reaches the panel only from the Kaggle archive
ingest (``scripts/ingest_training_pack.py``). The live path builds its panel
from ``player_game_logs`` via ``repository.load_player_panel``, that table has
no position column, and the live puller has none to write: the header list of
``src/ingestion/boxscores.py``'s ``leaguegamelog`` payload, recorded in
``tests/test_boxscore_ingest.py``, carries no starting lineup. (That list was
checked rather than assumed, because the first draft of this note asserted the
same endpoint had no PF column and the header list says it does — which is how
``fouls`` came to reach the live path and this layer does not.) So on a live
slate this layer finds no ``STARTING_POSITION``, logs that it is skipping, and
adds no columns, which is the designed behaviour for a missing input rather
than a silent zero.

Closing the gap needs a WRITER first, not a column. A ``starting_position``
column would sit empty on every row, and a column nothing writes states a fact
the system does not have. The candidate source is
``src/ingestion/espn_game.py``, whose ``BoxScoreRow`` is player-level and
carries a ``stats`` dict and a ``did_not_play`` flag, or ``boxscoresummaryv3``
where ``stats.nba.com`` is reachable. Until one of them is wired, DvP is a
research column measured on history.

RESEARCH ONLY. Nothing here is a betting signal.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.features.season import season_start_year

logger = logging.getLogger(__name__)

SOURCE_NAME = "dvp"

STARTING_POSITION_COLUMN = "STARTING_POSITION"
POS_BUCKET_COLUMN = "POS_BUCKET"

# The only three the archive records. Order is fixed so a tie in the modal
# count below resolves the same way on every run.
POSITION_BUCKETS: tuple[str, ...] = ("G", "F", "C")

# Position spellings other sources use -> the bucket they collapse onto.
# Exact, verified equivalences only: this is a rename, not a guess. "PF" is
# POWER FORWARD here and is unrelated to the panel's PF (personal fouls).
BUCKET_ALIASES: dict[str, str] = {
    "G": "G", "PG": "G", "SG": "G", "GUARD": "G",
    "F": "F", "SF": "F", "PF": "F", "FORWARD": "F",
    "C": "C", "CENTER": "C", "CENTRE": "C",
}

DVP_STATS: tuple[str, ...] = ("PTS", "REB", "AST", "FG3M", "STL", "BLK")

ROLL_WINDOW = 10
# Five of the defender's games, not the three defense.py uses. UNFITTED, and
# a judgement rather than a measurement: each entry in this window is itself a
# mean over a handful of player-games, so it is noisier than the team total
# defense.py rolls, and five entries buy back some of that at the cost of
# nulling two more games per team-season. Nothing has been measured about 3
# vs 5 vs 8 here; if the feature_ab arm is ever run it is worth sweeping.
ROLL_MIN_PERIODS = 5

ALLOWED_TEMPLATE = "DVP_{stat}_ALLOWED_L10"
INDEX_TEMPLATE = "DVP_{stat}_INDEX_L10"

DVP_FEATURE_COLS: tuple[str, ...] = (
    POS_BUCKET_COLUMN,
    *(ALLOWED_TEMPLATE.format(stat=s) for s in DVP_STATS),
    *(INDEX_TEMPLATE.format(stat=s) for s in DVP_STATS),
)

# Panel columns without which there is no defence-versus-position to describe.
REQUIRED_PANEL_COLS: tuple[str, ...] = (
    "PLAYER_ID", "GAME_ID", "GAME_DATE", "OPPONENT_ABBREVIATION",
)


class DvpFeatureError(ValueError):
    """Raised when DvP features are asked for without a position source."""


def normalise_bucket(value: object) -> str | None:
    """
    One of G, F, C, or None.

    Hybrids resolve to their FIRST listed position, which is the convention
    the source itself uses: an archive row reading "F-C" is a forward the
    scorer also considered a centre, not a coin flip. None is returned for
    anything unrecognised — a bucket guessed from an unknown spelling would
    put the player in the wrong defensive population and still produce a
    plausible number.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip().upper()
    if not text or text in {"NAN", "NONE", "NA", "<NA>"}:
        return None
    if text in BUCKET_ALIASES:
        return BUCKET_ALIASES[text]
    for token in text.replace("/", "-").split("-"):
        token = token.strip()
        if token in BUCKET_ALIASES:
            return BUCKET_ALIASES[token]
    return None


def _observed_bucket(panel: pd.DataFrame) -> pd.Series:
    """
    The bucket a past game actually recorded, for the aggregation side only.

    NEVER written to the panel. For the row being predicted this is current-game
    information — it says whether the player is in tonight's starting five, and
    that is a minutes signal. ``attach_dvp_features`` uses it to bucket the
    DEFENDER's completed games and then drops it.
    """
    if STARTING_POSITION_COLUMN not in panel.columns:
        return pd.Series(None, index=panel.index, dtype="object")
    return panel[STARTING_POSITION_COLUMN].map(normalise_bucket)


def assign_position_buckets(panel: pd.DataFrame) -> pd.Series:
    """
    Each row's AS-OF bucket: the modal bucket of that player's PRIOR starts.

    Expanding and shifted, so a player's first start contributes to his second
    game and never to itself. Players with no prior start get None and are
    excluded from the DvP join rather than assigned a guess.
    """
    observed = _observed_bucket(panel)
    if not observed.notna().any():
        return pd.Series(None, index=panel.index, dtype="object")

    work = pd.DataFrame(
        {
            "PLAYER_ID": panel["PLAYER_ID"],
            "GAME_DATE": pd.to_datetime(panel["GAME_DATE"], errors="coerce"),
            "_OBS": observed,
        },
        index=panel.index,
    )
    order = work.sort_values(["PLAYER_ID", "GAME_DATE"], kind="mergesort").index
    work = work.loc[order]

    grouped = work.groupby("PLAYER_ID", sort=False)["_OBS"]
    counts = np.column_stack(
        [
            grouped.transform(lambda s, b=bucket: (s == b).cumsum().shift(1)).to_numpy(
                dtype="float64"
            )
            for bucket in POSITION_BUCKETS
        ]
    )
    # All-NaN rows are a player's first game: nansum of nothing is 0, and the
    # total of 0 is what sends them to None below.
    total = np.nansum(counts, axis=1)
    filled = np.nan_to_num(counts, nan=-1.0)
    # argmax breaks a tie toward the earlier bucket, which is why
    # POSITION_BUCKETS is an ordered constant rather than a set.
    winner = np.asarray(POSITION_BUCKETS)[np.argmax(filled, axis=1)]
    asof = pd.Series(
        np.where(total > 0, winner, None), index=work.index, dtype="object"
    )
    return asof.reindex(panel.index)


def _aggregation_bucket(panel: pd.DataFrame) -> pd.Series:
    """
    The bucket a COMPLETED game is counted under: observed, else as-of.

    Two reasons it is not the observed designation alone, both measured on
    the real 214,381-row panel and both recorded under "MEASURED AND CHANGED"
    in this module's docstring:

      * ``startingPosition`` exists for starters only, so an observed-only
        aggregation describes what a defence allowed to opposing STARTERS
        while the feature is joined onto every bucketed player, bench
        included. The two populations have to be the same one.
      * A team faces exactly ONE starting centre per game. Any per-game
        minimum sample size therefore annihilates the C bucket, and the
        starters-only level makes a centre's bucket mean one player's night.

    Falling back to the as-of bucket is leakage-safe for the same reason the
    as-of bucket is: it reads the player's PRIOR starts only. It raises the
    population from the five starters to every bucketed player on the floor.
    """
    observed = _observed_bucket(panel)
    asof = assign_position_buckets(panel)
    return observed.where(observed.notna(), asof)


def build_opponent_allowed(panel: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (defending team, game, bucket) carrying PREGAME allowed means.

    The defending team is the player's ``OPPONENT_ABBREVIATION``: a player's
    row describes what he did TO that team, so grouping his rows by it yields
    what that team conceded. Getting this backwards would hand every model a
    full column of plausible numbers describing the wrong defence, which is
    the failure ``attach_defense_features`` carries the same warning about.

    The grid is the full cross product of the defender's games and all three
    buckets, not only the (game, bucket) pairs that were observed. A game in
    which the panel recorded no centre against this team must still appear, as
    a null contribution to the window, or that game would be skipped and a
    ten-game window would quietly span twelve.
    """
    missing = [c for c in REQUIRED_PANEL_COLS if c not in panel.columns]
    if missing:
        raise DvpFeatureError(
            f"DATA_NOT_AVAILABLE: panel missing {missing}, so there is no "
            "(defender, game, player) relationship to aggregate."
        )

    bucket = _aggregation_bucket(panel)
    if not bucket.notna().any():
        raise DvpFeatureError(
            "DATA_NOT_AVAILABLE: no row carries a usable "
            f"{STARTING_POSITION_COLUMN}. Positions are not derivable from the "
            "box score, and a bucket guessed from rebounds and assists would "
            "mix two defensive populations under one label."
        )

    stats = [s for s in DVP_STATS if s in panel.columns]
    if not stats:
        raise DvpFeatureError(
            f"DATA_NOT_AVAILABLE: none of {DVP_STATS} are in the panel, so "
            "there is nothing a defence could have allowed."
        )

    work = pd.DataFrame(
        {
            "team_abbr": panel["OPPONENT_ABBREVIATION"].astype("string"),
            "nba_game_id": panel["GAME_ID"].astype(str),
            "game_date": pd.to_datetime(panel["GAME_DATE"], errors="coerce"),
            "_BUCKET": bucket,
        },
        index=panel.index,
    )
    for stat in stats:
        work[stat] = pd.to_numeric(panel[stat], errors="coerce")
    work = work.dropna(subset=["team_abbr", "game_date"])

    # Every (defender, game) pair the panel knows about, whether or not a
    # bucket was observed in it.
    games = (
        work[["team_abbr", "nba_game_id", "game_date"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    games["SEASON_KEY"] = season_start_year(games["game_date"])

    bucketed = work.dropna(subset=["_BUCKET"])
    # MEAN per player-game in the bucket, never a sum -- see the docstring.
    # There is deliberately NO per-game minimum sample size: the ten-game
    # window over the DEFENDER's games is where thin samples are handled, by
    # ROLL_MIN_PERIODS, and a per-game floor of two was measured to null the
    # entire C bucket because a team faces one starting centre a night.
    per_game = bucketed.groupby(
        ["team_abbr", "nba_game_id", "_BUCKET"], as_index=False, observed=True
    ).agg({s: "mean" for s in stats})

    grid = games.merge(
        pd.DataFrame({"_BUCKET": list(POSITION_BUCKETS)}), how="cross"
    )
    merged = grid.merge(
        per_game, on=["team_abbr", "nba_game_id", "_BUCKET"], how="left"
    )
    merged = merged.sort_values(
        ["team_abbr", "_BUCKET", "SEASON_KEY", "game_date"], kind="mergesort"
    ).reset_index(drop=True)

    grouped = merged.groupby(["team_abbr", "_BUCKET", "SEASON_KEY"], sort=False)
    for stat in stats:
        merged[ALLOWED_TEMPLATE.format(stat=stat)] = grouped[stat].transform(
            lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=ROLL_MIN_PERIODS).mean()
        )
    for stat in stats:
        allowed = ALLOWED_TEMPLATE.format(stat=stat)
        merged[INDEX_TEMPLATE.format(stat=stat)] = _bucket_relative_index(
            merged, allowed
        )

    keep = ["nba_game_id", "team_abbr", "game_date", "_BUCKET", "SEASON_KEY"]
    keep += [ALLOWED_TEMPLATE.format(stat=s) for s in stats]
    keep += [INDEX_TEMPLATE.format(stat=s) for s in stats]
    out = merged[keep].copy()

    first = ALLOWED_TEMPLATE.format(stat=stats[0])
    known = int(out[first].notna().sum())
    logger.info(
        "DvP layer: %d (defender, game, bucket) rows, allowed mean known on %d "
        "(%.1f%%). The rest are a team's first %d games of a season against "
        "that bucket, which have no prior form and are left null.",
        len(out), known, 100.0 * known / max(len(out), 1), ROLL_MIN_PERIODS,
    )
    return out


def _bucket_relative_index(frame: pd.DataFrame, col: str) -> pd.Series:
    """
    ``col`` divided by what the league allowed THAT BUCKET as of that date.

    Per (season, bucket), never pooled across buckets: dividing a centre's
    rebounds-allowed by a league mean that includes guards would report every
    centre matchup as favourable. The baseline is an expanding daily mean,
    shifted one day so a team's own game-day value is out of its own
    denominator — the form ``defense._league_relative_index`` arrived at, and
    deliberately not the season-wide median the source implementation used.
    """
    daily = (
        frame.groupby(["SEASON_KEY", "_BUCKET", "game_date"], as_index=False,
                      observed=True)[col]
        .mean()
        .rename(columns={col: "_day_mean"})
        .sort_values(["SEASON_KEY", "_BUCKET", "game_date"], kind="mergesort")
        .reset_index(drop=True)
    )
    daily["_asof"] = daily.groupby(["SEASON_KEY", "_BUCKET"], sort=False, observed=True)[
        "_day_mean"
    ].transform(lambda s: s.expanding(min_periods=ROLL_MIN_PERIODS).mean().shift(1))
    joined = frame.merge(
        daily[["SEASON_KEY", "_BUCKET", "game_date", "_asof"]],
        on=["SEASON_KEY", "_BUCKET", "game_date"], how="left",
    )
    baseline = joined["_asof"].where(joined["_asof"] > 0)
    return (joined[col] / baseline).to_numpy()


def attach_dvp_features(
    panel: pd.DataFrame,
    allowed: pd.DataFrame | None = None,
    *,
    required: bool = False,
) -> pd.DataFrame:
    """
    Join the opponent's prior form AGAINST THIS PLAYER'S BUCKET onto each row.

    The join is on (game, defending team, the player's own as-of bucket), so
    two players in the same game against the same team receive DIFFERENT
    numbers when they play different positions. That is the entire point of
    the layer; a join that dropped the bucket would reproduce
    ``DEF_RATING_L10`` under a new name.

    Unmatched rows keep NaN. A league-average fill would read as a measured
    matchup against an average defence, and a bucket-less player would get a
    matchup number for a position he has not been observed playing.
    """
    if panel is None or panel.empty:
        if required:
            raise DvpFeatureError("DATA_NOT_AVAILABLE: panel is empty")
        logger.info("DvP layer skipped: empty panel.")
        return panel if panel is not None else pd.DataFrame()

    if STARTING_POSITION_COLUMN not in panel.columns:
        message = (
            f"panel has no {STARTING_POSITION_COLUMN} column. The Kaggle "
            "archive's startingPosition is mapped by src.ingestion.kaggle_nba; "
            "a panel built before that mapping existed must be rebuilt."
        )
        if required:
            raise DvpFeatureError(f"DATA_NOT_AVAILABLE: {message}")
        logger.info("DvP layer skipped: %s", message)
        return panel

    try:
        table = build_opponent_allowed(panel) if allowed is None else allowed
    except DvpFeatureError:
        if required:
            raise
        logger.info("DvP layer skipped: no usable position source in the panel.")
        return panel

    out = panel.copy()
    out[POS_BUCKET_COLUMN] = assign_position_buckets(out)
    out["GAME_ID"] = out["GAME_ID"].astype(str)

    lookup = table.rename(
        columns={
            "nba_game_id": "GAME_ID",
            "team_abbr": "OPPONENT_ABBREVIATION",
            "_BUCKET": POS_BUCKET_COLUMN,
        }
    ).drop(columns=[c for c in ("game_date", "SEASON_KEY") if c in table.columns])
    lookup["GAME_ID"] = lookup["GAME_ID"].astype(str)
    # The lookup's team codes are built as pandas "string"; the panel's are
    # "string" through kaggle_nba and plain object in a hand-built frame.
    # pandas DOES currently match across those two dtypes -- that was checked
    # by reverting this line and watching the join still work -- so this is
    # HARDENING, NOT A LIVE FAULT. It is here because the failure it forecloses
    # is the quietest one in this module: a merge that matches nothing returns
    # a full column of nulls that reads exactly like a league of teams with no
    # prior form, and the log line below would report 0.0% matched as though
    # that were a fact about the season. Aligning to the PANEL's dtype rather
    # than imposing one also means the frame handed back is the frame handed
    # in.
    lookup["OPPONENT_ABBREVIATION"] = lookup["OPPONENT_ABBREVIATION"].astype(
        out["OPPONENT_ABBREVIATION"].dtype
    )

    keys = ["GAME_ID", "OPPONENT_ABBREVIATION", POS_BUCKET_COLUMN]
    duplicated = int(lookup.duplicated(subset=keys).sum())
    if duplicated:
        raise DvpFeatureError(
            f"DvP lookup holds {duplicated} duplicate (game, defender, bucket) "
            "rows; joining it would silently multiply player rows."
        )

    before = len(out)
    joined = out.merge(
        lookup.assign(
            **{POS_BUCKET_COLUMN: lookup[POS_BUCKET_COLUMN].astype("object")}
        ),
        on=keys,
        how="left",
    )
    if len(joined) != before:
        raise DvpFeatureError(
            f"DvP join changed the row count ({before} -> {len(joined)})."
        )

    # The observed bucket never leaves this module — see _observed_bucket.
    assert "_BUCKET" not in joined.columns

    first = ALLOWED_TEMPLATE.format(stat=DVP_STATS[0])
    matched = int(joined[first].notna().sum()) if first in joined.columns else 0
    bucketed = int(joined[POS_BUCKET_COLUMN].notna().sum())
    logger.info(
        "DvP layer: %d of %d rows carry an as-of position bucket (%.1f%%), and "
        "%d matched their opponent's prior form against that bucket (%.1f%%). "
        "Unmatched rows are null, not league-average.",
        bucketed, len(joined), 100.0 * bucketed / max(len(joined), 1),
        matched, 100.0 * matched / max(len(joined), 1),
    )
    return joined


def attach_dvp_features_layer(panel: pd.DataFrame) -> pd.DataFrame:
    """Registry entry point. Same contract as every other additive layer."""
    return attach_dvp_features(panel, required=False)
