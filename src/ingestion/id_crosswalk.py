"""Match an external source's player names onto this panel's own.

RESEARCH_ONLY. Reads no network and no credential; it is a pure name map.

WHY THIS MODULE EXISTS. ``main._attach_prop_lines``,
``settlement.recorder._line_lookup`` and ``pipeline.scratches`` all join a
pick'em board to the player panel on an EXACT name, and all three told the
reader to route name variance "through ``ingestion/id_crosswalk.py``" — a path
that did not exist. The consequence was measured in
``docs/integration_audit.md`` §3: one diacritic (``Nikola Jokic`` against
``Nikola Jokić``) skips the row for want of a line source, so a systematic
format difference records ZERO gradeable rows while the run reports success.

THIS IS NOT A FUZZY MATCHER, AND THE MEASUREMENT IS WHY.

The obvious implementation — rapidfuzz ``token_sort_ratio`` above a score
cutoff — cannot work here, and the reference version this was adapted from used
exactly that at ``score_cutoff=85``. Scores on real NBA pairs:

    MUST MATCH                                    token_sort   ratio
    Nikola Jokic      / Nikola Jokić                    91.7    91.7
    Kristaps Porzingis/ Kristaps Porziņģis              88.9    88.9
    Luka Doncic       / Luka Dončić                     81.8    81.8
    Shai Gilgeous-Alexander / Shai Gilgeous Alexander   60.9    95.7

    MUST REFUSE
    Jalen Williams    / Jaylen Williams                 96.6    96.6
    Jalen Johnson     / Jaylen Johnson                  96.3    96.3
    Jaylen Brown      / Jalen Brown                     95.7    95.7
    Gary Payton II    / Gary Payton                     88.0    95.0

The classes OVERLAP, and not narrowly: the most dangerous pair in the league
scores 96.6 while Dončić scores 81.8. A cutoff of 85 therefore DROPS Dončić and
ACCEPTS Jalen Williams as Jaylen Williams — a wrong name on a wrong price in a
settlement ledger, which is not a near miss. No threshold separates them,
because the difference is not spelling distance: two different people have
similar names and one person has two spellings.

So the match is DETERMINISTIC. Unicode is decomposed and combining marks are
dropped, dots and apostrophes vanish, hyphens become spaces, and the result
must be EQUAL. That decides every case above correctly and never has to judge:

    Jokić -> jokic          P.J. Washington -> pj washington
    Dončić -> doncic        Gilgeous-Alexander -> gilgeous alexander
    Porziņģis -> porzingis  De'Aaron Fox -> deaaron fox

SUFFIXES ARE KEPT ON PURPOSE. ``Gary Payton II`` and ``Gary Payton`` are a son
and a father, as are ``Jabari Smith Jr.`` and ``Jabari Smith``. A
suffix-insensitive key merges them, so ``Jr``/``II`` stay part of the name and
those pairs correctly do not match.

A name that does not resolve is reported, never guessed: the result carries
``status`` and a reason, the caller keeps its own exact join, and a row with no
line is a row with no line.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

MATCHED = "MATCHED"
AMBIGUOUS = "AMBIGUOUS"
UNMATCHED = "DATA_NOT_AVAILABLE"

_STRIP = re.compile(r"[.'‘’ʼ]")
_TO_SPACE = re.compile(r"[-_/,]")

# WHERE THE NORMALISER STOPS AND A HUMAN STARTS.
#
# Everything above this line is derivable: a diacritic, a dot, an apostrophe,
# a hyphen. "Alexandre" -> "Alex" is not. It is a different given name for the
# same person, and no deterministic rule produces one from the other -- while
# the fuzzy score that would is precisely the score that also accepts Jalen
# Williams as Jaylen Williams (see the table at the top of this module). So
# the remaining cases are NAMED, one at a time, and each one is a claim about
# a specific person that someone checked.
#
# PROVENANCE IS THE WHOLE POINT. The panel stores ESPN's
# ``athlete.displayName`` (``src/ingestion/espn_game.py:387``), so an entry is
# only admissible if ESPN's own roster carries the value spelling and carries
# no player under the key spelling. Every entry below was checked against
# ESPN's NBA rosters on 2026-11-05 and was logged unmatched by the 2026-11-04
# slate -- which is also the bar for adding one: a name production actually
# missed, verified against the source the panel is built from, never a
# plausible-looking pair.
#
#   board "Alexandre Sarr" -> ESPN "Alex Sarr"        (WSH; the only Sarr)
#   board "Joshua Giddey"  -> ESPN "Josh Giddey"      (CHI; the only Giddey)
#   board "Jaime Jaquez"   -> ESPN "Jaime Jaquez Jr." (MIL; the only Jaquez)
#
# THE JAQUEZ ENTRY IS NOT A SUFFIX RULE. Suffixes stay part of the key on
# purpose, because ``Gary Payton II`` and ``Gary Payton`` are two people. This
# names one player whose book spelling drops his suffix; it does not make
# ``Jr``/``II`` optional for anyone else, and
# ``test_suffix_stripping_is_still_not_a_rule`` fails if it ever does.
#
# Keys and values are in canonical (normalised) form, because that is what
# they are compared against. Lookup is ONE HOP: a value may not itself be a
# key, so resolution cannot depend on dict order.
# GENERATIONAL SUFFIXES, FOR THE DIAGNOSTIC HINT ONLY.
#
# These are NOT stripped from a matching key -- `Gary Payton II` and `Gary
# Payton` stay two different people, which is the whole point of the module
# docstring's last paragraph. They are listed here because
# `names_sharing_surname` needs to know which trailing token is a suffix
# rather than a surname. Its first version compared the last token outright,
# so the 2026-11-05 slate logged "Jase Richardson Jr. -> ['Tim Hardaway Jr.',
# 'Wendell Carter Jr.']" (three unrelated people whose names end in "Jr.")
# and "Wendell Carter -> ['Jevon Carter']", omitting the panel's own "Wendell
# Carter Jr." -- the one name that was the answer.
_GENERATIONAL_SUFFIXES = frozenset({"jr", "sr", "ii", "iii", "iv", "v"})

# How many unmatched names the warning spells out. High enough that a real
# slate's misses all appear (2026-11-04 had 23), bounded so a board describing
# an entirely different slate cannot write a thousand names into the log.
_UNMATCHED_LOG_CAP = 60

BOARD_NAME_ALIASES: dict[str, str] = {
    "alexandre sarr": "alex sarr",
    "joshua giddey": "josh giddey",
    "jaime jaquez": "jaime jaquez jr",
    # Both logged unmatched by the 2026-11-05 slate (ORL at PHX), both
    # checked the same way, and note they MISMATCH IN OPPOSITE DIRECTIONS --
    # which is the argument against a blanket suffix rule as much as for one:
    #
    #   board "Wendell Carter"      -> ESPN "Wendell Carter Jr." (ORL)
    #       ESPN has no bare Wendell Carter; the other Carters are Jevon
    #       (ORL), Devin (BOS) and Carter Bryant (SA), none of them him.
    #   board "Jase Richardson Jr." -> ESPN "Jase Richardson"    (ORL)
    #       ESPN has no Jase Richardson Jr.; the book adds a suffix the
    #       panel does not carry.
    "wendell carter": "wendell carter jr",
    "jase richardson jr": "jase richardson",
}


def normalise_player_name(name: Any) -> str:
    """
    The canonical form both sides are compared in. Deterministic, not fuzzy.

    Decompose Unicode and drop combining marks (ć -> c, ņ -> n), lowercase,
    delete dots and apostrophes so ``P.J.`` and ``PJ`` agree and ``De'Aaron``
    and ``DeAaron`` agree, turn hyphens into spaces so a hyphenated surname
    matches its spaced form, then collapse whitespace.

    Returns "" for anything empty or unusable, which callers treat as a miss
    rather than as a key.
    """
    decomposed = unicodedata.normalize("NFKD", str(name or ""))
    without_marks = "".join(c for c in decomposed if not unicodedata.combining(c))
    lowered = without_marks.lower()
    stripped = _STRIP.sub("", lowered)
    spaced = _TO_SPACE.sub(" ", stripped)
    return " ".join(spaced.split())


def _surname_for_hint(name: Any) -> str:
    """
    The surname to compare NAMES BY EYE, suffix set aside. Diagnosis only.

    Nothing resolves through this. It exists so the unmatched-name warning
    can point at "Wendell Carter Jr." when the board says "Wendell Carter",
    instead of at everyone else whose name ends in the same suffix.

    A name that is nothing but a suffix keeps it, because dropping it would
    leave no surname at all to compare.
    """
    tokens = normalise_player_name(name).split()
    while len(tokens) > 1 and tokens[-1] in _GENERATIONAL_SUFFIXES:
        tokens.pop()
    return tokens[-1] if tokens else ""


@dataclass(frozen=True)
class PlayerRecord:
    """One canonical player, as the panel knows them."""

    name: str
    player_id: str | None = None
    team: str | None = None

    @property
    def key(self) -> str:
        return normalise_player_name(self.name)


@dataclass(frozen=True)
class CrosswalkMatch:
    """One resolution attempt, including why it failed."""

    query_name: str
    status: str = UNMATCHED
    matched_name: str | None = None
    player_id: str | None = None
    team: str | None = None
    reason: str | None = None

    @property
    def resolved(self) -> bool:
        return self.status == MATCHED

    def as_dict(self) -> dict[str, Any]:
        return {
            "query_name": self.query_name,
            "status": self.status,
            "matched_name": self.matched_name,
            "player_id": self.player_id,
            "team": self.team,
            "reason": self.reason,
        }


@dataclass
class CrosswalkReport:
    """What a batch resolved to, and every name it could not place."""

    matches: dict[str, CrosswalkMatch] = field(default_factory=dict)
    n_queried: int = 0
    n_matched: int = 0
    unmatched: list[str] = field(default_factory=list)
    ambiguous: list[str] = field(default_factory=list)

    @property
    def name_map(self) -> dict[str, str]:
        """Query name -> canonical panel name, for the resolved ones only."""
        return {
            q: m.matched_name
            for q, m in self.matches.items()
            if m.resolved and m.matched_name
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_queried": self.n_queried,
            "n_matched": self.n_matched,
            "n_unmatched": len(self.unmatched),
            "n_ambiguous": len(self.ambiguous),
            "unmatched_sample": self.unmatched[:10],
            "ambiguous_sample": self.ambiguous[:10],
        }


class PlayerIdCrosswalk:
    """
    Resolve an outside source's player names onto the panel's canonical names.

    Built from the panel rather than from a roster endpoint: the panel is the
    side whose names the rest of this pipeline keys on, and it already carries
    the PLAYER_ID a caller needs. A roster fetch would add a second name
    vocabulary to reconcile rather than removing one.
    """

    def __init__(self, roster: Iterable[PlayerRecord]) -> None:
        self._by_key: dict[str, list[PlayerRecord]] = defaultdict(list)
        for record in roster:
            key = record.key
            if not key:
                continue
            # The same player appearing many times in a panel is one player.
            if not any(
                existing.name == record.name
                and existing.player_id == record.player_id
                and existing.team == record.team
                for existing in self._by_key[key]
            ):
                self._by_key[key].append(record)

    def __len__(self) -> int:
        return sum(len(v) for v in self._by_key.values())

    def match_one(self, query_name: Any, *, team: Any | None = None) -> CrosswalkMatch:
        """
        Resolve one name, optionally disambiguated by team.

        TEAM IS A TIEBREAK, NOT A REQUIREMENT. Two players sharing a normalised
        name is the only case it is consulted for; requiring it everywhere would
        refuse every row from a board that does not publish a team, which is
        most of them.
        """
        raw = str(query_name or "").strip()
        key = normalise_player_name(raw)
        if not key:
            return CrosswalkMatch(
                query_name=raw, status=UNMATCHED, reason="empty player name"
            )

        candidates = self._by_key.get(key, [])
        # EXACT FIRST, ALWAYS. The alias table exists to reach a name the
        # panel HAS under a spelling the book uses; it must never rewrite a
        # name the panel already carries, or a future panel gaining the
        # board's own spelling would start resolving to somebody else.
        alias_key: str | None = None
        if not candidates:
            aliased = BOARD_NAME_ALIASES.get(key)
            if aliased:
                candidates = self._by_key.get(aliased, [])
                if candidates:
                    alias_key = aliased
        if not candidates:
            return CrosswalkMatch(
                query_name=raw,
                status=UNMATCHED,
                reason=(
                    "no panel player normalises to this name, and no entry in "
                    "BOARD_NAME_ALIASES reaches one. Deliberately not "
                    "fuzzy-matched: on real NBA names a score high enough to "
                    "catch a diacritic also catches Jalen/Jaylen."
                ),
            )
        if len(candidates) == 1:
            hit = candidates[0]
            return CrosswalkMatch(
                query_name=raw, status=MATCHED, matched_name=hit.name,
                player_id=hit.player_id, team=hit.team,
                reason=(
                    f"book spelling aliased to {alias_key!r}"
                    if alias_key else None
                ),
            )

        wanted_team = normalise_player_name(team)
        if wanted_team:
            on_team = [
                c for c in candidates
                if normalise_player_name(c.team) == wanted_team
            ]
            if len(on_team) == 1:
                hit = on_team[0]
                return CrosswalkMatch(
                    query_name=raw, status=MATCHED, matched_name=hit.name,
                    player_id=hit.player_id, team=hit.team,
                    reason="disambiguated by team",
                )

        teams = sorted({str(c.team) for c in candidates})
        return CrosswalkMatch(
            query_name=raw,
            status=AMBIGUOUS,
            reason=(
                f"{len(candidates)} panel players normalise to {key!r} "
                f"(teams: {teams}). Abstaining: matching the wrong player onto "
                "a price is a wrong record, not a near miss."
            ),
        )

    def names_sharing_surname(self, query_name: Any) -> list[str]:
        """
        Panel names whose last token equals this one's, for DIAGNOSIS ONLY.

        This is not a matching rule and nothing resolves through it. It
        answers the one question an unmatched name raises and the warning
        could not previously distinguish: is the panel carrying this player
        under another spelling, or not carrying them at all? Those have
        different fixes -- an alias entry versus a missing game -- and
        guessing between them is how a day gets spent on the wrong one.
        """
        surname = _surname_for_hint(query_name)
        if not surname:
            return []
        return sorted({
            record.name
            for panel_key, records in self._by_key.items()
            if _surname_for_hint(panel_key) == surname
            for record in records
        })

    def match_many(
        self,
        names: Iterable[Any],
        *,
        teams: Sequence[Any] | None = None,
    ) -> CrosswalkReport:
        """Resolve a batch, reporting every miss by name."""
        name_list = list(names)
        team_list = list(teams) if teams is not None else [None] * len(name_list)
        if len(team_list) != len(name_list):
            raise ValueError(
                f"{len(name_list)} name(s) against {len(team_list)} team(s) — "
                "the two sequences must be parallel"
            )

        report = CrosswalkReport(n_queried=len(name_list))
        for name, team in zip(name_list, team_list, strict=True):
            raw = str(name or "").strip()
            if raw in report.matches:
                continue
            match = self.match_one(raw, team=team)
            report.matches[raw] = match
            if match.resolved:
                report.n_matched += 1
            elif match.status == AMBIGUOUS:
                report.ambiguous.append(raw)
            else:
                report.unmatched.append(raw)
        return report


def crosswalk_from_frame(
    frame: pd.DataFrame | None,
    *,
    name_col: str = "PLAYER_NAME",
    id_col: str | None = "PLAYER_ID",
    team_col: str | None = "TEAM_ABBREVIATION",
) -> PlayerIdCrosswalk:
    """Build the crosswalk from the panel (or projections) frame itself."""
    if frame is None or getattr(frame, "empty", True) or name_col not in frame.columns:
        return PlayerIdCrosswalk([])

    records: list[PlayerRecord] = []
    for _, row in frame.iterrows():
        records.append(PlayerRecord(
            name=str(row.get(name_col) or "").strip(),
            player_id=(
                str(row.get(id_col)) if id_col and pd.notna(row.get(id_col)) else None
            ),
            team=(
                str(row.get(team_col)) if team_col and pd.notna(row.get(team_col)) else None
            ),
        ))
    return PlayerIdCrosswalk(records)


def resolve_board_names(
    board: pd.DataFrame | None,
    panel: pd.DataFrame | None,
    *,
    board_name_col: str = "player_name",
    board_team_col: str | None = "team",
    panel_name_col: str = "PLAYER_NAME",
    panel_team_col: str | None = "TEAM_ABBREVIATION",
) -> tuple[dict[str, str], CrosswalkReport]:
    """
    ``{board name -> canonical panel name}`` plus the report.

    The intended use: a caller keeps its existing exact-match join and rewrites
    the board's name column through this map first, so the join itself stays
    exact and every unresolved name is visible rather than absorbed.
    """
    empty = CrosswalkReport()
    if board is None or getattr(board, "empty", True):
        return {}, empty
    if board_name_col not in board.columns:
        logger.warning(
            "Board frame has no %s column — no name resolution attempted.",
            board_name_col,
        )
        return {}, empty

    crosswalk = crosswalk_from_frame(
        panel, name_col=panel_name_col, team_col=panel_team_col
    )
    if not len(crosswalk):
        return {}, empty

    teams = (
        list(board[board_team_col])
        if board_team_col and board_team_col in board.columns
        else None
    )
    report = crosswalk.match_many(list(board[board_name_col]), teams=teams)

    # A name that already equals its canonical form is not worth logging as a
    # rewrite; only the ones the normaliser actually changed are interesting.
    rewrites = {q: c for q, c in report.name_map.items() if q != c}
    if rewrites:
        logger.info(
            "Name crosswalk: %d of %d board name(s) resolved, %d rewritten "
            "(e.g. %s)",
            report.n_matched, report.n_queried, len(rewrites),
            list(rewrites.items())[:3],
        )
    if report.unmatched:
        # EVERY NAME, NOT FIVE OF THEM. This warning used to print
        # `report.unmatched[:5]`, which is how 2026-11-04's 23 misses stayed
        # undiagnosable: the five shown happened to include two players the
        # panel did not carry at all, so the list read as a spelling problem
        # when half of it was not one. A warning you cannot act on is a
        # warning that gets ignored.
        #
        # Each name is classified by whether the panel holds anyone with the
        # same surname. "Alexandre Sarr -> has Alex Sarr" is an alias to add
        # (see BOARD_NAME_ALIASES); "Kon Knueppel -> none" is this slate not
        # carrying him, which no name table can fix.
        detail = []
        for name in report.unmatched[:_UNMATCHED_LOG_CAP]:
            near = crosswalk.names_sharing_surname(name)
            detail.append(f"{name} -> {near if near else 'no such surname in panel'}")
        elided = max(0, len(report.unmatched) - _UNMATCHED_LOG_CAP)
        logger.warning(
            "Name crosswalk: %d board name(s) match no panel player, so those "
            "rows carry no line. Each is shown with the panel name(s) sharing "
            "its surname: a name beside a candidate is a spelling to add to "
            "BOARD_NAME_ALIASES, a name beside none is a player this slate's "
            "panel does not carry. %s%s",
            len(report.unmatched),
            "; ".join(detail),
            f" (+{elided} more)" if elided else "",
        )
    if report.ambiguous:
        logger.warning(
            "Name crosswalk: %d board name(s) are ambiguous and were refused: "
            "%s. Matching the wrong player onto a price is a wrong record.",
            len(report.ambiguous), report.ambiguous[:5],
        )
    return report.name_map, report


def apply_name_map(
    board: pd.DataFrame,
    name_map: Mapping[str, str],
    *,
    name_col: str = "player_name",
) -> pd.DataFrame:
    """
    Return a copy of ``board`` with its name column rewritten to canonical form.

    A name absent from the map is left exactly as it was: the caller's join then
    misses it, which is the honest outcome for a name nothing could resolve.
    """
    if board is None or getattr(board, "empty", True) or name_col not in board.columns:
        return board
    if not name_map:
        return board
    out = board.copy()
    out[name_col] = [
        name_map.get(str(v).strip(), v) for v in out[name_col]
    ]
    return out
