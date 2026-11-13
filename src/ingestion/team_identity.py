"""Derive the report's team names from its own matchups. No hardcoded names.

THE PROBLEM THIS SOLVES, and why it blocked the absence bridge.

`model_input_archive.build_absence_export` crosswalks an availability row to a
panel identity by matching on the REPORT's vocabulary: its `team` field, which
is a full club name such as "Orlando Magic". The panel carries
`TEAM_ABBREVIATION` ("ORL") and `OPPONENT_ABBREVIATION`, and nothing in this
repository maps one to the other. That missing map -- not the parsing, and not
the feature layer -- is what stood between an archived availability report and
a usable absence feature.

WHY THIS IS NOT A HARDCODED TABLE OF THIRTY TEAMS. Writing the names out from
memory looks like the obvious fix and is the wrong one, because an exact-match
join fails silently on a near-miss: the team simply never resolves, its rows
go to `unresolved`, and the feature abstains for that club with no error
anywhere. The report's own string for one club is "LA Clippers" -- not "Los
Angeles Clippers", which is what a from-memory table would almost certainly
say, and LAC would have gone quietly missing for the season.

HOW IT WORKS INSTEAD. Every report row carries both a full name and a
`matchup` written in abbreviations, "MIL@ORL". So each row says "this club is
one of these two abbreviations", and the club's identity falls out of the
constraints:

  1. INTERSECTION. A club seen in several matchups must be the abbreviation
     common to all of them. "Chicago Bulls" in ATL@CHI and CHI@MIN can only be
     CHI, because {ATL, CHI} & {CHI, MIN} = {CHI}.

  2. ELIMINATION. Two clubs in one matchup cannot share an abbreviation, so a
     resolved club removes its own from its opponent's candidates. Once
     "Chicago Bulls" is CHI, "Atlanta Hawks" in ATL@CHI must be ATL. This
     propagates until nothing changes.

On the 2026-11-13 01:15AM report alone, intersection resolves 5 of the 21
clubs present and elimination takes it to 11. The remaining 10 are clubs that
appear in exactly one matchup whose opponent is also unresolved -- GSW@SAS,
HOU@UTA, LAC@OKC, POR@SAC -- and no amount of reasoning about one night can
tell those pairs apart. They stay AMBIGUOUS, and resolve as further reports
are archived, because a club plays different opponents.

So the map is derived, self-correcting, and contains no club name this
repository invented. An unresolved club abstains; it never guesses.

A reviewed `known` mapping may be supplied, and it WINS over inference. That
is the operator's lever for the one-matchup pairs above, and the reason this
returns provenance for every entry: a reader must be able to tell a club this
module worked out from one a human asserted.

RESEARCH_ONLY. This resolves identities. It does not price or place a wager.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

logger = logging.getLogger(__name__)

RESEARCH_STATUS = "RESEARCH_ONLY"

#: Separator the official report writes between the away and home clubs.
MATCHUP_SEPARATOR = "@"

#: Ceiling on propagation rounds. Each round resolves at least one club or
#: stops, and there are 30 clubs, so this cannot be reached by correct input;
#: it exists so a malformed report cannot spin.
MAX_PROPAGATION_ROUNDS = 64

SOURCE_INFERRED = "inferred"
SOURCE_REVIEWED = "reviewed"


@dataclass(frozen=True)
class TeamIdentityMap:
    """Full club name -> abbreviation, with how each entry was decided."""

    resolved: dict[str, str] = field(default_factory=dict)
    ambiguous: dict[str, tuple[str, ...]] = field(default_factory=dict)
    provenance: dict[str, str] = field(default_factory=dict)

    def abbreviation(self, team_name: Any) -> str | None:
        """The abbreviation for a club, or None when it is not resolved."""
        return self.resolved.get(str(team_name or "").strip())

    def summary(self) -> dict[str, Any]:
        return {
            "research_status": RESEARCH_STATUS,
            "n_resolved": len(self.resolved),
            "n_ambiguous": len(self.ambiguous),
            "n_reviewed": sum(
                1 for v in self.provenance.values() if v == SOURCE_REVIEWED
            ),
            "resolved": dict(sorted(self.resolved.items())),
            "ambiguous": {k: list(v) for k, v in sorted(self.ambiguous.items())},
        }


def _matchup_abbreviations(matchup: Any) -> set[str] | None:
    """{'MIL', 'ORL'} from 'MIL@ORL'; None when it is not a two-club matchup."""
    text = str(matchup or "").strip().upper()
    if MATCHUP_SEPARATOR not in text:
        return None
    parts = [p.strip() for p in text.split(MATCHUP_SEPARATOR)]
    if len(parts) != 2 or not all(parts) or parts[0] == parts[1]:
        return None
    return set(parts)


def infer_team_abbreviations(
    rows: Iterable[Mapping[str, Any]],
    *,
    known: Mapping[str, str] | None = None,
) -> TeamIdentityMap:
    """
    Work out each club's abbreviation from the matchups it appears in.

    `rows` are availability rows carrying `team` (a full club name) and
    `matchup` ("AWY@HOM"). Rows missing either, or whose matchup is malformed,
    contribute nothing rather than being guessed at.

    `known` is a reviewed name -> abbreviation mapping. It is applied first and
    is never overridden, and it also seeds elimination, so asserting one club
    of a stuck pair resolves its opponent too.
    """
    observed: dict[str, list[set[str]]] = {}
    per_matchup: dict[str, set[str]] = {}

    for row in rows or ():
        name = str(row.get("team") or "").strip()
        abbreviations = _matchup_abbreviations(row.get("matchup"))
        if not name or abbreviations is None:
            continue
        key = MATCHUP_SEPARATOR.join(sorted(abbreviations))
        observed.setdefault(name, []).append(abbreviations)
        per_matchup.setdefault(key, set()).add(name)

    candidates: dict[str, set[str]] = {}
    provenance: dict[str, str] = {}

    for name, seen in observed.items():
        intersection = set.intersection(*seen)
        candidates[name] = intersection
        provenance[name] = SOURCE_INFERRED

    # A reviewed assertion wins, and is recorded as such. It may name a club
    # that never appeared in these rows, which is how an operator pins a club
    # the reports have not yet distinguished.
    for name, abbreviation in (known or {}).items():
        cleaned = str(name or "").strip()
        value = str(abbreviation or "").strip().upper()
        if not cleaned or not value:
            continue
        candidates[cleaned] = {value}
        provenance[cleaned] = SOURCE_REVIEWED

    # ELIMINATION. Two clubs in one matchup cannot be the same abbreviation.
    for _ in range(MAX_PROPAGATION_ROUNDS):
        changed = False
        for names in per_matchup.values():
            settled = {
                next(iter(candidates[n]))
                for n in names
                if n in candidates and len(candidates[n]) == 1
            }
            for name in names:
                current = candidates.get(name)
                if current is None or len(current) == 1:
                    continue
                # Never strip a reviewed club's only value, and never empty a
                # candidate set: an empty set would read as "no such club"
                # where the truth is "these constraints disagree".
                reduced = current - settled
                if reduced and reduced != current:
                    candidates[name] = reduced
                    changed = True
        if not changed:
            break

    resolved = {n: next(iter(a)) for n, a in candidates.items() if len(a) == 1}
    ambiguous = {
        n: tuple(sorted(a)) for n, a in candidates.items() if len(a) != 1
    }

    # Two clubs resolving to ONE abbreviation is a contradiction, not a map.
    # It means the report named the same club two ways, or a reviewed entry is
    # wrong. Both are reported rather than silently collapsed, because the
    # merge downstream would then attach one club's absences to the other.
    taken: dict[str, list[str]] = {}
    for name, abbreviation in resolved.items():
        taken.setdefault(abbreviation, []).append(name)
    for abbreviation, names in taken.items():
        if len(names) > 1:
            logger.warning(
                "team_identity: %s claimed by %d clubs (%s); all are left "
                "unresolved rather than attaching one club's absences to "
                "another", abbreviation, len(names), ", ".join(sorted(names)),
            )
            for name in names:
                ambiguous[name] = (abbreviation,)
                resolved.pop(name, None)

    logger.info(
        "team_identity: resolved %d club(s), %d ambiguous, from %d named in "
        "the reports", len(resolved), len(ambiguous), len(observed),
    )
    return TeamIdentityMap(
        resolved=resolved, ambiguous=ambiguous,
        provenance={n: provenance.get(n, SOURCE_INFERRED) for n in candidates},
    )
