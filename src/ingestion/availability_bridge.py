"""Turn archived availability reports into the absence input the panel reads.

WHAT WAS MISSING, precisely.

`model_input_archive` archives official availability reports with honest
timestamps and refuses to guess identities. `features/absences.py` reads a
cached inactive list and hands it to `inactive_players.attach_absence_features`,
which produces `BBS_TEAMMATES_OUT` and `BBS_VACATED_USAGE` -- the usage
redistribution signal `teammate_cascade` consumes. Nothing joined the two, so
the archive filled up while every absence column stayed null.

The archive's own docstring declines the join and says why:

    This does NOT write the existing inactive cache, whose missing-coverage
    semantics differ.

THAT REFUSAL WAS CORRECT, and this module exists to satisfy it rather than
override it. `attach_absence_features` decided coverage per GAME: any row of a
game appearing anywhere in the inactive frame was filled, so a team with no
absences read a true 0. That is right for the official INACTIVE LIST, which is
fetched per game and returns both rosters at once. The availability report is
filed PER TEAM, and one team can submit while its opponent has not. Under
per-game coverage the non-submitting team's rows would be filled with 0 and
assert a full-strength roster from no evidence -- on the 2026-11-13 01:15AM
report that is 10 of 99 rows.

So `attach_absence_features` now takes `covered_team_games`, and this module
supplies exactly the (game, team) pairs that actually filed. Everything else
stays `pd.NA` and reports DATA_NOT_AVAILABLE, which is what "we do not know"
has always meant in that layer.

ONLY `OUT` COUNTS AS AN ABSENCE, and this is the decision most worth arguing
with. The report's vocabulary is OUT, DOUBTFUL, QUESTIONABLE, PROBABLE,
AVAILABLE. Only OUT is a statement that the player will not play. DOUBTFUL is
a probability, and mapping it to an absence would publish a guess as a fact --
in a column named "teammates out", feeding a usage-redistribution feature, for
a player who may well take 30 minutes. `out_statuses` is a parameter so that
including DOUBTFUL can be MEASURED by `scripts/feature_ab.py` rather than
assumed, and the report counts every status either way so the question stays
answerable.

LEAKAGE. Every absence here is drawn from a report published and collected
before the cutoff -- `archive.as_of(mode="system")` enforces both -- and the
player identities come from PRIOR panel appearances, because the panel holds
only players who appeared and an absent player has no row for the game he
missed. Nothing reads the game being predicted.

RESEARCH_ONLY. This resolves absences. It does not price or place a wager.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from src.ingestion.model_input_archive import build_absence_export
from src.ingestion.team_identity import TeamIdentityMap, infer_team_abbreviations

logger = logging.getLogger(__name__)

RESEARCH_STATUS = "RESEARCH_ONLY"

#: The only status this treats as an absence. See the module docstring.
DEFAULT_OUT_STATUSES: tuple[str, ...] = ("OUT",)

#: The marker a team that has not filed carries. Such a pair is NOT covered.
NOT_SUBMITTED = "NOT_YET_SUBMITTED"

PANEL_COLUMNS = (
    "GAME_ID", "GAME_DATE", "TEAM_ABBREVIATION", "OPPONENT_ABBREVIATION",
    "IS_HOME", "PLAYER_ID", "PLAYER_NAME",
)


@dataclass(frozen=True)
class BridgeResult:
    """The absence input, the coverage it is entitled to, and what it cost."""

    inactives: pd.DataFrame
    covered_team_games: tuple[tuple[str, str], ...] = ()
    identity: TeamIdentityMap | None = None
    report: dict[str, Any] = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        return not self.inactives.empty or bool(self.covered_team_games)


def _matchup(team: Any, opponent: Any, is_home: Any) -> str | None:
    """'MIL@ORL' -- away first, matching the official report's spelling."""
    a, b = str(team or "").strip().upper(), str(opponent or "").strip().upper()
    if not a or not b or a == b:
        return None
    try:
        home = bool(is_home)
    except (TypeError, ValueError):
        return None
    return f"{b}@{a}" if home else f"{a}@{b}"


def _panel_vocabulary(
    panel: pd.DataFrame, identity: TeamIdentityMap,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Re-express the panel in the REPORT's vocabulary so its crosswalk can read it.

    `build_absence_export` keys on the report's own `(game_date, matchup,
    team)` and `(team, player_name)`, where `team` is a full club name. The
    panel speaks abbreviations, so each panel row is labelled with the club
    name the identity map resolved -- and a club the map could not resolve
    contributes nothing, which is how an unresolved club abstains instead of
    attaching its absences to the wrong roster.
    """
    by_abbreviation = {v: k for k, v in (identity.resolved or {}).items()}
    games: dict[tuple[str, str, str], dict[str, Any]] = {}
    players: dict[tuple[str, str], dict[str, Any]] = {}
    skipped = {"unresolved_club": 0, "no_matchup": 0}

    dates = pd.to_datetime(panel["GAME_DATE"], errors="coerce")
    for position, (_, row) in enumerate(panel.iterrows()):
        abbreviation = str(row.get("TEAM_ABBREVIATION") or "").strip().upper()
        club = by_abbreviation.get(abbreviation)
        if not club:
            skipped["unresolved_club"] += 1
            continue
        matchup = _matchup(
            abbreviation, row.get("OPPONENT_ABBREVIATION"), row.get("IS_HOME")
        )
        when = dates.iloc[position]
        if matchup is None or pd.isna(when):
            skipped["no_matchup"] += 1
            continue
        game_date = when.strftime("%Y-%m-%d")
        game_id = str(row.get("GAME_ID") or "").strip()
        if game_id:
            games[(game_date, matchup, club)] = {
                "game_date": game_date, "matchup": matchup, "team": club,
                "game_id": game_id, "team_abbreviation": abbreviation,
            }
        name, player_id = row.get("PLAYER_NAME"), row.get("PLAYER_ID")
        if name is not None and player_id is not None and str(player_id).strip():
            players[(club, str(name))] = {
                "team": club, "player_name": str(name),
                "player_id": str(player_id).strip(),
            }
    return list(games.values()), list(players.values()), skipped


def _covered_pairs(
    snapshots: Sequence[Mapping[str, Any]],
    games: Sequence[Mapping[str, Any]],
    identity: TeamIdentityMap,
    unresolved: Sequence[Mapping[str, Any]] = (),
    wanted: frozenset[str] = frozenset(DEFAULT_OUT_STATUSES),
) -> tuple[set[tuple[str, str]], dict[str, int]]:
    """The (game_id, abbreviation) pairs that actually filed a report.

    A club is covered for a game when it appears on an OK snapshot for that
    game with at least one row that is not NOT_YET_SUBMITTED. Appearing only
    as NOT_YET_SUBMITTED is the report stating outright that it has not filed.

    A CLUB THAT FILED AND DESIGNATED NOBODY is indistinguishable here from a
    club the report never mentions, because both contribute no row -- so both
    are left uncovered and read `pd.NA`. That is a deliberate undercount: it
    costs coverage on fully healthy rosters and it never invents a 0 for a
    roster whose status is unknown. The alternative mistakes silence for
    health, which is the fabrication this whole path exists to avoid.

    A CLUB WHOSE ABSENCES DID NOT ALL CROSSWALK is also uncovered, and this
    one was found by running the bridge against the real 2026-11-13 report
    rather than by reasoning about it. Filing is not the same as being
    understood: Miami filed that night and, against a panel that did not carry
    its players, every one of its OUT rows landed in `unresolved` -- so the
    pair counted as covered and the count collapsed to 0, reporting a healthy
    Miami roster from five absences we had simply failed to resolve. A pair is
    therefore covered only when EVERY absence row it filed resolved to a panel
    identity. One unresolved absence makes the count unknowable, and unknown
    is `pd.NA`, not zero.
    """
    index = {
        (g["game_date"], g["matchup"], g["team"]): (g["game_id"], g["team_abbreviation"])
        for g in games
    }
    covered: set[tuple[str, str]] = set()
    counts = {"filed": 0, "not_submitted": 0, "unmatched": 0}

    for snapshot in snapshots or ():
        if snapshot.get("status") != "OK":
            continue
        for row in snapshot.get("rows") or ():
            club = str(row.get("team") or "").strip()
            key = (row.get("game_date"), row.get("matchup"), club)
            if str(row.get("status") or "").strip().upper() == NOT_SUBMITTED:
                counts["not_submitted"] += 1
                continue
            located = index.get(key)
            if located is None:
                counts["unmatched"] += 1
                continue
            covered.add(located)
            counts["filed"] += 1

    # Remove any pair that filed an absence we could not resolve.
    spoiled: set[tuple[str, str]] = set()
    for row in unresolved or ():
        if str(row.get("status") or "").strip().upper() not in wanted:
            continue
        key = (row.get("game_date"), row.get("matchup"), str(row.get("team") or "").strip())
        located = index.get(key)
        if located is not None and located in covered:
            spoiled.add(located)
    if spoiled:
        logger.info(
            "availability_bridge: %d (game, team) pair(s) filed an absence that "
            "did not crosswalk; left uncovered rather than counted short",
            len(spoiled),
        )
    counts["spoiled_by_unresolved_absence"] = len(spoiled)
    return covered - spoiled, counts


def availability_absences(
    snapshots: Sequence[Mapping[str, Any]],
    panel: pd.DataFrame,
    *,
    identity: TeamIdentityMap | None = None,
    known_teams: Mapping[str, str] | None = None,
    out_statuses: Iterable[str] = DEFAULT_OUT_STATUSES,
) -> BridgeResult:
    """
    Build the inactive-shaped frame and the coverage it is entitled to.

    `snapshots` come from `InputArchive.as_of("availability", cutoff)`, which
    has already enforced that each was published AND collected before the
    cutoff. `panel` supplies the identities; its rows are prior appearances,
    never the game being predicted.

    The identity map is inferred from the snapshots' own matchups when none is
    supplied -- see `team_identity` -- and `known_teams` is the reviewed
    override for clubs the reports cannot yet separate.
    """
    wanted = {str(s).strip().upper() for s in out_statuses if str(s).strip()}
    missing = [c for c in PANEL_COLUMNS if c not in panel.columns]
    empty = pd.DataFrame(columns=["GAME_ID", "PLAYER_ID", "TEAM_ABBREVIATION"])
    if missing:
        return BridgeResult(
            inactives=empty,
            report={"status": "DATA_NOT_AVAILABLE",
                    "reason": f"panel is missing {sorted(missing)}",
                    "research_status": RESEARCH_STATUS},
        )

    rows_for_identity = [
        row for snapshot in (snapshots or ()) for row in (snapshot.get("rows") or ())
    ]
    resolved_identity = identity or infer_team_abbreviations(
        rows_for_identity, known=known_teams
    )

    games, players, skipped = _panel_vocabulary(panel, resolved_identity)
    exported, unresolved = build_absence_export(snapshots, games, players)
    covered, coverage_counts = _covered_pairs(
        snapshots, games, resolved_identity, unresolved, frozenset(wanted)
    )

    absences = [r for r in exported if str(r.get("STATUS") or "").upper() in wanted]
    inactives = (
        pd.DataFrame(absences)[["GAME_ID", "PLAYER_ID", "TEAM_ABBREVIATION"]]
        if absences else empty
    )
    # Only pairs we have coverage for may contribute an absence; an absence on
    # an uncovered pair would be filled against a pd.NA count downstream.
    if not inactives.empty:
        keep = [
            (str(g), str(t)) in covered
            for g, t in zip(inactives["GAME_ID"], inactives["TEAM_ABBREVIATION"])
        ]
        inactives = inactives.loc[keep].reset_index(drop=True)

    status_counts: dict[str, int] = {}
    for row in exported:
        key = str(row.get("STATUS") or "UNKNOWN").upper()
        status_counts[key] = status_counts.get(key, 0) + 1

    report = {
        "status": "OK" if covered else "DATA_NOT_AVAILABLE",
        "research_status": RESEARCH_STATUS,
        "out_statuses": sorted(wanted),
        "n_absences": int(len(inactives)),
        "n_covered_team_games": len(covered),
        "status_counts": status_counts,
        "panel_rows_skipped": skipped,
        "coverage_counts": coverage_counts,
        "n_unresolved_report_rows": len(unresolved),
        "identity": resolved_identity.summary(),
    }
    logger.info(
        "availability_bridge: %d absence(s) over %d covered (game, team) pair(s); "
        "%d report row(s) unresolved; %d club(s) still ambiguous",
        len(inactives), len(covered), len(unresolved),
        resolved_identity.summary()["n_ambiguous"],
    )
    return BridgeResult(
        inactives=inactives,
        covered_team_games=tuple(sorted(covered)),
        identity=resolved_identity,
        report=report,
    )
