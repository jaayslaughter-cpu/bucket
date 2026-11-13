"""
tests/test_availability_bridge.py -- the archive filled up and nothing read it.

WHAT THIS CLOSES. `model_input_archive` archived official availability reports
nightly; `features/absences.py` read a cached inactive list that nothing wrote.
So every BBS_ column stayed null and `teammate_cascade` reported
DATA_NOT_AVAILABLE on every row -- usage redistribution, which is one of the
largest effects in player props, contributed nothing.

The archive refused the join on purpose: "This does NOT write the existing
inactive cache, whose missing-coverage semantics differ." That refusal was
right, and these tests are mostly about the two ways of being wrong it was
protecting against -- both of which produce a plausible 0 where the truth is
"we do not know".

RESEARCH_ONLY. Absences. No wager.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.ingestion.availability_bridge import (
    DEFAULT_OUT_STATUSES,
    NOT_SUBMITTED,
    availability_absences,
)
from src.ingestion.inactive_players import attach_absence_features
from src.ingestion.team_identity import (
    SOURCE_INFERRED,
    SOURCE_REVIEWED,
    infer_team_abbreviations,
)


def report_row(matchup, team, player, status, date="2026-11-13"):
    return {"game_date": date, "matchup": matchup, "team": team,
            "player_name": player, "status": status, "reason": "Injury/Illness"}


def snapshot(rows, status="OK"):
    return {"kind": "availability", "status": status, "rows": rows,
            "observed_at_utc": "2026-11-13T01:15:00+00:00",
            "collected_at_utc": "2026-11-13T01:20:00+00:00",
            "source": "official", "source_url": "https://example.test/r.pdf"}


def panel_row(game_id, team, opponent, is_home, player_id, player_name,
              date="2026-11-13", minutes=28.0):
    return {"GAME_ID": game_id, "GAME_DATE": date, "TEAM_ABBREVIATION": team,
            "OPPONENT_ABBREVIATION": opponent, "IS_HOME": is_home,
            "PLAYER_ID": player_id, "PLAYER_NAME": player_name, "MIN": minutes}


# ---------------------------------------------------------------------------
# team identity, derived rather than remembered
# ---------------------------------------------------------------------------


def test_a_club_in_two_matchups_is_pinned_by_intersection() -> None:
    """{ATL, CHI} & {CHI, MIN} = {CHI}."""
    identity = infer_team_abbreviations([
        report_row("ATL@CHI", "Chicago Bulls", "A", "OUT"),
        report_row("CHI@MIN", "Chicago Bulls", "A", "OUT"),
    ])
    assert identity.abbreviation("Chicago Bulls") == "CHI"
    assert identity.provenance["Chicago Bulls"] == SOURCE_INFERRED


def test_a_resolved_club_eliminates_its_opponents_candidate() -> None:
    """Two clubs in one matchup cannot share an abbreviation."""
    identity = infer_team_abbreviations([
        report_row("ATL@CHI", "Chicago Bulls", "A", "OUT"),
        report_row("CHI@MIN", "Chicago Bulls", "A", "OUT"),
        report_row("ATL@CHI", "Atlanta Hawks", "B", "OUT"),
    ])
    assert identity.abbreviation("Chicago Bulls") == "CHI"
    assert identity.abbreviation("Atlanta Hawks") == "ATL"


def test_a_club_seen_in_one_matchup_stays_ambiguous() -> None:
    """
    WHY THERE IS NO HARDCODED TABLE OF THIRTY CLUBS. Nothing about one night
    says which of GSW/SAS is the Warriors, so the honest answer is neither.
    It resolves once the club appears against a different opponent.
    """
    identity = infer_team_abbreviations([
        report_row("GSW@SAS", "Golden State Warriors", "A", "OUT"),
    ])
    assert identity.abbreviation("Golden State Warriors") is None
    assert identity.ambiguous["Golden State Warriors"] == ("GSW", "SAS")


def test_a_reviewed_mapping_wins_and_seeds_elimination() -> None:
    """The operator's lever for a pair the reports cannot separate."""
    rows = [
        report_row("LAC@OKC", "LA Clippers", "A", "OUT"),
        report_row("LAC@OKC", "Oklahoma City Thunder", "B", "OUT"),
    ]
    assert infer_team_abbreviations(rows).abbreviation("LA Clippers") is None

    identity = infer_team_abbreviations(rows, known={"LA Clippers": "LAC"})
    assert identity.abbreviation("LA Clippers") == "LAC"
    assert identity.provenance["LA Clippers"] == SOURCE_REVIEWED
    # And elimination carries it to the opponent.
    assert identity.abbreviation("Oklahoma City Thunder") == "OKC"


def test_the_report_spells_the_clippers_without_los_angeles() -> None:
    """
    THE EXACT TRAP A FROM-MEMORY TABLE FALLS INTO, recorded because this
    project nearly wrote one. The official report says "LA Clippers". A table
    saying "Los Angeles Clippers" would never match, LAC would silently never
    resolve, and the feature would abstain for that club all season with no
    error anywhere. Deriving the map from the report's own text cannot make
    that mistake.
    """
    identity = infer_team_abbreviations(
        [report_row("LAC@OKC", "LA Clippers", "A", "OUT")],
        known={"LA Clippers": "LAC"},
    )
    assert identity.abbreviation("LA Clippers") == "LAC"
    assert identity.abbreviation("Los Angeles Clippers") is None


@pytest.mark.parametrize("matchup", ["", None, "CHI", "CHI@CHI", "A@B@C"])
def test_a_malformed_matchup_contributes_nothing(matchup) -> None:
    identity = infer_team_abbreviations([report_row(matchup, "Some Club", "A", "OUT")])
    assert identity.abbreviation("Some Club") is None


def test_two_clubs_claiming_one_abbreviation_are_both_refused() -> None:
    """
    A contradiction is not a map. Letting it stand would attach one club's
    absences to the other's roster, which is worse than abstaining.
    """
    identity = infer_team_abbreviations(
        [report_row("ATL@CHI", "Chicago Bulls", "A", "OUT")],
        known={"Chicago Bulls": "CHI", "Chicago Bulls Reserve": "CHI"},
    )
    assert identity.abbreviation("Chicago Bulls") is None
    assert identity.abbreviation("Chicago Bulls Reserve") is None


# ---------------------------------------------------------------------------
# only OUT is an absence
# ---------------------------------------------------------------------------


def test_only_out_becomes_an_absence() -> None:
    """
    DOUBTFUL is a probability. Mapping it to "teammates out" would publish a
    guess as a fact, in a column that feeds usage redistribution, for a player
    who may take thirty minutes.
    """
    assert DEFAULT_OUT_STATUSES == ("OUT",)
    rows = [
        report_row("CLE@MIA", "Cleveland Cavaliers", "Out Guy", "OUT"),
        report_row("CLE@MIA", "Cleveland Cavaliers", "Doubtful Guy", "DOUBTFUL"),
        report_row("CLE@MIA", "Cleveland Cavaliers", "Quest Guy", "QUESTIONABLE"),
        report_row("DAL@CLE", "Cleveland Cavaliers", "Out Guy", "OUT"),
    ]
    panel = pd.DataFrame([
        panel_row("1", "CLE", "MIA", False, "p1", "Out Guy"),
        panel_row("1", "CLE", "MIA", False, "p2", "Doubtful Guy"),
        panel_row("1", "CLE", "MIA", False, "p3", "Quest Guy"),
        panel_row("1", "CLE", "MIA", False, "p9", "Played Guy"),
    ])
    result = availability_absences([snapshot(rows)], panel,
                                   known_teams={"Cleveland Cavaliers": "CLE"})
    assert result.report["n_absences"] == 1
    assert result.inactives["PLAYER_ID"].tolist() == ["p1"]


def test_including_doubtful_is_possible_so_it_can_be_measured() -> None:
    """`out_statuses` is a parameter so `scripts/feature_ab.py` can answer
    whether DOUBTFUL earns inclusion, rather than this module assuming it."""
    rows = [
        report_row("CLE@MIA", "Cleveland Cavaliers", "Out Guy", "OUT"),
        report_row("CLE@MIA", "Cleveland Cavaliers", "Doubtful Guy", "DOUBTFUL"),
    ]
    panel = pd.DataFrame([
        panel_row("1", "CLE", "MIA", False, "p1", "Out Guy"),
        panel_row("1", "CLE", "MIA", False, "p2", "Doubtful Guy"),
    ])
    result = availability_absences(
        [snapshot(rows)], panel, known_teams={"Cleveland Cavaliers": "CLE"},
        out_statuses=("OUT", "DOUBTFUL"),
    )
    assert result.report["n_absences"] == 2
    assert set(result.inactives["PLAYER_ID"]) == {"p1", "p2"}


# ---------------------------------------------------------------------------
# THE TWO WAYS OF FABRICATING A ZERO
# ---------------------------------------------------------------------------


def test_a_club_that_did_not_file_reads_unknown_rather_than_zero() -> None:
    """
    THE REASON THE ARCHIVE REFUSED TO WRITE THE INACTIVE CACHE.

    `attach_absence_features` decided coverage per GAME, which is right for the
    inactive list (fetched per game, both rosters at once) and wrong for a
    report filed per team. Under the per-game rule the non-submitting club's
    rows are filled with 0 and assert a full-strength roster from no evidence.
    """
    rows = [
        report_row("CLE@MIA", "Cleveland Cavaliers", "Out Guy", "OUT"),
        # Miami states outright that it has not filed.
        report_row("CLE@MIA", "Miami Heat", None, NOT_SUBMITTED),
    ]
    panel = pd.DataFrame([
        panel_row("1", "CLE", "MIA", False, "p1", "Out Guy"),
        panel_row("1", "CLE", "MIA", False, "p9", "Played Guy"),
        panel_row("1", "MIA", "CLE", True, "m1", "Heat Guy"),
    ])
    result = availability_absences(
        [snapshot(rows)], panel,
        known_teams={"Cleveland Cavaliers": "CLE", "Miami Heat": "MIA"},
    )
    out = attach_absence_features(
        panel, result.inactives, covered_team_games=result.covered_team_games
    )
    cle = out[out["TEAM_ABBREVIATION"] == "CLE"]["BBS_TEAMMATES_OUT"]
    mia = out[out["TEAM_ABBREVIATION"] == "MIA"]["BBS_TEAMMATES_OUT"]
    assert (cle == 1).all(), cle.tolist()
    assert mia.isna().all(), (
        "a club that did not file was reported as having nobody out"
    )
    assert (
        out[out["TEAM_ABBREVIATION"] == "MIA"]["BBS_INACTIVE_SOURCE"]
        == "DATA_NOT_AVAILABLE"
    ).all()


def test_a_club_whose_absence_did_not_crosswalk_reads_unknown_too() -> None:
    """
    THE SECOND FABRICATION, found by running the bridge against the real
    2026-11-13 report rather than by reasoning about it.

    Filing is not the same as being understood. Miami filed that night, and
    against a panel that did not carry its players every Miami OUT row landed
    in `unresolved` -- so the pair counted as covered and the count collapsed
    to 0, reporting a healthy Miami roster from absences we had merely failed
    to resolve. A pair is covered only when EVERY absence it filed resolved.
    """
    rows = [
        report_row("CLE@MIA", "Cleveland Cavaliers", "Out Guy", "OUT"),
        # Miami filed a real absence, but this player is not in the panel.
        report_row("CLE@MIA", "Miami Heat", "Unknown To Panel", "OUT"),
    ]
    panel = pd.DataFrame([
        panel_row("1", "CLE", "MIA", False, "p1", "Out Guy"),
        panel_row("1", "MIA", "CLE", True, "m1", "Heat Guy"),
    ])
    result = availability_absences(
        [snapshot(rows)], panel,
        known_teams={"Cleveland Cavaliers": "CLE", "Miami Heat": "MIA"},
    )
    assert result.report["coverage_counts"]["spoiled_by_unresolved_absence"] == 1
    out = attach_absence_features(
        panel, result.inactives, covered_team_games=result.covered_team_games
    )
    mia = out[out["TEAM_ABBREVIATION"] == "MIA"]["BBS_TEAMMATES_OUT"]
    assert mia.isna().all(), (
        "a club whose filed absence did not crosswalk was counted as having "
        "nobody out"
    )


def test_the_per_game_default_is_unchanged_for_the_inactive_list() -> None:
    """
    The new parameter must narrow nothing when absent. The official inactive
    list depends on the per-game rule: a club with nobody out in a fetched
    game is a real 0, and breaking that would make every healthy roster
    abstain.
    """
    panel = pd.DataFrame([
        panel_row("1", "CLE", "MIA", False, "p1", "A"),
        panel_row("1", "MIA", "CLE", True, "m1", "B"),
    ])
    inactives = pd.DataFrame(
        [{"GAME_ID": "1", "PLAYER_ID": "p1", "TEAM_ABBREVIATION": "CLE"}]
    )
    out = attach_absence_features(panel, inactives)
    assert (out[out["TEAM_ABBREVIATION"] == "CLE"]["BBS_TEAMMATES_OUT"] == 1).all()
    assert (out[out["TEAM_ABBREVIATION"] == "MIA"]["BBS_TEAMMATES_OUT"] == 0).all(), (
        "the per-game default must still read a fetched game with nobody out "
        "as a real zero"
    )


def test_coverage_can_only_narrow_never_widen() -> None:
    """Naming a pair that contributed no absence still reads 0 -- that is what
    'this club filed and listed nobody' means -- but a pair left out of the
    list can never be filled."""
    panel = pd.DataFrame([
        panel_row("1", "CLE", "MIA", False, "p1", "A"),
        panel_row("1", "MIA", "CLE", True, "m1", "B"),
    ])
    inactives = pd.DataFrame(
        [{"GAME_ID": "1", "PLAYER_ID": "p1", "TEAM_ABBREVIATION": "CLE"}]
    )
    out = attach_absence_features(
        panel, inactives, covered_team_games=(("1", "CLE"), ("1", "MIA")),
    )
    assert (out[out["TEAM_ABBREVIATION"] == "MIA"]["BBS_TEAMMATES_OUT"] == 0).all()


# ---------------------------------------------------------------------------
# abstention
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dropped", ["GAME_ID", "TEAM_ABBREVIATION",
                                     "OPPONENT_ABBREVIATION", "IS_HOME",
                                     "PLAYER_ID", "PLAYER_NAME", "GAME_DATE"])
def test_a_panel_missing_a_crosswalk_column_abstains_by_name(dropped) -> None:
    panel = pd.DataFrame([panel_row("1", "CLE", "MIA", False, "p1", "A")]).drop(
        columns=[dropped]
    )
    result = availability_absences([snapshot([])], panel)
    assert result.report["status"] == "DATA_NOT_AVAILABLE"
    assert dropped in result.report["reason"]
    assert result.inactives.empty


def test_a_failed_report_contributes_no_coverage() -> None:
    """A snapshot that is not OK is not evidence about anybody."""
    rows = [report_row("CLE@MIA", "Cleveland Cavaliers", "Out Guy", "OUT")]
    panel = pd.DataFrame([panel_row("1", "CLE", "MIA", False, "p1", "Out Guy")])
    result = availability_absences(
        [snapshot(rows, status="FAILED")], panel,
        known_teams={"Cleveland Cavaliers": "CLE"},
    )
    assert result.covered_team_games == ()
    assert result.inactives.empty


def test_an_unresolved_club_contributes_nothing_rather_than_a_wrong_roster() -> None:
    """A club the identity map could not pin must not have its absences
    attached to whichever abbreviation happened to be in the matchup."""
    rows = [report_row("GSW@SAS", "Golden State Warriors", "Out Guy", "OUT")]
    panel = pd.DataFrame([
        panel_row("1", "GSW", "SAS", False, "p1", "Out Guy"),
        panel_row("1", "SAS", "GSW", True, "s1", "Spur"),
    ])
    result = availability_absences([snapshot(rows)], panel)
    assert result.inactives.empty
    assert result.covered_team_games == ()


def test_the_bridge_never_changes_the_panels_row_count() -> None:
    panel = pd.DataFrame([
        panel_row("1", "CLE", "MIA", False, "p1", "Out Guy"),
        panel_row("1", "CLE", "MIA", False, "p9", "Played"),
    ])
    rows = [report_row("CLE@MIA", "Cleveland Cavaliers", "Out Guy", "OUT")]
    result = availability_absences([snapshot(rows)], panel,
                                   known_teams={"Cleveland Cavaliers": "CLE"})
    out = attach_absence_features(
        panel, result.inactives, covered_team_games=result.covered_team_games
    )
    assert len(out) == len(panel)


# ---------------------------------------------------------------------------
# IDENTITY ACCUMULATES; ABSENCES DO NOT
# ---------------------------------------------------------------------------


def test_identity_resolves_across_reports_that_one_report_cannot() -> None:
    """
    THE DIFFERENCE BETWEEN THIS WORKING AND ABSTAINING FOREVER.

    `team_identity` separates two clubs by seeing them against DIFFERENT
    opponents. A single report cannot tell GSW from SAS when they play only
    each other, so a bridge that infers the map from one report leaves those
    clubs unresolved -- and since the as-of window is six hours and `as_of`
    selects the latest full report, inferring from the window means inferring
    from exactly one. Every run would start over and most clubs would never
    resolve.

    Across two nights the constraint closes: Golden State seen in GSW@SAS and
    then GSW@LAL can only be GSW.
    """
    one_night = [report_row("GSW@SAS", "Golden State Warriors", "A", "OUT")]
    assert infer_team_abbreviations(one_night).abbreviation(
        "Golden State Warriors"
    ) is None

    two_nights = one_night + [
        report_row("GSW@LAL", "Golden State Warriors", "A", "OUT", date="2026-11-15"),
    ]
    assert infer_team_abbreviations(two_nights).abbreviation(
        "Golden State Warriors"
    ) == "GSW"


def test_the_archive_reader_verifies_checksums_and_filters_nothing() -> None:
    """
    `all_snapshots` is the unfiltered reference read, and it must still refuse
    a corrupted file -- the integrity check cannot exist in `as_of` and not
    here, which is why `as_of` now selects from this.
    """
    import json as _json

    from src.ingestion.model_input_archive import InputArchive, InputError

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        archive = InputArchive(tmp)
        archive.write(
            kind="availability", source="official",
            source_url="https://example.test/r.pdf",
            observed_at="2026-11-13T01:15:00+00:00",
            collected_at="2026-11-13T01:20:00+00:00",
            status="OK",
            rows=[report_row("CLE@MIA", "Cleveland Cavaliers", "A", "OUT")],
        )
        assert len(archive.all_snapshots("availability")) == 1

        # Corrupt it: the content no longer hashes to its own filename.
        import pathlib
        target = next((pathlib.Path(tmp) / "snapshots" / "availability").glob("*.json"))
        payload = _json.loads(target.read_text())
        payload["status"] = "FAILED"
        target.write_text(_json.dumps(payload))
        with pytest.raises(InputError, match="checksum"):
            archive.all_snapshots("availability")


def test_an_unknown_snapshot_kind_is_refused() -> None:
    import tempfile

    from src.ingestion.model_input_archive import InputArchive, InputError

    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(InputError, match="unknown snapshot kind"):
            InputArchive(tmp).all_snapshots("not_a_kind")


def test_the_loader_itself_runs_and_abstains_without_an_archive(monkeypatch) -> None:
    """
    THE BUG THIS EXISTS TO PREVENT, which 30 passing tests did not see.

    Every test above calls `availability_absences` directly. The function the
    feature layer actually calls is
    `absences.load_availability_absences`, and its first spelling was
    `Path(resolve_state_root()) / "model_inputs"` -- but
    `resolve_state_root` returns `(path, how)`, so that raises TypeError on a
    tuple. The whole path sits behind a default-off toggle, so it would have
    raised on the day someone enabled it, not on the day it was written. This
    drives the real entry point.
    """
    from src.features import absences as absences_module

    panel = pd.DataFrame([panel_row("1", "CLE", "MIA", False, "p1", "A")])
    assert absences_module.load_availability_absences(panel) is None


def test_the_layer_abstains_cleanly_with_the_toggle_on_and_no_archive(
    monkeypatch, tmp_path,
) -> None:
    """
    Enabling the fallback on a deployment whose archive is empty -- which is
    every deployment until the collection job has run -- must abstain by name,
    not raise into the feature build.
    """
    from src.features import absences as absences_module

    monkeypatch.setenv(absences_module.ENV_AVAILABILITY_FALLBACK, "1")
    monkeypatch.setenv("PROPIQ_STATE_DIR", str(tmp_path))
    panel = pd.DataFrame([panel_row("1", "CLE", "MIA", False, "p1", "A")])
    out = absences_module.attach_absence_features_layer(
        panel, cache_root=tmp_path / "no-cache"
    )
    assert out["BBS_INACTIVE_SOURCE"].iloc[0] == "DATA_NOT_AVAILABLE"
    assert out["BBS_TEAMMATES_OUT"].isna().all()
    assert len(out) == len(panel)
