"""
tests/test_pbp_shot_layers_wired.py — the pbp and shot_zones layers in the
default builder.

WHY THESE EXIST. Both layers were reachable only through
`scripts/feature_ab.py --layer pbp|shot_zones`, which measures a layer with it
on and off. They were wired into `build_feature_matrix` on request,
2026-10-25, WITHOUT that measurement having been run on this panel. So the
properties asserted here are not "the layers help" — nobody knows that yet —
they are the properties that make the wiring safe either way:

  * supplying a frame attaches its columns, and NOT supplying one leaves the
    matrix byte-identical to before, so the deployment is unchanged until
    somebody puts the data there;
  * the schema version differs between those two cases, so an artifact trained
    with the layers cannot be confused with one trained without;
  * no same-game value survives into a feature, which is the one thing a
    play-by-play layer can get catastrophically wrong.

AND ONE HAZARD, asserted so it stays visible: `resolve_feature_cols` DROPS a
feature the panel lacks, with a warning and no error. An artifact trained with
these columns and scored on a panel without them therefore degrades quietly.
That is a real trap in this deployment, because the frames come from
`data/external/**`, which is gitignored and absent on a container.

RESEARCH_ONLY.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.features.builder import build_feature_matrix, resolved_feature_schema_version
from src.models.data_audit import make_demo_panel


@pytest.fixture
def panel() -> pd.DataFrame:
    return make_demo_panel(n_players=6, n_games=14)


def _pbp_summaries(panel: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (game, player), in the shape `attach_pbp_rolling_features`
    expects: `gameId` / `personId` plus the rate columns.
    """
    from src.features.pbp import PBP_RATE_COLS

    rows = []
    for i, (_, r) in enumerate(panel.iterrows()):
        row = {"gameId": str(r["GAME_ID"]), "personId": str(r["PLAYER_ID"])}
        # Values that vary per row, so a shift that failed to happen would
        # show up as a feature equal to its own game's value.
        for j, col in enumerate(PBP_RATE_COLS):
            row[col] = 0.1 + ((i + j) % 7) / 10.0
        rows.append(row)
    return pd.DataFrame(rows)


def _shot_summaries(panel: pd.DataFrame) -> pd.DataFrame:
    from src.features.shot_zones import SUMMARY_COLS

    rows = []
    for i, (_, r) in enumerate(panel.iterrows()):
        row = {"_GAME": str(r["GAME_ID"]), "_PLAYER": str(r["PLAYER_ID"])}
        for j, col in enumerate(SUMMARY_COLS):
            row[col] = 0.2 + ((i + j) % 5) / 10.0
        rows.append(row)
    return pd.DataFrame(rows)


# --- the deployment is unchanged until the data is there ------------------


def test_without_the_frames_no_layer_column_exists_at_all(panel):
    """
    THE PROPERTY THAT MAKES THIS SAFE TO SHIP TODAY, asserted ABSOLUTELY.

    The container has no play-by-play or shot export — `data/external/**` is
    gitignored and absent — so the wiring must add nothing there.

    THE FIRST VERSION OF THIS TEST COULD NOT FAIL. It compared
    `build_feature_matrix(panel)` against the same call with both frames
    passed as None — but those take the SAME code path, so a mutation that
    attached a column unconditionally changed both sides equally and the
    comparison still passed. Verified: `if True:` plus an unconditional
    `df["PBP_RIM_RATE_L5"] = nan` left all ten tests green.

    So the assertion is now against zero, not against another call.
    """
    bare = build_feature_matrix(panel)
    leaked = [c for c in bare.columns if c.startswith(("PBP_", "SZ_"))]
    assert not leaked, (
        f"the layers attached {leaked} with no frame supplied; the live "
        f"deployment's feature matrix just changed shape"
    )

    explicit_none = build_feature_matrix(
        panel, pbp_player_games=None, shot_summary=None
    )
    assert not [
        c for c in explicit_none.columns if c.startswith(("PBP_", "SZ_"))
    ]
    # And the two spellings of "absent" agree with each other.
    pd.testing.assert_frame_equal(bare, explicit_none)


def test_an_empty_frame_counts_as_absent(panel):
    """`is not None and not .empty` — an export that parsed to zero rows must
    skip exactly as a missing one does, not attach a column of NaN.

    Asserted absolutely for the same reason as the test above: comparing two
    calls that share a code path cannot catch a change to that path."""
    empty = build_feature_matrix(
        panel,
        pbp_player_games=pd.DataFrame(),
        shot_summary=pd.DataFrame(),
    )
    assert not [c for c in empty.columns if c.startswith(("PBP_", "SZ_"))]


# --- supplying a frame attaches its columns ------------------------------


def test_the_pbp_frame_attaches_pbp_columns(panel):
    out = build_feature_matrix(panel, pbp_player_games=_pbp_summaries(panel))
    attached = [c for c in out.columns if c.startswith("PBP_")]
    assert attached, "the pbp layer attached nothing"
    # ROLLED AND SHIFTED, not raw. A bare PBP_RIM_RATE column would be the
    # same-game value, which is the leak.
    assert any(c.endswith(("_L5", "_L10")) for c in attached)
    assert "PBP_RIM_RATE" not in out.columns, (
        "a same-game pbp column reached the matrix"
    )


def test_the_shot_frame_attaches_shot_columns(panel):
    out = build_feature_matrix(panel, shot_summary=_shot_summaries(panel))
    attached = [c for c in out.columns if c.startswith("SZ_")]
    assert attached, "the shot-zone layer attached nothing"
    assert "SZ_RIM_RATE" not in out.columns, (
        "a same-game shot column reached the matrix"
    )


def test_both_layers_can_attach_together(panel):
    out = build_feature_matrix(
        panel,
        pbp_player_games=_pbp_summaries(panel),
        shot_summary=_shot_summaries(panel),
    )
    assert [c for c in out.columns if c.startswith("PBP_")]
    assert [c for c in out.columns if c.startswith("SZ_")]


# --- the schema version keeps the two feature sets apart -----------------


def test_a_run_with_the_layers_does_not_claim_the_bare_schema_version(panel):
    """
    THE GUARD AGAINST A SILENT CONTRACT CHANGE. Two runs with different
    feature sets must not both report `fs_v1_shift1_l2`, or an artifact
    trained with these columns is indistinguishable from one trained without
    and `resolve_feature_cols` will quietly drop them at scoring time.
    """
    bare = build_feature_matrix(panel)
    rich = build_feature_matrix(
        panel,
        pbp_player_games=_pbp_summaries(panel),
        shot_summary=_shot_summaries(panel),
    )
    bare_version = bare["FEATURE_SCHEMA_VERSION"].iloc[0]
    rich_version = rich["FEATURE_SCHEMA_VERSION"].iloc[0]
    assert bare_version != rich_version
    assert "+layers." in rich_version


def test_the_layer_names_reach_the_schema_digest(panel):
    """The digest is of the attached layer NAMES, so it must change when the
    set changes — not merely be present."""
    only_pbp = resolved_feature_schema_version(["pbp"])
    only_shots = resolved_feature_schema_version(["shot_zones"])
    both = resolved_feature_schema_version(["pbp", "shot_zones"])
    assert len({only_pbp, only_shots, both}) == 3


# --- no same-game value survives -----------------------------------------


def test_no_lookahead_with_the_layers_on(panel):
    """
    The project's own leakage assertion, run against the richer matrix. A
    play-by-play layer describes what happened DURING a game; if the shift
    inside the attachers ever stopped happening, this is what catches it.
    """
    from src.features.builder import assert_no_lookahead

    out = build_feature_matrix(
        panel,
        pbp_player_games=_pbp_summaries(panel),
        shot_summary=_shot_summaries(panel),
    )
    assert_no_lookahead(out)


def test_a_players_first_game_has_no_pbp_history(panel):
    """
    Null, not self-describing. The shift happens before the rolling window, so
    the first row a player has cannot carry his own game's rate.
    """
    out = build_feature_matrix(panel, pbp_player_games=_pbp_summaries(panel))
    rolled = [c for c in out.columns if c.startswith("PBP_") and c.endswith("_L5")]
    assert rolled
    ordered = out.sort_values(["PLAYER_ID", "GAME_DATE"])
    first_rows = ordered.groupby("PLAYER_ID").head(1)
    for col in rolled:
        assert first_rows[col].isna().all(), (
            f"{col} is populated on a player's first game, so the shift did "
            f"not happen"
        )


# --- the hazard, asserted so it stays visible ----------------------------


def test_a_panel_missing_these_columns_drops_them_rather_than_failing(caplog):
    """
    THE TRAP THIS WIRING INTRODUCES, pinned so nobody rediscovers it in
    production. `resolve_feature_cols` drops a feature the panel lacks with a
    WARNING and no error. So an artifact trained with PBP columns, scored on a
    container that has no play-by-play export, scores on a narrower feature
    set than it was fitted on — and the only trace is a log line.

    This test does not fix that. It records it, so the next person wiring a
    layer knows the training and scoring paths must get the same frames.
    """
    import logging

    from src.models.compare import resolve_feature_cols

    frame = pd.DataFrame({"PTS_L5": [1.0, 2.0], "REB_L5": [3.0, 4.0]})
    wanted = ["PTS_L5", "REB_L5", "PBP_RIM_RATE_L10", "SZ_CORNER3_RATE_L10"]

    with caplog.at_level(logging.WARNING):
        present, absent = resolve_feature_cols(frame, wanted)

    assert present == ["PTS_L5", "REB_L5"]
    assert absent == ["PBP_RIM_RATE_L10", "SZ_CORNER3_RATE_L10"]
    assert "Dropping 2 feature(s)" in caplog.text
    # NOT zero-filled — that part is right and must stay right.
    assert "NOT zero-filled" in caplog.text
