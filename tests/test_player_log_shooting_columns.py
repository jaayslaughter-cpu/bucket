"""
The PTS model could not score a single row, and three places had to agree.

FOUND 2026-10-18, two days before opening night. Measured against a
2026-04-10 slate built from `player_game_logs`:

    REB  scored 325/325
    AST  scored 325/325
    PTS  MISSING ['TS_PCT_L10', 'TS_PCT_TREND', 'FT_RATE_L10']

`features/scoring_efficiency` declares REQUIRED_COLS = ("PTS", "FGA", "FTA")
and abstains without them. `player_game_logs` stored neither attempt column,
so the layer attached nothing, three features the PTS contract names were
absent, and `main`'s train/serve check skipped P(Over) for every player on
every slate. Not a freshness problem: the artifacts were fit on a panel built
from full box scores, so the features existed at training time and could not
exist at serving time.

THREE PLACES, AND MISSING ANY ONE OF THEM LOOKS EXACTLY LIKE SUCCESS:

  1. the schema            migrations/009 + the ORM model
  2. the WRITE path        `upsert_player_game_logs`, including the
                           ON CONFLICT set -- 214,381 rows already existed, so
                           every one took DO UPDATE and a column left out of
                           that set stays NULL for ever
  3. the READ path         `load_player_panel`, which builds the panel from an
                           explicit dict; a column absent there is absent from
                           the feature matrix however well it is stored

I missed (3) on the first pass: the migration applied, the backfill wrote
fga on all 214,381 rows, and PTS went on abstaining. Each of these tests
fails if one of the three regresses.

RESEARCH_ONLY. Box-score attempt counts.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MIGRATION = REPO_ROOT / "migrations" / "009_player_game_log_shooting.sql"


# --- 1. the schema ----------------------------------------------------------


def test_the_migration_adds_both_attempt_columns():
    sql = MIGRATION.read_text(encoding="utf-8")
    for col in ("fga", "fta"):
        assert f"ADD COLUMN IF NOT EXISTS {col} INTEGER" in sql


def test_the_columns_are_nullable_with_no_default():
    """
    A default of 0 is a fabrication with teeth here: TS% divides by
    (2 * (FGA + 0.44 * FTA)), so a backfilled zero is a division by zero or a
    silent efficiency figure for a game nobody shot in. The layer abstains on
    a null; it cannot abstain on a zero.
    """
    sql = MIGRATION.read_text(encoding="utf-8")
    assert "DEFAULT 0" not in sql.upper()
    assert "NOT NULL" not in sql.upper()


def test_the_orm_model_carries_them():
    from src.db.models import PlayerGameLog

    for col in ("fga", "fta"):
        assert col in PlayerGameLog.__table__.columns, f"{col} missing from the model"
        assert PlayerGameLog.__table__.columns[col].nullable


# --- 2. the write path ------------------------------------------------------


def test_the_write_path_sends_both_columns():
    from src.db import repository

    body = inspect.getsource(repository.upsert_player_game_logs)
    for col in ("fga", "fta"):
        assert f'"{col}": pd.to_numeric(df.get("{col.upper()}")' in body, (
            f"{col} is not written from the panel's {col.upper()} column"
        )


def test_both_columns_are_in_the_on_conflict_update_set():
    """
    THE ONE I ALMOST MISSED, and the module's own comment warns about it:
    `pf` was once left out of this set, which made "re-ingesting a season
    corrects rows" false for that column alone. 214,381 rows existed when
    migration 009 ran, so every row took the DO UPDATE path -- left out, these
    would have stayed NULL with the columns in place and the model still
    abstaining.
    """
    from src.db import repository

    body = inspect.getsource(repository.upsert_player_game_logs)
    conflict = body[body.index("on_conflict_do_update"):]
    for col in ("fga", "fta"):
        assert f'"{col}"' in conflict, f"{col} is written but never updated on conflict"


def test_the_integer_coercion_covers_them():
    """The model's columns are Integer and a float like 8.0 is rejected."""
    from src.db import repository

    body = inspect.getsource(repository.upsert_player_game_logs)
    line = next(ln for ln in body.splitlines() if "for col in (" in ln)
    for col in ("fga", "fta"):
        assert f'"{col}"' in line, f"{col} is not coerced to Int64 before the insert"


# --- 3. the read path -------------------------------------------------------


def test_the_read_path_returns_both_columns():
    """
    A column stored and never read is the shape of this whole defect: the
    migration applied, the backfill wrote fga on all 214,381 rows, and PTS
    went on abstaining because the panel dict did not mention it.
    """
    from src.db import repository

    body = inspect.getsource(repository.load_player_panel)
    for col in ("FGA", "FTA"):
        assert f'"{col}": r.{col.lower()}' in body, (
            f"load_player_panel does not put {col} on the panel, so "
            f"features/scoring_efficiency cannot attach TS_PCT or FT_RATE"
        )


# --- what the layer needs, and what the contract names ----------------------


def test_the_layer_requires_exactly_what_the_panel_now_carries():
    """
    Read from `scoring_efficiency` rather than restated here: if it ever
    requires a third box-score column, this fails instead of the model
    silently abstaining again.
    """
    from src.db import repository
    from src.features.scoring_efficiency import REQUIRED_COLS

    body = inspect.getsource(repository.load_player_panel)
    for col in REQUIRED_COLS:
        assert f'"{col}"' in body, (
            f"scoring_efficiency requires {col} and the panel does not carry it"
        )


def test_the_pts_contract_still_names_the_features_these_columns_feed():
    """
    The reason the columns were added. If a future PTS artifact drops these
    features this test fails and whoever dropped them decides whether the
    columns are still earning their place -- rather than leaving a migration
    nobody can account for.
    """
    import json

    art = REPO_ROOT / "data/external/model_runs/comparison/xgboost_PTS.meta.json"
    if not art.is_file():  # pragma: no cover - artifacts are gitignored
        pytest.skip("model artifacts are not tracked; present only where fitted")
    cols = json.loads(art.read_text(encoding="utf-8"))["feature_cols"]
    fed = [c for c in cols if c.startswith(("TS_PCT", "FT_RATE", "SHOT_VOLUME", "FGA_"))]
    assert fed, "no PTS feature reads the attempt columns any more"


def test_the_migration_states_what_each_range_guards_and_what_it_does_not():
    """
    The fga bound separates a player's attempts from a team total (team FGA
    runs 66-122). The fta bound CANNOT -- team FTA runs 2-59 and overlaps the
    player range entirely -- and the migration says so rather than implying a
    guard that is not there. A comment that overclaims is the defect this
    repository keeps finding.
    """
    sql = MIGRATION.read_text(encoding="utf-8")
    assert "DOES guard against a team total" in sql
    assert "DOES NOT guard against that" in sql
    assert "66-122" in sql and "2-59" in sql


def test_the_ranges_admit_every_value_the_real_data_holds():
    """Measured maxima: player FGA 47, FTA 43. A bound below either would
    reject real box scores."""
    sql = MIGRATION.read_text(encoding="utf-8")
    assert "fga <= 60" in sql and "fta <= 60" in sql
    # 47 and 43 are the measured maxima; the bound must sit above both.
    assert "0-47" in sql and "0-43" in sql
