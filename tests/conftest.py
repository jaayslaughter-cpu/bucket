"""
Shared test fixtures.

WHY THE CACHE IS ISOLATED. ``src.features.absences`` reads whatever inactive
parquet it finds under ``data/external/inactive_players/``, and
``build_feature_matrix`` runs that layer. A unit test must not depend on whether
this machine has run ``fetch-inactives``: a suite that passes on a clean checkout
and behaves differently once real data lands is a suite that cannot be trusted
about the feature it is guarding. So the whole session gets an empty cache
directory, and a test wanting a populated one passes ``cache_root`` explicitly.

SESSION SCOPE, deliberately. ``feature_matrix`` in tests/test_wave_modules.py is
module-scoped and pytest builds higher-scoped fixtures first, so a
function-scoped ``monkeypatch`` here would be applied AFTER that panel had
already been built — isolating every test except the one that reads the cache.

(The demo panel's ids are ``00DEMO000001``-shaped and cannot collide with real
NBA game ids, so that layer abstains either way today. That is an accident of the
demo id scheme rather than a property of the test, and it is not what the
isolation is for.)
"""

from __future__ import annotations

import pytest

from src.features.absences import ENV_CACHE_DIR


@pytest.fixture(scope="session", autouse=True)
def _isolated_inactive_cache(tmp_path_factory):
    empty = tmp_path_factory.mktemp("no_inactive_cache")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(ENV_CACHE_DIR, str(empty))
        yield empty


@pytest.fixture
def run_settlement():
    """
    Drive the real `settle_pending_props` over fake rows — no DB, no HTTP.

    SHARED BECAUSE THE ALTERNATIVE WAS GREPPING runner.py. Several tests used
    to assert that a substring or an AST shape still appeared in the source,
    which confirms the code LOOKS right and runs none of it: an inverted guard
    satisfies "an `if` mentioning closing_line and None" exactly as well as
    the correct one. Running the function found two live bugs those checks
    could not see, so the harness lives here where every settlement test can
    reach it.

    Pass `rows` (anything with the PropResult attributes the runner touches),
    and optionally a closing `lookup`, a `capture_raises` exception, the
    `player_stats` the box-score fetch should return, and `dry_run`.
    """
    from contextlib import contextmanager

    def _run(monkeypatch, rows, *, lookup=None, capture_raises=None,
             player_stats=None, dry_run=False):
        from src.settlement import runner as rn

        class _Db:
            def execute(self, _statement):
                return self

            def scalars(self):
                return self

            def all(self):
                return rows

        @contextmanager
        def _scope():
            yield _Db()

        def _capture(_dates, **_kwargs):
            if capture_raises is not None:
                raise capture_raises
            return lookup if lookup is not None else {}

        stats = player_stats if player_stats is not None else {
            # A real, final box score: Fox for 20, over a 13.5 line.
            "De'Aaron Fox": {
                "points": 20, "reboundsTotal": 4, "assists": 6,
                "minutes_played": 33.0, "did_not_play": False,
            },
        }

        monkeypatch.setattr(rn, "session_scope", _scope)
        monkeypatch.setattr(rn, "capture_for_dates", _capture)
        monkeypatch.setattr(
            rn, "fetch_player_stats_for_game",
            lambda *_a, **_k: (stats, {"gameId": "0022600001"}),
        )
        return rn.settle_pending_props(session_http=object(), dry_run=dry_run)

    return _run
