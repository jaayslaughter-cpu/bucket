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
