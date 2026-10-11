"""
tests/test_artifact_store.py — the boot-time artifact fetch.

WHAT THIS CLOSES, and why it needed closing. The trained artifacts live on the
mounted volume and nothing in this repository could put them there: `data/**`
is gitignored, `.dockerignore` drops `data/`, and there was no storage client.
Every deploy needed a person to run `scripts/seed_volume.py`, and that was the
one manual step that decided whether the slate scored anything at all.

The failure this guards against is specific and it is NOT "the download
failed". It is a HALF-fetched family: `xgb_adapter.load` sets
`mean_model = None` when `.mean.json` is absent and only warns, so a booster
installed without its mean head returns NULL PROJECTIONS beside live
probabilities — a slate that looks like it worked. So the tests here are mostly
about what the fetch REFUSES to install.

The HTTP layer is faked at the `requests.Session` boundary, so everything above
it — grouping into families, completeness, size comparison, staging, atomic
move, sidecar validation, redaction — is the real code.

RESEARCH_ONLY. No odds, no wager.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import src.models.artifact_store as store

BUCKET_ENV = {
    store.ENV_STORE_URL: "https://project.supabase.co",
    # DELIBERATELY NOT CREDENTIAL-SHAPED. This used Supabase's real
    # personal-access-token prefix followed by hex, and GitHub push protection
    # blocked the push -- correctly: the fixture was pattern-identical to a
    # real key, and the prefix is not repeated here for the same reason. A
    # test fixture that trips
    # secret scanning is one that trains people to click "allow this secret",
    # which is the habit that leaks the next real one. Long enough that
    # `_redact`'s blind str.replace cannot mangle ordinary prose, and shaped
    # so no scanner can mistake it.
    store.ENV_STORE_KEY: "not-a-real-key-only-a-redaction-fixture",
    store.ENV_STORE_BUCKET: "model-artifacts",
    store.ENV_STORE_PREFIX: "comparison",
}


class _Response:
    def __init__(self, status=200, payload=None, body=b"", text=""):
        self.status_code = status
        self._payload = payload
        self._body = body
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise AssertionError("should have been handled before raise_for_status")

    def iter_content(self, chunk_size=0):
        yield self._body


class _FakeSession:
    """Records every call and serves canned objects."""

    def __init__(self, listing, files, *, fail_on=None, status=200):
        self.listing = listing
        self.files = files
        self.fail_on = fail_on or set()
        self.status = status
        self.calls: list[tuple[str, str]] = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url))
        if any(f in url for f in self.fail_on):
            import requests

            raise requests.exceptions.ConnectionError("link went away")
        if "/object/list/" in url:
            return _Response(self.status, payload=self.listing, text="listing")
        name = url.split("/comparison/")[-1]
        if method == "POST":
            # An UPLOAD, not a fetch. Supabase uses POST to the object path for
            # both; the fake has to make the same distinction or the upload
            # tests exercise the 404 branch instead of the code they name.
            self.files[name] = kwargs.get("data", b"")
            return _Response(self.status, payload={"Key": name}, text="ok")
        if name not in self.files:
            return _Response(404, text=f"not found: {name}")
        return _Response(self.status, body=self.files[name])


def _meta(rows: int = 200_678) -> bytes:
    return json.dumps({"train_row_count": rows, "feature_cols": ["a", "b"]}).encode()


def _listing(names_to_bytes: dict[str, bytes]) -> list[dict]:
    return [
        {"name": f"comparison/{n}", "updated_at": "2026-10-10T00:00:00Z",
         "metadata": {"size": len(b)}}
        for n, b in names_to_bytes.items()
    ]


def _complete(market: str = "PTS") -> dict[str, bytes]:
    return {
        f"xgboost_{market}.json": b"booster-bytes",
        f"xgboost_{market}.mean.json": b"mean-head-bytes",
        f"xgboost_{market}.meta.json": _meta(),
    }


@pytest.fixture
def configured(monkeypatch):
    for k, v in BUCKET_ENV.items():
        monkeypatch.setenv(k, v)
    return BUCKET_ENV


# ---------------------------------------------------------------------------
# an unconfigured store must not break a deployment that was working
# ---------------------------------------------------------------------------

def test_an_unconfigured_store_is_not_an_error(monkeypatch):
    """
    Every deployment that seeds by hand was working yesterday. Failing the boot
    over an unset variable would break all of them to add a feature none of
    them asked for.
    """
    monkeypatch.delenv(store.ENV_STORE_URL, raising=False)
    monkeypatch.delenv(store.ENV_STORE_KEY, raising=False)
    assert store.is_configured() is False

    import scripts.fetch_artifacts as cli

    assert cli.main(["--pull"]) == 0


def test_half_configured_is_also_unconfigured(monkeypatch):
    """A URL with no key cannot authenticate, so it is not a store."""
    monkeypatch.setenv(store.ENV_STORE_URL, "https://project.supabase.co")
    monkeypatch.delenv(store.ENV_STORE_KEY, raising=False)
    assert store.is_configured() is False
    with pytest.raises(store.ArtifactStoreError):
        store.list_remote()


# ---------------------------------------------------------------------------
# what it refuses to install — the point of the module
# ---------------------------------------------------------------------------

def test_a_complete_family_lands(configured, tmp_path):
    files = _complete("PTS")
    session = _FakeSession(_listing(files), files)
    report = store.fetch_artifacts(tmp_path, session=session)

    assert [f["market"] for f in report["fetched"]] == ["PTS"]
    assert report["fetched"][0]["train_row_count"] == 200_678
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(files)


def test_a_booster_without_its_mean_head_is_refused_entirely(configured, tmp_path):
    """
    THE FAILURE THIS MODULE EXISTS FOR. `xgb_adapter.load` sets
    `mean_model = None` when `.mean.json` is absent and only WARNS, so
    installing the booster alone produces null projections beside live
    probabilities — a slate that looks like it worked. Nothing lands.
    """
    files = {k: v for k, v in _complete("PTS").items() if "mean" not in k}
    session = _FakeSession(_listing(files), files)
    report = store.fetch_artifacts(tmp_path, session=session)

    assert report["fetched"] == []
    assert report["refused"] and report["refused"][0]["market"] == "PTS"
    assert "null projections" in report["refused"][0]["reason"]
    assert list(tmp_path.iterdir()) == [], "a partial family was installed"


def test_a_booster_without_its_sidecar_is_refused(configured, tmp_path):
    """Scoring is refused without the feature contract, so it is not a model."""
    files = {k: v for k, v in _complete("PTS").items() if "meta" not in k}
    session = _FakeSession(_listing(files), files)
    report = store.fetch_artifacts(tmp_path, session=session)

    assert report["fetched"] == []
    assert report["refused"][0]["market"] == "PTS"
    assert list(tmp_path.iterdir()) == []


def test_an_unparseable_sidecar_leaves_the_volume_untouched(configured, tmp_path):
    """
    VALIDATED BEFORE IT MOVES. A sidecar that will not parse is a file, not a
    contract, and installing it turns a visible "nothing resolved" into a
    silent "resolved and unusable".
    """
    files = _complete("PTS")
    files["xgboost_PTS.meta.json"] = b"{ this is not json"
    session = _FakeSession(_listing(files), files)
    report = store.fetch_artifacts(tmp_path, session=session)

    assert report["fetched"] == []
    assert report["errors"] and report["errors"][0]["market"] == "PTS"
    assert list(tmp_path.iterdir()) == [], "a broken family was installed"


def test_a_sidecar_with_no_feature_cols_is_refused(configured, tmp_path):
    files = _complete("PTS")
    files["xgboost_PTS.meta.json"] = json.dumps({"train_row_count": 1}).encode()
    session = _FakeSession(_listing(files), files)
    report = store.fetch_artifacts(tmp_path, session=session)

    assert report["fetched"] == []
    assert "feature_cols" in str(report["errors"])
    assert list(tmp_path.iterdir()) == []


def test_a_dropped_connection_leaves_nothing_behind(configured, tmp_path):
    """
    ATOMIC PER FAMILY. Files stage beside the destination and move only once
    the family is complete and validated, so a link that dies half way leaves
    the volume exactly as it was.
    """
    files = _complete("PTS")
    session = _FakeSession(_listing(files), files, fail_on={"mean"})
    report = store.fetch_artifacts(tmp_path, session=session)

    assert report["fetched"] == []
    assert report["errors"]
    assert list(tmp_path.iterdir()) == [], "a staging directory or partial file survived"


# ---------------------------------------------------------------------------
# idempotence — the boot runs this every time
# ---------------------------------------------------------------------------

def test_the_second_boot_downloads_nothing(configured, tmp_path):
    files = _complete("PTS")
    listing = _listing(files)

    first = _FakeSession(listing, files)
    assert store.fetch_artifacts(tmp_path, session=first)["fetched"]

    second = _FakeSession(listing, files)
    report = store.fetch_artifacts(tmp_path, session=second)
    assert report["fetched"] == []
    assert report["skipped"] == [{"market": "PTS", "files": 3}]
    downloads = [c for c in second.calls if "/object/list/" not in c[1]]
    assert downloads == [], f"re-downloaded despite being current: {downloads}"


def test_a_changed_remote_size_is_re_fetched(configured, tmp_path):
    """Size, not a checksum: the list endpoint gives a size and not a hash, and
    a manifest carrying hashes would be a second source of truth."""
    files = _complete("PTS")
    store.fetch_artifacts(tmp_path, session=_FakeSession(_listing(files), files))

    files["xgboost_PTS.json"] = b"a-retrained-booster-of-a-different-length"
    report = store.fetch_artifacts(
        tmp_path, session=_FakeSession(_listing(files), files),
    )
    assert [f["market"] for f in report["fetched"]] == ["PTS"]
    assert (tmp_path / "xgboost_PTS.json").read_bytes() == files["xgboost_PTS.json"]


def test_force_re_fetches_a_current_family(configured, tmp_path):
    files = _complete("PTS")
    store.fetch_artifacts(tmp_path, session=_FakeSession(_listing(files), files))
    report = store.fetch_artifacts(
        tmp_path, session=_FakeSession(_listing(files), files), force=True,
    )
    assert [f["market"] for f in report["fetched"]] == ["PTS"]


# ---------------------------------------------------------------------------
# one market must not stop the others
# ---------------------------------------------------------------------------

def test_one_broken_market_does_not_stop_the_rest(configured, tmp_path):
    """
    A boot step, not a batch job: one market missing from the bucket should not
    keep the other two from scoring tonight.
    """
    files = {**_complete("PTS"), **_complete("REB")}
    del files["xgboost_REB.mean.json"]
    files.update(_complete("AST"))
    session = _FakeSession(_listing(files), files)
    report = store.fetch_artifacts(tmp_path, session=session)

    assert sorted(f["market"] for f in report["fetched"]) == ["AST", "PTS"]
    assert [r["market"] for r in report["refused"]] == ["REB"]
    assert (tmp_path / "xgboost_PTS.json").exists()
    assert not (tmp_path / "xgboost_REB.json").exists()


def test_markets_can_be_narrowed(configured, tmp_path):
    files = {**_complete("PTS"), **_complete("REB")}
    report = store.fetch_artifacts(
        tmp_path, markets=["PTS"], session=_FakeSession(_listing(files), files),
    )
    assert [f["market"] for f in report["fetched"]] == ["PTS"]
    assert not (tmp_path / "xgboost_REB.json").exists()


# ---------------------------------------------------------------------------
# the credential
# ---------------------------------------------------------------------------

def test_the_service_key_never_appears_in_an_error(configured, tmp_path, caplog):
    """
    Anyone holding it can read and write the bucket. The key goes in a header,
    so it should never reach an exception — "should never" is why `_redact`
    exists, and why this asserts on the whole report and the log.
    """
    import logging

    key = BUCKET_ENV[store.ENV_STORE_KEY]
    files = _complete("PTS")

    class _LeakySession(_FakeSession):
        def request(self, method, url, **kwargs):
            import requests

            raise requests.exceptions.ConnectionError(
                f"TLS handshake failed for {url}?token={key}"
            )

    with caplog.at_level(logging.WARNING):
        with pytest.raises(store.ArtifactStoreError) as exc:
            store.list_remote(session=_LeakySession(_listing(files), files))

    assert key not in str(exc.value)
    assert "***" in str(exc.value)
    assert key not in caplog.text


def test_a_4xx_body_is_redacted_and_not_retried(configured):
    """
    A wrong request does not become right. Retrying a 404 bucket three times
    makes a boot slower and no more correct, and the body could echo the key.
    """
    key = BUCKET_ENV[store.ENV_STORE_KEY]

    class _Forbidden(_FakeSession):
        def request(self, method, url, **kwargs):
            self.calls.append((method, url))
            return _Response(403, text=f"denied for apikey={key}")

    session = _Forbidden([], {})
    with pytest.raises(store.ArtifactStoreError) as exc:
        store.list_remote(session=session)

    assert key not in str(exc.value)
    assert len(session.calls) == 1, "a 4xx was retried"


def test_a_429_is_retried(configured, monkeypatch):
    """The one 4xx that IS transient: the request is fine, there have just been
    too many of them."""
    monkeypatch.setattr(store.time, "sleep", lambda s: None)
    seen: list[int] = []

    class _Limited(_FakeSession):
        def request(self, method, url, **kwargs):
            seen.append(1)
            if len(seen) < 3:
                return _Response(429, text="slow down")
            return _Response(200, payload=[], text="ok")

    assert store.list_remote(session=_Limited([], {})) == []
    assert len(seen) == 3, "the 429 was not retried"


# ---------------------------------------------------------------------------
# the upload half, which is a human action
# ---------------------------------------------------------------------------

def test_upload_refuses_an_incomplete_family(configured, tmp_path):
    (tmp_path / "xgboost_PTS.json").write_bytes(b"booster")
    session = _FakeSession([], {})
    report = store.upload_artifacts(tmp_path, session=session)

    assert report["uploaded"] == []
    assert report["refused"][0]["market"] == "PTS"
    assert ".meta.json" in str(report["refused"][0]["missing"])
    assert session.calls == [], "an incomplete family was uploaded anyway"


def test_upload_sends_the_whole_family(configured, tmp_path):
    for name, body in _complete("PTS").items():
        (tmp_path / name).write_bytes(body)
    (tmp_path / "catboost_PTS.cbm").write_bytes(b"catboost")
    session = _FakeSession([], {})
    report = store.upload_artifacts(tmp_path, session=session)

    assert report["uploaded"] == [{"market": "PTS", "files": 4}]
    assert len(session.calls) == 4


def test_the_boot_pulls_and_a_human_pushes():
    """
    AST-walked over scripts/start.sh: the boot must pull, never push. A
    scheduler that uploaded would replace the artifact that produced every
    probability now in the database, while nobody was looking.
    """
    body = [
        ln for ln in Path(__file__).resolve().parents[1].joinpath(
            "scripts/start.sh").read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    fetch = [ln for ln in body if "scripts.fetch_artifacts" in ln]
    assert fetch, "the boot does not fetch artifacts at all"
    assert all("--pull" in ln for ln in fetch)
    assert not any("--push" in ln for ln in fetch), "the boot pushes artifacts"

    # And before the probe, so the probe reports the volume after the fetch.
    probe = next(i for i, ln in enumerate(body) if "railway_healthcheck" in ln)
    pull = next(i for i, ln in enumerate(body) if "fetch_artifacts" in ln)
    assert pull < probe
