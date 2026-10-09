"""
src/db/session.py — engine/session management for PostgreSQL / Supabase.

Connection precedence:
1. ``DATABASE_URL`` env var (full SQLAlchemy URL) — use this for Supabase.
2. Individual ``PGHOST``/``PGPORT``/``PGUSER``/``PGPASSWORD``/``PGDATABASE`` vars.
3. Hard failure. There is deliberately NO SQLite fallback: silently
   falling back to a local file when Postgres is misconfigured would let
   a "successful" run write nowhere the rest of the stack can read.

Supabase notes:
- Use the **connection pooler** URL (port 6543) for short-lived jobs like
  this pipeline; the direct connection (port 5432) is fine for migrations
  and long sessions. Supabase's dashboard shows both under
  Project Settings -> Database.
- ``sslmode=require`` is appended automatically for non-local hosts if
  the URL does not already specify an sslmode.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from urllib.parse import urlparse, urlunparse

from sqlalchemy import Engine, create_engine
from sqlalchemy.engine import URL
from sqlalchemy.orm import Session, sessionmaker

from src.db.models import Base

_engine: Engine | None = None
_SessionFactory: sessionmaker[Session] | None = None


class DatabaseConfigError(RuntimeError):
    """Raised when no usable Postgres connection can be assembled."""


def _build_url_from_parts() -> str | None:
    host = os.environ.get("PGHOST")
    if not host:
        return None
    user = os.environ.get("PGUSER", "postgres")
    password = os.environ.get("PGPASSWORD", "")
    port = os.environ.get("PGPORT", "5432")
    database = os.environ.get("PGDATABASE", "postgres")

    # Built with URL.create rather than an f-string: a password containing
    # @, :, / or # — all common in generated credentials — produces a URL
    # that parses to the wrong host, and the resulting connection error
    # names neither the cause nor, thankfully, the password.
    return URL.create(
        "postgresql+psycopg",
        username=user,
        password=password or None,
        host=host,
        port=int(port) if str(port).isdigit() else None,
        database=database,
    ).render_as_string(hide_password=False)


def _ensure_sslmode(url: str) -> str:
    """Append sslmode=require for remote hosts (Supabase requires TLS)."""
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    is_local = hostname in {"localhost", "127.0.0.1", "::1"}
    if is_local or "sslmode=" in (parsed.query or ""):
        return url
    query = f"{parsed.query}&sslmode=require" if parsed.query else "sslmode=require"
    return urlunparse(parsed._replace(query=query))


def get_database_url() -> str:
    url = os.environ.get("DATABASE_URL") or _build_url_from_parts()
    if not url:
        raise DatabaseConfigError(
            "No database connection configured. Set DATABASE_URL (recommended, "
            "e.g. your Supabase pooler URL) or PGHOST/PGUSER/PGPASSWORD/PGDATABASE "
            "in your .env — see .env.example. There is no SQLite fallback by design."
        )
    if url.startswith("postgres://"):  # normalize legacy scheme
        url = url.replace("postgres://", "postgresql+psycopg://", 1)
    elif url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return _ensure_sslmode(url)


#: Recycle a pooled connection after this many seconds. SQLAlchemy's default is
#: -1, which is "never": the deployed worker runs a slate at 09:00 PT and a
#: settlement at 03:30 PT, so a pooled connection sits idle for eighteen hours
#: between them. ``pool_pre_ping`` already discards one the server has closed,
#: at the cost of a round trip per checkout; recycling means the long-dead ones
#: are not kept in the first place. Thirty minutes is comfortably inside
#: Supabase's pooler idle timeout.
POOL_RECYCLE_SECONDS = 1800

#: How long to wait for a free slot in the pool before raising. SQLAlchemy's
#: default is 30 and this restates it, because the value that matters is below.
POOL_TIMEOUT_SECONDS = 30

#: TCP/handshake timeout, passed to libpq. WITHOUT THIS THERE IS NO BOUND. A
#: pooler host that accepts the connection and never completes the handshake —
#: a region outage, a security group change — blocks the caller indefinitely,
#: and the slate job has no timeout of its own: it would hang past every tip-off
#: and be noticed as a worker that produced nothing, with no error anywhere.
#: Ten seconds is long for a pooler in the same region and short enough that the
#: failure is reported while the slate still has hours of runway.
CONNECT_TIMEOUT_SECONDS = 10

#: Shows up in ``pg_stat_activity`` and Supabase's dashboard, so a connection
#: can be attributed to this worker rather than to "psycopg".
APPLICATION_NAME = "propiq-worker"


def _int_env(name: str, default: int, *, low: int, high: int) -> int:
    """
    A bounded integer env var, defaulting rather than raising.

    Same contract as ``main._int_env`` and ``scheduler_worker._int_env``, and
    deliberately a third small implementation rather than an import: this
    module is the bottom of the dependency graph — the migration runner and
    every repository import it — and reaching up to the orchestrator or the
    worker to parse a number would make `src.db` depend on APScheduler.

    BOUNDED, not merely positive. ``max_overflow=0`` is legitimate (no bursting
    past ``pool_size``) so the floor cannot be 1, and a mistyped
    ``pool_size=500`` against a Supabase pooler would exhaust the project's
    connection allowance for every other client — which presents as unrelated
    things failing, not as a bad value here.
    """
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if low <= value <= high else default


def get_engine(echo: bool = False) -> Engine:
    global _engine
    if _engine is None:
        _engine = create_engine(
            get_database_url(),
            echo=echo,
            pool_pre_ping=True,   # survives Supabase pooler dropping idle connections
            pool_size=_int_env("PROPIQ_DB_POOL_SIZE", 5, low=1, high=50),
            max_overflow=_int_env("PROPIQ_DB_MAX_OVERFLOW", 5, low=0, high=50),
            pool_recycle=_int_env(
                "PROPIQ_DB_POOL_RECYCLE", POOL_RECYCLE_SECONDS,
                low=60, high=86_400,
            ),
            pool_timeout=_int_env(
                "PROPIQ_DB_POOL_TIMEOUT", POOL_TIMEOUT_SECONDS, low=1, high=300,
            ),
            connect_args={
                "connect_timeout": _int_env(
                    "PROPIQ_DB_CONNECT_TIMEOUT", CONNECT_TIMEOUT_SECONDS,
                    low=1, high=120,
                ),
                "application_name": APPLICATION_NAME,
            },
        )
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _SessionFactory


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope — commits on success, rolls back on exception."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db() -> None:
    """Create all tables. Safe to re-run (CREATE TABLE IF NOT EXISTS semantics)."""
    Base.metadata.create_all(bind=get_engine())


def healthcheck() -> tuple[bool, str]:
    """Return (ok, message). Used by main.py preflight before doing real work."""
    from sqlalchemy import text

    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, "Database reachable."
    except DatabaseConfigError as exc:
        return False, str(exc)
    except Exception as exc:  # noqa: BLE001 — surface the real driver error to the user
        return False, f"Database connection failed: {exc}"
