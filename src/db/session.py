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


def get_engine(echo: bool = False) -> Engine:
    global _engine
    if _engine is None:
        _engine = create_engine(
            get_database_url(),
            echo=echo,
            pool_pre_ping=True,   # survives Supabase pooler dropping idle connections
            pool_size=5,
            max_overflow=5,
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
