"""Timezone helpers — UTC storage, America/Los_Angeles display only.

Project rule: never present the original feed timezone to users.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

DISPLAY_TZ_NAME = "America/Los_Angeles"
STORAGE_TZ_NAME = "UTC"

DISPLAY_TZ = ZoneInfo(DISPLAY_TZ_NAME)
UTC = timezone.utc


def now_utc() -> datetime:
    return datetime.now(UTC)


def now_pacific() -> datetime:
    return datetime.now(DISPLAY_TZ)


def to_utc(dt: datetime, *, assume: str = "utc") -> datetime:
    """Normalize any aware/naive datetime to timezone-aware UTC.

    ``assume`` decides what a NAIVE input means, and the caller must choose,
    because the two readings differ by a calendar day at the boundary:
    a date-only cutoff of 2025-02-01 read as UTC displays as
    2025-01-31T16:00 Pacific — the previous day, which is what a reader
    sees in `data_cutoff_pt`. Pass ``assume="pacific"`` for a value that
    was a Pacific calendar date to begin with.
    """
    if dt.tzinfo is None:
        if assume == "pacific":
            return dt.replace(tzinfo=DISPLAY_TZ).astimezone(UTC)
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def pacific_midnight_utc(day: date) -> datetime:
    """Start of a Pacific calendar day, as an aware UTC datetime.

    Use this for slate dates and data cutoffs — anything that was a
    calendar date rather than an instant.
    """
    return datetime(day.year, day.month, day.day, tzinfo=DISPLAY_TZ).astimezone(UTC)


def to_pacific(dt: datetime) -> datetime:
    """Convert a datetime to America/Los_Angeles for user-facing display."""
    return to_utc(dt).astimezone(DISPLAY_TZ)


def format_pacific(
    dt: datetime | None,
    *,
    fmt: str = "%Y-%m-%d %H:%M:%S %Z",
) -> str | None:
    if dt is None:
        return None
    return to_pacific(dt).strftime(fmt)


def format_pacific_iso(dt: datetime | None) -> str | None:
    """ISO-8601 string in Pacific (includes offset, e.g. -08:00 / -07:00)."""
    if dt is None:
        return None
    return to_pacific(dt).isoformat()


def pacific_calendar_date(dt: datetime | None = None) -> date:
    """Today's calendar date in Los Angeles (default: now)."""
    if dt is None:
        return now_pacific().date()
    return to_pacific(dt).date()


def parse_slate_date(value: str) -> date:
    """
    Parse a user slate date string as a Pacific calendar date (YYYY-MM-DD).

    Slate cutoffs are always interpreted in America/Los_Angeles, never UTC midnight.
    """
    return date.fromisoformat(value.strip())


def slate_date_bounds_utc(slate: date) -> tuple[datetime, datetime]:
    """
    Return [start, end] UTC datetimes covering one Pacific calendar day.

    Useful when filtering tipoff timestamps stored in UTC.
    """
    start_pt = datetime(slate.year, slate.month, slate.day, 0, 0, 0, tzinfo=DISPLAY_TZ)
    end_pt = datetime(slate.year, slate.month, slate.day, 23, 59, 59, 999999, tzinfo=DISPLAY_TZ)
    return start_pt.astimezone(UTC), end_pt.astimezone(UTC)
