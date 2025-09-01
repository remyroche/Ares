"""Typed time utilities used across the project."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Tuple

UTC = timezone.utc


def get_current_timestamp_ms() -> int:
    """Return current UTC timestamp in milliseconds."""
    return int(datetime.now(UTC).timestamp() * 1000)


def parse_datetime_to_ms(dt_str: str) -> Optional[int]:
    """Parse a date/time string into UTC milliseconds. Returns None if invalid.

    Tries a few common formats; extend as needed.
    """
    if not dt_str:
        return None
    fmts = [
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S",
    ]
    for fmt in fmts:
        try:
            dt = datetime.strptime(dt_str, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            else:
                dt = dt.astimezone(UTC)
            return int(dt.timestamp() * 1000)
        except Exception:
            continue
    # Fallback: try fromisoformat without 'Z'
    try:
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        else:
            dt = dt.astimezone(UTC)
        return int(dt.timestamp() * 1000)
    except Exception:
        return None


def format_timestamp_ms(timestamp_ms: int) -> str:
    """Format UTC milliseconds as ISO8601 string."""
    dt = datetime.fromtimestamp(timestamp_ms / 1000, UTC)
    return dt.isoformat()


def validate_timestamp_ms(timestamp_ms: int) -> bool:
    """Basic sanity check: non-negative and not beyond 10 years in future."""
    if timestamp_ms < 0:
        return False
    max_future = get_current_timestamp_ms() + (10 * 365 * 24 * 60 * 60 * 1000)
    if timestamp_ms > max_future:
        return False
    return True


def calculate_duration_ms(start_ms: int, end_ms: int) -> int:
    return end_ms - start_ms


def format_duration_ms(duration_ms: int) -> str:
    if duration_ms < 1000:
        return f"{duration_ms}ms"
    if duration_ms < 60_000:
        return f"{duration_ms / 1000:.1f}s"
    if duration_ms < 3_600_000:
        return f"{duration_ms / 60_000:.1f}m"
    return f"{duration_ms / 3_600_000:.1f}h"
