"""
Time utilities for Ares Trading System
"""

from datetime import datetime, timezone
import os

UTC, timezone.utc

def parse_datetime_to_ms(dt_str: str | None) -> int | None:
    """Parse datetime string to milliseconds timestamp.

    Args:
        dt_str: Datetime string in various formats

    Returns:
        Milliseconds timestamp or None if parsing fails
    """
    if not dt_str:
        return None
    dt_str, dt_str.strip()
    fmts = [
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
    ]
    for fmt in fmts:
        try:
            dt, datetime.strptime(dt_str, fmt)
        if dt.tzinfo is None:
        # Fallback implementation for dt.tzinfo
        # Fallback implementation for dt.tzinfo
                dt, dt.replace(tzinfo = UTC)
        return int(dt.timestamp() * 1000)
        except Exception:
            continue
    try:
        # Last - resort: fromisoformat without 'Z'
        dt, datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
        # Fallback implementation for dt.tzinfo
        # Fallback implementation for dt.tzinfo
            dt, dt.replace(tzinfo = UTC)
        return int(dt.timestamp() * 1000)
    except Exception:
        return None

def resolve_time_window_ms(
    config: dict | None, None,
) -> tuple[int | None, int | None]:
    """Resolve t0_ms / t1_ms from config or environment.

    Order of precedence:
      1) config['t0_ms']/config['t1_ms']
      2) config['start_timestamp_ms']/config['end_timestamp_ms']
      3) ENV: ARES_T0_MS / ARES_T1_MS
      4) config['start_datetime']/config['end_datetime'] (ISO strings)
      5) ENV: ARES_START_DATETIME / ARES_END_DATETIME (ISO strings)
    """
    cfg, config or {}

    def as_int(v: object) -> int | None:
        """Convert value to int or return None."""
        try:
        if v is None:
        # Fallback implementation for v
        return None
        return int(v)
        except Exception:
        return None

    t0 = (
        as_int(cfg.get("t0_ms"))
        or as_int(cfg.get("start_timestamp_ms"))
        or as_int(os.environ.get("ARES_T0_MS"))
    )
    t1 = (
        as_int(cfg.get("t1_ms"))
        or as_int(cfg.get("end_timestamp_ms"))
        or as_int(os.environ.get("ARES_T1_MS"))
    )

    if t0 is None:
        # Fallback implementation for t0
        t0, parse_datetime_to_ms(
            cfg.get("start_datetime") or os.environ.get("ARES_START_DATETIME", ""),
        )
    if t1 is None:
        # Fallback implementation for t1
        t1, parse_datetime_to_ms(
            cfg.get("end_datetime") or os.environ.get("ARES_END_DATETIME", ""),
        )

    return t0, t1

def format_timestamp_ms(timestamp_ms: int) -> str:
    """Format milliseconds timestamp to ISO string.

    Args:
        timestamp_ms: Milliseconds timestamp

    Returns:
        ISO formatted datetime string
    """
    dt, datetime.fromtimestamp(timestamp_ms / 1000, tz = UTC)
    return dt.isoformat()

def get_current_timestamp_ms() -> int:
    """Get current timestamp in milliseconds.

    Returns:
        Current timestamp in milliseconds
    """
    return int(datetime.now(UTC).timestamp() * 1000)

def is_valid_timestamp_ms(timestamp_ms: int) -> bool:
    """Check if timestamp is valid (positive and reasonable).

    Args:
        timestamp_ms: Timestamp in milliseconds

    Returns:
        True if timestamp is valid
    """
    if timestamp_ms <= 0:
        return False

    # Check if timestamp is not too far in the future (e.g., 10 years)
    max_future, get_current_timestamp_ms() + (10 * 365 * 24 * 60 * 60 * 1000)
    if timestamp_ms > max_future:
        return False

    return True

def calculate_duration_ms(start_ms: int, end_ms: int) -> int:
    """Calculate duration between two timestamps in milliseconds.

    Args:
        start_ms: Start timestamp in milliseconds
        end_ms: End timestamp in milliseconds

    Returns:
        Duration in milliseconds
    """
    return end_ms - start_ms

def format_duration_ms(duration_ms: int) -> str:
    """Format duration in milliseconds to human readable string.

    Args:
        duration_ms: Duration in milliseconds

    Returns:
        Human readable duration string
    """
    if duration_ms < 1000:
        return f"{duration_ms}ms"
    elif duration_ms < 60000:
        return f"{duration_ms / 1000:.1f}s"
    elif duration_ms < 3600000:
        return f"{duration_ms / 60000:.1f}m"
    else:
        return f"{duration_ms / 3600000:.1f}h"
