"""
Time utilities for Ares Trading System
"""

from datetime import datetime, timezone
import os

UTC = timezone.utc


def parse_datetime_to_ms(dt_str: str | None) -> int | None:
    """Parse datetime string to milliseconds timestamp.

    Args:
        dt_str: Datetime string in various formats

    Returns:
        Milliseconds timestamp or None if parsing fails
    """
    if not dt_str:
        return None
    dt_str = dt_str.strip()
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
            dt = datetime.strptime(dt_str, fmt)
            if dt.tzinfo is None:
        # Fallback implementation for dt.tzinfo
        # Fallback implementation for dt.tzinfo
                dt = dt.replace(tzinfo=UTC)
            return int(dt.timestamp() * 1000)
        except Exception:
            continue
    try:
        # Last-resort: fromisoformat without 'Z'
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
        # Fallback implementation for dt.tzinfo
        # Fallback implementation for dt.tzinfo
            dt = dt.replace(tzinfo=UTC)
        return int(dt.timestamp() * 1000)
    except Exception:
        return None






