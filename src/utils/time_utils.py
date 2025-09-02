"""
Time utilities for Ares Trading System

This module provides comprehensive time handling utilities for trading operations,
including datetime parsing, timestamp conversion, and time window management.
"""

from datetime import datetime, timezone
import os
from typing import Optional, Tuple, Union


# Global UTC timezone constant
UTC = timezone.utc


def parse_datetime_to_ms(dt_str: str) -> Optional[int]:
    """
    Parse datetime string to milliseconds timestamp.
    
    Args:
        dt_str: Datetime string in various formats
        
    Returns:
        Milliseconds timestamp or None if parsing fails
    """
    if not dt_str:
        return None
    
    dt_str = dt_str.strip()
    
    # Supported datetime formats
    fmts = [
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
    ]
    
    # Try standard formats first
    for fmt in fmts:
        try:
            dt = datetime.strptime(dt_str, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return int(dt.timestamp() * 1000)
        except ValueError:
            continue
    
    # Try ISO format parsing
    try:
        # Handle 'Z' suffix for UTC
        if dt_str.endswith('Z'):
            dt_str = dt_str[:-1] + '+00:00'
        
        dt = datetime.fromisoformat(dt_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return int(dt.timestamp() * 1000)
    except ValueError:
        pass
    
    return None


def resolve_time_window_ms(config: Optional[dict] = None) -> Tuple[Optional[int], Optional[int]]:
    """
    Resolve time window from configuration or environment variables.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (start_timestamp_ms, end_timestamp_ms)
    """
    cfg = config or {}
    
    def as_int(v: Union[str, int, None]) -> Optional[int]:
        """Convert value to integer safely."""
        try:
            if v is None:
                return None
            return int(v)
        except (ValueError, TypeError):
            return None
    
    # Try to get timestamps from config
    t0 = (
        as_int(cfg.get("t0_ms")) or
        as_int(cfg.get("start_timestamp_ms")) or
        as_int(os.environ.get("ARES_T0_MS"))
    )
    
    t1 = (
        as_int(cfg.get("t1_ms")) or
        as_int(cfg.get("end_timestamp_ms")) or
        as_int(os.environ.get("ARES_T1_MS"))
    )
    
    # If timestamps not found, try parsing datetime strings
    if t0 is None:
        t0 = parse_datetime_to_ms(
            cfg.get("start_datetime") or os.environ.get("ARES_START_DATETIME", "")
        )
    
    if t1 is None:
        t1 = parse_datetime_to_ms(
            cfg.get("end_datetime") or os.environ.get("ARES_END_DATETIME", "")
        )
    
    return t0, t1


def format_timestamp_ms(timestamp_ms: int) -> str:
    """
    Format milliseconds timestamp to ISO string.
    
    Args:
        timestamp_ms: Milliseconds timestamp
        
    Returns:
        ISO formatted datetime string
    """
    dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC)
    return dt.isoformat()


def get_current_timestamp_ms() -> int:
    """
    Get current timestamp in milliseconds.
    
    Returns:
        Current milliseconds timestamp
    """
    return int(datetime.now(UTC).timestamp() * 1000)


def is_valid_timestamp_ms(timestamp_ms: int) -> bool:
    """
    Validate if timestamp is reasonable.
    
    Args:
        timestamp_ms: Milliseconds timestamp to validate
        
    Returns:
        True if timestamp is valid, False otherwise
    """
    if timestamp_ms <= 0:
        return False
    
    # Check if timestamp is not too far in the future (e.g., 10 years)
    max_future = get_current_timestamp_ms() + (10 * 365 * 24 * 60 * 60 * 1000)
    if timestamp_ms > max_future:
        return False
    
    return True


def calculate_duration_ms(start_ms: int, end_ms: int) -> int:
    """
    Calculate duration between two timestamps in milliseconds.
    
    Args:
        start_ms: Start timestamp in milliseconds
        end_ms: End timestamp in milliseconds
        
    Returns:
        Duration in milliseconds
    """
    return end_ms - start_ms


def format_duration_ms(duration_ms: int) -> str:
    """
    Format duration in milliseconds to human-readable string.
    
    Args:
        duration_ms: Duration in milliseconds
        
    Returns:
        Human-readable duration string
    """
    if duration_ms < 1000:
        return f"{duration_ms}ms"
    elif duration_ms < 60000:
        return f"{duration_ms / 1000:.1f}s"
    elif duration_ms < 3600000:
        return f"{duration_ms / 60000:.1f}m"
    else:
        return f"{duration_ms / 3600000:.1f}h"


def get_trading_hours() -> Tuple[int, int]:
    """
    Get trading hours in milliseconds since midnight UTC.
    
    Returns:
        Tuple of (market_open_ms, market_close_ms) in milliseconds since midnight
    """
    # Default to 9:30 AM - 4:00 PM EST (14:30 - 21:00 UTC)
    market_open_ms = 14 * 60 * 60 * 1000 + 30 * 60 * 1000  # 14:30 UTC
    market_close_ms = 21 * 60 * 60 * 1000  # 21:00 UTC
    
    return market_open_ms, market_close_ms


def is_market_open(timestamp_ms: Optional[int] = None) -> bool:
    """
    Check if market is open at given timestamp.
    
    Args:
        timestamp_ms: Timestamp to check (defaults to current time)
        
    Returns:
        True if market is open, False otherwise
    """
    if timestamp_ms is None:
        timestamp_ms = get_current_timestamp_ms()
    
    # Get current time components
    dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC)
    current_ms = dt.hour * 60 * 60 * 1000 + dt.minute * 60 * 1000 + dt.second * 1000
    
    # Check if it's a weekday
    if dt.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return False
    
    # Check if it's within trading hours
    market_open_ms, market_close_ms = get_trading_hours()
    
    return market_open_ms <= current_ms <= market_close_ms


def get_next_market_open(timestamp_ms: Optional[int] = None) -> int:
    """
    Get next market open timestamp.
    
    Args:
        timestamp_ms: Current timestamp (defaults to current time)
        
    Returns:
        Next market open timestamp in milliseconds
    """
    if timestamp_ms is None:
        timestamp_ms = get_current_timestamp_ms()
    
    dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC)
    market_open_ms, _ = get_trading_hours()
    
    # Calculate next market open
    current_ms = dt.hour * 60 * 60 * 1000 + dt.minute * 60 * 1000 + dt.second * 1000
    
    if current_ms < market_open_ms:
        # Market opens today
        next_open = dt.replace(hour=14, minute=30, second=0, microsecond=0)
    else:
        # Market opens tomorrow
        next_open = dt.replace(hour=14, minute=30, second=0, microsecond=0)
        next_open = next_open.replace(day=next_open.day + 1)
    
    # Adjust for weekends
    while next_open.weekday() >= 5:
        next_open = next_open.replace(day=next_open.day + 1)
    
    return int(next_open.timestamp() * 1000)
