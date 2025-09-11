"""
Datetime Utilities for Ares Trading System

This module provides datetime utilities for the trading system,
including current time retrieval and formatting functions.
"""

from datetime import datetime, date
from typing import Optional, Union


def get_current_datetime() -> datetime:
    """Get the current datetime.

    Returns:
        Current datetime object
    """
    return datetime.now()


def get_today() -> date:
    """Get today's date.

    Returns:
        Current date object
    """
    return date.today()


def format_datetime(dt: Union[datetime, str], format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Format a datetime object or string to a string.

    Args:
        dt: Datetime object or string to format
        format_str: Format string to use

    Returns:
        Formatted datetime string
    """
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt.replace('Z', '+00:00'))
        except ValueError:
            # Try common formats
            for fmt in ["%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"]:
                try:
                    dt = datetime.strptime(dt, fmt)
                    break
                except ValueError:
                    continue
            else:
                raise ValueError(f"Could not parse datetime string: {dt}")

    return dt.strftime(format_str)


def parse_datetime(dt_str: str, format_str: str = "%Y-%m-%d %H:%M:%S") -> datetime:
    """Parse a datetime string to a datetime object.

    Args:
        dt_str: Datetime string to parse
        format_str: Format string to use

    Returns:
        Parsed datetime object
    """
    return datetime.strptime(dt_str, format_str)


def get_current_timestamp() -> int:
    """Get current timestamp in seconds.

    Returns:
        Current timestamp as integer
    """
    return int(get_current_datetime().timestamp())


def format_timestamp(timestamp: Union[int, float], format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Format a timestamp to a datetime string.

    Args:
        timestamp: Timestamp in seconds
        format_str: Format string to use

    Returns:
        Formatted datetime string
    """
    dt = datetime.fromtimestamp(timestamp)
    return dt.strftime(format_str)
