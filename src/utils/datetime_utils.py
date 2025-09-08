"""
DateTime utilities with passthrough functions for common operations.
"""

from datetime import datetime
from typing import Optional

def format_datetime(dt: Optional[datetime] = None, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Format datetime to string."""
    if dt is None:
        dt = datetime.now()
    return dt.strftime(format_str)

def get_current_datetime() -> datetime:
    """Get current datetime."""
    return datetime.now()

def get_current_timestamp() -> float:
    """Get current timestamp."""
    return datetime.now().timestamp()

def parse_datetime(date_str: str, format_str: str = "%Y-%m-%d %H:%M:%S") -> datetime:
    """Parse datetime from string."""
    return datetime.strptime(date_str, format_str)

def is_valid_datetime(date_str: str, format_str: str = "%Y-%m-%d %H:%M:%S") -> bool:
    """Check if string is valid datetime."""
    try:
        datetime.strptime(date_str, format_str)
        return True
    except ValueError:
        return False

# Export all functions
__all__ = [
    'format_datetime',
    'get_current_datetime',
    'get_current_timestamp',
    'parse_datetime',
    'is_valid_datetime'
]
