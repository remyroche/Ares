from src.utils.tprint import tprint

"""Warning symbols and constants.

Provide both string symbols and callable helpers expected by logger and other modules.
"""
import logging

# Symbol constants
warning_symbol = "⚠️"
invalid_symbol = "❌"
missing_symbol = "❓"
error_symbol = "⛔"
failed_symbol = "💥"
critical_symbol = "🔥"

# Callable helpers used by logger fallback pattern
def _print_with(label: str, msg: object) -> None:
    try:
        tprint(f"{label}: {msg}")
    except Exception:
        tprint(f"{label}")

def warning(msg: object) -> None:  # type: ignore[func-assign]
    _print_with("WARNING", msg)

def invalid(msg: object) -> None:  # type: ignore[func-assign]
    _print_with("INVALID", msg)

def error(msg: object) -> None:  # type: ignore[func-assign]
    _print_with("ERROR", msg)

def failed(msg: object) -> None:  # type: ignore[func-assign]
    _print_with("FAILED", msg)

def critical(msg: object) -> None:  # type: ignore[func-assign]
    _print_with("CRITICAL", msg)

def missing(msg: object) -> None:  # type: ignore[func-assign]
    _print_with("MISSING", msg)

# Export all symbols and functions
__all__ = [
    'warning',
    'invalid',
    'error',
    'failed',
    'critical',
    'missing',
    # Symbol constants
    'warning_symbol',
    'invalid_symbol',
    'missing_symbol',
    'error_symbol',
    'failed_symbol',
    'critical_symbol'
]
