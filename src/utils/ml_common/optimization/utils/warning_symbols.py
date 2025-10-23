"""
Warning symbols and utilities for ML Common optimization components.
"""

# Warning symbols for consistent logging
error = "❌"
failed = "❌"
warning = "⚠️"
initialization_error = "🚫"
success = "✅"
info = "ℹ️"
processing = "🔄"
completed = "🎉"

# Status indicators
running = "🏃"
waiting = "⏳"
stopped = "🛑"
paused = "⏸️"

# Performance indicators
fast = "🚀"
slow = "🐌"
optimized = "⚡"
bottleneck = "🔥"

def format_error_message(message: str, symbol: str = error) -> str:
    """Format an error message with appropriate symbol."""
    return f"{symbol} {message}"

def format_success_message(message: str, symbol: str = success) -> str:
    """Format a success message with appropriate symbol."""
    return f"{symbol} {message}"

def format_warning_message(message: str, symbol: str = warning) -> str:
    """Format a warning message with appropriate symbol."""
    return f"{symbol} {message}"

def format_info_message(message: str, symbol: str = info) -> str:
    """Format an info message with appropriate symbol."""
    return f"{symbol} {message}"
