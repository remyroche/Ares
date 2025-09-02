"""Warning symbols and color utilities for enhanced logging output.

This module provides warning symbols, color codes, and formatting utilities
for making log messages more visually distinctive and informative.
"""

import os
import sys
from typing import Any


class ColorCodes:
    """ANSI color codes for terminal output."""

    # Reset
    RESET = "\033[0m"

    # Bold
    BOLD = "\033[1m"

    # Colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright colors
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    # Background colors
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"

    # Bright background colors
    BG_BRIGHT_BLACK = "\033[100m"
    BG_BRIGHT_RED = "\033[101m"
    BG_BRIGHT_GREEN = "\033[102m"
    BG_BRIGHT_YELLOW = "\033[103m"
    BG_BRIGHT_BLUE = "\033[104m"
    BG_BRIGHT_MAGENTA = "\033[105m"
    BG_BRIGHT_CYAN = "\033[106m"
    BG_BRIGHT_WHITE = "\033[107m"


class WarningSymbols:
    """Warning symbols and status indicators for logging."""

    # Status symbols
    SUCCESS = "✅"
    ERROR = "❌"
    WARNING = "⚠️"
    INFO = "ℹ️"
    DEBUG = "🔍"
    TRACE = "🔎"

    # Process symbols
    START = "🚀"
    STOP = "🛑"
    RELOAD = "🔄"
    SAVE = "💾"
    LOAD = "📋"
    NOTIFY = "📢"
    WAIT = "⏳"
    DONE = "🎯"

    # Data symbols
    DATABASE = "🗄️"
    FILE = "📁"
    CONFIG = "⚙️"
    MODEL = "🤖"
    TRAINING = "🎓"
    VALIDATION = "✅"
    TESTING = "🧪"
    DEPLOYMENT = "🚀"

    # Error symbols
    CRITICAL = "💥"
    FAILED = "💀"
    TIMEOUT = "⏰"
    CONNECTION = "🔌"
    MEMORY = "🧠"
    CPU = "⚡"
    NETWORK = "🌐"
    SECURITY = "🔒"

    # Color mappings for different log levels
    LEVEL_COLORS = {
        "DEBUG": ColorCodes.CYAN,
        "INFO": ColorCodes.GREEN,
        "WARNING": ColorCodes.YELLOW,
        "ERROR": ColorCodes.RED,
        "CRITICAL": ColorCodes.BRIGHT_RED,
    }

    @classmethod
    def colorize(cls, text: str, color: str) -> str:
        """Apply color to text."""
        if not cls._supports_color():
            return text
        return f"{color}{text}{ColorCodes.RESET}"

    @classmethod
    def _supports_color(cls) -> bool:
        """Check if terminal supports color output."""
        return (
            hasattr(sys.stdout, "isatty")
            and sys.stdout.isatty()
            and os.environ.get("TERM") != "dumb"
        )


# Convenience functions for common warning symbols
def success(message: str) -> str:
    """Format success message."""
    return f"{WarningSymbols.SUCCESS} {message}"


def error(message: str) -> str:
    """Format error message."""
    return f"{WarningSymbols.ERROR} {message}"


def warning(message: str) -> str:
    """Format warning message."""
    return f"{WarningSymbols.WARNING} {message}"


def info(message: str) -> str:
    """Format info message."""
    return f"{WarningSymbols.INFO} {message}"


def debug(message: str) -> str:
    """Format debug message."""
    return f"{WarningSymbols.DEBUG} {message}"


def failed(message: str) -> str:
    """Format failed message."""
    return f"{WarningSymbols.FAILED} {message}"


def initialization_error(message: str) -> str:
    """Format initialization error message."""
    return f"{WarningSymbols.CRITICAL} {message}"


def invalid(message: str) -> str:
    """Format invalid message."""
    return f"{WarningSymbols.WARNING} {message}"


def missing(message: str) -> str:
    """Format missing message."""
    return f"{WarningSymbols.INFO} {message}"


# Color utility functions
def color_success(text: str) -> str:
    """Color text as success (green)."""
    return WarningSymbols.colorize(text, ColorCodes.GREEN)


def color_error(text: str) -> str:
    """Color text as error (red)."""
    return WarningSymbols.colorize(text, ColorCodes.RED)


def color_warning(text: str) -> str:
    """Color text as warning (yellow)."""
    return WarningSymbols.colorize(text, ColorCodes.YELLOW)


def color_info(text: str) -> str:
    """Color text as info (blue)."""
    return WarningSymbols.colorize(text, ColorCodes.BLUE)


def color_debug(text: str) -> str:
    """Color text as debug (cyan)."""
    return WarningSymbols.colorize(text, ColorCodes.CYAN)


def color_bold(text: str) -> str:
    """Make text bold."""
    return WarningSymbols.colorize(text, ColorCodes.BOLD)


def color_highlight(text: str) -> str:
    """Highlight text with bright yellow background."""
    return WarningSymbols.colorize(text, ColorCodes.BG_BRIGHT_YELLOW + ColorCodes.BLACK)


# Status formatting functions
def format_status(status: str, message: str) -> str:
    """Format status with appropriate symbol and color."""
    status_map = {
        "success": (WarningSymbols.SUCCESS, ColorCodes.GREEN),
        "error": (WarningSymbols.ERROR, ColorCodes.RED),
        "warning": (WarningSymbols.WARNING, ColorCodes.YELLOW),
        "info": (WarningSymbols.INFO, ColorCodes.BLUE),
        "debug": (WarningSymbols.DEBUG, ColorCodes.CYAN),
        "failed": (WarningSymbols.FAILED, ColorCodes.RED),
        "critical": (WarningSymbols.CRITICAL, ColorCodes.BRIGHT_RED),
    }
    
    symbol, color = status_map.get(status.lower(), (WarningSymbols.INFO, ColorCodes.WHITE))
    return WarningSymbols.colorize(f"{symbol} {message}", color)


def format_progress(current: int, total: int, message: str = "") -> str:
    """Format progress indicator."""
    percentage = (current / total) * 100 if total > 0 else 0
    progress_bar = "█" * int(percentage / 5) + "░" * (20 - int(percentage / 5))
    return f"{WarningSymbols.WAIT} [{progress_bar}] {percentage:.1f}% {message}"


def format_table(headers: list[str], rows: list[list[str]], title: str = "") -> str:
    """Format data as a table."""
    if not rows:
        return f"{WarningSymbols.INFO} {title}: No data"
    
    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    
    # Create separator
    separator = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    
    # Build table
    table = []
    if title:
        table.append(f"{WarningSymbols.INFO} {title}")
    table.append(separator)
    
    # Headers
    header_row = "|" + "|".join(f" {h:<{w}} " for h, w in zip(headers, col_widths)) + "|"
    table.append(header_row)
    table.append(separator)
    
    # Data rows
    for row in rows:
        data_row = "|" + "|".join(f" {str(cell):<{w}} " for cell, w in zip(row, col_widths)) + "|"
        table.append(data_row)
    
    table.append(separator)
    return "\n".join(table)
