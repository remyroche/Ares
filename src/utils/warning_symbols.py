"""Warning symbols and color utilities for enhanced logging output.

This module provides warning symbols = color codes = and formatting utilities
for making log messages more visually distinctive and informative.
"""

import os
from src.utils.pipeline_standards import PipelineStandards, pipeline_standards
import sys


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


class WarningSymbols:
    """Unicode warning symbols for enhanced visual feedback."""

    # Success symbols
    CHECKMARK = "✅"
    SUCCESS_CIRCLE = "🟢"
    THUMBS_UP = "👍"

    # Warning symbols
    WARNING_TRIANGLE = "⚠️"
    WARNING_SIGN = "🚨"
    EXCLAMATION = "❗"

    # Error symbols
    RED_CROSS = "❌"
    FAILURE_SYMBOL = "💥"
    PROBLEM_SYMBOL = "🚫"
    ERROR_SYMBOL = "🔥"

    # Info symbols
    INFO_CIRCLE = "ℹ️"
    LIGHT_BULB = "💡"
    MAGNIFYING_GLASS = "🔍"

    # Process symbols
    GEAR = "⚙️"
    HOURGLASS = "⏳"
    CLOCK = "🕐"
    ARROW = "➡️"

    # Status symbols
    PLAY = "▶️"
    PAUSE = "⏸️"
    STOP = "⏹️"
    LOADING = "🔄"

    # Data symbols
    DATABASE = "🗄️"
    FILE = "📁"
    CHART = "📊"
    GRAPH = "📈"

    # Network symbols
    GLOBE = "🌐"
    WIFI = "📶"
    SERVER = "🖥️"
    CONNECTION = "🔗"


def should_use_colors() -> bool:
    """Determine if colors should be used based on environment.

    Returns:
        bool: True if colors should be used

    """
    # Check if we're in a terminal
    if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
        return False

    # Check for NO_COLOR environment variable
    if os.environ.get("NO_COLOR"):
        return False

    # Check for TERM environment variable
    term = os.environ.get("TERM", "").lower()
    return term not in ("dumb", "unknown")


def colorize(text: str, color: str, bold: bool = False) -> str:
    """Add color to text if colors are enabled.

    Args:
        text: Text to colorize
        color: Color code to apply
        bold: Whether to make text bold

    Returns:
        Colorized text or original text if colors disabled

    """
    if not should_use_colors():
        return text

    result = text
    if bold:
        result = f"{ColorCodes.BOLD}{result}"

    return f"{color}{result}{ColorCodes.RESET}"


def format_warning_message(
    message: str,
    symbol: str = WarningSymbols.WARNING_TRIANGLE,
    color: str = ColorCodes.BRIGHT_YELLOW,
    bold: bool = True,
) -> str:
    """Format a warning message with symbol and color.

    Args:
        message: The warning message
        symbol: Warning symbol to use
        color: Color code to apply
        bold: Whether to make text bold

    Returns:
        Formatted warning message

    """
    formatted_symbol = colorize(symbol, color, bold)
    formatted_message = colorize(message, color, bold)
    return f"{formatted_symbol} {formatted_message}"


def format_error_message(
    message: str,
    symbol: str = WarningSymbols.RED_CROSS,
    color: str = ColorCodes.BRIGHT_RED,
    bold: bool = True,
) -> str:
    """Format an error message with symbol and color.

    Args:
        message: The error message
        symbol: Error symbol to use
        color: Color code to apply
        bold: Whether to make text bold

    Returns:
        Formatted error message

    """
    formatted_symbol = colorize(symbol, color, bold)
    formatted_message = colorize(message, color, bold)
    return f"{formatted_symbol} {formatted_message}"


def format_critical_message(
    message: str,
    symbol: str = WarningSymbols.FAILURE_SYMBOL,
    color: str = ColorCodes.BRIGHT_RED,
    bold: bool = True,
) -> str:
    """Format a critical error message with symbol and color.

    Args:
        message: The critical error message
        symbol: Critical error symbol to use
        color: Color code to apply
        bold: Whether to make text bold

    Returns:
        Formatted critical error message

    """
    formatted_symbol = colorize(symbol, color, bold)
    formatted_message = colorize(message, color, bold)
    return f"{formatted_symbol} {formatted_message}"


def format_problem_message(
    message: str,
    symbol: str = WarningSymbols.PROBLEM_SYMBOL,
    color: str = ColorCodes.BRIGHT_RED,
    bold: bool = True,
) -> str:
    """Format a problem message with symbol and color.

    Args:
        message: The problem message
        symbol: Problem symbol to use
        color: Color code to apply
        bold: Whether to make text bold

    Returns:
        Formatted problem message

    """
    formatted_symbol = colorize(symbol, color, bold)
    formatted_message = colorize(message, color, bold)
    return f"{formatted_symbol} {formatted_message}"


def format_success_message(
    message: str,
    symbol: str = WarningSymbols.CHECKMARK,
    color: str = ColorCodes.BRIGHT_GREEN,
    bold: bool = True,
) -> str:
    """Format a success message with symbol and color.

    Args:
        message: The success message
        symbol: Success symbol to use
        color: Color code to apply
        bold: Whether to make text bold

    Returns:
        Formatted success message

    """
    formatted_symbol = colorize(symbol, color, bold)
    formatted_message = colorize(message, color, bold)
    return f"{formatted_symbol} {formatted_message}"


def format_info_message(
    message: str,
    symbol: str = WarningSymbols.INFO_CIRCLE,
    color: str = ColorCodes.BRIGHT_BLUE,
    bold: bool = False,
) -> str:
    """Format an info message with symbol and color.

    Args:
        message: The info message
        symbol: Info symbol to use
        color: Color code to apply
        bold: Whether to make text bold

    Returns:
        Formatted info message

    """
    formatted_symbol = colorize(symbol, color, bold)
    formatted_message = colorize(message, color, bold)
    return f"{formatted_symbol} {formatted_message}"


# Convenience functions for common warning types
def warning(message: str) -> str:
    """Format a warning message."""
    return format_warning_message(message)


def error(message: str) -> str:
    """Format an error message."""
    return format_error_message(message)


def critical(message: str) -> str:
    """Format a critical error message."""
    return format_critical_message(message)


def failed(message: str) -> str:
    """Format a failure message."""
    return format_problem_message(message)


def success(message: str) -> str:
    """Format a success message."""
    return format_success_message(message)


def info(message: str) -> str:
    """Format an info message."""
    return format_info_message(message)


def initialization_error(message: str) -> str:
    """Format an initialization error message."""
    return format_error_message(message)


def invalid(message: str) -> str:
    """Format an invalid input message."""
    return format_problem_message(message)


def missing(message: str) -> str:
    """Format a missing data message."""
    return format_warning_message(message)


def problem(message: str) -> str:
    """Format a problem message."""
    return format_problem_message(message)


def timeout(message: str) -> str:
    """Format a timeout message."""
    return format_error_message(message)


def connection_error(message: str) -> str:
    """Format a connection error message."""
    return format_error_message(message)


def validation_error(message: str) -> str:
    """Format a validation error message."""
    return format_error_message(message)


def execution_error(message: str) -> str:
    """Format an execution error message."""
    return format_error_message(message)
