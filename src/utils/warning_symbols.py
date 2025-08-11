"""
Warning symbols and colors for logging when things aren't working as expected.

This module provides standardized warning symbols and color codes
for consistent error and warning logging throughout the Ares trading bot.
"""

import os


class WarningSymbols:
    """Standardized warning symbols for logging."""

    # Red cross symbols for errors
    RED_CROSS = "❌"
    RED_X = "✗"
    RED_CROSS_BOLD = "✖"

    # Warning symbols
    WARNING_TRIANGLE = "⚠️"
    WARNING_SIGN = "⚠"
    WARNING_EXCLAMATION = "❗"
    WARNING_DOUBLE_EXCLAMATION = "‼"

    # Error symbols
    ERROR_SYMBOL = "🚨"
    ERROR_CIRCLE = "⭕"
    ERROR_STOP = "🛑"

    # Failure symbols
    FAILURE_SYMBOL = "💥"
    FAILURE_BOMB = "💣"
    FAILURE_SKULL = "💀"

    # Problem indicators
    PROBLEM_SYMBOL = "🔴"
    PROBLEM_DOT = "●"
    PROBLEM_CIRCLE = "🔴"

    # Status indicators
    FAILED = "❌"
    ERROR = "🚨"
    WARNING = "⚠️"
    PROBLEM = "🔴"
    CRITICAL = "💥"


class ColorCodes:
    """ANSI color codes for terminal output."""

    # Reset all formatting
    RESET = "\033[0m"

    # Text colors
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
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"

    # Text formatting
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    BLINK = "\033[5m"
    REVERSE = "\033[7m"
    HIDDEN = "\033[8m"


def should_use_colors() -> bool:
    """Check if colors should be used based on environment."""
    # Check if NO_COLOR environment variable is set
    if os.environ.get("NO_COLOR"):
        return False

    # Check if TERM is set and supports colors
    term = os.environ.get("TERM", "")
    if term in ("dumb", "unknown"):
        return False

    # Check if we're in a non-interactive environment
    if not os.isatty(1):  # stdout
        return False

    return True


def colorize(text: str, color: str, bold: bool = False) -> str:
    """
    Add color to text if colors are enabled.

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
    """
    Format a warning message with symbol and color.

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
    """
    Format an error message with symbol and color.

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
    """
    Format a critical error message with symbol and color.

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
    """
    Format a problem message with symbol and color.

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


# Convenience functions for common warning patterns
def warning(message: str) -> str:
    """Format a warning message."""
    return format_warning_message(message)


def error(message: str) -> str:
    """Format an error message."""
    return format_error_message(message)


def critical(message: str) -> str:
    """Format a critical error message."""
    return format_critical_message(message)


def problem(message: str) -> str:
    """Format a problem message."""
    return format_problem_message(message)


def failed(message: str) -> str:
    """Format a failure message."""
    return format_error_message(message, WarningSymbols.FAILED)


def invalid(message: str) -> str:
    """Format an invalid configuration/input message."""
    return format_error_message(message, WarningSymbols.RED_X)


def missing(message: str) -> str:
    """Format a missing data/configuration message."""
    return format_warning_message(message, WarningSymbols.WARNING_EXCLAMATION)


def timeout(message: str) -> str:
    """Format a timeout message."""
    return format_warning_message(
        message,
        WarningSymbols.WARNING_TRIANGLE,
        ColorCodes.BRIGHT_YELLOW,
    )


def connection_error(message: str) -> str:
    """Format a connection error message."""
    return format_error_message(message, WarningSymbols.ERROR_SYMBOL)


def validation_error(message: str) -> str:
    """Format a validation error message."""
    return format_error_message(message, WarningSymbols.RED_X)


def initialization_error(message: str) -> str:
    """Format an initialization error message."""
    return format_error_message(message, WarningSymbols.RED_CROSS)


def execution_error(message: str) -> str:
    """Format an execution error message."""
    return format_error_message(message, WarningSymbols.ERROR_SYMBOL)
