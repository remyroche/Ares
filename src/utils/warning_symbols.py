"""
Warning symbols and color codes for terminal output.
"""

from typing import Any, Dict, Optional
import logging

# Get system logger
try:
    from src.utils.logger import system_logger
except ImportError:
    system_logger = logging.getLogger(__name__)


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


from src.utils.logger import system_logger

class WarningSymbols:
    """Warning symbols and formatting utilities."""


    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize WarningSymbols."""
        self.config = config or {}
        self.logger = system_logger.getChild("WarningSymbols")
        self.is_initialized = True

    def warning(self, message: str) -> str:
        """Format a warning message."""
        return f"{ColorCodes.YELLOW}⚠️  WARNING: {message}{ColorCodes.RESET}"

    def error(self, message: str) -> str:
        """Format an error message."""
        return f"{ColorCodes.RED}❌ ERROR: {message}{ColorCodes.RESET}"

    def success(self, message: str) -> str:
        """Format a success message."""
        return f"{ColorCodes.GREEN}✅ SUCCESS: {message}{ColorCodes.RESET}"

    def info(self, message: str) -> str:
        """Format an info message."""
        return f"{ColorCodes.BLUE}ℹ️  INFO: {message}{ColorCodes.RESET}"

    def critical(self, message: str) -> str:
        """Format a critical message."""
        return f"{ColorCodes.BRIGHT_RED}🚨 CRITICAL: {message}{ColorCodes.RESET}"

    def debug(self, message: str) -> str:
        """Format a debug message."""
        return f"{ColorCodes.CYAN}🐛 DEBUG: {message}{ColorCodes.RESET}"


# Global instances
color_codes = ColorCodes()
warning_symbols = WarningSymbols()

# Convenience functions
def warning(message: str) -> str:
    """Format a warning message."""
    return warning_symbols.warning(message)


def error(message: str) -> str:
    """Format an error message."""
    return warning_symbols.error(message)


def success(message: str) -> str:
    """Format a success message."""
    return warning_symbols.success(message)


def info(message: str) -> str:
    """Format an info message."""
    return warning_symbols.info(message)


def critical(message: str) -> str:
    """Format a critical message."""
    return warning_symbols.critical(message)


def debug(message: str) -> str:
    """Format a debug message."""
    return warning_symbols.debug(message)


def failed(message: str) -> str:
    """Format a failure message."""
    return warning_symbols.error(message)

