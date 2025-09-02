"""
Warning symbols and color codes for terminal output.

This module provides ANSI color codes and warning symbols for consistent
terminal output formatting across the Ares trading bot.
"""

import logging
from typing import Any, Dict, Optional

try:
    from .logger import system_logger
except ImportError:
    # Fallback for when running as standalone
    from src.utils.logger import system_logger

# ANSI color codes for terminal output
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


class ColorCodes:
    """ANSI color codes for terminal output."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize ColorCodes."""
        self.config = config or {}
        self.logger = system_logger.getChild("ColorCodes")
        self.is_initialized = False

    async def initialize(self) -> bool:
        """Initialize ColorCodes."""
        try:
            class_name = self.__class__.__name__
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {class_name}: {e}")
            return False

    def get_color(self, color_name: str) -> str:
        """Get a color code by name."""
        return getattr(self, color_name.upper(), RESET)

    def format_text(self, text: str, color: str, bold: bool = False) -> str:
        """Format text with color and optional bold."""
        result = ""
        if bold:
            result += BOLD
        result += color + text + RESET
        return result


class WarningSymbols:
    """Warning symbols and formatting for terminal output."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize WarningSymbols."""
        self.config = config or {}
        self.logger = system_logger.getChild("WarningSymbols")
        self.is_initialized = False

    async def initialize(self) -> bool:
        """Initialize WarningSymbols."""
        try:
            class_name = self.__class__.__name__
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {class_name}: {e}")
            return False

    def get_symbol(self, symbol_name: str) -> str:
        """Get a warning symbol by name."""
        symbols = {
            "error": "❌",
            "warning": "⚠️",
            "success": "✅",
            "info": "ℹ️",
            "debug": "🔍",
            "critical": "🚨",
            "stop": "🛑",
            "start": "🚀",
            "check": "✓",
            "cross": "✗",
            "arrow": "→",
            "star": "⭐",
            "fire": "🔥",
            "rocket": "🚀",
            "brain": "🧠",
            "money": "💰",
            "chart": "📊",
            "gear": "⚙️",
            "lock": "🔒",
            "unlock": "🔓",
        }
        return symbols.get(symbol_name, "•")


# Global instances
color_codes = ColorCodes()
warning_symbols = WarningSymbols()


# Convenience functions for backward compatibility
def error(text: str) -> str:
    """Format error text."""
    return f"{BRIGHT_RED}{text}{RESET}"


def warning(text: str) -> str:
    """Format warning text."""
    return f"{YELLOW}{text}{RESET}"


def success(text: str) -> str:
    """Format success text."""
    return f"{BRIGHT_GREEN}{text}{RESET}"


def info(text: str) -> str:
    """Format info text."""
    return f"{CYAN}{text}{RESET}"


def critical(text: str) -> str:
    """Format critical text."""
    return f"{BRIGHT_RED}{BOLD}{text}{RESET}"


def invalid(text: str) -> str:
    """Format invalid text."""
    return f"{BRIGHT_RED}{text}{RESET}"


def missing(text: str) -> str:
    """Format missing text."""
    return f"{YELLOW}{text}{RESET}"


# Initialize global instances
async def initialize_warning_symbols():
    """Initialize warning symbols globally."""
    await color_codes.initialize()
    await warning_symbols.initialize()


# Auto-initialize if running as main
if __name__ == "__main__":
    import asyncio
    asyncio.run(initialize_warning_symbols())
