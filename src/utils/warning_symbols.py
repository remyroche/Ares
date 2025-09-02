"""
Warning Symbols and Color Codes for Enhanced Logging

This module provides warning symbols, color codes, and enhanced logging capabilities
for the Ares trading bot system.
"""

import logging
from typing import Any, Dict, Optional
from src.utils.logger import system_logger

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
    
    def __init__(self, config: Dict[str, Any] | None = None):
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

class WarningSymbols:
    """Warning symbols and emojis for enhanced logging."""
    
    # Status symbols
    SUCCESS = "✅"
    WARNING = "⚠️"
    ERROR = "❌"
    INFO = "ℹ️"
    DEBUG = "🔍"
    LOADING = "🔄"
    COMPLETE = "🎯"
    FAILED = "💥"
    SKIPPED = "⏭️"
    VALIDATED = "🔒"
    
    # Process symbols
    START = "🚀"
    STOP = "🛑"
    PAUSE = "⏸️"
    RESUME = "▶️"
    CLEANUP = "🧹"
    OPTIMIZATION = "⚡"
    MONITORING = "📊"
    ALERT = "🚨"
    
    # Data symbols
    DATA = "📊"
    FILE = "📁"
    DATABASE = "🗄️"
    NETWORK = "🌐"
    CACHE = "💾"
    QUEUE = "📋"
    STACK = "📚"
    
    def __init__(self, config: Dict[str, Any] | None = None):
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

# Create global instances
color_codes = ColorCodes()
warning_symbols = WarningSymbols()

# Convenience functions for easy access
def error(message: str, **kwargs) -> str:
    """Format error message with color and symbol."""
    return f"{color_codes.BRIGHT_RED}{warning_symbols.ERROR} {message}{color_codes.RESET}"

def warning(message: str, **kwargs) -> str:
    """Format warning message with color and symbol."""
    return f"{color_codes.BRIGHT_YELLOW}{warning_symbols.WARNING} {message}{color_codes.RESET}"

def critical(message: str, **kwargs) -> str:
    """Format critical message with color and symbol."""
    return f"{color_codes.BRIGHT_RED}{color_codes.BOLD}{warning_symbols.ALERT} {message}{color_codes.RESET}"

def success(message: str, **kwargs) -> str:
    """Format success message with color and symbol."""
    return f"{color_codes.BRIGHT_GREEN}{warning_symbols.SUCCESS} {message}{color_codes.RESET}"

def info(message: str, **kwargs) -> str:
    """Format info message with color and symbol."""
    return f"{color_codes.BRIGHT_BLUE}{warning_symbols.INFO} {message}{color_codes.RESET}"

def debug(message: str, **kwargs) -> str:
    """Format debug message with color and symbol."""
    return f"{color_codes.CYAN}{warning_symbols.DEBUG} {message}{color_codes.RESET}"

# Export the main symbols and functions
__all__ = [
    "ColorCodes",
    "WarningSymbols",
    "color_codes",
    "warning_symbols",
    "error",
    "warning",
    "critical",
    "success",
    "info",
    "debug"
]
