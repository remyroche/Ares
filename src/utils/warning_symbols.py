"""
Warning Symbols Module

This module provides warning symbols and color codes for terminal output,
including functions for creating warning messages and error indicators.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:

from src.utils.logger import system_logger

# ANSI color codes for terminal output

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
    """Warning symbols and message formatting utilities."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize WarningSymbols."""
        self.config = config or {}
        self.logger = system_logger.getChild("WarningSymbols")
        self.is_initialized = False
        
        # Default warning symbols
        self.symbols = {
            "warning": "⚠️",
            "error": "❌",
            "success": "✅",
            "info": "ℹ️",
            "loading": "🔄",
            "check": "✓",
            "cross": "✗",
            "arrow": "→",
            "star": "⭐",
            "fire": "🔥",
            "rocket": "🚀",
            "gear": "⚙️",
            "lock": "🔒",
            "unlock": "🔓",
            "clock": "⏰",
            "calendar": "📅",
            "chart": "📊",
            "database": "💾",
            "network": "🌐",
            "security": "🔐"
        }
        
        # Color mappings
        self.colors = {
            "warning": ColorCodes.YELLOW,
            "error": ColorCodes.RED,
            "success": ColorCodes.GREEN,
            "info": ColorCodes.BLUE,
            "loading": ColorCodes.CYAN,
            "default": ColorCodes.WHITE
        }
        
        self.is_initialized = True
        self.logger.info("WarningSymbols initialized successfully")
    
    def get_symbol(self, symbol_name: str) -> str:
        """Get a warning symbol by name."""
        try:
            return self.symbols.get(symbol_name, "•")
        except Exception as e:
            self.logger.error(f"Error getting symbol {symbol_name}: {e}")
            return "•"
    
    def get_color(self, color_name: str) -> str:
        """Get a color code by name."""
        try:
            return self.colors.get(color_name, ColorCodes.WHITE)
        except Exception as e:
            self.logger.error(f"Error getting color {color_name}: {e}")
            return ColorCodes.WHITE
    
    def format_message(self, message: str, symbol: str = "info", color: str = "default", bold: bool = False) -> str:
        """Format a message with symbol and color."""
        try:
            symbol_char = self.get_symbol(symbol)
            color_code = self.get_color(color)
            bold_code = ColorCodes.BOLD if bold else ""
            
            formatted = f"{color_code}{bold_code}{symbol_char} {message}{ColorCodes.RESET}"
            return formatted
        except Exception as e:
            self.logger.error(f"Error formatting message: {e}")
            return message
    
    def warning(self, message: str, bold: bool = False) -> str:
        """Create a warning message."""
        return self.format_message(message, "warning", "warning", bold)
    
    def error(self, message: str, bold: bool = False) -> str:
        """Create an error message."""
        return self.format_message(message, "error", "error", bold)
    
    def success(self, message: str, bold: bool = False) -> str:
        """Create a success message."""
        return self.format_message(message, "success", "success", bold)
    
    def info(self, message: str, bold: bool = False) -> str:
        """Create an info message."""
        return self.format_message(message, "info", "info", bold)
    
    def loading(self, message: str, bold: bool = False) -> str:
        """Create a loading message."""
        return self.format_message(message, "loading", "loading", bold)
    
    def custom(self, message: str, symbol: str, color: str = "default", bold: bool = False) -> str:
        """Create a custom formatted message."""
        return self.format_message(message, symbol, color, bold)
    
    def add_symbol(self, name: str, symbol: str) -> None:
        """Add a custom symbol."""
        try:
            self.symbols[name] = symbol
            self.logger.info(f"Added custom symbol: {name} = {symbol}")
        except Exception as e:
            self.logger.error(f"Error adding symbol {name}: {e}")
    
    def add_color(self, name: str, color_code: str) -> None:
        """Add a custom color."""
        try:
            self.colors[name] = color_code
            self.logger.info(f"Added custom color: {name} = {color_code}")
        except Exception as e:
            self.logger.error(f"Error adding color {name}: {e}")
    
    def get_all_symbols(self) -> Dict[str, str]:
        """Get all available symbols."""
        return self.symbols.copy()
    
    def get_all_colors(self) -> Dict[str, str]:
        """Get all available colors."""
        return self.colors.copy()
    
    def reset_colors(self) -> None:
        """Reset colors to default."""
        try:
            self.colors = {
                "warning": ColorCodes.YELLOW,
                "error": ColorCodes.RED,
                "success": ColorCodes.GREEN,
                "info": ColorCodes.BLUE,
                "loading": ColorCodes.CYAN,
                "default": ColorCodes.WHITE
            }
            self.logger.info("Colors reset to defaults")
        except Exception as e:
            self.logger.error(f"Error resetting colors: {e}")

# Global warning symbols instance
warning_symbols = WarningSymbols()

# Convenience functions for backward compatibility
def warning(message: str, bold: bool = False) -> str:
    """Create a warning message."""
    return warning_symbols.warning(message, bold)

def error(message: str, bold: bool = False) -> str:
    """Create an error message."""
    return warning_symbols.error(message, bold)

def success(message: str, bold: bool = False) -> str:
    """Create a success message."""
    return warning_symbols.success(message, bold)

def info(message: str, bold: bool = False) -> str:
    """Create an info message."""
    return warning_symbols.info(message, bold)

def loading(message: str, bold: bool = False) -> str:
    """Create a loading message."""
    return warning_symbols.loading(message, bold)

def failed(message: str, bold: bool = False) -> str:
    """Create a failed message (alias for error)."""
    return warning_symbols.error(message, bold)

def missing(message: str, bold: bool = False) -> str:
    """Create a missing message (alias for warning)."""
    return warning_symbols.warning(message, bold)

# Convenience function for creating warning symbols instance
def create_warning_symbols(config: Optional[Dict[str, Any]] = None) -> WarningSymbols:
    """Create a new WarningSymbols instance."""
    return WarningSymbols(config)

