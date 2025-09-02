"""
Warning Symbols and Color Codes
Provides standardized warning symbols and ANSI color codes for terminal output.
"""

from typing import Any, Dict, Optional
import logging

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
    
    # Background colors
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"

class WarningSymbols:
    """Standardized warning symbols for consistent messaging."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize WarningSymbols."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.is_initialized = False
    
    async def initialize(self) -> bool:
        """Initialize WarningSymbols."""
        try:
            self.logger.info("🚀 Initializing WarningSymbols...")
            self.is_initialized = True
            self.logger.info("✅ WarningSymbols initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing WarningSymbols: {e}")
            return False
    
    def get_symbol(self, level: str) -> str:
        """Get warning symbol for a given level."""
        symbols = {
            'info': 'ℹ️',
            'warning': '⚠️',
            'error': '❌',
            'critical': '🚨',
            'success': '✅',
            'debug': '🐛',
            'trace': '🔍',
            'fatal': '💀'
        }
        return symbols.get(level.lower(), '❓')

# Global instances
color_codes = ColorCodes()
warning_symbols = WarningSymbols()

# Convenience functions for easy access
def error(message: str, use_color: bool = True) -> str:
    """Format error message with symbol and optional color."""
    symbol = warning_symbols.get_symbol('error')
    if use_color:
        return f"{color_codes.BRIGHT_RED}{symbol} {message}{color_codes.RESET}"
    return f"{symbol} {message}"

def warning(message: str, use_color: bool = True) -> str:
    """Format warning message with symbol and optional color."""
    symbol = warning_symbols.get_symbol('warning')
    if use_color:
        return f"{color_codes.BRIGHT_YELLOW}{symbol} {message}{color_codes.RESET}"
    return f"{symbol} {message}"

def critical(message: str, use_color: bool = True) -> str:
    """Format critical message with symbol and optional color."""
    symbol = warning_symbols.get_symbol('critical')
    if use_color:
        return f"{color_codes.BRIGHT_RED}{color_codes.BOLD}{symbol} {message}{color_codes.RESET}"
    return f"{symbol} {message}"

def success(message: str, use_color: bool = True) -> str:
    """Format success message with symbol and optional color."""
    symbol = warning_symbols.get_symbol('success')
    if use_color:
        return f"{color_codes.BRIGHT_GREEN}{symbol} {message}{color_codes.RESET}"
    return f"{symbol} {message}"

def info(message: str, use_color: bool = True) -> str:
    """Format info message with symbol and optional color."""
    symbol = warning_symbols.get_symbol('info')
    if use_color:
        return f"{color_codes.BRIGHT_BLUE}{symbol} {message}{color_codes.RESET}"
    return f"{symbol} {message}"

def debug(message: str, use_color: bool = True) -> str:
    """Format debug message with symbol and optional color."""
    symbol = warning_symbols.get_symbol('debug')
    if use_color:
        return f"{color_codes.CYAN}{symbol} {message}{color_codes.RESET}"
    return f"{symbol} {message}"

def trace(message: str, use_color: bool = True) -> str:
    """Format trace message with symbol and optional color."""
    symbol = warning_symbols.get_symbol('trace')
    if use_color:
        return f"{color_codes.MAGENTA}{symbol} {message}{color_codes.RESET}"
    return f"{symbol} {message}"

def fatal(message: str, use_color: bool = True) -> str:
    """Format fatal message with symbol and optional color."""
    symbol = warning_symbols.get_symbol('fatal')
    if use_color:
        return f"{color_codes.BRIGHT_RED}{color_codes.BOLD}{color_codes.BG_BLACK}{symbol} {message}{color_codes.RESET}"
    return f"{symbol} {message}"

# Color formatting functions
def colorize(text: str, color: str, bold: bool = False) -> str:
    """Colorize text with specified color and optional bold."""
    if bold:
        return f"{color_codes.BOLD}{color}{text}{color_codes.RESET}"
    return f"{color}{text}{color_codes.RESET}"

def highlight(text: str, bg_color: str = None, fg_color: str = None) -> str:
    """Highlight text with background and/or foreground color."""
    result = text
    if bg_color:
        result = f"{bg_color}{result}"
    if fg_color:
        result = f"{fg_color}{result}"
    if bg_color or fg_color:
        result = f"{result}{color_codes.RESET}"
    return result

# Status indicators
def status_indicator(status: str, use_color: bool = True) -> str:
    """Get status indicator for common statuses."""
    indicators = {
        'running': '🔄',
        'completed': '✅',
        'failed': '❌',
        'pending': '⏳',
        'cancelled': '🚫',
        'skipped': '⏭️',
        'warning': '⚠️',
        'info': 'ℹ️'
    }
    
    symbol = indicators.get(status.lower(), '❓')
    if use_color:
        if status.lower() in ['completed', 'success']:
            return f"{color_codes.BRIGHT_GREEN}{symbol}"
        elif status.lower() in ['failed', 'error', 'critical']:
            return f"{color_codes.BRIGHT_RED}{symbol}"
        elif status.lower() in ['warning']:
            return f"{color_codes.BRIGHT_YELLOW}{symbol}"
        elif status.lower() in ['running', 'pending']:
            return f"{color_codes.BRIGHT_BLUE}{symbol}"
        else:
            return f"{color_codes.WHITE}{symbol}"
    return symbol

# Progress indicators
def progress_bar(current: int, total: int, width: int = 50, use_color: bool = True) -> str:
    """Create a progress bar."""
    if total == 0:
        return "[]"
    
    percentage = current / total
    filled_width = int(width * percentage)
    bar = "█" * filled_width + "░" * (width - filled_width)
    
    if use_color:
        if percentage >= 0.8:
            color = color_codes.BRIGHT_GREEN
        elif percentage >= 0.6:
            color = color_codes.BRIGHT_YELLOW
        elif percentage >= 0.4:
            color = color_codes.YELLOW
        else:
            color = color_codes.BRIGHT_RED
        
        return f"[{color}{bar}{color_codes.RESET}] {percentage:.1%}"
    
    return f"[{bar}] {percentage:.1%}"

# Export all symbols and functions
__all__ = [
    'ColorCodes',
    'WarningSymbols',
    'color_codes',
    'warning_symbols',
    'error',
    'warning',
    'critical',
    'success',
    'info',
    'debug',
    'trace',
    'fatal',
    'colorize',
    'highlight',
    'status_indicator',
    'progress_bar'
]

if __name__ == "__main__":
    # Test the warning symbols and color codes
    print("Testing warning symbols and color codes...")
    
    print(error("This is an error message"))
    print(warning("This is a warning message"))
    print(critical("This is a critical message"))
    print(success("This is a success message"))
    print(info("This is an info message"))
    print(debug("This is a debug message"))
    
    print(f"\nStatus indicators:")
    print(f"Running: {status_indicator('running')}")
    print(f"Completed: {status_indicator('completed')}")
    print(f"Failed: {status_indicator('failed')}")
    
    print(f"\nProgress bar:")
    print(progress_bar(75, 100))
    
    print("\nAll tests completed successfully!")
