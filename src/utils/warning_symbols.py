"""ANSI color codes and simple formatting helpers for logging output."""
from __future__ import annotations

# Reset and styles
RESET = "\033[0m"
BOLD = "\033[1m"

# Basic colors
BLACK = "\033[30m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"

# Bright (high-intensity) colors
BRIGHT_RED = "\033[91m"
BRIGHT_GREEN = "\033[92m"
BRIGHT_YELLOW = "\033[93m"
BRIGHT_BLUE = "\033[94m"
BRIGHT_MAGENTA = "\033[95m"
BRIGHT_CYAN = "\033[96m"
BRIGHT_WHITE = "\033[97m"


def colorize(text: str, color: str = RESET, *, bold: bool = False) -> str:
    styled = f"{color}{text}{RESET}"
    if bold:
        styled = f"{BOLD}{styled}"
    return styled


def format_warning_message(message: str) -> str:
    return colorize(message, BRIGHT_YELLOW, bold=True)


def format_error_message(message: str) -> str:
    return colorize(message, BRIGHT_RED, bold=True)


def format_info_message(message: str) -> str:
    return colorize(message, CYAN, bold=False)
