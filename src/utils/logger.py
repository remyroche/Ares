"""
Centralized logging configuration with Standardized Import Management.

This module provides a unified logging system with JSON formatting,
file rotation, and console output capabilities.
"""

import logging
import logging.handlers
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from contextlib import contextmanager
import threading

# Create a basic system logger
system_logger = logging.getLogger("ares_system")
system_logger.setLevel(logging.INFO)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
console_handler.setFormatter(formatter)

# Add handler to logger
system_logger.addHandler(console_handler)

# Prevent duplicate logs
system_logger.propagate = False

# Basic logging functions
def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name."""
    return system_logger.getChild(name)

def set_log_level(level: str) -> None:
    """Set the logging level."""
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    if level.upper() in level_map:
        system_logger.setLevel(level_map[level.upper()])
        console_handler.setLevel(level_map[level.upper()])

# Export the main logger
__all__ = ['system_logger', 'get_logger', 'set_log_level']
