"""
ML Common - Logger Module

This module provides logging utilities for the ML Common package.
"""

import logging
from ...utils.logger import get_logger as base_get_logger

# Re-export the base logger function
def get_logger(name: str = "MLCommon") -> logging.Logger:
    """Get a logger instance with the specified name."""
    try:
        return base_get_logger(name)
    except Exception:
        # Fallback to standard logging
        return logging.getLogger(name)

def setup_logger(name: str = "MLCommon", level: int = logging.INFO) -> logging.Logger:
    """Setup and return a logger with the specified configuration."""
    logger = get_logger(name)
    logger.setLevel(level)
    return logger

__all__ = ['get_logger', 'setup_logger']
