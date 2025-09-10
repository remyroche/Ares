from src.training.steps.standardized_parquet_handler import standardized_parquet_handler
"""
Step 12 Modular: Logger Setup

This module provides logging configuration for Step 12.
"""

import logging

from .imports import system_logger

def setup_step12_logger(name: str = "step12_analyst_enhancement") -> logging.Logger:
    """Setup logger for Step 12."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)
        logger.parent = system_logger

    return logger

__all__ = ['setup_step12_logger']
