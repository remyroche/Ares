"""Utilities for obtaining ML-common loggers.

This module centralizes creation of loggers used throughout
``src.utils.ml_common`` so that other modules do not need to replicate
fallback logic or import low-level logging primitives directly.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.utils.logger import get_system_logger


def get_ml_logger(name: str = "MLCommon", *, level: Optional[int] = None) -> logging.Logger:
    """Return a logger scoped to the ML common package.

    The function prefers the application's system logger hierarchy when
    available and gracefully falls back to the standard library's logging
    module if the unified logger has not been initialised yet.
    """
    try:
        base_logger = get_system_logger()
    except Exception:  # pragma: no cover - defensive fallback
        base_logger = logging.getLogger("Ares")

    if name:
        logger = base_logger.getChild(name)
    else:
        logger = base_logger

    if level is not None:
        logger.setLevel(level)

    return logger


def setup_ml_logger(name: str = "MLCommon", level: int = logging.INFO) -> logging.Logger:
    """Compatibility wrapper that configures and returns a child logger."""
    return get_ml_logger(name, level=level)


__all__ = ["get_ml_logger", "setup_ml_logger"]
