"""Typed logging utilities for the project.

- `system_logger`: process-wide logger
- `get_logger(name)`: child loggers
"""
from __future__ import annotations

import logging
from logging import Logger
from typing import Optional


def _ensure_stream_handler(logger: Logger) -> None:
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setLevel(logging.NOTSET)
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)


def setup_logging(level: int | str = logging.INFO, *, name: str = "System") -> Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level if isinstance(level, int) else logging.getLevelName(level))
    _ensure_stream_handler(logger)
    logger.propagate = False
    return logger


system_logger: Logger = setup_logging()


def get_logger(name: str, *, parent: Optional[Logger] = None) -> Logger:
    base = parent or system_logger
    child = base.getChild(name)
    _ensure_stream_handler(child)
    child.propagate = False
    return child
