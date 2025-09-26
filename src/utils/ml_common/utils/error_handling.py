"""Centralised error handling utilities for ml_common."""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Callable, Iterator, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def log_and_raise(message: str, exc: BaseException) -> None:
    """Log an exception and re-raise it with context."""
    logger.error(message, exc_info=exc)
    raise exc


def log_and_return_default(message: str, exc: BaseException, default: T) -> T:
    """Log an exception and return a safe default value."""
    logger.warning(message, exc_info=exc)
    return default


@contextmanager
def suppress_with_logging(message: str) -> Iterator[None]:
    """Context manager that logs exceptions and prevents silent failures."""
    try:
        yield
    except Exception as exc:  # pragma: no cover - defensive path
        logger.exception(message, exc_info=exc)


def guard_execution(action: Callable[[], T], message: str, default: T) -> T:
    """Execute an action, logging failures and returning a default value."""
    try:
        return action()
    except Exception as exc:  # pragma: no cover - defensive path
        logger.exception(message, exc_info=exc)
        return default
