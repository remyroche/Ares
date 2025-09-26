"""Memory management helpers for training steps.

The audit highlighted that several training loops create large intermediate
structures without an explicit clean-up strategy.  The utilities in this module
provide lightweight context managers and helpers that make it trivial for the
training steps to monitor and release memory deterministically.
"""

from __future__ import annotations

import gc
import logging
import os
from contextlib import contextmanager
from typing import Iterator, Optional

try:  # pragma: no cover - best effort optional dependency
    import psutil  # type: ignore
except Exception:  # pragma: no cover - psutil may not be installed
    psutil = None  # type: ignore


def _current_memory_mb() -> Optional[float]:
    """Return the current process memory usage in megabytes if possible."""

    if psutil is None:  # pragma: no cover - depends on environment
        return None

    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


@contextmanager
def memory_guard(description: str, logger: Optional[logging.Logger] = None) -> Iterator[None]:
    """Context manager that monitors memory usage for a code block.

    The guard records the process RSS before and after the block, triggers a
    garbage collection cycle, and logs the delta.  It is intentionally very
    lightweight so it can wrap fine-grained operations such as per-regime
    training loops.
    """

    start = _current_memory_mb()
    try:
        yield
    finally:
        gc.collect()
        end = _current_memory_mb()

        if start is not None and end is not None and logger is not None:
            delta = end - start
            logger.debug(
                "Memory guard '%s' delta: %.2f MB (start=%.2f MB, end=%.2f MB)",
                description,
                delta,
                start,
                end,
            )


def aggressive_gc(logger: Optional[logging.Logger] = None, context: str = "") -> None:
    """Trigger an explicit garbage collection cycle with optional logging."""

    collected = gc.collect()
    if logger is not None:
        logger.debug("Garbage collection (%s) reclaimed %s objects", context, collected)


__all__ = ["aggressive_gc", "memory_guard"]

