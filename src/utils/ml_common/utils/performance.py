"""Performance safeguards for retry loops and monitoring utilities."""
from __future__ import annotations

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


def bounded_sleep(delay: float, maximum: Optional[float] = None) -> None:
    """Sleep for a bounded amount of time, logging when limits are enforced."""
    if delay < 0:
        logger.warning("Negative sleep delay requested; defaulting to 0s")
        delay = 0
    if maximum is not None and delay > maximum:
        logger.warning("Requested sleep %.2fs exceeds maximum %.2fs; truncating", delay, maximum)
        delay = maximum
    time.sleep(delay)
