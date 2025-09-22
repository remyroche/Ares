"""
Compatibility wrappers for Cross-Validation utilities.

This module re-exports the canonical implementation from
`src.utils.ml_common.validation.cv_utils` to eliminate duplication while
preserving existing import paths and APIs used by downstream code.
"""

import logging
from typing import Optional

from src.utils.ml_common.validation.cv_utils import (
    CrossValidationUtilities as _CoreCrossValidationUtilities,
)

logger = logging.getLogger(__name__)


# Backwards-compatible alias
class CrossValidationUtilities(_CoreCrossValidationUtilities):
    """Alias for the canonical CrossValidationUtilities implementation."""

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config=config)
        try:
            logger.getChild('CrossValidationUtilities').info("Using canonical CV utils via compatibility wrapper")
        except Exception:
            pass


_cv_instance: Optional[CrossValidationUtilities] = None


def get_cross_validation_utilities() -> CrossValidationUtilities:
    """Return a process-wide singleton instance for compatibility."""
    global _cv_instance
    if _cv_instance is None:
        _cv_instance = CrossValidationUtilities()
    return _cv_instance


__all__ = [
    'CrossValidationUtilities',
    'get_cross_validation_utilities',
]
