"""
Evaluation module for ML common utilities.
"""

import logging
from typing import Optional

from .evaluation_utils import EvaluationUtils

# Global evaluation utils instance
_evaluation_utils: Optional['EvaluationUtils'] = None

def get_evaluation_utils() -> 'EvaluationUtils':
    """Get or create the global evaluation utilities instance."""
    global _evaluation_utils

    if _evaluation_utils is None:
        _evaluation_utils = EvaluationUtils()
        logging.info("✅ Evaluation utilities initialized")

    return _evaluation_utils

__all__ = [
    'EvaluationUtils',
    'get_evaluation_utils'
]
