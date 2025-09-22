"""
Deprecated re-export. Use one of the canonical imports instead:

- from src.utils.parallel_processing_optimizer import ParallelProcessor
- from src.utils.ml_common.utils import ParallelProcessor
"""

from src.utils.parallel_processing_optimizer import ParallelProcessor  # kept for backward compatibility

__all__ = ['ParallelProcessor']
