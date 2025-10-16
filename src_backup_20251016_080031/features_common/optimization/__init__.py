"""Optimization utilities shared across feature systems."""

# Import common utilities
from ..utils import (
    TPRINT_AVAILABLE, tprint,
    VECTORBT_OPTIMIZER_AVAILABLE, VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer,
    UnifiedVectorizationManager, get_unified_vectorization_manager
)

from .cv_base import BaseCVSplitter, PurgedCVSplitter

__all__ = [
    'BaseCVSplitter',
    'PurgedCVSplitter',
]

# Add VectorBT optimization components to __all__ if available
if VECTORBT_OPTIMIZER_AVAILABLE:
    __all__.extend([
        'VectorBTRollingOptimizer',
        'get_vectorbt_rolling_optimizer',
        'UnifiedVectorizationManager',
        'get_unified_vectorization_manager',
    ])

if TPRINT_AVAILABLE:
    tprint(f"🔧 [optimization] Module initialized with {len(__all__)} exports", color="cyan")
else:
    print(f"🔧 [optimization] Module initialized with {len(__all__)} exports")
