"""Optimization utilities shared across feature systems."""

# Import common utilities
from ..utils import (
    TPRINT_AVAILABLE, tprint,
    VECTORBT_OPTIMIZER_AVAILABLE, VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer,
    UnifiedVectorizationManager, get_unified_vectorization_manager
)

# Import consolidated CV implementations
try:
    from src.utils.ml_common.validation.consolidated_cv import (
        ConsolidatedCrossValidator, ConsolidatedCVConfig, ValidationType,
        create_purged_cv, create_temporal_cv
    )
    # Legacy aliases for backward compatibility
    BaseCVSplitter = ConsolidatedCrossValidator
    PurgedCVSplitter = ConsolidatedCrossValidator
    CV_AVAILABLE = True
except ImportError:
    BaseCVSplitter = None
    PurgedCVSplitter = None
    CV_AVAILABLE = False

__all__ = [
    'BaseCVSplitter',
    'PurgedCVSplitter',
]

# Add consolidated CV exports if available
if CV_AVAILABLE:
    __all__.extend([
        'ConsolidatedCrossValidator', 'ConsolidatedCVConfig', 'ValidationType',
        'create_purged_cv', 'create_temporal_cv'
    ])

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
