"""Optimization utilities shared across feature systems."""

# Import utility functions
try:
    from src.utils.tprint import tprint
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False

from .cv_base import BaseCVSplitter, PurgedCVSplitter

# Import VectorBT optimization components
try:
    from src.feature_generation.utils.vectorbt_rolling_optimizer import VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer
    from src.feature_generation.utils.unified_vectorization_manager import UnifiedVectorizationManager, get_unified_vectorization_manager
    VECTORBT_OPTIMIZER_AVAILABLE = True
    if TPRINT_AVAILABLE:
        tprint("✅ [optimization] VectorBT optimization modules loaded successfully", color="green")
except ImportError:
    VECTORBT_OPTIMIZER_AVAILABLE = False
    VectorBTRollingOptimizer = None
    get_vectorbt_rolling_optimizer = None
    UnifiedVectorizationManager = None
    get_unified_vectorization_manager = None
    if TPRINT_AVAILABLE:
        tprint("⚠️  [optimization] VectorBT optimization modules not available", color="yellow")

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
