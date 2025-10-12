"""Transform and scaling utilities shared across feature systems."""

# Import utility functions
try:
    from src.utils.tprint import tprint
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False

from .base_scaler import BaseScaler, create_optimized_scaler, create_optimized_batch_scaler
from .vectorbt_scaler import VectorBTScaler, VectorBTBatchScaler

# Import VectorBT optimization components
try:
    from src.feature_generation.utils.vectorbt_rolling_optimizer import VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer
    from src.feature_generation.utils.unified_vectorization_manager import UnifiedVectorizationManager, get_unified_vectorization_manager
    VECTORBT_OPTIMIZER_AVAILABLE = True
    if TPRINT_AVAILABLE:
        tprint("✅ [transforms] VectorBT optimization modules loaded successfully", color="green")
except ImportError:
    VECTORBT_OPTIMIZER_AVAILABLE = False
    VectorBTRollingOptimizer = None
    get_vectorbt_rolling_optimizer = None
    UnifiedVectorizationManager = None
    get_unified_vectorization_manager = None
    if TPRINT_AVAILABLE:
        tprint("⚠️  [transforms] VectorBT optimization modules not available", color="yellow")

__all__ = [
    'BaseScaler',
    'VectorBTScaler',
    'VectorBTBatchScaler',
    'create_optimized_scaler',
    'create_optimized_batch_scaler',
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
    tprint(f"🔧 [transforms] Module initialized with {len(__all__)} exports", color="cyan")
