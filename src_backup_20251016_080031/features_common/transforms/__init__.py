"""Transform and scaling utilities shared across feature systems."""

# Import common utilities
from ..utils import (
    TPRINT_AVAILABLE, tprint,
    VECTORBT_OPTIMIZER_AVAILABLE, VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer,
    UnifiedVectorizationManager, get_unified_vectorization_manager
)

from .base_scaler import BaseScaler, create_optimized_scaler, create_optimized_batch_scaler
from .vectorbt_scaler import VectorBTScaler, VectorBTBatchScaler
from .scaling_normalization import ScalingNormalizer
from .categorical_encoding import CategoricalEncoder

__all__ = [
    'BaseScaler',
    'VectorBTScaler',
    'VectorBTBatchScaler',
    'create_optimized_scaler',
    'create_optimized_batch_scaler',
    'ScalingNormalizer',
    'CategoricalEncoder',
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
else:
    print(f"🔧 [transforms] Module initialized with {len(__all__)} exports")
